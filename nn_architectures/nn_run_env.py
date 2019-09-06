# from OpenGL import GLU
import gym
import numpy as np
import datetime
import sys
import os
import argparse
import mlp_module
import lstm_module

env_table = {
    "invpend": "RoboschoolInvertedPendulum-v1",
    "cheetah": "HalfCheetah-v2",
    "mountaincar": "MountainCarContinuous-v0",
}

class NNsearchEnv:
    def __init__(self,env_name,filter_len,mean_len,nn_type):
        self.env_name = env_name
        assert env_name in env_table.keys(), "Envname must be one of "+str(env_table.keys())
        if(env_name == "invpend"):
            import roboschool
        
        self.env = gym.make(env_table[env_name])
        self.nn_type = nn_type
        self.create_nn(self.nn_type)

        self.filter_len = filter_len
        self.mean_len = mean_len

    def preprocess_observations(self,obs):
        if(self.env_name == "invpend"):
            return np.array([np.arcsin(obs[3]),obs[0]])
        if(self.env_name == "mountaincar"):
            return np.array([obs[0],obs[1]])
        if(self.env_name == "cheetah"):
            return np.dot(obs,self.w_in)
        raise ValueError("This should not happen")

    def preprocess_actions(self,action):
        if(self.env_name == "cheetah"):
            return np.dot(action,self.w_out)
        else:
            return action

    def run_one_episode(self,do_render=False):
        ### Uncomment to optimize a fake cost function for debugging
        # self.nn.reset_state()
        # c = self.nn.step(np.zeros(2))[0]
        # return np.tanh(c)

        total_reward = 0
        obs = self.env.reset()
        self.nn.reset_state()

        while 1:
            obs = self.preprocess_observations(obs)
            action = self.nn.step(obs)
            action = self.preprocess_actions(action)

            obs, r, done, info = self.env.step(action)

            if(self.env_name == "invpend"):
                max_bonus = 200.0/1000.0
                bonus = (1.0-abs(float(obs[0])))*max_bonus
                if(r>0.0):
                    total_reward+=bonus

            total_reward += r

            if(done):
                break
        # print('Return: '+str(total_reward))
        return total_reward

    def evaluate_avg(self):
        N = 50

        returns = np.zeros(N)
        for i in range(0,N):
            returns[i]= self.run_one_episode()

        return np.mean(returns)

    def input_size(self):
        return int(self.env.observation_space.shape[0])

    def output_size(self):
        return int(self.env.action_space.shape[0])

    def run_multiple_episodes(self):
        returns = np.zeros(self.filter_len)
        for i in range(0,self.filter_len):
            returns[i]= self.run_one_episode()

        sort = np.sort(returns)
        worst_cases = sort[0:self.mean_len]

        return [np.mean(worst_cases),np.mean(returns)]


    def optimize(self,max_steps):
        # Break symmetry by adding noise
        self.nn.add_noise(0.01)

        r_values = np.zeros(1000000)
        r_counter=0

        (current_return,mean_ret) =  self.run_multiple_episodes()
        r_values[r_counter]=mean_ret
        r_counter+=1

        steps_since_last_improvement = 0
        starttime = datetime.datetime.now()
        steps=-1
        log_freq=250
        amplitude = 0.01
        while steps < max_steps:
            steps+=1
            
            self.nn.add_noise(amplitude)
            if(self.env_name == "cheetah"):
                self.w_backup = [np.copy(self.w_in),np.copy(self.w_out)]
                self.w_in += np.random.normal(0,amplitude,size=[self.input_size(),2])
                self.w_out += np.random.normal(0,amplitude,size=[2, self.output_size()])


            (new_return,mean_ret) =  self.run_multiple_episodes()
            r_values[r_counter]=mean_ret
            r_counter+=1
            # print('Stochastic Return: '+str(new_return))
            if(new_return > current_return):
                # print('Improvement! New Return: '+str(new_return))
                current_return=new_return
                steps_since_last_improvement = 0
                amplitude /= 2
                if(amplitude < 0.01):
                    amplitude = 0.01
            else:
                self.nn.undo_noise()
                if(self.env_name == "cheetah"):
                    self.w_in,self.w_out = self.w_backup
                
                steps_since_last_improvement+=1

                # no improvement seen for 100 steps
                if(steps_since_last_improvement>50):
                    amplitude = amplitude*5
                    if(amplitude > 1):
                        amplitude = 1
                    steps_since_last_improvement=0
                    # reevaluate return
                    (current_return,mean_ret) =  self.run_multiple_episodes()


            if(steps % log_freq == 0 and self.csvlogfile != None):
                elapsed = datetime.datetime.now()-starttime
                avg_cost = self.evaluate_avg()
                performance_r = np.mean(r_values[0:r_counter])
                self.csvlogfile.write(str(steps)+';'+str(avg_cost)+';'+str(performance_r)+';'+str(elapsed.total_seconds())+'\n')
                self.csvlogfile.flush()
                # outfile = logdir+'/tw-'+str(worker_id)+'_steps-'+str(steps)+'.bnn'
                # lif.WriteToFile(outfile)
                    # print('Set Distortion to '+str(num_distortions))
        if(self.logfile != None):
            self.logfile.write('Total steps done: '+str(steps)+'\n')
            self.logfile.close()
        if(self.csvlogfile != None):
            elapsed = datetime.datetime.now()-starttime
            avg_cost = self.evaluate_avg()
            performance_r = np.mean(r_values[0:r_counter])
            self.csvlogfile.write(str(steps)+';'+str(avg_cost)+';'+str(performance_r)+';'+str(elapsed.total_seconds())+'\n')
            self.csvlogfile.flush()

    def create_nn(self,nn_type):
        input_dim = 2
        output_dim = 1
        if(self.env_name == "cheetah"):
            output_dim = 2
            self.w_in = np.zeros([self.input_size(),2])
            self.w_out = np.zeros([2, self.output_size()])

        if(nn_type == "mlp"):
            self.nn = mlp_module.MLP(input_dim,12,output_dim)
        elif(nn_type == "lstm"):
            self.nn = lstm_module.LSTM(input_dim,12,output_dim)
        else:
            raise ValueError("This should not happen")

    def run_optimization(self,worker_id,max_steps):

        root_path = 'results/'+self.env_name+'_'+self.nn_type+'/filter_'+str(self.filter_len)+'_'+str(self.mean_len)
        log_path = root_path+'/logs_csv'
        log_path_txt = root_path+'/logs_txt'
        store_path = root_path+'/final'

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(log_path_txt):
            os.makedirs(log_path_txt)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        log_file=log_path_txt+'/textlog_'+worker_id+'.log'
        csv_log=log_path+'/csvlog_'+worker_id+'.log'
        self.logfile = open(log_file, 'w')
        self.csvlogfile = open(csv_log, 'w')


        print('Begin Return of '+worker_id+': '+str(self.run_multiple_episodes()))
        self.optimize(max_steps)
        print('End Return: of '+worker_id+': '+str(self.run_multiple_episodes()))

        # outfile = store_path+'/'+self.env_name+'_'+self.nn_type+'-optimized_'+worker_id+ '.npz'

        # self.lif.WriteToFile(outfile)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',default="invpend")
    parser.add_argument('--nn',default="mlp")
    parser.add_argument('--steps',default=20000,type=int)
    parser.add_argument('--filter',default=10,type=int)
    parser.add_argument('--mean',default=5,type=int)
    parser.add_argument('--optimize',action="store_true")
    parser.add_argument('--id',default="0")
    args = parser.parse_args()

    nnenv = NNsearchEnv(
        env_name = args.env,
        filter_len = args.filter,
        mean_len = args.mean,
        nn_type = args.nn
    )
    print("Optimize")
    nnenv.run_optimization(args.id,args.steps)
    # if(args.optimize):
    # else:
    #     print("Replay")
    #     nnenv.replay(args.file)

