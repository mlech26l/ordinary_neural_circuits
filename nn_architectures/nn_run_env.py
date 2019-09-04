# from OpenGL import GLU
from gym import wrappers
import numpy as np
import random as rng
from PIL import Image,ImageDraw
import datetime
import sys
import os
import argparse
import mlp_module
import lstm_module

env_table = {
    "invpend": "RoboschoolInvertedPendulum-v1",
    "cheetah": "HalfCheetah-v1",
    "mountaincar": "MountainCarContinuous-v0",
}

class NNsearchEnv:
    def __init__(self,env_name,filter,mean,nn_type):
        self.env_name = env_name
        assert env_name in env_table.keys(), "Envname must be one of "+str(env_table.keys())
        if(env_name == "invpend"):
            import gym, roboschool
        
        self.env = gym.make(env_table[env_name])
        self.nn_type = nn_type
        self.create_nn(self.nn_type)

        self.filter_len = filter
        self.mean_len=mean

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
        total_reward = 0
        obs = self.env.reset()
        self.nn.Reset()

        total_reward=np.zeros(1)
        while 1:
            obs = self.preprocess_observations(obs)
            action = self.nn.step(obs)
            action = self.preprocess_actions(action)

            obs, r, done, info = self.env.step(action)

            if(self.env_name == "inv_pend"):
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
        self.lif.AddNoise(0.5,15)
        self.lif.AddNoiseVleak(8,8)
        self.lif.AddNoiseGleak(0.2,8)
        self.lif.AddNoiseSigma(0.2,10)
        self.lif.AddNoiseCm(0.1,10)
        self.lif.CommitNoise()

        r_values = np.zeros(1000000)
        r_counter=0

        (current_return,mean_ret) =  self.run_multiple_episodes()
        r_values[r_counter]=mean_ret
        r_counter+=1

        num_distortions = 4
        num_distortions_sigma=3
        num_distortions_vleak=2
        num_distortions_gleak=2
        num_distortions_cm=2
        steps_since_last_improvement=0

        starttime = datetime.datetime.now()
        endtime = starttime + ts
        steps=-1
        log_freq=250
        while endtime>datetime.datetime.now() and steps < max_steps:
            steps+=1

            # weight
            distortions = rng.randint(0,num_distortions)
            variance = rng.uniform(0.01,0.4)

            # sigma
            distortions_sigma = rng.randint(0,num_distortions_sigma)
            variance_sigma = rng.uniform(0.01,0.05)

            # vleak
            distortions_vleak = rng.randint(0,num_distortions_vleak)
            variance_vleak = rng.uniform(0.1,3)

            # vleak
            distortions_gleak = rng.randint(0,num_distortions_gleak)
            variance_gleak = rng.uniform(0.05,0.5)

            #cm
            distortions_cm = rng.randint(0,num_distortions_cm)
            variance_cm = rng.uniform(0.01,0.1)

            self.lif.AddNoise(variance,distortions)
            self.lif.AddNoiseSigma(variance_sigma,distortions_sigma)
            self.lif.AddNoiseVleak(variance_vleak,distortions_vleak)
            self.lif.AddNoiseCm(variance_cm,distortions_cm)
            self.lif.AddNoiseGleak(variance_gleak,distortions_gleak)

            (new_return,mean_ret) =  self.run_multiple_episodes()
            r_values[r_counter]=mean_ret
            r_counter+=1
            # print('Stochastic Return: '+str(new_return))
            if(new_return > current_return):
                # print('Improvement! New Return: '+str(new_return))
                if(self.logfile != None):
                    elapsed = datetime.datetime.now()-starttime
                    self.logfile.write('Improvement after: '+str(steps)+' steps, with return '+str(new_return)+', Elapsed: '+str(elapsed.total_seconds())+'\n')
                    self.logfile.flush()

                current_return=new_return
                self.lif.CommitNoise()
                steps_since_last_improvement=0

                num_distortions-=1
                if(num_distortions<4):
                    num_distortions=4

                num_distortions_sigma-=1
                if(num_distortions_sigma<3):
                    num_distortions_sigma=3

                num_distortions_vleak-=1
                if(num_distortions_vleak<2):
                    num_distortions_vleak=2

                num_distortions_gleak-=1
                if(num_distortions_gleak<2):
                    num_distortions_gleak=2

                num_distortions_cm-=1
                if(num_distortions_cm<2):
                    num_distortions_cm=2
                # print('Set Distortion to '+str(num_distortions))
            else:
                steps_since_last_improvement+=1
                self.lif.UndoNoise()

                # no improvement seen for 100 steps
                if(steps_since_last_improvement>50):
                    steps_since_last_improvement=0

                    # reevaluate return
                    (current_return,mean_ret) =  self.run_multiple_episodes()
                    r_values[r_counter]=mean_ret
                    r_counter+=1
                    # print('Reevaluate to: '+str(current_return))
                    if(self.logfile != None):
                        self.logfile.write('Reevaluate after: '+str(steps)+' steps, with return '+str(new_return)+'\n')
                        self.logfile.flush()


                    # Increase variance
                    num_distortions+=1
                    if(num_distortions>12):
                        num_distortions=12
                    # Increase variance sigma
                    num_distortions_sigma+=1
                    if(num_distortions_sigma>8):
                        num_distortions_sigma=8
                    # Increase variance vleak
                    num_distortions_vleak+=1
                    if(num_distortions_vleak>6):
                        num_distortions_vleak=6
                    # Increase variance vleak
                    num_distortions_gleak+=1
                    if(num_distortions_gleak>6):
                        num_distortions_gleak=6
                    # Increase variance cm
                    num_distortions_cm+=1
                    if(num_distortions_cm>4):
                        num_distortions_cm=4
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
            self.w_in = np.random.normal(0,1,size=[self.input_size(),2])
            self.w_out = np.random.normal(0,1,size=[2, self.output_size()])

        if(nn_type == "mlp"):
            self.nn = mlp_module.MLP(input_dim,12,output_dim)
        elif(nn_type == "lstm"):
            self.nn = lstm_module.LSTM(input_dim,12,output_dim)
        else:
            raise ValueError("This should not happen")

    def run_optimization(self,worker_id,max_steps):
        seed = int(worker_id)+20*datetime.datetime.now().microsecond+23115
        rng.seed(seed)

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
        self.optimize(ts=datetime.timedelta(hours=12),max_steps=50000)
        print('End Return: of '+worker_id+': '+str(self.run_multiple_episodes()))

        outfile = store_path+'/tw-optimized_'+worker_id+ '.bnn'

        self.lif.WriteToFile(outfile)


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
        filter = args.filter,
        mean = args.mean,
        nn_type = args.nn
    )
    if(args.optimize):
        print("Optimize")
        nnenv.run_optimization(args.id,args.steps)
    else:
        print("Replay")
        nnenv.replay(args.file)

