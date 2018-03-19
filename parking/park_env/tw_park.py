# from OpenGL import GLU
#import gym, roboschool
#from gym import wrappers
import numpy as np
import pybnn
import random as rng
import park_env
import datetime
import sys
import os

class TWsearchEnv:
    def __init__(self,env):
        self.env = env

    def set_observations_for_lif(self,obs,observations,time):
        observations[0] = obs[0]
        observations[1] = obs[1]
        observations[3] = obs[2]
        observations[2]=0.0
        if(time>=3.0):
            observations[2]=1.0

    def run_one_episode(self,do_render=False):
        total_reward = 0
        obs = self.env.reset()
        self.lif.Reset()
        if(do_render):
            print('dump lif')
            rewardlog = open('rewardlog.log','w')
            self.lif.DumpClear('lif-dump.csv')

        observations = []
        for i in range(0,4):
            observations.append(float(0))

        self.set_observations_for_lif(obs,observations,0)
        actions = [0.0,0.0]
        self.lif.Update(observations,0.01,10)

        total_reward=np.zeros(1)
        gamma = 1.0
        time =0.0

        start_pos=0
        has_started=False
        while 1:
            actions = self.lif.Update(observations,0.01,10)
            obs, r, done, info = self.env.step(actions)

            time += 0.1
            self.set_observations_for_lif(obs,observations,time)


            total_reward += r*gamma
                #gamma = gamma*gamma
            if(do_render):
                rewardlog.write(str(total_reward)+'\n')
                rewardlog.flush()
                self.lif.DumpState('lif-dump.csv')

            if(done):
                break
        # print('Return: '+str(total_reward))
        return np.sum(total_reward)




    def load_tw(self,filename):
        self.lif = pybnn.LifNet(filename)
        #lif.AddBiSensoryNeuron(1,6,-0.3,0.3)
        # lif.AddBiSensoryNeuron(1,6,-0.03,0.03)
        # lif.AddBiSensoryNeuron(7,0,-0.15,0.15)


        # lif.AddBiSensoryNeuron(1,6,-0.03,0.03)
        self.lif.AddSensoryNeuron(0,1)
        self.lif.AddSensoryNeuron(1,1)
        self.lif.AddSensoryNeuron(6,1)
        self.lif.AddSensoryNeuron(7,1)

        self.lif.AddMotorNeuron(11,1)
        #self.lif.AddBiMotorNeuron(11,12,-1,1)
        self.lif.AddBiMotorNeuron(9,10,-1,1)

        self.lif.Reset()


    def optimize(self,ts=datetime.timedelta(seconds=60)):
        # Break symmetry by adding noise
        self.lif.AddNoise(0.5,15)
        self.lif.AddNoiseVleak(8,8)
        self.lif.AddNoiseGleak(0.2,8)
        self.lif.AddNoiseSigma(0.2,10)
        self.lif.AddNoiseCm(0.1,10)
        self.lif.CommitNoise()

        r_values = np.zeros(1000000)
        r_counter=0

        current_return =  self.run_one_episode()
        r_values[r_counter]=current_return
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
        while endtime>datetime.datetime.now():
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

            new_return =  self.run_one_episode()
            r_values[r_counter]=new_return
            r_counter+=1
            # print('Stochastic Return: '+str(new_return))
            if(new_return > current_return):
                # print('Improvement! New Return: '+str(new_return))
                if(self.logfile != None):
                    elapsed = datetime.datetime.now()-starttime
                    self.logfile.write('Improvement after: '+str(steps)+' steps, with return '+str(new_return)+', Elapsed: '+str(elapsed.total_seconds())+'\n')

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
                avg_cost = self.run_one_episode()
                performance_r = np.mean(r_values[0:r_counter])
                self.csvlogfile.write(str(steps)+';'+str(avg_cost)+';'+str(performance_r)+';'+str(elapsed.total_seconds())+'\n')
                self.csvlogfile.flush()
                # outfile = logdir+'/tw-'+str(worker_id)+'_steps-'+str(steps)+'.bnn'
                # lif.WriteToFile(outfile)
                    # print('Set Distortion to '+str(num_distortions))
        if(self.logfile != None):
            self.logfile.write('Total steps done: '+str(steps)+'\n')
            self.logfile.close()

    def create_and_optimize(self,):
        self.lif=load_tw('final/tw-optimized_1.bnn')

        self.lif.SeedRandomNumberGenerator(23661);

        print('Begin Return: '+str(self.run_one_episode(env,lif)))
        self.optimize(ts=datetime.timedelta(minutes=20))
        print('End Return: '+str(self.run_one_episode(env,lif)))


        self.lif.WriteToFile('tw-optimized_parking.bnn')
        self.run_one_episode(True)

    def replay(self,filename):
        self.load_tw(filename)

        print('Replay Return: '+str(self.run_one_episode()))

        self.run_one_episode(True)


    def replay_arg(self):

        worker_id =1
        if(len(sys.argv)>1):
            worker_id = int(sys.argv[1])

        filename = 'bnn1/tw-optimized_'+str(worker_id)+'.bnn'
        self.load_tw(filename)

        print('Replay Return: '+str(self.run_one_episode()))

        self.run_one_episode(True)


    def optimize_and_store(self):

        self.load_tw('tw_pure.bnn')

        worker_id =1
        if(len(sys.argv)>1):
            worker_id = int(sys.argv[1])

        seed = worker_id+20*datetime.datetime.now().microsecond+23115
        self.lif.SeedRandomNumberGenerator(seed);
        rng.seed(seed)

        root_path = 'results'
        log_path = root_path+'/logs_csv'
        log_path_txt = root_path+'/logs_txt'
        store_path = root_path+'/final'

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(log_path_txt):
            os.makedirs(log_path_txt)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        log_file=log_path_txt+'/textlog_'+str(worker_id)+'.log'
        csv_log=log_path+'/csvlog_'+str(worker_id)+'.log'
        #self.logfile = open(log_file, 'w')
        #self.csvlogfile = open(csv_log, 'w')
        self.logfile=None
        self.csvlogfile=None


        print('Begin Return of '+str(worker_id)+': '+str(self.run_one_episode()))
        self.optimize(ts=datetime.timedelta(hours=0,minutes=5))
        print('End Return: of '+str(worker_id)+': '+str(self.run_one_episode()))

        outfile = store_path+'/tw-optimized_'+str(worker_id)+ '.bnn';
        self.lif.WriteToFile(outfile)
        print('Stored!')



def demo_run():
    env = park_env.ParkEnv()

    env.reset();

    twenv = TWsearchEnv(env)
    #twenv.replay('results/final/tw-optimized_1.bnn')
    twenv.optimize_and_store()

if __name__=="__main__":
    demo_run()
