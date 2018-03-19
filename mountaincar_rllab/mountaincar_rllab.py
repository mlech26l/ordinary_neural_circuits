from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.normalized_env import normalize
import numpy as np
from PIL import Image,ImageDraw
import pybnn
import random as rng
import datetime
import sys
import os
import matplotlib.pyplot as plt
from time import sleep

class TWsearchEnv:
    def __init__(self,env,filter_len, mean_len):
        self.env = env
        self.filter_len = filter_len
        self.mean_len=mean_len

        # total_episodes = 50000
        # self.episode_limit = total_episodes/self.filter_len
        self.episode_limit=50000

    def set_observations_for_lif(self,obs,observations):
        observations[0] = float(obs[0])
        observations[1] = float(obs[1])

    def run_one_episode(self,do_render=False):
        total_reward = 0
        obs = self.env.reset()
        self.lif.Reset()
        if(do_render):
            rewardlog = open('rewardlog.log','w')
            self.lif.DumpClear('lif-dump.csv')
            N = 500
            obs1 = np.zeros(N)
            obs2 = np.zeros(N)

        observations = []
        for i in range(0,2):
            observations.append(float(0))

        self.set_observations_for_lif(obs,observations)
        actions = np.zeros(1)
        self.lif.Update(observations,0.01,10)

        total_reward=np.zeros(1)
        gamma = 1.0
        time =0.0

        start_pos=0
        has_started=False
        i = 0

        # maximum of 1000 steps
        T = 500
        for t in range(T):
            action = self.lif.Update(observations,0.01,10)
            actions=np.array(action)
            if(do_render):
                print('Step: '+str(t)+', t: '+str(time)+', total r '+str(total_reward)+': '+str(obs[0])+': '+str(obs[1]))
                time += 1.0/24.0
                obs1[i] = obs[0]
                obs2[i] = obs[1]
            obs, r, done, info = self.env.step(actions)
            self.set_observations_for_lif(obs,observations)


            total_reward += r*gamma
            #gamma = gamma*gamma

            if(do_render):
                rewardlog.write(str(total_reward)+'\n')
                rewardlog.flush()
                self.lif.DumpState('lif-dump.csv')
                # pic = env.render()
                screen = self.env.render()
                sleep(0.016)
                # pic = TensorRGBToImage(screen)
                # pic.save('vid/img_'+str(i).zfill(5)+'.png')
            elif(done):
                break
            i+=1
        if(do_render):
            plt.clf()
            plt.hist(obs1, 50, normed=1, facecolor='green', alpha=0.75)

            plt.xlabel('Observation')
            plt.ylabel('Probability')
            plt.title('Observation[0]')
            plt.grid(True)

            plt.savefig('f10_5_obs1.png')

            plt.clf()
            plt.hist(obs2, 50, normed=1, facecolor='green', alpha=0.75)

            plt.xlabel('Observation')
            plt.ylabel('Probability')
            plt.title('Observation[1]')
            plt.grid(True)

            plt.savefig('f10_5_obs2.png')

        # print('Return: '+str(total_reward))
        return np.sum(total_reward)

    def evaluate_avg(self):
        N = 50

        returns = np.zeros(N)
        for i in range(0,N):
            returns[i]= self.run_one_episode()

        return np.mean(returns)



    def run_multiple_episodes(self):
        returns = np.zeros(self.filter_len)
        for i in range(0,self.filter_len):
            returns[i]= self.run_one_episode()

        sort = np.sort(returns)
        worst_cases = sort[0:self.mean_len]

        return [np.mean(worst_cases),np.mean(returns)]



    def load_tw(self,filename):
        self.lif = pybnn.LifNet(filename)
        #lif.AddBiSensoryNeuron(1,6,-0.3,0.3)
        # self.lif.AddBiSensoryNeuron(1,6,-0.2,0.2)
        # self.lif.AddBiSensoryNeuron(7,0,-0.5,0.5)

        self.lif.AddBiSensoryNeuron(1,6,-0.8,0.8)
        self.lif.AddBiSensoryNeuron(7,0,-1.5,1.5)

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

        r_values = np.zeros(self.episode_limit)
        r_counter=0

        (current_return,mean_ret) =  self.run_multiple_episodes()
        r_values[r_counter]=mean_ret
        r_counter+=1

        avg_cost = self.evaluate_avg()
        self.csvlogfile.write(str(0)+';'+str(0)+';'+str(avg_cost)+';'+str(mean_ret)+';'+str(0)+'\n')
        self.csvlogfile.flush()

        num_distortions = 4
        num_distortions_sigma=3
        num_distortions_vleak=2
        num_distortions_gleak=2
        num_distortions_cm=2
        steps_since_last_improvement=0

        starttime = datetime.datetime.now()
        endtime = starttime + ts
        steps=-1
        log_freq=1000
        # while endtime>datetime.datetime.now():
        while True:
            steps+=1

            if(r_counter >= self.episode_limit):
                break

            # weight
            distortions = rng.randint(0,num_distortions)
            variance = rng.uniform(0.05,0.5)

            # sigma
            distortions_sigma = rng.randint(0,num_distortions_sigma)
            variance_sigma = rng.uniform(0.01,0.05)

            # vleak
            distortions_vleak = rng.randint(0,num_distortions_vleak)
            variance_vleak = rng.uniform(0.5,5)

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
                if(num_distortions<5):
                    num_distortions=5

                num_distortions_sigma-=1
                if(num_distortions_sigma<4):
                    num_distortions_sigma=4

                num_distortions_vleak-=1
                if(num_distortions_vleak<4):
                    num_distortions_vleak=4

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
                if(steps_since_last_improvement>50 and r_counter % log_freq != 0):
                    steps_since_last_improvement=0

                    # reevaluate return
                    if(r_counter >= self.episode_limit):
                        break
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
                    if(num_distortions_vleak>8):
                        num_distortions_vleak=8
                    # Increase variance vleak
                    num_distortions_gleak+=1
                    if(num_distortions_gleak>6):
                        num_distortions_gleak=6
                    # Increase variance cm
                    num_distortions_cm+=1
                    if(num_distortions_cm>4):
                        num_distortions_cm=4
            if(r_counter % log_freq == 0 and self.csvlogfile != None):
                elapsed = datetime.datetime.now()-starttime
                avg_cost = self.evaluate_avg()
                performance_r = np.mean(r_values[0:r_counter])
                self.csvlogfile.write(str(steps)+';'+str(r_counter)+';'+str(avg_cost)+';'+str(performance_r)+';'+str(elapsed.total_seconds())+'\n')
                self.csvlogfile.flush()
                # outfile = logdir+'/tw-'+str(worker_id)+'_steps-'+str(steps)+'.bnn'
                # lif.WriteToFile(outfile)
                    # print('Set Distortion to '+str(num_distortions))
        if(self.csvlogfile != None):
            elapsed = datetime.datetime.now()-starttime
            avg_cost = self.evaluate_avg()
            performance_r = np.mean(r_values[0:r_counter])
            self.csvlogfile.write(str(steps)+';'+str(r_counter)+';'+str(avg_cost)+';'+str(performance_r)+';'+str(elapsed.total_seconds())+'\n')
            self.csvlogfile.flush()

        if(self.logfile != None):
            self.logfile.write('Total steps done: '+str(steps)+'\n')
            self.logfile.close()

    def replay(self,filename):
        self.load_tw(filename)
        if not os.path.exists('vid'):
            os.makedirs('vid')

        # starttime = datetime.datetime.now()
        print('Average Reward: '+str(self.evaluate_avg()))
        # elapsed = datetime.datetime.now()-starttime
        # print('Performance: ', str(500.0/elapsed.total_seconds()),' episodes per second')
        # sys.exit()

        print('Replay Return: '+str(self.run_multiple_episodes()))

        self.run_one_episode(True)


    def optimize_and_store(self):

        self.load_tw('tw_pure.bnn')

        worker_id =1
        if(len(sys.argv)>1):
            worker_id = int(sys.argv[1])

        seed = worker_id+20*datetime.datetime.now().microsecond+23115
        self.lif.SeedRandomNumberGenerator(seed);
        rng.seed(seed)

        root_path = 'results/filter_'+str(self.filter_len)+'_'+str(self.mean_len)
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
        self.logfile = open(log_file, 'w')
        self.csvlogfile = open(csv_log, 'w')


        print('Begin Return of '+str(worker_id)+': '+str(self.run_multiple_episodes()))
        self.optimize(ts=datetime.timedelta(hours=16))
        print('End Return: of '+str(worker_id)+': '+str(self.run_multiple_episodes()))

        outfile = store_path+'/tw-optimized_'+str(worker_id)+ '.bnn';

        self.lif.WriteToFile(outfile)



if __name__=="__main__":
    env = normalize(MountainCarEnv())

    fitler_len = 10
    mean_len = 5
    if(len(sys.argv)>2):
        fitler_len = int(sys.argv[2])
        mean_len = int(sys.argv[2])
    if(len(sys.argv)>3):
        mean_len = int(sys.argv[3])

    twenv = TWsearchEnv(env,fitler_len,mean_len)
    twenv.replay('final/tw-optimized.bnn')
    #twenv.optimize_and_store()

