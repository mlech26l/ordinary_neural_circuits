import gym
from gym import wrappers
import numpy as np
import pybnn
import random as rng
from PIL import Image,ImageDraw
import datetime
import sys
import os
import argparse

class TWsearchEnv:
    def __init__(self,env,filter_len, mean_len):
        self.env = env
        self.filter_len = filter_len
        self.mean_len=mean_len

    def TensorRGBToImage(self,tensor):
        new_im = Image.new("RGB",(tensor.shape[1],tensor.shape[0]))
        pixels=[]
        for y in range(tensor.shape[0]):
            for x in range(tensor.shape[1]):
                r = tensor[y][x][0]
                g = tensor[y][x][1]
                b = tensor[y][x][2]
                pixels.append((r,g,b))
        new_im.putdata(pixels)
        return new_im

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
        i=0

        done2 = False
        while 1:
            action = self.lif.Update(observations,0.01,10)
            actions[0]=action[0]
            obs, r, done, info = self.env.step(actions)
            self.set_observations_for_lif(obs,observations)


            total_reward += r*gamma
            time += 0.0165

            if(do_render):
                rewardlog.write(str(total_reward)+'\n')
                rewardlog.flush()
                self.lif.DumpState('lif-dump.csv')
                self.env.render()
                print("R: {:0.3f}".format(float(total_reward)))
                # screen = env.render(mode='rgb_array')
                # print('Img shape: '+str(screen.shape))
                # pic = TensorRGBToImage(screen)
                # pic.save('vid/img_'+str(i).zfill(5)+'.png')
                # phi = np.arcsin(obs[3])
                # print('Obs: '+str(phi)+', '+str(obs[4])+' Act: '+str(actions[0]))

                if(time >= 16.5):
                    return
            elif(done):
                break
            i+=1
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
        self.lif.AddBiSensoryNeuron(1,6,-0.3,0.3)
        # self.lif.AddBiSensoryNeuron(1,6,-1.0,1.0)
        self.lif.AddBiSensoryNeuron(7,0,-0.02,0.02)

        self.lif.AddBiMotorNeuron(9,10,-1,1)

        self.lif.Reset()



    def optimize(self,ts=datetime.timedelta(seconds=60),max_steps=1000000):
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

        num_distortions = 6
        num_distortions_sigma=5
        num_distortions_vleak=5
        num_distortions_gleak=4
        num_distortions_cm=4
        steps_since_last_improvement=0

        starttime = datetime.datetime.now()
        endtime = starttime + ts
        steps=-1
        log_freq=250
        while endtime>datetime.datetime.now() and steps < max_steps:
            steps+=1

            # weight
            distortions = rng.randint(0,num_distortions)
            variance = rng.uniform(0.01,0.8)

            # sigma
            distortions_sigma = rng.randint(0,num_distortions_sigma)
            variance_sigma = rng.uniform(0.01,0.08)

            # vleak
            distortions_vleak = rng.randint(0,num_distortions_vleak)
            variance_vleak = rng.uniform(0.1,3)

            # vleak
            distortions_gleak = rng.randint(0,num_distortions_gleak)
            variance_gleak = rng.uniform(0.05,0.8)

            #cm
            distortions_cm = rng.randint(0,num_distortions_cm)
            variance_cm = rng.uniform(0.01,0.3)

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
                print('Improvement! New Return: '+str(new_return))
                if(self.logfile != None):
                    elapsed = datetime.datetime.now()-starttime
                    self.logfile.write('Improvement after: '+str(steps)+' steps, with return '+str(new_return)+', Elapsed: '+str(elapsed.total_seconds())+'\n')
                    self.logfile.flush()

                current_return=new_return
                self.lif.CommitNoise()
                steps_since_last_improvement=0

                num_distortions-=1
                if(num_distortions<6):
                    num_distortions=6

                num_distortions_sigma-=1
                if(num_distortions_sigma<5):
                    num_distortions_sigma=5

                num_distortions_vleak-=1
                if(num_distortions_vleak<4):
                    num_distortions_vleak=4

                num_distortions_gleak-=1
                if(num_distortions_gleak<4):
                    num_distortions_gleak=4

                num_distortions_cm-=1
                if(num_distortions_cm<4):
                    num_distortions_cm=4
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
                    if(num_distortions>16):
                        num_distortions=12
                    # Increase variance sigma
                    num_distortions_sigma+=1
                    if(num_distortions_sigma>12):
                        num_distortions_sigma=12
                    # Increase variance vleak
                    num_distortions_vleak+=1
                    if(num_distortions_vleak>8):
                        num_distortions_vleak=8
                    # Increase variance vleak
                    num_distortions_gleak+=1
                    if(num_distortions_gleak>8):
                        num_distortions_gleak=8
                    # Increase variance cm
                    num_distortions_cm+=1
                    if(num_distortions_cm>7):
                        num_distortions_cm=7
                if(steps_since_last_improvement > 300):
                    self.lif.AddNoise(0.5,15)
                    self.lif.AddNoiseVleak(8,8)
                    self.lif.AddNoiseGleak(0.2,8)
                    self.lif.AddNoiseSigma(0.2,10)
                    self.lif.AddNoiseCm(0.1,10)
                    self.lif.CommitNoise()
                    
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

    def replay(self,filename):
        self.load_tw(filename)
        if not os.path.exists('vid'):
            os.makedirs('vid')
        print('Average Reward: '+str(self.evaluate_avg()))
        print('Replay Return: '+str(self.run_multiple_episodes()))

        self.run_one_episode(True)


    def replay_arg(self):

        worker_id =1
        if(len(sys.argv)>1):
            worker_id = int(sys.argv[1])

        filename = 'bnn1/tw-optimized_'+str(worker_id)+'.bnn'
        self.load_tw(filename)

        print('Replay Return: '+str(self.run_multiple_episodes()))

        self.run_one_episode(True)

    def optimize_and_store(self,worker_id,in_file='tw_pure.bnn'):
        self.load_tw(in_file)

        if(worker_id.isdigit()):
            seed = int(worker_id)+20*datetime.datetime.now().microsecond+23115
        else:
            seed = 20*datetime.datetime.now().microsecond+23115

        self.lif.SeedRandomNumberGenerator(seed)
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

        log_file=log_path_txt+'/textlog_'+worker_id+'.log'
        csv_log=log_path+'/csvlog_'+worker_id+'.log'
        self.logfile = open(log_file, 'w')
        self.csvlogfile = open(csv_log, 'w')


        print('Begin Return of '+worker_id+': '+str(self.run_multiple_episodes()))
        self.optimize(ts=datetime.timedelta(minutes=3),max_steps=50000)
        print('End Return: of '+worker_id+': '+str(self.run_multiple_episodes()))

        outfile = store_path+'/tw-optimized_'+worker_id+ '.bnn'

        self.lif.WriteToFile(outfile)


def demo_run():
    env = gym.make("MountainCarContinuous-v0")
    # print('Observation space: '+str(env.observation_space.shape[0]))
    # print('Action space: '+str(env.action_space.shape[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument('--filter',default=20,type=int)
    parser.add_argument('--mean',default=20,type=int)
    parser.add_argument('--file',default="tw_pure.bnn")
    parser.add_argument('--optimize',action="store_true")
    parser.add_argument('--id',default="0")
    args = parser.parse_args()

    twenv = TWsearchEnv(env,args.filter,args.mean)
    if(args.optimize):
        print("Optimize")
        twenv.optimize_and_store(str(args.id),args.file)
    else:
        print("Replay")
        twenv.replay(args.file)

if __name__=="__main__":
    demo_run()
