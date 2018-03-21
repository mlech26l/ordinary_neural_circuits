import numpy as np
import pybnn
import datetime
import sys
import os

import pyparkgym

class ParkEnv:
    def __init__(self):
        self.nsteps = 10
        self.robot = pyparkgym.ParkGym()
        self.robot.LoadReferenceTrace('learn_tw.dat')
    def reset(self,):
        self.robot.Reset()
        return np.zeros([3])

    # obs, r, done, info = env.step(actions)
    def step(self,action):
        self.robot.Actuate(action[0],action[1])
        self.robot.UpdatePhysics(self.nsteps)

        obs = np.zeros([3])
        obs[0] = self.robot.GetX()
        obs[1] = self.robot.GetY()
        obs[2] = self.robot.GetTheta()

        done = self.robot.IsDone()

        r = self.robot.GetReward()

        return [obs,r,done,None]

    def render(self,):
        return None
