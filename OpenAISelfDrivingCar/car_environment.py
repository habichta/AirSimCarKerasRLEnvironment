
import airsim
import time
from gym import spaces
from gym import Env
import math
import numpy as np
from airsim_interface import AirSimInterface
from gym import spaces

class CarEnvironment(Env):


    def __init__(self):
        
        #throttle,steering,brake,gear
        action_space = spaces.Box(low=np.array([-1.0,-1.0,-1.0,-1.0]), high=np.array([1.0,1.0,1.0,1.0]) , dtype=np.float32)
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(84,84), dtype=np.float32)
        self._action_space = action_space
        self._observation_space = observation_space
        self.airi = AirSimInterface()
    

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reward_range(self):
        pass

    @property
    def metadata(self):
        pass

    def seed(self,seed):
        pass

    def close():
        pass

    def compute_reward(self,achieved_goal, desired_goal):
        pass

    @property
    def unwrapped(self):
        pass

    @property
    def spec(self):
        pass

    def step(self,a):
    
        ob, reward, done = self.airi.act(a)
    
        return ob, reward, done, {'dummy': 0}

    def reset(self):

        return self.airi.reset()

    def render(self,mode,**kwargs):
        pass

    
    
    def get_observation():
        return self.airi.getDepthImage()

    
