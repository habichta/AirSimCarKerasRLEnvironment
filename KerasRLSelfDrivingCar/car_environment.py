
import airsim
import time
from gym import spaces
import math
import numpy as np

class AirSimInterface():

    def __init__(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.initialCarControlSettings()
        self.blocked_detection = 0
        self.collision_detection = 0
        self.reverse_count = 0
        self.last_collision_object = None

    def initialCarControlSettings(self):
        self.car_controls.is_manual_gear=False
        self.car_controls.gear_immediate=True
        self.client.setCarControls(self.car_controls)

        position = airsim.Vector3r(0.0 , 0.0, 0.0)
        
        heading = airsim.utils.to_quaternion(0.0, 0.0, np.random.uniform(0.0,6.5))
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)




    def getMinimalActionSet(self):
        return range(0,4)



    def act(self,action):
        
        self._set_controls(action)


        ob = self.getNextState()

        reward,done = self._calculate_reward_and_done()

        print('action',action,'reward',reward)

        return ob,reward,done

    def _set_controls(self,action):
       
        self.car_controls.throttle = 1.0       
        self.car_controls.brake = 0.0
        self.car_controls.is_manual_gear = False;

        if self.client.getCarState().speed > 5.5:
            self.car_controls.brake = 1.0
        
        if action == 0:
            self.car_controls.steering = 0.0
        elif action == 1:
            self.car_controls.steering = 0.8
        elif action == 2:
            self.car_controls.steering = -0.8
        elif action == 3:
            self.car_controls.brake = 1.0

        self.client.setCarControls(self.car_controls)

    def _car_break_free(self,t=3):
        
        print('Breaking free')

        self.car_controls.throttle = -0.8
        self.car_controls.is_manual_gear = True;
        self.car_controls.manual_gear = -1
        self.car_controls.steering = 0
        self.client.setCarControls(self.car_controls)
        time.sleep(t)
        self.car_controls.throttle = 1.0
        self.car_controls.manual_gear = 1
        self.client.setCarControls(self.car_controls)
        time.sleep(1.5)
        self.car_controls.is_manual_gear = False;
        self.client.setCarControls(self.car_controls)

    def reset(self):
        self.client.reset()
        self.blocked_detection = 0
        self.collision_detection = 0
        self._set_controls(0)
        time.sleep(1)
        self.initialCarControlSettings()
        ob = self.getNextState()
        return ob


    def _has_collided(self):
        collision_info = self.client.simGetCollisionInfo()
        collision_timestamp = math.floor(float(collision_info.time_stamp)/pow(10,9))
        current_time =  math.floor(time.time())

        if collision_timestamp == current_time:
            print(collision_timestamp,current_time)
            self.collision_detection +=1
            return True
        return False


    def _calculate_reward_and_done(self):
        
        car_state = self.client.getCarState()
        collided = self._has_collided()
        car_speed = car_state.speed

        if abs(car_speed) < 0.6:
            self.blocked_detection += 1
        else:
            self.blocked_detection = 0
        
        print('blocked/collision counter:', self.blocked_detection,self.collision_detection)
        print('collided:', collided)
        if self.blocked_detection > 15:
                return -1,True
        elif collided:
            if car_speed < 0.1 and self.blocked_detection < 5:
                self._car_break_free()
            if self.collision_detection > 5:
                return -1, True
            else:
                return -1, False

        elif car_state.speed > 2:
                return 1,False
        else:
                return 0,False

    def getNextState(self):
        state = None
        while(not state or state[0].width==0 or state[0].height==0):
            state = self.getDepthImage()
        return state

    def getDepthImage(self):
        return self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False)])



class CarEnvironment():


    def __init__(self):

        self.airi = AirSimInterface()
        self._action_set = self.airi.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))


    def seed(self,seed):
        pass


    def step(self,a):
    
        ob, reward, done = self.airi.act(a)
    
        return ob, reward, done, {'dummy': 0}

    def reset(self):

        return self.airi.reset()
    
    def get_observation():
        return self.airi.getDepthImage()

    @property   
    def _n_actions(self):
        return len(self._action_set)
