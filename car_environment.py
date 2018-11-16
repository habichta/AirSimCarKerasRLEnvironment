
import airsim
import time
from gym import spaces

class AirSimInterface():

    def __init__(self, action_time_delay=1):
        self.action_time_delay = action_time_delay
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.initialCarControlSettings()

    def initialCarControlSettings(self):
        self.car_controls.is_manual_gear=False
        self.car_controls.gear_immediate=True
        self.client.setCarControls(self.car_controls)

    def getMinimalActionSet(self):
        return range(0,6)



    def act(self,action):
        
        self._set_controls(action)


        reward,done = self._calculate_reward_and_done()

        print('action',action,'reward',reward)

        return reward,done

    def _set_controls(self,action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1
        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:   
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.steering = 0.25
        else:
            self.car_controls.steering = -0.25
        
        self.client.setCarControls(self.car_controls)


    def reset(self):
        self.client.reset()
        self._set_controls(1)
        time.sleep(1)
        ob = self.getNextState()
        return ob



    def _calculate_reward_and_done(self):
        
        collision_info = self.client.simGetCollisionInfo()
        car_state = self.client.getCarState()
        
        if collision_info.has_collided:
                return -1,True
        elif car_state.speed > 4:
                return 1,False
        else:
                return 0,False

    def getNextState(self):
        state = None
        while(not state):
            print('Get New State')
            state = self.getDepthImage()
        return state

    def getDepthImage(self):
        return self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])



class CarEnvironment():


    def __init__(self):

        self.airi = AirSimInterface()
        self._action_set = self.airi.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))


    def seed(self,seed):
        pass


    def step(self,a):
    
        reward, done = self.airi.act(a)
        ob = self.airi.getNextState()
        return ob, reward, done, {'dummy': 0}

    def reset(self):
        return self.airi.reset()
    
    def get_observation():
        return self.airi.getDepthImage()

    @property   
    def _n_actions(self):
        return len(self._action_set)
