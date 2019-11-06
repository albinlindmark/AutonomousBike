import gym
import numpy as np

class BikeLQREnv(gym.Env):
    """
    Description:

    Observation: 
        
    Actions:

    Reward:


    Starting State:


    Episode Termination:

    """

    def __init__(self):
        self.state = None
        self.reward = None
        # The ss matrices were created with Ts = 0.04 s and v = 5 m/s
        self.A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32)
        self.B = np.array([[0.210654076615514], [1.042632885238932]], dtype=np.float32)
        self.Q = np.array([[10, 0], [0, 0]], dtype=np.float32)
        self.R = 1


    def step(self, action):
        self.reward += -(self.state.transpose() @ self.Q @ self.state + self.R*action**2)
        self.state = self.A @ self.state + self.B @ action
        # If roll angle larger than 30 degrees, then terminate:
        if self.state[0] > 30*np.pi/180:
            self.done = True
        
        return self.state, self.reward, self.done, _
        

    def reset(self):
        # Start with random roll angle, zero roll angle rate and  zero steering angle.
        # I.e. phi = uniform(-15, 15), dphi = 0 and delta = 0.
        # Note that x1 = phi and x2 =  b*h*dphi - a*v*delta.
        phi_0 = np.pi/180 * np.random.uniform(-15, 15)
        self.state = np.array([[phi_0], [0]], dtype=np.float32)
        self.reward = 0
        self.done = False
        return self.state
        
    def render(self, mode='human'):
        print('Reward:', self.reward)
