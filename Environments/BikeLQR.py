import gym
import numpy as np

class BikeLQREnv(gym.Env):
    """
    Description:

    Observation: 
        
    Actions:
        Type: Discrete(4)

    Reward:


    Starting State:


    Episode Termination:

    """

    def __init__(self):
        self.state = None
        self.Ts = 0.04
        self.A = np.array([[1.0151, 0.0707], [0.4318, 1.0151]], dtype=np.float32)
        self.B = np.array([[0.0018], [0.0005]], dtype=np.float32)
        self.Q = np.array([[10, 0], [0, 0]], dtype=np.float32)
        self.R = 1
        self.reward = 0


    def step(self, action):
        state = self.state
        state = self.A @ state + self.B @ action
        self.reward += -(state.transpose() @ self.Q @ state + self.R**2*action)
        
        
        

    def reset(self):
        1
        
    def render(self, mode='human'):
        print('hej')
