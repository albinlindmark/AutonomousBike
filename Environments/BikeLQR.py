import gym
import numpy as np
from gym import spaces

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
        high = np.array([np.pi/2, np.finfo(np.float32).max, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-np.pi/2, np.pi/2, shape=(1,), dtype=np.float32)
        
        # The state space matrices were created with Ts = 0.04 s and v = 5 m/s
        self.A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32)
        self.B = np.array([0.210654076615514, 1.042632885238932], dtype=np.float32)
        
        self.Q = np.array([[10, 0], [0, 0]])
        self.R = 1

        # The state space matrices with dynamic velocity
        self.A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32)
        self.Ainv = np.array([[0, 0.093092967291786], [0.568852500000000, 0]], dtype=np.float32)
        self.Bc_without_v = np.array([0.872633942893808, 1], dtype=np.float32)

        

    def step(self, action):

        if self.const_vel == False: 
            self.v = np.max([self.v + np.random.uniform(-0.01, 0.01), 0.5])

        # Define the B matrix for the current velocity
        Bc = self.Bc_without_v*(np.array([self.v, self.v**2], dtype = np.float32))
        B = self.Ainv.dot(self.A - np.eye(2)).dot(Bc)
        

        # Make sure that the action (delta) is in interval [-pi/2, pi/2]:
        action = np.clip(action, -np.pi/2, np.pi/2)[0]
        cost = (self.state[0:2].transpose() @ self.Q @ self.state[0:2] + action**2*self.R)
        self.reward = -np.log(1e5*cost)
        
        states_wo_v = self.A @ self.state[0:2] + B * action # action is scalar
        self.state = np.array([states_wo_v[0], states_wo_v[1], self.v], dtype = np.float32)

        # If roll angle larger than 30 degrees, then terminate:
        if abs(self.state[0]) > 30*np.pi/180:
            self.done = True
            worst_case_phi = np.pi/180*15
            worst_case_state = np.array([worst_case_phi, 0], dtype=np.float32)
            worst_case_action = np.pi/2
            cost = (worst_case_state.transpose() @ self.Q @ worst_case_state + worst_case_action**2*self.R)
            self.reward = -100 * np.log(1e5*cost)
        
        # If roll angle around 1 degrees, increase the reward
        #if abs(self.state[0]) < 1*np.pi/180:
        #    self.reward += 0.1 - abs(0.1*self.state[0])
        
        return self.state, self.reward, self.done, {}
        

    def reset(self, init_state=None, const_vel=False):
        # Start with random roll angle, zero roll angle rate and  zero steering angle.
        # I.e. phi(0) = uniform(-15, 15), dphi(0) = 0 and delta(0) = 0.
        # Note that x1 = phi and x2 =  b*h*dphi - a*v*delta.
        #phi_0 = np.pi/180 * np.random.choice([np.random.uniform(-15, -1), np.random.uniform(1, 15)])
        
        self.const_vel = const_vel

        if type(init_state) == np.ndarray:
            phi_0 = np.pi/180*(init_state[0])
            self.v = init_state[2] 

        else:
            phi_0 = np.pi/180*(np.random.uniform(-0.5, 0.5))
            self.v = np.random.uniform(0.5, 5)
            
        self.state = np.array([phi_0, 0, self.v], dtype=np.float32)
        self.reward = 0
        self.done = False
        #phi_0 = np.pi/180*15
        
        return self.state
        
    def render(self, mode='human'):

        print('Reward:', self.reward)
