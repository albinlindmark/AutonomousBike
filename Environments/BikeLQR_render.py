import gym
import numpy as np
from gym import spaces
from torch.distributions import Normal
import matplotlib.pyplot as plt
import time
import pyglet
from pyglet import gl


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
        self.viewer = None
        self.action = None
        self.length = 0.5
        
        high = np.array([np.pi/2, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-np.pi/2, np.pi/2, shape=(1,), dtype=np.float32)
        
        # The state space matrices were created with Ts = 0.04 s and v = 5 m/s
        self.A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32)
        self.B = np.array([0.210654076615514, 1.042632885238932], dtype=np.float32)
        
        self.Q = np.array([[10, 0], [0, 0]])
        self.R = 1


    def step(self, action):
        # Make sure that the action (delta) is in interval [-pi/2, pi/2]:
        self.action = np.clip(action, -np.pi/2, np.pi/2)
        cost = (self.state.transpose() @ self.Q @ self.state + self.action**2*self.R)
        self.reward = -cost
        
        self.state = self.A @ self.state + self.B * self.action # action is scalar

        # If roll angle larger than 30 degrees, then terminate:
        if abs(self.state[0]) > 30*np.pi/180:
            self.done = True
            worst_case_phi = np.pi/180*15
            worst_case_state = np.array([worst_case_phi, 0], dtype=np.float32)
            worst_case_action = np.pi/2
            cost = (worst_case_state.transpose() @ self.Q @ worst_case_state + worst_case_action**2*self.R)
            self.reward = -100 * cost
        
        return self.state, self.reward, self.done, {}
        

    def reset(self):
        # Start with random roll angle, zero roll angle rate and  zero steering angle.
        # I.e. phi(0) = uniform(-15, 15), dphi(0) = 0 and delta(0) = 0.
        # Note that x1 = phi and x2 =  b*h*dphi - a*v*delta.
        #phi_0 = np.pi/180 * np.random.choice([np.random.uniform(-15, -1), np.random.uniform(1, 15)]) 

        #phi_0 = np.pi/180*(np.random.uniform(-15, 15)) 
        phi_0 = np.pi/6
        self.state = np.array([phi_0, 0], dtype=np.float32)
        self.reward = 0
        self.done = False
        return self.state
        
    def render(self, mode='human'):
        screen_width  = 600
        screen_height = 400

        axleoffset = 30.0/4.0
        world_width = 2.4*2
        scale = screen_width/world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Text
            #self.score_label = pyglet.text.Label("Nu går det fort!!", font_size=12,
            #    x=0, y=0, anchor_x='center', anchor_y='center',
            #    color=(255,255,255,255))
            #self.score_label.draw()

            # Cykeln
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(500, 100))
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(pole)
            self._pole_geom = pole

            # Styret
            styre1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            styre2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            styre1.set_color(.8,.6,.4)
            styre2.set_color(.8,.6,.4)
            self.styre1trans = rendering.Transform(translation=(200, 100), rotation=3.14/2)
            self.styre2trans = rendering.Transform(translation=(200, 100), rotation=-3.14/2)
            styre1.add_attr(self.styre1trans)
            styre2.add_attr(self.styre2trans)
            self.viewer.add_geom(styre1)
            self.viewer.add_geom(styre2)
            self._styre1_geom = styre1
            self._styre2_geom = styre2
        
        if self.state is None: return None

        pole = self._pole_geom
        styre1 = self._styre1_geom
        styre2 = self._styre2_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]
        styre1.v = [(l,b), (l,t), (r,t), (r,b)]
        styre2.v = [(l,b), (l,t), (r,t), (r,b)]
        
        x = self.state
        self.poletrans.set_rotation(-x[0])

        #self.score_label.text = "Nu går det fort!!"
        #self.score_label.text = "%04i" % self.reward
        #self.score_label.draw()

        if self.action is not None:
            self.styre1trans.set_rotation(self.action+np.pi/2)
            self.styre2trans.set_rotation(self.action-np.pi/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    plot_curves = False
    time_steps = 25
    env = BikeLQREnv()
    env.reset()
    K = np.array([2.153841222845478, 0.298295153292258], dtype=np.float32)
    next_state = env.state
    states = []
    actions = []
    states.append(next_state[0])
    
    for t in range(time_steps):
        env.render()
        action = -K@next_state
        next_state, reward, done, _ = env.step(action)
        states.append(next_state[0])
        actions.append(action)
        time.sleep(0.15)
    
    env.close()

    if plot_curves:
        fig, axes = plt.subplots(figsize=(14,8))
        axes.plot(range(time_steps), states[:-1])
        axes.plot(range(time_steps), actions)
        plt.legend(['x','u'])
        plt.show()



        
