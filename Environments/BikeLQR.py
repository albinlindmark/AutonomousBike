import gym
import numpy as np
from gym import spaces
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
        self.v_delta = 0

        high = np.array([np.pi/2, np.finfo(np.float32).max, 15])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-np.pi/2, np.pi/2, shape=(1,), dtype=np.float32)
        
        # The state space matrices were created with Ts = 0.04 s
        self.A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32) # is independent of the speed
        #self.B = np.array([0.210654076615514, 1.042632885238932], dtype=np.float32)

        # B depends on the velocity, so it will vary for each step, so the ss equation becomes x_k+1 = A x_k + B_k delta, where B_k depends on the speed v.
        # We have that B_k = B_k_wo_v * [v; v^2] (element-wise operation) where Bk_wo_v is the same as inv(Ac) (A - I) Bc(1) where Bc(1) is Bc for v=1.
        self.B_k_wo_v = np.array([0.036491277663333, 0.047719661231268], dtype=np.float32)
   
        self.Q = np.array([[10, 0], [0, 0]])
        self.R = 1
        self.cost_scale_factor = 1e5


    def step(self, action):
        # Make sure that the action (delta) is in interval [-pi/2, pi/2]:
        action = np.clip(action, -np.pi/2, np.pi/2)[0]
        self.action = action.copy() # For the rendering

        state_wo_v = self.state[0:2]
        v = self.state[2]

        cost = self.cost_scale_factor * (state_wo_v.transpose() @ self.Q @ state_wo_v + action**2*self.R)
        #cost = self.cost_scale_factor * (self.state.transpose() @ self.Q @ self.state + action**2*self.R)**(1/2)
        log_cost = np.log(cost + np.finfo(np.float32).eps)
        self.reward = -log_cost

        B_k = self.B_k_wo_v * np.array([v, v**2], dtype=np.float32)
        state_wo_v = self.A @ state_wo_v + B_k * action # action is scalar
        if self.changing_speed:
            v = v + self.v_delta
            v = np.max([0.5, v])
            v = np.min([10, va])
        else:
            v = np.max([0.5, v])
            v = np.min([10, v])
        self.state = np.array([state_wo_v[0], state_wo_v[1], v], dtype=np.float32)

        # If roll angle larger than 30 degrees, then terminate:
        if abs(self.state[0]) > 30*np.pi/180:
            self.done = True
            worst_case_phi = np.pi/180*15
            worst_case_state = np.array([worst_case_phi, 0], dtype=np.float32)
            worst_case_action = np.pi/2
            cost = self.cost_scale_factor * (worst_case_state.transpose() @ self.Q @ worst_case_state + worst_case_action**2*self.R)
            self.reward = -100 * np.log(cost + np.finfo(np.float32).eps)
        
        return self.state, self.reward, self.done, {}
        

    def reset(self, init_state=None, changing_speed = False):
        # Start with random roll angle, zero roll angle rate and  zero steering angle.
        # I.e. phi(0) = uniform(min, max), dphi(0) = 0 and delta(0) = 0.
        # Note that x1 = phi and x2 =  b*h*dphi - a*v*delta.
        if type(init_state) == np.ndarray:
            self.state = init_state
        else:
            phi_0 = np.pi/180*np.random.uniform(-5, 5)
            #v_0 = np.random.choice([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            #v_0 = np.random.uniform(0.5, 10)
            v_0 = 5
            self.state = np.array([phi_0, 0, v_0], dtype=np.float32)        
        
        self.changing_speed = changing_speed
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
