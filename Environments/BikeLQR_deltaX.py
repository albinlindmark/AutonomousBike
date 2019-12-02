import gym
import numpy as np
from gym import spaces

class BikeLQREnv_deltaX(gym.Env):
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

        self.x = 0
        self.y = 0
        self.xref = 0
        self.deltax = self.x-self.xref 
        
        high = np.array([np.pi/2, np.finfo(np.float32).max, np.finfo(np.float32).max,np.finfo(np.float32).max])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-np.pi/2, np.pi/2, shape=(1,), dtype=np.float32)
        
        # The state space matrices were created with Ts = 0.04 s and v = 5 m/s
        self.A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32)
        self.B = np.array([0.210654076615514, 1.042632885238932], dtype=np.float32)
        
        self.Q = np.array([[10, 0], [0, 0]])
        self.R = 1
        self.S = 2

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
        self.action = action
        cost = (self.state[0:2].transpose() @ self.Q @ self.state[0:2] + action**2*self.R + self.deltax**2*self.S)
        self.reward = -np.log(1e5*cost)

        # Update states of bike model
        states_wo_v = self.A @ self.state[0:2] + B * action # action is scalar

        # Update states of bike position
        self.x = self.x + self.v*np.sin(self.action)
        self.y = self.y + self.v*np.cos(self.action)
        self.deltax = self.x-self.xref

        self.state = np.array([states_wo_v[0], states_wo_v[1], self.v, self.deltax], dtype = np.float32)
        

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
            self.v = np.random.uniform(5, 5)
            

        self.x = 0
        self.y = 0
        self.deltax = 0
        self.state = np.array([phi_0, 0, self.v,self.deltax], dtype=np.float32)
        self.reward = 0
        self.done = False
        
        
        #phi_0 = np.pi/180*15
        
        return self.state
        
    def render(self, mode='human'):
        screen_width  = 600
        screen_height = 400

        axleoffset = 30.0/4.0
        world_width = 2.4*2
        scale = screen_width/world_width
        polewidth = 10.0
        polelen = scale * (2 * 0.5)

        carty = 100
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Text
            #self.score_label = pyglet.text.Label("Nu går det fort!!", font_size=12,
            #    x=0, y=0, anchor_x='center', anchor_y='center',
            #    color=(255,255,255,255))
            #self.score_label.draw()

            # Gamla Cykeln
            '''
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(500, 100))
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(pole)
            self._pole_geom = pole
            '''

            # Nya cykeln
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
            self._pole_geom = pole
            
            # Styret
            styre1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            styre2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            styre1.set_color(.8,.6,.4)
            styre2.set_color(.8,.6,.4)
            self.styre1trans = rendering.Transform(translation=(200, 250), rotation=3.14/2)
            self.styre2trans = rendering.Transform(translation=(200, 250), rotation=-3.14/2)
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
        cartx = x[3]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
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
