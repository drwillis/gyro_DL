# Modified from Symplectic-ODENet's Pendulum environment
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/myenv/pendulum.py

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class GyroEnvV1(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0, ori_rep = 'angle', friction=False):
        # constants
        self.m=0.001  # mass of vibrating mass
        self.Jzz=.1  # moment of inertia of vibrating mass about rotational axis
        self.ky=10  # spring constant in vibrating direction
        self.by=.1  # viscous friction coefficient in vibrating direction
        self.kx=10  # spring constant in sensing direction
        self.bx=1  # viscous friction coefficient in sensing direction
        self.y0=.1  # amplitude of excitation signal
        self.wy=500  # frequency of excitation signal
        self.cy=0.00001  # Coulombic Friction in excitation direction
        self.cx=0.00001  # Coulombic Friction in sensing direction
        self.params=np.array([self.m,self.Jzz,self.ky,self.ky,self.kx,self.bx,self.y0,self.wy,self.cy,self.cx]) # Group constants into a vector to send to ODE function

        self.max_speed=100.
        self.max_torque=5.
        # self.dt=.05
        self.dt= 0.0001
        self.g = g
        self.viewer = None

        # high = np.array([1., 1., self.max_speed])
        high = np.array([1., 1., self.max_speed])
        # self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.double)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.double)

        # Orientation representations: 'angle', 'rotmat'
        self.ori_rep = ori_rep
        self.friction = friction
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def dynamics(self, t, b, tau):
        # y[0] is angle, y[1] is angular velocity
        # f[0] = angular velocity
        # f[1] = angular acceleration
        bdot = np.zeros_like(b)

        #state variable assignemnts
        x = b[0]
        y = b[1]
        theta = b[2]
        dotx = b[3]
        doty = b[4]
        dottheta = b[5]

        # Configuration coordinate vector
        gamma = np.array([x, y, theta]).T
        # System Speeds
        dotgamma = np.array([dotx, doty, dottheta]).T

        # System Mass matrix
        H = np.array([[self.m, 0, -self.m*y],
                    [0, self.m, self.m*x],
                    [-self.m*y, self.m*x, self.m*(x**2+y**2)+self.Jzz]])

        # Coriolis and Centripetal force vector
        d=np.array([-2*self.m*doty*dottheta-self.m*x*dottheta**2,
                    2*self.m*dotx*dottheta-self.m*y*dottheta**2,
                    2*self.m*x*dotx*dottheta+2*self.m*y*doty*dottheta]).T

        # Spring constant matrix
        K = np.diag([self.kx, self.ky, 0])

        # Viscous damping constant matrix
        B = np.diag([self.bx, self.by, 0])

        # Coulombic Friction constant matrix
        C = np.diag([self.cx, self.cy, 0])

        F = np.array([0,  # no external force applied in sensing directin (x)
                      self.y0 * np.cos(self.wy * t),  # excitation signal applied in vibrating direction (y)
                      tau]).T  # torque applied to the gyroscope (so that it rotates)

        # Rate of change of state vector
        bdot[0:3] = b[3:6]
        bdot[3:6] = np.matmul(np.linalg.inv(H),
                              (F - d - np.matmul(K, gamma) - np.matmul(B, dotgamma) - np.matmul(C, np.sign(dotgamma))))
        return bdot

    def get_state(self):
        return self._get_obs()

    def step(self,u):
        theta = self.state[2] # theta := theta
        thetadot = self.state[5]

        dt = self.dt
        tau = u[0]
        # u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u[0] # for rendering
        costs = angle_normalize(theta)**2 + .1*thetadot**2 + .001*(tau**2)

        ivp = solve_ivp(fun=lambda t, y:self.dynamics(t, y, tau), t_span=[0, self.dt], y0=self.state)
        self.state = ivp.y[:, -1]

        return self._get_obs(), -costs, False, {}

    def reset(self, ori_rep = 'angle'):
        high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        self.state = np.array([0,0,0,0,0,0]).T
        self.last_u = None
        # Orientation representations: 'angle', 'rotmat'
        self.ori_rep = ori_rep
        return self._get_obs()

    def _get_obs(self):
        x=self.state[0]
        y=self.state[1]
        theta = self.state[2] # th := theta
        xdot=self.state[3]
        ydot=self.state[4]
        thetadot = self.state[5]
        w_vec = np.array([0.0, 0.0, thetadot])
        if self.ori_rep == 'angle':
            ret = np.array([np.cos(theta), np.sin(theta), x**2 + y**2, xdot**2 + ydot**2, thetadot])
        if self.ori_rep == 'rotmat':
            x_vec = np.array([x, y, 0])
            R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                          [np.sin(theta),  np.cos(theta), 0.0],
                          [0.0,            0.0,           1.0]])
            v_vec = np.array([xdot, ydot, 0])
            ret = np.concatenate((x_vec, R.flatten(), v_vec, w_vec))
        return ret

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] - np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None