import gym
from gym import spaces
import numpy as np
from dynamics import MassDamperSystem
from cost import compute_cost

class MassDamperEnv(gym.Env):
    def __init__(self, dt=0.01, max_steps=1000):
        super(MassDamperEnv, self).__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.target = 1.0
        self.system = MassDamperSystem(dt=self.dt)

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-self.system.u_max]),
            high=np.array([self.system.u_max]),
            dtype=np.float32
        )

    def reset(self):
        self.system.reset()
        self.step_count = 0
        obs = np.array([self.target - self.system.x, self.system.v], dtype=np.float32)
        return obs

    def step(self, action):
        action = np.clip(action, -self.system.u_max, self.system.u_max)
        x, v = self.system.step(action)
        self.step_count += 1

        obs = np.array([self.target - x, v], dtype=np.float32)
        reward = -compute_cost({"positions":[x], "controls":[action]}, target=self.target)
        done = (self.step_count >= self.max_steps) or (abs(self.target - x) < 0.001)
        return obs, reward, done, {}
