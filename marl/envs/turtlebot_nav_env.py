import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TurtleBotNavEnv(gym.Env):
    def __init__(self, render_mode=False):
        super().__init__()
        self.render_mode = render_mode

        # World size and goal location
        self.map_size = 10.0  # 10x10 grid
        self.goal = np.array([8.0, 8.0])

        # Observation: [x, y, theta] + fake LiDAR (8 rays)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # Action: [linear velocity, angular velocity]
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([1.0, 1.0])
        self.agent_theta = 0.0
        self.timestep = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        v, omega = action
        dt = 0.1  # seconds

        # Update orientation and position
        self.agent_theta += omega * dt
        dx = v * np.cos(self.agent_theta) * dt
        dy = v * np.sin(self.agent_theta) * dt
        self.agent_pos += np.array([dx, dy])
        self.timestep += 1

        # Compute reward
        dist_to_goal = np.linalg.norm(self.goal - self.agent_pos)
        done = dist_to_goal < 0.5 or self.timestep > 200
        reward = -dist_to_goal

        if dist_to_goal < 0.5:
            reward += 10.0
        elif self._is_collision():
            reward -= 10.0
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        lidar = self._simulate_lidar()
        return np.concatenate([self.agent_pos, [self.agent_theta], lidar])

    def _simulate_lidar(self, num_rays=8, max_range=3.0):
        # Fake lidar for now: all distances = max_range
        return np.ones(num_rays) * max_range

    def _is_collision(self):
        # Simple boundary check
        return np.any(self.agent_pos < 0) or np.any(self.agent_pos > self.map_size)

    def render(self):
        if not self.render_mode:
            return
        print(f"Agent at {self.agent_pos}, heading {self.agent_theta:.2f}")
