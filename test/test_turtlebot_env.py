import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marl.envs.turtlebot_nav_env import TurtleBotNavEnv
import numpy as np

env = TurtleBotNavEnv()
obs, _ = env.reset()

done = False
total_reward = 0

while not done:
    action = np.array([0.5, 0.0])  # forward
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()

print("✅ Test run complete. Total reward:", total_reward)
