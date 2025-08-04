import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create a simple environment
env = make_vec_env("CartPole-v1", n_envs=1)

# Initialize PPO
model = PPO("MlpPolicy", env, verbose=1)

# Train for a few steps
model.learn(total_timesteps=1000)

print("✅ Training complete.")
