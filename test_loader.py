from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer
from sb3_contrib import TQC
import time
import gymnasium as gym

env = gym.make('PandaReachCam-v3', render_mode="human") # rgb_array

# HER must be loaded with the env
model = DDPG.load("ddpg_her_panda", env=env)

obs, _ = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
      print("terminated")
      print(i)
      obs, info = env.reset()
    elif truncated:
      print("truncated")
      print(info)
      obs, info = env.reset()