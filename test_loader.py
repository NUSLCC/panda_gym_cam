from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3 import SAC, HerReplayBuffer
from sb3_contrib import TQC
import time
import gymnasium as gym

env = gym.make('PandaReachCam-v3', render_mode="human", control_type="joints") # rgb_array
print(env.action_space)
print(env.observation_space)
# # HER must be loaded with the env
model = SAC.load("sac_her_panda_mask_only", env=env)

obs, _ = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print(action)
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