from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer, SAC
from sb3_contrib import TQC
import time
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('PandaMovingReachCamJoints-v3', render_mode="human", control_type="joints") # rgb_array
print(env.action_space)
# HER must be loaded with the env
model = TQC.load("philip4_tqc_deep_movingreach_joints.zip", env=env)

obs, _ = env.reset()
# print(obs['observation'])

# plt.imshow(obs["observation"])
# plt.title('First reset')
# plt.show()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)

    # plt.imshow(obs["observation"])
    # plt.title('Subsequent resets')
    # plt.show()

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