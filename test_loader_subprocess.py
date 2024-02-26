from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer, SAC
from sb3_contrib import TQC
import time
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('PandaReachCamJoints-v3', render_mode="human", control_type="joints") # rgb_array
print(env.action_space)
# HER must be loaded with the env
model = TQC.load("logs/philip4_tqc_deep_reach_obsonly_joints_21_16_24_750000_steps_works", env=env)

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
    if terminated or truncated:
      break