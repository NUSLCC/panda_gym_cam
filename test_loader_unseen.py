from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer, SAC
from sb3_contrib import TQC
import time
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('PandaPickandPlaceCam-v3', render_mode="human", control_type="ee") # rgb_array
print(env.action_space)
# HER must be loaded with the env

#model = TQC.load("philip4_tqc_deep_push_joints.zip", env=env)
model = TQC.load("philip4_tqc_deep_pickandplace.zip", env=env)
#model = TQC.load("philip4_tqc_deep_movingreach_joints.zip", env=env)
#model = TQC.load("philip4_tqc_deep_reach_obsonly_joints.zip", env=env)

obs, _ = env.reset()
# print(obs['observation'])

# plt.imshow(obs["observation"])
# plt.title('First reset')
# plt.show()
success_number = 0
failure_number = 0
episode_number = 0
new_episode_flag = False

for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)

    # plt.imshow(obs["observation"])
    # plt.title('Subsequent resets')
    # plt.show()

   # print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if new_episode_flag == True:
      print(episode_number, success_number, failure_number)
      new_episode_flag = False
    if episode_number == 100:
      print(success_number)
      print(failure_number)
      print(success_number / (success_number + failure_number))
      break
    if terminated:
      print("terminated")
      if info['is_success'] == True:
         success_number += 1
      if info['is_failure'] == True:
         failure_number += 1
      episode_number += 1
      new_episode_flag = True
      obs, info = env.reset()
   #   take from info!
    elif truncated:
      print("truncated")
      failure_number += 1
      episode_number += 1
      new_episode_flag = True
      obs, info = env.reset()