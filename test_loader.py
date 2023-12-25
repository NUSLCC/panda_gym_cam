from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer, SAC
from sb3_contrib import TQC
import time
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('PandaReachCam-v3', render_mode="human", control_type="joints") # rgb_array
#print(env.action_space)
# HER must be loaded with the env
model = SAC.load("sac_her_panda", env=env)

obs, _ = env.reset()
# print(obs['observation'])


for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, _ = env.reset()
    x = obs["observation"][:160,:,:] - obs["observation"][160:,:,:]

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(obs["observation"].reshape(90,320,3))
    ax[1].imshow(x.reshape(90,160,3))
    plt.show()

    # print(action)
    # obs, reward, terminated, truncated, info = env.step(action)
    # env.render()
    # if terminated:
    #   print("terminated")
    #   print(i)
    #   obs, info = env.reset()
    # elif truncated:
    #   print("truncated")
    #   print(info)
    #   obs, info = env.reset()