from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, SAC, HerReplayBuffer
from sb3_contrib import TQC
import time
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('PandaReachCam-v3', render_mode="human", control_type="joints") # rgb_array
# print(env.action_space)
# print(env.observation_space)
# HER must be loaded with the env
model = SAC.load("sac_cross_attention_panda", env=env)

obs, _ = env.reset()

plt.imshow(obs["observation"])
plt.title('First reset')
plt.show()

# policy = model.policy
# print(policy)
# features_extractor = policy.actor.features_extractor
# print(f'Features extractor: {features_extractor}')

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    # print(action)
    
    obs, _ = env.reset()

    plt.imshow(obs["observation"])
    plt.title('Subsequent resets')
    plt.show()
    #obs, reward, terminated, truncated, info = env.step(action)
    env.render()
