from panda_gym.envs import PandaReachCamEnv
from sb3_contrib import TQC
import time
import gymnasium as gym

env = gym.make('PandaReachCam-v3', render_mode="human") # rgb_array

# HER must be loaded with the env
model = TQC.load("tqc_her_panda", env=env)

obs, _ = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
      obs, info = env.reset()
    time.sleep(0.1)