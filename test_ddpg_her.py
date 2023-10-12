from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.logger import configure
from datetime import datetime

import gymnasium as gym

env = gym.make('PandaReachCam-v3', render_mode="human") # rgb_array

model = DDPG(policy="MultiInputPolicy",env=env, batch_size=2048, gamma=0.95, learning_rate=1e-4,
            train_freq=64, gradient_steps=64, tau=0.05, replay_buffer_class=HerReplayBuffer, verbose=1, 
            buffer_size=300000, tensorboard_log="./tmp", learning_starts=100, 
            # Parameters for HER
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"))

tmp_path = "./tmp/"+datetime.now().strftime('ddpg_her_modified_%H_%M_%d')
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

model.learn(total_timesteps=50_000, progress_bar=True)
model.save("ddpg_her_panda")