from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.logger import configure
from sb3_contrib import TQC
from datetime import datetime

import gymnasium as gym

env = gym.make('PandaReachCam-v3', render_mode="human") # rgb_array

model = TQC(policy="MultiInputPolicy",env=env, batch_size=2048, gamma=0.95, learning_rate=1e-4, verbose=1, 
            train_freq=64, gradient_steps=64, tau=0.05, tensorboard_log="./tmp", learning_starts=100,
            buffer_size=300000, replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
            # Parameters for TQC
            top_quantiles_to_drop_per_net=2, 
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2, n_quantiles=25))


tmp_path = "./tmp/"+datetime.now().strftime('tqc_her_%H_%M_%d')
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

model.learn(total_timesteps=50_000, progress_bar=True)
model.save("tqc_her_panda")
