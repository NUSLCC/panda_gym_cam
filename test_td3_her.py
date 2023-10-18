from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.logger import configure
from datetime import datetime

import gymnasium as gym

env = gym.make('PandaReachCam-v3', render_mode="human") # rgb_array

model = TD3(policy="MultiInputPolicy",env=env, batch_size=2048, gamma=0.95, learning_rate=1e-4, verbose=1, 
            train_freq=64, gradient_steps=64, tau=0.05, tensorboard_log="./tmp", learning_starts=100,
            buffer_size=300000, replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
            # Parameters for TD3
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2))


tmp_path = "./tmp/"+datetime.now().strftime('td3_dual_table_%H_%M_%d')
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

model.learn(total_timesteps=30_000, progress_bar=True)
model.save("td3_her_dual")