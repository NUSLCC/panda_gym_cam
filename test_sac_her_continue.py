from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
import gymnasium as gym
from panda_gym.utils import CustomCombinedExtractor
import torch

if __name__=="__main__":
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./logs/",
    name_prefix="rgb_moving_obj",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )

    # env = gym.make('PandaReachCam-v3', render_mode="human") #, control_type="joints") # rgb_array
    env_id = "PandaReachCamJoints-v3"
    num_cpu = 32
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    
    model = SAC.load("models/sac_rgb_sine_moving_lstm", env = env, device="cuda:1")
    model.load_replay_buffer("logs/rgb_moving_obj_replay_buffer_700000_steps", truncate_last_traj=False)
    print(f'Replay buffer size is {model.replay_buffer.size()}')
    tmp_path = "./tmp/sac_dual_"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback, reset_num_timesteps=False, progress_bar=True)
    
    model.save("sac_rgb_sine_moving_lstm")
