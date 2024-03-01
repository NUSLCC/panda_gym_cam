from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import SAC, HerReplayBuffer
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import configure_logger
from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.torch_layers import CombinedExtractor
from panda_gym.utils import CustomCombinedExtractor, CustomConvNextExtractor, DeepConvNetCombinedExtractor, DeepConvNetCombinedExtractorObservationOnly
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__=="__main__":
    # env = gym.make('PandaReachCam-v3', render_mode="human") #, control_type="joints") # rgb_array
    env_id = "PandaMovingReachCamDense-v3"
    num_cpu = 16
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    print(f'Action space: {env.action_space}')
    
    # Save a checkpoint every 100000 steps
    checkpoint_callback = CheckpointCallback(
    save_freq=max(50000 // num_cpu, 1),
    save_path="./logs/",
    name_prefix=datetime.now().strftime('philip4_tqc_deep_movingreach_obsonly_%H_%M_%d'),
    save_replay_buffer=True,
    save_vecnormalize=True,
    )
    
    # Training from scratch:
    model = TQC(policy="MultiInputPolicy",env=env, batch_size=256, gamma=0.95, learning_rate=1e-4, verbose=1, 
                train_freq=64, gradient_steps=64, tau=0.05, tensorboard_log="./tmp", learning_starts=1500,
                buffer_size=20000, replay_buffer_class=None, device="cuda:0",
                # Parameters for HER
              #  replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
                # Parameters for TQC
                top_quantiles_to_drop_per_net=2, 
                policy_kwargs=dict(
                    features_extractor_class=DeepConvNetCombinedExtractorObservationOnly,
                    net_arch=[512, 512, 512], 
                    n_critics=2, 
                    n_quantiles=25)
                )
    
    tmp_path = "./tmp/"+datetime.now().strftime('tqc_dual_philip4_movingreach_obsonly_%H_%M_%d')
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=2_500_000, callback=checkpoint_callback, progress_bar=True)

    # Loading model:

    # model = TQC.load("logs/philip4_tqc_deep_movingreach_joints_850000_steps", env = env)
    # model.load_replay_buffer("logs/philip4_tqc_deep_movingreach_joints_replay_buffer_850000_steps", truncate_last_traj=False)
    # print(f'Replay buffer size is {model.replay_buffer.size()}')
    # tmp_path = "./tmp/"+"tqc_dual_philip4_movingreach_joints_22_23_21"
    # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # model.set_logger(new_logger)
    # model.learn(total_timesteps=2_500_000, callback=checkpoint_callback, reset_num_timesteps=False, progress_bar=True)
    
    model.save("tqc_her_philip4_push")
