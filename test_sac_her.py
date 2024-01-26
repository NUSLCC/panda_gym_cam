from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.torch_layers import CombinedExtractor
from panda_gym.utils import CustomCombinedExtractor, CustomConvNextExtractor
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__=="__main__":
    # env = gym.make('PandaReachCam-v3', render_mode="human") #, control_type="joints") # rgb_array
    env_id = "PandaPickandPlaceCam-v3"
    num_cpu = 16
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    print(f'Action space: {env.action_space}')
    
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
    save_freq=max(50000 // num_cpu, 1),
    save_path="./logs/",
    name_prefix="philip4_pick_and_place",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )
    
    model = SAC(policy="MultiInputPolicy",env=env, batch_size=2048, gamma=0.95, learning_rate=1e-4, verbose=1, 
                train_freq=64, gradient_steps=64, tau=0.05, tensorboard_log="./tmp", learning_starts=1500,
                buffer_size=20000, replay_buffer_class=HerReplayBuffer, device="cuda:0",
                # Parameters for HER
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
                # Parameters for SAC
                policy_kwargs=dict(
                    features_extractor_class=CustomCombinedExtractor,
                    features_extractor_kwargs=dict(cnn_output_dim = 512),
                    net_arch=[512, 512, 512], 
                    n_critics=2)
                )
    
    # model = SAC.load("logs/philip4_pick_and_place_300000_steps", env = env) #760000+300000 steps
    # model.load_replay_buffer("logs/philip4_pick_and_place_replay_buffer_300000_steps")
    # print(f'Replay buffer size is {model.replay_buffer.size()}')
    
    # print(model.policy)
    
    tmp_path = "./tmp/"+datetime.now().strftime('sac_dual_philip4_pickandplace_%H_%M_%d')
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=1_500_000, callback=checkpoint_callback, progress_bar=True)
    model.save("sac_her_philip4_pickandplace")
