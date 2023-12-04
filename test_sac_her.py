from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from datetime import datetime
import gymnasium as gym
from attention.modules import CustomFeatureExtractor

if __name__=="__main__":
    # env = gym.make('PandaReachCam-v3', render_mode="human") #, control_type="joints") # rgb_array
    env_id = "PandaReachCamJoints-v3"
    num_cpu = 16
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = SAC(policy="CnnPolicy",env=env, batch_size=128, gamma=0.95, learning_rate=1e-4, verbose=1, 
                train_freq=64, gradient_steps=64, tau=0.05, tensorboard_log="./tmp", learning_starts=1000,
                buffer_size=600000, replay_buffer_class=ReplayBuffer,
                policy_kwargs=dict(
                    features_extractor_class=CustomFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=76608), # output size of custom feature extractor
                    net_arch=[512, 512, 512], 
                    n_critics=2)
    )

    tmp_path = "./tmp/"+datetime.now().strftime('sac_dual_attention_table_%H_%M_%d')
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=700_000, progress_bar=True)
    model.save("sac_dual_attention_panda")
