from panda_gym.envs import PandaReachCamEnv
from panda_gym.envs import PandaPickandPlaceCamEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from datetime import datetime
import gymnasium as gym

if __name__=="__main__":
    env_id = "PandaReachCamJoints-v3"
    num_cpu = 16
    env = make_vec_env('PandaReachCam-v3', n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = DDPG(policy="MultiInputPolicy",env=env, batch_size=2048, gamma=0.95, learning_rate=1e-4, verbose=1, 
                train_freq=64, gradient_steps=64, tau=0.05, tensorboard_log="./tmp", learning_starts=1000,
                buffer_size=600000, replay_buffer_class=DictReplayBuffer)

    tmp_path = "./tmp/"+datetime.now().strftime('ddpg_dual_table_only_vision_%H_%M_%d')
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=700_000, progress_bar=True)
    model.save("ddpg_dual_only_vision_panda")