from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer

env = PandaReachCamEnv(render_mode="human") # rgb_array

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1, buffer_size=300000, tensorboard_log="./tmp")

model.learn(total_timesteps=20000, progress_bar=True)