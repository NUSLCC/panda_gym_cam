from panda_gym.envs import PandaReachCamEnv
from panda_gym.envs import PandaReachCamObstacleEnv
from panda_gym.envs import PandaPickandPlaceCamEnv
from panda_gym.envs import PandaLiftCamEnv
from panda_gym.envs import PandaPushCamEnv
from panda_gym.envs import PandaSlideCamEnv
# import numpy as np
# np.set_printoptions(threshold = np.inf)

env = PandaSlideCamEnv(render_mode="human") # rgb_array

observation, info = env.reset()

print(observation["observation"].shape)

for _ in range(500):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if _ % 100 == 0:
        observation, info = env.reset()