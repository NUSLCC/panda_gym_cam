import matplotlib.pyplot as plt
import sys
sys.path.append("/home/lcc/GitRepo/panda-gym")
sys.path.append("/Users/chenchen/GitRepo/panda_gym_cam")
from panda_gym.envs import PandaReachCamEnv

env = PandaReachCamEnv(render_mode="human") # rgb_array

observation, info = env.reset()

print(observation["observation"].shape)
print(observation["observation"].dtype)
# print(observation["achieved_goal"])
# print(observation["desired_goal"])

plt.imshow(observation["observation"][:, :, 0:3].astype(int))
plt.title('RGB Image (without alpha)')
plt.show()

plt.imshow(observation["observation"][:, :, 3], cmap='gray')
plt.title('RGBD Image (Depth channel)')
plt.show()

plt.imshow(observation["observation"][:, :, 4:7].astype(int))
plt.title('RGB Image (without alpha)')
plt.show()

plt.imshow(observation["observation"][:, :, 7], cmap='gray')
plt.title('RGBD Image (Depth channel)')
plt.show()

# for _ in range(500):
#     action = env.action_space.sample() # random action
#     observation, reward, terminated, truncated, info = env.step(action)

#     if _ % 100 == 0:
#         observation, info = env.reset()