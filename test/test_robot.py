import matplotlib.pyplot as plt
import sys
sys.path.append("/home/lcc/GitRepo/panda-gym")
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_cam import PandaCam

sim = PyBullet(render_mode="human")
robot = PandaCam(sim)
obs = robot.get_obs()

# print(rgb.shape)
# print(dep.shape)
print(obs.shape)
# print(rgb[1,1])
# print(dep[1,1])
# print(obs[:, :, 3])
print(obs.dtype)


plt.imshow(obs[:, :, :3].astype(int))
plt.title('RGB Image (without alpha)')
plt.show()

plt.imshow(obs[:, :, 3], cmap='gray')
plt.title('RGBD Image (Depth channel)')
plt.show()