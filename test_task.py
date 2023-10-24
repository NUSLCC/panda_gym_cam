import numpy as np
from panda_gym.pybullet import PyBullet
from panda_gym.envs.tasks.reach import Reach
from panda_gym.envs.tasks.reach_cam import ReachCam
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.robots.panda_cam import PandaCam

sim = PyBullet(render_mode="human")
robot1 = PandaCam(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type="joints")
task1= ReachCam(sim, get_ee_position=robot1.get_ee_position)

task1.reset()
print(task1.get_obs().shape)
print(task1.get_achieved_goal())
print(task1.get_goal())
# print(task1.is_success(task1.get_achieved_goal(), task1.get_goal()))
# print(task1.compute_reward(task1.get_achieved_goal(), task1.get_goal(), {}))

# print("--------------------")

# robot2 = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type="joints")
# task2= Reach(sim, get_ee_position=robot2.get_ee_position)
# task2.reset()
# print(task2.get_obs().shape)
# print(task2.get_achieved_goal())
# print(task2.get_goal())

