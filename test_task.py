import numpy as np
from panda_gym.pybullet import PyBullet
from panda_gym.envs.tasks.reach_cam import ReachCam
from panda_gym.envs.robots.panda_camera import PandaWithCamera

sim = PyBullet(render_mode="rgb_array")
robot = PandaWithCamera(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type="joints")
task1= ReachCam(sim, get_ee_position=robot.get_ee_position)

task1.reset()
print(task1.get_obs())
print(task1.get_achieved_goal())
print(task1.is_success(task1.get_achieved_goal(), task1.get_goal()))
print(task1.compute_reward(task1.get_achieved_goal(), task1.get_goal(), {}))
