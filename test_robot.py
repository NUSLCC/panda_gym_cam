import numpy as np

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_cam import PandaCam

sim = PyBullet(render_mode="human")
robot = PandaCam(sim)

for _ in range(10000):
    robot.set_action(np.array([1.0]))
    sim.step()