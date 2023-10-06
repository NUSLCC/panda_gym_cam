import numpy as np

from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_camera import PandaWithCamera

sim = PyBullet(render_mode="human")
robot = PandaWithCamera(sim)

for _ in range(1000):
    robot.set_action(np.array([1.0]))
    sim.step()