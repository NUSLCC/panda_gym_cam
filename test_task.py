from panda_gym.pybullet import PyBullet
from panda_gym.envs.tasks.reach_cam import ReachCam
from panda_gym.envs.tasks.slide import Slide
sim = PyBullet(render_mode="rgb_array")

task2 = ReachCam(sim)

# task1 = Slide(sim)
# task1.reset()
# print(task1.get_obs())
# print(task1.get_achieved_goal())
# print(task1.is_success(task1.get_achieved_goal(), task1.get_goal()))
# print(task1.compute_reward(task1.get_achieved_goal(), task1.get_goal()))
# print("----------------------------------------------------------------")
task2.reset()
for _ in range(1000):
    print(task2.get_obs())
# print(task2.get_achieved_goal())
# print(task2.get_goal())