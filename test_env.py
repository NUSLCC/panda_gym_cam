from panda_gym.envs import PandaReachCamEnv

env = PandaReachCamEnv(render_mode="human") # rgb_array

observation, info = env.reset()

print(observation["observation"].shape)

for _ in range(500):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if _ % 100 == 0:
        observation, info = env.reset()