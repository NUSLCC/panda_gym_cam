from panda_gym.envs import PandaReachCamEnv

env = PandaReachCamEnv(render_mode="human") # rgb_array

observation, info = env.reset()


for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()