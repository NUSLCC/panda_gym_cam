# panda-gym

Set of robotic environments based on PyBullet physics engine and gymnasium.

## Documentation

Check out the [documentation](https://panda-gym.readthedocs.io/en/latest/).

## Installation

```bash
pip install panda-gym
```

## Usage

```python
import gymnasium as gym
import panda_gym

env = gym.make('PandaReach-v3', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Baselines results

Baselines results are available in [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) and the pre-trained agents in the [Hugging Face Hub](https://huggingface.co/sb3).