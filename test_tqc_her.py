from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import TQC

env = PandaReachCamEnv(render_mode="human")

# Initialize the model
policy_kwargs = dict(n_critics=2, n_quantiles=25)

model = TQC(policy="MultiInputPolicy",env=env, replay_buffer_class=HerReplayBuffer, verbose=1, buffer_size=300000, tensorboard_log="./tmp", learning_starts=30000, 
    # Parameters for HER
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    # Parameters for TQC
    top_quantiles_to_drop_per_net=2, policy_kwargs=policy_kwargs)

# Train the model
model.learn(total_timesteps=20_000, log_interval=4, progress_bar=True)
model.save("tqc_her_panda")
