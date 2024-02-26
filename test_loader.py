from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import SAC, HerReplayBuffer
import gymnasium as gym
from panda_gym.utils import CustomCombinedExtractor
import time

start_time = time.time()

env = gym.make('PandaReachCam-v3', render_mode="human", control_type="ee")
# print(env.action_space)
# print(env.observation_space)
# HER must be loaded with the env

# Remember to change the feature extractors in utils
model_name = "sac_rgb_randbig_moving_lstm_ee_model"
# model_name = "sac_rgb_randbig_moving_lstm_ee_kine_model"

# model_name = "sac_rgb_randbig_moving_deformcnn_ee_model"
# model_name = "sac_rgb_randbig_moving_deformcnn_ee_kine_model"

# model_name = "sac_rgb_randbig_moving_dualcnn_ee_model"
# model_name = "sac_rgb_randbig_moving_dualcnn_ee_kine_model"

# model_name = "sac_rgb_randbig_moving_naturecnn_ee_model"
# model_name = "sac_rgb_randbig_moving_naturecnn_ee_kine_model"

model = SAC.load(model_name, env=env)
obs, _ = env.reset()

success_count = 0
failure_count = 0
term_count = 0
trun_count = 0

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
      term_count += 1
      print("Current Episode No.", term_count + trun_count)
      if info["is_success"] == True and info["is_failure"] == False:
         success_count += 1
      elif info["is_success"] == False and info["is_failure"] == True:
         failure_count += 1
      obs, info = env.reset()

    elif truncated:
      trun_count += 1
      print("truncated info:", info)
      print("Current Episode No.", term_count + trun_count)
      obs, info = env.reset()
    
    if (term_count + trun_count) == 100:
      break

end_time = time.time()
print("******************************************************************")
print("Model name:", model_name)
print("Running time:", end_time-start_time)
print("success count:", success_count)
print("failure count:", failure_count)
print("terminate count:", term_count)
print("truncated count:", trun_count)
print("******************************************************************")

# Open the file in append mode
with open('test_loader_results.txt', 'a') as file:
    file.write("******************************************************************\n")
    file.write("Model name:" + model_name + "\n")
    file.write("Running time:" + str(end_time-start_time) + "s\n")
    file.write("success count:" + str(success_count) + "\n")
    file.write("failure count:" + str(failure_count) + "\n")
    file.write("terminate count:" + str(term_count) + "\n")
    file.write("truncated count:" + str(trun_count) + "\n")