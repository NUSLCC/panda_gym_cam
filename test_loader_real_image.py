from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer, SAC
from sb3_contrib import TQC
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re

active_image = cv2.imread('active1_Color.png')
static_image = cv2.imread('static1_Color.png')

static_image = static_image[50:580,100:900,:]

resized_active_image = cv2.resize(active_image, (160,90))
resized_static_image = cv2.resize(static_image, (160,90))

final_image = np.concatenate([resized_active_image, resized_static_image], axis=-1)
final_image = np.transpose(final_image, (2, 0, 1))

#print("final img shape", final_image.shape)

observation = {
    "observation": final_image.astype(np.uint8),
    "desired_goal": np.random.uniform(-10, 10, (3,)).astype(np.float32),
    "achieved_goal": np.random.uniform(-10, 10, (3,)).astype(np.float32),
    "state": np.random.uniform(-10, 10, (10,)).astype(np.float32)
}

print("Observation: ", observation)

env = gym.make('PandaReachCamJoints-v3', render_mode="human", control_type="joints") # rgb_array
#print(env.action_space)
# HER must be loaded with the env
model = TQC.load("philip4_tqc_deep_reach_obsonly_joints", env=env)

obs, _ = env.reset() # don't use this obs 

action, _states = model.predict(observation, deterministic=True)

action_array = []

for i in action:
    action_array.append(i)

print("Original action array: ", action_array)

multiplied_array = np.array(action_array) * 0.05
print("Modified action array: ", multiplied_array)
neutral_joints = np.array([0, 0.41, 0, -1.85, 0, 2.26, 0.79])
current_joints = neutral_joints + multiplied_array

current_joints_array = []
for i in current_joints:
    current_joints_array.append(i)

print("Absolute joint command: ", current_joints_array)

# for element in action_array:
#     # Convert string to list of floats
#     float_list = [float(num) for num in element.strip('[]').split(',')]

#     # Multiply each float by 0.05
#     multiplied_list = [num * 0.05 for num in float_list] # panda-gym multiplies joint action by 0.05 to limit change in pose
    
#     for i in range(len(current_joints)):
#         current_joints[i] += multiplied_list[i]

#     # Convert back to string
#     multiplied_string= '[' + ','.join(map(str, multiplied_list)) + ']'
#     multiplied_absolute_string = '[' + ','.join(map(str, current_joints)) + ']'

#     # Append to result list
#     actual_joints.append(multiplied_absolute_string)
#     multiplied_joints.append(multiplied_string)

# print("Actual joint list:", actual_joints)