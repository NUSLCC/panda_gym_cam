import cv2
import numpy as np
import subprocess
import time
from panda_gym.envs import PandaReachCamEnv
from stable_baselines3 import DDPG, HerReplayBuffer, SAC
from sb3_contrib import TQC
import gymnasium as gym
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def capture_and_save_images():
    ctx = rs.context()
    devices = ctx.query_devices()
    dev0 = ctx.query_devices()[0]  
    dev1 = ctx.query_devices()[1]

    device_model0 = str(dev0.get_info(rs.camera_info.name))
    device_model1 = str(dev1.get_info(rs.camera_info.name))

    # print(f'device_model0: {device_model0}') 
    # print(f'device_model1: {device_model1}')

    pipe2 = rs.pipeline()
    cfg2 = rs.config()
    cfg2.enable_device('f1271479')

    pipe1 = rs.pipeline()
    cfg1 = rs.config()
    cfg1.enable_device('f1321229')

    cfg1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming from both cameras
    pipe1.start(cfg1)
    pipe2.start(cfg2)

    # Camera 1
    time.sleep(0.5)
    frames_1 = pipe1.wait_for_frames()
    color_frame_1 = frames_1.get_color_frame()
    color_image_1 = np.asanyarray(color_frame_1.get_data())

    file_name = f"a_Color.png"
    
    # Get the current working directory
    output_dir = os.getcwd()
    
    # Save the image as a PNG file in the current directory
    file_path = os.path.join(output_dir, file_name)
    cv2.imwrite(file_path, color_image_1)
# print(f"Saved image: {file_path}")

    # Wait for 5 seconds
    time.sleep(0.5)

    # Camera 2
    frames_2 = pipe2.wait_for_frames()
    color_frame_2 = frames_2.get_color_frame()
    color_image_2 = np.asanyarray(color_frame_2.get_data())

    file_name = f"s_Color.png"
    
    # Get the current working directory
    output_dir = os.getcwd()
    
    # Save the image as a PNG file in the current directory
    file_path = os.path.join(output_dir, file_name)
    cv2.imwrite(file_path, color_image_2)
# print(f"Saved image: {file_path}")

# cv2.imshow('Final image', static_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


    # Process images
    static_image = color_image_2[150:580, 300:900, :]

    resized_active_image = cv2.resize(color_image_1, (160, 90))
    resized_static_image = cv2.resize(static_image, (160, 90))
    
    # cv2.imshow('Final image', resized_static_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Concatenate images
    final_image = np.concatenate([resized_active_image, resized_static_image], axis=-1)
    final_image = np.transpose(final_image, (2, 0, 1))

    return final_image


def generate_roslaunch_command(current_joints_array):
    command = [f'roslaunch franka_example_controllers move_to_start_philip.launch panda_joint1:={current_joints_array[0]} panda_joint2:={current_joints_array[1]} panda_joint3:={current_joints_array[2]} panda_joint4:={current_joints_array[3]} panda_joint5:={current_joints_array[4]} panda_joint6:={current_joints_array[5]} panda_joint7:={current_joints_array[6]} robot_ip:=172.16.0.2']
    return command

def main():
    first_iteration = True
    while True:
        # Capture and save images
        final_image = capture_and_save_images()

        # Generate observation
        observation = {
            "observation": final_image.astype(np.uint8),
            "desired_goal": np.random.uniform(-10, 10, (3,)).astype(np.float32),
            "achieved_goal": np.random.uniform(-10, 10, (3,)).astype(np.float32),
            "state": np.random.uniform(-10, 10, (10,)).astype(np.float32)
        }

        # Load environment and model
        if first_iteration:
            env = gym.make('PandaReachCamJoints-v3', render_mode='rgb_array', control_type="joints")
            model = TQC.load("reach_blacktable_jitter", env=env)

        # Predict action
        action, _ = model.predict(observation, deterministic=True)

        action_array = []

        for i in action:
            action_array.append(i)

        # Update current joints
        multiplied_array = np.array(action_array) * 0.05
        if first_iteration:
            first_iteration = False
            current_joints = np.array([0, 0.41, 0, -1.85, 0, 2.26, 0.79])
        current_joints += multiplied_array

        current_joints_array = []
        for i in current_joints:
            current_joints_array.append(i)

        # Generate and execute roslaunch command
        command = generate_roslaunch_command(current_joints_array)

        # env = os.environ.copy()
        # env['PATH'] = '/opt/ros/noetic/bin:' + env['PATH'] 
     #   subprocess.run(command, shell=True, cwd='/home/franka/catkin_ws')

        print(command[0])

        # Sleep for 5 seconds
        time.sleep(12)

if __name__ == "__main__":
    main()
