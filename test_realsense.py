import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

ctx = rs.context()
devices = ctx.query_devices()
dev0 = ctx.query_devices()[0]  
dev1 = ctx.query_devices()[1]

device_model0 = str(dev0.get_info(rs.camera_info.name))
device_model1 = str(dev1.get_info(rs.camera_info.name))

print(f'device_model0: {device_model0}') 
print(f'device_model1: {device_model1}')

pipe1 = rs.pipeline()
cfg1 = rs.config()
cfg1.enable_device('f1321229')

pipe2 = rs.pipeline()
cfg2 = rs.config()
cfg2.enable_device('f1271479')

cfg1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming from both cameras
pipe1.start(cfg1)
pipe2.start(cfg2)

try:
    while True:

        # Camera 1
        frames_1 = pipe1.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()

        # Convert images to numpy arrays
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        file_name = f"a_Color.png"
        
        # Get the current working directory
        output_dir = os.getcwd()
        
        # Save the image as a PNG file in the current directory
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, color_image_1)
       # print(f"Saved image: {file_path}")

        # Wait for 5 seconds
        time.sleep(5)

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

        # Wait for 5 seconds
        time.sleep(5)

        # # Stack all images horizontally
        # images = np.hstack((color_image_1, color_image_2))

        # # Show images from both cameras
        # cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        # cv2.imshow('RealSense', images)
        # cv2.waitKey(1)

finally:

    # Stop streaming
    pipe1.stop()
    pipe2.stop()

    # https://github.com/IntelRealSense/librealsense/issues/1735