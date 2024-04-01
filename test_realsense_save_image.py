import cv2
import pyrealsense2 as rs
import os
from datetime import datetime
import time
import numpy as np

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a new frame from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert the frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        file_name = f"s_Color.png"
        
        # Get the current working directory
        output_dir = os.getcwd()
        
        # Save the image as a PNG file in the current directory
        file_path = os.path.join(output_dir, file_name)
        cv2.imwrite(file_path, color_image)
       # print(f"Saved image: {file_path}")

        # Wait for 5 seconds
        time.sleep(5)

except KeyboardInterrupt:
    pass

finally:
    # Stop the pipeline and clean up
    pipeline.stop()
