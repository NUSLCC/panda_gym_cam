import pybullet as p
import numpy as np
p.connect(p.GUI)
# Initialize PyBullet and create a scene

# Define camera parameters
width = 640
height = 480
fov = 60
near = 0.1
far = 100.0
camera_target_position = [0, 0, 0]
camera_distance = 2.0

# Capture an image from the camera
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=camera_target_position,
    distance=camera_distance,
    yaw=0,
    pitch=-10,
    roll=0,
    upAxisIndex=2,
)
projection_matrix = p.computeProjectionMatrixFOV(
    fov=fov,
    aspect=width / height,
    nearVal=near,
    farVal=far,
)
width, height, rgb_array, depth_array, seg_mask = p.getCameraImage(
    width=10,
    height=10,
    viewMatrix=view_matrix,
    projectionMatrix=projection_matrix,
    renderer=p.ER_BULLET_HARDWARE_OPENGL,
)

# Access the data
print(f"Image width: {width}")
print(f"Image height: {height}")
rgb_array = np.array(rgb_array, dtype=np.uint8)
print(f"RGB array shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
print(rgb_array)
# print(f"Depth array shape: {depth_array.shape}, dtype: {depth_array.dtype}")
# print(f"Segmentation mask shape: {seg_mask.shape}, dtype: {seg_mask.dtype}")
p.disconnect()