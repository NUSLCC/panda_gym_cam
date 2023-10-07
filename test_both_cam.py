import pybullet as p
import pybullet_data
import time
import numpy as np
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF('plane.urdf')
camId = p.loadURDF('URDF_files/d405_cam_with_stand.urdf', [1.5,0,0], p.getQuaternionFromEuler([0,0,0]))
pandaId = p.loadURDF('URDF_files/panda_modified.urdf',[0,0,0], p.getQuaternionFromEuler([0,0,0]))


for i in range(10000):
    """
    If the code is run without modification, the lefthand side box of 'Synthetic Camera RGB Data' displays the 
    rendering from the robot-cam (camera_state) and stationary camera away from the robot (camera_state2). Comment out one of these if you 
    only want to see one rendering.
    """


    """
    Robot cam block below
    """
    camera_state = p.getLinkState(pandaId, 12)
   # print(f"Camera state is {camera_state}")
    camera_pos = np.array(camera_state[0])
    camera_pos[2] = camera_pos[2] - 0.0185 # 11.5 mm is half of D405 cam thickness, but need to extend 18.5mm out to not see cam material itself
    camera_orn = np.array(camera_state[1])
    rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orn)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
    forward_vec = rot_matrix.dot(np.array((0, 0, -1)))
    up_vec = rot_matrix.dot(np.array((0, 1, 0)))
    target_position = camera_pos + forward_vec
    view_matrix = p.computeViewMatrix(camera_pos, target_position, up_vec)

    """
    Stationary cam block below
    """
    camera_state2 = p.getLinkState(camId, 1)
    camera_pos2 = np.array(camera_state2[0])
    camera_pos2[2] = camera_pos2[2] - 0.0185 # 11.5 mm is half of D405 cam thickness, but need to extend 18.5mm out to not see cam material itself
    camera_orn2 = np.array(camera_state2[1])

    rot_matrix2 = np.array(p.getMatrixFromQuaternion(camera_orn2)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
    forward_vec2 = rot_matrix2.dot(np.array((0, 0, -1)))
    up_vec2 = rot_matrix2.dot(np.array((0, 1, 0)))
    target_position2 = camera_pos2 + 0.1 * forward_vec2
    view_matrix2 = p.computeViewMatrix(camera_pos2, target_position2, up_vec2)

    width = 1280
    height = 720
    aspect_ratio = width / height
    fov = 58
    nearVal = 0.01
    farVal = 100

    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal)

    """
    Comment out one of these to see only one rendering, or display both to see both rendering
    """
    rgb_img = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[2] # COMMENT OUT THIS LINE TO SHOW ROBOT-CAM RENDERING
    rgb_img2 = p.getCameraImage(width, height, view_matrix2, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[2]
    
    #rgb_img = np.array(self.rgb_img).reshape(self.height, self.width, 4)[:, :, :3]
    #rgb_img2 = np.array(self.rgb_img).reshape(self.height, self.width, 4)[:, :, :3]

    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()