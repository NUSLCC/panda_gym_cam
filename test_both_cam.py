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

    camera_state = p.getLinkState(pandaId, 12)
   # print(f"Camera state is {camera_state}")
    camera_pos = np.array(camera_state[0])
    camera_pos[2] = camera_pos[2] - 0.016
    camera_orn = np.array(camera_state[1])
    rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orn)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
    forward_vec = rot_matrix.dot(np.array((0, 0, -1)))
    up_vec = rot_matrix.dot(np.array((0, 1, 0)))

    # pitch_angle = -math.pi/4
    # camera_state2 = p.getLinkState(camId, 1)
    # #print(f"Camera state is {camera_state2}")
    # camera_pos2 = np.array(camera_state2[0])
    # camera_pos2[2] = camera_pos2[2] - 0.04
    # # camera_orn2 = np.array(camera_state2[1])
    # # camera_orn2 = np.array(p.getEulerFromQuaternion(camera_orn2))
    # # camera_orn2[1] = camera_orn2[1] + math.pi/4


    # camera_orn2 = p.getQuaternionFromEuler([0, pitch_angle, -math.pi])
    # rot_matrix2 = np.array(p.getMatrixFromQuaternion(camera_orn2)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
    # forward_vec2 = rot_matrix2.dot(np.array((0, 0, -1)))
    # up_vec2 = rot_matrix2.dot(np.array((0, 1, 0)))
    # # forward_vec2 = rot_matrix2.dot(np.array((0, np.sin(pitch_angle), -np.cos(pitch_angle))))
    # # up_vec2 = rot_matrix2.dot(np.array((0, np.cos(pitch_angle), np.sin(pitch_angle))))

    target_position = camera_pos + forward_vec
   # target_position2 = camera_pos2 + 0.1 * forward_vec2

    view_matrix = p.computeViewMatrix(camera_pos, target_position, up_vec)
   # view_matrix2 = p.computeViewMatrix(camera_pos2, target_position2, up_vec2)

    width = 1280
    height = 720
    aspect_ratio = width / height
    fov = 58
    nearVal = 0.01
    farVal = 100

    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal)

    rgb_img = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[2]
   # rgb_img2 = p.getCameraImage(width, height, view_matrix2, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[2]
    #rgb_img = np.array(self.rgb_img).reshape(self.height, self.width, 4)[:, :, :3]

    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()