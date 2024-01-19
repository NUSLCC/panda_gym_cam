
import pybullet as p
import pybullet_data
import time
import numpy as np
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF('plane.urdf')
camId = p.loadURDF('URDF_files/L515_cam_with_stand.urdf', [1.5,0,0], p.getQuaternionFromEuler([0,0,0]))
pandaId = p.loadURDF('URDF_files/panda_modified.urdf',[-0.6,0,0], p.getQuaternionFromEuler([0,0,0]), useFixedBase = True)

neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
for i in range(len(neutral_joint_values)):  
    p.resetJointState(pandaId, i, targetValue = neutral_joint_values[i])

for i in range(10000):
    """
    If the code is run without modification, the lefthand side box of 'Synthetic Camera RGB Data' displays the 
    rendering from the robot-cam (camera_state) and stationary camera away from the robot (camera_state2). Comment out one of these if you 
    only want to see one rendering.
    """


    """
    Robot cam block below
    """
    camera_state = p.getLinkState(pandaId, 13)
    camera_pos = np.array(camera_state[0])
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
    camera_pos2[0] = camera_pos2[0] - 0.013*math.cos(math.pi/4) # 13 mm is half of L515 cam thickness, but need to use trigonometry because the camera is rotated 45 deg
    camera_pos2[2] = camera_pos2[2] - 0.013*math.sin(math.pi/4) 
    camera_orn2 = np.array(p.getQuaternionFromEuler([math.pi/4, 0, math.pi/2]))

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
    

    ee_state = p.getLinkState(pandaId, 11)
    print(ee_state[0])

    # joint_poses = p.calculateInverseKinematics(pandaId, 11, [0.155, 0.255, 0], p.getQuaternionFromEuler([0,-math.pi, math.pi/2]))
    # for i in range(7):
    #     p.setJointMotorControl2(pandaId, i, controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i])

    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()

