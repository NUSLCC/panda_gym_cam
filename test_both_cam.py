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

neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79]) # taken from panda gym
for i in range(len(neutral_joint_values)):  
    p.resetJointState(pandaId, i, targetValue = neutral_joint_values[i])
    
horiz_fov_deg = 87
vert_fov_deg = 58
horiz_fov = math.radians(horiz_fov_deg)
vert_fov = math.radians(vert_fov_deg)

dis_to_table = 0.2349 # z distance from the neutral_joint_values in panda-gym
horiz_total_dis = 2*dis_to_table*math.tan(horiz_fov/2)
vert_total_dis = 2*dis_to_table*math.tan(vert_fov/2)
object_size = 0.02 # object size of box to generate later 

generate_obj_extremities = True # generates four objects at four corners of robot-cam projection according to calculation below
generate_rand_obj = True # toggle to True to demonstrate generating objects randomly inside the bounding box projection

print(f"Horizontal total dis: {horiz_total_dis}")
print(f"Vertical total dis: {vert_total_dis}")

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
    camera_pos[2] = camera_pos[2] - 0.0115 - 0.001 # 11.5 mm is half of D405 cam thickness, but need to minus off another 0.001 to avoid seeing edges of itself
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
    camera_pos2[0] = camera_pos2[0] - 0.0115*math.cos(math.pi/4) -0.001 # 11.5 mm is half of D405 cam thickness, but need to use trigonometry because the camera is rotated 45 deg
    camera_pos2[2] = camera_pos2[2] - 0.0115*math.sin(math.pi/4) - 0.001
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

   # print(f"Camera pose is {camera_pos}")
    x_coord = camera_pos[0]
    y_coord = camera_pos[1]
    x_min = x_coord - vert_total_dis/2
    x_max = x_coord + vert_total_dis/2
    y_min = y_coord - horiz_total_dis/2
    y_max = y_coord + horiz_total_dis/2

    print(f"x boundaries: {(x_min, x_max)}")
    print(f"y boundaries: {(y_min, y_max)}")

    obj_range_low = (x_min, y_min, 0)
    obj_range_high = (x_max, y_max, 0)
    object_position = np.array([0, 0, object_size / 2])
    noise = np.random.uniform(obj_range_low, obj_range_high)
    object_position += noise
    object_position_extremities = [(x_min, y_min, object_size/2), (x_max, y_max, object_size/2), (x_max, y_min, object_size/2), (x_min, y_max, object_size/2)]

    col_boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents = [object_size/2, object_size/2, object_size/2])
    
    if generate_obj_extremities:
        for i in object_position_extremities:
            boxId = p.createMultiBody(1.0, col_boxId, basePosition = i, baseOrientation = p.getQuaternionFromEuler([0, 0, 0]))
        
    if generate_rand_obj:
        boxId = p.createMultiBody(1.0, col_boxId, basePosition = object_position, baseOrientation = p.getQuaternionFromEuler([0, 0, 0]))

    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()