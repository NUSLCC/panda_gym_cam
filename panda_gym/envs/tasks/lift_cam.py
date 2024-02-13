from typing import Any, Dict, Optional

import pybullet as p
import numpy as np
import math
import matplotlib.pyplot as plt

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from panda_gym.utils import generate_object_range
from panda_gym.utils import generate_semicircle_object_range
from panda_gym.utils import calculate_object_range


class LiftCam(Task):
    def __init__(
        self,
        sim: PyBullet,
        get_ee_position,
        reward_type: str = "dense",
        distance_threshold: float = 0.02,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.far_distance_threshold = 0.8 
        self.object_size = 0.04
        self.get_ee_position = get_ee_position
        self.lift_distance = 0.1 # distance to lift object to achieve success
        self.goal_range_low = None
        self.goal_range_high = None
        self.obj_range_low = None
        self.obj_range_high = None
        self.cam_width: int = 160
        self.cam_height: int = 90
        self.cam_link = 13
        self.stationary_cam_link = 1
        self.stationary_cam_pitch_angle = 40
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_box(
            body_name="black_panda_table",
            half_extents=np.array([0.32, 0.32, 0.398/2]),
            mass=0.0,
            position=np.array([-0.68, 0, -0.398/2]),
            rgba_color=np.array([0, 0, 0, 1]),
        )
        self.sim.create_box(
            body_name="silver_table_block",
            half_extents=np.array([0.2, 0.2, 0.01]),
            mass=0.0,
            position=np.array([-0.68, 0, -0.0001]),
            rgba_color=np.array([192/255, 192/255, 192/255, 1]),
        )
        self.sim.create_box(
            body_name="white_table",
            half_extents=np.array([0.4, 0.64, 0.398/2]), 
            mass=0.0,
            position=np.array([0.04, 0, -0.398/2]),
            rgba_color=np.array([1, 1, 1, 1]),
        )
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.0, 1.0, 0.0, 1.0]),
        )
        self.sim.loadURDF( 
            body_name="stationary_camera",
            fileName="URDF_files/L515_cam_with_stand.urdf",
            basePosition=[0.65, 0, 0.5-0.3],
            useFixedBase=True,
        )

    def get_obs(self) -> np.ndarray:
        return self.render_from_stationary_cam() 
    
    def render_from_stationary_cam(
        self,
        cam_width: int = 160,
        cam_height: int = 90,
    ) -> Optional[np.ndarray]:
        """
        Stationary camera that is directly in front of the robot arm
        """
        cam_pos = self.sim.get_link_position("stationary_camera", self.stationary_cam_link)
        cam_orn = np.array(p.getQuaternionFromEuler([math.radians(90-self.stationary_cam_pitch_angle), 0, math.pi/2]))
        cam_pos[0] = cam_pos[0] - 0.013*math.cos(math.radians(90-self.stationary_cam_pitch_angle)) # 13 mm is half of L515 cam thickness, but need to use trigonometry because the camera is rotated 45 deg
        cam_pos[2] = cam_pos[2] - 0.013*math.sin(math.radians(90-self.stationary_cam_pitch_angle)) 
        rot_matrix = np.array(self.sim.physics_client.getMatrixFromQuaternion(cam_orn)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
        forward_vec = rot_matrix.dot(np.array((0, 0, -1)))
        up_vec = rot_matrix.dot(np.array((0, 1, 0)))
        target_position = cam_pos + 0.1 * forward_vec
        view_matrix = self.sim.physics_client.computeViewMatrix(cam_pos, target_position, up_vec)
        aspect_ratio = cam_width / cam_height
        fov = 43
        nearVal = 0.01
        farVal = 100
        proj_matrix = self.sim.physics_client.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal)

        rgb_img = self.sim.physics_client.getCameraImage(cam_width, cam_height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[2]
        rgb_img = np.array(rgb_img).reshape(cam_height, cam_width, 4)[:, :, :3]

        depth_img = self.sim.physics_client.getCameraImage(cam_width, cam_height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[3]
        depth_img = np.array(depth_img).reshape((cam_height, cam_width))
        depth_img = farVal * nearVal / (farVal - (farVal - nearVal) * depth_img)
        depth_img = depth_img[..., np.newaxis]

        return rgb_img, depth_img

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def get_obj_pos_rotation(self) -> np.ndarray:
        # position, rotation of the object
     #   object_position = self.sim.get_base_position("object") 
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        return np.concatenate([object_rotation, object_velocity, object_angular_velocity])

    def reset(self) -> None:
        # self.obj_range_low, self.obj_range_high = generate_semicircle_object_range()
        # self.goal_range_low, self.goal_range_high = generate_semicircle_object_range() # both object and goal must be within reach of Panda arm
        self.robot_cam_initial_x, self.robot_cam_initial_y, self.robot_cam_initial_z = self.sim.get_link_position("panda_camera", self.cam_link)
        self.goal_range_low, self.goal_range_high = calculate_object_range(initial_x_coord=self.robot_cam_initial_x, initial_y_coord=self.robot_cam_initial_y, initial_z_coord=self.robot_cam_initial_z)
        object_position = self._sample_object()
        self.goal = self._sample_goal(object_position)
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self, object_position) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.lift_distance])  # z offset for the sphere center
        goal += object_position
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the sphere center
        noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def is_terminated(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal) # distance between object cube and target cube
        ee_position = np.array(self.get_ee_position())
        ee_distance = distance(achieved_goal, ee_position) # distance between end effector and object cube
      #  height = self.sim.get_base_position("object")[2]
       # return np.array(d < self.distance_threshold or d > self.far_distance_threshold or height < self.object_size/2, dtype=bool)
        return np.array(d < self.distance_threshold or ee_distance > self.far_distance_threshold, dtype=bool)

    def is_failure(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        ee_distance = distance(achieved_goal, ee_position) # distance between end effector and object cube
       # d = distance(achieved_goal, desired_goal)
        return np.array(ee_distance > self.far_distance_threshold, dtype=bool)
    
    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal).astype(np.float32)
        ee_position = np.array(self.get_ee_position()).astype(np.float32)
        ee_distance = distance(achieved_goal, ee_position).astype(np.float32)
        # print("ee_pos", ee_position, ee_position.shape)
        # print("achieved_goal", achieved_goal, achieved_goal.shape)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            reward = np.float32(0)
            reward_reaching = 1-np.tanh(9*ee_distance)
            reward_lifting = np.float32(0)
            reward_lifting_weight = np.float32(1.5)
            if achieved_goal[2] >= 0.03: # center of mass of object is off the table
                reward_lifting = reward_lifting_weight*(1-np.tanh(20*(np.linalg.norm(d-self.distance_threshold))))
                reward += reward_reaching + reward_lifting - 0.05
                reward = np.clip(reward, 0, 1+reward_lifting_weight)
            else:
                reward += reward_reaching + reward_lifting - 0.02
                reward = np.clip(reward, 0, 1+reward_lifting_weight)
            # print(f'Reward: {reward}')
            # print(f'Reach: {reward_reaching}, lift: {reward_lifting}')
            return reward.astype(np.float32)
    

