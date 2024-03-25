from typing import Any, Dict, Optional

import pybullet as p
import numpy as np
import math
import matplotlib.pyplot as plt

from panda_gym.envs.core import Task
from panda_gym.utils import distance
from panda_gym.utils import calculate_object_range
from panda_gym.utils import generate_object_range
from panda_gym.utils import generate_semicircle_object_range
from panda_gym.utils import colorjitter
from panda_gym.utils import masked_auto_encoder
from panda_gym.utils import velocity_calculator
from panda_gym.utils import sine_velocity

class ReachCam(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="dense",
        distance_threshold=0.05,
        image_overlap_threshold=0.80,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.image_overlap_threshold = image_overlap_threshold
        self.distance_threshold=distance_threshold
        self.far_distance_threshold = 1.0
        self.object_size = 0.04
        self.object_velocity_max = [0.15, 0.15, 0] # (x,y,z) velocity 
        self.get_ee_position = get_ee_position
        self.goal_range_low = None
        self.goal_range_high = None
        self.cam_width: int = 160
        self.cam_height: int = 90
        self.cam_link = 13
        self.stationary_cam_link = 1
        self.stationary_cam_pitch_angle = 40
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
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
            rgba_color=np.array([0.3, 0.3, 0.3, 1]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.array([0.03, 0.03, 0.03]),
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.5, 0.9, 0.5, 1]),
        )
        self.sim.loadURDF( 
            body_name="stationary_camera",
            fileName="URDF_files/L515_cam_with_stand.urdf",
            basePosition=[0.65, 0, 0.5-0.3],
            useFixedBase=True,
        )
        self.object_initial_velocity = np.random.uniform(np.array(self.object_velocity_max) / 2, self.object_velocity_max)

        # self.sim.create_sphere(
        #     body_name="outer",
        #     radius=self.object_size/2,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.array([-0.05094756, -0.21310885, self.object_size / 2]),
        #     rgba_color=np.array([0, 0, 0, 1]),
        # )
        # self.sim.create_sphere(
        #     body_name="outer",
        #     radius=self.object_size/2,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.array([-0.05094756, 0.21278672, self.object_size / 2]),
        #     rgba_color=np.array([0, 0, 0, 1]),
        # )
        # self.sim.create_sphere(
        #     body_name="outer",
        #     radius=self.object_size/2,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.array([0.1978265, 0.21278672, self.object_size / 2]),
        #     rgba_color=np.array([0, 0, 0, 1]),
        # )
        # self.sim.create_sphere(
        #     body_name="outer",
        #     radius=self.object_size/2,
        #     mass=0.0,
        #     ghost=True,
        #     position=np.array([0.1978265, -0.21310885, self.object_size / 2]),
        #     rgba_color=np.array([0, 0, 0, 1]),
        # )

    def get_obs(self) -> np.ndarray:
        # jittered_img = colorjitter(rgb_img, brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.3)
        # mae_img = masked_auto_encoder(jittered_img)
        # return mae_img
        # target_position = self.sim.get_base_position("target")
        # self.object_initial_velocity = sine_velocity(target_position, np.array(self.object_initial_velocity))
        # self.object_initial_velocity = velocity_calculator(target_position, np.array(self.object_initial_velocity))
        # target_velocity = self.object_initial_velocity
        # self.sim.set_base_velocity("target", target_velocity)
        return self.render_from_stationary_cam()

    def render_from_stationary_cam(
        self,
        # cam_width: int = 400,
        # cam_height: int = 224,
        cam_width: int = 160,
        cam_height: int = 90,
    ) -> Optional[np.ndarray]:
        """
        Stationary camera that is directly in front of the robot arm
        """
        cam_pos = self.sim.get_link_position("stationary_camera", self.stationary_cam_link)
        cam_orn = np.array(p.getQuaternionFromEuler([math.radians(90-self.stationary_cam_pitch_angle), 0, math.pi/2]))
        cam_pos[0] = cam_pos[0] - 0.04*math.cos(math.radians(90-self.stationary_cam_pitch_angle)) # 13 mm is half of L515 cam thickness, but need to use trigonometry because the camera is rotated 45 deg
        cam_pos[2] = cam_pos[2] - 0.04*math.sin(math.radians(90-self.stationary_cam_pitch_angle)) 
        rot_matrix = np.array(self.sim.physics_client.getMatrixFromQuaternion(cam_orn)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
        forward_vec = rot_matrix.dot(np.array((0, 0, -1)))
        up_vec = rot_matrix.dot(np.array((0, 1, 0)))

    #     # Define standard deviation for noise
    #     std_dev = 0.02 # for cam pose
    #     std_dev_first = 0.08  # for the first component
    #     std_dev_second = 0.01  # for the second component
    #     std_dev_third = 0.08  # for the third component
    #     # Generate random noise for each component with different standard deviations
    #     noise_first_look = np.random.uniform(-std_dev_first, std_dev_first)
    #     noise_second_look = np.random.uniform(-std_dev_second, std_dev_second)
    #     noise_third_look = np.random.uniform(-std_dev_third, std_dev_third)
    #     noise_first_up = np.random.uniform(-std_dev_first, std_dev_first)
    #     noise_second_up = np.random.uniform(-std_dev_second, std_dev_second)
    #     noise_third_up = np.random.uniform(-std_dev_third, std_dev_third)
    #     # Combine the noise components into a noise vector for look and up vectors
    #     noise_look = np.array([noise_first_look, noise_second_look, noise_third_look])
    #     noise_up = np.array([noise_first_up, noise_second_up, noise_third_up])
    #     # Generate random noise for camera position
    #     noise_cam_pos = np.random.uniform(-std_dev, std_dev, size=(3,))
    #     # Apply noise to camera position, look, and up vectors
    #     cam_pos += noise_cam_pos
    #     forward_vec += noise_look
    #    # print(forward_vec)
    #     up_vec += noise_up
    #   #  print(up_vec)
    #     # Normalize the look and up vectors
    #     forward_vec /= np.linalg.norm(forward_vec)
    #     up_vec /= np.linalg.norm(up_vec)

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
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.num_timesteps = 0
        self.hover_list = []
        self.robot_cam_initial_x, self.robot_cam_initial_y, self.robot_cam_initial_z = self.sim.get_link_position("panda_camera", self.cam_link)
        self.goal_range_low, self.goal_range_high = generate_semicircle_object_range()
        self.goal = self._sample_goal()
       # self.goal = np.array([0, 0, self.object_size / 2]) # fixed goal for testing
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
       # self.object_initial_velocity = np.random.uniform(np.array(self.object_velocity_max) / 2, self.object_velocity_max)
      #  self.object_initial_velocity = np.array([0, 0.1, 0]) # for sin function 

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the sphere center
        noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def is_terminated(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold or d > self.far_distance_threshold, dtype=bool)

    def is_failure(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d > self.far_distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
      #  self.num_timesteps += 1
       # print(self.num_timesteps)
        d = distance(achieved_goal, desired_goal).astype(np.float32)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)

    def get_obj_pos_rotation(self) -> np.ndarray:
        return np.array([])  # no obj related pos or rotation
