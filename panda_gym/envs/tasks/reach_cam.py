import numpy as np
import math
from typing import Any, Dict, Optional

import pybullet as p
from panda_gym.envs.core import Task
from panda_gym.utils import distance

class ReachCam(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="dense",
        distance_threshold=0.02,
        goal_range=0.5,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold=distance_threshold
        self.far_distance_threshold = 1.0
        self.object_size = 0.04
        self.get_ee_position = get_ee_position 
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.cam_link = 13
        self.stationary_cam_link = 1
        self.stationary_cam_pitch_angle = 40

        self.target_start_x_position = 0.0
        self.target_start_y_position = -0.5
        self.target_start_position = np.array([self.target_start_x_position, self.target_start_y_position, self.object_size / 2])
        self.target_current_x_position = self.target_start_x_position
        self.target_current_y_position = self.target_start_y_position
        
        self.target_current_x_step = 0.0
        self.target_x_step_length = 2*math.pi/50 # 100 is the max of steps per run
        self.target_y_step_length = 0.01

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_box(
            body_name="black_panda_table",
            half_extents=np.array([0.32, 0.32, 0.38/2]),
            mass=0.0,
            position=np.array([-0.68, 0, -0.21]),
            rgba_color=np.array([0, 0, 0, 1]),
        )
        self.sim.create_box(
            body_name="silver_table_block",
            half_extents=np.array([0.2, 0.2, 0.01]),
            mass=0.0,
            position=np.array([-0.68, 0, -0.01]),
            rgba_color=np.array([192/255, 192/255, 192/255, 1]),
        )
        self.sim.create_box(
            body_name="white_target_table",
            half_extents=np.array([0.4, 0.64, 0.4/2]), 
            mass=0.0,
            position=np.array([0.04, 0, -0.4/2]),
            rgba_color=np.array([1, 1, 1, 1]),
        )
        self.sim.create_sphere(
            body_name="target",
            radius=self.object_size/2,
            mass=0.0,
            ghost=True,
            position=self.target_start_position,
            rgba_color=np.array([0.1, 0.9, 0.1, 1]),
        )
        self.sim.loadURDF(
            body_name="stationary_camera",
            fileName="URDF_files/L515_cam_with_stand.urdf",
            basePosition=[0.65, 0, 0.5-0.3],
            useFixedBase=True,
        )

    def get_obs(self) -> np.ndarray:
        rgb_img = self.render_from_stationary_cam() 
        return rgb_img

    def render_from_stationary_cam(
        self,
        cam_width: int = 160,
        cam_height: int = 160,
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
        rgb_img = np.array(rgb_img).reshape((cam_height, cam_width, 4))[:, :, :3]

        depth_img = self.sim.physics_client.getCameraImage(cam_width, cam_height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[3]
        depth_img = np.array(depth_img).reshape((cam_height, cam_width))
        depth_img = farVal * nearVal / (farVal - (farVal - nearVal) * depth_img)
        depth_img = depth_img[..., np.newaxis]

        # global_cam = np.concatenate((rgb_img, depth_img), axis=-1)

        return rgb_img, depth_img

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position
    
    def get_goal(self) -> np.ndarray:
        goal = self.sim.get_base_position("target")
        return goal 

    def reset(self) -> None:
        self.goal = self.get_goal()
        self.sim.set_base_pose("target", self.target_start_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.target_current_x_position = self.target_start_x_position
        self.target_current_y_position = self.target_start_y_position

    def set_target_position(self) -> None:
        self.target_current_x_step += self.target_x_step_length
        self.target_current_x_position = 0.2 * math.sin(self.target_current_x_step)
        self.target_current_y_position += self.target_y_step_length
        self.sim.set_base_pose("target", np.array([self.target_current_x_position, self.target_current_y_position, self.object_size / 2]), np.array([0.0, 0.0, 0.0, 1.0]))

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def is_failure(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d > self.far_distance_threshold, dtype=bool)
    
    def is_terminated(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold or d > self.far_distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)

    def get_obj_pos_rotation(self) -> np.ndarray:
        return np.array([])  # no obj related pos or rotation
