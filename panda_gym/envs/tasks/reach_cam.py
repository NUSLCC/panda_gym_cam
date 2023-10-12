from typing import Any, Dict, Optional

import pybullet as p
import numpy as np
import math

from panda_gym.envs.core import Task
from panda_gym.utils import distance
from panda_gym.utils import calculate_object_range
from panda_gym.utils import generate_object_range


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
        self.get_ee_position = get_ee_position
        self.goal_range_low = None
        self.goal_range_high = None
        self.cam_width: int = 160
        self.cam_height: int = 90
        self.cam_link = 13
        self.stationary_cam_link = 1
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=self.object_size/2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.loadURDF( 
            body_name="stationary_camera",
            fileName="URDF_files/d405_cam_with_stand.urdf",
            basePosition=[1.5, 0.0, 0.0],
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
        cam_orn = self.sim.get_link_orientation("stationary_camera", self.stationary_cam_link)
        cam_pos[0] = cam_pos[0] - 0.0115*math.cos(math.pi/4) -0.001 # 11.5 mm is half of D405 cam thickness, but need to use trigonometry because the camera is rotated 45 deg
        cam_pos[2] = cam_pos[2] - 0.0115*math.sin(math.pi/4) - 0.001
        rot_matrix = np.array(self.sim.physics_client.getMatrixFromQuaternion(cam_orn)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
        forward_vec = rot_matrix.dot(np.array((0, 0, -1)))
        up_vec = rot_matrix.dot(np.array((0, 1, 0)))
        target_position = cam_pos + 0.1 * forward_vec
        view_matrix = self.sim.physics_client.computeViewMatrix(cam_pos, target_position, up_vec)
        aspect_ratio = cam_width / cam_height
        fov = 58
        nearVal = 0.01
        farVal = 100
        proj_matrix = self.sim.physics_client.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal)
        rgb_img = self.sim.physics_client.getCameraImage(cam_width, cam_height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[2]
        rgb_img = np.array(rgb_img).reshape(cam_width, cam_height, 4)[:, :, :3]
        return rgb_img

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.robot_cam_initial_x, self.robot_cam_initial_y, self.robot_cam_initial_z = self.sim.get_link_position("panda_camera", self.cam_link)
        self.goal_range_low, self.goal_range_high = generate_object_range()
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

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
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
