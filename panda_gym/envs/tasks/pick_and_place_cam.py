from typing import Any, Dict, Optional

import pybullet as p
import numpy as np
import math
import matplotlib.pyplot as plt

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from panda_gym.utils import generate_object_range


class PickAndPlaceCam(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "dense",
        distance_threshold: float = 0.05,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.far_distance_threshold = 1.0
        self.object_size = 0.04
        self.goal_range_low = None
        self.goal_range_high = None
        self.obj_range_low = None
        self.obj_range_high = None
        self.cam_width: int = 160
        self.cam_height: int = 90
        self.cam_link = 13
        self.stationary_cam_link = 1
        self.stationary_cam_pitch_angle = 60
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
            half_extents=np.array([0.4, 0.4, 0.398/2]),
            mass=0.0,
            position=np.array([0.04, 0, -0.398/2]),
            rgba_color=np.array([1, 1, 1, 1]),
        )
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1]),
        )
        self.sim.loadURDF( 
            body_name="stationary_camera",
            fileName="URDF_files/L515_cam_with_stand.urdf",
            basePosition=[0.55, 0, 0.5-0.4],
            useFixedBase=True,
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
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
        rgb_img = np.array(rgb_img).reshape(cam_width, cam_height, 4)[:, :, :3]
        return rgb_img

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def get_obj_pos_rotation(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object") 
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        return np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])

    def reset(self) -> None:
        self.obj_range_low, self.obj_range_high = generate_object_range()
        self.goal_range_low, self.goal_range_high = generate_object_range() # both object and goal must be within reach of Panda arm
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

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
