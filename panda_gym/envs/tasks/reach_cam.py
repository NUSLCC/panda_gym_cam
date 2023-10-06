import sys
print(sys.path)

from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance
from panda_gym.utils import calculate_coverage_ratio_nparray


class ReachCam(Task):
    def __init__(
        self,
        sim,
        reward_type="dense",
        image_overlap_threshold=0.80,
        obj_xy_range=0.3, # CHANGE THIS LATER!!
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.image_overlap_threshold = image_overlap_threshold
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0]) # np.random.uniform(obj_range_low and obj_range_high)
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.26, 0.13, 0.02, 1.0]), # dark brown color 
        )

        # put the other camera here? Use body_name? Load URDF here 

    def get_obs(self) -> np.ndarray:
        return self.sim.render_from_stationary_cam(width=self.render_width, height=self.render_height)

    def get_achieved_goal(self) -> np.ndarray:
        coverage_ratio = np.array(self.get_image_coverage_ratio())
        return coverage_ratio

    def reset(self) -> None:
        object_position = self._sample_object()
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.image_overlap_threshold, dtype=bool)

    def compute_reward(self, camera_viewarea, object_viewarea, info: Dict[str, Any]) -> np.ndarray:
        c = calculate_coverage_ratio_nparray(camera_viewarea, object_viewarea)
        if self.reward_type == "sparse":
            return -np.array(c > self.image_overlap_threshold, dtype=np.float32)
        else:
            return -c.astype(np.float32)
