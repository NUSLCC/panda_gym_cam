from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import calculate_coverage_ratio_nparray


class ReachCam(Task):
    def __init__(
        self,
        sim,
        reward_type="dense",
        image_overlap_threshold=0.80,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.image_overlap_threshold = image_overlap_threshold
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.render_width: int = 480
        self.render_height: int = 480
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        print("get_obs_none")
        return 

    def get_achieved_goal(self) -> np.ndarray:
        ## return achieved goal
        return 

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def is_success(self, camera_viewarea: np.ndarray, object_viewarea: np.ndarray) -> np.ndarray:
        c = calculate_coverage_ratio_nparray(camera_viewarea, object_viewarea)
        return np.array(c < self.image_overlap_threshold, dtype=bool)

    def compute_reward(self, camera_viewarea, object_viewarea, info: Dict[str, Any]) -> np.ndarray:
        c = calculate_coverage_ratio_nparray(camera_viewarea, object_viewarea)
        if self.reward_type == "sparse":
            return -np.array(c > self.image_overlap_threshold, dtype=np.float32)
        else:
            return -c.astype(np.float32)
