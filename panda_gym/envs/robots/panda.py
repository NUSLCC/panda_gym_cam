import numpy as np
from gym import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class Panda(PyBulletRobot):
    """Panda in PyBullet.

    Args:
        sim (Any): Simulation engine.
        block_gripper (bool, optional): Whether the gripper is blocked.
            Defaults to False.
        base_position ((x, y, z), optionnal): Position of the base base of the robot.
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
    FINGERS_INDICES = np.array([9, 10])
    NEUTRAL_JOINT_VALUES = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
    JOINT_FORCES = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        control_type: str = "ee",
    ) -> None:
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        self.ee_link = 11
        super().__init__(
            sim, body_name="panda", file_name="franka_panda/panda.urdf", base_position=base_position, action_space=action_space
        )
        self.sim.set_lateral_friction(self.body_name, self.FINGERS_INDICES[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.FINGERS_INDICES[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.FINGERS_INDICES[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.FINGERS_INDICES[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        if self.control_type == "ee":
            return self.set_ee_action(action)
        else:
            return self.set_joints_action(action)

    def set_ee_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ee_ctrl = action[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_ctrl
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_angles = self._inverse_kinematics(position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        if not self.block_gripper:
            fingers_ctrl = action[3] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl
            target_angles[-2:] = [target_fingers_width / 2, target_fingers_width / 2]
        self.control_joints(target_angles=target_angles)

    def set_joints_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joints_ctrl = action[:7] * 0.05  # limit maximum change in position
        # get the current position and the target position
        joints_position = [self.sim.get_joint_angle(self.body_name, joint=i) for i in range(7)]
        target_joints = joints_position + joints_ctrl
        # Clip the height target. For some reason, it has a great impact on learning
        # target_joints_position[2] = max(0, target_joints_position[2])

        if not self.block_gripper:
            fingers_ctrl = action[7] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl
        else:
            target_fingers_width = 0
        target_angles = np.concatenate((target_joints, [target_fingers_width / 2, target_fingers_width / 2]))

        self.control_joints(target_angles=target_angles)

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            obs = np.concatenate((ee_position, ee_velocity))
        return obs

    def reset(self):
        self.set_joint_neutral()

    def get_fingers_width(self):
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.FINGERS_INDICES[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.FINGERS_INDICES[1])
        return finger1 + finger2

    def get_ee_position(self):
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self):
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def _inverse_kinematics(self, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values. The last two
        coordinates (fingers) are [0, 0].

        Args:
            position (x, y, z): Desired position of the end-effector.
            orientation (x, y, z, w): Desired orientation of the end-effector.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(
            self.body_name, ee_link=11, position=position, orientation=orientation
        )
        # Replace the fingers coef by [0, 0]
        inverse_kinematics[7:9] = np.zeros(2)
        return inverse_kinematics

    def set_joint_neutral(self):
        """Set the robot to its neutral pose."""
        self.set_joint_values(self.NEUTRAL_JOINT_VALUES)

    def set_ee_position(self, position: np.ndarray):
        """Set the end-effector position. Can induce collisions.

        Warning:
            Make sure that the position does not induce collision.

        Args:
            position (x, y, z): Desired position of the gripper.
        """
        # compute the new joint angles
        angles = self._inverse_kinematics(position=position, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self.set_joint_values(angles=angles)

    def set_joint_values(self, angles):
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.JOINT_INDICES, angles=angles)
