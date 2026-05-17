import numpy as np
from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv
from safe_bimanual_rl.utils.quaternions import quat_to_mat


class TrayPickUpReachEnv(TrayPickUpBaseEnv):
    """
    Reach phase: both grippers must reach the tray handles with correct orientation.
    Episode ends on success or on excessive contact force.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 1000,
        n_substeps: int = 4,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        contact_cost_weight: float = -1e-4,
        handle_distance_weight: float = 1.0,
        contact_threshold: float = 3.0,
        control_cost_weight: float = -1e-4,
        reach_sharpness: float = 0.5,
        rotation_reward_weight: float = 1.0,
        orientation_sharpness: float = 0.3,
        success_position_reward: float = 10.0,
        success_orientation_reward: float = 50.0,
        success_position_threshold: float = 0.06,
        success_orientation_threshold: float = 0.4,
        success_steps: int = 10,
        **viewer_params,
    ):
        """
        Args:
            gamma (float): Discount factor.
            horizon (int): Maximum number of steps per episode.
            n_substeps (int): Number of MuJoCo simulation substeps per environment step.
            contact_force_range (tuple[float, float]): Clipping range for contact forces.
            contact_cost_weight (float): Weight applied to the contact force cost (negative).
            handle_distance_weight (float): Weight for the handle-distance reward.
            contact_threshold (float): Contact force magnitude above which the episode terminates.
            control_cost_weight (float): Weight penalizing large actions (negative).
            reach_sharpness (float): Controls how sharply the tanh reward drops off with distance.
            rotation_reward_weight (float): Weight for the gripper orientation reward.
            orientation_sharpness (float): Controls how sharply the tanh orientation
                reward drops off.
            success_position_reward (float): Bonus reward when both end-effectors reach
                their targets.
            success_orientation_reward (float): Bonus reward when both grippers reach
                target orientation.
            success_position_threshold (float): Distance threshold (metres) for position
                success.
            success_orientation_threshold (float): Angle threshold (radians) for
                orientation success.
            success_steps (int): Number of consecutive steps required for success.
        """
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            contact_force_range=contact_force_range,
            contact_cost_weight=contact_cost_weight,
            handle_distance_weight=handle_distance_weight,
            reach_sharpness=reach_sharpness,
            control_cost_weight=control_cost_weight,
            **viewer_params,
        )
        self._contact_threshold = contact_threshold
        self._rotation_reward_weight = rotation_reward_weight
        self._orientation_sharpness = orientation_sharpness
        self._success_position_reward = success_position_reward
        self._success_orientation_reward = success_orientation_reward
        self._success_position_threshold = success_position_threshold
        self._success_orientation_threshold = success_orientation_threshold
        self._success_steps = success_steps
        self._consecutive_success_steps = 0
        self._absorbing_counts = {
            "position_reached": 0,
            "contact_force": 0,
        }

    def _get_handle_distance_reward(self, obs):
        # Reward uses site-to-hand-end distance, not the handle-centre obs
        right_arm_pos = self._read_data("right_hande_robotiq_hande_end_pos")
        left_arm_pos = self._read_data("left_hande_robotiq_hande_end_pos")
        rel_right = self._read_data("right_grasp_target_pos") - right_arm_pos
        rel_left = self._read_data("left_grasp_target_pos") - left_arm_pos
        reward = (1 - np.tanh(np.linalg.norm(rel_right) / self._reach_sharpness)) + (
            1 - np.tanh(np.linalg.norm(rel_left) / self._reach_sharpness)
        )
        return self._handle_distance_weight * reward

    def _get_gripper_rotation_reward(self, obs):
        """
        Compute the orientation reward, active only when the gripper is within
        proximity_threshold of the handle.
        """
        right_handle_mat = quat_to_mat(
            self.obs_helper.get_from_obs(obs, "right_handle_rot")
        )
        left_handle_mat = quat_to_mat(
            self.obs_helper.get_from_obs(obs, "left_handle_rot")
        )
        right_gripper_mat = quat_to_mat(
            self._read_data("right_hande_robotiq_hande_end_rot")
        )
        left_gripper_mat = quat_to_mat(
            self._read_data("left_hande_robotiq_hande_end_rot")
        )

        proximity_threshold = 0.10
        right_arm_pos = self._read_data("right_hande_robotiq_hande_end_pos")
        left_arm_pos = self._read_data("left_hande_robotiq_hande_end_pos")
        right_dist = np.linalg.norm(
            self._read_data("right_grasp_target_pos") - right_arm_pos
        )
        left_dist = np.linalg.norm(
            self._read_data("left_grasp_target_pos") - left_arm_pos
        )

        # Left: gripper_y parallel to handle_y
        left_angle = np.arccos(
            np.clip(
                abs(np.dot(left_gripper_mat[:, 1], left_handle_mat[:, 1])), -1.0, 1.0
            )
        )
        # Right: gripper_y parallel to handle_y
        right_angle = np.arccos(
            np.clip(
                abs(np.dot(right_gripper_mat[:, 1], right_handle_mat[:, 1])), -1.0, 1.0
            )
        )

        right_reward = (
            (1 - np.tanh(right_angle / self._orientation_sharpness))
            if right_dist <= proximity_threshold
            else 0.0
        )
        left_reward = (
            (1 - np.tanh(left_angle / self._orientation_sharpness))
            if left_dist <= proximity_threshold
            else 0.0
        )

        return self._rotation_reward_weight * (right_reward + left_reward)

    def _position_reached(self, obs):
        """Check if both end effectors are within success_threshold of their grasp targets."""
        right_dist = np.linalg.norm(
            self._read_data("right_grasp_target_pos")
            - self._read_data("right_hande_robotiq_hande_end_pos")
        )
        left_dist = np.linalg.norm(
            self._read_data("left_grasp_target_pos")
            - self._read_data("left_hande_robotiq_hande_end_pos")
        )
        return (
            right_dist < self._success_position_threshold
            and left_dist < self._success_position_threshold
        )

    def _orientation_reached(self, obs):
        """
        Check if both grippers are within success_orientation_threshold
        of their handle orientations.
        """
        right_handle_mat = quat_to_mat(
            self.obs_helper.get_from_obs(obs, "right_handle_rot")
        )
        left_handle_mat = quat_to_mat(
            self.obs_helper.get_from_obs(obs, "left_handle_rot")
        )
        right_gripper_mat = quat_to_mat(
            self._read_data("right_hande_robotiq_hande_end_rot")
        )
        left_gripper_mat = quat_to_mat(
            self._read_data("left_hande_robotiq_hande_end_rot")
        )
        # Left: gripper_y parallel to handle_y
        left_angle = np.arccos(
            np.clip(
                abs(np.dot(left_gripper_mat[:, 1], left_handle_mat[:, 1])), -1.0, 1.0
            )
        )
        # Right: gripper_y parallel to handle_y
        right_angle = np.arccos(
            np.clip(
                abs(np.dot(right_gripper_mat[:, 1], right_handle_mat[:, 1])), -1.0, 1.0
            )
        )
        return (
            right_angle < self._success_orientation_threshold
            and left_angle < self._success_orientation_threshold
        )

    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute the reward for the reach environment.

        Args:
            obs (np.ndarray): The observation of the environment.
            action (np.ndarray): The action taken by the agent.
        Returns:
            reward (float): The reward for the current state and action.
        """
        handle_distance_reward = self._get_handle_distance_reward(next_obs)
        contact_table_cost = self._get_contact_cost(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        rotation_reward = self._get_gripper_rotation_reward(next_obs)
        both_reached = self._position_reached(next_obs) and self._orientation_reached(
            next_obs
        )
        position_bonus = self._success_position_reward if both_reached else 0.0
        orientation_bonus = self._success_orientation_reward if both_reached else 0.0

        reward = (
            handle_distance_reward
            + contact_table_cost
            + ctrl_cost
            + rotation_reward
            + position_bonus
            + orientation_bonus
        )

        return reward

    def setup(self, obs):
        super().setup(obs)
        self._consecutive_success_steps = 0

    def is_absorbing(self, obs):
        """
        Check if the current state is absorbing.

        Args:
            obs (np.ndarray): The observation of the environment.
        Returns:
            is_absorbing (bool): True if the current state is absorbing, False otherwise.
        """
        if self._position_reached(obs) and self._orientation_reached(obs):
            self._consecutive_success_steps += 1
            if self._consecutive_success_steps >= self._success_steps:
                print("Position and orientation success")
                self._absorbing_counts["position_reached"] += 1
                return True
            return False
        self._consecutive_success_steps = 0
        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")[0]
        if contact_force > self._contact_threshold:
            self._absorbing_counts["contact_force"] += 1
            print("Contact force exceeded threshold")
            return True
        return False

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        both_reached = self._position_reached(obs) and self._orientation_reached(obs)
        info["handle_distance_reward"] = self._get_handle_distance_reward(obs)
        info["contact_table_cost"] = self._get_contact_cost(obs)
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        info["rotation_reward"] = self._get_gripper_rotation_reward(obs)
        info["position_bonus"] = self._success_position_reward if both_reached else 0.0
        info["orientation_bonus"] = (
            self._success_orientation_reward if both_reached else 0.0
        )
        return info


if __name__ == "__main__":
    env = TrayPickUpReachEnv()
    env.reset()
    env.render()
    while True:
        action = np.zeros(env.info.action_space.shape[0])
        env.step(action)
        env.render()
