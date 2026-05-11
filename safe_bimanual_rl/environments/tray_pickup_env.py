import os
import mujoco
import numpy as np
from safe_bimanual_rl.environments.bimanual_table_env import BimanualTableEnv
from mushroom_rl.environments.mujoco import ObservationType
from safe_bimanual_rl.utils.quaternions import quat_to_mat
from mushroom_rl.rl_utils.spaces import Box


# Reach a point environment with two arms
class TrayPickUpEnv(BimanualTableEnv):
    """
    A reach environment for two arms,
    where the goal is to pick up a cube on a tray.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 1000,
        n_substeps: int = 5,
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
        success_steps: int = 40,
        **viewer_params,
    ):
        """
        Initialize the tray pick-up environment.

        Args:
            gamma (float): Discount factor.
            horizon (int): Maximum number of steps per episode.
            n_substeps (int): Number of MuJoCo simulation substeps per environment step.
            contact_force_range (tuple[float, float]): Clipping range for contact forces used in
                the observation.
            contact_cost_weight (float): Weight applied to the contact force cost (should be
                negative).
            handle_distance_weight (float): Weight for the handle-distance reward.
            contact_threshold (float): Contact force magnitude above which the episode terminates.
            control_cost_weight (float): Weight penalizing large actions (should be negative).
            reach_sharpness (float): Controls how sharply the tanh reward drops off with distance
                to the grasp target.
            rotation_reward_weight (float): Weight for the gripper orientation reward.
            orientation_sharpness (float): Controls how sharply the tanh orientation reward drops
                off with angular error.
            success_position_reward (float): Bonus reward granted when both end-effectors reach
                their grasp targets.
            success_orientation_reward (float): Bonus reward granted when both grippers reach the
                target orientation.
            success_position_threshold (float): Distance threshold (metres) for position success.
            success_orientation_threshold (float): Angle threshold (radians) for orientation
                success.
        """

        additional_data_spec = [
            ("cube_pos", "cube", ObservationType.BODY_POS),
            ("tray_pos", "tray", ObservationType.BODY_POS),
            ("right_grasp_target_pos", "right_grasp_target", ObservationType.SITE_POS),
            ("left_grasp_target_pos", "left_grasp_target", ObservationType.SITE_POS),
            ("right_handle_rot", "right_handle", ObservationType.BODY_ROT),
            ("left_handle_rot", "left_handle", ObservationType.BODY_ROT),
            (
                "right_hande_robotiq_hande_end_pos",
                "right_hande_robotiq_hande_end",
                ObservationType.BODY_POS,
            ),
            (
                "left_hande_robotiq_hande_end_pos",
                "left_hande_robotiq_hande_end",
                ObservationType.BODY_POS,
            ),
            (
                "right_hande_robotiq_hande_end_rot",
                "right_hande_robotiq_hande_end",
                ObservationType.BODY_ROT,
            ),
            (
                "left_hande_robotiq_hande_end_rot",
                "left_hande_robotiq_hande_end",
                ObservationType.BODY_ROT,
            ),
        ]

        collision_groups = [
            ("cube", ["cube"]),
            (
                "tray",
                [
                    "tray_base",
                    "tray_wall_front",
                    "tray_wall_back",
                    "tray_wall_right",
                    "tray_wall_left",
                ],
            ),
            ("table", ["table_base_link_collision"]),
            ("right_handle", ["right_handle_bar"]),
            ("left_handle", ["left_handle_bar"]),
            (
                "right_hand_right_finger",
                ["right_hande_robotiq_hande_right_finger_collision"],
            ),
            (
                "right_hand_left_finger",
                ["right_hande_robotiq_hande_left_finger_collision"],
            ),
            (
                "left_hand_right_finger",
                ["left_hande_robotiq_hande_right_finger_collision"],
            ),
            (
                "left_hand_left_finger",
                ["left_hande_robotiq_hande_left_finger_collision"],
            ),
        ]

        scene_xml = os.path.join(
            os.path.dirname(__file__), "data", "tray_pickup_env.xml"
        )

        self._contact_cost_weight = contact_cost_weight
        self._handle_distance_weight = handle_distance_weight
        self._contact_force_range = contact_force_range
        self._contact_threshold = contact_threshold
        self._control_cost_weight = control_cost_weight
        self._reach_sharpness = reach_sharpness
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

        super().__init__(
            scene_xml=scene_xml,
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        self.obs_helper.add_obs("rel_right_handle_pos", 3)
        self.obs_helper.add_obs("rel_left_handle_pos", 3)
        self.obs_helper.add_obs("contact_force", 1)
        self.obs_helper.add_obs("right_handle_rot", 4)
        self.obs_helper.add_obs("left_handle_rot", 4)

        # Update dimensions of the observation space
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())

        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        # Relative positions to grasp targets (5 cm from each handle, tracked via MuJoCo sites)
        right_grasp_target = self._read_data("right_grasp_target_pos")
        left_grasp_target = self._read_data("left_grasp_target_pos")

        right_arm_pos = self._read_data("right_hande_robotiq_hande_end_pos")
        left_arm_pos = self._read_data("left_hande_robotiq_hande_end_pos")

        rel_right_handle_pos = right_grasp_target - right_arm_pos
        rel_left_handle_pos = left_grasp_target - left_arm_pos

        # Create contact force observation (robot+hand / table)
        contact_force = self._get_contact_force(
            "robot", "table", self._contact_force_range
        ) + self._get_contact_force("hand", "table", self._contact_force_range)

        # Create handle rotation observations
        right_handle_rot = self._read_data("right_handle_rot")
        left_handle_rot = self._read_data("left_handle_rot")

        # Concatenate the new observations to the original observation
        obs = np.concatenate(
            [
                obs,
                rel_right_handle_pos,
                rel_left_handle_pos,
                contact_force,
                right_handle_rot,
                left_handle_rot,
            ]
        )

        return obs

    def _get_contact_cost(self, obs):
        """
        Compute the cost based on the contact force exceeding the threshold.

        Args:
            obs: The observation of the environment.

        Returns:
            cost: The computed cost based on the excess contact force beyond the threshold.
                  Returns 0 when contact_force <= contact_threshold.
        """

        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")[0]
        return self._contact_cost_weight * contact_force

    def _get_handle_distance_reward(self, obs):
        """
        Compute the reward based on the distance between the handles and the end effectors.

        Args:
            obs: The observation of the environment.

        Returns:
            reward: The computed reward based on the distance between the handles
            and the end effectors
        """

        rel_handle_pos_right = self.obs_helper.get_from_obs(obs, "rel_right_handle_pos")
        rel_handle_pos_left = self.obs_helper.get_from_obs(obs, "rel_left_handle_pos")

        right_arm_distance = np.linalg.norm(rel_handle_pos_right)
        left_arm_distance = np.linalg.norm(rel_handle_pos_left)

        reward = (1 - np.tanh(right_arm_distance / self._reach_sharpness)) + (
            1 - np.tanh(left_arm_distance / self._reach_sharpness)
        )

        return self._handle_distance_weight * reward

    def _get_gripper_rotation_reward(self, obs):
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

        # Left: gripper_y parallel to handle_y, gripper_z same direction as handle_x
        left_angle_y = np.arccos(
            np.clip(
                abs(np.dot(left_gripper_mat[:, 1], left_handle_mat[:, 1])), -1.0, 1.0
            )
        )
        # left_angle_zx = np.arccos(
        #     np.clip(np.dot(left_gripper_mat[:, 2], left_handle_mat[:, 0]), -1.0, 1.0)
        # )

        # Right: gripper_y parallel to handle_y, gripper_z opposite direction to handle_x
        right_angle_y = np.arccos(
            np.clip(
                abs(np.dot(right_gripper_mat[:, 1], right_handle_mat[:, 1])), -1.0, 1.0
            )
        )
        # right_angle_zx = np.arccos(
        #     np.clip(-np.dot(right_gripper_mat[:, 2], right_handle_mat[:, 0]), -1.0, 1.0)
        # )

        # left_reward = (
        #     (1 - np.tanh(left_angle_y / 0.4)) + (1 - np.tanh(left_angle_zx / 0.4))
        # ) / 2
        #
        # right_reward = (
        #     (1 - np.tanh(right_angle_y / 0.4)) + (1 - np.tanh(right_angle_zx / 0.4))
        # ) / 2

        proximity_threshold = 0.10
        right_dist = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_right_handle_pos")
        )
        left_dist = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_left_handle_pos")
        )

        right_reward = (
            (1 - np.tanh(right_angle_y / self._orientation_sharpness))
            if right_dist <= proximity_threshold
            else 0.0
        )
        left_reward = (
            (1 - np.tanh(left_angle_y / self._orientation_sharpness))
            if left_dist <= proximity_threshold
            else 0.0
        )

        return self._rotation_reward_weight * (right_reward + left_reward)

    def _get_ctrl_cost(self, action):
        """
        Compute the control cost penalizing large actions.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            float: The weighted sum of squared action values.
        """
        ctrl_cost = np.sum(np.square(action))
        return self._control_cost_weight * ctrl_cost

    def _position_reached(self, obs):
        """
        Check if both end effectors are within success_threshold of their grasp targets.
        """
        right_dist = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_right_handle_pos")
        )
        left_dist = np.linalg.norm(
            self.obs_helper.get_from_obs(obs, "rel_left_handle_pos")
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

        right_angle = np.arccos(
            np.clip(
                abs(np.dot(right_gripper_mat[:, 1], right_handle_mat[:, 1])), -1.0, 1.0
            )
        )
        left_angle = np.arccos(
            np.clip(
                abs(np.dot(left_gripper_mat[:, 1], left_handle_mat[:, 1])), -1.0, 1.0
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
                self._absorbing_counts["position_reached"] += 1
                print("Position and orientation success")
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
    env = TrayPickUpEnv()
    env.reset()
    env.render()
    while True:
        action = np.zeros(env.info.action_space.shape[0])
        env.step(action)
        env.render()