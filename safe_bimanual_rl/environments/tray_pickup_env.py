import os
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
        cube_fell_off_tray_penalty: float = -5.0,
        contact_threshold: float = 2.0,
        control_cost_weight: float = -1e-4,
        reach_sharpness: float = 0.5,
        grasp_reward: float = 5.0,
        rotation_reward_weight: float = 1.0,
        **viewer_params,
    ):
        """
        Initialize the reach environment.

        Args:
            gamma (float): The discounting factor of the environment.
            horizon (int): The maximum horizon for the environment
            n_substeps (int): The number of substeps to use by the MuJoCo simulator.
            contact_force_range (tuple[float, float]): The range of contact forces to consider.
            contact_cost_weight (float): The weight for the contact cost.
            cube_distance_weight (float): The weight for the cube distance cost.
            cube_touched_reward (float): The reward for touching the cube.
            contact_threshold (float): The threshold for considering a contact as significant.
            reach_sharpness (float): Controls how sharply the tanh reward drops off with distance.
            cube_displacement_weight (float): Penalty weight for displacing the cube from its
                initial position. Should be negative.
        """

        additional_data_spec = [
            ("tray_pos", "tray", ObservationType.BODY_POS),
            ("right_handle_pos", "right_handle", ObservationType.BODY_POS),
            ("left_handle_pos", "left_handle", ObservationType.BODY_POS),
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
        self._grasp_reward = grasp_reward
        self._cube_fell_off_tray_penalty = cube_fell_off_tray_penalty
        self._rotation_reward_weight = rotation_reward_weight

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

        # Create relative positions observations of the handles with respect to the end effectors
        right_handle_pos = self._read_data("right_handle_pos")
        left_handle_pos = self._read_data("left_handle_pos")

        right_arm_pos = self._read_data("right_hande_robotiq_hande_end_pos")
        left_arm_pos = self._read_data("left_hande_robotiq_hande_end_pos")

        rel_right_handle_pos = right_handle_pos - right_arm_pos
        rel_left_handle_pos = left_handle_pos - left_arm_pos

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

    def _cube_on_tray(self):
        """
        Check if the cube is on the tray.

        Returns:
            bool: True if the cube is on the tray, False otherwise.
        """
        return self._check_collision("cube", "tray")

    def _get_cube_fell_off_tray_cost(self):
        """
        Compute the penalty if the cube has fallen off the tray.

        Returns:
            float: The penalty value if the cube is not on the tray, 0 otherwise.
        """
        if not self._cube_on_tray():
            return self._cube_fell_off_tray_penalty
        return 0

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
        left_angle_zx = np.arccos(
            np.clip(np.dot(left_gripper_mat[:, 2], left_handle_mat[:, 0]), -1.0, 1.0)
        )

        # Right: gripper_y parallel to handle_y, gripper_z opposite direction to handle_x
        right_angle_y = np.arccos(
            np.clip(
                abs(np.dot(right_gripper_mat[:, 1], right_handle_mat[:, 1])), -1.0, 1.0
            )
        )
        right_angle_zx = np.arccos(
            np.clip(-np.dot(right_gripper_mat[:, 2], right_handle_mat[:, 0]), -1.0, 1.0)
        )

        left_reward = (
            (1 - np.tanh(left_angle_y / 0.4)) + (1 - np.tanh(left_angle_zx / 0.4))
        ) / 2

        right_reward = (
            (1 - np.tanh(right_angle_y / 0.4)) + (1 - np.tanh(right_angle_zx / 0.4))
        ) / 2

        return self._rotation_reward_weight * (right_reward + left_reward)

    def _get_grasp_reward(self):
        """
        Compute the reward for grasping both tray handles with the correct fingers.

        Returns:
            float: Reward accumulated for each hand that has both fingers on its handle.
        """
        right_hand_right_finger_on_handle = self._check_collision(
            "right_handle", "right_hand_right_finger"
        )
        right_hand_left_finger_on_handle = self._check_collision(
            "right_handle", "right_hand_left_finger"
        )
        left_hand_right_finger_on_handle = self._check_collision(
            "left_handle", "left_hand_right_finger"
        )
        left_hand_left_finger_on_handle = self._check_collision(
            "left_handle", "left_hand_left_finger"
        )

        reward = 0

        if right_hand_right_finger_on_handle and right_hand_left_finger_on_handle:
            reward += self._grasp_reward
        if left_hand_right_finger_on_handle and left_hand_left_finger_on_handle:
            reward += self._grasp_reward
        return reward

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
        cube_fell_off_tray_cost = self._get_cube_fell_off_tray_cost()
        grasp_reward = self._get_grasp_reward()
        rotation_reward = self._get_gripper_rotation_reward(next_obs)
        reward = (
            handle_distance_reward
            + contact_table_cost
            + ctrl_cost
            # + cube_fell_off_tray_cost
            + grasp_reward
            + rotation_reward
        )

        return reward

    def is_absorbing(self, obs):
        """
        Check if the current state is absorbing.

        Args:
            obs (np.ndarray): The observation of the environment.
        Returns:
            is_absorbing (bool): True if the current state is absorbing, False otherwise.
        """

        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")[0]
        if contact_force > self._contact_threshold:
            return True
        return False

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)

        info["right_handle_pos"] = self._read_data("right_handle_pos")
        info["left_handle_pos"] = self._read_data("left_handle_pos")
        info["right_handle_rot"] = self._read_data("right_handle_rot")
        info["left_handle_rot"] = self._read_data("left_handle_rot")
        info["right_hande_robotiq_hande_end_pos"] = self.obs_helper.get_from_obs(
            obs, "right_hande_robotiq_hande_end_pos"
        )
        info["left_hande_robotiq_hande_end_pos"] = self.obs_helper.get_from_obs(
            obs, "left_hande_robotiq_hande_end_pos"
        )

        info["handle_distance_reward"] = self._get_handle_distance_reward(obs)
        info["contact_table_cost"] = self._get_contact_cost(obs)
        info["ctrl_cost"] = self._get_ctrl_cost(action)
        info["grasp_reward"] = self._get_grasp_reward()
        info["rotation_reward"] = self._get_gripper_rotation_reward(obs)
        return info


if __name__ == "__main__":
    # Action ordering (18 values):
    #   [0:7]   right arm A1-A7
    #   [7:14]  left arm A1-A7
    #   [14:16] left fingers      ← obs indices 16-17 (swap vs obs)
    #   [16:18] right fingers     ← obs indices 14-15 (swap vs obs)
    #
    # Observation ordering:
    #   [0:7]   right arm positions,  [7:14]  left arm positions
    #   [14:16] right finger pos,     [16:18] left finger pos
    #   [18:25] right arm vel,        [25:32] left arm vel
    #   [32:34] right finger vel,     [34:36] left finger vel

    # IK solution: pos (-0.6, -0.207, 0.913) + orientation aligned with handle
    # gripper_y ∥ handle_y (-worldX), gripper_z ∥ handle_x (worldY)
    left_arm_target = np.array(
        [-1.5696, -0.2335, 2.7197, -1.9536, 2.4092, -1.0705, -0.5063]
    )
    left_finger_target = np.array([0.0, 0.0])  # open gripper

    right_arm_target = -left_arm_target
    right_finger_target = np.array([0.0, 0.0])

    Kp_arm = 0.5
    Kp_finger = 0.1

    env = TrayPickUpEnv()
    obs, _ = env.reset()
    env.render()

    step = 0

    while True:
        right_pos = obs[0:7]
        right_finger_pos = obs[14:16]

        left_pos = obs[7:14]
        left_finger_pos = obs[16:18]

        left_vel_cmd = np.clip(Kp_arm * (left_arm_target - left_pos), -1.0, 1.0)
        left_finger_vel_cmd = np.clip(
            Kp_finger * (left_finger_target - left_finger_pos), -0.5, 0.5
        )

        right_vel_cmd = np.clip(Kp_arm * (right_arm_target - right_pos), -1.0, 1.0)

        right_finger_vel_cmd = np.clip(
            Kp_finger * (right_finger_target - right_finger_pos), -0.5, 0.5
        )

        action = np.concatenate(
            [
                right_vel_cmd,  # [0:7]
                left_vel_cmd,  # [7:14]
                left_finger_vel_cmd,  # [14:16]
                right_finger_vel_cmd,  # [16:18]
            ]
        )

        obs, reward, absorbing, info = env.step(action)
        env.render()

        if step % 10 == 0:
            print(f"left_handle_pos    : {np.round(info['left_handle_pos'], 3)}")
            print(
                f"left_grasp_pos     : {np.round(info['left_hande_robotiq_hande_end_pos'], 3)}"
            )
            print(f"handle_dist_reward : {info['handle_distance_reward']:.4f}")
            print(f"rotation_reward    : {info['rotation_reward']:.4f}")
            print(f"grasp_reward       : {info['grasp_reward']:.4f}")
            print(f"contact_cost       : {info['contact_table_cost']:.4f}")
            print(f"ctrl_cost          : {info['ctrl_cost']:.4f}")
            print(f"total_reward       : {reward:.4f}")
            print()

        step += 1
