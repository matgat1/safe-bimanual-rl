import os
import numpy as np
from safe_bimanual_rl.environments.bimanual_table_env import BimanualTableEnv
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box


# Reach a point environment with two arms
class TrayPickUpEnv(BimanualTableEnv):
    """
    A reach environment for two arms, where the goal is to reach the cube on the table.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 1000,
        n_substeps: int = 5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        contact_cost_weight: float = -1e-4,
        cube_distance_weight: float = 1.0,
        cube_touched_reward: float = 3.0,
        contact_threshold: float = 2.0,
        control_cost_weight: float = -1e-4,
        reach_sharpness: float = 0.5,
        cube_displacement_weight: float = -1.0,
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
        actuation_spec = [
            "right_arm_A1_ctrl",
            "right_arm_A2_ctrl",
            "right_arm_A3_ctrl",
            "right_arm_A4_ctrl",
            "right_arm_A5_ctrl",
            "right_arm_A6_ctrl",
            "right_arm_A7_ctrl",
            "left_arm_A1_ctrl",
            "left_arm_A2_ctrl",
            "left_arm_A3_ctrl",
            "left_arm_A4_ctrl",
            "left_arm_A5_ctrl",
            "left_arm_A6_ctrl",
            "left_arm_A7_ctrl",
        ]

        additional_data_spec = [
            ("cube_pos", "cube", ObservationType.BODY_POS),
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
        ]

        collision_groups = [
            ("cube", ["cube"]),
            ("table", ["table_base_link_collision"]),
        ]

        scene_xml = os.path.join(
            os.path.dirname(__file__), "data", "tray_pickup_env.xml"
        )

        self._contact_cost_weight = contact_cost_weight
        self._cube_distance_weight = cube_distance_weight
        self._contact_force_range = contact_force_range
        self._contact_threshold = contact_threshold
        self._cube_touched_reward = cube_touched_reward
        self._control_cost_weight = control_cost_weight
        self._reach_sharpness = reach_sharpness
        self._cube_displacement_weight = cube_displacement_weight
        self._prev_cube_pos = None

        super().__init__(
            scene_xml=scene_xml,
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            actuation_spec=actuation_spec,
            **viewer_params,
        )

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        self.obs_helper.add_obs("rel_cube_pos_right_arm", 3)
        self.obs_helper.add_obs("rel_cube_pos_left_arm", 3)
        self.obs_helper.add_obs("contact_force", 1)

        # Update dimensions of the observation space
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())

        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        cube_pos = self._read_data("cube_pos")

        right_arm_pos = self._read_data("right_hande_robotiq_hande_end_pos")
        left_arm_pos = self._read_data("left_hande_robotiq_hande_end_pos")

        rel_cube_pos_right_arm = cube_pos - right_arm_pos
        rel_cube_pos_left_arm = cube_pos - left_arm_pos

        contact_force = self._get_contact_force(
            "robot", "table", self._contact_force_range
        ) + self._get_contact_force("hand", "table", self._contact_force_range)

        obs = np.concatenate(
            [obs, rel_cube_pos_right_arm, rel_cube_pos_left_arm, contact_force]
        )

        return obs

    def _is_cube_touched(self):
        """
        Check if the cube is touched by each arm independently.

        Returns:
            tuple[bool, bool]: (right_touched, left_touched), each True if the
                corresponding hand is in contact with the cube.
        """
        right_touched = self._check_collision("cube", "right_hand_fingers")
        left_touched = self._check_collision("cube", "left_hand_fingers")
        return right_touched, left_touched

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

    def _get_cube_distance_reward(self, obs):
        """
        Compute the reward based on the distance between the cube and the end effectors.

        Args:
            obs: The observation of the environment.

        Returns:
            reward: The computed reward based on the distance between the cube and the end effectors
        """

        rel_cube_pos_right = self.obs_helper.get_from_obs(obs, "rel_cube_pos_right_arm")
        rel_cube_pos_left = self.obs_helper.get_from_obs(obs, "rel_cube_pos_left_arm")

        right_arm_distance = np.linalg.norm(rel_cube_pos_right)
        left_arm_distance = np.linalg.norm(rel_cube_pos_left)

        reward = (1 - np.tanh(right_arm_distance / self._reach_sharpness)) + (
            1 - np.tanh(left_arm_distance / self._reach_sharpness)
        )

        return self._cube_distance_weight * reward

    def _get_ctrl_cost(self, action):
        ctrl_cost = np.sum(np.square(action))
        return self._control_cost_weight * ctrl_cost

    def _get_cube_push_cost(self):
        cube_pos = self._read_data("cube_pos")
        if self._prev_cube_pos is None:
            self._prev_cube_pos = cube_pos.copy()
            return 0.0
        displacement = np.linalg.norm(cube_pos - self._prev_cube_pos)
        self._prev_cube_pos = cube_pos.copy()
        return self._cube_displacement_weight * displacement

    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute the reward for the reach environment.

        Args:
            obs (np.ndarray): The observation of the environment.
            action (np.ndarray): The action taken by the agent.
        Returns:
            reward (float): The reward for the current state and action.
        """

        cube_hand_distance_reward = self._get_cube_distance_reward(next_obs)
        right_touched, left_touched = self._is_cube_touched()
        cube_touched_reward = (self._cube_touched_reward / 2) * (
            right_touched + left_touched
        )
        contact_table_cost = self._get_contact_cost(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        cube_push_penalty = self._get_cube_push_cost()

        reward = (
            cube_hand_distance_reward
            + cube_touched_reward
            + contact_table_cost
            + ctrl_cost
            + cube_push_penalty
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


if __name__ == "__main__":
    env = TrayPickUpEnv()
    env.reset()
    env.render()
    while True:
        action = np.zeros((14,))
        # action = np.random.uniform(-2.0, 2.0, size=(14,))
        env.step(action)
        env.render()
