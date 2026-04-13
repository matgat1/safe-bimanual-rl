import os
import numpy as np
from safe_bimanual_rl.environments.bimanual_table_env import BimanualTableEnv
from mushroom_rl.environments.mujoco import ObservationType
from mushroom_rl.rl_utils.spaces import Box


# Reach a point environment with two arms
class ReachEnv(BimanualTableEnv):
    """
    A reach environment for two arms, where the goal is to reach the cube on the table.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 1000,
        n_substeps: int = 5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        contact_cost_weight: float = -0.1,
        cube_distance_weight: float = 1.0,
        cube_touched_reward: float = 10.0,
        contact_threshold: float = 100,
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
            os.path.dirname(__file__), "data", "arms_tray_scene.xml"
        )

        self._contact_cost_weight = contact_cost_weight
        self._cube_distance_weight = cube_distance_weight
        self._contact_force_range = contact_force_range
        self._contact_threshold = contact_threshold
        self._cube_touched_reward = cube_touched_reward

        super().__init__(
            scene_xml=scene_xml,
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            actuation_spec=actuation_spec,
        )

    # TODO Create function is cube touched (use it in reward and absorbing state)

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
        # cube_pos = np.array([-0.92, 0.0, 0.86499277])
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
        Check if the cube is touched by either of the arms.

        Args:
            obs: The observation of the environment.
        Returns:
            is_touched: True if the cube is touched, False otherwise.
        """
        is_touched = self._check_collision("cube", "hand")
        return is_touched

    def _get_contact_cost(self, obs):
        """
        Compute the cost based on the contact force.

        Args:
            obs: The observation of the environment.

        Returns:
            cost: The computed cost based on the contact force.
        """

        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")
        return self._contact_cost_weight * contact_force

    def _get_cube_distance_cost(self, obs):
        """
        Compute the cost based on the distance between the cube and the end effectors.

        Args:
            obs: The observation of the environment.

        Returns:
            cost: The computed cost based on the distance between the cube and the end effectors.
        """

        rel_cube_pos_right = self.obs_helper.get_from_obs(obs, "rel_cube_pos_right_arm")
        rel_cube_pos_left = self.obs_helper.get_from_obs(obs, "rel_cube_pos_left_arm")

        right_arm_distance = np.linalg.norm(rel_cube_pos_right)
        left_arm_distance = np.linalg.norm(rel_cube_pos_left)

        cost = -(right_arm_distance + left_arm_distance)

        return self._cube_distance_weight * cost

    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute the reward for the reach environment.

        Args:
            obs (np.ndarray): The observation of the environment.
            action (np.ndarray): The action taken by the agent.
        Returns:
            reward (float): The reward for the current state and action.
        """

        cube_hand_distance_reward = self._get_cube_distance_cost(next_obs)
        cube_touched_reward = (
            self._cube_touched_reward if self._is_cube_touched() else 0.0
        )
        contact_table_cost = self._get_contact_cost(next_obs)[0]

        reward = cube_hand_distance_reward + cube_touched_reward + contact_table_cost

        return reward

    def is_absorbing(self, obs):
        """
        Check if the current state is absorbing.

        Args:
            obs (np.ndarray): The observation of the environment.
        Returns:
            is_absorbing (bool): True if the current state is absorbing, False otherwise.
        """

        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")
        cube_touched = self._is_cube_touched()
        if (contact_force > self._contact_threshold) or cube_touched:
            return True

        return False


if __name__ == "__main__":
    env = ReachEnv()
    env.reset()
    env.render()
    while True:
        action = np.zeros((14,))
        # action = np.random.uniform(-2.0, 2.0, size=(14,))
        env.step(action)
        env.render()
