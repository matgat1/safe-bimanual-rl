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

    def __init__(self, gamma: float = 0.99, horizon: int = 1000, n_substeps: int = 5):
        """
        Initialize the reach environment.

        Args:
            gamma (float): The discounting factor of the environment.
            horizon (int): The maximum horizon for the environment
            n_substeps (int): The number of substeps to use by the MuJoCo simulator.
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

        scene_xml = os.path.join(
            os.path.dirname(__file__), "data", "arms_tray_scene.xml"
        )

        super().__init__(
            scene_xml=scene_xml,
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            additional_data_spec=additional_data_spec,
            actuation_spec=actuation_spec,
        )

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        self.obs_helper.add_obs("rel_cube_pos_right_arm", 3)
        self.obs_helper.add_obs("rel_cube_pos_left_arm", 3)

        # Update dimensions of the observation space to include the cube position
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())

        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)

        cube_pos = self._read_data("cube_pos")
        cube_pos = np.array([0.7, 0.1, 0.86499277])
        right_arm_pos = self._read_data("right_hande_robotiq_hande_end_pos")
        left_arm_pos = self._read_data("left_hande_robotiq_hande_end_pos")

        rel_cube_pos_right_arm = cube_pos - right_arm_pos
        rel_cube_pos_left_arm = cube_pos - left_arm_pos

        obs = np.concatenate([obs, rel_cube_pos_right_arm, rel_cube_pos_left_arm])

        return obs

    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute the reward for the reach environment.

        Args:
            obs (np.ndarray): The observation of the environment.
            action (np.ndarray): The action taken by the agent.
        Returns:
            reward (float): The reward for the current state and action.
        """

        rel_cube_pos_right = self.obs_helper.get_from_obs(
            next_obs, "rel_cube_pos_right_arm"
        )
        rel_cube_pos_left = self.obs_helper.get_from_obs(
            next_obs, "rel_cube_pos_left_arm"
        )

        right_arm_distance = np.linalg.norm(rel_cube_pos_right)
        left_arm_distance = np.linalg.norm(rel_cube_pos_left)

        reward = -(right_arm_distance + left_arm_distance)

        return reward

    def is_absorbing(self, obs):
        """
        Check if the current state is absorbing.

        Args:
            obs (np.ndarray): The observation of the environment.
        Returns:
            is_absorbing (bool): True if the current state is absorbing, False otherwise.
        """
        is_absorbing = False
        return is_absorbing


if __name__ == "__main__":
    env = ReachEnv()
    env.reset()
    env.render()
    while True:
        action = np.zeros((14,))
        # action = np.random.uniform(-2.0, 2.0, size=(14,))
        env.step(action)
        env.render()
