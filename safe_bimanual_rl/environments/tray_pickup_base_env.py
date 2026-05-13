import os
import numpy as np
from mushroom_rl.rl_utils.spaces import Box
from safe_bimanual_rl.environments.bimanual_table_env import BimanualTableEnv
from mushroom_rl.environments.mujoco import ObservationType


class TrayPickUpBaseEnv(BimanualTableEnv):
    """
    Base environment for the bimanual tray pick-up task.

    Provides the shared scene setup, collision groups, additional data specs,
    and reward helper methods. Each phase environment defines its own
    observation space and reward.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 1000,
        n_substeps: int = 5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        contact_cost_weight: float = -1e-4,
        handle_distance_weight: float = 1.0,
        reach_sharpness: float = 0.5,
        control_cost_weight: float = -1e-4,
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
            reach_sharpness (float): Controls how sharply the tanh reward drops off with distance.
            control_cost_weight (float): Weight penalizing large actions (negative).
        """
        additional_data_spec = [
            ("cube_pos", "cube", ObservationType.BODY_POS),
            ("tray_pos", "tray", ObservationType.BODY_POS),
            ("right_grasp_target_pos", "right_grasp_target", ObservationType.SITE_POS),
            ("left_grasp_target_pos", "left_grasp_target", ObservationType.SITE_POS),
            ("right_grip_point_pos", "right_grip_point", ObservationType.SITE_POS),
            ("left_grip_point_pos", "left_grip_point", ObservationType.SITE_POS),
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
        self._reach_sharpness = reach_sharpness
        self._control_cost_weight = control_cost_weight

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
        self.obs_helper.add_obs("tray_contact_force", 1)
        self.obs_helper.add_obs("cube_pos", 3)
        self.obs_helper.add_obs("right_handle_rot", 4)
        self.obs_helper.add_obs("left_handle_rot", 4)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)
        right_grip_pos = self._read_data("right_grip_point_pos")
        left_grip_pos = self._read_data("left_grip_point_pos")
        rel_right = self._read_data("right_handle_pos") - right_grip_pos
        rel_left = self._read_data("left_handle_pos") - left_grip_pos
        contact_force = self._get_contact_force(
            "robot", "table", self._contact_force_range
        ) + self._get_contact_force("hand", "table", self._contact_force_range)
        tray_contact_force = self._get_contact_force(
            "tray", "table", self._contact_force_range
        ) - 0.9  # subtract gravity baseline so the observation is ~0 at rest
        cube_pos = self._read_data("cube_pos")
        right_handle_rot = self._read_data("right_handle_rot")
        left_handle_rot = self._read_data("left_handle_rot")
        return np.concatenate(
            [
                obs,
                rel_right,
                rel_left,
                contact_force,
                tray_contact_force,
                cube_pos,
                right_handle_rot,
                left_handle_rot,
            ]
        )

    def _get_contact_cost(self, obs):
        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")[0]
        return self._contact_cost_weight * contact_force

    def _get_handle_distance_reward(self, obs):
        rel_right = self.obs_helper.get_from_obs(obs, "rel_right_handle_pos")
        rel_left = self.obs_helper.get_from_obs(obs, "rel_left_handle_pos")
        reward = (1 - np.tanh(np.linalg.norm(rel_right) / self._reach_sharpness)) + (
            1 - np.tanh(np.linalg.norm(rel_left) / self._reach_sharpness)
        )
        return self._handle_distance_weight * reward

    def _get_ctrl_cost(self, action):
        return self._control_cost_weight * np.sum(np.square(action))
