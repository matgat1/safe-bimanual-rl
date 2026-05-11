import os
import mujoco
import numpy as np
from mushroom_rl.rl_utils.spaces import Box
from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv


class TrayPickUpGraspEnv(TrayPickUpBaseEnv):
    """
    Grasp phase: both grippers must close around the tray handles.
    Episodes start with grippers already positioned near the handles (reach phase complete).
    """

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 1000,
        n_substeps: int = 5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        handle_distance_weight: float = 1.0,
        reach_sharpness: float = 0.5,
        grasp_force_threshold: float = 0.5,
        success_grasp_reward: float = 10.0,
        contact_threshold: float = 3.0,
        **viewer_params,
    ):
        """
        Args:
            gamma (float): Discount factor.
            horizon (int): Maximum number of steps per episode.
            n_substeps (int): Number of MuJoCo simulation substeps per environment step.
            contact_force_range (tuple[float, float]): Clipping range for contact forces.
            handle_distance_weight (float): Weight for the handle-distance reward.
            reach_sharpness (float): Controls how sharply the tanh reward drops off with distance.
            grasp_force_threshold (float): Contact force above which the episode is a success.
            success_grasp_reward (float): Bonus reward when both grippers grasp the handles.
            contact_threshold (float): Contact force magnitude above which the episode terminates.
        """
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            contact_force_range=contact_force_range,
            handle_distance_weight=handle_distance_weight,
            reach_sharpness=reach_sharpness,
            **viewer_params,
        )
        self._grasp_force_threshold = grasp_force_threshold
        self._success_grasp_reward = success_grasp_reward
        self._contact_threshold = contact_threshold
        self._last_right_grasp_force = np.zeros(1)
        self._last_left_grasp_force = np.zeros(1)
        self._absorbing_counts = {
            "grasp_reached": 0,
            "contact_force": 0,
        }
        self.init_states_path = os.path.join(
            os.path.dirname(__file__), "data", "initial_states", "grasp_init_states.npz"
        )
        data = np.load(self.init_states_path)
        self._init_states = list(zip(data["qpos"], data["qvel"], data["act"]))

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        self.obs_helper.add_obs("rel_right_handle_pos", 3)
        self.obs_helper.add_obs("rel_left_handle_pos", 3)
        self.obs_helper.add_obs("contact_force", 1)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)
        right_arm_pos = self._read_data("right_hande_robotiq_hande_end_pos")
        left_arm_pos = self._read_data("left_hande_robotiq_hande_end_pos")
        rel_right = self._read_data("right_handle_pos") - right_arm_pos
        rel_left = self._read_data("left_handle_pos") - left_arm_pos
        # Create contact force observation (robot+hand / table)
        contact_force = self._get_contact_force(
            "robot", "table", self._contact_force_range
        ) + self._get_contact_force("hand", "table", self._contact_force_range)
        return np.concatenate([obs, rel_right, rel_left, contact_force])

    def _grasp_reached(self) -> bool:
        """Check if both grippers are applying sufficient force on the handles."""
        return bool(
            self._last_right_grasp_force > self._grasp_force_threshold
            and self._last_left_grasp_force > self._grasp_force_threshold
        )

    def _get_grasp_contact_reward(self):
        right_finger_right = self._get_contact_force(
            "right_hand_right_finger", "right_handle", self._contact_force_range
        )
        right_finger_left = self._get_contact_force(
            "right_hand_left_finger", "right_handle", self._contact_force_range
        )
        left_finger_right = self._get_contact_force(
            "left_hand_right_finger", "left_handle", self._contact_force_range
        )
        left_finger_left = self._get_contact_force(
            "left_hand_left_finger", "left_handle", self._contact_force_range
        )
        # Both fingers must contact the handle: product is zero if either finger misses.
        right_grasp = np.tanh(right_finger_right) * np.tanh(right_finger_left)
        left_grasp = np.tanh(left_finger_right) * np.tanh(left_finger_left)

        # Track the weaker finger on each hand for success detection.
        self._last_right_grasp_force = np.minimum(right_finger_right, right_finger_left)
        self._last_left_grasp_force = np.minimum(left_finger_right, left_finger_left)
        return (right_grasp + left_grasp).item()

    def setup(self, obs):
        super().setup(obs)
        qpos, qvel, act = self._init_states[np.random.randint(len(self._init_states))]
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.act[:] = act
        self._last_right_grasp_force = np.zeros(1)
        self._last_left_grasp_force = np.zeros(1)
        mujoco.mj_forward(self._model, self._data)

    def reward(self, obs, action, next_obs, absorbing):
        handle_distance_reward = self._get_handle_distance_reward(next_obs)
        grasp_contact_reward = self._get_grasp_contact_reward()
        grasp_bonus = self._success_grasp_reward if self._grasp_reached() else 0.0

        reward = handle_distance_reward + grasp_contact_reward + grasp_bonus

        return reward

    def is_absorbing(self, obs):
        if self._grasp_reached():
            self._absorbing_counts["grasp_reached"] += 1
            print("Grasp success")
            return True
        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")[0]
        if contact_force > self._contact_threshold:
            self._absorbing_counts["contact_force"] += 1
            print("Contact force exceeded threshold")
            return True
        return False

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        info["grasp_contact_reward"] = self._get_grasp_contact_reward()
        info["distance_reward"] = self._get_handle_distance_reward(obs)
        return info


if __name__ == "__main__":
    env = TrayPickUpGraspEnv()
    obs = env.reset()
    env.render()
    while True:
        action = np.zeros((18,))
        env.step(action)
        env.render()
