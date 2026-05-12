import os
import mujoco
import numpy as np
from mushroom_rl.rl_utils.spaces import Box
from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv


class TrayPickUpLiftEnv(TrayPickUpBaseEnv):
    """
    Lift phase: both arms lift the tray to a target height.
    Episodes start with grippers grasping the handles (grasp phase complete).
    """

    def __init__(
        self,
        gamma: float = 0.99,
        horizon: int = 1000,
        n_substeps: int = 5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        lift_height_target: float = 1.0,
        lift_height_weight: float = 2.0,
        lift_sharpness: float = 0.10,
        grasp_keep_weight: float = 1.0,
        grasp_sharpness: float = 0.05,
        success_lift_reward: float = 50.0,
        ee_max_distance: float = 0.15,
        **viewer_params,
    ):
        """
        Args:
            gamma (float): Discount factor.
            horizon (int): Maximum number of steps per episode.
            n_substeps (int): Number of MuJoCo simulation substeps per environment step.
            contact_force_range (tuple[float, float]): Clipping range for contact forces.
            lift_height_target (float): Target cube height in metres for success.
            lift_height_weight (float): Weight for the lift height reward.
            lift_sharpness (float): tanh sharpness for lift reward (metres above init).
            grasp_keep_weight (float): Weight for the grasp-maintenance reward.
            grasp_sharpness (float): tanh sharpness for EE-to-handle distance (metres).
            success_lift_reward (float): Bonus reward when cube reaches the target height.
            ee_max_distance (float): Max EE-to-handle distance before episode terminates.
        """
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            contact_force_range=contact_force_range,
            **viewer_params,
        )
        self._lift_height_target = lift_height_target
        self._lift_height_weight = lift_height_weight
        self._lift_sharpness = lift_sharpness
        self._grasp_keep_weight = grasp_keep_weight
        self._grasp_sharpness = grasp_sharpness
        self._success_lift_reward = success_lift_reward
        self._ee_max_distance = ee_max_distance
        self._init_cube_z = 0.0
        self._absorbing_counts = {
            "lift_success": 0,
            "grasp_lost": 0,
        }
        self.init_states_path = os.path.join(
            os.path.dirname(__file__), "data", "initial_states", "lift_init_states.npz"
        )
        data = np.load(self.init_states_path)
        self._init_states = list(zip(data["qpos"], data["qvel"], data["act"]))

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        self.obs_helper.add_obs("cube_pos", 3)
        self.obs_helper.add_obs("rel_right_handle_pos", 3)
        self.obs_helper.add_obs("rel_left_handle_pos", 3)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        obs = super()._create_observation(obs)
        cube_pos = self._read_data("cube_pos")
        right_arm_pos = self._read_data("right_hande_robotiq_hande_end_pos")
        left_arm_pos = self._read_data("left_hande_robotiq_hande_end_pos")
        rel_right = self._read_data("right_handle_pos") - right_arm_pos
        rel_left = self._read_data("left_handle_pos") - left_arm_pos
        return np.concatenate([obs, cube_pos, rel_right, rel_left])

    def _get_lift_reward(self, obs):
        cube_z = self.obs_helper.get_from_obs(obs, "cube_pos")[2]
        delta_z = max(cube_z - self._init_cube_z, 0.0)
        return self._lift_height_weight * np.tanh(delta_z / self._lift_sharpness)

    def _get_grasp_keep_reward(self, obs):
        rel_right = self.obs_helper.get_from_obs(obs, "rel_right_handle_pos")
        rel_left = self.obs_helper.get_from_obs(obs, "rel_left_handle_pos")
        d_right = np.linalg.norm(rel_right)
        d_left = np.linalg.norm(rel_left)
        keep = (1 - np.tanh(d_right / self._grasp_sharpness)) + (
            1 - np.tanh(d_left / self._grasp_sharpness)
        )
        return self._grasp_keep_weight * keep

    def _lift_success(self, obs):
        cube_z = self.obs_helper.get_from_obs(obs, "cube_pos")[2]
        return cube_z > self._lift_height_target

    def _grasp_lost(self, obs):
        rel_right = self.obs_helper.get_from_obs(obs, "rel_right_handle_pos")
        rel_left = self.obs_helper.get_from_obs(obs, "rel_left_handle_pos")
        return (
            np.linalg.norm(rel_right) > self._ee_max_distance
            or np.linalg.norm(rel_left) > self._ee_max_distance
        )

    def setup(self, obs):
        super().setup(obs)
        qpos, qvel, act = self._init_states[np.random.randint(len(self._init_states))]
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.act[:] = act
        mujoco.mj_forward(self._model, self._data)
        self._init_cube_z = self._read_data("cube_pos")[2]

    def reward(self, obs, action, next_obs, absorbing):
        lift_reward = self._get_lift_reward(next_obs)
        grasp_keep_reward = self._get_grasp_keep_reward(next_obs)
        success_bonus = (
            self._success_lift_reward if self._lift_success(next_obs) else 0.0
        )
        return lift_reward + grasp_keep_reward + success_bonus

    def is_absorbing(self, obs):
        if self._lift_success(obs):
            self._absorbing_counts["lift_success"] += 1
            print("Lift success")
            return True
        if self._grasp_lost(obs):
            self._absorbing_counts["grasp_lost"] += 1
            print("Grasp lost")
            return True
        return False

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        info["lift_reward"] = self._get_lift_reward(obs)
        info["grasp_keep_reward"] = self._get_grasp_keep_reward(obs)
        info["success_bonus"] = (
            self._success_lift_reward if self._lift_success(obs) else 0.0
        )
        info["total_reward"] = (
            info["lift_reward"] + info["grasp_keep_reward"] + info["success_bonus"]
        )
        return info


if __name__ == "__main__":
    env = TrayPickUpLiftEnv()
    obs = env.reset()
    env.render()

    while True:
        action = np.zeros(env.info.action_space.shape[0])
        obs, _, _, info = env.step(action)
        env.render()

        print(
            f"\r  lift={info['lift_reward']:+.3f}"
            f"  grasp_keep={info['grasp_keep_reward']:+.3f}"
            f"  bonus={info['success_bonus']:+.3f}"
            f"  total={info['total_reward']:+.3f}",
            end="",
            flush=True,
        )
