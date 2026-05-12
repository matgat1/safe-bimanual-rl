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
        right_grip_pos = self._read_data("right_grip_point_pos")
        left_grip_pos = self._read_data("left_grip_point_pos")
        rel_right = self._read_data("right_handle_pos") - right_grip_pos
        rel_left = self._read_data("left_handle_pos") - left_grip_pos
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
        # Reward equals the weaker finger
        right_grasp = np.minimum(np.tanh(right_finger_right), np.tanh(right_finger_left))
        left_grasp = np.minimum(np.tanh(left_finger_right), np.tanh(left_finger_left))

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
        grasp_contact = self._get_grasp_contact_reward()
        distance = self._get_handle_distance_reward(obs)
        grasp_bonus = self._success_grasp_reward if self._grasp_reached() else 0.0
        info["grasp_contact_reward"] = grasp_contact
        info["distance_reward"] = distance
        info["grasp_bonus"] = grasp_bonus
        info["total_reward"] = grasp_contact + distance + grasp_bonus
        return info


def _grasp_controller(env):
    """
    Jacobian-transpose controller: drives each grip_point site to the handle
    centre (+1 cm overshoot), then closes the fingers.

    The grip_point sites sit at the finger level (z=0.099 in the link frame),
    so targeting them at the handle positions the fingers around the bar.
    Fingers close once the grip_point is within CLOSE_THRESHOLD of the handle.

    Action layout (BimanualTableEnv order):
      [0:7]   right arm A1-A7   ctrlrange [-1, 1]
      [7:14]  left arm A1-A7    ctrlrange [-1, 1]
      [14:16] left fingers       ctrlrange [-0.5, 0.5]
      [16:18] right fingers      ctrlrange [-0.5, 0.5]
    """
    model = env._model
    data = env._data
    nv = model.nv

    right_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grip_point")
    left_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grip_point")
    right_handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_handle")
    left_handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_handle")

    right_arm_dofs = [model.joint(f"right_arm_A{i}").dofadr[0] for i in range(1, 8)]
    left_arm_dofs = [model.joint(f"left_arm_A{i}").dofadr[0] for i in range(1, 8)]

    OVERSHOOT = 0.00    # push 1 cm past the handle centre along approach axis
    CLOSE_THRESHOLD = 0.005  # start closing when grip_point is within 4 cm

    # Overshoot: push 1 cm past the handle in the current approach direction.
    right_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hande_robotiq_hande_link")
    left_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hande_robotiq_hande_link")
    right_z = data.xmat[right_link_id].reshape(3, 3)[:, 2]
    left_z = data.xmat[left_link_id].reshape(3, 3)[:, 2]

    right_target = data.xpos[right_handle_id] + OVERSHOOT * right_z
    left_target = data.xpos[left_handle_id] + OVERSHOOT * left_z

    right_err = right_target - data.site_xpos[right_site_id]
    left_err = left_target - data.site_xpos[left_site_id]

    jacp_right = np.zeros((3, nv))
    jacp_left = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp_right, None, right_site_id)
    mujoco.mj_jacSite(model, data, jacp_left, None, left_site_id)

    K = 4.0
    right_dq = K * jacp_right[:, right_arm_dofs].T @ right_err
    left_dq = K * jacp_left[:, left_arm_dofs].T @ left_err

    right_finger_cmd = -0.5 if np.linalg.norm(right_err) < CLOSE_THRESHOLD else 0.5
    left_finger_cmd = -0.5 if np.linalg.norm(left_err) < CLOSE_THRESHOLD else 0.5

    action = np.zeros(18)
    action[0:7] = np.clip(right_dq, -1.0, 1.0)
    action[7:14] = np.clip(left_dq, -1.0, 1.0)
    action[14:16] = left_finger_cmd
    action[16:18] = right_finger_cmd
    return action


if __name__ == "__main__":
    env = TrayPickUpGraspEnv()
    obs = env.reset()
    env.render()

    while True:
        action = _grasp_controller(env)
        obs, _, _, info = env.step(action)
        env.render()

        print(
            f"\r  distance={info['distance_reward']:+.3f}"
            f"  contact={info['grasp_contact_reward']:+.3f}"
            f"  bonus={info['grasp_bonus']:+.3f}"
            f"  total={info['total_reward']:+.3f}",
            end="",
            flush=True,
        )
