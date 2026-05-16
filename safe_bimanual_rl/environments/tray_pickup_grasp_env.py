import os
import mujoco
import numpy as np
from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv


class TrayPickUpGraspEnv(TrayPickUpBaseEnv):
    """
    Grasp and lift phase: both grippers must close around the tray handles and lift it.
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
        contact_cost_weight: float = -1e-4,
        tray_contact_threshold: float = 3.0,
        tray_contact_cost_weight: float = -1e-4,
        lift_height_weight: float = 2.0,
        lift_sharpness: float = 0.1,
        lift_target_height: float = 0.5,
        success_lift_reward: float = 100.0,
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
            grasp_force_threshold (float): Contact force threshold for grasp quality gate.
            success_grasp_reward (float): Bonus reward when both grippers first grasp the handles.
            contact_threshold (float): Robot/hand-table contact force magnitude above which the episode terminates.
            contact_cost_weight (float): Weight applied to the robot/hand-table contact force cost (negative).
            tray_contact_threshold (float): Tray-table contact force (baseline-subtracted) above which the episode terminates.
            tray_contact_cost_weight (float): Weight applied to the tray-table contact force cost (negative).
            lift_height_weight (float): Weight for the continuous tray height reward.
            lift_sharpness (float): Tanh sharpness for the height reward (meters).
            lift_target_height (float): Height above initial position (m) that triggers lift success.
            success_lift_reward (float): Bonus reward when the tray is lifted to target height.
        """
        super().__init__(
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            contact_force_range=contact_force_range,
            contact_cost_weight=contact_cost_weight,
            handle_distance_weight=handle_distance_weight,
            reach_sharpness=reach_sharpness,
            **viewer_params,
        )
        self._grasp_force_threshold = grasp_force_threshold
        self._success_grasp_reward = success_grasp_reward
        self._contact_threshold = contact_threshold
        self._tray_contact_threshold = tray_contact_threshold
        self._tray_contact_cost_weight = tray_contact_cost_weight
        self._lift_height_weight = lift_height_weight
        self._lift_sharpness = lift_sharpness
        self._lift_target_height = lift_target_height
        self._success_lift_reward = success_lift_reward
        self._last_right_grasp_force = np.zeros(1)
        self._last_left_grasp_force = np.zeros(1)
        self._initial_cube_height = None
        self._grasp_bonus_given = False
        self._absorbing_counts = {
            "lift_reached": 0,
            "contact_force": 0,
            "tray_contact_force": 0,
            "grasp_reached": 0,
        }
        self.init_states_path = os.path.join(
            os.path.dirname(__file__), "data", "initial_states", "grasp_init_states.npz"
        )
        data = np.load(self.init_states_path)
        self._init_states = list(zip(data["qpos"], data["qvel"], data["act"]))

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
        right_grasp = np.minimum(
            np.tanh(right_finger_right), np.tanh(right_finger_left)
        )
        left_grasp = np.minimum(np.tanh(left_finger_right), np.tanh(left_finger_left))

        # Track the weaker finger on each hand: both fingers must contact for success.
        self._last_right_grasp_force = np.maximum(right_finger_right, right_finger_left)
        self._last_left_grasp_force = np.maximum(left_finger_right, left_finger_left)

        return (2 * np.minimum(right_grasp, left_grasp)).item()

    def _lift_reached(self) -> bool:
        cube_z = self._read_data("cube_pos")[2]
        return bool(cube_z - self._initial_cube_height > self._lift_target_height)

    def _get_lift_reward(self):
        cube_z = self._read_data("cube_pos")[2]
        height = max(0.0, cube_z - self._initial_cube_height)
        return self._lift_height_weight * np.tanh(height / self._lift_sharpness)

    def _get_tray_contact_cost(self, obs):
        # observation already has the gravity baseline subtracted (~0 at rest)
        tray_contact_force = self.obs_helper.get_from_obs(obs, "tray_contact_force")[0]
        return self._tray_contact_cost_weight * max(0.0, tray_contact_force)

    def setup(self, obs):
        super().setup(obs)
        qpos, qvel, act = self._init_states[np.random.randint(len(self._init_states))]
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.act[:] = act
        self._last_right_grasp_force = np.zeros(1)
        self._last_left_grasp_force = np.zeros(1)
        self._grasp_bonus_given = False
        mujoco.mj_forward(self._model, self._data)
        self._initial_cube_height = self._read_data("cube_pos")[2]

    def reward(self, obs, action, next_obs, absorbing):
        handle_distance_reward = self._get_handle_distance_reward(next_obs)
        grasp_contact_reward = self._get_grasp_contact_reward()
        lift_reward = self._get_lift_reward()
        lift_bonus = self._success_lift_reward if self._lift_reached() else 0.0
        contact_table_cost = self._get_contact_cost(next_obs)
        tray_contact_cost = self._get_tray_contact_cost(next_obs)

        # One-time bonus the first time both arms achieve a solid grasp.
        grasp_bonus = 0.0
        if not self._grasp_bonus_given and self._grasp_reached():
            grasp_bonus = self._success_grasp_reward
            self._grasp_bonus_given = True
            self._absorbing_counts["grasp_reached"] += 1

        reward = (
            handle_distance_reward
            + grasp_contact_reward
            + grasp_bonus
            + lift_reward
            + lift_bonus
            + contact_table_cost
            + tray_contact_cost
        )
        return reward

    def is_absorbing(self, obs):
        if self._lift_reached():
            self._absorbing_counts["lift_reached"] += 1
            print("Lift success")
            return True
        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")[0]
        if contact_force > self._contact_threshold:
            self._absorbing_counts["contact_force"] += 1
            print("Contact force exceeded threshold")
            return True
        tray_contact_force = self.obs_helper.get_from_obs(obs, "tray_contact_force")[0]
        if tray_contact_force > self._tray_contact_threshold:
            self._absorbing_counts["tray_contact_force"] += 1
            print("Tray contact force exceeded threshold")
            return True
        return False

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        cube_z = self._read_data("cube_pos")[2]
        info["grasp_contact_reward"] = self._get_grasp_contact_reward()
        info["distance_reward"] = self._get_handle_distance_reward(obs)
        info["lift_reward"] = self._get_lift_reward()
        info["cube_height"] = cube_z - self._initial_cube_height
        info["grasp_reached"] = self._grasp_reached()
        info["lift_reached"] = self._lift_reached()
        return info


def _make_grasp_controller():
    """
    Factory returning a two-phase Jacobian-transpose controller.

    Phase 1 – Grasp: drives each grip_point site to the handle centre, then
    closes the fingers.

    Phase 2 – Lift: once both grippers confirm contact (env._grasp_reached()),
    keeps fingers fully closed and drives both grippers straight up by 50 cm
    above their position at the moment grasp was confirmed.

    All controller state lives inside this closure; nothing is stored on env.

    Action layout (BimanualTableEnv order):
      [0:7]   right arm A1-A7   ctrlrange [-1, 1]
      [7:14]  left arm A1-A7    ctrlrange [-1, 1]
      [14:16] left fingers       ctrlrange [-0.5, 0.5]
      [16:18] right fingers      ctrlrange [-0.5, 0.5]
    """
    lift_target_z = None

    def controller(env):
        nonlocal lift_target_z

        model = env._model
        data = env._data
        nv = model.nv

        right_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_grip_point"
        )
        left_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_grip_point"
        )
        right_handle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_handle"
        )
        left_handle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_handle"
        )

        right_arm_dofs = [model.joint(f"right_arm_A{i}").dofadr[0] for i in range(1, 8)]
        left_arm_dofs = [model.joint(f"left_arm_A{i}").dofadr[0] for i in range(1, 8)]

        OVERSHOOT = 0.00
        CLOSE_THRESHOLD = 0.005
        LIFT_HEIGHT = 0.5  # 50 cm

        # Latch into lift phase on first grasp detection — never revert.
        if lift_target_z is None and env._grasp_reached():
            lift_target_z = data.xpos[right_handle_id][2] + LIFT_HEIGHT

        if lift_target_z is not None:
            # ── Phase 2: lift ─────────────────────────────────────────────
            right_err = np.array(
                [0.0, 0.0, lift_target_z - data.site_xpos[right_site_id][2]]
            )
            left_err = np.array(
                [0.0, 0.0, lift_target_z - data.site_xpos[left_site_id][2]]
            )
            right_finger_cmd = -0.5
            left_finger_cmd = -0.5
            K = 2.0

        else:
            # ── Phase 1: grasp ────────────────────────────────────────────
            right_link_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "right_hande_robotiq_hande_link"
            )
            left_link_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "left_hande_robotiq_hande_link"
            )
            right_z = data.xmat[right_link_id].reshape(3, 3)[:, 2]
            left_z = data.xmat[left_link_id].reshape(3, 3)[:, 2]

            right_err = (
                data.xpos[right_handle_id]
                + OVERSHOOT * right_z
                - data.site_xpos[right_site_id]
            )
            left_err = (
                data.xpos[left_handle_id]
                + OVERSHOOT * left_z
                - data.site_xpos[left_site_id]
            )

            right_finger_cmd = (
                -0.5 if np.linalg.norm(right_err) < CLOSE_THRESHOLD else 0.5
            )
            left_finger_cmd = (
                -0.5 if np.linalg.norm(left_err) < CLOSE_THRESHOLD else 0.5
            )
            K = 4.0

        # ── Jacobian-transpose IK ─────────────────────────────────────────
        jacp_right = np.zeros((3, nv))
        jacp_left = np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, jacp_right, None, right_site_id)
        mujoco.mj_jacSite(model, data, jacp_left, None, left_site_id)

        right_dq = K * jacp_right[:, right_arm_dofs].T @ right_err
        left_dq = K * jacp_left[:, left_arm_dofs].T @ left_err

        action = np.zeros(18)
        action[0:7] = np.clip(right_dq, -1.0, 1.0)
        action[7:14] = np.clip(left_dq, -1.0, 1.0)
        action[14:16] = left_finger_cmd
        action[16:18] = right_finger_cmd
        return action

    return controller


_grasp_controller = _make_grasp_controller()


if __name__ == "__main__":
    env = TrayPickUpGraspEnv()
    obs = env.reset()
    env.render()

    was_grasping = False

    while True:
        action = _grasp_controller(env)
        obs, reward, absorbing, info = env.step(action)
        env.render()

        grasping = info["grasp_reached"]
        if grasping and not was_grasping:
            print("\n  [GRASP OK] démarrage du lift")
        was_grasping = grasping

        print(
            f"\r  reward={reward:+.3f}"
            f"  dist={info['distance_reward']:+.3f}"
            f"  contact={info['grasp_contact_reward']:+.3f}"
            f"  lift={info['lift_reward']:+.3f}"
            f"  cube_h={info['cube_height']:+.3f}"
            f"  grasp={'Y' if grasping else 'N'}"
            f"  lift_ok={'Y' if info['lift_reached'] else 'N'}",
            end="",
            flush=True,
        )
