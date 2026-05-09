import mujoco
import numpy as np
from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv


class TrayPickUpReachEnv(TrayPickUpBaseEnv):
    """
    Reach phase: both grippers must reach the tray handles with correct orientation.
    Episode ends on success or on excessive contact force.
    """

    def reward(self, obs, action, next_obs, absorbing):
        handle_distance_reward = self._get_handle_distance_reward(next_obs)
        contact_table_cost = self._get_contact_cost(next_obs)
        ctrl_cost = self._get_ctrl_cost(action)
        rotation_reward = self._get_gripper_rotation_reward(next_obs)
        tray_push_penalty = self._tray_push_penalty if self._tray_pushed() else 0.0
        both_reached = self._position_reached(next_obs) and self._orientation_reached(
            next_obs
        )
        position_bonus = self._success_position_reward if both_reached else 0.0
        orientation_bonus = self._success_orientation_reward if both_reached else 0.0

        return (
            handle_distance_reward
            + contact_table_cost
            + ctrl_cost
            + rotation_reward
            + tray_push_penalty
            + position_bonus
            + orientation_bonus
        )

    def is_absorbing(self, obs):
        if self._position_reached(obs) and self._orientation_reached(obs):
            self._absorbing_counts["position_reached"] += 1
            print("Success! Episode will terminate.")
            return True
        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")[0]
        if contact_force > self._contact_threshold:
            self._absorbing_counts["contact_force"] += 1
            print(
                f"Contact force {contact_force:.2f} exceeded threshold! Episode will terminate."
            )
            return True
        if self._tray_pushed():
            self._absorbing_counts["tray_pushed"] += 1
            print("Tray pushed! Episode will terminate.")
            return True
        return False


if __name__ == "__main__":
    env = TrayPickUpReachEnv()
    model = env._model
    space = env.info.action_space

    right_ee_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_hande_robotiq_hande_end"
    )
    left_ee_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_hande_robotiq_hande_end"
    )
    right_handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_handle")
    left_handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_handle")

    right_grasp_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_target"
    )
    left_grasp_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_target"
    )

    right_dof_ids = np.array(
        [model.joint(f"right_arm_A{i}").dofadr[0] for i in range(1, 8)]
    )
    left_dof_ids = np.array(
        [model.joint(f"left_arm_A{i}").dofadr[0] for i in range(1, 8)]
    )

    RIGHT_ACT = slice(0, 7)
    LEFT_ACT = slice(7, 14)
    nv = model.nv
    KP_POS = 5.0
    KP_ROT = 2.0
    LAM = 0.01

    def _compute_action():
        data = env._data
        rel_right = data.site_xpos[right_grasp_site] - data.xpos[right_ee_id]
        rel_left = data.site_xpos[left_grasp_site] - data.xpos[left_ee_id]

        right_handle_mat = data.xmat[right_handle_id].reshape(3, 3)
        left_handle_mat = data.xmat[left_handle_id].reshape(3, 3)
        right_gripper_mat = data.xmat[right_ee_id].reshape(3, 3)
        left_gripper_mat = data.xmat[left_ee_id].reshape(3, 3)

        r_hy = right_handle_mat[:, 1] * np.sign(
            np.dot(right_gripper_mat[:, 1], right_handle_mat[:, 1])
        )
        l_hy = left_handle_mat[:, 1] * np.sign(
            np.dot(left_gripper_mat[:, 1], left_handle_mat[:, 1])
        )
        right_rot_err = np.cross(right_gripper_mat[:, 1], r_hy) + np.cross(
            right_gripper_mat[:, 2], -right_handle_mat[:, 0]
        )
        left_rot_err = np.cross(left_gripper_mat[:, 1], l_hy) + np.cross(
            left_gripper_mat[:, 2], left_handle_mat[:, 0]
        )

        Jp_r, Jr_r = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBody(model, data, Jp_r, Jr_r, right_ee_id)
        Jp_l, Jr_l = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBody(model, data, Jp_l, Jr_l, left_ee_id)

        J_r = np.vstack([Jp_r[:, right_dof_ids], Jr_r[:, right_dof_ids]])
        J_l = np.vstack([Jp_l[:, left_dof_ids], Jr_l[:, left_dof_ids]])
        q_dot_r = J_r.T @ np.linalg.solve(
            J_r @ J_r.T + LAM * np.eye(6),
            np.concatenate([KP_POS * rel_right, KP_ROT * right_rot_err]),
        )
        q_dot_l = J_l.T @ np.linalg.solve(
            J_l @ J_l.T + LAM * np.eye(6),
            np.concatenate([KP_POS * rel_left, KP_ROT * left_rot_err]),
        )

        action = np.zeros(space.shape[0])
        action[RIGHT_ACT] = np.clip(
            q_dot_r, space.low[RIGHT_ACT], space.high[RIGHT_ACT]
        )
        action[LEFT_ACT] = np.clip(q_dot_l, space.low[LEFT_ACT], space.high[LEFT_ACT])
        return action

    obs = env.reset()
    env.render()
    step = 0
    while True:
        action = _compute_action()
        obs, reward, absorbing, info = env.step(action)
        if absorbing:
            obs = env.reset()
        print(
            f"Step {step:4d} | reward: {reward:+.3f} | "
            f"dist: {info['handle_distance_reward']:+.3f}  "
            f"rot: {info['rotation_reward']:+.3f}  "
            f"ctrl: {info['ctrl_cost']:+.3f}  "
            f"contact: {info['contact_table_cost']:+.3f}  "
            f"tray: {info['tray_push_penalty']:+.3f}  "
            f"pos_bonus: {info['position_bonus']:+.3f}  "
            f"ori_bonus: {info['orientation_bonus']:+.3f}"
        )
        step += 1
        env.render()
