"""
Generate start states for the tray pick-up along a scripted (differential IK) demonstration.

From every state of ``grasp_init_states_0.npz`` (grippers open, ~10 cm from the handles):
    1. both grip points are moved onto their handle with damped least-squares IK,
       keeping the gripper orientation;
    2. both grippers are closed;
    3. both grip points are moved up together, which lifts the tray.

States are recorded at each stage, with their stage index:
    0 near:     the input state (grippers open, near the handles)
    1 pregrasp: grip points on the handles, grippers open
    2 grasped:  both handles squeezed by both fingers
    3 lifting:  tray held and lifted, recorded every ``--lift_record_every`` m

Only demonstrations where the grasp holds are kept. These states are only used as
starting points of part of the training episodes of the single PPO policy
(TrayPickUpWarpEnv ``start_states_file``), the policy is never trained on the script actions.

Usage:
    python -m safe_bimanual_rl.utils.generate_pickup_start_states
"""

import argparse
import os

import mujoco
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "environments", "data")
SIDES = ("right", "left")
FINGERS = ("left_finger", "right_finger")

STAGE_NEAR, STAGE_PREGRASP, STAGE_GRASPED, STAGE_LIFTING = range(4)
STAGE_NAMES = ["near", "pregrasp", "grasped", "lifting"]


class ScriptedPickup:
    def __init__(self, model, n_substeps=5, max_joint_vel=0.5, finger_speed=0.2, damping=0.05):
        self.m = model
        self.d = mujoco.MjData(model)
        self.n_substeps = n_substeps
        self.max_joint_vel = max_joint_vel
        self.finger_speed = finger_speed
        self.damping = damping
        self.limit_margin = 0.02

        self.arm = {}
        for side in SIDES:
            joints = [model.joint(f"{side}_arm_A{i}") for i in range(1, 8)]
            self.arm[side] = {
                "dof": np.array([j.dofadr[0] for j in joints]),
                "qpos": np.array([j.qposadr[0] for j in joints]),
                "range": np.array([j.range for j in joints]),
                "ctrl": np.array([model.actuator(f"{j.name}_ctrl").id for j in joints]),
                "site": model.site(f"{side}_grip_point").id,
                "handle": model.body(f"{side}_handle").id,
                "finger_ctrl": np.array(
                    [
                        model.actuator(f"{side}_hande_robotiq_hande_{f}_joint_ctrl").id
                        for f in FINGERS
                    ]
                ),
                "finger_geoms": [
                    model.geom(f"{side}_hande_robotiq_hande_{f}_collision").id for f in FINGERS
                ],
                "bar": model.geom(f"{side}_handle_bar").id,
            }
        self.cube = model.body("cube").id
        self.tray = model.body("tray").id
        self.force = np.zeros(6)

    # ------------------------------------------------------------------
    def load(self, qpos, qvel, act):
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        self.d.act[:] = act
        self.d.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def state(self):
        return self.d.qpos.copy(), self.d.qvel.copy(), self.d.act.copy()

    def grip_pos(self, side):
        return self.d.site_xpos[self.arm[side]["site"]].copy()

    def grip_rot(self, side):
        return self.d.site_xmat[self.arm[side]["site"]].reshape(3, 3).copy()

    def handle_pos(self, side):
        return self.d.xpos[self.arm[side]["handle"]].copy()

    def handle_dist(self, side):
        return np.linalg.norm(self.handle_pos(side) - self.grip_pos(side))

    def cube_z(self):
        return self.d.xpos[self.cube][2]

    def tray_angle(self):
        z_axis = self.d.xmat[self.tray].reshape(3, 3)[:, 2]
        return np.arccos(np.clip(abs(z_axis[2]), -1.0, 1.0))

    def finger_forces(self, side):
        """Normal force of each finger of the hand on its handle bar."""
        arm = self.arm[side]
        forces = []
        for finger_geom in arm["finger_geoms"]:
            total = 0.0
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                if {c.geom1, c.geom2} == {finger_geom, arm["bar"]}:
                    mujoco.mj_contactForce(self.m, self.d, i, self.force)
                    total += self.force[0]
            forces.append(total)
        return np.array(forces)

    def grasping(self, side, force_threshold, grasp_distance):
        return (
            self.finger_forces(side).min() > force_threshold
            and self.handle_dist(side) < grasp_distance
        )

    # ------------------------------------------------------------------
    def _ik_ctrl(self, side, target_pos, target_rot, gain=4.0):
        """Joint velocities moving the grip point to the target pose (damped least squares)."""
        arm = self.arm[side]
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, arm["site"])
        J = np.vstack([jacp[:, arm["dof"]], jacr[:, arm["dof"]]])

        pos_err = target_pos - self.grip_pos(side)
        rot_err_mat = target_rot @ self.grip_rot(side).T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot_err_mat.flatten())
        rot_err = np.zeros(3)
        mujoco.mju_quat2Vel(rot_err, quat, 1.0)

        # The handle bars are capsules along the world x axis: the rotation about that
        # axis is left free, which frees a degree of freedom for the position
        J = np.delete(J, 3, axis=0)
        err = gain * np.concatenate([pos_err, rot_err[1:]])

        # Joints at a limit (e.g. the elbow, fully bent in the input states) that the
        # solution pushes further are removed from the Jacobian and the IK is solved again
        q = self.d.qpos[arm["qpos"]]
        low, high = arm["range"][:, 0] + self.limit_margin, arm["range"][:, 1] - self.limit_margin
        free = np.ones(len(q), dtype=bool)
        for _ in range(len(q)):
            Jf = J * free
            qdot = Jf.T @ np.linalg.solve(Jf @ Jf.T + self.damping**2 * np.eye(len(err)), err)
            blocked = free & (((q <= low) & (qdot < 0)) | ((q >= high) & (qdot > 0)))
            if not blocked.any():
                break
            free &= ~blocked
        return np.clip(qdot * free, -self.max_joint_vel, self.max_joint_vel)

    def step(self, targets, gripper):
        """
        One environment step (n_substeps simulation steps).

        Args:
            targets (dict): side -> (target_pos, target_rot), or None to hold the arm.
            gripper (str): "open" or "close", applied to both hands.
        """
        finger_vel = self.finger_speed if gripper == "open" else -self.finger_speed  # closed at q=0
        for side in SIDES:
            arm = self.arm[side]
            target = targets.get(side)
            self.d.ctrl[arm["ctrl"]] = 0.0 if target is None else self._ik_ctrl(side, *target)
            self.d.ctrl[arm["finger_ctrl"]] = finger_vel
        mujoco.mj_step(self.m, self.d, self.n_substeps)
        return np.isfinite(self.d.qpos).all()


def demonstrate(sim, qpos, qvel, act, args):
    """Run one scripted pick-up, returns a list of (stage, qpos, qvel, act, cube_rest_z)."""
    sim.load(qpos, qvel, act)
    cube_rest_z = sim.cube_z()
    records = [(STAGE_NEAR, *sim.state(), cube_rest_z)]
    rot = {side: sim.grip_rot(side) for side in SIDES}

    # 1. Move the grip points onto the handles, grippers open: first to a point backed
    # off along the gripper axis, then straight along that axis
    for backoff, tol in ((args.backoff, args.backoff_tol), (-args.depth, args.pregrasp_tol)):
        for _ in range(args.approach_steps):
            targets = {
                side: (sim.handle_pos(side) - backoff * rot[side][:, 2], rot[side])
                for side in SIDES
            }
            if not sim.step(targets, "open"):
                return [], "unstable"
            errors = [np.linalg.norm(targets[side][0] - sim.grip_pos(side)) for side in SIDES]
            if max(errors) < tol:
                break
        else:
            return [], "approach"
    for _ in range(args.settle_steps):  # stop the arms before recording
        sim.step({side: (sim.handle_pos(side), rot[side]) for side in SIDES}, "open")
    records.append((STAGE_PREGRASP, *sim.state(), cube_rest_z))

    # 2. Close the grippers, arms holding their position
    hold = {side: (sim.grip_pos(side), rot[side]) for side in SIDES}
    for _ in range(args.close_steps):
        if not sim.step(hold, "close"):
            return [], "unstable"
    if not all(
        sim.grasping(side, args.grasp_force_threshold, args.grasp_distance) for side in SIDES
    ):
        return [], "grasp"
    records.append((STAGE_GRASPED, *sim.state(), cube_rest_z))

    # 3. Lift both grip points together
    start = {side: sim.grip_pos(side) for side in SIDES}
    next_record = args.lift_record_every
    lost_steps = 0
    for k in range(1, args.lift_steps + 1):
        dz = min(k * args.lift_speed * sim.m.opt.timestep * sim.n_substeps, args.lift_height)
        targets = {side: (start[side] + np.array([0.0, 0.0, dz]), rot[side]) for side in SIDES}
        if not sim.step(targets, "close"):
            return [], "unstable"
        held = all(
            sim.grasping(side, args.grasp_force_threshold, args.grasp_distance) for side in SIDES
        )
        # Finger contacts flicker on the capsule bars: only a grasp lost for several
        # consecutive steps ends the demonstration (keeping the states recorded so far)
        lost_steps = 0 if held else lost_steps + 1
        if lost_steps > args.max_lost_steps or sim.tray_angle() > args.max_tray_angle:
            return records, "lift"
        height = sim.cube_z() - cube_rest_z
        if held and height >= next_record:
            records.append((STAGE_LIFTING, *sim.state(), cube_rest_z))
            next_record += args.lift_record_every
        if dz >= args.lift_height and height >= args.lift_height - 0.02:
            break
    return records, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="grasp_init_states_0.npz")
    parser.add_argument("--output", default="pickup_start_states.npz")
    parser.add_argument("--approach_steps", type=int, default=150)
    parser.add_argument("--backoff", type=float, default=0.06, help="pre-grasp offset (m)")
    parser.add_argument("--backoff_tol", type=float, default=0.01)
    parser.add_argument(
        "--depth", type=float, default=-0.005, help="grip point target beyond the handle center (m)"
    )
    parser.add_argument("--pregrasp_tol", type=float, default=0.004)
    parser.add_argument("--settle_steps", type=int, default=5)
    parser.add_argument("--close_steps", type=int, default=20)
    parser.add_argument("--lift_steps", type=int, default=400)
    parser.add_argument("--lift_speed", type=float, default=0.05, help="m/s")
    parser.add_argument("--max_lost_steps", type=int, default=3)
    parser.add_argument("--lift_height", type=float, default=0.40)
    parser.add_argument("--lift_record_every", type=float, default=0.025)
    parser.add_argument("--grasp_force_threshold", type=float, default=1.0)
    parser.add_argument("--grasp_distance", type=float, default=0.03)
    parser.add_argument("--max_tray_angle", type=float, default=0.4)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.join(DATA_DIR, "tray_pickup_env.xml"))
    sim = ScriptedPickup(model)
    inputs = np.load(os.path.join(DATA_DIR, "initial_states", args.input))

    out = {"qpos": [], "qvel": [], "act": [], "stage": [], "cube_rest_z": []}
    outcomes = {}
    max_height = []
    for i in range(len(inputs["qpos"])):
        records, outcome = demonstrate(
            sim, inputs["qpos"][i], inputs["qvel"][i], inputs["act"][i], args
        )
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if records:
            max_height.append(sim.cube_z() - records[0][4])
        for stage, qpos, qvel, act, cube_rest_z in records:
            out["qpos"].append(qpos)
            out["qvel"].append(qvel)
            out["act"].append(act)
            out["stage"].append(stage)
            out["cube_rest_z"].append(cube_rest_z)

    result = {key: np.array(value) for key, value in out.items()}
    result["stage"] = result["stage"].astype(np.int64)
    path = os.path.join(DATA_DIR, "initial_states", args.output)
    np.savez(path, **result)

    print(f"Demonstrations: {outcomes}")
    if max_height:
        print(
            f"Tray height at the end: mean {np.mean(max_height):.3f} m, "
            f"max {np.max(max_height):.3f} m"
        )
    counts = np.bincount(result["stage"], minlength=len(STAGE_NAMES))
    print("States per stage: " + ", ".join(f"{n}={c}" for n, c in zip(STAGE_NAMES, counts)))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
