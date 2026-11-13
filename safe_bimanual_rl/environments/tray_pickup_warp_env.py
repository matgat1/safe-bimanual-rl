import os
import mujoco
import numpy as np
import torch
import mujoco_warp as mjw
import warp as wp

from mushroom_rl.core.spaces import Box
from mushroom_rl.environments.mujoco_warp import MuJoCoWarp
from mushroom_rl.utils.mujoco import ObservationType, MujocoViewer


ARM_JOINTS = [f"{side}_arm_A{i}" for side in ("right", "left") for i in range(1, 8)]
RIGHT_FINGERS = [
    "right_hande_robotiq_hande_left_finger_joint",
    "right_hande_robotiq_hande_right_finger_joint",
]
LEFT_FINGERS = [
    "left_hande_robotiq_hande_left_finger_joint",
    "left_hande_robotiq_hande_right_finger_joint",
]

# Same action/observation layout as BimanualTableEnv so that agents and
# controllers are interchangeable between the CPU and GPU environments.
ACTION_SPEC = [f"{j}_ctrl" for j in ARM_JOINTS + LEFT_FINGERS + RIGHT_FINGERS]

OBSERVATION_SPEC = (
    [(f"{j}_pos", j, ObservationType.JOINT_POS) for j in ARM_JOINTS]
    + [(f"{j}_pos", j, ObservationType.JOINT_POS) for j in RIGHT_FINGERS]
    + [(f"{j}_pos", j, ObservationType.JOINT_POS) for j in LEFT_FINGERS]
    + [(f"{j}_vel", j, ObservationType.JOINT_VEL) for j in ARM_JOINTS]
    + [(f"{j}_vel", j, ObservationType.JOINT_VEL) for j in LEFT_FINGERS]
    + [(f"{j}_vel", j, ObservationType.JOINT_VEL) for j in RIGHT_FINGERS]
    + [
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
)

ADDITIONAL_DATA_SPEC = [
    ("cube_pos", "cube", ObservationType.BODY_POS),
    ("tray_rot", "tray", ObservationType.BODY_ROT),
    ("right_grip_point_pos", "right_grip_point", ObservationType.SITE_POS),
    ("left_grip_point_pos", "left_grip_point", ObservationType.SITE_POS),
    ("right_handle_pos", "right_handle", ObservationType.BODY_POS),
    ("left_handle_pos", "left_handle", ObservationType.BODY_POS),
    ("right_handle_rot", "right_handle", ObservationType.BODY_ROT),
    ("left_handle_rot", "left_handle", ObservationType.BODY_ROT),
]

COLLISION_GROUPS = [
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
    *[
        (f"{side}_hand_{finger}", [f"{side}_hande_robotiq_hande_{finger}_collision"])
        for side in ("right", "left")
        for finger in ("left_finger", "right_finger")
    ],
    (
        "robot",
        [f"{side}_arm_link_{i}_collision" for side in ("left", "right") for i in range(8)]
        + [
            "left_hande_robotiq_hande_coupler_collision",
            "right_hande_robotiq_hande_coupler_collision",
        ],
    ),
    (
        "hand",
        [
            f"{side}_hande_robotiq_hande_{part}_collision"
            for side in ("left", "right")
            for part in ("link", "left_finger", "right_finger")
        ],
    ),
]




class TrayPickUpWarpEnv(MuJoCoWarp):
    """
    GPU-vectorized (MuJoCo Warp) environment for the full bimanual tray pick-up:
    a single policy reaches the handles, grasps them with both grippers and lifts
    the tray while keeping it level.

    Simulates ``num_envs`` worlds in parallel. Observations, rewards and absorbing
    flags are batched torch tensors of shape (num_envs, ...).

    The reward is staged (reach -> grasp -> lift -> level), each stage only paying
    once the previous one is achieved, and normalized so that the per-step reward
    is at most 1. Once the tray is held lifted and level the per-step reward is
    maxed out, and after ``success_hold_steps`` such steps the episode terminates
    with a reward of 1 / (1 - gamma), i.e. the value of staying at the maximum
    forever, so terminating on success is never worse than continuing.
    """

    def __init__(
        self,
        num_envs: int,
        gamma: float = 0.99,
        horizon: int = 600,
        n_substeps: int = 5,
        reach_weight: float = 1.0,
        reach_sharpness: float = 0.3,
        grasp_weight: float = 1.0,
        grasp_force_threshold: float = 1.0,
        grasp_distance: float = 0.03,
        lift_weight: float = 2.0,
        lift_sharpness: float = 0.1,
        lift_target_height: float = 0.35,
        tray_orientation_weight: float = 1.0,
        tray_orientation_sharpness: float = 0.3,
        tray_orientation_threshold: float = 0.4,
        success_hold_steps: int = 20,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        contact_threshold: float = 3.0,
        contact_cost_weight: float = -0.05,
        tray_contact_threshold: float = 2.0,
        tray_contact_cost_weight: float = -0.05,
        action_cost_weight: float = -0.01,
        joint_noise: float = 0.05,
        tray_pos_noise: float = 0.02,
        grasp_start_prob: float = 0.0,
        max_joint_velocity: float = 100.0,
        grasp_init_states_file: str = "grasp_init_states_0.npz",
        nconmax: int = 64,
        njmax: int = 256,
        use_graph_capture: bool = False,
        device: str = None,
        **viewer_params,
    ):
        """
        Args:
            num_envs (int): Number of parallel worlds simulated on the GPU.
            gamma (float): Discount factor.
            horizon (int): Maximum number of steps per episode.
            n_substeps (int): Number of simulation steps per environment step.
            reach_weight (float): Weight of the gripper-to-handle distance stage.
            reach_sharpness (float): Distance (m) scale of the tanh reach reward.
            grasp_weight (float): Weight of the grasp stage (fraction of hands grasping).
            grasp_force_threshold (float): Normal force (N) each of the two fingers of a
                hand must apply on its handle to count as grasping (the palm is ignored).
            grasp_distance (float): Max distance (m) between grip point and handle to
                count as grasping.
            lift_weight (float): Weight of the lift stage, paid only while both hands grasp.
            lift_sharpness (float): Height (m) scale of the tanh lift reward.
            lift_target_height (float): Tray height (m) above its initial position for success.
            tray_orientation_weight (float): Weight of the level-tray stage, paid only
                while both hands grasp.
            tray_orientation_sharpness (float): Angle (rad) scale of the tanh orientation reward.
            tray_orientation_threshold (float): Max tray tilt (rad) for success.
            success_hold_steps (int): Consecutive success steps before the episode ends.
            contact_force_range (tuple[float, float]): Clipping range of the table contact forces.
            contact_threshold (float): Robot/hand-table contact above which the episode
                terminates.
            contact_cost_weight (float): Weight of the robot/hand-table contact cost.
            tray_contact_threshold (float): Tray-table contact (baseline subtracted) above
                which the episode terminates while the tray is on the table.
            tray_contact_cost_weight (float): Weight of the tray-table contact cost.
            action_cost_weight (float): Weight of the squared action cost.
            joint_noise (float): Uniform noise (rad) added to the home arm configuration.
            tray_pos_noise (float): Uniform noise (m) added to the tray x/y position.
            max_joint_velocity (float): A world whose state is not finite or has a joint
                velocity above this value is considered unstable, and its episode ends.
            grasp_start_prob (float): Probability of starting an episode from a saved
                state with the grippers already near the handles instead of the home pose.
            grasp_init_states_file (str): File in data/initial_states/ with those states.
            nconmax (int): Contacts allocated per world (CPU peak is ~24).
            njmax (int): Constraints allocated per world (CPU peak is ~80).
            use_graph_capture (bool): Replay the simulation step as a CUDA graph.
            device (str): Torch device of the simulation (default torch device if None).
        """
        self._reach_weight = reach_weight
        self._reach_sharpness = reach_sharpness
        self._grasp_weight = grasp_weight
        self._grasp_force_threshold = grasp_force_threshold
        self._grasp_distance = grasp_distance
        self._lift_weight = lift_weight
        self._lift_sharpness = lift_sharpness
        self._lift_target_height = lift_target_height
        self._tray_orientation_weight = tray_orientation_weight
        self._tray_orientation_sharpness = tray_orientation_sharpness
        self._tray_orientation_threshold = tray_orientation_threshold
        self._success_hold_steps = success_hold_steps
        self._contact_force_range = contact_force_range
        self._contact_threshold = contact_threshold
        self._contact_cost_weight = contact_cost_weight
        self._tray_contact_threshold = tray_contact_threshold
        self._tray_contact_cost_weight = tray_contact_cost_weight
        self._action_cost_weight = action_cost_weight
        self._joint_noise = joint_noise
        self._tray_pos_noise = tray_pos_noise
        self._grasp_start_prob = grasp_start_prob
        self._max_joint_velocity = max_joint_velocity
        self._max_stage_reward = (
            reach_weight + grasp_weight + lift_weight + tray_orientation_weight
        )

        viewer_params.setdefault("camera_params", MujocoViewer.get_default_camera_params())
        viewer_params.setdefault("hide_menu_on_startup", True)
        viewer_params.setdefault("width", 1280)
        viewer_params.setdefault("height", 720)
        viewer_params["camera_params"]["static"].update(
            {
                "distance": 3,
                "elevation": -30.0,
                "azimuth": 30.0,
                "lookat": np.array([-0.92, 0.0, 0.9]),
            }
        )

        scene_xml = os.path.join(os.path.dirname(__file__), "data", "tray_pickup_env.xml")

        super().__init__(
            xml_file=scene_xml,
            actuation_spec=ACTION_SPEC,
            observation_spec=OBSERVATION_SPEC,
            gamma=gamma,
            horizon=horizon,
            num_envs=num_envs,
            n_substeps=n_substeps,
            additional_data_spec=ADDITIONAL_DATA_SPEC,
            collision_groups=COLLISION_GROUPS,
            nconmax=nconmax,
            njmax=njmax,
            use_graph_capture=use_graph_capture,
            device=device,
            **viewer_params,
        )

        dev = self.info.device
        self._group_masks = {}
        for name, geom_ids in self.collision_groups.items():
            mask = torch.zeros(self._model.ngeom, dtype=torch.bool, device=dev)
            mask[list(geom_ids)] = True
            self._group_masks[name] = mask
        self._contacts = None
        self._env_mask = torch.ones(num_envs, dtype=torch.bool, device=dev)

        # Home pose, with the qpos/act addresses of the arm joints to perturb
        dtype = wp.to_torch(self._data_wp.qpos).dtype
        home = self._model.keyframe("home")
        self._home_qpos = torch.as_tensor(home.qpos, dtype=dtype, device=dev)
        self._home_qvel = torch.as_tensor(home.qvel, dtype=dtype, device=dev)
        self._home_act = torch.as_tensor(home.act, dtype=dtype, device=dev)
        arm_qpos_adr, arm_act_adr = [], []
        for joint in ARM_JOINTS:
            joint_id = self._model.joint(joint).id
            actuator_id = int(np.flatnonzero(self._model.actuator_trnid[:, 0] == joint_id)[0])
            arm_qpos_adr.append(self._model.jnt_qposadr[joint_id])
            arm_act_adr.append(self._model.actuator_actadr[actuator_id])
        self._arm_qpos_adr = torch.as_tensor(arm_qpos_adr, device=dev)
        self._arm_act_adr = torch.as_tensor(arm_act_adr, device=dev)
        # x/y of every free body (tray and cube), shifted together
        free_qpos_adr = self._model.jnt_qposadr[self._model.jnt_type == mujoco.mjtJoint.mjJNT_FREE]
        self._free_xy_adr = torch.as_tensor(
            [[adr, adr + 1] for adr in free_qpos_adr], device=dev
        )

        grasp_states = np.load(
            os.path.join(
                os.path.dirname(__file__), "data", "initial_states", grasp_init_states_file
            )
        )
        self._grasp_qpos = torch.as_tensor(grasp_states["qpos"], dtype=dtype, device=dev)
        self._grasp_qvel = torch.as_tensor(grasp_states["qvel"], dtype=dtype, device=dev)
        self._grasp_act = torch.as_tensor(grasp_states["act"], dtype=dtype, device=dev)

        # Per-world episode state
        self._initial_cube_height = torch.zeros(num_envs, device=dev)
        self._success_steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self._success = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self._absorbing_counts = {
            key: torch.zeros((), dtype=torch.long, device=dev)
            for key in ("success", "contact_force", "tray_contact_force", "unstable")
        }
        # Number of active steps (and started episodes) in each stage, to monitor training
        self._step_counts = {
            key: torch.zeros((), dtype=torch.long, device=dev)
            for key in ("episodes", "steps", "right_grasp", "left_grasp", "both_grasp", "lifted")
        }

    def get_absorbing_counts(self) -> dict:
        return {key: int(value.item()) for key, value in self._absorbing_counts.items()}

    def get_step_counts(self) -> dict:
        return {key: int(value.item()) for key, value in self._step_counts.items()}

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def _update_contacts(self):
        """Compute the 6D force of every active contact, in the contact frame."""
        nacon = int(wp.to_torch(self._data_wp.nacon)[0].item())
        nacon = min(nacon, self._data_wp.naconmax)
        if nacon == 0:
            self._contacts = None
            return

        dev = self.info.device
        contact_ids = torch.arange(nacon, dtype=torch.int32, device=dev)
        force = wp.zeros(nacon, dtype=wp.spatial_vector, device=wp.device_from_torch(dev))
        mjw.contact_force(self._model_wp, self._data_wp, wp.from_torch(contact_ids), False, force)

        self._contacts = (
            wp.to_torch(self._data_wp.contact.geom)[:nacon].long(),
            wp.to_torch(self._data_wp.contact.worldid)[:nacon].long(),
            wp.to_torch(force).clone(),
        )

    def _match_contacts(self, group1, group2):
        """Indices of the active contacts between the two groups, None if there are none."""
        if self._contacts is None:
            return None

        geom, _, _ = self._contacts
        m1, m2 = self._group_masks[group1], self._group_masks[group2]
        g1, g2 = geom[:, 0], geom[:, 1]
        valid = (g1 >= 0) & (g2 >= 0)
        g1, g2 = g1.clamp(min=0), g2.clamp(min=0)
        match = valid & ((m1[g1] & m2[g2]) | (m1[g2] & m2[g1]))

        contact_idx = torch.nonzero(match, as_tuple=True)[0]
        return contact_idx if contact_idx.numel() > 0 else None

    def _get_normal_force(self, group1, group2):
        """Total contact normal force (N) between the two groups, shape (num_envs,)."""
        result = torch.zeros(self._num_envs, device=self.info.device)
        contact_idx = self._match_contacts(group1, group2)
        if contact_idx is None:
            return result

        _, worldid, force = self._contacts
        result.index_add_(0, worldid[contact_idx], force[contact_idx, 0].to(result.dtype))
        return result

    def _get_collision_force(self, group1, group2):
        """
        Force/torque of the first contact between the two groups in every world,
        matching MuJoCo._get_collision_force. Shape (num_envs, 6), zero if no contact.
        """
        result = torch.zeros((self._num_envs, 6), device=self.info.device)
        contact_idx = self._match_contacts(group1, group2)
        if contact_idx is None:
            return result

        _, worldid, force = self._contacts
        n_contacts = force.shape[0]
        first = torch.full((self._num_envs,), n_contacts, dtype=torch.long, device=self.info.device)
        first.scatter_reduce_(0, worldid[contact_idx], contact_idx, reduce="amin")
        has_contact = first < n_contacts
        result[has_contact] = force[first[has_contact]].to(result.dtype)
        return result

    def _get_contact_force(self, group1, group2):
        """Sum of the squared clipped contact force/torque, shape (num_envs, 1)."""
        force = self._get_collision_force(group1, group2).clamp(*self._contact_force_range)
        return force.square().sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        self.obs_helper.add_obs("rel_right_handle_pos", 3)
        self.obs_helper.add_obs("rel_left_handle_pos", 3)
        self.obs_helper.add_obs("contact_force", 1)
        self.obs_helper.add_obs("tray_contact_force", 1)
        self.obs_helper.add_obs("cube_pos", 3)
        self.obs_helper.add_obs("right_handle_rot", 4)
        self.obs_helper.add_obs("left_handle_rot", 4)
        self.obs_helper.add_obs("finger_forces", 4)
        self.obs_helper.add_obs("tray_height", 1)
        mdp_info.observation_space = Box(*self.obs_helper.get_obs_limits())
        return mdp_info

    def _create_observation(self, obs):
        self._update_contacts()
        rel_right = self._read_data("right_handle_pos") - self._read_data("right_grip_point_pos")
        rel_left = self._read_data("left_handle_pos") - self._read_data("left_grip_point_pos")
        contact_force = self._get_contact_force("robot", "table") + self._get_contact_force(
            "hand", "table"
        )
        # subtract gravity baseline so the observation is ~0 at rest
        tray_contact_force = self._get_contact_force("tray", "table") - 0.9
        # Normal force of each finger on its handle: (right hand: left, right, left hand: left, right)
        finger_forces = torch.stack(
            [
                self._get_normal_force(f"{side}_hand_{finger}", f"{side}_handle")
                for side in ("right", "left")
                for finger in ("left_finger", "right_finger")
            ],
            dim=1,
        )
        obs = torch.cat(
            [
                obs,
                rel_right.to(obs.dtype),
                rel_left.to(obs.dtype),
                contact_force.to(obs.dtype),
                tray_contact_force.to(obs.dtype),
                self._read_data("cube_pos").to(obs.dtype),
                self._read_data("right_handle_rot").to(obs.dtype),
                self._read_data("left_handle_rot").to(obs.dtype),
                finger_forces.to(obs.dtype),
                self._cube_height().unsqueeze(1).to(obs.dtype),
            ],
            dim=1,
        )

        # A diverged world would put NaNs in the dataset and the networks: flag it,
        # end its episode and hand out a zero observation instead.
        qvel = wp.to_torch(self._data_wp.qvel)
        self._unstable = (
            ~torch.isfinite(obs).all(dim=1)
            | ~torch.isfinite(qvel).all(dim=1)
            | (qvel.abs().amax(dim=1) > self._max_joint_velocity)
        )
        return torch.where(self._unstable.unsqueeze(1), torch.zeros_like(obs), obs)

    # ------------------------------------------------------------------
    # Task quantities
    # ------------------------------------------------------------------

    def _cube_height(self):
        return self._read_data("cube_pos")[:, 2] - self._initial_cube_height

    def _grip_forces(self, obs):
        """
        Per hand, the force of its weaker finger on the handle: the handle is only
        held when it is squeezed between both fingers. Two (num_envs,) tensors.
        """
        forces = self.obs_helper.get_from_obs(obs, "finger_forces")
        return forces[:, :2].amin(dim=1), forces[:, 2:].amin(dim=1)

    def _hands_grasping(self, obs):
        """
        Per hand, whether it holds its handle: handle close to the grip point and
        squeezed by both fingers. Two (num_envs,) masks.
        """
        right_force, left_force = self._grip_forces(obs)
        rel_right = self.obs_helper.get_from_obs(obs, "rel_right_handle_pos").norm(dim=-1)
        rel_left = self.obs_helper.get_from_obs(obs, "rel_left_handle_pos").norm(dim=-1)
        right = (right_force > self._grasp_force_threshold) & (rel_right < self._grasp_distance)
        left = (left_force > self._grasp_force_threshold) & (rel_left < self._grasp_distance)
        return right, left

    def _tray_angle(self):
        # z component of the tray z-axis, from the (w, x, y, z) quaternion
        _, x, y, _ = self._read_data("tray_rot").unbind(dim=-1)
        cos_angle = (1 - 2 * (x.square() + y.square())).abs().clamp(-1.0, 1.0)
        return torch.arccos(cos_angle)

    def _lifted(self, obs):
        """Tray held by both hands, above the target height and level."""
        right_grasp, left_grasp = self._hands_grasping(obs)
        height = self.obs_helper.get_from_obs(obs, "tray_height")[:, 0]
        return (
            right_grasp
            & left_grasp
            & (height > self._lift_target_height)
            & (self._tray_angle() < self._tray_orientation_threshold)
        )

    # ------------------------------------------------------------------
    # MuJoCoWarp interface
    # ------------------------------------------------------------------

    def _preprocess_action(self, action):
        # Unbounded policies (e.g. PPO's Gaussian) can exceed the action space
        low = torch.as_tensor(self.info.action_space.low, dtype=action.dtype, device=action.device)
        high = torch.as_tensor(self.info.action_space.high, dtype=action.dtype, device=action.device)
        return torch.clamp(action, low, high)

    def step_all(self, env_mask, action):
        # All worlds are simulated, keep the mask to only count active episodes
        self._env_mask = env_mask
        return super().step_all(env_mask, action)

    def setup(self, env_indices, obs):
        super().setup(env_indices, obs)
        idx = env_indices.long()
        n, dev = len(idx), idx.device
        qpos = wp.to_torch(self._data_wp.qpos)
        qvel = wp.to_torch(self._data_wp.qvel)
        act = wp.to_torch(self._data_wp.act)

        # Home pose with noise on the arm joints (and their integrated targets)
        home_qpos = self._home_qpos.repeat(n, 1)
        home_act = self._home_act.repeat(n, 1)
        joint_noise = (torch.rand(n, len(ARM_JOINTS), device=dev) * 2 - 1) * self._joint_noise
        home_qpos[:, self._arm_qpos_adr] += joint_noise
        home_act[:, self._arm_act_adr] += joint_noise
        tray_noise = (torch.rand(n, 1, 2, device=dev) * 2 - 1) * self._tray_pos_noise
        home_qpos[:, self._free_xy_adr] += tray_noise
        qpos[idx] = home_qpos
        qvel[idx] = self._home_qvel
        act[idx] = home_act

        # Some episodes start with the grippers already near the handles
        from_grasp = torch.rand(n, device=dev) < self._grasp_start_prob
        if from_grasp.any():
            grasp_idx = idx[from_grasp]
            sample = torch.randint(len(self._grasp_qpos), (len(grasp_idx),), device=dev)
            qpos[grasp_idx] = self._grasp_qpos[sample]
            qvel[grasp_idx] = self._grasp_qvel[sample]
            act[grasp_idx] = self._grasp_act[sample]

        self._mj_warp.forward(self._model_wp, self._data_wp)

        self._initial_cube_height[idx] = self._read_data("cube_pos")[idx, 2].to(
            self._initial_cube_height.dtype
        )
        self._success_steps[idx] = 0
        self._success[idx] = False
        self._step_counts["episodes"] += n

    def _stage_reward(self, obs):
        """Staged task reward in [0, max_stage_reward]."""
        rel_right = self.obs_helper.get_from_obs(obs, "rel_right_handle_pos").norm(dim=-1)
        rel_left = self.obs_helper.get_from_obs(obs, "rel_left_handle_pos").norm(dim=-1)
        right_grasp, left_grasp = self._hands_grasping(obs)
        both_grasp = (right_grasp & left_grasp).float()

        # A grasping hand gets the full reach reward, so reach never fights lift
        reach_right = torch.where(
            right_grasp, 1.0, 1 - torch.tanh(rel_right / self._reach_sharpness)
        )
        reach_left = torch.where(left_grasp, 1.0, 1 - torch.tanh(rel_left / self._reach_sharpness))
        reach = self._reach_weight * (reach_right + reach_left) / 2
        grasp = self._grasp_weight * (right_grasp.float() + left_grasp.float()) / 2

        height = self.obs_helper.get_from_obs(obs, "tray_height")[:, 0].clamp(min=0.0)
        lift = both_grasp * self._lift_weight * torch.tanh(height / self._lift_sharpness)
        level = (
            both_grasp
            * self._tray_orientation_weight
            * (1 - torch.tanh(self._tray_angle() / self._tray_orientation_sharpness))
        )

        stage_reward = reach + grasp + lift + level
        return torch.where(
            self._lifted(obs), torch.full_like(stage_reward, self._max_stage_reward), stage_reward
        )

    def reward(self, obs, action, next_obs, absorbing):
        task_reward = self._stage_reward(next_obs) / self._max_stage_reward

        contact_cost = (
            self._contact_cost_weight * self.obs_helper.get_from_obs(next_obs, "contact_force")[:, 0]
        )
        tray_contact_force = self.obs_helper.get_from_obs(next_obs, "tray_contact_force")[:, 0]
        tray_contact_cost = self._tray_contact_cost_weight * tray_contact_force.clamp(min=0.0)
        action_t = torch.as_tensor(action, dtype=task_reward.dtype, device=task_reward.device)
        action_cost = self._action_cost_weight * action_t.square().sum(dim=-1)

        reward = task_reward + contact_cost + tray_contact_cost + action_cost
        # Ending on success is worth staying at the maximum reward forever
        reward = torch.where(
            self._success, torch.full_like(reward, 1 / (1 - self.info.gamma)), reward
        )
        reward = torch.where(self._unstable, torch.zeros_like(reward), reward)
        return reward.to(obs.dtype)

    def is_absorbing(self, obs):
        lifted = self._lifted(obs)
        self._success_steps = torch.where(
            lifted, self._success_steps + 1, torch.zeros_like(self._success_steps)
        )
        self._success = self._success_steps >= self._success_hold_steps

        contact_force = self.obs_helper.get_from_obs(obs, "contact_force")[:, 0]
        contact_exceeded = ~self._success & (contact_force > self._contact_threshold)

        # Skip tray contact termination while the tray is being lifted
        # to avoid false terminations from simulation force artifacts mid-air.
        height = self.obs_helper.get_from_obs(obs, "tray_height")[:, 0]
        tray_contact_force = self.obs_helper.get_from_obs(obs, "tray_contact_force")[:, 0]
        tray_contact_exceeded = (
            ~self._success
            & ~contact_exceeded
            & (height < 0.05)
            & (tray_contact_force > self._tray_contact_threshold)
        )

        self._absorbing_counts["success"] += (self._success & self._env_mask).sum()
        self._absorbing_counts["unstable"] += (self._unstable & self._env_mask).sum()
        self._absorbing_counts["contact_force"] += (contact_exceeded & self._env_mask).sum()
        self._absorbing_counts["tray_contact_force"] += (
            tray_contact_exceeded & self._env_mask
        ).sum()

        right_grasp, left_grasp = self._hands_grasping(obs)
        self._step_counts["steps"] += self._env_mask.sum()
        self._step_counts["right_grasp"] += (right_grasp & self._env_mask).sum()
        self._step_counts["left_grasp"] += (left_grasp & self._env_mask).sum()
        self._step_counts["both_grasp"] += (right_grasp & left_grasp & self._env_mask).sum()
        self._step_counts["lifted"] += (lifted & self._env_mask).sum()

        return self._success | contact_exceeded | tray_contact_exceeded | self._unstable

    def _create_info_dictionary(self, obs):
        right_grasp, left_grasp = self._hands_grasping(obs)
        right_force, left_force = self._grip_forces(obs)
        return {
            "tray_height": self.obs_helper.get_from_obs(obs, "tray_height")[:, 0],
            "tray_angle": self._tray_angle(),
            "right_handle_dist": self.obs_helper.get_from_obs(obs, "rel_right_handle_pos").norm(
                dim=-1
            ),
            "left_handle_dist": self.obs_helper.get_from_obs(obs, "rel_left_handle_pos").norm(
                dim=-1
            ),
            "right_grip_force": right_force,
            "left_grip_force": left_force,
            "right_grasp": right_grasp,
            "left_grasp": left_grasp,
            "both_grasp": right_grasp & left_grasp,
            "lifted": self._lifted(obs),
            "success": self._success,
        }


if __name__ == "__main__":
    # Throughput benchmark: python -m safe_bimanual_rl.environments.tray_pickup_warp_env 64 128 256
    import sys
    import time

    n_envs_list = [int(arg) for arg in sys.argv[1:]] or [64, 128, 256, 512, 1024]
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for n_envs in n_envs_list:
        env = TrayPickUpWarpEnv(
            num_envs=n_envs,
            use_graph_capture=use_cuda,
            device="cuda:0" if use_cuda else "cpu",
        )
        mask = torch.ones(n_envs, dtype=torch.bool, device=env.info.device)
        env.reset_all(mask)
        low, high = (
            torch.as_tensor(bound, dtype=torch.float32, device=env.info.device)
            for bound in (env.info.action_space.low, env.info.action_space.high)
        )

        def run(n_steps):
            for _ in range(n_steps):
                action = low + torch.rand(n_envs, len(low), device=env.info.device) * (high - low)
                env.step_all(mask, action)
            if use_cuda:
                torch.cuda.synchronize()

        run(5)  # warm-up: kernel compilation and CUDA graph capture
        n_steps = 100
        start = time.time()
        run(n_steps)
        elapsed = time.time() - start
        memory = f", {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB" if use_cuda else ""
        print(f"{n_envs:5d} worlds: {n_envs * n_steps / elapsed:8.0f} env steps/s{memory}")
        del env
