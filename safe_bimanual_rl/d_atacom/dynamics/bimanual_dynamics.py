from safe_bimanual_rl.d_atacom.dynamics.dynamics import AccelerationControlSystem


class BimanualArmDynamics(AccelerationControlSystem):
    """
    Control-affine dynamics for the dual IIWA 14-DOF bimanual robot.

    State layout (from BimanualTableEnv observation_spec):
      [0:14]   joint positions  (right 0-6, left 7-13)
      [14:28]  joint velocities (right 14-20, left 21-27)
      [28:34]  end-effector positions (right 28-30, left 31-33)
      [34:37]  rel_cube_pos_right_arm   (ReachEnv extra)
      [37:40]  rel_cube_pos_left_arm    (ReachEnv extra)
      [40]     contact_force            (ReachEnv extra)

    q = [joint_pos, joint_vel]  →  index_q = 0..27  (dim_q = 28 in AccelerationControlSystem)
    x = end-effector + task obs →  index_x = 28..40 (dim_x = 13)
    """

    def __init__(self, acc_limit: float = 1.0, n_task_obs: int = 13):
        n_joints = 14
        index_q = list(range(2 * n_joints))          # 0-27: [pos, vel]
        index_x = list(range(2 * n_joints, 2 * n_joints + n_task_obs))  # 28-40

        super().__init__(
            dim_q=n_joints,
            index_q=index_q,
            acc_limit=acc_limit,
            dim_x=n_task_obs,
            index_x=index_x,
        )
