import numpy as np
from safe_bimanual_rl.d_atacom.dynamics.dynamics import ControlAffineSystem


class BimanualArmDynamics(ControlAffineSystem):
    """
    Velocity-control dynamics for the dual IIWA bimanual robot.

    The robot uses intvelocity actuators: the policy outputs a velocity target u,
    which integrates to a position reference tracked by a PD controller. At the
    policy timestep level the effective dynamics are first-order:

        dq_arm_pos/dt = vel_limit * u_arm

    State decomposition:
      q = arm joint positions  [0:14]   dim_q = 14  (controllable, constraint-relevant)
      x = EE positions         [36:42]  dim_x = 6   (uncontrollable task context)

    The full policy action u has 18 dims (14 arm + 4 finger joints):
      dim_u = 18

    G is (dim_q × dim_u) = (14 × 18):
      - First 14 columns: vel_limit * I  (arm joints drive arm positions)
      - Last  4 columns:  zeros          (finger joints don't affect arm positions)

    Because J_G[:, 14:] = 0, ATACOM puts no constraint on finger actions — they
    pass through the null space freely, exactly as the policy dictates.

    Observation layout (BimanualTableEnv base):
      [0:7]    right arm joint positions  (A1-A7)
      [7:14]   left  arm joint positions  (A1-A7)
      [14:18]  finger joint positions     (not controlled by policy arm outputs)
      [18:32]  arm joint velocities       (14 dims)
      [32:36]  finger joint velocities    (4 dims)
      [36:39]  right end-effector position
      [39:42]  left  end-effector position
      [42+]    task-specific observations
    """

    _N_ARM_JOINTS = 14
    _N_FINGER_JOINTS = 4   # 2 per gripper × 2 grippers
    _EE_INDEX_START = 36
    _EE_DIM = 6            # right (3) + left (3)

    def __init__(self, vel_limit: float = 1.0):
        if np.isscalar(vel_limit):
            self.vel_limit = np.ones(self._N_ARM_JOINTS) * vel_limit
        else:
            self.vel_limit = np.asarray(vel_limit, dtype=float)

        self._add_save_attr(vel_limit="primitive")

        n_arm = self._N_ARM_JOINTS
        n_total = n_arm + self._N_FINGER_JOINTS   # 18 = full action space

        index_q = list(range(n_arm))
        index_x = list(range(self._EE_INDEX_START, self._EE_INDEX_START + self._EE_DIM))

        super().__init__(
            dim_q=n_arm,
            dim_u=n_total,
            index_q=index_q,
            dim_x=self._EE_DIM,
            index_x=index_x,
        )

    def f(self, q):
        assert q.shape[-1] == self.dim_q
        return np.zeros((q.shape[0], self.dim_q, 1))

    def G(self, q):
        assert q.shape[-1] == self.dim_q
        # Shape (dim_q, dim_u) = (14, 18)
        G = np.zeros((self.dim_q, self.dim_u))
        G[:, : self._N_ARM_JOINTS] = np.diag(self.vel_limit)
        return G
