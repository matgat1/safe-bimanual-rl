import numpy as np
import mujoco
from safe_bimanual_rl.environments.bimanual_table_env import BimanualTableEnv

_LEFT_SPHERE_NAMES = [
    "left_arm_link_0_sphere",
    "left_arm_A1_joint_sphere",
    "left_arm_link_1_sphere",
    "left_arm_A2_joint_sphere",
    "left_arm_link_2_sphere",
    "left_arm_A3_joint_sphere",
    "left_arm_link_3_sphere",
    "left_arm_A4_joint_sphere",
    "left_arm_link_4_sphere",
    "left_arm_A5_joint_sphere",
    "left_arm_link_5_sphere",
    "left_arm_A6_joint_sphere",
    "left_arm_link_7_sphere",
]
_RIGHT_SPHERE_NAMES = [
    "right_arm_link_0_sphere",
    "right_arm_A1_joint_sphere",
    "right_arm_link_1_sphere",
    "right_arm_A2_joint_sphere",
    "right_arm_link_2_sphere",
    "right_arm_A3_joint_sphere",
    "right_arm_link_3_sphere",
    "right_arm_A4_joint_sphere",
    "right_arm_link_4_sphere",
    "right_arm_A5_joint_sphere",
    "right_arm_link_5_sphere",
    "right_arm_A6_joint_sphere",
    "right_arm_link_7_sphere",
]

_TABLE_GEOM_NAME = "table_base_link_collision"


class SafeBimanualTableEnv(BimanualTableEnv):
    """
    Extends BimanualTableEnv with safety costs from:
      1. Arm self-collision: penetration depth between left and right arm spheres.
      2. Table collision: any arm sphere whose centre drops below z_table + radius.

    Cost convention: cost > 0 means constraint violated, 0 means safe.
    _get_safety_cost() returns the maximum violation across both checks and is
    the single value to pass to D-ATACOM as the cost signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._left_sphere_ids = self._resolve_sphere_ids(_LEFT_SPHERE_NAMES)
        self._right_sphere_ids = self._resolve_sphere_ids(_RIGHT_SPHERE_NAMES)
        self._table_surface_z = self._compute_table_surface_z()

    def _resolve_sphere_ids(self, names):
        ids = []
        for name in names:
            site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id == -1:
                raise ValueError(f"Safety sphere site '{name}' not found in model.")
            radius = float(self._model.site_size[site_id, 0])
            ids.append((site_id, radius))
        return ids

    def _compute_table_surface_z(self):
        """Read table surface z from the collision geom: geom_center_z + half_height."""
        geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, _TABLE_GEOM_NAME
        )
        if geom_id == -1:
            raise ValueError(f"Table geom '{_TABLE_GEOM_NAME}' not found in model.")
        # geom_xpos is the world-frame centre position (valid after mj_forward in __init__)
        # geom_size[2] is the half-height for a box geom
        return float(self._data.geom_xpos[geom_id, 2] + self._model.geom_size[geom_id, 2])

    def _get_self_collision_cost(self):
        """Maximum penetration depth across all left-right arm sphere pairs."""
        max_violation = 0.0
        for left_id, left_r in self._left_sphere_ids:
            left_pos = self._data.site_xpos[left_id]
            for right_id, right_r in self._right_sphere_ids:
                right_pos = self._data.site_xpos[right_id]
                dist = np.linalg.norm(left_pos - right_pos)
                violation = left_r + right_r - dist
                if violation > max_violation:
                    max_violation = violation
        return float(max_violation)

    def _get_table_collision_cost(self):
        """Maximum penetration depth of any arm sphere below the table surface."""
        max_violation = 0.0
        for site_id, radius in self._left_sphere_ids + self._right_sphere_ids:
            sphere_z = self._data.site_xpos[site_id, 2]
            # violation > 0 when sphere centre is within radius of the table surface
            violation = self._table_surface_z + radius - sphere_z
            if violation > max_violation:
                max_violation = violation
        return float(max_violation)

    def _get_safety_cost(self):
        """Combined cost: max of arm self-collision and arm-table collision."""
        return max(self._get_self_collision_cost(), self._get_table_collision_cost())
