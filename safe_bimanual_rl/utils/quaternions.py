import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
