import os
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType


class BimanualTableEnv(MuJoCo):
    """
    A bimanual table environment for mujoco simulation.
    Using mushroom_rl's MuJoCo environment as a base class
    """

    def __init__(self, gamma=1, horizon=1000, n_substeps=1):
        """
        Initialize the bimanual table environment.

        Args:
            gamma (float): The discounting factor of the environment.
            horizon (int): The maximum horizon for the environment
            n_substeps (int): The number of substeps to use by the MuJoCo simulator.
        """

        scene_xml = os.path.join(
            os.path.dirname(__file__), "data", "dual_arm_iiwa_mujoco.xml"
        )

        action_spec = [
            "right_arm_A1_ctrl",
            "right_arm_A2_ctrl",
            "right_arm_A3_ctrl",
            "right_arm_A4_ctrl",
            "right_arm_A5_ctrl",
            "right_arm_A6_ctrl",
            "right_arm_A7_ctrl",
            "left_arm_A1_ctrl",
            "left_arm_A2_ctrl",
            "left_arm_A3_ctrl",
            "left_arm_A4_ctrl",
            "left_arm_A5_ctrl",
            "left_arm_A6_ctrl",
            "left_arm_A7_ctrl",
            "left_hande_robotiq_hande_left_finger_joint_ctrl",
            "left_hande_robotiq_hande_right_finger_joint_ctrl",
            "right_hande_robotiq_hande_left_finger_joint_ctrl",
            "right_hande_robotiq_hande_right_finger_joint_ctrl",
        ]

        observation_spec = [
            ("right_arm_A1_pos", "right_arm_A1", ObservationType.JOINT_POS),
            ("right_arm_A2_pos", "right_arm_A2", ObservationType.JOINT_POS),
            ("right_arm_A3_pos", "right_arm_A3", ObservationType.JOINT_POS),
            ("right_arm_A4_pos", "right_arm_A4", ObservationType.JOINT_POS),
            ("right_arm_A5_pos", "right_arm_A5", ObservationType.JOINT_POS),
            ("right_arm_A6_pos", "right_arm_A6", ObservationType.JOINT_POS),
            ("right_arm_A7_pos", "right_arm_A7", ObservationType.JOINT_POS),
            ("left_arm_A1_pos", "left_arm_A1", ObservationType.JOINT_POS),
            ("left_arm_A2_pos", "left_arm_A2", ObservationType.JOINT_POS),
            ("left_arm_A3_pos", "left_arm_A3", ObservationType.JOINT_POS),
            ("left_arm_A4_pos", "left_arm_A4", ObservationType.JOINT_POS),
            ("left_arm_A5_pos", "left_arm_A5", ObservationType.JOINT_POS),
            ("left_arm_A6_pos", "left_arm_A6", ObservationType.JOINT_POS),
            ("left_arm_A7_pos", "left_arm_A7", ObservationType.JOINT_POS),
            (
                "right_hande_robotiq_hande_left_finger_joint_pos",
                "right_hande_robotiq_hande_left_finger_joint",
                ObservationType.JOINT_POS,
            ),
            (
                "right_hande_robotiq_hande_right_finger_joint_pos",
                "right_hande_robotiq_hande_right_finger_joint",
                ObservationType.JOINT_POS,
            ),
            (
                "left_hande_robotiq_hande_left_finger_joint_pos",
                "left_hande_robotiq_hande_left_finger_joint",
                ObservationType.JOINT_POS,
            ),
            (
                "left_hande_robotiq_hande_right_finger_joint_pos",
                "left_hande_robotiq_hande_right_finger_joint",
                ObservationType.JOINT_POS,
            ),
            ("right_arm_A1_vel", "right_arm_A1", ObservationType.JOINT_VEL),
            ("right_arm_A2_vel", "right_arm_A2", ObservationType.JOINT_VEL),
            ("right_arm_A3_vel", "right_arm_A3", ObservationType.JOINT_VEL),
            ("right_arm_A4_vel", "right_arm_A4", ObservationType.JOINT_VEL),
            ("right_arm_A5_vel", "right_arm_A5", ObservationType.JOINT_VEL),
            ("right_arm_A6_vel", "right_arm_A6", ObservationType.JOINT_VEL),
            ("right_arm_A7_vel", "right_arm_A7", ObservationType.JOINT_VEL),
            ("left_arm_A1_vel", "left_arm_A1", ObservationType.JOINT_VEL),
            ("left_arm_A2_vel", "left_arm_A2", ObservationType.JOINT_VEL),
            ("left_arm_A3_vel", "left_arm_A3", ObservationType.JOINT_VEL),
            ("left_arm_A4_vel", "left_arm_A4", ObservationType.JOINT_VEL),
            ("left_arm_A5_vel", "left_arm_A5", ObservationType.JOINT_VEL),
            ("left_arm_A6_vel", "left_arm_A6", ObservationType.JOINT_VEL),
            ("left_arm_A7_vel", "left_arm_A7", ObservationType.JOINT_VEL),
            (
                "left_hande_robotiq_hande_left_finger_joint_vel",
                "left_hande_robotiq_hande_left_finger_joint",
                ObservationType.JOINT_VEL,
            ),
            (
                "left_hande_robotiq_hande_right_finger_joint_vel",
                "left_hande_robotiq_hande_right_finger_joint",
                ObservationType.JOINT_VEL,
            ),
            (
                "right_hande_robotiq_hande_left_finger_joint_vel",
                "right_hande_robotiq_hande_left_finger_joint",
                ObservationType.JOINT_VEL,
            ),
            (
                "right_hande_robotiq_hande_right_finger_joint_vel",
                "right_hande_robotiq_hande_right_finger_joint",
                ObservationType.JOINT_VEL,
            ),
        ]

        super().__init__(
            xml_file=scene_xml,
            actuation_spec=action_spec,
            observation_spec=observation_spec,
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
        )

    def is_absorbing(self, obs):
        return False

    def reward(self, obs, action, next_obs, absorbing):
        return 0


if __name__ == "__main__":
    import numpy as np

    env = BimanualTableEnv()
    env.reset()
    env.render()
    while True:
        action = np.zeros((18,))
        env.step(action)
        env.render()
