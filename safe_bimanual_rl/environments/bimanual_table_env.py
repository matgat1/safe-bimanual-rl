import os

import mujoco
import numpy as np

from mushroom_rl.environments.mujoco import MuJoCo, ObservationType, MujocoViewer


class BimanualTableEnv(MuJoCo):

    def __init__(self, xml_path, gamma, horizon, n_substeps, collision_groups=None):
        """

        Returns:
            _type_: _description_
        """

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
        
        # TODO: Add collision
        
        # TODO: keyframe ?
        
        viewer_params = {}
        viewer_params.setdefault(
            "camera_params", MujocoViewer.get_default_camera_params()
        )
        
        
        super().__init__(
            xml_path,
            gamma=gamma,
            horizon=horizon,
            actuation_spec=action_spec,
            observation_spec=observation_spec,
            collision_groups=collision_groups,
            n_substeps=n_substeps,
        )
