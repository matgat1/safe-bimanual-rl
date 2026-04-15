from mushroom_rl.environments.mujoco import MuJoCo, ObservationType, MujocoViewer

import numpy as np


class BimanualTableEnv(MuJoCo):
    """
    A bimanual table environment for mujoco simulation.
    Using mushroom_rl's MuJoCo environment as a base class
    """

    def __init__(
        self,
        scene_xml,
        gamma,
        horizon,
        n_substeps,
        actuation_spec=None,
        collision_groups=None,
        additional_data_spec=None,
        **viewer_params,
    ):
        """
        Initialize the bimanual table environment.

        Args:
            scene_xml (str): The path to the MuJoCo scene XML file.
            gamma (float): The discounting factor of the environment.
            horizon (int): The maximum horizon for the environment
            n_substeps (int): The number of substeps to use by the MuJoCo simulator.
            actuation_spec (list): The list of actuation variables for the environment.
            observation_spec (list): The list of observation variables for the environment.
        """

        action_spec = actuation_spec or [
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

        additional_data_spec = additional_data_spec or []

        collision_groups = collision_groups or []

        collision_groups += [
            (
                "robot",
                [
                    "left_arm_link_0_collision",
                    "left_arm_link_1_collision",
                    "left_arm_link_2_collision",
                    "left_arm_link_3_collision",
                    "left_arm_link_4_collision",
                    "left_arm_link_5_collision",
                    "left_arm_link_6_collision",
                    "left_arm_link_7_collision",
                    "left_hande_robotiq_hande_coupler_collision",
                    "right_arm_link_0_collision",
                    "right_arm_link_1_collision",
                    "right_arm_link_2_collision",
                    "right_arm_link_3_collision",
                    "right_arm_link_4_collision",
                    "right_arm_link_5_collision",
                    "right_arm_link_6_collision",
                    "right_arm_link_7_collision",
                    "right_hande_robotiq_hande_coupler_collision",
                ],
            ),
            (
                "hand",
                [
                    "left_hande_robotiq_hande_link_collision",
                    "left_hande_robotiq_hande_left_finger_collision",
                    "left_hande_robotiq_hande_right_finger_collision",
                    "right_hande_robotiq_hande_link_collision",
                    "right_hande_robotiq_hande_left_finger_collision",
                    "right_hande_robotiq_hande_right_finger_collision",
                ],
            ),
        ]

        viewer_params = viewer_params or {}
        viewer_params.setdefault(
            "camera_params", MujocoViewer.get_default_camera_params()
        )

        viewer_params["camera_params"]["static"].update(
            {
                "distance": 4,
                "elevation": -50.0,
                "azimuth": 70.0,
                "lookat": np.array([-0.92, 0.0, 0.9]),
            }
        )

        super().__init__(
            xml_file=scene_xml,
            actuation_spec=action_spec,
            observation_spec=observation_spec,
            additional_data_spec=additional_data_spec,
            collision_groups=collision_groups,
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            **viewer_params,
        )

    def _get_contact_force(self, group1, group2, contact_force_range):
        """
        Get the contact force between two collision groups.
        Clip the contact force to the specified range.
        Return the sum of the squared contact forces.
        """

        collision_force = self._get_collision_force(group1, group2)
        contact_force = np.clip(collision_force, *contact_force_range)
        contact_force = np.sum(np.square(contact_force), keepdims=True)
        return contact_force
