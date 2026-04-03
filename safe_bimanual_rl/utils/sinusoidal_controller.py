import numpy as np
import mujoco


# pylint: disable=too-few-public-methods
class SinusoidalController:
    """
    A controller that generates mirrored sinusoidal references for both arms of a
    bimanual robotic system.
    """

    def __init__(self, model):

        self.model = model

        # Mapping joints / actuators
        self.left_A1_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "left_arm_A1"
        )
        self.right_A1_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "right_arm_A1"
        )

        self.left_A1_act = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_arm_A1_ctrl"
        )
        self.right_A1_act = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_arm_A1_ctrl"
        )

        self.kp = 5.0

    def step(self, data):
        """
        Generate sinusoidal references for both arms, with a mirrored pattern.
        Args:
            data: The current simulation data, used to access joint positions and
                  apply control signals.
        """
        t = data.time

        wave = 0.4 * np.sin(t)

        q_left = data.qpos[
            self.left_A1_joint
        ]  # Get current position of left arm's A1 joint
        data.ctrl[self.left_A1_act] = self.kp * (
            wave - q_left
        )  # Apply control to left arm's A1 actuator

        q_right = data.qpos[self.right_A1_joint]
        data.ctrl[self.right_A1_act] = self.kp * (-wave - q_right)


if __name__ == "__main__":
    import os
    import mujoco.viewer

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(BASE_DIR, "environments", "data", "scene.xml")

    scene_model = mujoco.MjModel.from_xml_path(xml_path)
    scene_data = mujoco.MjData(scene_model)

    controller = SinusoidalController(scene_model)
    
    cam_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_CAMERA, "my_cam")

    with mujoco.viewer.launch_passive(scene_model, scene_data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_cam")

        while viewer.is_running():
            controller.step(scene_data)
            mujoco.mj_step(scene_model, scene_data)
            viewer.sync()

