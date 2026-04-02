# pylint: disable=no-member
import os
import mujoco
import mujoco.viewer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "data", "dual_arm_iiwa_mujoco.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"Bodies : {model.nbody}")
print(f"Joints : {model.njnt}")
print(f"Actuators : {model.nu}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
