import os
import mujoco
import mujoco.viewer


def test_mujoco_import_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    xml_path = os.path.join(
        PROJECT_ROOT,
        "safe_bimanual_rl",
        "environments",
        "data",
        "dual_arm_iiwa_mujoco.xml",
    )

    assert os.path.exists(xml_path), f"XML not found at {xml_path}"

    model = mujoco.MjModel.from_xml_path(xml_path)

    bodies = model.nbody
    joints = model.njnt
    actuators = model.nu

    assert bodies == 38, f"The model should have 38 bodies, but found {bodies}"
    assert joints == 18, f"The model should have 18 joints, but found {joints}"
    assert actuators == 18, f"The model should have 18 actuators, but found {actuators}"
