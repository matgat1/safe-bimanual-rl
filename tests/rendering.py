from safe_bimanual_rl.environments.bimanual_table_env import BimanualTableEnv

env = BimanualTableEnv(
    xml_path="safe_bimanual_rl/environments/data/dual_arm_iiwa_mujoco.xml",
    gamma=0.99,
    horizon=1000,
    n_substeps=20,
)

env.render(record=True)
