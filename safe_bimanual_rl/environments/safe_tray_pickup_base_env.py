from safe_bimanual_rl.environments.safe_bimanual_table_env import SafeBimanualTableEnv
from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv


class SafeTrayPickUpBaseEnv(SafeBimanualTableEnv, TrayPickUpBaseEnv):
    """
    Tray pick-up base environment extended with sphere-based self-collision cost.

    Inherits all scene setup and reward helpers from TrayPickUpBaseEnv,
    and gains _get_self_collision_cost() from SafeBimanualTableEnv.
    The MRO ensures SafeBimanualTableEnv.__init__ runs after the full
    model is initialized, so sphere site IDs are resolved correctly.
    """

    pass
