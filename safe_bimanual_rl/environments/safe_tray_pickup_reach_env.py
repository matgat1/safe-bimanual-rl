import numpy as np
from safe_bimanual_rl.environments.safe_tray_pickup_base_env import (
    SafeTrayPickUpBaseEnv,
)
from safe_bimanual_rl.environments.tray_pickup_reach_env import TrayPickUpReachEnv


class SafeTrayPickUpReachEnv(SafeTrayPickUpBaseEnv, TrayPickUpReachEnv):
    """
    Reach phase environment with sphere-based safety cost for D-ATACOM.

    MRO: SafeTrayPickUpReachEnv → SafeTrayPickUpBaseEnv → SafeBimanualTableEnv
         → TrayPickUpReachEnv → TrayPickUpBaseEnv → BimanualTableEnv → MuJoCo

    All reward, observation, and termination logic is inherited from
    TrayPickUpReachEnv. _create_info_dictionary adds info["cost"] as the
    maximum penetration depth across two checks:
      - arm self-collision (left-right sphere pairs)
      - arm-table collision (any sphere below table surface + radius)
    Cost is 0 when safe, > 0 when either constraint is violated.
    """

    def _create_info_dictionary(self, obs, action):
        info = super()._create_info_dictionary(obs, action)
        info["cost"] = self._get_safety_cost()
        return info


if __name__ == "__main__":
    env = SafeTrayPickUpReachEnv()
    env.reset()
    env.render()
    while True:
        action = np.zeros(env.info.action_space.shape[0])
        obs, reward, absorbing, info = env.step(action)
        print(f"cost: {info['cost']:.4f}")
        env.render()
