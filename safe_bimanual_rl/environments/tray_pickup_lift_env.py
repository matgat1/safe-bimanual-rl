from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv


class TrayPickUpLiftEnv(TrayPickUpBaseEnv):
    """
    Lift phase: both arms lift the tray to a target height.
    Episodes start with grippers grasping the handles (grasp phase complete).
    """

    def reward(self, obs, action, next_obs, absorbing):
        raise NotImplementedError

    def is_absorbing(self, obs):
        raise NotImplementedError
