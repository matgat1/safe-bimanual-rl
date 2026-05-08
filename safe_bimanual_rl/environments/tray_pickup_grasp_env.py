from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv
import numpy as np

class TrayPickUpGraspEnv(TrayPickUpBaseEnv):
    """
    Grasp phase: both grippers must close around the tray handles.
    Episodes start with grippers already positioned at the handles (reach phase complete).
    """

    def __init__(self, keyframe: str = "grasp_start", **kwargs):
        super().__init__(keyframe=keyframe, **kwargs)

    def reward(self, obs, action, next_obs, absorbing):
        return 0

    def is_absorbing(self, obs):
        return False


if __name__ == "__main__":
    env = TrayPickUpGraspEnv()
    obs = env.reset()
    env.render()
    while True:
        action = np.zeros((18,))
        env.step(action)
        env.render()