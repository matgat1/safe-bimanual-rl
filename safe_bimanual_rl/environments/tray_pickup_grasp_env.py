import os
import mujoco
import numpy as np
from safe_bimanual_rl.environments.tray_pickup_base_env import TrayPickUpBaseEnv


class TrayPickUpGraspEnv(TrayPickUpBaseEnv):
    """
    Grasp phase: both grippers must close around the tray handles.
    Episodes start with grippers already positioned at the handles (reach phase complete).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_states_path = os.path.join(
            os.path.dirname(__file__), "data", "initial_states","grasp_init_states.npz"
        )
        data = np.load(self.init_states_path)
        self._init_states = list(zip(data["qpos"], data["qvel"], data["act"]))

    def setup(self, obs):
        super().setup(obs)
        qpos, qvel, act = self._init_states[np.random.randint(len(self._init_states))]
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._data.act[:] = act
        mujoco.mj_forward(self._model, self._data)

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
