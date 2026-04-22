import numpy as np
import pytest
from safe_bimanual_rl.environments.reach_env import ReachEnv


@pytest.fixture(scope="module")
def env():
    return ReachEnv()


def test_env_instantiation(env):
    assert env is not None


def test_reset_returns_obs(env):
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert len(obs) > 0


def test_obs_shape_consistent(env):
    obs1, _ = env.reset()
    obs2, _ = env.reset()
    assert obs1.shape == obs2.shape


def test_step_zero_action_returns_tuple(env):
    env.reset()
    action = np.zeros(14)
    next_obs, reward, absorbing, info = env.step(action)
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(float(reward), float)
    assert isinstance(absorbing, (bool, np.bool_))


def test_reward_is_finite(env):
    env.reset()
    action = np.zeros(14)
    _, reward, _, _ = env.step(action)
    assert np.isfinite(reward)


def test_is_absorbing_returns_bool(env):
    obs, _ = env.reset()
    result = env.is_absorbing(obs)
    assert isinstance(result, (bool, np.bool_))


def test_ctrl_cost_zero_for_zero_action(env):
    cost = env._get_ctrl_cost(np.zeros(14))
    assert cost == 0.0


def test_ctrl_cost_negative_for_nonzero_action(env):
    cost = env._get_ctrl_cost(np.ones(14))
    assert cost < 0.0


def test_cube_distance_reward_formula():
    """Test the distance reward formula directly without env state."""
    sharpness = 0.5
    weight = 1.0

    def distance_reward(right_dist, left_dist):
        return weight * (
            (1 - np.tanh(right_dist / sharpness)) + (1 - np.tanh(left_dist / sharpness))
        )

    reward_near = distance_reward(0.0, 0.0)
    reward_far = distance_reward(1.0, 1.0)
    assert reward_near > reward_far
    assert reward_near == pytest.approx(2.0)
    assert reward_far < 2.0


def test_multiple_steps_without_crash(env):
    env.reset()
    for _ in range(10):
        action = np.random.uniform(-0.1, 0.1, size=14)
        env.step(action)
