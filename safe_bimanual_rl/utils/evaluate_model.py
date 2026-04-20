from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from safe_bimanual_rl.environments.bimanual_table_env import BimanualTableEnv
from safe_bimanual_rl.environments.reach_env import ReachEnv
from safe_bimanual_rl.environments.tray_pickup_env import TrayPickUpEnv
from safe_bimanual_rl.rl_utils.actor_critic_sac_networks import (  # noqa: F401
    ActorNetwork,
    CriticNetwork,
)
import numpy as np
import argparse

ENV_REGISTRY = {
    "reach_cube": ReachEnv,
    "tray_pickup": TrayPickUpEnv,
}


def _detect_env(model_path: str) -> str:
    lower = model_path.lower()
    for key in ENV_REGISTRY:
        if key in lower:
            return key
    raise ValueError(
        f"Could not detect environment from model path '{model_path}'. "
        f"Use --env with one of: {list(ENV_REGISTRY.keys())}"
    )


def evaluate(
    environment: BimanualTableEnv,
    agent,
    n_episodes: int = 3,
    record: bool = False,
):
    core = Core(agent, environment)
    dataset = core.evaluate(n_episodes=n_episodes, render=True, record=record)
    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    print(f"J={J:.3f}, R={R:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=list(ENV_REGISTRY.keys()),
        help="Environment to use. Auto-detected from model_path if not provided.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model_path is None:
        raise ValueError(
            "Model path is required : "
            "Use --model_path to specify the path to the trained model."
        )
    env_key = args.env if args.env is not None else _detect_env(args.model_path)
    print(f"Using environment: {env_key}")
    env = ENV_REGISTRY[env_key](gamma=0.99, horizon=200, n_substeps=4)
    agent = SAC.load(args.model_path)
    evaluate(
        environment=env, agent=agent, n_episodes=args.n_episodes, record=args.record
    )
