from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core
from safe_bimanual_rl.environments.reach_env import ReachEnv
from safe_bimanual_rl.reach_point_experiment import (  # noqa: F401
    ActorNetwork,
    CriticNetwork,
)
import numpy as np
import argparse


def evaluate(
    environment: ReachEnv,
    agent,
    n_episodes: int = 5,
    record: bool = False,
):
    """
    Evaluate the agent on the environment.

    Args:
        environment (ReachEnv): The environment to evaluate on.
        agent (SAC): The agent to evaluate.
        n_episodes (int): The number of episodes to evaluate for.
    """

    mdp = environment

    core = Core(agent, mdp)

    # Evaluate the agent
    dataset = core.evaluate(n_episodes=n_episodes, render=True, record=record)
    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    print(f"J={J:.3f}, R={R:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--record", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model_path is None:
        raise ValueError(
            "Model path is required : "
            "Use --model_path to specify the path to the trained model."
        )
    env = ReachEnv(gamma=0.99, horizon=200, n_substeps=4)
    agent = SAC.load(args.model_path)
    evaluate(
        environment=env, agent=agent, n_episodes=args.n_episodes, record=args.record
    )
