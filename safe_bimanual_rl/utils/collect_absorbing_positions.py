import numpy as np
import argparse
from pathlib import Path
from mushroom_rl.algorithms.actor_critic import SAC
from safe_bimanual_rl.environments import TrayPickUpEnv
from safe_bimanual_rl.rl_utils.actor_critic_sac_networks import (  # noqa: F401
    ActorNetwork,
    CriticNetwork,
)


def collect_absorbing_positions(
    environment,
    agent,
    n_episodes: int = 20,
    output_path: str = "absorbing_positions.npz",
    render: bool = False,
):
    """
    Run the agent and record qpos/qvel whenever a success absorbing state is reached
    (position + orientation reached, not contact force failure).

    Args:
        environment: A BimanualTableEnv subclass with _position_reached and
                     _orientation_reached methods.
        agent: A mushroom_rl agent (e.g. SAC) with a draw_action method.
        n_episodes: Number of episodes to run.
        output_path: Path to save the collected states as a .npz file.
        render: Whether to render the environment.

    Returns:
        dict with keys 'qpos', 'qvel', 'act', each an array of shape (n_successes, n).
    """
    qpos_list = []
    qvel_list = []
    act_list = []

    for ep in range(n_episodes):
        obs, _ = environment.reset()
        if render:
            environment.render()

        done = False
        while not done:
            action, _ = agent.draw_action(obs)
            next_obs, _, absorbing, _ = environment.step(action)

            if absorbing:
                done = True
                success = environment._position_reached(
                    next_obs
                ) and environment._orientation_reached(next_obs)
                if success:
                    qpos_list.append(environment._data.qpos.copy())
                    qvel_list.append(environment._data.qvel.copy())
                    act_list.append(environment._data.act.copy())
                    print(
                        f"Episode {ep + 1}/{n_episodes}: success — state recorded "
                        f"(total: {len(qpos_list)})"
                    )
                else:
                    print(f"Episode {ep + 1}/{n_episodes}: failure (contact force)")
            else:
                obs = next_obs

            if render:
                environment.render()

    if qpos_list:
        result = {
            "qpos": np.array(qpos_list),
            "qvel": np.array(qvel_list),
            "act": np.array(act_list),
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)
        print(
            f"\nSaved {len(qpos_list)}/{n_episodes} success states to '{output_path}'"
        )
    else:
        result = {"qpos": np.empty((0,)), "qvel": np.empty((0,)), "act": np.empty((0,))}
        print(f"\nNo success states recorded over {n_episodes} episodes.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="absorbing_positions.npz")
    parser.add_argument("--render", action="store_true", default=False)
    args = parser.parse_args()

    env = TrayPickUpEnv(gamma=0.99, horizon=200, n_substeps=4)
    agent = SAC.load(args.model_path)

    collect_absorbing_positions(
        environment=env,
        agent=agent,
        n_episodes=args.n_episodes,
        output_path=args.output_path,
        render=args.render,
    )
