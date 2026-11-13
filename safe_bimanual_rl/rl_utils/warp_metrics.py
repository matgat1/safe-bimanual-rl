import torch


def episode_metrics(dataset) -> dict:
    """
    Per-episode statistics of a TrayPickUpWarpEnv evaluation dataset, averaged over
    the complete episodes: how often each stage of the task is reached and how close
    the policy gets to it.

    Args:
        dataset (Dataset): dataset returned by Core.evaluate, whose step info is
            filled by TrayPickUpWarpEnv._create_info_dictionary.

    Returns:
        A dictionary of floats.
    """
    info = dataset.info
    last = dataset.last.bool()
    ends = dataset.last_or_boundary.bool()

    # Episode index of every step, episodes cut by the end of the dataset are dropped
    episode = torch.cumsum(ends.long(), dim=0) - ends.long()
    complete = last[ends]
    n_episodes = int(ends.sum().item())

    def per_episode(values, reduce):
        out = torch.zeros(n_episodes, device=values.device)
        out = out.scatter_reduce(0, episode, values.float(), reduce=reduce, include_self=False)
        return out[complete]

    length = per_episode(torch.ones_like(info["tray_height"]), "sum")
    metrics = {
        "episodes": int(complete.sum().item()),
        "episode_length": length.mean(),
        # Fraction of episodes reaching each stage at least once
        "right_grasp_rate": per_episode(info["right_grasp"], "amax").mean(),
        "left_grasp_rate": per_episode(info["left_grasp"], "amax").mean(),
        "both_grasp_rate": per_episode(info["both_grasp"], "amax").mean(),
        "lifted_rate": per_episode(info["lifted"], "amax").mean(),
        "success_rate": per_episode(info["success"], "amax").mean(),
        # Fraction of the steps spent holding the tray with both hands
        "both_grasp_steps": (per_episode(info["both_grasp"], "sum") / length).mean(),
        # Best value reached during the episode
        "right_min_handle_dist": per_episode(info["right_handle_dist"], "amin").mean(),
        "left_min_handle_dist": per_episode(info["left_handle_dist"], "amin").mean(),
        "right_max_grip_force": per_episode(info["right_grip_force"], "amax").mean(),
        "left_max_grip_force": per_episode(info["left_grip_force"], "amax").mean(),
        "max_tray_height": per_episode(info["tray_height"], "amax").mean(),
        "max_tray_angle": per_episode(info["tray_angle"], "amax").mean(),
    }
    return {key: float(value) for key, value in metrics.items()}


def counts_delta(before: dict, after: dict) -> dict:
    """Difference between two snapshots of the environment counters."""
    return {key: after[key] - before[key] for key in after}


def termination_rates(absorbing_delta: dict, step_delta: dict) -> dict:
    """
    Fraction of the episodes started in the interval ending with each termination
    cause (the rest reach the horizon or are still running).
    """
    n_episodes = max(step_delta["episodes"], 1)
    return {key: value / n_episodes for key, value in absorbing_delta.items()}


def stage_step_rates(step_delta: dict) -> dict:
    """Fraction of the simulated steps spent in each stage of the task."""
    n_steps = max(step_delta["steps"], 1)
    return {
        key: value / n_steps
        for key, value in step_delta.items()
        if key not in ("episodes", "steps")
    }
