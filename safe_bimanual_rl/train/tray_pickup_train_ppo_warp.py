import os
import sys
from time import perf_counter
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import wandb

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.approximators.parametric.networks import ActorNetwork, FeedForwardNetwork
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.rl_utils.preprocessors import StandardizationPreprocessor
from mushroom_rl.utils.torch_utils import TorchUtils
from tqdm import trange

from safe_bimanual_rl.environments.tray_pickup_warp_env import (
    N_ARM_ACTIONS,
    START_GROUPS,
    TrayPickUpWarpEnv,
)
from safe_bimanual_rl.rl_utils.persistent_core import PersistentVectorizedCore
from safe_bimanual_rl.rl_utils.plotting import save_plots
from safe_bimanual_rl.rl_utils.warp_metrics import (
    counts_delta,
    episode_metrics,
    stage_step_rates,
    termination_rates,
)


def evaluate(core, agent, mdp, n_episodes, quiet, greedy=False):
    """
    Evaluate the policy, with the task-stage and termination statistics of the episodes.
    With greedy=True the policy plays its mean action (no exploration noise), the metrics
    are prefixed by "eval_greedy/" instead of "eval/".
    """
    prefix = "eval_greedy" if greedy else "eval"
    absorbing_before, steps_before = mdp.get_absorbing_counts(), mdp.get_step_counts()
    dataset = core.evaluate(n_episodes=n_episodes, render=False, quiet=quiet, greedy=greedy)
    J = dataset.discounted_return.mean().item()
    R = dataset.undiscounted_return.mean().item()
    H = agent.policy.entropy().item()

    absorbing_delta = counts_delta(absorbing_before, mdp.get_absorbing_counts())
    step_delta = counts_delta(steps_before, mdp.get_step_counts())
    metrics = {
        **{f"{prefix}/{k}": v for k, v in episode_metrics(dataset).items()},
        **{
            f"{prefix}/termination/{k}": v
            for k, v in termination_rates(absorbing_delta, step_delta).items()
        },
    }
    return J, R, H, metrics


def policy_std_metrics(agent, action_dim):
    """Std of the Gaussian policy (state independent), overall and for the arm/gripper actions."""
    std = agent.policy.get_weights()[-action_dim:].exp()
    # Action layout: 14 arm joints then the grippers (2 actions, or 4 fingers)
    return {
        "policy/std_mean": std.mean().item(),
        "policy/std_min": std.min().item(),
        "policy/std_arm": std[:N_ARM_ACTIONS].mean().item(),
        "policy/std_fingers": std[N_ARM_ACTIONS:].mean().item(),
    }


def std_clamp_callback(policy, std_min, std_max):
    """
    Fit callback keeping every std of the Gaussian policy in [std_min, std_max]: the floor
    keeps exploring, the cap stops the entropy bonus from pushing the std above the action
    limit, where most actions would be clipped to +-action_space_limit.
    """
    log_std_min, log_std_max = float(np.log(std_min)), float(np.log(std_max))

    def callback(_dataset):
        with torch.no_grad():
            # pylint: disable=protected-access
            policy._log_sigma.clamp_(min=log_std_min, max=log_std_max)

    return callback


def start_group_metrics(before, after):
    """
    Training statistics per start of the episodes (home pose or demonstration stage):
    share of the steps, steps with both hands grasping, mean tray height, steps above 5 cm.
    """
    delta = {key: [a - b for a, b in zip(after[key], before[key])] for key in after}
    total = max(sum(delta["steps"]), 1.0)
    metrics = {}
    for i, group in enumerate(START_GROUPS):
        steps = delta["steps"][i]
        metrics[f"train/start/{group}/step_share"] = steps / total
        if steps > 0:
            metrics[f"train/start/{group}/both_grasp_steps"] = delta["both_grasp"][i] / steps
            metrics[f"train/start/{group}/mean_tray_height"] = delta["height_sum"][i] / steps
            metrics[f"train/start/{group}/height_5cm_steps"] = delta["height_5cm"][i] / steps
    return metrics


def start_state_prob(cfg, epoch):
    """Linear decay of the probability of starting from a demonstration state (curriculum)."""
    progress = min(epoch / max(cfg.start_state_decay_epochs, 1), 1.0)
    p0, p1 = cfg.reward.start_state_prob, cfg.start_state_prob_final
    return p0 + progress * (p1 - p0)


def experiment(cfg: DictConfig):
    """Train a single PPO policy for the full tray pick-up (reach, grasp, lift) on GPU."""
    hydra_cfg = HydraConfig.get()
    save_dir = hydra_cfg.runtime.output_dir
    # Extract date, time, job_num from multirun output path: .../multirun/date/time/job_num
    date, time, job_num = save_dir.split(os.sep)[-3:]
    run_name = f"{cfg.model_name}_{date}_{time}_{job_num}"

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    TorchUtils.set_default_device(cfg.device)

    logger = Logger(PPO.__name__, results_dir=None)
    logger.strong_line()
    logger.info(f"Experiment Algorithm: PPO - TrayPickUpWarpEnv ({cfg.n_envs} envs)")

    def make_env(n_envs, **overrides):
        env = TrayPickUpWarpEnv(
            num_envs=n_envs,
            gamma=cfg.gamma,
            horizon=cfg.horizon,
            use_graph_capture=cfg.use_graph_capture,
            nconmax=cfg.nconmax,
            njmax=cfg.njmax,
            device=cfg.device,
            **{**cfg.reward, **overrides},
        )
        # Actions outside [-action_space_limit, action_space_limit] are clipped by the env
        # (with the binary gripper only the sign of the gripper actions matters)
        env.info.action_space.low[:] = -cfg.action_space_limit
        env.info.action_space.high[:] = cfg.action_space_limit
        return env

    mdp = make_env(cfg.n_envs)
    # Separate evaluation worlds, so that evaluating never cuts the training episodes.
    # Evaluation always starts from the home pose: it measures the full task.
    eval_mdp = make_env(cfg.n_envs_eval, start_state_prob=0.0)

    policy = GaussianTorchPolicy(
        ActorNetwork,
        mdp.info.observation_space.shape,
        mdp.info.action_space.shape,
        std_0=cfg.std_0,
        n_features=cfg.n_features,
        n_layers=cfg.n_layers,
        activation="elu",
        gain_scale=0.1,
    )

    critic_params = {
        "network": FeedForwardNetwork,
        "optimizer": {"class": optim.Adam, "params": {"lr": cfg.lr_critic}},
        "loss": F.mse_loss,
        "n_features": cfg.n_features,
        "n_layers": cfg.n_layers,
        "activation": "elu",
        "batch_size": cfg.batch_size,
        "input_shape": mdp.info.observation_space.shape,
        "output_shape": (1,),
    }

    agent = PPO(
        mdp.info,
        policy,
        actor_optimizer={"class": optim.Adam, "params": {"lr": cfg.lr_actor}},
        critic_params=critic_params,
        n_epochs_policy=cfg.n_epochs_policy,
        batch_size=cfg.batch_size,
        eps_ppo=cfg.eps_ppo,
        lam=cfg.lam,
        ent_coeff=cfg.ent_coeff,
        critic_fit_params={"n_epochs": cfg.n_epochs_critic},
    )
    agent.add_core_preprocessor(StandardizationPreprocessor(mdp.info))

    core = PersistentVectorizedCore(
        agent, mdp, callbacks_fit=[std_clamp_callback(policy, cfg.std_min, cfg.std_max)]
    )
    eval_core = Core(agent, eval_mdp)

    run = wandb.init(
        entity="matgat1-lth",
        project="safe_bimanual_rl",
        name=run_name,
        group=f"{cfg.model_name}_{date}_{time}",
        mode="online" if cfg.use_wandb else "disabled",
        settings=wandb.Settings(silent=True),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    J_values, R_values, H_values = [], [], []
    grasp_values, success_values = [], []
    best_J = -np.inf
    action_dim = mdp.info.action_space.shape[0]

    def run_evaluations():
        """Stochastic and greedy evaluations, returns J, R, H, metrics, greedy J, greedy R."""
        J, R, H, metrics = evaluate(
            eval_core, agent, eval_mdp, cfg.n_episodes_test, cfg.use_cluster
        )
        J_greedy, R_greedy, _, metrics_greedy = evaluate(
            eval_core, agent, eval_mdp, cfg.n_episodes_test, cfg.use_cluster, greedy=True
        )
        return J, R, H, {**metrics, **metrics_greedy}, J_greedy, R_greedy

    def log_epoch(epoch, J, R, H, metrics, J_greedy, R_greedy):
        logger.epoch_info(
            epoch,
            J=J,
            R=R,
            J_greedy=J_greedy,
            entropy=H,
            right_grasp=metrics["eval/right_grasp_rate"],
            left_grasp=metrics["eval/left_grasp_rate"],
            both_grasp=metrics["eval/both_grasp_rate"],
            both_grasp_greedy=metrics["eval_greedy/both_grasp_rate"],
            max_height=metrics["eval/max_tray_height"],
            success=metrics["eval/success_rate"],
            success_greedy=metrics["eval_greedy/success_rate"],
        )
        run.log(
            {
                "Discounted Return (J)": J,
                "Undiscounted Return (R)": R,
                "Discounted Return greedy (J)": J_greedy,
                "Undiscounted Return greedy (R)": R_greedy,
                "Entropy (H)": H,
                "epoch": epoch,
                **{f"absorbing/{k}": v for k, v in mdp.get_absorbing_counts().items()},
                **metrics,
                **policy_std_metrics(agent, action_dim),
            }
        )
        J_values.append(J)
        R_values.append(R)
        H_values.append(H)
        grasp_values.append(metrics["eval/both_grasp_rate"])
        success_values.append(metrics["eval/success_rate"])

    # Evaluation before training
    log_epoch(0, *run_evaluations())

    # Training loop
    for n in trange(cfg.n_epochs, leave=False):
        mdp.set_start_state_prob(start_state_prob(cfg, n))
        absorbing_before, steps_before = mdp.get_absorbing_counts(), mdp.get_step_counts()
        groups_before = mdp.get_group_counts()
        start = perf_counter()
        core.learn(
            n_steps=cfg.n_steps, n_steps_per_fit=cfg.n_steps_per_fit, quiet=cfg.use_cluster
        )
        learn_time = perf_counter() - start
        logger.info(f"Learn: {learn_time:.1f}s ({cfg.n_steps / learn_time:.0f} env steps/s)")

        # Statistics of the training rollouts of this epoch
        absorbing_delta = counts_delta(absorbing_before, mdp.get_absorbing_counts())
        step_delta = counts_delta(steps_before, mdp.get_step_counts())
        run.log(
            {
                "time/learn": learn_time,
                "epoch": n + 1,
                "train/episodes": step_delta["episodes"],
                "train/start_state_prob": start_state_prob(cfg, n),
                "train/demo_episodes": step_delta["demo_episodes"] / max(step_delta["episodes"], 1),
                **{
                    f"train/termination/{k}": v
                    for k, v in termination_rates(absorbing_delta, step_delta).items()
                },
                **{f"train/{k}_steps": v for k, v in stage_step_rates(step_delta).items()},
                **start_group_metrics(groups_before, mdp.get_group_counts()),
                **policy_std_metrics(agent, action_dim),
            }
        )

        # Evaluation simulates every world, so it costs as much as n_envs episodes
        if (n + 1) % cfg.eval_every != 0 and n + 1 != cfg.n_epochs:
            continue
        start = perf_counter()
        J, R, H, metrics, J_greedy, R_greedy = run_evaluations()
        logger.info(f"Evaluation: {perf_counter() - start:.1f}s")
        log_epoch(n + 1, J, R, H, metrics, J_greedy, R_greedy)

        # The best model is the one whose mean action (the deployed policy) does best
        if cfg.save_model and J_greedy > best_J:
            best_J = J_greedy
            best_file = os.path.join(save_dir, f"{cfg.model_name}_best.msh")
            agent.save(best_file)
            logger.info(f"New best model saved (greedy J={J_greedy:.2f}): {best_file}")

    for key, value in mdp.get_absorbing_counts().items():
        run.summary[f"absorbing/{key}"] = value
    for key, value in mdp.get_step_counts().items():
        run.summary[f"steps/{key}"] = value
    run.finish()

    save_plots(
        {
            "Discounted Return (J)": J_values,
            "Undiscounted Return (R)": R_values,
            "Entropy (H)": H_values,
            "Both-hands grasp rate": grasp_values,
            "Success rate": success_values,
        },
        save_dir=save_dir,
        run_name=run_name,
    )

    if cfg.save_model:
        last_file = os.path.join(save_dir, f"{cfg.model_name}_last.msh")
        agent.save(last_file)
        logger.info(f"Last model saved: {last_file}")


@hydra.main(version_base=None, config_path="../configs", config_name="tray_pickup_ppo_warp")
def main(cfg: DictConfig):
    """Entry point: parse Hydra config and run the experiment."""
    # Flush every line, so that the logs show up immediately when piped (e.g. into tee)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    print(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    experiment(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
