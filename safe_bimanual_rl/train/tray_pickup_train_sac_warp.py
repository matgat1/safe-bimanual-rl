import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import wandb

from mushroom_rl.core import Core, Logger
from mushroom_rl.utils.torch_utils import TorchUtils
from tqdm import trange

from safe_bimanual_rl.environments.tray_pickup_warp_env import TrayPickUpWarpEnv
from safe_bimanual_rl.rl_utils.actor_critic_sac_networks import (
    ActorNetwork,
    CriticNetwork,
)
from safe_bimanual_rl.rl_utils.multi_update_sac import MultiUpdateSAC
from safe_bimanual_rl.rl_utils.plotting import save_plots


def evaluate(core, agent, n_episodes, quiet):
    dataset = core.evaluate(n_episodes=n_episodes, render=False, quiet=quiet)
    J = dataset.discounted_return.mean().item()
    R = dataset.undiscounted_return.mean().item()
    H = agent.policy.entropy(dataset.state).item()
    return J, R, H


def experiment(cfg: DictConfig):
    """Train a single SAC policy for the full tray pick-up (reach, grasp, lift) on GPU."""
    hydra_cfg = HydraConfig.get()
    save_dir = hydra_cfg.runtime.output_dir
    # Extract date, time, job_num from multirun output path: .../multirun/date/time/job_num
    date, time, job_num = save_dir.split(os.sep)[-3:]
    run_name = f"{cfg.model_name}_{date}_{time}_{job_num}"

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    TorchUtils.set_default_device(cfg.device)

    logger = Logger(MultiUpdateSAC.__name__, results_dir=None)
    logger.strong_line()
    logger.info(f"Experiment Algorithm: SAC - TrayPickUpWarpEnv ({cfg.n_envs} envs)")

    mdp = TrayPickUpWarpEnv(
        num_envs=cfg.n_envs,
        horizon=cfg.horizon,
        use_graph_capture=cfg.use_graph_capture,
        nconmax=cfg.nconmax,
        njmax=cfg.njmax,
        device=cfg.device,
        **cfg.reward,
    )

    # Actor
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = {
        "network": ActorNetwork,
        "n_features": cfg.n_features,
        "input_shape": actor_input_shape,
        "output_shape": mdp.info.action_space.shape,
    }
    actor_sigma_params = dict(actor_mu_params)
    actor_optimizer = {"class": optim.Adam, "params": {"lr": cfg.lr_actor}}

    # Critic
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = {
        "network": CriticNetwork,
        "optimizer": {"class": optim.Adam, "params": {"lr": cfg.lr_critic}},
        "loss": F.mse_loss,
        "n_features": cfg.n_features,
        "input_shape": critic_input_shape,
        "output_shape": (1,),
    }

    # Action space normalization to [-action_space_limit, action_space_limit]
    mdp.info.action_space.low[:] = -cfg.action_space_limit
    mdp.info.action_space.high[:] = cfg.action_space_limit

    # Agent
    agent = MultiUpdateSAC(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        cfg.batch_size,
        cfg.initial_replay_size,
        cfg.max_replay_size,
        cfg.warmup_transitions,
        cfg.tau,
        cfg.lr_alpha,
        target_entropy=cfg.target_entropy,
        critic_fit_params=None,
        n_updates_per_fit=cfg.n_updates_per_fit,
    )

    core = Core(agent, mdp)

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
    best_J = -np.inf

    def log_epoch(epoch, J, R, H):
        logger.epoch_info(epoch, J=J, R=R, entropy=H)
        run.log(
            {
                "Discounted Return (J)": J,
                "Undiscounted Return (R)": R,
                "Entropy (H)": H,
                "epoch": epoch,
                **{f"absorbing/{k}": v for k, v in mdp.get_absorbing_counts().items()},
            }
        )
        J_values.append(J)
        R_values.append(R)
        H_values.append(H)

    # Evaluation before training
    log_epoch(0, *evaluate(core, agent, cfg.n_episodes_test, cfg.use_cluster))

    # Replay initialisation
    core.learn(
        n_steps=cfg.initial_replay_size,
        n_steps_per_fit=cfg.initial_replay_size,
        quiet=cfg.use_cluster,
    )

    # Training loop
    for n in trange(cfg.n_epochs, leave=False):
        core.learn(
            n_steps=cfg.n_steps, n_steps_per_fit=cfg.n_steps_per_fit, quiet=cfg.use_cluster
        )
        J, R, H = evaluate(core, agent, cfg.n_episodes_test, cfg.use_cluster)
        log_epoch(n + 1, J, R, H)

        if cfg.save_model and J > best_J:
            best_J = J
            best_file = os.path.join(save_dir, f"{cfg.model_name}_best.msh")
            agent.save(best_file)
            logger.info(f"New best model saved (J={J:.2f}): {best_file}")

    for key, value in mdp.get_absorbing_counts().items():
        run.summary[f"absorbing/{key}"] = value
    run.finish()

    save_plots(
        {
            "Discounted Return (J)": J_values,
            "Undiscounted Return (R)": R_values,
            "Entropy (H)": H_values,
        },
        save_dir=save_dir,
        run_name=run_name,
    )

    if cfg.save_model:
        last_file = os.path.join(save_dir, f"{cfg.model_name}_last.msh")
        agent.save(last_file)
        logger.info(f"Last model saved: {last_file}")


@hydra.main(
    version_base=None, config_path="../configs", config_name="tray_pickup_sac_warp"
)
def main(cfg: DictConfig):
    """Entry point: parse Hydra config and run the experiment."""
    print(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    experiment(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
