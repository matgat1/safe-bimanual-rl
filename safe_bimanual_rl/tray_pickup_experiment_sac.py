import os
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import wandb

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from tqdm import trange

from safe_bimanual_rl.environments.tray_pickup_env import TrayPickUpEnv
from safe_bimanual_rl.rl_utils.actor_critic_sac_networks import (
    ActorNetwork,
    CriticNetwork,
)
from safe_bimanual_rl.rl_utils.plotting import save_plots


def experiment(
    n_epochs=100,
    n_steps=10000,
    n_steps_per_fit=1,
    n_episodes_test=5,
    initial_replay_size=10000,
    max_replay_size=200000,
    batch_size=256,
    n_features=128,
    warmup_transitions=10000,
    tau=0.003,
    lr_alpha=1e-4,
    lr_actor=3e-4,
    lr_critic=3e-4,
    use_cluster=False,
    save_model=False,
    model_name: str = "tray_pickup_agent",
    contact_cost_weight: float = -1e-4,
    handle_distance_weight: float = 3.0,
    contact_threshold: float = 2.0,
    control_cost_weight: float = -1e-4,
    reach_sharpness: float = 0.3,
    tray_push_penalty: float = -10.0,
    rotation_reward_weight: float = 1.0,
    orientation_sharpness: float = 0.5,
    success_reward: float = 50.0,
    success_threshold: float = 0.03,
    success_orientation_threshold: float = 0.3,
    action_space_limit: float = 0.4,
    use_wandb: bool = True,
):

    hydra_cfg = HydraConfig.get()
    save_dir = hydra_cfg.runtime.output_dir
    # Extract date, time, job_num from multirun output path: .../multirun/date/time/job_num
    date, time, job_num = save_dir.split(os.sep)[-3:]
    run_name = f"{model_name}_{date}_{time}_{job_num}"

    np.random.seed()

    logger = Logger(SAC.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: SAC - TrayPickUpEnv")

    # Load Environment
    mdp = TrayPickUpEnv(
        gamma=0.99,
        horizon=150,
        n_substeps=4,
        contact_cost_weight=contact_cost_weight,
        handle_distance_weight=handle_distance_weight,
        contact_threshold=contact_threshold,
        control_cost_weight=control_cost_weight,
        reach_sharpness=reach_sharpness,
        tray_push_penalty=tray_push_penalty,
        rotation_reward_weight=rotation_reward_weight,
        orientation_sharpness=orientation_sharpness,
        success_reward=success_reward,
        success_threshold=success_threshold,
        success_orientation_threshold=success_orientation_threshold,
    )

    # Actor
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = {
        "network": ActorNetwork,
        "n_features": n_features,
        "input_shape": actor_input_shape,
        "output_shape": mdp.info.action_space.shape,
    }
    actor_sigma_params = {
        "network": ActorNetwork,
        "n_features": n_features,
        "input_shape": actor_input_shape,
        "output_shape": mdp.info.action_space.shape,
    }
    actor_optimizer = {"class": optim.Adam, "params": {"lr": lr_actor}}

    # Critic
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr_critic}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Action space normalization to [-action_space_limit, action_space_limit]
    mdp.info.action_space.low[:] = -action_space_limit
    mdp.info.action_space.high[:] = action_space_limit

    # Agent
    agent = SAC(
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        critic_fit_params=None,
    )

    core = Core(agent, mdp)

    # wandb initialization
    run = wandb.init(
        entity="matgat1-lth",
        project="safe_bimanual_rl",
        name=run_name,
        group=f"{model_name}_{date}_{time}",
        mode="online" if use_wandb else "disabled",
        settings=wandb.Settings(silent=True),
        config={
            "n_epochs": n_epochs,
            "n_steps": n_steps,
            "n_steps_per_fit": n_steps_per_fit,
            "n_episodes_test": n_episodes_test,
            "initial_replay_size": initial_replay_size,
            "max_replay_size": max_replay_size,
            "batch_size": batch_size,
            "n_features": n_features,
            "warmup_transitions": warmup_transitions,
            "tau": tau,
            "lr_alpha": lr_alpha,
            "lr_actor": lr_actor,
            "lr_critic": lr_critic,
            "contact_cost_weight": contact_cost_weight,
            "handle_distance_weight": handle_distance_weight,
            "contact_threshold": contact_threshold,
            "control_cost_weight": control_cost_weight,
            "reach_sharpness": reach_sharpness,
            "rotation_reward_weight": rotation_reward_weight,
            "orientation_sharpness": orientation_sharpness,
            "action_space_limit": action_space_limit,
        },
    )

    J_values, R_values, H_values = [], [], []

    # Evaluation before training
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    H = agent.policy.entropy(dataset.state).item()

    logger.epoch_info(0, J=J, R=R, entropy=H)
    run.log(
        {
            "Discounted Return (J)": J,
            "Undiscounted Return (R)": R,
            "Entropy (H)": H,
            "epoch": 0,
        }
    )
    J_values.append(J)
    R_values.append(R)
    H_values.append(H)

    # Replay initialisation
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # Training loop
    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=5, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        H = agent.policy.entropy(dataset.state).item()
        logger.epoch_info(n + 1, J=J, R=R, entropy=H)

        run.log(
            {
                "Discounted Return (J)": J,
                "Undiscounted Return (R)": R,
                "Entropy (H)": H,
                "epoch": n + 1,
            }
        )
        J_values.append(J)
        R_values.append(R)
        H_values.append(H)

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

    if not use_cluster:
        # Final visualization
        logger.info("Press a button to visualize")
        input()
        core.evaluate(n_episodes=5, render=True)
    else:
        core.evaluate(n_episodes=5, render=False)
        logger.info("Experiment finished.")

    if save_model:
        file_name = f"{model_name}.msh"
        agent.save(os.path.join(save_dir, file_name))

        logger.info(f"Model saved : {save_dir}/{file_name}")


@hydra.main(version_base=None, config_path="configs", config_name="tray_pickup_sac")
def main(cfg: DictConfig):
    print(f"Running with config:\n{cfg}")
    experiment(
        n_epochs=cfg.n_epochs,
        n_steps=cfg.n_steps,
        n_steps_per_fit=cfg.n_steps_per_fit,
        n_episodes_test=cfg.n_episodes_test,
        initial_replay_size=cfg.initial_replay_size,
        max_replay_size=cfg.max_replay_size,
        batch_size=cfg.batch_size,
        n_features=cfg.n_features,
        warmup_transitions=cfg.warmup_transitions,
        tau=cfg.tau,
        lr_alpha=cfg.lr_alpha,
        lr_actor=cfg.lr_actor,
        lr_critic=cfg.lr_critic,
        use_cluster=cfg.use_cluster,
        save_model=cfg.save_model,
        model_name=cfg.model_name,
        use_wandb=cfg.use_wandb,
        contact_cost_weight=cfg.contact_cost_weight,
        handle_distance_weight=cfg.handle_distance_weight,
        contact_threshold=cfg.contact_threshold,
        control_cost_weight=cfg.control_cost_weight,
        reach_sharpness=cfg.reach_sharpness,
        tray_push_penalty=cfg.tray_push_penalty,
        rotation_reward_weight=cfg.rotation_reward_weight,
        orientation_sharpness=cfg.orientation_sharpness,
        success_reward=cfg.success_reward,
        success_threshold=cfg.success_threshold,
        success_orientation_threshold=cfg.success_orientation_threshold,
        action_space_limit=cfg.action_space_limit,
    )


if __name__ == "__main__":
    main()
