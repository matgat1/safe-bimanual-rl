import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import wandb

from mushroom_rl.core import Logger
from tqdm import trange

from safe_bimanual_rl.environments.reach_env import ReachEnv
from safe_bimanual_rl.rl_utils.actor_critic_sac_networks import (
    ActorNetwork,
    CriticNetwork,
)
from safe_bimanual_rl.rl_utils.plotting import save_plots
from safe_bimanual_rl.d_atacom.algorithms.datacom_sac import DatacomSAC
from safe_bimanual_rl.d_atacom.dynamics.bimanual_dynamics import BimanualArmDynamics
from safe_bimanual_rl.d_atacom.utils.safe_core import SafeCore


class ConstraintNetwork(nn.Module):
    """Two-headed network for the learned CBF: outputs (mean, log_std)."""

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._mu = nn.Linear(n_features, output_shape[0])
        self._log_std = nn.Linear(n_features, output_shape[0])

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._mu.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self._log_std.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        x = F.relu(self._h1(state.float()))
        x = F.relu(self._h2(x))
        return self._mu(x), self._log_std(x)


def experiment(
    n_epochs=100,
    n_steps=8000,
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
    lr_constraint=5e-4,
    accepted_risk=0.9,
    atacom_lam=25.0,
    atacom_beta=3.0,
    cost_budget=1.0,
    lr_delta=0.05,
    init_delta=3.0,
    delta_warmup_transitions=50000,
    acc_limit=1.0,
    use_cluster=False,
    save_model=False,
    model_name: str = "reach_cube_datacom_agent",
    contact_cost_weight: float = -1e-4,
    cube_distance_weight: float = 3.0,
    cube_touched_reward: float = 3.0,
    contact_threshold: float = 2.0,
    control_cost_weight: float = -1e-4,
    reach_sharpness: float = 0.5,
    cube_displacement_weight: float = -5.0,
    action_space_limit: float = 0.4,
    use_wandb: bool = True,
):
    hydra_cfg = HydraConfig.get()
    save_dir = hydra_cfg.runtime.output_dir
    date, time, job_num = save_dir.split(os.sep)[-3:]
    run_name = f"{model_name}_{date}_{time}_{job_num}"

    np.random.seed()
    torch.manual_seed(0)

    logger = Logger(DatacomSAC.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: DatacomSAC - ReachEnv")

    # Environment
    mdp = ReachEnv(
        gamma=0.99,
        horizon=200,
        n_substeps=4,
        contact_cost_weight=contact_cost_weight,
        cube_distance_weight=cube_distance_weight,
        cube_touched_reward=cube_touched_reward,
        contact_threshold=contact_threshold,
        control_cost_weight=control_cost_weight,
        reach_sharpness=reach_sharpness,
        cube_displacement_weight=cube_displacement_weight,
    )
    mdp.info.action_space.low[:] = -action_space_limit
    mdp.info.action_space.high[:] = action_space_limit

    # Control-affine dynamics for ATACOM
    # State layout: [joint_pos(0-13), joint_vel(14-27), ee_pos(28-33),
    #                rel_cube_right(34-36), rel_cube_left(37-39), contact_force(40)]
    control_system = BimanualArmDynamics(acc_limit=acc_limit)

    actor_input_shape = mdp.info.observation_space.shape

    # Actor networks
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

    # Critic network (takes state + action as input)
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr_critic}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Constraint network: takes [joint_pos, joint_vel, ee_pos] = 34 dims
    # (dim_q + dim_x from BimanualArmDynamics = 28 + 13 = 41... but ee_pos is 6, not 13)
    # Adjust: the constraint state is get_q(obs) + get_x(obs) concatenated
    constraint_input_dim = control_system.dim_q + control_system.dim_x  # 28 + 13 = 41
    constraint_params = dict(
        network=ConstraintNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": lr_constraint}},
        n_features=n_features,
        input_shape=(constraint_input_dim,),
        output_shape=(1,),
    )

    # Agent
    agent = DatacomSAC(
        mdp_info=mdp.info,
        control_system=control_system,
        accepted_risk=accepted_risk,
        actor_mu_params=actor_mu_params,
        actor_sigma_params=actor_sigma_params,
        actor_optimizer=actor_optimizer,
        critic_params=critic_params,
        batch_size=batch_size,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        warmup_transitions=warmup_transitions,
        tau=tau,
        lr_alpha=lr_alpha,
        cost_budget=cost_budget,
        constraint_params=constraint_params,
        atacom_lam=atacom_lam,
        atacom_beta=atacom_beta,
        lr_delta=lr_delta,
        init_delta=init_delta,
        delta_warmup_transitions=delta_warmup_transitions,
    )

    core = SafeCore(agent, mdp)

    # Wandb
    run = wandb.init(
        entity="matgat1-lth",
        project="safe_bimanual_rl",
        name=run_name,
        group=f"{model_name}_{date}_{time}",
        mode="online" if use_wandb else "disabled",
        settings=wandb.Settings(silent=True),
        config=dict(
            n_epochs=n_epochs, n_steps=n_steps, batch_size=batch_size,
            accepted_risk=accepted_risk, atacom_lam=atacom_lam, atacom_beta=atacom_beta,
            cost_budget=cost_budget, init_delta=init_delta, acc_limit=acc_limit,
        ),
    )

    J_values, cost_values = [], []

    # Initial evaluation
    eval_data = core.evaluate(n_episodes=n_episodes_test, render=False)
    logger.epoch_info(0, J=eval_data["J"], mean_cost=eval_data["mean_cost"])
    run.log({"J": eval_data["J"], "mean_cost": eval_data["mean_cost"], "epoch": 0})
    J_values.append(eval_data["J"])
    cost_values.append(eval_data["mean_cost"])

    # Replay buffer warm-up
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # Training loop
    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        eval_data = core.evaluate(n_episodes=n_episodes_test, render=False)

        logger.epoch_info(n + 1, J=eval_data["J"], mean_cost=eval_data["mean_cost"])
        run.log({"J": eval_data["J"], "mean_cost": eval_data["mean_cost"], "epoch": n + 1})
        J_values.append(eval_data["J"])
        cost_values.append(eval_data["mean_cost"])

    run.finish()

    save_plots(
        {"Discounted Return (J)": J_values, "Mean Cost": cost_values},
        save_dir=save_dir,
        run_name=run_name,
    )

    if save_model:
        file_name = f"{model_name}.msh"
        agent.save(os.path.join(save_dir, file_name))
        logger.info(f"Model saved: {save_dir}/{file_name}")

    if not use_cluster:
        logger.info("Press a button to visualize")
        input()
        core.evaluate(n_episodes=5, render=True)
    else:
        core.evaluate(n_episodes=5, render=False)
        logger.info("Experiment finished.")


@hydra.main(version_base=None, config_path="configs", config_name="reach_cube_datacom")
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
        lr_constraint=cfg.lr_constraint,
        accepted_risk=cfg.accepted_risk,
        atacom_lam=cfg.atacom_lam,
        atacom_beta=cfg.atacom_beta,
        cost_budget=cfg.cost_budget,
        lr_delta=cfg.lr_delta,
        init_delta=cfg.init_delta,
        delta_warmup_transitions=cfg.delta_warmup_transitions,
        acc_limit=cfg.acc_limit,
        use_cluster=cfg.use_cluster,
        save_model=cfg.save_model,
        model_name=cfg.model_name,
        use_wandb=cfg.use_wandb,
        contact_cost_weight=cfg.contact_cost_weight,
        cube_distance_weight=cfg.cube_distance_weight,
        cube_touched_reward=cfg.cube_touched_reward,
        contact_threshold=cfg.contact_threshold,
        control_cost_weight=cfg.control_cost_weight,
        reach_sharpness=cfg.reach_sharpness,
        cube_displacement_weight=cfg.cube_displacement_weight,
        action_space_limit=cfg.action_space_limit,
    )


if __name__ == "__main__":
    main()
