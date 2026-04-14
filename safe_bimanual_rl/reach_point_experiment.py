import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from tqdm import trange
import argparse

from safe_bimanual_rl.environments.reach_env import ReachEnv


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        return torch.squeeze(self._h3(features2))


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        return self._h3(features2)


def experiment(
    n_epochs=100,
    n_steps=4000,
    n_steps_test=2000,
    use_cluster=False,
    save_model=False,
    model_name: str = "sac_agent",
):
    np.random.seed()

    logger = Logger(SAC.__name__, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: SAC - ReachEnv")

    # Load Environment
    mdp = ReachEnv(gamma=0.99, horizon=200, n_substeps=4)

    # Hyperparameters
    initial_replay_size = 5000
    max_replay_size = 200000
    batch_size = 256
    n_features = 128
    warmup_transitions = 10000
    tau = 0.001
    lr_alpha = 3e-4

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
    actor_optimizer = {"class": optim.Adam, "params": {"lr": 5e-4}}

    # Critic
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": 5e-4}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
    )

    # Action space normalization to [-1, 1]
    mdp.info.action_space.low[:] = -1.0
    mdp.info.action_space.high[:] = 1.0

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

    # Evaluation before training
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    logger.epoch_info(0, J=J, R=R)

    # Replay initialisation
    core.learn(
        n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True
    )

    # Training loop
    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)
        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        logger.epoch_info(n + 1, J=J, R=R)

    if not use_cluster:
        # Final visualization
        logger.info("Press a button to visualize")
        input()
        core.evaluate(n_episodes=5, render=True)
    else:
        core.evaluate(n_episodes=5, render=False)
        logger.info("Experiment finished.")

    if save_model:
        import os

        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{model_name}.msh"
        agent.save(os.path.join(save_dir, file_name))

        logger.info(f"Model saved : {save_dir}/{file_name}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_cluster", action="store_true", default=False)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=4000)
    parser.add_argument("--nsteps_tests", type=int, default=2000)
    parser.add_argument("--model_name", type=str, default="sac_agent")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Running experiment with the following parameters:")
    print(
        f"use_cluster={args.use_cluster}, save_model={args.save_model}, "
        f"n_epochs={args.n_epochs}, n_steps={args.n_steps}, nsteps_tests={args.nsteps_tests}"
    )
    experiment(
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        n_steps_test=args.nsteps_tests,
        use_cluster=args.use_cluster,
        save_model=args.save_model,
        model_name=args.model_name,
    )
