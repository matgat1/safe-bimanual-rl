"""
Visualize a PPO policy trained with train/tray_pickup_train_ppo_warp.py in TrayPickUpWarpEnv.

The environment is rebuilt with the exact parameters of the training run, read from the
Hydra config saved next to the model (<run_dir>/.hydra/config.yaml), so copy the whole run
directory, not only the .msh file.

Examples:
    # Window (needs a display), mean action of the policy
    python -m safe_bimanual_rl.utils.visualize_ppo_warp \
        --model_path multirun/2026-11-14/14-44-54/0/tray_pickup_ppo_warp_best.msh

    # Headless (e.g. over ssh): record a video instead of opening a window
    MUJOCO_GL=egl python -m safe_bimanual_rl.utils.visualize_ppo_warp \
        --model_path .../tray_pickup_ppo_warp_best.msh --record --n_episodes 3
"""

import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.utils.record import VideoRecorder
from mushroom_rl.utils.torch_utils import TorchUtils

from safe_bimanual_rl.environments.tray_pickup_warp_env import TrayPickUpWarpEnv

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "tray_pickup_ppo_warp.yaml"
)
DEFAULT_REWARD_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "reward", "tray_pickup_warp.yaml"
)


def load_config(model_path, config_path):
    """Training config of the run (Hydra), or the default config of the repository."""
    if config_path is None:
        run_config = os.path.join(os.path.dirname(model_path), ".hydra", "config.yaml")
        if os.path.exists(run_config):
            config_path = run_config
    if config_path is not None:
        print(f"Config: {config_path}")
        return OmegaConf.load(config_path)

    print("No .hydra/config.yaml next to the model: using the default config of the repository")
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    cfg.reward = OmegaConf.load(DEFAULT_REWARD_CONFIG)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", default=None, help="default: <model dir>/.hydra/config.yaml")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="cuda:0 (default when available) or cpu",
    )
    parser.add_argument("--stochastic", action="store_true",
                        help="sample the actions (with exploration noise) instead of the mean")
    parser.add_argument("--record", action="store_true",
                        help="save a video instead of opening a window (works without display)")
    parser.add_argument("--video_dir", default="videos")
    args = parser.parse_args()

    cfg = load_config(args.model_path, args.config)
    TorchUtils.set_default_device(args.device)
    use_cuda = args.device.startswith("cuda")

    env = TrayPickUpWarpEnv(
        num_envs=1,
        gamma=cfg.gamma,
        horizon=cfg.horizon,
        use_graph_capture=use_cuda,
        nconmax=cfg.nconmax,
        njmax=cfg.njmax,
        device=args.device,
        headless=args.record,
        **{**cfg.reward, "start_state_prob": 0.0},  # always start from the home pose
    )
    env.info.action_space.low[:] = -cfg.action_space_limit
    env.info.action_space.high[:] = cfg.action_space_limit

    agent = PPO.load(args.model_path)
    preprocessor = agent.core_preprocessors[0]  # observation standardization (not updated here)
    draw = agent.draw_action if args.stochastic else agent.draw_action_greedy

    recorder = None
    if args.record:
        name = os.path.splitext(os.path.basename(args.model_path))[0]
        recorder = VideoRecorder(path=args.video_dir, video_name=name, fps=int(round(1 / env.dt)))

    mask = torch.ones(1, dtype=torch.bool, device=args.device)
    for episode in range(args.n_episodes):
        obs, _ = env.reset_all(mask)
        info = {}
        for step in range(cfg.horizon):
            action = draw(preprocessor(obs))
            if isinstance(action, tuple):
                action = action[0]
            obs, _, absorbing, info = env.step_all(mask, action)

            frame = env.render_all(mask, record=args.record)
            if recorder is not None:
                recorder(frame)

            if bool(absorbing[0]):
                break

        print(
            f"Episode {episode + 1}: {step + 1} steps, "
            f"success={bool(info['success'][0])}, "
            f"max tray height reached at the end={float(info['tray_height'][0]):.3f} m, "
            f"both hands grasping={bool(info['both_grasp'][0])}"
        )

    if recorder is not None:
        path = recorder.stop()
        print(f"Video saved to {path}")
    env.stop()


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
