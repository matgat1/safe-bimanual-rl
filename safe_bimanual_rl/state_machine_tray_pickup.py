from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.utils.record import VideoRecorder

from safe_bimanual_rl.environments.tray_pickup_reach_env import TrayPickUpReachEnv
from safe_bimanual_rl.rl_utils.actor_critic_sac_networks import (  # noqa: F401
    ActorNetwork,
    CriticNetwork,
)

import argparse


def grasping_position_reached(env, obs, consecutive_steps):
    # Same absorbing condition as TrayPickUpReachEnv.is_absorbing (position_reached case)
    return (
        env._position_reached(obs)
        and env._orientation_reached(obs)
        and consecutive_steps >= env._success_steps
    )


def lifting_position_reached(env, initial_cube_height):
    # Same absorbing condition as TrayPickUpGraspEnv._lift_reached()
    lift_target_height = 0.5
    cube_z = env._read_data("cube_pos")[2]
    return cube_z - initial_cube_height > lift_target_height


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--grasping_position_model_path", type=str, default=None)
    parser.add_argument("--grasp_model_path", type=str, default=None)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--n_episodes", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.grasping_position_model_path is None or args.grasp_model_path is None:
        raise ValueError(
            "Both --grasping_position_model_path and --grasp_model_path are required."
        )

    phase1_horizon = 200
    phase2_horizon = 300

    env = TrayPickUpReachEnv(gamma=0.99, horizon=phase1_horizon, n_substeps=4)

    grasping_position_agent = SAC.load(args.grasping_position_model_path)
    grasp_agent = SAC.load(args.grasp_model_path)

    recorder = VideoRecorder(video_name="state_machine") if args.record else None

    n_phase1_success = 0
    n_phase2_success = 0

    for episode in range(args.n_episodes):
        obs, _ = env.reset()

        # Phase 1: Use grasping position model to reach grasping position
        consecutive_success_steps = 0
        phase1_success = False

        for _ in range(phase1_horizon):
            action, _ = grasping_position_agent.draw_action(obs)
            obs, _, absorbing, _ = env.step(action)

            frame = env.render(args.record)
            if args.record:
                recorder(frame)

            if env._position_reached(obs) and env._orientation_reached(obs):
                consecutive_success_steps += 1
            else:
                consecutive_success_steps = 0

            if grasping_position_reached(env, obs, consecutive_success_steps):
                print(f"[Episode {episode + 1}] Grasping position reached")
                phase1_success = True
                n_phase1_success += 1
                break

            if absorbing:
                print(f"[Episode {episode + 1}] Phase 1 failed (contact force)")
                break

        if not phase1_success:
            continue

        # Phase 2: Use grasp model to grasp and lift the tray.
        initial_cube_height = env._read_data("cube_pos")[2]

        for _ in range(phase2_horizon):
            action, _ = grasp_agent.draw_action(obs)
            obs, _, _, _ = env.step(action)

            frame = env.render(args.record)
            if args.record:
                recorder(frame)

            if lifting_position_reached(env, initial_cube_height):
                print(f"[Episode {episode + 1}] Tray lifted successfully!")
                n_phase2_success += 1
                break

    if args.record:
        recorder.stop()
        print("Video saved to ./mushroom_rl_recordings/")

    print(
        f"\nResults over {args.n_episodes} episodes: "
        f"phase1={n_phase1_success}/{args.n_episodes}, "
        f"phase2={n_phase2_success}/{args.n_episodes}"
    )
