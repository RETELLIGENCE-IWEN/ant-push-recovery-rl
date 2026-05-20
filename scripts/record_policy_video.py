from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gymnasium as gym
import imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from stable_directional_ant import (
    ControlledLocomotionAntWrapper,
    ControlledLocomotionRewardConfig,
    LateralErrorObservationWrapper,
    ObservationHistoryStackWrapper,
    WellTrainedLocomotionAntWrapper,
    WellTrainedLocomotionRewardConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runs/baseline_ppo_ant_50k/models/final_model.zip",
    )
    parser.add_argument("--out", type=str, default="videos/baseline_ppo_ant_50k.mp4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--include-lateral-error-observation", action="store_true")
    parser.add_argument("--lateral-error-observation-clip", type=float, default=5.0)
    parser.add_argument("--use-controlled-locomotion-wrapper", action="store_true")
    parser.add_argument("--controlled-target-forward-velocity", type=float, default=2.0)
    parser.add_argument("--controlled-target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--controlled-target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--controlled-target-yaw", type=float, default=0.0)
    parser.add_argument("--controlled-target-height", type=float, default=0.53)
    parser.add_argument("--controlled-target-velocity-obs-scale", type=float, default=3.0)
    parser.add_argument("--controlled-target-yaw-rate-obs-scale", type=float, default=2.0)
    parser.add_argument("--controlled-include-command-observation", action="store_true")
    parser.add_argument("--use-well-trained-wrapper", action="store_true")
    parser.add_argument("--well-trained-target-forward-velocity", type=float, default=2.0)
    parser.add_argument("--well-trained-target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--well-trained-target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--well-trained-target-height", type=float, default=0.53)
    parser.add_argument("--well-trained-velocity-obs-scale", type=float, default=3.0)
    parser.add_argument("--well-trained-include-command-observation", action="store_true")
    parser.add_argument("--history-stack-size", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make("Ant-v5", render_mode="rgb_array")
    if args.use_well_trained_wrapper:
        env = WellTrainedLocomotionAntWrapper(
            env,
            reward_config=WellTrainedLocomotionRewardConfig(
                target_forward_velocity=args.well_trained_target_forward_velocity,
                target_lateral_velocity=args.well_trained_target_lateral_velocity,
                target_yaw_rate=args.well_trained_target_yaw_rate,
                target_height=args.well_trained_target_height,
                velocity_obs_scale=args.well_trained_velocity_obs_scale,
                include_command_observation=args.well_trained_include_command_observation,
            ),
        )
    elif args.use_controlled_locomotion_wrapper:
        env = ControlledLocomotionAntWrapper(
            env,
            reward_config=ControlledLocomotionRewardConfig(
                target_forward_velocity=args.controlled_target_forward_velocity,
                target_lateral_velocity=args.controlled_target_lateral_velocity,
                target_yaw_rate=args.controlled_target_yaw_rate,
                target_yaw=args.controlled_target_yaw,
                target_height=args.controlled_target_height,
                target_velocity_obs_scale=args.controlled_target_velocity_obs_scale,
                target_yaw_rate_obs_scale=args.controlled_target_yaw_rate_obs_scale,
                lateral_position_clip=args.lateral_error_observation_clip,
                include_command_observation=args.controlled_include_command_observation,
            ),
        )
    elif args.include_lateral_error_observation:
        env = LateralErrorObservationWrapper(
            env,
            clip=args.lateral_error_observation_clip,
        )
    if args.history_stack_size > 1:
        env = ObservationHistoryStackWrapper(env, stack_size=args.history_stack_size)
    obs, info = env.reset(seed=args.seed)

    model = PPO.load(args.model, device="cpu")

    total_reward = 0.0
    steps = 0

    print(f"[record] model: {args.model}")
    print(f"[record] output: {out_path}")

    with imageio.get_writer(str(out_path), fps=args.fps) as writer:
        frame = env.render()
        writer.append_data(frame)

        for step in range(args.max_steps):
            action, _state = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            writer.append_data(frame)

            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                print(
                    f"[episode end] step={step}, "
                    f"terminated={terminated}, truncated={truncated}"
                )
                break

    env.close()

    print("[done] video saved")
    print("[summary] steps:", steps)
    print("[summary] total_reward:", total_reward)


if __name__ == "__main__":
    main()
