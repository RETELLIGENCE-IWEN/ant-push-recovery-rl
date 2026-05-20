from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from stable_directional_ant import (
    DomainRandomizationConfig,
    DomainRandomizationWrapper,
    PushDisturbanceConfig,
    PushDisturbanceWrapper,
    WellTrainedLocomotionAntWrapper,
    WellTrainedLocomotionRewardConfig,
)


def make_env(
    seed: int,
    rank: int,
    reward_config: WellTrainedLocomotionRewardConfig,
    push_config: PushDisturbanceConfig,
    dr_config: DomainRandomizationConfig,
):
    def _init():
        env = gym.make("Ant-v5")
        env = DomainRandomizationWrapper(env, config=dr_config)
        env = PushDisturbanceWrapper(env, config=push_config)
        env = WellTrainedLocomotionAntWrapper(env, reward_config=reward_config)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(args, reward_config, push_config, dr_config):
    env_fns = [
        make_env(
            seed=args.seed,
            rank=i,
            reward_config=reward_config,
            push_config=push_config,
            dr_config=dr_config,
        )
        for i in range(args.n_envs)
    ]
    if args.vec_env == "dummy":
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="fork")
    return VecMonitor(env)


def build_model(args: argparse.Namespace, env, log_dir: Path) -> PPO:
    if args.resume_from is None:
        raise ValueError(
            "train_robust_locomotion requires --resume-from a well-trained "
            "nominal policy (e.g. v3h_s2_seed42)."
        )

    print("[stage] warm-starting from:", args.resume_from)
    return PPO.load(
        args.resume_from,
        env=env,
        device="cpu",
        tensorboard_log=str(log_dir),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="robust_locomotion_v4a")
    parser.add_argument("--resume-from", type=str, required=True)
    parser.add_argument("--total-steps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument(
        "--vec-env", type=str, default="subproc", choices=["dummy", "subproc"]
    )

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=None)

    # Reward config (matches v3h Stage 2 final settings by default)
    parser.add_argument("--target-forward-velocity", type=float, default=2.0)
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--target-height", type=float, default=0.53)
    parser.add_argument("--w-track-vx", type=float, default=1.5)
    parser.add_argument("--w-track-vy", type=float, default=2.5)
    parser.add_argument("--w-track-omega-z", type=float, default=0.5)
    parser.add_argument("--w-progress-vx", type=float, default=1.0)
    parser.add_argument("--w-heading-alignment", type=float, default=1.0)
    parser.add_argument("--w-lateral-position", type=float, default=0.1)
    parser.add_argument("--w-action-rate", type=float, default=0.05)
    parser.add_argument("--w-action-accel", type=float, default=0.02)

    # Push disturbance
    parser.add_argument("--push-enabled", type=int, default=1)
    parser.add_argument("--push-force-max", type=float, default=10.0)
    parser.add_argument("--push-duration-steps", type=int, default=5)
    parser.add_argument("--push-interval-steps-min", type=int, default=500)
    parser.add_argument("--push-interval-steps-max", type=int, default=1000)
    parser.add_argument("--push-curriculum-ramp-steps", type=int, default=300_000)
    parser.add_argument("--push-initial-quiet-steps", type=int, default=10_000)
    parser.add_argument("--push-torque-z-max", type=float, default=0.0)
    parser.add_argument("--push-duration-max-steps", type=int, default=0)

    # Domain randomization
    parser.add_argument("--dr-enabled", type=int, default=1)
    parser.add_argument("--dr-mass-scale-min", type=float, default=0.8)
    parser.add_argument("--dr-mass-scale-max", type=float, default=1.2)
    parser.add_argument("--dr-friction-scale-min", type=float, default=0.5)
    parser.add_argument("--dr-friction-scale-max", type=float, default=1.5)
    parser.add_argument("--dr-damping-scale-min", type=float, default=0.8)
    parser.add_argument("--dr-damping-scale-max", type=float, default=1.2)
    parser.add_argument("--dr-motor-scale-min", type=float, default=0.85)
    parser.add_argument("--dr-motor-scale-max", type=float, default=1.15)
    parser.add_argument("--dr-action-noise-std", type=float, default=0.02)

    parser.add_argument("--save-every-env-steps", type=int, default=100_000)
    args = parser.parse_args()

    torch.set_num_threads(1)
    if args.target_kl is not None and args.target_kl <= 0.0:
        args.target_kl = None

    reward_config = WellTrainedLocomotionRewardConfig(
        target_forward_velocity=args.target_forward_velocity,
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        target_height=args.target_height,
        w_track_vx=args.w_track_vx,
        w_track_vy=args.w_track_vy,
        w_track_omega_z=args.w_track_omega_z,
        w_progress_vx=args.w_progress_vx,
        w_heading_alignment=args.w_heading_alignment,
        w_lateral_position=args.w_lateral_position,
        w_action_rate=args.w_action_rate,
        w_action_accel=args.w_action_accel,
    )
    push_config = PushDisturbanceConfig(
        enabled=bool(args.push_enabled),
        push_force_max=args.push_force_max,
        push_duration_steps=args.push_duration_steps,
        push_interval_steps_min=args.push_interval_steps_min,
        push_interval_steps_max=args.push_interval_steps_max,
        curriculum_ramp_steps=args.push_curriculum_ramp_steps,
        initial_quiet_steps=args.push_initial_quiet_steps,
        push_torque_z_max=args.push_torque_z_max,
        push_duration_max_steps=args.push_duration_max_steps,
    )
    dr_config = DomainRandomizationConfig(
        enabled=bool(args.dr_enabled),
        mass_scale_min=args.dr_mass_scale_min,
        mass_scale_max=args.dr_mass_scale_max,
        friction_scale_min=args.dr_friction_scale_min,
        friction_scale_max=args.dr_friction_scale_max,
        damping_scale_min=args.dr_damping_scale_min,
        damping_scale_max=args.dr_damping_scale_max,
        motor_scale_min=args.dr_motor_scale_min,
        motor_scale_max=args.dr_motor_scale_max,
        action_noise_std=args.dr_action_noise_std,
    )

    run_dir = Path("runs") / args.run_name
    model_dir = run_dir / "models"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "tb"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "reward_config": asdict(reward_config),
                "push_config": asdict(push_config),
                "dr_config": asdict(dr_config),
            },
            f,
            indent=2,
        )

    print("[config] run_name:", args.run_name)
    print("[config] resume_from:", args.resume_from)
    print("[config] total_steps:", args.total_steps)
    print("[config] push_config:", push_config)
    print("[config] dr_config:", dr_config)

    env = build_vec_env(args, reward_config, push_config, dr_config)
    model = build_model(args=args, env=env, log_dir=log_dir)

    save_freq_calls = max(args.save_every_env_steps // max(args.n_envs, 1), 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_robust_locomotion_ant",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("[stage] starting learning")
    model.learn(
        total_timesteps=args.total_steps,
        reset_num_timesteps=False,
        tb_log_name=args.run_name,
        callback=checkpoint_callback,
        progress_bar=False,
    )

    save_path = model_dir / "final_model"
    model.save(str(save_path))
    env.close()

    print("[done] final num_timesteps:", model.num_timesteps)
    print(f"[done] saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()
