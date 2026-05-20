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
    ControlledLocomotionAntWrapper,
    ControlledLocomotionRewardConfig,
)


def make_env(seed: int, rank: int, reward_config: ControlledLocomotionRewardConfig):
    def _init():
        env = gym.make("Ant-v5")
        env = ControlledLocomotionAntWrapper(env, reward_config=reward_config)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(
    seed: int,
    n_envs: int,
    vec_env_type: str,
    reward_config: ControlledLocomotionRewardConfig,
):
    env_fns = [
        make_env(seed=seed, rank=i, reward_config=reward_config)
        for i in range(n_envs)
    ]

    if vec_env_type == "dummy":
        env = DummyVecEnv(env_fns)
    elif vec_env_type == "subproc":
        env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        raise ValueError(f"Unknown vec_env_type: {vec_env_type}")

    return VecMonitor(env)


def copy_policy_weights_with_expanded_input(
    source_model: PPO,
    target_model: PPO,
) -> tuple[int, int, list[str]]:
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()

    copied = 0
    skipped: list[str] = []

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            skipped.append(key)
            continue

        if source_tensor.shape == target_tensor.shape:
            target_state[key] = source_tensor.detach().clone()
            copied += 1
            continue

        can_expand_input = (
            source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] < target_tensor.shape[1]
        )
        if can_expand_input:
            expanded = target_tensor.detach().clone()
            expanded[:, : source_tensor.shape[1]] = source_tensor.detach()
            expanded[:, source_tensor.shape[1] :] = 0.0
            target_state[key] = expanded
            copied += 1
            continue

        skipped.append(key)

    target_model.policy.load_state_dict(target_state)
    return copied, len(skipped), skipped


def build_model(
    args: argparse.Namespace,
    env,
    log_dir: Path,
    reset_num_timesteps: bool,
) -> PPO:
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        )
    )

    if args.resume_from is not None:
        if args.warm_start_on_observation_mismatch:
            print("[stage] warm-starting from:", args.resume_from)
            print("[note] Optimizer state is reset; compatible policy tensors are copied.")
            source_model = PPO.load(args.resume_from, device="cpu")
            policy_kwargs = dict(source_model.policy_kwargs)
            model = PPO(
                policy="MlpPolicy",
                env=env,
                device="cpu",
                verbose=1,
                seed=args.seed,
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
                max_grad_norm=args.max_grad_norm,
                policy_kwargs=policy_kwargs,
            )
            copied, skipped_count, skipped = copy_policy_weights_with_expanded_input(
                source_model=source_model,
                target_model=model,
            )
            model.num_timesteps = source_model.num_timesteps
            print("[warm-start] copied tensors:", copied)
            print("[warm-start] skipped tensors:", skipped_count)
            if skipped:
                print("[warm-start] skipped names:", skipped[:10])
            return model

        print("[stage] loading same-observation model:", args.resume_from)
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

    print("[stage] creating new controlled-locomotion PPO model")
    return PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        verbose=1,
        seed=args.seed,
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
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="controlled_locomotion_v3")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--warm-start-on-observation-mismatch", action="store_true")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.08)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Controlled locomotion objective
    parser.add_argument("--target-forward-velocity", type=float, default=2.0)
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--target-yaw", type=float, default=0.0)
    parser.add_argument("--target-height", type=float, default=0.53)
    parser.add_argument("--target-velocity-obs-scale", type=float, default=3.0)
    parser.add_argument("--target-yaw-rate-obs-scale", type=float, default=2.0)
    parser.add_argument("--randomize-commands", action="store_true")
    parser.add_argument("--command-forward-velocity-min", type=float, default=1.6)
    parser.add_argument("--command-forward-velocity-max", type=float, default=2.2)
    parser.add_argument("--command-lateral-velocity-min", type=float, default=0.0)
    parser.add_argument("--command-lateral-velocity-max", type=float, default=0.0)
    parser.add_argument("--command-yaw-rate-min", type=float, default=0.0)
    parser.add_argument("--command-yaw-rate-max", type=float, default=0.0)
    parser.add_argument("--lateral-position-clip", type=float, default=5.0)
    parser.add_argument("--lateral-position-reward-clip", type=float, default=5.0)
    parser.add_argument("--lateral-position-soft-limit", type=float, default=0.0)
    parser.add_argument("--include-command-observation", action="store_true")
    parser.add_argument("--w-alive", type=float, default=0.8)
    parser.add_argument("--w-velocity-tracking", type=float, default=1.6)
    parser.add_argument("--velocity-tracking-sigma", type=float, default=0.25)
    parser.add_argument("--w-lateral-velocity-tracking", type=float, default=0.0)
    parser.add_argument("--lateral-velocity-tracking-sigma", type=float, default=0.25)
    parser.add_argument("--w-yaw-rate-tracking", type=float, default=0.0)
    parser.add_argument("--yaw-rate-tracking-sigma", type=float, default=0.25)
    parser.add_argument("--w-heading-tracking", type=float, default=1.2)
    parser.add_argument("--heading-tracking-sigma", type=float, default=0.50)
    parser.add_argument("--w-course-tracking", type=float, default=0.0)
    parser.add_argument("--course-tracking-sigma", type=float, default=0.25)
    parser.add_argument("--course-tracking-min-speed", type=float, default=0.25)
    parser.add_argument("--w-lateral-position", type=float, default=0.035)
    parser.add_argument("--w-lateral-velocity", type=float, default=0.10)
    parser.add_argument("--w-lateral-away-velocity", type=float, default=0.0)
    parser.add_argument("--w-yaw-rate", type=float, default=0.03)
    parser.add_argument("--w-roll-pitch", type=float, default=0.12)
    parser.add_argument("--w-roll-pitch-rate", type=float, default=0.03)
    parser.add_argument("--w-height", type=float, default=0.60)
    parser.add_argument("--w-vertical-velocity", type=float, default=0.04)
    parser.add_argument("--w-action-energy", type=float, default=0.02)
    parser.add_argument("--w-action-rate", type=float, default=0.05)
    parser.add_argument("--w-action-accel", type=float, default=0.015)

    parser.add_argument("--save-every-env-steps", type=int, default=100_000)
    args = parser.parse_args()

    torch.set_num_threads(1)

    reward_config = ControlledLocomotionRewardConfig(
        target_forward_velocity=args.target_forward_velocity,
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        target_yaw=args.target_yaw,
        target_height=args.target_height,
        target_velocity_obs_scale=args.target_velocity_obs_scale,
        target_yaw_rate_obs_scale=args.target_yaw_rate_obs_scale,
        randomize_commands=args.randomize_commands,
        command_forward_velocity_min=args.command_forward_velocity_min,
        command_forward_velocity_max=args.command_forward_velocity_max,
        command_lateral_velocity_min=args.command_lateral_velocity_min,
        command_lateral_velocity_max=args.command_lateral_velocity_max,
        command_yaw_rate_min=args.command_yaw_rate_min,
        command_yaw_rate_max=args.command_yaw_rate_max,
        lateral_position_clip=args.lateral_position_clip,
        lateral_position_reward_clip=args.lateral_position_reward_clip,
        lateral_position_soft_limit=args.lateral_position_soft_limit,
        include_command_observation=args.include_command_observation,
        w_alive=args.w_alive,
        w_velocity_tracking=args.w_velocity_tracking,
        velocity_tracking_sigma=args.velocity_tracking_sigma,
        w_lateral_velocity_tracking=args.w_lateral_velocity_tracking,
        lateral_velocity_tracking_sigma=args.lateral_velocity_tracking_sigma,
        w_yaw_rate_tracking=args.w_yaw_rate_tracking,
        yaw_rate_tracking_sigma=args.yaw_rate_tracking_sigma,
        w_heading_tracking=args.w_heading_tracking,
        heading_tracking_sigma=args.heading_tracking_sigma,
        w_course_tracking=args.w_course_tracking,
        course_tracking_sigma=args.course_tracking_sigma,
        course_tracking_min_speed=args.course_tracking_min_speed,
        w_lateral_position=args.w_lateral_position,
        w_lateral_velocity=args.w_lateral_velocity,
        w_lateral_away_velocity=args.w_lateral_away_velocity,
        w_yaw_rate=args.w_yaw_rate,
        w_roll_pitch=args.w_roll_pitch,
        w_roll_pitch_rate=args.w_roll_pitch_rate,
        w_height=args.w_height,
        w_vertical_velocity=args.w_vertical_velocity,
        w_action_energy=args.w_action_energy,
        w_action_rate=args.w_action_rate,
        w_action_accel=args.w_action_accel,
    )

    run_dir = Path("runs") / args.run_name
    model_dir = run_dir / "models"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "tb"

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "reward_config": asdict(reward_config),
            },
            f,
            indent=2,
        )

    print("[config] run_name:", args.run_name)
    print("[config] total_steps:", args.total_steps)
    print("[config] seed:", args.seed)
    print("[config] n_envs:", args.n_envs)
    print("[config] vec_env:", args.vec_env)
    print("[config] reward_config:", reward_config)
    print("[config] saved:", config_path)
    print("[torch] version:", torch.__version__)
    print("[torch] threads:", torch.get_num_threads())

    env = build_vec_env(
        seed=args.seed,
        n_envs=args.n_envs,
        vec_env_type=args.vec_env,
        reward_config=reward_config,
    )

    reset_num_timesteps = args.resume_from is None
    model = build_model(
        args=args,
        env=env,
        log_dir=log_dir,
        reset_num_timesteps=reset_num_timesteps,
    )
    if args.resume_from is not None:
        reset_num_timesteps = False

    save_freq_calls = max(args.save_every_env_steps // max(args.n_envs, 1), 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_controlled_locomotion_ant",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("[stage] starting learning")
    model.learn(
        total_timesteps=args.total_steps,
        reset_num_timesteps=reset_num_timesteps,
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
