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
    LateralErrorObservationWrapper,
    StableDirectionalAntWrapper,
    StableDirectionalRewardConfig,
)


def make_env(
    seed: int,
    rank: int,
    reward_config: StableDirectionalRewardConfig,
    include_lateral_error_observation: bool,
    lateral_error_observation_clip: float,
):
    def _init():
        env = gym.make("Ant-v5")
        env = StableDirectionalAntWrapper(env, reward_config=reward_config)
        if include_lateral_error_observation:
            env = LateralErrorObservationWrapper(
                env,
                clip=lateral_error_observation_clip,
            )
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(
    seed: int,
    n_envs: int,
    vec_env_type: str,
    reward_config: StableDirectionalRewardConfig,
    include_lateral_error_observation: bool,
    lateral_error_observation_clip: float,
):
    env_fns = [
        make_env(
            seed=seed,
            rank=i,
            reward_config=reward_config,
            include_lateral_error_observation=include_lateral_error_observation,
            lateral_error_observation_clip=lateral_error_observation_clip,
        )
        for i in range(n_envs)
    ]

    if vec_env_type == "dummy":
        env = DummyVecEnv(env_fns)
    elif vec_env_type == "subproc":
        env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        raise ValueError(f"Unknown vec_env_type: {vec_env_type}")

    env = VecMonitor(env)
    return env


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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="stable_directional_ppo_ant_800k")
    parser.add_argument("--total-steps", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"])

    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--warm-start-on-observation-mismatch", action="store_true")
    parser.add_argument("--include-lateral-error-observation", action="store_true")
    parser.add_argument("--lateral-error-observation-clip", type=float, default=5.0)

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.04)

    # Reward weights
    parser.add_argument("--w-heading", type=float, default=0.35)
    parser.add_argument("--w-lateral-velocity", type=float, default=0.08)
    parser.add_argument("--w-lateral-position", type=float, default=0.0)
    parser.add_argument("--lateral-position-clip", type=float, default=5.0)
    parser.add_argument("--w-yaw-rate", type=float, default=0.03)
    parser.add_argument("--target-forward-velocity", type=float, default=None)
    parser.add_argument("--w-forward-velocity", type=float, default=0.0)
    parser.add_argument("--w-roll-pitch", type=float, default=0.12)
    parser.add_argument("--w-height", type=float, default=0.40)
    parser.add_argument("--w-vertical-velocity", type=float, default=0.03)
    parser.add_argument("--w-action-smooth", type=float, default=0.015)
    parser.add_argument("--target-height", type=float, default=None)

    parser.add_argument("--save-every-env-steps", type=int, default=100_000)

    args = parser.parse_args()

    torch.set_num_threads(1)

    reward_config = StableDirectionalRewardConfig(
        target_yaw=0.0,
        target_height=args.target_height,
        w_heading=args.w_heading,
        w_lateral_velocity=args.w_lateral_velocity,
        w_lateral_position=args.w_lateral_position,
        lateral_position_clip=args.lateral_position_clip,
        w_yaw_rate=args.w_yaw_rate,
        target_forward_velocity=args.target_forward_velocity,
        w_forward_velocity=args.w_forward_velocity,
        w_roll_pitch=args.w_roll_pitch,
        w_height=args.w_height,
        w_vertical_velocity=args.w_vertical_velocity,
        w_action_smooth=args.w_action_smooth,
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
        include_lateral_error_observation=args.include_lateral_error_observation,
        lateral_error_observation_clip=args.lateral_error_observation_clip,
    )

    if args.resume_from is not None:
        if args.warm_start_on_observation_mismatch:
            print("[stage] warm-starting from existing model:", args.resume_from)
            print("[note] Optimizer state is reset; compatible policy weights are copied.")
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
                max_grad_norm=0.5,
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
        else:
            print("[stage] loading existing model:", args.resume_from)
            print("[note] Network architecture is inherited from the loaded model.")
            model = PPO.load(
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
        reset_num_timesteps = False
    else:
        print("[stage] creating new PPO model with larger MLP policy")

        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        )

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
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
        )
        reset_num_timesteps = True

    save_freq_calls = max(args.save_every_env_steps // max(args.n_envs, 1), 1)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_stable_directional_ant",
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
