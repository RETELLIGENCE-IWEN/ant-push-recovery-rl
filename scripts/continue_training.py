from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def make_env(seed: int):
    def _init():
        env = gym.make("Ant-v5")
        env.reset(seed=seed)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="baseline_ppo_ant_continued")
    parser.add_argument("--additional-steps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-freq", type=int, default=50_000)
    args = parser.parse_args()

    torch.set_num_threads(1)

    run_dir = Path("runs") / args.run_name
    model_dir = run_dir / "models"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "tb"

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("[config] resume_from:", args.resume_from)
    print("[config] run_name:", args.run_name)
    print("[config] additional_steps:", args.additional_steps)
    print("[config] seed:", args.seed)

    env = DummyVecEnv([make_env(args.seed)])
    env = VecMonitor(env)

    print("[stage] loading model")

    model = PPO.load(
        args.resume_from,
        env=env,
        device="cpu",
        tensorboard_log=str(log_dir),
    )

    print("[model] existing num_timesteps:", model.num_timesteps)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_ant_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("[stage] continuing learning")

    model.learn(
        total_timesteps=args.additional_steps,
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