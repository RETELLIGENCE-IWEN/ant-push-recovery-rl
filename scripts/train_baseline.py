from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def make_env(seed: int):
    def _init():
        env = gym.make("Ant-v5")
        env.reset(seed=seed)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="baseline_ppo_ant_50k")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    model_dir = run_dir / "models"
    log_dir = run_dir / "tb"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(args.seed)])
    env = VecMonitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(log_dir),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
    )

    model.learn(
        total_timesteps=args.total_steps,
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    save_path = model_dir / "final_model"
    model.save(str(save_path))

    env.close()

    print(f"[done] saved model to: {save_path}.zip")


if __name__ == "__main__":
    main()