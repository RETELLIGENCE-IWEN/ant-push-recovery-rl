from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO


def get_x_position(env: gym.Env) -> float:
    return float(env.unwrapped.data.qpos[0])


def evaluate_episode(
    model: PPO,
    seed: int,
    max_steps: int,
    deterministic: bool,
) -> dict:
    env = gym.make("Ant-v5")
    obs, info = env.reset(seed=seed)

    initial_x = get_x_position(env)
    total_reward = 0.0
    action_energy_sum = 0.0
    action_norm_sum = 0.0
    steps = 0
    terminated = False
    truncated = False

    dt = float(env.unwrapped.dt)

    for step in range(max_steps):
        action, _state = model.predict(obs, deterministic=deterministic)
        action_np = np.asarray(action, dtype=np.float64)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        action_energy_sum += float(np.sum(np.square(action_np)))
        action_norm_sum += float(np.linalg.norm(action_np))
        steps += 1

        if terminated or truncated:
            break

    final_x = get_x_position(env)
    distance_x = final_x - initial_x
    duration_s = steps * dt

    env.close()

    return {
        "seed": seed,
        "return": total_reward,
        "steps": steps,
        "duration_s": duration_s,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "survived_to_max_steps": bool(steps >= max_steps and not terminated),
        "initial_x": initial_x,
        "final_x": final_x,
        "distance_x": distance_x,
        "mean_forward_velocity": distance_x / max(duration_s, 1e-9),
        "mean_action_norm": action_norm_sum / max(steps, 1),
        "mean_action_energy": action_energy_sum / max(steps, 1),
    }


def summarize(values: list[float]) -> dict:
    if len(values) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    if len(values) == 1:
        return {
            "mean": values[0],
            "std": 0.0,
            "min": values[0],
            "max": values[0],
        }
    return {
        "mean": mean(values),
        "std": stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runs/baseline_ppo_ant_50k/models/final_model.zip",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default="reports/baseline_clean_50k")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device="cpu")

    rows = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        row = evaluate_episode(
            model=model,
            seed=ep_seed,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )
        rows.append(row)
        print(
            f"[eval] ep={i:03d} seed={ep_seed} "
            f"return={row['return']:.3f} "
            f"distance_x={row['distance_x']:.3f} "
            f"survived={row['survived_to_max_steps']}"
        )

    csv_path = out_dir / "episodes.csv"
    json_path = out_dir / "summary.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "deterministic": args.deterministic,
        "return": summarize([r["return"] for r in rows]),
        "distance_x": summarize([r["distance_x"] for r in rows]),
        "mean_forward_velocity": summarize([r["mean_forward_velocity"] for r in rows]),
        "mean_action_norm": summarize([r["mean_action_norm"] for r in rows]),
        "mean_action_energy": summarize([r["mean_action_energy"] for r in rows]),
        "survival_rate": sum(r["survived_to_max_steps"] for r in rows) / len(rows),
        "termination_rate": sum(r["terminated"] for r in rows) / len(rows),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] csv:", csv_path)
    print("[done] summary:", json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()