"""
Three-tier evaluation for robust locomotion policy.

Tier A: Quiet eval (no push, no DR) — verify nominal locomotion preserved.
Tier B: Single-push grid (force x direction) — controlled disturbance response.
Tier C: Random-push stress (training-like) — random pushes + DR.

All tiers use the WellTrainedLocomotionAntWrapper observation/reward (same as
the policy was trained with). Push disturbance / DR are added externally for
Tier B / C without re-using the curriculum wrapper, since those vary by env
step rather than the controlled scenarios this eval needs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from stable_directional_ant import (
    DomainRandomizationConfig,
    DomainRandomizationWrapper,
    PushDisturbanceConfig,
    PushDisturbanceWrapper,
    WellTrainedLocomotionAntWrapper,
    WellTrainedLocomotionRewardConfig,
    quat_wxyz_to_rpy,
    wrap_angle_rad,
)


def get_torso_id(env: gym.Env, name: str = "torso") -> int:
    model = env.unwrapped.model
    for i in range(model.nbody):
        if model.body(i).name == name:
            return i
    raise ValueError(f"Body {name} not found")


def get_root_state(env: gym.Env) -> dict:
    qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
    roll, pitch, yaw = quat_wxyz_to_rpy(qpos[3:7])
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    vx_b = cos_y * qvel[0] + sin_y * qvel[1]
    vy_b = -sin_y * qvel[0] + cos_y * qvel[1]
    return {
        "x": float(qpos[0]),
        "y": float(qpos[1]),
        "z": float(qpos[2]),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "vx": float(qvel[0]),
        "vy": float(qvel[1]),
        "vz": float(qvel[2]),
        "vx_body": float(vx_b),
        "vy_body": float(vy_b),
        "yaw_rate": float(qvel[5]),
    }


def build_eval_env(use_dr: bool, dr_action_noise: float) -> gym.Env:
    env = gym.make("Ant-v5")
    if use_dr:
        env = DomainRandomizationWrapper(
            env,
            config=DomainRandomizationConfig(
                enabled=True,
                action_noise_std=dr_action_noise,
            ),
        )
    env = WellTrainedLocomotionAntWrapper(
        env,
        reward_config=WellTrainedLocomotionRewardConfig(),
    )
    return env


def summarize(vals):
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "min": vals[0], "max": vals[0]}
    return {
        "mean": mean(vals),
        "std": stdev(vals),
        "min": min(vals),
        "max": max(vals),
    }


def run_episode(
    model: PPO,
    seed: int,
    max_steps: int,
    use_dr: bool,
    dr_action_noise: float,
    push_schedule: list[dict] | None,
) -> dict:
    """
    push_schedule is a list of {"step_start": int, "duration": int, "force_xy": np.array(2)}
    Forces are applied to torso between [step_start, step_start+duration).
    """
    env = build_eval_env(use_dr=use_dr, dr_action_noise=dr_action_noise)
    torso_id = get_torso_id(env)

    obs, info = env.reset(seed=seed)
    initial = get_root_state(env)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    yaw_abs_sum = 0.0
    vx_body_sum = 0.0
    vy_body_abs_sum = 0.0
    roll_sq_sum = 0.0
    pitch_sq_sum = 0.0
    height_sum = 0.0
    heights = []

    # For Tier B: track post-push recovery
    post_push_states = []  # list of (steps_after_push, vx_body, vy_body, drift_from_pre_push)
    pre_push_y = None
    push_applied_at = None
    if push_schedule:
        push_applied_at = min(p["step_start"] for p in push_schedule)

    dt = float(env.unwrapped.dt)

    for t in range(max_steps):
        # Apply scheduled push (overrides anything else; reset each step)
        env.unwrapped.data.xfrc_applied[torso_id, :] = 0.0
        if push_schedule:
            for p in push_schedule:
                if p["step_start"] <= t < p["step_start"] + p["duration"]:
                    env.unwrapped.data.xfrc_applied[torso_id, 0] = p["force_xy"][0]
                    env.unwrapped.data.xfrc_applied[torso_id, 1] = p["force_xy"][1]

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        s = get_root_state(env)

        total_reward += float(reward)
        steps += 1

        yaw_err = wrap_angle_rad(s["yaw"])
        yaw_abs_sum += abs(yaw_err)
        vx_body_sum += s["vx_body"]
        vy_body_abs_sum += abs(s["vy_body"])
        roll_sq_sum += s["roll"] ** 2
        pitch_sq_sum += s["pitch"] ** 2
        height_sum += s["z"]
        heights.append(s["z"])

        if push_applied_at is not None:
            if t == push_applied_at:
                pre_push_y = s["y"]
            if t >= push_applied_at and pre_push_y is not None:
                post_push_states.append(
                    (t - push_applied_at, s["vx_body"], s["vy_body"], s["y"] - pre_push_y)
                )

        if terminated or truncated:
            break

    final = get_root_state(env)
    env.close()

    drift = final["y"] - initial["y"]
    duration_s = steps * dt

    # Tier B specific: post-push recovery metrics
    recovery = {}
    if post_push_states:
        # Time to recover vx_body to within 0.2 of target (=2.0)
        target_vx = 2.0
        recovery_step = None
        for step_off, vxb, _, _ in post_push_states:
            if step_off > 10 and abs(vxb - target_vx) < 0.2:
                recovery_step = step_off
                break
        recovery["recovery_steps_vx"] = recovery_step if recovery_step is not None else -1
        recovery["max_post_push_drift"] = max(
            (abs(d) for _, _, _, d in post_push_states), default=0.0
        )

    return {
        "seed": seed,
        "return": total_reward,
        "steps": steps,
        "survived_to_max_steps": bool(steps >= max_steps and not terminated),
        "terminated": bool(terminated),
        "distance_x": final["x"] - initial["x"],
        "mean_forward_velocity": (final["x"] - initial["x"]) / max(duration_s, 1e-9),
        "lateral_drift": drift,
        "abs_lateral_drift": abs(drift),
        "vx_body_mean": vx_body_sum / max(steps, 1),
        "vy_body_mean_abs": vy_body_abs_sum / max(steps, 1),
        "yaw_abs_mean": yaw_abs_sum / max(steps, 1),
        "roll_rms": math.sqrt(roll_sq_sum / max(steps, 1)),
        "pitch_rms": math.sqrt(pitch_sq_sum / max(steps, 1)),
        "height_mean": height_sum / max(steps, 1),
        "height_std": float(np.std(heights)) if len(heights) >= 2 else 0.0,
        **recovery,
    }


def tier_a_quiet(model: PPO, out_dir: Path, episodes: int = 10) -> dict:
    print("=== Tier A: Quiet eval (no push, no DR) ===")
    rows = []
    for i in range(episodes):
        row = run_episode(
            model=model,
            seed=1000 + i,
            max_steps=1000,
            use_dr=False,
            dr_action_noise=0.0,
            push_schedule=None,
        )
        rows.append(row)
        print(
            f"  ep={i:02d} return={row['return']:.0f} "
            f"vx={row['vx_body_mean']:.2f} "
            f"drift={row['abs_lateral_drift']:.2f} "
            f"yaw={row['yaw_abs_mean']:.3f} "
            f"survived={row['survived_to_max_steps']}"
        )
    csv_path = out_dir / "tier_a_quiet.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    summary = {
        "tier": "A_quiet",
        "episodes": episodes,
        "survival_rate": sum(r["survived_to_max_steps"] for r in rows) / len(rows),
        **{k: summarize([r[k] for r in rows]) for k in numeric},
    }
    return summary


def tier_b_push_grid(
    model: PPO,
    out_dir: Path,
    magnitudes: list[float],
    directions_deg: list[float],
    push_at_step: int = 250,
    push_duration: int = 5,
    episodes_per_cell: int = 5,
) -> dict:
    print(f"=== Tier B: Single push grid (push at t={push_at_step*0.01:.1f}s, dur={push_duration*0.01:.2f}s) ===")
    rows = []
    for mag in magnitudes:
        for d_deg in directions_deg:
            theta = math.radians(d_deg)
            force_xy = np.array([mag * math.cos(theta), mag * math.sin(theta)])
            cell_results = []
            for ep in range(episodes_per_cell):
                row = run_episode(
                    model=model,
                    seed=2000 + ep,
                    max_steps=1000,
                    use_dr=False,
                    dr_action_noise=0.0,
                    push_schedule=[{
                        "step_start": push_at_step,
                        "duration": push_duration,
                        "force_xy": force_xy,
                    }],
                )
                row["push_magnitude"] = mag
                row["push_direction_deg"] = d_deg
                rows.append(row)
                cell_results.append(row)
            surv = sum(r["survived_to_max_steps"] for r in cell_results) / len(cell_results)
            rec = mean(
                [r.get("recovery_steps_vx", -1) for r in cell_results if r.get("recovery_steps_vx", -1) > 0]
                or [-1]
            )
            print(f"  mag={mag:.1f}N dir={d_deg:5.1f}° surv={surv:.2f} rec_steps={rec:.0f}")
    csv_path = out_dir / "tier_b_push_grid.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # Summarize by magnitude
    by_mag = {}
    for mag in magnitudes:
        cell_rows = [r for r in rows if r["push_magnitude"] == mag]
        by_mag[f"mag_{mag:.1f}N"] = {
            "survival_rate": sum(r["survived_to_max_steps"] for r in cell_rows) / len(cell_rows),
            "mean_recovery_steps": mean([r["recovery_steps_vx"] for r in cell_rows if r["recovery_steps_vx"] > 0]) if any(r["recovery_steps_vx"] > 0 for r in cell_rows) else None,
            "mean_max_drift_post_push": mean([r["max_post_push_drift"] for r in cell_rows]),
            "mean_final_drift": mean([r["abs_lateral_drift"] for r in cell_rows]),
        }
    return {"tier": "B_push_grid", "magnitudes": magnitudes, "directions_deg": directions_deg, "by_magnitude": by_mag}


def tier_c_random_push(model: PPO, out_dir: Path, episodes: int = 10) -> dict:
    print("=== Tier C: Random push + DR (training-like) ===")
    rows = []
    for i in range(episodes):
        # Use the curriculum wrapper at max strength + DR
        env = gym.make("Ant-v5")
        env = DomainRandomizationWrapper(
            env,
            config=DomainRandomizationConfig(enabled=True, action_noise_std=0.02),
        )
        env = PushDisturbanceWrapper(
            env,
            config=PushDisturbanceConfig(
                enabled=True,
                push_force_max=10.0,
                push_duration_steps=5,
                push_interval_steps_min=500,
                push_interval_steps_max=1000,
                curriculum_ramp_steps=1,  # already at max immediately
                initial_quiet_steps=0,
            ),
        )
        env = WellTrainedLocomotionAntWrapper(
            env,
            reward_config=WellTrainedLocomotionRewardConfig(),
        )
        obs, info = env.reset(seed=3000 + i)
        initial = get_root_state(env)
        total_r = 0.0
        steps = 0
        terminated = False
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += float(r)
            steps += 1
            if terminated or truncated:
                break
        final = get_root_state(env)
        env.close()
        row = {
            "seed": 3000 + i,
            "return": total_r,
            "steps": steps,
            "survived_to_max_steps": bool(steps >= 1000 and not terminated),
            "terminated": bool(terminated),
            "abs_lateral_drift": abs(final["y"] - initial["y"]),
            "mean_forward_velocity": (final["x"] - initial["x"]) / max(steps * float(env.unwrapped.dt), 1e-9) if steps > 0 else 0.0,
        }
        rows.append(row)
        print(f"  ep={i:02d} return={row['return']:.0f} steps={steps} survived={row['survived_to_max_steps']} drift={row['abs_lateral_drift']:.2f}")
    csv_path = out_dir / "tier_c_random_push.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {
        "tier": "C_random_push",
        "episodes": episodes,
        "survival_rate": sum(r["survived_to_max_steps"] for r in rows) / len(rows),
        "return": summarize([r["return"] for r in rows]),
        "abs_lateral_drift": summarize([r["abs_lateral_drift"] for r in rows]),
        "mean_forward_velocity": summarize([r["mean_forward_velocity"] for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--tiers", type=str, default="abc", help="Subset of 'abc' to run")
    parser.add_argument("--tier-a-episodes", type=int, default=10)
    parser.add_argument("--tier-b-episodes", type=int, default=3)
    parser.add_argument("--tier-c-episodes", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device="cpu")

    summaries = {}
    if "a" in args.tiers:
        summaries["A"] = tier_a_quiet(model, out_dir, args.tier_a_episodes)
    if "b" in args.tiers:
        summaries["B"] = tier_b_push_grid(
            model,
            out_dir,
            magnitudes=[2.0, 4.0, 6.0, 8.0, 10.0],
            directions_deg=[0, 45, 90, 135, 180, 225, 270, 315],
            push_at_step=250,
            push_duration=5,
            episodes_per_cell=args.tier_b_episodes,
        )
    if "c" in args.tiers:
        summaries["C"] = tier_c_random_push(model, out_dir, args.tier_c_episodes)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2)
    print()
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
