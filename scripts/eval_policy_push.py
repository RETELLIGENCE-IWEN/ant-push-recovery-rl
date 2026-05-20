from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from eval_policy_metrics import get_root_state, summarize, wrap_angle_rad
from stable_directional_ant import (
    ControlledLocomotionAntWrapper,
    ControlledLocomotionRewardConfig,
    LateralErrorObservationWrapper,
)


def get_body_id(env: gym.Env, body_name: str) -> int:
    model = env.unwrapped.model

    if hasattr(model, "body"):
        try:
            return int(model.body(body_name).id)
        except KeyError:
            pass

    try:
        import mujoco

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
    except Exception as exc:  # pragma: no cover - depends on MuJoCo binding details
        raise ValueError(f"Could not resolve MuJoCo body: {body_name}") from exc

    if body_id < 0:
        raise ValueError(f"Unknown MuJoCo body: {body_name}")
    return int(body_id)


def make_push_force(
    mode: str,
    magnitude: float,
    episode_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "lateral":
        sign = 1.0 if episode_index % 2 == 0 else -1.0
        return np.array([0.0, sign * magnitude, 0.0], dtype=np.float64)
    if mode == "backward":
        return np.array([-magnitude, 0.0, 0.0], dtype=np.float64)
    if mode == "forward":
        return np.array([magnitude, 0.0, 0.0], dtype=np.float64)
    if mode == "random_xy":
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        return np.array(
            [magnitude * math.cos(angle), magnitude * math.sin(angle), 0.0],
            dtype=np.float64,
        )
    if mode == "random_cardinal":
        forces = [
            np.array([magnitude, 0.0, 0.0], dtype=np.float64),
            np.array([-magnitude, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, magnitude, 0.0], dtype=np.float64),
            np.array([0.0, -magnitude, 0.0], dtype=np.float64),
        ]
        return forces[int(rng.integers(0, len(forces)))]

    raise ValueError(f"Unknown force mode: {mode}")


def set_external_force(env: gym.Env, body_id: int, force_xyz: np.ndarray) -> None:
    data = env.unwrapped.data
    data.xfrc_applied[:] = 0.0
    data.xfrc_applied[body_id, :3] = force_xyz


def evaluate_episode(
    model: PPO,
    seed: int,
    episode_index: int,
    max_steps: int,
    deterministic: bool,
    force_body: str,
    force_mode: str,
    force_magnitude: float,
    push_start: int,
    push_duration: int,
    push_start_jitter: int,
    pre_window: int,
    post_window: int,
    recovery_fraction: float,
    recovery_angle: float,
    include_lateral_error_observation: bool,
    lateral_error_observation_clip: float,
    use_controlled_locomotion_wrapper: bool,
    controlled_reward_config: ControlledLocomotionRewardConfig,
) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    env = gym.make("Ant-v5")
    if use_controlled_locomotion_wrapper:
        env = ControlledLocomotionAntWrapper(
            env,
            reward_config=controlled_reward_config,
        )
    elif include_lateral_error_observation:
        env = LateralErrorObservationWrapper(
            env,
            clip=lateral_error_observation_clip,
        )
    obs, _info = env.reset(seed=seed)

    body_id = get_body_id(env, force_body)

    if push_start_jitter > 0:
        jitter = int(rng.integers(-push_start_jitter, push_start_jitter + 1))
    else:
        jitter = 0
    episode_push_start = max(1, push_start + jitter)
    push_end = min(max_steps, episode_push_start + push_duration)
    force_xyz = make_push_force(
        mode=force_mode,
        magnitude=force_magnitude,
        episode_index=episode_index,
        rng=rng,
    )

    initial = get_root_state(env)
    state_at_push_start: dict[str, float] | None = None

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    pre_push_vx: list[float] = []
    post_push_vx: list[float] = []
    recovery_vx_window: list[float] = []

    max_abs_roll_after_push = 0.0
    max_abs_pitch_after_push = 0.0
    max_abs_yaw_after_push = 0.0
    min_height_after_push = float("inf")
    recovered_after_push = False
    recovery_steps = max(max_steps - push_end, 0)

    dt = float(env.unwrapped.dt)

    for step in range(max_steps):
        if step == episode_push_start:
            state_at_push_start = get_root_state(env)

        action, _state = model.predict(obs, deterministic=deterministic)

        if episode_push_start <= step < push_end:
            set_external_force(env, body_id=body_id, force_xyz=force_xyz)
        else:
            set_external_force(env, body_id=body_id, force_xyz=np.zeros(3))

        obs, reward, terminated, truncated, _info = env.step(action)
        state = get_root_state(env)

        total_reward += float(reward)
        steps += 1

        if episode_push_start - pre_window <= step < episode_push_start:
            pre_push_vx.append(state["vx"])

        if push_end <= step < push_end + post_window:
            post_push_vx.append(state["vx"])
            yaw_abs = abs(wrap_angle_rad(state["yaw"]))
            max_abs_roll_after_push = max(max_abs_roll_after_push, abs(state["roll"]))
            max_abs_pitch_after_push = max(max_abs_pitch_after_push, abs(state["pitch"]))
            max_abs_yaw_after_push = max(max_abs_yaw_after_push, yaw_abs)
            min_height_after_push = min(min_height_after_push, state["z"])

            recovery_vx_window.append(state["vx"])
            recovery_vx_window = recovery_vx_window[-10:]

            pre_mean = mean(pre_push_vx) if pre_push_vx else 0.0
            target_vx = recovery_fraction * max(pre_mean, 0.1)
            vx_recovered = mean(recovery_vx_window) >= target_vx
            attitude_recovered = (
                abs(state["roll"]) <= recovery_angle
                and abs(state["pitch"]) <= recovery_angle
            )
            healthy_height = 0.2 <= state["z"] <= 1.0
            if (
                not recovered_after_push
                and vx_recovered
                and attitude_recovered
                and healthy_height
            ):
                recovered_after_push = True
                recovery_steps = step - push_end + 1

        if terminated or truncated:
            break

    set_external_force(env, body_id=body_id, force_xyz=np.zeros(3))
    final = get_root_state(env)
    env.close()

    if state_at_push_start is None:
        state_at_push_start = final

    pre_mean = float(mean(pre_push_vx)) if pre_push_vx else 0.0
    post_mean = float(mean(post_push_vx)) if post_push_vx else 0.0
    velocity_retention = post_mean / max(abs(pre_mean), 1e-9)
    duration_s = steps * dt
    distance_x = final["x"] - initial["x"]

    if not math.isfinite(min_height_after_push):
        min_height_after_push = final["z"]

    return {
        "seed": seed,
        "return": total_reward,
        "steps": steps,
        "duration_s": duration_s,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "survived_to_max_steps": bool(steps >= max_steps and not terminated),
        "was_pushed": bool(steps > episode_push_start),
        "survived_push_window": bool(
            steps >= push_end and not (terminated and steps < push_end)
        ),
        "recovered_after_push": bool(recovered_after_push),
        "push_start": episode_push_start,
        "push_duration": push_duration,
        "push_force_x": float(force_xyz[0]),
        "push_force_y": float(force_xyz[1]),
        "push_force_z": float(force_xyz[2]),
        "push_force_norm": float(np.linalg.norm(force_xyz)),
        "distance_x": distance_x,
        "mean_forward_velocity": distance_x / max(duration_s, 1e-9),
        "lateral_drift": final["y"] - initial["y"],
        "abs_lateral_drift": abs(final["y"] - initial["y"]),
        "push_start_x": state_at_push_start["x"],
        "push_start_y": state_at_push_start["y"],
        "post_push_distance_x": final["x"] - state_at_push_start["x"],
        "post_push_lateral_drift": final["y"] - state_at_push_start["y"],
        "pre_push_vx_mean": pre_mean,
        "post_push_vx_mean": post_mean,
        "post_push_velocity_retention": velocity_retention,
        "recovery_steps": recovery_steps,
        "recovery_time_s": recovery_steps * dt,
        "max_abs_roll_after_push": max_abs_roll_after_push,
        "max_abs_pitch_after_push": max_abs_pitch_after_push,
        "max_abs_yaw_after_push": max_abs_yaw_after_push,
        "min_height_after_push": min_height_after_push,
        "final_yaw_abs": abs(wrap_angle_rad(final["yaw"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--force-body", type=str, default="torso")
    parser.add_argument(
        "--force-mode",
        type=str,
        default="lateral",
        choices=["lateral", "backward", "forward", "random_xy", "random_cardinal"],
    )
    parser.add_argument("--force-magnitude", type=float, default=80.0)
    parser.add_argument("--push-start", type=int, default=200)
    parser.add_argument("--push-duration", type=int, default=10)
    parser.add_argument("--push-start-jitter", type=int, default=0)
    parser.add_argument("--pre-window", type=int, default=50)
    parser.add_argument("--post-window", type=int, default=150)
    parser.add_argument("--recovery-fraction", type=float, default=0.8)
    parser.add_argument("--recovery-angle", type=float, default=0.6)
    parser.add_argument("--include-lateral-error-observation", action="store_true")
    parser.add_argument("--lateral-error-observation-clip", type=float, default=5.0)
    parser.add_argument("--use-controlled-locomotion-wrapper", action="store_true")
    parser.add_argument("--controlled-target-forward-velocity", type=float, default=2.0)
    parser.add_argument("--controlled-target-yaw", type=float, default=0.0)
    parser.add_argument("--controlled-target-height", type=float, default=0.53)
    parser.add_argument("--controlled-target-velocity-obs-scale", type=float, default=3.0)
    args = parser.parse_args()

    torch.set_num_threads(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    controlled_reward_config = ControlledLocomotionRewardConfig(
        target_forward_velocity=args.controlled_target_forward_velocity,
        target_yaw=args.controlled_target_yaw,
        target_height=args.controlled_target_height,
        target_velocity_obs_scale=args.controlled_target_velocity_obs_scale,
        lateral_position_clip=args.lateral_error_observation_clip,
    )

    rows = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        row = evaluate_episode(
            model=model,
            seed=ep_seed,
            episode_index=i,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            force_body=args.force_body,
            force_mode=args.force_mode,
            force_magnitude=args.force_magnitude,
            push_start=args.push_start,
            push_duration=args.push_duration,
            push_start_jitter=args.push_start_jitter,
            pre_window=args.pre_window,
            post_window=args.post_window,
            recovery_fraction=args.recovery_fraction,
            recovery_angle=args.recovery_angle,
            include_lateral_error_observation=args.include_lateral_error_observation,
            lateral_error_observation_clip=args.lateral_error_observation_clip,
            use_controlled_locomotion_wrapper=args.use_controlled_locomotion_wrapper,
            controlled_reward_config=controlled_reward_config,
        )
        rows.append(row)

        print(
            f"[eval-push] ep={i:03d} seed={ep_seed} "
            f"return={row['return']:.1f} "
            f"survived={row['survived_to_max_steps']} "
            f"push_ok={row['survived_push_window']} "
            f"recovered={row['recovered_after_push']} "
            f"retention={row['post_push_velocity_retention']:.2f}"
        )

    csv_path = out_dir / "episodes.csv"
    json_path = out_dir / "summary.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    config_like_numeric_keys = {"push_duration"}
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and key not in config_like_numeric_keys
    ]

    pushed_rows = [r for r in rows if r["was_pushed"]]
    push_window_rows = [r for r in rows if r["survived_push_window"]]
    recovered_rows = [r for r in rows if r["recovered_after_push"]]

    push_exposure_rate = len(pushed_rows) / len(rows)
    push_window_survival_rate = len(push_window_rows) / len(rows)
    recovery_rate = len(recovered_rows) / len(rows)

    summary = {
        "model": args.model,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "deterministic": args.deterministic,
        "force_body": args.force_body,
        "force_mode": args.force_mode,
        "force_magnitude": args.force_magnitude,
        "push_duration": args.push_duration,
        "include_lateral_error_observation": args.include_lateral_error_observation,
        "lateral_error_observation_clip": args.lateral_error_observation_clip,
        "use_controlled_locomotion_wrapper": args.use_controlled_locomotion_wrapper,
        "push_exposure_rate": push_exposure_rate,
        "early_failure_rate": 1.0 - push_exposure_rate,
        "push_window_survival_rate": push_window_survival_rate,
        "push_window_survival_rate_given_pushed": (
            len(push_window_rows) / len(pushed_rows) if pushed_rows else None
        ),
        "recovery_rate": recovery_rate,
        "recovery_rate_given_pushed": (
            len(recovered_rows) / len(pushed_rows) if pushed_rows else None
        ),
        "recovery_rate_given_push_window_survival": (
            len(recovered_rows) / len(push_window_rows) if push_window_rows else None
        ),
        "survival_rate": sum(r["survived_to_max_steps"] for r in rows) / len(rows),
        "termination_rate": sum(r["terminated"] for r in rows) / len(rows),
    }

    for key in numeric_keys:
        summary[key] = summarize([float(r[key]) for r in rows])

    if recovered_rows:
        summary["recovered_recovery_time_s"] = summarize(
            [float(r["recovery_time_s"]) for r in recovered_rows]
        )
    else:
        summary["recovered_recovery_time_s"] = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] csv:", csv_path)
    print("[done] summary:", json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
