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
    ControlledLocomotionAntWrapper,
    ControlledLocomotionRewardConfig,
    LateralErrorObservationWrapper,
    WellTrainedLocomotionAntWrapper,
    WellTrainedLocomotionRewardConfig,
)


def wrap_angle_rad(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def quat_wxyz_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in q]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


def get_root_state(env: gym.Env) -> dict[str, float]:
    qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)

    x = float(qpos[0])
    y = float(qpos[1])
    z = float(qpos[2])

    roll, pitch, yaw = quat_wxyz_to_rpy(qpos[3:7])

    vx_world = float(qvel[0])
    vy_world = float(qvel[1])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vx_body = cos_yaw * vx_world + sin_yaw * vy_world
    vy_body = -sin_yaw * vx_world + cos_yaw * vy_world

    dof_vel = qvel[6:].copy()

    return {
        "x": x,
        "y": y,
        "z": z,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "vx": vx_world,
        "vy": vy_world,
        "vz": float(qvel[2]),
        "vx_body": float(vx_body),
        "vy_body": float(vy_body),
        "roll_rate": float(qvel[3]),
        "pitch_rate": float(qvel[4]),
        "yaw_rate": float(qvel[5]),
        "dof_vel_sq_sum": float(np.sum(dof_vel * dof_vel)),
    }


def summarize(values: list[float]) -> dict:
    if len(values) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "min": values[0], "max": values[0]}
    return {
        "mean": mean(values),
        "std": stdev(values),
        "min": min(values),
        "max": max(values),
    }


def evaluate_episode(
    model: PPO,
    seed: int,
    max_steps: int,
    deterministic: bool,
    include_lateral_error_observation: bool,
    lateral_error_observation_clip: float,
    use_controlled_locomotion_wrapper: bool,
    controlled_reward_config: ControlledLocomotionRewardConfig,
    use_well_trained_wrapper: bool,
    well_trained_reward_config: WellTrainedLocomotionRewardConfig,
) -> dict:
    env = gym.make("Ant-v5")
    if use_well_trained_wrapper:
        env = WellTrainedLocomotionAntWrapper(
            env,
            reward_config=well_trained_reward_config,
        )
    elif use_controlled_locomotion_wrapper:
        env = ControlledLocomotionAntWrapper(
            env,
            reward_config=controlled_reward_config,
        )
    elif include_lateral_error_observation:
        env = LateralErrorObservationWrapper(
            env,
            clip=lateral_error_observation_clip,
        )
    obs, info = env.reset(seed=seed)

    initial = get_root_state(env)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    action_norm_sum = 0.0
    action_energy_sum = 0.0
    action_delta_sq_sum = 0.0
    prev_action: np.ndarray | None = None

    yaw_abs_sum = 0.0
    yaw_rate_sq_sum = 0.0
    heading_alignment_sum = 0.0
    course_yaw_abs_sum = 0.0
    course_alignment_sum = 0.0
    speed_xy_sum = 0.0

    roll_sq_sum = 0.0
    pitch_sq_sum = 0.0
    roll_rate_sq_sum = 0.0
    pitch_rate_sq_sum = 0.0
    vertical_velocity_sq_sum = 0.0
    lateral_velocity_abs_sum = 0.0
    lateral_velocity_sq_sum = 0.0

    vx_body_sum = 0.0
    vy_body_abs_sum = 0.0
    vy_body_sq_sum = 0.0
    vx_body_err_sq_sum = 0.0
    dof_vel_sq_sum = 0.0

    heights: list[float] = []

    dt = float(env.unwrapped.dt)
    target_course_yaw = 0.0
    target_vx_body = 0.0
    target_vy_body = 0.0
    if use_well_trained_wrapper:
        target_vx_body = well_trained_reward_config.target_forward_velocity
        target_vy_body = well_trained_reward_config.target_lateral_velocity
    elif use_controlled_locomotion_wrapper:
        command_speed = math.hypot(
            controlled_reward_config.target_forward_velocity,
            controlled_reward_config.target_lateral_velocity,
        )
        if command_speed > 1e-9:
            target_course_yaw = math.atan2(
                controlled_reward_config.target_lateral_velocity,
                controlled_reward_config.target_forward_velocity,
            )

    for _ in range(max_steps):
        action, _state = model.predict(obs, deterministic=deterministic)
        action_np = np.asarray(action, dtype=np.float64)

        obs, reward, terminated, truncated, info = env.step(action)
        state = get_root_state(env)

        yaw_error = wrap_angle_rad(state["yaw"] - 0.0)
        speed_xy = math.hypot(state["vx"], state["vy"])
        course_error = 0.0
        if speed_xy > 1e-9:
            course_error = wrap_angle_rad(
                math.atan2(state["vy"], state["vx"]) - target_course_yaw
            )

        total_reward += float(reward)
        steps += 1

        action_norm_sum += float(np.linalg.norm(action_np))
        action_energy_sum += float(np.sum(action_np * action_np))

        if prev_action is not None:
            da = action_np - prev_action
            action_delta_sq_sum += float(np.sum(da * da))
        prev_action = action_np.copy()

        yaw_abs_sum += abs(yaw_error)
        yaw_rate_sq_sum += state["yaw_rate"] ** 2
        heading_alignment_sum += math.cos(yaw_error)
        course_yaw_abs_sum += abs(course_error)
        course_alignment_sum += math.cos(course_error)
        speed_xy_sum += speed_xy

        roll_sq_sum += state["roll"] ** 2
        pitch_sq_sum += state["pitch"] ** 2
        roll_rate_sq_sum += state["roll_rate"] ** 2
        pitch_rate_sq_sum += state["pitch_rate"] ** 2
        vertical_velocity_sq_sum += state["vz"] ** 2
        lateral_velocity_abs_sum += abs(state["vy"])
        lateral_velocity_sq_sum += state["vy"] ** 2

        vx_body_sum += state["vx_body"]
        vy_body_abs_sum += abs(state["vy_body"])
        vy_body_sq_sum += state["vy_body"] ** 2
        vx_body_err_sq_sum += (state["vx_body"] - target_vx_body) ** 2
        dof_vel_sq_sum += state["dof_vel_sq_sum"]

        heights.append(state["z"])

        if terminated or truncated:
            break

    final = get_root_state(env)
    env.close()

    duration_s = steps * dt
    distance_x = final["x"] - initial["x"]
    lateral_drift = final["y"] - initial["y"]

    height_mean = float(mean(heights)) if heights else 0.0
    height_std = float(stdev(heights)) if len(heights) >= 2 else 0.0

    return {
        "seed": seed,
        "return": total_reward,
        "steps": steps,
        "duration_s": duration_s,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "survived_to_max_steps": bool(steps >= max_steps and not terminated),

        "distance_x": distance_x,
        "mean_forward_velocity": distance_x / max(duration_s, 1e-9),

        "lateral_drift": lateral_drift,
        "abs_lateral_drift": abs(lateral_drift),
        "mean_abs_lateral_velocity": lateral_velocity_abs_sum / max(steps, 1),
        "lateral_velocity_rms": math.sqrt(lateral_velocity_sq_sum / max(steps, 1)),

        "final_yaw_abs": abs(wrap_angle_rad(final["yaw"])),
        "yaw_abs_mean": yaw_abs_sum / max(steps, 1),
        "yaw_rate_rms": math.sqrt(yaw_rate_sq_sum / max(steps, 1)),
        "heading_alignment_mean": heading_alignment_sum / max(steps, 1),
        "course_yaw_abs_mean": course_yaw_abs_sum / max(steps, 1),
        "course_alignment_mean": course_alignment_sum / max(steps, 1),
        "mean_xy_speed": speed_xy_sum / max(steps, 1),

        "roll_rms": math.sqrt(roll_sq_sum / max(steps, 1)),
        "pitch_rms": math.sqrt(pitch_sq_sum / max(steps, 1)),
        "roll_rate_rms": math.sqrt(roll_rate_sq_sum / max(steps, 1)),
        "pitch_rate_rms": math.sqrt(pitch_rate_sq_sum / max(steps, 1)),
        "height_mean": height_mean,
        "height_std": height_std,
        "vertical_velocity_rms": math.sqrt(vertical_velocity_sq_sum / max(steps, 1)),

        "vx_body_mean": vx_body_sum / max(steps, 1),
        "vx_body_err_rms": math.sqrt(vx_body_err_sq_sum / max(steps, 1)),
        "vy_body_mean_abs": vy_body_abs_sum / max(steps, 1),
        "vy_body_rms": math.sqrt(vy_body_sq_sum / max(steps, 1)),
        "dof_vel_rms": math.sqrt(dof_vel_sq_sum / max(steps, 1)),

        "mean_action_norm": action_norm_sum / max(steps, 1),
        "mean_action_energy": action_energy_sum / max(steps, 1),
        "action_delta_rms": math.sqrt(action_delta_sq_sum / max(steps - 1, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, required=True)
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
    args = parser.parse_args()

    torch.set_num_threads(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    controlled_reward_config = ControlledLocomotionRewardConfig(
        target_forward_velocity=args.controlled_target_forward_velocity,
        target_lateral_velocity=args.controlled_target_lateral_velocity,
        target_yaw_rate=args.controlled_target_yaw_rate,
        target_yaw=args.controlled_target_yaw,
        target_height=args.controlled_target_height,
        target_velocity_obs_scale=args.controlled_target_velocity_obs_scale,
        target_yaw_rate_obs_scale=args.controlled_target_yaw_rate_obs_scale,
        lateral_position_clip=args.lateral_error_observation_clip,
        include_command_observation=args.controlled_include_command_observation,
    )
    well_trained_reward_config = WellTrainedLocomotionRewardConfig(
        target_forward_velocity=args.well_trained_target_forward_velocity,
        target_lateral_velocity=args.well_trained_target_lateral_velocity,
        target_yaw_rate=args.well_trained_target_yaw_rate,
        target_height=args.well_trained_target_height,
        velocity_obs_scale=args.well_trained_velocity_obs_scale,
        include_command_observation=args.well_trained_include_command_observation,
    )

    rows = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        row = evaluate_episode(
            model=model,
            seed=ep_seed,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            include_lateral_error_observation=args.include_lateral_error_observation,
            lateral_error_observation_clip=args.lateral_error_observation_clip,
            use_controlled_locomotion_wrapper=args.use_controlled_locomotion_wrapper,
            controlled_reward_config=controlled_reward_config,
            use_well_trained_wrapper=args.use_well_trained_wrapper,
            well_trained_reward_config=well_trained_reward_config,
        )
        rows.append(row)

        print(
            f"[eval-metrics] ep={i:03d} seed={ep_seed} "
            f"return={row['return']:.1f} "
            f"vx_body={row['vx_body_mean']:.2f} "
            f"vy_body_abs={row['vy_body_mean_abs']:.3f} "
            f"yaw_abs={row['yaw_abs_mean']:.3f} "
            f"drift={row['abs_lateral_drift']:.2f} "
            f"h_std={row['height_std']:.3f} "
            f"act_dr={row['action_delta_rms']:.3f} "
            f"survived={row['survived_to_max_steps']}"
        )

    csv_path = out_dir / "episodes.csv"
    json_path = out_dir / "summary.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]

    summary = {
        "model": args.model,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "deterministic": args.deterministic,
        "include_lateral_error_observation": args.include_lateral_error_observation,
        "lateral_error_observation_clip": args.lateral_error_observation_clip,
        "use_controlled_locomotion_wrapper": args.use_controlled_locomotion_wrapper,
        "controlled_target_forward_velocity": args.controlled_target_forward_velocity,
        "controlled_target_yaw": args.controlled_target_yaw,
        "controlled_target_height": args.controlled_target_height,
        "use_well_trained_wrapper": args.use_well_trained_wrapper,
        "well_trained_target_forward_velocity": args.well_trained_target_forward_velocity,
        "well_trained_target_lateral_velocity": args.well_trained_target_lateral_velocity,
        "well_trained_target_yaw_rate": args.well_trained_target_yaw_rate,
        "well_trained_target_height": args.well_trained_target_height,
        "survival_rate": sum(r["survived_to_max_steps"] for r in rows) / len(rows),
        "termination_rate": sum(r["terminated"] for r in rows) / len(rows),
    }

    for key in numeric_keys:
        summary[key] = summarize([float(r[key]) for r in rows])

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] csv:", csv_path)
    print("[done] summary:", json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
