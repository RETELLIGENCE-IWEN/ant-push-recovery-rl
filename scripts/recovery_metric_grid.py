"""
E2: Recovery Metric Grid.

Re-evaluate v4b and v4c using metrics that distinguish recovery *quality*, not
just survival. Uses running-window averages so that Ant's intrinsic gait
oscillation does not prevent the recovery condition from ever being satisfied.

Metrics per episode:
  survival_to_end
  recovery_time_vx_smoothed   (steps until 50-step running |vx_body-target| < 0.4)
  recovery_time_heading       (steps until 50-step running |yaw| < 0.25)
  recovery_time_yaw_rate      (steps until 50-step running |wz| < 0.6)
  post_push_max_lateral_deviation
  post_push_integrated_vx_error  (sum |vx_body - target| over 5s after push)
  post_push_integrated_yaw_abs   (sum |yaw| over 5s after push)
  post_push_max_yaw
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from stable_directional_ant import (
    ObservationHistoryStackWrapper,
    WellTrainedLocomotionAntWrapper,
    WellTrainedLocomotionRewardConfig,
    quat_wxyz_to_rpy,
    wrap_angle_rad,
)


def root_state(env: gym.Env) -> dict:
    qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
    roll, pitch, yaw = quat_wxyz_to_rpy(qpos[3:7])
    cy, sy = math.cos(yaw), math.sin(yaw)
    return {
        "x": float(qpos[0]),
        "y": float(qpos[1]),
        "z": float(qpos[2]),
        "yaw": yaw,
        "roll": roll,
        "pitch": pitch,
        "vx_body": float(cy * qvel[0] + sy * qvel[1]),
        "vy_body": float(-sy * qvel[0] + cy * qvel[1]),
        "wz": float(qvel[5]),
    }


def get_torso_id(env: gym.Env) -> int:
    m = env.unwrapped.model
    for i in range(m.nbody):
        if m.body(i).name == "torso":
            return i
    raise ValueError("torso not found")


def run_episode(
    model: PPO,
    seed: int,
    push_step: int,
    push_duration: int,
    force_xy: np.ndarray,
    torque_z: float,
    target_vx: float,
    max_steps: int = 800,
    window_steps: int = 50,            # smoothing window for recovery checks
    recovery_horizon_steps: int = 500, # max steps post-push to look for recovery
    vx_tol: float = 0.4,
    yaw_tol: float = 0.25,
    wz_tol: float = 0.6,
    history_stack_size: int = 1,
) -> dict:
    env = gym.make("Ant-v5")
    env = WellTrainedLocomotionAntWrapper(env, reward_config=WellTrainedLocomotionRewardConfig())
    if history_stack_size > 1:
        env = ObservationHistoryStackWrapper(env, stack_size=history_stack_size)
    torso_id = get_torso_id(env)

    obs, info = env.reset(seed=seed)
    initial = root_state(env)

    # Running windows for smoothed conditions
    vx_err_window = deque(maxlen=window_steps)
    yaw_window = deque(maxlen=window_steps)
    wz_window = deque(maxlen=window_steps)

    push_end_step = push_step + push_duration
    pre_push_y = None
    pre_push_state = None
    post_push_max_lateral_dev = 0.0
    integrated_vx_err = 0.0
    integrated_yaw_abs = 0.0
    post_push_max_yaw = 0.0
    recovery_time_vx = -1
    recovery_time_heading = -1
    recovery_time_yaw_rate = -1
    survived = True
    terminated = False
    truncated = False

    for t in range(max_steps):
        env.unwrapped.data.xfrc_applied[torso_id, :] = 0.0
        if t == push_step:
            pre_push_state = root_state(env)
            pre_push_y = pre_push_state["y"]
        if push_step <= t < push_end_step:
            env.unwrapped.data.xfrc_applied[torso_id, 0] = force_xy[0]
            env.unwrapped.data.xfrc_applied[torso_id, 1] = force_xy[1]
            env.unwrapped.data.xfrc_applied[torso_id, 5] = torque_z

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        s = root_state(env)

        vx_err_window.append(abs(s["vx_body"] - target_vx))
        yaw_window.append(abs(wrap_angle_rad(s["yaw"])))
        wz_window.append(abs(s["wz"]))

        if pre_push_y is not None and t >= push_step:
            steps_after_push_start = t - push_step
            steps_after_push_end = t - push_end_step

            # post-push lateral deviation max
            post_push_max_lateral_dev = max(
                post_push_max_lateral_dev, abs(s["y"] - pre_push_y)
            )

            # post-push integrated metrics in first 5s (500 steps after push end)
            if 0 <= steps_after_push_end < recovery_horizon_steps:
                integrated_vx_err += abs(s["vx_body"] - target_vx)
                integrated_yaw_abs += abs(wrap_angle_rad(s["yaw"]))
                post_push_max_yaw = max(post_push_max_yaw, abs(wrap_angle_rad(s["yaw"])))

                # Recovery: smoothed window mean within tolerance
                if len(vx_err_window) == window_steps:
                    if recovery_time_vx < 0 and (sum(vx_err_window) / window_steps) < vx_tol:
                        recovery_time_vx = steps_after_push_end
                    if recovery_time_heading < 0 and (sum(yaw_window) / window_steps) < yaw_tol:
                        recovery_time_heading = steps_after_push_end
                    if recovery_time_yaw_rate < 0 and (sum(wz_window) / window_steps) < wz_tol:
                        recovery_time_yaw_rate = steps_after_push_end

        if terminated or truncated:
            survived = bool(t >= max_steps - 1 and not terminated)
            break
    else:
        survived = True

    final = root_state(env)
    env.close()

    return {
        "seed": seed,
        "push_force_x": float(force_xy[0]),
        "push_force_y": float(force_xy[1]),
        "push_force_magnitude": float(math.hypot(force_xy[0], force_xy[1])),
        "push_torque_z": float(torque_z),
        "survived_to_end": survived,
        "fall_step": int(t) if terminated else -1,
        "recovery_time_vx_smoothed": recovery_time_vx,
        "recovery_time_heading": recovery_time_heading,
        "recovery_time_yaw_rate": recovery_time_yaw_rate,
        "post_push_max_lateral_deviation": post_push_max_lateral_dev,
        "post_push_integrated_vx_error": integrated_vx_err,
        "post_push_integrated_yaw_abs": integrated_yaw_abs,
        "post_push_max_yaw": post_push_max_yaw,
        "final_drift": final["y"] - initial["y"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True,
                   help="label=path entries comma-separated")
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--forces", type=str, default="5,10,15")
    p.add_argument("--directions-deg", type=str, default="0,90,180,270")
    p.add_argument("--torque-z", type=float, default=0.20)
    p.add_argument("--randomize-torque-sign", action="store_true")
    p.add_argument("--episodes-per-cell", type=int, default=4)
    p.add_argument("--push-step", type=int, default=250)
    p.add_argument("--push-duration", type=int, default=30)
    p.add_argument("--target-vx", type=float, default=2.0)
    p.add_argument("--history-stack-size", type=int, default=1,
                   help="If >1, stack last N observations to match training-time obs space.")
    p.add_argument("--per-model-history-sizes", type=str, default="",
                   help="Optional: comma-separated history sizes per model (same order as --models). "
                        "Overrides --history-stack-size when set.")
    args = p.parse_args()

    torch.set_num_threads(1)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    forces = [float(x) for x in args.forces.split(",")]
    directions = [float(x) for x in args.directions_deg.split(",")]
    model_entries = [tuple(e.split("=", 1)) for e in args.models.split(",")]

    print(f"[E2] models={[m[0] for m in model_entries]}")
    print(f"[E2] forces={forces} dirs={directions} torque={args.torque_z}")
    print(f"[E2] episodes/cell={args.episodes_per_cell}")

    per_model_history = {}
    if args.per_model_history_sizes:
        sizes = [int(s) for s in args.per_model_history_sizes.split(",")]
        for (label, _), sz in zip(model_entries, sizes):
            per_model_history[label] = sz

    rows = []
    for label, model_path in model_entries:
        hs = per_model_history.get(label, args.history_stack_size)
        print(f"\n=== {label} ({model_path}) [history_stack={hs}] ===")
        model = PPO.load(model_path, device="cpu")
        for force in forces:
            for d_deg in directions:
                theta = math.radians(d_deg)
                force_xy = np.array([force * math.cos(theta), force * math.sin(theta)])
                cell_rows = []
                for ep in range(args.episodes_per_cell):
                    if args.randomize_torque_sign:
                        torque_z = args.torque_z * (1.0 if ep % 2 == 0 else -1.0)
                    else:
                        torque_z = args.torque_z
                    row = run_episode(
                        model=model,
                        seed=6000 + ep,
                        push_step=args.push_step,
                        push_duration=args.push_duration,
                        force_xy=force_xy,
                        torque_z=torque_z,
                        target_vx=args.target_vx,
                        history_stack_size=hs,
                    )
                    row["policy"] = label
                    row["direction_deg"] = d_deg
                    rows.append(row)
                    cell_rows.append(row)
                surv = sum(r["survived_to_end"] for r in cell_rows) / len(cell_rows)
                vx_rec = [r["recovery_time_vx_smoothed"] for r in cell_rows if r["recovery_time_vx_smoothed"] >= 0]
                head_rec = [r["recovery_time_heading"] for r in cell_rows if r["recovery_time_heading"] >= 0]
                int_vx = np.mean([r["post_push_integrated_vx_error"] for r in cell_rows])
                int_yaw = np.mean([r["post_push_integrated_yaw_abs"] for r in cell_rows])
                print(
                    f"  F={force:5.1f}N dir={d_deg:5.1f}°  surv={surv:.2f}  "
                    f"rec_vx={np.mean(vx_rec) if vx_rec else -1:6.1f}  "
                    f"rec_head={np.mean(head_rec) if head_rec else -1:6.1f}  "
                    f"int_vx_err={int_vx:6.1f}  int_yaw={int_yaw:6.1f}"
                )
        del model

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[E2] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
