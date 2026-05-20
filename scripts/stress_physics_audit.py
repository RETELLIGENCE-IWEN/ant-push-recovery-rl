"""
E1: Stress Physics Audit.

For each (policy, force, direction, torque) condition, apply a single controlled
push to the torso at t=2.5s, log root state immediately before, during, and after
the push window, and follow the policy through the rest of the episode while
measuring recovery and fall behavior.

Goal: replace first-order impulse calculations with *measured* effective root
velocity changes, so we know how much disturbance the policy actually experiences.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
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
    WellTrainedLocomotionAntWrapper,
    WellTrainedLocomotionRewardConfig,
    quat_wxyz_to_rpy,
    wrap_angle_rad,
)


def get_root_state(env: gym.Env) -> dict:
    qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
    roll, pitch, yaw = quat_wxyz_to_rpy(qpos[3:7])
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
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
        "vx_body": float(cos_y * qvel[0] + sin_y * qvel[1]),
        "vy_body": float(-sin_y * qvel[0] + cos_y * qvel[1]),
        "wx": float(qvel[3]),
        "wy": float(qvel[4]),
        "wz": float(qvel[5]),
    }


def get_torso_id(env: gym.Env) -> int:
    m = env.unwrapped.model
    for i in range(m.nbody):
        if m.body(i).name == "torso":
            return i
    raise ValueError("torso not found")


def is_alive(s: dict) -> bool:
    # Ant healthy_range is roughly z in [0.2, 1.0] in Ant-v5 default termination
    return 0.2 <= s["z"] <= 1.0


def run_audit_episode(
    model: PPO,
    seed: int,
    push_step: int,
    push_duration: int,
    force_xy: np.ndarray,
    torque_z: float,
    target_vx: float,
    max_steps: int = 800,
    recovery_window_steps: int = 500,  # 5s @ 100Hz
) -> dict:
    env = gym.make("Ant-v5")
    env = WellTrainedLocomotionAntWrapper(
        env, reward_config=WellTrainedLocomotionRewardConfig()
    )
    torso_id = get_torso_id(env)
    dt = float(env.unwrapped.dt)

    obs, info = env.reset(seed=seed)
    initial = get_root_state(env)

    pre_push_state = None
    end_push_state = None
    end_push_step_actual = None
    max_roll_1s = 0.0
    max_pitch_1s = 0.0
    fall_time = -1
    recovery_step = -1  # steps after push end at which recovery condition first met for 50 contiguous steps
    recovery_buffer = 0  # contiguous steps satisfying recovery condition
    terminated = False
    truncated = False

    for t in range(max_steps):
        env.unwrapped.data.xfrc_applied[torso_id, :] = 0.0

        if t == push_step:
            pre_push_state = get_root_state(env)

        if push_step <= t < push_step + push_duration:
            env.unwrapped.data.xfrc_applied[torso_id, 0] = force_xy[0]
            env.unwrapped.data.xfrc_applied[torso_id, 1] = force_xy[1]
            env.unwrapped.data.xfrc_applied[torso_id, 5] = torque_z

        if t == push_step + push_duration:
            end_push_state = get_root_state(env)
            end_push_step_actual = t

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        s = get_root_state(env)

        # Track max roll/pitch within 1s after push end
        if end_push_step_actual is not None:
            steps_after = t - end_push_step_actual
            if 0 <= steps_after < 100:
                max_roll_1s = max(max_roll_1s, abs(s["roll"]))
                max_pitch_1s = max(max_pitch_1s, abs(s["pitch"]))

            # Recovery: |vx_body - target| < 0.4 AND |vy_body| < 0.4 AND
            #          |yaw| < 0.35 AND |yaw_rate| < 0.6, sustained 50 steps
            if 0 <= steps_after < recovery_window_steps:
                cond = (
                    abs(s["vx_body"] - target_vx) < 0.4
                    and abs(s["vy_body"]) < 0.4
                    and abs(wrap_angle_rad(s["yaw"])) < 0.35
                    and abs(s["wz"]) < 0.6
                )
                if cond:
                    recovery_buffer += 1
                    if recovery_buffer >= 50 and recovery_step < 0:
                        recovery_step = steps_after - 50
                else:
                    recovery_buffer = 0

        if not is_alive(s) and fall_time < 0:
            fall_time = t - (push_step + push_duration) if end_push_step_actual else t

        if terminated or truncated:
            break

    final = get_root_state(env)
    env.close()

    if pre_push_state is None or end_push_state is None:
        return None

    delta_vx = end_push_state["vx"] - pre_push_state["vx"]
    delta_vy = end_push_state["vy"] - pre_push_state["vy"]
    delta_vz = end_push_state["vz"] - pre_push_state["vz"]
    delta_v_xy_norm = math.hypot(delta_vx, delta_vy)
    delta_wz = end_push_state["wz"] - pre_push_state["wz"]
    push_impulse_lin_N_s = math.hypot(force_xy[0], force_xy[1]) * (push_duration * dt)
    push_impulse_ang_Nms = abs(torque_z) * (push_duration * dt)

    return {
        "seed": seed,
        "push_force_x": float(force_xy[0]),
        "push_force_y": float(force_xy[1]),
        "push_force_magnitude": float(math.hypot(force_xy[0], force_xy[1])),
        "push_torque_z": float(torque_z),
        "push_duration_steps": push_duration,
        "push_duration_s": push_duration * dt,
        "push_impulse_lin_N_s": push_impulse_lin_N_s,
        "push_impulse_ang_Nms": push_impulse_ang_Nms,
        "pre_vx": pre_push_state["vx"],
        "pre_vy": pre_push_state["vy"],
        "pre_vz": pre_push_state["vz"],
        "pre_wz": pre_push_state["wz"],
        "pre_yaw": pre_push_state["yaw"],
        "pre_z": pre_push_state["z"],
        "end_vx": end_push_state["vx"],
        "end_vy": end_push_state["vy"],
        "end_vz": end_push_state["vz"],
        "end_wz": end_push_state["wz"],
        "end_z": end_push_state["z"],
        "delta_vx": delta_vx,
        "delta_vy": delta_vy,
        "delta_vz": delta_vz,
        "delta_v_xy_norm": delta_v_xy_norm,
        "delta_wz": delta_wz,
        "max_roll_1s_after_push": max_roll_1s,
        "max_pitch_1s_after_push": max_pitch_1s,
        "fall_time_steps": fall_time,
        "recovery_steps_after_push": recovery_step,
        "survived_to_end": bool(not terminated and not truncated),
        "final_x": final["x"],
        "final_y": final["y"],
        "final_drift": final["y"] - initial["y"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True,
                   help="Comma-separated label=path entries, e.g. 'v4b=runs/.../final_model.zip,v4c=runs/.../final_model.zip'")
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--forces", type=str, default="5,10,15")
    p.add_argument("--directions-deg", type=str, default="0,90,180,270")
    p.add_argument("--torques", type=str, default="0,0.20", help="Comma-separated torque_z magnitudes (signs alternated per ep)")
    p.add_argument("--episodes-per-cell", type=int, default=3)
    p.add_argument("--push-step", type=int, default=250)
    p.add_argument("--push-duration", type=int, default=30)
    p.add_argument("--target-vx", type=float, default=2.0)
    args = p.parse_args()

    torch.set_num_threads(1)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    forces = [float(x) for x in args.forces.split(",")]
    directions = [float(x) for x in args.directions_deg.split(",")]
    torques = [float(x) for x in args.torques.split(",")]
    model_entries = []
    for entry in args.models.split(","):
        label, path = entry.split("=", 1)
        model_entries.append((label, path))

    print(f"[E1] forces={forces} directions={directions} torques={torques}")
    print(f"[E1] models: {model_entries}")
    print(f"[E1] episodes/cell={args.episodes_per_cell}, push_step={args.push_step}, duration={args.push_duration}")

    rows = []
    for label, model_path in model_entries:
        print(f"\n=== Auditing {label} ({model_path}) ===")
        model = PPO.load(model_path, device="cpu")
        for force in forces:
            for d_deg in directions:
                theta = math.radians(d_deg)
                force_xy = np.array([force * math.cos(theta), force * math.sin(theta)])
                for torque_mag in torques:
                    for ep in range(args.episodes_per_cell):
                        # Alternate torque sign across episodes
                        torque_z = torque_mag * (1.0 if ep % 2 == 0 else -1.0)
                        row = run_audit_episode(
                            model=model,
                            seed=5000 + ep,
                            push_step=args.push_step,
                            push_duration=args.push_duration,
                            force_xy=force_xy,
                            torque_z=torque_z,
                            target_vx=args.target_vx,
                        )
                        if row is None:
                            continue
                        row["policy"] = label
                        row["direction_deg"] = d_deg
                        rows.append(row)
                # one line per (force, dir, torque_mag) cell summary
                cells = [r for r in rows if r["policy"] == label
                         and r["push_force_magnitude"] == force
                         and r["direction_deg"] == d_deg]
                if cells:
                    last_cells = cells[-args.episodes_per_cell * len(torques):]
                    dv = np.mean([abs(r["delta_v_xy_norm"]) for r in last_cells])
                    dwz = np.mean([abs(r["delta_wz"]) for r in last_cells])
                    rec_ok = [r["recovery_steps_after_push"] for r in last_cells if r["recovery_steps_after_push"] >= 0]
                    rec = np.mean(rec_ok) if rec_ok else -1
                    surv = sum(r["survived_to_end"] for r in last_cells) / len(last_cells)
                    print(f"  F={force:5.1f}N dir={d_deg:5.1f}°  |dv|={dv:.2f}m/s  |dwz|={dwz:.2f}rad/s  rec_steps={rec:.0f}  surv={surv:.2f}")
        del model

    if not rows:
        raise RuntimeError("No rows produced.")

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[E1] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
