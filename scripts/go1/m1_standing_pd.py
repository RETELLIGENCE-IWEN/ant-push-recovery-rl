"""
M1: Standing PD controller sanity check for Go1.

Reset to the model's `home` keyframe (standing pose) and hold the standing
joint targets via the position actuators (built-in PD with kp=100) for a
fixed duration. Measure whether the body remains upright and joints stay
near the target. If this fails, RL on top is almost guaranteed to fail too.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import imageio
import mujoco
import numpy as np

MODEL_SCENE = Path("/workspace/external/mujoco_menagerie/unitree_go1/scene.xml")
OUT_METRICS = Path("reports/go1_standing_pd_metrics.json")
OUT_VIDEO = Path("videos/go1_standing_pd.mp4")


def quat_wxyz_to_rpy(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(MODEL_SCENE))
    data = mujoco.MjData(model)

    # Find the "home" keyframe
    home_key_id = -1
    for i in range(model.nkey):
        if model.key(i).name == "home":
            home_key_id = i
            break
    assert home_key_id >= 0, "Expected a `home` keyframe in the Go1 model"

    target_qpos = np.array(model.key_qpos[home_key_id], copy=True)
    target_ctrl = np.array(model.key_ctrl[home_key_id], copy=True)
    print(f"[M1] home keyframe ctrl (12 joint targets): {target_ctrl}")
    print(f"[M1] home keyframe qpos[:7] (free joint): {target_qpos[:7]}")

    # Reset to home
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)

    # Sanity: are we in a stable initial state? Check ctrl matches qpos[7:]
    init_joints = np.array(data.qpos[7:], copy=True)
    print(f"[M1] after reset, joint qpos = {init_joints}")
    print(f"[M1] ctrl will be = {target_ctrl}")

    # Hold standing pose for `duration_s` seconds; record per-step state
    duration_s = 5.0
    sim_steps = int(duration_s / model.opt.timestep)
    fps = 30
    sim_steps_per_frame = max(int((1.0 / fps) / model.opt.timestep), 1)
    total_frames = int(duration_s * fps)

    renderer = mujoco.Renderer(model, height=480, width=640)

    z_history = []
    roll_history = []
    pitch_history = []
    yaw_history = []
    joint_err_max_history = []
    joint_err_rms_history = []
    fall = False
    fall_time_s = -1.0

    foot_keywords = ["foot", "calf"]  # Go1 doesn't have a body literally named foot
    foot_geom_ids = []
    for i in range(model.ngeom):
        gname = model.geom(i).name or ""
        if any(k in gname.lower() for k in ["foot"]):
            foot_geom_ids.append(i)
    # foot collisions in Go1 are spheres attached at the calf end via default class
    # so we identify them by their classname pattern instead — fall back: use last 4
    # bodies named *_calf and report their position z.
    calf_body_ids = [
        i for i in range(model.nbody)
        if (model.body(i).name or "").endswith("_calf")
    ]
    print(f"[M1] calf body ids (foot-attached): {calf_body_ids}")

    with imageio.get_writer(str(OUT_VIDEO), fps=fps) as writer:
        for f_idx in range(total_frames):
            for _ in range(sim_steps_per_frame):
                data.ctrl[:] = target_ctrl
                mujoco.mj_step(model, data)

                z = float(data.qpos[2])
                roll, pitch, yaw = quat_wxyz_to_rpy(data.qpos[3:7])
                joint_err = data.qpos[7:] - target_ctrl
                z_history.append(z)
                roll_history.append(roll)
                pitch_history.append(pitch)
                yaw_history.append(yaw)
                joint_err_max_history.append(float(np.max(np.abs(joint_err))))
                joint_err_rms_history.append(float(np.sqrt(np.mean(joint_err**2))))

                if not fall and z < 0.15:
                    fall = True
                    fall_time_s = float(data.time)

            renderer.update_scene(data, camera="tracking")
            frame = renderer.render()
            writer.append_data(frame)

    renderer.close()

    # Summary metrics
    metrics = {
        "duration_s": duration_s,
        "timestep_s": float(model.opt.timestep),
        "fell": fall,
        "fall_time_s": fall_time_s,
        "z_mean": statistics.mean(z_history),
        "z_std": statistics.stdev(z_history) if len(z_history) > 1 else 0.0,
        "z_min": min(z_history),
        "z_max": max(z_history),
        "z_initial_target": float(target_qpos[2]),
        "roll_rms": math.sqrt(sum(r*r for r in roll_history) / len(roll_history)),
        "pitch_rms": math.sqrt(sum(p*p for p in pitch_history) / len(pitch_history)),
        "yaw_drift_abs": abs(yaw_history[-1] - yaw_history[0]),
        "joint_err_max_overall": max(joint_err_max_history),
        "joint_err_rms_overall": math.sqrt(
            sum(e*e for e in joint_err_rms_history) / len(joint_err_rms_history)
        ),
    }

    with OUT_METRICS.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== M1 standing PD result ===")
    print(f"  fell within {duration_s}s? {metrics['fell']} (fall_time={metrics['fall_time_s']:.3f}s)")
    print(f"  z: mean={metrics['z_mean']:.4f} std={metrics['z_std']:.4f} "
          f"range=[{metrics['z_min']:.4f}, {metrics['z_max']:.4f}]  target={metrics['z_initial_target']:.4f}")
    print(f"  roll_rms={metrics['roll_rms']:.4f} pitch_rms={metrics['pitch_rms']:.4f}")
    print(f"  yaw drift over episode: {metrics['yaw_drift_abs']:.4f} rad")
    print(f"  joint pos error: max={metrics['joint_err_max_overall']:.4f} rad, "
          f"rms over time={metrics['joint_err_rms_overall']:.4f} rad")
    print(f"\n[M1] video: {OUT_VIDEO}")
    print(f"[M1] metrics: {OUT_METRICS}")

    # Success criteria
    success = (
        not metrics["fell"]
        and abs(metrics["z_mean"] - metrics["z_initial_target"]) < 0.04
        and metrics["z_std"] < 0.02
        and metrics["roll_rms"] < 0.1
        and metrics["pitch_rms"] < 0.1
        and metrics["joint_err_rms_overall"] < 0.10
    )
    print(f"\n[M1] standing PD success: {success}")
    if not success:
        print("[M1] failure analysis:")
        if metrics["fell"]:
            print(f"  - body fell to z<0.15 at t={metrics['fall_time_s']:.3f}s")
        if abs(metrics["z_mean"] - metrics["z_initial_target"]) >= 0.04:
            print(f"  - z_mean deviates from target by {abs(metrics['z_mean'] - metrics['z_initial_target']):.4f}m (>0.04)")
        if metrics["z_std"] >= 0.02:
            print(f"  - z_std={metrics['z_std']:.4f} > 0.02 (bouncing/jitter)")
        if metrics["roll_rms"] >= 0.1 or metrics["pitch_rms"] >= 0.1:
            print(f"  - body tilt: roll {metrics['roll_rms']:.3f}, pitch {metrics['pitch_rms']:.3f}")
        if metrics["joint_err_rms_overall"] >= 0.10:
            print(f"  - joint tracking poor: rms err {metrics['joint_err_rms_overall']:.4f} rad")


if __name__ == "__main__":
    main()
