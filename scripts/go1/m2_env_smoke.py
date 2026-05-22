"""
M2 smoke tests: verify Go1Env behaves sanely before training.

A — Zero action standing: action=0, 10s, expect no fall, ~M1 metrics.
B — Small random action: action ~ U(-0.1,0.1), 10s, expect mostly standing.
C — Full random action: action ~ U(-1,1), 5s, OK to fall but no NaN/explosion.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from go1_env import Go1Env, Go1EnvConfig, N_JOINTS


def run_smoke(
    label: str,
    action_fn,
    duration_s: float,
    seed: int,
    record_video: bool = True,
) -> dict:
    cfg = Go1EnvConfig(
        command_vx=0.0,
        command_vy=0.0,
        command_wz=0.0,
        max_episode_steps=int(duration_s / (Go1EnvConfig.decimation * 0.002)),
    )
    env = Go1Env(config=cfg, render_mode="rgb_array" if record_video else None)
    obs, info = env.reset(seed=seed)
    print(f"\n=== M2-{label} ===")
    print(f"  initial z={info['z']:.4f}, sim_time={info['sim_time_s']:.3f}s")

    z_log = []
    roll_pitch_log = []
    joint_err_log = []
    nan_seen = False
    terminated = False
    term_reason = ""
    rng = np.random.default_rng(seed)
    actions_log = []

    n_policy_steps = int(duration_s / (cfg.decimation * 0.002))
    fps = 25
    sim_steps_per_frame = max(int(round(50 / fps)), 1)

    frames = []
    for t in range(n_policy_steps):
        a = action_fn(rng).astype(np.float32)
        actions_log.append(a)
        obs, reward, terminated, truncated, info = env.step(a)
        z_log.append(info["z"])

        if record_video and (t % sim_steps_per_frame == 0):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if np.any(np.isnan(obs)):
            nan_seen = True
            term_reason = "nan_in_obs"
            break
        if terminated:
            term_reason = info.get("term_reason", "terminated")
            break

    final_t_s = (t + 1) * cfg.decimation * 0.002
    print(f"  stopped after {t+1} policy steps = {final_t_s:.3f}s")
    print(f"  terminated={terminated}, reason='{term_reason}'")
    print(f"  z: mean={np.mean(z_log):.4f} std={np.std(z_log):.4f} "
          f"min={np.min(z_log):.4f} max={np.max(z_log):.4f}")

    actions_arr = np.stack(actions_log) if actions_log else np.zeros((0, N_JOINTS))
    metrics = {
        "label": label,
        "duration_requested_s": duration_s,
        "duration_actual_s": final_t_s,
        "policy_steps_completed": t + 1,
        "terminated": bool(terminated),
        "term_reason": term_reason,
        "nan_seen": nan_seen,
        "z_mean": float(np.mean(z_log)),
        "z_std": float(np.std(z_log)),
        "z_min": float(np.min(z_log)),
        "z_max": float(np.max(z_log)),
        "action_mean_abs": float(np.mean(np.abs(actions_arr))) if actions_arr.size else 0.0,
        "action_max_abs": float(np.max(np.abs(actions_arr))) if actions_arr.size else 0.0,
    }

    if record_video and frames:
        out_video = Path(f"videos/go1_m2_{label}.mp4")
        out_video.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(out_video), fps=fps) as w:
            for fr in frames:
                w.append_data(fr)
        print(f"  video: {out_video}")
        metrics["video"] = str(out_video)

    env.close()
    return metrics


def main():
    results = {}

    # A: zero action standing
    results["A_zero"] = run_smoke(
        "A_zero", lambda rng: np.zeros(N_JOINTS), duration_s=10.0, seed=42
    )

    # B: small random action ~ U(-0.1, 0.1)
    results["B_small_random"] = run_smoke(
        "B_small_random",
        lambda rng: rng.uniform(-0.1, 0.1, size=N_JOINTS),
        duration_s=10.0,
        seed=42,
    )

    # C: full random action ~ U(-1, 1)
    results["C_full_random"] = run_smoke(
        "C_full_random",
        lambda rng: rng.uniform(-1.0, 1.0, size=N_JOINTS),
        duration_s=5.0,
        seed=42,
    )

    out = Path("reports/go1_m2_env_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print("\n=== M2 summary ===")
    for k, m in results.items():
        ok = "OK" if not m["nan_seen"] else "NaN"
        print(f"  {k:18s} {ok}  t={m['duration_actual_s']:5.2f}s  z=[{m['z_min']:.3f},{m['z_max']:.3f}]  reason='{m['term_reason']}'")
    print(f"\n[M2] results: {out}")


if __name__ == "__main__":
    main()
