from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from stable_baselines3 import PPO

from eval_policy_metrics import summarize
from eval_policy_push import evaluate_episode
from stable_directional_ant import ControlledLocomotionRewardConfig


def force_label(value: float) -> str:
    label = f"{value:g}"
    return label.replace(".", "p")


def write_run_outputs(out_dir: Path, rows: list[dict], args, magnitude: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "episodes.csv"
    json_path = out_dir / "summary.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pushed_rows = [r for r in rows if r["was_pushed"]]
    push_window_rows = [r for r in rows if r["survived_push_window"]]
    recovered_rows = [r for r in rows if r["recovered_after_push"]]

    summary = {
        "model": args.model,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "deterministic": args.deterministic,
        "force_body": args.force_body,
        "force_mode": args.force_mode,
        "force_magnitude": magnitude,
        "push_duration": args.push_duration,
        "include_lateral_error_observation": args.include_lateral_error_observation,
        "lateral_error_observation_clip": args.lateral_error_observation_clip,
        "use_controlled_locomotion_wrapper": args.use_controlled_locomotion_wrapper,
        "push_exposure_rate": len(pushed_rows) / len(rows),
        "early_failure_rate": 1.0 - (len(pushed_rows) / len(rows)),
        "push_window_survival_rate": len(push_window_rows) / len(rows),
        "push_window_survival_rate_given_pushed": (
            len(push_window_rows) / len(pushed_rows) if pushed_rows else None
        ),
        "recovery_rate": len(recovered_rows) / len(rows),
        "recovery_rate_given_pushed": (
            len(recovered_rows) / len(pushed_rows) if pushed_rows else None
        ),
        "recovery_rate_given_push_window_survival": (
            len(recovered_rows) / len(push_window_rows) if push_window_rows else None
        ),
        "survival_rate": sum(r["survived_to_max_steps"] for r in rows) / len(rows),
        "termination_rate": sum(r["terminated"] for r in rows) / len(rows),
    }

    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key != "push_duration"
    ]
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

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[20, 30, 40, 50, 60])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--force-body", type=str, default="torso")
    parser.add_argument(
        "--force-mode",
        type=str,
        default="lateral",
        choices=["lateral", "backward", "forward", "random_xy", "random_cardinal"],
    )
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

    root_out_dir = Path(args.out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    controlled_reward_config = ControlledLocomotionRewardConfig(
        target_forward_velocity=args.controlled_target_forward_velocity,
        target_yaw=args.controlled_target_yaw,
        target_height=args.controlled_target_height,
        target_velocity_obs_scale=args.controlled_target_velocity_obs_scale,
        lateral_position_clip=args.lateral_error_observation_clip,
    )

    sweep_rows = []
    for magnitude in args.magnitudes:
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
                force_magnitude=magnitude,
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

        run_out_dir = root_out_dir / f"force_{force_label(magnitude)}"
        summary = write_run_outputs(run_out_dir, rows, args=args, magnitude=magnitude)

        compact = {
            "force_magnitude": magnitude,
            "push_exposure_rate": summary["push_exposure_rate"],
            "push_window_survival_rate": summary["push_window_survival_rate"],
            "push_window_survival_rate_given_pushed": summary[
                "push_window_survival_rate_given_pushed"
            ],
            "recovery_rate": summary["recovery_rate"],
            "recovery_rate_given_pushed": summary["recovery_rate_given_pushed"],
            "survival_rate": summary["survival_rate"],
            "steps_mean": summary["steps"]["mean"],
            "return_mean": summary["return"]["mean"],
            "post_push_velocity_retention_mean": summary[
                "post_push_velocity_retention"
            ]["mean"],
            "final_yaw_abs_mean": summary["final_yaw_abs"]["mean"],
        }
        sweep_rows.append(compact)

        print(
            f"[sweep] force={magnitude:g} "
            f"push_window={compact['push_window_survival_rate']:.2f} "
            f"recovery={compact['recovery_rate']:.2f} "
            f"survival={compact['survival_rate']:.2f}"
        )

    sweep_csv_path = root_out_dir / "sweep_summary.csv"
    sweep_json_path = root_out_dir / "sweep_summary.json"

    with sweep_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    with sweep_json_path.open("w", encoding="utf-8") as f:
        json.dump(sweep_rows, f, indent=2)

    print("[done] sweep csv:", sweep_csv_path)
    print("[done] sweep json:", sweep_json_path)


if __name__ == "__main__":
    main()
