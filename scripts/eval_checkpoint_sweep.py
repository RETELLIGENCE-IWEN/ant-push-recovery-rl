from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import torch
from stable_baselines3 import PPO

from eval_policy_metrics import evaluate_episode, summarize
from stable_directional_ant import ControlledLocomotionRewardConfig


def safe_label(path: Path) -> str:
    if path.name == "final_model.zip":
        return f"{path.parent.parent.name}_final"
    return path.stem


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    model_path: Path,
    rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    summary = {
        "model": str(model_path),
        "label": safe_label(model_path),
        "episodes": args.episodes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "deterministic": args.deterministic,
        "survival_rate": sum(r["survived_to_max_steps"] for r in rows) / len(rows),
        "termination_rate": sum(r["terminated"] for r in rows) / len(rows),
    }
    for key in numeric_keys:
        summary[key] = summarize([float(r[key]) for r in rows])

    summary["selection_score"] = selection_score(summary, args)
    return summary


def mean_value(summary: dict, key: str, default: float = 0.0) -> float:
    value = summary.get(key)
    if not isinstance(value, dict):
        return default
    mean = value.get("mean")
    return float(mean) if mean is not None else default


def max_value(summary: dict, key: str, default: float = 0.0) -> float:
    value = summary.get(key)
    if not isinstance(value, dict):
        return default
    max_item = value.get("max")
    return float(max_item) if max_item is not None else default


def selection_score(summary: dict, args: argparse.Namespace) -> float:
    survival = float(summary["survival_rate"])
    drift_mean = mean_value(summary, "abs_lateral_drift")
    drift_max = max_value(summary, "abs_lateral_drift")
    velocity = mean_value(summary, "mean_forward_velocity")
    heading = mean_value(summary, "heading_alignment_mean")
    course = mean_value(summary, "course_alignment_mean")
    height_std = mean_value(summary, "height_std")
    action_delta = mean_value(summary, "action_delta_rms")
    velocity_error = abs(velocity - args.score_target_velocity)

    return (
        args.score_survival_weight * survival
        + args.score_heading_weight * heading
        + args.score_course_weight * course
        - args.score_drift_weight * drift_mean
        - args.score_drift_max_weight * drift_max
        - args.score_velocity_error_weight * velocity_error
        - args.score_height_std_weight * height_std
        - args.score_action_delta_weight * action_delta
    )


def summary_flat_row(summary: dict) -> dict:
    keys = [
        "selection_score",
        "survival_rate",
        "termination_rate",
        "mean_forward_velocity",
        "abs_lateral_drift",
        "lateral_velocity_rms",
        "heading_alignment_mean",
        "yaw_abs_mean",
        "course_alignment_mean",
        "course_yaw_abs_mean",
        "height_std",
        "vertical_velocity_rms",
        "mean_action_energy",
        "action_delta_rms",
        "return",
    ]
    row = {
        "label": summary["label"],
        "model": summary["model"],
    }
    for key in keys:
        value = summary.get(key)
        if isinstance(value, dict):
            row[f"{key}.mean"] = value.get("mean")
            row[f"{key}.std"] = value.get("std")
            row[f"{key}.max"] = value.get("max")
        else:
            row[key] = value
    return row


def find_models(args: argparse.Namespace) -> list[Path]:
    checkpoint_dir = Path(args.checkpoint_dir)
    models = sorted(checkpoint_dir.glob(args.model_glob))
    for model_path in args.include_model:
        path = Path(model_path)
        if path not in models:
            models.append(path)
    if args.limit is not None:
        models = models[: args.limit]
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--model-glob", type=str, default="*.zip")
    parser.add_argument("--include-model", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--raw-ant-env", action="store_true")
    parser.add_argument("--include-lateral-error-observation", action="store_true")
    parser.add_argument("--lateral-error-observation-clip", type=float, default=5.0)
    parser.add_argument("--controlled-target-forward-velocity", type=float, default=2.0)
    parser.add_argument("--controlled-target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--controlled-target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--controlled-target-yaw", type=float, default=0.0)
    parser.add_argument("--controlled-target-height", type=float, default=0.53)
    parser.add_argument("--controlled-target-velocity-obs-scale", type=float, default=3.0)
    parser.add_argument("--controlled-target-yaw-rate-obs-scale", type=float, default=2.0)
    parser.add_argument("--controlled-include-command-observation", action="store_true")

    parser.add_argument("--score-target-velocity", type=float, default=2.0)
    parser.add_argument("--score-survival-weight", type=float, default=10.0)
    parser.add_argument("--score-heading-weight", type=float, default=1.0)
    parser.add_argument("--score-course-weight", type=float, default=1.0)
    parser.add_argument("--score-drift-weight", type=float, default=0.08)
    parser.add_argument("--score-drift-max-weight", type=float, default=0.04)
    parser.add_argument("--score-velocity-error-weight", type=float, default=0.5)
    parser.add_argument("--score-height-std-weight", type=float, default=1.0)
    parser.add_argument("--score-action-delta-weight", type=float, default=0.4)
    args = parser.parse_args()

    torch.set_num_threads(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = find_models(args)
    if not models:
        raise FileNotFoundError(f"No models matched in {args.checkpoint_dir}")

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

    summaries = []
    for model_path in models:
        label = safe_label(model_path)
        print(f"[sweep] evaluating {label}: {model_path}")
        model = PPO.load(str(model_path), device="cpu")
        rows = []
        for i in range(args.episodes):
            row = evaluate_episode(
                model=model,
                seed=args.seed + i,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
                include_lateral_error_observation=args.include_lateral_error_observation,
                lateral_error_observation_clip=args.lateral_error_observation_clip,
                use_controlled_locomotion_wrapper=not args.raw_ant_env,
                controlled_reward_config=controlled_reward_config,
            )
            rows.append(row)
        write_rows(out_dir / f"{label}_episodes.csv", rows)

        summary = build_summary(model_path=model_path, rows=rows, args=args)
        summaries.append(summary)
        print(
            f"[sweep] score={summary['selection_score']:.3f} "
            f"survival={summary['survival_rate']:.2f} "
            f"vx={mean_value(summary, 'mean_forward_velocity'):.2f} "
            f"drift_mean={mean_value(summary, 'abs_lateral_drift'):.2f} "
            f"drift_max={max_value(summary, 'abs_lateral_drift'):.2f} "
            f"heading={mean_value(summary, 'heading_alignment_mean'):.3f} "
            f"course={mean_value(summary, 'course_alignment_mean'):.3f}"
        )

    summaries = sorted(
        summaries,
        key=lambda item: item["selection_score"],
        reverse=True,
    )
    flat_rows = [summary_flat_row(summary) for summary in summaries]
    write_rows(out_dir / "sweep_summary.csv", flat_rows)
    with (out_dir / "sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("[done] best:", summaries[0]["label"])
    print("[done] best model:", summaries[0]["model"])
    print("[done] csv:", out_dir / "sweep_summary.csv")
    print("[done] json:", out_dir / "sweep_summary.json")


if __name__ == "__main__":
    main()
