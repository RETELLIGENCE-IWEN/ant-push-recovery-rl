#!/usr/bin/env bash
# Evaluate v3g final models for all 3 seeds against well-trained criteria.

set -euo pipefail

SEEDS=(42 123 7)

for s in "${SEEDS[@]}"; do
  MODEL="runs/well_trained_v3g_seed${s}_1500k/models/final_model.zip"
  OUT="reports/metrics_well_trained_v3g_seed${s}_clean"
  if [[ ! -f "$MODEL" ]]; then
    echo "[skip] model missing: $MODEL"
    continue
  fi
  echo "=== eval v3g seed=${s} ==="
  python scripts/eval_policy_metrics.py \
    --model "$MODEL" \
    --out-dir "$OUT" \
    --episodes 10 \
    --seed 1000 \
    --deterministic \
    --use-well-trained-wrapper
done

echo ""
echo "=== summary across seeds ==="
python - <<'PY'
import json
from pathlib import Path

ABS_CRITERIA = {
    "survival_rate":           (">=", 1.00),
    "vx_body_err_rms":         ("<=", 0.15),
    "vy_body_mean_abs":        ("<=", 0.15),
    "yaw_rate_rms":            ("<=", 0.20),
    "abs_lateral_drift":       ("<=", 2.0),
    "yaw_abs_mean":            ("<=", 0.15),
    "course_alignment_mean":   (">=", 0.97),
    "roll_rms":                ("<=", 0.08),
    "pitch_rms":               ("<=", 0.08),
    "height_std":              ("<=", 0.035),
    "vertical_velocity_rms":   ("<=", 0.40),
    "action_delta_rms":        ("<=", 0.60),
    "mean_action_energy":      ("<=", 0.50),
}

def get(s, k):
    if k == "survival_rate":
        return s.get(k)
    v = s.get(k)
    return v.get("mean") if isinstance(v, dict) else v

rows = []
for seed in [42, 123, 7]:
    p = Path(f"reports/metrics_well_trained_v3g_seed{seed}_clean/summary.json")
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    row = {"seed": seed}
    pass_count = 0
    for k, (op, t) in ABS_CRITERIA.items():
        v = get(s, k)
        if v is None:
            ok = "?"
        else:
            ok = "PASS" if ((op == ">=" and v >= t) or (op == "<=" and v <= t)) else "FAIL"
            if ok == "PASS":
                pass_count += 1
        row[k] = (v, ok)
    row["passed"] = f"{pass_count}/{len(ABS_CRITERIA)}"
    rows.append(row)

if not rows:
    print("[error] no eval results found")
    raise SystemExit(1)

hdr = ["seed"] + list(ABS_CRITERIA.keys()) + ["passed"]
print("\t".join(hdr))
for row in rows:
    cells = [str(row["seed"])]
    for k in ABS_CRITERIA:
        v, ok = row[k]
        cells.append(f"{v:.3f}[{ok}]" if isinstance(v, (int, float)) else f"NA[{ok}]")
    cells.append(row["passed"])
    print("\t".join(cells))
PY
