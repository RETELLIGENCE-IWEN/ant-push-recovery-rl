#!/usr/bin/env bash
set -euo pipefail
for s in 42 123 7; do
  MODEL="runs/well_trained_v3h_s1_seed${s}_500k/models/final_model.zip"
  OUT="reports/metrics_v3h_s1_seed${s}_clean"
  python scripts/eval_policy_metrics.py \
    --model "$MODEL" --out-dir "$OUT" \
    --episodes 5 --seed 1000 --deterministic --use-well-trained-wrapper 2>&1 | grep '\[eval-metrics\]'
done
