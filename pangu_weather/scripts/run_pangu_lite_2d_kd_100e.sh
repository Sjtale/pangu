#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

phase="${1:-preflight}"
teacher="${PANGU_2D_TEACHER:-./pangu_backups/model_bak.pth}"
baseline="${PANGU_2D_BASELINE_RMSE:-./data/official_baseline_rmse.npy}"
initial="${PANGU_2D_INITIAL:-./data/checkpoints/model_pangu_lite_2d_pos288_hybrid.pth}"
prefix="${PANGU_2D_PREFIX:-pangu_lite_2d_pos288_kd100}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

prepare() {
  require_file "$teacher"
  python scripts/initialize_pangu_lite_2d.py --teacher "$teacher" --output "$initial"
}

export PANGU_STUDENT_PROFILE=pangu_lite_2d_pos288
export PANGU_DISTILL_INIT_CHECKPOINT="$initial"
export PANGU_DISTILL_CHECKPOINT_PREFIX="$prefix"
export PANGU_DISTILL_MAX_EPOCH=100
export PANGU_DISTILL_STEPS_PER_EPOCH=2048
export PANGU_DISTILL_GRADIENT_ACCUMULATION=4
export PANGU_DISTILL_WARMUP_STEPS=1024
export PANGU_DISTILL_MIN_LR_RATIO=0.01
export PANGU_DISTILL_LEARNING_RATE=5e-5
export PANGU_DISTILL_HINT_WEIGHT=0
export PANGU_DISTILL_HINT_LAYERS=""
export PANGU_SCORE_ALIGNED=1
export PANGU_SCORE_BASELINE_RMSE="$baseline"
export PANGU_SCORE_LOSS_WEIGHTS=0.55,0.30,0.10,0.05
export PANGU_SCORE_STAGE=all
export PANGU_DISTILL_DISABLE_EARLY_STOPPING=1
export PANGU_DISTILL_CHECKPOINT_INTERVAL=256
export PANGU_DISTILL_REQUIRE_PROTOCOL_MATCH=1
export WORLD_SIZE=1
export LOCAL_RANK=0

preflight() {
  require_file "$teacher"
  require_file "$baseline"
  require_file "$initial"
  python - <<'PY'
import os
import numpy as np
import torch
from pangu_lite_2d import PanguLite2DAttentionPosEmbed

checkpoint = torch.load(os.environ["PANGU_DISTILL_INIT_CHECKPOINT"], map_location="cpu", weights_only=False)
model = PanguLite2DAttentionPosEmbed()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
assert tuple(model.absolute_pos_embed.shape) == (1, 91, 288)
assert not any("earth_position_bias" in name for name, _ in model.named_parameters())
baseline = np.load(os.environ["PANGU_SCORE_BASELINE_RMSE"]).reshape(-1)
assert baseline.size in (15, 69) and np.isfinite(baseline).all() and (baseline > 0).all()
assert torch.cuda.is_available(), "single DCU is not visible through torch.cuda"
assert torch.cuda.device_count() == 1, "exactly one DCU must be visible"
print("2D KD preflight passed")
PY
}

case "$phase" in
  prepare)
    prepare
    ;;
  preflight)
    preflight
    ;;
  train)
    preflight
    python distill_train.py
    ;;
  launch)
    preflight
    mkdir -p logs
    log="logs/${prefix}_$(date +%Y%m%d_%H%M%S).log"
    screen -dmS "$prefix" bash -lc "cd '$PWD'; source ../earth_env.sh; scripts/run_pangu_lite_2d_kd_100e.sh train >> '$log' 2>&1"
    printf '%s\n' "$log" > "logs/${prefix}.logpath"
    screen -list
    ;;
  status)
    screen -list || true
    if [[ -f "logs/${prefix}.logpath" ]]; then
      tail -n 40 "$(cat "logs/${prefix}.logpath")"
    fi
    ;;
  *)
    echo "Usage: $0 {prepare|preflight|train|launch|status}" >&2
    exit 2
    ;;
esac
