#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile="${PANGU_SCORE_PROFILE:-pgw_lite_pruned_96}"
source_name="${PANGU_SCORE_SOURCE_CHECKPOINT:-model_fp16_alias_compact.pth}"
prefix="${PANGU_SCORE_CANDIDATE_PREFIX:-score_recovery}"
stage1_steps="${PANGU_SCORE_STAGE1_STEPS:-410}"
stage2_steps="${PANGU_SCORE_STAGE2_STEPS:-1638}"
sensitivity="data/quant_sensitivity.json"
baseline_rmse="${PANGU_SCORE_BASELINE_RMSE:-data/official_baseline_rmse.npy}"

if [[ "$source_name" == */* ]]; then
  echo "PANGU_SCORE_SOURCE_CHECKPOINT must be a filename under data/checkpoints" >&2
  exit 2
fi

required=(
  "data/checkpoints/$source_name"
  "$sensitivity"
  "$baseline_rmse"
  "distill_train.py"
  "score_training_utils.py"
  "scripts/quantize_mixed_precision.py"
  "scripts/compact_fuser_alias_checkpoint.py"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 2
  fi
done

stage1_prefix="${prefix}_stage1"
stage2_prefix="${prefix}_stage2"
quantized="data/checkpoints/${prefix}_keep5.pth"
compact="data/checkpoints/${prefix}_keep5_compact.pth"
outputs=(
  "data/checkpoints/${stage1_prefix}_latest.pth"
  "data/checkpoints/${stage1_prefix}_train.pth"
  "data/checkpoints/${stage1_prefix}_fp16.pth"
  "data/checkpoints/${stage2_prefix}_latest.pth"
  "data/checkpoints/${stage2_prefix}_train.pth"
  "data/checkpoints/${stage2_prefix}_fp16.pth"
  "$quantized"
  "$compact"
)
for path in "${outputs[@]}"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite candidate artifact: $path" >&2
    exit 2
  fi
done

echo "Stage 1/2: scored head + top-5 sensitive recovery ($stage1_steps steps)"
PANGU_STUDENT_PROFILE="$profile" \
PANGU_DISTILL_CHECKPOINT_PREFIX="$stage1_prefix" \
PANGU_DISTILL_INIT_CHECKPOINT="$source_name" \
PANGU_DISTILL_RESUME_FROM=latest \
PANGU_DISTILL_MAX_EPOCH=1 \
PANGU_DISTILL_STEPS_PER_EPOCH="$stage1_steps" \
PANGU_DISTILL_WARMUP_STEPS=40 \
PANGU_DISTILL_LEARNING_RATE=1e-5 \
PANGU_DISTILL_HINT_WEIGHT=0 \
PANGU_SCORE_ALIGNED=1 \
PANGU_SCORE_STAGE=head \
PANGU_SCORE_SENSITIVITY_PATH="$sensitivity" \
PANGU_SCORE_SENSITIVE_COUNT=5 \
PANGU_SCORE_BASELINE_RMSE="$baseline_rmse" \
PANGU_SCORE_PROJECT_QUANTIZED=0 \
python distill_train.py

echo "Stage 2/2: scored whole-model low-LR recovery ($stage2_steps steps)"
PANGU_STUDENT_PROFILE="$profile" \
PANGU_DISTILL_CHECKPOINT_PREFIX="$stage2_prefix" \
PANGU_DISTILL_INIT_CHECKPOINT="${stage1_prefix}_fp16.pth" \
PANGU_DISTILL_RESUME_FROM=latest \
PANGU_DISTILL_MAX_EPOCH=1 \
PANGU_DISTILL_STEPS_PER_EPOCH="$stage2_steps" \
PANGU_DISTILL_WARMUP_STEPS=80 \
PANGU_DISTILL_LEARNING_RATE=2e-6 \
PANGU_DISTILL_HINT_WEIGHT=0 \
PANGU_SCORE_ALIGNED=1 \
PANGU_SCORE_STAGE=all \
PANGU_SCORE_SENSITIVITY_PATH="$sensitivity" \
PANGU_SCORE_SENSITIVE_COUNT=5 \
PANGU_SCORE_BASELINE_RMSE="$baseline_rmse" \
PANGU_SCORE_PROJECT_QUANTIZED=1 \
PANGU_SCORE_PROJECT_INTERVAL=1 \
python distill_train.py

echo "Rebuilding the accepted keep-count=5 storage format"
PANGU_QUANTIZE_PROFILE="$profile" python scripts/quantize_mixed_precision.py \
  --keep-count 5 \
  --checkpoint "data/checkpoints/${stage2_prefix}_fp16.pth" \
  --output "$quantized"

python scripts/compact_fuser_alias_checkpoint.py \
  --source "$quantized" \
  --output "$compact"

echo "Candidate ready: $compact"
echo "Accepted checkpoint and calibration were not modified."
echo "Next: run inference with this candidate, then blocked calibration evaluation."
