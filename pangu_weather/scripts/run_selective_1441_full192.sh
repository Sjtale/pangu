#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

phase="${1:-}"
if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 {archive|prepare|probe|train|pack|probe-packed}" >&2
  exit 2
fi
case "$phase" in
  archive|prepare|probe|train|pack|probe-packed) ;;
  *) echo "Usage: $0 {archive|prepare|probe|train|pack|probe-packed}" >&2; exit 2 ;;
esac

profile="pangu_selective_1441_full192"
prefix="${PANGU_SELECTIVE_PREFIX:-selective_1441_full192_recovery}"
if [[ ! "$prefix" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "PANGU_SELECTIVE_PREFIX must be a safe basename, got: $prefix" >&2
  exit 2
fi

local_full="data/checkpoints/model_bak.pth"
official_full="pangu_backups/model_bak.pth"
source_checkpoint="${PANGU_FULL192_CHECKPOINT:-$official_full}"
if [[ -f "$local_full" && -z "${PANGU_FULL192_CHECKPOINT:-}" ]]; then
  source_checkpoint="$local_full"
fi

init_train="data/checkpoints/${profile}_init_train.pth"
init_fp16="data/checkpoints/${profile}_init_fp16.pth"
latest_train="data/checkpoints/${prefix}_latest.pth"
trained_fp16="data/checkpoints/${prefix}_fp16.pth"
packed_fp16="data/checkpoints/${prefix}_fp16_compact.pth"

if [[ "$phase" == "archive" ]]; then
  python scripts/archive_selective_1441_full192.py --prefix "$prefix"
  exit 0
fi

if [[ "${PANGU_ALLOW_REJECTED_SELECTIVE_1441:-0}" != "1" ]]; then
  echo "$profile is archived/rejected (validation=0.2003)." >&2
  echo "Use '$0 archive' to archive server artifacts." >&2
  echo "Set PANGU_ALLOW_REJECTED_SELECTIVE_1441=1 only for reproduction." >&2
  exit 2
fi

if [[ "$phase" == "prepare" ]]; then
  if [[ ! -f "$source_checkpoint" ]]; then
    echo "Missing official full_192 checkpoint: $source_checkpoint" >&2
    exit 2
  fi
  python scripts/initialize_selective_1441_full192.py \
    --source "$source_checkpoint" \
    --output "$init_train" \
    --inference-output "$init_fp16"
  exit 0
fi

if [[ "$phase" == "probe" ]]; then
  if [[ ! -f "$init_fp16" ]]; then
    echo "Missing $init_fp16; run '$0 prepare' first" >&2
    exit 2
  fi
  python scripts/probe_uv_runtime_sweep.py \
      --preset baseline \
      --buffer-intern 1 \
      --fp16-checkpoint model_fp16_alias_compact.pth \
      --candidate-fp16-checkpoint "$(basename "$init_fp16")" \
      --max-batches 5 \
      --repeat 1 \
      --skip-output-compare \
      --log-file logs/selective_1441_full192_init.jsonl
  exit 0
fi

if [[ "$phase" == "pack" ]]; then
  if [[ ! -f "$trained_fp16" ]]; then
    echo "Missing trained FP16 checkpoint: $trained_fp16" >&2
    exit 2
  fi
  python scripts/compact_fuser_alias_checkpoint.py \
    --source "$trained_fp16" \
    --output "$packed_fp16"
  exit 0
fi

if [[ "$phase" == "probe-packed" ]]; then
  if [[ ! -f "$packed_fp16" ]]; then
    echo "Missing $packed_fp16; run '$0 pack' first" >&2
    exit 2
  fi
  python scripts/probe_uv_runtime_sweep.py \
      --preset baseline \
      --buffer-intern 1 \
      --fp16-checkpoint model_fp16_alias_compact.pth \
      --candidate-fp16-checkpoint "$(basename "$packed_fp16")" \
      --max-batches 5 \
      --repeat 5 \
      --log-file logs/selective_1441_full192_trained_compact.jsonl
  exit 0
fi

if [[ ! -f "$init_train" ]]; then
  echo "Missing $init_train; run '$0 prepare' first" >&2
  exit 2
fi
if [[ ! -e "$latest_train" ]]; then
  for output in \
    "data/checkpoints/${prefix}_train.pth" \
    "$trained_fp16" \
    "$packed_fp16"; do
    if [[ -e "$output" ]]; then
      echo "Refusing fresh run because $output already exists" >&2
      exit 2
    fi
  done
else
  echo "Resuming from $latest_train"
fi

training_env=(
  "PANGU_STUDENT_PROFILE=$profile"
  "PANGU_RECOVERY_ONLY=1"
  "PANGU_DISTILL_CHECKPOINT_PREFIX=$prefix"
  "PANGU_DISTILL_FRESH_OFFICIAL=0"
  "PANGU_DISTILL_INIT_CHECKPOINT=$(basename "$init_train")"
  "PANGU_DISTILL_MAX_EPOCH=4"
  "PANGU_DISTILL_STEPS_PER_EPOCH=2048"
  "PANGU_DISTILL_CHECKPOINT_INTERVAL=256"
  "PANGU_DISTILL_WARMUP_STEPS=256"
  "PANGU_DISTILL_LEARNING_RATE=1e-5"
  "PANGU_DISTILL_MIN_LR_RATIO=0.1"
  "PANGU_DISTILL_GROUND_TRUTH_WEIGHT=1.0"
  "PANGU_DISTILL_TEACHER_WEIGHT=0.0"
  "PANGU_DISTILL_HINT_WEIGHT=0.0"
  "PANGU_DISTILL_HINT_LAYERS="
  "PANGU_DISTILL_GRADIENT_ACCUMULATION=1"
  "PANGU_DISTILL_REQUIRE_PROTOCOL_MATCH=1"
  "PANGU_DISTILL_DISABLE_EARLY_STOPPING=1"
  "PANGU_DISTILL_EXTRA_EPOCHS=0"
  "PANGU_SCORE_ALIGNED=0"
  "PANGU_SCORE_PROJECT_QUANTIZED=0"
  "PANGU_SCORED_ONLY_RECOVERY=0"
  "PANGU_USE_SWIGLU=0"
  "PANGU_USE_RMSNORM=0"
  "PANGU_USE_GQA=0"
  "PANGU_SHARE_DEEP_BLOCKS=0"
  "PANGU_COMPACT_ATTN_MASK=0"
)

env "${training_env[@]}" python distill_train.py
