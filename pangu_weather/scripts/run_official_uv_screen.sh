#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

candidate="${1:-}"
phase="${2:-}"
epochs="${3:-1}"
if [[ "$#" -gt 3 ]]; then
  echo "Usage: $0 {A|S96|B|E|C|D} {prepare|probe|train|pack|probe-packed} [epochs]" >&2
  exit 2
fi
case "$candidate" in
  A) profile="uv_a_patch8_w80_shallow"; candidate_slug="a" ;;
  S96) profile="uv_s96_patch8_w96_shallow"; candidate_slug="s96" ;;
  B) profile="uv_b_patch8_w64_shallow"; candidate_slug="b" ;;
  E) profile="uv_e_patch8_w80_ultrashallow"; candidate_slug="e" ;;
  C) profile="uv_c_patch12_w80_shallow"; candidate_slug="c" ;;
  D) profile="uv_d_patch16_w80_shallow"; candidate_slug="d" ;;
  *) echo "Usage: $0 {A|S96|B|E|C|D} {prepare|probe|train|pack|probe-packed} [epochs]" >&2; exit 2 ;;
esac
case "$phase" in
  prepare|probe|train|pack|probe-packed) ;;
  *) echo "Usage: $0 {A|S96|B|E|C|D} {prepare|probe|train|pack|probe-packed} [epochs]" >&2; exit 2 ;;
esac
if [[ "$phase" == "train" && ! "$epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "epochs must be a positive integer, got: $epochs" >&2
  exit 2
fi

if [[ "$candidate" == "S96" ]]; then
  screen_checkpoint="data/checkpoints/${profile}_pgw96_exact_init_fp16.pth"
elif [[ "$candidate" == "A" ]]; then
  screen_checkpoint="data/checkpoints/${profile}_pgw96_structured_init_fp16.pth"
else
  screen_checkpoint="data/checkpoints/${profile}_official_init_fp16.pth"
fi
prefix="${PANGU_UV_SCREEN_PREFIX:-uv_screen_${candidate_slug}}"
baseline_rmse="${PANGU_SCORE_BASELINE_RMSE:-data/official_baseline_rmse.npy}"
trained_fp16="data/checkpoints/${prefix}_fp16.pth"
packed_fp16="data/checkpoints/${prefix}_fp16_compact.pth"
if [[ ! "$prefix" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "PANGU_UV_SCREEN_PREFIX must be a safe basename, got: $prefix" >&2
  exit 2
fi

if [[ "$phase" == "prepare" ]]; then
  if [[ "$candidate" == "S96" || "$candidate" == "A" ]]; then
    s96_source="data/checkpoints/model_pgw_lite_pruned_96_fp16.pth"
    if [[ ! -f "$s96_source" ]]; then
      echo "Missing PGW-Lite Width-96 source checkpoint: $s96_source" >&2
      exit 2
    fi
    prune_args=(
      --source "$s96_source"
      --output "$screen_checkpoint"
      --target-profile "$profile"
      --dtype fp16
      --require-unquantized-source
    )
    if [[ "$candidate" == "S96" ]]; then
      prune_args+=(--strict-exact-depth)
    fi
    python scripts/prune_structured.py "${prune_args[@]}"
  else
    python scripts/prepare_official_uv_student.py \
      --profile "$profile" \
      --output "$screen_checkpoint"
  fi
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
    echo "Missing packed checkpoint: $packed_fp16; run '$0 $candidate pack' first" >&2
    exit 2
  fi
  python scripts/probe_uv_runtime_sweep.py \
    --preset baseline \
    --fp16-checkpoint model_fp16_alias_compact.pth \
    --candidate-fp16-checkpoint "$(basename "$packed_fp16")" \
    --max-batches 4 \
    --repeat 5 \
    --log-file "logs/uv_arch_${candidate_slug}_trained_compact.jsonl"
  exit 0
fi

if [[ ! -f "$screen_checkpoint" ]]; then
  echo "Missing $screen_checkpoint; run '$0 $candidate prepare' first" >&2
  exit 2
fi

if [[ "$phase" == "probe" ]]; then
  python scripts/probe_uv_runtime_sweep.py \
    --preset baseline \
    --fp16-checkpoint model_fp16_alias_compact.pth \
    --candidate-fp16-checkpoint "$(basename "$screen_checkpoint")" \
    --max-batches 4 \
    --repeat 5 \
    --skip-output-compare \
    --log-file "logs/uv_arch_${candidate_slug}.jsonl"
  exit 0
fi

if [[ "$candidate" != "S96" && ! -f "$baseline_rmse" ]]; then
  echo "Missing official baseline RMSE: $baseline_rmse" >&2
  exit 2
fi
latest_output="data/checkpoints/${prefix}_latest.pth"
fresh_official=1
if [[ -e "$latest_output" ]]; then
  fresh_official=0
  echo "Resuming from $latest_output"
else
  for suffix in train fp16; do
    output="data/checkpoints/${prefix}_${suffix}.pth"
    if [[ -e "$output" ]]; then
      echo "Refusing fresh run because $output already exists" >&2
      exit 2
    fi
  done
fi

training_env=(
  "PANGU_STUDENT_PROFILE=$profile"
  "PANGU_DISTILL_CHECKPOINT_PREFIX=$prefix"
  "PANGU_DISTILL_MAX_EPOCH=$epochs"
  "PANGU_DISTILL_STEPS_PER_EPOCH=2048"
  "PANGU_DISTILL_CHECKPOINT_INTERVAL=256"
  "PANGU_DISTILL_WARMUP_STEPS=256"
  "PANGU_DISTILL_LEARNING_RATE=1e-5"
  "PANGU_DISTILL_REQUIRE_PROTOCOL_MATCH=1"
  "PANGU_DISTILL_STAGNATION_WARN_EPOCHS=2"
  "PANGU_DISTILL_DISABLE_EARLY_STOPPING=1"
  "PANGU_DISTILL_HINT_WEIGHT=0"
  "PANGU_SCORE_PROJECT_QUANTIZED=0"
  "PANGU_USE_SWIGLU=0"
  "PANGU_USE_RMSNORM=0"
  "PANGU_USE_GQA=0"
  "PANGU_SHARE_DEEP_BLOCKS=0"
)

if [[ "$candidate" == "S96" ]]; then
  training_env+=(
    "PANGU_DISTILL_FRESH_OFFICIAL=0"
    "PANGU_DISTILL_INIT_CHECKPOINT=$(basename "$screen_checkpoint")"
    "PANGU_DISTILL_GROUND_TRUTH_WEIGHT=0.5"
    "PANGU_DISTILL_TEACHER_WEIGHT=0.5"
    "PANGU_DISTILL_HINT_LAYERS="
    "PANGU_SCORE_ALIGNED=0"
  )
elif [[ "$candidate" == "A" ]]; then
  training_env+=(
    "PANGU_DISTILL_FRESH_OFFICIAL=0"
    "PANGU_DISTILL_INIT_CHECKPOINT=$(basename "$screen_checkpoint")"
    "PANGU_SCORE_ALIGNED=1"
    "PANGU_SCORE_STAGE=all"
    "PANGU_SCORE_LOSS_WEIGHTS=0.45,0.20,0.25,0.10"
    "PANGU_SCORE_BASELINE_RMSE=$baseline_rmse"
  )
else
  training_env+=(
    "PANGU_DISTILL_FRESH_OFFICIAL=$fresh_official"
    "PANGU_SCORE_ALIGNED=1"
    "PANGU_SCORE_STAGE=all"
    "PANGU_SCORE_LOSS_WEIGHTS=0.45,0.20,0.25,0.10"
    "PANGU_SCORE_BASELINE_RMSE=$baseline_rmse"
  )
fi

env "${training_env[@]}" python distill_train.py
