#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

action="${1:-}"
source_checkpoint="${PANGU_SELECTIVE_MLP96_SOURCE:-data/checkpoints/model_pgw_lite_pruned_96_fp16.pth}"
teacher_checkpoint="${PANGU_SELECTIVE_MLP96_TEACHER:-pangu_backups/model_bak.pth}"
init_checkpoint="${PANGU_SELECTIVE_MLP96_INIT:-data/checkpoints/selective_mlp96_init.pth}"
recovery_prefix="${PANGU_SELECTIVE_MLP96_RECOVERY_PREFIX:-selective_mlp96_recovery}"
teacher_prefix="${PANGU_SELECTIVE_MLP96_TEACHER_PREFIX:-selective_mlp96_fullteacher}"
recovery_train="data/checkpoints/${recovery_prefix}_train.pth"

usage() {
  echo "Usage: $0 prepare|recover|distill|runtime|audit|validate-metrics [args...]" >&2
  exit 2
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

run_stage() {
  local stage="$1"
  local input_checkpoint="$2"
  local prefix="$3"
  local epochs="$4"
  local steps="$5"
  local learning_rate="$6"
  local warmup="$7"

  require_file "$input_checkpoint"
  env \
    PANGU_STUDENT_PROFILE=selective_mlp96 \
    PANGU_SELECTIVE_MLP96_STAGE="$stage" \
    PANGU_SELECTIVE_MLP96_SOURCE_CHECKPOINT="$source_checkpoint" \
    PANGU_DISTILL_INIT_CHECKPOINT="$input_checkpoint" \
    PANGU_DISTILL_CHECKPOINT_PREFIX="$prefix" \
    PANGU_DISTILL_MAX_EPOCH="$epochs" \
    PANGU_DISTILL_STEPS_PER_EPOCH="$steps" \
    PANGU_DISTILL_LEARNING_RATE="$learning_rate" \
    PANGU_DISTILL_WARMUP_STEPS="$warmup" \
    PANGU_DISTILL_MIN_LR_RATIO=0.1 \
    PANGU_DISTILL_GROUND_TRUTH_WEIGHT=0 \
    PANGU_DISTILL_TEACHER_WEIGHT=1 \
    PANGU_DISTILL_HINT_WEIGHT=0 \
    PANGU_DISTILL_HINT_LAYERS= \
    PANGU_DISTILL_GRADIENT_ACCUMULATION=1 \
    PANGU_DISTILL_CHECKPOINT_INTERVAL=256 \
    PANGU_DISTILL_REQUIRE_PROTOCOL_MATCH=1 \
    PANGU_DISTILL_DISABLE_EARLY_STOPPING=1 \
    PANGU_DISTILL_EXTRA_EPOCHS=0 \
    PANGU_DISTILL_FRESH_OFFICIAL=0 \
    PANGU_SCORE_ALIGNED=0 \
    PANGU_SCORE_PROJECT_QUANTIZED=0 \
    PANGU_RECOVERY_ONLY=0 \
    PANGU_SCORED_ONLY_RECOVERY=0 \
    PANGU_SHARE_DEEP_BLOCKS=0 \
    PANGU_USE_SWIGLU=0 \
    PANGU_USE_RMSNORM=0 \
    PANGU_USE_GQA=0 \
    python distill_train.py
}

case "$action" in
  prepare)
    require_file "$source_checkpoint"
    require_file "$teacher_checkpoint"
    python scripts/initialize_selective_mlp96.py \
      --source "$source_checkpoint" \
      --teacher "$teacher_checkpoint" \
      --output "$init_checkpoint"
    ;;
  recover)
    require_file "$source_checkpoint"
    run_stage source_recovery "$init_checkpoint" "$recovery_prefix" 1 512 2e-5 64
    ;;
  distill)
    run_stage full_teacher "$recovery_train" "$teacher_prefix" 3 1024 5e-6 128
    ;;
  runtime)
    baseline="${2:-data/checkpoints/model_fp16_alias_compact.pth}"
    candidate="${3:-data/checkpoints/${teacher_prefix}_step3072_fp16.pth}"
    runtime_log="${PANGU_SELECTIVE_MLP96_RUNTIME_LOG:-logs/selective_mlp96_runtime_$(date +%Y%m%d_%H%M%S).jsonl}"
    runtime_report="${runtime_log%.jsonl}_gate.json"
    require_file "$baseline"
    require_file "$candidate"
    if [[ -e "$runtime_log" || -e "$runtime_report" ]]; then
      echo "Refusing to overwrite SelectiveMLP-96 runtime artifact" >&2
      exit 2
    fi
    PANGU_COMPLIANT_FULL69_BOUNDARY=1 \
      python scripts/probe_uv_runtime_sweep.py \
        --preset baseline \
        --repeat 5 \
        --max-batches 5 \
        --fp16-checkpoint "$baseline" \
        --candidate-fp16-checkpoint "$candidate" \
        --buffer-intern 1 \
        --log-file "$runtime_log"
    python scripts/validate_selective_mlp96_runtime.py \
      --log "$runtime_log" \
      --checkpoint "$candidate" \
      --output "$runtime_report"
    ;;
  audit)
    checkpoint="${2:-data/checkpoints/${teacher_prefix}_step3072_fp16.pth}"
    require_file "$checkpoint"
    python scripts/audit_selective_mlp96_checkpoint.py "$checkpoint"
    ;;
  validate-metrics)
    shift
    python blocked_w_validator.py "$@"
    ;;
  *)
    usage
    ;;
esac
