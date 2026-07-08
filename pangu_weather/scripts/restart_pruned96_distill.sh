#!/usr/bin/env bash
set -euo pipefail

# Restart pgw_lite_pruned_96 distillation from a floating structural checkpoint.
#
# Run from pangu_weather or anywhere inside this repository:
#   bash scripts/restart_pruned96_distill.sh smoke
#   bash scripts/restart_pruned96_distill.sh full
#   bash scripts/restart_pruned96_distill.sh smoke-full
#   bash scripts/restart_pruned96_distill.sh quant-smoke

usage() {
  cat <<'EOF'
Usage: bash scripts/restart_pruned96_distill.sh [mode]

Modes:
  check            Verify that the floating structural init checkpoint exists and is not quantized.
  smoke            Backup resume artifacts, then run 1 epoch x 20 steps from floating init.
  full             Backup resume artifacts, then run 30 epochs x 512 steps from floating init.
  smoke-full       Run smoke, backup smoke outputs, then run the full floating-init schedule.
  quant-check      Verify quantized recovery init at data/checkpoints/model_fp16.pth.
  quant-smoke      Run 1 epoch x 20 steps from quantized model_fp16.pth recovery init.
  quant-full       Run 30 epochs x 512 steps from quantized model_fp16.pth recovery init.
  quant-smoke-full Run quant-smoke, backup outputs, then run quant-full.

Environment overrides:
  PANGU_RESTART_PROFILE=pgw_lite_pruned_96
  PANGU_RESTART_INIT_CHECKPOINT=model_pgw_lite_pruned_96.pth
  PANGU_RESTART_GROUND_TRUTH_WEIGHT=0.3
  PANGU_RESTART_TEACHER_WEIGHT=0.5
  PANGU_RESTART_HINT_WEIGHT=0
  PANGU_RESTART_SMOKE_STEPS=20
  PANGU_RESTART_FULL_STEPS=512
  PANGU_RESTART_FULL_EPOCHS=30
  PANGU_RESTART_ALLOW_QUANTIZED_INIT=0

By default, this script refuses INT8/scale-bearing checkpoints. That protects
the "structural restart" path from accidentally using a quantized submission
checkpoint as init. Use quant-* modes for intentional quantized-checkpoint
recovery distillation from data/checkpoints/model_fp16.pth.
EOF
}

mode="${1:-smoke}"
case "$mode" in
  check|smoke|full|smoke-full|quant-check|quant-smoke|quant-full|quant-smoke-full) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
is_quant_mode=0
if [[ "$mode" == quant-* ]]; then
  is_quant_mode=1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pangu_dir="$(cd "${script_dir}/.." && pwd)"
cd "$pangu_dir"

profile="${PANGU_RESTART_PROFILE:-pgw_lite_pruned_96}"
init_checkpoint="${PANGU_RESTART_INIT_CHECKPOINT:-model_pgw_lite_pruned_96.pth}"
checkpoint_dir="${PANGU_RESTART_CHECKPOINT_DIR:-data/checkpoints}"
ground_truth_weight="${PANGU_RESTART_GROUND_TRUTH_WEIGHT:-0.3}"
teacher_weight="${PANGU_RESTART_TEACHER_WEIGHT:-0.5}"
hint_weight="${PANGU_RESTART_HINT_WEIGHT:-0}"
smoke_steps="${PANGU_RESTART_SMOKE_STEPS:-20}"
full_steps="${PANGU_RESTART_FULL_STEPS:-512}"
full_epochs="${PANGU_RESTART_FULL_EPOCHS:-30}"
allow_quantized_init="${PANGU_RESTART_ALLOW_QUANTIZED_INIT:-0}"

if [[ "$is_quant_mode" == "1" ]]; then
  init_checkpoint="${PANGU_RESTART_INIT_CHECKPOINT:-model_fp16.pth}"
  allow_quantized_init="${PANGU_RESTART_ALLOW_QUANTIZED_INIT:-1}"
fi

if [[ "$init_checkpoint" = /* ]]; then
  init_path="$init_checkpoint"
  init_for_distill="$init_checkpoint"
else
  init_path="${checkpoint_dir}/${init_checkpoint}"
  init_for_distill="$init_checkpoint"
fi

if [[ ! -f "$init_path" ]]; then
  cat >&2 <<EOF
Missing floating structural init checkpoint:
  ${init_path}

Generate or restore it first, for example with scripts/prune_structured.py,
then rerun this script.
EOF
  exit 1
fi

checkpoint_quant_info="$(
  INIT_PATH="$init_path" python -c '
import os
import torch

path = os.environ["INIT_PATH"]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state = checkpoint.get("model_state_dict", checkpoint)
int8_count = sum(1 for value in state.values() if torch.is_tensor(value) and value.dtype == torch.int8)
scale_count = sum(1 for key in state if str(key).endswith("_scale"))
print(f"{int8_count} {scale_count}")
'
)"
read -r int8_tensors scale_keys <<<"$checkpoint_quant_info"
if [[ "$is_quant_mode" == "1" && "${int8_tensors}" == "0" ]]; then
  cat >&2 <<EOF
Quantized recovery mode expected INT8 tensors, but none were found:
  ${init_path}

Use smoke/full modes for floating structural-init distillation, or point
PANGU_RESTART_INIT_CHECKPOINT at the quantized checkpoint you want to recover.
EOF
  exit 1
fi
if [[ "${int8_tensors}" != "0" && "$allow_quantized_init" != "1" ]]; then
  cat >&2 <<EOF
Refusing quantized init for structural restart:
  ${init_path}

Detected INT8 tensors: ${int8_tensors}
Detected *_scale keys: ${scale_keys}

This script is for a clean structural-init distillation restart. Use a floating
FP16/FP32 structural checkpoint, or explicitly set:
  PANGU_RESTART_ALLOW_QUANTIZED_INIT=1
if you intentionally want quantized-checkpoint recovery distillation.
EOF
  exit 1
fi

echo "[restart-distill] pangu_dir=${pangu_dir}"
echo "[restart-distill] mode=${mode}"
echo "[restart-distill] profile=${profile}"
echo "[restart-distill] init=${init_path}"
echo "[restart-distill] init_int8_tensors=${int8_tensors}, init_scale_keys=${scale_keys}"
echo "[restart-distill] loss_weights=(hard=${ground_truth_weight}, teacher=${teacher_weight}, hint=${hint_weight})"

if [[ "$mode" == "check" || "$mode" == "quant-check" ]]; then
  echo "[restart-distill] check passed"
  exit 0
fi

backup_resume_artifacts() {
  local tag="$1"
  local backup_dir="${checkpoint_dir}/restart_backup_${profile}_${tag}_$(date +%Y%m%d_%H%M%S)"
  local moved=0

  mkdir -p "$backup_dir"
  for path in \
    "${checkpoint_dir}/model_${profile}_latest.pth" \
    "${checkpoint_dir}/model_${profile}_train.pth" \
    "${checkpoint_dir}/model_${profile}_fp16.pth"
  do
    if [[ -e "$path" ]]; then
      mv "$path" "$backup_dir/"
      echo "[restart-distill] backed up ${path} -> ${backup_dir}/"
      moved=1
    fi
  done

  if [[ "$moved" -eq 0 ]]; then
    rmdir "$backup_dir"
    echo "[restart-distill] no ${profile} resume artifacts to back up"
  else
    echo "[restart-distill] backup_dir=${backup_dir}"
  fi
}

latest_log() {
  ls -t logs/distill_train_*.log 2>/dev/null | head -n 1
}

assert_log_contains() {
  local log_path="$1"
  local needle="$2"
  if ! grep -F "$needle" "$log_path" >/dev/null; then
    echo "[restart-distill] expected log line not found: ${needle}" >&2
    echo "[restart-distill] log: ${log_path}" >&2
    exit 1
  fi
}

validate_run_log() {
  local log_path
  local expected_init
  local expected_loss
  log_path="$(latest_log)"
  if [[ -z "$log_path" ]]; then
    echo "[restart-distill] no distill_train log found" >&2
    exit 1
  fi

  if [[ "$init_for_distill" = /* ]]; then
    expected_init="init=${init_for_distill}"
  else
    expected_init="init=./${checkpoint_dir}/${init_for_distill}"
  fi
  expected_loss="$(printf 'loss_weights=(hard=%.2f, teacher=%.2f, hint=%.2f)' \
    "$ground_truth_weight" "$teacher_weight" "$hint_weight")"

  assert_log_contains "$log_path" "student_profile=${profile}"
  assert_log_contains "$log_path" "$expected_init"
  assert_log_contains "$log_path" "$expected_loss"
  assert_log_contains "$log_path" "Epoch schedule: start_epoch=0"

  echo "[restart-distill] acceptance log checks passed: ${log_path}"
  grep -E "Distillation starts|Epoch schedule|Train [0-9]+-1/|Epoch [0-9]+:" "$log_path" | tail -n 20
}

run_distill() {
  local steps="$1"
  local epochs="$2"
  local label="$3"

  echo "[restart-distill] starting ${label}: epochs=${epochs}, steps_per_epoch=${steps}"
  HDF5_USE_FILE_LOCKING=FALSE \
  PANGU_STUDENT_PROFILE="$profile" \
  PANGU_DISTILL_INIT_CHECKPOINT="$init_for_distill" \
  PANGU_DISTILL_GROUND_TRUTH_WEIGHT="$ground_truth_weight" \
  PANGU_DISTILL_TEACHER_WEIGHT="$teacher_weight" \
  PANGU_DISTILL_HINT_WEIGHT="$hint_weight" \
  PANGU_DISTILL_STEPS_PER_EPOCH="$steps" \
  PANGU_DISTILL_MAX_EPOCH="$epochs" \
  python -u distill_train.py

  validate_run_log
}

if [[ "$mode" == "smoke" || "$mode" == "quant-smoke" ]]; then
  backup_resume_artifacts "smoke"
  run_distill "$smoke_steps" 1 "smoke"
elif [[ "$mode" == "full" || "$mode" == "quant-full" ]]; then
  backup_resume_artifacts "full"
  run_distill "$full_steps" "$full_epochs" "full"
else
  backup_resume_artifacts "smoke"
  run_distill "$smoke_steps" 1 "smoke"
  backup_resume_artifacts "full"
  run_distill "$full_steps" "$full_epochs" "full"
fi
