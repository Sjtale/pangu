#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"
checkpoint="${PANGU_DIAG_CHECKPOINT:-model_fp16.pth}"
repeat="${PANGU_DIAG_REPEAT:-5}"
max_batches="${PANGU_DIAG_MAX_BATCHES:-4}"

case "$mode" in
  static|runtime|report|all) ;;
  *)
    echo "Usage: bash scripts/run_pruned96_uv_diagnosis.sh [static|runtime|report|all]" >&2
    exit 2
    ;;
esac

checkpoint_path="$checkpoint"
if [[ "$checkpoint" != /* ]]; then
  checkpoint_path="data/checkpoints/$checkpoint"
fi

run_static() {
  python scripts/audit_pruned96_uv.py static \
    --checkpoint "$checkpoint_path" \
    --output logs/pruned96_static_audit.json
}

run_runtime() {
  run_static

  PANGU_PROFILE_MEMORY=1 \
  PANGU_SCORED_ONLY_RECOVERY=0 \
  PANGU_DIRECT_RECOVERY_WIDTH_CHUNK=16 \
    python scripts/probe_uv_runtime_sweep.py \
      --preset baseline \
      --fp16-checkpoint "$checkpoint" \
      --repeat "$repeat" \
      --max-batches "$max_batches" \
      --log-file logs/pruned96_runtime.jsonl

  PANGU_PROFILE_MEMORY=1 \
  PANGU_SCORED_ONLY_RECOVERY=0 \
  PANGU_DIRECT_RECOVERY_WIDTH_CHUNK=16 \
    python module_test_scripts/probe_vram_breakdown.py \
      --checkpoint "$checkpoint" \
      --output-json logs/pruned96_vram_breakdown.json
}

run_report() {
  python scripts/audit_pruned96_uv.py report \
    --static-json logs/pruned96_static_audit.json \
    --vram-json logs/pruned96_vram_breakdown.json \
    --runtime-jsonl logs/pruned96_runtime.jsonl \
    --output logs/pruned96_uv_bottleneck_report.md
}

case "$mode" in
  static) run_static ;;
  runtime) run_runtime ;;
  report) run_report ;;
  all)
    run_runtime
    run_report
    ;;
esac
