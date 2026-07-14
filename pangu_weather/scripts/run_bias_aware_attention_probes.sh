#!/usr/bin/env bash
set -u

stamp="${PANGU_ATTN_PROBE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
output_dir="${PANGU_ATTN_PROBE_DIR:-logs/bias_aware_attention_${stamp}}"
onescience_root="${PANGU_ONESCIENCE_ROOT:-../../onescience}"
python_bin="${PANGU_PYTHON:-python}"
jax_python="${PANGU_JAX_PYTHON:-$python_bin}"

mkdir -p "$output_dir"

run_probe() {
  interpreter="$1"
  backend="$2"
  output="$3"
  shift 3
  echo "[probe] python=$interpreter backend=$backend output=$output"
  "$interpreter" scripts/probe_bias_aware_attention_backends.py \
    --backend "$backend" \
    --onescience-root "$onescience_root" \
    --output "$output" \
    "$@"
}

# Keep compiler/framework failures isolated. A failed row must not prevent the
# next independent compatibility route from producing evidence.
run_probe \
  "$python_bin" \
  pytorch-triton \
  "$output_dir/pytorch_triton_l144.json" || true
if [[ "${PANGU_RETRY_CRASHED_FLEX:-0}" == "1" ]]; then
  ulimit -c 0
  run_probe "$python_bin" flex "$output_dir/flex_l144.json" || true
else
  echo "[probe] skip flex: torch.compile segfault already reproduced on this DCU stack"
fi
if ! command -v "$jax_python" >/dev/null 2>&1; then
  echo "[probe] skip OneScience Triton: PANGU_JAX_PYTHON is not executable: $jax_python"
elif ! "$jax_python" -c 'import jax, jax_triton' >/dev/null 2>&1; then
  echo "[probe] skip OneScience Triton: interpreter lacks jax or jax_triton: $jax_python"
else
  run_probe \
    "$jax_python" \
    onescience-triton \
    "$output_dir/onescience_triton_l144_l192.json" || true
fi
run_probe \
  "$python_bin" \
  transformer-engine \
  "$output_dir/transformer_engine_l144.json" \
  2>"$output_dir/transformer_engine_debug.log" || true

echo "[probe] reports=$output_dir"
