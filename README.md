# Xiandao Cup AI4S — Pangu-Weather Compression

*English · [简体中文](README.zh-CN.md)*

Model compression and inference acceleration for the Pangu-Weather forecasting
model, built for the Xiandao Cup AI4S track. Structured pruning, full-69-channel
knowledge distillation, mixed-precision quantization, and DCU/HIP kernel
optimization cut both the checkpoint size and the per-forecast latency, while
preserving the timing boundary of the organizers' original `inference.py`.

## Layout

```
.
├── AI4S_ERA5NetCDF_to_HDF5.py   # ERA5 NetCDF → training HDF5 conversion
├── env.sh / earth_env.sh        # dataset & model paths, DTK/Conda environment
├── export_DDP_vars.sh           # distributed training environment variables
├── onedatasets/                 # ERA5 data and normalization statistics (large files untracked)
├── onemodels/                   # OneScience model directory
└── pangu_weather/               # main project: training, distillation, quantization, inference
```

Main entry points under `pangu_weather/`:

| Path | Purpose |
| --- | --- |
| `train.py` | Full-model baseline training |
| `scripts/prune_structured.py` | Derive the `pruned_96` student initialization from Pangu weights |
| `distill_train.py` | Full-69-channel knowledge distillation |
| `scripts/quantize_mixed_precision.py` | Mixed-precision quantization (submission uses `--keep-count 67`) |
| `scripts/compact_fuser_alias_checkpoint.py` | Losslessly drop duplicated OneFuser alias weights |
| `inference.py` | Inference entry point, preserving the template timing boundary |
| `result.py` | Compute RMSE / ACC from inference output |
| `scripts/build_submission.sh` | Build and audit the submission package |

## Quick start

```bash
source env.sh
cd pangu_weather

# Smoke test: load the final checkpoint and run a single batch
HDF5_USE_FILE_LOCKING=FALSE \
PANGU_CHECKPOINT=model_fp16_alias_compact.pth \
PANGU_MAX_INFERENCE_BATCHES=1 \
python inference.py

# Full inference, then metrics
HDF5_USE_FILE_LOCKING=FALSE \
PANGU_CHECKPOINT=model_fp16_alias_compact.pth \
python inference.py
python result.py
```

The complete pruning → distillation → quantization → inference command chain is
documented in [`pangu_weather/蒸馏与推理说明.md`](pangu_weather/蒸馏与推理说明.md)
(Chinese).

## Tests

Run from the repository root, so that `pangu_weather` is importable as a package:

```bash
python -m unittest discover -s pangu_weather/tests -p 'test_*.py'
```

`pangu_weather/tests/` holds `unittest`-based static and lightweight runtime
contract tests that do not require a DCU. The cases in
`pangu_weather/module_test_scripts/` need an actual DCU runtime.

## Compliance

The student model predicts all 69 output channels directly. It does not predict
the residual from input to ground truth, does not exploit correlations between
test samples, and does not apply post-training external slope, affine, or
global-mean correction coefficients. The timed region contains only the model
forward pass and the required DCU synchronization. HIP kernels ship as source
and are compiled by `hipcc` on first run; no closed-source binaries are bundled.

See [`pangu_weather/COMPLIANCE_README.md`](pangu_weather/COMPLIANCE_README.md)
for details.
