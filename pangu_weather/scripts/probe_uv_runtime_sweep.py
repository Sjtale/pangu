#!/usr/bin/env python3
"""Run controlled U/V runtime probes for Pangu-Weather inference.

The script intentionally drives the existing inference.py entrypoint in a
subprocess so timing boundaries, model loading, and output generation stay
identical to the submission path.
"""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


FULL_GRID = {
    "PANGU_DIRECT_RECOVERY_WIDTH_CHUNK": ["4", "8", "12", "16", "24", "32"],
    "PANGU_ATTN_CHUNK_SIZE": ["3", "4", "5"],
    "PANGU_MLP_CHUNK_SIZE": ["16384", "32768", "65536"],
    "PANGU_SPLIT_RECOVERY": ["0", "1"],
    "PANGU_CACHE_EARTH_BIAS": ["0", "1"],
    "PANGU_LAYERWISE_EMPTY_CACHE": ["0", "1"],
}

FOCUSED_GRID = {
    # Keep every accepted high-score default fixed except the attention chunk.
    # The DCU/platform history already rejected broad width/MLP/cache sweeps.
    "PANGU_ATTN_CHUNK_SIZE": ["3", "4", "5"],
}

BASELINE_GRID = {}

COMPACT_MASK_GRID = {
    "PANGU_COMPACT_ATTN_MASK": ["1"],
}

DIRECT_MASK_GRID = {
    "PANGU_DIRECT_MASK_SLICE": ["1"],
}

CUDA_GRAPH_GRID = {
    "PANGU_DISABLE_CUDA_GRAPH": ["0"],
    "PANGU_LAYERWISE_CUDA_GRAPH": ["1"],
}

CPU_RECOVERY_GRID = {
    "PANGU_CPU_RECOVERY_OUTPUT": ["1"],
}

FULL_RECOVERY_GRID = {
    # Width 16 is emitted as the baseline. Larger chunks trade a small
    # temporary for fewer full-channel recovery GEMM launches.
    "PANGU_DIRECT_RECOVERY_WIDTH_CHUNK": ["24", "32", "48"],
}

HIP_CANDIDATES = [
    {"PANGU_HIP_SCHEDULE_SPIN": "1"},
    {"PANGU_HIP_PREFER_L1": "1"},
    {"PANGU_HIP_STREAM_SPIN": "1"},
    {
        "PANGU_HIP_SCHEDULE_SPIN": "1",
        "PANGU_HIP_PREFER_L1": "1",
        "PANGU_HIP_STREAM_SPIN": "1",
    },
]

STAGEWISE_CANDIDATES = [
    {
        "PANGU_ATTN_CHUNK_SIZE_LAYER2": "4",
        "PANGU_ATTN_CHUNK_SIZE_LAYER3": "4",
    },
    {
        "PANGU_ATTN_CHUNK_SIZE_LAYER2": "8",
        "PANGU_ATTN_CHUNK_SIZE_LAYER3": "8",
        "PANGU_CHUNKED_QKV_LAYER2": "0",
        "PANGU_CHUNKED_QKV_LAYER3": "0",
        "PANGU_CHUNKED_PROJ_LAYER2": "0",
        "PANGU_CHUNKED_PROJ_LAYER3": "0",
        "PANGU_MLP_CHUNK_SIZE_LAYER2": "65536",
        "PANGU_MLP_CHUNK_SIZE_LAYER3": "65536",
    },
    {
        "PANGU_ATTN_CHUNK_SIZE_LAYER2": "0",
        "PANGU_ATTN_CHUNK_SIZE_LAYER3": "0",
        "PANGU_CHUNKED_QKV_LAYER2": "0",
        "PANGU_CHUNKED_QKV_LAYER3": "0",
        "PANGU_CHUNKED_PROJ_LAYER2": "0",
        "PANGU_CHUNKED_PROJ_LAYER3": "0",
        "PANGU_MLP_CHUNK_SIZE_LAYER2": "0",
        "PANGU_MLP_CHUNK_SIZE_LAYER3": "0",
    },
]

PANGU_LITE_2D_ENV = {
    # The 2D student has its own forward path. Disable optimizations that patch
    # OneScience's 3D Pangu modules so the probe measures the architecture as-is.
    "PANGU_MODEL_ARCHITECTURE": "PanguLite2DAttentionPosEmbed",
    "PANGU_LAYERWISE_INFERENCE": "0",
    "PANGU_DIRECT_RECOVERY": "0",
    "PANGU_CHUNKED_ATTENTION": "0",
    "PANGU_CHUNKED_QKV": "0",
    "PANGU_CHUNKED_PROJ": "0",
    "PANGU_CHUNKED_MLP": "0",
    "PANGU_INPLACE_BLOCK": "0",
    "PANGU_RESET_PEAK_AFTER_LOAD": "1",
}

BASE_ENV = {
    "PANGU_AUTO_SCAN_CHECKPOINT": "0",
    "PANGU_DISABLE_CUDA_GRAPH": "1",
    "PANGU_LAYERWISE_INFERENCE": "1",
    "PANGU_RECOMPUTE_SKIP": "0",
    "PANGU_DIRECT_RECOVERY": "1",
    "PANGU_DIRECT_RECOVERY_WIDTH_CHUNK": "16",
    "PANGU_SCORED_ONLY_RECOVERY": "0",
    "PANGU_CHUNKED_ATTENTION": "1",
    "PANGU_ATTN_CHUNK_SIZE": "3",
    "PANGU_CHUNKED_QKV": "1",
    "PANGU_CHUNKED_PROJ": "1",
    "PANGU_CHUNKED_MLP": "1",
    "PANGU_MLP_CHUNK_SIZE": "32768",
    "PANGU_DISABLE_AFFINE_CALIBRATION": "1",
    "PANGU_GLOBAL_MEAN_CORRECTION": "0",
    "PANGU_STREAM_WEIGHTS": "0",
    "PANGU_SPLIT_RECOVERY": "0",
    "PANGU_CACHE_EARTH_BIAS": "0",
    "PANGU_INPLACE_BLOCK": "1",
    "PANGU_CLEAR_INPUT_REFS": "1",
    "PANGU_LAYERWISE_EMPTY_CACHE": "0",
    "PANGU_USE_ONNX": "0",
    "PANGU_PROFILE_MEMORY": "1",
    "PANGU_COMPACT_ATTN_MASK": "0",
    "PANGU_DIRECT_MASK_SLICE": "0",
    "PANGU_GRAPH_DIRECT_INPUT": "1",
    "PANGU_CPU_RECOVERY_OUTPUT": "0",
    "PANGU_HIP_SCHEDULE_SPIN": "0",
    "PANGU_HIP_PREFER_L1": "0",
    "PANGU_HIP_STREAM_SPIN": "0",
    # P2 is an explicit full-model A/B candidate; keep every other preset
    # production-safe even if the caller's shell exports the flag.
    "PANGU_P2_TILED_ATTENTION": "0",
    "PANGU_P2_TILED_MODE": "online",
    "PANGU_P2_FULL_WIDTH": "1",
    # Buffer interning is the platform-verified 90.1048 guardrail default.
    "PANGU_INTERN_IMMUTABLE_BUFFERS": "1",
}

for _stage in ("LAYER1", "LAYER2", "LAYER3", "LAYER4"):
    BASE_ENV[f"PANGU_ATTN_CHUNK_SIZE_{_stage}"] = "3"
    BASE_ENV[f"PANGU_CHUNKED_QKV_{_stage}"] = "1"
    BASE_ENV[f"PANGU_CHUNKED_PROJ_{_stage}"] = "1"
    BASE_ENV[f"PANGU_MLP_CHUNK_SIZE_{_stage}"] = "32768"

FORBIDDEN_VALUES = {
    "PANGU_GLOBAL_MEAN_CORRECTION": "1",
    "PANGU_STREAM_WEIGHTS": {"stage", "block"},
    "PANGU_USE_ONNX": "1",
    "PANGU_ATTN_CHUNK_SIZE": "2",
}


def candidate_label(env):
    if env.get("PANGU_MODEL_ARCHITECTURE") == "PanguLite2DAttentionPosEmbed":
        return f"pangu_lite_2d_pos288_reset{env.get('PANGU_RESET_PEAK_AFTER_LOAD', '0')}"
    return (
        f"w{env['PANGU_DIRECT_RECOVERY_WIDTH_CHUNK']}"
        f"_a{env['PANGU_ATTN_CHUNK_SIZE']}"
        f"_m{env['PANGU_MLP_CHUNK_SIZE']}"
        f"_split{env['PANGU_SPLIT_RECOVERY']}"
        f"_cache{env['PANGU_CACHE_EARTH_BIAS']}"
        f"_empty{env['PANGU_LAYERWISE_EMPTY_CACHE']}"
        f"_inplace{env['PANGU_INPLACE_BLOCK']}"
        f"_clear{env['PANGU_CLEAR_INPUT_REFS']}"
        f"_scored{env['PANGU_SCORED_ONLY_RECOVERY']}"
        f"_mask{env['PANGU_COMPACT_ATTN_MASK']}"
        f"_directmask{env['PANGU_DIRECT_MASK_SLICE']}"
        f"_graph{int(env['PANGU_DISABLE_CUDA_GRAPH'] == '0')}"
        f"_graphinput{env['PANGU_GRAPH_DIRECT_INPUT']}"
        f"_cpuout{env['PANGU_CPU_RECOVERY_OUTPUT']}"
        f"_intern{env['PANGU_INTERN_IMMUTABLE_BUFFERS']}"
        f"_hip{env['PANGU_HIP_SCHEDULE_SPIN']}"
        f"{env['PANGU_HIP_PREFER_L1']}"
        f"{env['PANGU_HIP_STREAM_SPIN']}"
        f"_p2{env.get('PANGU_P2_TILED_ATTENTION', '0')}"
        f"_l23a{env['PANGU_ATTN_CHUNK_SIZE_LAYER2']}"
        f"q{env['PANGU_CHUNKED_QKV_LAYER2']}"
        f"p{env['PANGU_CHUNKED_PROJ_LAYER2']}"
        f"m{env['PANGU_MLP_CHUNK_SIZE_LAYER2']}"
        f"_reset{env.get('PANGU_RESET_PEAK_AFTER_LOAD', '0')}"
    )


def validate_env(env):
    for name, forbidden in FORBIDDEN_VALUES.items():
        value = env.get(name)
        if isinstance(forbidden, set):
            if value in forbidden:
                raise ValueError(f"{name}={value} is forbidden for this sweep")
        elif value == forbidden:
            raise ValueError(f"{name}={value} is forbidden for this sweep")


def iter_candidates(preset="baseline"):
    if preset == "pangu-lite-2d":
        merged = dict(BASE_ENV)
        merged.update(PANGU_LITE_2D_ENV)
        validate_env(merged)
        yield {
            "label": candidate_label(merged),
            "kind": "architecture",
            "env": merged,
        }
        return

    if preset == "p2-tiled":
        for value, kind in (("0", "baseline"), ("1", "p2-tiled")):
            env = dict(BASE_ENV)
            env["PANGU_P2_TILED_ATTENTION"] = value
            if value == "1":
                env["PANGU_P2_TILED_MODE"] = "full-row-fast"
            validate_env(env)
            yield {
                "label": candidate_label(env),
                "kind": kind,
                "env": env,
            }
        return

    if preset in {"hip", "buffer-intern", "stagewise"}:
        candidate_envs = {
            "hip": HIP_CANDIDATES,
            "buffer-intern": [{"PANGU_INTERN_IMMUTABLE_BUFFERS": "1"}],
            "stagewise": STAGEWISE_CANDIDATES,
        }[preset]
        baseline_env = dict(BASE_ENV)
        if preset == "buffer-intern":
            # Retain the historical off/on diagnostic without weakening the
            # HIP and stage-wise guardrail baseline.
            baseline_env["PANGU_INTERN_IMMUTABLE_BUFFERS"] = "0"
        yield {
            "label": candidate_label(baseline_env),
            "kind": "baseline",
            "env": baseline_env,
        }
        for values in candidate_envs:
            env = dict(BASE_ENV)
            env.update(values)
            validate_env(env)
            yield {
                "label": candidate_label(env),
                "kind": preset,
                "env": env,
            }
        return
    if preset == "baseline":
        grid = BASELINE_GRID
        include_regression = False
        include_reset_probe = False
    elif preset == "focused":
        grid = FOCUSED_GRID
        include_regression = False
        include_reset_probe = False
    elif preset == "compact-mask":
        grid = COMPACT_MASK_GRID
        include_regression = False
        include_reset_probe = False
    elif preset == "direct-mask":
        grid = DIRECT_MASK_GRID
        include_regression = False
        include_reset_probe = False
    elif preset == "cuda-graph":
        grid = CUDA_GRAPH_GRID
        include_regression = False
        include_reset_probe = False
    elif preset == "cpu-recovery":
        grid = CPU_RECOVERY_GRID
        include_regression = False
        include_reset_probe = False
    elif preset == "full-recovery":
        grid = FULL_RECOVERY_GRID
        include_regression = False
        include_reset_probe = False
    elif preset == "full":
        grid = FULL_GRID
        include_regression = True
        include_reset_probe = True
    else:
        raise ValueError(f"Unknown preset: {preset}")

    seen = set()

    def emit(env, kind):
        merged = dict(BASE_ENV)
        merged.update(env)
        validate_env(merged)
        label = candidate_label(merged)
        if label in seen:
            return None
        seen.add(label)
        return {"label": label, "kind": kind, "env": merged}

    baseline = emit({}, "baseline")
    if baseline:
        yield baseline

    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        candidate = emit(dict(zip(keys, values)), "grid")
        if candidate:
            yield candidate

    if include_regression:
        regression = emit({"PANGU_INPLACE_BLOCK": "0"}, "inplace_regression")
        if regression:
            yield regression

    if include_reset_probe:
        reset_probe = emit({"PANGU_RESET_PEAK_AFTER_LOAD": "1"}, "reset_peak_probe")
        if reset_probe:
            yield reset_probe


def checkpoint_ab_candidates(candidates, baseline_checkpoint, candidate_checkpoint):
    if len(candidates) != 1:
        raise ValueError("Checkpoint A/B requires the baseline preset")
    baseline = dict(candidates[0])
    baseline["env"] = dict(baseline["env"])
    baseline["env"]["PANGU_FP16_CHECKPOINT"] = baseline_checkpoint
    baseline["label"] += "_ckptbaseline"

    candidate = dict(baseline)
    candidate["kind"] = "checkpoint_candidate"
    candidate["env"] = dict(baseline["env"])
    candidate["env"]["PANGU_FP16_CHECKPOINT"] = candidate_checkpoint
    candidate["label"] = candidates[0]["label"] + "_ckptcandidate"
    return [baseline, candidate]


def reset_output_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_stdout(stdout):
    max_vram_mb = None
    current_vram_mb = None
    reserved_values = []
    for line in stdout.splitlines():
        match = re.search(r"Max VRAM:\s*([0-9.]+)\s*MB", line)
        if match:
            max_vram_mb = float(match.group(1))
        match = re.search(r"Current VRAM:\s*([0-9.]+)\s*MB", line)
        if match:
            current_vram_mb = float(match.group(1))
        match = re.search(r"reserved=([0-9.]+)\s*MB", line)
        if match:
            reserved_values.append(float(match.group(1)))
    return {
        "max_vram_mb": max_vram_mb,
        "current_vram_mb": current_vram_mb,
        "reserved_mb": max(reserved_values) if reserved_values else None,
    }


def read_time_record(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        values = json.load(f)
    return [float(value) for value in values]


def compare_outputs(candidate_dir, baseline_dir):
    if baseline_dir is None or not baseline_dir.exists():
        return {"output_max_abs": None, "output_max_rel": None, "output_files": 0}
    max_abs = 0.0
    max_rel = 0.0
    matched = 0
    for baseline_file in sorted(baseline_dir.glob("*.npy")):
        candidate_file = candidate_dir / baseline_file.name
        if not candidate_file.exists():
            continue
        baseline = np.load(baseline_file)
        candidate = np.load(candidate_file)
        diff = np.abs(candidate - baseline)
        denom = np.maximum(np.abs(baseline), 1.0e-6)
        max_abs = max(max_abs, float(np.max(diff)))
        max_rel = max(max_rel, float(np.max(diff / denom)))
        matched += 1
    return {
        "output_max_abs": max_abs if matched else None,
        "output_max_rel": max_rel if matched else None,
        "output_files": matched,
    }


def run_one(candidate, *, args, pangu_dir, output_dir, baseline_dir):
    env = os.environ.copy()
    env.update(candidate["env"])
    env["PANGU_MAX_INFERENCE_BATCHES"] = str(args.max_batches)
    if args.fp16_checkpoint and "PANGU_FP16_CHECKPOINT" not in candidate["env"]:
        env["PANGU_FP16_CHECKPOINT"] = args.fp16_checkpoint

    if env.get("PANGU_MODEL_ARCHITECTURE") == "PanguLite2DAttentionPosEmbed":
        checkpoint = Path(env["PANGU_FP16_CHECKPOINT"])
        if not checkpoint.is_absolute():
            checkpoint = pangu_dir / "data" / "checkpoints" / checkpoint
        if not checkpoint.is_file():
            return {
                "label": candidate["label"],
                "kind": candidate["kind"],
                "env": candidate["env"],
                "returncode": 2,
                "error": f"2D architecture checkpoint not found: {checkpoint}",
                "stdout_tail": "Refusing to fall back to the official 3D teacher.",
            }

    latency_runs_ms = []
    steady_latency_runs_ms = []
    parsed_runs = []
    stdout_tail = ""
    start_wall = time.perf_counter()
    for repeat_index in range(args.repeat):
        reset_output_dir(output_dir)
        command = [args.python, "inference.py"]
        process = subprocess.run(
            command,
            cwd=str(pangu_dir),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stdout_tail = "\n".join(process.stdout.splitlines()[-80:])
        if process.returncode != 0:
            return {
                "label": candidate["label"],
                "kind": candidate["kind"],
                "env": candidate["env"],
                "returncode": process.returncode,
                "error": "inference.py failed",
                "stdout_tail": stdout_tail,
            }
        times = read_time_record(pangu_dir / "result" / "time_record.json")
        latency_runs_ms.extend(value * 1000.0 for value in times)
        steady_times = times[1:] if len(times) > 1 else times
        steady_latency_runs_ms.extend(value * 1000.0 for value in steady_times)
        parsed_runs.append(parse_stdout(process.stdout))

    output_metrics = compare_outputs(output_dir, baseline_dir)
    max_values = [item["max_vram_mb"] for item in parsed_runs if item["max_vram_mb"] is not None]
    reserved_values = [item["reserved_mb"] for item in parsed_runs if item["reserved_mb"] is not None]
    current_values = [item["current_vram_mb"] for item in parsed_runs if item["current_vram_mb"] is not None]
    return {
        "label": candidate["label"],
        "kind": candidate["kind"],
        "env": candidate["env"],
        "returncode": 0,
        "repeat": args.repeat,
        "max_batches": args.max_batches,
        "latency_ms_values": latency_runs_ms,
        "latency_avg_ms": float(np.mean(latency_runs_ms)) if latency_runs_ms else None,
        "latency_min_ms": float(np.min(latency_runs_ms)) if latency_runs_ms else None,
        "latency_p50_ms": float(np.median(latency_runs_ms)) if latency_runs_ms else None,
        "steady_latency_ms_values": steady_latency_runs_ms,
        "steady_latency_avg_ms": (
            float(np.mean(steady_latency_runs_ms)) if steady_latency_runs_ms else None
        ),
        "steady_latency_p50_ms": (
            float(np.median(steady_latency_runs_ms)) if steady_latency_runs_ms else None
        ),
        "steady_latency_p90_ms": (
            float(np.percentile(steady_latency_runs_ms, 90))
            if steady_latency_runs_ms else None
        ),
        "steady_latency_std_ms": (
            float(np.std(steady_latency_runs_ms)) if steady_latency_runs_ms else None
        ),
        "max_vram_mb": max(max_values) if max_values else None,
        "reserved_mb": max(reserved_values) if reserved_values else None,
        "current_vram_mb": current_values[-1] if current_values else None,
        "wall_time_s": time.perf_counter() - start_wall,
        "stdout_tail": stdout_tail,
        **output_metrics,
    }


def make_log_path(path):
    if path:
        return Path(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"uv_runtime_sweep_{stamp}.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Probe U/V runtime candidates.")
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fp16-checkpoint", default=None)
    parser.add_argument(
        "--candidate-fp16-checkpoint",
        default=None,
        help=(
            "Run a two-checkpoint A/B against --fp16-checkpoint or "
            "model_fp16.pth when using the baseline preset."
        ),
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--buffer-intern",
        choices=("0", "1"),
        default=None,
        help="Force immutable-buffer interning off/on for every emitted candidate.",
    )
    parser.add_argument(
        "--preset",
        choices=[
            "baseline", "compact-mask", "direct-mask", "cuda-graph", "cpu-recovery",
            "full-recovery", "focused", "full", "pangu-lite-2d", "hip",
            "buffer-intern", "stagewise",
            "p2-tiled",
        ],
        default="baseline",
        help=(
            "baseline measures only the fixed defaults; compact-mask runs an "
            "isolated off/on A/B; direct-mask removes mask index kernels; "
            "cuda-graph isolates graph replay; full-recovery sweeps only direct-recovery "
            "width; pangu-lite-2d measures the 2D positional-embedding student; "
            "hip isolates the three HIP runtime controls; buffer-intern shares "
            "identical FP16 masks/indexes; stagewise screens three layer2/3 "
            "chunk schedules; p2-tiled runs an explicit off/on full-model A/B "
            "for pgw_lite_pruned_96; focused sweeps attention chunk size; full "
            "is the broad diagnostic grid."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-output-compare", action="store_true")
    args = parser.parse_args()

    if args.preset == "pangu-lite-2d" and args.fp16_checkpoint is None:
        args.fp16_checkpoint = "model_pangu_lite_2d_pos288_hybrid.pth"

    pangu_dir = Path(__file__).resolve().parents[1]
    log_path = make_log_path(args.log_file)
    if not log_path.is_absolute():
        log_path = pangu_dir / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = list(iter_candidates(args.preset))
    if args.buffer_intern is not None:
        for candidate in candidates:
            candidate["env"] = dict(candidate["env"])
            candidate["env"]["PANGU_INTERN_IMMUTABLE_BUFFERS"] = args.buffer_intern
            candidate["label"] = candidate_label(candidate["env"])
    if args.preset == "pangu-lite-2d":
        for candidate in candidates:
            candidate["env"] = dict(candidate["env"])
            candidate["env"]["PANGU_FP16_CHECKPOINT"] = args.fp16_checkpoint
    if args.candidate_fp16_checkpoint:
        candidates = checkpoint_ab_candidates(
            candidates,
            args.fp16_checkpoint or "model_fp16.pth",
            args.candidate_fp16_checkpoint,
        )
    if args.limit > 0:
        candidates = candidates[: args.limit]

    if args.dry_run:
        for candidate in candidates:
            print(json.dumps(candidate, ensure_ascii=False, sort_keys=True))
        print(f"# candidates={len(candidates)}")
        return

    output_dir = pangu_dir / "result" / "output"
    baseline_dir = None
    with log_path.open("w", encoding="utf-8") as log_file:
        for index, candidate in enumerate(candidates, start=1):
            print(f"[{index}/{len(candidates)}] {candidate['label']}")
            result = run_one(
                candidate,
                args=args,
                pangu_dir=pangu_dir,
                output_dir=output_dir,
                baseline_dir=None if args.skip_output_compare else baseline_dir,
            )
            if baseline_dir is None and result.get("returncode") == 0:
                baseline_dir = pangu_dir / "result" / "uv_sweep_baseline"
                reset_output_dir(baseline_dir)
                for npy_path in output_dir.glob("*.npy"):
                    shutil.copy2(npy_path, baseline_dir / npy_path.name)
            log_file.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            log_file.flush()
            print(
                f"  rc={result.get('returncode')} "
                f"lat={result.get('latency_avg_ms')} "
                f"steady={result.get('steady_latency_avg_ms')} "
                f"vram={result.get('max_vram_mb')} "
                f"err={result.get('output_max_abs')}"
            )

    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
