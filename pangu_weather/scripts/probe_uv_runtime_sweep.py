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

BASE_ENV = {
    "PANGU_AUTO_SCAN_CHECKPOINT": "0",
    "PANGU_DISABLE_CUDA_GRAPH": "1",
    "PANGU_LAYERWISE_INFERENCE": "1",
    "PANGU_RECOMPUTE_SKIP": "0",
    "PANGU_DIRECT_RECOVERY": "1",
    "PANGU_DIRECT_RECOVERY_WIDTH_CHUNK": "16",
    "PANGU_SCORED_ONLY_RECOVERY": "1",
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
}

FORBIDDEN_VALUES = {
    "PANGU_GLOBAL_MEAN_CORRECTION": "1",
    "PANGU_STREAM_WEIGHTS": {"stage", "block"},
    "PANGU_USE_ONNX": "1",
    "PANGU_ATTN_CHUNK_SIZE": "2",
}


def candidate_label(env):
    return (
        f"w{env['PANGU_DIRECT_RECOVERY_WIDTH_CHUNK']}"
        f"_a{env['PANGU_ATTN_CHUNK_SIZE']}"
        f"_m{env['PANGU_MLP_CHUNK_SIZE']}"
        f"_split{env['PANGU_SPLIT_RECOVERY']}"
        f"_cache{env['PANGU_CACHE_EARTH_BIAS']}"
        f"_empty{env['PANGU_LAYERWISE_EMPTY_CACHE']}"
        f"_inplace{env['PANGU_INPLACE_BLOCK']}"
        f"_clear{env['PANGU_CLEAR_INPUT_REFS']}"
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
    if preset == "baseline":
        grid = BASELINE_GRID
        include_regression = False
        include_reset_probe = False
    elif preset == "focused":
        grid = FOCUSED_GRID
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
    if args.fp16_checkpoint:
        env["PANGU_FP16_CHECKPOINT"] = args.fp16_checkpoint

    latency_runs_ms = []
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
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fp16-checkpoint", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--preset",
        choices=["baseline", "focused", "full"],
        default="baseline",
        help=(
            "baseline measures only the 89.3716 fixed defaults; focused sweeps "
            "attention chunk size; full is the broad diagnostic grid."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-output-compare", action="store_true")
    args = parser.parse_args()

    pangu_dir = Path(__file__).resolve().parents[1]
    log_path = make_log_path(args.log_file)
    if not log_path.is_absolute():
        log_path = pangu_dir / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = list(iter_candidates(args.preset))
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
                f"vram={result.get('max_vram_mb')} "
                f"err={result.get('output_max_abs')}"
            )

    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
