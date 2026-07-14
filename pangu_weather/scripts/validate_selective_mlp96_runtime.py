#!/usr/bin/env python3
"""Fail-closed runtime gate for the SelectiveMLP-96 checkpoint A/B probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REQUIRED_REPEATS = 5
REQUIRED_BATCHES = 5
REQUIRED_STEADY_POINTS = 20
ABSOLUTE_MEAN_LIMIT_MS = 95.0
BASELINE_RATIO_LIMIT = 0.95
MAX_CHECKPOINT_MIB = 29.1
MIB = 1024 ** 2


def _finite_number(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"runtime row has no numeric {key}")
    value = float(value)
    if not (value >= 0.0 and value < float("inf")):
        raise ValueError(f"runtime row has invalid {key}={value!r}")
    return value


def load_probe_rows(path):
    path = Path(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 2:
        raise ValueError(f"checkpoint A/B log must contain exactly two rows, got {len(rows)}")
    baseline = next((row for row in rows if row.get("kind") == "baseline"), None)
    candidate = next(
        (row for row in rows if row.get("kind") == "checkpoint_candidate"), None
    )
    if baseline is None or candidate is None:
        raise ValueError("runtime log must contain baseline and checkpoint_candidate rows")
    return baseline, candidate


def validate_runtime_rows(baseline, candidate, *, checkpoint_size_bytes):
    for name, row in (("baseline", baseline), ("candidate", candidate)):
        if row.get("returncode") != 0:
            raise ValueError(f"{name} inference process failed")
        if row.get("repeat") != REQUIRED_REPEATS:
            raise ValueError(f"{name} must use {REQUIRED_REPEATS} independent processes")
        if row.get("max_batches") != REQUIRED_BATCHES:
            raise ValueError(f"{name} must run one warmup plus four timed samples")
        points = row.get("steady_latency_ms_values")
        if not isinstance(points, list) or len(points) != REQUIRED_STEADY_POINTS:
            raise ValueError(
                f"{name} must contain {REQUIRED_STEADY_POINTS} steady latency points"
            )

    baseline_mean = _finite_number(baseline, "steady_latency_avg_ms")
    candidate_mean = _finite_number(candidate, "steady_latency_avg_ms")
    baseline_p90 = _finite_number(baseline, "steady_latency_p90_ms")
    candidate_p90 = _finite_number(candidate, "steady_latency_p90_ms")
    baseline_vram = _finite_number(baseline, "max_vram_mb")
    candidate_vram = _finite_number(candidate, "max_vram_mb")
    checkpoint_mib = float(checkpoint_size_bytes) / MIB
    mean_limit = min(ABSOLUTE_MEAN_LIMIT_MS, BASELINE_RATIO_LIMIT * baseline_mean)

    gates = {
        "steady_mean": {
            "actual_ms": candidate_mean,
            "limit_ms": mean_limit,
            "passed": candidate_mean <= mean_limit,
        },
        "steady_p90": {
            "actual_ms": candidate_p90,
            "limit_ms": baseline_p90,
            "passed": candidate_p90 <= baseline_p90,
        },
        "peak_vram": {
            "actual_mb": candidate_vram,
            "limit_mb": baseline_vram,
            "passed": candidate_vram <= baseline_vram,
        },
        "checkpoint_size": {
            "actual_mib": checkpoint_mib,
            "limit_mib": MAX_CHECKPOINT_MIB,
            "passed": checkpoint_mib <= MAX_CHECKPOINT_MIB,
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "passed": passed,
        "protocol": {
            "independent_processes": REQUIRED_REPEATS,
            "warmup_per_process": 1,
            "steady_samples_per_process": REQUIRED_BATCHES - 1,
            "steady_points": REQUIRED_STEADY_POINTS,
        },
        "baseline": {
            "steady_mean_ms": baseline_mean,
            "steady_p90_ms": baseline_p90,
            "peak_vram_mb": baseline_vram,
        },
        "candidate": {
            "steady_mean_ms": candidate_mean,
            "steady_p90_ms": candidate_p90,
            "peak_vram_mb": candidate_vram,
            "checkpoint_mib": checkpoint_mib,
        },
        "gates": gates,
    }


def validate_runtime_log(log_path, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    baseline, candidate = load_probe_rows(log_path)
    return validate_runtime_rows(
        baseline,
        candidate,
        checkpoint_size_bytes=checkpoint_path.stat().st_size,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    report = validate_runtime_log(args.log, args.checkpoint)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite runtime gate report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
