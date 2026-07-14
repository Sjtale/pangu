#!/usr/bin/env python3
"""Diagnose the next inference-time (U) and lightweight (V) directions.

The clean A/B always executes the repository's real ``inference.py`` in
isolated subprocess working directories.  CUDA-event attribution and memory
inventory run separately so their synchronization and hooks cannot influence
the latency samples used for score projections.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import runpy
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MIB = 1024 ** 2
SCHEMA_VERSION = 1
EXPECTED_PROFILE = {
    "patch_size": [2, 8, 8],
    "embed_dim": 96,
    "num_heads": [3, 6, 6, 3],
    "depth_blocks": [2, 6, 6, 2],
}
SCORE_REFERENCE = {
    "total": 90.1743,
    "score_inference_time": 17.9758,
    "score_lightweight": 35.9786,
    "score_prediction": 36.2200,
    "metric_mapping": {
        "U": "inference_time",
        "V": "lightweight",
        "W": "prediction",
    },
    "rounding_note": (
        "Displayed components sum to 90.1744; preserve the platform total "
        "90.1743 verbatim."
    ),
}
KNOWN_PACKAGE = {
    "bytes": 65701,
    "sha256": "553948caf2e97285a6794c81cba9a1ca5ef37a9ec68d2404a9e7f74af3793cc4",
}
EXPECTED_COMPACT_CHECKPOINT_BYTES = 36190484
EXPECTED_BASELINE = {
    "p2_on_mean_ms": 77.2945,
    "p2_on_peak_mb": 504.6,
    "p2_on_current_mb": 109.9,
    "latency_tolerance_fraction": 0.02,
    "memory_tolerance_mb": 5.0,
}
SOURCE_PATHS = (
    "inference.py",
    "p2_tiled_attention.py",
    "hip_earth_attention_tiled.py",
    "hip_kernels/earth_attention_tiled_fwd.hip",
    "pangu_profile_model.py",
    "conf/config.yaml",
)
MEMORY_RE = re.compile(
    r"^\[MEM\]\s+(?P<tag>.*?):\s+allocated=(?P<allocated>[0-9.]+)\s+MB,\s+"
    r"reserved=(?P<reserved>[0-9.]+)\s+MB,\s+peak=(?P<peak>[0-9.]+)\s+MB$"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_script_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile_value) / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "std_ms": None,
            "cv": None,
            "min_ms": None,
            "max_ms": None,
            "values_ms": [],
        }
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean_ms": mean,
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "std_ms": std,
        "cv": std / mean if mean else None,
        "min_ms": min(values),
        "max_ms": max(values),
        "values_ms": values,
    }


def paired_bootstrap_ci(candidate_values, oracle_values, samples=10000, seed=901743):
    if len(candidate_values) != len(oracle_values) or not candidate_values:
        raise ValueError("Paired bootstrap requires equal, non-empty sample vectors")
    deltas = [
        float(candidate) - float(oracle)
        for candidate, oracle in zip(candidate_values, oracle_values)
    ]
    generator = random.Random(seed)
    means = []
    for _ in range(int(samples)):
        means.append(
            statistics.fmean(
                deltas[generator.randrange(len(deltas))]
                for _ in range(len(deltas))
            )
        )
    return {
        "metric": "candidate_minus_oracle_ms",
        "paired_count": len(deltas),
        "mean_delta_ms": statistics.fmean(deltas),
        "lower_95_ms": percentile(means, 2.5),
        "upper_95_ms": percentile(means, 97.5),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def latency_score_targets(current_latency_ms, current_score=None):
    """Return the local elasticity targets used for the 90.1743 guardrail."""

    current_score = (
        SCORE_REFERENCE["score_inference_time"]
        if current_score is None
        else float(current_score)
    )
    targets = []
    for score_gain in (0.05, 0.10, 0.20):
        reduction_fraction = score_gain / 2.0
        targets.append(
            {
                "score_gain": score_gain,
                "target_score": current_score + score_gain,
                "required_latency_reduction_pct": reduction_fraction * 100.0,
                "target_latency_ms": float(current_latency_ms)
                * (1.0 - reduction_fraction),
            }
        )
    return targets


def classify_cache_attribute(attribute_name, index=None):
    if attribute_name == "_pangu_p2_tiled_bias_index":
        return {1: "packed_bias", 2: "compact_index"}.get(index, "cache_metadata")
    if attribute_name == "_pangu_p2_tiled_region_ids":
        return "region_ids" if index == 1 else "cache_metadata"
    if attribute_name == "earth_position_bias_table":
        return "original_bias"
    if attribute_name == "earth_position_index":
        return "original_index"
    return "other"


def dedupe_tensor_records(records):
    unique = {}
    by_kind_keys = defaultdict(set)
    modules_by_storage = defaultdict(set)
    for record in records:
        storage_key = str(record["storage_key"])
        unique.setdefault(storage_key, record)
        by_kind_keys[record["kind"]].add(storage_key)
        modules_by_storage[storage_key].add(record["module"])

    by_kind = {}
    for kind, keys in sorted(by_kind_keys.items()):
        logical_bytes = sum(
            int(record["logical_bytes"])
            for record in records
            if record["kind"] == kind
        )
        unique_bytes = sum(int(unique[key]["storage_bytes"]) for key in keys)
        by_kind[kind] = {
            "logical_bytes": logical_bytes,
            "unique_storage_bytes": unique_bytes,
            "unique_storage_mb": unique_bytes / MIB,
            "storage_count": len(keys),
        }

    share_groups = [
        {
            "storage_key": key,
            "modules": sorted(modules),
            "module_count": len(modules),
            "storage_bytes": int(unique[key]["storage_bytes"]),
        }
        for key, modules in modules_by_storage.items()
        if len(modules) > 1
    ]
    share_groups.sort(key=lambda item: item["storage_bytes"], reverse=True)
    return {
        "logical_bytes": sum(int(item["logical_bytes"]) for item in records),
        "unique_storage_bytes": sum(
            int(item["storage_bytes"]) for item in unique.values()
        ),
        "by_kind": by_kind,
        "share_groups": share_groups,
    }


def _shareable_bytes(records, kind):
    relevant = [record for record in records if record["kind"] == kind]
    if not relevant:
        return 0
    unique_by_storage = {}
    for record in relevant:
        unique_by_storage.setdefault(str(record["storage_key"]), record)
    total = sum(int(record["storage_bytes"]) for record in unique_by_storage.values())
    keep_by_signature = {}
    for record in unique_by_storage.values():
        signature = (tuple(record["shape"]), record["dtype"])
        keep_by_signature[signature] = max(
            keep_by_signature.get(signature, 0), int(record["storage_bytes"])
        )
    return max(0, total - sum(keep_by_signature.values()))


def build_cache_summary(records, patched_modules):
    deduped = dedupe_tensor_records(records)
    by_kind = deduped["by_kind"]
    packed_bytes = by_kind.get("packed_bias", {}).get("unique_storage_bytes", 0)
    deduped["patched_modules"] = int(patched_modules)
    deduped["records"] = records
    deduped["reclaim_scenarios"] = {
        "eliminate_packed_bias_double_residency_bytes": packed_bytes,
        "share_compact_index_bytes": _shareable_bytes(records, "compact_index"),
        "share_region_ids_bytes": _shareable_bytes(records, "region_ids"),
    }
    return deduped


def assess_baseline_drift(p2_mean_ms, p2_peak_mb, p2_current_mb):
    values = {
        "latency": (
            p2_mean_ms,
            EXPECTED_BASELINE["p2_on_mean_ms"],
            EXPECTED_BASELINE["latency_tolerance_fraction"]
            * EXPECTED_BASELINE["p2_on_mean_ms"],
        ),
        "peak_memory": (
            p2_peak_mb,
            EXPECTED_BASELINE["p2_on_peak_mb"],
            EXPECTED_BASELINE["memory_tolerance_mb"],
        ),
        "current_memory": (
            p2_current_mb,
            EXPECTED_BASELINE["p2_on_current_mb"],
            EXPECTED_BASELINE["memory_tolerance_mb"],
        ),
    }
    checks = {}
    for name, (actual, expected, tolerance) in values.items():
        checks[name] = {
            "actual": actual,
            "expected": expected,
            "tolerance": tolerance,
            "passed": actual is not None and abs(float(actual) - expected) <= tolerance,
        }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def correctness_gate(correctness_records):
    if not correctness_records:
        return {"passed": False, "reason": "no output comparisons"}
    passed = all(
        record.get("exact")
        and record.get("output_files", 0) > 0
        and record.get("all_outputs_have_69_channels")
        and record.get("nan_count", 0) == 0
        and record.get("inf_count", 0) == 0
        for record in correctness_records
    )
    return {
        "passed": passed,
        "rounds": len(correctness_records),
        "total_mismatch_count": sum(
            int(record.get("mismatch_count", 0)) for record in correctness_records
        ),
    }


def _git_output(repo, *args):
    process = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return process.returncode, process.stdout.strip(), process.stderr.strip()


def collect_git_provenance(repo):
    head_rc, head, head_error = _git_output(repo, "rev-parse", "HEAD")
    status_rc, status, status_error = _git_output(repo, "status", "--short", "--branch")
    return {
        "head": head if head_rc == 0 else None,
        "status": status if status_rc == 0 else None,
        "clean": status_rc == 0
        and not any(
            line and not line.startswith("##") for line in status.splitlines()
        ),
        "errors": [error for error in (head_error, status_error) if error],
    }


def collect_environment():
    environment = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        environment.update(
            {
                "torch": torch.__version__,
                "torch_hip": getattr(torch.version, "hip", None),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            environment["device"] = torch.cuda.get_device_name(0)
            environment["total_memory_bytes"] = int(properties.total_memory)
            environment["gcn_arch"] = getattr(properties, "gcnArchName", None)
    except ImportError:
        environment["torch"] = None
        environment["cuda_available"] = False

    for tool in ("hipcc", "hipprof", "rocprof", "rocm-smi", "amd-smi"):
        environment[f"{tool}_path"] = shutil.which(tool)
    return environment


def run_static(checkpoint, package_zip, pangu_dir):
    checkpoint = Path(checkpoint).resolve()
    audit_checkpoint_module = _load_script_module(
        "diagnose_uv_audit_checkpoint",
        pangu_dir / "scripts" / "audit_pruned96_uv.py",
    )
    checkpoint_report = audit_checkpoint_module.audit_checkpoint(checkpoint)
    normalized = {
        key: checkpoint_report["profile"].get(key) for key in EXPECTED_PROFILE
    }
    if normalized != EXPECTED_PROFILE:
        raise ValueError(f"Unexpected checkpoint profile: {normalized}")
    checkpoint_report["sha256"] = sha256_file(checkpoint)
    checkpoint_report["file_bytes"] = checkpoint.stat().st_size

    package_report = None
    if package_zip is not None:
        package_path = Path(package_zip).resolve()
        audit_package_module = _load_script_module(
            "diagnose_uv_audit_package",
            pangu_dir / "scripts" / "audit_submission_package.py",
        )
        package_report = audit_package_module.audit_zip(package_path, checkpoint)
        package_report["matches_known_scored_package"] = (
            package_report["package_bytes"] == KNOWN_PACKAGE["bytes"]
            and package_report["package_sha256"] == KNOWN_PACKAGE["sha256"]
        )

    source_hashes = {}
    for relative_path in SOURCE_PATHS:
        path = pangu_dir / relative_path
        source_hashes[relative_path] = sha256_file(path) if path.is_file() else None

    repo = pangu_dir.parent
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "score_reference": SCORE_REFERENCE,
        "provenance": {
            "status": "user_confirmed",
            "git": collect_git_provenance(repo),
            "source_sha256": source_hashes,
            "checkpoint_expected_basename": "model_fp16_alias_compact.pth",
            "checkpoint_expected_bytes": EXPECTED_COMPACT_CHECKPOINT_BYTES,
            "checkpoint_matches_expected_bytes": (
                checkpoint.stat().st_size == EXPECTED_COMPACT_CHECKPOINT_BYTES
            ),
            "known_package": KNOWN_PACKAGE,
        },
        "environment": collect_environment(),
        "checkpoint": checkpoint_report,
        "package": package_report,
    }


def parse_memory_stdout(stdout):
    lifecycle = []
    max_vram_mb = None
    current_vram_mb = None
    patched_modules = None
    profile = None
    for line in stdout.splitlines():
        match = MEMORY_RE.match(line.strip())
        if match:
            lifecycle.append(
                {
                    "tag": match.group("tag"),
                    "allocated_mb": float(match.group("allocated")),
                    "reserved_mb": float(match.group("reserved")),
                    "peak_mb": float(match.group("peak")),
                }
            )
        match = re.search(r"Max VRAM:\s*([0-9.]+)\s*MB", line)
        if match:
            max_vram_mb = float(match.group(1))
        match = re.search(r"Current VRAM:\s*([0-9.]+)\s*MB", line)
        if match:
            current_vram_mb = float(match.group(1))
        match = re.search(r"patched=(\d+)", line)
        if match and "P2" in line:
            patched_modules = int(match.group(1))
        match = re.search(
            r"profile=(\S+)\s+patch=\[([^]]+)\]\s+embed=(\d+)", line
        )
        if match:
            profile = {
                "name": match.group(1),
                "patch_size": [int(value.strip()) for value in match.group(2).split(",")],
                "embed_dim": int(match.group(3)),
            }
    return {
        "lifecycle": lifecycle,
        "max_vram_mb": max_vram_mb,
        "current_vram_mb": current_vram_mb,
        "patched_modules": patched_modules,
        "profile": profile,
    }


def _prepare_run_directory(path, pangu_dir):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for name in ("conf", "data", "pangu_backups"):
        source = pangu_dir / name
        if source.exists():
            (path / name).symlink_to(source, target_is_directory=True)
    (path / "result" / "output").mkdir(parents=True)
    return path


def _variant_env(checkpoint, p2_enabled, max_batches, profile_memory=False):
    return {
        "PANGU_CHECKPOINT": str(Path(checkpoint).resolve()),
        "PANGU_AUTO_SCAN_CHECKPOINT": "0",
        "PANGU_MAX_INFERENCE_BATCHES": str(max_batches),
        "PANGU_P2_TILED_ATTENTION": "1" if p2_enabled else "0",
        "PANGU_P2_TILED_MODE": "full-row-fast" if p2_enabled else "online",
        "PANGU_P2_FULL_WIDTH": "1",
        "PANGU_P2_TILED_DEBUG": "0",
        "PANGU_PROFILE_MEMORY": "1" if profile_memory else "0",
        "PANGU_INTERN_IMMUTABLE_BUFFERS": "1",
        "PANGU_DISABLE_CUDA_GRAPH": "1",
    }


def _read_time_record(run_dir):
    path = Path(run_dir) / "result" / "time_record.json"
    if not path.is_file():
        return []
    values = load_json(path)
    return [float(value) * 1000.0 for value in values]


def run_process(
    *,
    python,
    pangu_dir,
    run_dir,
    env_overrides,
    worker_output=None,
):
    env = os.environ.copy()
    env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pangu_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if worker_output is None:
        command = [python, str(pangu_dir / "inference.py")]
    else:
        command = [
            python,
            str(Path(__file__).resolve()),
            "_worker",
            "--pangu-dir",
            str(pangu_dir),
            "--worker-output",
            str(worker_output),
        ]
    env["PYTHONUNBUFFERED"] = "1"
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(run_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_lines = []
    memory_timeline = []
    for line in process.stdout:
        output_lines.append(line)
        match = MEMORY_RE.match(line.strip())
        if match:
            memory_timeline.append(
                {
                    "elapsed_s": time.perf_counter() - started,
                    "allocated_mb": float(match.group("allocated")),
                }
            )
    returncode = process.wait()
    wall_time_s = time.perf_counter() - started
    stdout = "".join(output_lines)
    parsed = parse_memory_stdout(stdout)
    time_weighted_proxy = None
    if memory_timeline:
        weighted = 0.0
        start_time = memory_timeline[0]["elapsed_s"]
        for index, sample in enumerate(memory_timeline):
            end_time = (
                memory_timeline[index + 1]["elapsed_s"]
                if index + 1 < len(memory_timeline)
                else wall_time_s
            )
            weighted += sample["allocated_mb"] * max(
                0.0, end_time - sample["elapsed_s"]
            )
        duration = max(0.0, wall_time_s - start_time)
        if duration > 0:
            time_weighted_proxy = weighted / duration
    return {
        "returncode": returncode,
        "wall_time_s": wall_time_s,
        "stdout_tail": "\n".join(stdout.splitlines()[-120:]),
        "latency_ms": _read_time_record(run_dir),
        "output_dir": str(Path(run_dir) / "result" / "output"),
        "time_weighted_allocated_proxy_mb": time_weighted_proxy,
        "memory_timeline": memory_timeline,
        **parsed,
    }


def compare_output_dirs(candidate_dir, oracle_dir, include_hashes=False):
    import numpy as np

    candidate_dir = Path(candidate_dir)
    oracle_dir = Path(oracle_dir)
    oracle_files = sorted(oracle_dir.glob("*.npy"))
    candidate_files = {path.name: path for path in candidate_dir.glob("*.npy")}
    mismatch_count = 0
    max_abs = 0.0
    max_rel = 0.0
    nan_count = 0
    inf_count = 0
    all_69 = bool(oracle_files)
    files = []
    for oracle_path in oracle_files:
        candidate_path = candidate_files.get(oracle_path.name)
        if candidate_path is None:
            all_69 = False
            files.append({"name": oracle_path.name, "missing_candidate": True})
            continue
        oracle = np.load(oracle_path, mmap_mode="r")
        candidate = np.load(candidate_path, mmap_mode="r")
        shape_equal = oracle.shape == candidate.shape
        dtype_equal = oracle.dtype == candidate.dtype
        channels = int(oracle.shape[1]) if oracle.ndim >= 2 else None
        all_69 = all_69 and shape_equal and channels == 69
        if not shape_equal:
            mismatch_count += max(oracle.size, candidate.size)
            files.append(
                {
                    "name": oracle_path.name,
                    "oracle_shape": list(oracle.shape),
                    "candidate_shape": list(candidate.shape),
                    "shape_equal": False,
                }
            )
            continue
        for channel_index in range(oracle.shape[1]):
            oracle_chunk = np.asarray(oracle[:, channel_index])
            candidate_chunk = np.asarray(candidate[:, channel_index])
            mismatch_count += int(
                np.count_nonzero(candidate_chunk != oracle_chunk)
            )
            delta = np.abs(
                candidate_chunk.astype(np.float32)
                - oracle_chunk.astype(np.float32)
            )
            finite_delta = delta[np.isfinite(delta)]
            if finite_delta.size:
                max_abs = max(max_abs, float(np.max(finite_delta)))
                denominator = np.maximum(
                    np.abs(oracle_chunk.astype(np.float32)), 1.0e-6
                )
                relative = delta / denominator
                finite_relative = relative[np.isfinite(relative)]
                if finite_relative.size:
                    max_rel = max(max_rel, float(np.max(finite_relative)))
            nan_count += int(np.isnan(candidate_chunk).sum())
            inf_count += int(np.isinf(candidate_chunk).sum())
        files.append(
            {
                "name": oracle_path.name,
                "shape": list(oracle.shape),
                "dtype": str(oracle.dtype),
                "dtype_equal": bool(dtype_equal),
                "oracle_sha256": (
                    sha256_file(oracle_path) if include_hashes else None
                ),
                "candidate_sha256": (
                    sha256_file(candidate_path) if include_hashes else None
                ),
            }
        )
    all_dtypes_equal = all(
        item.get("dtype_equal", False)
        for item in files
        if not item.get("missing_candidate") and item.get("shape_equal", True)
    )
    exact = (
        bool(oracle_files)
        and len(candidate_files) == len(oracle_files)
        and mismatch_count == 0
        and nan_count == 0
        and inf_count == 0
        and all_69
        and all_dtypes_equal
    )
    return {
        "exact": exact,
        "output_files": len(oracle_files),
        "candidate_output_files": len(candidate_files),
        "all_outputs_have_69_channels": all_69,
        "all_dtypes_equal": all_dtypes_equal,
        "mismatch_count": mismatch_count,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "files": files,
    }


class _EventCollector:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.current_forward = 0
        self._stacks = defaultdict(list)
        self._pairs = []

    def start(self, category, label):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        self._stacks[(category, label)].append(
            (event, self.current_forward)
        )

    def stop(self, category, label):
        stack = self._stacks[(category, label)]
        if not stack:
            raise RuntimeError(f"Missing CUDA event start for {category}/{label}")
        start, forward_index = stack.pop()
        stop = self.torch.cuda.Event(enable_timing=True)
        stop.record()
        self._pairs.append((category, label, forward_index, start, stop))

    def summarize(self):
        self.torch.cuda.synchronize()
        raw = []
        aggregate = defaultdict(list)
        for category, label, forward_index, start, stop in self._pairs:
            elapsed = float(start.elapsed_time(stop))
            phase = "cold" if forward_index <= 1 else "steady"
            raw.append(
                {
                    "category": category,
                    "label": label,
                    "forward_index": forward_index,
                    "phase": phase,
                    "elapsed_ms": elapsed,
                }
            )
            aggregate[(category, label, phase)].append(elapsed)
        rows = []
        steady_forward_count = max(0, self.current_forward - 1)
        for (category, label, phase), values in aggregate.items():
            rows.append(
                {
                    "category": category,
                    "label": label,
                    "phase": phase,
                    "calls": len(values),
                    "total_ms": sum(values),
                    "mean_call_ms": statistics.fmean(values),
                    "per_forward_mean_ms": (
                        sum(values) / steady_forward_count
                        if phase == "steady" and steady_forward_count
                        else sum(values)
                    ),
                }
            )
        rows.sort(key=lambda item: item["total_ms"], reverse=True)
        return {
            "forward_count": self.current_forward,
            "steady_forward_count": steady_forward_count,
            "events": raw,
            "summary": rows,
        }


def _register_timed_hook(module, collector, category, label, handles):
    def before(_module, _inputs):
        collector.start(category, label)

    def after(_module, _inputs, _output):
        collector.stop(category, label)

    handles.append(module.register_forward_pre_hook(before))
    handles.append(module.register_forward_hook(after))


def _install_attribution_hooks(model, collector):
    handles = []

    def root_before(_module, _inputs):
        collector.current_forward += 1
        collector.start("forward", "model")

    def root_after(_module, _inputs, _output):
        collector.stop("forward", "model")

    handles.append(model.register_forward_pre_hook(root_before))
    handles.append(model.register_forward_hook(root_after))

    for name in ("patchembed2d", "patchembed3d"):
        module = getattr(model, name, None)
        if module is not None:
            _register_timed_hook(module, collector, "embed", name, handles)
    for name in ("downsample", "upsample"):
        module = getattr(model, name, None)
        if module is not None:
            _register_timed_hook(module, collector, "stage", name, handles)
    for name in ("patchrecovery2d", "patchrecovery3d"):
        module = getattr(model, name, None)
        if module is not None:
            _register_timed_hook(module, collector, "recovery", name, handles)

    for module_name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name == "EarthTransformer3DBlock":
            _register_timed_hook(module, collector, "block", module_name, handles)
        if class_name == "EarthAttention3D":
            _register_timed_hook(module, collector, "attention", module_name, handles)
            if hasattr(module, "qkv"):
                _register_timed_hook(
                    module.qkv, collector, "qkv", f"{module_name}.qkv", handles
                )
            if hasattr(module, "proj"):
                _register_timed_hook(
                    module.proj, collector, "projection", f"{module_name}.proj", handles
                )
        if module_name.endswith(".mlp"):
            _register_timed_hook(module, collector, "mlp", module_name, handles)
        elif module_name.endswith((".norm1", ".norm2")):
            _register_timed_hook(module, collector, "norm", module_name, handles)
    return handles


def _tensor_record(module_name, kind, tensor):
    storage = tensor.untyped_storage()
    storage_key = f"{tensor.device}:{storage.data_ptr()}:{storage.nbytes()}"
    return {
        "module": module_name,
        "kind": kind,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "logical_bytes": int(tensor.numel() * tensor.element_size()),
        "storage_bytes": int(storage.nbytes()),
        "storage_key": storage_key,
        "tensor_data_ptr": int(tensor.data_ptr()),
    }


def _collect_p2_cache_inventory(model, torch_module):
    records = []
    patched = 0
    for module_name, module in model.named_modules():
        if not hasattr(module, "_pangu_p2_original_forward"):
            continue
        patched += 1
        for attribute in ("earth_position_bias_table", "earth_position_index"):
            value = getattr(module, attribute, None)
            if isinstance(value, torch_module.Tensor):
                records.append(
                    _tensor_record(
                        module_name,
                        classify_cache_attribute(attribute),
                        value,
                    )
                )
        bias_index = getattr(module, "_pangu_p2_tiled_bias_index", None)
        if isinstance(bias_index, tuple):
            for index in (1, 2):
                if index < len(bias_index) and isinstance(
                    bias_index[index], torch_module.Tensor
                ):
                    records.append(
                        _tensor_record(
                            module_name,
                            classify_cache_attribute(
                                "_pangu_p2_tiled_bias_index", index
                            ),
                            bias_index[index],
                        )
                    )
        region_ids = getattr(module, "_pangu_p2_tiled_region_ids", None)
        if (
            isinstance(region_ids, tuple)
            and len(region_ids) > 1
            and isinstance(region_ids[1], torch_module.Tensor)
        ):
            records.append(
                _tensor_record(
                    module_name,
                    classify_cache_attribute("_pangu_p2_tiled_region_ids", 1),
                    region_ids[1],
                )
            )
    return build_cache_summary(records, patched)


def _worker_main(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pangu-dir", required=True)
    parser.add_argument("--worker-output", required=True)
    args = parser.parse_args(argv)
    pangu_dir = Path(args.pangu_dir).resolve()
    sys.path.insert(0, str(pangu_dir))

    import torch
    import p2_tiled_attention
    import pangu_profile_model

    collector = _EventCollector(torch)
    model_holder = {}
    handles = []
    original_enable = p2_tiled_attention.enable_p2_tiled_attention
    original_backend = p2_tiled_attention._backend
    original_run_fuser = pangu_profile_model._run_fuser_layerwise
    original_embed_sequence = pangu_profile_model._embed_sequence
    original_recover_outputs = pangu_profile_model._recover_outputs

    def timed_backend():
        compact_index, tiled_forward, pack_bias, mask_to_regions = original_backend()

        def timed_tiled_forward(*call_args, **call_kwargs):
            collector.start("p2_kernel", "earth_attention_tiled")
            try:
                return tiled_forward(*call_args, **call_kwargs)
            finally:
                collector.stop("p2_kernel", "earth_attention_tiled")

        return compact_index, timed_tiled_forward, pack_bias, mask_to_regions

    def diagnostic_enable(model, *call_args, **call_kwargs):
        patched = original_enable(model, *call_args, **call_kwargs)
        model_holder["model"] = model
        handles.extend(_install_attribution_hooks(model, collector))
        return patched

    def timed_run_fuser(owner, fuser, x, empty_cache=False, label=None):
        stage_label = label or "fuser"
        collector.start("stage", stage_label)
        try:
            return original_run_fuser(owner, fuser, x, empty_cache, label)
        finally:
            collector.stop("stage", stage_label)

    def timed_embed_sequence(model, inputs):
        collector.start("stage", "embed_sequence")
        try:
            return original_embed_sequence(model, inputs)
        finally:
            collector.stop("stage", "embed_sequence")

    def timed_recover_outputs(model, sequence, batch, pressure, height, width):
        collector.start("stage", "recovery")
        try:
            return original_recover_outputs(
                model, sequence, batch, pressure, height, width
            )
        finally:
            collector.stop("stage", "recovery")

    p2_tiled_attention._backend = timed_backend
    p2_tiled_attention.enable_p2_tiled_attention = diagnostic_enable
    pangu_profile_model._run_fuser_layerwise = timed_run_fuser
    pangu_profile_model._embed_sequence = timed_embed_sequence
    pangu_profile_model._recover_outputs = timed_recover_outputs
    payload = {"schema_version": SCHEMA_VERSION, "generated_at": utc_now()}
    try:
        runpy.run_path(str(pangu_dir / "inference.py"), run_name="__main__")
        model = model_holder.get("model")
        if model is None:
            raise RuntimeError("Attribution worker did not capture the P2 model")
        payload["patched_modules"] = int(
            getattr(model, "_pangu_p2_tiled_attention_count", 0)
        )
        payload["event_attribution"] = collector.summarize()
        payload["p2_cache_inventory"] = _collect_p2_cache_inventory(model, torch)
        payload["success"] = True
    except Exception as error:
        payload.update(
            {
                "success": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(args.worker_output, payload)
        raise
    finally:
        for handle in handles:
            handle.remove()
        p2_tiled_attention._backend = original_backend
        p2_tiled_attention.enable_p2_tiled_attention = original_enable
        pangu_profile_model._run_fuser_layerwise = original_run_fuser
        pangu_profile_model._embed_sequence = original_embed_sequence
        pangu_profile_model._recover_outputs = original_recover_outputs
    write_json(args.worker_output, payload)
    return 0


def _clean_run_record(process_result, round_index, variant, order_index):
    values = process_result["latency_ms"]
    return {
        "round": round_index,
        "variant": variant,
        "order_index": order_index,
        "returncode": process_result["returncode"],
        "wall_time_s": process_result["wall_time_s"],
        "cold_latency_ms": values[0] if values else None,
        "steady_latency_ms": values[1:] if len(values) > 1 else [],
        "all_latency_ms": values,
        "max_vram_mb": process_result["max_vram_mb"],
        "current_vram_mb": process_result["current_vram_mb"],
        "profile": process_result["profile"],
        "patched_modules": process_result["patched_modules"],
        "output_dir": process_result["output_dir"],
        "outputs_removed_after_compare": False,
        "stdout_tail": process_result["stdout_tail"],
    }


def _require_process_success(result, label):
    if result["returncode"] != 0:
        raise RuntimeError(
            f"{label} failed with return code {result['returncode']}:\n"
            + result["stdout_tail"]
        )
    if len(result["latency_ms"]) < 2:
        raise RuntimeError(f"{label} did not produce cold and steady latency samples")


def run_runtime(checkpoint, repeat, max_batches, python, output_dir, pangu_dir):
    if repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if max_batches < 2:
        raise ValueError("--max-batches must be at least 2")
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    work_root = output_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    clean_runs = []
    correctness = []
    for round_index in range(1, repeat + 1):
        order = (
            ("p2_off", "p2_on")
            if round_index % 2 == 1
            else ("p2_on", "p2_off")
        )
        round_records = {}
        for order_index, variant in enumerate(order, start=1):
            run_dir = _prepare_run_directory(
                work_root / f"round_{round_index:02d}_{variant}", pangu_dir
            )
            result = run_process(
                python=python,
                pangu_dir=pangu_dir,
                run_dir=run_dir,
                env_overrides=_variant_env(
                    checkpoint,
                    p2_enabled=variant == "p2_on",
                    max_batches=max_batches,
                    profile_memory=False,
                ),
            )
            _require_process_success(result, f"round {round_index} {variant}")
            record = _clean_run_record(
                result, round_index, variant, order_index
            )
            clean_runs.append(record)
            round_records[variant] = record
        comparison = {
            "round": round_index,
            **compare_output_dirs(
                round_records["p2_on"]["output_dir"],
                round_records["p2_off"]["output_dir"],
                include_hashes=round_index == 1,
            ),
        }
        correctness.append(comparison)
        if comparison["exact"]:
            for record in round_records.values():
                shutil.rmtree(record["output_dir"])
                record["outputs_removed_after_compare"] = True

    by_variant = {}
    for variant in ("p2_off", "p2_on"):
        records = [record for record in clean_runs if record["variant"] == variant]
        cold = [record["cold_latency_ms"] for record in records]
        steady = [
            value for record in records for value in record["steady_latency_ms"]
        ]
        max_values = [
            record["max_vram_mb"]
            for record in records
            if record["max_vram_mb"] is not None
        ]
        by_variant[variant] = {
            "cold": summarize_samples(cold),
            "steady": summarize_samples(steady),
            "max_vram_mb": max(max_values) if max_values else None,
            "current_vram_mb": records[-1]["current_vram_mb"],
        }

    paired_on = []
    paired_off = []
    for round_index in range(1, repeat + 1):
        off = next(
            record for record in clean_runs
            if record["round"] == round_index and record["variant"] == "p2_off"
        )
        on = next(
            record for record in clean_runs
            if record["round"] == round_index and record["variant"] == "p2_on"
        )
        paired_off.extend(off["steady_latency_ms"])
        paired_on.extend(on["steady_latency_ms"])
    paired_ci = paired_bootstrap_ci(paired_on, paired_off)

    memory_runs = {}
    off_dir = _prepare_run_directory(work_root / "memory_p2_off", pangu_dir)
    off_result = run_process(
        python=python,
        pangu_dir=pangu_dir,
        run_dir=off_dir,
        env_overrides=_variant_env(
            checkpoint, p2_enabled=False, max_batches=max_batches, profile_memory=True
        ),
    )
    _require_process_success(off_result, "P2-off memory lifecycle")
    shutil.rmtree(off_result["output_dir"])
    memory_runs["p2_off"] = {
        key: off_result[key]
        for key in (
            "lifecycle",
            "max_vram_mb",
            "current_vram_mb",
            "time_weighted_allocated_proxy_mb",
            "memory_timeline",
            "stdout_tail",
        )
    }

    on_dir = _prepare_run_directory(work_root / "attribution_p2_on", pangu_dir)
    worker_output = on_dir / "attribution.json"
    on_result = run_process(
        python=python,
        pangu_dir=pangu_dir,
        run_dir=on_dir,
        env_overrides=_variant_env(
            checkpoint, p2_enabled=True, max_batches=max_batches, profile_memory=True
        ),
        worker_output=worker_output,
    )
    _require_process_success(on_result, "P2-on attribution")
    worker = load_json(worker_output)
    if not worker.get("success"):
        raise RuntimeError(f"P2-on attribution failed: {worker.get('error')}")
    shutil.rmtree(on_result["output_dir"])
    memory_runs["p2_on"] = {
        key: on_result[key]
        for key in (
            "lifecycle",
            "max_vram_mb",
            "current_vram_mb",
            "time_weighted_allocated_proxy_mb",
            "memory_timeline",
            "stdout_tail",
        )
    }

    p2_mean = by_variant["p2_on"]["steady"]["mean_ms"]
    drift = assess_baseline_drift(
        p2_mean,
        memory_runs["p2_on"]["max_vram_mb"],
        memory_runs["p2_on"]["current_vram_mb"],
    )
    correctness_result = correctness_gate(correctness)
    patched_modules = worker.get("patched_modules")
    exact_profile = all(
        record.get("profile", {}).get("name") == "pgw_lite_pruned_96"
        for record in clean_runs
    )
    integrity = {
        "exact_profile": exact_profile,
        "patched_modules": patched_modules,
        "patched_modules_passed": patched_modules == 16,
        "correctness": correctness_result,
        "baseline_drift": drift,
    }
    integrity["passed"] = (
        exact_profile
        and patched_modules == 16
        and correctness_result["passed"]
        and drift["passed"]
    )

    cache_inventory = worker["p2_cache_inventory"]
    memory_delta = None
    if (
        memory_runs["p2_on"]["current_vram_mb"] is not None
        and memory_runs["p2_off"]["current_vram_mb"] is not None
    ):
        memory_delta = (
            memory_runs["p2_on"]["current_vram_mb"]
            - memory_runs["p2_off"]["current_vram_mb"]
        )
    cache_bytes = sum(
        cache_inventory.get("by_kind", {}).get(kind, {}).get(
            "unique_storage_bytes", 0
        )
        for kind in ("packed_bias", "compact_index", "region_ids")
    )
    cache_inventory["observed_current_delta_mb"] = memory_delta
    cache_inventory["cache_explained_pct"] = (
        100.0 * cache_bytes / (memory_delta * MIB)
        if memory_delta is not None and memory_delta > 0
        else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "configuration": {
            "checkpoint": str(checkpoint),
            "repeat": repeat,
            "max_batches": max_batches,
            "paired_order_alternates": True,
            "clean_pass_instrumented": False,
            "attribution_pass_separate": True,
            "oracle_variant": "p2_off",
        },
        "clean_runs": clean_runs,
        "variants": by_variant,
        "paired_bootstrap_95_ci": paired_ci,
        "correctness_rounds": correctness,
        "memory_lifecycle": memory_runs,
        "event_attribution": worker["event_attribution"],
        "p2_cache_inventory": cache_inventory,
        "integrity": integrity,
    }


def _event_candidates(runtime):
    allowed = {"qkv", "p2_kernel", "projection", "mlp"}
    rows = [
        row
        for row in runtime.get("event_attribution", {}).get("summary", [])
        if row.get("phase") == "steady" and row.get("category") in allowed
    ]
    rows.sort(key=lambda row: row.get("per_forward_mean_ms", 0), reverse=True)
    return rows


def build_diagnosis(static, runtime):
    static_profile = static.get("checkpoint", {}).get("profile", {})
    static_profile_passed = all(
        static_profile.get(key) == expected
        for key, expected in EXPECTED_PROFILE.items()
    )
    integrity_passed = bool(
        runtime.get("integrity", {}).get("passed") and static_profile_passed
    )
    p2_mean = runtime["variants"]["p2_on"]["steady"]["mean_ms"]
    off_mean = runtime["variants"]["p2_off"]["steady"]["mean_ms"]
    latency_reduction_pct = 100.0 * (off_mean - p2_mean) / off_mean
    event_candidates = _event_candidates(runtime)
    top_time = event_candidates[0] if event_candidates else None

    cache = runtime.get("p2_cache_inventory", {})
    packed_mb = (
        cache.get("by_kind", {})
        .get("packed_bias", {})
        .get("unique_storage_bytes", 0)
        / MIB
    )
    recommendations = []
    if integrity_passed:
        recommendations.append(
            {
                "priority": 1,
                "metric": "V/lightweight",
                "direction": "eliminate_packed_bias_double_residency",
                "evidence": f"{packed_mb:.3f} MiB unique packed-bias storage",
                "target": "recover at least 16 MiB with <=1% latency regression",
            }
        )
        if top_time is not None:
            recommendations.append(
                {
                    "priority": 2,
                    "metric": "U/inference_time",
                    "direction": top_time["category"],
                    "label": top_time["label"],
                    "evidence": (
                        f"{top_time['per_forward_mean_ms']:.4f} ms per steady forward"
                    ),
                    "target": (
                        ">=2.5% mean improvement, CI below zero, no P90 regression, exact output"
                    ),
                }
            )
    else:
        recommendations.append(
            {
                "priority": 1,
                "metric": "guardrail",
                "direction": "resolve_integrity_or_baseline_drift",
                "evidence": runtime.get("integrity"),
                "target": "all fail-closed integrity gates must pass before ranking",
            }
        )

    amdahl = []
    if integrity_passed:
        for row in event_candidates[:8]:
            for factor in (1.25, 1.5, 2.0):
                saved = row["per_forward_mean_ms"] * (1.0 - 1.0 / factor)
                amdahl.append(
                    {
                        "category": row["category"],
                        "label": row["label"],
                        "speedup_factor": factor,
                        "estimated_saved_ms": saved,
                        "estimated_end_to_end_latency_ms": max(0.0, p2_mean - saved),
                        "estimated_end_to_end_reduction_pct": 100.0 * saved / p2_mean,
                    }
                )

    projections = None
    if integrity_passed:
        projections = {
            "score_inference_time_targets": latency_score_targets(p2_mean),
            "observed_p2_latency_reduction_pct": latency_reduction_pct,
            "observed_p2_score_delta_proxy": 2.0 * latency_reduction_pct / 100.0,
            "amdahl_candidates": amdahl,
            "warning": "Proxy only; official platform scoring remains authoritative.",
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "score_reference": SCORE_REFERENCE,
        "provenance": static.get("provenance"),
        "static": static,
        "runtime": runtime,
        "diagnosis": {
            "valid_for_ranking": integrity_passed,
            "static_profile_passed": static_profile_passed,
            "latency_reduction_pct": latency_reduction_pct,
            "projections": projections,
            "recommendations": recommendations,
            "rejected_directions": [
                {
                    "direction": "P2 QKV/projection width chunk",
                    "reason": (
                        "Previously measured slower at 80.0917 ms and did not recover persistent memory."
                    ),
                }
            ],
        },
    }


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _numeric_delta(candidate, oracle):
    if candidate is None or oracle is None:
        return None
    return float(candidate) - float(oracle)


def render_markdown(diagnosis):
    runtime = diagnosis["runtime"]
    integrity = runtime["integrity"]
    off = runtime["variants"]["p2_off"]["steady"]
    on = runtime["variants"]["p2_on"]["steady"]
    ci = runtime["paired_bootstrap_95_ci"]
    cache = runtime["p2_cache_inventory"]
    memory = runtime["memory_lifecycle"]
    lines = [
        "# 90.1743 U/V 深度诊断",
        "",
        "## 结论",
        "",
    ]
    for item in diagnosis["diagnosis"]["recommendations"]:
        lines.append(
            f"- P{item['priority']} **{item['metric']}**：`{item['direction']}`；"
            f"证据：{item.get('evidence')}；门槛：{item['target']}。"
        )
    lines.extend(
        [
            "",
            "## U：推理时长",
            "",
            "| 变体 | steady 样本 | mean ms | P50 ms | P90 ms | CV |",
            "|---|---:|---:|---:|---:|---:|",
            f"| P2-off oracle | {off['count']} | {_fmt(off['mean_ms'])} | {_fmt(off['p50_ms'])} | {_fmt(off['p90_ms'])} | {_fmt(off['cv'], 6)} |",
            f"| P2-current | {on['count']} | {_fmt(on['mean_ms'])} | {_fmt(on['p50_ms'])} | {_fmt(on['p90_ms'])} | {_fmt(on['cv'], 6)} |",
            "",
            f"- 配对差值（P2-on - P2-off）：`{_fmt(ci['mean_delta_ms'])} ms`，"
            f"bootstrap 95% CI `[{_fmt(ci['lower_95_ms'])}, {_fmt(ci['upper_95_ms'])}] ms`。",
            f"- baseline drift：`{'PASS' if integrity['baseline_drift']['passed'] else 'FAIL'}`。",
            "",
            "### CUDA Event steady waterfall（inclusive）",
            "",
            "| 类别 | 模块 | 每 forward ms | calls |",
            "|---|---|---:|---:|",
        ]
    )
    for row in _event_candidates(runtime)[:15]:
        lines.append(
            f"| {row['category']} | `{row['label']}` | "
            f"{_fmt(row['per_forward_mean_ms'])} | {row['calls']} |"
        )

    by_kind = cache.get("by_kind", {})
    lines.extend(
        [
            "",
            "## V：模型轻量化",
            "",
            "| 项目 | P2-off | P2-on | 差值 |",
            "|---|---:|---:|---:|",
            f"| peak allocated MiB | {_fmt(memory['p2_off']['max_vram_mb'], 1)} | {_fmt(memory['p2_on']['max_vram_mb'], 1)} | {_fmt(_numeric_delta(memory['p2_on']['max_vram_mb'], memory['p2_off']['max_vram_mb']), 1)} |",
            f"| current allocated MiB | {_fmt(memory['p2_off']['current_vram_mb'], 1)} | {_fmt(memory['p2_on']['current_vram_mb'], 1)} | {_fmt(cache.get('observed_current_delta_mb'), 1)} |",
            f"| time-weighted allocated proxy MiB | {_fmt(memory['p2_off'].get('time_weighted_allocated_proxy_mb'), 1)} | {_fmt(memory['p2_on'].get('time_weighted_allocated_proxy_mb'), 1)} | n/a |",
            "",
            "| P2 常驻项 | unique MiB |",
            "|---|---:|",
        ]
    )
    for kind in ("packed_bias", "compact_index", "region_ids"):
        lines.append(
            f"| {kind} | {_fmt(by_kind.get(kind, {}).get('unique_storage_mb'))} |"
        )
    lines.extend(
        [
            "",
            f"缓存对 current 增量的解释比例：`{_fmt(cache.get('cache_explained_pct'), 2)}%`。",
            "",
            "## 守门状态",
            "",
            f"- exact profile：`{integrity['exact_profile']}`",
            f"- static full profile：`{diagnosis['diagnosis']['static_profile_passed']}`",
            f"- P2 patched modules：`{integrity['patched_modules']}` / 16",
            f"- 69 通道 bitwise exact：`{integrity['correctness']['passed']}`",
            f"- 可进行方向排名：`{diagnosis['diagnosis']['valid_for_ranking']}`",
            "",
            "已否决方向：不重新尝试 P2 QKV/projection width chunk；历史结果更慢且未回收常驻缓存。",
        ]
    )
    projections = diagnosis["diagnosis"].get("projections")
    if projections:
        lines.extend(
            [
                "",
                "## U 分数弹性（本地 proxy）",
                "",
                "| ΔU | 所需时延下降 | 目标时延 ms |",
                "|---:|---:|---:|",
            ]
        )
        for target in projections["score_inference_time_targets"]:
            lines.append(
                f"| +{target['score_gain']:.2f} | "
                f"{target['required_latency_reduction_pct']:.2f}% | "
                f"{target['target_latency_ms']:.4f} |"
            )
        lines.append("")
        lines.append("分数外推仅为 proxy，正式平台结果始终优先。")
    return "\n".join(lines)


def _resolve_path(path, pangu_dir):
    path = Path(path)
    return path if path.is_absolute() else pangu_dir / path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return _worker_main(argv[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("static", "runtime", "report", "all"))
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints/model_fp16_alias_compact.pth",
    )
    parser.add_argument("--package-zip", default=None)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="logs/diagnose_uv_90_1743")
    args = parser.parse_args(argv)

    pangu_dir = Path(__file__).resolve().parents[1]
    checkpoint = _resolve_path(args.checkpoint, pangu_dir)
    package_zip = (
        _resolve_path(args.package_zip, pangu_dir)
        if args.package_zip is not None
        else None
    )
    output_dir = _resolve_path(args.output_dir, pangu_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    static_path = output_dir / "static.json"
    runtime_path = output_dir / "runtime.json"

    static = None
    runtime = None
    if args.command in {"static", "all"}:
        static = run_static(checkpoint, package_zip, pangu_dir)
        write_json(static_path, static)
        print(f"wrote {static_path}")
    if args.command in {"runtime", "all"}:
        runtime = run_runtime(
            checkpoint=checkpoint,
            repeat=args.repeat,
            max_batches=args.max_batches,
            python=args.python,
            output_dir=output_dir,
            pangu_dir=pangu_dir,
        )
        write_json(runtime_path, runtime)
        print(f"wrote {runtime_path}")
    if args.command in {"report", "all"}:
        static = static if static is not None else load_json(static_path)
        runtime = runtime if runtime is not None else load_json(runtime_path)
        diagnosis = build_diagnosis(static, runtime)
        diagnosis_path = output_dir / "diagnosis.json"
        markdown_path = output_dir / "diagnosis.md"
        write_json(diagnosis_path, diagnosis)
        write_text(markdown_path, render_markdown(diagnosis))
        print(f"wrote {diagnosis_path}")
        print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
