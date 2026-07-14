#!/usr/bin/env python3
"""Run controlled U/V runtime probes for Pangu-Weather inference.

The script intentionally drives the existing inference.py entrypoint in a
subprocess so timing boundaries, model loading, and output generation stay
identical to the submission path.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
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

P2_RUNTIME_CANDIDATES = [
    {"PANGU_HIP_NEAREST_CPU": "1"},
    {"PANGU_HIP_SHARED_MEM_BANK_BYTES": "8"},
    {"PANGU_HIP_STREAM_PRIORITY": "greatest"},
]

P2_RUNTIME_CONTROL_ENV = {
    "nearest": {"PANGU_HIP_NEAREST_CPU": "1"},
    "bank8": {"PANGU_HIP_SHARED_MEM_BANK_BYTES": "8"},
    "priority": {"PANGU_HIP_STREAM_PRIORITY": "greatest"},
    "stream-spin": {"PANGU_HIP_STREAM_SPIN": "1"},
}

P2_PLATFORM_KERNEL_FLAGS = (
    "-DPANGU_FULL_ROW_DIRECT_SCORE_STORE=1 "
    "-DPANGU_FULL_ROW_PV_DOUBLE_BUFFER=1"
)

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
    "PANGU_HIP_NEAREST_CPU": "0",
    "PANGU_HIP_SHARED_MEM_BANK_BYTES": "0",
    "PANGU_HIP_STREAM_PRIORITY": "0",
    # P2 is an explicit full-model A/B candidate; keep every other preset
    # production-safe even if the caller's shell exports the flag.
    "PANGU_P2_TILED_ATTENTION": "0",
    "PANGU_P2_TILED_MODE": "online",
    "PANGU_P2_FULL_WIDTH": "1",
    "PANGU_P2_RELEASE_ORIGINAL_BIAS": "0",
    "PANGU_P2_RETAIN_CPU_BIAS_BACKUP": "0",
    "PANGU_P2_FULL_ROW_SCORE_STRIDE": "144",
    "PANGU_P2_FULL_ROW_QK_TILE": "16",
    "PANGU_P2_REGION_RELEASE": "0",
    "PANGU_P2_PREBUILD_HIP": "0",
    # Every full-model row owns its compiler flags. Never inherit an isolated
    # kernel experiment from the caller's shell into both sides of an A/B.
    "PANGU_TILED_HIP_EXTRA_FLAGS": "",
    # Buffer interning is part of the compliant 90.7763 guardrail default.
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

INTERLEAVED_PRESETS = {
    "p2-bias-reclaim",
    "p2-runtime",
    "p2-stride",
    "p2-kernel-pipeline",
    "p2-qk32",
    "p2-region-release",
    "p2-hip-prebuild",
}

FROZEN_OFFICIAL_TIMER_SHA256 = (
    "fa7d46a8ea3a3da93f5348bbb6b237409da16a68b20708331d4d9b0f4adb61ad"
)


def should_run_interleaved(preset, candidate_checkpoint=None):
    return preset in INTERLEAVED_PRESETS or candidate_checkpoint is not None


def official_timer_block_sha256(inference_path):
    source = Path(inference_path).read_text(encoding="utf-8")
    loop = source.index("for batch_index, data in enumerate")
    marker = "#----------------------AI4S(时间度量不可更改)---------------------------"
    start = source.index(marker, loop)
    end_marker = "#---------------------------------------------------------------------"
    end = source.index(end_marker, start) + len(end_marker)
    return hashlib.sha256(source[start:end].encode()).hexdigest()


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
        f"_near{env['PANGU_HIP_NEAREST_CPU']}"
        f"_bank{env['PANGU_HIP_SHARED_MEM_BANK_BYTES']}"
        f"_prio{env['PANGU_HIP_STREAM_PRIORITY']}"
        f"_p2{env.get('PANGU_P2_TILED_ATTENTION', '0')}"
        f"_bias{env.get('PANGU_P2_RELEASE_ORIGINAL_BIAS', '0')}"
        f"_stride{env.get('PANGU_P2_FULL_ROW_SCORE_STRIDE', '144')}"
        f"_qkt{env.get('PANGU_P2_FULL_ROW_QK_TILE', '16')}"
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

    if preset == "p2-runtime":
        baseline_env = dict(BASE_ENV)
        baseline_env.update(
            {
                "PANGU_P2_TILED_ATTENTION": "1",
                "PANGU_P2_TILED_MODE": "full-row-fast",
                "PANGU_P2_RELEASE_ORIGINAL_BIAS": "1",
            }
        )
        validate_env(baseline_env)
        yield {
            "label": candidate_label(baseline_env),
            "kind": "baseline",
            "env": baseline_env,
        }
        for values in P2_RUNTIME_CANDIDATES:
            env = dict(baseline_env)
            env.update(values)
            validate_env(env)
            yield {
                "label": candidate_label(env),
                "kind": "p2-runtime",
                "env": env,
            }
        return

    if preset == "p2-bias-reclaim":
        for release, kind in (("0", "baseline"), ("1", "p2-bias-reclaim")):
            env = dict(BASE_ENV)
            env.update(
                {
                    "PANGU_P2_TILED_ATTENTION": "1",
                    "PANGU_P2_TILED_MODE": "full-row-fast",
                    "PANGU_P2_RELEASE_ORIGINAL_BIAS": release,
                }
            )
            validate_env(env)
            yield {
                "label": candidate_label(env),
                "kind": kind,
                "env": env,
            }
        return

    if preset == "p2-region-release":
        for release, kind in (("0", "baseline"), ("1", "p2-region-release")):
            env = dict(BASE_ENV)
            env.update(
                {
                    "PANGU_P2_TILED_ATTENTION": "1",
                    "PANGU_P2_TILED_MODE": "full-row-fast",
                    "PANGU_P2_RELEASE_ORIGINAL_BIAS": "1",
                    "PANGU_P2_REGION_RELEASE": release,
                    "PANGU_TILED_HIP_EXTRA_FLAGS": P2_PLATFORM_KERNEL_FLAGS,
                }
            )
            validate_env(env)
            yield {
                "label": candidate_label(env) + f"_regionrelease{release}",
                "kind": kind,
                "env": env,
            }
        return

    if preset == "p2-hip-prebuild":
        for prebuild, kind in (("0", "baseline"), ("1", "p2-hip-prebuild")):
            env = dict(BASE_ENV)
            env.update(
                {
                    "PANGU_P2_TILED_ATTENTION": "1",
                    "PANGU_P2_TILED_MODE": "full-row-fast",
                    "PANGU_P2_RELEASE_ORIGINAL_BIAS": "1",
                    "PANGU_P2_PREBUILD_HIP": prebuild,
                    "PANGU_TILED_HIP_EXTRA_FLAGS": P2_PLATFORM_KERNEL_FLAGS,
                }
            )
            validate_env(env)
            yield {
                "label": candidate_label(env) + f"_hipprebuild{prebuild}",
                "kind": kind,
                "env": env,
                "fresh_hip_build": True,
            }
        return

    if preset == "p2-stride":
        for stride, kind in (("144", "baseline"), ("148", "p2-stride"), ("156", "p2-stride")):
            env = dict(BASE_ENV)
            env.update(
                {
                    "PANGU_P2_TILED_ATTENTION": "1",
                    "PANGU_P2_TILED_MODE": "full-row-fast",
                    "PANGU_P2_RELEASE_ORIGINAL_BIAS": "1",
                    "PANGU_P2_FULL_ROW_SCORE_STRIDE": stride,
                }
            )
            validate_env(env)
            yield {
                "label": candidate_label(env),
                "kind": kind,
                "env": env,
            }
        return

    if preset == "p2-kernel-pipeline":
        base_env = dict(BASE_ENV)
        base_env.update(
            {
                "PANGU_P2_TILED_ATTENTION": "1",
                "PANGU_P2_TILED_MODE": "full-row-fast",
                "PANGU_P2_RELEASE_ORIGINAL_BIAS": "1",
            }
        )
        candidate_env = dict(base_env)
        candidate_env["PANGU_TILED_HIP_EXTRA_FLAGS"] = (
            "-DPANGU_FULL_ROW_DIRECT_SCORE_STORE=1 "
            "-DPANGU_FULL_ROW_PV_DOUBLE_BUFFER=1"
        )
        validate_env(base_env)
        validate_env(candidate_env)
        yield {
            "label": candidate_label(base_env) + "_kpipe0",
            "kind": "baseline",
            "env": base_env,
        }
        yield {
            "label": candidate_label(candidate_env) + "_kpipe1",
            "kind": "p2-kernel",
            "env": candidate_env,
        }
        return

    if preset == "p2-qk32":
        base_env = dict(BASE_ENV)
        base_env.update(
            {
                "PANGU_P2_TILED_ATTENTION": "1",
                "PANGU_P2_TILED_MODE": "full-row-fast",
                "PANGU_P2_RELEASE_ORIGINAL_BIAS": "1",
                "PANGU_P2_FULL_ROW_QK_TILE": "16",
                "PANGU_TILED_HIP_EXTRA_FLAGS": (
                    "-DPANGU_FULL_ROW_DIRECT_SCORE_STORE=1 "
                    "-DPANGU_FULL_ROW_PV_DOUBLE_BUFFER=1"
                ),
            }
        )
        candidate_env = dict(base_env)
        candidate_env["PANGU_P2_FULL_ROW_QK_TILE"] = "32"
        validate_env(base_env)
        validate_env(candidate_env)
        yield {
            "label": candidate_label(base_env) + "_qk16",
            "kind": "baseline",
            "env": base_env,
        }
        yield {
            "label": candidate_label(candidate_env) + "_qk32",
            "kind": "p2-qk32",
            "env": candidate_env,
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
    if len(candidates) == 1:
        template = candidates[0]
    elif (
        len(candidates) == 2
        and candidates[0].get("kind") == "baseline"
        and candidates[1].get("kind") == "p2-region-release"
    ):
        # After region release is accepted, hold that runtime fixed on both
        # sides while comparing only the checkpoint representation.
        template = candidates[1]
    else:
        raise ValueError(
            "Checkpoint A/B requires baseline or p2-region-release preset"
        )
    baseline = dict(template)
    baseline["kind"] = "baseline"
    baseline["env"] = dict(baseline["env"])
    baseline["env"]["PANGU_FP16_CHECKPOINT"] = baseline_checkpoint
    baseline["label"] += "_ckptbaseline"

    candidate = dict(baseline)
    candidate["kind"] = "checkpoint_candidate"
    candidate["env"] = dict(baseline["env"])
    candidate["env"]["PANGU_FP16_CHECKPOINT"] = candidate_checkpoint
    candidate["label"] = template["label"] + "_ckptcandidate"
    return [baseline, candidate]


def p2_runtime_greedy_candidates(spec):
    """Build an incremental A/B such as ``nearest:bank8``."""

    if spec.count(":") != 1:
        raise ValueError("greedy step must use BASE_CONTROLS:ADDED_CONTROL")
    raw_base, added = spec.split(":", 1)
    base_controls = [value for value in raw_base.split(",") if value]
    if added not in P2_RUNTIME_CONTROL_ENV:
        raise ValueError(f"unknown added P2 runtime control: {added}")
    if len(base_controls) != len(set(base_controls)):
        raise ValueError("greedy base controls must be unique")
    unknown = [
        control for control in base_controls if control not in P2_RUNTIME_CONTROL_ENV
    ]
    if unknown:
        raise ValueError(f"unknown base P2 runtime control: {unknown[0]}")
    if added in base_controls:
        raise ValueError("added P2 runtime control is already in the base")

    base_env = dict(next(iter(iter_candidates("p2-runtime")))["env"])
    for control in base_controls:
        base_env.update(P2_RUNTIME_CONTROL_ENV[control])
    candidate_env = dict(base_env)
    candidate_env.update(P2_RUNTIME_CONTROL_ENV[added])
    return [
        {
            "label": candidate_label(base_env),
            "kind": "baseline",
            "env": base_env,
            "greedy_controls": base_controls,
            "greedy_incremental": False,
        },
        {
            "label": candidate_label(candidate_env),
            "kind": "p2-runtime",
            "env": candidate_env,
            "greedy_controls": [*base_controls, added],
            "greedy_incremental": True,
        },
    ]


def reset_output_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_stdout(stdout):
    max_vram_mb = None
    current_vram_mb = None
    reserved_values = []
    p2_region_setup = None
    hip_prebuild_report = None
    report_parse_errors = []
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
        if line.startswith("[P2_REGION_SETUP] "):
            try:
                p2_region_setup = json.loads(
                    line.removeprefix("[P2_REGION_SETUP] ")
                )
            except json.JSONDecodeError as error:
                report_parse_errors.append(f"P2_REGION_SETUP: {error}")
        prebuild_marker = "prepared HIP library: "
        if prebuild_marker in line:
            try:
                hip_prebuild_report = json.loads(line.split(prebuild_marker, 1)[1])
            except json.JSONDecodeError as error:
                report_parse_errors.append(f"P2_PREBUILD_HIP: {error}")
    return {
        "max_vram_mb": max_vram_mb,
        "current_vram_mb": current_vram_mb,
        "reserved_mb": max(reserved_values) if reserved_values else None,
        "p2_region_setup": p2_region_setup,
        "hip_prebuild_report": hip_prebuild_report,
        "report_parse_errors": report_parse_errors,
    }


def read_time_record(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        values = json.load(f)
    return [float(value) for value in values]


def _sha256_output_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_output_file(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    nan_count = 0
    inf_count = 0
    if np.issubdtype(array.dtype, np.inexact):
        iterator = np.nditer(
            array,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"]],
            order="C",
            buffersize=262144,
        )
        for values in iterator:
            nan_count += int(np.count_nonzero(np.isnan(values)))
            inf_count += int(np.count_nonzero(np.isinf(values)))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "file_bytes": path.stat().st_size,
        "sha256": _sha256_output_file(path),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "all_finite": nan_count == 0 and inf_count == 0,
    }


def _compare_output_arrays(baseline_path, candidate_path):
    baseline = np.load(baseline_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    shape_equal = baseline.shape == candidate.shape
    dtype_equal = baseline.dtype == candidate.dtype
    if not shape_equal or not dtype_equal:
        return {
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "array_equal": False,
            "max_abs": None,
            "max_rel": None,
        }

    array_equal = True
    numeric = np.issubdtype(baseline.dtype, np.number)
    max_abs = 0.0 if numeric else None
    max_rel = 0.0 if numeric else None
    iterator = np.nditer(
        [baseline, candidate],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        order="C",
        buffersize=262144,
    )
    for baseline_values, candidate_values in iterator:
        if not np.array_equal(baseline_values, candidate_values):
            array_equal = False
        if not numeric:
            continue
        finite = np.isfinite(baseline_values) & np.isfinite(candidate_values)
        if not np.any(finite):
            continue
        conversion_dtype = (
            np.complex128
            if np.issubdtype(baseline.dtype, np.complexfloating)
            else np.float64
        )
        baseline_finite = baseline_values[finite].astype(
            conversion_dtype, copy=False
        )
        candidate_finite = candidate_values[finite].astype(
            conversion_dtype, copy=False
        )
        difference = np.abs(candidate_finite - baseline_finite)
        denominator = np.maximum(np.abs(baseline_finite), 1.0e-6)
        max_abs = max(max_abs, float(np.max(difference)))
        max_rel = max(max_rel, float(np.max(difference / denominator)))
    return {
        "shape_equal": True,
        "dtype_equal": True,
        "array_equal": array_equal,
        "max_abs": max_abs,
        "max_rel": max_rel,
    }


def _inspect_output_set(paths, side):
    metadata = {}
    errors = {}
    for name, path in paths.items():
        try:
            metadata[name] = _inspect_output_file(path)
        except (OSError, TypeError, ValueError) as error:
            errors[f"{side}:{name}"] = f"{type(error).__name__}: {error}"
    return metadata, errors


def compare_outputs(candidate_dir, baseline_dir):
    candidate_paths = {
        path.name: path for path in sorted(candidate_dir.glob("*.npy"))
    }
    candidate_metadata, load_errors = _inspect_output_set(
        candidate_paths, "candidate"
    )
    candidate_nan_count = sum(
        item["nan_count"] for item in candidate_metadata.values()
    )
    candidate_inf_count = sum(
        item["inf_count"] for item in candidate_metadata.values()
    )
    candidate_all_finite = bool(candidate_paths) and not load_errors and all(
        item["all_finite"] for item in candidate_metadata.values()
    )
    if baseline_dir is None or not baseline_dir.exists():
        return {
            "output_compare_available": False,
            "output_exact": None,
            "output_set_equal": None,
            "output_shapes_equal": None,
            "output_dtypes_equal": None,
            "output_array_equal": None,
            "output_sha256_equal": None,
            "output_all_finite": candidate_all_finite,
            "output_baseline_all_finite": None,
            "output_nan_count": candidate_nan_count,
            "output_inf_count": candidate_inf_count,
            "output_baseline_nan_count": None,
            "output_baseline_inf_count": None,
            "output_missing_files": [],
            "output_extra_files": [],
            "output_load_errors": load_errors,
            "output_file_checks": [
                {"filename": name, "candidate": candidate_metadata.get(name)}
                for name in sorted(candidate_paths)
            ],
            "output_max_abs": None,
            "output_max_rel": None,
            "output_files": 0,
            "output_candidate_files": len(candidate_paths),
            "output_baseline_files": None,
        }

    baseline_paths = {
        path.name: path for path in sorted(baseline_dir.glob("*.npy"))
    }
    baseline_metadata, baseline_errors = _inspect_output_set(
        baseline_paths, "baseline"
    )
    load_errors.update(baseline_errors)
    baseline_nan_count = sum(
        item["nan_count"] for item in baseline_metadata.values()
    )
    baseline_inf_count = sum(
        item["inf_count"] for item in baseline_metadata.values()
    )
    baseline_all_finite = bool(baseline_paths) and not baseline_errors and all(
        item["all_finite"] for item in baseline_metadata.values()
    )

    baseline_names = set(baseline_paths)
    candidate_names = set(candidate_paths)
    matched_names = sorted(baseline_names & candidate_names)
    missing_files = sorted(baseline_names - candidate_names)
    extra_files = sorted(candidate_names - baseline_names)
    file_checks = []
    for name in sorted(baseline_names | candidate_names):
        record = {
            "filename": name,
            "baseline": baseline_metadata.get(name),
            "candidate": candidate_metadata.get(name),
        }
        if name in matched_names and not any(
            key in load_errors for key in (f"baseline:{name}", f"candidate:{name}")
        ):
            record.update(
                _compare_output_arrays(baseline_paths[name], candidate_paths[name])
            )
            record["sha256_equal"] = (
                baseline_metadata[name]["sha256"]
                == candidate_metadata[name]["sha256"]
            )
        else:
            record.update(
                {
                    "shape_equal": False,
                    "dtype_equal": False,
                    "array_equal": False,
                    "sha256_equal": False,
                    "max_abs": None,
                    "max_rel": None,
                }
            )
        file_checks.append(record)

    output_set_equal = baseline_names == candidate_names
    output_shapes_equal = output_set_equal and all(
        item["shape_equal"] for item in file_checks
    )
    output_dtypes_equal = output_set_equal and all(
        item["dtype_equal"] for item in file_checks
    )
    output_array_equal = output_set_equal and all(
        item["array_equal"] for item in file_checks
    )
    output_sha256_equal = output_set_equal and all(
        item["sha256_equal"] for item in file_checks
    )
    output_exact = (
        bool(baseline_names)
        and not load_errors
        and output_set_equal
        and output_shapes_equal
        and output_dtypes_equal
        and output_array_equal
        and output_sha256_equal
        and candidate_all_finite
        and baseline_all_finite
    )
    max_abs_values = [
        item["max_abs"] for item in file_checks if item["max_abs"] is not None
    ]
    max_rel_values = [
        item["max_rel"] for item in file_checks if item["max_rel"] is not None
    ]
    return {
        "output_compare_available": True,
        "output_exact": output_exact,
        "output_set_equal": output_set_equal,
        "output_shapes_equal": output_shapes_equal,
        "output_dtypes_equal": output_dtypes_equal,
        "output_array_equal": output_array_equal,
        "output_sha256_equal": output_sha256_equal,
        "output_all_finite": candidate_all_finite,
        "output_baseline_all_finite": baseline_all_finite,
        "output_nan_count": candidate_nan_count,
        "output_inf_count": candidate_inf_count,
        "output_baseline_nan_count": baseline_nan_count,
        "output_baseline_inf_count": baseline_inf_count,
        "output_missing_files": missing_files,
        "output_extra_files": extra_files,
        "output_load_errors": load_errors,
        "output_file_checks": file_checks,
        "output_max_abs": max(max_abs_values) if max_abs_values else None,
        "output_max_rel": max(max_rel_values) if max_rel_values else None,
        "output_files": len(matched_names),
        "output_candidate_files": len(candidate_paths),
        "output_baseline_files": len(baseline_paths),
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
    fresh_hip_build_dirs = []
    start_wall = time.perf_counter()
    for repeat_index in range(args.repeat):
        reset_output_dir(output_dir)
        command = [args.python, "inference.py"]
        run_env = env
        temporary_build_dir = None
        if candidate.get("fresh_hip_build", False):
            temporary_build_dir = tempfile.TemporaryDirectory(
                prefix="pangu_uv_fresh_hip_"
            )
            run_env = dict(env)
            run_env["PANGU_TILED_HIP_BUILD_DIR"] = temporary_build_dir.name
            fresh_hip_build_dirs.append(Path(temporary_build_dir.name).name)
        try:
            process = subprocess.run(
                command,
                cwd=str(pangu_dir),
                env=run_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finally:
            if temporary_build_dir is not None:
                temporary_build_dir.cleanup()
        stdout_tail = "\n".join(process.stdout.splitlines()[-80:])
        if process.returncode != 0:
            return {
                "label": candidate["label"],
                "kind": candidate["kind"],
                "env": candidate["env"],
                "fresh_hip_build": candidate.get("fresh_hip_build", False),
                "fresh_hip_build_dirs": fresh_hip_build_dirs,
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
    region_reports = [
        item["p2_region_setup"]
        for item in parsed_runs
        if item["p2_region_setup"] is not None
    ]
    hip_prebuild_reports = [
        item["hip_prebuild_report"]
        for item in parsed_runs
        if item["hip_prebuild_report"] is not None
    ]
    return {
        "label": candidate["label"],
        "kind": candidate["kind"],
        "env": candidate["env"],
        "fresh_hip_build": candidate.get("fresh_hip_build", False),
        "fresh_hip_build_dirs": fresh_hip_build_dirs,
        "greedy_controls": candidate.get("greedy_controls"),
        "greedy_incremental": candidate.get("greedy_incremental", False),
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
        "max_vram_rounds_mb": [item["max_vram_mb"] for item in parsed_runs],
        "reserved_rounds_mb": [item["reserved_mb"] for item in parsed_runs],
        "current_vram_rounds_mb": [
            item["current_vram_mb"] for item in parsed_runs
        ],
        "p2_region_setup": region_reports[-1] if region_reports else None,
        "p2_region_setup_rounds": [
            item["p2_region_setup"] for item in parsed_runs
        ],
        "hip_prebuild_report": (
            hip_prebuild_reports[-1] if hip_prebuild_reports else None
        ),
        "hip_prebuild_report_rounds": [
            item["hip_prebuild_report"] for item in parsed_runs
        ],
        "report_parse_errors": [
            {"round": round_index, "errors": item["report_parse_errors"]}
            for round_index, item in enumerate(parsed_runs, start=1)
            if item["report_parse_errors"]
        ],
        "wall_time_s": time.perf_counter() - start_wall,
        "stdout_tail": stdout_tail,
        **output_metrics,
    }


def _aggregate_interleaved_runs(candidate, runs, repeat, max_batches):
    failure = next((run for run in runs if run.get("returncode") != 0), None)
    if failure is not None:
        return failure
    latency_values = [
        value for run in runs for value in run.get("latency_ms_values", [])
    ]
    latency_rounds = [run.get("latency_ms_values", []) for run in runs]
    steady_rounds = [run.get("steady_latency_ms_values", []) for run in runs]
    steady_values = [value for values in steady_rounds for value in values]

    def maximum(key):
        values = [run.get(key) for run in runs if run.get(key) is not None]
        return max(values) if values else None

    def all_available(key):
        values = [run.get(key) for run in runs if run.get(key) is not None]
        return all(values) if values else None

    def union(key):
        return sorted(
            {
                value
                for run in runs
                for value in (run.get(key) or [])
            }
        )

    output_abs = [
        run.get("output_max_abs")
        for run in runs
        if run.get("output_max_abs") is not None
    ]
    output_rel = [
        run.get("output_max_rel")
        for run in runs
        if run.get("output_max_rel") is not None
    ]
    output_load_errors = {
        f"round_{round_index}:{key}": value
        for round_index, run in enumerate(runs, start=1)
        for key, value in (run.get("output_load_errors") or {}).items()
    }
    output_comparison_rounds = [
        {
            "round": round_index,
            "available": run.get("output_compare_available", False),
            "exact": run.get("output_exact"),
            "set_equal": run.get("output_set_equal"),
            "shapes_equal": run.get("output_shapes_equal"),
            "dtypes_equal": run.get("output_dtypes_equal"),
            "array_equal": run.get("output_array_equal"),
            "sha256_equal": run.get("output_sha256_equal"),
            "all_finite": run.get("output_all_finite"),
            "nan_count": run.get("output_nan_count"),
            "inf_count": run.get("output_inf_count"),
            "candidate_files": run.get("output_candidate_files"),
            "baseline_files": run.get("output_baseline_files"),
            "file_checks": run.get("output_file_checks", []),
        }
        for round_index, run in enumerate(runs, start=1)
    ]
    region_reports = [
        run.get("p2_region_setup")
        for run in runs
        if run.get("p2_region_setup") is not None
    ]
    hip_prebuild_reports = [
        run.get("hip_prebuild_report")
        for run in runs
        if run.get("hip_prebuild_report") is not None
    ]
    return {
        "label": candidate["label"],
        "kind": candidate["kind"],
        "env": candidate["env"],
        "fresh_hip_build": candidate.get("fresh_hip_build", False),
        "fresh_hip_build_dirs": [
            value
            for run in runs
            for value in run.get("fresh_hip_build_dirs", [])
        ],
        "greedy_controls": candidate.get("greedy_controls"),
        "greedy_incremental": candidate.get("greedy_incremental", False),
        "returncode": 0,
        "repeat": repeat,
        "max_batches": max_batches,
        "interleaved": True,
        "latency_ms_values": latency_values,
        "latency_avg_ms": float(np.mean(latency_values)) if latency_values else None,
        "latency_min_ms": float(np.min(latency_values)) if latency_values else None,
        "latency_p50_ms": float(np.median(latency_values)) if latency_values else None,
        "latency_rounds_ms": latency_rounds,
        "steady_latency_rounds_ms": steady_rounds,
        "steady_latency_ms_values": steady_values,
        "steady_latency_avg_ms": float(np.mean(steady_values)) if steady_values else None,
        "steady_latency_p50_ms": float(np.median(steady_values)) if steady_values else None,
        "steady_latency_p90_ms": (
            float(np.percentile(steady_values, 90)) if steady_values else None
        ),
        "steady_latency_std_ms": float(np.std(steady_values)) if steady_values else None,
        "max_vram_mb": maximum("max_vram_mb"),
        "reserved_mb": maximum("reserved_mb"),
        "current_vram_mb": runs[-1].get("current_vram_mb"),
        "max_vram_rounds_mb": [run.get("max_vram_mb") for run in runs],
        "reserved_rounds_mb": [run.get("reserved_mb") for run in runs],
        "current_vram_rounds_mb": [run.get("current_vram_mb") for run in runs],
        "p2_region_setup": region_reports[-1] if region_reports else None,
        "p2_region_setup_rounds": [
            run.get("p2_region_setup") for run in runs
        ],
        "hip_prebuild_report": (
            hip_prebuild_reports[-1] if hip_prebuild_reports else None
        ),
        "hip_prebuild_report_rounds": [
            run.get("hip_prebuild_report") for run in runs
        ],
        "report_parse_errors": [
            {"round": round_index, "errors": run.get("report_parse_errors")}
            for round_index, run in enumerate(runs, start=1)
            if run.get("report_parse_errors")
        ],
        "wall_time_s": sum(run.get("wall_time_s", 0.0) for run in runs),
        "stdout_tail": runs[-1].get("stdout_tail", ""),
        "output_max_abs": max(output_abs) if output_abs else None,
        "output_max_rel": max(output_rel) if output_rel else None,
        "output_files": maximum("output_files"),
        "output_candidate_files": maximum("output_candidate_files"),
        "output_baseline_files": maximum("output_baseline_files"),
        "output_compare_available": any(
            run.get("output_compare_available", False) for run in runs
        ),
        "output_exact": all_available("output_exact"),
        "output_set_equal": all_available("output_set_equal"),
        "output_shapes_equal": all_available("output_shapes_equal"),
        "output_dtypes_equal": all_available("output_dtypes_equal"),
        "output_array_equal": all_available("output_array_equal"),
        "output_sha256_equal": all_available("output_sha256_equal"),
        "output_all_finite": all_available("output_all_finite"),
        "output_baseline_all_finite": all_available(
            "output_baseline_all_finite"
        ),
        "output_nan_count": maximum("output_nan_count"),
        "output_inf_count": maximum("output_inf_count"),
        "output_baseline_nan_count": maximum("output_baseline_nan_count"),
        "output_baseline_inf_count": maximum("output_baseline_inf_count"),
        "output_missing_files": union("output_missing_files"),
        "output_extra_files": union("output_extra_files"),
        "output_load_errors": output_load_errors,
        "output_file_checks": runs[-1].get("output_file_checks", []),
        "output_comparison_rounds": output_comparison_rounds,
    }


def run_interleaved(candidates, *, args, pangu_dir, output_dir):
    """Alternate candidate order by subprocess round to control drift."""

    single_args = argparse.Namespace(**vars(args))
    single_args.repeat = 1
    runs_by_label = {candidate["label"]: [] for candidate in candidates}
    baseline_dir = None
    for round_index in range(args.repeat):
        ordered = candidates if round_index % 2 == 0 else list(reversed(candidates))
        for candidate in ordered:
            print(
                f"[round {round_index + 1}/{args.repeat}] "
                f"{candidate['label']}"
            )
            result = run_one(
                candidate,
                args=single_args,
                pangu_dir=pangu_dir,
                output_dir=output_dir,
                baseline_dir=baseline_dir,
            )
            runs_by_label[candidate["label"]].append(result)
            if (
                baseline_dir is None
                and candidate["kind"] == "baseline"
                and result.get("returncode") == 0
            ):
                baseline_dir = pangu_dir / "result" / "uv_sweep_baseline"
                reset_output_dir(baseline_dir)
                for npy_path in output_dir.glob("*.npy"):
                    shutil.copy2(npy_path, baseline_dir / npy_path.name)
            if result.get("returncode") != 0:
                return [
                    _aggregate_interleaved_runs(
                        item,
                        runs_by_label[item["label"]],
                        args.repeat,
                        args.max_batches,
                    )
                    for item in candidates
                    if runs_by_label[item["label"]]
                ]
    return [
        _aggregate_interleaved_runs(
            candidate,
            runs_by_label[candidate["label"]],
            args.repeat,
            args.max_batches,
        )
        for candidate in candidates
    ]


def _round_means(rounds):
    return [float(np.mean(values)) for values in rounds if values]


def _paired_bootstrap_summary(
    baseline_rounds,
    candidate_rounds,
    *,
    seed,
    samples=10000,
):
    baseline_means = _round_means(baseline_rounds)
    candidate_means = _round_means(candidate_rounds)
    if (
        not baseline_means
        or len(baseline_means) != len(baseline_rounds)
        or len(candidate_means) != len(candidate_rounds)
        or len(baseline_means) != len(candidate_means)
    ):
        return {
            "complete": False,
            "baseline_round_means_ms": baseline_means,
            "candidate_round_means_ms": candidate_means,
            "seed": seed,
            "bootstrap_samples": samples,
        }
    deltas = [
        candidate - baseline
        for baseline, candidate in zip(baseline_means, candidate_means)
    ]
    generator = random.Random(seed)
    bootstrap_means = []
    for _ in range(samples):
        bootstrap_means.append(
            sum(deltas[generator.randrange(len(deltas))] for _ in deltas)
            / len(deltas)
        )
    bootstrap_means.sort()
    lower_index = int(0.025 * (len(bootstrap_means) - 1))
    upper_index = int(0.975 * (len(bootstrap_means) - 1))
    baseline_mean = float(np.mean(baseline_means))
    candidate_mean = float(np.mean(candidate_means))
    delta_mean = float(np.mean(deltas))
    return {
        "complete": True,
        "baseline_round_means_ms": baseline_means,
        "candidate_round_means_ms": candidate_means,
        "paired_round_deltas_ms": deltas,
        "baseline_mean_ms": baseline_mean,
        "candidate_mean_ms": candidate_mean,
        "paired_delta_mean_ms": delta_mean,
        "paired_delta_pct": (
            100.0 * delta_mean / baseline_mean if baseline_mean else None
        ),
        "paired_bootstrap_ci95_ms": [
            bootstrap_means[lower_index],
            bootstrap_means[upper_index],
        ],
        "seed": seed,
        "bootstrap_samples": samples,
    }


def _paired_round_values(baseline_values, candidate_values):
    complete = (
        bool(baseline_values)
        and len(baseline_values) == len(candidate_values)
        and all(value is not None for value in baseline_values)
        and all(value is not None for value in candidate_values)
    )
    deltas = (
        [
            float(candidate) - float(baseline)
            for baseline, candidate in zip(baseline_values, candidate_values)
        ]
        if complete
        else []
    )
    return {
        "complete": complete,
        "baseline_round_values": baseline_values,
        "candidate_round_values": candidate_values,
        "paired_round_deltas": deltas,
        "paired_delta_mean": float(np.mean(deltas)) if deltas else None,
    }


def _exact_output_round_gate(result):
    rounds = result.get("output_comparison_rounds") or []
    if not rounds:
        return False
    for check in rounds:
        if check.get("candidate_files") != result.get("max_batches"):
            return False
        if check.get("all_finite") is not True:
            return False
        if check.get("available"):
            if check.get("baseline_files") != result.get("max_batches"):
                return False
            if check.get("exact") is not True:
                return False
        elif result.get("kind") != "baseline" or check.get("round") != 1:
            return False
    return True


def build_interleaved_summary(results, *, seed=20260715, samples=10000):
    baseline = next(
        (result for result in results if result.get("kind") == "baseline"),
        None,
    )
    comparisons = []
    if baseline is not None:
        for index, candidate in enumerate(
            (result for result in results if result is not baseline), start=1
        ):
            comparisons.append(
                {
                    "candidate_label": candidate.get("label"),
                    "candidate_kind": candidate.get("kind"),
                    "all_batches": _paired_bootstrap_summary(
                        baseline.get("latency_rounds_ms", []),
                        candidate.get("latency_rounds_ms", []),
                        seed=seed + index * 2,
                        samples=samples,
                    ),
                    "steady": _paired_bootstrap_summary(
                        baseline.get("steady_latency_rounds_ms", []),
                        candidate.get("steady_latency_rounds_ms", []),
                        seed=seed + index * 2 + 1,
                        samples=samples,
                    ),
                    "max_vram_mb": _paired_round_values(
                        baseline.get("max_vram_rounds_mb", []),
                        candidate.get("max_vram_rounds_mb", []),
                    ),
                    "current_vram_mb": _paired_round_values(
                        baseline.get("current_vram_rounds_mb", []),
                        candidate.get("current_vram_rounds_mb", []),
                    ),
                    "reserved_mb": _paired_round_values(
                        baseline.get("reserved_rounds_mb", []),
                        candidate.get("reserved_rounds_mb", []),
                    ),
                    "baseline_region_setup_rounds": baseline.get(
                        "p2_region_setup_rounds", []
                    ),
                    "candidate_region_setup_rounds": candidate.get(
                        "p2_region_setup_rounds", []
                    ),
                    "baseline_hip_prebuild_rounds": baseline.get(
                        "hip_prebuild_report_rounds", []
                    ),
                    "candidate_hip_prebuild_rounds": candidate.get(
                        "hip_prebuild_report_rounds", []
                    ),
                    "baseline_exact_output_all_rounds": (
                        _exact_output_round_gate(baseline)
                    ),
                    "candidate_exact_output_all_rounds": (
                        _exact_output_round_gate(candidate)
                    ),
                    "exact_output_all_rounds": (
                        _exact_output_round_gate(baseline)
                        and _exact_output_round_gate(candidate)
                    ),
                }
            )
    return {
        "record_type": "interleaved_ab_summary",
        "label": "__interleaved_ab_summary__",
        "kind": "ab_summary",
        "returncode": None,
        "baseline_label": baseline.get("label") if baseline is not None else None,
        "repeat": baseline.get("repeat") if baseline is not None else None,
        "max_batches": baseline.get("max_batches") if baseline is not None else None,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
    }


def _relative_change_pct(baseline, candidate):
    if baseline is None or candidate is None or baseline == 0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def build_stage1_hard_gate(results, *, preset, timer_sha256):
    """Evaluate the frozen 5x4 Stage-1 gates without hiding failed evidence."""

    checks = []

    def add(name, passed, observed=None, requirement=None):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "requirement": requirement,
            }
        )

    baseline = next(
        (result for result in results if result.get("kind") == "baseline"), None
    )
    candidate = next(
        (result for result in results if result.get("kind") != "baseline"), None
    )
    add("one_baseline_one_candidate", len(results) == 2 and baseline is not None and candidate is not None, len(results), 2)
    if baseline is None or candidate is None:
        return {"passed": False, "checks": checks}

    completed = sum(
        len(result.get("latency_rounds_ms", [])) for result in (baseline, candidate)
    )
    add(
        "ten_successful_subprocesses",
        completed == 10
        and all(result.get("returncode") == 0 for result in (baseline, candidate)),
        completed,
        10,
    )
    add(
        "five_rounds_four_batches",
        all(
            result.get("repeat") == 5 and result.get("max_batches") == 4
            for result in (baseline, candidate)
        ),
        {
            "baseline": [baseline.get("repeat"), baseline.get("max_batches")],
            "candidate": [candidate.get("repeat"), candidate.get("max_batches")],
        },
        {"repeat": 5, "max_batches": 4},
    )
    add(
        "official_timer_hash",
        timer_sha256 == FROZEN_OFFICIAL_TIMER_SHA256,
        timer_sha256,
        FROZEN_OFFICIAL_TIMER_SHA256,
    )
    exact = _exact_output_round_gate(baseline) and _exact_output_round_gate(candidate)
    add(
        "four_exact_finite_outputs_every_round",
        exact,
        {
            "baseline": _exact_output_round_gate(baseline),
            "candidate": _exact_output_round_gate(candidate),
        },
        True,
    )
    add(
        "report_parse_errors_empty",
        not baseline.get("report_parse_errors")
        and not candidate.get("report_parse_errors"),
        {
            "baseline": baseline.get("report_parse_errors"),
            "candidate": candidate.get("report_parse_errors"),
        },
        [],
    )

    steady_mean_pct = _relative_change_pct(
        baseline.get("steady_latency_avg_ms"),
        candidate.get("steady_latency_avg_ms"),
    )
    steady_p90_pct = _relative_change_pct(
        baseline.get("steady_latency_p90_ms"),
        candidate.get("steady_latency_p90_ms"),
    )
    max_vram_deltas = [
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(
            baseline.get("max_vram_rounds_mb", []),
            candidate.get("max_vram_rounds_mb", []),
        )
        if baseline_value is not None and candidate_value is not None
    ]
    current_vram_deltas = [
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(
            baseline.get("current_vram_rounds_mb", []),
            candidate.get("current_vram_rounds_mb", []),
        )
        if baseline_value is not None and candidate_value is not None
    ]

    if preset == "p2-region-release":
        reports = candidate.get("p2_region_setup_rounds", [])
        topology_ok = len(reports) == 5 and all(
            isinstance(report, dict)
            and report.get("attention_modules") == 16
            and report.get("shifted_mask_owners") == 8
            and report.get("dense_mask_logical_bytes_after") == 0
            and report.get("dense_mask_unique_bytes_after") == 0
            for report in reports
        )
        add("region_exact_16_8_dense_after_zero", topology_ok, reports, True)
        reclaimed = [
            report.get("actual_cuda_dense_mask_reclaimed_bytes")
            if isinstance(report, dict)
            and "actual_cuda_dense_mask_reclaimed_bytes" in report
            else report.get("actual_cuda_allocated_reclaimed_bytes")
            if isinstance(report, dict)
            else None
            for report in reports
        ]
        current_reclaim_ok = (
            len(current_vram_deltas) == 5
            and len(reclaimed) == 5
            and all(
                isinstance(value, int)
                and value > 0
                and (-delta_mb * 2**20) >= 0.9 * value
                for delta_mb, value in zip(current_vram_deltas, reclaimed)
            )
        )
        add(
            "region_current_allocated_drop_at_least_0_9r",
            current_reclaim_ok,
            {"current_delta_mb": current_vram_deltas, "reclaimed_bytes": reclaimed},
            "drop >= 0.9 * measured R in every round",
        )
        add(
            "region_peak_increase_at_most_1mb",
            len(max_vram_deltas) == 5 and max(max_vram_deltas) <= 1.0,
            max_vram_deltas,
            "<= 1 MiB every round",
        )
        add(
            "region_steady_mean_regression_at_most_0_5pct",
            steady_mean_pct is not None and steady_mean_pct <= 0.5,
            steady_mean_pct,
            "<= 0.5%",
        )
        add(
            "region_steady_p90_regression_at_most_1pct",
            steady_p90_pct is not None and steady_p90_pct <= 1.0,
            steady_p90_pct,
            "<= 1.0%",
        )
    elif preset == "p2-hip-prebuild":
        prebuild_reports = candidate.get("hip_prebuild_report_rounds", [])
        build_dirs = [
            *baseline.get("fresh_hip_build_dirs", []),
            *candidate.get("fresh_hip_build_dirs", []),
        ]
        add(
            "hip_prebuild_report_every_candidate_round",
            len(prebuild_reports) == 5
            and all(isinstance(report, dict) and report.get("fingerprint") for report in prebuild_reports),
            prebuild_reports,
            5,
        )
        add(
            "hip_fresh_independent_build_dirs",
            len(build_dirs) == 10 and len(set(build_dirs)) == 10,
            build_dirs,
            "10 unique directories",
        )
        all_batch = _paired_bootstrap_summary(
            baseline.get("latency_rounds_ms", []),
            candidate.get("latency_rounds_ms", []),
            seed=20260717,
        )
        ci95 = all_batch.get("paired_bootstrap_ci95_ms") or [None, None]
        add(
            "hip_all_batch_speedup_at_least_0_5pct",
            all_batch.get("complete") is True
            and all_batch.get("paired_delta_pct") is not None
            and all_batch["paired_delta_pct"] <= -0.5,
            all_batch.get("paired_delta_pct"),
            "candidate-baseline <= -0.5%",
        )
        add(
            "hip_bootstrap_ci95_confirms_improvement",
            ci95[1] is not None and ci95[1] < 0,
            ci95,
            "upper bound < 0 ms",
        )
        add(
            "hip_peak_increase_at_most_2mb",
            len(max_vram_deltas) == 5 and max(max_vram_deltas) <= 2.0,
            max_vram_deltas,
            "<= 2 MiB every round",
        )
    else:
        add(
            "checkpoint_memory_does_not_increase",
            len(max_vram_deltas) == 5
            and len(current_vram_deltas) == 5
            and max(max_vram_deltas) <= 0
            and max(current_vram_deltas) <= 0,
            {
                "peak_delta_mb": max_vram_deltas,
                "current_delta_mb": current_vram_deltas,
            },
            "all deltas <= 0 MiB",
        )
        add(
            "checkpoint_steady_mean_regression_at_most_0_5pct",
            steady_mean_pct is not None and steady_mean_pct <= 0.5,
            steady_mean_pct,
            "<= 0.5%",
        )
        add(
            "checkpoint_steady_p90_regression_at_most_0_5pct",
            steady_p90_pct is not None and steady_p90_pct <= 0.5,
            steady_p90_pct,
            "<= 0.5%",
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "delta_convention": "candidate_minus_baseline; negative latency is faster",
        "checks": checks,
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
        "--p2-runtime-greedy-step",
        default=None,
        help=(
            "With --preset p2-runtime, compare BASE_CONTROLS:ADDED_CONTROL; "
            "controls are nearest,bank8,priority,stream-spin. Example: "
            "nearest:bank8. Use :nearest for the first step."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=[
            "baseline", "compact-mask", "direct-mask", "cuda-graph", "cpu-recovery",
            "full-recovery", "focused", "full", "pangu-lite-2d", "hip",
            "buffer-intern", "stagewise",
            "p2-tiled", "p2-bias-reclaim", "p2-runtime", "p2-stride",
            "p2-kernel-pipeline", "p2-qk32", "p2-region-release",
            "p2-hip-prebuild",
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
            "for pgw_lite_pruned_96; p2-runtime fixes P2 plus bias reclaim and "
            "p2-bias-reclaim isolates original-bias GPU release; "
            "isolates NUMA, LDS-bank, and stream-priority controls; focused "
            "p2-stride compares full-row LDS strides 144/148/156; "
            "p2-kernel-pipeline compares canonical P2 with the exact direct-score "
            "store plus PV-double-buffer kernel; "
            "p2-qk32 compares that platform kernel with a 32-key QK tile while "
            "retaining the 16-key PV double buffer; "
            "p2-region-release enables one-time region-ID preparation plus "
            "dense-mask release; p2-hip-prebuild compares lazy and "
            "pre-timed source builds in fresh per-process build directories; "
            "sweeps attention chunk size; full "
            "is the broad diagnostic grid."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-output-compare", action="store_true")
    args = parser.parse_args()

    if args.preset == "pangu-lite-2d" and args.fp16_checkpoint is None:
        args.fp16_checkpoint = "model_pangu_lite_2d_pos288_hybrid.pth"

    pangu_dir = Path(__file__).resolve().parents[1]
    timer_sha256 = official_timer_block_sha256(pangu_dir / "inference.py")
    if timer_sha256 != FROZEN_OFFICIAL_TIMER_SHA256:
        raise RuntimeError(
            "Official timer block hash drifted: "
            f"expected={FROZEN_OFFICIAL_TIMER_SHA256} actual={timer_sha256}"
        )
    log_path = make_log_path(args.log_file)
    if not log_path.is_absolute():
        log_path = pangu_dir / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = list(iter_candidates(args.preset))
    if args.p2_runtime_greedy_step is not None:
        if args.preset != "p2-runtime":
            parser.error("--p2-runtime-greedy-step requires --preset p2-runtime")
        try:
            candidates = p2_runtime_greedy_candidates(
                args.p2_runtime_greedy_step
            )
        except ValueError as error:
            parser.error(str(error))
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
    if should_run_interleaved(args.preset, args.candidate_fp16_checkpoint):
        results = run_interleaved(
            candidates,
            args=args,
            pangu_dir=pangu_dir,
            output_dir=output_dir,
        )
        for result in results:
            result["official_timer_block_sha256"] = timer_sha256
        with log_path.open("w", encoding="utf-8") as log_file:
            for result in results:
                log_file.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
                print(
                    f"  {result.get('label')} rc={result.get('returncode')} "
                    f"steady={result.get('steady_latency_avg_ms')} "
                    f"vram={result.get('max_vram_mb')} "
                    f"err={result.get('output_max_abs')} "
                    f"exact={result.get('output_exact')}"
                )
            summary = build_interleaved_summary(results)
            summary["preset"] = args.preset
            summary["official_timer_block_sha256"] = timer_sha256
            if (
                args.preset in {"p2-region-release", "p2-hip-prebuild"}
                or args.candidate_fp16_checkpoint is not None
            ):
                gate_preset = (
                    "checkpoint"
                    if args.candidate_fp16_checkpoint is not None
                    else args.preset
                )
                summary["stage1_hard_gate"] = build_stage1_hard_gate(
                    results,
                    preset=gate_preset,
                    timer_sha256=timer_sha256,
                )
            log_file.write(
                json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n"
            )
        print(f"wrote {log_path}")
        return

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
            result["official_timer_block_sha256"] = timer_sha256
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
