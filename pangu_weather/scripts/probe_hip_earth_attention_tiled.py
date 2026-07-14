#!/usr/bin/env python3
"""Compile and gate the exact-pruned tiled HIP EarthAttention prototype.

This probe intentionally stays separate from the production forward path.  It
uses the exact ``pruned_96`` attention geometries, compact relative-position
bias/index inputs, and compact shifted-window region labels.
"""

import argparse
import json
import math
import os
import resource
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


ATOL = 5e-3
RTOL = 1e-2
MIN_REPRESENTATIVE_SPEEDUP = 1.5
KERNEL_MODES = ("online", "full-row-fast", "full-row-expf")
PROFILE = {
    "name": "pgw_lite_pruned_96",
    "embed_dim": 96,
    "num_heads": [3, 6, 6, 3],
    "window_size": [2, 6, 12],
    "head_dim": 32,
}

CASES = {
    "small_l32_unshifted": {
        "name": "small_l32_unshifted",
        "width": 1,
        "heads": 1,
        "pressure_height": 1,
        "window_size": [2, 4, 4],
        "input_resolution": [2, 4, 4],
        "shifted": False,
    },
    "shallow_l144_unshifted": {
        "name": "shallow_l144_unshifted",
        "width": 3,
        "heads": 3,
        "pressure_height": 64,
        "window_size": [2, 6, 12],
        "input_resolution": [8, 96, 36],
        "shifted": False,
    },
    "shallow_l144_shifted": {
        "name": "shallow_l144_shifted",
        "width": 3,
        "heads": 3,
        "pressure_height": 64,
        "window_size": [2, 6, 12],
        "input_resolution": [8, 96, 36],
        "shifted": True,
    },
    "deep_l144_unshifted": {
        "name": "deep_l144_unshifted",
        "width": 3,
        "heads": 6,
        "pressure_height": 32,
        "window_size": [2, 6, 12],
        "input_resolution": [8, 48, 36],
        "shifted": False,
    },
    "deep_l144_shifted": {
        "name": "deep_l144_shifted",
        "width": 3,
        "heads": 6,
        "pressure_height": 32,
        "window_size": [2, 6, 12],
        "input_resolution": [8, 48, 36],
        "shifted": True,
    },
}


def _position_index(window_size):
    """Mirror OneScience's EarthAttention3D index without importing it."""

    win_pl, win_lat, win_lon = window_size
    coords_zi = torch.arange(win_pl)
    coords_zj = -torch.arange(win_pl) * win_pl
    coords_hi = torch.arange(win_lat)
    coords_hj = -torch.arange(win_lat) * win_lat
    coords_w = torch.arange(win_lon)
    coords_1 = torch.stack(
        torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij")
    )
    coords_2 = torch.stack(
        torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij")
    )
    flattened_1 = torch.flatten(coords_1, 1)
    flattened_2 = torch.flatten(coords_2, 1)
    coords = (flattened_1[:, :, None] - flattened_2[:, None, :]).permute(
        1, 2, 0
    )
    coords = coords.contiguous()
    coords[:, :, 2] += win_lon - 1
    coords[:, :, 1] *= 2 * win_lon - 1
    coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat
    return coords.sum(-1)


def _shift_region_ids(input_resolution, window_size):
    """Return the compact labels used to construct OneScience's -100 mask."""

    pressure, latitude, longitude = input_resolution
    win_pl, win_lat, win_lon = window_size
    shift_pl, shift_lat, shift_lon = (value // 2 for value in window_size)
    if pressure % win_pl or latitude % win_lat or longitude % win_lon:
        raise ValueError("probe input resolution must be divisible by window size")

    image = torch.zeros(
        (1, pressure, latitude, longitude + shift_lon, 1), dtype=torch.uint8
    )
    pl_slices = (
        slice(0, -win_pl),
        slice(-win_pl, -shift_pl),
        slice(-shift_pl, None),
    )
    lat_slices = (
        slice(0, -win_lat),
        slice(-win_lat, -shift_lat),
        slice(-shift_lat, None),
    )
    lon_slices = (
        slice(0, -win_lon),
        slice(-win_lon, -shift_lon),
        slice(-shift_lon, None),
    )
    label = 0
    for pl_slice in pl_slices:
        for lat_slice in lat_slices:
            for lon_slice in lon_slices:
                image[:, pl_slice, lat_slice, lon_slice, :] = label
                label += 1
    image = image[:, :, :, :longitude, :]
    image = image.view(
        1,
        pressure // win_pl,
        win_pl,
        latitude // win_lat,
        win_lat,
        longitude // win_lon,
        win_lon,
        1,
    )
    windows = image.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous()
    return windows.view(
        longitude // win_lon,
        (pressure // win_pl) * (latitude // win_lat),
        win_pl * win_lat * win_lon,
    )


def _make_inputs(case, device, seed):
    width = case["width"]
    heads = case["heads"]
    pressure_height = case["pressure_height"]
    window_size = case["window_size"]
    sequence_length = math.prod(window_size)
    relative_positions = (
        (window_size[0] ** 2)
        * (window_size[1] ** 2)
        * (2 * window_size[2] - 1)
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    qkv = torch.randn(
        width,
        pressure_height,
        sequence_length,
        3,
        heads,
        PROFILE["head_dim"],
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    bias_table = (
        torch.randn(
            relative_positions,
            pressure_height,
            heads,
            generator=generator,
            device=device,
            dtype=torch.float16,
        )
        * 0.02
    )
    index_i64 = _position_index(window_size).to(device=device)
    region_ids = None
    if case["shifted"]:
        region_ids = _shift_region_ids(
            case["input_resolution"], window_size
        ).to(device=device)
    return qkv, bias_table, index_i64, region_ids


def _dense_inputs(bias_table, index_i64, region_ids):
    sequence_length = index_i64.shape[0]
    pressure_height = bias_table.shape[1]
    bias = bias_table[index_i64.reshape(-1)].view(
        sequence_length, sequence_length, pressure_height, -1
    )
    bias = bias.permute(3, 2, 0, 1).contiguous().unsqueeze(0)
    mask = None
    if region_ids is not None:
        mask = (region_ids.unsqueeze(-1) != region_ids.unsqueeze(-2)).to(
            dtype=torch.float16
        )
        mask.mul_(-100.0)
    return bias, mask


def _reference(qkv, bias, mask, scale):
    qkv_heads = qkv.permute(3, 0, 4, 1, 2, 5)
    # Production EarthAttention scales Q while it is FP16, before QK.
    q = (qkv_heads[0] * scale).float()
    k = qkv_heads[1].float()
    v = qkv_heads[2].float()
    scores = torch.matmul(q, k.transpose(-2, -1)) + bias.float()
    if mask is not None:
        scores = scores + mask[:, None].float()
    output = torch.matmul(torch.softmax(scores, dim=-1), v)
    return output.permute(0, 2, 3, 1, 4).reshape(
        qkv.shape[0], qkv.shape[1], qkv.shape[2], -1
    )


def _eager_half(qkv, bias, mask, scale):
    qkv_heads = qkv.permute(3, 0, 4, 1, 2, 5)
    q = qkv_heads[0] * scale
    scores = torch.matmul(q, qkv_heads[1].transpose(-2, -1)) + bias
    if mask is not None:
        scores = scores + mask[:, None]
    output = torch.matmul(torch.softmax(scores, dim=-1), qkv_heads[2])
    return output.permute(0, 2, 3, 1, 4).reshape(
        qkv.shape[0], qkv.shape[1], qkv.shape[2], -1
    )


def _eager_half_stages(qkv, bias, mask, scale):
    qkv_heads = qkv.permute(3, 0, 4, 1, 2, 5)
    q = qkv_heads[0] * scale
    qk_half = torch.matmul(q, qkv_heads[1].transpose(-2, -1))
    biased_half = qk_half + bias
    if mask is not None:
        biased_half = biased_half + mask[:, None]
    probability_half = torch.softmax(biased_half, dim=-1)
    output = torch.matmul(probability_half, qkv_heads[2])
    output = output.permute(0, 2, 3, 1, 4).reshape(
        qkv.shape[0], qkv.shape[1], qkv.shape[2], -1
    )
    return {
        "qk_half": qk_half,
        "biased_half": biased_half,
        "probability_half": probability_half,
        "pv_output_half": output,
    }


def _percentile_nearest_rank(values, fraction):
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _measure(call, device, warmup, samples, launches_per_sample):
    output = None
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize(device)
    baseline_allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
    torch.cuda.reset_peak_memory_stats(device)

    gpu_samples = []
    wall_samples = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record()
        for _ in range(launches_per_sample):
            output = call()
        stop.record()
        stop.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        gpu_samples.append(start.elapsed_time(stop) / launches_per_sample)
        wall_samples.append(wall_ms / launches_per_sample)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return output, {
        "gpu_median_ms": statistics.median(gpu_samples),
        "gpu_p90_ms": _percentile_nearest_rank(gpu_samples, 0.90),
        "gpu_mean_ms": statistics.fmean(gpu_samples),
        "wall_median_ms": statistics.median(wall_samples),
        "wall_p90_ms": _percentile_nearest_rank(wall_samples, 0.90),
        "baseline_allocated_mb": baseline_allocated_mb,
        "peak_allocated_mb": peak_mb,
        "incremental_peak_mb": peak_mb - baseline_allocated_mb,
        "samples_ms": gpu_samples,
    }


def _comparison_metrics(left, right):
    difference = (left.float() - right.float()).abs()
    return {
        "exact": bool(torch.equal(left, right)),
        "mismatch_count": int((left != right).sum().item()),
        "elements": left.numel(),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
    }


def _run_case(call_hip, pack_bias, compact_index, case, device, args):
    qkv, bias_table, index_i64, region_ids = _make_inputs(
        case, device, args.seed
    )
    dense_bias, dense_mask = _dense_inputs(bias_table, index_i64, region_ids)
    packed_bias = pack_bias(bias_table)
    index_u16 = compact_index(index_i64, bias_table.shape[0])
    scale = PROFILE["head_dim"] ** -0.5

    reference = _reference(qkv, dense_bias, dense_mask, scale)
    # Candidate peak must not retain the dense bias/mask that the fused ABI is
    # designed to eliminate.  Keep only the compact inputs and the small
    # reference output; reconstruct eager-only dense inputs after this timing.
    torch.cuda.synchronize(device)
    del dense_bias, dense_mask
    candidate, tiled_metrics = _measure(
        lambda: call_hip(
            qkv,
            packed_bias,
            index_u16,
            region_ids,
            scale=scale,
            mask_width=0 if region_ids is None else region_ids.shape[0],
            width_offset=0,
            mode=args.kernel_mode,
        ),
        device,
        args.warmup,
        args.samples,
        args.launches_per_sample,
    )
    finite = bool(torch.isfinite(candidate).all().item())
    output_contract = (
        tuple(candidate.shape) == tuple(reference.shape)
        and candidate.dtype == torch.float16
        and candidate.is_contiguous()
    )
    fp32_compatible = finite and output_contract and bool(
        torch.allclose(reference, candidate.float(), atol=ATOL, rtol=RTOL)
    )
    candidate_contiguous = bool(candidate.is_contiguous())
    candidate_vs_reference = _comparison_metrics(candidate.float(), reference)

    dense_bias, dense_mask = _dense_inputs(bias_table, index_i64, region_ids)
    eager_output, eager_metrics = _measure(
        lambda: _eager_half(qkv, dense_bias, dense_mask, scale),
        device,
        args.warmup,
        args.samples,
        args.launches_per_sample,
    )
    candidate_vs_eager = _comparison_metrics(candidate, eager_output)
    eager_vs_reference = _comparison_metrics(eager_output.float(), reference)
    stage_diagnostics = None
    if args.diagnostic_stages:
        diagnostic_output, candidate_stages = call_hip(
            qkv,
            packed_bias,
            index_u16,
            region_ids,
            scale=scale,
            mask_width=0 if region_ids is None else region_ids.shape[0],
            width_offset=0,
            mode=args.kernel_mode,
            return_diagnostics=True,
        )
        eager_stages = _eager_half_stages(
            qkv,
            dense_bias,
            dense_mask,
            scale,
        )
        alternate_mode = (
            "full-row-expf"
            if args.kernel_mode == "full-row-fast"
            else "full-row-fast"
        )
        alternate_output, alternate_stages = call_hip(
            qkv,
            packed_bias,
            index_u16,
            region_ids,
            scale=scale,
            mask_width=0 if region_ids is None else region_ids.shape[0],
            width_offset=0,
            mode=alternate_mode,
            return_diagnostics=True,
        )
        stage_diagnostics = {
            "qk_half": _comparison_metrics(
                candidate_stages["qk_half"],
                eager_stages["qk_half"],
            ),
            "biased_half": _comparison_metrics(
                candidate_stages["biased_half"],
                eager_stages["biased_half"],
            ),
            "probability_half": _comparison_metrics(
                candidate_stages["probability_half"],
                eager_stages["probability_half"],
            ),
            "pv_output_half": _comparison_metrics(
                diagnostic_output,
                eager_stages["pv_output_half"],
            ),
            "diagnostic_output_matches_timed_candidate": _comparison_metrics(
                diagnostic_output,
                candidate,
            ),
            "exp_variant_comparison": {
                "candidate_mode": args.kernel_mode,
                "alternate_mode": alternate_mode,
                "probability_half": _comparison_metrics(
                    candidate_stages["probability_half"],
                    alternate_stages["probability_half"],
                ),
                "pv_output_half": _comparison_metrics(
                    diagnostic_output,
                    alternate_output,
                ),
            },
        }
    production_exact = candidate_vs_eager["exact"]
    requires_production_exact = args.kernel_mode != "online"
    compatible = fp32_compatible and (
        production_exact or not requires_production_exact
    )
    speedup = eager_metrics["gpu_median_ms"] / tiled_metrics["gpu_median_ms"]
    p90_ok = tiled_metrics["gpu_p90_ms"] <= eager_metrics["gpu_p90_ms"]
    peak_ok = (
        tiled_metrics["peak_allocated_mb"] <= eager_metrics["peak_allocated_mb"]
    )
    sequence_length = math.prod(case["window_size"])
    return {
        **case,
        "kernel_mode": args.kernel_mode,
        "tokens": sequence_length,
        "relative_positions": bias_table.shape[0],
        "status": "PASS" if compatible else "NUMERICAL_MISMATCH",
        "finite": finite,
        "output_contract": bool(output_contract),
        "numerically_compatible": compatible,
        "fp32_compatible": fp32_compatible,
        "production_exact": production_exact,
        "production_exact_required": requires_production_exact,
        "atol": ATOL,
        "rtol": RTOL,
        "max_abs_error": candidate_vs_reference["max_abs"],
        "mean_abs_error": candidate_vs_reference["mean_abs"],
        "candidate_vs_eager_half": candidate_vs_eager,
        "candidate_vs_reference_fp32": candidate_vs_reference,
        "eager_half_vs_reference_fp32": eager_vs_reference,
        "stage_diagnostics": stage_diagnostics,
        "tiled": tiled_metrics,
        "eager": eager_metrics,
        "speedup_median": speedup,
        "p90_not_slower_than_eager": p90_ok,
        "peak_not_above_eager": peak_ok,
        "qkv_contiguous": qkv.is_contiguous(),
        "output_contiguous": candidate_contiguous,
        "output_shape": [
            case["width"],
            case["pressure_height"],
            sequence_length,
            case["heads"] * PROFILE["head_dim"],
        ],
    }


def _disable_core_dumps():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_isolated_case(case_name, args):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker-case",
        case_name,
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--launches-per-sample",
        str(args.launches_per_sample),
        "--seed",
        str(args.seed),
        "--kernel-mode",
        args.kernel_mode,
        "--score-stride",
        str(args.score_stride),
        "--qk-tile",
        str(args.qk_tile),
    ]
    if args.allow_extra_flags:
        command.append("--allow-extra-flags")
    if args.diagnostic_stages:
        command.append("--diagnostic-stages")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        preexec_fn=_disable_core_dumps,
    )
    if completed.returncode == 0:
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        try:
            return json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as error:
            return {
                **CASES[case_name],
                "status": "FAIL",
                "numerically_compatible": False,
                "error": f"invalid worker JSON: {error}",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
    signal_name = None
    if completed.returncode < 0:
        try:
            signal_name = signal.Signals(-completed.returncode).name
        except ValueError:
            signal_name = f"SIGNAL_{-completed.returncode}"
    return {
        **CASES[case_name],
        "status": "FAIL",
        "numerically_compatible": False,
        "error": "isolated tiled-HIP worker failed",
        "returncode": completed.returncode,
        "signal": signal_name,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _default_output():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "logs" / f"hip_earth_attention_tiled_probe_{stamp}.json"


def _write_report(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(payload, end="")


def _validate_args(parser, args):
    if args.warmup < 1:
        parser.error("--warmup must be positive")
    if args.samples < 2:
        parser.error("--samples must be at least 2 for a P90 gate")
    if args.launches_per_sample < 1:
        parser.error("--launches-per-sample must be positive")
    if args.samples * args.launches_per_sample < 100:
        parser.error("P2 requires at least 100 timed launches per case")
    inherited_extra_flags = os.environ.get("PANGU_TILED_HIP_EXTRA_FLAGS", "").strip()
    if inherited_extra_flags and not args.allow_extra_flags:
        parser.error(
            "PANGU_TILED_HIP_EXTRA_FLAGS is non-empty; clear it for the canonical "
            "gate or pass --allow-extra-flags for an explicitly non-canonical run"
        )
    if args.diagnostic_stages and args.kernel_mode == "online":
        parser.error("--diagnostic-stages requires a full-row kernel mode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--launches-per-sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--allow-extra-flags", action="store_true")
    parser.add_argument("--kernel-mode", choices=KERNEL_MODES, default="online")
    parser.add_argument(
        "--score-stride", type=int, choices=(144, 148, 156), default=144
    )
    parser.add_argument("--qk-tile", type=int, choices=(16, 32), default=16)
    parser.add_argument("--diagnostic-stages", action="store_true")
    parser.add_argument("--worker-case", choices=tuple(CASES))
    args = parser.parse_args()
    _validate_args(parser, args)
    os.environ["PANGU_P2_FULL_ROW_SCORE_STRIDE"] = str(args.score_stride)
    os.environ["PANGU_P2_FULL_ROW_QK_TILE"] = str(args.qk_tile)

    if not torch.cuda.is_available():
        raise RuntimeError("HIP device is unavailable")
    device = torch.device("cuda:0")
    pangu_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(pangu_root))
    from hip_earth_attention_tiled import (
        build_hip_earth_attention_tiled,
        compact_earth_position_index,
        get_hip_earth_attention_tiled_info,
        hip_earth_attention_tiled,
        pack_earth_bias_table,
    )

    if args.worker_case:
        try:
            result = _run_case(
                hip_earth_attention_tiled,
                pack_earth_bias_table,
                compact_earth_position_index,
                CASES[args.worker_case],
                device,
                args,
            )
        except Exception as error:
            result = {
                **CASES[args.worker_case],
                "status": "FAIL",
                "numerically_compatible": False,
                "error": f"{type(error).__name__}: {error}",
            }
        print(json.dumps(result, ensure_ascii=False))
        return

    output_path = args.output or _default_output()
    started = time.perf_counter()
    try:
        library_path = build_hip_earth_attention_tiled(force=args.force_rebuild)
        kernel_info = get_hip_earth_attention_tiled_info(
            device,
            mode=args.kernel_mode,
        )
        compile_seconds = time.perf_counter() - started
    except Exception as error:
        _write_report(
            output_path,
            {
                "profile": PROFILE,
                "kernel_mode": args.kernel_mode,
                "full_row_score_stride": args.score_stride,
                "acceptance": "The isolated tiled HIP kernel must compile first",
                "device": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "torch_hip_version": torch.version.hip,
                "compile_seconds": time.perf_counter() - started,
                "results": [],
                "build_error": f"{type(error).__name__}: {error}",
                "decision": "DO_NOT_INTEGRATE",
            },
        )
        return

    results = []
    for case_name in CASES:
        result = _run_isolated_case(case_name, args)
        results.append(result)
        if result["status"] == "FAIL":
            break

    complete = len(results) == len(CASES)
    numerical_ok = complete and all(
        result.get("numerically_compatible", False) for result in results
    )
    memory_ok = complete and all(
        result.get("peak_not_above_eager", False) for result in results
    )
    p90_ok = complete and all(
        result.get("p90_not_slower_than_eager", False) for result in results
    )
    representative = [result for result in results if result.get("tokens") == 144]
    speed_ok = len(representative) == 4 and all(
        result.get("speedup_median", 0.0) >= MIN_REPRESENTATIVE_SPEEDUP
        for result in representative
    )
    kernel_config = kernel_info.get("config", {})
    required_occupancy = 8 if args.kernel_mode != "online" else 1
    kernel_ok = (
        kernel_config.get("q_tile") == 16
        and kernel_config.get("k_tile") == args.qk_tile
        and kernel_config.get("head_dim") == 32
        and kernel_config.get("block_threads") == 256
        and kernel_config.get("dynamic_smem_bytes", 0) > 0
        and kernel_info.get("occupancy", {}).get(
            "active_blocks_per_multiprocessor", 0
        )
        >= required_occupancy
    )
    report = {
        "profile": PROFILE,
        "kernel_mode": args.kernel_mode,
        "full_row_score_stride": args.score_stride,
        "full_row_qk_tile": args.qk_tile,
        "scope": "isolated prototype; production forward remains unchanged",
        "acceptance": {
            "numerical": (
                "bitwise exact against production eager-half and "
                f"allclose against FP32(atol={ATOL}, rtol={RTOL})"
                if args.kernel_mode != "online"
                else f"finite and allclose against FP32(atol={ATOL}, rtol={RTOL})"
            ),
            "timing": "10 warmups and >=100 timed launches; report median/P90",
            "speed": f"every exact L144 case >= {MIN_REPRESENTATIVE_SPEEDUP}x",
            "tail_latency": "tiled GPU-event P90 must not exceed eager P90",
            "memory": "tiled peak allocated memory must not exceed eager",
            "occupancy": f"active blocks per multiprocessor >= {required_occupancy}",
        },
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "library": str(library_path),
        "kernel": kernel_info,
        "compile_seconds": compile_seconds,
        "timing_config": {
            "warmup": args.warmup,
            "samples": args.samples,
            "launches_per_sample": args.launches_per_sample,
        },
        "results": results,
        "gates": {
            "numerical": numerical_ok,
            "memory": memory_ok,
            "p90": p90_ok,
            "representative_speed": speed_ok,
            "kernel_config_and_occupancy": kernel_ok,
        },
        "decision": (
            (
                "PROFILE_TILED_CORE_AND_PROBE_MFMA"
                if args.kernel_mode == "online"
                else "READY_FOR_FULL_MODEL_PARITY_AUDIT"
            )
            if numerical_ok and memory_ok and p90_ok and speed_ok and kernel_ok
            else "DO_NOT_INTEGRATE"
        ),
    }
    _write_report(output_path, report)


if __name__ == "__main__":
    main()
