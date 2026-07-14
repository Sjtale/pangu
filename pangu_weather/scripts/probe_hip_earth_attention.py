#!/usr/bin/env python3
"""Compile and probe the repository-owned bias-aware HIP attention kernel."""

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


ATOL = 5e-3
RTOL = 1e-2

CASES = {
    "small_l32": {
        "name": "small_l32",
        "width": 1,
        "heads": 1,
        "pressure_height": 1,
        "tokens": 32,
    },
    "representative_l144": {
        "name": "representative_l144",
        "width": 3,
        "heads": 3,
        "pressure_height": 64,
        "tokens": 144,
    },
}


def _make_inputs(width, heads, pressure_height, tokens, head_dim, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    qkv_base = torch.randn(
        width,
        pressure_height,
        tokens,
        3,
        heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    qkv = qkv_base.permute(3, 0, 4, 1, 2, 5)
    q, k, v = qkv[0], qkv[1], qkv[2]
    earth = torch.randn(
        1,
        heads,
        pressure_height,
        tokens,
        tokens,
        generator=generator,
        device=device,
        dtype=torch.float16,
    ) * 0.02
    shifted = torch.zeros(
        width,
        pressure_height,
        tokens,
        tokens,
        device=device,
        dtype=torch.float16,
    )
    split = tokens // 2
    shifted[..., :split, split:] = -100.0
    shifted[..., split:, :split] = -100.0
    return q, k, v, earth, shifted


def _reference(q, k, v, earth, shifted, scale):
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    scores += earth.float() + shifted[:, None].float()
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def _eager_half(q, k, v, earth, shifted, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores += earth + shifted[:, None]
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _measure(call, device, warmup, repeat):
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(repeat):
        output = call()
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0 / repeat
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return output, latency_ms, peak_mb


def _run_case(call_hip, case, device, args):
    inputs = _make_inputs(
        case["width"],
        case["heads"],
        case["pressure_height"],
        case["tokens"],
        32,
        device,
        args.seed,
    )
    scale = 32**-0.5
    reference = _reference(*inputs, scale)
    candidate, hip_ms, hip_peak = _measure(
        lambda: call_hip(*inputs, scale), device, args.warmup, args.repeat
    )
    difference = (reference - candidate.float()).abs()
    compatible = bool(
        torch.allclose(reference, candidate.float(), atol=ATOL, rtol=RTOL)
    )
    max_abs_error = float(difference.max().item())
    mean_abs_error = float(difference.mean().item())
    del candidate, difference
    _, eager_ms, eager_peak = _measure(
        lambda: _eager_half(*inputs, scale), device, args.warmup, args.repeat
    )
    return {
        **case,
        "status": "PASS" if compatible else "FAIL",
        "numerically_compatible": compatible,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "hip_latency_ms": hip_ms,
        "eager_latency_ms": eager_ms,
        "speedup": eager_ms / hip_ms,
        "hip_peak_allocated_mb": hip_peak,
        "eager_peak_allocated_mb": eager_peak,
        "q_stride": list(inputs[0].stride()),
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
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
    ]
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
        "error": "isolated HIP worker failed",
        "returncode": completed.returncode,
        "signal": signal_name,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_report(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(payload, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--worker-case", choices=tuple(CASES))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("HIP device is unavailable")
    device = torch.device("cuda:0")
    pangu_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(pangu_root))
    from hip_earth_attention import build_hip_earth_attention, hip_earth_attention

    if args.worker_case:
        try:
            result = _run_case(
                hip_earth_attention, CASES[args.worker_case], device, args
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
    if not args.output:
        parser.error("--output is required unless --worker-case is used")

    started = time.perf_counter()
    try:
        library_path = build_hip_earth_attention(force=args.force_rebuild)
    except Exception as error:
        _write_report(
            args.output,
            {
                "acceptance": "The repository-owned HIP kernel must compile first",
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
    compile_seconds = time.perf_counter() - started
    results = []
    for case_name in CASES:
        result = _run_isolated_case(case_name, args)
        results.append(result)
        if result["status"] != "PASS":
            break

    report = {
        "acceptance": (
            "Require numerical PASS, representative speedup > 1, hipprof one-kernel "
            "evidence, then isolated full-model A/B against 90.1048"
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "library": str(library_path),
        "compile_seconds": compile_seconds,
        "results": results,
        "decision": (
            "PROFILE_WITH_HIPPROF"
            if len(results) == len(CASES)
            and all(result["status"] == "PASS" for result in results)
            and results[-1]["speedup"] > 1.0
            else "DO_NOT_INTEGRATE"
        ),
    }
    _write_report(args.output, report)


if __name__ == "__main__":
    main()
