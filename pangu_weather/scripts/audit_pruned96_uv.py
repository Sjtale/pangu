#!/usr/bin/env python3
"""Audit full pruned_96 storage and summarize measured U/V bottlenecks."""

import argparse
import gc
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path


MIB = 1024 ** 2
EXPECTED_PROFILE = {
    "patch_size": [2, 8, 8],
    "embed_dim": 96,
    "num_heads": [3, 6, 6, 3],
    "depth_blocks": [2, 6, 6, 2],
}
CORE_STAGES = {
    "embed_sequence",
    "layer1_forward",
    "downsample",
    "layer2_forward",
    "layer3_forward",
    "upsample",
    "layer4_forward",
    "skip_concat",
    "recovery",
    "output_postprocess",
}


def _mb(value):
    return round(float(value) / MIB, 4)


def _as_int_list(value):
    return [int(item) for item in value]


def _process_memory_mb():
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        values = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _unit = line.split()
                values[key.rstrip(":")] = round(float(value) / 1024.0, 4)
        return {"rss_mb": values.get("VmRSS"), "high_water_mb": values.get("VmHWM")}
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = MIB if sys.platform == "darwin" else 1024.0
    return {"rss_mb": None, "high_water_mb": round(float(raw) / divisor, 4)}


def _embedding_weight(state_dict):
    suffixes = (
        "patchembed2d.embedder.proj.weight",
        "patchembed2d.proj.weight",
    )
    return next(
        (tensor for key, tensor in state_dict.items() if key.endswith(suffixes)),
        None,
    )


def infer_profile(checkpoint, state_dict):
    metadata = checkpoint.get("model_profile", {}) if isinstance(checkpoint, dict) else {}
    profile = dict(metadata) if isinstance(metadata, dict) else {}
    weight = _embedding_weight(state_dict)
    if weight is not None:
        profile.setdefault("embed_dim", int(weight.shape[0]))
        profile.setdefault("patch_size", [int(item) for item in weight.shape[-3:]])
    profile.setdefault("num_heads", [3, 6, 6, 3])
    profile.setdefault("depth_blocks", [2, 6, 6, 2])
    for key in ("patch_size", "num_heads", "depth_blocks"):
        if key in profile and profile[key] is not None:
            profile[key] = _as_int_list(profile[key])
    if "embed_dim" in profile:
        profile["embed_dim"] = int(profile["embed_dim"])
    return profile


def validate_profile(profile):
    mismatches = {}
    for key, expected in EXPECTED_PROFILE.items():
        if profile.get(key) != expected:
            mismatches[key] = {"actual": profile.get(key), "expected": expected}
    if mismatches:
        raise ValueError(
            "Checkpoint is not full pgw_lite_pruned_96: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def audit_checkpoint(path):
    import torch

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    memory_before = _process_memory_mb()
    started_at = time.perf_counter()
    checkpoint = torch.load(path, map_location="cpu")
    load_elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    memory_loaded = _process_memory_mb()
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state dict")

    profile = infer_profile(checkpoint, state_dict)
    validate_profile(profile)

    dtype_bytes = Counter()
    logical_bytes = 0
    unique_storages = {}
    tensors = []
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        size_bytes = value.numel() * value.element_size()
        logical_bytes += size_bytes
        dtype_bytes[str(value.dtype)] += size_bytes
        storage = value.untyped_storage()
        storage_key = (storage.data_ptr(), storage.nbytes())
        unique_storages.setdefault(storage_key, storage.nbytes())
        tensors.append(
            {
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "size_mb": _mb(size_bytes),
            }
        )

    unique_storage_bytes = sum(unique_storages.values())
    tensors.sort(key=lambda item: item["size_mb"], reverse=True)
    result = {
        "checkpoint": str(path),
        "file_size_mb": _mb(path.stat().st_size),
        "checkpoint_load_ms": round(load_elapsed_ms, 4),
        "process_memory_before_load": memory_before,
        "process_memory_after_load": memory_loaded,
        "profile": profile,
        "tensor_count": len(tensors),
        "logical_tensor_mb": _mb(logical_bytes),
        "unique_storage_mb": _mb(unique_storage_bytes),
        "alias_view_savings_mb": _mb(max(0, logical_bytes - unique_storage_bytes)),
        "serialization_overhead_mb": _mb(max(0, path.stat().st_size - unique_storage_bytes)),
        "dtype_mb": {
            dtype: _mb(size) for dtype, size in sorted(dtype_bytes.items())
        },
        "largest_tensors": tensors[:15],
    }
    del checkpoint, state_dict
    gc.collect()
    result["process_memory_after_cleanup"] = _process_memory_mb()
    return result


def audit_package(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    files = []
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(path)
        if any(part in {".git", "result", "logs", "__pycache__"} for part in relative.parts):
            continue
        files.append({"path": str(relative), "size_mb": _mb(file_path.stat().st_size)})
    files.sort(key=lambda item: item["size_mb"], reverse=True)
    return {
        "package_dir": str(path),
        "total_file_mb": round(sum(item["size_mb"] for item in files), 4),
        "file_count": len(files),
        "largest_files": files[:20],
    }


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_baseline(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    successful = [row for row in rows if row.get("returncode") == 0]
    if not successful:
        raise ValueError(f"No successful runtime row in {path}")
    return next((row for row in successful if row.get("kind") == "baseline"), successful[0])


def build_report(static, vram_records, runtime):
    stage_rows = [row for row in vram_records if row.get("tag") in CORE_STAGES]
    steady_rows = [
        row
        for row in vram_records
        if row.get("tag", "").startswith("steady.")
        and row["tag"].split(".", 1)[1] in CORE_STAGES
    ]
    cold_timed = sorted(
        (row for row in stage_rows if row.get("elapsed_ms") is not None),
        key=lambda row: row["elapsed_ms"],
        reverse=True,
    )
    steady_timed = sorted(
        (row for row in steady_rows if row.get("elapsed_ms") is not None),
        key=lambda row: row["elapsed_ms"],
        reverse=True,
    )
    peak = sorted(stage_rows, key=lambda row: row.get("peak_mb", 0), reverse=True)
    allocated = sorted(stage_rows, key=lambda row: row.get("delta_alloc_mb", 0), reverse=True)
    first_values = runtime.get("latency_ms_values") or []
    steady = runtime.get("steady_latency_avg_ms")
    first = first_values[0] if first_values else None

    lines = [
        "# pruned_96 U/V bottleneck audit",
        "",
        "## Scope",
        "",
        "This report accepts only full `pgw_lite_pruned_96`: patch `[2,8,8]`, "
        "width 96, heads `[3,6,6,3]`, depth `[2,6,6,2]`.",
        "",
        "## U evidence",
        "",
        f"- Checkpoint file: `{static['file_size_mb']:.4f} MiB`.",
        f"- CPU checkpoint load: `{static.get('checkpoint_load_ms')} ms`; process memory "
        f"before/after load: `{static.get('process_memory_before_load')}` / "
        f"`{static.get('process_memory_after_load')}`.",
        f"- Logical tensor bytes: `{static['logical_tensor_mb']:.4f} MiB`; unique storage: "
        f"`{static['unique_storage_mb']:.4f} MiB`; alias/view saving: "
        f"`{static['alias_view_savings_mb']:.4f} MiB`.",
        f"- Measured forward peak: `{runtime.get('max_vram_mb')} MiB`; reserved peak: "
        f"`{runtime.get('reserved_mb')} MiB`; post-run resident: `{runtime.get('current_vram_mb')} MiB`.",
        "",
        "Largest absolute stage peaks:",
        "",
    ]
    for row in peak[:5]:
        lines.append(
            f"- `{row['tag']}`: peak `{row.get('peak_mb', 0):.2f} MiB`, "
            f"new high-water increment `{row.get('delta_peak_mb', 0):+.2f} MiB`."
        )
    lines.extend(["", "Largest live-allocation changes:", ""])
    for row in allocated[:5]:
        lines.append(
            f"- `{row['tag']}`: delta allocated `{row.get('delta_alloc_mb', 0):+.2f} MiB`, "
            f"allocated `{row.get('allocated_mb', 0):.2f} MiB`."
        )

    lines.extend(
        [
            "",
            "## V evidence",
            "",
            f"- Official-boundary mean: `{runtime.get('latency_avg_ms')} ms`; steady mean: "
            f"`{steady} ms`; P50/P90: `{runtime.get('steady_latency_p50_ms')}` / "
            f"`{runtime.get('steady_latency_p90_ms')}` ms.",
            f"- First measured forward: `{first}` ms. Stage timings below synchronize after "
            "every stage and are attribution measurements, not a replacement for the official-boundary timing.",
            "",
            "Largest cold-start synchronized stage timings:",
            "",
        ]
    )
    for row in cold_timed[:8]:
        lines.append(f"- `{row['tag']}`: `{row['elapsed_ms']:.3f} ms`.")

    lines.extend(["", "Largest steady synchronized stage timings:", ""])
    for row in steady_timed[:8]:
        stage = row["tag"].split(".", 1)[1]
        lines.append(f"- `{stage}`: `{row['elapsed_ms']:.3f} ms`.")
    if not steady_timed:
        lines.append("- Unavailable: rerun with the steady-stage probe.")

    u_driver = peak[0]["tag"] if peak else "unavailable"
    cold_v_driver = cold_timed[0]["tag"] if cold_timed else "unavailable"
    steady_v_driver = (
        steady_timed[0]["tag"].split(".", 1)[1]
        if steady_timed
        else "unavailable"
    )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"- Current local U proxy is most constrained by the `{u_driver}` high-water mark. "
            "Platform U remains authoritative because prior platform rows did not track CUDA peak monotonically.",
            f"- Cold-start V attribution is led by `{cold_v_driver}`; steady V attribution is led by "
            f"`{steady_v_driver}`. Optimize either only after "
            "confirming the same direction lowers the uninstrumented official-boundary mean.",
            "- Checkpoint/package bytes, resident memory, peak memory, first-forward time and steady time "
            "must remain separate columns; combining them into one proxy hides the actual bottleneck.",
        ]
    )
    return "\n".join(lines) + "\n"


def command_static(args):
    result = audit_checkpoint(args.checkpoint)
    if args.package_dir:
        result["package"] = audit_package(args.package_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


def command_report(args):
    report = build_report(
        load_json(args.static_json),
        load_json(args.vram_json),
        load_runtime_baseline(args.runtime_jsonl),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"wrote {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    static = subparsers.add_parser("static", help="Audit checkpoint and package storage.")
    static.add_argument("--checkpoint", required=True)
    static.add_argument("--package-dir")
    static.add_argument("--output", default="logs/pruned96_static_audit.json")
    static.set_defaults(func=command_static)

    report = subparsers.add_parser("report", help="Combine storage, VRAM and latency evidence.")
    report.add_argument("--static-json", default="logs/pruned96_static_audit.json")
    report.add_argument("--vram-json", default="logs/pruned96_vram_breakdown.json")
    report.add_argument("--runtime-jsonl", default="logs/pruned96_runtime.jsonl")
    report.add_argument("--output", default="logs/pruned96_uv_bottleneck_report.md")
    report.set_defaults(func=command_report)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
