"""Audit and create a minimal FP16 inference checkpoint.

The OneScience Pangu state dict contains aliases. Converting every key with an
independent ``tensor.half()`` call expands shared storages and can make the FP16
file much larger than necessary. This tool converts each unique floating-point
storage once, reconstructs all tensor views, and verifies the saved checkpoint.

Examples:
    python scripts/convert_fp16.py
    python scripts/convert_fp16.py --source data/checkpoints/model_fp16.pth --audit-only
    python scripts/convert_fp16.py --output data/checkpoints/model_fp16_compact.pth
"""

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict
from collections.abc import Mapping

import torch


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return (storage._cdata, str(tensor.device), str(tensor.dtype))


def _storage_groups(state_dict):
    groups = OrderedDict()
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            continue
        groups.setdefault(_storage_key(tensor), []).append((name, tensor))
    return groups


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("model_state_dict must be a mapping")
    non_tensors = [name for name, value in state_dict.items()
                   if not isinstance(value, torch.Tensor)]
    if non_tensors:
        preview = ", ".join(non_tensors[:5])
        raise TypeError(f"model_state_dict contains non-tensors: {preview}")
    return state_dict


def audit_checkpoint(path, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    groups = _storage_groups(state_dict)

    tensor_dtypes = Counter()
    element_dtypes = Counter()
    logical_bytes = 0
    ungrouped_bytes = 0
    for tensor in state_dict.values():
        tensor_dtypes[str(tensor.dtype)] += 1
        element_dtypes[str(tensor.dtype)] += tensor.numel()
        logical_bytes += tensor.numel() * tensor.element_size()
        if tensor.layout != torch.strided:
            ungrouped_bytes += tensor.numel() * tensor.element_size()

    unique_storage_bytes = ungrouped_bytes + sum(
        tensors[0][1].untyped_storage().nbytes() for tensors in groups.values()
    )
    alias_groups = [
        [name for name, _ in tensors]
        for tensors in groups.values()
        if len(tensors) > 1
    ]
    top_level_keys = list(checkpoint) if checkpoint is not state_dict else []
    extra_fields = [key for key in top_level_keys if key != "model_state_dict"]

    report = {
        "path": os.path.abspath(path),
        "file_bytes": os.path.getsize(path),
        "top_level_keys": top_level_keys,
        "extra_fields": extra_fields,
        "tensor_count": len(state_dict),
        "tensor_dtypes": dict(sorted(tensor_dtypes.items())),
        "elements_by_dtype": dict(sorted(element_dtypes.items())),
        "logical_tensor_bytes": logical_bytes,
        "unique_storage_count": len(groups),
        "unique_storage_bytes": unique_storage_bytes,
        "alias_group_count": len(alias_groups),
        "aliased_tensor_count": sum(len(names) for names in alias_groups),
        "largest_alias_groups": sorted(alias_groups, key=len, reverse=True)[:10],
    }
    return report


def print_report(report, title):
    mib = 1024 ** 2
    print(f"\n{title}: {report['path']}")
    print(f"  file size:            {report['file_bytes'] / mib:.1f} MiB")
    print(f"  tensors:              {report['tensor_count']}")
    print(f"  tensor dtypes:        {report['tensor_dtypes']}")
    print(f"  logical tensor bytes: {report['logical_tensor_bytes'] / mib:.1f} MiB")
    print(f"  unique storage bytes: {report['unique_storage_bytes'] / mib:.1f} MiB")
    print(f"  unique storages:      {report['unique_storage_count']}")
    print(f"  alias groups:         {report['alias_group_count']}")
    print(f"  extra fields:         {report['extra_fields']}")
    for names in report["largest_alias_groups"][:5]:
        preview = ", ".join(names[:4])
        suffix = " ..." if len(names) > 4 else ""
        print(f"    alias x{len(names)}: {preview}{suffix}")


def convert_state_dict_to_fp16(state_dict):
    """Convert each dense floating storage once and preserve all tensor views."""
    converted = OrderedDict()
    groups = _storage_groups(state_dict)

    for tensors in groups.values():
        representative = tensors[0][1]
        if not representative.is_floating_point():
            for name, tensor in tensors:
                converted[name] = tensor
            continue

        storage = representative.untyped_storage()
        storage_elements = storage.nbytes() // representative.element_size()
        base = torch.empty(0, dtype=representative.dtype, device="cpu")
        base = base.set_(storage, 0, (storage_elements,), (1,))
        fp16_storage = base.half()
        for name, tensor in tensors:
            converted[name] = fp16_storage.as_strided(
                tuple(tensor.size()), tuple(tensor.stride()), tensor.storage_offset()
            )

    for name, tensor in state_dict.items():
        if name in converted:
            continue
        converted[name] = tensor.half() if tensor.is_floating_point() else tensor

    converted = state_dict.__class__((name, converted[name]) for name in state_dict)
    if hasattr(state_dict, "_metadata"):
        converted._metadata = state_dict._metadata
    return converted


def _alias_partitions(state_dict):
    return sorted(
        sorted(name for name, _ in tensors)
        for tensors in _storage_groups(state_dict).values()
        if len(tensors) > 1
    )


def verify_checkpoint(source_state, candidate_path):
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    candidate_state = extract_state_dict(candidate)
    if list(candidate_state) != list(source_state):
        raise ValueError("Candidate state_dict keys or ordering differ from source")

    for name, source in source_state.items():
        actual = candidate_state[name]
        expected = source.half() if source.is_floating_point() else source
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"Tensor metadata mismatch: {name}")
        if not torch.equal(actual, expected):
            raise ValueError(f"Tensor value mismatch: {name}")

    if _alias_partitions(candidate_state) != _alias_partitions(source_state):
        raise ValueError("Candidate tensor storage aliases differ from source")


def convert_to_fp16(source_path, destination_path, report_path=None):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = extract_state_dict(checkpoint)
    source_report = audit_checkpoint(source_path, checkpoint)
    print_report(source_report, "Source checkpoint")
    del checkpoint

    fp16_state = convert_state_dict_to_fp16(source_state)
    destination_path = os.path.abspath(destination_path)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    temporary_path = destination_path + ".tmp"
    try:
        torch.save({"model_state_dict": fp16_state}, temporary_path)
        verify_checkpoint(source_state, temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    output_report = audit_checkpoint(destination_path)
    print_report(output_report, "Verified FP16 checkpoint")
    reduction = 100 * (1 - output_report["file_bytes"] / source_report["file_bytes"])
    print(f"  file reduction:       {reduction:.1f}%")

    if report_path:
        payload = {"source": source_report, "output": output_report}
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return output_report


def resolve_source(checkpoint_dir, backup_dir, explicit_source):
    if explicit_source:
        return explicit_source
    local_source = os.path.join(checkpoint_dir, "model_bak.pth")
    backup_source = os.path.join(backup_dir, "model_bak.pth")
    return local_source if os.path.exists(local_source) else backup_source


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit or create a storage-preserving FP16 Pangu checkpoint"
    )
    parser.add_argument("--checkpoint_dir", default="./data/checkpoints")
    parser.add_argument("--backup_dir", default="./pangu_backups")
    parser.add_argument("--source", help="Explicit checkpoint to read")
    parser.add_argument("--output", help="Output path (default: model_fp16.pth)")
    parser.add_argument("--report", help="Optional JSON audit report path")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = resolve_source(args.checkpoint_dir, args.backup_dir, args.source)
    if not os.path.exists(source_path):
        print(f"Source checkpoint does not exist: {source_path}", file=sys.stderr)
        return 1

    if args.audit_only:
        report = audit_checkpoint(source_path)
        print_report(report, "Checkpoint audit")
        if args.report:
            with open(args.report, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        return 0

    output_path = args.output or os.path.join(args.checkpoint_dir, "model_fp16.pth")
    if os.path.abspath(source_path) == os.path.abspath(output_path):
        print("Source and output paths must differ", file=sys.stderr)
        return 1
    convert_to_fp16(source_path, output_path, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
