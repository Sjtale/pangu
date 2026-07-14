#!/usr/bin/env python3
"""Remove verified, constructor-deterministic earth-position index buffers."""

import argparse
import hashlib
import os
import sys
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch


PANGU_DIR = Path(__file__).resolve().parents[1]
if str(PANGU_DIR) not in sys.path:
    sys.path.insert(0, str(PANGU_DIR))

INDEX_SUFFIX = "earth_position_index"
LOWER_ALIAS = ".fuser."
UPPER_ALIAS = ".Fuser."
STORAGE_SCHEMA_VERSION = 1
OPTIMIZATION_KEY = "deterministic_buffer_elision"
ELISION_METHOD = "constructor-earth-position-index-v1"
EXPECTED_PROFILE = {
    "name": "pgw_lite_pruned_96",
    "patch_size": [2, 8, 8],
    "embed_dim": 96,
    "num_heads": [3, 6, 6, 3],
    "window_size": [2, 6, 12],
    "depth_blocks": [2, 6, 6, 2],
}


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def tensor_sha256(tensor):
    """Hash contiguous CPU tensor value bytes for loader-side verification."""

    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a metadata mapping")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Checkpoint must contain a non-empty model_state_dict")
    invalid = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if invalid:
        raise TypeError(f"model_state_dict contains non-tensors: {invalid[:5]}")
    return state


def validate_profile(checkpoint):
    profile = checkpoint.get("model_profile")
    if not isinstance(profile, Mapping):
        raise TypeError("Checkpoint must contain model_profile metadata")

    mismatches = {}
    for key, expected in EXPECTED_PROFILE.items():
        actual = profile.get(key)
        if isinstance(expected, list) and actual is not None:
            actual = [int(value) for value in actual]
        elif isinstance(expected, int) and actual is not None:
            actual = int(actual)
        if actual != expected:
            mismatches[key] = {"actual": actual, "expected": expected}
    if mismatches:
        raise ValueError(
            "Checkpoint is not exact pgw_lite_pruned_96: "
            + repr(mismatches)
        )
    return profile


def build_runtime_model(checkpoint, state):
    """Build the exact CPU runtime model without loading checkpoint tensors."""

    from pangu_profile_model import build_pangu_model

    profile = validate_profile(checkpoint)
    keys = tuple(state)
    model = build_pangu_model(
        img_size=[721, 1440],
        patch_size=profile["patch_size"],
        embed_dim=profile["embed_dim"],
        num_heads=profile["num_heads"],
        window_size=profile["window_size"],
        depth_blocks=profile["depth_blocks"],
        use_swiglu=any(".mlp.w1." in key for key in keys),
        use_rmsnorm=(
            any(".norm1.weight" in key for key in keys)
            and not any(".norm1.bias" in key for key in keys)
        ),
        use_gqa=any(".q_proj." in key for key in keys),
        share_deep_blocks=profile.get("share_deep_blocks"),
        chunked_attention=False,
    )
    return model.cpu().eval()


def _same_storage(left, right):
    return left.untyped_storage()._cdata == right.untyped_storage()._cdata


def _upper_alias_key(key):
    if LOWER_ALIAS not in key:
        return None
    return key.replace(LOWER_ALIAS, UPPER_ALIAS, 1)


def _dtype_bytes(state):
    totals = Counter()
    for value in state.values():
        totals[str(value.dtype)] += tensor_bytes(value)
    return dict(sorted(totals.items()))


def audit_source(checkpoint, state, model):
    """Return a complete elision manifest or fail before any file is written."""

    validate_profile(checkpoint)
    storage = checkpoint.get("storage_optimization") or {}
    if not isinstance(storage, Mapping):
        raise TypeError("storage_optimization metadata must be a mapping")
    if OPTIMIZATION_KEY in storage or "deterministic_index_elision" in storage:
        raise ValueError("Checkpoint already contains deterministic index elision")

    invalid_dtypes = []
    for key, value in state.items():
        expected_dtype = torch.int64 if key.endswith(INDEX_SUFFIX) else torch.float16
        if value.dtype != expected_dtype:
            invalid_dtypes.append((key, str(value.dtype), str(expected_dtype)))
    if invalid_dtypes:
        raise ValueError(f"Unexpected source tensor dtypes: {invalid_dtypes[:5]}")

    model_state = model.state_dict()
    source_keys = set(state)
    runtime_keys = set(model_state)
    unexpected = sorted(source_keys - runtime_keys)
    if unexpected:
        raise ValueError(f"Checkpoint has keys absent from runtime model: {unexpected[:5]}")

    for key, value in state.items():
        if tuple(value.shape) != tuple(model_state[key].shape):
            raise ValueError(
                f"Source/runtime shape mismatch for {key}: "
                f"{tuple(value.shape)} vs {tuple(model_state[key].shape)}"
            )

    preexisting_missing = sorted(runtime_keys - source_keys)
    for key in preexisting_missing:
        upper_key = _upper_alias_key(key)
        if upper_key is None or upper_key not in state:
            raise ValueError(f"Unsupported source missing key: {key}")
        if upper_key not in model_state or not _same_storage(
            model_state[key], model_state[upper_key]
        ):
            raise ValueError(f"Runtime alias does not share storage: {key}")

    removed_keys = sorted(key for key in state if key.endswith(INDEX_SUFFIX))
    runtime_index_keys = sorted(key for key in model_state if key.endswith(INDEX_SUFFIX))
    if not removed_keys or not runtime_index_keys:
        raise ValueError("No earth_position_index tensors found")

    for key in runtime_index_keys:
        source_key = key if key in state else _upper_alias_key(key)
        if source_key is None or source_key not in state:
            raise ValueError(f"Runtime index has no verified checkpoint source: {key}")
        expected = model_state[key]
        actual = state[source_key]
        if expected.dtype != torch.int64 or actual.dtype != torch.int64:
            raise ValueError(f"Index dtype mismatch for {key}")
        if tuple(expected.shape) != tuple(actual.shape):
            raise ValueError(f"Index shape mismatch for {key}")
        if not torch.equal(expected, actual):
            raise ValueError(f"Index value mismatch for {key}")

    output_state = OrderedDict(
        (key, value) for key, value in state.items() if key not in removed_keys
    )
    generated_indices = {
        key: {
            "shape": list(model_state[key].shape),
            "dtype": str(model_state[key].dtype),
            "sha256": tensor_sha256(model_state[key]),
            "bytes": tensor_bytes(model_state[key]),
        }
        for key in runtime_index_keys
    }
    removed_logical_bytes = sum(tensor_bytes(state[key]) for key in removed_keys)
    return {
        "method": ELISION_METHOD,
        "profile": EXPECTED_PROFILE["name"],
        "source_tensor_count": len(state),
        "source_dtype_bytes": _dtype_bytes(state),
        "output_tensor_count": len(output_state),
        "output_dtype_bytes": _dtype_bytes(output_state),
        "removed_tensor_count": len(removed_keys),
        "removed_logical_bytes": removed_logical_bytes,
        "removed_checkpoint_keys": removed_keys,
        # Alias-compacted lower-case weights are proven independently by the
        # runtime loader.  This allowlist contains deterministic indices only.
        "expected_runtime_missing_keys": runtime_index_keys,
        "generated_indices": generated_indices,
    }


def elide_state_dict(state, removed_keys):
    removed = set(removed_keys)
    output = OrderedDict((key, value) for key, value in state.items() if key not in removed)
    if hasattr(state, "_metadata"):
        output._metadata = state._metadata
    return output


def build_candidate(checkpoint, output_state, manifest):
    existing = checkpoint.get("storage_optimization")
    if existing is None:
        storage_optimization = {}
    elif isinstance(existing, Mapping):
        storage_optimization = dict(existing)
    else:
        raise TypeError("storage_optimization metadata must be a mapping")
    existing_schema = storage_optimization.get("schema_version")
    if existing_schema not in (None, STORAGE_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported storage_optimization schema: {existing_schema}"
        )
    if (
        OPTIMIZATION_KEY in storage_optimization
        or "deterministic_index_elision" in storage_optimization
    ):
        raise ValueError("Checkpoint already contains deterministic index elision")

    storage_optimization["schema_version"] = STORAGE_SCHEMA_VERSION
    storage_optimization[OPTIMIZATION_KEY] = manifest
    candidate = dict(checkpoint)
    candidate["model_state_dict"] = output_state
    candidate["storage_optimization"] = storage_optimization
    return candidate


def _values_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and list(left) == list(right)
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_values_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def verify_saved_candidate(expected, output_path):
    actual = load_checkpoint(output_path)
    if not _values_equal(expected, actual):
        raise ValueError("Saved checkpoint differs from the verified candidate")


def _print_audit(source_path, state, manifest):
    print(f"Source: {Path(source_path).resolve()}")
    print(f"Source size: {Path(source_path).stat().st_size} bytes")
    print(f"Source SHA256: {sha256_file(source_path)}")
    print(f"Source tensors: {len(state)}")
    print(f"Removed index tensors: {manifest['removed_tensor_count']}")
    print(f"Removed index bytes: {manifest['removed_logical_bytes']}")
    print(f"Output tensors: {manifest['output_tensor_count']}")


def elide_checkpoint(source_path, output_path, audit_only=False):
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path == output_path:
        raise ValueError("Source and output paths must differ")
    if not audit_only and output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    checkpoint = load_checkpoint(source_path)
    state = extract_state_dict(checkpoint)
    validate_profile(checkpoint)
    model = build_runtime_model(checkpoint, state)
    try:
        manifest = audit_source(checkpoint, state, model)
    finally:
        del model

    output_state = elide_state_dict(state, manifest["removed_checkpoint_keys"])
    candidate = build_candidate(checkpoint, output_state, manifest)
    _print_audit(source_path, state, manifest)
    if audit_only:
        print("Audit only: no output written")
        return manifest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    try:
        torch.save(candidate, temporary_name)
        verify_saved_candidate(candidate, temporary_name)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)

    print(f"Verified output: {output_path}")
    print(f"Output size: {output_path.stat().st_size} bytes")
    print(f"Output SHA256: {sha256_file(output_path)}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="data/checkpoints/model_fp16_alias_compact.pth"
    )
    parser.add_argument(
        "--output", default="data/checkpoints/model_fp16_index_elided.pth"
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    elide_checkpoint(args.source, args.output, audit_only=args.audit_only)


if __name__ == "__main__":
    main()
