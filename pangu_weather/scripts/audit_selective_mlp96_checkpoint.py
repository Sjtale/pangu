#!/usr/bin/env python3
"""Fail-closed audit for a parameter-only SelectiveMLP-96 checkpoint."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import torch


PANGU_DIR = Path(__file__).resolve().parents[1]
if str(PANGU_DIR) not in sys.path:
    sys.path.insert(0, str(PANGU_DIR))

from selective_mlp96 import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    INITIALIZATION_METHOD,
    MLP_RATIO_BLOCKS,
    PROFILE_NAME,
    PROFILE_SPEC,
    TARGET_MLP_BLOCKS,
    validate_profile,
)


MIB = 1024 ** 2
DEFAULT_MAX_FILE_MIB = 29.1
EXPECTED_IMPORTANCE_FORMULA = (
    "sqrt(mean(GELU(fc1(x))^2))*l2_norm(fc2[:,neuron])"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PARAMETER_SUFFIXES = (".weight", ".bias", "earth_position_bias_table")
FORBIDDEN_BUFFER_SUFFIXES = ("earth_position_index", "attn_mask")
FORBIDDEN_QUANT_METADATA = ("quantization", "int4_storage")


def _mib(value):
    return round(float(value) / MIB, 6)


def canonical_parameter_key(key):
    """Collapse OneScience's duplicate wrapper registrations to one key."""

    return (
        str(key)
        .replace(".Fuser.", ".fuser.")
        .replace(".Sampler.", ".sampler.")
        .replace(".Reconvery.", ".recovery.")
    )


def canonical_tensor_state(state):
    if not isinstance(state, dict):
        raise TypeError("model_state_dict must be a tensor mapping")
    canonical = OrderedDict()
    original_keys = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State value is not a tensor: {key}")
        normalized = canonical_parameter_key(key)
        if normalized in canonical:
            raise ValueError(
                "Parameter-only export contains duplicate alias keys: "
                f"{original_keys[normalized]!r}, {str(key)!r}"
            )
        canonical[normalized] = tensor
        original_keys[normalized] = str(key)
    if not canonical:
        raise ValueError("model_state_dict is empty")
    return canonical


def expected_mlp_shapes():
    shapes = OrderedDict()
    stage_dims = (96, 192, 192, 96)
    for stage_index, (dim, ratios) in enumerate(
        zip(stage_dims, MLP_RATIO_BLOCKS), start=1
    ):
        for block_index, ratio in enumerate(ratios):
            prefix = (
                f"layer{stage_index}.fuser.blocks.{block_index}.transformer.mlp"
            )
            hidden = dim * int(ratio)
            shapes[prefix + ".fc1.weight"] = (hidden, dim)
            shapes[prefix + ".fc1.bias"] = (hidden,)
            shapes[prefix + ".fc2.weight"] = (dim, hidden)
            shapes[prefix + ".fc2.bias"] = (dim,)
    return shapes


def validate_initialization_metadata(metadata):
    """Validate the initializer's emitted provenance and full coverage record."""

    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint has no initialization provenance")
    expected = {
        "method": INITIALIZATION_METHOD,
        "profile_name": PROFILE_NAME,
        "human_label": "SelectiveMLP-96",
        "source": "full_depth_ratio4_pruned96",
        "teacher": "official_full192",
        "mlp_ratio_blocks": MLP_RATIO_BLOCKS,
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Initialization provenance mismatch: actual={actual}, expected={expected}"
        )

    for key in ("source_sha256", "teacher_sha256", "init_sha256"):
        if SHA256_PATTERN.fullmatch(str(metadata.get(key, ""))) is None:
            raise ValueError(f"Initialization has invalid {key}")
    if metadata.get("strict_coverage") is not True:
        raise ValueError("Initialization strict_coverage must be true")
    random_count = metadata.get("random_initialized_parameters")
    if isinstance(random_count, bool) or random_count != 0:
        raise ValueError("Initialization must record random_initialized_parameters=0")

    for total_key, covered_key in (
        ("state_tensor_keys", "covered_state_tensor_keys"),
        ("parameter_state_keys", "covered_parameter_state_keys"),
    ):
        total = metadata.get(total_key)
        covered = metadata.get(covered_key)
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
            or covered != total
        ):
            raise ValueError(
                f"Initialization coverage mismatch: {covered_key}={covered!r}, "
                f"{total_key}={total!r}"
            )

    target_prefixes = {
        canonical_parameter_key(
            f"{stage}.Fuser.blocks.{block}.transformer.mlp"
        )
        for stage, block in TARGET_MLP_BLOCKS
    }
    raw_indices = metadata.get("neuron_indices")
    if not isinstance(raw_indices, dict):
        raise ValueError("Initialization neuron_indices are missing")
    indices = {
        canonical_parameter_key(key): value for key, value in raw_indices.items()
    }
    if set(indices) != target_prefixes:
        raise ValueError("Initialization neuron_indices do not cover the 11-block schedule")
    for prefix, selected in indices.items():
        if (
            not isinstance(selected, list)
            or len(selected) != 384
            or selected != sorted(set(selected))
            or isinstance(selected[0], bool)
            or selected[0] < 0
            or selected[-1] >= 768
        ):
            raise ValueError(f"Invalid paired-neuron indices for {prefix}")

    calibration = metadata.get("activation_calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Initialization activation_calibration is missing")
    expected_calibration = {
        "dataset": "official_train",
        "input_count": 32,
        "tokens_per_input": 4096,
        "tokens_per_mlp": 131072,
        "seed": 20260713,
        "importance": EXPECTED_IMPORTANCE_FORMULA,
    }
    actual_calibration = {
        key: calibration.get(key) for key in expected_calibration
    }
    if actual_calibration != expected_calibration:
        raise ValueError(
            "Initialization activation calibration mismatch: "
            f"actual={actual_calibration}, expected={expected_calibration}"
        )
    input_indices = calibration.get("input_indices")
    if (
        not isinstance(input_indices, list)
        or len(input_indices) != 32
        or input_indices != sorted(set(input_indices))
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in input_indices
        )
    ):
        raise ValueError("Initialization must record 32 unique ordered train indices")


def validate_parameter_only_fp16_state(state):
    forbidden_buffers = [
        key for key in state if key.endswith(FORBIDDEN_BUFFER_SUFFIXES)
    ]
    non_parameters = [
        key for key in state if not key.endswith(PARAMETER_SUFFIXES)
    ]
    quantized_keys = [
        key
        for key, tensor in state.items()
        if (
            "int4" in key.lower()
            or "packed" in key.lower()
            or key.lower().endswith(("_scale", "zero_point"))
            or bool(getattr(tensor, "is_quantized", False))
        )
    ]
    if forbidden_buffers or non_parameters or quantized_keys:
        raise ValueError(
            "Checkpoint is not a parameter-only unquantized export: "
            f"buffers={forbidden_buffers[:5]}, non_parameters={non_parameters[:5]}, "
            f"quantized={quantized_keys[:5]}"
        )

    bad_tensors = [
        key
        for key, tensor in state.items()
        if (
            tensor.dtype != torch.float16
            or tensor.device.type != "cpu"
            or tensor.is_sparse
        )
    ]
    if bad_tensors:
        raise ValueError(
            "All exported parameters must be dense CPU FP16 tensors: "
            f"{bad_tensors[:5]}"
        )


def validate_mlp_shapes(state):
    expected = expected_mlp_shapes()
    missing = sorted(set(expected) - set(state))
    mismatched = sorted(
        key
        for key, shape in expected.items()
        if key in state and tuple(state[key].shape) != shape
    )
    if missing or mismatched:
        raise ValueError(
            "SelectiveMLP-96 MLP schedule/state-shape mismatch: "
            f"missing={missing[:5]}, shape_mismatch={mismatched[:5]}"
        )
    return len(expected)


def validate_model_parameter_keys(state, model):
    expected = OrderedDict()
    for name, parameter in model.named_parameters():
        key = canonical_parameter_key(name)
        if key in expected:
            raise ValueError(f"Built model has duplicate canonical parameter key: {key}")
        expected[key] = parameter

    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Checkpoint trainable-key mismatch against built model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={mismatched[:5]}"
        )
    return {
        "performed": True,
        "expected_trainable_keys": len(expected),
        "missing_trainable_keys": 0,
        "unexpected_trainable_keys": 0,
        "shape_mismatches": 0,
        "model_parameter_count": sum(parameter.numel() for parameter in expected.values()),
    }


def audit_payload(
    checkpoint,
    *,
    file_size_bytes,
    checkpoint_path=None,
    model=None,
    max_file_mib=DEFAULT_MAX_FILE_MIB,
):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    forbidden_metadata = [key for key in FORBIDDEN_QUANT_METADATA if key in checkpoint]
    if forbidden_metadata:
        raise ValueError(f"Quantization metadata is forbidden: {forbidden_metadata}")

    profile = checkpoint.get("model_profile")
    if not isinstance(profile, dict):
        raise ValueError("Checkpoint has no model_profile metadata")
    validate_profile(profile)
    validate_initialization_metadata(checkpoint.get("initialization"))

    state = canonical_tensor_state(checkpoint.get("model_state_dict"))
    validate_parameter_only_fp16_state(state)
    verified_mlp_tensors = validate_mlp_shapes(state)

    parameter_count = sum(tensor.numel() for tensor in state.values())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"SelectiveMLP-96 parameter count mismatch: "
            f"{parameter_count} != {EXPECTED_PARAMETER_COUNT}"
        )

    if model is None:
        model_check = {
            "performed": False,
            "expected_trainable_keys": None,
            "missing_trainable_keys": None,
            "unexpected_trainable_keys": None,
            "shape_mismatches": None,
            "model_parameter_count": None,
        }
    else:
        model_check = validate_model_parameter_keys(state, model)
        if model_check["model_parameter_count"] != EXPECTED_PARAMETER_COUNT:
            raise ValueError(
                "Built SelectiveMLP-96 model parameter count mismatch: "
                f"{model_check['model_parameter_count']} != {EXPECTED_PARAMETER_COUNT}"
            )

    file_size_bytes = int(file_size_bytes)
    max_file_mib = float(max_file_mib)
    if file_size_bytes < 0 or not math.isfinite(max_file_mib) or max_file_mib <= 0:
        raise ValueError("Checkpoint size and max_file_mib must be positive and finite")
    max_file_bytes = max_file_mib * MIB
    if file_size_bytes > max_file_bytes:
        raise ValueError(
            f"Checkpoint file-size gate failed: {_mib(file_size_bytes)} MiB "
            f"> {max_file_mib:.4f} MiB"
        )

    logical_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in state.values()
    )
    dtype_counts = Counter(str(tensor.dtype) for tensor in state.values())
    return {
        "passed": True,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "profile": profile,
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_tensor_count": len(state),
        "verified_mlp_tensor_shapes": verified_mlp_tensors,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "parameter_only": True,
        "unquantized_fp16": True,
        "logical_bytes": logical_bytes,
        "logical_mib": _mib(logical_bytes),
        "file_bytes": file_size_bytes,
        "file_mib": _mib(file_size_bytes),
        "max_file_mib": max_file_mib,
        "file_size_gate_passed": True,
        "initialization": {
            "method": checkpoint["initialization"]["method"],
            "source_sha256": checkpoint["initialization"]["source_sha256"],
            "teacher_sha256": checkpoint["initialization"]["teacher_sha256"],
            "init_sha256": checkpoint["initialization"]["init_sha256"],
            "random_initialized_parameters": 0,
            "strict_coverage": True,
        },
        "model_key_check": model_check,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checkpoint(path, *, model=None, max_file_mib=DEFAULT_MAX_FILE_MIB):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    report = audit_payload(
        checkpoint,
        file_size_bytes=path.stat().st_size,
        checkpoint_path=path,
        model=model,
        max_file_mib=max_file_mib,
    )
    report["checkpoint_sha256"] = sha256_file(path)
    return report


def build_expected_model():
    from pangu_profile_model import build_pangu_model, selective_mlp_96_profile

    profile = selective_mlp_96_profile()
    env_names = ("PANGU_CHUNKED_MLP", "PANGU_COMPACT_ATTN_MASK")
    previous = {name: os.environ.get(name) for name in env_names}
    os.environ.update({name: "0" for name in env_names})
    try:
        return build_pangu_model(
            img_size=[721, 1440],
            patch_size=profile["patch_size"],
            embed_dim=profile["embed_dim"],
            num_heads=profile["num_heads"],
            window_size=profile["window_size"],
            depth_blocks=profile["depth_blocks"],
            mlp_ratio_blocks=profile["mlp_ratio_blocks"],
            use_swiglu=False,
            use_rmsnorm=False,
            use_gqa=False,
            share_deep_blocks=False,
            chunked_attention=False,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Parameter-only SelectiveMLP-96 .pth")
    parser.add_argument(
        "--max-file-mib",
        type=float,
        default=DEFAULT_MAX_FILE_MIB,
        help="Fail when the checkpoint exceeds this binary-MiB size (default: 29.1)",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip exact trainable-key comparison against a freshly built model",
    )
    parser.add_argument("--json-out", help="Optional path for the JSON audit report")
    return parser.parse_args(argv)


def _write_json(path, payload):
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    args = parse_args(argv)
    try:
        model = None if args.skip_model_check else build_expected_model()
        report = audit_checkpoint(
            args.checkpoint,
            model=model,
            max_file_mib=args.max_file_mib,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        failure = {
            "passed": False,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(args.json_out, failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1

    _write_json(args.json_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
