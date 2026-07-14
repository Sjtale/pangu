#!/usr/bin/env python3
"""Build the selective 1-4-4-1 student only from official full_192 weights."""

import argparse
import hashlib
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F


PANGU_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANGU_DIR))

from onescience.utils.YParams import YParams
from pangu_profile_model import build_pangu_model
from scripts.prune_structured import (
    _attention_heads,
    _migrate_tensor,
    _mlp_indices,
    _residual_indices,
    _state_depths,
)


PROFILE_NAME = "pangu_selective_1441_full192"
SOURCE_PATCH_SIZE = [2, 4, 4]
SOURCE_EMBED_DIM = 192
SOURCE_NUM_HEADS = [6, 12, 12, 6]
SOURCE_DEPTHS = [2, 6, 6, 2]
TARGET_PATCH_SIZE = [2, 8, 8]
TARGET_EMBED_DIM = 96
TARGET_NUM_HEADS = [3, 6, 6, 3]
TARGET_DEPTHS = [1, 4, 4, 1]
TARGET_WINDOW_SIZE = [2, 6, 12]
TARGET_MLP_RATIO = 4
EXPECTED_PARAMETER_COUNT = 10_575_945
BLOCK_MAP = [[0], [0, 1, 4, 5], [0, 1, 4, 5], [0]]
BLOCK_PATTERN = re.compile(r"^(layer([1-4])\.(?:fuser|Fuser)\.blocks\.)(\d+)(\..+)$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Official checkpoint must contain a tensor state dict")
    cleaned = OrderedDict()
    for key, value in state.items():
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        if clean_key in cleaned:
            raise ValueError(f"Duplicate state key after module-prefix removal: {clean_key}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State value is not a tensor: {key}")
        cleaned[clean_key] = value
    return cleaned


def validate_full192_source(checkpoint, source_state):
    forbidden_metadata = sorted(
        key
        for key in ("distillation", "pruning", "quantization", "int4_storage")
        if key in checkpoint
    )
    if forbidden_metadata:
        raise ValueError(
            "Initialization source is not the official full_192 checkpoint; "
            f"found metadata={forbidden_metadata}"
        )
    if any(
        key.endswith("_scale")
        or key.endswith(".int4_scale")
        or value.dtype in {torch.int8, torch.uint8}
        for key, value in source_state.items()
    ):
        raise ValueError("Official full_192 initialization source must be unquantized")

    profile = checkpoint.get("model_profile")
    if profile is not None:
        actual = {
            "patch_size": [int(value) for value in profile.get("patch_size", [])],
            "embed_dim": int(profile.get("embed_dim", -1)),
            "num_heads": [int(value) for value in profile.get("num_heads", [])],
            "depth_blocks": [
                int(value) for value in profile.get("depth_blocks", SOURCE_DEPTHS)
            ],
        }
        expected = {
            "patch_size": SOURCE_PATCH_SIZE,
            "embed_dim": SOURCE_EMBED_DIM,
            "num_heads": SOURCE_NUM_HEADS,
            "depth_blocks": SOURCE_DEPTHS,
        }
        if actual != expected:
            raise ValueError(
                f"Official full_192 model_profile mismatch: actual={actual}, expected={expected}"
            )

    embed2d_key = "patchembed2d.embedder.proj.weight"
    embed3d_key = "patchembed3d.embedder.proj.weight"
    if embed2d_key not in source_state or embed3d_key not in source_state:
        raise ValueError("Official full_192 checkpoint is missing patch embedding weights")
    if tuple(source_state[embed2d_key].shape) != (192, 7, 1, 4, 4):
        raise ValueError(
            f"Official 2D patch embedding shape mismatch: {tuple(source_state[embed2d_key].shape)}"
        )
    if tuple(source_state[embed3d_key].shape) != (192, 5, 2, 4, 4):
        raise ValueError(
            f"Official 3D patch embedding shape mismatch: {tuple(source_state[embed3d_key].shape)}"
        )
    actual_depths = _state_depths(source_state)
    if actual_depths != SOURCE_DEPTHS:
        raise ValueError(
            f"Official full_192 depth mismatch: {actual_depths} != {SOURCE_DEPTHS}"
        )


def validate_full192_structure(source_state, expected_state):
    expected_shapes = {
        key: tuple(value.shape) for key, value in expected_state.items()
    }
    missing = sorted(set(expected_shapes) - set(source_state))
    unexpected = sorted(set(source_state) - set(expected_shapes))
    mismatched = sorted(
        key
        for key, shape in expected_shapes.items()
        if key in source_state and tuple(source_state[key].shape) != shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Official full_192 state structure mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={mismatched[:5]}"
        )


def validate_target_model(model):
    state = model.state_dict()
    if _state_depths(state) != TARGET_DEPTHS:
        raise ValueError("Target model does not have exact [1,4,4,1] depth")
    expected_shifts = [
        [(0, 0, 0) if index % 2 == 0 else (1, 3, 6) for index in range(depth)]
        for depth in TARGET_DEPTHS
    ]
    actual_shifts = []
    for stage_index in range(1, 5):
        fuser = getattr(model, f"layer{stage_index}").fuser
        actual_shifts.append(
            [tuple(int(value) for value in block.shift_size) for block in fuser.blocks]
        )
    if actual_shifts != expected_shifts:
        raise ValueError(
            f"Target shift phases mismatch: {actual_shifts} != {expected_shifts}"
        )
    if tuple(state["patchembed2d.embedder.proj.weight"].shape[-2:]) != (8, 8):
        raise ValueError("Target surface patch embedding is not patch8")
    if tuple(state["patchembed3d.embedder.proj.weight"].shape[-3:]) != (2, 8, 8):
        raise ValueError("Target upper-air patch embedding is not [2,8,8]")
    surface_channels = int(state["patchrecovery2d.recovery.proj.bias"].numel())
    upper_variables = int(state["patchrecovery3d.recovery.proj.bias"].numel())
    if surface_channels + 13 * upper_variables != 69:
        raise ValueError("Target model does not recover all 69 output channels")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"Target parameter count mismatch: {parameter_count} != {EXPECTED_PARAMETER_COUNT}"
        )


def source_key_for_target(target_key):
    match = BLOCK_PATTERN.match(target_key)
    if match is None:
        return target_key
    stage = int(match.group(2)) - 1
    target_block = int(match.group(3))
    if target_block >= len(BLOCK_MAP[stage]):
        raise ValueError(f"Target block is outside fixed 1-4-4-1 map: {target_key}")
    source_block = BLOCK_MAP[stage][target_block]
    return f"{match.group(1)}{source_block}{match.group(4)}"


def resize_patch_weight(value, target_shape, preserve_embedding_scale):
    if value.ndim != len(target_shape) or value.ndim < 4:
        raise ValueError(
            f"Patch weight rank mismatch: source={tuple(value.shape)}, target={tuple(target_shape)}"
        )
    if tuple(value.shape[:-2]) != tuple(target_shape[:-2]):
        raise ValueError(
            "Patch weight non-spatial dimensions must already be selected: "
            f"source={tuple(value.shape)}, target={tuple(target_shape)}"
        )
    source_height, source_width = value.shape[-2:]
    target_height, target_width = target_shape[-2:]
    resized = F.interpolate(
        value.float().reshape(-1, 1, source_height, source_width),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    ).reshape(target_shape)
    if preserve_embedding_scale:
        resized = resized * (
            (source_height * source_width) / (target_height * target_width)
        )
    return resized.to(value.dtype)


def interpolate_earth_bias(value, target_shape):
    if value.ndim != 3 or len(target_shape) != 3:
        raise ValueError("Earth-position bias must be rank three")
    if value.shape[0] != target_shape[0] or value.shape[2] != target_shape[2]:
        raise ValueError(
            f"Earth-position bias dimensions mismatch: {tuple(value.shape)} -> {tuple(target_shape)}"
        )
    source_windows = int(value.shape[1])
    target_windows = int(target_shape[1])
    heads = int(value.shape[2])
    if source_windows % 4 == 0 and target_windows % 4 == 0:
        source_latitude = source_windows // 4
        target_latitude = target_windows // 4
        series = value.float().view(value.shape[0], 4, source_latitude, heads)
        series = series.permute(0, 3, 1, 2).reshape(-1, 1, source_latitude)
        resized = F.interpolate(
            series,
            size=target_latitude,
            mode="linear",
            align_corners=False,
        )
        resized = resized.view(value.shape[0], heads, 4, target_latitude)
        resized = resized.permute(0, 2, 3, 1).reshape(target_shape)
    else:
        series = value.float().permute(0, 2, 1).reshape(-1, 1, source_windows)
        resized = F.interpolate(
            series,
            size=target_windows,
            mode="linear",
            align_corners=False,
        )
        resized = resized.view(value.shape[0], heads, target_windows).permute(0, 2, 1)
    return resized.to(value.dtype)


def migrate_parameter(
    target_key,
    target_tensor,
    source_state,
    residual,
    attention_heads,
    mlp_hidden,
):
    source_key = source_key_for_target(target_key)
    if source_key not in source_state:
        raise KeyError(f"Official full_192 checkpoint is missing {source_key} for {target_key}")
    migrated = _migrate_tensor(
        source_key,
        source_state[source_key],
        target_tensor,
        residual,
        attention_heads,
        mlp_hidden,
        SOURCE_EMBED_DIM,
        SOURCE_NUM_HEADS,
    )
    if target_key.endswith(".proj.weight") and target_key.startswith(
        ("patchembed2d.", "patchembed3d.", "patchrecovery2d.", "patchrecovery3d.")
    ):
        migrated = resize_patch_weight(
            migrated,
            target_tensor.shape,
            preserve_embedding_scale=target_key.startswith("patchembed"),
        )
    elif target_key.endswith("earth_position_bias_table"):
        migrated = interpolate_earth_bias(migrated, target_tensor.shape)
    if tuple(migrated.shape) != tuple(target_tensor.shape):
        raise ValueError(
            f"Deterministic migration shape mismatch for {target_key}: "
            f"{tuple(migrated.shape)} != {tuple(target_tensor.shape)}"
        )
    return migrated, source_key


def parameter_state_keys(model):
    return list(dict(model.named_parameters(remove_duplicate=False)))


def index_metadata(indices):
    return {
        key: [int(value) for value in tensor.tolist()]
        for key, tensor in indices.items()
    }


def atomic_deterministic_save(payload, output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = output_path.with_suffix(output_path.suffix + ".tmp1")
    second = output_path.with_suffix(output_path.suffix + ".tmp2")
    try:
        torch.save(payload, first)
        torch.save(payload, second)
        first_hash = sha256_file(first)
        second_hash = sha256_file(second)
        if first_hash != second_hash:
            raise RuntimeError(
                "torch.save did not produce byte-reproducible checkpoint payloads: "
                f"{first_hash} != {second_hash}"
            )
        os.replace(first, output_path)
    finally:
        for temporary in (first, second):
            if temporary.exists():
                temporary.unlink()
    return first_hash


def build_profile(cfg):
    configured = cfg.student_profiles[PROFILE_NAME]
    actual = {
        "patch_size": [int(value) for value in configured.patch_size],
        "embed_dim": int(configured.embed_dim),
        "num_heads": [int(value) for value in configured.num_heads],
        "depth_blocks": [int(value) for value in configured.depth_blocks],
        "window_size": [int(value) for value in configured.window_size],
        "mlp_ratio": int(configured.mlp_ratio),
    }
    expected = {
        "patch_size": TARGET_PATCH_SIZE,
        "embed_dim": TARGET_EMBED_DIM,
        "num_heads": TARGET_NUM_HEADS,
        "depth_blocks": TARGET_DEPTHS,
        "window_size": TARGET_WINDOW_SIZE,
        "mlp_ratio": TARGET_MLP_RATIO,
    }
    if actual != expected:
        raise ValueError(
            f"Configured {PROFILE_NAME} mismatch: actual={actual}, expected={expected}"
        )
    return {
        "name": PROFILE_NAME,
        **expected,
    }


def initialize(args):
    source_path = Path(args.source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Official full_192 checkpoint not found: {source_path}")
    if source_path.name != "model_bak.pth":
        raise ValueError("Initialization source filename must be model_bak.pth")
    outputs = [Path(args.output)]
    if args.inference_output:
        outputs.append(Path(args.inference_output))
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Training and inference outputs must be different files")
    existing_outputs = [str(path) for path in outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {existing_outputs}"
        )
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = clean_state_dict(checkpoint.get("model_state_dict", checkpoint))
    validate_full192_source(checkpoint, source_state)

    cfg = YParams(args.config, "model")
    profile = build_profile(cfg)
    os.environ["PANGU_COMPACT_ATTN_MASK"] = "0"
    source_model = build_pangu_model(
        img_size=[721, 1440],
        patch_size=SOURCE_PATCH_SIZE,
        embed_dim=SOURCE_EMBED_DIM,
        num_heads=SOURCE_NUM_HEADS,
        window_size=TARGET_WINDOW_SIZE,
        depth_blocks=SOURCE_DEPTHS,
        use_swiglu=False,
        use_rmsnorm=False,
        use_gqa=False,
        share_deep_blocks=False,
        chunked_attention=False,
    )
    validate_full192_structure(source_state, source_model.state_dict())
    del source_model
    target_model = build_pangu_model(
        img_size=[721, 1440],
        patch_size=TARGET_PATCH_SIZE,
        embed_dim=TARGET_EMBED_DIM,
        num_heads=TARGET_NUM_HEADS,
        window_size=TARGET_WINDOW_SIZE,
        depth_blocks=TARGET_DEPTHS,
        use_swiglu=False,
        use_rmsnorm=False,
        use_gqa=False,
        share_deep_blocks=False,
        chunked_attention=False,
    )
    validate_target_model(target_model)
    target_state = target_model.state_dict()
    target_parameter_keys = parameter_state_keys(target_model)
    target_parameter_key_set = set(target_parameter_keys)

    residual = _residual_indices(source_state, SOURCE_EMBED_DIM, TARGET_EMBED_DIM)
    attention_heads = _attention_heads(
        source_state,
        SOURCE_EMBED_DIM,
        TARGET_EMBED_DIM,
        SOURCE_NUM_HEADS,
    )
    mlp_hidden = _mlp_indices(source_state, SOURCE_EMBED_DIM, TARGET_EMBED_DIM)

    migrated_state = OrderedDict()
    source_key_map = OrderedDict()
    for target_key, target_tensor in target_state.items():
        if target_key in target_parameter_key_set:
            migrated, source_key = migrate_parameter(
                target_key,
                target_tensor,
                source_state,
                residual,
                attention_heads,
                mlp_hidden,
            )
            migrated_state[target_key] = migrated
            source_key_map[target_key] = source_key
        else:
            migrated_state[target_key] = target_tensor

    covered = set(source_key_map)
    missing_parameters = sorted(target_parameter_key_set - covered)
    if missing_parameters:
        raise RuntimeError(
            f"Randomly initialized target parameters remain: {missing_parameters[:10]}"
        )
    target_model.load_state_dict(migrated_state, strict=True)
    target_model.half()
    full_state = OrderedDict(
        (key, value.detach().cpu()) for key, value in target_model.state_dict().items()
    )
    inference_state = OrderedDict(
        (key, full_state[key]) for key in parameter_state_keys(target_model)
    )

    initialization = {
        "method": "full192_selective_1441_deterministic",
        "source": "official_full_192",
        "source_file": source_path.name,
        "source_sha256": sha256_file(source_path),
        "source_profile": {
            "patch_size": SOURCE_PATCH_SIZE,
            "embed_dim": SOURCE_EMBED_DIM,
            "num_heads": SOURCE_NUM_HEADS,
            "depth_blocks": SOURCE_DEPTHS,
        },
        "target_profile": profile,
        "block_map": BLOCK_MAP,
        "residual_channels": index_metadata(residual),
        "attention_heads": index_metadata(attention_heads),
        "mlp_hidden": index_metadata(mlp_hidden),
        "source_key_map": source_key_map,
        "transforms": {
            "patch_embedding": "bicubic_spatial_resize_with_area_scale",
            "patch_recovery": "bicubic_spatial_resize_preserve_coefficient_scale",
            "earth_position_bias": "head_selection_then_linear_latitude_interpolation",
            "width": "global_residual_complete_head_paired_mlp_selection",
        },
        "parameter_state_keys": len(target_parameter_keys),
        "covered_parameter_state_keys": len(covered),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "random_initialized_parameters": 0,
        "deterministic_geometry_buffers": len(full_state) - len(inference_state),
    }
    training_payload = {
        "model_state_dict": full_state,
        "model_profile": profile,
        "initialization": initialization,
    }
    inference_payload = {
        "model_state_dict": inference_state,
        "model_profile": profile,
        "initialization": initialization,
    }

    training_hash = atomic_deterministic_save(training_payload, args.output)
    saved = torch.load(args.output, map_location="cpu", weights_only=False)
    verification_model = build_pangu_model(
        img_size=[721, 1440],
        patch_size=TARGET_PATCH_SIZE,
        embed_dim=TARGET_EMBED_DIM,
        num_heads=TARGET_NUM_HEADS,
        window_size=profile["window_size"],
        depth_blocks=TARGET_DEPTHS,
        use_swiglu=False,
        use_rmsnorm=False,
        use_gqa=False,
        share_deep_blocks=False,
        chunked_attention=False,
    ).half()
    verification_model.load_state_dict(saved["model_state_dict"], strict=True)

    inference_hash = None
    if args.inference_output:
        inference_hash = atomic_deterministic_save(
            inference_payload, args.inference_output
        )

    parameters = sum(parameter.numel() for parameter in target_model.parameters())
    fp16_parameter_bytes = sum(
        parameter.numel() * 2 for parameter in target_model.parameters()
    )
    print(f"profile={PROFILE_NAME}")
    print(f"parameters={parameters}")
    print(f"fp16_parameter_mib={fp16_parameter_bytes / 2**20:.4f}")
    print(f"parameter_coverage={len(covered)}/{len(target_parameter_keys)}")
    print("random_initialized_parameters=0")
    print(f"training_output={args.output}")
    print(f"training_sha256={training_hash}")
    if args.inference_output:
        print(f"inference_output={args.inference_output}")
        print(f"inference_sha256={inference_hash}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inference-output")
    parser.add_argument(
        "--config", default=str(PANGU_DIR / "conf" / "config.yaml")
    )
    return parser.parse_args()


if __name__ == "__main__":
    initialize(parse_args())
