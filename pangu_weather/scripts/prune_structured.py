"""Create a width-pruned Pangu-Weather checkpoint from official weights.

The pruning keeps the original per-head width (32) and removes complete
attention heads together with globally consistent residual channels. This
preserves every residual, sampling, and skip-connection shape dependency.

Run from ``pangu_weather`` after activating the competition environment:

    python scripts/prune_structured.py
"""

import argparse
import hashlib
import itertools
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pangu_profile_model import build_pangu_model
from onescience.utils.YParams import YParams


S96_PROFILE = "uv_s96_patch8_w96_shallow"
S96_SOURCE_PATCH_SIZE = [2, 8, 8]
S96_SOURCE_EMBED_DIM = 96
S96_SOURCE_NUM_HEADS = [3, 6, 6, 3]
S96_SOURCE_DEPTHS = [2, 6, 6, 2]
S96_TARGET_DEPTHS = [1, 2, 2, 1]


def _index(values, dim, indices):
    return torch.index_select(values, dim, indices.to(values.device))


def _grouped(indices, groups, source_width):
    return torch.cat([indices + group * source_width for group in range(groups)])


def _stage(key):
    if key.startswith(("layer1.", "layer4.")):
        return "shallow"
    if key.startswith(("layer2.", "layer3.")):
        return "deep"
    return None


def _top_indices(scores, count):
    selected = torch.topk(scores, count, largest=True, sorted=False).indices
    return selected.sort().values.cpu()


def _residual_indices(state, source_width, target_width):
    """Rank globally shared residual channels by adjacent weight magnitude."""
    shallow = torch.zeros(source_width, dtype=torch.float64)
    deep = torch.zeros(source_width * 2, dtype=torch.float64)

    def add(score, value):
        score += value.detach().float().cpu().double()

    for key, tensor in state.items():
        stage = _stage(key)
        if key.startswith(("patchembed2d.", "patchembed3d.")) and key.endswith(
            "proj.weight"
        ):
            add(shallow, tensor.abs().flatten(1).sum(1))
        elif stage is not None:
            score = shallow if stage == "shallow" else deep
            width = len(score)
            if key.endswith(("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")):
                add(score, tensor.abs())
            elif key.endswith("qkv.weight"):
                add(score, tensor.abs().sum(0))
            elif key.endswith("proj.weight"):
                add(score, tensor.abs().sum(1))
            elif key.endswith("proj.bias"):
                add(score, tensor.abs())
            elif key.endswith("mlp.fc1.weight"):
                add(score, tensor.abs().sum(0))
            elif key.endswith("mlp.fc2.weight"):
                add(score, tensor.abs().sum(1))
            elif key.endswith("mlp.fc2.bias") and tensor.numel() == width:
                add(score, tensor.abs())
        elif key.startswith("downsample."):
            if key.endswith("linear.weight"):
                add(deep, tensor.abs().sum(1))
                add(shallow, tensor.abs().reshape(source_width * 2, 4, source_width).sum((0, 1)))
            elif key.endswith(("norm.weight", "norm.bias")):
                add(shallow, tensor.abs().reshape(4, source_width).sum(0))
        elif key.startswith("upsample."):
            if key.endswith("linear1.weight"):
                add(shallow, tensor.abs().reshape(4, source_width, source_width * 2).sum((0, 2)))
                add(deep, tensor.abs().sum(0))
            elif key.endswith("linear2.weight"):
                add(shallow, tensor.abs().sum(0) + tensor.abs().sum(1))
            elif key.endswith(("norm.weight", "norm.bias")):
                add(shallow, tensor.abs())
        elif key.startswith(("patchrecovery2d.", "patchrecovery3d.")) and key.endswith(
            "proj.weight"
        ):
            add(shallow, tensor.abs().flatten(1).sum(1).reshape(2, source_width).sum(0))

    return {
        "shallow": _top_indices(shallow, target_width),
        "deep": _top_indices(deep, target_width * 2),
    }


def _attention_heads(state, source_width, target_width, source_heads):
    selected = {}
    for key, tensor in state.items():
        if not key.endswith("qkv.weight"):
            continue
        stage = _stage(key)
        if stage is None:
            continue
        width = source_width if stage == "shallow" else source_width * 2
        pruned_width = target_width if stage == "shallow" else target_width * 2
        heads = source_heads[0] if stage == "shallow" else source_heads[1]
        target_heads = pruned_width // (width // heads)
        head_dim = width // heads
        qkv = tensor.detach().float().reshape(3, heads, head_dim, width)
        scores = qkv.abs().sum((0, 2, 3)).cpu().double()
        prefix = key[: -len(".qkv.weight")]
        selected[prefix] = _top_indices(scores, target_heads)
    return selected


def _mlp_indices(state, source_width, target_width):
    selected = {}
    for key, fc1 in state.items():
        if not key.endswith("mlp.fc1.weight"):
            continue
        stage = _stage(key)
        if stage is None:
            continue
        width = source_width if stage == "shallow" else source_width * 2
        pruned_width = target_width if stage == "shallow" else target_width * 2
        prefix = key[: -len(".fc1.weight")]
        fc2 = state[prefix + ".fc2.weight"]
        scores = fc1.detach().float().abs().sum(1)
        scores += fc2.detach().float().abs().sum(0)
        selected[prefix] = _top_indices(scores.cpu().double(), pruned_width * 4)
    return selected


def _migrate_tensor(
    key,
    source,
    target,
    residual,
    heads,
    hidden,
    source_width,
    source_heads,
):
    stage = _stage(key)
    if key.startswith(("patchembed2d.", "patchembed3d.")) and key.endswith(
        ("proj.weight", "proj.bias")
    ):
        return _index(source, 0, residual["shallow"])

    if stage is not None:
        channels = residual[stage]
        width = source_width if stage == "shallow" else source_width * 2
        num_heads = source_heads[0] if stage == "shallow" else source_heads[1]
        head_dim = width // num_heads
        attn_prefix = key.split(".qkv.")[0] if ".qkv." in key else None
        if attn_prefix is None and ".proj." in key and ".attn." in key:
            attn_prefix = key.split(".proj.")[0]
        if attn_prefix is None and key.endswith("earth_position_bias_table"):
            attn_prefix = key[: -len(".earth_position_bias_table")]

        if attn_prefix is not None:
            head_ids = heads[attn_prefix]
            head_channels = torch.cat(
                [torch.arange(head * head_dim, (head + 1) * head_dim) for head in head_ids]
            )
            if key.endswith("qkv.weight"):
                qkv_rows = torch.cat(
                    [head_channels + part * width for part in range(3)]
                )
                return _index(_index(source, 0, qkv_rows), 1, channels)
            if key.endswith("qkv.bias"):
                qkv_rows = torch.cat(
                    [head_channels + part * width for part in range(3)]
                )
                return _index(source, 0, qkv_rows)
            if key.endswith("earth_position_bias_table"):
                return _index(source, source.ndim - 1, head_ids)
            if key.endswith("proj.weight"):
                return _index(_index(source, 0, channels), 1, head_channels)
            if key.endswith("proj.bias"):
                return _index(source, 0, channels)

        mlp_prefix = key.split(".fc1.")[0] if ".fc1." in key else None
        if mlp_prefix is None and ".fc2." in key:
            mlp_prefix = key.split(".fc2.")[0]
        if mlp_prefix is not None:
            hidden_ids = hidden[mlp_prefix]
            if key.endswith("fc1.weight"):
                return _index(_index(source, 0, hidden_ids), 1, channels)
            if key.endswith("fc1.bias"):
                return _index(source, 0, hidden_ids)
            if key.endswith("fc2.weight"):
                return _index(_index(source, 0, channels), 1, hidden_ids)
            if key.endswith("fc2.bias"):
                return _index(source, 0, channels)

        if source.ndim == 1 and source.numel() == width:
            return _index(source, 0, channels)

    if key.startswith("downsample."):
        shallow4 = _grouped(residual["shallow"], 4, source_width)
        if key.endswith("linear.weight"):
            return _index(_index(source, 0, residual["deep"]), 1, shallow4)
        if key.endswith(("norm.weight", "norm.bias")):
            return _index(source, 0, shallow4)

    if key.startswith("upsample."):
        shallow = residual["shallow"]
        if key.endswith("linear1.weight"):
            shallow4 = _grouped(shallow, 4, source_width)
            return _index(_index(source, 0, shallow4), 1, residual["deep"])
        if key.endswith("linear2.weight"):
            return _index(_index(source, 0, shallow), 1, shallow)
        if key.endswith(("norm.weight", "norm.bias")):
            return _index(source, 0, shallow)

    if key.startswith(("patchrecovery2d.", "patchrecovery3d.")):
        if key.endswith("proj.weight"):
            skip_channels = _grouped(residual["shallow"], 2, source_width)
            return _index(source, 0, skip_channels)

    if tuple(source.shape) == tuple(target.shape):
        return source
    raise ValueError(
        f"No pruning rule for {key}: {tuple(source.shape)} -> {tuple(target.shape)}"
    )


def _resize_patch_weight(value, target_shape, preserve_embedding_scale):
    """Resize only the horizontal patch kernel dimensions deterministically."""

    if value.ndim != len(target_shape) or value.ndim < 4:
        raise ValueError(
            f"Patch weight rank mismatch: {tuple(value.shape)} -> {tuple(target_shape)}"
        )
    if tuple(value.shape[:-2]) != tuple(target_shape[:-2]):
        raise ValueError(
            "Patch weight non-spatial dimensions do not match after pruning: "
            f"{tuple(value.shape)} -> {tuple(target_shape)}"
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
        resized *= (source_height * source_width) / (target_height * target_width)
    return resized.to(value.dtype)


def _interpolate_earth_bias(value, target_shape):
    """Resize the latitude-window axis while preserving longitude groups."""

    if value.ndim != 3 or len(target_shape) != 3:
        raise ValueError("Earth-position bias must be rank three")
    if value.shape[0] != target_shape[0] or value.shape[2] != target_shape[2]:
        raise ValueError(
            f"Earth-position bias dimensions mismatch: "
            f"{tuple(value.shape)} -> {tuple(target_shape)}"
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


def get_depth_block_map(source_depths, target_depths):
    """Select type-compatible entrance/exit blocks for depth pruning."""

    if len(source_depths) != len(target_depths):
        raise ValueError("source_depths and target_depths must have equal length")
    block_map = []
    for source_depth, target_depth in zip(source_depths, target_depths):
        source_depth = int(source_depth)
        target_depth = int(target_depth)
        if source_depth <= 0 or target_depth <= 0 or target_depth > source_depth:
            raise ValueError(
                f"Invalid depth reduction: {source_depth} -> {target_depth}"
            )
        if target_depth == source_depth:
            selected = list(range(source_depth))
        elif target_depth == 1:
            # Target block zero is always the non-shifted variant.
            selected = [0]
        else:
            step = (source_depth - 1) / (target_depth - 1)
            ideals = [index * step for index in range(target_depth)]
            candidates = [
                list(candidate)
                for candidate in itertools.combinations(range(source_depth), target_depth)
                if all(
                    (target % 2) == (source % 2)
                    for target, source in enumerate(candidate)
                )
            ]
            if not candidates:
                raise ValueError(
                    "No shift-compatible depth selection exists for "
                    f"{source_depth} -> {target_depth}"
                )
            selected = min(
                candidates,
                key=lambda candidate: sum(
                    (source - ideal) ** 2
                    for source, ideal in zip(candidate, ideals)
                ),
            )
        block_map.append(selected)
    return block_map


def get_source_key_for_target(key, source_depths, target_depths):
    if ".blocks." not in key:
        return key
    parts = key.split(".")
    try:
        blocks_idx = parts.index("blocks")
    except ValueError:
        return key
        
    t = int(parts[blocks_idx + 1])
    
    # Look back to find the layer stage name (e.g. layer1, layer2, etc.)
    layer_name = None
    for idx in range(blocks_idx - 1, -1, -1):
        token = parts[idx]
        if token.startswith("layer") and token[-1].isdigit():
            layer_name = token
            break
            
    if layer_name is None:
        raise ValueError(f"Could not parse layer stage name from key: {key}")
    
    # layer name to index: "layer1" -> 0, "layer2" -> 1, etc.
    layer_idx = int(layer_name[-1]) - 1
    block_map = get_depth_block_map(source_depths, target_depths)
    s = block_map[layer_idx][t]
        
    parts[blocks_idx + 1] = str(s)
    return ".".join(parts)


def _state_depths(state):
    depths = []
    for layer_index in range(1, 5):
        prefix = f"layer{layer_index}."
        block_indices = {
            int(key.split(".blocks.", 1)[1].split(".", 1)[0])
            for key in state
            if key.startswith(prefix) and ".blocks." in key
        }
        if not block_indices:
            raise ValueError(f"Source checkpoint has no blocks for layer{layer_index}")
        expected = set(range(max(block_indices) + 1))
        if block_indices != expected:
            raise ValueError(
                f"Source layer{layer_index} block indices are not contiguous: "
                f"{sorted(block_indices)}"
            )
        depths.append(len(block_indices))
    return depths


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_s96_source(checkpoint, source_state, cfg):
    if any(
        str(key).endswith("_scale")
        or (isinstance(value, torch.Tensor) and value.dtype == torch.int8)
        for key, value in source_state.items()
    ):
        raise ValueError("S96 initialization source must be unquantized FP16 weights")

    profile = checkpoint.get("model_profile")
    if profile is not None:
        actual = {
            "patch_size": [int(value) for value in profile.get("patch_size", [])],
            "embed_dim": int(profile.get("embed_dim", -1)),
            "num_heads": [int(value) for value in profile.get("num_heads", [])],
            "depth_blocks": [
                int(value)
                for value in profile.get("depth_blocks", S96_SOURCE_DEPTHS)
            ],
        }
        expected = {
            "patch_size": S96_SOURCE_PATCH_SIZE,
            "embed_dim": S96_SOURCE_EMBED_DIM,
            "num_heads": S96_SOURCE_NUM_HEADS,
            "depth_blocks": S96_SOURCE_DEPTHS,
        }
        if actual != expected:
            raise ValueError(
                f"S96 source model_profile mismatch: actual={actual}, expected={expected}"
            )

    actual_depths = _state_depths(source_state)
    if actual_depths != S96_SOURCE_DEPTHS:
        raise ValueError(
            f"S96 source depth mismatch: {actual_depths} != {S96_SOURCE_DEPTHS}"
        )

    expected_source = build_pangu_model(
        img_size=cfg.img_size if hasattr(cfg, "img_size") else (721, 1440),
        patch_size=S96_SOURCE_PATCH_SIZE,
        embed_dim=S96_SOURCE_EMBED_DIM,
        num_heads=S96_SOURCE_NUM_HEADS,
        window_size=cfg.window_size,
    ).state_dict()
    missing = sorted(set(expected_source) - set(source_state))
    mismatched = sorted(
        key
        for key, expected_tensor in expected_source.items()
        if key in source_state
        and tuple(source_state[key].shape) != tuple(expected_tensor.shape)
    )
    if missing or mismatched:
        raise ValueError(
            "S96 source structure mismatch: "
            f"missing={missing[:5]}, shape_mismatch={mismatched[:5]}"
        )


def _exact_s96_depth_state(source_state, target_state):
    block_map = get_depth_block_map(S96_SOURCE_DEPTHS, S96_TARGET_DEPTHS)
    expected_map = [[0], [0, 5], [0, 5], [0]]
    if block_map != expected_map:
        raise AssertionError(f"Unexpected S96 depth map: {block_map}")

    migrated = OrderedDict()
    source_keys = {}
    for target_key, target_tensor in target_state.items():
        source_key = get_source_key_for_target(
            target_key, S96_SOURCE_DEPTHS, S96_TARGET_DEPTHS
        )
        if source_key not in source_state:
            raise KeyError(f"S96 source is missing {source_key} for {target_key}")
        source_tensor = source_state[source_key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"S96 exact-load shape mismatch for {target_key}: "
                f"source {source_key} {tuple(source_tensor.shape)} != "
                f"target {tuple(target_tensor.shape)}"
            )
        migrated[target_key] = source_tensor
        source_keys[target_key] = source_key
    if len(migrated) != len(target_state):
        raise AssertionError("S96 exact initialization did not cover every target tensor")
    return migrated, source_keys, block_map


def prune_checkpoint(args):
    cfg = YParams(args.config, "model")
    source_path = args.source
    if not os.path.exists(source_path) and args.strict_exact_depth:
        raise FileNotFoundError(
            f"Strict S96 initialization source not found: {source_path}"
        )
    if not os.path.exists(source_path):
        backup_path = os.path.join(cfg.official_checkpoint_dir, "model_bak.pth")
        if not os.path.exists(backup_path):
            raise FileNotFoundError(
                f"Neither pruning source {source_path} nor backup {backup_path} exists"
            )
        source_path = backup_path
        print(f"FP16 source not found; use official FP32 backup: {source_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = checkpoint["model_state_dict"]

    # Infer normal pruning sources from metadata. Strict S96 is fixed and then
    # independently audited against a full expected state dict below.
    profile = checkpoint.get("model_profile", None)
    if args.strict_exact_depth:
        source_width = S96_SOURCE_EMBED_DIM
        source_heads = tuple(S96_SOURCE_NUM_HEADS)
        patch_size = list(S96_SOURCE_PATCH_SIZE)
    elif profile is not None:
        source_width = int(profile.get("embed_dim", cfg.embed_dim))
        source_heads = tuple(int(value) for value in profile.get("num_heads", cfg.num_heads))
        patch_size = list(profile.get("patch_size", cfg.patch_size))
        print(f"Inferred source profile from checkpoint: patch_size={patch_size}, embed_dim={source_width}, num_heads={source_heads}")
    else:
        source_width = int(cfg.embed_dim)
        source_heads = tuple(int(value) for value in cfg.num_heads)
        patch_size = cfg.patch_size

    head_dim = source_width // source_heads[0]
    if source_width % source_heads[0] or source_width * 2 % source_heads[1]:
        raise ValueError("Source embed dimensions must be divisible by attention heads")

    source_depths = [2, 6, 6, 2]
    if args.target_profile:
        profiles = getattr(cfg, "student_profiles", {})
        if args.target_profile not in profiles:
            raise ValueError(f"Unknown target profile: {args.target_profile}")
        target_profile_cfg = profiles[args.target_profile]
        target_patch_size = [int(v) for v in target_profile_cfg.patch_size]
        target_width = int(target_profile_cfg.embed_dim)
        target_heads = tuple(int(h) for h in target_profile_cfg.num_heads)
        target_depth_blocks = getattr(target_profile_cfg, "depth_blocks", None)
    else:
        target_patch_size = list(patch_size)
        target_width = args.target_embed_dim
        target_heads = (
            target_width // head_dim,
            target_width * 2 // head_dim,
            target_width * 2 // head_dim,
            target_width // head_dim,
        )
        target_depth_blocks = None

    if target_depth_blocks is not None:
        target_depth_blocks = [int(v) for v in target_depth_blocks]
    target_depths = target_depth_blocks if target_depth_blocks is not None else source_depths

    if target_width > source_width or target_width % head_dim:
        raise ValueError(
            f"target embed dim must be smaller than or equal to {source_width} and divisible by {head_dim}"
        )

    img_size = cfg.img_size if hasattr(cfg, "img_size") else (721, 1440)

    source_model = build_pangu_model(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=source_width,
        num_heads=source_heads,
        window_size=cfg.window_size,
    )
    source_parameters = sum(parameter.numel() for parameter in source_model.parameters())
    del source_model

    target_model = build_pangu_model(
        img_size=img_size,
        patch_size=target_patch_size,
        embed_dim=target_width,
        num_heads=target_heads,
        window_size=cfg.window_size,
        depth_blocks=target_depth_blocks,
    )
    target_state = target_model.state_dict()
    if args.strict_exact_depth:
        if args.target_profile != S96_PROFILE:
            raise ValueError(
                f"--strict-exact-depth only supports --target-profile {S96_PROFILE}"
            )
        if (
            patch_size != S96_SOURCE_PATCH_SIZE
            or source_width != S96_SOURCE_EMBED_DIM
            or list(source_heads) != S96_SOURCE_NUM_HEADS
            or target_depths != S96_TARGET_DEPTHS
        ):
            raise ValueError("Strict S96 initialization received an incompatible profile")
        _validate_s96_source(checkpoint, source_state, cfg)
        migrated, source_keys, block_map = _exact_s96_depth_state(
            source_state, target_state
        )
        residual = {
            "shallow": torch.arange(source_width),
            "deep": torch.arange(source_width * 2),
        }
    else:
        residual = _residual_indices(source_state, source_width, target_width)
        attention_heads = _attention_heads(
            source_state, source_width, target_width, source_heads
        )
        mlp_hidden = _mlp_indices(source_state, source_width, target_width)

        migrated = OrderedDict()
        resized_patch_keys = []
        interpolated_bias_keys = []
        regenerated_buffer_keys = []
        for key, target_tensor in target_state.items():
            source_key = get_source_key_for_target(key, source_depths, target_depths)
            if source_key not in source_state:
                raise KeyError(f"Source checkpoint is missing {source_key} for target {key}")
            if key.endswith(("attn_mask", "earth_position_index")):
                tensor = target_tensor
                regenerated_buffer_keys.append(key)
            else:
                tensor = _migrate_tensor(
                    source_key,
                    source_state[source_key],
                    target_tensor,
                    residual,
                    attention_heads,
                    mlp_hidden,
                    source_width,
                    source_heads,
                )
                if key.endswith(".proj.weight") and key.startswith(
                    (
                        "patchembed2d.",
                        "patchembed3d.",
                        "patchrecovery2d.",
                        "patchrecovery3d.",
                    )
                ) and tuple(tensor.shape) != tuple(target_tensor.shape):
                    tensor = _resize_patch_weight(
                        tensor,
                        target_tensor.shape,
                        preserve_embedding_scale=key.startswith("patchembed"),
                    )
                    resized_patch_keys.append(key)
                elif key.endswith("earth_position_bias_table") and (
                    tuple(tensor.shape) != tuple(target_tensor.shape)
                ):
                    tensor = _interpolate_earth_bias(tensor, target_tensor.shape)
                    interpolated_bias_keys.append(key)
            if tuple(tensor.shape) != tuple(target_tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {key}: got {tuple(tensor.shape)}, "
                    f"expected {tuple(target_tensor.shape)}"
                )
            migrated[key] = tensor

    target_model.load_state_dict(migrated, strict=True)
    if args.dtype == "fp16":
        target_model.half()
    output_state = target_model.state_dict()
    metadata = {
        "method": (
            "pgw_lite_width96_exact_depth_selection"
            if args.strict_exact_depth
            else "structured_head_width_pruning"
        ),
        "source_embed_dim": source_width,
        "target_embed_dim": target_width,
        "source_num_heads": source_heads,
        "target_num_heads": target_heads,
        "source_patch_size": list(patch_size),
        "target_patch_size": target_patch_size,
        "shallow_channels": residual["shallow"].tolist(),
        "deep_channels": residual["deep"].tolist(),
    }
    if target_depth_blocks is not None:
        metadata["target_depth_blocks"] = target_depth_blocks
    if not args.strict_exact_depth:
        metadata.update(
            {
                "resized_patch_keys": resized_patch_keys,
                "interpolated_bias_keys": interpolated_bias_keys,
                "regenerated_buffer_keys": regenerated_buffer_keys,
            }
        )
    if args.strict_exact_depth:
        metadata.update(
            {
                "source_file": os.path.basename(source_path),
                "source_sha256": _sha256(source_path),
                "source_depth_blocks": S96_SOURCE_DEPTHS,
                "block_map": block_map,
                "loaded_tensors": len(migrated),
                "target_tensors": len(target_state),
                "interpolated_tensors": 0,
                "resized_tensors": 0,
                "source_key_map": source_keys,
            }
        )
        
    model_profile = {
        "name": args.target_profile if args.target_profile else (f"pgw_lite_pruned_{target_width}" if target_patch_size == [2, 8, 8] else f"student_{target_width}"),
        "patch_size": target_patch_size,
        "embed_dim": target_width,
        "num_heads": list(target_heads),
        "depth_blocks": list(target_depths),
        "window_size": [int(value) for value in cfg.window_size],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": output_state,
            "pruning": metadata,
            "model_profile": model_profile,
        },
        args.output,
    )

    target_parameters = sum(parameter.numel() for parameter in target_model.parameters())
    source_size = os.path.getsize(source_path)
    target_size = os.path.getsize(args.output)
    print(f"Source parameters: {source_parameters:,}")
    print(f"Pruned parameters: {target_parameters:,}")
    print(f"Parameter reduction: {(1 - target_parameters / source_parameters) * 100:.2f}%")
    print(f"Source checkpoint size: {source_size / 1024**2:.1f} MiB")
    print(f"Pruned checkpoint size: {target_size / 1024**2:.1f} MiB")
    print(f"Checkpoint size reduction: {(1 - target_size / source_size) * 100:.2f}%")
    print(f"Saved pruned checkpoint: {args.output}")
    print(f"Target patch_size: {target_patch_size}")
    print(f"Target embed_dim: {target_width}, num_heads: {target_heads}")
    if target_depth_blocks is not None:
        print(f"Target depth blocks: {target_depth_blocks}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--source", default="data/checkpoints/model_fp16.pth")
    parser.add_argument("--output", default="data/checkpoints/model_pruned_fp16.pth")
    parser.add_argument("--target-embed-dim", type=int, default=160)
    parser.add_argument("--target-profile", default=None)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--strict-exact-depth",
        action="store_true",
        help="Require exact Width-96 S96 depth selection without fallback or resizing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prune_checkpoint(parse_args())
