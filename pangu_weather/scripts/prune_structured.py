"""Create a width-pruned Pangu-Weather checkpoint from official weights.

The pruning keeps the original per-head width (32) and removes complete
attention heads together with globally consistent residual channels. This
preserves every residual, sampling, and skip-connection shape dependency.

Run from ``pangu_weather`` after activating the competition environment:

    python scripts/prune_structured.py
"""

import argparse
import os
from collections import OrderedDict

import torch

from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams


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


def prune_checkpoint(args):
    cfg = YParams(args.config, "model")
    source_width = int(cfg.embed_dim)
    source_heads = tuple(int(value) for value in cfg.num_heads)
    head_dim = source_width // source_heads[0]
    if source_width % source_heads[0] or source_width * 2 % source_heads[1]:
        raise ValueError("Source embed dimensions must be divisible by attention heads")
    if args.target_embed_dim >= source_width or args.target_embed_dim % head_dim:
        raise ValueError(
            f"target embed dim must be smaller than {source_width} and divisible by {head_dim}"
        )

    target_heads = (
        args.target_embed_dim // head_dim,
        args.target_embed_dim * 2 // head_dim,
        args.target_embed_dim * 2 // head_dim,
        args.target_embed_dim // head_dim,
    )
    source_path = args.source
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
    img_size = cfg.img_size if hasattr(cfg, "img_size") else (721, 1440)

    source_model = Pangu(
        img_size=img_size,
        patch_size=cfg.patch_size,
        embed_dim=source_width,
        num_heads=source_heads,
        window_size=cfg.window_size,
    )
    source_parameters = sum(parameter.numel() for parameter in source_model.parameters())
    del source_model

    target_model = Pangu(
        img_size=img_size,
        patch_size=cfg.patch_size,
        embed_dim=args.target_embed_dim,
        num_heads=target_heads,
        window_size=cfg.window_size,
    )
    target_state = target_model.state_dict()
    residual = _residual_indices(source_state, source_width, args.target_embed_dim)
    attention_heads = _attention_heads(
        source_state, source_width, args.target_embed_dim, source_heads
    )
    mlp_hidden = _mlp_indices(source_state, source_width, args.target_embed_dim)

    migrated = OrderedDict()
    for key, target_tensor in target_state.items():
        if key not in source_state:
            raise KeyError(f"Source checkpoint is missing {key}")
        tensor = _migrate_tensor(
            key,
            source_state[key],
            target_tensor,
            residual,
            attention_heads,
            mlp_hidden,
            source_width,
            source_heads,
        )
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
        "method": "structured_head_width_pruning",
        "source_embed_dim": source_width,
        "target_embed_dim": args.target_embed_dim,
        "source_num_heads": source_heads,
        "target_num_heads": target_heads,
        "shallow_channels": residual["shallow"].tolist(),
        "deep_channels": residual["deep"].tolist(),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"model_state_dict": output_state, "pruning": metadata}, args.output)

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
    print(f"Target embed_dim: {args.target_embed_dim}, num_heads: {target_heads}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--source", default="data/checkpoints/model_fp16.pth")
    parser.add_argument("--output", default="data/checkpoints/model_pruned_fp16.pth")
    parser.add_argument("--target-embed-dim", type=int, default=160)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


if __name__ == "__main__":
    prune_checkpoint(parse_args())
