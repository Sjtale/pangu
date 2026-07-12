"""Hybrid-transfer the official 3D Pangu checkpoint into the 2D KD student."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pangu_lite_2d import PanguLite2DAttentionPosEmbed


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(checkpoint):
    state = checkpoint.get("model_state_dict", checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}


def _find(state, prefix, suffix):
    matches = [key for key in state if key.startswith(prefix) and key.endswith(suffix)]
    if not matches:
        raise KeyError(f"missing teacher tensor {prefix}*{suffix}")
    return state[min(matches, key=len)]


def _copy_overlap(target, source):
    if target.ndim != source.ndim:
        raise ValueError(f"rank mismatch {target.shape} <- {source.shape}")
    slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, source.shape))
    target[slices].copy_(source[slices].to(target.dtype))
    copied = 1
    for item in slices:
        copied *= item.stop
    return copied


def _resize_kernel(weight, size):
    old_size = weight.shape[-2:]
    resized = F.interpolate(
        weight.reshape(-1, 1, *old_size).float(),
        size=size,
        mode="bicubic",
        align_corners=False,
    ).reshape(*weight.shape[:-2], *size)
    return resized * ((old_size[0] * old_size[1]) / (size[0] * size[1]))


def hybrid_transfer(student, teacher):
    report = {"categories": {}, "random_only": []}
    state = student.state_dict()
    with torch.no_grad():
        surface_embed = _find(teacher, "patchembed2d.", ".proj.weight").squeeze(2)
        upper_embed = _find(teacher, "patchembed3d.", ".proj.weight")
        target = state["patchembed.weight"]
        copied = _copy_overlap(target[:, :7], _resize_kernel(surface_embed, (8, 8)))
        for variable in range(5):
            for level in range(13):
                source = upper_embed[:, variable, level % upper_embed.shape[2]]
                copied += _copy_overlap(
                    target[:, 7 + variable * 13 + level], _resize_kernel(source, (8, 8))
                )
        surface_bias = _find(teacher, "patchembed2d.", ".proj.bias")
        copied += _copy_overlap(state["patchembed.bias"], surface_bias)
        report["categories"]["patch_embedding"] = copied

        surface_recovery = _find(teacher, "patchrecovery2d.", ".proj.weight").squeeze(2)
        upper_recovery = _find(teacher, "patchrecovery3d.", ".proj.weight")
        recovery = state["patchrecovery.weight"]
        copied = _copy_overlap(recovery[:, :4], _resize_kernel(surface_recovery, (8, 8)))
        for variable in range(5):
            for level in range(13):
                source = upper_recovery[:, variable, level % upper_recovery.shape[2]]
                copied += _copy_overlap(
                    recovery[:, 4 + variable * 13 + level], _resize_kernel(source, (8, 8))
                )
        surface_recovery_bias = _find(teacher, "patchrecovery2d.", ".proj.bias")
        upper_recovery_bias = _find(teacher, "patchrecovery3d.", ".proj.bias")
        state["patchrecovery.bias"][:4].copy_(surface_recovery_bias[:4])
        for variable in range(5):
            state["patchrecovery.bias"][4 + variable * 13:4 + (variable + 1) * 13].fill_(upper_recovery_bias[variable])
        report["categories"]["patch_recovery"] = copied + 69

        category_counts = {"mlp": 0, "downsample": 0, "upsample": 0, "normalization": 0}
        for key, target_tensor in state.items():
            if ".attn." in key or key == "absolute_pos_embed" or key.startswith(("patchembed", "patchrecovery")):
                continue
            if ".mlp." in key:
                category = "mlp"
                stage, remainder = key.split(".blocks.", 1)
                block, leaf = remainder.split(".", 1)
                source_prefix = f"{stage}."
                teacher_suffix = f"blocks.{block}.transformer.{leaf}"
            elif key.startswith("downsample."):
                category = "downsample"
                source_prefix = "downsample."
                teacher_suffix = key[len("downsample."):]
            elif key.startswith("upsample."):
                category = "upsample"
                source_prefix = "upsample."
                teacher_suffix = key[len("upsample."):]
            elif ".norm" in key:
                category = "normalization"
                stage, remainder = key.split(".blocks.", 1)
                block, leaf = remainder.split(".", 1)
                source_prefix = f"{stage}."
                teacher_suffix = f"blocks.{block}.transformer.{leaf}"
            else:
                continue
            candidates = [
                value
                for source_key, value in teacher.items()
                if source_key.startswith(source_prefix) and source_key.endswith(teacher_suffix)
            ]
            candidates = [value for value in candidates if value.ndim == target_tensor.ndim]
            if candidates:
                category_counts[category] += _copy_overlap(target_tensor, candidates[0])
        report["categories"].update(category_counts)

        for key in state:
            if ".attn." in key or key == "absolute_pos_embed":
                report["random_only"].append(key)
        student.load_state_dict(state, strict=True)

    required = ("patch_embedding", "patch_recovery", "mlp", "downsample", "upsample")
    missing = [name for name in required if report["categories"].get(name, 0) <= 0]
    if missing:
        raise RuntimeError(f"hybrid transfer missed required categories: {missing}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
    student = PanguLite2DAttentionPosEmbed()
    report = hybrid_transfer(student, _state(checkpoint))
    report["teacher_sha256"] = _sha256(args.teacher)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": student.state_dict(),
        "model_profile": {
            "name": "pangu_lite_2d_pos288",
            "architecture": "PanguLite2DAttentionPosEmbed",
            "patch_size": [8, 8],
            "embed_dim": 288,
            "num_heads": [6, 12, 12, 6],
        },
        "hybrid_transfer": report,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    report["output_sha256"] = _sha256(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
