#!/usr/bin/env python3
"""Remove redundant OneFuser alias tensors without changing loaded weights.

OneFuser registers the same module first as ``fuser`` and then as ``Fuser``.
The current mixed checkpoint stores both namespaces. In inference, the source
state is copied in order, so the later ``Fuser`` tensor is the final value in
the shared runtime storage. This tool keeps that final writer and removes the
earlier ``fuser`` tensor plus any orphaned quantization scale.
"""

import argparse
import os
from collections import OrderedDict
from collections.abc import Mapping

import torch


LOWER_ALIAS = ".fuser."
UPPER_ALIAS = ".Fuser."


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint must contain model_state_dict")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("model_state_dict must contain tensors only")
    return state


def plan_alias_compaction(state):
    keys = list(state)
    positions = {key: index for index, key in enumerate(keys)}
    pairs = []
    drop_keys = set()

    for lower_key in keys:
        if LOWER_ALIAS not in lower_key or lower_key.endswith("_scale"):
            continue
        upper_key = lower_key.replace(LOWER_ALIAS, UPPER_ALIAS, 1)
        if upper_key not in state:
            continue
        if positions[upper_key] <= positions[lower_key]:
            raise ValueError(f"Final alias writer is not {upper_key}")
        if tuple(state[lower_key].shape) != tuple(state[upper_key].shape):
            raise ValueError(f"Alias shape mismatch: {lower_key} vs {upper_key}")
        pairs.append((lower_key, upper_key))
        drop_keys.add(lower_key)
        scale_key = lower_key + "_scale"
        if scale_key in state:
            drop_keys.add(scale_key)

    if not pairs:
        raise ValueError("No ordered fuser/Fuser alias pairs found")
    return pairs, drop_keys


def validate_runtime_aliases(model_state, pairs):
    for lower_key, upper_key in pairs:
        if lower_key not in model_state or upper_key not in model_state:
            raise ValueError(f"Runtime model is missing alias pair: {lower_key}")
        lower = model_state[lower_key]
        upper = model_state[upper_key]
        if lower.untyped_storage()._cdata != upper.untyped_storage()._cdata:
            raise ValueError(f"Runtime aliases do not share storage: {lower_key}")


def compact_state_dict(state, drop_keys):
    compacted = OrderedDict(
        (key, value) for key, value in state.items() if key not in drop_keys
    )
    if hasattr(state, "_metadata"):
        compacted._metadata = state._metadata
    return compacted


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def build_runtime_model(checkpoint, state):
    from pangu_profile_model import build_pangu_model

    profile = checkpoint.get("model_profile")
    if not isinstance(profile, dict):
        raise TypeError("Checkpoint must contain model_profile metadata")
    required = ("patch_size", "embed_dim", "num_heads", "window_size")
    if any(name not in profile for name in required):
        raise ValueError("model_profile is incomplete")

    keys = tuple(state)
    return build_pangu_model(
        img_size=[721, 1440],
        patch_size=[int(value) for value in profile["patch_size"]],
        embed_dim=int(profile["embed_dim"]),
        num_heads=[int(value) for value in profile["num_heads"]],
        window_size=[int(value) for value in profile["window_size"]],
        depth_blocks=(
            [int(value) for value in profile["depth_blocks"]]
            if profile.get("depth_blocks") is not None
            else None
        ),
        use_swiglu=any(".mlp.w1." in key for key in keys),
        use_rmsnorm=(
            any(".norm1.weight" in key for key in keys)
            and not any(".norm1.bias" in key for key in keys)
        ),
        use_gqa=any(".q_proj." in key for key in keys),
        share_deep_blocks=profile.get("share_deep_blocks"),
    )


def verify_saved_candidate(expected_state, output_path):
    candidate = torch.load(output_path, map_location="cpu", weights_only=False)
    actual_state = extract_state_dict(candidate)
    if list(actual_state) != list(expected_state):
        raise ValueError("Saved candidate key order differs from compacted state")
    for key, expected in expected_state.items():
        actual = actual_state[key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"Saved tensor metadata mismatch: {key}")
        if not torch.equal(actual, expected):
            raise ValueError(f"Saved tensor value mismatch: {key}")


def print_plan(source_path, state, pairs, drop_keys):
    logical_bytes = sum(tensor_bytes(value) for value in state.values())
    dropped_bytes = sum(tensor_bytes(state[key]) for key in drop_keys)
    scale_count = sum(key.endswith("_scale") for key in drop_keys)
    print(f"Source: {os.path.abspath(source_path)}")
    print(f"File size: {os.path.getsize(source_path) / 1024**2:.2f} MiB")
    print(f"Alias pairs: {len(pairs)}")
    print(f"Dropped tensors: {len(drop_keys)} ({scale_count} scales)")
    print(f"Dropped logical bytes: {dropped_bytes / 1024**2:.2f} MiB")
    print(f"Remaining logical bytes: {(logical_bytes - dropped_bytes) / 1024**2:.2f} MiB")


def compact_checkpoint(source_path, output_path, audit_only=False):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint)
    pairs, drop_keys = plan_alias_compaction(state)

    model = build_runtime_model(checkpoint, state)
    validate_runtime_aliases(model.state_dict(), pairs)
    del model

    print_plan(source_path, state, pairs, drop_keys)
    if audit_only:
        return None

    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    compacted_state = compact_state_dict(state, drop_keys)
    output_checkpoint = dict(checkpoint)
    output_checkpoint["model_state_dict"] = compacted_state
    output_checkpoint["alias_compaction"] = {
        "method": "keep-later-OneFuser-alias",
        "dropped_tensor_count": len(drop_keys),
        "alias_pair_count": len(pairs),
    }

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    try:
        torch.save(output_checkpoint, temporary_path)
        verify_saved_candidate(compacted_state, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print(f"Verified output: {output_path}")
    print(f"Output size: {os.path.getsize(output_path) / 1024**2:.2f} MiB")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/checkpoints/model_fp16.pth")
    parser.add_argument(
        "--output", default="data/checkpoints/model_fp16_alias_compact.pth"
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.source):
        raise FileNotFoundError(args.source)
    if os.path.abspath(args.source) == os.path.abspath(args.output):
        raise ValueError("Source and output paths must differ")
    compact_checkpoint(args.source, args.output, audit_only=args.audit_only)


if __name__ == "__main__":
    main()
