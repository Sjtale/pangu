#!/usr/bin/env python3
"""Pack existing per-output INT8 Linear weights as groupwise INT4."""

import argparse
import os
from collections import OrderedDict

import torch


def pack_groupwise_int4(qweight, source_scale, group_size=64):
    if qweight.ndim != 2 or qweight.dtype != torch.int8:
        raise TypeError("Expected a 2D int8 weight")
    rows, columns = qweight.shape
    source_scale = source_scale.to(torch.float32)
    if source_scale.ndim == 1:
        source_scale = source_scale.view(-1, 1)
    if source_scale.shape != (rows, 1):
        raise ValueError(f"Unsupported source scale shape: {tuple(source_scale.shape)}")

    padded_columns = ((columns + group_size - 1) // group_size) * group_size
    weight = qweight.to(torch.float32) * source_scale
    if padded_columns != columns:
        weight = torch.nn.functional.pad(weight, (0, padded_columns - columns))
    grouped = weight.reshape(rows, -1, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1.0e-12) / 7.0
    quantized = torch.round(grouped / scales.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
    encoded = quantized.reshape(rows, padded_columns).to(torch.int16) & 0x0F
    packed = (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).to(torch.uint8)
    return packed, scales.to(torch.float16)


def unpack_groupwise_int4(packed, scales, shape, group_size):
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    quantized = torch.stack((low, high), dim=-1).reshape(shape[0], -1)
    quantized = torch.where(quantized >= 8, quantized - 16, quantized)
    restored = quantized.to(torch.float32).reshape(shape[0], scales.shape[1], group_size)
    return (restored * scales.to(torch.float32).unsqueeze(-1)).reshape(shape[0], -1)[:, : shape[1]]


def pack_checkpoint(source, output, group_size=64):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain model_state_dict")
    if os.path.exists(output):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    packed_state = OrderedDict()
    packed_count = 0
    for key, value in state.items():
        if key.endswith("_scale"):
            continue
        scale_key = key + "_scale"
        if value.dtype == torch.int8 and value.ndim == 2 and scale_key in state:
            packed, scales = pack_groupwise_int4(value, state[scale_key], group_size)
            packed_state[key] = packed
            packed_state[key + ".int4_scale"] = scales
            packed_state[key + ".int4_shape"] = torch.tensor(value.shape, dtype=torch.int32)
            packed_state[key + ".int4_group_size"] = torch.tensor(group_size, dtype=torch.int16)
            packed_count += 1
        else:
            packed_state[key] = value

    if packed_count == 0:
        raise ValueError("No per-output INT8 Linear weights found")
    output_checkpoint = dict(checkpoint)
    output_checkpoint["model_state_dict"] = packed_state
    output_checkpoint["int4_storage"] = {
        "scheme": "symmetric-groupwise-int4",
        "group_size": group_size,
        "packed_weight_count": packed_count,
        "runtime": "restore-fp16-before-timing",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(output_checkpoint, output)
    source_size = os.path.getsize(source)
    output_size = os.path.getsize(output)
    print(f"Packed weights: {packed_count}")
    print(f"Size: {source_size / 2**20:.2f} -> {output_size / 2**20:.2f} MiB")
    print(f"Reduction: {(1.0 - output_size / source_size) * 100.0:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/checkpoints/model_fp16_alias_compact.pth")
    parser.add_argument("--output", default="data/checkpoints/model_int4_group64.pth")
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    args = parser.parse_args()
    pack_checkpoint(args.source, args.output, args.group_size)


if __name__ == "__main__":
    main()
