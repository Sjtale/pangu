#!/usr/bin/env python3
"""Force PyTorch Flash SDPA on representative Pangu attention tensors."""

import argparse
import inspect
import json
import time

import torch
import torch.nn.functional as F


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(call, device, warmup, repeat):
    for _ in range(warmup):
        call()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = None
    for _ in range(repeat):
        output = call()
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeat
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None
    )
    return output, elapsed_ms, peak_mb


def _make_inputs(args, device):
    shape = (
        args.width_windows,
        args.pressure_height_windows,
        args.heads,
        args.tokens,
        args.head_dim,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    q = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    k = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
    v = torch.randn(shape, device=device, dtype=torch.float16, generator=generator)

    bias = torch.randn(
        1,
        args.pressure_height_windows,
        args.heads,
        args.tokens,
        args.tokens,
        device=device,
        dtype=torch.float16,
        generator=generator,
    ) * 0.02
    shifted_mask = torch.zeros(
        args.width_windows,
        args.pressure_height_windows,
        1,
        args.tokens,
        args.tokens,
        device=device,
        dtype=torch.float16,
    )
    split = args.tokens // 2
    shifted_mask[..., :split, split:] = -100.0
    shifted_mask[..., split:, :split] = -100.0
    combined_mask = (bias + shifted_mask).reshape(
        args.width_windows * args.pressure_height_windows,
        args.heads,
        args.tokens,
        args.tokens,
    )

    flat_shape = (
        args.width_windows * args.pressure_height_windows,
        args.heads,
        args.tokens,
        args.head_dim,
    )
    return q.reshape(flat_shape), k.reshape(flat_shape), v.reshape(flat_shape), combined_mask


def _eager(q, k, v, mask, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _forced_flash(q, k, v, mask, scale):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )


def _probe_case(name, q, k, v, mask, scale, args, device):
    reference, eager_ms, eager_peak = _measure(
        lambda: _eager(q, k, v, mask, scale), device, args.warmup, args.repeat
    )
    try:
        candidate, flash_ms, flash_peak = _measure(
            lambda: _forced_flash(q, k, v, mask, scale),
            device,
            args.warmup,
            args.repeat,
        )
        difference = (reference.float() - candidate.float()).abs()
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        compatible = bool(
            torch.allclose(reference.float(), candidate.float(), atol=5e-3, rtol=1e-2)
        )
        return {
            "name": name,
            "forced_flash_supported": True,
            "numerically_compatible": compatible,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "eager_latency_ms": eager_ms,
            "flash_latency_ms": flash_ms,
            "speedup": eager_ms / flash_ms,
            "eager_peak_allocated_mb": eager_peak,
            "flash_peak_allocated_mb": flash_peak,
        }
    except Exception as error:
        return {
            "name": name,
            "forced_flash_supported": False,
            "numerically_compatible": False,
            "error": f"{type(error).__name__}: {error}",
            "eager_latency_ms": eager_ms,
            "eager_peak_allocated_mb": eager_peak,
        }


def _flash_attn_inventory():
    try:
        import flash_attn
        from flash_attn import flash_attn_func

        return {
            "version": getattr(flash_attn, "__version__", None),
            "flash_attn_func_signature": str(inspect.signature(flash_attn_func)),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width-windows", type=int, default=3)
    parser.add_argument("--pressure-height-windows", type=int, default=64)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=144)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/HIP device is required for forced Flash SDPA")
    device = torch.device("cuda:0")
    q, k, v, combined_mask = _make_inputs(args, device)
    scale = args.head_dim**-0.5

    basic = _probe_case("no_mask", q, k, v, None, scale, args, device)
    combined = _probe_case(
        "learned_bias_plus_shifted_mask",
        q,
        k,
        v,
        combined_mask,
        scale,
        args,
        device,
    )
    adapter_candidate = bool(
        combined["forced_flash_supported"] and combined["numerically_compatible"]
    )
    report = {
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(device),
        "representative_shape": {
            "width_windows": args.width_windows,
            "pressure_height_windows": args.pressure_height_windows,
            "heads": args.heads,
            "tokens": args.tokens,
            "head_dim": args.head_dim,
        },
        "direct_flash_attn": _flash_attn_inventory(),
        "cases": [basic, combined],
        "adapter_candidate": adapter_candidate,
        "profiler_confirmation_required": adapter_candidate,
        "decision": (
            "PROFILE_FOR_FUSED_KERNEL"
            if adapter_candidate
            else "STOP_PYTORCH_FLASH_AND_TEST_HIP_STAGEWISE"
        ),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(args.output, "x", encoding="utf-8") as stream:
        stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
