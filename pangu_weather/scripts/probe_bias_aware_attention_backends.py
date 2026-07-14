#!/usr/bin/env python3
"""Probe fused attention backends against Pangu earth-bias semantics."""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import torch


ATOL = 5e-3
RTOL = 1e-2


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(call, device, warmup, repeat):
    for _ in range(warmup):
        output = call()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(repeat):
        output = call()
    _synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0 / repeat
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None
    )
    return output, latency_ms, peak_mb


def _make_inputs(args, device, tokens=None):
    tokens = args.tokens if tokens is None else tokens
    generator = torch.Generator(device=device).manual_seed(args.seed)
    batch = args.width_windows * args.pressure_height_windows
    qkv_shape = (batch, args.heads, tokens, args.head_dim)
    q = torch.randn(qkv_shape, generator=generator, device=device, dtype=torch.float16)
    k = torch.randn(qkv_shape, generator=generator, device=device, dtype=torch.float16)
    v = torch.randn(qkv_shape, generator=generator, device=device, dtype=torch.float16)
    earth_bias = torch.randn(
        args.pressure_height_windows,
        args.heads,
        tokens,
        tokens,
        generator=generator,
        device=device,
        dtype=torch.float16,
    ) * 0.02
    shifted_additive = torch.zeros(
        args.width_windows,
        args.pressure_height_windows,
        1,
        tokens,
        tokens,
        device=device,
        dtype=torch.float16,
    )
    split = tokens // 2
    shifted_additive[..., :split, split:] = -100.0
    shifted_additive[..., split:, :split] = -100.0
    return q, k, v, earth_bias, shifted_additive


def _combined_bias(earth_bias, shifted_additive):
    width, pressure_height = shifted_additive.shape[:2]
    return (earth_bias.unsqueeze(0) + shifted_additive).reshape(
        width * pressure_height, *earth_bias.shape[1:]
    )


def _reference(q, k, v, earth_bias, shifted_additive, scale):
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    scores = scores + _combined_bias(earth_bias, shifted_additive).float()
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def _eager_half(q, k, v, earth_bias, shifted_additive, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores + _combined_bias(earth_bias, shifted_additive)
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _errors(reference, candidate):
    difference = (reference.float() - candidate.float()).abs()
    return {
        "numerically_compatible": bool(
            torch.allclose(reference.float(), candidate.float(), atol=ATOL, rtol=RTOL)
        ),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


def _profile_names(call):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        call()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return sorted({event.name for event in profile.events()})


def _probe_flex(args, device):
    result = {"backend": "pytorch_flex_attention", "tokens": args.tokens}
    try:
        from torch.nn.attention.flex_attention import flex_attention

        q, k, v, earth_bias, shifted_additive = _make_inputs(args, device)
        scale = args.head_dim**-0.5
        reference = _reference(q, k, v, earth_bias, shifted_additive, scale)

        pressure_height = args.pressure_height_windows
        heads = args.heads
        tokens = args.tokens
        earth_bias_flat = earth_bias.flatten()
        shifted_additive_flat = shifted_additive.flatten()

        def score_mod(score, batch, head, query_index, key_index):
            width_index = batch // pressure_height
            pressure_index = batch % pressure_height
            earth_offset = (
                ((pressure_index * heads + head) * tokens + query_index) * tokens
                + key_index
            )
            shifted_offset = (
                (
                    (width_index * pressure_height + pressure_index) * tokens
                    + query_index
                )
                * tokens
                + key_index
            )
            return (
                score
                + earth_bias_flat[earth_offset]
                + shifted_additive_flat[shifted_offset]
            )

        compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)

        def call():
            return compiled(q, k, v, score_mod=score_mod, scale=scale)

        candidate, latency_ms, peak_mb = _measure(
            call, device, args.warmup, args.repeat
        )
        profile_names = _profile_names(call)
        errors = _errors(reference, candidate)
        result.update(
            {
                "status": "PASS" if errors["numerically_compatible"] else "FAIL",
                "latency_ms": latency_ms,
                "peak_allocated_mb": peak_mb,
                "profile_events": profile_names,
                "profiler_confirmation_required": True,
                **errors,
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "FAIL",
                "numerically_compatible": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return result


def _probe_pytorch_triton(args, device):
    result = {"backend": "pytorch_native_triton", "tokens": args.tokens}
    try:
        pangu_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(pangu_root))
        from triton_earth_attention import triton_earth_attention

        q, k, v, earth_bias, shifted_additive = _make_inputs(args, device)
        scale = args.head_dim**-0.5
        reference = _reference(q, k, v, earth_bias, shifted_additive, scale)
        width = args.width_windows
        pressure_height = args.pressure_height_windows

        def model_layout(tensor):
            return tensor.reshape(
                width,
                pressure_height,
                args.heads,
                args.tokens,
                args.head_dim,
            ).permute(0, 2, 1, 3, 4)

        q_model = model_layout(q)
        k_model = model_layout(k)
        v_model = model_layout(v)
        earth_model = earth_bias.permute(1, 0, 2, 3).unsqueeze(0)

        _, eager_ms, eager_peak_mb = _measure(
            lambda: _eager_half(
                q, k, v, earth_bias, shifted_additive, scale
            ),
            device,
            args.warmup,
            args.repeat,
        )

        def call():
            return triton_earth_attention(
                q_model,
                k_model,
                v_model,
                earth_model,
                shifted_additive,
                scale,
            )

        candidate, latency_ms, peak_mb = _measure(
            call, device, args.warmup, args.repeat
        )
        candidate = candidate.permute(0, 2, 1, 3, 4).reshape_as(q)
        errors = _errors(reference, candidate)
        try:
            profile_fields = {"profile_events": _profile_names(call)}
        except Exception as profile_error:
            profile_fields = {
                "profile_error": f"{type(profile_error).__name__}: {profile_error}"
            }
        result.update(
            {
                "status": "PASS" if errors["numerically_compatible"] else "FAIL",
                "eager_latency_ms": eager_ms,
                "latency_ms": latency_ms,
                "speedup": eager_ms / latency_ms,
                "eager_peak_allocated_mb": eager_peak_mb,
                "peak_allocated_mb": peak_mb,
                "uses_torch_compile": False,
                "materializes_attention_scores": False,
                "profiler_confirmation_required": True,
                **profile_fields,
                **errors,
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "FAIL",
                "numerically_compatible": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return result


def _infer_onescience_root():
    return Path(__file__).resolve().parents[3] / "onescience"


def _pad_for_triton(inputs, padded_tokens):
    q, k, v, earth_bias, shifted_additive = inputs
    original_tokens = q.shape[-2]
    if padded_tokens < original_tokens:
        raise ValueError("padded token count must not be smaller than the original")
    if padded_tokens == original_tokens:
        allowed = shifted_additive == 0
        return inputs, allowed

    def pad_qkv(tensor):
        output = torch.zeros(
            *tensor.shape[:-2], padded_tokens, tensor.shape[-1], dtype=tensor.dtype
        )
        output[..., :original_tokens, :] = tensor
        return output

    padded_bias = torch.zeros(
        *earth_bias.shape[:-2], padded_tokens, padded_tokens, dtype=earth_bias.dtype
    )
    padded_bias[..., :original_tokens, :original_tokens] = earth_bias
    padded_shift = torch.zeros(
        *shifted_additive.shape[:-2],
        padded_tokens,
        padded_tokens,
        dtype=shifted_additive.dtype,
    )
    padded_shift[..., :original_tokens, :original_tokens] = shifted_additive
    allowed = padded_shift == 0
    allowed[..., original_tokens:, :] = False
    allowed[..., :, original_tokens:] = False
    allowed[..., original_tokens:, 0] = True
    return (
        pad_qkv(q),
        pad_qkv(k),
        pad_qkv(v),
        padded_bias,
        padded_shift,
    ), allowed


def _probe_onescience_triton_case(args, padded_tokens):
    result = {
        "backend": "onescience_alphafold3_triton",
        "original_tokens": args.tokens,
        "kernel_tokens": padded_tokens,
    }
    try:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        root = Path(args.onescience_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"OneScience root not found: {root}")
        sys.path.insert(0, str(root / "src"))
        import jax
        import jax.numpy as jnp
        import numpy as np
        from onescience.flax_models.alphafold3.jax.attention import attention

        if jax.default_backend() != "gpu":
            raise RuntimeError(f"JAX GPU backend required, got {jax.default_backend()}")

        cpu_inputs = _make_inputs(args, torch.device("cpu"))
        reference = _reference(*cpu_inputs, scale=args.head_dim**-0.5)
        padded, allowed = _pad_for_triton(cpu_inputs, padded_tokens)
        q, k, v, earth_bias, _ = padded
        batch = args.width_windows * args.pressure_height_windows
        bias = earth_bias.unsqueeze(0).expand(
            args.width_windows, -1, -1, -1, -1
        ).reshape(batch, args.heads, padded_tokens, padded_tokens)
        mask = allowed.expand(-1, -1, args.heads, -1, -1).reshape(
            batch, args.heads, padded_tokens, padded_tokens
        )

        q_jax = jax.device_put(np.asarray(q.permute(0, 2, 1, 3)))
        k_jax = jax.device_put(np.asarray(k.permute(0, 2, 1, 3)))
        v_jax = jax.device_put(np.asarray(v.permute(0, 2, 1, 3)))
        bias_jax = jax.device_put(np.asarray(bias))
        mask_jax = jax.device_put(np.asarray(mask, dtype=np.bool_))

        call = jax.jit(
            lambda: attention.dot_product_attention(
                q_jax,
                k_jax,
                v_jax,
                bias=bias_jax,
                mask=mask_jax,
                implementation="triton",
                logits_dtype=jnp.float32,
            )
        )
        for _ in range(args.warmup):
            output = call()
            output.block_until_ready()
        started = time.perf_counter()
        for _ in range(args.repeat):
            output = call()
        output.block_until_ready()
        latency_ms = (time.perf_counter() - started) * 1000.0 / args.repeat
        candidate = torch.from_numpy(np.asarray(output)).permute(0, 2, 1, 3)
        candidate = candidate[..., : args.tokens, :]
        errors = _errors(reference, candidate)
        result.update(
            {
                "status": "PASS" if errors["numerically_compatible"] else "FAIL",
                "latency_ms": latency_ms,
                "explicit_triton_no_xla_fallback": True,
                "profiler_confirmation_required": True,
                **errors,
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "FAIL",
                "numerically_compatible": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return result


def _probe_transformer_engine(args, device):
    result = {"backend": "transformer_engine_fused_attention", "tokens": args.tokens}
    try:
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_UNFUSED_ATTN"] = "0"
        os.environ.setdefault("NVTE_DEBUG", "1")
        os.environ.setdefault("NVTE_DEBUG_LEVEL", "2")
        from transformer_engine import pytorch as te

        forward_parameters = inspect.signature(
            te.DotProductAttention.forward
        ).parameters
        required = {"core_attention_bias_type", "core_attention_bias"}
        if not required.issubset(forward_parameters):
            raise RuntimeError(
                "Transformer Engine API has no post-scale arbitrary bias arguments"
            )

        q, k, v, earth_bias, shifted_additive = _make_inputs(args, device)
        scale = args.head_dim**-0.5
        reference = _reference(q, k, v, earth_bias, shifted_additive, scale)
        combined = _combined_bias(earth_bias, shifted_additive)
        q_bshd = q.permute(0, 2, 1, 3)
        k_bshd = k.permute(0, 2, 1, 3)
        v_bshd = v.permute(0, 2, 1, 3)
        module = te.DotProductAttention(
            num_attention_heads=args.heads,
            kv_channels=args.head_dim,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            softmax_scale=scale,
        ).to(device)

        def call():
            output = module(
                q_bshd,
                k_bshd,
                v_bshd,
                qkv_format="bshd",
                core_attention_bias_type="post_scale_bias",
                core_attention_bias=combined,
            )
            return output.reshape(q_bshd.shape).permute(0, 2, 1, 3)

        candidate, latency_ms, peak_mb = _measure(
            call, device, args.warmup, args.repeat
        )
        profile_names = _profile_names(call)
        errors = _errors(reference, candidate)
        result.update(
            {
                "status": "PASS" if errors["numerically_compatible"] else "FAIL",
                "latency_ms": latency_ms,
                "peak_allocated_mb": peak_mb,
                "forced_environment": {
                    "NVTE_FLASH_ATTN": "0",
                    "NVTE_FUSED_ATTN": "1",
                    "NVTE_UNFUSED_ATTN": "0",
                    "NVTE_DEBUG": os.environ["NVTE_DEBUG"],
                    "NVTE_DEBUG_LEVEL": os.environ["NVTE_DEBUG_LEVEL"],
                },
                "profile_events": profile_names,
                "profiler_confirmation_required": True,
                **errors,
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "SKIP_OR_FAIL",
                "numerically_compatible": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backend",
        choices=(
            "all",
            "pytorch-triton",
            "flex",
            "onescience-triton",
            "transformer-engine",
        ),
        default="all",
    )
    parser.add_argument("--onescience-root", default=str(_infer_onescience_root()))
    parser.add_argument("--width-windows", type=int, default=3)
    parser.add_argument("--pressure-height-windows", type=int, default=64)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=144)
    parser.add_argument("--padded-tokens", type=int, default=192)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/HIP device is required")
    device = torch.device("cuda:0")
    results = []
    if args.backend in ("all", "pytorch-triton"):
        results.append(_probe_pytorch_triton(args, device))
    if args.backend in ("all", "flex"):
        results.append(_probe_flex(args, device))
    if args.backend in ("all", "onescience-triton"):
        results.append(_probe_onescience_triton_case(args, args.tokens))
        results.append(_probe_onescience_triton_case(args, args.padded_tokens))
    if args.backend in ("all", "transformer-engine"):
        results.append(_probe_transformer_engine(args, device))

    report = {
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(device),
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "shape": {
            "width_windows": args.width_windows,
            "pressure_height_windows": args.pressure_height_windows,
            "heads": args.heads,
            "tokens": args.tokens,
            "head_dim": args.head_dim,
        },
        "results": results,
        "acceptance": (
            "PASS is only a numerical gate; accept a backend only after hipprof "
            "confirms a fused kernel and full-model A/B beats the 90.1048 guardrail"
        ),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(args.output, "x", encoding="utf-8") as stream:
        stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
