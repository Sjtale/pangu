"""Inference-only Triton EarthAttention forward for the Pangu DCU path."""

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _earth_attention_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        earth_bias_ptr,
        shifted_mask_ptr,
        output_ptr,
        stride_qw,
        stride_qh,
        stride_qp,
        stride_qm,
        stride_qd,
        stride_kw,
        stride_kh,
        stride_kp,
        stride_kn,
        stride_kd,
        stride_vw,
        stride_vh,
        stride_vp,
        stride_vn,
        stride_vd,
        stride_eh,
        stride_ep,
        stride_em,
        stride_en,
        stride_sw,
        stride_sp,
        stride_sm,
        stride_sn,
        stride_ow,
        stride_oh,
        stride_op,
        stride_om,
        stride_od,
        scale: tl.constexpr,
        heads: tl.constexpr,
        pressure_height: tl.constexpr,
        tokens: tl.constexpr,
        head_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        program_attention = tl.program_id(0)
        program_m = tl.program_id(1)

        pressure_index = program_attention % pressure_height
        width_head = program_attention // pressure_height
        head_index = width_head % heads
        width_index = width_head // heads

        offsets_m = program_m * block_m + tl.arange(0, block_m)
        offsets_n = tl.arange(0, block_n)
        offsets_d = tl.arange(0, block_d)
        valid_m = offsets_m < tokens
        valid_d = offsets_d < head_dim

        q_base = (
            width_index * stride_qw
            + head_index * stride_qh
            + pressure_index * stride_qp
        )
        q_offsets = (
            q_base
            + offsets_m[:, None] * stride_qm
            + offsets_d[None, :] * stride_qd
        )
        q = tl.load(
            q_ptr + q_offsets,
            mask=valid_m[:, None] & valid_d[None, :],
            other=0.0,
        )

        running_max = tl.full([block_m], -float("inf"), tl.float32)
        running_sum = tl.zeros([block_m], tl.float32)
        accumulator = tl.zeros([block_m, block_d], tl.float32)

        for start_n in range(0, tokens, block_n):
            current_n = start_n + offsets_n
            valid_n = current_n < tokens
            valid_mn = valid_m[:, None] & valid_n[None, :]

            k_base = (
                width_index * stride_kw
                + head_index * stride_kh
                + pressure_index * stride_kp
            )
            k_offsets = (
                k_base
                + offsets_d[:, None] * stride_kd
                + current_n[None, :] * stride_kn
            )
            k = tl.load(
                k_ptr + k_offsets,
                mask=valid_d[:, None] & valid_n[None, :],
                other=0.0,
            )

            scores = tl.dot(q, k) * scale
            earth_base = head_index * stride_eh + pressure_index * stride_ep
            earth_offsets = (
                earth_base
                + offsets_m[:, None] * stride_em
                + current_n[None, :] * stride_en
            )
            earth_bias = tl.load(
                earth_bias_ptr + earth_offsets,
                mask=valid_mn,
                other=0.0,
            )
            shifted_base = width_index * stride_sw + pressure_index * stride_sp
            shifted_offsets = (
                shifted_base
                + offsets_m[:, None] * stride_sm
                + current_n[None, :] * stride_sn
            )
            shifted_mask = tl.load(
                shifted_mask_ptr + shifted_offsets,
                mask=valid_mn,
                other=0.0,
            )
            scores += earth_bias.to(tl.float32) + shifted_mask.to(tl.float32)
            scores = tl.where(valid_mn, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            new_max = tl.where(valid_m, new_max, 0.0)
            correction = tl.where(
                valid_m, tl.exp(running_max - new_max), 0.0
            )
            probabilities = tl.where(
                valid_mn, tl.exp(scores - new_max[:, None]), 0.0
            )
            running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
            accumulator *= correction[:, None]

            v_base = (
                width_index * stride_vw
                + head_index * stride_vh
                + pressure_index * stride_vp
            )
            v_offsets = (
                v_base
                + current_n[:, None] * stride_vn
                + offsets_d[None, :] * stride_vd
            )
            values = tl.load(
                v_ptr + v_offsets,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            )
            accumulator += tl.dot(probabilities.to(tl.float16), values)
            running_max = new_max

        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        output = accumulator / denominator[:, None]
        output_base = (
            width_index * stride_ow
            + head_index * stride_oh
            + pressure_index * stride_op
        )
        output_offsets = (
            output_base
            + offsets_m[:, None] * stride_om
            + offsets_d[None, :] * stride_od
        )
        tl.store(
            output_ptr + output_offsets,
            output,
            mask=valid_m[:, None] & valid_d[None, :],
        )

else:
    _earth_attention_fwd_kernel = None


def _validate_inputs(q, k, v, earth_bias, shifted_mask):
    if q.ndim != 5:
        raise ValueError(
            "q/k/v must have shape [width, heads, pressure_height, tokens, dim]"
        )
    if tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        raise ValueError("q, k, and v must have identical shapes")
    width, heads, pressure_height, tokens, head_dim = q.shape
    if head_dim != 32:
        raise ValueError(
            f"Pangu Triton kernel currently requires head_dim=32, got {head_dim}"
        )
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must all be FP16")

    if earth_bias.ndim == 5 and earth_bias.shape[0] == 1:
        earth_bias = earth_bias[0]
    expected_earth = (heads, pressure_height, tokens, tokens)
    if tuple(earth_bias.shape) != expected_earth:
        raise ValueError(
            f"earth_bias must have shape [1, H, PH, L, L] or {expected_earth}"
        )

    if shifted_mask.ndim == 5 and shifted_mask.shape[2] == 1:
        shifted_mask = shifted_mask[:, :, 0]
    expected_mask = (width, pressure_height, tokens, tokens)
    if tuple(shifted_mask.shape) != expected_mask:
        raise ValueError(
            "shifted_mask must have shape [W, PH, L, L] or "
            f"[W, PH, 1, L, L], got {tuple(shifted_mask.shape)}"
        )
    if earth_bias.dtype != q.dtype or shifted_mask.dtype != q.dtype:
        raise TypeError("earth_bias and shifted_mask must match the FP16 q dtype")
    tensors = (q, k, v, earth_bias, shifted_mask)
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all attention tensors must be on the same device")
    return earth_bias, shifted_mask, (width, heads, pressure_height, tokens, head_dim)


@torch.no_grad()
def triton_earth_attention(q, k, v, earth_bias, shifted_mask, scale):
    """Compute fused Pangu attention without compile or score materialization."""

    earth_bias, shifted_mask, shape = _validate_inputs(
        q, k, v, earth_bias, shifted_mask
    )
    if triton is None:
        raise RuntimeError("The triton package is not installed")
    if q.device.type != "cuda":
        raise RuntimeError("A CUDA/HIP device is required for the Triton kernel")

    width, heads, pressure_height, tokens, head_dim = shape
    output = torch.empty_like(q)
    block_m = 32
    block_n = 32
    block_d = 32
    grid = (width * heads * pressure_height, triton.cdiv(tokens, block_m))
    _earth_attention_fwd_kernel[grid](
        q,
        k,
        v,
        earth_bias,
        shifted_mask,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *earth_bias.stride(),
        *shifted_mask.stride(),
        *output.stride(),
        scale=float(scale),
        heads=heads,
        pressure_height=pressure_height,
        tokens=tokens,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )
    return output
