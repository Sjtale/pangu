#!/usr/bin/env python3
"""Isolate direct Triton compiler/runtime failures in supervised subprocesses."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


WORKER = "--worker" in sys.argv

if WORKER:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _vector_add_kernel(x_ptr, y_ptr, output_ptr, size, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        valid = offsets < size
        x = tl.load(x_ptr + offsets, mask=valid)
        y = tl.load(y_ptr + offsets, mask=valid)
        tl.store(output_ptr + offsets, x + y, mask=valid)

    @triton.jit
    def _dot_kernel(a_ptr, b_ptr, output_ptr, size: tl.constexpr):
        offsets_m = tl.arange(0, size)
        offsets_n = tl.arange(0, size)
        offsets_k = tl.arange(0, size)
        a = tl.load(a_ptr + offsets_m[:, None] * size + offsets_k[None, :])
        b = tl.load(b_ptr + offsets_k[:, None] * size + offsets_n[None, :])
        output = tl.dot(a, b)
        tl.store(output_ptr + offsets_m[:, None] * size + offsets_n[None, :], output)

    @triton.jit
    def _attention_component_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        mask_ptr,
        output_ptr,
        scale: tl.constexpr,
        mode: tl.constexpr,
        max_fp16: tl.constexpr,
        dot_barrier: tl.constexpr,
        size: tl.constexpr,
        head_dim: tl.constexpr,
    ):
        offsets_m = tl.arange(0, size)
        offsets_n = tl.arange(0, size)
        offsets_d = tl.arange(0, head_dim)
        q = tl.load(
            q_ptr + offsets_m[:, None] * head_dim + offsets_d[None, :]
        )
        k = tl.load(
            k_ptr + offsets_d[:, None] + offsets_n[None, :] * head_dim
        )
        if dot_barrier:
            scores = tl.dot((q * scale).to(k.dtype), k)
        else:
            scores = tl.dot(q, k) * scale
        if mode >= 1:
            if dot_barrier:
                # Match the OneScience AlphaFold3 kernel's accumulator
                # materialization workaround before adding arbitrary bias.
                scores = scores.to(tl.uint32, bitcast=True) & 0xFFFFFFFF
                scores = scores.to(tl.float32, bitcast=True)
            matrix_offsets = offsets_m[:, None] * size + offsets_n[None, :]
            bias = tl.load(bias_ptr + matrix_offsets)
            shifted_mask = tl.load(mask_ptr + matrix_offsets)
            scores += bias.to(tl.float32) + shifted_mask.to(tl.float32)
        if mode >= 2:
            if max_fp16:
                row_max = tl.max(scores.to(tl.float16), axis=1).to(tl.float32)
            else:
                row_max = tl.max(scores, axis=1)
            scores -= row_max[:, None]
        if mode >= 3:
            probabilities = tl.exp(scores)
        if mode >= 4:
            denominator = tl.sum(probabilities, axis=1)
        if mode >= 5:
            probabilities /= denominator[:, None]
        if mode <= 2:
            output_offsets = offsets_m[:, None] * size + offsets_n[None, :]
            tl.store(output_ptr + output_offsets, scores)
        elif mode == 3:
            output_offsets = offsets_m[:, None] * size + offsets_n[None, :]
            tl.store(output_ptr + output_offsets, probabilities)
        elif mode == 4:
            tl.store(output_ptr + offsets_m, denominator)
        elif mode == 5:
            output_offsets = offsets_m[:, None] * size + offsets_n[None, :]
            tl.store(output_ptr + output_offsets, probabilities)
        else:
            values = tl.load(
                v_ptr + offsets_n[:, None] * head_dim + offsets_d[None, :]
            )
            output = tl.dot(probabilities.to(tl.float16), values)
            output_offsets = (
                offsets_m[:, None] * head_dim + offsets_d[None, :]
            )
            tl.store(output_ptr + output_offsets, output)


def _worker(stage):
    print(f"stage={stage} phase=imported", flush=True)
    print(
        f"torch={torch.__version__} hip={torch.version.hip} triton={triton.__version__}",
        flush=True,
    )
    if stage == "import":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is unavailable")
    device = torch.device("cuda:0")
    print(f"stage={stage} phase=device device={torch.cuda.get_device_name(device)}", flush=True)

    if stage == "vector":
        size = 1024
        x = torch.randn(size, device=device, dtype=torch.float32)
        y = torch.randn_like(x)
        output = torch.empty_like(x)
        print("stage=vector phase=launch", flush=True)
        _vector_add_kernel[(triton.cdiv(size, 256),)](
            x, y, output, size, block=256
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output, x + y)
        print("stage=vector phase=verified", flush=True)
        return

    if stage == "dot":
        size = 32
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)
        output = torch.empty_like(a)
        print("stage=dot phase=launch", flush=True)
        _dot_kernel[(1,)](a, b, output, size=size, num_warps=4)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output.float(), (a @ b).float(), atol=5e-3, rtol=1e-2)
        print("stage=dot phase=verified", flush=True)
        return

    component_modes = {
        "attention_qk": (0, False, False),
        "attention_bias": (1, False, False),
        "attention_max_fp16": (2, True, False),
        "attention_bias_bitcast": (1, False, True),
        "attention_max_bitcast": (2, False, True),
        "attention_exp_bitcast": (3, False, True),
        "attention_sum_bitcast": (4, False, True),
        "attention_normalize_bitcast": (5, False, True),
        "attention_pv_bitcast": (6, False, True),
    }
    if stage in component_modes:
        size = 32
        head_dim = 32
        q = torch.randn(size, head_dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        bias = torch.randn(size, size, device=device, dtype=torch.float16) * 0.02
        shifted_mask = torch.zeros_like(bias)
        shifted_mask[: size // 2, size // 2 :] = -100.0
        mode, max_fp16, dot_barrier = component_modes[stage]
        if mode == 4:
            output_shape = (size,)
        elif mode <= 5:
            output_shape = (size, size)
        else:
            output_shape = (size, head_dim)
        output_dtype = torch.float32 if mode <= 5 else torch.float16
        output = torch.empty(output_shape, device=device, dtype=output_dtype)
        print(f"stage={stage} phase=launch", flush=True)
        _attention_component_kernel[(1,)](
            q,
            k,
            v,
            bias,
            shifted_mask,
            output,
            scale=head_dim**-0.5,
            mode=mode,
            max_fp16=max_fp16,
            dot_barrier=dot_barrier,
            size=size,
            head_dim=head_dim,
            num_warps=4,
        )
        torch.cuda.synchronize(device)
        if dot_barrier:
            scores = (q * (head_dim**-0.5)).float() @ k.float().T
        else:
            scores = (q.float() @ k.float().T) * head_dim**-0.5
        if mode >= 1:
            scores += bias.float() + shifted_mask.float()
        if mode >= 2:
            if max_fp16:
                row_max = scores.half().max(dim=-1, keepdim=True).values.float()
            else:
                row_max = scores.max(dim=-1, keepdim=True).values
            scores -= row_max
        if mode >= 3:
            probabilities = torch.exp(scores)
        if mode >= 4:
            denominator = probabilities.sum(dim=-1)
        if mode >= 5:
            probabilities /= denominator[:, None]
        if mode <= 2:
            reference = scores
        elif mode == 3:
            reference = probabilities
        elif mode == 4:
            reference = denominator
        elif mode == 5:
            reference = probabilities
        else:
            reference = probabilities @ v.float()
        torch.testing.assert_close(
            output.float(), reference, atol=5e-3, rtol=1e-2
        )
        print(f"stage={stage} phase=verified", flush=True)
        return

    if stage in ("earth32", "earth144"):
        pangu_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(pangu_root))
        from triton_earth_attention import triton_earth_attention

        tokens = 32 if stage == "earth32" else 144
        shape = (1, 1, 1, tokens, 32)
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        earth = torch.randn(
            1, 1, 1, tokens, tokens, device=device, dtype=torch.float16
        ) * 0.02
        shifted = torch.zeros(
            1, 1, tokens, tokens, device=device, dtype=torch.float16
        )
        print(f"stage={stage} phase=launch", flush=True)
        output = triton_earth_attention(q, k, v, earth, shifted, 32**-0.5)
        torch.cuda.synchronize(device)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * 32**-0.5
        reference = torch.matmul(
            torch.softmax(scores + earth.float() + shifted.float(), dim=-1),
            v.float(),
        )
        torch.testing.assert_close(
            output.float(), reference, atol=5e-3, rtol=1e-2
        )
        print(f"stage={stage} phase=verified", flush=True)
        return
    raise ValueError(f"unknown worker stage: {stage}")


def _disable_core_dump():
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_supervisor(output_path):
    stages = (
        "import",
        "vector",
        "dot",
        "attention_qk",
        "attention_bias",
        "attention_bias_bitcast",
        "attention_max_bitcast",
        "attention_exp_bitcast",
        "attention_sum_bitcast",
        "attention_normalize_bitcast",
        "attention_pv_bitcast",
        "earth32",
        "earth144",
    )
    results = []
    stop = False
    for stage in stages:
        if stop:
            results.append({"stage": stage, "status": "SKIPPED_AFTER_FAILURE"})
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", stage]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
                preexec_fn=_disable_core_dump,
            )
            signal_name = None
            if completed.returncode < 0:
                signal_name = signal.Signals(-completed.returncode).name
            status = "PASS" if completed.returncode == 0 else "FAIL"
            results.append(
                {
                    "stage": stage,
                    "status": status,
                    "returncode": completed.returncode,
                    "signal": signal_name,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            stop = completed.returncode != 0
        except subprocess.TimeoutExpired as error:
            results.append(
                {
                    "stage": stage,
                    "status": "TIMEOUT",
                    "stdout": error.stdout,
                    "stderr": error.stderr,
                }
            )
            stop = True
    report = {
        "stages": results,
        "decision": (
            "DIRECT_TRITON_STACK_REQUIRES_REPAIR"
            if any(result["status"] != "PASS" for result in results)
            else "RERUN_REPRESENTATIVE_EARTH_ATTENTION"
        ),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    with open(output_path, "x", encoding="utf-8") as stream:
        stream.write(payload)
    print(payload, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument(
        "--worker",
        choices=(
            "import",
            "vector",
            "dot",
            "attention_qk",
            "attention_bias",
            "attention_max_fp16",
            "attention_bias_bitcast",
            "attention_max_bitcast",
            "attention_exp_bitcast",
            "attention_sum_bitcast",
            "attention_normalize_bitcast",
            "attention_pv_bitcast",
            "earth32",
            "earth144",
        ),
    )
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
        return
    if not args.output:
        parser.error("--output is required in supervisor mode")
    _run_supervisor(args.output)


if __name__ == "__main__":
    main()
