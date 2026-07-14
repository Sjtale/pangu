"""Inference-only bias-aware EarthAttention backed by a local HIP shared library."""

import ctypes
import os
import subprocess
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "hip_kernels" / "earth_attention_fwd.hip"
_DEFAULT_BUILD_DIR = _ROOT / "logs" / "hip_earth_attention_build"
_LIBRARY_NAME = "libpangu_earth_attention.so"
_LIBRARY = None


def build_hip_earth_attention(force=False):
    """Compile the repository-owned HIP source without installing a package."""

    hipcc = Path(os.environ.get("PANGU_HIPCC", "/opt/dtk/bin/hipcc"))
    if not hipcc.is_file():
        raise FileNotFoundError(f"HIP compiler not found: {hipcc}")
    build_dir = Path(
        os.environ.get("PANGU_HIP_ATTENTION_BUILD_DIR", _DEFAULT_BUILD_DIR)
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / _LIBRARY_NAME
    if (
        force
        or not library_path.is_file()
        or library_path.stat().st_mtime < _SOURCE.stat().st_mtime
    ):
        command = [
            str(hipcc),
            "-O3",
            "-std=c++17",
            "-fPIC",
            "--shared",
            str(_SOURCE),
            "-Wl,-rpath,/opt/dtk/lib",
            "-o",
            str(library_path),
        ]
        subprocess.run(command, check=True)
    return library_path


def _load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    library = ctypes.CDLL(str(build_hip_earth_attention()))
    stride_pointer = ctypes.POINTER(ctypes.c_int64)
    library.pangu_earth_attention_fwd_fp16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        stride_pointer,
        stride_pointer,
        stride_pointer,
        stride_pointer,
        stride_pointer,
        stride_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    library.pangu_earth_attention_fwd_fp16.restype = ctypes.c_int
    library.pangu_hip_error_string.argtypes = [ctypes.c_int]
    library.pangu_hip_error_string.restype = ctypes.c_char_p
    _LIBRARY = library
    return library


def _validate_inputs(q, k, v, earth_bias, shifted_mask):
    if q.ndim != 5:
        raise ValueError("q/k/v must have shape [W, H, PH, L, D]")
    if tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        raise ValueError("q, k, and v must have identical shapes")
    width, heads, pressure_height, tokens, head_dim = q.shape
    if head_dim != 32:
        raise ValueError(f"HIP kernel requires head_dim=32, got {head_dim}")
    if tokens > 256:
        raise ValueError(f"HIP kernel supports at most 256 tokens, got {tokens}")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must all be FP16")

    if earth_bias.ndim == 5 and earth_bias.shape[0] == 1:
        earth_bias = earth_bias[0]
    expected_earth = (heads, pressure_height, tokens, tokens)
    if tuple(earth_bias.shape) != expected_earth:
        raise ValueError(
            "earth_bias must have shape [1, H, PH, L, L] or "
            f"{expected_earth}, got {tuple(earth_bias.shape)}"
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
    return earth_bias, shifted_mask, (
        width,
        heads,
        pressure_height,
        tokens,
        head_dim,
    )


def _stride_array(tensor):
    return (ctypes.c_int64 * tensor.ndim)(*tensor.stride())


@torch.no_grad()
def hip_earth_attention(q, k, v, earth_bias, shifted_mask, scale):
    """Run fused FP16/D32 EarthAttention on the current PyTorch HIP stream."""

    earth_bias, shifted_mask, shape = _validate_inputs(
        q, k, v, earth_bias, shifted_mask
    )
    if q.device.type != "cuda":
        raise RuntimeError("A CUDA/HIP device is required for the HIP kernel")
    width, heads, pressure_height, tokens, head_dim = shape
    output = torch.empty_like(q, memory_format=torch.preserve_format)
    strides = tuple(
        _stride_array(tensor)
        for tensor in (q, k, v, earth_bias, shifted_mask, output)
    )
    stream = torch.cuda.current_stream(q.device).cuda_stream
    library = _load_library()
    status = library.pangu_earth_attention_fwd_fp16(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(earth_bias.data_ptr()),
        ctypes.c_void_p(shifted_mask.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        *strides,
        width,
        heads,
        pressure_height,
        tokens,
        head_dim,
        ctypes.c_float(float(scale)),
        ctypes.c_void_p(stream),
    )
    if status != 0:
        message = library.pangu_hip_error_string(status)
        detail = message.decode("utf-8", errors="replace") if message else "unknown"
        raise RuntimeError(f"HIP EarthAttention launch failed ({status}): {detail}")
    return output
