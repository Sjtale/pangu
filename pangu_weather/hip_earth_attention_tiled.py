"""Inference-only wrapper for the repository-owned tiled HIP EarthAttention."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import shlex
import subprocess
import uuid
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "hip_kernels" / "earth_attention_tiled_fwd.hip"
_DEFAULT_BUILD_DIR = _ROOT / "logs" / "hip_earth_attention_tiled_build"
_LIBRARY_PREFIX = "libpangu_earth_attention_tiled"
_FIXED_COMPILE_FLAGS = (
    "-O3",
    "-std=c++17",
    "-fPIC",
    "--shared",
    "-Wl,-rpath,/opt/dtk/lib",
)
_LIBRARIES: dict[Path, ctypes.CDLL] = {}
_BUILD_INFO: dict[Path, dict] = {}
_ACTIVE_LIBRARY: ctypes.CDLL | None = None
_ACTIVE_LIBRARY_PATH: Path | None = None
_VALIDATED_INDEX_KEYS: set[tuple] = set()
_SUPPORTED_TOKENS = {32, 144}
_HEAD_DIM = 32
_BLOCKED_MASK_VALUE = -100.0
_INT32_MAX = 2**31 - 1
_KERNEL_MODES = {"online", "full-row-fast", "full-row-expf"}


def _compiler_configuration():
    if not _SOURCE.is_file():
        raise FileNotFoundError(f"Tiled HIP source not found: {_SOURCE}")

    hipcc = Path(os.environ.get("PANGU_HIPCC", "/opt/dtk/bin/hipcc")).expanduser()
    if not hipcc.is_file():
        raise FileNotFoundError(f"HIP compiler not found: {hipcc}")
    hipcc = hipcc.resolve()
    version_result = subprocess.run(
        [str(hipcc), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    version_bytes = version_result.stdout or b""

    raw_extra_flags = os.environ.get("PANGU_TILED_HIP_EXTRA_FLAGS", "")
    try:
        extra_flags = shlex.split(raw_extra_flags)
    except ValueError as error:
        raise ValueError(f"Invalid PANGU_TILED_HIP_EXTRA_FLAGS: {error}") from error

    raw_arch = os.environ.get("PANGU_TILED_HIP_ARCH")
    arch = raw_arch.strip() if raw_arch and raw_arch.strip() else None
    source_bytes = _SOURCE.read_bytes()
    fingerprint_payload = {
        "hipcc_path": str(hipcc),
        "hipcc_version": version_bytes.decode("utf-8", errors="replace"),
        "fixed_flags": list(_FIXED_COMPILE_FLAGS),
        "extra_flags": extra_flags,
        "arch": arch,
    }
    digest = hashlib.sha256()
    digest.update(source_bytes)
    digest.update(b"\0")
    digest.update(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return hipcc, extra_flags, arch, digest.hexdigest(), fingerprint_payload


def build_hip_earth_attention_tiled(force=False):
    """Build and cache the tiled HIP shared library without installing it."""

    hipcc, extra_flags, arch, fingerprint, metadata = _compiler_configuration()
    build_dir = Path(
        os.environ.get("PANGU_TILED_HIP_BUILD_DIR", _DEFAULT_BUILD_DIR)
    ).expanduser()
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / f"{_LIBRARY_PREFIX}-{fingerprint}.so"
    _BUILD_INFO[library_path.resolve()] = {
        **metadata,
        "fingerprint": fingerprint,
        "source": str(_SOURCE),
        "library": str(library_path.resolve()),
    }
    if library_path.is_file() and not force:
        return library_path

    temporary_path = build_dir / (
        f".{library_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    command = [str(hipcc), *_FIXED_COMPILE_FLAGS[:4]]
    if arch is not None:
        command.append(f"--offload-arch={arch}")
    command.extend(extra_flags)
    command.extend(
        [
            str(_SOURCE),
            _FIXED_COMPILE_FLAGS[4],
            "-o",
            str(temporary_path),
        ]
    )
    try:
        subprocess.run(command, check=True)
        os.replace(temporary_path, library_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return library_path


def _bind_library(library):
    void_pointer = ctypes.c_void_p
    library.pangu_earth_attention_tiled_fwd_fp16.argtypes = [
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        void_pointer,
    ]
    library.pangu_earth_attention_tiled_fwd_fp16.restype = ctypes.c_int

    library.pangu_earth_attention_tiled_full_row_fwd_fp16.argtypes = [
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        void_pointer,
    ]
    library.pangu_earth_attention_tiled_full_row_fwd_fp16.restype = ctypes.c_int
    library.pangu_earth_attention_tiled_full_row_diagnostic_fp16.argtypes = [
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        void_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        void_pointer,
    ]
    library.pangu_earth_attention_tiled_full_row_diagnostic_fp16.restype = (
        ctypes.c_int
    )

    library.pangu_earth_attention_tiled_get_config.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.pangu_earth_attention_tiled_get_config.restype = ctypes.c_int
    library.pangu_earth_attention_tiled_get_occupancy.argtypes = [
        ctypes.POINTER(ctypes.c_int)
    ]
    library.pangu_earth_attention_tiled_get_occupancy.restype = ctypes.c_int
    library.pangu_earth_attention_tiled_full_row_get_config.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.pangu_earth_attention_tiled_full_row_get_config.restype = ctypes.c_int
    library.pangu_earth_attention_tiled_full_row_get_occupancy.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.pangu_earth_attention_tiled_full_row_get_occupancy.restype = ctypes.c_int
    library.pangu_earth_attention_tiled_implementation_kind.argtypes = []
    library.pangu_earth_attention_tiled_implementation_kind.restype = ctypes.c_char_p
    library.pangu_earth_attention_tiled_full_row_implementation_kind.argtypes = [
        ctypes.c_int
    ]
    library.pangu_earth_attention_tiled_full_row_implementation_kind.restype = (
        ctypes.c_char_p
    )
    library.pangu_earth_attention_tiled_error_string.argtypes = [ctypes.c_int]
    library.pangu_earth_attention_tiled_error_string.restype = ctypes.c_char_p
    return library


def _load_library():
    global _ACTIVE_LIBRARY, _ACTIVE_LIBRARY_PATH
    if _ACTIVE_LIBRARY is not None:
        return _ACTIVE_LIBRARY, _ACTIVE_LIBRARY_PATH

    library_path = build_hip_earth_attention_tiled().resolve()
    library = _LIBRARIES.get(library_path)
    if library is None:
        library = _bind_library(ctypes.CDLL(str(library_path)))
        _LIBRARIES[library_path] = library
    _ACTIVE_LIBRARY = library
    _ACTIVE_LIBRARY_PATH = library_path
    return _ACTIVE_LIBRARY, _ACTIVE_LIBRARY_PATH


def _error_detail(library, status):
    message = library.pangu_earth_attention_tiled_error_string(status)
    return message.decode("utf-8", errors="replace") if message else "unknown"


def _check_status(library, status, operation):
    if status != 0:
        raise RuntimeError(
            f"Tiled HIP EarthAttention {operation} failed ({status}): "
            f"{_error_detail(library, status)}"
        )


def _resolve_kernel_mode(mode):
    if mode not in _KERNEL_MODES:
        supported = ", ".join(sorted(_KERNEL_MODES))
        raise ValueError(f"kernel mode must be one of: {supported}")
    return mode, int(mode == "full-row-expf")


def _require_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_nonempty(name, tensor):
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")


def _integer_min_max(tensor):
    values = tensor.detach().to(dtype=torch.int64)
    return int(values.min().item()), int(values.max().item())


def pack_earth_bias_table(earth_position_bias_table):
    """Pack a Pangu bias table from ``[R, PH, H]`` to ``[H, PH, R]``."""

    _require_tensor("earth_position_bias_table", earth_position_bias_table)
    if earth_position_bias_table.ndim != 3:
        raise ValueError("earth_position_bias_table must have shape [R, PH, H]")
    _require_nonempty("earth_position_bias_table", earth_position_bias_table)
    if earth_position_bias_table.dtype != torch.float16:
        raise TypeError("earth_position_bias_table must be FP16")
    bias_rows, pressure_height, heads = earth_position_bias_table.shape
    if bias_rows > 32768:
        raise ValueError(
            f"int16 position indices support at most 32768 bias rows, got {bias_rows}"
        )
    if min(bias_rows, pressure_height, heads) <= 0:
        raise ValueError("earth_position_bias_table dimensions must be positive")
    if not torch.isfinite(earth_position_bias_table).all().item():
        raise ValueError("earth_position_bias_table must contain only finite values")
    return earth_position_bias_table.permute(2, 1, 0).contiguous()


def compact_earth_position_index(earth_position_index, bias_rows=None):
    """Validate and compact a square Pangu position-index matrix to int16."""

    _require_tensor("earth_position_index", earth_position_index)
    if earth_position_index.ndim != 2:
        raise ValueError("earth_position_index must have shape [L, L]")
    tokens, key_tokens = earth_position_index.shape
    if tokens != key_tokens:
        raise ValueError("earth_position_index must be square")
    if tokens not in _SUPPORTED_TOKENS:
        raise ValueError(
            f"tiled HIP kernel supports L=32 or L=144, got L={tokens}"
        )
    _require_nonempty("earth_position_index", earth_position_index)
    if earth_position_index.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("earth_position_index must have an integer dtype")
    minimum, maximum = _integer_min_max(earth_position_index)
    if minimum < 0 or maximum > 32767:
        raise ValueError(
            "earth_position_index values must be in the int16 range [0, 32767]"
        )
    if bias_rows is not None:
        if isinstance(bias_rows, bool) or not isinstance(bias_rows, int):
            raise TypeError("bias_rows must be a positive integer")
        if bias_rows <= 0 or bias_rows > 32768:
            raise ValueError("bias_rows must be in [1, 32768]")
        if maximum >= bias_rows:
            raise ValueError(
                f"earth_position_index contains {maximum}, but bias_rows is "
                f"only {bias_rows}"
            )
    compacted = earth_position_index.to(dtype=torch.int16).contiguous()
    if bias_rows is not None:
        _VALIDATED_INDEX_KEYS.add(_position_index_key(compacted, bias_rows))
    return compacted


def shifted_mask_to_region_ids(shifted_mask):
    """Convert an exactly reconstructable 0/-100 dense mask to uint8 labels."""

    _require_tensor("shifted_mask", shifted_mask)
    if shifted_mask.ndim == 5:
        if shifted_mask.shape[2] != 1:
            raise ValueError(
                "5D shifted_mask must have shape [maskW, PH, 1, L, L]"
            )
        shifted_mask = shifted_mask[:, :, 0]
    if shifted_mask.ndim != 4:
        raise ValueError("shifted_mask must have shape [maskW, PH, L, L]")
    mask_width, pressure_height, tokens, key_tokens = shifted_mask.shape
    if min(mask_width, pressure_height) <= 0 or tokens != key_tokens:
        raise ValueError("shifted_mask dimensions must be positive and square")
    if tokens not in _SUPPORTED_TOKENS:
        raise ValueError(
            f"tiled HIP kernel supports L=32 or L=144, got L={tokens}"
        )
    if not torch.is_floating_point(shifted_mask):
        raise TypeError("shifted_mask must have a floating-point dtype")

    valid_values = (shifted_mask == 0) | (
        shifted_mask == _BLOCKED_MASK_VALUE
    )
    if not valid_values.all().item():
        raise ValueError("shifted_mask must contain only exact 0 and -100 values")

    zero_relation = (shifted_mask == 0).detach().to(device="cpu")
    flat_relations = zero_relation.reshape(-1, tokens, tokens)
    flat_labels = torch.empty(
        (flat_relations.shape[0], tokens), dtype=torch.uint8, device="cpu"
    )
    for matrix_index, relation in enumerate(flat_relations):
        labels = torch.full((tokens,), -1, dtype=torch.int16)
        region_id = 0
        for query_index in range(tokens):
            if labels[query_index].item() >= 0:
                continue
            members = relation[query_index]
            if not members[query_index].item():
                raise ValueError("shifted_mask zero relation must contain its diagonal")
            if region_id > 255:
                raise ValueError("shifted_mask requires more than 256 region IDs")
            labels[members] = region_id
            region_id += 1
        reconstructed = labels[:, None] == labels[None, :]
        if not torch.equal(reconstructed, relation):
            raise ValueError(
                "shifted_mask zero entries are not reconstructable from region IDs"
            )
        flat_labels[matrix_index].copy_(labels.to(dtype=torch.uint8))

    labels = flat_labels.reshape(mask_width, pressure_height, tokens)
    return labels.to(device=shifted_mask.device).contiguous()


def _position_index_key(position_index, bias_rows):
    try:
        version = int(position_index._version)
    except RuntimeError:
        version = None
    return (
        str(position_index.device),
        position_index.data_ptr(),
        tuple(position_index.shape),
        version,
        int(bias_rows),
    )


def _require_registered_position_index(position_index, bias_rows):
    if _position_index_key(position_index, bias_rows) not in _VALIDATED_INDEX_KEYS:
        raise ValueError(
            "position_index is not registered for this packed_bias; call "
            "compact_earth_position_index(index, bias_rows) on the final-device "
            "tensor before timed inference"
        )


def _validate_forward_inputs(
    qkv,
    packed_bias,
    position_index,
    region_ids,
    scale,
    mask_width,
    width_offset,
):
    for name, tensor in (
        ("qkv", qkv),
        ("packed_bias", packed_bias),
        ("position_index", position_index),
    ):
        _require_tensor(name, tensor)
    if qkv.ndim != 6:
        raise ValueError("qkv must have shape [W, PH, L, 3, H, 32]")
    width, pressure_height, tokens, qkv_parts, heads, head_dim = qkv.shape
    if min(width, pressure_height, heads) <= 0:
        raise ValueError("qkv W, PH, and H dimensions must be positive")
    if max(width, pressure_height, heads) > _INT32_MAX:
        raise ValueError("qkv W, PH, and H dimensions must fit int32")
    if qkv_parts != 3 or head_dim != _HEAD_DIM:
        raise ValueError("qkv must have trailing shape [3, H, 32]")
    if tokens not in _SUPPORTED_TOKENS:
        raise ValueError(f"tiled HIP kernel supports L=32 or L=144, got L={tokens}")
    if qkv.dtype != torch.float16:
        raise TypeError("qkv must be FP16")
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous")
    if qkv.device.type != "cuda":
        raise RuntimeError("qkv must be on a CUDA/HIP device")

    expected_bias_prefix = (heads, pressure_height)
    if packed_bias.ndim != 3 or tuple(packed_bias.shape[:2]) != expected_bias_prefix:
        raise ValueError(
            f"packed_bias must have shape [H, PH, R] with prefix "
            f"{expected_bias_prefix}, got {tuple(packed_bias.shape)}"
        )
    bias_rows = packed_bias.shape[2]
    if bias_rows <= 0 or bias_rows > 32768:
        raise ValueError("packed_bias R must be in [1, 32768]")
    if packed_bias.dtype != torch.float16 or not packed_bias.is_contiguous():
        raise TypeError("packed_bias must be contiguous FP16")

    if tuple(position_index.shape) != (tokens, tokens):
        raise ValueError(f"position_index must have shape [{tokens}, {tokens}]")
    if position_index.dtype != torch.int16 or not position_index.is_contiguous():
        raise TypeError("position_index must be contiguous int16")

    tensors = [packed_bias, position_index]
    if region_ids is None:
        if mask_width not in (None, 0):
            raise ValueError("mask_width must be None or 0 when region_ids is None")
        if width_offset != 0:
            raise ValueError("width_offset must be 0 when region_ids is None")
        resolved_mask_width = 0
    else:
        _require_tensor("region_ids", region_ids)
        expected_region_suffix = (pressure_height, tokens)
        if region_ids.ndim != 3 or tuple(region_ids.shape[1:]) != expected_region_suffix:
            raise ValueError(
                "region_ids must have shape [maskW, PH, L] with suffix "
                f"{expected_region_suffix}, got {tuple(region_ids.shape)}"
            )
        if region_ids.shape[0] <= 0:
            raise ValueError("region_ids maskW must be positive")
        if region_ids.shape[0] > _INT32_MAX:
            raise ValueError("region_ids maskW must fit int32")
        if region_ids.dtype != torch.uint8 or not region_ids.is_contiguous():
            raise TypeError("region_ids must be contiguous uint8")
        if mask_width is not None and (
            isinstance(mask_width, bool) or not isinstance(mask_width, int)
        ):
            raise TypeError("mask_width must be an integer or None")
        resolved_mask_width = region_ids.shape[0] if mask_width is None else mask_width
        if resolved_mask_width != region_ids.shape[0]:
            raise ValueError("mask_width must equal region_ids.shape[0]")
        if (
            isinstance(width_offset, bool)
            or not isinstance(width_offset, int)
            or width_offset < 0
        ):
            raise ValueError("width_offset must be a non-negative integer")
        tensors.append(region_ids)

    if any(tensor.device != qkv.device for tensor in tensors):
        raise ValueError("all tiled EarthAttention tensors must be on the same device")
    if isinstance(scale, bool):
        raise TypeError("scale must be a finite positive float")
    try:
        scale_value = float(scale)
    except (TypeError, ValueError) as error:
        raise TypeError("scale must be a finite positive float") from error
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("scale must be a finite positive float")
    if (
        isinstance(width_offset, bool)
        or not isinstance(width_offset, int)
        or width_offset > _INT32_MAX
        or width_offset > _INT32_MAX - width + 1
    ):
        raise ValueError("width_offset must fit a non-negative int32")

    _require_registered_position_index(position_index, bias_rows)
    return (
        width,
        pressure_height,
        heads,
        tokens,
        bias_rows,
        resolved_mask_width,
        scale_value,
    )


@torch.no_grad()
def hip_earth_attention_tiled(
    qkv,
    packed_bias,
    position_index,
    region_ids=None,
    *,
    scale,
    mask_width=None,
    width_offset=0,
    mode="online",
    return_diagnostics=False,
):
    """Run tiled FP16/D32 EarthAttention on PyTorch's current HIP stream."""

    mode, standard_exp = _resolve_kernel_mode(mode)
    if return_diagnostics and mode == "online":
        raise ValueError("stage diagnostics are available only for full-row modes")
    shape = _validate_forward_inputs(
        qkv,
        packed_bias,
        position_index,
        region_ids,
        scale,
        mask_width,
        width_offset,
    )
    (
        width,
        pressure_height,
        heads,
        tokens,
        bias_rows,
        resolved_mask_width,
        scale_value,
    ) = shape
    output = torch.empty(
        (width, pressure_height, tokens, heads * _HEAD_DIM),
        dtype=torch.float16,
        device=qkv.device,
    )
    stream = torch.cuda.current_stream(qkv.device).cuda_stream
    library, _ = _load_library()
    region_pointer = (
        ctypes.c_void_p() if region_ids is None else ctypes.c_void_p(region_ids.data_ptr())
    )
    arguments = (
        ctypes.c_void_p(qkv.data_ptr()),
        ctypes.c_void_p(packed_bias.data_ptr()),
        ctypes.c_void_p(position_index.data_ptr()),
        region_pointer,
        ctypes.c_void_p(output.data_ptr()),
        width,
        pressure_height,
        heads,
        tokens,
        bias_rows,
        resolved_mask_width,
        width_offset,
        ctypes.c_float(scale_value),
    )
    if return_diagnostics:
        stage_shape = (width, heads, pressure_height, tokens, tokens)
        diagnostics = {
            "qk_half": torch.empty(stage_shape, dtype=torch.float16, device=qkv.device),
            "biased_half": torch.empty(
                stage_shape,
                dtype=torch.float16,
                device=qkv.device,
            ),
            "probability_half": torch.empty(
                stage_shape,
                dtype=torch.float16,
                device=qkv.device,
            ),
        }
        status = library.pangu_earth_attention_tiled_full_row_diagnostic_fp16(
            *arguments[:5],
            ctypes.c_void_p(diagnostics["qk_half"].data_ptr()),
            ctypes.c_void_p(diagnostics["biased_half"].data_ptr()),
            ctypes.c_void_p(diagnostics["probability_half"].data_ptr()),
            *arguments[5:],
            standard_exp,
            ctypes.c_void_p(stream),
        )
    elif mode == "online":
        status = library.pangu_earth_attention_tiled_fwd_fp16(
            *arguments,
            ctypes.c_void_p(stream),
        )
    else:
        status = library.pangu_earth_attention_tiled_full_row_fwd_fp16(
            *arguments,
            standard_exp,
            ctypes.c_void_p(stream),
        )
    _check_status(library, status, "launch")
    if return_diagnostics:
        return output, diagnostics
    return output


def _device_context(device):
    if device is None:
        return torch.cuda.device(torch.cuda.current_device())
    if isinstance(device, int) and not isinstance(device, bool):
        return torch.cuda.device(device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("device must identify a CUDA/HIP device")
    return torch.cuda.device(resolved)


def get_hip_earth_attention_tiled_info(device=None, mode="online"):
    """Return compiled implementation, tile configuration, and occupancy."""

    mode, standard_exp = _resolve_kernel_mode(mode)
    library, library_path = _load_library()
    q_tile = ctypes.c_int()
    k_tile = ctypes.c_int()
    head_dim = ctypes.c_int()
    block_threads = ctypes.c_int()
    dynamic_smem = ctypes.c_size_t()
    config_function = (
        library.pangu_earth_attention_tiled_get_config
        if mode == "online"
        else library.pangu_earth_attention_tiled_full_row_get_config
    )
    status = config_function(
        ctypes.byref(q_tile),
        ctypes.byref(k_tile),
        ctypes.byref(head_dim),
        ctypes.byref(block_threads),
        ctypes.byref(dynamic_smem),
    )
    _check_status(library, status, "configuration query")

    active_blocks = ctypes.c_int()
    with _device_context(device):
        if mode == "online":
            status = library.pangu_earth_attention_tiled_get_occupancy(
                ctypes.byref(active_blocks)
            )
        else:
            status = library.pangu_earth_attention_tiled_full_row_get_occupancy(
                standard_exp,
                ctypes.byref(active_blocks),
            )
    _check_status(library, status, "occupancy query")

    raw_kind = (
        library.pangu_earth_attention_tiled_implementation_kind()
        if mode == "online"
        else library.pangu_earth_attention_tiled_full_row_implementation_kind(
            standard_exp
        )
    )
    implementation_kind = (
        raw_kind.decode("utf-8", errors="replace") if raw_kind else "unknown"
    )
    build_info = _BUILD_INFO.get(library_path, {})
    return {
        "mode": mode,
        "implementation_kind": implementation_kind,
        "config": {
            "q_tile": q_tile.value,
            "k_tile": k_tile.value,
            "head_dim": head_dim.value,
            "block_threads": block_threads.value,
            "dynamic_smem_bytes": dynamic_smem.value,
        },
        "occupancy": {
            "active_blocks_per_multiprocessor": active_blocks.value,
        },
        "build": dict(build_info),
    }
