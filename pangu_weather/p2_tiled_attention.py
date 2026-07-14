"""Opt-in full-model adapter for the isolated gfx936 tiled EarthAttention.

This module is deliberately separate from the production model builder.  A
caller must explicitly patch a model with :func:`enable_p2_tiled_attention`;
all other model construction and inference paths remain unchanged.
"""

from __future__ import annotations

import types
import json
import os
from contextlib import contextmanager

import torch


_SUPPORTED_TOKENS = {32, 144}
_HEAD_DIM = 32
_KERNEL_MODES = {"online", "full-row-fast", "full-row-expf"}

_GLOBAL_REGION_IDS_CACHE = {}


def _backend():
    try:
        from .hip_earth_attention_tiled import (
            compact_earth_position_index,
            hip_earth_attention_tiled,
            pack_earth_bias_table,
            shifted_mask_to_region_ids,
        )
    except ImportError:
        from hip_earth_attention_tiled import (
            compact_earth_position_index,
            hip_earth_attention_tiled,
            pack_earth_bias_table,
            shifted_mask_to_region_ids,
        )
    return (
        compact_earth_position_index,
        hip_earth_attention_tiled,
        pack_earth_bias_table,
        shifted_mask_to_region_ids,
    )


def _module_is_supported(module):
    window_size = tuple(int(value) for value in module.window_size)
    heads = int(module.num_heads)
    dim = int(module.dim)
    return (
        window_size[0] * window_size[1] * window_size[2] in _SUPPORTED_TOKENS
        and dim == heads * _HEAD_DIM
        and heads in (3, 6)
        and hasattr(module, "qkv")
        and hasattr(module, "proj")
    )


def _cached_packed_bias_and_index(module, x):
    cached = getattr(module, "_pangu_p2_tiled_bias_index", None)
    if getattr(module, "_pangu_p2_tiled_bias_index_prepared", False):
        packed_bias, compacted_index = cached[1], cached[2]
        if packed_bias.device != x.device or compacted_index.device != x.device:
            raise RuntimeError("prepared P2 bias/index cache is on the wrong device")
        return packed_bias, compacted_index

    compact_index, _, pack_bias, _ = _backend()
    bias_table = module.earth_position_bias_table
    position_index = module.earth_position_index
    if bias_table.device != x.device:
        bias_table = bias_table.to(device=x.device)
    if bias_table.dtype != torch.float16:
        bias_table = bias_table.to(dtype=torch.float16)
    if position_index.device != x.device:
        position_index = position_index.to(device=x.device)

    bias_key = (
        bias_table.data_ptr(),
        int(getattr(bias_table, "_version", 0)),
        str(bias_table.device),
        tuple(bias_table.shape),
    )
    index_key = (
        position_index.data_ptr(),
        int(getattr(position_index, "_version", 0)),
        str(position_index.device),
        tuple(position_index.shape),
    )
    cache_key = (bias_key, index_key)
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]

    bias_rows = int(bias_table.shape[0])
    packed_bias = pack_bias(bias_table)
    compacted_index = compact_index(position_index, bias_rows=bias_rows)
    module._pangu_p2_tiled_bias_index = (
        cache_key,
        packed_bias,
        compacted_index,
    )
    return packed_bias, compacted_index


@torch.no_grad()
def _prepare_packed_bias_and_index(module, release_original_bias, retain_cpu_backup):
    """Materialize the immutable P2 cache before the first model forward."""

    bias_table = module.earth_position_bias_table
    if getattr(module, "_pangu_p2_tiled_bias_index_prepared", False):
        if release_original_bias:
            if not getattr(module, "_pangu_p2_original_bias_released", False):
                _release_original_bias(module, bias_table, retain_cpu_backup)
            if not getattr(module, "_pangu_p2_original_index_released", False):
                _release_original_index(module, module.earth_position_index, retain_cpu_backup)
        return

    compact_index, _, pack_bias, _ = _backend()
    position_index = module.earth_position_index
    if bias_table.dtype != torch.float16:
        bias_table = bias_table.to(dtype=torch.float16)
    if position_index.device != bias_table.device:
        position_index = position_index.to(device=bias_table.device)

    bias_rows = int(bias_table.shape[0])
    packed_bias = pack_bias(bias_table)
    compacted_index = compact_index(position_index, bias_rows=bias_rows)
    module._pangu_p2_tiled_bias_index = (
        ("prepared", str(bias_table.device), tuple(bias_table.shape)),
        packed_bias,
        compacted_index,
    )
    module._pangu_p2_tiled_bias_index_prepared = True

    if release_original_bias:
        _release_original_bias(module, bias_table, retain_cpu_backup)
        _release_original_index(module, position_index, retain_cpu_backup)


def _release_original_bias(module, bias_table, retain_cpu_backup):
    """Drop the original device Parameter after its packed cache exists."""

    module._pangu_p2_original_bias_device = bias_table.device
    module._pangu_p2_original_bias_dtype = bias_table.dtype
    module._pangu_p2_original_bias_requires_grad = bool(
        module.earth_position_bias_table.requires_grad
    )
    if retain_cpu_backup:
        module._pangu_p2_original_bias_table_cpu = bias_table.detach().to(
            device="cpu", copy=True
        )
    elif hasattr(module, "_pangu_p2_original_bias_table_cpu"):
        del module._pangu_p2_original_bias_table_cpu
    module.earth_position_bias_table = torch.nn.Parameter(
        torch.empty(0, dtype=bias_table.dtype, device="cpu"),
        requires_grad=False,
    )
    module._pangu_p2_original_bias_released = True


def _release_original_index(module, position_index, retain_cpu_backup):
    """Drop the original device Buffer after its packed cache exists."""

    module._pangu_p2_original_index_device = position_index.device
    module._pangu_p2_original_index_dtype = position_index.dtype
    if retain_cpu_backup:
        module._pangu_p2_original_index_cpu = position_index.detach().to(
            device="cpu", copy=True
        )
    elif hasattr(module, "_pangu_p2_original_index_cpu"):
        del module._pangu_p2_original_index_cpu
    module.register_buffer(
        "earth_position_index",
        torch.empty(0, dtype=position_index.dtype, device="cpu"),
        persistent=True,
    )
    module._pangu_p2_original_index_released = True


def _restore_original_bias(module, consume_backup=False):
    if not getattr(module, "_pangu_p2_original_bias_released", False):
        return
    backup = getattr(module, "_pangu_p2_original_bias_table_cpu", None)
    if backup is None:
        raise RuntimeError(
            "cannot restore released P2 bias without a retained CPU backup"
        )
    restored = backup.to(
        device=module._pangu_p2_original_bias_device,
        dtype=module._pangu_p2_original_bias_dtype,
        copy=True,
    )
    module.earth_position_bias_table = torch.nn.Parameter(
        restored,
        requires_grad=module._pangu_p2_original_bias_requires_grad,
    )
    module._pangu_p2_original_bias_released = False
    if consume_backup:
        del module._pangu_p2_original_bias_table_cpu


def _restore_original_index(module, consume_backup=False):
    if not getattr(module, "_pangu_p2_original_index_released", False):
        return
    backup = getattr(module, "_pangu_p2_original_index_cpu", None)
    if backup is None:
        raise RuntimeError(
            "cannot restore released P2 index without a retained CPU backup"
        )
    restored = backup.to(
        device=module._pangu_p2_original_index_device,
        dtype=module._pangu_p2_original_index_dtype,
        copy=True,
    )
    module.register_buffer("earth_position_index", restored, persistent=True)
    module._pangu_p2_original_index_released = False
    if consume_backup:
        del module._pangu_p2_original_index_cpu


@contextmanager
def _temporary_original_bias(module):
    released_bias = getattr(module, "_pangu_p2_original_bias_released", False)
    released_index = getattr(module, "_pangu_p2_original_index_released", False)
    if released_bias:
        _restore_original_bias(module)
    if released_index:
        _restore_original_index(module)
    try:
        yield
    finally:
        if released_bias:
            original = module.earth_position_bias_table
            module.earth_position_bias_table = torch.nn.Parameter(
                torch.empty(0, dtype=original.dtype, device="cpu"),
                requires_grad=False,
            )
            module._pangu_p2_original_bias_released = True
        if released_index:
            original = module.earth_position_index
            module.register_buffer(
                "earth_position_index",
                torch.empty(0, dtype=original.dtype, device="cpu"),
                persistent=True,
            )
            module._pangu_p2_original_index_released = True


def _cached_region_ids(module, mask, x):
    if mask is None:
        return None
    original_mask = mask
    cache_key = (
        original_mask.data_ptr(),
        int(getattr(original_mask, "_version", 0)),
        str(original_mask.device),
        tuple(original_mask.shape),
        str(original_mask.dtype),
    )
    if cache_key in _GLOBAL_REGION_IDS_CACHE:
        return _GLOBAL_REGION_IDS_CACHE[cache_key]
    if mask.device != x.device:
        mask = mask.to(device=x.device)
    if not torch.is_floating_point(mask):
        mask = mask.to(dtype=torch.float16)
    _, _, _, mask_to_regions = _backend()
    region_ids = mask_to_regions(mask)
    _GLOBAL_REGION_IDS_CACHE[cache_key] = region_ids
    return region_ids


def _capture_reference_projection_input(projection, original_forward, x, mask):
    projection_inputs = []

    def capture_pre_projection(_module, inputs):
        projection_inputs.append(inputs[0])

    handle = projection.register_forward_pre_hook(capture_pre_projection)
    try:
        reference = original_forward(x, mask)
    finally:
        handle.remove()
    if not projection_inputs:
        raise RuntimeError("P2 module audit did not capture projection input")
    return torch.cat(projection_inputs, dim=0), reference


def _run_p2_tiled_chunks(
    self,
    x,
    *,
    packed_bias,
    position_index,
    region_ids,
    mask_width,
    tiled_forward,
    capture_attention_output,
):
    batch_windows, pressure_height, tokens, channels = x.shape
    configured_chunk_size = int(
        getattr(self, "_pangu_attention_chunk_size", 3)
    )
    chunked_qkv = bool(getattr(self, "_pangu_chunked_qkv", False))
    chunked_proj = bool(getattr(self, "_pangu_chunked_proj", False))
    if bool(getattr(self, "_pangu_p2_full_width", False)):
        chunked_qkv = False
        chunked_proj = False
    if configured_chunk_size <= 0 or not (chunked_qkv or chunked_proj):
        chunk_size = batch_windows
    else:
        chunk_size = max(1, configured_chunk_size)

    qkv = None
    if not chunked_qkv:
        qkv = self.qkv(x).reshape(
            batch_windows,
            pressure_height,
            tokens,
            3,
            int(self.num_heads),
            _HEAD_DIM,
        ).contiguous()

    attention_output = None
    projected = None
    if chunked_proj:
        projected = x.new_empty(
            batch_windows,
            pressure_height,
            tokens,
            channels,
        )
        if capture_attention_output:
            attention_output = x.new_empty(
                batch_windows,
                pressure_height,
                tokens,
                channels,
            )
    else:
        attention_output = x.new_empty(
            batch_windows,
            pressure_height,
            tokens,
            channels,
        )

    for start in range(0, batch_windows, chunk_size):
        end = min(start + chunk_size, batch_windows)
        if chunked_qkv:
            qkv_chunk = self.qkv(x[start:end]).reshape(
                end - start,
                pressure_height,
                tokens,
                3,
                int(self.num_heads),
                _HEAD_DIM,
            ).contiguous()
        else:
            qkv_chunk = qkv[start:end]
        attention_chunk = tiled_forward(
            qkv_chunk,
            packed_bias,
            position_index,
            region_ids,
            scale=self.scale,
            mask_width=mask_width,
            width_offset=start if region_ids is not None else 0,
            mode=self._pangu_p2_tiled_kernel_mode,
        )
        if capture_attention_output and chunked_proj:
            attention_output[start:end].copy_(attention_chunk)
        if chunked_proj:
            projected[start:end].copy_(self.proj(attention_chunk))
        else:
            attention_output[start:end].copy_(attention_chunk)
        del qkv_chunk, attention_chunk

    if chunked_proj:
        result = self.proj_drop(projected)
    else:
        result = self.proj_drop(self.proj(attention_output))
    return result, attention_output if capture_attention_output else None


@torch.no_grad()
def _forward_p2_tiled(self, x, mask=None):
    original_forward = self._pangu_p2_original_forward
    if not _module_is_supported(self):
        if self._pangu_p2_tiled_strict:
            raise RuntimeError("P2 tiled attention received an unsupported module")
        return original_forward(x, mask)
    if self.training:
        raise RuntimeError("P2 tiled attention is inference-only")
    if x.ndim != 4 or x.dtype != torch.float16 or x.device.type != "cuda":
        if self._pangu_p2_tiled_strict:
            raise RuntimeError("P2 tiled attention requires contiguous-device FP16 input")
        return original_forward(x, mask)
    if not x.is_contiguous():
        x = x.contiguous()

    _, _, tokens, _ = x.shape
    expected_tokens = 1
    for value in self.window_size:
        expected_tokens *= int(value)
    if tokens != expected_tokens or tokens not in _SUPPORTED_TOKENS:
        if self._pangu_p2_tiled_strict:
            raise RuntimeError(f"unsupported P2 token length: {tokens}")
        return original_forward(x, mask)

    packed_bias, position_index = _cached_packed_bias_and_index(self, x)
    region_ids = _cached_region_ids(self, mask, x)
    mask_width = None if region_ids is None else int(region_ids.shape[0])
    _, tiled_forward, _, _ = _backend()
    audit_enabled = os.environ.get("PANGU_P2_TILED_DEBUG", "0").lower() not in {
        "0", "false", "no", "off",
    }
    result, attention_output = _run_p2_tiled_chunks(
        self,
        x,
        packed_bias=packed_bias,
        position_index=position_index,
        region_ids=region_ids,
        mask_width=mask_width,
        tiled_forward=tiled_forward,
        capture_attention_output=audit_enabled,
    )
    if audit_enabled:
        with _temporary_original_bias(self):
            reference_pre_projection, reference = (
                _capture_reference_projection_input(
                    self.proj,
                    original_forward,
                    x,
                    mask,
                )
            )
        if reference_pre_projection.shape != attention_output.shape:
            raise RuntimeError(
                "P2 module audit projection shape mismatch: "
                f"candidate={tuple(attention_output.shape)} "
                f"reference={tuple(reference_pre_projection.shape)}"
            )

        def metrics(candidate, expected):
            delta = (candidate.float() - expected.float()).abs()
            return {
                "exact": bool(torch.equal(candidate, expected)),
                "mismatch_count": int((candidate != expected).sum().item()),
                "elements": candidate.numel(),
                "max_abs": float(delta.max().item()),
                "mean_abs": float(delta.mean().item()),
            }

        audit = {
            "module": getattr(self, "_pangu_p2_module_name", "?"),
            "mode": self._pangu_p2_tiled_kernel_mode,
            "shape": list(x.shape),
            "pre_projection": metrics(
                attention_output,
                reference_pre_projection,
            ),
            "post_projection": metrics(result, reference),
        }
        self._pangu_p2_last_audit = audit
        print("[P2_AUDIT] " + json.dumps(audit, sort_keys=True))
    return result


def enable_p2_tiled_attention(
    model,
    strict=True,
    kernel_mode="online",
    force_full_width=False,
    release_original_bias=False,
    retain_cpu_backup=False,
):
    """Patch exact-profile EarthAttention modules for an explicit A/B run.

    The function does not alter production defaults.  It returns the number
    of patched attention modules.  Rollback after bias release requires an
    explicit retained CPU backup.
    """

    if kernel_mode not in _KERNEL_MODES:
        supported = ", ".join(sorted(_KERNEL_MODES))
        raise ValueError(f"P2 kernel_mode must be one of: {supported}")
    if release_original_bias and not retain_cpu_backup:
        debug_enabled = os.environ.get("PANGU_P2_TILED_DEBUG", "0").lower() not in {
            "0", "false", "no", "off",
        }
        if debug_enabled:
            raise ValueError(
                "P2 debug with released bias requires retain_cpu_backup=True"
            )

    patched = 0
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "EarthAttention3D":
            continue
        if not _module_is_supported(module):
            if strict:
                raise RuntimeError(
                    f"unsupported EarthAttention3D module for P2: {module_name}"
                )
            continue
        already_released = getattr(
            module, "_pangu_p2_original_bias_released", False
        )
        if already_released and not release_original_bias:
            raise RuntimeError(
                "cannot disable P2 bias release by re-enabling an already "
                "released module"
            )
        if (
            already_released
            and retain_cpu_backup
            and not hasattr(module, "_pangu_p2_original_bias_table_cpu")
        ):
            raise RuntimeError(
                "cannot add a CPU bias backup after the original bias was released"
            )
        if not hasattr(module, "_pangu_p2_original_forward"):
            module._pangu_p2_original_forward = module.forward
        module._pangu_p2_module_name = module_name
        module._pangu_p2_tiled_strict = bool(strict)
        module._pangu_p2_tiled_kernel_mode = kernel_mode
        module._pangu_p2_full_width = bool(force_full_width)
        _prepare_packed_bias_and_index(
            module,
            release_original_bias=bool(release_original_bias),
            retain_cpu_backup=bool(retain_cpu_backup),
        )
        module.forward = types.MethodType(_forward_p2_tiled, module)
        patched += 1
    if patched == 0:
        raise RuntimeError("no exact-profile EarthAttention3D module was patched")
    model._pangu_p2_tiled_attention_count = patched
    model._pangu_p2_tiled_kernel_mode = kernel_mode
    model._pangu_p2_full_width = bool(force_full_width)
    model._pangu_p2_original_bias_released = bool(release_original_bias)
    return patched


def disable_p2_tiled_attention(model):
    """Restore every attention forward previously patched by this module."""

    global _GLOBAL_REGION_IDS_CACHE
    _GLOBAL_REGION_IDS_CACHE.clear()

    restored = 0
    for module in model.modules():
        original = getattr(module, "_pangu_p2_original_forward", None)
        if original is None:
            continue
        _restore_original_bias(module, consume_backup=True)
        _restore_original_index(module, consume_backup=True)
        module.forward = original
        for attribute in (
            "_pangu_p2_tiled_bias_index",
            "_pangu_p2_tiled_bias_index_prepared",
            "_pangu_p2_tiled_region_ids",
            "_pangu_p2_original_bias_device",
            "_pangu_p2_original_bias_dtype",
            "_pangu_p2_original_bias_requires_grad",
            "_pangu_p2_original_bias_released",
            "_pangu_p2_original_index_device",
            "_pangu_p2_original_index_dtype",
            "_pangu_p2_original_index_released",
        ):
            if hasattr(module, attribute):
                delattr(module, attribute)
        restored += 1
    model._pangu_p2_tiled_attention_count = 0
    model._pangu_p2_original_bias_released = False
    return restored
