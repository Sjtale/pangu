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
_EXPECTED_REGION_SETUP_ATTENTIONS = 16
_EXPECTED_SHIFTED_MASK_OWNERS = 8
_BLOCKED_MASK_VALUE = -100.0

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


def _mask_cache_key(mask):
    return (
        mask.data_ptr(),
        int(getattr(mask, "_version", 0)),
        str(mask.device),
        tuple(mask.shape),
        str(mask.dtype),
    )


def _storage_summary(tensors):
    logical_bytes = 0
    unique_storages = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        logical_bytes += tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        unique_storages.setdefault(storage_key, storage.nbytes())
    return {
        "logical_bytes": int(logical_bytes),
        "unique_storage_bytes": int(sum(unique_storages.values())),
        "unique_storage_count": len(unique_storages),
    }


def _snapshot_module_state(module):
    state = dict(module.__dict__)
    for attribute in (
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
    ):
        value = state.get(attribute)
        if value is not None:
            state[attribute] = value.copy()
    return state


def _restore_module_state(module, state):
    module.__dict__.clear()
    module.__dict__.update(state)


def _cuda_memory_allocated():
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.memory_allocated())


def _unwrap_block_attention(block):
    attention = getattr(block, "attn", None)
    if attention is None:
        return None
    return getattr(attention, "attentioner", attention)


def _region_ids_match_mask(region_ids, mask):
    if mask.ndim == 4:
        expected_shape = (mask.shape[0], mask.shape[1], mask.shape[2])
    elif mask.ndim == 5 and mask.shape[2] == 1:
        expected_shape = (mask.shape[0], mask.shape[1], mask.shape[3])
    else:
        return False
    return (
        isinstance(region_ids, torch.Tensor)
        and tuple(region_ids.shape) == tuple(expected_shape)
        and region_ids.dtype == torch.uint8
        and region_ids.device == mask.device
        and region_ids.is_contiguous()
    )


def _validate_region_ids_reconstruction(region_ids, mask, owner_name):
    if not _region_ids_match_mask(region_ids, mask):
        raise RuntimeError(f"invalid prepared region IDs for P2 owner: {owner_name}")
    if not torch.is_floating_point(mask):
        raise RuntimeError(f"P2 dense mask is not floating point: {owner_name}")

    dense_mask = mask[:, :, 0] if mask.ndim == 5 else mask
    sequence_length = dense_mask.shape[-1]
    flat_masks = dense_mask.reshape(-1, sequence_length, sequence_length)
    flat_regions = region_ids.reshape(-1, sequence_length)
    if flat_masks.shape[0] != flat_regions.shape[0]:
        raise RuntimeError(f"P2 region matrix count mismatch: {owner_name}")

    for matrix_index in range(flat_masks.shape[0]):
        matrix = flat_masks[matrix_index]
        zero_relation = matrix == 0
        valid_values = zero_relation | (matrix == _BLOCKED_MASK_VALUE)
        if not bool(valid_values.all().item()):
            raise RuntimeError(
                f"P2 dense mask has invalid values: {owner_name} "
                f"matrix={matrix_index}"
            )
        labels = flat_regions[matrix_index]
        reconstructed = labels[:, None] == labels[None, :]
        if not torch.equal(reconstructed, zero_relation):
            raise RuntimeError(
                f"P2 region IDs do not reconstruct dense mask: {owner_name} "
                f"matrix={matrix_index}"
            )


@torch.no_grad()
def _prepare_region_release_plan(model, retain_cpu_mask_backup):
    attention_modules = [
        (module_name, module)
        for module_name, module in model.named_modules()
        if module.__class__.__name__ == "EarthAttention3D"
    ]
    owners = {}
    for owner_name, owner in model.named_modules():
        if owner.__class__.__name__ != "EarthTransformer3DBlock":
            continue
        attention = _unwrap_block_attention(owner)
        if attention is None:
            raise RuntimeError(f"P2 region setup owner has no attention: {owner_name}")
        attention_id = id(attention)
        if attention_id in owners:
            raise RuntimeError("P2 region setup found duplicate attention owners")
        owners[attention_id] = (owner_name, owner)

    if len(attention_modules) != _EXPECTED_REGION_SETUP_ATTENTIONS:
        raise RuntimeError(
            "P2 region setup requires exactly "
            f"{_EXPECTED_REGION_SETUP_ATTENTIONS} EarthAttention3D modules, "
            f"got {len(attention_modules)}"
        )
    if len(owners) != _EXPECTED_REGION_SETUP_ATTENTIONS:
        raise RuntimeError(
            "P2 region setup requires exactly "
            f"{_EXPECTED_REGION_SETUP_ATTENTIONS} EarthTransformer3DBlock owners, "
            f"got {len(owners)}"
        )

    attention_ids = {id(module) for _, module in attention_modules}
    if attention_ids != set(owners):
        raise RuntimeError("P2 region setup attention/owner mapping is not one-to-one")
    if any(not _module_is_supported(module) for _, module in attention_modules):
        raise RuntimeError("P2 region setup found an unsupported EarthAttention3D module")
    if any(
        getattr(module, "_pangu_p2_region_ids_prepared", False)
        for _, module in attention_modules
    ):
        raise RuntimeError("P2 region IDs are already prepared")

    _, _, _, mask_to_regions = _backend()
    canonical_masks = []
    plans = []
    shifted_count = 0
    for module_name, module in attention_modules:
        owner_name, owner = owners[id(module)]
        mask = owner._buffers.get("attn_mask")
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise TypeError(f"P2 owner mask is not a tensor: {owner_name}")

        entry = {
            "module_name": module_name,
            "module": module,
            "owner_name": owner_name,
            "owner": owner,
            "mask": mask,
            "mask_key": None,
            "region_ids": None,
            "persistent": "attn_mask" not in owner._non_persistent_buffers_set,
            "cpu_backup": None,
        }
        if mask is not None:
            shifted_count += 1
            mask_key = _mask_cache_key(mask)
            region_ids = None
            for canonical_mask, canonical_region_ids in canonical_masks:
                same_metadata = (
                    canonical_mask.shape == mask.shape
                    and canonical_mask.dtype == mask.dtype
                    and canonical_mask.device == mask.device
                )
                if same_metadata and (
                    canonical_mask.data_ptr() == mask.data_ptr()
                    or torch.equal(canonical_mask, mask)
                ):
                    region_ids = canonical_region_ids
                    break
            if region_ids is None:
                region_ids = mask_to_regions(mask)
                canonical_masks.append((mask, region_ids))
            _validate_region_ids_reconstruction(region_ids, mask, owner_name)
            entry["mask_key"] = mask_key
            entry["region_ids"] = region_ids
        plans.append(entry)

    if shifted_count != _EXPECTED_SHIFTED_MASK_OWNERS:
        raise RuntimeError(
            "P2 region setup requires exactly "
            f"{_EXPECTED_SHIFTED_MASK_OWNERS} shifted-mask owners, "
            f"got {shifted_count}"
        )

    if retain_cpu_mask_backup:
        backups = {}
        for entry in plans:
            mask = entry["mask"]
            if mask is None:
                continue
            storage = mask.untyped_storage()
            storage_key = (str(mask.device), storage.data_ptr(), storage.nbytes())
            if storage_key not in backups:
                backups[storage_key] = mask.detach().to(device="cpu", copy=True)
            entry["cpu_backup"] = backups[storage_key]

    dense_summary = _storage_summary(
        entry["mask"] for entry in plans if entry["mask"] is not None
    )
    region_summary = _storage_summary(
        entry["region_ids"]
        for entry in plans
        if entry["region_ids"] is not None
    )
    return plans, dense_summary, region_summary


def _clear_prepared_region_attributes(module):
    for attribute in (
        "_pangu_p2_tiled_region_ids",
        "_pangu_p2_region_ids_prepared",
        "_pangu_p2_region_ids_expected",
        "_pangu_p2_original_mask_released",
    ):
        if hasattr(module, attribute):
            delattr(module, attribute)


def _clear_owner_mask_release_attributes(owner):
    for attribute in (
        "_pangu_p2_original_attn_mask_cpu",
        "_pangu_p2_original_attn_mask_device",
        "_pangu_p2_original_attn_mask_dtype",
        "_pangu_p2_original_attn_mask_persistent",
        "_pangu_p2_original_attn_mask_released",
    ):
        if hasattr(owner, attribute):
            delattr(owner, attribute)


def _validate_committed_region_setup(plans, release_original_masks):
    if len(plans) != _EXPECTED_REGION_SETUP_ATTENTIONS:
        raise RuntimeError("P2 committed region setup lost attention modules")
    shifted = [entry for entry in plans if entry["region_ids"] is not None]
    if len(shifted) != _EXPECTED_SHIFTED_MASK_OWNERS:
        raise RuntimeError("P2 committed region setup lost shifted-mask owners")

    released_count = 0
    for entry in plans:
        module = entry["module"]
        region_ids = entry["region_ids"]
        cached = getattr(module, "_pangu_p2_tiled_region_ids", None)
        if not getattr(module, "_pangu_p2_region_ids_prepared", False):
            raise RuntimeError("P2 committed region setup is missing its marker")
        if not isinstance(cached, tuple) or len(cached) != 2:
            raise RuntimeError("P2 committed region setup is missing its cache")
        if region_ids is None:
            if cached != (None, None):
                raise RuntimeError("unshifted P2 module retained region IDs")
            continue
        if (
            cached[1] is not region_ids
            or region_ids.dtype != torch.uint8
            or not region_ids.is_contiguous()
            or region_ids.device != entry["mask"].device
            or (torch.cuda.is_available() and region_ids.device.type != "cuda")
        ):
            raise RuntimeError("prepared shifted P2 region IDs failed layout checks")
        if release_original_masks:
            if entry["owner"]._buffers.get("attn_mask") is not None:
                raise RuntimeError("P2 dense attention mask release did not commit")
            if not getattr(module, "_pangu_p2_original_mask_released", False):
                raise RuntimeError("P2 attention release marker is missing")
            released_count += 1

    expected_released = _EXPECTED_SHIFTED_MASK_OWNERS if release_original_masks else 0
    if released_count != expected_released:
        raise RuntimeError(
            f"P2 committed {released_count} dense-mask releases; "
            f"expected {expected_released}"
        )


def _commit_region_release_plan(
    plans,
    *,
    release_original_masks,
    retain_cpu_mask_backup,
):
    released = []
    try:
        for entry in plans:
            module = entry["module"]
            region_ids = entry["region_ids"]
            module._pangu_p2_tiled_region_ids = (
                entry["mask_key"],
                region_ids,
            )
            module._pangu_p2_region_ids_expected = region_ids is not None
            module._pangu_p2_original_mask_released = False
            module._pangu_p2_region_ids_prepared = True

        if release_original_masks:
            for entry in plans:
                mask = entry["mask"]
                if mask is None:
                    continue
                owner = entry["owner"]
                owner._pangu_p2_original_attn_mask_device = mask.device
                owner._pangu_p2_original_attn_mask_dtype = mask.dtype
                owner._pangu_p2_original_attn_mask_persistent = entry["persistent"]
                if retain_cpu_mask_backup:
                    owner._pangu_p2_original_attn_mask_cpu = entry["cpu_backup"]
                owner.register_buffer(
                    "attn_mask",
                    None,
                    persistent=entry["persistent"],
                )
                owner._pangu_p2_original_attn_mask_released = True
                entry["module"]._pangu_p2_original_mask_released = True
                released.append(entry)
        _validate_committed_region_setup(plans, release_original_masks)
    except Exception:
        for entry in released:
            entry["owner"].register_buffer(
                "attn_mask",
                entry["mask"],
                persistent=entry["persistent"],
            )
        for entry in plans:
            _clear_prepared_region_attributes(entry["module"])
            _clear_owner_mask_release_attributes(entry["owner"])
        raise

    for entry in plans:
        if entry["region_ids"] is not None:
            _GLOBAL_REGION_IDS_CACHE[entry["mask_key"]] = entry["region_ids"]
        entry["mask"] = None


def _restore_original_attention_mask(owner, consume_backup=False):
    if not getattr(owner, "_pangu_p2_original_attn_mask_released", False):
        return
    backup = getattr(owner, "_pangu_p2_original_attn_mask_cpu", None)
    if backup is None:
        raise RuntimeError(
            "cannot restore released P2 attention mask without a retained CPU backup"
        )
    restored = backup.to(
        device=owner._pangu_p2_original_attn_mask_device,
        dtype=owner._pangu_p2_original_attn_mask_dtype,
        copy=True,
    )
    owner.register_buffer(
        "attn_mask",
        restored,
        persistent=owner._pangu_p2_original_attn_mask_persistent,
    )
    owner._pangu_p2_original_attn_mask_released = False
    if consume_backup:
        _clear_owner_mask_release_attributes(owner)


def _cached_region_ids(module, mask, x):
    prepared = getattr(module, "_pangu_p2_region_ids_prepared", False)
    released = getattr(module, "_pangu_p2_original_mask_released", False)
    if released and not prepared:
        raise RuntimeError("released P2 mask has no prepared region-ID marker")
    if prepared:
        cached = getattr(module, "_pangu_p2_tiled_region_ids", None)
        expected = bool(getattr(module, "_pangu_p2_region_ids_expected", False))
        if not isinstance(cached, tuple) or len(cached) != 2:
            raise RuntimeError("prepared P2 region-ID cache is missing")
        cache_key, region_ids = cached
        if not expected:
            if region_ids is not None or mask is not None:
                raise RuntimeError("unshifted P2 attention received a mask or region IDs")
            return None
        if not isinstance(region_ids, torch.Tensor):
            raise RuntimeError("prepared shifted P2 attention has no region IDs")
        if mask is None:
            if not released:
                raise RuntimeError("prepared P2 attention unexpectedly lost its dense mask")
        elif _mask_cache_key(mask) != cache_key:
            raise RuntimeError("prepared P2 attention received a different dense mask")
        if (
            region_ids.dtype != torch.uint8
            or not region_ids.is_contiguous()
            or region_ids.device != x.device
        ):
            raise RuntimeError("prepared P2 region IDs have an invalid runtime layout")
        return region_ids
    if mask is None:
        return None
    original_mask = mask
    cache_key = _mask_cache_key(original_mask)
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
    precompute_region_ids=False,
    release_original_masks=False,
    retain_cpu_mask_backup=False,
):
    """Patch exact-profile EarthAttention modules for an explicit A/B run.

    The function does not alter production defaults.  It returns the number
    of patched attention modules.  Rollback after bias release requires an
    explicit retained CPU backup.  Region precomputation is fail-closed for
    the submission's exact 16-attention/8-shifted-mask topology; dense masks
    can be released only after every region-ID tensor has been validated.
    """

    if kernel_mode not in _KERNEL_MODES:
        supported = ", ".join(sorted(_KERNEL_MODES))
        raise ValueError(f"P2 kernel_mode must be one of: {supported}")
    if release_original_masks and not precompute_region_ids:
        raise ValueError(
            "release_original_masks=True requires precompute_region_ids=True"
        )
    if retain_cpu_mask_backup and not release_original_masks:
        raise ValueError(
            "retain_cpu_mask_backup=True requires release_original_masks=True"
        )
    debug_enabled = os.environ.get("PANGU_P2_TILED_DEBUG", "0").lower() not in {
        "0", "false", "no", "off",
    }
    if release_original_masks and debug_enabled:
        raise ValueError("P2 debug is incompatible with released dense attention masks")
    if release_original_bias and not retain_cpu_backup:
        if debug_enabled:
            raise ValueError(
                "P2 debug with released bias requires retain_cpu_backup=True"
            )

    region_plan = None
    dense_summary = None
    region_summary = None
    allocated_before = None
    if precompute_region_ids:
        region_plan, dense_summary, region_summary = _prepare_region_release_plan(
            model,
            retain_cpu_mask_backup=bool(retain_cpu_mask_backup),
        )

    attention_modules = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "EarthAttention3D":
            continue
        if not _module_is_supported(module):
            if strict:
                raise RuntimeError(
                    f"unsupported EarthAttention3D module for P2: {module_name}"
                )
            continue
        attention_modules.append((module_name, module))

    if not attention_modules:
        raise RuntimeError("no exact-profile EarthAttention3D module was patched")
    if (
        region_plan is not None
        and len(attention_modules) != _EXPECTED_REGION_SETUP_ATTENTIONS
    ):
        raise RuntimeError(
            "P2 region setup patched-count drift: "
            f"expected {_EXPECTED_REGION_SETUP_ATTENTIONS}, "
            f"got {len(attention_modules)}"
        )

    released_auxiliary_summary = _storage_summary(
        tensor
        for _, module in attention_modules
        for tensor in (
            getattr(module, "earth_position_bias_table", None),
            getattr(module, "earth_position_index", None),
        )
        if release_original_bias
        and isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cuda"
    )

    for _, module in attention_modules:
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

    transaction_modules = [model]
    transaction_modules.extend(module for _, module in attention_modules)
    if region_plan is not None:
        transaction_modules.extend(entry["owner"] for entry in region_plan)
    deduplicated_modules = []
    seen_module_ids = set()
    for module in transaction_modules:
        if id(module) in seen_module_ids:
            continue
        seen_module_ids.add(id(module))
        deduplicated_modules.append(module)
    transaction_states = [
        (module, _snapshot_module_state(module))
        for module in deduplicated_modules
    ]
    region_cache_before = dict(_GLOBAL_REGION_IDS_CACHE)

    patched = 0
    report = None
    try:
        for module_name, module in attention_modules:
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

        if region_plan is not None:
            # Measure immediately around the dense-mask release. Region IDs
            # and the existing bias/index preparation are already resident,
            # so this delta cannot be contaminated by those one-time changes.
            allocated_before = _cuda_memory_allocated()
            _commit_region_release_plan(
                region_plan,
                release_original_masks=bool(release_original_masks),
                retain_cpu_mask_backup=bool(retain_cpu_mask_backup),
            )
            dense_after_summary = _storage_summary(
                entry["owner"]._buffers.get("attn_mask")
                for entry in region_plan
            )
            theoretical_dense_reclaimed = (
                dense_summary["unique_storage_bytes"]
                - dense_after_summary["unique_storage_bytes"]
            )
            theoretical_net_reclaimed = (
                theoretical_dense_reclaimed
                - region_summary["unique_storage_bytes"]
            )
            report = {
                "precomputed_region_ids": True,
                "released_original_masks": bool(release_original_masks),
                "attention_modules": patched,
                "shifted_mask_owners": _EXPECTED_SHIFTED_MASK_OWNERS,
                "dense_mask_logical_bytes_before": dense_summary["logical_bytes"],
                "dense_mask_unique_bytes_before": dense_summary[
                    "unique_storage_bytes"
                ],
                "dense_mask_unique_storage_count_before": dense_summary[
                    "unique_storage_count"
                ],
                "dense_mask_logical_bytes_after": dense_after_summary[
                    "logical_bytes"
                ],
                "dense_mask_unique_bytes_after": dense_after_summary[
                    "unique_storage_bytes"
                ],
                "dense_mask_unique_storage_count_after": dense_after_summary[
                    "unique_storage_count"
                ],
                "region_ids_logical_bytes": region_summary["logical_bytes"],
                "region_ids_unique_bytes": region_summary[
                    "unique_storage_bytes"
                ],
                "region_ids_unique_storage_count": region_summary[
                    "unique_storage_count"
                ],
                "theoretical_dense_mask_reclaimed_bytes": (
                    theoretical_dense_reclaimed
                ),
                "theoretical_net_reclaimed_bytes": theoretical_net_reclaimed,
                "actual_cuda_allocated_reclaimed_bytes": None,
                "actual_cuda_dense_mask_reclaimed_bytes": None,
                "actual_cuda_allocated_reclaimed_total_bytes": None,
                "transaction_auxiliary_release_unique_bytes": (
                    released_auxiliary_summary["unique_storage_bytes"]
                ),
                "net_unique_bytes_reclaimed": theoretical_net_reclaimed,
                "retained_cpu_mask_backup": bool(retain_cpu_mask_backup),
                "cuda_memory_allocated_before": allocated_before,
                "cuda_memory_allocated_after": None,
                "cuda_memory_allocated_delta": None,
            }
            model._pangu_p2_region_setup_report = report

        model._pangu_p2_tiled_attention_count = patched
        model._pangu_p2_tiled_kernel_mode = kernel_mode
        model._pangu_p2_full_width = bool(force_full_width)
        model._pangu_p2_original_bias_released = bool(release_original_bias)
    except Exception:
        for module, state in reversed(transaction_states):
            _restore_module_state(module, state)
        _GLOBAL_REGION_IDS_CACHE.clear()
        _GLOBAL_REGION_IDS_CACHE.update(region_cache_before)
        raise

    # Drop rollback references before observing allocator state. The setup is
    # fully committed at this point, so these tensors must no longer keep the
    # released masks or bias/index buffers alive.
    transaction_states.clear()
    if report is not None:
        allocated_after = _cuda_memory_allocated()
        actual_total_reclaimed = (
            allocated_before - allocated_after
            if allocated_before is not None and allocated_after is not None
            else None
        )
        actual_region_reclaimed = (
            actual_total_reclaimed
            - released_auxiliary_summary["unique_storage_bytes"]
            if actual_total_reclaimed is not None
            else None
        )
        report["actual_cuda_allocated_reclaimed_bytes"] = actual_region_reclaimed
        report["actual_cuda_dense_mask_reclaimed_bytes"] = actual_region_reclaimed
        report["actual_cuda_allocated_reclaimed_total_bytes"] = (
            actual_total_reclaimed
        )
        report["cuda_memory_allocated_after"] = allocated_after
        report["cuda_memory_allocated_delta"] = (
            -actual_total_reclaimed
            if actual_total_reclaimed is not None
            else None
        )
        print("[P2_REGION_SETUP] " + json.dumps(report, sort_keys=True))
    return patched


def disable_p2_tiled_attention(model):
    """Restore every attention forward previously patched by this module."""

    global _GLOBAL_REGION_IDS_CACHE
    _GLOBAL_REGION_IDS_CACHE.clear()

    for module in model.modules():
        _restore_original_attention_mask(module, consume_backup=True)

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
            "_pangu_p2_region_ids_prepared",
            "_pangu_p2_region_ids_expected",
            "_pangu_p2_original_mask_released",
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
    if hasattr(model, "_pangu_p2_region_setup_report"):
        del model._pangu_p2_region_setup_report
    return restored
