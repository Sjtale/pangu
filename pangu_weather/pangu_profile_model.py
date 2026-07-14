"""Profile-aware Pangu model construction for submission-local variants.

The competition package only ships ``pangu_weather/`` while OneScience remains
the platform-provided dependency. Keep architecture adaptations that are needed
by profile checkpoints here instead of relying on local OneScience edits.
"""

import types
import os
import math
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.models.pangu import Pangu
from onescience.modules import OneRecovery, OneFuser
from onescience.modules.attention.earthattention3d import EarthAttention3D


SELECTIVE_MLP_96_PARAMETER_COUNT = 14_768_265
SELECTIVE_MLP_96_MLP_RATIO_BLOCKS = (
    (4, 4),
    (4, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
    (4, 4),
)


def selective_mlp_96_profile():
    """Return an independent copy of the exact SelectiveMLP-96 profile."""

    return {
        "name": "selective_mlp96",
        "patch_size": [2, 8, 8],
        "embed_dim": 96,
        "num_heads": [3, 6, 6, 3],
        "window_size": [2, 6, 12],
        "depth_blocks": [2, 6, 6, 2],
        "mlp_ratio_blocks": [
            list(stage) for stage in SELECTIVE_MLP_96_MLP_RATIO_BLOCKS
        ],
    }


def _is_enabled(name, default=False):
    default_value = "1" if default else "0"
    return os.environ.get(name, default_value).lower() not in {"0", "false", "no"}


def _env_share_deep_blocks():
    raw_value = os.environ.get("PANGU_SHARE_DEEP_BLOCKS", "0").strip().lower()
    if raw_value in {"0", "false", "no", "off", ""}:
        return None
    if raw_value in {"1", "true", "yes", "on"}:
        return "layer2_to_layer3"
    return raw_value


def _env_int(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _module_stage(module_name):
    for stage in ("layer1", "layer2", "layer3", "layer4"):
        if module_name == stage or module_name.startswith(stage + "."):
            return stage
    return None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GQAEarthAttention3D(nn.Module):
    def __init__(self, original_attn, kv_group_size=2):
        super().__init__()
        self.dim = original_attn.dim
        self.window_size = original_attn.window_size
        self.num_heads = original_attn.num_heads
        self.kv_group_size = kv_group_size
        self.num_kv_heads = self.num_heads // kv_group_size

        head_dim = self.dim // self.num_heads
        self.scale = original_attn.scale
        self.num_pressure_height_windows = original_attn.num_pressure_height_windows

        self.earth_position_bias_table = original_attn.earth_position_bias_table
        self.register_buffer("earth_position_index", original_attn.earth_position_index)

        qkv_bias = original_attn.qkv.bias is not None
        self.q_proj = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        kv_dim = self.num_kv_heads * head_dim
        self.kv_proj = nn.Linear(self.dim, kv_dim * 2, bias=qkv_bias)

        self.attn_drop = original_attn.attn_drop
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop
        self.softmax = original_attn.softmax

    def forward(self, x, mask=None):
        BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens, Channels = x.shape
        head_dim = Channels // self.num_heads

        q = self.q_proj(x).reshape(
            BatchTimesWidthWindows,
            NumPressureHeightWindows,
            WindowTokens,
            self.num_heads,
            head_dim
        ).permute(0, 3, 1, 2, 4)

        kv = self.kv_proj(x).reshape(
            BatchTimesWidthWindows,
            NumPressureHeightWindows,
            WindowTokens,
            2,
            self.num_kv_heads,
            head_dim
        ).permute(3, 0, 4, 1, 2, 5)
        k, v = kv[0], kv[1]

        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)
            if k.shape[1] < self.num_heads:
                diff = self.num_heads - k.shape[1]
                last_k = k[:, -1:, :, :, :]
                last_v = v[:, -1:, :, :, :]
                k = torch.cat([k, last_k.repeat(1, diff, 1, 1, 1)], dim=1)
                v = torch.cat([v, last_v.repeat(1, diff, 1, 1, 1)], dim=1)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.num_pressure_height_windows,
            -1,
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            NumWidthWindows = mask.shape[0]
            attn = attn.view(
                BatchTimesWidthWindows // NumWidthWindows,
                NumWidthWindows,
                self.num_heads,
                NumPressureHeightWindows,
                WindowTokens,
                WindowTokens,
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(
                -1,
                self.num_heads,
                NumPressureHeightWindows,
                WindowTokens,
                WindowTokens,
            )
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(
            BatchTimesWidthWindows,
            NumPressureHeightWindows,
            WindowTokens,
            Channels,
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def apply_architectural_upgrades(model, use_swiglu=False, use_rmsnorm=False, use_gqa=False, kv_group_size=2):
    from onescience.modules.transformer.onetransformer import OneTransformer
    for name, module in model.named_modules():
        if isinstance(module, OneTransformer) and hasattr(module, "transformer"):
            trans = module.transformer
            if trans.__class__.__name__ == "EarthTransformer3DBlock":
                if use_rmsnorm:
                    if hasattr(trans, "norm1") and isinstance(trans.norm1, nn.LayerNorm):
                        trans.norm1 = RMSNorm(trans.norm1.normalized_shape[0], eps=trans.norm1.eps)
                    if hasattr(trans, "norm2") and isinstance(trans.norm2, nn.LayerNorm):
                        trans.norm2 = RMSNorm(trans.norm2.normalized_shape[0], eps=trans.norm2.eps)
                if use_swiglu:
                    if hasattr(trans, "mlp"):
                        old_mlp = trans.mlp
                        if old_mlp.__class__.__name__ == "Mlp":
                            in_features = old_mlp.fc1.in_features
                            out_features = old_mlp.fc2.out_features
                            old_hidden = old_mlp.fc1.out_features
                            new_hidden = int(old_hidden * 2 / 3)
                            trans.mlp = SwiGLUMlp(
                                in_features=in_features,
                                hidden_features=new_hidden,
                                out_features=out_features,
                                drop=old_mlp.drop.p
                            )
                if use_gqa:
                    if hasattr(trans, "attn") and hasattr(trans.attn, "attentioner"):
                        old_attn = trans.attn.attentioner
                        if old_attn.__class__.__name__ == "EarthAttention3D":
                            trans.attn.attentioner = GQAEarthAttention3D(old_attn, kv_group_size=kv_group_size)


def adapt_qkv_for_gqa(source_state, model):
    new_state = {}
    for key, val in source_state.items():
        if key.endswith(".attn.qkv.weight") or key.endswith(".attn.qkv.bias"):
            base_key = key.rsplit(".qkv.", 1)[0]
            parts = base_key.split(".")
            curr = model
            for part in parts:
                if part.isdigit():
                    curr = curr[int(part)]
                else:
                    curr = getattr(curr, part, None)
                    if curr is None:
                        break
            attn_module = None
            if hasattr(curr, "transformer"):
                attn_module = curr.transformer
            elif hasattr(curr, "attentioner"):
                attn_module = curr.attentioner
            elif hasattr(curr, "attn"):
                attn_module = curr.attn
                if hasattr(attn_module, "attentioner"):
                    attn_module = attn_module.attentioner

            if attn_module.__class__.__name__ == "GQAEarthAttention3D":
                is_bias = key.endswith(".bias")
                suffix = ".bias" if is_bias else ".weight"
                q_key = f"{base_key}.q_proj{suffix}"
                kv_key = f"{base_key}.kv_proj{suffix}"

                dim = val.shape[0] // 3
                q_val = val[:dim]
                k_val = val[dim:2 * dim]
                v_val = val[2 * dim:]

                kv_group_size = attn_module.kv_group_size
                num_heads = attn_module.num_heads
                num_kv_heads = attn_module.num_kv_heads
                head_dim = dim // num_heads

                if not is_bias:
                    k_reshaped = k_val.view(num_heads, head_dim, -1)
                    indices = [i * kv_group_size for i in range(num_kv_heads)]
                    k_pooled = k_reshaped[indices, :, :].reshape(num_kv_heads * head_dim, -1)

                    v_reshaped = v_val.view(num_heads, head_dim, -1)
                    v_pooled = v_reshaped[indices, :, :].reshape(num_kv_heads * head_dim, -1)
                else:
                    k_reshaped = k_val.view(num_heads, head_dim)
                    indices = [i * kv_group_size for i in range(num_kv_heads)]
                    k_pooled = k_reshaped[indices, :].reshape(num_kv_heads * head_dim)

                    v_reshaped = v_val.view(num_heads, head_dim)
                    v_pooled = v_reshaped[indices, :].reshape(num_kv_heads * head_dim)

                new_state[q_key] = q_val
                new_state[kv_key] = torch.cat([k_pooled, v_pooled], dim=0)
            else:
                new_state[key] = val
        else:
            new_state[key] = val
    return new_state



def _as_int_list(value):
    return [int(v) for v in value]


def _embed_sequence(model, x):
    if isinstance(x, list):
        SurfaceInput = x[0]
        UpperAirInput = x[1]
        x.clear()
    elif isinstance(x, tuple):
        SurfaceInput, UpperAirInput = x
    else:
        SurfaceInput = x[:, :7, :, :]
        UpperAirInput = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])

    SurfaceFeatures = model.patchembed2d(SurfaceInput)
    del SurfaceInput
    UpperAirFeatures = model.patchembed3d(UpperAirInput)
    del UpperAirInput

    CombinedFeatures = torch.concat(
        [SurfaceFeatures.unsqueeze(2), UpperAirFeatures], dim=2
    )
    del SurfaceFeatures, UpperAirFeatures
    Batch, Channels, PressureLevels, Height, Width = CombinedFeatures.shape
    sequence = CombinedFeatures.reshape(Batch, Channels, -1).transpose(1, 2)
    return sequence, Batch, PressureLevels, Height, Width


def _maybe_empty_cache(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stream_weights_mode(value=None):
    raw_value = os.environ.get("PANGU_STREAM_WEIGHTS", "0") if value is None else value
    raw_value = str(raw_value).strip().lower()
    if raw_value in {"", "0", "false", "no", "off"}:
        return None
    if raw_value not in {"stage", "block"}:
        raise ValueError(
            "PANGU_STREAM_WEIGHTS must be one of: 0, stage, block; "
            f"got {raw_value!r}"
        )
    return raw_value


def _module_tensor_bytes(module):
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        total += tensor.numel() * tensor.element_size()
    return total


def _pin_cpu_module_tensors(module):
    if not torch.cuda.is_available():
        return
    for param in module.parameters(recurse=True):
        if param.data.device.type == "cpu" and not param.data.is_pinned():
            try:
                param.data = param.data.pin_memory()
            except RuntimeError:
                pass
    for child in module.modules():
        for name, buffer in list(child._buffers.items()):
            if buffer is None or buffer.device.type != "cpu":
                continue
            if buffer.is_pinned():
                continue
            try:
                child._buffers[name] = buffer.pin_memory()
            except RuntimeError:
                pass


def _offload_stream_module(module, pin_memory=True):
    module.to("cpu")
    if pin_memory:
        _pin_cpu_module_tensors(module)


def _streamed_stage_modules(model):
    return [
        ("layer1", model.layer1),
        ("downsample", model.downsample),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("upsample", model.upsample),
        ("layer4", model.layer4),
    ]


def _streamed_block_modules(model):
    modules = [("downsample", model.downsample), ("upsample", model.upsample)]
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        blocks = _get_fuser_blocks(getattr(model, layer_name))
        if blocks is None:
            modules.append((layer_name, getattr(model, layer_name)))
            continue
        for idx, block in enumerate(blocks):
            modules.append((f"{layer_name}.block{idx}", block))
    return modules


def _stream_module_to_device(module, device):
    module.to(device)
    return module


def _run_streamed_module(owner, module, x, label=None):
    mode = getattr(owner, "_pangu_stream_weights", None)
    if mode not in {"stage", "block"}:
        return module(x)

    device = x.device
    _stream_module_to_device(module, device)
    if label is not None:
        _profile_layerwise_memory(f"{label}.stream_to_device", reset=True)
    try:
        out = module(x)
    finally:
        _offload_stream_module(
            module,
            pin_memory=bool(getattr(owner, "_pangu_stream_pin_memory", True)),
        )
        _maybe_empty_cache(bool(getattr(owner, "_pangu_stream_empty_cache", True)))
    return out


def enable_streamed_weight_residency(model, mode="stage", pin_memory=True, empty_cache=True):
    """Keep selected backbone weights on CPU and move only the active stage/block to device."""

    mode = _stream_weights_mode(mode)
    if mode is None:
        return 0, 0

    model._pangu_stream_weights = mode
    model._pangu_stream_pin_memory = bool(pin_memory)
    model._pangu_stream_empty_cache = bool(empty_cache)

    modules = _streamed_stage_modules(model) if mode == "stage" else _streamed_block_modules(model)
    seen = set()
    offloaded_count = 0
    offloaded_bytes = 0
    for _, module in modules:
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        offloaded_bytes += _module_tensor_bytes(module)
        _offload_stream_module(module, pin_memory=pin_memory)
        offloaded_count += 1

    _maybe_empty_cache(empty_cache)
    return offloaded_count, offloaded_bytes


def _profile_layerwise_memory(tag, reset=False):
    if not _is_enabled("PANGU_PROFILE_LAYERWISE_MEMORY"):
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"[LAYER-MEM] {tag}: allocated={allocated:.1f} MB, "
        f"reserved={reserved:.1f} MB, peak={peak:.1f} MB"
    )
    if reset:
        torch.cuda.reset_peak_memory_stats()


def _crop_recovery_output(recovery, output):
    _, _, PressureLevels, Height, Width = output.shape

    PressureLevelsPad = PressureLevels - recovery.img_size[0]
    HeightPad = Height - recovery.img_size[1]
    WidthPad = Width - recovery.img_size[2]

    if PressureLevelsPad < 0 or HeightPad < 0 or WidthPad < 0:
        raise ValueError("Recovered feature map is smaller than the target img_size")

    PaddingFront = PressureLevelsPad // 2
    PaddingBack = PressureLevelsPad - PaddingFront
    PaddingTop = HeightPad // 2
    PaddingBottom = HeightPad - PaddingTop
    PaddingLeft = WidthPad // 2
    PaddingRight = WidthPad - PaddingLeft

    return output[
        :,
        :,
        PaddingFront : PressureLevels - PaddingBack,
        PaddingTop : Height - PaddingBottom,
        PaddingLeft : Width - PaddingRight,
    ]


def _recovery_crop_bounds(recovery, output_shape):
    PressureLevels, Height, Width = output_shape

    PressureLevelsPad = PressureLevels - recovery.img_size[0]
    HeightPad = Height - recovery.img_size[1]
    WidthPad = Width - recovery.img_size[2]

    if PressureLevelsPad < 0 or HeightPad < 0 or WidthPad < 0:
        raise ValueError("Recovered feature map is smaller than the target img_size")

    PaddingFront = PressureLevelsPad // 2
    PaddingTop = HeightPad // 2
    PaddingLeft = WidthPad // 2
    return (
        PaddingFront,
        PaddingFront + recovery.img_size[0],
        PaddingTop,
        PaddingTop + recovery.img_size[1],
        PaddingLeft,
        PaddingLeft + recovery.img_size[2],
    )


def _direct_patch_unembed_chunk(recovery, x):
    proj = recovery.proj
    patch_pressure, patch_height, patch_width = tuple(proj.kernel_size)
    Batch, Channels, PressureLevels, Height, Width = x.shape
    weight = proj.weight.reshape(
        Channels,
        recovery.out_chans * patch_pressure * patch_height * patch_width,
    )
    blocks = x.permute(0, 2, 3, 4, 1).reshape(-1, Channels).matmul(weight)
    blocks = blocks.reshape(
        Batch,
        PressureLevels,
        Height,
        Width,
        recovery.out_chans,
        patch_pressure,
        patch_height,
        patch_width,
    )
    output = blocks.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        Batch,
        recovery.out_chans,
        PressureLevels * patch_pressure,
        Height * patch_height,
        Width * patch_width,
    )
    if proj.bias is not None:
        output = output + proj.bias.view(1, -1, 1, 1, 1)
    return output


def _direct_patch_recovery(
    recovery_module, x, width_chunk_size=None, output_to_cpu=False
):
    """Recover non-overlapping patches without ConvTranspose3d workspace."""

    recovery = getattr(recovery_module, "recovery", recovery_module)
    squeeze_pressure_dim = False
    if x.ndim == 4:
        x = x.unsqueeze(2)
        squeeze_pressure_dim = True
    elif x.ndim != 5:
        return recovery_module(x)

    if x.shape[1] != recovery.in_chans:
        raise ValueError(
            f"Expected input channels {recovery.in_chans}, but received {x.shape[1]}"
        )

    proj = recovery.proj
    unsupported = (
        getattr(proj, "groups", 1) != 1
        or tuple(proj.stride) != tuple(proj.kernel_size)
        or tuple(proj.padding) != (0, 0, 0)
        or tuple(proj.output_padding) != (0, 0, 0)
        or tuple(proj.dilation) != (1, 1, 1)
    )
    if unsupported:
        return recovery_module(x.squeeze(2) if squeeze_pressure_dim else x)

    patch_pressure, patch_height, patch_width = tuple(proj.kernel_size)
    full_shape = (
        x.shape[2] * patch_pressure,
        x.shape[3] * patch_height,
        x.shape[4] * patch_width,
    )
    (
        crop_pressure_start,
        crop_pressure_end,
        crop_height_start,
        crop_height_end,
        crop_width_start,
        crop_width_end,
    ) = _recovery_crop_bounds(recovery, full_shape)

    width_chunk_size = (
        x.shape[4]
        if width_chunk_size is None or int(width_chunk_size) <= 0
        else min(int(width_chunk_size), x.shape[4])
    )
    output_shape = (x.shape[0], recovery.out_chans, *tuple(recovery.img_size))
    if output_to_cpu:
        output = torch.empty(
            output_shape,
            dtype=x.dtype,
            device="cpu",
            pin_memory=bool(x.is_cuda),
        )
    else:
        output = x.new_empty(output_shape)

    for start in range(0, x.shape[4], width_chunk_size):
        end = min(start + width_chunk_size, x.shape[4])
        chunk = _direct_patch_unembed_chunk(recovery, x[:, :, :, :, start:end])
        chunk_width_start = start * patch_width
        chunk_width_end = end * patch_width
        overlap_start = max(chunk_width_start, crop_width_start)
        overlap_end = min(chunk_width_end, crop_width_end)
        if overlap_start < overlap_end:
            source_width = slice(
                overlap_start - chunk_width_start,
                overlap_end - chunk_width_start,
            )
            dest_width = slice(
                overlap_start - crop_width_start,
                overlap_end - crop_width_start,
            )
            output[:, :, :, :, dest_width].copy_(
                chunk[
                    :,
                    :,
                    crop_pressure_start:crop_pressure_end,
                    crop_height_start:crop_height_end,
                    source_width,
                ],
                non_blocking=bool(output_to_cpu and x.is_cuda),
            )
        del chunk

    if squeeze_pressure_dim:
        output = output.squeeze(2)
    return output


def _chunked_patchrecovery3d(recovery_module, x, chunk_size):
    """Recover upper-air variables in output-channel chunks to lower peak VRAM."""

    recovery = getattr(recovery_module, "recovery", recovery_module)
    if x.ndim != 5:
        return recovery_module(x)
    if x.shape[1] != recovery.in_chans:
        raise ValueError(
            f"Expected input channels {recovery.in_chans}, but received {x.shape[1]}"
        )

    proj = recovery.proj
    if getattr(proj, "groups", 1) != 1:
        return recovery_module(x)

    out_chans = int(recovery.out_chans)
    chunk_size = max(1, min(int(chunk_size), out_chans))
    if chunk_size >= out_chans:
        return recovery_module(x)

    output = x.new_empty((x.shape[0], out_chans, *tuple(recovery.img_size)))
    bias = proj.bias
    for start in range(0, out_chans, chunk_size):
        end = min(start + chunk_size, out_chans)
        chunk = _direct_patch_unembed_chunk(recovery, x[:, :, :, :, :])
        output[:, start:end, :, :, :].copy_(_crop_recovery_output(recovery, chunk))
        del chunk
        _profile_layerwise_memory(f"recover.upper_air.chunk{start}:{end}", reset=True)
    return output


def _direct_recovery_width_chunk(default=16):
    return _env_int("PANGU_DIRECT_RECOVERY_WIDTH_CHUNK", default)


def _recover_surface(model, output_surface):
    if _is_enabled("PANGU_DIRECT_RECOVERY"):
        return _direct_patch_recovery(
            model.patchrecovery2d,
            output_surface,
            width_chunk_size=_direct_recovery_width_chunk(),
            output_to_cpu=_is_enabled("PANGU_CPU_RECOVERY_OUTPUT"),
        )
    return model.patchrecovery2d(output_surface)


def _recover_upper_air(model, output_upper_air):
    if _is_enabled("PANGU_DIRECT_RECOVERY"):
        return _direct_patch_recovery(
            model.patchrecovery3d,
            output_upper_air,
            width_chunk_size=_direct_recovery_width_chunk(),
            output_to_cpu=_is_enabled("PANGU_CPU_RECOVERY_OUTPUT"),
        )
    if _is_enabled("PANGU_CHUNKED_RECOVERY"):
        chunk_size = _env_int("PANGU_RECOVERY_CHUNK_SIZE", 1)
        return _chunked_patchrecovery3d(model.patchrecovery3d, output_upper_air, chunk_size)
    return model.patchrecovery3d(output_upper_air)


def _run_fuser_layerwise(owner, fuser, x, empty_cache=False, label=None):
    stream_mode = getattr(owner, "_pangu_stream_weights", None)
    if stream_mode == "stage":
        x = _run_streamed_module(owner, fuser, x, label)
        if label is not None:
            _profile_layerwise_memory(label, reset=True)
        return x

    blocks = getattr(fuser, "blocks", None)
    if blocks is None:
        blocks = getattr(getattr(fuser, "fuser", None), "blocks", None)
    if blocks is None:
        x = _run_streamed_module(owner, fuser, x, label)
        if label is not None:
            _profile_layerwise_memory(label, reset=True)
        return x
    for idx, block in enumerate(blocks):
        if stream_mode == "block":
            x = _run_streamed_module(owner, block, x, f"{label}.block{idx}" if label else None)
        else:
            x = block(x)
        if label is not None:
            _profile_layerwise_memory(f"{label}.block{idx}", reset=True)
        _maybe_empty_cache(empty_cache)
    return x


def _run_sample_layerwise(owner, sampler, x, empty_cache=False, label=None):
    x = _run_streamed_module(owner, sampler, x, label)
    if label is not None:
        _profile_layerwise_memory(label, reset=True)
    _maybe_empty_cache(empty_cache)
    return x


def _recover_outputs(model, sequence, Batch, PressureLevels, Height, Width):
    OutputFeatures = sequence.transpose(1, 2).reshape(
        Batch, -1, PressureLevels, Height, Width
    )
    output_surface = OutputFeatures[:, :, 0, :, :]
    output_upper_air = OutputFeatures[:, :, 1:, :, :]
    _profile_layerwise_memory("recover.reshape_views", reset=True)

    output_surface = _recover_surface(model, output_surface)
    _profile_layerwise_memory("recover.surface", reset=True)
    output_upper_air = _recover_upper_air(model, output_upper_air)
    _profile_layerwise_memory("recover.upper_air", reset=True)
    return output_surface, output_upper_air


def _forward_recompute_skip(self, x):
    sequence, Batch, PressureLevels, Height, Width = _embed_sequence(self, x)

    sequence = self.layer1(sequence)
    sequence = self.downsample(sequence)
    sequence = self.layer2(sequence)
    sequence = self.layer3(sequence)
    sequence = self.upsample(sequence)
    sequence = self.layer4(sequence)

    skip_sequence, _, _, _, _ = _embed_sequence(self, x)
    skip_sequence = self.layer1(skip_sequence)

    OutputFeatures = torch.concat([sequence, skip_sequence], dim=-1)
    OutputFeatures = OutputFeatures.transpose(1, 2).reshape(
        Batch, -1, PressureLevels, Height, Width
    )
    output_surface = OutputFeatures[:, :, 0, :, :]
    output_upper_air = OutputFeatures[:, :, 1:, :, :]

    output_surface = _recover_surface(self, output_surface)
    output_upper_air = _recover_upper_air(self, output_upper_air)
    return output_surface, output_upper_air


def _forward_layerwise(self, x):
    empty_cache = bool(getattr(self, "_layerwise_empty_cache", False))
    _profile_layerwise_memory("forward.start", reset=True)
    sequence, Batch, PressureLevels, Height, Width = _embed_sequence(self, x)
    _profile_layerwise_memory("embed_sequence", reset=True)

    sequence = _run_fuser_layerwise(self, self.layer1, sequence, empty_cache, "layer1")
    skip_sequence = sequence
    _maybe_empty_cache(empty_cache)

    sequence = _run_sample_layerwise(self, self.downsample, sequence, empty_cache, "downsample")
    sequence = _run_fuser_layerwise(self, self.layer2, sequence, empty_cache, "layer2")
    sequence = _run_fuser_layerwise(self, self.layer3, sequence, empty_cache, "layer3")
    sequence = _run_sample_layerwise(self, self.upsample, sequence, empty_cache, "upsample")
    sequence = _run_fuser_layerwise(self, self.layer4, sequence, empty_cache, "layer4")

    sequence = torch.concat([sequence, skip_sequence], dim=-1)
    del skip_sequence
    _profile_layerwise_memory("skip_concat", reset=True)
    _maybe_empty_cache(empty_cache)
    return _recover_outputs(self, sequence, Batch, PressureLevels, Height, Width)


def _recover_outputs_split(model, sequence, skip_sequence, Batch, PressureLevels, Height, Width):
    """Recover surface and upper-air outputs without full skip concatenation.

    Instead of concatenating the entire sequence and skip_sequence along the
    channel dimension (which creates a transient tensor of size
    [B, P*H*W, 2*embed_dim]), this function splits the token dimension into
    surface (pressure_level=0) and upper-air (pressure_level=1+) parts and
    concatenates only the smaller slices independently. This reduces peak
    memory by avoiding the full-width concatenation.
    """
    HW = Height * Width
    embed_dim = sequence.shape[-1]

    # Surface: tokens [0, H*W) correspond to pressure_level=0
    surface_seq = sequence[:, :HW, :]        # [B, H*W, C] - view, no alloc
    surface_skip = skip_sequence[:, :HW, :]  # [B, H*W, C] - view, no alloc
    surface_combined = torch.cat([surface_seq, surface_skip], dim=-1)  # [B, H*W, 2C]
    # Transpose+reshape to recovery input format [B, 2C, H, W]
    surface_input = surface_combined.reshape(Batch, Height, Width, 2 * embed_dim)
    surface_input = surface_input.permute(0, 3, 1, 2).contiguous()
    del surface_combined, surface_seq, surface_skip
    _profile_layerwise_memory("split_recover.surface_prep", reset=True)

    output_surface = _recover_surface(model, surface_input)
    del surface_input
    _profile_layerwise_memory("split_recover.surface", reset=True)

    # Upper air: tokens [H*W, P*H*W) correspond to pressure_levels 1+
    upper_seq = sequence[:, HW:, :]        # [B, (P-1)*H*W, C] - view
    upper_skip = skip_sequence[:, HW:, :]  # [B, (P-1)*H*W, C] - view
    upper_combined = torch.cat([upper_seq, upper_skip], dim=-1)  # [B, (P-1)*H*W, 2C]
    del sequence, skip_sequence, upper_seq, upper_skip
    # Reshape to recovery input format [B, 2C, P-1, H, W]
    upper_input = upper_combined.reshape(
        Batch, PressureLevels - 1, Height, Width, 2 * embed_dim
    )
    upper_input = upper_input.permute(0, 4, 1, 2, 3).contiguous()
    del upper_combined
    _profile_layerwise_memory("split_recover.upper_prep", reset=True)

    output_upper_air = _recover_upper_air(model, upper_input)
    del upper_input
    _profile_layerwise_memory("split_recover.upper_air", reset=True)

    return output_surface, output_upper_air


def _forward_layerwise_split(self, x):
    """Layerwise forward with split recovery to avoid full skip concatenation."""
    empty_cache = bool(getattr(self, "_layerwise_empty_cache", False))
    _profile_layerwise_memory("forward.start", reset=True)
    sequence, Batch, PressureLevels, Height, Width = _embed_sequence(self, x)
    _profile_layerwise_memory("embed_sequence", reset=True)

    sequence = _run_fuser_layerwise(self, self.layer1, sequence, empty_cache, "layer1")
    skip_sequence = sequence
    _maybe_empty_cache(empty_cache)

    sequence = _run_sample_layerwise(self, self.downsample, sequence, empty_cache, "downsample")
    sequence = _run_fuser_layerwise(self, self.layer2, sequence, empty_cache, "layer2")
    sequence = _run_fuser_layerwise(self, self.layer3, sequence, empty_cache, "layer3")
    sequence = _run_sample_layerwise(self, self.upsample, sequence, empty_cache, "upsample")
    sequence = _run_fuser_layerwise(self, self.layer4, sequence, empty_cache, "layer4")

    _profile_layerwise_memory("before_split_recovery", reset=True)
    _maybe_empty_cache(empty_cache)
    return _recover_outputs_split(
        self, sequence, skip_sequence, Batch, PressureLevels, Height, Width
    )


def _forward_layerwise_recompute_skip(self, x):
    empty_cache = bool(getattr(self, "_layerwise_empty_cache", False))
    _profile_layerwise_memory("forward.start", reset=True)
    sequence, Batch, PressureLevels, Height, Width = _embed_sequence(self, x)
    _profile_layerwise_memory("embed_sequence", reset=True)

    sequence = _run_fuser_layerwise(self, self.layer1, sequence, empty_cache, "layer1.main")
    sequence = _run_sample_layerwise(self, self.downsample, sequence, empty_cache, "downsample")
    sequence = _run_fuser_layerwise(self, self.layer2, sequence, empty_cache, "layer2")
    sequence = _run_fuser_layerwise(self, self.layer3, sequence, empty_cache, "layer3")
    sequence = _run_sample_layerwise(self, self.upsample, sequence, empty_cache, "upsample")
    sequence = _run_fuser_layerwise(self, self.layer4, sequence, empty_cache, "layer4")

    skip_sequence, _, _, _, _ = _embed_sequence(self, x)
    _profile_layerwise_memory("skip.embed_sequence", reset=True)
    skip_sequence = _run_fuser_layerwise(
        self, self.layer1, skip_sequence, empty_cache, "layer1.skip"
    )

    sequence = torch.concat([sequence, skip_sequence], dim=-1)
    del skip_sequence
    _profile_layerwise_memory("skip_concat", reset=True)
    _maybe_empty_cache(empty_cache)
    return _recover_outputs(self, sequence, Batch, PressureLevels, Height, Width)


def enable_skip_recompute(model):
    """Trade extra layer1 compute for a shorter-lived skip activation."""

    model.forward = types.MethodType(_forward_recompute_skip, model)
    return model


def enable_layerwise_inference(model, recompute_skip=False, empty_cache=False,
                               split_recovery=False):
    """Run Pangu stages and fuser blocks explicitly for memory A/B tests."""

    model._layerwise_empty_cache = bool(empty_cache)
    if recompute_skip:
        forward = _forward_layerwise_recompute_skip
    elif split_recovery:
        forward = _forward_layerwise_split
    else:
        forward = _forward_layerwise
    model.forward = types.MethodType(forward, model)
    return model


def _get_fuser_blocks(fuser):
    if hasattr(fuser, "blocks"):
        return getattr(fuser, "blocks")
    inner = getattr(fuser, "fuser", None)
    if inner is not None and hasattr(inner, "blocks"):
        return getattr(inner, "blocks")
    return None


def _set_fuser_blocks(fuser, blocks):
    shared_blocks = nn.ModuleList(list(blocks))
    if hasattr(fuser, "blocks"):
        fuser.blocks = shared_blocks
        return
    inner = getattr(fuser, "fuser", None)
    if inner is not None and hasattr(inner, "blocks"):
        inner.blocks = shared_blocks
        return
    raise AttributeError("Pangu fuser does not expose a blocks ModuleList")


def validate_mlp_ratio_blocks(mlp_ratio_blocks, depth_blocks, embed_dim):
    """Validate and normalize a four-stage, block-level MLP-ratio schedule."""

    if isinstance(mlp_ratio_blocks, (str, bytes)):
        raise TypeError("mlp_ratio_blocks must be a four-stage nested sequence")
    try:
        stages = list(mlp_ratio_blocks)
    except TypeError as exc:
        raise TypeError(
            "mlp_ratio_blocks must be a four-stage nested sequence"
        ) from exc
    if len(stages) != 4:
        raise ValueError(
            f"mlp_ratio_blocks must contain 4 stages, got {len(stages)}"
        )

    if isinstance(depth_blocks, (str, bytes)):
        raise TypeError("depth_blocks must be a four-element sequence")
    try:
        raw_depths = list(depth_blocks)
    except TypeError as exc:
        raise TypeError("depth_blocks must be a four-element sequence") from exc
    if len(raw_depths) != 4:
        raise ValueError(f"depth_blocks must contain 4 values, got {len(raw_depths)}")

    depths = []
    for stage_idx, depth in enumerate(raw_depths):
        if (
            isinstance(depth, bool)
            or not isinstance(depth, Real)
            or not math.isfinite(float(depth))
            or int(depth) != float(depth)
            or int(depth) <= 0
        ):
            raise ValueError(
                f"depth_blocks[{stage_idx}] must be a positive integer, got {depth!r}"
            )
        depths.append(int(depth))

    embed_dim = int(embed_dim)
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    stage_dims = (embed_dim, embed_dim * 2, embed_dim * 2, embed_dim)
    normalized = []
    for stage_idx, (stage, depth, dim) in enumerate(
        zip(stages, depths, stage_dims)
    ):
        if isinstance(stage, (str, bytes)):
            raise TypeError(
                f"mlp_ratio_blocks[{stage_idx}] must be a sequence of ratios"
            )
        try:
            ratios = list(stage)
        except TypeError as exc:
            raise TypeError(
                f"mlp_ratio_blocks[{stage_idx}] must be a sequence of ratios"
            ) from exc
        if len(ratios) != depth:
            raise ValueError(
                f"mlp_ratio_blocks[{stage_idx}] must contain {depth} ratios, "
                f"got {len(ratios)}"
            )

        normalized_stage = []
        for block_idx, ratio in enumerate(ratios):
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, Real)
                or not math.isfinite(float(ratio))
                or float(ratio) <= 0
            ):
                raise ValueError(
                    f"mlp_ratio_blocks[{stage_idx}][{block_idx}] must be a "
                    f"positive finite number, got {ratio!r}"
                )
            ratio = float(ratio)
            hidden_features = dim * ratio
            if not hidden_features.is_integer():
                raise ValueError(
                    f"mlp_ratio_blocks[{stage_idx}][{block_idx}]={ratio} does "
                    f"not produce an integer hidden size for dim={dim}"
                )
            normalized_stage.append(int(ratio) if ratio.is_integer() else ratio)
        normalized.append(normalized_stage)
    return normalized


def apply_mlp_ratio_blocks(model, mlp_ratio_blocks, embed_dim):
    """Replace selected native Pangu MLPs according to a block-level schedule."""

    stage_names = ("layer1", "layer2", "layer3", "layer4")
    stage_dims = (
        int(embed_dim),
        int(embed_dim) * 2,
        int(embed_dim) * 2,
        int(embed_dim),
    )
    stage_blocks = []
    for stage_name in stage_names:
        blocks = _get_fuser_blocks(getattr(model, stage_name, None))
        if blocks is None:
            raise ValueError(f"{stage_name} does not expose Pangu fuser blocks")
        stage_blocks.append(blocks)

    schedule = validate_mlp_ratio_blocks(
        mlp_ratio_blocks,
        [len(blocks) for blocks in stage_blocks],
        embed_dim,
    )

    from onescience.modules.func_utils.pangu_utils import Mlp

    replacements = []
    for stage_name, blocks, dim, ratios in zip(
        stage_names, stage_blocks, stage_dims, schedule
    ):
        for block_idx, (block, ratio) in enumerate(zip(blocks, ratios)):
            transformer = getattr(block, "transformer", None)
            old_mlp = getattr(transformer, "mlp", None)
            if not isinstance(old_mlp, Mlp):
                raise TypeError(
                    f"{stage_name}.blocks.{block_idx} must contain a native Mlp"
                )
            if (
                old_mlp.fc1.in_features != dim
                or old_mlp.fc2.out_features != dim
                or old_mlp.fc2.in_features != old_mlp.fc1.out_features
            ):
                raise ValueError(
                    f"{stage_name}.blocks.{block_idx} has incompatible MLP shapes"
                )
            target_hidden = int(dim * ratio)
            replacements.append((transformer, old_mlp, target_hidden, ratio))

    replaced = 0
    for transformer, old_mlp, target_hidden, ratio in replacements:
        if old_mlp.fc1.out_features != target_hidden:
            new_mlp = Mlp(
                in_features=old_mlp.fc1.in_features,
                hidden_features=target_hidden,
                out_features=old_mlp.fc2.out_features,
                act_layer=type(old_mlp.act),
                drop=float(old_mlp.drop.p),
            )
            new_mlp.to(
                device=old_mlp.fc1.weight.device,
                dtype=old_mlp.fc1.weight.dtype,
            )
            new_mlp.train(old_mlp.training)
            transformer.mlp = new_mlp
            replaced += 1
        transformer.mlp_ratio = ratio

    model._pangu_mlp_ratio_blocks = tuple(
        tuple(stage) for stage in schedule
    )
    model._pangu_mlp_ratio_replaced = replaced
    return replaced


def enable_deep_block_sharing(model, mode="layer2_to_layer3"):
    """Share same-resolution deep blocks to reduce resident parameters.

    The patch8+96 profile has matching ``layer2`` and ``layer3`` dimensions and
    block counts, making this the least invasive weight-sharing experiment. It
    should be followed by distillation/fine-tuning before any scoring run.
    """

    if mode in {None, "", "0", "false", "no", "off"}:
        return model
    if mode != "layer2_to_layer3":
        raise ValueError(f"Unsupported PANGU_SHARE_DEEP_BLOCKS mode: {mode}")

    layer2_blocks = _get_fuser_blocks(model.layer2)
    layer3_blocks = _get_fuser_blocks(model.layer3)
    if layer2_blocks is None or layer3_blocks is None:
        raise AttributeError("Cannot share deep blocks because layer2/layer3 blocks are unavailable")
    if len(layer2_blocks) != len(layer3_blocks):
        raise ValueError(
            "Cannot share layer2/layer3 blocks with different depths: "
            f"{len(layer2_blocks)} != {len(layer3_blocks)}"
        )
    _set_fuser_blocks(model.layer3, layer2_blocks)
    model._share_deep_blocks = mode
    return model


def _forward_memory_efficient(self, x):
    if isinstance(x, (tuple, list)):
        SurfaceInput, UpperAirInput = x
    else:
        SurfaceInput = x[:, :7, :, :]
        UpperAirInput = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])

    SurfaceFeatures = self.patchembed2d(SurfaceInput)
    UpperAirFeatures = self.patchembed3d(UpperAirInput)

    CombinedFeatures = torch.concat(
        [SurfaceFeatures.unsqueeze(2), UpperAirFeatures], dim=2
    )
    Batch, Channels, PressureLevels, Height, Width = CombinedFeatures.shape
    sequence = CombinedFeatures.reshape(Batch, Channels, -1).transpose(1, 2)

    sequence = self.layer1(sequence)
    skip_sequence = sequence

    sequence = self.downsample(sequence)
    sequence = self.layer2(sequence)
    sequence = self.layer3(sequence)
    sequence = self.upsample(sequence)
    sequence = self.layer4(sequence)

    OutputFeatures = torch.concat([sequence, skip_sequence], dim=-1)
    OutputFeatures = OutputFeatures.transpose(1, 2).reshape(
        Batch, -1, PressureLevels, Height, Width
    )
    output_surface = OutputFeatures[:, :, 0, :, :]
    output_upper_air = OutputFeatures[:, :, 1:, :, :]

    output_surface = _recover_surface(self, output_surface)
    output_upper_air = _recover_upper_air(self, output_upper_air)
    return output_surface, output_upper_air


def enable_memory_efficient_forward(model):
    """Accept tuple inputs so inference can avoid building a 72-channel tensor."""

    model.forward = types.MethodType(_forward_memory_efficient, model)
    return model


Pangu.forward = _forward_memory_efficient


def _forward_chunked_earth_attention_3d(self, x, mask=None):
    BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens, Channels = x.shape
    bias_shape = (
        1,
        self.num_heads,
        NumPressureHeightWindows,
        WindowTokens,
        WindowTokens,
    )

    # T0.5: Cache earth_position_bias to avoid recomputing every forward.
    # The bias depends only on earth_position_bias_table and earth_position_index,
    # both of which are fixed model parameters. Caching eliminates redundant
    # indexing, permutation, and contiguous copy on every forward call.
    cache_bias = bool(getattr(self, "_pangu_cache_earth_bias", False))
    cached = getattr(self, "_cached_earth_position_bias", None)
    if (
        cache_bias
        and cached is not None
        and cached.device == x.device
        and cached.dtype == x.dtype
        and tuple(cached.shape) == bias_shape
    ):
        earth_position_bias = cached
    else:
        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.num_pressure_height_windows,
            -1,
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        earth_position_bias = earth_position_bias.unsqueeze(0)
        if earth_position_bias.dtype != x.dtype:
            earth_position_bias = earth_position_bias.to(dtype=x.dtype)
        if cache_bias:
            self._cached_earth_position_bias = earth_position_bias

    configured_chunk_size = int(getattr(self, "_pangu_attention_chunk_size", 3))
    chunk_size = (
        BatchTimesWidthWindows
        if configured_chunk_size <= 0
        else max(1, configured_chunk_size)
    )
    chunked_qkv = bool(getattr(self, "_pangu_chunked_qkv", False))
    chunked_proj = bool(getattr(self, "_pangu_chunked_proj", False))
    direct_mask_slice = bool(getattr(self, "_pangu_direct_mask_slice", False))

    def select_mask_chunk(start, end):
        """Select a repeating mask range without launching arange/remainder kernels."""

        width_windows = mask.shape[0]
        chunk_length = end - start
        if chunk_length > width_windows:
            indices = torch.arange(start, end, device=mask.device) % width_windows
            return mask.index_select(0, indices)
        offset = start % width_windows
        first_length = min(chunk_length, width_windows - offset)
        first = mask[offset:offset + first_length]
        if first_length == chunk_length:
            return first
        remaining = chunk_length - first_length
        return torch.cat((first, mask[:remaining]), dim=0)

    def run_attention_chunk(start, end, q_chunk, k_chunk, v_chunk):
        attn_chunk = q_chunk @ k_chunk.transpose(-2, -1)
        attn_chunk = attn_chunk + earth_position_bias
        if mask is not None:
            if direct_mask_slice:
                mask_chunk = select_mask_chunk(start, end)
            else:
                NumWidthWindows = mask.shape[0]
                mask_indices = (
                    torch.arange(start, end, device=mask.device) % NumWidthWindows
                )
                mask_chunk = mask.index_select(0, mask_indices)
            attn_chunk = attn_chunk + mask_chunk.unsqueeze(1)
        attn_chunk = self.softmax(attn_chunk)
        attn_chunk = self.attn_drop(attn_chunk)
        return (
            (attn_chunk @ v_chunk)
            .permute(0, 2, 3, 1, 4)
            .reshape(q_chunk.shape[0], NumPressureHeightWindows, WindowTokens, Channels)
        )

    if chunked_proj:
        projected = x.new_empty(
            BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens, Channels
        )
    else:
        out = x.new_empty(
            BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens, Channels
        )

    if chunked_qkv:
        for start in range(0, BatchTimesWidthWindows, chunk_size):
            end = min(start + chunk_size, BatchTimesWidthWindows)
            qkv_chunk = (
                self.qkv(x[start:end])
                .reshape(
                    end - start,
                    NumPressureHeightWindows,
                    WindowTokens,
                    3,
                    self.num_heads,
                    Channels // self.num_heads,
                )
                .permute(3, 0, 4, 1, 2, 5)
            )
            q_chunk = qkv_chunk[0] * self.scale
            attn_out = run_attention_chunk(start, end, q_chunk, qkv_chunk[1], qkv_chunk[2])
            if chunked_proj:
                projected[start:end].copy_(self.proj(attn_out))
            else:
                out[start:end].copy_(attn_out)
            del qkv_chunk, q_chunk, attn_out
    else:
        qkv = (
            self.qkv(x)
            .reshape(
                BatchTimesWidthWindows,
                NumPressureHeightWindows,
                WindowTokens,
                3,
                self.num_heads,
                Channels // self.num_heads,
            )
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        for start in range(0, BatchTimesWidthWindows, chunk_size):
            end = min(start + chunk_size, BatchTimesWidthWindows)
            attn_out = run_attention_chunk(start, end, q[start:end], k[start:end], v[start:end])
            if chunked_proj:
                projected[start:end].copy_(self.proj(attn_out))
            else:
                out[start:end].copy_(attn_out)
            del attn_out
        del qkv, q, k, v

    if chunked_proj:
        x = projected
    else:
        x = self.proj(out)
    x = self.proj_drop(x)
    return x


def enable_chunked_attention(
    model,
    chunk_size=3,
    cache_earth_bias=False,
    chunked_qkv=False,
    chunked_proj=False,
    direct_mask_slice=False,
    stage_options=None,
):
    """Patch model-local EarthAttention3D instances for memory A/B testing."""

    patched = 0
    chunk_size = int(chunk_size)
    stage_options = stage_options or {}
    for module_name, module in model.named_modules():
        if module.__class__.__name__ == "EarthAttention3D":
            options = stage_options.get(_module_stage(module_name), {})
            module._pangu_attention_chunk_size = int(
                options.get("chunk_size", chunk_size)
            )
            module._pangu_cache_earth_bias = bool(cache_earth_bias)
            module._pangu_chunked_qkv = bool(
                options.get("chunked_qkv", chunked_qkv)
            )
            module._pangu_chunked_proj = bool(
                options.get("chunked_proj", chunked_proj)
            )
            module._pangu_direct_mask_slice = bool(direct_mask_slice)
            module.forward = types.MethodType(_forward_chunked_earth_attention_3d, module)
            patched += 1
    return patched


def compact_attention_masks(model):
    """Store fixed 0/-100 shifted-window masks as int8 buffers."""

    compacted = 0
    saved_bytes = 0
    for module in model.modules():
        mask = getattr(module, "attn_mask", None)
        if mask is None or not torch.is_floating_point(mask):
            continue
        if not torch.all((mask == 0) | (mask == -100)).item():
            continue
        old_bytes = mask.numel() * mask.element_size()
        module.attn_mask = mask.to(dtype=torch.int8)
        saved_bytes += old_bytes - module.attn_mask.numel() * module.attn_mask.element_size()
        compacted += 1
    return compacted, saved_bytes


def intern_immutable_buffers(
    model,
    buffer_names=("attn_mask", "earth_position_index"),
):
    """Share storage only for byte-identical, read-only Pangu buffers.

    Run this after ``model.to(device)``. PyTorch applies device conversion to
    each registered buffer independently, so CPU-side aliasing would otherwise
    be lost during the move to DCU.
    """

    canonical_by_metadata = {}
    replaced = 0
    saved_bytes = 0
    shared_groups = set()
    for module_name, module in model.named_modules():
        for buffer_name in buffer_names:
            buffer = module._buffers.get(buffer_name)
            if not isinstance(buffer, torch.Tensor):
                continue
            metadata = (
                buffer_name,
                tuple(buffer.shape),
                buffer.dtype,
                buffer.device,
            )
            candidates = canonical_by_metadata.setdefault(metadata, [])
            canonical = next(
                (item for item in candidates if torch.equal(item[1], buffer)),
                None,
            )
            if canonical is None:
                candidates.append((module_name, buffer))
                continue
            canonical_name, canonical_buffer = canonical
            if canonical_buffer.data_ptr() == buffer.data_ptr():
                continue
            module._buffers[buffer_name] = canonical_buffer
            replaced += 1
            saved_bytes += buffer.numel() * buffer.element_size()
            shared_groups.add((buffer_name, canonical_name))
    return {
        "replaced": replaced,
        "saved_bytes": saved_bytes,
        "shared_groups": len(shared_groups),
    }


def _forward_chunked_mlp_block(self, x: torch.Tensor):
    from onescience.modules.func_utils import crop3d, window_partition, window_reverse
    PressureLevels, Height, Width = self.input_resolution
    Batch, NumTokens, Channels = x.shape

    shortcut = x
    x = self.norm1(x)
    x = x.view(Batch, PressureLevels, Height, Width, Channels)

    x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
    _, PaddedPressureLevels, PaddedHeight, PaddedWidth, _ = x.shape

    ShiftPressureLevels, ShiftHeight, ShiftWidth = self.shift_size
    if self.use_roll:
        shifted_x = torch.roll(
            x,
            shifts=(-ShiftPressureLevels, -ShiftHeight, -ShiftWidth),
            dims=(1, 2, 3),
        )
        x_windows = window_partition(shifted_x, self.window_size)
    else:
        shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)

    WindowPressureLevels, WindowHeight, WindowWidth = self.window_size
    x_windows = x_windows.view(
        x_windows.shape[0],
        x_windows.shape[1],
        WindowPressureLevels * WindowHeight * WindowWidth,
        Channels,
    )

    attn_windows = self.attn(x_windows, mask=self.attn_mask)

    attn_windows = attn_windows.view(
        attn_windows.shape[0],
        attn_windows.shape[1],
        WindowPressureLevels,
        WindowHeight,
        WindowWidth,
        Channels,
    )

    if self.use_roll:
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            Pl=PaddedPressureLevels,
            Lat=PaddedHeight,
            Lon=PaddedWidth,
        )
        x = torch.roll(
            shifted_x,
            shifts=(ShiftPressureLevels, ShiftHeight, ShiftWidth),
            dims=(1, 2, 3),
        )
    else:
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            Pl=PaddedPressureLevels,
            Lat=PaddedHeight,
            Lon=PaddedWidth,
        )
        x = shifted_x

    x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(
        0, 2, 3, 4, 1
    )

    x = x.reshape(Batch, PressureLevels * Height * Width, Channels)
    # T0.6: In-place residual update to avoid allocating a new tensor.
    # Under torch.inference_mode(), drop_path is identity (eval mode, no dropout),
    # so shortcut.add_(x) is safe — shortcut is not referenced elsewhere after this.
    inplace = bool(getattr(self, "_pangu_inplace_block", False))
    if inplace:
        shortcut.add_(self.drop_path(x))
        x = shortcut
    else:
        x = shortcut + self.drop_path(x)

    # Chunked MLP computation to reduce peak dynamic activations VRAM
    configured_chunk_size = int(getattr(self, "_pangu_mlp_chunk_size", 32768))
    chunk_size = NumTokens if configured_chunk_size <= 0 else configured_chunk_size

    if chunk_size >= NumTokens:
        x_mlp = self.mlp(self.norm2(x))
    else:
        x_mlp = x.new_empty(Batch, NumTokens, Channels)
        for start in range(0, NumTokens, chunk_size):
            end = min(start + chunk_size, NumTokens)
            x_mlp[:, start:end].copy_(self.mlp(self.norm2(x[:, start:end])))

    if inplace:
        x.add_(self.drop_path(x_mlp))
    else:
        x = x + self.drop_path(x_mlp)
    return x


def enable_chunked_mlp(
    model,
    chunk_size=32768,
    inplace_block=False,
    stage_chunk_sizes=None,
):
    """Patch model-local EarthTransformer3DBlock instances to save memory on MLP forward pass."""
    patched = 0
    chunk_size = int(chunk_size)
    stage_chunk_sizes = stage_chunk_sizes or {}
    for module_name, module in model.named_modules():
        if module.__class__.__name__ == "EarthTransformer3DBlock":
            stage = _module_stage(module_name)
            module._pangu_mlp_chunk_size = int(
                stage_chunk_sizes.get(stage, chunk_size)
            )
            module._pangu_inplace_block = bool(inplace_block)
            module.forward = types.MethodType(_forward_chunked_mlp_block, module)
            patched += 1
    return patched




def build_pangu_model(
    img_size,
    patch_size,
    embed_dim,
    num_heads,
    window_size,
    depth_blocks=None,
    recompute_skip=False,
    layerwise_inference=False,
    layerwise_empty_cache=False,
    use_swiglu=None,
    use_rmsnorm=None,
    use_gqa=None,
    kv_group_size=None,
    share_deep_blocks=None,
    chunked_attention=None,
    attention_chunk_size=None,
    mlp_ratio_blocks=None,
):
    """Create a Pangu model and patch submission-local profile differences.

    The upstream OneScience Pangu implementation hardcodes patch recovery for
    the original ``[2, 4, 4]`` patch size. PGW-Lite uses ``[2, 8, 8]``, so we
    replace only the recovery heads inside the pangu_weather submission code.
    State-dict key names remain compatible because the replacement uses the
    same ``OneRecovery`` wrapper attributes.
    """

    patch_size = _as_int_list(patch_size)
    img_size = _as_int_list(img_size)
    embed_dim = int(embed_dim)
    num_heads = _as_int_list(num_heads)
    window_size = _as_int_list(window_size)

    model = Pangu(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_size=window_size,
    )

    if depth_blocks is not None:
        depth_blocks = _as_int_list(depth_blocks)
        import numpy as np
        import math

        patched_input_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )
        patched_downsampled_shape = (
            8,
            math.ceil(patched_input_shape[1] / 2),
            math.ceil(patched_input_shape[2] / 2),
        )

        total_depth = sum(depth_blocks)
        drop_path = np.linspace(0, 0.2, total_depth).tolist() if total_depth > 0 else []

        dp_idx = 0

        # layer1
        d1 = depth_blocks[0]
        model.layer1 = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=patched_input_shape,
            depth=d1,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[dp_idx : dp_idx + d1],
        )
        dp_idx += d1

        # layer2
        d2 = depth_blocks[1]
        model.layer2 = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=patched_downsampled_shape,
            depth=d2,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[dp_idx : dp_idx + d2],
        )
        dp_idx += d2

        # layer3
        d3 = depth_blocks[2]
        model.layer3 = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=patched_downsampled_shape,
            depth=d3,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[dp_idx : dp_idx + d3],
        )
        dp_idx += d3

        # layer4
        d4 = depth_blocks[3]
        model.layer4 = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=patched_input_shape,
            depth=d4,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[dp_idx : dp_idx + d4],
        )

    if patch_size != [2, 4, 4]:
        model.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=tuple(img_size),
            patch_size=tuple(patch_size[1:]),
            in_chans=embed_dim * 2,
            out_chans=4,
        )
        model.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(13, *tuple(img_size)),
            patch_size=tuple(patch_size),
            in_chans=embed_dim * 2,
            out_chans=5,
        )

    if use_swiglu is None:
        use_swiglu = _is_enabled("PANGU_USE_SWIGLU")
    if use_rmsnorm is None:
        use_rmsnorm = _is_enabled("PANGU_USE_RMSNORM")
    if use_gqa is None:
        use_gqa = _is_enabled("PANGU_USE_GQA")
    if kv_group_size is None:
        kv_group_size = _env_int("PANGU_GQA_GROUP_SIZE", 2)
    if share_deep_blocks is None:
        share_deep_blocks = _env_share_deep_blocks()

    if mlp_ratio_blocks is not None:
        if use_swiglu:
            raise ValueError(
                "mlp_ratio_blocks requires native Mlp modules; use_swiglu must be False"
            )
        if share_deep_blocks:
            raise ValueError(
                "mlp_ratio_blocks is incompatible with shared deep blocks"
            )
        apply_mlp_ratio_blocks(model, mlp_ratio_blocks, embed_dim)

    apply_architectural_upgrades(
        model,
        use_swiglu=use_swiglu,
        use_rmsnorm=use_rmsnorm,
        use_gqa=use_gqa,
        kv_group_size=kv_group_size
    )

    if share_deep_blocks:
        enable_deep_block_sharing(model, mode=share_deep_blocks)

    if chunked_attention is None:
        chunked_attention = _is_enabled("PANGU_CHUNKED_ATTENTION")
    if attention_chunk_size is None:
        attention_chunk_size = _env_int("PANGU_ATTN_CHUNK_SIZE", 3)
    cache_earth_bias = _is_enabled("PANGU_CACHE_EARTH_BIAS")
    chunked_qkv = _is_enabled("PANGU_CHUNKED_QKV")
    chunked_proj = _is_enabled("PANGU_CHUNKED_PROJ", default=chunked_qkv)
    direct_mask_slice = _is_enabled("PANGU_DIRECT_MASK_SLICE")
    if chunked_attention:
        attention_stage_options = {}
        for stage in ("layer1", "layer2", "layer3", "layer4"):
            suffix = stage.upper()
            attention_stage_options[stage] = {
                "chunk_size": _env_int(
                    f"PANGU_ATTN_CHUNK_SIZE_{suffix}", attention_chunk_size
                ),
                "chunked_qkv": _is_enabled(
                    f"PANGU_CHUNKED_QKV_{suffix}", default=chunked_qkv
                ),
                "chunked_proj": _is_enabled(
                    f"PANGU_CHUNKED_PROJ_{suffix}", default=chunked_proj
                ),
            }
        model._pangu_chunked_attention_count = enable_chunked_attention(
            model, chunk_size=attention_chunk_size,
            cache_earth_bias=cache_earth_bias,
            chunked_qkv=chunked_qkv,
            chunked_proj=chunked_proj,
            direct_mask_slice=direct_mask_slice,
            stage_options=attention_stage_options,
        )
        if chunked_qkv or chunked_proj:
            print(
                "🧠  PANGU_CHUNKED_ATTENTION extra: "
                f"qkv={int(chunked_qkv)} proj={int(chunked_proj)} "
                f"direct_mask={int(direct_mask_slice)}"
            )

    if _is_enabled("PANGU_COMPACT_ATTN_MASK"):
        compacted, saved_bytes = compact_attention_masks(model)
        model._pangu_compact_mask_count = compacted
        model._pangu_compact_mask_saved_bytes = saved_bytes
        print(
            "🗜️  PANGU_COMPACT_ATTN_MASK=1，"
            f"compacted={compacted}，pre-FP16 saved={saved_bytes / 1024**2:.1f} MB"
        )

    inplace_block = _is_enabled("PANGU_INPLACE_BLOCK")
    if _is_enabled("PANGU_CHUNKED_MLP", default=True):
        mlp_chunk_size = _env_int("PANGU_MLP_CHUNK_SIZE", 32768)
        mlp_stage_chunk_sizes = {
            stage: _env_int(f"PANGU_MLP_CHUNK_SIZE_{stage.upper()}", mlp_chunk_size)
            for stage in ("layer1", "layer2", "layer3", "layer4")
        }
        model._pangu_chunked_mlp_count = enable_chunked_mlp(
            model,
            chunk_size=mlp_chunk_size,
            inplace_block=inplace_block,
            stage_chunk_sizes=mlp_stage_chunk_sizes,
        )
        print(f"🧠  PANGU_CHUNKED_MLP=1，chunk_size={mlp_chunk_size}，patched={model._pangu_chunked_mlp_count}")
        if inplace_block:
            print("🔧  PANGU_INPLACE_BLOCK=1，启用 in-place 残差更新")

    split_recovery = _is_enabled("PANGU_SPLIT_RECOVERY")
    if layerwise_inference:
        enable_layerwise_inference(
            model,
            recompute_skip=recompute_skip,
            empty_cache=layerwise_empty_cache,
            split_recovery=split_recovery,
        )
    elif recompute_skip:
        enable_skip_recompute(model)
    else:
        enable_memory_efficient_forward(model)

    return model
