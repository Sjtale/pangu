"""Profile-aware Pangu model construction for submission-local variants.

The competition package only ships ``pangu_weather/`` while OneScience remains
the platform-provided dependency. Keep architecture adaptations that are needed
by profile checkpoints here instead of relying on local OneScience edits.
"""

import types
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.models.pangu import Pangu
from onescience.modules import OneRecovery, OneFuser
from onescience.modules.attention.earthattention3d import EarthAttention3D


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


def _direct_patch_recovery(recovery_module, x, width_chunk_size=None):
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
    output = x.new_empty((x.shape[0], recovery.out_chans, *tuple(recovery.img_size)))

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
                ]
            )
        del chunk

    if squeeze_pressure_dim:
        output = output.squeeze(2)
    return output


def _direct_patch_recovery_scored_only(recovery_module, x, width_chunk_size=None):
    """Recover ONLY the 15 scored channels of Pangu-Weather to maximize speed and minimize memory."""
    recovery = getattr(recovery_module, "recovery", recovery_module)

    if x.ndim != 5:
        return _direct_patch_recovery(recovery_module, x, width_chunk_size)

    proj = recovery.proj
    patch_pressure, patch_height, patch_width = tuple(proj.kernel_size)
    Batch, Channels, PressureLevels, Height, Width = x.shape

    if PressureLevels != 7 or recovery.out_chans != 5:
        return _direct_patch_recovery(recovery_module, x, width_chunk_size)

    output = x.new_zeros((Batch, 5, 13, *tuple(recovery.img_size[1:])))

    weight = proj.weight.reshape(Channels, 5, patch_pressure, patch_height, patch_width)
    bias = proj.bias if proj.bias is not None else None

    width_chunk_size = (
        Width
        if width_chunk_size is None or int(width_chunk_size) <= 0
        else min(int(width_chunk_size), Width)
    )

    (
        crop_pressure_start,
        crop_pressure_end,
        crop_height_start,
        crop_height_end,
        crop_width_start,
        crop_width_end,
    ) = _recovery_crop_bounds(recovery, (14, Height * patch_height, Width * patch_width))

    for start in range(0, Width, width_chunk_size):
        end = min(start + width_chunk_size, Width)
        chunk_w = end - start

        # 1. p_idx = 1 (contributes to output levels 2 and 3)
        # variables 0:3 (Z, Q, T)
        w_sub1 = weight[:, 0:3, :, :, :].reshape(Channels, 3 * 2 * patch_height * patch_width)
        x_sub1 = x[:, :, 1:2, :, start:end]

        blocks1 = x_sub1.permute(0, 2, 3, 4, 1).reshape(-1, Channels).matmul(w_sub1)
        blocks1 = blocks1.reshape(Batch, 1, Height, chunk_w, 3, 2, patch_height, patch_width)
        chunk1 = blocks1.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(Batch, 3, 2, Height * patch_height, chunk_w * patch_width)
        if bias is not None:
            chunk1 = chunk1 + bias[0:3].view(1, 3, 1, 1, 1)

        # 2. p_idx = 2 (contributes to output level 5)
        # variables 0:5 (Z, Q, T, U, V) at p_offset = 1
        w_sub2 = weight[:, :, 1:2, :, :].reshape(Channels, 5 * 1 * patch_height * patch_width)
        x_sub2 = x[:, :, 2:3, :, start:end]

        blocks2 = x_sub2.permute(0, 2, 3, 4, 1).reshape(-1, Channels).matmul(w_sub2)
        blocks2 = blocks2.reshape(Batch, 1, Height, chunk_w, 5, 1, patch_height, patch_width)
        chunk2 = blocks2.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(Batch, 5, 1, Height * patch_height, chunk_w * patch_width)
        if bias is not None:
            chunk2 = chunk2 + bias.view(1, 5, 1, 1, 1)

        chunk_width_start = start * patch_width
        chunk_width_end = end * patch_width
        overlap_start = max(chunk_width_start, crop_width_start)
        overlap_end = min(chunk_width_end, crop_width_end)

        if overlap_start < overlap_end:
            source_width = slice(overlap_start - chunk_width_start, overlap_end - chunk_width_start)
            dest_width = slice(overlap_start - crop_width_start, overlap_end - crop_width_start)
            h_slice_src = slice(crop_height_start, crop_height_end)

            output[:, 0:3, 2, :, dest_width].copy_(chunk1[:, :, 0, h_slice_src, source_width])
            output[:, 0:3, 3, :, dest_width].copy_(chunk1[:, :, 1, h_slice_src, source_width])
            output[:, 0:5, 5, :, dest_width].copy_(chunk2[:, :, 0, h_slice_src, source_width])

        del chunk1, chunk2

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
        )
    return model.patchrecovery2d(output_surface)


def _recover_upper_air(model, output_upper_air):
    if _is_enabled("PANGU_DIRECT_RECOVERY"):
        if _is_enabled("PANGU_SCORED_ONLY_RECOVERY"):
            return _direct_patch_recovery_scored_only(
                model.patchrecovery3d,
                output_upper_air,
                width_chunk_size=_direct_recovery_width_chunk(),
            )
        return _direct_patch_recovery(
            model.patchrecovery3d,
            output_upper_air,
            width_chunk_size=_direct_recovery_width_chunk(),
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

    chunk_size = max(1, int(getattr(self, "_pangu_attention_chunk_size", 3)))
    chunked_qkv = bool(getattr(self, "_pangu_chunked_qkv", False))
    chunked_proj = bool(getattr(self, "_pangu_chunked_proj", False))

    def run_attention_chunk(start, end, q_chunk, k_chunk, v_chunk):
        attn_chunk = q_chunk @ k_chunk.transpose(-2, -1)
        attn_chunk = attn_chunk + earth_position_bias
        if mask is not None:
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
):
    """Patch model-local EarthAttention3D instances for memory A/B testing."""

    patched = 0
    chunk_size = max(1, int(chunk_size))
    for module in model.modules():
        if module.__class__.__name__ == "EarthAttention3D":
            module._pangu_attention_chunk_size = chunk_size
            module._pangu_cache_earth_bias = bool(cache_earth_bias)
            module._pangu_chunked_qkv = bool(chunked_qkv)
            module._pangu_chunked_proj = bool(chunked_proj)
            module.forward = types.MethodType(_forward_chunked_earth_attention_3d, module)
            patched += 1
    return patched


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
    chunk_size = getattr(self, "_pangu_mlp_chunk_size", 32768)

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


def enable_chunked_mlp(model, chunk_size=32768, inplace_block=False):
    """Patch model-local EarthTransformer3DBlock instances to save memory on MLP forward pass."""
    patched = 0
    chunk_size = max(1, int(chunk_size))
    for module in model.modules():
        if module.__class__.__name__ == "EarthTransformer3DBlock":
            module._pangu_mlp_chunk_size = chunk_size
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

    apply_architectural_upgrades(
        model,
        use_swiglu=use_swiglu,
        use_rmsnorm=use_rmsnorm,
        use_gqa=use_gqa,
        kv_group_size=kv_group_size
    )

    if share_deep_blocks is None:
        share_deep_blocks = _env_share_deep_blocks()
    if share_deep_blocks:
        enable_deep_block_sharing(model, mode=share_deep_blocks)

    if chunked_attention is None:
        chunked_attention = _is_enabled("PANGU_CHUNKED_ATTENTION")
    if attention_chunk_size is None:
        attention_chunk_size = _env_int("PANGU_ATTN_CHUNK_SIZE", 3)
    cache_earth_bias = _is_enabled("PANGU_CACHE_EARTH_BIAS")
    chunked_qkv = _is_enabled("PANGU_CHUNKED_QKV")
    chunked_proj = _is_enabled("PANGU_CHUNKED_PROJ", default=chunked_qkv)
    if chunked_attention:
        model._pangu_chunked_attention_count = enable_chunked_attention(
            model, chunk_size=attention_chunk_size,
            cache_earth_bias=cache_earth_bias,
            chunked_qkv=chunked_qkv,
            chunked_proj=chunked_proj,
        )
        if chunked_qkv or chunked_proj:
            print(
                "🧠  PANGU_CHUNKED_ATTENTION extra: "
                f"qkv={int(chunked_qkv)} proj={int(chunked_proj)}"
            )

    inplace_block = _is_enabled("PANGU_INPLACE_BLOCK")
    if _is_enabled("PANGU_CHUNKED_MLP", default=True):
        mlp_chunk_size = _env_int("PANGU_MLP_CHUNK_SIZE", 32768)
        model._pangu_chunked_mlp_count = enable_chunked_mlp(
            model, chunk_size=mlp_chunk_size,
            inplace_block=inplace_block,
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
