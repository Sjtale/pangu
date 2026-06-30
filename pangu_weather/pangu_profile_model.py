"""Profile-aware Pangu model construction for submission-local variants.

The competition package only ships ``pangu_weather/`` while OneScience remains
the platform-provided dependency. Keep architecture adaptations that are needed
by profile checkpoints here instead of relying on local OneScience edits.
"""

import types

from onescience.models.pangu import Pangu
from onescience.modules import OneRecovery, OneFuser
import torch


def _as_int_list(value):
    return [int(v) for v in value]


def _embed_sequence(model, x):
    SurfaceInput = x[:, :7, :, :]
    UpperAirInput = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])

    SurfaceFeatures = model.patchembed2d(SurfaceInput)
    UpperAirFeatures = model.patchembed3d(UpperAirInput)
    CombinedFeatures = torch.concat(
        [SurfaceFeatures.unsqueeze(2), UpperAirFeatures], dim=2
    )
    Batch, Channels, PressureLevels, Height, Width = CombinedFeatures.shape
    sequence = CombinedFeatures.reshape(Batch, Channels, -1).transpose(1, 2)
    return sequence, Batch, PressureLevels, Height, Width


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

    output_surface = self.patchrecovery2d(output_surface)
    output_upper_air = self.patchrecovery3d(output_upper_air)
    return output_surface, output_upper_air


def enable_skip_recompute(model):
    """Trade extra layer1 compute for a shorter-lived skip activation."""

    model.forward = types.MethodType(_forward_recompute_skip, model)
    return model


def build_pangu_model(
    img_size,
    patch_size,
    embed_dim,
    num_heads,
    window_size,
    depth_blocks=None,
    recompute_skip=False,
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

    if recompute_skip:
        enable_skip_recompute(model)

    return model
