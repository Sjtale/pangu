"""FAQ-compliant, device-resident Pangu inference boundary.

The wrapper accepts the raw physical 69-channel tensor after its transfer to
the accelerator.  Every required transformation from normalization through the
final physical 69-channel prediction stays inside ``forward``.  The caller may
copy the returned tensor to the CPU after the timed call.
"""

from __future__ import annotations

import torch
from torch import nn


NUM_CHANNELS = 69
NUM_SURFACE_CHANNELS = 4
NUM_UPPER_AIR_VARIABLES = 5
NUM_PRESSURE_LEVELS = 13
NUM_STATIC_SURFACE_CHANNELS = 3


def _channel_buffer(name: str, value) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.numel() != NUM_CHANNELS:
        raise ValueError(
            f"{name} must contain {NUM_CHANNELS} channel values, "
            f"got {tensor.numel()}"
        )
    return tensor.reshape(1, NUM_CHANNELS, 1, 1).contiguous()


def build_standard_output_recovery(
    means, stds
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only the organizer-provided physical-unit recovery statistics."""

    return _channel_buffer("stds", stds), _channel_buffer("means", means)


class CompliantInferenceWrapper(nn.Module):
    """Keep all required Pangu preprocessing and postprocessing in ``forward``.

    ``compute_dtype`` controls the raw input and static-mask dtype presented to
    ``core_model``.  FP16 is the submission default; FP32 is available for a
    controlled accuracy/debug run.  Physical recovery is always performed in
    FP32 because pressure-like channels exceed the finite FP16 range.
    """

    def __init__(
        self,
        core_model: nn.Module,
        static_surface_mask,
        input_means,
        input_stds,
        *,
        output_means=None,
        output_stds=None,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if compute_dtype not in (torch.float16, torch.float32):
            raise ValueError("compute_dtype must be torch.float16 or torch.float32")
        mask = torch.as_tensor(static_surface_mask, dtype=compute_dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if (
            mask.ndim != 4
            or mask.shape[0] != 1
            or mask.shape[1] != NUM_STATIC_SURFACE_CHANNELS
        ):
            raise ValueError("static_surface_mask must have shape [3,H,W] or [1,3,H,W]")

        input_mean = _channel_buffer("input_means", input_means)
        input_std = _channel_buffer("input_stds", input_stds)
        if not torch.isfinite(input_mean).all() or not torch.isfinite(input_std).all():
            raise ValueError("input normalization statistics must be finite")
        if torch.any(input_std == 0):
            raise ValueError("input_stds must be non-zero")
        output_mean_values = input_means if output_means is None else output_means
        output_std_values = input_stds if output_stds is None else output_stds
        output_scale, output_bias = build_standard_output_recovery(
            output_mean_values, output_std_values
        )
        self.core_model = core_model
        self.compute_dtype = compute_dtype
        self.register_buffer("static_surface_mask", mask.contiguous())
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_inv_std", input_std.reciprocal().contiguous())
        self.register_buffer("output_scale", output_scale)
        self.register_buffer("output_bias", output_bias)

    def forward(self, raw_physical: torch.Tensor) -> torch.Tensor:
        """Return contiguous physical fields in official 69-channel order."""

        if raw_physical.ndim != 4 or raw_physical.shape[1] != NUM_CHANNELS:
            raise ValueError("raw_physical must have shape [B,69,H,W]")
        batch, _, height, width = raw_physical.shape
        if tuple(self.static_surface_mask.shape[-2:]) != (height, width):
            raise ValueError("raw_physical spatial shape does not match static mask")
        if raw_physical.device != self.static_surface_mask.device:
            raise ValueError("raw_physical and wrapper buffers must be on the same device")

        # Per-sample normalization is preprocessing, so it deliberately lives
        # inside this timed forward rather than inside the CPU data loader.
        raw = raw_physical.to(torch.float32)
        raw = raw.sub(self.input_mean).mul(self.input_inv_std)
        raw = raw.to(dtype=self.compute_dtype)
        static_mask = self.static_surface_mask.expand(batch, -1, -1, -1)
        surface_input = torch.cat((raw[:, :NUM_SURFACE_CHANNELS], static_mask), dim=1)
        upper_air_input = raw[:, NUM_SURFACE_CHANNELS:].reshape(
            batch,
            NUM_UPPER_AIR_VARIABLES,
            NUM_PRESSURE_LEVELS,
            height,
            width,
        )

        output_surface, output_upper_air = self.core_model(
            (surface_input, upper_air_input)
        )
        if tuple(output_surface.shape) != (
            batch,
            NUM_SURFACE_CHANNELS,
            height,
            width,
        ):
            raise ValueError("core surface output must have shape [B,4,H,W]")
        if tuple(output_upper_air.shape) != (
            batch,
            NUM_UPPER_AIR_VARIABLES,
            NUM_PRESSURE_LEVELS,
            height,
            width,
        ):
            raise ValueError("core upper-air output must have shape [B,5,13,H,W]")
        if (
            output_surface.device != raw_physical.device
            or output_upper_air.device != raw_physical.device
        ):
            raise ValueError("core outputs must remain on the input device")

        normalized_output = torch.cat(
            (
                output_surface,
                output_upper_air.reshape(batch, -1, height, width),
            ),
            dim=1,
        )
        physical_output = normalized_output.to(torch.float32)
        physical_output = physical_output.mul(self.output_scale).add(self.output_bias)
        return physical_output.contiguous()
