import unittest
from pathlib import Path

import torch
from torch import nn

from pangu_weather.compliant_inference_wrapper import (
    CompliantInferenceWrapper,
    fold_output_recovery_affine,
)


class FakePanguCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_surface = None
        self.last_upper_air = None

    def forward(self, inputs):
        surface, upper_air = inputs
        self.last_surface = surface
        self.last_upper_air = upper_air
        return surface[:, :4] + 10, upper_air + 100


def channel_values(start=0.0, step=1.0):
    return torch.arange(69, dtype=torch.float32) * step + start


class CompliantInferenceWrapperTests(unittest.TestCase):
    def make_wrapper(self, *, compute_dtype=torch.float16):
        core = FakePanguCore()
        mask = torch.stack(
            (
                torch.full((2, 3), 201.0),
                torch.full((2, 3), 202.0),
                torch.full((2, 3), 203.0),
            )
        )
        means = channel_values(start=1000.0, step=2.0)
        stds = channel_values(start=1.0, step=0.25)
        slopes = channel_values(start=0.5, step=0.01)
        affine_scale = channel_values(start=0.8, step=0.005)
        affine_bias = channel_values(start=-3.0, step=0.1)
        wrapper = CompliantInferenceWrapper(
            core,
            mask,
            torch.zeros(69),
            torch.ones(69),
            output_means=means,
            output_stds=stds,
            slope_coefficients=slopes,
            affine_scale=affine_scale,
            affine_bias=affine_bias,
            compute_dtype=compute_dtype,
        )
        return wrapper, core, means, stds, slopes, affine_scale, affine_bias

    def test_shape_order_affine_contiguity_and_default_dtype(self):
        wrapper, core, means, stds, slopes, affine_scale, affine_bias = (
            self.make_wrapper()
        )
        raw = torch.arange(2 * 69 * 2 * 3, dtype=torch.float32).reshape(2, 69, 2, 3)

        result = wrapper(raw)

        self.assertEqual(core.last_surface.shape, (2, 7, 2, 3))
        self.assertEqual(core.last_upper_air.shape, (2, 5, 13, 2, 3))
        self.assertEqual(core.last_surface.dtype, torch.float16)
        self.assertEqual(core.last_upper_air.dtype, torch.float16)
        torch.testing.assert_close(core.last_surface[:, :4], raw[:, :4].half())
        torch.testing.assert_close(
            core.last_surface[:, 4:],
            wrapper.static_surface_mask.expand(2, -1, -1, -1),
        )
        torch.testing.assert_close(
            core.last_upper_air.reshape(2, 65, 2, 3), raw[:, 4:].half()
        )

        normalized = torch.cat(
            (raw[:, :4].half() + 10, raw[:, 4:].half() + 100), dim=1
        ).float()
        expected_scale = (stds * slopes * affine_scale).reshape(1, 69, 1, 1)
        expected_bias = (means + affine_bias).reshape(1, 69, 1, 1)
        expected = normalized * expected_scale + expected_bias
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape, (2, 69, 2, 3))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(result.is_contiguous())

    def test_optional_fp32_compute_dtype(self):
        wrapper, core, *_ = self.make_wrapper(compute_dtype=torch.float32)
        raw = torch.arange(69 * 2 * 3, dtype=torch.float16).reshape(1, 69, 2, 3)

        result = wrapper(raw)

        self.assertEqual(core.last_surface.dtype, torch.float32)
        self.assertEqual(core.last_upper_air.dtype, torch.float32)
        self.assertEqual(result.dtype, torch.float32)

    def test_raw_physical_input_is_normalized_inside_forward(self):
        means = channel_values(start=100.0, step=0.5)
        stds = channel_values(start=2.0, step=0.1)
        core = FakePanguCore()
        wrapper = CompliantInferenceWrapper(
            core,
            torch.zeros(3, 1, 2),
            means,
            stds,
            output_means=torch.zeros(69),
            output_stds=torch.ones(69),
        )
        normalized = torch.arange(69 * 2, dtype=torch.float32).reshape(1, 69, 1, 2)
        physical = normalized * stds.reshape(1, 69, 1, 1) + means.reshape(
            1, 69, 1, 1
        )

        wrapper(physical)

        torch.testing.assert_close(
            core.last_surface[:, :4], normalized[:, :4].half()
        )
        torch.testing.assert_close(
            core.last_upper_air.reshape(1, 65, 1, 2),
            normalized[:, 4:].half(),
        )

    def test_fold_helper_is_deterministic_and_matches_pointwise_affine(self):
        means = channel_values(start=10.0, step=0.5)
        stds = channel_values(start=1.0, step=0.1)
        slopes = channel_values(start=0.75, step=0.002)
        affine_scale = channel_values(start=0.9, step=0.001)
        affine_bias = channel_values(start=-1.0, step=0.02)

        first = fold_output_recovery_affine(
            means,
            stds,
            slope_coefficients=slopes,
            affine_scale=affine_scale,
            affine_bias=affine_bias,
        )
        second = fold_output_recovery_affine(
            means,
            stds,
            slope_coefficients=slopes,
            affine_scale=affine_scale,
            affine_bias=affine_bias,
        )

        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
        normalized = channel_values(start=-2.0, step=0.1).reshape(1, 69, 1, 1)
        legacy = means.reshape(1, 69, 1, 1) + (
            affine_scale.reshape(1, 69, 1, 1)
            * (
                stds.reshape(1, 69, 1, 1)
                * slopes.reshape(1, 69, 1, 1)
                * normalized
            )
        ) + affine_bias.reshape(1, 69, 1, 1)
        folded = normalized * first[0] + first[1]
        torch.testing.assert_close(folded, legacy)

    def test_forward_never_materializes_on_cpu(self):
        wrapper, _, *_ = self.make_wrapper()
        wrapper = wrapper.to("meta")
        raw = torch.empty((1, 69, 2, 3), device="meta", dtype=torch.float32)

        result = wrapper(raw)

        self.assertEqual(result.device.type, "meta")
        self.assertEqual(result.shape, (1, 69, 2, 3))

    def test_global_mean_correction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "global mean correction is forbidden"):
            CompliantInferenceWrapper(
                FakePanguCore(),
                torch.zeros(3, 2, 3),
                torch.zeros(69),
                torch.ones(69),
                global_mean_correction=object(),
            )

    def test_submission_entrypoint_keeps_only_h2d_and_d2h_outside_timer(self):
        source = (
            Path(__file__).parents[1] / "inference.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CompliantInferenceWrapper", source)
        self.assertIn("mlp_ratio_blocks=model_profile.get", source)
        self.assertIn("_validate_selective_mlp96_state_load", source)
        selective_source = (
            Path(__file__).parents[1] / "selective_mlp96.py"
        ).read_text(encoding="utf-8")
        self.assertIn('(\"Fuser\", \"fuser\")', selective_source)
        self.assertIn("duplicate_aliases", selective_source)
        self.assertIn("normalize=not compliant_boundary", source)
        loop = source[source.index("for batch_index, data") :]
        timer_start = loop.index("start_time = time.perf_counter()")
        timer_end = loop.index("end_time = time.perf_counter()")
        self.assertLess(
            loop.index('invar = invar.to("cuda:0", non_blocking=True)'),
            timer_start,
        )
        compliant_start = loop.index("if compliant_boundary:")
        compliant_pre_timer = loop[
            compliant_start : loop.index("elif graph_direct_input:", compliant_start)
        ]
        self.assertNotIn("dtype=target_dtype", compliant_pre_timer)
        self.assertGreater(loop.index("model_output = model(invar)"), timer_start)
        self.assertLess(loop.index("model_output = model(invar)"), timer_end)
        self.assertGreater(
            loop.index("pred_var = pred_tensor.detach().cpu().numpy()"), timer_end
        )
        compliant_post = loop[
            loop.index("if compliant_boundary:", timer_end) : loop.index(
                "else:", loop.index("if compliant_boundary:", timer_end)
            )
        ]
        self.assertNotIn("apply_affine_calibration", compliant_post)
        self.assertNotIn("apply_global_mean_correction", compliant_post)


if __name__ == "__main__":
    unittest.main()
