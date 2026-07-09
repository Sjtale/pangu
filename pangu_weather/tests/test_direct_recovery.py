import importlib
import sys
import types
import unittest

import torch
import torch.nn as nn


def _install_onescience_stubs():
    onescience = types.ModuleType("onescience")
    models = types.ModuleType("onescience.models")
    pangu = types.ModuleType("onescience.models.pangu")
    modules = types.ModuleType("onescience.modules")
    attention = types.ModuleType("onescience.modules.attention")
    earthattention3d = types.ModuleType(
        "onescience.modules.attention.earthattention3d"
    )

    class Pangu(nn.Module):
        pass

    class OneRecovery(nn.Module):
        pass

    class OneFuser(nn.Module):
        pass

    class EarthAttention3D(nn.Module):
        pass

    pangu.Pangu = Pangu
    modules.OneRecovery = OneRecovery
    modules.OneFuser = OneFuser
    earthattention3d.EarthAttention3D = EarthAttention3D

    sys.modules.setdefault("onescience", onescience)
    sys.modules.setdefault("onescience.models", models)
    sys.modules.setdefault("onescience.models.pangu", pangu)
    sys.modules.setdefault("onescience.modules", modules)
    sys.modules.setdefault("onescience.modules.attention", attention)
    sys.modules.setdefault(
        "onescience.modules.attention.earthattention3d", earthattention3d
    )


_install_onescience_stubs()
pangu_profile_model = importlib.import_module("pangu_weather.pangu_profile_model")


class TinyPanguPatchRecovery(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        if len(img_size) == 2:
            img_size = (1, *img_size)
        if len(patch_size) == 2:
            patch_size = (1, *patch_size)
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.proj = nn.ConvTranspose3d(
            in_chans,
            out_chans,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        squeeze_pressure_dim = False
        if x.ndim == 4:
            x = x.unsqueeze(2)
            squeeze_pressure_dim = True
        output = self.proj(x)
        output = pangu_profile_model._crop_recovery_output(self, output)
        if squeeze_pressure_dim:
            output = output.squeeze(2)
        return output


class DirectRecoveryTests(unittest.TestCase):
    def test_direct_recovery_matches_convtranspose3d_with_crop(self):
        torch.manual_seed(7)
        recovery = TinyPanguPatchRecovery(
            img_size=(3, 5, 7),
            patch_size=(2, 3, 2),
            in_chans=3,
            out_chans=4,
        )
        x = torch.randn(2, 3, 2, 2, 4)

        expected = recovery(x)
        actual = pangu_profile_model._direct_patch_recovery(
            recovery, x, width_chunk_size=2
        )

        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)

    def test_direct_recovery_matches_2d_surface_recovery_with_crop(self):
        torch.manual_seed(11)
        recovery = TinyPanguPatchRecovery(
            img_size=(5, 7),
            patch_size=(3, 2),
            in_chans=3,
            out_chans=2,
        )
        x = torch.randn(2, 3, 2, 4)

        expected = recovery(x)
        actual = pangu_profile_model._direct_patch_recovery(
            recovery, x, width_chunk_size=1
        )

        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertLessEqual((actual - expected).abs().max().item(), 1e-6)

    def test_scored_only_recovery_matches_subset_channels(self):
        torch.manual_seed(42)
        recovery = TinyPanguPatchRecovery(
            img_size=(13, 16, 16),
            patch_size=(2, 8, 8),
            in_chans=3,
            out_chans=5,
        )
        x = torch.randn(1, 3, 7, 2, 2)

        expected = recovery(x)
        actual = pangu_profile_model._direct_patch_recovery_scored_only(
            recovery, x, width_chunk_size=1
        )

        self.assertEqual(tuple(actual.shape), tuple(expected.shape))

        # Verify the 11 scored channels match expected perfectly
        # Variables: Z (0), Q (1), T (2) at levels 2, 3, 5
        for v in [0, 1, 2]:
            for lvl in [2, 3, 5]:
                diff = (actual[:, v, lvl] - expected[:, v, lvl]).abs().max().item()
                self.assertLessEqual(diff, 1e-6)

        # Variables: U (3), V (4) at level 5
        for v in [3, 4]:
            diff = (actual[:, v, 5] - expected[:, v, 5]).abs().max().item()
            self.assertLessEqual(diff, 1e-6)

        # Verify other channels are filled with zeros
        for v in range(5):
            for lvl in range(13):
                if v in [0, 1, 2] and lvl in [2, 3, 5]:
                    continue
                if v in [3, 4] and lvl == 5:
                    continue
                self.assertEqual(actual[:, v, lvl].abs().max().item(), 0.0)


class StreamedWeightResidencyTests(unittest.TestCase):
    def test_run_streamed_module_preserves_output(self):
        torch.manual_seed(123)
        owner = types.SimpleNamespace(
            _pangu_stream_weights="stage",
            _pangu_stream_pin_memory=False,
            _pangu_stream_empty_cache=False,
        )
        module = nn.Linear(4, 3)
        x = torch.randn(2, 4)

        expected = module(x)
        actual = pangu_profile_model._run_streamed_module(owner, module, x, "linear")

        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(next(module.parameters()).device.type, "cpu")

    def test_enable_streamed_weight_residency_block_mode_marks_model(self):
        class TinyFuser(nn.Module):
            def __init__(self, depth):
                super().__init__()
                self.blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(depth))

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = TinyFuser(2)
                self.downsample = nn.Linear(4, 4)
                self.layer2 = TinyFuser(1)
                self.layer3 = TinyFuser(1)
                self.upsample = nn.Linear(4, 4)
                self.layer4 = TinyFuser(1)

        model = TinyModel()
        count, bytes_offloaded = pangu_profile_model.enable_streamed_weight_residency(
            model, mode="block", pin_memory=False, empty_cache=False
        )

        self.assertEqual(model._pangu_stream_weights, "block")
        self.assertEqual(count, 7)
        self.assertGreater(bytes_offloaded, 0)


class ChunkedAttentionTests(unittest.TestCase):
    def test_chunked_qkv_and_proj_match_full_qkv_path(self):
        class TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.scale = 0.5
                self.num_pressure_height_windows = 2
                self.window_size = (1, 1, 3)
                self.qkv = nn.Linear(4, 12)
                self.proj = nn.Linear(4, 4)
                self.attn_drop = nn.Identity()
                self.proj_drop = nn.Identity()
                self.softmax = nn.Softmax(dim=-1)
                table_len = 3 * 3 * self.num_pressure_height_windows
                self.earth_position_bias_table = nn.Parameter(
                    torch.randn(table_len, self.num_heads)
                )
                self.register_buffer(
                    "earth_position_index", torch.arange(table_len)
                )

        torch.manual_seed(2026)
        attention = TinyAttention()
        x = torch.randn(5, 2, 3, 4)

        attention._pangu_attention_chunk_size = 2
        attention._pangu_cache_earth_bias = False
        attention._pangu_chunked_qkv = False
        attention._pangu_chunked_proj = False
        expected = pangu_profile_model._forward_chunked_earth_attention_3d(
            attention, x
        )

        attention._pangu_chunked_qkv = True
        attention._pangu_chunked_proj = True
        actual = pangu_profile_model._forward_chunked_earth_attention_3d(
            attention, x
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
