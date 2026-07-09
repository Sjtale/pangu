import importlib
import sys
import types
import unittest
import torch
import torch.nn as nn

def _install_onescience_stubs():
    try:
        import onescience
        import onescience.models.pangu
        import onescience.modules.func_utils
        return
    except ImportError:
        pass

    onescience = types.ModuleType("onescience")
    models = types.ModuleType("onescience.models")
    pangu = types.ModuleType("onescience.models.pangu")
    modules = types.ModuleType("onescience.modules")
    func_utils = types.ModuleType("onescience.modules.func_utils")
    attention = types.ModuleType("onescience.modules.attention")
    earthattention3d = types.ModuleType("onescience.modules.attention.earthattention3d")

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

    def crop3d(x, res):
        pressure, height, width = res
        return x[:, :, :pressure, :height, :width]

    def window_partition(x, win):
        batch, pressure, height, width, channels = x.shape
        wp, wh, ww = win
        x = x.view(
            batch,
            pressure // wp,
            wp,
            height // wh,
            wh,
            width // ww,
            ww,
            channels,
        )
        return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
            batch,
            -1,
            wp,
            wh,
            ww,
            channels,
        )

    def window_reverse(windows, win, Pl, Lat, Lon):
        batch = windows.shape[0]
        wp, wh, ww = win
        windows = windows.view(
            batch,
            Pl // wp,
            Lat // wh,
            Lon // ww,
            wp,
            wh,
            ww,
            windows.shape[-1],
        )
        return windows.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(
            batch,
            Pl,
            Lat,
            Lon,
            windows.shape[-1],
        )

    func_utils.crop3d = crop3d
    func_utils.window_partition = window_partition
    func_utils.window_reverse = window_reverse

    sys.modules.setdefault("onescience", onescience)
    sys.modules.setdefault("onescience.models", models)
    sys.modules.setdefault("onescience.models.pangu", pangu)
    sys.modules.setdefault("onescience.modules", modules)
    sys.modules.setdefault("onescience.modules.func_utils", func_utils)
    sys.modules.setdefault("onescience.modules.attention", attention)
    sys.modules.setdefault("onescience.modules.attention.earthattention3d", earthattention3d)

_install_onescience_stubs()
pangu_profile_model = importlib.import_module("pangu_weather.pangu_profile_model")


class DummyRecovery(nn.Module):
    def __init__(self, out_chans):
        super().__init__()
        self.out_chans = out_chans

    def forward(self, x):
        return x * 2.0


class T0OptimizationsTests(unittest.TestCase):

    def test_split_recovery_equivalence(self):
        """Test that split recovery produces identical output to full skip concat recovery."""
        torch.manual_seed(42)

        # Build mock model with recovery units
        model = types.SimpleNamespace()
        model.patchrecovery2d = DummyRecovery(out_chans=4)
        model.patchrecovery3d = DummyRecovery(out_chans=5)

        # Direct recovery env flags must be mocked out or disabled
        # We can hook the functions or set mock methods on model
        original_recover_surface = pangu_profile_model._recover_surface
        original_recover_upper_air = pangu_profile_model._recover_upper_air
        pangu_profile_model._recover_surface = lambda m, x: m.patchrecovery2d(x)
        pangu_profile_model._recover_upper_air = lambda m, x: m.patchrecovery3d(x)

        try:
            Batch = 2
            PressureLevels = 4
            Height = 8
            Width = 8
            embed_dim = 16
            P_tokens = PressureLevels * Height * Width

            sequence = torch.randn(Batch, P_tokens, embed_dim)
            skip_sequence = torch.randn(Batch, P_tokens, embed_dim)

            # 1. Run baseline full recovery
            full_seq = torch.concat([sequence, skip_sequence], dim=-1)
            expected_surface, expected_upper = pangu_profile_model._recover_outputs(
                model, full_seq, Batch, PressureLevels, Height, Width
            )

            # 2. Run split recovery
            actual_surface, actual_upper = pangu_profile_model._recover_outputs_split(
                model, sequence.clone(), skip_sequence.clone(), Batch, PressureLevels, Height, Width
            )

            # Check shapes
            self.assertEqual(actual_surface.shape, expected_surface.shape)
            self.assertEqual(actual_upper.shape, expected_upper.shape)

            # Check values
            self.assertTrue(torch.allclose(actual_surface, expected_surface, atol=1e-6))
            self.assertTrue(torch.allclose(actual_upper, expected_upper, atol=1e-6))

        finally:
            pangu_profile_model._recover_surface = original_recover_surface
            pangu_profile_model._recover_upper_air = original_recover_upper_air

    def test_earth_attention_bias_caching(self):
        """Test that Earth position bias caching works and returns identical outputs."""
        torch.manual_seed(100)

        # Setup mock EarthAttention3D properties
        class MockEarthAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.scale = 0.125
                self.num_pressure_height_windows = 2
                self.window_size = (2, 2, 2)
                # table: [N_table, num_heads]
                self.earth_position_bias_table = nn.Parameter(torch.randn(30, 2))
                self.earth_position_index = torch.randint(0, 30, (8, 8, 2))
                self.attn_drop = nn.Identity()
                self.softmax = nn.Softmax(dim=-1)
                self.proj = nn.Linear(4, 4)
                self.proj_drop = nn.Identity()

                # Stub qkv projection
                self.qkv = nn.Linear(4, 12)

        # Create instance and patch forward method
        module = MockEarthAttention()
        module.forward = types.MethodType(pangu_profile_model._forward_chunked_earth_attention_3d, module)

        x = torch.randn(4, 2, 8, 4) # [BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens, Channels]

        # 1. Run without cache
        module._pangu_cache_earth_bias = False
        module._pangu_attention_chunk_size = 2
        out_no_cache = module(x)
        self.assertFalse(hasattr(module, "_cached_earth_position_bias"))

        # 2. Run with cache
        module._pangu_cache_earth_bias = True
        out_cache_first = module(x)
        self.assertTrue(hasattr(module, "_cached_earth_position_bias"))
        cached_bias = module._cached_earth_position_bias

        # 3. Run with cache again, verify identical output and tensor reuse
        out_cache_second = module(x)
        self.assertTrue(module._cached_earth_position_bias is cached_bias)

        # Verify numerical equivalence
        self.assertTrue(torch.allclose(out_no_cache, out_cache_first, atol=1e-5))
        self.assertTrue(torch.allclose(out_no_cache, out_cache_second, atol=1e-5))

    def test_inplace_residual_updates(self):
        """Test that in-place residual updates produce identical output to baseline out-of-place."""
        torch.manual_seed(200)

        # Setup mock EarthTransformer3DBlock properties
        class MockTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_resolution = (2, 4, 4)
                self.shift_size = (0, 0, 0)
                self.window_size = (2, 2, 2)
                self.use_roll = False

                self.norm1 = nn.Identity()
                self.norm2 = nn.Identity()
                self.pad = nn.Identity()
                self.drop_path = nn.Identity()

                # Attention mock return x
                self.attn = lambda x_windows, mask: x_windows
                self.attn_mask = None

                # MLP
                self.mlp = nn.Linear(8, 8)

        block = MockTransformerBlock()
        block.forward = types.MethodType(pangu_profile_model._forward_chunked_mlp_block, block)

        x = torch.randn(1, 32, 8) # [Batch, NumTokens, Channels]

        with torch.inference_mode():
            # 1. Out of place
            block._pangu_inplace_block = False
            block._pangu_mlp_chunk_size = 16
            expected = block(x.clone())

            # 2. In place
            block._pangu_inplace_block = True
            actual = block(x.clone())

            # Check outputs
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
