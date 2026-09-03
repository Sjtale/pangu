import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import p2_tiled_attention as p2_adapter
import hip_earth_attention_tiled as hip_adapter


class EarthAttention3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = (2, 4, 4)
        self.num_heads = 3
        self.dim = 96
        self.scale = 32**-0.5
        self.qkv = torch.nn.Identity()
        self.proj = torch.nn.Identity()
        self.proj_drop = torch.nn.Identity()
        self.earth_position_bias_table = torch.nn.Parameter(
            torch.arange(42, dtype=torch.float16).view(7, 2, 3)
        )
        self.register_buffer(
            "earth_position_index",
            torch.arange(32 * 32, dtype=torch.int64).remainder(7).view(32, 32),
        )

    def forward(self, x, mask=None):
        return x


class OneAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attentioner = EarthAttention3D()

    def forward(self, *args, **kwargs):
        return self.attentioner(*args, **kwargs)


class EarthTransformer3DBlock(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.attn = OneAttention()
        self.register_buffer("attn_mask", mask)


class MinimalRegionModel(torch.nn.Module):
    def __init__(self, shared_mask):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                EarthTransformer3DBlock(shared_mask if index % 2 else None)
                for index in range(4)
            ]
        )


def _zero_mask():
    return torch.zeros((2, 1, 32, 32), dtype=torch.float16)


def _mock_backend(convert_calls):
    def compact(index, bias_rows=None):
        del bias_rows
        return index.to(dtype=torch.int16).contiguous()

    def pack(table):
        return table.permute(2, 1, 0).contiguous()

    def convert(mask):
        convert_calls.append(mask.data_ptr())
        return torch.zeros(mask.shape[:-1], dtype=torch.uint8)

    return compact, object(), pack, convert


def _attention(block):
    return block.attn.attentioner


class MinimalRegionReleaseTests(unittest.TestCase):
    def setUp(self):
        p2_adapter._GLOBAL_REGION_IDS_CACHE.clear()

    def tearDown(self):
        p2_adapter._GLOBAL_REGION_IDS_CACHE.clear()

    def test_shared_dense_mask_is_converted_once_and_released(self):
        model = MinimalRegionModel(_zero_mask())
        convert_calls = []
        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(convert_calls),
        ):
            patched = p2_adapter.enable_p2_tiled_attention(
                model,
                release_original_masks=True,
            )

        self.assertEqual(patched, 4)
        self.assertEqual(len(convert_calls), 1)
        shifted = [model.blocks[1], model.blocks[3]]
        unshifted = [model.blocks[0], model.blocks[2]]
        self.assertTrue(all(block.attn_mask is None for block in shifted))

        region_ids = [
            _attention(block)._pangu_p2_tiled_region_ids[1] for block in shifted
        ]
        self.assertIs(region_ids[0], region_ids[1])
        x = torch.zeros((1, 1, 32, 96), dtype=torch.float16)
        for block in shifted:
            self.assertIs(
                p2_adapter._cached_region_ids(_attention(block), None, x),
                region_ids[0],
            )
        for block in unshifted:
            self.assertIsNone(
                p2_adapter._cached_region_ids(_attention(block), None, x)
            )

        report = model._pangu_p2_region_setup_report
        self.assertEqual(report["attention_modules"], 4)
        self.assertEqual(report["shifted_mask_owners"], 2)
        self.assertEqual(report["dense_mask_unique_bytes_before"], 4096)
        self.assertEqual(report["dense_mask_unique_bytes_after"], 0)
        self.assertEqual(report["region_ids_unique_bytes"], 64)

    def test_cpu_prepass_is_reused_after_weights_are_loaded(self):
        model = MinimalRegionModel(_zero_mask())
        convert_calls = []
        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(convert_calls),
        ):
            report = p2_adapter.prepare_p2_region_masks_cpu(model)
            for block in model.blocks:
                _attention(block).earth_position_bias_table.data.fill_(9)
            patched = p2_adapter.enable_p2_tiled_attention(
                model,
                release_original_bias=True,
                release_original_masks=True,
            )

        self.assertEqual(patched, 4)
        self.assertEqual(len(convert_calls), 1)
        self.assertTrue(report["prepared_on_cpu"])
        self.assertEqual(report["region_ids_device_move_count"], 1)
        self.assertTrue(all(block.attn_mask is None for block in model.blocks))
        shifted = [_attention(model.blocks[1]), _attention(model.blocks[3])]
        self.assertIs(
            shifted[0]._pangu_p2_tiled_region_ids[1],
            shifted[1]._pangu_p2_tiled_region_ids[1],
        )
        for attention in shifted:
            packed_bias = attention._pangu_p2_tiled_bias_index[1]
            self.assertTrue(torch.all(packed_bias == 9))
            self.assertTrue(attention._pangu_p2_original_bias_released)

    def test_region_off_keeps_mother_template_masks_and_can_disable(self):
        mask = _zero_mask()
        model = MinimalRegionModel(mask)
        convert_calls = []
        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(convert_calls),
        ):
            patched = p2_adapter.enable_p2_tiled_attention(model)
            restored = p2_adapter.disable_p2_tiled_attention(model)

        self.assertEqual(patched, 4)
        self.assertEqual(restored, 4)
        self.assertEqual(convert_calls, [])
        self.assertIs(model.blocks[1].attn_mask, mask)
        self.assertIs(model.blocks[3].attn_mask, mask)

    def test_released_masks_cannot_enter_unsupported_rollback(self):
        model = MinimalRegionModel(_zero_mask())
        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend([]),
        ):
            p2_adapter.enable_p2_tiled_attention(
                model,
                release_original_masks=True,
            )
        with self.assertRaisesRegex(RuntimeError, "cannot disable P2"):
            p2_adapter.disable_p2_tiled_attention(model)


class InferenceWiringTests(unittest.TestCase):
    def test_region_and_hip_prebuild_wiring_are_before_timer(self):
        source = (ROOT / "inference.py").read_text(encoding="utf-8")
        self.assertIn(
            'os.environ.setdefault("PANGU_P2_REGION_RELEASE", "1")',
            source,
        )
        self.assertIn("release_original_masks=p2_region_release", source)
        self.assertIn(
            'os.environ.setdefault("PANGU_P2_PREBUILD_HIP", "1")',
            source,
        )
        self.assertIn("prepare_hip_earth_attention_tiled(", source)
        self.assertIn("prepare_p2_region_masks_cpu(model)", source)
        self.assertNotIn("_validate_constructor_attention_masks", source)
        self.assertNotIn("PANGU_P2_ELIDE_INDICES", source)
        setup = source.index("prepare_hip_earth_attention_tiled(")
        cpu_mask_intern = source.index('buffer_names=("attn_mask",)')
        cpu_region_setup = source.index("prepare_p2_region_masks_cpu(model)")
        model_to_cuda = source.index("model = model.to('cuda:0')")
        p2_enable = source.index("patched_tiled_attention = enable_p2_tiled_attention(")
        timer = source.index("start_time = time.perf_counter()")
        self.assertLess(cpu_mask_intern, cpu_region_setup)
        self.assertLess(cpu_region_setup, model_to_cuda)
        self.assertLess(model_to_cuda, p2_enable)
        self.assertLess(setup, timer)


class HipPrebuildTests(unittest.TestCase):
    def test_prebuild_loads_and_validates_expected_library(self):
        info = {
            "mode": "full-row-fast",
            "config": {
                "q_tile": 16,
                "k_tile": 16,
                "head_dim": 32,
                "block_threads": 256,
            },
            "occupancy": {"active_blocks_per_multiprocessor": 8},
            "build": {"fingerprint": "abc123"},
        }
        with mock.patch.object(
            hip_adapter,
            "get_hip_earth_attention_tiled_info",
            return_value=info,
        ) as get_info:
            report = hip_adapter.prepare_hip_earth_attention_tiled(
                device="cuda:0",
                mode="full-row-fast",
            )

        get_info.assert_called_once_with(device="cuda:0", mode="full-row-fast")
        self.assertEqual(report["fingerprint"], "abc123")

    def test_prebuild_rejects_unexpected_library_configuration(self):
        info = {
            "config": {
                "q_tile": 32,
                "k_tile": 16,
                "head_dim": 32,
                "block_threads": 256,
            },
            "occupancy": {"active_blocks_per_multiprocessor": 8},
            "build": {"fingerprint": "abc123"},
        }
        with mock.patch.object(
            hip_adapter,
            "get_hip_earth_attention_tiled_info",
            return_value=info,
        ):
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                hip_adapter.prepare_hip_earth_attention_tiled()


if __name__ == "__main__":
    unittest.main()
