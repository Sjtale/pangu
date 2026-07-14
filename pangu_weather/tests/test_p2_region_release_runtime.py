import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import hip_earth_attention_tiled as hip_adapter
import p2_tiled_attention as p2_adapter


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

    def __setattr__(self, name, value):
        if name == "forward" and self.__dict__.get("_reject_forward_patch", False):
            raise RuntimeError("synthetic forward patch failure")
        super().__setattr__(name, value)


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


class ExactP2Model(torch.nn.Module):
    def __init__(self, blocks=16, mask_factory=None):
        super().__init__()
        mask_factory = mask_factory or (lambda _index: _zero_mask())
        self.blocks = torch.nn.ModuleList(
            [
                EarthTransformer3DBlock(
                    mask_factory(index) if index % 2 else None
                )
                for index in range(blocks)
            ]
        )


def _zero_mask():
    return torch.zeros((2, 1, 32, 32), dtype=torch.float16)


def _two_region_mask():
    labels = torch.arange(32).div(16, rounding_mode="floor")
    relation = labels[:, None] == labels[None, :]
    matrix = torch.where(relation, 0.0, -100.0).to(dtype=torch.float16)
    return matrix.view(1, 1, 32, 32).repeat(2, 1, 1, 1)


def _mixed_region_mask():
    return torch.cat((_zero_mask()[:1], _two_region_mask()[1:]), dim=0)


def _mock_backend(mask_converter):
    def compact(index, bias_rows=None):
        del bias_rows
        return index.to(dtype=torch.int16).contiguous()

    def pack(table):
        return table.permute(2, 1, 0).contiguous()

    return compact, object(), pack, mask_converter


def _attention(block):
    return block.attn.attentioner


class P2RegionReleaseTests(unittest.TestCase):
    def setUp(self):
        p2_adapter._GLOBAL_REGION_IDS_CACHE.clear()

    def tearDown(self):
        p2_adapter._GLOBAL_REGION_IDS_CACHE.clear()

    def test_exact_setup_dedupes_releases_and_restores_masks(self):
        model = ExactP2Model()
        calls = []

        def convert(mask):
            calls.append(mask.data_ptr())
            return torch.zeros(mask.shape[:2] + mask.shape[2:3], dtype=torch.uint8)

        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(convert),
        ), mock.patch.object(
            p2_adapter,
            "_cuda_memory_allocated",
            side_effect=(1_000_000, 967_000),
        ):
            patched = p2_adapter.enable_p2_tiled_attention(
                model,
                precompute_region_ids=True,
                release_original_masks=True,
                retain_cpu_mask_backup=True,
            )

            self.assertEqual(patched, 16)
            self.assertEqual(len(calls), 1)
            shifted = [block for index, block in enumerate(model.blocks) if index % 2]
            unshifted = [block for index, block in enumerate(model.blocks) if not index % 2]
            self.assertTrue(all(block.attn_mask is None for block in shifted))

            region_ids = [
                _attention(block)._pangu_p2_tiled_region_ids[1] for block in shifted
            ]
            self.assertTrue(all(item.data_ptr() == region_ids[0].data_ptr() for item in region_ids))
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
            self.assertEqual(report["attention_modules"], 16)
            self.assertEqual(report["shifted_mask_owners"], 8)
            self.assertEqual(report["dense_mask_logical_bytes_after"], 0)
            self.assertEqual(report["dense_mask_unique_bytes_after"], 0)
            self.assertEqual(report["dense_mask_unique_storage_count_after"], 0)
            self.assertEqual(
                report["theoretical_dense_mask_reclaimed_bytes"],
                32_768,
            )
            self.assertEqual(report["theoretical_net_reclaimed_bytes"], 32_704)
            self.assertEqual(report["net_unique_bytes_reclaimed"], 32_704)
            self.assertEqual(
                report["actual_cuda_allocated_reclaimed_bytes"],
                33_000,
            )
            self.assertEqual(report["region_ids_unique_storage_count"], 1)
            self.assertEqual(report["cuda_memory_allocated_before"], 1_000_000)
            self.assertEqual(report["cuda_memory_allocated_after"], 967_000)
            self.assertEqual(report["cuda_memory_allocated_delta"], -33_000)

            restored = p2_adapter.disable_p2_tiled_attention(model)
            self.assertEqual(restored, 16)
            self.assertTrue(
                all(torch.equal(block.attn_mask, _zero_mask()) for block in shifted)
            )
            self.assertFalse(hasattr(model, "_pangu_p2_region_setup_report"))

    def test_conversion_failure_releases_nothing_and_patches_nothing(self):
        shifted_index = 0

        def mask_factory(_index):
            nonlocal shifted_index
            shifted_index += 1
            return _zero_mask() if shifted_index < 5 else _two_region_mask()

        model = ExactP2Model(mask_factory=mask_factory)
        calls = 0

        def convert(mask):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("synthetic conversion failure")
            return torch.zeros(mask.shape[:2] + mask.shape[2:3], dtype=torch.uint8)

        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(convert),
        ):
            with self.assertRaisesRegex(ValueError, "synthetic conversion failure"):
                p2_adapter.enable_p2_tiled_attention(
                    model,
                    precompute_region_ids=True,
                    release_original_masks=True,
                )

        shifted = [block for index, block in enumerate(model.blocks) if index % 2]
        self.assertTrue(all(block.attn_mask is not None for block in shifted))
        self.assertTrue(
            all(
                not hasattr(_attention(block), "_pangu_p2_original_forward")
                for block in model.blocks
            )
        )
        self.assertEqual(p2_adapter._GLOBAL_REGION_IDS_CACHE, {})

    def test_commit_failure_restores_every_dense_mask(self):
        model = ExactP2Model()
        original_biases = [
            _attention(block).earth_position_bias_table for block in model.blocks
        ]
        original_indices = [
            _attention(block).earth_position_index for block in model.blocks
        ]

        def convert(mask):
            return torch.zeros(mask.shape[:2] + mask.shape[2:3], dtype=torch.uint8)

        victim = model.blocks[9]
        original_register_buffer = victim.register_buffer

        def reject_release(name, tensor, persistent=True):
            if name == "attn_mask" and tensor is None:
                raise RuntimeError("synthetic release failure")
            return original_register_buffer(name, tensor, persistent=persistent)

        victim.register_buffer = reject_release
        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(convert),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic release failure"):
                p2_adapter.enable_p2_tiled_attention(
                    model,
                    release_original_bias=True,
                    precompute_region_ids=True,
                    release_original_masks=True,
                )

        shifted = [block for index, block in enumerate(model.blocks) if index % 2]
        self.assertTrue(all(block.attn_mask is not None for block in shifted))
        self.assertTrue(
            all(
                not hasattr(_attention(block), "_pangu_p2_region_ids_prepared")
                for block in model.blocks
            )
        )
        self.assertTrue(
            all(
                _attention(block).forward.__func__ is EarthAttention3D.forward
                for block in model.blocks
            )
        )
        self.assertTrue(
            all(
                _attention(block).earth_position_bias_table is original_bias
                for block, original_bias in zip(model.blocks, original_biases)
            )
        )
        self.assertTrue(
            all(
                _attention(block).earth_position_index is original_index
                for block, original_index in zip(model.blocks, original_indices)
            )
        )
        self.assertFalse(hasattr(model, "_pangu_p2_tiled_attention_count"))
        self.assertEqual(p2_adapter._GLOBAL_REGION_IDS_CACHE, {})

    def test_forward_patch_failure_restores_complete_pre_call_state(self):
        model = ExactP2Model()
        victim = _attention(model.blocks[8])
        victim._reject_forward_patch = True
        original_biases = [
            _attention(block).earth_position_bias_table for block in model.blocks
        ]
        original_indices = [
            _attention(block).earth_position_index for block in model.blocks
        ]

        def convert(mask):
            return torch.zeros(mask.shape[:2] + mask.shape[2:3], dtype=torch.uint8)

        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(convert),
        ):
            with self.assertRaisesRegex(RuntimeError, "forward patch failure"):
                p2_adapter.enable_p2_tiled_attention(
                    model,
                    release_original_bias=True,
                    precompute_region_ids=True,
                    release_original_masks=True,
                )

        shifted = [block for index, block in enumerate(model.blocks) if index % 2]
        self.assertTrue(all(block.attn_mask is not None for block in shifted))
        self.assertTrue(
            all(
                _attention(block).forward.__func__ is EarthAttention3D.forward
                for block in model.blocks
            )
        )
        self.assertTrue(
            all(
                _attention(block).earth_position_bias_table is original_bias
                for block, original_bias in zip(model.blocks, original_biases)
            )
        )
        self.assertTrue(
            all(
                _attention(block).earth_position_index is original_index
                for block, original_index in zip(model.blocks, original_indices)
            )
        )
        self.assertTrue(
            all(
                not hasattr(_attention(block), "_pangu_p2_original_forward")
                for block in model.blocks
            )
        )
        self.assertFalse(hasattr(model, "_pangu_p2_tiled_attention_count"))
        self.assertEqual(p2_adapter._GLOBAL_REGION_IDS_CACHE, {})

    def test_every_dense_matrix_must_reconstruct_before_release(self):
        model = ExactP2Model(mask_factory=lambda _index: _mixed_region_mask())

        def invalid_convert(mask):
            return torch.zeros(mask.shape[:2] + mask.shape[2:3], dtype=torch.uint8)

        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(invalid_convert),
        ):
            with self.assertRaisesRegex(RuntimeError, "matrix=1"):
                p2_adapter.enable_p2_tiled_attention(
                    model,
                    precompute_region_ids=True,
                    release_original_masks=True,
                )

        shifted = [block for index, block in enumerate(model.blocks) if index % 2]
        self.assertTrue(all(block.attn_mask is not None for block in shifted))
        self.assertTrue(
            all(
                not hasattr(_attention(block), "_pangu_p2_original_forward")
                for block in model.blocks
            )
        )
        self.assertEqual(p2_adapter._GLOBAL_REGION_IDS_CACHE, {})

    def test_exact_topology_and_prepared_marker_fail_closed(self):
        short_model = ExactP2Model(blocks=15)

        def convert(mask):
            return torch.zeros(mask.shape[:2] + mask.shape[2:3], dtype=torch.uint8)

        with mock.patch.object(
            p2_adapter,
            "_backend",
            return_value=_mock_backend(convert),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly 16"):
                p2_adapter.enable_p2_tiled_attention(
                    short_model,
                    precompute_region_ids=True,
                )

            model = ExactP2Model()
            p2_adapter.enable_p2_tiled_attention(
                model,
                precompute_region_ids=True,
                release_original_masks=True,
                retain_cpu_mask_backup=True,
            )
            shifted_attention = _attention(model.blocks[1])
            prepared_cache = shifted_attention._pangu_p2_tiled_region_ids
            del shifted_attention._pangu_p2_tiled_region_ids
            x = torch.zeros((1, 1, 32, 96), dtype=torch.float16)
            with self.assertRaisesRegex(RuntimeError, "cache is missing"):
                p2_adapter._cached_region_ids(shifted_attention, None, x)
            shifted_attention._pangu_p2_tiled_region_ids = prepared_cache
            p2_adapter.disable_p2_tiled_attention(model)

    def test_release_requires_precompute_and_backup_requires_release(self):
        model = ExactP2Model()
        with self.assertRaisesRegex(ValueError, "requires precompute_region_ids"):
            p2_adapter.enable_p2_tiled_attention(
                model,
                release_original_masks=True,
            )
        with self.assertRaisesRegex(ValueError, "requires release_original_masks"):
            p2_adapter.enable_p2_tiled_attention(
                model,
                retain_cpu_mask_backup=True,
            )


class HipPreparationTests(unittest.TestCase):
    def test_prepare_returns_fingerprint_and_config_without_forward(self):
        info = {
            "mode": "full-row-fast",
            "implementation_kind": "test",
            "config": {
                "q_tile": 16,
                "k_tile": 16,
                "head_dim": 32,
                "block_threads": 256,
            },
            "occupancy": {"active_blocks_per_multiprocessor": 8},
            "build": {"fingerprint": "abc123", "library": "/tmp/test.so"},
        }
        with mock.patch.object(
            hip_adapter,
            "get_hip_earth_attention_tiled_info",
            return_value=info,
        ) as get_info:
            prepared = hip_adapter.prepare_hip_earth_attention_tiled(
                device="cuda:0",
                mode="full-row-fast",
            )

        get_info.assert_called_once_with(device="cuda:0", mode="full-row-fast")
        self.assertEqual(prepared["fingerprint"], "abc123")
        self.assertEqual(prepared["config"], info["config"])

    def test_prepare_rejects_wrong_compiled_tile(self):
        info = {
            "mode": "full-row-fast",
            "implementation_kind": "test",
            "config": {
                "q_tile": 16,
                "k_tile": 32,
                "head_dim": 32,
                "block_threads": 256,
            },
            "occupancy": {"active_blocks_per_multiprocessor": 8},
            "build": {"fingerprint": "abc123", "library": "/tmp/test.so"},
        }
        with mock.patch.object(
            hip_adapter,
            "get_hip_earth_attention_tiled_info",
            return_value=info,
        ):
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                hip_adapter.prepare_hip_earth_attention_tiled(
                    device="cuda:0",
                    mode="full-row-fast",
                )


class InferenceWiringTests(unittest.TestCase):
    def test_promoted_region_and_experimental_prebuild_defaults_before_timer(self):
        source = (ROOT / "inference.py").read_text(encoding="utf-8")
        self.assertIn(
            'os.environ.setdefault("PANGU_P2_REGION_RELEASE", "1")',
            source,
        )
        self.assertIn(
            'os.environ.setdefault("PANGU_P2_PREBUILD_HIP", "0")',
            source,
        )
        self.assertIn("precompute_region_ids=p2_region_release", source)
        self.assertIn("release_original_masks=p2_region_release", source)
        setup = source.index("prepare_hip_earth_attention_tiled(")
        timer = source.index("start_time = time.perf_counter()")
        self.assertLess(setup, timer)


if __name__ == "__main__":
    unittest.main()
