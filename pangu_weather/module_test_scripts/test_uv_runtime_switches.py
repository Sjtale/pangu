import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PANGU_DIR = Path(__file__).resolve().parents[1]


def _install_onescience_stubs():
    try:
        import onescience
        import onescience.models.pangu
        import onescience.modules
        return
    except ImportError:
        pass

    onescience = types.ModuleType("onescience")
    models = types.ModuleType("onescience.models")
    pangu = types.ModuleType("onescience.models.pangu")
    modules = types.ModuleType("onescience.modules")
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

    sys.modules.setdefault("onescience", onescience)
    sys.modules.setdefault("onescience.models", models)
    sys.modules.setdefault("onescience.models.pangu", pangu)
    sys.modules.setdefault("onescience.modules", modules)
    sys.modules.setdefault("onescience.modules.attention", attention)
    sys.modules.setdefault("onescience.modules.attention.earthattention3d", earthattention3d)


def _load_script(name):
    path = PANGU_DIR / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_install_onescience_stubs()
pangu_profile_model = importlib.import_module("pangu_weather.pangu_profile_model")
probe_uv_runtime_sweep = _load_script("probe_uv_runtime_sweep")
rank_uv_candidates = _load_script("rank_uv_candidates")


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
        self.earth_position_bias_table = nn.Parameter(torch.randn(table_len, self.num_heads))
        self.register_buffer("earth_position_index", torch.arange(table_len))


class UVRuntimeSwitchTests(unittest.TestCase):
    def test_default_uv_probe_only_measures_baseline(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates())
        self.assertEqual(len(candidates), 1)
        env = candidates[0]["env"]
        self.assertEqual(candidates[0]["kind"], "baseline")
        self.assertEqual(env["PANGU_DIRECT_RECOVERY_WIDTH_CHUNK"], "16")
        self.assertEqual(env["PANGU_ATTN_CHUNK_SIZE"], "3")
        self.assertEqual(env["PANGU_MLP_CHUNK_SIZE"], "32768")
        self.assertEqual(env["PANGU_SPLIT_RECOVERY"], "0")
        self.assertEqual(env["PANGU_CACHE_EARTH_BIAS"], "0")
        self.assertEqual(env["PANGU_LAYERWISE_EMPTY_CACHE"], "0")
        self.assertEqual(env["PANGU_INPLACE_BLOCK"], "1")
        self.assertEqual(env["PANGU_CLEAR_INPUT_REFS"], "1")
        self.assertEqual(env["PANGU_GLOBAL_MEAN_CORRECTION"], "0")
        self.assertEqual(env["PANGU_STREAM_WEIGHTS"], "0")
        self.assertEqual(env["PANGU_USE_ONNX"], "0")
        self.assertEqual(env["PANGU_COMPACT_ATTN_MASK"], "0")

    def test_compact_mask_preset_is_isolated_off_on_ab(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("compact-mask"))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate["env"]["PANGU_COMPACT_ATTN_MASK"] for candidate in candidates],
            ["0", "1"],
        )
        self.assertEqual(len({candidate["label"] for candidate in candidates}), 2)
        for candidate in candidates:
            env = candidate["env"]
            self.assertEqual(env["PANGU_ATTN_CHUNK_SIZE"], "3")
            self.assertEqual(env["PANGU_CHUNKED_QKV"], "1")
            self.assertEqual(env["PANGU_CHUNKED_PROJ"], "1")
            self.assertEqual(env["PANGU_GLOBAL_MEAN_CORRECTION"], "0")
            self.assertEqual(env["PANGU_STREAM_WEIGHTS"], "0")
            self.assertEqual(env["PANGU_USE_ONNX"], "0")

    def test_focused_uv_grid_only_sweeps_attention_chunk_when_explicit(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("focused"))
        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            {candidate["env"]["PANGU_ATTN_CHUNK_SIZE"] for candidate in candidates},
            {"3", "4", "5"},
        )
        for candidate in candidates:
            env = candidate["env"]
            self.assertEqual(env["PANGU_DIRECT_RECOVERY_WIDTH_CHUNK"], "16")
            self.assertEqual(env["PANGU_MLP_CHUNK_SIZE"], "32768")
            self.assertEqual(env["PANGU_SPLIT_RECOVERY"], "0")
            self.assertEqual(env["PANGU_CACHE_EARTH_BIAS"], "0")
            self.assertEqual(env["PANGU_LAYERWISE_EMPTY_CACHE"], "0")
            self.assertEqual(env["PANGU_INPLACE_BLOCK"], "1")
            self.assertEqual(env["PANGU_CLEAR_INPUT_REFS"], "1")
            self.assertEqual(env["PANGU_GLOBAL_MEAN_CORRECTION"], "0")
            self.assertEqual(env["PANGU_STREAM_WEIGHTS"], "0")
            self.assertEqual(env["PANGU_USE_ONNX"], "0")
            self.assertNotEqual(env["PANGU_ATTN_CHUNK_SIZE"], "2")

    def test_full_uv_grid_still_blocks_rejected_switches(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("full"))
        self.assertGreater(len(candidates), 3)
        for candidate in candidates:
            env = candidate["env"]
            self.assertEqual(env["PANGU_GLOBAL_MEAN_CORRECTION"], "0")
            self.assertEqual(env["PANGU_STREAM_WEIGHTS"], "0")
            self.assertEqual(env["PANGU_USE_ONNX"], "0")
            self.assertNotEqual(env["PANGU_ATTN_CHUNK_SIZE"], "2")

    def test_attention_cache_qkv_proj_combinations_match_baseline(self):
        torch.manual_seed(20260709)
        base = TinyAttention()
        x = torch.randn(5, 2, 3, 4)

        base._pangu_attention_chunk_size = 2
        base._pangu_cache_earth_bias = False
        base._pangu_chunked_qkv = False
        base._pangu_chunked_proj = False
        expected = pangu_profile_model._forward_chunked_earth_attention_3d(base, x)

        for cache_bias in (False, True):
            for chunked_qkv in (False, True):
                for chunked_proj in (False, True):
                    module = TinyAttention()
                    module.load_state_dict(base.state_dict())
                    module._pangu_attention_chunk_size = 2
                    module._pangu_cache_earth_bias = cache_bias
                    module._pangu_chunked_qkv = chunked_qkv
                    module._pangu_chunked_proj = chunked_proj
                    actual = pangu_profile_model._forward_chunked_earth_attention_3d(module, x)
                    self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
                    if cache_bias:
                        cached = module._cached_earth_position_bias
                        again = pangu_profile_model._forward_chunked_earth_attention_3d(module, x)
                        self.assertIs(module._cached_earth_position_bias, cached)
                        self.assertTrue(torch.allclose(again, expected, atol=1e-6, rtol=1e-6))

    def test_compact_attention_mask_is_exact_and_smaller(self):
        torch.manual_seed(20260710)
        base = TinyAttention().half()
        compact = TinyAttention()
        compact.load_state_dict(base.state_dict())

        float_mask = torch.zeros(3, 2, 3, 3, dtype=torch.float16)
        float_mask[:, :, 0, 2] = -100
        base.register_buffer("attn_mask", float_mask.clone())
        compact.register_buffer("attn_mask", float_mask.float())

        compacted, saved_bytes = pangu_profile_model.compact_attention_masks(compact)
        compact.half()

        self.assertEqual(compacted, 1)
        self.assertGreater(saved_bytes, 0)
        self.assertEqual(compact.attn_mask.dtype, torch.int8)
        self.assertLess(
            compact.attn_mask.numel() * compact.attn_mask.element_size(),
            base.attn_mask.numel() * base.attn_mask.element_size(),
        )

        x = torch.randn(6, 2, 3, 4, dtype=torch.float16)
        for module in (base, compact):
            module._pangu_attention_chunk_size = 2
            module._pangu_cache_earth_bias = False
            module._pangu_chunked_qkv = True
            module._pangu_chunked_proj = True

        expected = pangu_profile_model._forward_chunked_earth_attention_3d(
            base, x, mask=base.attn_mask
        )
        actual = pangu_profile_model._forward_chunked_earth_attention_3d(
            compact, x, mask=compact.attn_mask
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_ranker_prefers_platform_total_over_local_proxy(self):
        rows = [
            {"label": "fast_local", "returncode": 0, "max_vram_mb": 500.0, "latency_avg_ms": 80.0},
            {"label": "platform_best", "returncode": 0, "max_vram_mb": 700.0, "latency_avg_ms": 120.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "sweep.jsonl"
            csv_path = Path(tmp) / "platform.csv"
            jsonl.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            csv_path.write_text(
                "label,total,u,v,w\nplatform_best,90.1,36.0,18.0,36.1\n",
                encoding="utf-8",
            )
            merged = rank_uv_candidates.merge_platform(
                rank_uv_candidates.load_jsonl(jsonl),
                rank_uv_candidates.load_platform_csv(csv_path),
            )
            ranked = sorted(merged, key=rank_uv_candidates.sort_key)
            self.assertEqual(ranked[0]["label"], "platform_best")


if __name__ == "__main__":
    unittest.main()
