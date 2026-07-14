import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
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


class EarthAttention3D(TinyAttention):
    pass


class EarthTransformer3DBlock(nn.Module):
    def forward(self, x):
        return x


class BufferHolder(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("attn_mask", mask.clone())
        self.register_buffer("earth_position_index", torch.arange(9))


class UVRuntimeSwitchTests(unittest.TestCase):
    def test_runtime_promotion_gates_enforce_plan_thresholds(self):
        rows = [
            {
                "kind": "baseline", "steady_latency_avg_ms": 100.0,
                "max_vram_mb": 500.0, "output_max_abs": 0.0, "env": {},
            },
            {
                "kind": "grid", "steady_latency_avg_ms": 93.9,
                "max_vram_mb": 509.0, "output_max_abs": 0.0,
                "env": {"PANGU_DISABLE_CUDA_GRAPH": "0"},
            },
            {
                "kind": "grid", "steady_latency_avg_ms": 96.9,
                "max_vram_mb": 500.0, "output_max_abs": 0.0,
                "env": {"PANGU_DIRECT_MASK_SLICE": "1"},
            },
        ]
        gated = rank_uv_candidates.add_runtime_gates(rows)
        self.assertEqual([row["promotion_gate"] for row in gated], ["baseline", "pass", "pass"])

    def test_runtime_ranking_uses_steady_latency(self):
        rows = [
            {
                "label": "cold-fast", "kind": "grid", "returncode": 0,
                "latency_avg_ms": 50.0, "steady_latency_avg_ms": 101.0,
                "max_vram_mb": 500.0, "output_max_abs": 0.0,
            },
            {
                "label": "steady-fast", "kind": "grid", "returncode": 0,
                "latency_avg_ms": 200.0, "steady_latency_avg_ms": 99.0,
                "max_vram_mb": 500.0, "output_max_abs": 0.0,
            },
        ]
        ranked = sorted(rows, key=rank_uv_candidates.sort_key)
        self.assertEqual(ranked[0]["label"], "steady-fast")

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
        self.assertEqual(env["PANGU_SCORED_ONLY_RECOVERY"], "0")
        self.assertEqual(env["PANGU_GLOBAL_MEAN_CORRECTION"], "0")
        self.assertEqual(env["PANGU_STREAM_WEIGHTS"], "0")
        self.assertEqual(env["PANGU_USE_ONNX"], "0")
        self.assertEqual(env["PANGU_COMPACT_ATTN_MASK"], "0")
        self.assertEqual(env["PANGU_DIRECT_MASK_SLICE"], "0")
        self.assertEqual(env["PANGU_GRAPH_DIRECT_INPUT"], "1")
        self.assertEqual(env["PANGU_CPU_RECOVERY_OUTPUT"], "0")
        self.assertEqual(env["PANGU_HIP_SCHEDULE_SPIN"], "0")
        self.assertEqual(env["PANGU_HIP_PREFER_L1"], "0")
        self.assertEqual(env["PANGU_HIP_STREAM_SPIN"], "0")
        self.assertEqual(env["PANGU_INTERN_IMMUTABLE_BUFFERS"], "1")

    def test_p2_tiled_preset_is_explicit_off_on_ab(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("p2-tiled"))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate["kind"] for candidate in candidates],
            ["baseline", "p2-tiled"],
        )
        self.assertEqual(
            [candidate["env"]["PANGU_P2_TILED_ATTENTION"] for candidate in candidates],
            ["0", "1"],
        )
        self.assertEqual(
            [candidate["env"]["PANGU_P2_TILED_MODE"] for candidate in candidates],
            ["online", "full-row-fast"],
        )
        self.assertEqual(len({candidate["label"] for candidate in candidates}), 2)
        for candidate in candidates:
            env = candidate["env"]
            self.assertEqual(env["PANGU_CHUNKED_ATTENTION"], "1")
            self.assertEqual(env["PANGU_INTERN_IMMUTABLE_BUFFERS"], "1")

    def test_hip_probe_isolates_each_control_and_combination(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("hip"))
        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0]["kind"], "baseline")
        self.assertTrue(
            all(
                candidate["env"]["PANGU_INTERN_IMMUTABLE_BUFFERS"] == "1"
                for candidate in candidates
            )
        )
        enabled = [
            tuple(
                candidate["env"][name]
                for name in (
                    "PANGU_HIP_SCHEDULE_SPIN",
                    "PANGU_HIP_PREFER_L1",
                    "PANGU_HIP_STREAM_SPIN",
                )
            )
            for candidate in candidates
        ]
        self.assertEqual(
            enabled,
            [
                ("0", "0", "0"),
                ("1", "0", "0"),
                ("0", "1", "0"),
                ("0", "0", "1"),
                ("1", "1", "1"),
            ],
        )

    def test_stagewise_probe_keeps_outer_stages_at_guardrail(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("stagewise"))
        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            env = candidate["env"]
            self.assertEqual(env["PANGU_INTERN_IMMUTABLE_BUFFERS"], "1")
            for stage in ("LAYER1", "LAYER4"):
                self.assertEqual(env[f"PANGU_ATTN_CHUNK_SIZE_{stage}"], "3")
                self.assertEqual(env[f"PANGU_CHUNKED_QKV_{stage}"], "1")
                self.assertEqual(env[f"PANGU_CHUNKED_PROJ_{stage}"], "1")
                self.assertEqual(env[f"PANGU_MLP_CHUNK_SIZE_{stage}"], "32768")

    def test_buffer_intern_probe_preserves_historical_off_on_ab(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("buffer-intern"))
        self.assertEqual(
            [candidate["env"]["PANGU_INTERN_IMMUTABLE_BUFFERS"] for candidate in candidates],
            ["0", "1"],
        )

    def test_stage_options_apply_only_to_named_stage(self):
        model = nn.Module()
        for stage in ("layer1", "layer2", "layer3", "layer4"):
            container = nn.Module()
            container.attn = EarthAttention3D()
            container.block = EarthTransformer3DBlock()
            setattr(model, stage, container)

        pangu_profile_model.enable_chunked_attention(
            model,
            chunk_size=3,
            chunked_qkv=True,
            chunked_proj=True,
            stage_options={
                "layer2": {
                    "chunk_size": 0,
                    "chunked_qkv": False,
                    "chunked_proj": False,
                }
            },
        )
        pangu_profile_model.enable_chunked_mlp(
            model,
            chunk_size=32768,
            stage_chunk_sizes={"layer2": 0, "layer3": 65536},
        )
        self.assertEqual(model.layer1.attn._pangu_attention_chunk_size, 3)
        self.assertEqual(model.layer2.attn._pangu_attention_chunk_size, 0)
        self.assertFalse(model.layer2.attn._pangu_chunked_qkv)
        self.assertFalse(model.layer2.attn._pangu_chunked_proj)
        self.assertEqual(model.layer2.block._pangu_mlp_chunk_size, 0)
        self.assertEqual(model.layer3.block._pangu_mlp_chunk_size, 65536)

    def test_immutable_buffer_interning_shares_only_equal_buffers(self):
        model = nn.Module()
        model.layer1 = BufferHolder(torch.tensor([0.0, -100.0]))
        model.layer2 = BufferHolder(torch.tensor([0.0, -100.0]))
        model.layer3 = BufferHolder(torch.tensor([0.0, 0.0]))
        layer3_mask_pointer = model.layer3.attn_mask.data_ptr()

        report = pangu_profile_model.intern_immutable_buffers(model)

        self.assertGreaterEqual(report["replaced"], 2)
        self.assertEqual(
            model.layer1.attn_mask.data_ptr(), model.layer2.attn_mask.data_ptr()
        )
        self.assertNotEqual(model.layer1.attn_mask.data_ptr(), layer3_mask_pointer)
        self.assertEqual(
            model.layer1.earth_position_index.data_ptr(),
            model.layer3.earth_position_index.data_ptr(),
        )
        before = model.layer1.attn_mask.clone()
        _ = model.layer2.attn_mask + 1
        self.assertTrue(torch.equal(model.layer1.attn_mask, before))

    def test_cpu_recovery_preset_is_isolated_off_on_ab(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("cpu-recovery"))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate["env"]["PANGU_CPU_RECOVERY_OUTPUT"] for candidate in candidates],
            ["0", "1"],
        )
        for candidate in candidates:
            self.assertEqual(candidate["env"]["PANGU_DISABLE_CUDA_GRAPH"], "1")
            self.assertEqual(candidate["env"]["PANGU_COMPACT_ATTN_MASK"], "0")

    def test_pangu_lite_2d_preset_disables_3d_runtime_patches(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("pangu-lite-2d"))
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["kind"], "architecture")
        self.assertEqual(candidate["label"], "pangu_lite_2d_pos288_reset1")
        env = candidate["env"]
        self.assertEqual(
            env["PANGU_MODEL_ARCHITECTURE"],
            "PanguLite2DAttentionPosEmbed",
        )
        self.assertEqual(env["PANGU_LAYERWISE_INFERENCE"], "0")
        self.assertEqual(env["PANGU_DIRECT_RECOVERY"], "0")
        self.assertEqual(env["PANGU_CHUNKED_ATTENTION"], "0")
        self.assertEqual(env["PANGU_CHUNKED_MLP"], "0")
        self.assertEqual(env["PANGU_RESET_PEAK_AFTER_LOAD"], "1")

    def test_pangu_lite_2d_missing_checkpoint_fails_before_inference_fallback(self):
        candidate = list(probe_uv_runtime_sweep.iter_candidates("pangu-lite-2d"))[0]
        candidate["env"] = dict(candidate["env"])
        candidate["env"]["PANGU_FP16_CHECKPOINT"] = "missing_2d_checkpoint.pth"
        with tempfile.TemporaryDirectory() as tmp:
            result = probe_uv_runtime_sweep.run_one(
                candidate,
                args=Namespace(max_batches=1, repeat=1, python=sys.executable, fp16_checkpoint=None),
                pangu_dir=Path(tmp),
                output_dir=Path(tmp) / "output",
                baseline_dir=None,
            )
        self.assertEqual(result["returncode"], 2)
        self.assertIn("2D architecture checkpoint not found", result["error"])
        self.assertIn("Refusing to fall back", result["stdout_tail"])

    def test_submission_defaults_reject_platform_negative_graph_candidate(self):
        source = (PANGU_DIR / "inference.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.setdefault("PANGU_DISABLE_CUDA_GRAPH", "1")', source)
        self.assertIn('os.environ.setdefault("PANGU_LAYERWISE_CUDA_GRAPH", "0")', source)
        self.assertIn('os.environ.setdefault("PANGU_GRAPH_DIRECT_INPUT", "1")', source)
        self.assertIn('os.environ.setdefault("PANGU_COMPACT_ATTN_MASK", "0")', source)
        self.assertIn('os.environ.setdefault("PANGU_DIRECT_MASK_SLICE", "0")', source)
        self.assertIn('os.environ.setdefault("PANGU_HIP_SCHEDULE_SPIN", "0")', source)
        self.assertIn('os.environ.setdefault("PANGU_HIP_PREFER_L1", "0")', source)
        self.assertIn('os.environ.setdefault("PANGU_HIP_STREAM_SPIN", "0")', source)
        self.assertIn(
            'os.environ.setdefault("PANGU_INTERN_IMMUTABLE_BUFFERS", "1")',
            source,
        )
        self.assertLess(
            source.index("model.eval()"),
            source.index("intern_report = intern_immutable_buffers(model)"),
        )

    def test_direct_mask_preset_is_isolated_off_on_ab(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("direct-mask"))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate["env"]["PANGU_DIRECT_MASK_SLICE"] for candidate in candidates],
            ["0", "1"],
        )
        for candidate in candidates:
            self.assertEqual(candidate["env"]["PANGU_COMPACT_ATTN_MASK"], "0")
            self.assertEqual(candidate["env"]["PANGU_DISABLE_CUDA_GRAPH"], "1")

    def test_cuda_graph_preset_is_isolated_off_on_ab(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("cuda-graph"))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate["env"]["PANGU_DISABLE_CUDA_GRAPH"] for candidate in candidates],
            ["1", "0"],
        )
        self.assertEqual(candidates[1]["env"]["PANGU_LAYERWISE_CUDA_GRAPH"], "1")
        for candidate in candidates:
            self.assertEqual(candidate["env"]["PANGU_DIRECT_MASK_SLICE"], "0")
            self.assertEqual(candidate["env"]["PANGU_COMPACT_ATTN_MASK"], "0")
            self.assertEqual(candidate["env"]["PANGU_GRAPH_DIRECT_INPUT"], "1")

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

    def test_full_recovery_preset_only_sweeps_width(self):
        candidates = list(probe_uv_runtime_sweep.iter_candidates("full-recovery"))
        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            [candidate["env"]["PANGU_DIRECT_RECOVERY_WIDTH_CHUNK"]
             for candidate in candidates],
            ["16", "24", "32", "48"],
        )
        self.assertEqual(len({candidate["label"] for candidate in candidates}), 4)
        for candidate in candidates:
            env = candidate["env"]
            self.assertEqual(env["PANGU_SCORED_ONLY_RECOVERY"], "0")
            self.assertEqual(env["PANGU_ATTN_CHUNK_SIZE"], "3")
            self.assertEqual(env["PANGU_MLP_CHUNK_SIZE"], "32768")
            self.assertEqual(env["PANGU_COMPACT_ATTN_MASK"], "0")
            self.assertEqual(env["PANGU_GLOBAL_MEAN_CORRECTION"], "0")
            self.assertEqual(env["PANGU_STREAM_WEIGHTS"], "0")
            self.assertEqual(env["PANGU_USE_ONNX"], "0")

    def test_checkpoint_ab_uses_distinct_explicit_files(self):
        candidates = probe_uv_runtime_sweep.checkpoint_ab_candidates(
            list(probe_uv_runtime_sweep.iter_candidates("baseline")),
            "model_fp16.pth",
            "model_fp16_alias_compact.pth",
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["kind"], "baseline")
        self.assertEqual(candidates[1]["kind"], "checkpoint_candidate")
        self.assertEqual(
            [candidate["env"]["PANGU_FP16_CHECKPOINT"] for candidate in candidates],
            ["model_fp16.pth", "model_fp16_alias_compact.pth"],
        )
        self.assertNotEqual(candidates[0]["label"], candidates[1]["label"])

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

    def test_direct_mask_slicing_matches_index_select_across_wrap(self):
        torch.manual_seed(20260711)
        base = TinyAttention()
        direct = TinyAttention()
        direct.load_state_dict(base.state_dict())
        x = torch.randn(7, 2, 3, 4)
        mask = torch.zeros(4, 2, 3, 3)
        mask[:, :, 0, 2] = torch.arange(4).view(4, 1)
        for module, enabled in ((base, False), (direct, True)):
            module._pangu_attention_chunk_size = 3
            module._pangu_cache_earth_bias = False
            module._pangu_chunked_qkv = True
            module._pangu_chunked_proj = True
            module._pangu_direct_mask_slice = enabled
        expected = pangu_profile_model._forward_chunked_earth_attention_3d(base, x, mask)
        actual = pangu_profile_model._forward_chunked_earth_attention_3d(direct, x, mask)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_attention_chunk_means_one_full_batch_chunk(self):
        torch.manual_seed(20260713)
        chunked = TinyAttention()
        full = TinyAttention()
        full.load_state_dict(chunked.state_dict())
        x = torch.randn(7, 2, 3, 4)
        for module, chunk_size in ((chunked, 3), (full, 0)):
            module._pangu_attention_chunk_size = chunk_size
            module._pangu_cache_earth_bias = False
            module._pangu_chunked_qkv = True
            module._pangu_chunked_proj = True
            module._pangu_direct_mask_slice = False
        expected = pangu_profile_model._forward_chunked_earth_attention_3d(chunked, x)
        actual = pangu_profile_model._forward_chunked_earth_attention_3d(full, x)
        self.assertTrue(torch.equal(actual, expected))

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
