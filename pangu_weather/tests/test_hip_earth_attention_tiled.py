"""Static and CPU-side gates for the isolated pruned_96 tiled HIP path."""

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path
from unittest import mock


PANGU_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = PANGU_ROOT / "hip_kernels" / "earth_attention_tiled_fwd.hip"
WRAPPER_PATH = PANGU_ROOT / "hip_earth_attention_tiled.py"
PROBE_PATH = PANGU_ROOT / "scripts" / "probe_hip_earth_attention_tiled.py"
MFMA_PROBE_PATH = PANGU_ROOT / "scripts" / "probe_hip_mfma_capability.py"

try:
    import torch
except ImportError:  # Local source-only validation does not require PyTorch.
    torch = None


class TiledHipSourceTests(unittest.TestCase):
    def test_new_path_is_complete_and_isolated(self):
        self.assertTrue(KERNEL_PATH.is_file())
        self.assertTrue(WRAPPER_PATH.is_file())
        self.assertTrue(PROBE_PATH.is_file())
        self.assertTrue(MFMA_PROBE_PATH.is_file())
        inference_source = (PANGU_ROOT / "inference.py").read_text(encoding="utf-8")
        self.assertIn("PANGU_P2_TILED_ATTENTION", inference_source)
        self.assertIn('if _is_enabled("PANGU_P2_TILED_ATTENTION"):', inference_source)
        self.assertIn(
            'os.environ.setdefault("PANGU_P2_TILED_ATTENTION", "1")',
            inference_source,
        )
        self.assertIn(
            'os.environ.setdefault("PANGU_P2_TILED_MODE", "full-row-fast")',
            inference_source,
        )
        self.assertIn(
            'os.environ.setdefault("PANGU_P2_FULL_WIDTH", "1")',
            inference_source,
        )
        self.assertIn(
            'os.environ.setdefault("PANGU_TILED_HIP_ARCH", '
            '"gfx936:sramecc+:xnack-")',
            inference_source,
        )
        self.assertIn(
            'os.environ.setdefault("PANGU_TILED_HIP_BUILD_DIR", '
            '"/tmp/pangu_tiled_hip_p2_full")',
            inference_source,
        )
        self.assertIn(
            'model_profile.get("name") != "pgw_lite_pruned_96"',
            inference_source,
        )
        self.assertIn('"PANGU_P2_TILED_MODE"', inference_source)
        self.assertIn("kernel_mode=p2_kernel_mode", inference_source)
        self.assertIn("force_full_width=p2_full_width", inference_source)
        profile_source = (PANGU_ROOT / "pangu_profile_model.py").read_text(encoding="utf-8")
        self.assertNotIn("PANGU_P2_TILED_ATTENTION", profile_source)

    def test_kernel_has_tiled_online_softmax_and_compact_inputs(self):
        source = KERNEL_PATH.read_text(encoding="utf-8")
        required = (
            "pangu_earth_attention_tiled_fwd_fp16",
            "pangu_earth_attention_tiled_get_config",
            "pangu_earth_attention_tiled_get_occupancy",
            "pangu_earth_attention_tiled_implementation_kind",
            "position_index",
            "region_id",
            "running_max",
            "running_sum",
            "hipLaunchKernelGGL",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertIn("kQueryTile = 16", source)
        self.assertIn("kKeyTile = 16", source)
        self.assertIn("kBlockThreads = 256", source)
        self.assertIn("typedef _Float16 my_half4", source)
        self.assertIn("typedef float my_float4", source)
        self.assertIn("__builtin_amdgcn_mmac_f32_16x16x16f16", source)
        self.assertIn('"matrix_tiled_mmac_online_fp32_gfx936"', source)
        self.assertIn("score_shared[row * 16 + col] = d_0[i]", source)
        self.assertIn("score_shared[row * 16 + col] += d_1[i]", source)
        self.assertIn("std::int16_t", source)
        self.assertNotIn("float* scores", source)
        self.assertNotIn("__builtin_amdgcn_mfma", source)
        self.assertNotIn("std::uint16_t", source)
        self.assertIn("    return;\n}\n\n} // namespace", source)

    def test_kernel_has_isolated_full_row_fp16_parity_path(self):
        source = KERNEL_PATH.read_text(encoding="utf-8")
        required = (
            "earth_attention_tiled_full_row_fwd_fp16_kernel",
            "pangu_earth_attention_tiled_full_row_fwd_fp16",
            "pangu_earth_attention_tiled_full_row_diagnostic_fp16",
            "pangu_earth_attention_tiled_full_row_get_config",
            "pangu_earth_attention_tiled_full_row_get_occupancy",
            "pangu_earth_attention_tiled_full_row_implementation_kind",
            "kParityDynamicSharedBytes == 6656",
            "kParityScoreElements = kQueryTile * kMaxTokens",
            "full_row_softmax<kStandardExp, 64, 4>",
            "full_row_softmax<kStandardExp, 32, 1>",
            "__shfl_xor",
            "elements[iteration] / sum);\n        }\n    }\n    return;",
            "full_scores[row * kMaxTokens + key_index] = rounded_score",
            "__half2float(rounded_score) - 100.0f",
            "mmac_f32_16x16x16f16(a_low, b_low, d)",
            "mmac_f32_16x16x16f16(a_high, b_high, d)",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertIn(
            '"matrix_tiled_mmac_full_row_fp16_fast_exp_gfx936"', source
        )
        self.assertIn(
            '"matrix_tiled_mmac_full_row_fp16_standard_exp_gfx936"', source
        )
        full_row_source = source.split(
            "earth_attention_tiled_full_row_fwd_fp16_kernel", 1
        )[1].split("float pv_acc_0_x", 1)[0]
        self.assertNotIn("qk_scratch[row * 16 + col_base + item] +=", full_row_source)

    def test_wrapper_fingerprints_build_and_uses_current_stream(self):
        source = WRAPPER_PATH.read_text(encoding="utf-8")
        required = (
            "hashlib",
            "subprocess",
            "fixed_flags",
            "PANGU_TILED_HIP_EXTRA_FLAGS",
            "PANGU_TILED_HIP_BUILD_DIR",
            "torch.cuda.current_stream",
            "os.replace",
            "_ACTIVE_LIBRARY",
            "pack_earth_bias_table",
            "compact_earth_position_index",
            "shifted_mask_to_region_ids",
            "_require_registered_position_index",
            "get_hip_earth_attention_tiled_info",
            "full-row-fast",
            "full-row-expf",
            "pangu_earth_attention_tiled_full_row_fwd_fp16",
            "pangu_earth_attention_tiled_full_row_diagnostic_fp16",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertNotIn("torch.cuda.synchronize", source)
        self.assertIn("torch.int16", source)
        self.assertNotIn("torch.uint16", source)
        load_source = source.split("def _load_library", 1)[1].split(
            "def _error_detail", 1
        )[0]
        self.assertIn("if _ACTIVE_LIBRARY is not None", load_source)
        self.assertLess(
            load_source.index("if _ACTIVE_LIBRARY is not None"),
            load_source.index("build_hip_earth_attention_tiled()"),
        )
        validation_source = source.split("def _validate_forward_inputs", 1)[1].split(
            "@torch.no_grad", 1
        )[0]
        self.assertNotIn(".item()", validation_source)

        adapter_source = (PANGU_ROOT / "p2_tiled_attention.py").read_text(
            encoding="utf-8"
        )
        for token in (
            'kernel_mode="online"',
            "mode=self._pangu_p2_tiled_kernel_mode",
            '"pre_projection"',
            '"post_projection"',
            '"[P2_AUDIT] "',
            "register_forward_pre_hook",
            "force_full_width=False",
            "_pangu_p2_full_width",
        ):
            with self.subTest(adapter_token=token):
                self.assertIn(token, adapter_source)

    def test_probe_uses_exact_pruned_shapes_and_strict_gates(self):
        source = PROBE_PATH.read_text(encoding="utf-8")
        required = (
            '"shallow_l144_shifted"',
            '"deep_l144_shifted"',
            '"pressure_height": 64',
            '"pressure_height": 32',
            '"heads": 3',
            '"heads": 6',
            '"input_resolution": [8, 96, 36]',
            '"input_resolution": [8, 48, 36]',
            "torch.cuda.Event",
            "gpu_median_ms",
            "gpu_p90_ms",
            "MIN_REPRESENTATIVE_SPEEDUP = 1.5",
            "samples * args.launches_per_sample < 100",
            "open(\"x\"",
            "preexec_fn=_disable_core_dumps",
            "--allow-extra-flags",
            "PANGU_TILED_HIP_EXTRA_FLAGS is non-empty",
            'if args.allow_extra_flags:',
            'command.append("--allow-extra-flags")',
            '"candidate_vs_eager_half"',
            '"candidate_vs_reference_fp32"',
            '"eager_half_vs_reference_fp32"',
            '"production_exact"',
            '"--kernel-mode"',
            '"--diagnostic-stages"',
            '"stage_diagnostics"',
            '"exp_variant_comparison"',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertIn("qkv_heads[0] * scale", source)
        self.assertNotIn("scores = torch.matmul(q.float(), k.float()) * scale", source)
        main_source = source.split("def main():", 1)[1]
        self.assertLess(
            main_source.index("compile_seconds = time.perf_counter() - started"),
            main_source.index("results = []"),
        )

    def test_mfma_probe_is_explicit_read_only_and_non_claiming(self):
        source = MFMA_PROBE_PATH.read_text(encoding="utf-8")
        required = (
            "--candidate-builtin",
            "--arch",
            "IDENTIFIER_RE.fullmatch",
            "ARCH_RE.fullmatch",
            "tempfile.TemporaryDirectory",
            "shell=False",
            '"status": "unconfirmed"',
            '"intrinsic_call_compiled": False',
            '"isa_emission_confirmed": False',
            'path.open("x"',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertNotIn("pip install", source)
        self.assertNotIn("shell=True", source)


class OnlineSoftmaxMathTests(unittest.TestCase):
    def test_tile16_recurrence_matches_direct_softmax_for_supported_lengths(self):
        generator = random.Random(17)
        for tokens in (32, 144):
            with self.subTest(tokens=tokens):
                scores = [generator.uniform(-8.0, 8.0) for _ in range(tokens)]
                values = [
                    [generator.uniform(-2.0, 2.0) for _ in range(4)]
                    for _ in range(tokens)
                ]
                direct_max = max(scores)
                direct_weights = [math.exp(score - direct_max) for score in scores]
                direct_sum = sum(direct_weights)
                direct = [
                    sum(weight * value[dim] for weight, value in zip(direct_weights, values))
                    / direct_sum
                    for dim in range(4)
                ]

                running_max = -math.inf
                running_sum = 0.0
                running_output = [0.0] * 4
                for key_start in range(0, tokens, 16):
                    tile_scores = scores[key_start : key_start + 16]
                    new_max = max(running_max, max(tile_scores))
                    alpha = (
                        math.exp(running_max - new_max)
                        if math.isfinite(running_max)
                        else 0.0
                    )
                    probabilities = [
                        math.exp(score - new_max) for score in tile_scores
                    ]
                    running_sum = running_sum * alpha + sum(probabilities)
                    for dim in range(4):
                        running_output[dim] = running_output[dim] * alpha + sum(
                            probability * values[key_start + index][dim]
                            for index, probability in enumerate(probabilities)
                        )
                    running_max = new_max
                online = [value / running_sum for value in running_output]
                for expected, actual in zip(direct, online):
                    self.assertAlmostEqual(expected, actual, places=12)


class FullRowSoftmaxLayoutTests(unittest.TestCase):
    def test_l144_wave64_lane_plan_covers_every_element_once(self):
        indices = [lane + iteration * 64 for lane in range(64) for iteration in range(4)]
        valid = sorted(index for index in indices if index < 144)
        self.assertEqual(valid, list(range(144)))

    def test_l32_logical_wave32_batch_plan_covers_all_rows(self):
        rows = [logical_group * 2 + batch for logical_group in range(8) for batch in range(2)]
        self.assertEqual(rows, list(range(16)))
        indices = [lane for lane in range(32)]
        self.assertEqual(indices, list(range(32)))


@unittest.skipIf(torch is None, "PyTorch is unavailable in this source-only environment")
class TiledHipCpuHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(PANGU_ROOT))
        import hip_earth_attention_tiled as wrapper
        import p2_tiled_attention as adapter

        cls.adapter = adapter
        cls.wrapper = wrapper
        spec = importlib.util.spec_from_file_location("tiled_probe", PROBE_PATH)
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    @classmethod
    def tearDownClass(cls):
        if sys.path and sys.path[0] == str(PANGU_ROOT):
            sys.path.pop(0)

    def test_bias_pack_preserves_all_values(self):
        table = torch.arange(7 * 3 * 2, dtype=torch.float16).view(7, 3, 2)
        packed = self.wrapper.pack_earth_bias_table(table)
        self.assertEqual(tuple(packed.shape), (2, 3, 7))
        self.assertTrue(packed.is_contiguous())
        self.assertTrue(torch.equal(packed, table.permute(2, 1, 0)))

    def test_library_load_reuses_process_cache_before_build(self):
        original_library = self.wrapper._ACTIVE_LIBRARY
        original_path = self.wrapper._ACTIVE_LIBRARY_PATH
        sentinel_library = object()
        sentinel_path = Path("/tmp/libpangu-tiled-cache-test.so")
        try:
            self.wrapper._ACTIVE_LIBRARY = sentinel_library
            self.wrapper._ACTIVE_LIBRARY_PATH = sentinel_path
            with mock.patch.object(
                self.wrapper,
                "build_hip_earth_attention_tiled",
                side_effect=AssertionError("cached load must not rebuild"),
            ):
                library, library_path = self.wrapper._load_library()
            self.assertIs(library, sentinel_library)
            self.assertEqual(library_path, sentinel_path)
        finally:
            self.wrapper._ACTIVE_LIBRARY = original_library
            self.wrapper._ACTIVE_LIBRARY_PATH = original_path

    def test_kernel_mode_is_explicit_and_online_remains_default(self):
        self.assertEqual(
            self.wrapper._resolve_kernel_mode("online"),
            ("online", 0),
        )
        self.assertEqual(
            self.wrapper._resolve_kernel_mode("full-row-fast"),
            ("full-row-fast", 0),
        )
        self.assertEqual(
            self.wrapper._resolve_kernel_mode("full-row-expf"),
            ("full-row-expf", 1),
        )
        with self.assertRaises(ValueError):
            self.wrapper._resolve_kernel_mode("unknown")

    def test_debug_capture_reassembles_chunked_projection_inputs(self):
        projection = torch.nn.Linear(4, 4, bias=False)
        x = torch.arange(15 * 4, dtype=torch.float32).reshape(15, 4)

        def original_forward(value, _mask):
            return torch.cat(
                [projection(value[start : start + 3]) for start in range(0, 15, 3)],
                dim=0,
            )

        captured, reference = self.adapter._capture_reference_projection_input(
            projection,
            original_forward,
            x,
            None,
        )

        self.assertTrue(torch.equal(captured, x))
        self.assertTrue(torch.equal(reference, original_forward(x, None)))
        self.assertEqual(len(projection._forward_pre_hooks), 0)

    def test_adapter_chunks_qkv_projection_and_tracks_mask_width_offset(self):
        class RecordingQkv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.widths = []

            def forward(self, value):
                self.widths.append(value.shape[0])
                return torch.cat((value, value, value), dim=-1)

        class RecordingProjection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.widths = []

            def forward(self, value):
                self.widths.append(value.shape[0])
                return value + 1

        class Module:
            pass

        module = Module()
        module.num_heads = 2
        module.scale = 0.5
        module.qkv = RecordingQkv()
        module.proj = RecordingProjection()
        module.proj_drop = torch.nn.Identity()
        module._pangu_attention_chunk_size = 2
        module._pangu_chunked_qkv = True
        module._pangu_chunked_proj = True
        module._pangu_p2_tiled_kernel_mode = "full-row-fast"
        x = torch.arange(5 * 2 * 64, dtype=torch.float32).reshape(5, 1, 2, 64)
        offsets = []

        def tiled_forward(qkv, *_args, width_offset, **_kwargs):
            offsets.append(width_offset)
            return qkv[:, :, :, 2].reshape(qkv.shape[0], 1, 2, 64)

        result, captured = self.adapter._run_p2_tiled_chunks(
            module,
            x,
            packed_bias=object(),
            position_index=object(),
            region_ids=object(),
            mask_width=3,
            tiled_forward=tiled_forward,
            capture_attention_output=True,
        )

        self.assertEqual(module.qkv.widths, [2, 2, 1])
        self.assertEqual(module.proj.widths, [2, 2, 1])
        self.assertEqual(offsets, [0, 2, 4])
        self.assertTrue(torch.equal(captured, x))
        self.assertTrue(torch.equal(result, x + 1))

        offsets.clear()
        self.adapter._run_p2_tiled_chunks(
            module,
            x,
            packed_bias=object(),
            position_index=object(),
            region_ids=None,
            mask_width=None,
            tiled_forward=tiled_forward,
            capture_attention_output=False,
        )
        self.assertEqual(offsets, [0, 0, 0])

        module.qkv.widths.clear()
        module.proj.widths.clear()
        module._pangu_p2_full_width = True
        offsets.clear()
        result, captured = self.adapter._run_p2_tiled_chunks(
            module,
            x,
            packed_bias=object(),
            position_index=object(),
            region_ids=object(),
            mask_width=3,
            tiled_forward=tiled_forward,
            capture_attention_output=True,
        )
        self.assertEqual(module.qkv.widths, [5])
        self.assertEqual(module.proj.widths, [5])
        self.assertEqual(offsets, [0])
        self.assertTrue(torch.equal(captured, x))
        self.assertTrue(torch.equal(result, x + 1))

        module.qkv.widths.clear()
        module.proj.widths.clear()
        module._pangu_p2_full_width = False
        module._pangu_chunked_qkv = False
        module._pangu_chunked_proj = False
        offsets.clear()
        result, captured = self.adapter._run_p2_tiled_chunks(
            module,
            x,
            packed_bias=object(),
            position_index=object(),
            region_ids=object(),
            mask_width=3,
            tiled_forward=tiled_forward,
            capture_attention_output=True,
        )
        self.assertEqual(module.qkv.widths, [5])
        self.assertEqual(module.proj.widths, [5])
        self.assertEqual(offsets, [0])
        self.assertTrue(torch.equal(captured, x))
        self.assertTrue(torch.equal(result, x + 1))

    def test_position_index_compacts_losslessly(self):
        index = self.probe._position_index([2, 6, 12])
        compact = self.wrapper.compact_earth_position_index(index, 3312)
        self.assertEqual(compact.dtype, torch.int16)
        self.assertTrue(compact.is_contiguous())
        self.assertTrue(torch.equal(compact.to(torch.int64), index))
        with self.assertRaises(ValueError):
            self.wrapper.compact_earth_position_index(index + 4000, 3312)

    def test_dense_shift_mask_round_trips_to_region_equivalence(self):
        labels = self.probe._shift_region_ids([8, 48, 36], [2, 6, 12])
        dense = (labels.unsqueeze(-1) != labels.unsqueeze(-2)).to(torch.float16)
        dense.mul_(-100.0)
        recovered = self.wrapper.shifted_mask_to_region_ids(dense)
        reconstructed = (
            recovered.unsqueeze(-1) != recovered.unsqueeze(-2)
        ).to(torch.float16)
        reconstructed.mul_(-100.0)
        self.assertEqual(recovered.dtype, torch.uint8)
        self.assertTrue(recovered.is_contiguous())
        self.assertTrue(torch.equal(reconstructed, dense))

    def test_probe_exact_geometry_is_self_consistent(self):
        for case in self.probe.CASES.values():
            with self.subTest(case=case["name"]):
                tokens = 1
                for value in case["window_size"]:
                    tokens *= value
                self.assertIn(tokens, (32, 144))
                ph = (
                    case["input_resolution"][0] // case["window_size"][0]
                ) * (case["input_resolution"][1] // case["window_size"][1])
                self.assertEqual(ph, case["pressure_height"])
                self.assertEqual(
                    case["input_resolution"][2] // case["window_size"][2],
                    case["width"],
                )


if __name__ == "__main__":
    unittest.main()
