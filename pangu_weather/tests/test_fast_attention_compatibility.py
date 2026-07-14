import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_fast_attention_compatibility.py"
BIAS_AWARE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_bias_aware_attention_backends.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_fast_attention_compatibility", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bias_aware_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_bias_aware_attention_backends", BIAS_AWARE_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastAttentionCompatibilityTests(unittest.TestCase):
    def test_representative_layout_combines_bias_and_shifted_mask(self):
        probe = _load_probe()
        args = SimpleNamespace(
            width_windows=2,
            pressure_height_windows=3,
            heads=2,
            tokens=4,
            head_dim=2,
            seed=7,
        )
        q, k, v, mask = probe._make_inputs(args, torch.device("cpu"))
        self.assertEqual(tuple(q.shape), (6, 2, 4, 2))
        self.assertEqual(tuple(k.shape), tuple(q.shape))
        self.assertEqual(tuple(v.shape), tuple(q.shape))
        self.assertEqual(tuple(mask.shape), (6, 2, 4, 4))
        self.assertTrue(torch.all(mask[..., :2, 2:] < -99.0))

    def test_flash_path_explicitly_disables_backend_fallback(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])", source)
        self.assertNotIn("SDPBackend.MATH", source)
        self.assertNotIn("SDPBackend.EFFICIENT_ATTENTION", source)

    def test_bias_aware_layout_keeps_earth_bias_and_shift_mask_separate(self):
        probe = _load_bias_aware_probe()
        args = SimpleNamespace(
            width_windows=2,
            pressure_height_windows=3,
            heads=2,
            tokens=4,
            head_dim=2,
            seed=7,
        )
        q, k, v, earth_bias, shifted = probe._make_inputs(args, torch.device("cpu"))
        self.assertEqual(tuple(q.shape), (6, 2, 4, 2))
        self.assertEqual(tuple(k.shape), tuple(q.shape))
        self.assertEqual(tuple(v.shape), tuple(q.shape))
        self.assertEqual(tuple(earth_bias.shape), (3, 2, 4, 4))
        self.assertEqual(tuple(shifted.shape), (2, 3, 1, 4, 4))
        self.assertEqual(
            tuple(probe._combined_bias(earth_bias, shifted).shape), (6, 2, 4, 4)
        )

    def test_padding_masks_added_keys_and_preserves_original_inputs(self):
        probe = _load_bias_aware_probe()
        args = SimpleNamespace(
            width_windows=2,
            pressure_height_windows=3,
            heads=2,
            tokens=4,
            head_dim=2,
            seed=7,
        )
        inputs = probe._make_inputs(args, torch.device("cpu"))
        padded, allowed = probe._pad_for_triton(inputs, padded_tokens=8)
        self.assertEqual(tuple(padded[0].shape), (6, 2, 8, 2))
        self.assertTrue(torch.equal(padded[0][..., :4, :], inputs[0]))
        self.assertFalse(bool(allowed[..., :, 4:].any()))
        self.assertTrue(bool(allowed[..., 4:, 0].all()))

    def test_bias_aware_backends_are_forced_without_xla_or_unfused_fallback(self):
        source = BIAS_AWARE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('implementation="triton"', source)
        self.assertIn('os.environ["NVTE_UNFUSED_ATTN"] = "0"', source)
        self.assertIn('os.environ.setdefault("NVTE_DEBUG", "1")', source)
        self.assertIn("torch.compile(flex_attention, fullgraph=True, dynamic=False)", source)

    def test_launcher_does_not_repeat_crashed_flex_by_default(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_bias_aware_attention_probes.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('PANGU_RETRY_CRASHED_FLEX:-0', launcher)
        self.assertIn('PANGU_JAX_PYTHON', launcher)
        self.assertIn('command -v "$jax_python"', launcher)
        self.assertIn("import jax, jax_triton", launcher)
        self.assertIn('transformer_engine_debug.log', launcher)


if __name__ == "__main__":
    unittest.main()
