import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_triton_runtime_stages.py"
)


class TritonRuntimeStageTests(unittest.TestCase):
    def test_supervisor_isolates_expected_compiler_stages(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for stage in (
            "import",
            "vector",
            "dot",
            "attention_qk",
            "attention_bias",
            "attention_bias_bitcast",
            "attention_max_bitcast",
            "attention_exp_bitcast",
            "attention_sum_bitcast",
            "attention_normalize_bitcast",
            "attention_pv_bitcast",
            "earth32",
            "earth144",
        ):
            self.assertIn(f'"{stage}"', source)
        self.assertIn("subprocess.run", source)
        self.assertIn("preexec_fn=_disable_core_dump", source)
        self.assertIn("signal.Signals", source)
        self.assertIn("SKIPPED_AFTER_FAILURE", source)

    def test_workers_cover_elementwise_dot_and_earth_attention(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("@triton.jit", source)
        self.assertIn("_vector_add_kernel", source)
        self.assertIn("_dot_kernel", source)
        self.assertIn("_attention_component_kernel", source)
        self.assertIn("scores.to(tl.float16)", source)
        self.assertIn("scores.to(tl.uint32, bitcast=True)", source)
        self.assertIn("scores.to(tl.float32, bitcast=True)", source)
        self.assertIn("triton_earth_attention", source)


if __name__ == "__main__":
    unittest.main()
