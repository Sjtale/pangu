import ast
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_quant_sensitivity.py"


class AnalyzeQuantSensitivityStaticTests(unittest.TestCase):
    def test_parameter_mutation_is_no_grad_and_restored_in_finally(self):
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertGreaterEqual(source.count("with torch.no_grad():"), 2)
        self.assertIn("finally:", source)
        self.assertIn("module.weight.copy_(original)", source)


if __name__ == "__main__":
    unittest.main()
