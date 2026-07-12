import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = load_script("freeze_pruned96_guardrail")
AUDIT = load_script("audit_submission_package")


class GuardrailPackageTests(unittest.TestCase):
    def test_freeze_copies_artifacts_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = []
            for label, filename in (
                ("submission_zip", "submission.zip"),
                ("checkpoint", "model.pth"),
                ("calibration", "calibration.npy"),
            ):
                path = root / filename
                path.write_bytes((label + "-verified").encode())
                artifacts.append((label, path))
            output = root / "guardrail"
            manifest = FREEZE.freeze_guardrail(output, artifacts)
            self.assertEqual(manifest["profile"]["depth_blocks"], [2, 6, 6, 2])
            self.assertEqual(manifest["platform_score"]["total"], 89.6297)
            on_disk = json.loads((output / "guardrail_manifest.json").read_text())
            self.assertEqual(len(on_disk["artifacts"]), 3)
            with self.assertRaises(FileExistsError):
                FREEZE.freeze_guardrail(output, artifacts)

    def test_package_audit_requires_exact_runtime_whitelist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for basename in sorted(AUDIT.REQUIRED_BASENAMES):
                    archive.writestr(f"pangu_weather/{basename}", basename)
            report = AUDIT.audit_zip(path)
            self.assertEqual(
                {Path(name).name for name in report["files"]},
                AUDIT.REQUIRED_BASENAMES,
            )

            invalid = Path(directory) / "invalid.zip"
            with zipfile.ZipFile(invalid, "w") as archive:
                for basename in sorted(AUDIT.REQUIRED_BASENAMES | {"README.md"}):
                    archive.writestr(f"pangu_weather/{basename}", basename)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                AUDIT.audit_zip(invalid)


if __name__ == "__main__":
    unittest.main()
