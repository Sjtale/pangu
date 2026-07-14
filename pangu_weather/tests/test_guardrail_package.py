import importlib.util
import json
import shutil
import subprocess
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
    @staticmethod
    def _write_valid_package(path):
        with zipfile.ZipFile(path, "w") as archive:
            for member in sorted(AUDIT.REQUIRED_PATHS):
                archive.writestr(member, member)

    @staticmethod
    def _prepare_build_tree(root, include_url=True):
        source_root = Path(__file__).parents[1]
        (root / "scripts").mkdir(parents=True)
        shutil.copy2(source_root / "scripts/build_submission.sh", root / "scripts")
        shutil.copy2(
            source_root / "scripts/audit_submission_package.py",
            root / "scripts",
        )
        for member in AUDIT.REQUIRED_PATHS:
            relative = Path(member).relative_to("pangu_weather")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == Path("data/download_model_url.txt"):
                if include_url:
                    target.write_text(
                        "https://example.invalid/model.zip\n", encoding="utf-8"
                    )
            elif relative == Path("conf/config.yaml"):
                target.write_text(
                    "stats_dir: old\n"
                    "static_dir: old\n"
                    "data_dir: old\n"
                    "train_ratio: old\n"
                    "val_ratio: old\n"
                    "test_ratio: old\n"
                    "checkpoint_dir: old\n"
                    "batch_size: old\n"
                    "world_size: old\n",
                    encoding="utf-8",
                )
            else:
                target.write_text(relative.as_posix(), encoding="utf-8")

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
            self._write_valid_package(path)
            report = AUDIT.audit_zip(path)
            self.assertEqual(set(report["files"]), AUDIT.REQUIRED_PATHS)

            invalid = Path(directory) / "invalid.zip"
            self._write_valid_package(invalid)
            with zipfile.ZipFile(invalid, "a") as archive:
                archive.writestr("pangu_weather/README.md", "unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                AUDIT.audit_zip(invalid)

    def test_package_audit_rejects_wrong_path_and_allows_empty_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong_path = root / "wrong-path.zip"
            self._write_valid_package(wrong_path)
            with zipfile.ZipFile(wrong_path, "a") as archive:
                archive.writestr(
                    "pangu_weather/hip_kernels/../earth_attention_tiled_fwd.hip",
                    "wrong path",
                )
            with self.assertRaisesRegex(ValueError, "unexpected"):
                AUDIT.audit_zip(wrong_path)

            empty_url = root / "empty-url.zip"
            with zipfile.ZipFile(empty_url, "w") as archive:
                for member in sorted(AUDIT.REQUIRED_PATHS):
                    archive.writestr(member, "")
            report = AUDIT.audit_zip(empty_url)
            self.assertEqual(set(report["files"]), AUDIT.REQUIRED_PATHS)

    def test_build_submission_produces_audited_p2_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pangu_weather"
            self._prepare_build_tree(root)
            result = subprocess.run(
                ["bash", str(root / "scripts/build_submission.sh")],
                cwd=directory,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            package = root / "submit_package/pangu_weather.zip"
            report = AUDIT.audit_zip(package)
            self.assertEqual(set(report["files"]), AUDIT.REQUIRED_PATHS)

    def test_build_submission_rejects_missing_model_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pangu_weather"
            self._prepare_build_tree(root, include_url=False)
            result = subprocess.run(
                ["bash", str(root / "scripts/build_submission.sh")],
                cwd=directory,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("缺少打包必需文件", result.stdout)
            self.assertFalse((root / "submit_package").exists())


if __name__ == "__main__":
    unittest.main()
