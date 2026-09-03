import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
import io
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
    def _write_required_directories(archive):
        for member in sorted(AUDIT.REQUIRED_DIRECTORIES):
            archive.writestr(member, b"")

    @staticmethod
    def _write_valid_package(path):
        with zipfile.ZipFile(path, "w") as archive:
            GuardrailPackageTests._write_required_directories(archive)
            for member in sorted(AUDIT.REQUIRED_PATHS):
                archive.writestr(member, member)

    @staticmethod
    def _prepare_build_tree(root, include_url=True):
        source_root = Path(__file__).parents[1]
        (root / "scripts").mkdir(parents=True)
        shutil.copy2(source_root / "scripts/build_submission.sh", root / "scripts")
        shutil.copy2(source_root / "COMPLIANCE_README.md", root)
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
        shutil.copy2(
            source_root / "scripts/audit_submission_package.py",
            root / "scripts",
        )

    def test_freeze_copies_artifacts_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = []
            for label, filename in (
                ("submission_zip", "submission.zip"),
                ("checkpoint", "model.pth"),
                ("static_audit", "static.json"),
            ):
                path = root / filename
                path.write_bytes((label + "-verified").encode())
                artifacts.append((label, path))
            output = root / "guardrail"
            manifest = FREEZE.freeze_guardrail(output, artifacts)
            self.assertEqual(manifest["profile"]["depth_blocks"], [2, 6, 6, 2])
            self.assertEqual(manifest["platform_score"]["total"], 90.7763)
            self.assertEqual(
                manifest["platform_score"]["metric_mapping"]["U"],
                "lightweight",
            )
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

            missing_output = Path(directory) / "missing-output.zip"
            with zipfile.ZipFile(missing_output, "w") as archive:
                for member in sorted(AUDIT.REQUIRED_PATHS):
                    archive.writestr(member, member)
            with self.assertRaisesRegex(ValueError, "missing_directories"):
                AUDIT.audit_zip(missing_output)

            invalid = Path(directory) / "invalid.zip"
            self._write_valid_package(invalid)
            with zipfile.ZipFile(invalid, "a") as archive:
                archive.writestr("pangu_weather/unexpected.py", "unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                AUDIT.audit_zip(invalid)

    def test_package_audit_rejects_forbidden_compliance_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.zip"
            with zipfile.ZipFile(path, "w") as archive:
                self._write_required_directories(archive)
                for member in sorted(AUDIT.REQUIRED_PATHS):
                    content = member
                    if member == "pangu_weather/inference.py":
                        content = "name = 'calibration_coeffs.npy'\n"
                    archive.writestr(member, content)
            with self.assertRaisesRegex(ValueError, "Forbidden compliance paths"):
                AUDIT.audit_zip(path)

    def test_package_audit_allows_fail_closed_scored_only_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.zip"
            with zipfile.ZipFile(path, "w") as archive:
                self._write_required_directories(archive)
                for member in sorted(AUDIT.REQUIRED_PATHS):
                    content = member
                    if member == "pangu_weather/distill_train.py":
                        content = (
                            'if env_enabled("PANGU_SCORED_ONLY_RECOVERY"):\n'
                            '    raise ValueError("scored-only is forbidden")\n'
                        )
                    archive.writestr(member, content)
            report = AUDIT.audit_zip(path)
            self.assertEqual(set(report["files"]), AUDIT.REQUIRED_PATHS)

    def test_model_archive_must_contain_only_root_checkpoint(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "submission.zip"
            self._write_valid_package(package)
            model = root / "model.zip"
            with zipfile.ZipFile(model, "w") as archive:
                archive.writestr("nested/model_fp16.pth", b"checkpoint")
            with self.assertRaisesRegex(ValueError, "exactly model_fp16.pth"):
                AUDIT.audit_zip(package, model)

            buffer = io.BytesIO()
            torch.save(
                {
                    "model_state_dict": {
                        **{f"linear{index}.weight": torch.ones(1, 1, dtype=torch.float16) for index in range(67)},
                    },
                    "model_profile": {"name": "pgw_lite_pruned_96"},
                    "distillation": {
                        "teacher_source": "organizer_pangu_full_model",
                        "ground_truth_weight": 0.5,
                        "teacher_weight": 0.5,
                        "all_69_channels": True,
                        "predict_residual": False,
                    },
                    "quantization": {
                        "fp16_keep_count": 67,
                        "quantized_keys_count": 0,
                    },
                    "alias_compaction": {"alias_pair_count": 224},
                },
                buffer,
            )
            with zipfile.ZipFile(model, "w") as archive:
                archive.writestr("model_fp16.pth", buffer.getvalue())
            report = AUDIT.audit_zip(package, model)
            self.assertEqual(report["model_members"], ["model_fp16.pth"])
            self.assertEqual(
                report["model_provenance"]["teacher_source"],
                "organizer_pangu_full_model",
            )

    def test_checkpoint_metadata_rejects_residual_or_wrong_fixed_weights(self):
        base = {
            "model_profile": {"name": "pgw_lite_pruned_96"},
            "distillation": {
                "teacher_source": "organizer_pangu_full_model",
                "ground_truth_weight": 0.5,
                "teacher_weight": 0.5,
                "all_69_channels": True,
                "predict_residual": True,
            },
            "quantization": {
                "fp16_keep_count": 67,
                "quantized_keys_count": 0,
            },
            "alias_compaction": {"alias_pair_count": 224},
        }
        with self.assertRaisesRegex(ValueError, "Residual-target"):
            AUDIT.audit_checkpoint_metadata(base)
        base["distillation"]["predict_residual"] = False
        base["distillation"]["ground_truth_weight"] = 0.8
        with self.assertRaisesRegex(ValueError, "must be 0.5/0.5"):
            AUDIT.audit_checkpoint_metadata(base)

        base["distillation"]["ground_truth_weight"] = 0.5
        base["distillation"]["hint_weight"] = 0.0
        with self.assertRaisesRegex(ValueError, "must not use hint"):
            AUDIT.audit_checkpoint_metadata(base)

    def test_checkpoint_metadata_allows_absent_optional_distillation(self):
        report = AUDIT.audit_checkpoint_metadata(
            {"model_profile": {"name": "pgw_lite_pruned_96"}}
        )
        self.assertFalse(report["distillation_metadata_present"])

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
                self._write_required_directories(archive)
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
            self.assertTrue(
                AUDIT.REQUIRED_DIRECTORIES.issubset(set(report["directories"]))
            )

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

    def test_build_submission_rejects_unsafe_alternate_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pangu_weather"
            self._prepare_build_tree(root)
            victim = Path(directory) / "do_not_delete"
            victim.mkdir()
            marker = victim / "marker.txt"
            marker.write_text("preserve", encoding="utf-8")
            environment = os.environ.copy()
            environment["PANGU_SUBMIT_DIR"] = str(victim)
            result = subprocess.run(
                ["bash", str(root / "scripts/build_submission.sh")],
                cwd=directory,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Refusing unsafe PANGU_SUBMIT_DIR", result.stdout)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
