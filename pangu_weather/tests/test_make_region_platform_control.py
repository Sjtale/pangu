import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import zipfile


SCRIPT = Path(__file__).parents[1] / "scripts/make_region_platform_control.py"
SPEC = importlib.util.spec_from_file_location("make_region_platform_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inference_source(default="1"):
    return (
        'os.environ.setdefault("PANGU_P2_REGION_RELEASE", "' + default + '")\n'
        "for batch_index, data in enumerate(loader):\n"
        + MODULE.TIMER_START
        + "\nstart_time = time.perf_counter()\n"
        + MODULE.TIMER_END
        + "\n"
    ).encode("utf-8")


def _write_source(path, inference=None, comment=b"source archive comment"):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.comment = comment
        archive.writestr("pangu_weather/", b"")
        archive.writestr(
            MODULE.INFERENCE_MEMBER,
            _inference_source() if inference is None else inference,
        )
        archive.writestr("pangu_weather/result.py", b"result\n")
        archive.writestr("pangu_weather/conf/config.yaml", b"config\n")


class RegionPlatformControlTests(unittest.TestCase):
    def test_control_changes_only_region_default_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "region_on.zip"
            output = root / "submit_package/control/pangu_weather.zip"
            report_path = root / "logs/control.json"
            _write_source(source)
            source_sha256 = MODULE._sha256_path(source)

            report = MODULE.make_region_off_control(
                source,
                output,
                source_sha256,
                report_path=report_path,
            )

            self.assertTrue(output.is_file())
            self.assertTrue(report_path.is_file())
            self.assertEqual(output.stat().st_mode & 0o777, 0o644)
            self.assertEqual(report_path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(report["changed_members"], [MODULE.INFERENCE_MEMBER])
            self.assertEqual(report["source_region_release_default"], 1)
            self.assertEqual(report["output_region_release_default"], 0)
            self.assertEqual(
                report["official_timer_block_sha256"],
                MODULE._timer_sha256(_inference_source()),
            )
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.comment, b"source archive comment")
                self.assertEqual(
                    archive.read(MODULE.INFERENCE_MEMBER),
                    _inference_source(default="0"),
                )
                self.assertEqual(archive.read("pangu_weather/result.py"), b"result\n")

    def test_rejects_wrong_source_hash_without_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "region_on.zip"
            output = root / "control.zip"
            _write_source(source)
            with self.assertRaisesRegex(ValueError, "Source ZIP SHA256 mismatch"):
                MODULE.make_region_off_control(source, output, "0" * 64)
            self.assertFalse(output.exists())

    def test_refuses_existing_output_without_overwriting_or_leaving_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "region_on.zip"
            output = root / "control.zip"
            report_path = root / "control.json"
            _write_source(source)
            output.write_bytes(b"existing output")

            with self.assertRaisesRegex(FileExistsError, "overwrite output"):
                MODULE.make_region_off_control(
                    source,
                    output,
                    MODULE._sha256_path(source),
                    report_path=report_path,
                )

            self.assertEqual(output.read_bytes(), b"existing output")
            self.assertFalse(report_path.exists())

    def test_refuses_existing_report_without_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "region_on.zip"
            output = root / "control.zip"
            report_path = root / "control.json"
            _write_source(source)
            report_path.write_bytes(b"existing report")

            with self.assertRaisesRegex(FileExistsError, "overwrite report"):
                MODULE.make_region_off_control(
                    source,
                    output,
                    MODULE._sha256_path(source),
                    report_path=report_path,
                )

            self.assertEqual(report_path.read_bytes(), b"existing report")
            self.assertFalse(output.exists())

    def test_keeps_valid_output_if_report_appears_during_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "region_on.zip"
            output = root / "control.zip"
            report_path = root / "control.json"
            _write_source(source)
            original_link = MODULE.os.link

            def publish_with_report_race(temporary, destination):
                if Path(destination) == report_path.resolve():
                    report_path.write_bytes(b"concurrent report")
                    raise FileExistsError(report_path)
                return original_link(temporary, destination)

            with mock.patch.object(
                MODULE.os, "link", side_effect=publish_with_report_race
            ):
                with self.assertRaisesRegex(RuntimeError, "left intact"):
                    MODULE.make_region_off_control(
                        source,
                        output,
                        MODULE._sha256_path(source),
                        report_path=report_path,
                    )

            self.assertTrue(output.is_file())
            self.assertEqual(report_path.read_bytes(), b"concurrent report")
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    archive.read(MODULE.INFERENCE_MEMBER),
                    _inference_source(default="0"),
                )

    def test_rejects_non_region_on_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "region_off.zip"
            output = root / "control.zip"
            _write_source(source, inference=_inference_source(default="0"))
            with self.assertRaisesRegex(ValueError, "exactly one Region-on"):
                MODULE.make_region_off_control(
                    source,
                    output,
                    MODULE._sha256_path(source),
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
