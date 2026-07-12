"""Tests for lossless checkpoint gzip packaging."""

import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compress_checkpoint_gzip.py"
SPEC = importlib.util.spec_from_file_location("compress_checkpoint_gzip", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompressCheckpointGzipTests(unittest.TestCase):
    def test_round_trip_is_byte_exact_and_smaller(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "source.pth.gz"
            source.write_bytes((b"pangu-weather-weights\0" * 4096) + bytes(range(256)))
            MODULE.compress_checkpoint(source, output)
            with gzip.open(output, "rb") as stream:
                self.assertEqual(stream.read(), source.read_bytes())
            self.assertLess(output.stat().st_size, source.stat().st_size)

    def test_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            output = Path(directory) / "source.pth.gz"
            source.write_bytes(b"source")
            output.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                MODULE.compress_checkpoint(source, output)


if __name__ == "__main__":
    unittest.main()
