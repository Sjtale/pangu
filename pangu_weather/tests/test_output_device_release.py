"""Static lifecycle contract for inference output device tensors."""

import hashlib
import unittest
from pathlib import Path


INFERENCE = Path(__file__).parents[1] / "inference.py"
TIMER_SHA256 = "fa7d46a8ea3a3da93f5348bbb6b237409da16a68b20708331d4d9b0f4adb61ad"


class OutputDeviceReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = INFERENCE.read_text(encoding="utf-8")
        loop_start = cls.source.index("for batch_index, data in enumerate")
        cls.loop = cls.source[loop_start:]

    def test_default_cpu_postprocess_releases_each_device_output_after_d2h(self):
        block_start = self.loop.index("if compliant_boundary:", self.loop.index("end_time"))
        block_end = self.loop.index("np.save", block_start)
        block = self.loop[block_start:block_end]

        unpack = block.index("out_surface, out_upper_air = model_output")
        release_tuple = block.index("del model_output", unpack)
        copy_surface = block.index(
            "out_surface_cpu = out_surface.detach().cpu()", release_tuple
        )
        release_surface = block.index("del out_surface", copy_surface)
        copy_upper = block.index(
            "out_upper_air_cpu = out_upper_air.detach().cpu()", release_surface
        )
        release_upper = block.index("del out_upper_air", copy_upper)
        concat = block.index(
            "[out_surface_cpu, out_upper_air_cpu]", release_upper
        )

        self.assertLess(
            unpack,
            release_tuple,
            "the output tuple otherwise keeps both device tensors alive",
        )
        self.assertLess(copy_surface, release_surface)
        self.assertLess(release_surface, copy_upper)
        self.assertLess(copy_upper, release_upper)
        self.assertLess(release_upper, concat)

    def test_all_output_paths_drop_device_references_before_save(self):
        output_start = self.loop.index("if compliant_boundary:", self.loop.index("end_time"))
        save = self.loop.index("np.save", output_start)
        output_block = self.loop[output_start:save]

        compliant_copy = output_block.index(
            "pred_var = model_output.detach().cpu().numpy()"
        )
        compliant_release = output_block.index("del model_output", compliant_copy)
        gpu_concat = output_block.index(
            "pred_var = torch.concat(", compliant_release
        )
        gpu_release = output_block.index(
            "del out_surface, out_upper_air", gpu_concat
        )

        self.assertLess(compliant_copy, compliant_release)
        self.assertLess(compliant_release, save - output_start)
        self.assertLess(gpu_concat, gpu_release)
        self.assertLess(gpu_release, save - output_start)

    def test_official_timer_block_remains_frozen(self):
        marker = "#----------------------AI4S(时间度量不可更改)---------------------------"
        loop_start = self.source.index("for batch_index, data in enumerate")
        start = self.source.index(marker, loop_start)
        end_marker = "#---------------------------------------------------------------------"
        end = self.source.index(end_marker, start) + len(end_marker)
        digest = hashlib.sha256(self.source[start:end].encode()).hexdigest()
        self.assertEqual(digest, TIMER_SHA256)


if __name__ == "__main__":
    unittest.main()
