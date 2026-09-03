import sys
import unittest
import weakref
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import p2_tiled_attention as p2_adapter


class RecordingQKV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.widths = []
        self.output_ref = lambda: None

    def forward(self, x):
        self.widths.append(x.shape[0])
        output = torch.cat((x, x, x), dim=-1)
        self.output_ref = weakref.ref(output)
        return output


class RecordingProjection(torch.nn.Module):
    def __init__(self, qkv):
        super().__init__()
        self.qkv = qkv
        self.inputs = []
        self.qkv_alive = []

    def forward(self, x):
        self.inputs.append(x)
        self.qkv_alive.append(self.qkv.output_ref() is not None)
        return x + 1


class RecordingDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 2


class MinimalAttention:
    def __init__(self, *, full_width):
        self.num_heads = 3
        self.scale = 32**-0.5
        self.qkv = RecordingQKV()
        self.proj = RecordingProjection(self.qkv)
        self.proj_drop = RecordingDropout()
        self._pangu_attention_chunk_size = 2
        self._pangu_chunked_qkv = True
        self._pangu_chunked_proj = True
        self._pangu_p2_full_width = full_width
        self._pangu_p2_tiled_kernel_mode = "full-row-fast"


class FullWidthOutputReuseTests(unittest.TestCase):
    def _run(self, *, full_width, capture_attention_output=True, chunked=None):
        module = MinimalAttention(full_width=full_width)
        if chunked is not None:
            module._pangu_chunked_qkv = chunked
            module._pangu_chunked_proj = chunked
        x = torch.arange(5 * 2 * 32 * 96, dtype=torch.float32).reshape(
            5, 2, 32, 96
        )
        hip_output = -x
        calls = []
        region_ids = object() if chunked else None

        def tiled_forward(qkv, *args, **kwargs):
            del args
            calls.append((tuple(qkv.shape), kwargs))
            start = kwargs["width_offset"]
            if start == 0 and qkv.shape[0] == hip_output.shape[0]:
                return hip_output
            return hip_output[start : start + qkv.shape[0]]

        result, captured = p2_adapter._run_p2_tiled_chunks(
            module,
            x,
            packed_bias=None,
            position_index=None,
            region_ids=region_ids,
            mask_width=7 if region_ids is not None else None,
            tiled_forward=tiled_forward,
            capture_attention_output=capture_attention_output,
        )
        return module, result, captured, hip_output, calls

    def test_full_width_reuses_hip_output_storage(self):
        module, result, captured, hip_output, calls = self._run(
            full_width=True
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], 5)
        self.assertEqual(module.qkv.widths, [5])
        self.assertEqual(len(module.proj.inputs), 1)
        self.assertIs(module.proj.inputs[0], hip_output)
        self.assertEqual(module.proj.inputs[0].data_ptr(), hip_output.data_ptr())
        self.assertEqual(module.proj.qkv_alive, [False])
        self.assertIs(captured, hip_output)
        self.assertEqual(captured.data_ptr(), hip_output.data_ptr())
        self.assertTrue(torch.equal(result, hip_output + 3))

    def test_full_width_without_capture_still_reuses_projection_input(self):
        module, result, captured, hip_output, calls = self._run(
            full_width=True,
            capture_attention_output=False,
        )

        self.assertEqual(len(calls), 1)
        self.assertIsNone(captured)
        self.assertIs(module.proj.inputs[0], hip_output)
        self.assertEqual(module.proj.qkv_alive, [False])
        self.assertTrue(torch.equal(result, hip_output + 3))

    def test_non_full_width_keeps_existing_copy_path(self):
        module, result, captured, hip_output, calls = self._run(
            full_width=False,
            chunked=False,
        )

        self.assertEqual(len(calls), 1)
        self.assertIsNot(captured, hip_output)
        self.assertTrue(torch.equal(captured, hip_output))
        self.assertIs(module.proj.inputs[0], captured)
        self.assertTrue(torch.equal(result, hip_output + 3))

    def test_chunked_path_keeps_tail_assembly_and_offsets(self):
        module, result, captured, hip_output, calls = self._run(
            full_width=False,
            chunked=True,
        )

        self.assertEqual(module.qkv.widths, [2, 2, 1])
        self.assertEqual([shape[0] for shape, _ in calls], [2, 2, 1])
        self.assertEqual(
            [kwargs["width_offset"] for _, kwargs in calls],
            [0, 2, 4],
        )
        self.assertTrue(
            all(kwargs["mask_width"] == 7 for _, kwargs in calls)
        )
        self.assertEqual(len(module.proj.inputs), 3)
        self.assertTrue(torch.equal(captured, hip_output))
        self.assertTrue(torch.equal(result, hip_output + 3))


if __name__ == "__main__":
    unittest.main()
