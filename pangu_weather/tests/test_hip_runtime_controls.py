import importlib
import sys
import unittest
from pathlib import Path


PANGU_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(PANGU_DIR))
MODULE = importlib.import_module("hip_runtime_controls")


class FakeFunction:
    def __init__(self, callback=None):
        self.callback = callback or (lambda *args: 0)
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


class FakeHip:
    def __init__(self):
        self.hipSetDeviceFlags = FakeFunction()
        self.hipDeviceSetCacheConfig = FakeFunction()
        self.hipStreamSetAttribute = FakeFunction()

        def get_attribute(_stream, _attribute, value):
            value._obj.syncPolicy = MODULE.HIP_SYNC_POLICY_SPIN
            return 0

        self.hipStreamGetAttribute = FakeFunction(get_attribute)


class FakeStream:
    def __init__(self, pointer):
        self.cuda_stream = pointer
        self.waited_for = None

    def wait_stream(self, stream):
        self.waited_for = stream


class FakeCuda:
    def __init__(self):
        self.previous = FakeStream(11)
        self.created = FakeStream(22)
        self.current = self.previous

    def current_stream(self):
        return self.current

    def Stream(self):
        return self.created

    def set_stream(self, stream):
        self.current = stream


class HipRuntimeControlsTests(unittest.TestCase):
    def test_schedule_and_cache_use_documented_enum_values(self):
        libhip = FakeHip()
        MODULE.set_schedule_spin(libhip)
        MODULE.set_prefer_l1(libhip)
        self.assertEqual(
            libhip.hipSetDeviceFlags.calls[0][0], MODULE.HIP_DEVICE_SCHEDULE_SPIN
        )
        self.assertEqual(
            libhip.hipDeviceSetCacheConfig.calls[0][0],
            MODULE.HIP_FUNC_CACHE_PREFER_L1,
        )

    def test_spin_stream_waits_for_previous_and_verifies_attribute(self):
        libhip = FakeHip()
        torch_module = type("FakeTorch", (), {"cuda": FakeCuda()})()
        stream = MODULE.create_spin_stream(libhip, torch_module)
        self.assertIs(stream.waited_for, torch_module.cuda.previous)
        self.assertIs(torch_module.cuda.current, stream)
        self.assertEqual(len(libhip.hipStreamSetAttribute.calls), 1)
        self.assertEqual(len(libhip.hipStreamGetAttribute.calls), 1)

    def test_nonzero_status_is_rejected(self):
        libhip = FakeHip()
        libhip.hipSetDeviceFlags = FakeFunction(lambda *_args: 100)
        with self.assertRaisesRegex(RuntimeError, "returned 100"):
            MODULE.set_schedule_spin(libhip)


if __name__ == "__main__":
    unittest.main()
