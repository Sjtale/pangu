"""Small, independently testable HIP runtime controls for DCU A/B probes."""

import ctypes


HIP_DEVICE_SCHEDULE_SPIN = 1
HIP_FUNC_CACHE_PREFER_L1 = 2
HIP_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3
HIP_SYNC_POLICY_SPIN = 2


class HipStreamAttrValue(ctypes.Union):
    _fields_ = [("syncPolicy", ctypes.c_int)]


def load_hip_runtime():
    for library_name in (
        "libamdhip64.so",
        "libhip_hcc.so",
        "libamdhip64.so.5",
        "libamdhip64.so.6",
    ):
        try:
            return ctypes.CDLL(library_name)
        except OSError:
            continue
    return None


def set_schedule_spin(libhip):
    """Set the process wait policy before any HIP context is created."""

    libhip.hipSetDeviceFlags.argtypes = [ctypes.c_uint]
    libhip.hipSetDeviceFlags.restype = ctypes.c_int
    status = libhip.hipSetDeviceFlags(HIP_DEVICE_SCHEDULE_SPIN)
    if status != 0:
        raise RuntimeError(f"hipSetDeviceFlags returned {status}")


def set_prefer_l1(libhip):
    libhip.hipDeviceSetCacheConfig.argtypes = [ctypes.c_int]
    libhip.hipDeviceSetCacheConfig.restype = ctypes.c_int
    status = libhip.hipDeviceSetCacheConfig(HIP_FUNC_CACHE_PREFER_L1)
    if status != 0:
        raise RuntimeError(f"hipDeviceSetCacheConfig returned {status}")


def create_spin_stream(libhip, torch_module):
    """Create an ordered PyTorch stream and verify its HIP wait policy."""

    previous_stream = torch_module.cuda.current_stream()
    stream = torch_module.cuda.Stream()
    stream.wait_stream(previous_stream)

    libhip.hipStreamSetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(HipStreamAttrValue),
    ]
    libhip.hipStreamSetAttribute.restype = ctypes.c_int
    libhip.hipStreamGetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(HipStreamAttrValue),
    ]
    libhip.hipStreamGetAttribute.restype = ctypes.c_int

    stream_pointer = ctypes.c_void_p(stream.cuda_stream)
    requested = HipStreamAttrValue()
    requested.syncPolicy = HIP_SYNC_POLICY_SPIN
    status = libhip.hipStreamSetAttribute(
        stream_pointer,
        HIP_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY,
        ctypes.byref(requested),
    )
    if status != 0:
        raise RuntimeError(f"hipStreamSetAttribute returned {status}")

    observed = HipStreamAttrValue()
    status = libhip.hipStreamGetAttribute(
        stream_pointer,
        HIP_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY,
        ctypes.byref(observed),
    )
    if status != 0:
        raise RuntimeError(f"hipStreamGetAttribute returned {status}")
    if observed.syncPolicy != HIP_SYNC_POLICY_SPIN:
        raise RuntimeError(
            "HIP stream synchronization policy verification failed: "
            f"expected {HIP_SYNC_POLICY_SPIN}, got {observed.syncPolicy}"
        )

    torch_module.cuda.set_stream(stream)
    return stream
