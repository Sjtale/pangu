"""Small, independently testable HIP runtime controls for DCU A/B probes."""

import ctypes
import os


HIP_DEVICE_SCHEDULE_SPIN = 1
HIP_FUNC_CACHE_PREFER_L1 = 2
HIP_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3
HIP_SYNC_POLICY_SPIN = 2
HIP_SHARED_MEM_BANK_SIZE_FOUR_BYTE = 1
HIP_SHARED_MEM_BANK_SIZE_EIGHT_BYTE = 2


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


def set_nearest_cpu_affinity(libhip, os_module=os):
    """Bind the current process to the closest allowed CPU reported by HIP."""

    libhip.hipExtGetNearstCPU.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libhip.hipExtGetNearstCPU.restype = ctypes.c_int
    nearest = ctypes.c_int()
    status = libhip.hipExtGetNearstCPU(ctypes.byref(nearest))
    if status != 0:
        raise RuntimeError(f"hipExtGetNearstCPU returned {status}")

    previous = set(os_module.sched_getaffinity(0))
    if nearest.value not in previous:
        raise RuntimeError(
            f"nearest HIP CPU {nearest.value} is outside allowed affinity "
            f"{sorted(previous)}"
        )
    os_module.sched_setaffinity(0, {nearest.value})
    observed = set(os_module.sched_getaffinity(0))
    if observed != {nearest.value}:
        raise RuntimeError(
            "CPU affinity verification failed: "
            f"expected {[nearest.value]}, got {sorted(observed)}"
        )
    return {
        "nearest_cpu": nearest.value,
        "previous_affinity": sorted(previous),
        "observed_affinity": sorted(observed),
    }


def set_shared_mem_bank_size(libhip, bank_bytes):
    configs = {
        4: HIP_SHARED_MEM_BANK_SIZE_FOUR_BYTE,
        8: HIP_SHARED_MEM_BANK_SIZE_EIGHT_BYTE,
    }
    if bank_bytes not in configs:
        raise ValueError("HIP shared-memory bank size must be 4 or 8 bytes")

    libhip.hipDeviceGetSharedMemConfig.argtypes = [ctypes.POINTER(ctypes.c_int)]
    libhip.hipDeviceGetSharedMemConfig.restype = ctypes.c_int
    libhip.hipDeviceSetSharedMemConfig.argtypes = [ctypes.c_int]
    libhip.hipDeviceSetSharedMemConfig.restype = ctypes.c_int
    previous = ctypes.c_int()
    status = libhip.hipDeviceGetSharedMemConfig(ctypes.byref(previous))
    if status != 0:
        raise RuntimeError(f"hipDeviceGetSharedMemConfig returned {status}")
    requested = configs[bank_bytes]
    status = libhip.hipDeviceSetSharedMemConfig(requested)
    if status != 0:
        raise RuntimeError(f"hipDeviceSetSharedMemConfig returned {status}")
    observed = ctypes.c_int()
    status = libhip.hipDeviceGetSharedMemConfig(ctypes.byref(observed))
    if status != 0:
        raise RuntimeError(f"hipDeviceGetSharedMemConfig returned {status}")
    if observed.value != requested:
        raise RuntimeError(
            "HIP shared-memory bank verification failed: "
            f"expected {requested}, got {observed.value}"
        )
    return {
        "bank_bytes": bank_bytes,
        "previous_config": previous.value,
        "observed_config": observed.value,
    }


def get_stream_priority_range(libhip):
    libhip.hipDeviceGetStreamPriorityRange.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    libhip.hipDeviceGetStreamPriorityRange.restype = ctypes.c_int
    least = ctypes.c_int()
    greatest = ctypes.c_int()
    status = libhip.hipDeviceGetStreamPriorityRange(
        ctypes.byref(least), ctypes.byref(greatest)
    )
    if status != 0:
        raise RuntimeError(f"hipDeviceGetStreamPriorityRange returned {status}")
    return least.value, greatest.value


def create_runtime_stream(libhip, torch_module, priority=None, spin=False):
    """Create one ordered stream with optional priority and wait policy."""

    previous_stream = torch_module.cuda.current_stream()
    report = {"spin": bool(spin), "priority": None, "priority_range": None}
    if priority is None:
        stream = torch_module.cuda.Stream()
    elif priority == "greatest":
        least, greatest = get_stream_priority_range(libhip)
        stream = torch_module.cuda.Stream(priority=greatest)
        report["priority"] = greatest
        report["priority_range"] = [least, greatest]
    else:
        raise ValueError("HIP stream priority must be None or 'greatest'")
    stream.wait_stream(previous_stream)

    if spin:
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
    return stream, report


def create_spin_stream(libhip, torch_module):
    """Backward-compatible wrapper for the existing spin-only control."""

    stream, _ = create_runtime_stream(libhip, torch_module, spin=True)
    return stream
