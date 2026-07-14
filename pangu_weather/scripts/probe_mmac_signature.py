#!/usr/bin/env python3
"""Probe the exact compiler signature of __builtin_amdgcn_mmac_f32_16x16x16f16."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Candidates to test: (A_type, B_type, C_type, output_type, extra_params_code)
CANDIDATES = [
    # 1. MFMA style with 3 extra ints (my_half4, my_half4, my_float4)
    ("my_half4", "my_half4", "my_float4", "my_float4", "0, 0, 0"),
    # 2. WMMA style without extra ints (my_half4, my_half4, my_float4)
    ("my_half4", "my_half4", "my_float4", "my_float4", ""),
    
    # 3. MFMA style with 3 extra ints (my_half8, my_half8, my_float4)
    ("my_half8", "my_half8", "my_float4", "my_float4", "0, 0, 0"),
    # 4. WMMA style without extra ints (my_half8, my_half8, my_float4)
    ("my_half8", "my_half8", "my_float4", "my_float4", ""),
    
    # 5. MFMA style with 3 extra ints (my_half4, my_half4, my_float8)
    ("my_half4", "my_half4", "my_float8", "my_float8", "0, 0, 0"),
    # 6. WMMA style without extra ints (my_half4, my_half4, my_float8)
    ("my_half4", "my_half4", "my_float8", "my_float8", ""),

    # 7. MFMA style with 3 extra ints (my_half8, my_half8, my_float8)
    ("my_half8", "my_half8", "my_float8", "my_float8", "0, 0, 0"),
    # 8. WMMA style without extra ints (my_half8, my_half8, my_float8)
    ("my_half8", "my_half8", "my_float8", "my_float8", ""),
    
    # 9. MFMA style with 3 extra ints (my_half16, my_half16, my_float4)
    ("my_half16", "my_half16", "my_float4", "my_float4", "0, 0, 0"),
    # 10. WMMA style without extra ints (my_half16, my_half16, my_float4)
    ("my_half16", "my_half16", "my_float4", "my_float4", ""),
]

def make_source(a_type, b_type, c_type, out_type, extra_params):
    args_list = ["a", "b", "c"]
    if extra_params:
        args_list.append(extra_params)
    args_str = ", ".join(args_list)
    
    return f"""\
#include <hip/hip_runtime.h>

typedef _Float16 my_half2 __attribute__((ext_vector_type(2)));
typedef _Float16 my_half4 __attribute__((ext_vector_type(4)));
typedef _Float16 my_half8 __attribute__((ext_vector_type(8)));
typedef _Float16 my_half16 __attribute__((ext_vector_type(16)));

typedef float my_float2 __attribute__((ext_vector_type(2)));
typedef float my_float4 __attribute__((ext_vector_type(4)));
typedef float my_float8 __attribute__((ext_vector_type(8)));
typedef float my_float16 __attribute__((ext_vector_type(16)));

extern "C" __global__ void test_probe_kernel({a_type} a, {b_type} b, {c_type} c, {out_type}* out) {{
#if defined(__HIP_DEVICE_COMPILE__)
    *out = __builtin_amdgcn_mmac_f32_16x16x16f16({args_str});
#endif
    return;
}}
"""

def test_compile(hipcc, arch, source):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_path = tmp_path / "probe.hip"
        object_path = tmp_path / "probe.o"
        source_path.write_text(source, encoding="utf-8")
        
        command = [
            hipcc,
            f"--offload-arch={arch}",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        return result.returncode == 0, result.stderr

def main():
    parser = argparse.ArgumentParser(description="Probe __builtin_amdgcn_mmac_f32_16x16x16f16 signature.")
    parser.add_argument("--hipcc", default="/opt/dtk/bin/hipcc", help="Path to hipcc")
    parser.add_argument("--arch", default="gfx936:sramecc+:xnack-", help="Target architecture")
    args = parser.parse_args()
    
    print(f"Probing signatures on compiler: {args.hipcc} for arch: {args.arch}...")
    
    found_any = False
    for i, (a_t, b_t, c_t, out_t, extra) in enumerate(CANDIDATES, 1):
        extra_desc = f"with extra: ({extra})" if extra else "without extra params"
        desc = f"Candidate {i}: {out_t} __builtin_amdgcn_mmac_f32_16x16x16f16({a_t}, {b_t}, {c_t}{', ' + extra if extra else ''})"
        
        source = make_source(a_t, b_t, c_t, out_t, extra)
        success, stderr = test_compile(args.hipcc, args.arch, source)
        
        if success:
            print(f"🟢 SUCCESS: {desc}")
            found_any = True
        else:
            # Check if it was a type error vs some other compiler issue
            if "too many arguments" in stderr or "too few arguments" in stderr or "parameter of incompatible type" in stderr:
                print(f"🔴 FAILED (Type Mismatch): {desc}")
            else:
                first_line = stderr.strip().split("\n")[0] if stderr else "unknown error"
                print(f"🔴 FAILED (Compile Error): {desc} -> {first_line}")
                
    if not found_any:
        print("\n❌ No candidates matched the compiler's expected signature.")
        sys.exit(1)
    else:
        print("\n✅ Probe complete.")

if __name__ == "__main__":
    main()
