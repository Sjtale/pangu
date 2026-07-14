#!/usr/bin/env python3
"""Diagnostic tool to probe matrix-to-lane layout of __builtin_amdgcn_mmac_f32_16x16x16f16."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

HIP_SOURCE = """
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <vector>

typedef _Float16 my_half2 __attribute__((ext_vector_type(2)));
typedef _Float16 my_half4 __attribute__((ext_vector_type(4)));
typedef float my_float4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void probe_layout_kernel(
    const float* A_in,
    const float* B_in,
    float* D_out)
{
    int lane_id = threadIdx.x; // 0 to 63
    
    my_half4 a;
    my_half4 b;
    my_float4 c;
    
    a.x = (_Float16)A_in[lane_id * 4 + 0];
    a.y = (_Float16)A_in[lane_id * 4 + 1];
    a.z = (_Float16)A_in[lane_id * 4 + 2];
    a.w = (_Float16)A_in[lane_id * 4 + 3];
    
    b.x = (_Float16)B_in[lane_id * 4 + 0];
    b.y = (_Float16)B_in[lane_id * 4 + 1];
    b.z = (_Float16)B_in[lane_id * 4 + 2];
    b.w = (_Float16)B_in[lane_id * 4 + 3];
    
    c.x = 0.0f;
    c.y = 0.0f;
    c.z = 0.0f;
    c.w = 0.0f;
    
#if defined(__HIP_DEVICE_COMPILE__)
    my_float4 d = __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, c);
#else
    my_float4 d = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
    
    D_out[lane_id * 4 + 0] = d.x;
    D_out[lane_id * 4 + 1] = d.y;
    D_out[lane_id * 4 + 2] = d.z;
    D_out[lane_id * 4 + 3] = d.w;
}

int main() {
    float* d_A;
    float* d_B;
    float* d_D;
    hipMalloc(&d_A, 256 * sizeof(float));
    hipMalloc(&d_B, 256 * sizeof(float));
    hipMalloc(&d_D, 256 * sizeof(float));
    
    std::vector<float> h_A(256, 0.0f);
    std::vector<float> h_B(256, 0.0f);
    std::vector<float> h_D(256, 0.0f);
    
    printf("START_PROBE\\n");
    for (int j = 0; j < 256; ++j) {
        // Clear host inputs
        for (int i = 0; i < 256; ++i) {
            h_A[i] = (float)(i + 1); // 1 to 256
            h_B[i] = (i == j) ? 1.0f : 0.0f;
        }
        
        hipMemcpy(d_A, h_A.data(), 256 * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), 256 * sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_D, 0, 256 * sizeof(float));
        
        hipLaunchKernelGGL(probe_layout_kernel, dim3(1), dim3(64), 0, 0, d_A, d_B, d_D);
        hipDeviceSynchronize();
        
        hipMemcpy(h_D.data(), d_D, 256 * sizeof(float), hipMemcpyDeviceToHost);
        
        // Print non-zero outputs
        for (int k = 0; k < 256; ++k) {
            if (h_D[k] != 0.0f) {
                // h_D[k] is the value from h_A, which is A_idx + 1
                int a_idx = (int)(h_D[k] + 0.5f) - 1;
                printf("MATCH: B_idx=%d -> D_idx=%d (val from A_idx=%d)\\n", j, k, a_idx);
            }
        }
    }
    printf("END_PROBE\\n");
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_D);
    return 0;
}
"""

def main():
    parser = argparse.ArgumentParser(description="Probe layout mapping of __builtin_amdgcn_mmac_f32_16x16x16f16.")
    parser.add_argument("--hipcc", default="/opt/dtk/bin/hipcc", help="Path to hipcc")
    parser.add_argument("--arch", default="gfx936:sramecc+:xnack-", help="Target architecture")
    args = parser.parse_args()

    print(f"Compiling layout probe using {args.hipcc} for arch {args.arch}...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_path = tmp_path / "probe_layout.hip"
        exec_path = tmp_path / "probe_layout"
        
        source_path.write_text(HIP_SOURCE, encoding="utf-8")
        
        command = [
            args.hipcc,
            f"--offload-arch={args.arch}",
            str(source_path),
            "-o",
            str(exec_path),
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Compilation failed!")
            print(result.stderr)
            return

        print("🟢 Compilation successful. Running probe on GPU...")
        
        run_result = subprocess.run([str(exec_path)], capture_output=True, text=True)
        if run_result.returncode != 0:
            print("❌ Execution failed!")
            print(run_result.stderr)
            return

        # Parse matches
        lines = run_result.stdout.strip().split("\n")
        matches = []
        for line in lines:
            if line.startswith("MATCH:"):
                # MATCH: B_idx=j -> D_idx=k (val from A_idx=i)
                parts = line.split()
                b_idx = int(parts[1].split("=")[1])
                d_idx = int(parts[3].split("=")[1])
                a_idx = int(parts[6].split("=")[1].replace(")", ""))
                matches.append((b_idx, d_idx, a_idx))

        if not matches:
            print("❌ No matches found! The intrinsic might not have executed.")
            return

        print(f"Captured {len(matches)} matches. Reconstructing matrix layouts...")

        # 1. Group D into rows using matched A
        d_by_matched_a = {}
        for d in range(256):
            matched = frozenset([a for b, d_val, a in matches if d_val == d])
            d_by_matched_a.setdefault(matched, []).append(d)
        
        row_D = [0] * 256
        row_A = [0] * 256
        for r, (matched, d_list) in enumerate(d_by_matched_a.items()):
            for d in d_list:
                row_D[d] = r
            for a in matched:
                row_A[a] = r
                
        # 2. Group D into columns using matched B
        d_by_matched_b = {}
        for d in range(256):
            matched = frozenset([b for b, d_val, a in matches if d_val == d])
            d_by_matched_b.setdefault(matched, []).append(d)
            
        col_D = [0] * 256
        col_B = [0] * 256
        for c, (matched, d_list) in enumerate(d_by_matched_b.items()):
            for d in d_list:
                col_D[d] = c
            for b in matched:
                col_B[b] = c
                
        # 3. Group B into rows using matched A
        b_by_matched_a = {}
        for b in range(256):
            matched = frozenset([a for b_val, d, a in matches if b_val == b])
            b_by_matched_a.setdefault(matched, []).append(b)
            
        row_B = [0] * 256
        col_A = [0] * 256
        for r_B, (matched, b_list) in enumerate(b_by_matched_a.items()):
            for b in b_list:
                row_B[b] = r_B
            for a in matched:
                col_A[a] = r_B

        # Analyze bit representations for each matrix
        # Let's print out the mapping function for each matrix
        def analyze_matrix(name, rows, cols):
            print(f"\n=== Layout analysis for Matrix {name} ===")
            print("Row mapping by index:")
            row_map_str = ", ".join(f"{i}->{rows[i]}" for i in range(16))
            print(f"  First 16 indices: {row_map_str}")
            
            print("Col mapping by index:")
            col_map_str = ", ".join(f"{i}->{cols[i]}" for i in range(16))
            print(f"  First 16 indices: {col_map_str}")
            
            # Find bit-level mapping: row_bit[b] = index_bit[p]
            row_bits = ["?"] * 4
            col_bits = ["?"] * 4
            
            for b in range(4):
                for p in range(8):
                    # Check if bit b of row matches bit p of index for all 256 indices
                    if all(((rows[idx] >> b) & 1) == ((idx >> p) & 1) for idx in range(256)):
                        row_bits[b] = f"idx_bit[{p}]"
                    elif all(((rows[idx] >> b) & 1) == (1 - ((idx >> p) & 1)) for idx in range(256)):
                        row_bits[b] = f"~idx_bit[{p}]"
            
            for b in range(4):
                for p in range(8):
                    if all(((cols[idx] >> b) & 1) == ((idx >> p) & 1) for idx in range(256)):
                        col_bits[b] = f"idx_bit[{p}]"
                    elif all(((cols[idx] >> b) & 1) == (1 - ((idx >> p) & 1)) for idx in range(256)):
                        col_bits[b] = f"~idx_bit[{p}]"
                        
            print("Bit Formulas (index = lane_id * 4 + element_index):")
            print(f"  Row bits: r3={row_bits[3]}, r2={row_bits[2]}, r1={row_bits[1]}, r0={row_bits[0]}")
            print(f"  Col bits: c3={col_bits[3]}, c2={col_bits[2]}, c1={col_bits[1]}, c0={col_bits[0]}")
            
            # Print a neat 16x16 grid showing which index (lane*4 + element) maps to each (r, c)
            grid = [["" for _ in range(16)] for _ in range(16)]
            for idx in range(256):
                r = rows[idx]
                c = cols[idx]
                lane = idx // 4
                el = idx % 4
                grid[r][c] = f"{lane:02d}.{el}"
            
            print("Grid Representation (Row x Col, cell value is lane.element):")
            for r in range(16):
                row_cells = " | ".join(grid[r])
                print(f"  Row {r:02d}: {row_cells}")

        analyze_matrix("A", row_A, col_A)
        analyze_matrix("B", row_B, col_B)
        analyze_matrix("D", row_D, col_D)

if __name__ == "__main__":
    main()
