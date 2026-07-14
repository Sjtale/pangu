#!/usr/bin/env python3
"""
Validation Script: test_p2_memory_optimizations.py

Consolidated test suite to validate the syntax and logic of the P2 memory optimizations
(global region_ids sharing and eager earth_position_index release/restore).

Usage (to be executed manually by the user on the competition server):
python module_test_scripts/test_p2_memory_optimizations.py
"""

import os
import sys
import unittest
import py_compile

def test_syntax():
    print("=" * 60)
    print("STEP 1: Verify syntax correctness of p2_tiled_attention.py")
    print("=" * 60)
    
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(parent_dir, "p2_tiled_attention.py"),
        os.path.join(parent_dir, "pangu_weather", "p2_tiled_attention.py"),
        os.path.join(parent_dir, "src", "xiandao2026-AI4S", "pangu_weather", "p2_tiled_attention.py"),
    ]
    adapter_path = None
    for path in candidates:
        if os.path.exists(path):
            adapter_path = path
            break
        
    if not adapter_path:
        print(f"  [ERROR] p2_tiled_attention.py not found! Tried candidates: {candidates}")
        return False
        
    try:
        py_compile.compile(adapter_path, doraise=True)
        print(f"  [SUCCESS] Python syntax is correct for: {adapter_path}")
        return True
    except py_compile.PyCompileError as e:
        print(f"  [ERROR] Syntax error detected in p2_tiled_attention.py:\n{e}")
        return False


def test_logic():
    print("\n" + "=" * 60)
    print("STEP 2: Run P2 Memory Optimization logic checks")
    print("=" * 60)

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        parent_dir,
        os.path.join(parent_dir, "pangu_weather"),
        os.path.join(parent_dir, "src", "xiandao2026-AI4S", "pangu_weather"),
    ]
    pangu_dir = None
    for path in candidates:
        if os.path.exists(os.path.join(path, "p2_tiled_attention.py")):
            pangu_dir = path
            break
        
    if not pangu_dir:
        print(f"  [ERROR] Cannot find directory containing p2_tiled_attention.py! Tried candidates: {candidates}")
        return False
        
    if pangu_dir not in sys.path:
        sys.path.insert(0, pangu_dir)

    try:
        import torch
        import p2_tiled_attention as adapter
        print("  - Successfully imported PyTorch and p2_tiled_attention.")
    except Exception as e:
        print(f"  [ERROR] Failed to import PyTorch or adapter: {e}")
        return False

    # Define a dummy EarthAttention3D module
    class EarthAttention3D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.window_size = (2, 4, 4)
            self.num_heads = 3
            self.dim = 96
            self.qkv = torch.nn.Identity()
            self.proj = torch.nn.Identity()
            self.earth_position_bias_table = torch.nn.Parameter(
                torch.arange(42, dtype=torch.float16).view(7, 2, 3)
            )
            self.register_buffer(
                "earth_position_index",
                torch.arange(32 * 32, dtype=torch.int64).remainder(7),
            )

        def forward(self, x, _mask=None):
            return x

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = EarthAttention3D()
            self.attn2 = EarthAttention3D()

    model = MockModel()
    
    # Save original buffers for comparison
    orig_bias = model.attn1.earth_position_bias_table.detach().clone()
    orig_index = model.attn1.earth_position_index.detach().clone()
    
    # Mock backend functions to avoid HIP loading on CPU/no-DTK environments
    mock_regions = torch.arange(288, dtype=torch.uint8).view(2, 144)
    original_backend = adapter._backend
    adapter._backend = lambda: (
        lambda index, bias_rows: index.to(dtype=torch.int16).contiguous(),
        object(),
        lambda table: table.permute(2, 1, 0).contiguous(),
        lambda _m: mock_regions,
    )
    
    try:
        print("  - Enabling P2 tiled attention with release_original_bias=True...")
        patched = adapter.enable_p2_tiled_attention(
            model,
            strict=True,
            release_original_bias=True,
            retain_cpu_backup=True,
        )
        
        print(f"  - Patched modules count: {patched}")
        if patched != 2:
            print(f"  [ERROR] Expected 2 modules to be patched, got {patched}")
            return False
            
        # Verify eager release of bias and index
        print("  - Verifying bias parameter and index buffer release...")
        if model.attn1.earth_position_bias_table.numel() != 0:
            print("  [ERROR] earth_position_bias_table was not eagerly released (size > 0)!")
            return False
        if model.attn1.earth_position_index.numel() != 0:
            print("  [ERROR] earth_position_index was not eagerly released (size > 0)!")
            return False
        print("    [SUCCESS] Both bias table and position index are successfully cleared from GPU memory.")
        
        # Verify region_ids global caching
        print("  - Verifying global region_ids cache sharing...")
        mask = torch.zeros(2, 144, 144, dtype=torch.float16)
        x = torch.zeros(2, 8, 144, 96, dtype=torch.float16)
        
        r1 = adapter._cached_region_ids(model.attn1, mask, x)
        r2 = adapter._cached_region_ids(model.attn2, mask, x)
        
        if r1.data_ptr() != r2.data_ptr() or r1 is not r2:
            print("  [ERROR] Global cache mismatch! Modules did not share the same region_ids tensor.")
            return False
            
        if len(adapter._GLOBAL_REGION_IDS_CACHE) != 1:
            print(f"  [ERROR] Expected cache size 1, got {len(adapter._GLOBAL_REGION_IDS_CACHE)}")
            return False
        print("    [SUCCESS] Region IDs are shared globally between modules (saving duplicate VRAM allocations).")
        
        # Verify rollback restores everything
        print("  - Disabling P2 tiled attention (rollback)...")
        restored = adapter.disable_p2_tiled_attention(model)
        if restored != 2:
            print(f"  [ERROR] Expected 2 restored modules, got {restored}")
            return False
            
        if not torch.equal(model.attn1.earth_position_bias_table, orig_bias):
            print("  [ERROR] Restored bias table does not match original values!")
            return False
        if not torch.equal(model.attn1.earth_position_index, orig_index):
            print("  [ERROR] Restored position index does not match original values!")
            return False
            
        if len(adapter._GLOBAL_REGION_IDS_CACHE) != 0:
            print("  [ERROR] Cache was not cleared after disable!")
            return False
            
        print("    [SUCCESS] Rollback successfully restored original bias tables and index buffers.")
        print("  [SUCCESS] All logic verification tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Logic verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        adapter._backend = original_backend


def main():
    success_syntax = test_syntax()
    success_logic = test_logic()
    
    print("\n" + "=" * 60)
    if success_syntax and success_logic:
        print("SUMMARY: ALL COMPREHENSIVE TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SUMMARY: TEST SUITE FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
