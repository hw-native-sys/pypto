# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Backend950 implementation."""

import tempfile
from pathlib import Path

import pytest
from pypto import ir
from pypto.backend import Backend950, BackendType


class TestBackend950Construction:
    """Tests for 950 backend construction and basic properties."""

    def test_backend_950_construction(self):
        """Test Backend950 singleton instance is accessible and valid."""
        backend = Backend950.instance()

        assert backend is not None
        assert backend.soc is not None
        assert backend.get_type_name() == "950"

    def test_backend_950_singleton(self):
        """Test Backend950 follows singleton pattern."""
        backend1 = Backend950.instance()
        backend2 = Backend950.instance()

        assert backend1 is backend2

    def test_soc_structure(self):
        """Test SoC structure matches 950 specification.

        950 SoC: 2 dies, each with 18 mix clusters (1 AIC + 2 AIV per cluster).
        Total cores per die = 18 * (1 + 2) = 54.
        Total cores = 2 * 54 = 108.
        """
        backend = Backend950.instance()
        soc = backend.soc

        # 950 has 2 dies
        assert soc.total_die_count() == 2
        # 18 clusters per die * 2 dies = 36 clusters total
        assert soc.total_cluster_count() == 36
        # 3 cores per cluster (1 AIC + 2 AIV) * 36 clusters = 108 total cores
        assert soc.total_core_count() == 108

    def test_backend_type_enum(self):
        """Test BackendType.Ascend950 enum value exists."""
        assert hasattr(BackendType, "Ascend950")


class TestBackend950MemorySize:
    """Tests for 950 backend memory size configuration."""

    def test_get_mem_sizes(self):
        """Test memory sizes match 950 hardware specification."""
        backend = Backend950.instance()

        # Test cases: (memory_type, expected_size_in_KB)
        # Based on Create950SoC() in soc.cpp:
        #   AIC core: Mat=512KB, Left=64KB, Right=64KB, Acc=256KB, Bias=4KB
        #   AIV core: Vec=248KB
        test_cases = [
            (ir.MemorySpace.Mat, 512),  # 512KB Mat per AIC core
            (ir.MemorySpace.Left, 64),  # 64KB Left per AIC core
            (ir.MemorySpace.Right, 64),  # 64KB Right per AIC core
            (ir.MemorySpace.Acc, 256),  # 256KB Acc per AIC core
            (ir.MemorySpace.Vec, 248),  # 248KB Vec per AIV core
            (ir.MemorySpace.DDR, 0),  # DDR not in core memory
        ]

        for mem_type, expected_kb in test_cases:
            mem_size = backend.get_mem_size(mem_type)
            expected_size = expected_kb * 1024
            assert mem_size == expected_size, (
                f"Memory size for {mem_type} should be {expected_kb}KB ({expected_size} bytes), "
                f"got {mem_size} bytes"
            )


class TestBackend950MemoryPath:
    """Tests for 950 backend memory path finding."""

    def test_find_mem_paths(self):
        """Test finding memory paths between different memory spaces.

        950 memory graph (same topology as 910B):
          DDR -> Vec, Mat
          Vec -> Mat, DDR
          Mat -> Left, Right
          Acc -> Vec, Mat, DDR
        """
        backend = Backend950.instance()

        # Test cases: (from, to, expected_path)
        test_cases = [
            # DDR connections
            (
                ir.MemorySpace.DDR,
                ir.MemorySpace.Left,
                [ir.MemorySpace.DDR, ir.MemorySpace.Mat, ir.MemorySpace.Left],
            ),
            (ir.MemorySpace.DDR, ir.MemorySpace.Vec, [ir.MemorySpace.DDR, ir.MemorySpace.Vec]),
            (ir.MemorySpace.DDR, ir.MemorySpace.Mat, [ir.MemorySpace.DDR, ir.MemorySpace.Mat]),
            # Vec connections
            (ir.MemorySpace.Vec, ir.MemorySpace.DDR, [ir.MemorySpace.Vec, ir.MemorySpace.DDR]),
            # Mat connections
            (ir.MemorySpace.Mat, ir.MemorySpace.Left, [ir.MemorySpace.Mat, ir.MemorySpace.Left]),
            (ir.MemorySpace.Mat, ir.MemorySpace.Right, [ir.MemorySpace.Mat, ir.MemorySpace.Right]),
            # Acc connections
            (ir.MemorySpace.Acc, ir.MemorySpace.Mat, [ir.MemorySpace.Acc, ir.MemorySpace.Mat]),
            (ir.MemorySpace.Acc, ir.MemorySpace.DDR, [ir.MemorySpace.Acc, ir.MemorySpace.DDR]),
            # Same memory
            (ir.MemorySpace.Vec, ir.MemorySpace.Vec, [ir.MemorySpace.Vec]),
            (ir.MemorySpace.Mat, ir.MemorySpace.Mat, [ir.MemorySpace.Mat]),
        ]

        for from_mem, to_mem, expected_path in test_cases:
            path = backend.find_mem_path(from_mem, to_mem)
            assert path == expected_path, (
                f"Path from {from_mem} to {to_mem} should be {expected_path}, got {path}"
            )

    def test_memory_hierarchy_path_lengths(self):
        """Test that memory path lengths match expected hop counts."""
        backend = Backend950.instance()

        # Test cases: (from, to, expected_length)
        test_cases = [
            # Direct connections (length 2)
            (ir.MemorySpace.DDR, ir.MemorySpace.Vec, 2),
            (ir.MemorySpace.DDR, ir.MemorySpace.Mat, 2),
            (ir.MemorySpace.Vec, ir.MemorySpace.DDR, 2),
            (ir.MemorySpace.Mat, ir.MemorySpace.Left, 2),
            (ir.MemorySpace.Mat, ir.MemorySpace.Right, 2),
            (ir.MemorySpace.Acc, ir.MemorySpace.Mat, 2),
            (ir.MemorySpace.Acc, ir.MemorySpace.DDR, 2),
            # Two-hop connections (length 3)
            (ir.MemorySpace.DDR, ir.MemorySpace.Left, 3),
            (ir.MemorySpace.DDR, ir.MemorySpace.Right, 3),
        ]

        for from_mem, to_mem, expected_len in test_cases:
            path = backend.find_mem_path(from_mem, to_mem)
            assert len(path) == expected_len, (
                f"Path from {from_mem} to {to_mem} should have length {expected_len}, got {len(path)}: {path}"
            )
            assert path[0] == from_mem, f"Path should start with {from_mem}"
            assert path[-1] == to_mem, f"Path should end with {to_mem}"


class TestBackend950Serialization:
    """Tests for 950 backend serialization."""

    def test_export_backend(self):
        """Test exporting Backend950 to file."""
        backend = Backend950.instance()

        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
            temp_path = f.name

        try:
            backend.export_to_file(temp_path)
            assert Path(temp_path).exists()
            assert Path(temp_path).stat().st_size > 0
        finally:
            Path(temp_path).unlink()

    def test_export_and_check_type_name(self):
        """Test that Backend950 type name is correctly reported as '950'."""
        backend = Backend950.instance()

        # Verify type name without needing to parse msgpack
        assert backend.get_type_name() == "950"

    def test_export_nonempty_file(self):
        """Test that exported msgpack file is non-empty."""
        backend = Backend950.instance()

        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
            temp_path = f.name

        try:
            backend.export_to_file(temp_path)
            assert Path(temp_path).stat().st_size > 0
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
