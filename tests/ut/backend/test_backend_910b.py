# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Backend910B_CCE and Backend910B_PTO implementation."""

import tempfile
from pathlib import Path

import pytest
from pypto import ir
from pypto.backend import Backend910B_CCE, Backend910B_PTO


class TestBackend910BConstruction:
    """Tests for 910B backend construction and basic properties."""

    def test_backend_910b_cce_construction(self):
        """Test Backend910B_CCE singleton instance."""
        backend = Backend910B_CCE.instance()

        # Backend910B_CCE should be accessible via singleton
        assert backend is not None
        assert backend.soc is not None
        assert backend.get_type_name() == "910B_CCE"

        # Verify singleton behavior
        backend2 = Backend910B_CCE.instance()
        assert backend is backend2

    def test_backend_910b_pto_construction(self):
        """Test Backend910B_PTO singleton instance."""
        backend = Backend910B_PTO.instance()

        # Backend910B_PTO should be accessible via singleton
        assert backend is not None
        assert backend.soc is not None
        assert backend.get_type_name() == "910B_PTO"

        # Verify singleton behavior
        backend2 = Backend910B_PTO.instance()
        assert backend is backend2

    def test_soc_structure(self):
        """Test SoC structure matches 910B specification."""
        backend = Backend910B_CCE.instance()
        soc = backend.soc

        # 910B has 1 die with 24 AIC cores + 48 AIV cores = 72 total cores
        assert soc.total_die_count() == 1
        assert soc.total_core_count() == 24 + 48


class TestBackend910BMemoryPath:
    """Tests for 910B backend memory path finding."""

    def test_find_mem_paths(self):
        """Test finding memory paths between different memory spaces."""
        backend = Backend910B_CCE.instance()

        # Test cases: (from, to, expected_path)
        test_cases = [
            # DDR connections
            (
                ir.MemorySpace.DDR,
                ir.MemorySpace.Left,
                [ir.MemorySpace.DDR, ir.MemorySpace.Mat, ir.MemorySpace.Left],
            ),
            (ir.MemorySpace.DDR, ir.MemorySpace.Vec, [ir.MemorySpace.DDR, ir.MemorySpace.Vec]),
            # UB connections
            (ir.MemorySpace.Vec, ir.MemorySpace.DDR, [ir.MemorySpace.Vec, ir.MemorySpace.DDR]),
            # L1 connections
            (ir.MemorySpace.Mat, ir.MemorySpace.Left, [ir.MemorySpace.Mat, ir.MemorySpace.Left]),
            (ir.MemorySpace.Mat, ir.MemorySpace.Right, [ir.MemorySpace.Mat, ir.MemorySpace.Right]),
            # Acc connections
            (ir.MemorySpace.Acc, ir.MemorySpace.Mat, [ir.MemorySpace.Acc, ir.MemorySpace.Mat]),
            (ir.MemorySpace.Acc, ir.MemorySpace.DDR, [ir.MemorySpace.Acc, ir.MemorySpace.DDR]),
            # Same memory
            (ir.MemorySpace.Mat, ir.MemorySpace.Mat, [ir.MemorySpace.Mat]),
        ]

        for from_mem, to_mem, expected_path in test_cases:
            path = backend.find_mem_path(from_mem, to_mem)
            assert path == expected_path, (
                f"Path from {from_mem} to {to_mem} should be {expected_path}, got {path}"
            )


class TestBackend910BMemorySize:
    """Tests for 910B backend memory size calculation."""

    def test_get_mem_sizes(self):
        """Test getting memory sizes for different memory types."""
        backend = Backend910B_CCE.instance()

        # Test cases: (memory_type, expected_size_in_KB)
        test_cases = [
            (ir.MemorySpace.Left, 64),  # 64KB per AIC core
            (ir.MemorySpace.Right, 64),  # 64KB per AIC core
            (ir.MemorySpace.Acc, 128),  # 128KB per AIC core
            (ir.MemorySpace.Mat, 512),  # 512KB per AIC core
            (ir.MemorySpace.Vec, 192),  # 192KB per AIV core
            (ir.MemorySpace.DDR, 0),  # DDR not in core memory
        ]

        for mem_type, expected_kb in test_cases:
            mem_size = backend.get_mem_size(mem_type)
            expected_size = expected_kb * 1024
            assert mem_size == expected_size, (
                f"Memory size for {mem_type} should be {expected_kb}KB ({expected_size} bytes), "
                f"got {mem_size} bytes"
            )


class TestBackend910BMemoryHierarchy:
    """Tests for 910B memory hierarchy configuration."""

    def test_memory_hierarchy_connections(self):
        """Test memory hierarchy connections are correctly configured."""
        backend = Backend910B_CCE.instance()

        # Test cases: (from, to, expected_path_length)
        test_cases = [
            # Direct connections (length 2)
            (ir.MemorySpace.DDR, ir.MemorySpace.Vec, 2),
            (ir.MemorySpace.DDR, ir.MemorySpace.Mat, 2),
            (ir.MemorySpace.Vec, ir.MemorySpace.DDR, 2),
            (ir.MemorySpace.Mat, ir.MemorySpace.Left, 2),
            (ir.MemorySpace.Mat, ir.MemorySpace.Right, 2),
            (ir.MemorySpace.Acc, ir.MemorySpace.Mat, 2),
            (ir.MemorySpace.Acc, ir.MemorySpace.DDR, 2),
        ]

        for from_mem, to_mem, expected_len in test_cases:
            path = backend.find_mem_path(from_mem, to_mem)
            assert len(path) == expected_len, (
                f"Path from {from_mem} to {to_mem} should have length {expected_len}, got {len(path)}: {path}"
            )
            assert path[0] == from_mem, f"Path should start with {from_mem}"
            assert path[-1] == to_mem, f"Path should end with {to_mem}"


class TestBackend910BSerialization:
    """Tests for 910B backend serialization."""

    def test_export_backend(self):
        """Test exporting Backend910B_CCE singleton."""
        backend = Backend910B_CCE.instance()

        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
            temp_path = f.name

        try:
            # Export backend (singleton can be serialized for inspection)
            backend.export_to_file(temp_path)

            # Verify file was created
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
