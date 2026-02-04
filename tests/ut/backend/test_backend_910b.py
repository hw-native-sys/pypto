# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for Backend910B implementation."""

import tempfile
from pathlib import Path

from pypto import ir
from pypto.backend import Backend910B


class TestBackend910BConstruction:
    """Tests for 910B backend construction and basic properties."""

    def test_backend_910b_construction(self):
        """Test Backend910B constructs successfully with standard configuration."""
        backend = Backend910B()

        # Backend910B should construct successfully
        assert backend is not None
        assert backend.soc is not None

    def test_soc_structure(self):
        """Test SoC structure matches 910B specification."""
        backend = Backend910B()
        soc = backend.soc

        # 910B has 1 die with 24 AIC cores + 48 AIV cores = 72 total cores
        assert soc.total_die_count() == 1
        assert soc.total_core_count() == 24 + 48


class TestBackend910BMemoryPath:
    """Tests for 910B backend memory path finding."""

    def test_find_mem_paths(self):
        """Test finding memory paths between different memory spaces."""
        backend = Backend910B()

        # Test cases: (from, to, expected_path)
        test_cases = [
            # DDR connections
            (
                ir.MemorySpace.DDR,
                ir.MemorySpace.L0A,
                [ir.MemorySpace.DDR, ir.MemorySpace.L1, ir.MemorySpace.L0A],
            ),
            (ir.MemorySpace.DDR, ir.MemorySpace.UB, [ir.MemorySpace.DDR, ir.MemorySpace.UB]),
            # UB connections
            (ir.MemorySpace.UB, ir.MemorySpace.DDR, [ir.MemorySpace.UB, ir.MemorySpace.DDR]),
            # L1 connections
            (ir.MemorySpace.L1, ir.MemorySpace.L0A, [ir.MemorySpace.L1, ir.MemorySpace.L0A]),
            (ir.MemorySpace.L1, ir.MemorySpace.L0B, [ir.MemorySpace.L1, ir.MemorySpace.L0B]),
            # L0C connections
            (ir.MemorySpace.L0C, ir.MemorySpace.L1, [ir.MemorySpace.L0C, ir.MemorySpace.L1]),
            (ir.MemorySpace.L0C, ir.MemorySpace.DDR, [ir.MemorySpace.L0C, ir.MemorySpace.DDR]),
            # Same memory
            (ir.MemorySpace.L1, ir.MemorySpace.L1, [ir.MemorySpace.L1]),
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
        backend = Backend910B()

        # Test cases: (memory_type, expected_size_in_KB)
        test_cases = [
            (ir.MemorySpace.L0A, 64),  # 64KB per AIC core
            (ir.MemorySpace.L0B, 64),  # 64KB per AIC core
            (ir.MemorySpace.L0C, 128),  # 128KB per AIC core
            (ir.MemorySpace.L1, 512),  # 512KB per AIC core
            (ir.MemorySpace.UB, 192),  # 192KB per AIV core
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
        backend = Backend910B()

        # Test cases: (from, to, expected_path_length)
        test_cases = [
            # Direct connections (length 2)
            (ir.MemorySpace.DDR, ir.MemorySpace.UB, 2),
            (ir.MemorySpace.DDR, ir.MemorySpace.L1, 2),
            (ir.MemorySpace.UB, ir.MemorySpace.DDR, 2),
            (ir.MemorySpace.L1, ir.MemorySpace.L0A, 2),
            (ir.MemorySpace.L1, ir.MemorySpace.L0B, 2),
            (ir.MemorySpace.L0C, ir.MemorySpace.L1, 2),
            (ir.MemorySpace.L0C, ir.MemorySpace.DDR, 2),
        ]

        for from_mem, to_mem, expected_len in test_cases:
            path = backend.find_mem_path(from_mem, to_mem)
            assert len(path) == expected_len, (
                f"Path from {from_mem} to {to_mem} should have length {expected_len}, got {len(path)}: {path}"
            )
            assert path[0] == from_mem, f"Path should start with {from_mem}"
            assert path[-1] == to_mem, f"Path should end with {to_mem}"


class TestBackend910BSerialization:
    """Tests for 910B backend serialization and deserialization."""

    def test_export_import_backend(self):
        """Test exporting and importing Backend910B."""
        backend = Backend910B()

        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
            temp_path = f.name

        try:
            # Export backend
            backend.export_to_file(temp_path)

            # Import backend
            restored = Backend910B.import_from_file(temp_path)

            # Verify structure is preserved
            assert restored.soc.total_die_count() == 1
            assert restored.soc.total_core_count() == 72

            # Verify memory sizes are preserved (single mem size, not total)
            assert restored.get_mem_size(ir.MemorySpace.L0A) == 64 * 1024
            assert restored.get_mem_size(ir.MemorySpace.UB) == 192 * 1024
        finally:
            Path(temp_path).unlink()

    def test_export_import_memory_hierarchy(self):
        """Test that memory hierarchy is preserved after serialization."""
        backend = Backend910B()

        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
            temp_path = f.name

        try:
            backend.export_to_file(temp_path)
            restored = Backend910B.import_from_file(temp_path)

            # Verify memory paths work after deserialization
            path = restored.find_mem_path(ir.MemorySpace.DDR, ir.MemorySpace.L0A)
            assert len(path) == 3
            assert path[0] == ir.MemorySpace.DDR
            assert path[-1] == ir.MemorySpace.L0A
        finally:
            Path(temp_path).unlink()
