# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comprehensive tests for MemRef, MemorySpace, and TileView."""

import pytest
from pypto import DataType, ir
from pypto.ir import IRBuilder


class TestMemorySpace:
    """Tests for MemorySpace enum."""

    def test_memory_space_values(self):
        """Test all MemorySpace enum values."""
        assert ir.MemorySpace.DDR is not None
        assert ir.MemorySpace.UB is not None
        assert ir.MemorySpace.L1 is not None
        assert ir.MemorySpace.L0A is not None
        assert ir.MemorySpace.L0B is not None
        assert ir.MemorySpace.L0C is not None

    def test_memory_space_equality(self):
        """Test MemorySpace enum equality."""
        assert ir.MemorySpace.DDR == ir.MemorySpace.DDR
        assert ir.MemorySpace.UB == ir.MemorySpace.UB
        assert ir.MemorySpace.DDR != ir.MemorySpace.UB

    def test_memory_space_in_dict(self):
        """Test using MemorySpace as dictionary keys."""
        memory_map = {
            ir.MemorySpace.DDR: "off-chip",
            ir.MemorySpace.UB: "on-chip",
            ir.MemorySpace.L1: "L1 cache",
        }
        assert memory_map[ir.MemorySpace.DDR] == "off-chip"
        assert memory_map[ir.MemorySpace.UB] == "on-chip"
        assert memory_map[ir.MemorySpace.L1] == "L1 cache"


class TestMemRef:
    """Tests for MemRef struct."""

    def test_memref_creation_with_params(self):
        """Test creating a MemRef with all parameters."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)
        memref = ir.MemRef(ir.MemorySpace.DDR, addr, 1024, 0)
        assert memref is not None
        assert memref.id_ == 0

    def test_memref_set_attributes(self):
        """Test setting MemRef attributes."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)

        memref = ir.MemRef(ir.MemorySpace.DDR, addr, 1024, 0)

        assert memref.memory_space_ == ir.MemorySpace.DDR
        assert memref.addr_.same_as(addr)
        assert memref.size_ == 1024

    def test_memref_different_memory_spaces(self):
        """Test MemRef with different memory spaces."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)

        # Test each memory space
        for mem_space in [
            ir.MemorySpace.DDR,
            ir.MemorySpace.UB,
            ir.MemorySpace.L1,
            ir.MemorySpace.L0A,
            ir.MemorySpace.L0B,
            ir.MemorySpace.L0C,
        ]:
            memref = ir.MemRef(mem_space, addr, 2048, 1)
            assert memref.memory_space_ == mem_space

    def test_memref_with_symbolic_address(self):
        """Test MemRef with symbolic address expression."""
        span = ir.Span.unknown()
        base_addr = ir.Var("base_addr", ir.ScalarType(DataType.INT64), span)
        offset = ir.ConstInt(128, DataType.INT64, span)
        addr_expr = ir.Add(base_addr, offset, DataType.INT64, span)

        memref = ir.MemRef(ir.MemorySpace.UB, addr_expr, 4096, 2)

        assert isinstance(memref.addr_, ir.Add)
        assert memref.size_ == 4096

    def test_memref_large_size(self):
        """Test MemRef with large size values (uint64)."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)

        memref = ir.MemRef(ir.MemorySpace.DDR, addr, 2**32, 3)  # 4GB

        assert memref.size_ == 2**32

    def test_memref_zero_address(self):
        """Test MemRef with zero address."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)

        memref = ir.MemRef(ir.MemorySpace.L1, addr, 512, 4)

        assert isinstance(memref.addr_, ir.ConstInt)
        assert memref.addr_.value == 0


class TestTileView:
    """Tests for TileView struct."""

    def test_tileview_creation_empty(self):
        """Test creating an empty TileView."""
        tile_view = ir.TileView()
        assert tile_view is not None

    def test_tileview_set_attributes(self):
        """Test setting TileView attributes."""
        span = ir.Span.unknown()
        valid_shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        start_offset = ir.ConstInt(0, DataType.INT64, span)

        tile_view = ir.TileView()
        tile_view.valid_shape = valid_shape
        tile_view.stride = stride
        tile_view.start_offset = start_offset

        assert len(tile_view.valid_shape) == 2
        assert len(tile_view.stride) == 2
        assert isinstance(tile_view.start_offset, ir.Expr)

    def test_tileview_symbolic_dimensions(self):
        """Test TileView with symbolic dimensions."""
        span = ir.Span.unknown()
        M = ir.Var("M", ir.ScalarType(DataType.INT64), span)
        N = ir.Var("N", ir.ScalarType(DataType.INT64), span)

        tile_view = ir.TileView()
        tile_view.valid_shape = [M, N]
        tile_view.stride = [ir.ConstInt(1, DataType.INT64, span), M]
        tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

        assert isinstance(tile_view.valid_shape[0], ir.Var)
        assert isinstance(tile_view.valid_shape[1], ir.Var)

    def test_tileview_non_contiguous_stride(self):
        """Test TileView with non-contiguous stride."""
        span = ir.Span.unknown()

        tile_view = ir.TileView()
        tile_view.valid_shape = [
            ir.ConstInt(8, DataType.INT64, span),
            ir.ConstInt(8, DataType.INT64, span),
        ]
        tile_view.stride = [
            ir.ConstInt(2, DataType.INT64, span),
            ir.ConstInt(32, DataType.INT64, span),
        ]
        tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

        # Verify non-unit stride in first dimension
        assert isinstance(tile_view.stride[0], ir.ConstInt)
        assert tile_view.stride[0].value == 2


class TestTensorTypeWithMemRef:
    """Tests for TensorType with MemRef."""

    def test_tensor_type_without_memref(self):
        """Test TensorType creation without MemRef."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(20, DataType.INT64, span),
        ]

        tensor_type = ir.TensorType(shape, DataType.FP32)
        assert tensor_type.dtype == DataType.FP32
        assert len(tensor_type.shape) == 2
        assert tensor_type.memref is None

    def test_tensor_type_with_memref(self):
        """Test TensorType creation with MemRef."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(10, DataType.INT64, span),
            ir.ConstInt(20, DataType.INT64, span),
        ]

        # Create MemRef
        memref = ir.MemRef(
            ir.MemorySpace.DDR,
            ir.ConstInt(0x1000, DataType.INT64, span),
            10 * 20 * 4,  # 10x20 FP32 elements
            32,  # ID
        )

        tensor_type = ir.TensorType(shape, DataType.FP32, memref)
        assert tensor_type.memref is not None
        assert tensor_type.memref.memory_space_ == ir.MemorySpace.DDR
        assert tensor_type.memref.size_ == 800

    def test_tensor_type_memref_different_spaces(self):
        """Test TensorType with MemRef in different memory spaces."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(32, DataType.INT64, span)]

        for mem_space in [ir.MemorySpace.DDR, ir.MemorySpace.UB, ir.MemorySpace.L1]:
            memref = ir.MemRef(mem_space, ir.ConstInt(0, DataType.INT64, span), 128, 10)

            tensor_type = ir.TensorType(shape, DataType.FP32, memref)
            assert tensor_type.memref is not None
            assert tensor_type.memref.memory_space_ == mem_space

    def test_tensor_var_with_memref(self):
        """Test Var with TensorType containing MemRef."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(64, DataType.INT64, span)]

        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0x2000, DataType.INT64, span), 256, 11)

        tensor_type = ir.TensorType(shape, DataType.FP16, memref)
        tensor_var = ir.Var("tensor_ub", tensor_type, span)

        assert isinstance(tensor_var.type, ir.TensorType)
        assert tensor_var.type.memref is not None
        assert tensor_var.type.memref.memory_space_ == ir.MemorySpace.UB


class TestTileTypeWithMemRef:
    """Tests for TileType with MemRef and TileView."""

    def test_tile_type_without_memref(self):
        """Test TileType creation without MemRef."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        tile_type = ir.TileType(shape, DataType.FP32)
        assert tile_type.dtype == DataType.FP32
        assert len(tile_type.shape) == 2
        assert tile_type.memref is None
        assert tile_type.tile_view is None

    def test_tile_type_with_memref(self):
        """Test TileType creation with MemRef."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16 * 16 * 4, 12)

        tile_type = ir.TileType(shape, DataType.FP32, memref)
        assert tile_type.memref is not None
        assert tile_type.memref.memory_space_ == ir.MemorySpace.UB

    def test_tile_type_with_memref_and_tileview(self):
        """Test TileType with both MemRef and TileView."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        # Create MemRef
        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 16 * 16 * 2, 13)

        # Create TileView
        tile_view = ir.TileView()
        tile_view.valid_shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        tile_view.stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

        tile_type = ir.TileType(shape, DataType.FP16, memref, tile_view)
        assert tile_type.memref is not None
        assert tile_type.tile_view is not None
        assert len(tile_type.tile_view.valid_shape) == 2

    def test_tile_type_1d_with_memref(self):
        """Test 1D TileType with MemRef."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(32, DataType.INT64, span)]

        memref = ir.MemRef(ir.MemorySpace.L0A, ir.ConstInt(0, DataType.INT64, span), 128, 14)

        tile_type = ir.TileType(shape, DataType.FP32, memref)
        assert len(tile_type.shape) == 1
        assert tile_type.memref is not None
        assert tile_type.memref.memory_space_ == ir.MemorySpace.L0A

    def test_tile_type_3d_now_supported(self):
        """Test that TileType now accepts 3D shapes (multi-dimensional support)."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(8, DataType.INT64, span),
            ir.ConstInt(8, DataType.INT64, span),
            ir.ConstInt(8, DataType.INT64, span),
        ]

        # This should now succeed
        tile_type = ir.TileType(shape, DataType.FP32)
        assert len(tile_type.shape) == 3
        assert tile_type.dtype == DataType.FP32

    def test_tile_type_4d_supported(self):
        """Test that TileType accepts 4D shapes."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(2, DataType.INT64, span),
            ir.ConstInt(4, DataType.INT64, span),
            ir.ConstInt(8, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        tile_type = ir.TileType(shape, DataType.FP16)
        assert len(tile_type.shape) == 4
        assert tile_type.dtype == DataType.FP16

    def test_tile_type_5d_supported(self):
        """Test that TileType accepts 5D shapes."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(2, DataType.INT64, span),
            ir.ConstInt(3, DataType.INT64, span),
            ir.ConstInt(4, DataType.INT64, span),
            ir.ConstInt(8, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        tile_type = ir.TileType(shape, DataType.FP32)
        assert len(tile_type.shape) == 5
        assert tile_type.dtype == DataType.FP32

    def test_tile_type_3d_with_memref(self):
        """Test that TileType with MemRef accepts 3D shapes."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(4, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 4 * 16 * 16 * 2, 50)

        tile_type = ir.TileType(shape, DataType.FP16, memref)
        assert len(tile_type.shape) == 3
        assert tile_type.memref is not None
        assert tile_type.memref.memory_space_ == ir.MemorySpace.UB

    def test_tile_var_with_memref_l0c(self):
        """Test Var with TileType containing MemRef in L0C."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        memref = ir.MemRef(ir.MemorySpace.L0C, ir.ConstInt(0, DataType.INT64, span), 512, 15)

        tile_type = ir.TileType(shape, DataType.FP16, memref)
        tile_var = ir.Var("output_tile", tile_type, span)

        assert isinstance(tile_var.type, ir.TileType)
        assert tile_var.type.memref is not None
        assert tile_var.type.memref.memory_space_ == ir.MemorySpace.L0C


class TestMemRefSerialization:
    """Tests for MemRef serialization and deserialization."""

    def test_serialize_tensor_with_memref(self):
        """Test serializing TensorType with MemRef."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(10, DataType.INT64, span)]

        memref = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0x1000, DataType.INT64, span), 40, 16)

        tensor_type = ir.TensorType(shape, DataType.FP32, memref)
        tensor_var = ir.Var("tensor", tensor_type, span)

        # Serialize
        data = ir.serialize(tensor_var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TensorType)
        assert restored.type.memref is not None
        assert restored.type.memref.memory_space_ == ir.MemorySpace.DDR

    def test_serialize_tile_with_memref_and_view(self):
        """Test serializing TileType with MemRef and TileView."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 512, 17)

        tile_view = ir.TileView()
        tile_view.valid_shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        tile_view.stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        tile_view.start_offset = ir.ConstInt(0, DataType.INT64, span)

        tile_type = ir.TileType(shape, DataType.FP16, memref, tile_view)
        tile_var = ir.Var("tile", tile_type, span)

        # Serialize
        data = ir.serialize(tile_var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TileType)
        assert restored.type.memref is not None
        assert restored.type.tile_view is not None
        assert len(restored.type.tile_view.valid_shape) == 2

    def test_serialize_tensor_with_tensorview(self):
        """Test serializing TensorType with TensorView."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(128, DataType.INT64, span),
            ir.ConstInt(256, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(128, DataType.INT64, span),
        ]

        tensor_view = ir.TensorView(stride, ir.TensorLayout.ND)
        tensor_type = ir.TensorType(shape, DataType.FP32, memref=None, tensor_view=tensor_view)
        var = ir.Var("t", tensor_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TensorType)
        assert restored.type.tensor_view is not None
        assert restored.type.tensor_view.layout == ir.TensorLayout.ND
        assert len(restored.type.tensor_view.stride) == 2

    def test_serialize_tensor_with_memref_and_tensorview(self):
        """Test serializing TensorType with both MemRef and TensorView."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(64, DataType.INT64, span),
            ir.ConstInt(64, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(64, DataType.INT64, span),
        ]

        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0x3000, DataType.INT64, span), 8192, 25)
        tensor_view = ir.TensorView(stride, ir.TensorLayout.DN)
        tensor_type = ir.TensorType(shape, DataType.FP16, memref=memref, tensor_view=tensor_view)
        var = ir.Var("t", tensor_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TensorType)
        assert restored.type.memref is not None
        assert restored.type.memref.memory_space_ == ir.MemorySpace.UB
        assert restored.type.tensor_view is not None
        assert restored.type.tensor_view.layout == ir.TensorLayout.DN
        assert len(restored.type.tensor_view.stride) == 2

    def test_serialize_assign_with_memref(self):
        """Test serializing AssignStmt with MemRef-enabled types."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(32, DataType.INT64, span)]

        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0x4000, DataType.INT64, span), 128, 18)

        tensor_type = ir.TensorType(shape, DataType.FP32, memref)
        lhs = ir.Var("result", tensor_type, span)
        rhs = ir.Var("input", tensor_type, span)

        stmt = ir.AssignStmt(lhs, rhs, span)

        # Serialize
        data = ir.serialize(stmt)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)
        assert isinstance(restored, ir.AssignStmt)
        assert isinstance(restored.var.type, ir.ShapedType)
        assert restored.var.type.memref is not None


class TestMemRefStandaloneSerialization:
    """Tests for standalone MemRef node serialization and deserialization."""

    def test_serialize_standalone_memref(self):
        """Serialize/deserialize a standalone MemRef and verify all fields."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0x1000, DataType.INT64, span)
        memref = ir.MemRef(ir.MemorySpace.UB, addr, 2048, 42)

        data = ir.serialize(memref)
        restored = ir.deserialize(data)

        assert isinstance(restored, ir.MemRef)
        assert restored.memory_space_ == ir.MemorySpace.UB
        assert isinstance(restored.addr_, ir.ConstInt)
        assert restored.addr_.value == 0x1000
        assert restored.size_ == 2048
        assert restored.id_ == 42

    def test_serialize_memref_all_memory_spaces(self):
        """Verify all MemorySpace values round-trip correctly."""
        span = ir.Span.unknown()
        for mem_space in [
            ir.MemorySpace.DDR,
            ir.MemorySpace.UB,
            ir.MemorySpace.L1,
            ir.MemorySpace.L0A,
            ir.MemorySpace.L0B,
            ir.MemorySpace.L0C,
        ]:
            addr = ir.ConstInt(0, DataType.INT64, span)
            memref = ir.MemRef(mem_space, addr, 1024, 1)

            data = ir.serialize(memref)
            restored = ir.deserialize(data)

            assert isinstance(restored, ir.MemRef)
            assert restored.memory_space_ == mem_space

    def test_serialize_memref_roundtrip_structural_equal(self):
        """Serialize a MemRef, deserialize, and verify structural equality."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(256, DataType.INT64, span)
        memref = ir.MemRef(ir.MemorySpace.L0A, addr, 4096, 7)

        data = ir.serialize(memref)
        restored = ir.deserialize(data)

        ir.assert_structural_equal(memref, restored, enable_auto_mapping=True)

    def test_serialize_program_with_memref(self):
        """Serialize a program containing MemRef nodes in function bodies."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(32, DataType.INT64, span)]

        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 128, 10)
        tensor_type = ir.TensorType(shape, DataType.FP32, memref)
        lhs = ir.Var("result", tensor_type, span)
        rhs = ir.Var("input", tensor_type, span)
        body = ir.AssignStmt(lhs, rhs, span)

        func = ir.Function("test_fn", [rhs], [tensor_type], body, span)
        program = ir.Program([func], "test_prog", span)

        data = ir.serialize(program)
        restored = ir.deserialize(data)

        ir.assert_structural_equal(program, restored, enable_auto_mapping=True)


class TestMemRefStructuralComparison:
    """Tests for structural comparison with MemRef."""

    def test_standalone_memref_structural_equal(self):
        """Two standalone MemRefs with same fields (different ids) should be equal."""
        span = ir.Span.unknown()
        addr1 = ir.ConstInt(0, DataType.INT64, span)
        addr2 = ir.ConstInt(0, DataType.INT64, span)
        memref1 = ir.MemRef(ir.MemorySpace.DDR, addr1, 1024, 1)
        memref2 = ir.MemRef(ir.MemorySpace.DDR, addr2, 1024, 99)

        ir.assert_structural_equal(memref1, memref2, enable_auto_mapping=True)

    def test_standalone_memref_different_memory_space_not_equal(self):
        """Two standalone MemRefs with different memory spaces should NOT be equal."""
        span = ir.Span.unknown()
        addr1 = ir.ConstInt(0, DataType.INT64, span)
        addr2 = ir.ConstInt(0, DataType.INT64, span)
        memref1 = ir.MemRef(ir.MemorySpace.DDR, addr1, 1024, 1)
        memref2 = ir.MemRef(ir.MemorySpace.UB, addr2, 1024, 2)

        assert not ir.structural_equal(memref1, memref2, enable_auto_mapping=True)

    def test_standalone_memref_different_size_not_equal(self):
        """Two standalone MemRefs with different sizes should NOT be equal."""
        span = ir.Span.unknown()
        addr1 = ir.ConstInt(0, DataType.INT64, span)
        addr2 = ir.ConstInt(0, DataType.INT64, span)
        memref1 = ir.MemRef(ir.MemorySpace.DDR, addr1, 1024, 1)
        memref2 = ir.MemRef(ir.MemorySpace.DDR, addr2, 2048, 2)

        assert not ir.structural_equal(memref1, memref2, enable_auto_mapping=True)

    def test_standalone_memref_different_addr_not_equal(self):
        """Two standalone MemRefs with different addr expressions should NOT be equal."""
        span = ir.Span.unknown()
        addr1 = ir.ConstInt(0, DataType.INT64, span)
        addr2 = ir.ConstInt(100, DataType.INT64, span)
        memref1 = ir.MemRef(ir.MemorySpace.DDR, addr1, 1024, 1)
        memref2 = ir.MemRef(ir.MemorySpace.DDR, addr2, 1024, 2)

        assert not ir.structural_equal(memref1, memref2, enable_auto_mapping=True)

    def test_standalone_memref_structural_hash_different_fields(self):
        """Two MemRefs with different memory_space produce different hashes."""
        span = ir.Span.unknown()
        addr1 = ir.ConstInt(0, DataType.INT64, span)
        addr2 = ir.ConstInt(0, DataType.INT64, span)
        memref1 = ir.MemRef(ir.MemorySpace.DDR, addr1, 512, 1)
        memref2 = ir.MemRef(ir.MemorySpace.UB, addr2, 512, 2)

        hash1 = ir.structural_hash(memref1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(memref2, enable_auto_mapping=True)
        assert hash1 != hash2

    def test_tensor_with_same_memref_structural_equal(self):
        """Test structural equality of tensors with identical MemRef."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(10, DataType.INT64, span)]

        memref1 = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 40, 19)
        memref2 = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 40, 20)

        tensor_type1 = ir.TensorType(shape, DataType.FP32, memref1)
        tensor_type2 = ir.TensorType(shape, DataType.FP32, memref2)

        var1 = ir.Var("t1", tensor_type1, span)
        var2 = ir.Var("t2", tensor_type2, span)

        ir.assert_structural_equal(var1, var2, enable_auto_mapping=True)

    def test_tensor_with_different_memref_not_equal(self):
        """Test that tensors with different MemRef are not structurally equal."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(10, DataType.INT64, span)]

        memref1 = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 40, 21)
        memref2 = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 40, 22)

        tensor_type1 = ir.TensorType(shape, DataType.FP32, memref1)
        tensor_type2 = ir.TensorType(shape, DataType.FP32, memref2)

        var1 = ir.Var("t1", tensor_type1, span)
        var2 = ir.Var("t2", tensor_type2, span)

        # Different memory spaces should make them not equal
        assert not ir.structural_equal(var1, var2)

    def test_tile_with_memref_structural_hash(self):
        """Test structural hash consistency for tiles with MemRef."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        memref1 = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 512, 23)
        memref2 = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 512, 24)

        tile_type1 = ir.TileType(shape, DataType.FP16, memref1)
        tile_type2 = ir.TileType(shape, DataType.FP16, memref2)

        var1 = ir.Var("tile1", tile_type1, span)
        var2 = ir.Var("tile2", tile_type2, span)

        hash1 = ir.structural_hash(var1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(var2, enable_auto_mapping=True)
        assert hash1 == hash2


class TestMemRefPythonPrinter:
    """Tests for Python printing with MemRef."""

    def test_print_tensor_with_memref(self):
        """Test Python printing of TensorType with MemRef."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(10, DataType.INT64, span)]

        memref = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 40, 25)

        tensor_type = ir.TensorType(shape, DataType.FP32, memref)
        tensor_var = ir.Var("tensor", tensor_type, span)
        stmt = ir.AssignStmt(tensor_var, ir.ConstInt(0, DataType.INT64, span), span)

        result = ir.python_print(stmt)
        # Just verify it doesn't crash and produces output
        assert result is not None
        assert len(result) > 0

    def test_print_tile_with_memref(self):
        """Test Python printing of TileType with MemRef."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        memref = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0, DataType.INT64, span), 512, 26)

        tile_type = ir.TileType(shape, DataType.FP16, memref)
        tile_var = ir.Var("tile", tile_type, span)
        stmt = ir.AssignStmt(tile_var, ir.ConstInt(0, DataType.INT64, span), span)

        result = ir.python_print(stmt)
        assert result is not None
        assert len(result) > 0


class TestMemRefIntegration:
    """Integration tests combining MemRef with other IR features."""

    def test_memref_in_function(self):
        """Test using MemRef in a function."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(10, DataType.INT64, span)]

        # Input tensor with DDR MemRef
        memref_in = ir.MemRef(ir.MemorySpace.DDR, ir.ConstInt(0, DataType.INT64, span), 40, 27)

        input_type = ir.TensorType(shape, DataType.FP32, memref_in)
        input_var = ir.Var("input", input_type, span)

        # Output tensor with UB MemRef
        memref_out = ir.MemRef(ir.MemorySpace.UB, ir.ConstInt(0x1000, DataType.INT64, span), 40, 28)

        output_type = ir.TensorType(shape, DataType.FP32, memref_out)
        output_var = ir.Var("output", output_type, span)

        # Create function body
        body = ir.AssignStmt(output_var, input_var, span)

        # Create function
        func = ir.Function("copy_ddr_to_ub", [input_var], [output_type], body, span)

        assert func is not None
        assert len(func.params) == 1
        assert isinstance(func.params[0].type, ir.ShapedType)
        assert func.params[0].type.memref is not None
        assert func.params[0].type.memref.memory_space_ == ir.MemorySpace.DDR

    def test_memref_with_ops(self):
        """Test MemRef with operator calls."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        # Create tile with L0A MemRef
        memref_a = ir.MemRef(ir.MemorySpace.L0A, ir.ConstInt(0, DataType.INT64, span), 512, 29)

        tile_type_a = ir.TileType(shape, DataType.FP16, memref_a)
        tile_a = ir.Var("tile_a", tile_type_a, span)

        # Create tile with L0B MemRef
        memref_b = ir.MemRef(ir.MemorySpace.L0B, ir.ConstInt(0x200, DataType.INT64, span), 512, 30)

        tile_type_b = ir.TileType(shape, DataType.FP16, memref_b)
        tile_b = ir.Var("tile_b", tile_type_b, span)

        # Create tile with L0C MemRef for output
        memref_c = ir.MemRef(ir.MemorySpace.L0C, ir.ConstInt(0x400, DataType.INT64, span), 512, 31)

        tile_type_c = ir.TileType(shape, DataType.FP32, memref_c)

        # Create matmul op
        op = ir.Op("matmul")
        call = ir.Call(op, [tile_a, tile_b], tile_type_c, span)

        assert call is not None
        assert isinstance(call.type, ir.ShapedType)
        assert call.type.memref is not None
        assert call.type.memref.memory_space_ == ir.MemorySpace.L0C


class TestMemRefConstructor:
    """Tests for MemRef constructor syntax."""

    def test_memref_constructor(self):
        """Test creating MemRef with constructor."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0x1000, DataType.INT64, span)

        # Create MemRef with constructor
        memref = ir.MemRef(ir.MemorySpace.DDR, addr, 1024, 5)

        assert memref.memory_space_ == ir.MemorySpace.DDR
        assert memref.addr_.same_as(addr)
        assert memref.size_ == 1024

    def test_memref_constructor_different_spaces(self):
        """Test MemRef constructor with different memory spaces."""
        span = ir.Span.unknown()

        for mem_space in [
            ir.MemorySpace.DDR,
            ir.MemorySpace.UB,
            ir.MemorySpace.L1,
            ir.MemorySpace.L0A,
            ir.MemorySpace.L0B,
            ir.MemorySpace.L0C,
        ]:
            addr = ir.ConstInt(0, DataType.INT64, span)
            memref = ir.MemRef(mem_space, addr, 2048, 6)
            assert memref.memory_space_ == mem_space
            assert memref.size_ == 2048


class TestTileViewConstructor:
    """Tests for TileView constructor syntax."""

    def test_tileview_constructor(self):
        """Test creating TileView with constructor."""
        span = ir.Span.unknown()
        valid_shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        start_offset = ir.ConstInt(0, DataType.INT64, span)

        # Create TileView with constructor
        tv = ir.TileView(valid_shape, stride, start_offset)

        assert len(tv.valid_shape) == 2
        assert len(tv.stride) == 2
        assert tv.start_offset.same_as(start_offset)

    def test_tileview_constructor_with_vars(self):
        """Test TileView constructor with symbolic expressions."""
        span = ir.Span.unknown()
        n = ir.Var("n", ir.ScalarType(DataType.INT64), span)
        m = ir.Var("m", ir.ScalarType(DataType.INT64), span)

        valid_shape = [n, m]
        stride = [ir.ConstInt(1, DataType.INT64, span), n]
        start_offset = ir.ConstInt(0, DataType.INT64, span)

        tv = ir.TileView(valid_shape, stride, start_offset)

        assert len(tv.valid_shape) == 2
        assert tv.valid_shape[0].same_as(n)
        assert tv.valid_shape[1].same_as(m)


class TestPythonSyntaxPrinting:
    """Tests for Python syntax printing with MemRef and TileView."""

    def test_tensor_type_with_memref_print(self):
        """Test printing TensorType with MemRef variable name."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(64, DataType.INT64, span),
            ir.ConstInt(128, DataType.INT64, span),
        ]
        addr = ir.ConstInt(0x1000, DataType.INT64, span)
        memref = ir.MemRef(ir.MemorySpace.DDR, addr, 1024, 7)

        tensor_type = ir.TensorType(shape, DataType.FP32, memref)
        printed = ir.python_print(tensor_type)

        assert "pl.Tensor" in printed
        assert "pl.FP32" in printed
        assert "memref=" in printed
        # MemRef prints as full constructor syntax: pl.MemRef(pl.MemorySpace.DDR, addr, size, id)
        assert "pl.MemRef" in printed
        assert "pl.MemorySpace.DDR" in printed
        assert "4096" in printed  # 0x1000 in decimal
        assert "1024" in printed  # size
        assert "7" in printed  # id

    def test_tile_type_with_memref_and_tileview_print(self):
        """Test printing TileType with MemRef variable name and TileView."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        addr = ir.ConstInt(0x2000, DataType.INT64, span)
        memref = ir.MemRef(ir.MemorySpace.L0A, addr, 512, 8)

        valid_shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]
        start_offset = ir.ConstInt(0, DataType.INT64, span)
        tv = ir.TileView(valid_shape, stride, start_offset)

        tile_type = ir.TileType(shape, DataType.FP16, memref, tv)
        printed = ir.python_print(tile_type)

        assert "pl.Tile" in printed
        assert "pl.FP16" in printed
        assert "memref=" in printed
        # MemRef with name "mem_l0a_8" (includes memory space) prints as variable reference
        assert "mem_l0a_8" in printed
        assert "tile_view=" in printed
        assert "pl.TileView" in printed
        assert "valid_shape=" in printed
        assert "stride=" in printed
        assert "start_offset=" in printed

    def test_memref_print_with_symbolic_addr(self):
        """Test printing MemRef with symbolic address as variable name."""
        span = ir.Span.unknown()
        base = ir.Var("base_addr", ir.ScalarType(DataType.INT64), span)
        offset = ir.ConstInt(128, DataType.INT64, span)
        addr = ir.Add(base, offset, DataType.INT64, span)

        memref = ir.MemRef(ir.MemorySpace.UB, addr, 2048, 9)

        shape = [ir.ConstInt(32, DataType.INT64, span)]
        tensor_type = ir.TensorType(shape, DataType.INT32, memref)
        printed = ir.python_print(tensor_type)

        # MemRef prints as full constructor syntax with symbolic address
        assert "pl.MemRef" in printed
        assert "pl.MemorySpace.UB" in printed
        assert "base_addr + 128" in printed  # symbolic address expression
        assert "2048" in printed  # size
        assert "9" in printed  # id

    def test_tensor_type_with_tensorview_print(self):
        """Test printing TensorType with TensorView."""
        span = ir.Span.unknown()
        shape = [128, 256]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(128, DataType.INT64, span),
        ]

        tensor_view = ir.TensorView(stride, ir.TensorLayout.DN)
        tensor_type = ir.TensorType(shape, DataType.FP32, memref=None, tensor_view=tensor_view)

        printed = ir.python_print(tensor_type)
        assert "pl.TensorView" in printed
        assert "pl.TensorLayout.DN" in printed

    def test_tensor_type_with_memref_and_tensorview_print(self):
        """Test printing TensorType with both MemRef and TensorView."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(64, DataType.INT64, span),
            ir.ConstInt(64, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(64, DataType.INT64, span),
        ]

        addr = ir.ConstInt(0x5000, DataType.INT64, span)
        memref = ir.MemRef(ir.MemorySpace.L0A, addr, 4096, 42)
        tensor_view = ir.TensorView(stride, ir.TensorLayout.NZ)
        tensor_type = ir.TensorType(shape, DataType.FP16, memref=memref, tensor_view=tensor_view)

        printed = ir.python_print(tensor_type)

        assert "pl.Tensor" in printed
        assert "pl.FP16" in printed
        assert "memref=" in printed
        assert "pl.MemRef" in printed
        assert "tensor_view=" in printed
        assert "pl.TensorView" in printed
        assert "pl.TensorLayout.NZ" in printed


class TestIRBuilderHelpers:
    """Tests for IR Builder helper methods."""

    def test_builder_memref(self):
        """Test IRBuilder.memref() helper."""
        ib = IRBuilder()

        # Create memref with int address
        memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 1024, 33)

        assert isinstance(memref, ir.MemRef)
        assert memref.memory_space_ == ir.MemorySpace.DDR
        assert memref.size_ == 1024

    def test_builder_tile_view(self):
        """Test IRBuilder.tile_view() helper."""
        ib = IRBuilder()

        # Create tile view with integer dimensions
        tv = ib.tile_view([16, 16], [1, 16], 0)

        assert isinstance(tv, ir.TileView)
        assert len(tv.valid_shape) == 2
        assert len(tv.stride) == 2

    def test_builder_tensor_type(self):
        """Test IRBuilder.tensor_type() helper."""
        ib = IRBuilder()

        # Simple tensor type
        tensor_t = ib.tensor_type([64, 128], DataType.FP32)

        assert isinstance(tensor_t, ir.TensorType)
        assert len(tensor_t.shape) == 2
        assert tensor_t.dtype == DataType.FP32
        assert tensor_t.memref is None

    def test_builder_tensor_type_with_memref(self):
        """Test IRBuilder.tensor_type() with memref."""
        ib = IRBuilder()

        # Create memref
        memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 1024, 34)

        # Tensor type with memref
        tensor_t = ib.tensor_type([64, 128], DataType.FP32, memref=memref)

        assert isinstance(tensor_t, ir.TensorType)
        assert tensor_t.memref is not None
        assert tensor_t.memref.memory_space_ == ir.MemorySpace.DDR

    def test_builder_tile_type(self):
        """Test IRBuilder.tile_type() helper."""
        ib = IRBuilder()

        # Simple tile type
        tile_t = ib.tile_type([16, 16], DataType.FP16)

        assert isinstance(tile_t, ir.TileType)
        assert len(tile_t.shape) == 2
        assert tile_t.dtype == DataType.FP16

    def test_builder_tile_type_with_memref_and_tileview(self):
        """Test IRBuilder.tile_type() with memref and tile_view."""
        ib = IRBuilder()

        # Create memref and tile view
        memref = ib.memref(ir.MemorySpace.L0A, 0, 512, 35)
        tv = ib.tile_view([16, 16], [1, 16], 0)

        # Tile type with memref and tile_view
        tile_t = ib.tile_type([16, 16], DataType.FP16, memref=memref, tile_view=tv)

        assert isinstance(tile_t, ir.TileType)
        assert tile_t.memref is not None
        assert tile_t.tile_view is not None
        assert tile_t.memref.memory_space_ == ir.MemorySpace.L0A

    def test_builder_round_trip(self):
        """Test round-trip: create with builder, print to Python syntax."""
        ib = IRBuilder()

        # Create complex tile type with builder
        memref = ib.memref(ir.MemorySpace.L0B, 0x200, 1024, 36)
        tv = ib.tile_view([32, 32], [1, 32], 0)
        tile_t = ib.tile_type([32, 32], DataType.FP32, memref=memref, tile_view=tv)

        # Print to Python syntax
        printed = ir.python_print(tile_t)

        # Verify output contains all expected elements
        assert "pl.Tile" in printed
        assert "pl.Tile[[32, 32], pl.FP32," in printed
        assert "pl.FP32" in printed
        # MemRef with name "mem_l0b_36" (includes memory space) prints as variable reference
        assert "memref=mem_l0b_36" in printed
        assert "tile_view=pl.TileView" in printed


class TestTensorLayout:
    """Tests for TensorLayout enum."""

    def test_layout_values(self):
        """Test all TensorLayout enum values."""
        assert ir.TensorLayout.ND is not None
        assert ir.TensorLayout.DN is not None
        assert ir.TensorLayout.NZ is not None

    def test_layout_equality(self):
        """Test TensorLayout enum equality."""
        assert ir.TensorLayout.ND == ir.TensorLayout.ND
        assert ir.TensorLayout.DN == ir.TensorLayout.DN
        assert ir.TensorLayout.NZ == ir.TensorLayout.NZ
        assert ir.TensorLayout.ND != ir.TensorLayout.DN
        assert ir.TensorLayout.DN != ir.TensorLayout.NZ

    def test_layout_in_dict(self):
        """Test using TensorLayout as dictionary keys."""
        layout_map = {
            ir.TensorLayout.ND: "ND layout",
            ir.TensorLayout.DN: "DN layout",
            ir.TensorLayout.NZ: "NZ layout",
        }
        assert layout_map[ir.TensorLayout.ND] == "ND layout"
        assert layout_map[ir.TensorLayout.DN] == "DN layout"
        assert layout_map[ir.TensorLayout.NZ] == "NZ layout"


class TestTensorView:
    """Tests for TensorView struct."""

    def test_tensorview_creation_empty(self):
        """Test creating an empty TensorView."""
        tensor_view = ir.TensorView()
        assert tensor_view is not None
        assert tensor_view.layout == ir.TensorLayout.ND  # Default layout is ND

    def test_tensorview_set_attributes(self):
        """Test setting TensorView attributes."""
        span = ir.Span.unknown()
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        tensor_view = ir.TensorView()
        tensor_view.stride = stride
        tensor_view.layout = ir.TensorLayout.DN

        assert len(tensor_view.stride) == 2
        assert tensor_view.layout == ir.TensorLayout.DN

    def test_tensorview_with_layout(self):
        """Test TensorView with different layouts."""
        span = ir.Span.unknown()
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(32, DataType.INT64, span),
        ]

        for layout in [ir.TensorLayout.ND, ir.TensorLayout.DN, ir.TensorLayout.NZ]:
            tensor_view = ir.TensorView(stride, layout)
            assert tensor_view.layout == layout
            assert len(tensor_view.stride) == 2

    def test_tensorview_symbolic_stride(self):
        """Test TensorView with symbolic stride."""
        span = ir.Span.unknown()
        M = ir.Var("M", ir.ScalarType(DataType.INT64), span)

        tensor_view = ir.TensorView()
        tensor_view.stride = [ir.ConstInt(1, DataType.INT64, span), M]
        tensor_view.layout = ir.TensorLayout.NZ

        assert isinstance(tensor_view.stride[0], ir.ConstInt)
        assert isinstance(tensor_view.stride[1], ir.Var)
        assert tensor_view.layout == ir.TensorLayout.NZ


class TestTensorTypeWithTensorView:
    """Tests for TensorType with TensorView."""

    def test_tensor_type_without_tensorview(self):
        """Test TensorType has tensor_view attribute."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(32, DataType.INT64, span),
        ]

        # Create TensorType - tensor_view defaults to None
        tensor_type = ir.TensorType(shape, DataType.FP16)

        assert tensor_type.dtype == DataType.FP16
        assert len(tensor_type.shape) == 2
        # Verify tensor_view attribute exists and defaults to None
        assert hasattr(tensor_type, "tensor_view")
        assert tensor_type.tensor_view is None

    def test_tensor_type_with_tensorview(self):
        """Test TensorType creation with TensorView."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(32, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        tensor_view = ir.TensorView(stride, ir.TensorLayout.ND)
        tensor_type = ir.TensorType(shape, DataType.FP16, memref=None, tensor_view=tensor_view)

        assert tensor_type.dtype == DataType.FP16
        assert len(tensor_type.shape) == 2
        assert tensor_type.tensor_view is not None
        assert tensor_type.tensor_view.layout == ir.TensorLayout.ND
        assert len(tensor_type.tensor_view.stride) == 2

    def test_tensorview_attribute_exists(self):
        """Test that TensorView and TensorLayout can be created."""
        span = ir.Span.unknown()
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span),
        ]

        # Create a TensorView
        tensor_view = ir.TensorView(stride, ir.TensorLayout.ND)
        assert tensor_view is not None
        assert tensor_view.layout == ir.TensorLayout.ND
        assert len(tensor_view.stride) == 2

    def test_tensor_type_tensorview_different_layouts(self):
        """Test TensorType with TensorView in different layouts."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(64, DataType.INT64, span),
            ir.ConstInt(64, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(64, DataType.INT64, span),
        ]

        for layout in [ir.TensorLayout.ND, ir.TensorLayout.DN, ir.TensorLayout.NZ]:
            tensor_view = ir.TensorView(stride, layout)
            tensor_type = ir.TensorType(shape, DataType.FP32, memref=None, tensor_view=tensor_view)

            assert tensor_type.tensor_view is not None
            assert tensor_type.tensor_view.layout == layout

    def test_tensor_var_with_tensorview(self):
        """Test Var with TensorType containing TensorView."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(128, DataType.INT64, span)]
        stride = [ir.ConstInt(1, DataType.INT64, span)]

        tensor_view = ir.TensorView(stride, ir.TensorLayout.DN)
        tensor_type = ir.TensorType(shape, DataType.FP32, memref=None, tensor_view=tensor_view)

        # Create a Var with this TensorType
        var = ir.Var("x", tensor_type, span)

        assert var.name == "x"
        assert isinstance(var.type, ir.TensorType)
        assert var.type.dtype == DataType.FP32
        assert var.type.tensor_view is not None
        assert var.type.tensor_view.layout == ir.TensorLayout.DN

    def test_tensor_type_with_memref_and_tensorview(self):
        """Test TensorType with both MemRef and TensorView."""
        span = ir.Span.unknown()
        shape = [
            ir.ConstInt(32, DataType.INT64, span),
            ir.ConstInt(32, DataType.INT64, span),
        ]
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(32, DataType.INT64, span),
        ]

        # Create MemRef
        memref = ir.MemRef(
            ir.MemorySpace.UB,
            ir.ConstInt(0x4000, DataType.INT64, span),
            32 * 32 * 2,  # 32x32 FP16 elements
            42,  # ID
        )

        # Create TensorView
        tensor_view = ir.TensorView(stride, ir.TensorLayout.NZ)

        # Create TensorType with both MemRef and TensorView
        tensor_type = ir.TensorType(shape, DataType.FP16, memref=memref, tensor_view=tensor_view)

        assert tensor_type.memref is not None
        assert tensor_type.memref.memory_space_ == ir.MemorySpace.UB
        assert tensor_type.tensor_view is not None
        assert tensor_type.tensor_view.layout == ir.TensorLayout.NZ

    def test_tensor_type_constant_shape_with_tensorview(self):
        """Test TensorType with constant shape and TensorView."""
        span = ir.Span.unknown()
        stride = [
            ir.ConstInt(1, DataType.INT64, span),
            ir.ConstInt(256, DataType.INT64, span),
        ]

        tensor_view = ir.TensorView(stride, ir.TensorLayout.ND)
        tensor_type = ir.TensorType([128, 256], DataType.FP32, memref=None, tensor_view=tensor_view)

        assert len(tensor_type.shape) == 2
        assert tensor_type.tensor_view is not None
        assert tensor_type.tensor_view.layout == ir.TensorLayout.ND


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
