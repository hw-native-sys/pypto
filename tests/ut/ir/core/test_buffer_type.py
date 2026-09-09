# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Physical buffer descriptors are independent of logical tile storage metadata."""

import ast
from typing import Any

import pypto
import pytest
from pypto import DataType, ir


def test_buffer_descriptor_has_no_storage_identity():
    descriptor = ir.BufferType([32, 64], DataType.FP32, ir.Mem.Vec)
    assert isinstance(descriptor, ir.Type)
    assert not isinstance(descriptor, ir.ShapedType)
    assert descriptor.shape == [32, 64]
    assert descriptor.valid_shape == [32, 64]
    assert descriptor.dtype == DataType.FP32
    assert descriptor.memory_space == ir.Mem.Vec
    for field in ("memref", "base", "addr", "tile_view", "start_offset"):
        assert not hasattr(descriptor, field)
    with pytest.raises(AttributeError):
        setattr(descriptor, "fractal", 1024)


@pytest.mark.parametrize("valid_shape", [[-1, 64], [0, 64], [16, 32]])
def test_valid_descriptor_has_no_runtime_ssa(valid_shape):
    descriptor = ir.BufferType([32, 64], DataType.FP16, ir.Mem.Vec, valid_shape)
    assert descriptor.valid_shape == valid_shape
    extent = ir.Var("extent", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    invalid_dims: Any = [extent, 64]
    with pytest.raises(TypeError):
        ir.BufferType([32, 64], DataType.FP16, ir.Mem.Vec, invalid_dims)


@pytest.mark.parametrize("shape", [[], [0, 64], [-1, 64], [32, -2]])
def test_invalid_physical_shape(shape):
    with pytest.raises(ValueError, match="physical (shape|extent)"):
        ir.BufferType(shape, DataType.FP32, ir.Mem.Vec)


@pytest.mark.parametrize("valid_shape", [[32], [33, 64], [32, 65], [-2, 64]])
def test_invalid_valid_descriptor(valid_shape):
    with pytest.raises(ValueError, match="valid"):
        ir.BufferType([32, 64], DataType.FP32, ir.Mem.Vec, valid_shape)


@pytest.mark.parametrize("space", [ir.Mem.DDR, ir.Mem.ScalarLocal])
def test_non_tile_memory_rejected(space):
    with pytest.raises(ValueError, match="on-chip tile memory"):
        ir.BufferType([32, 64], DataType.FP32, space)


def test_runtime_handle_dtype_rejected():
    with pytest.raises(ValueError, match="element data type"):
        ir.BufferType([32, 64], DataType.TASK_ID, ir.Mem.Vec)


def test_invalid_layout_and_fractal():
    with pytest.raises(ValueError, match="block layout"):
        ir.BufferType([32, 64], DataType.FP32, ir.Mem.Vec, blayout=ir.TileLayout.none_box)
    with pytest.raises(ValueError, match="fractal"):
        ir.BufferType([32, 64], DataType.FP32, ir.Mem.Vec, fractal=0)


@pytest.mark.parametrize("slot_count", [0, -1])
def test_invalid_multi_buffer_count(slot_count):
    descriptor = ir.BufferType([32, 64], DataType.FP32, ir.Mem.Vec)
    with pytest.raises(ValueError, match="slot_count"):
        ir.MultiBufferType(descriptor, slot_count)


@pytest.mark.parametrize("multi", [False, True])
def test_printer_preserves_physical_descriptor(multi):
    descriptor = ir.BufferType(
        [32, 64],
        DataType.FP16,
        ir.Mem.Mat,
        valid_shape=[-1, 48],
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=1024,
        pad=ir.PadValue.zero,
        compact=ir.CompactMode.normal,
    )
    original = ir.MultiBufferType(descriptor, 3) if multi else descriptor
    text = ir.python_print(original, format=False)
    assert "UnknownType" not in text
    assert "MemRef" not in text
    # Evaluate only the constructor expression emitted from this fixed test input.
    namespace = {"pypto": pypto}
    code = ast.parse(f"restored = {text}")
    exec(compile(code, "<buffer descriptor>", "exec"), namespace)
    assert namespace["restored"] == original


def test_program_dump_imports_native_buffer_types():
    span = ir.Span.unknown()
    descriptor = ir.BufferType([32, 64], DataType.FP32, ir.Mem.Vec)
    source = ir.Var("source", descriptor, span)
    function = ir.Function("kernel", [source], [], ir.ReturnStmt([], span), span)
    text = ir.Program([function], "BufferProgram", span).as_python(format=False)
    assert "import pypto\n" in text
    assert "pypto.ir.BufferType(" in text
    ast.parse(text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
