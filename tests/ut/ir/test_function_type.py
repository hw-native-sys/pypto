# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for function type attribute feature."""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir import IRBuilder


def test_function_type_enum():
    """Test FunctionType enum values."""
    assert ir.FunctionType.Opaque
    assert ir.FunctionType.Orchestration
    assert ir.FunctionType.InCore


def test_function_constructor_with_type():
    """Test Function constructor with type parameter."""
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Create function with Opaque type (default)
    params = [ir.Var("x", ir.ScalarType(dtype), span)]
    return_types = [ir.ScalarType(dtype)]
    body = ir.SeqStmts([], span)

    func_opaque = ir.Function("test_opaque", params, return_types, body, span)
    assert func_opaque.func_type == ir.FunctionType.Opaque

    # Create function with Orchestration type
    func_orch = ir.Function("test_orch", params, return_types, body, span, ir.FunctionType.Orchestration)
    assert func_orch.func_type == ir.FunctionType.Orchestration

    # Create function with InCore type
    func_incore = ir.Function("test_incore", params, return_types, body, span, ir.FunctionType.InCore)
    assert func_incore.func_type == ir.FunctionType.InCore


def test_ir_builder_with_function_type():
    """Test IR Builder with function type parameter."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Build function with Orchestration type
    with ib.function("orchestrator", span=span, type=ir.FunctionType.Orchestration) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func = f.get_result()
    assert func.name == "orchestrator"
    assert func.func_type == ir.FunctionType.Orchestration

    # Build function with InCore type
    with ib.function("aicore_kernel", span=span, type=ir.FunctionType.InCore) as f:
        y = f.param("y", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(y, span=span)

    func2 = f.get_result()
    assert func2.name == "aicore_kernel"
    assert func2.func_type == ir.FunctionType.InCore


def test_function_type_python_print():
    """Test that function type is correctly printed in Python syntax."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Opaque function should not print type parameter
    with ib.function("default_func", span=span) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_opaque = f.get_result()
    printed = ir.python_print(func_opaque, "pl")
    assert "@pl.function\n" in printed
    assert "type=" not in printed  # Opaque should not print type parameter

    # Orchestration function should print type parameter
    with ib.function("orchestrator", span=span, type=ir.FunctionType.Orchestration) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_orch = f.get_result()
    printed_orch = ir.python_print(func_orch, "pl")
    assert "@pl.function(type=pl.FunctionType.Orchestration)" in printed_orch

    # InCore function should print type parameter
    with ib.function("kernel", span=span, type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_incore = f.get_result()
    printed_incore = ir.python_print(func_incore, "pl")
    assert "@pl.function(type=pl.FunctionType.InCore)" in printed_incore


def test_function_type_decorator_parsing():
    """Test parsing functions with type parameter in decorator."""

    # Test Opaque (default)
    @pl.function
    def default_func(x: pl.Tensor[[4], pl.INT64]) -> pl.Tensor[[4], pl.INT64]:
        return x

    assert default_func.name == "default_func"
    assert default_func.func_type == ir.FunctionType.Opaque

    # Test Orchestration
    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(x: pl.Tensor[[4], pl.INT64]) -> pl.Tensor[[4], pl.INT64]:
        return x

    assert orchestrator.name == "orchestrator"
    assert orchestrator.func_type == ir.FunctionType.Orchestration

    # Test InCore
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(x: pl.Tensor[[4], pl.INT64]) -> pl.Tensor[[4], pl.INT64]:
        return x

    assert kernel.name == "kernel"
    assert kernel.func_type == ir.FunctionType.InCore


def test_function_type_serialization():
    """Test that function type is correctly serialized and deserialized."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Create function with Orchestration type
    with ib.function("test_func", span=span, type=ir.FunctionType.Orchestration) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    original = f.get_result()

    # Serialize and deserialize
    serialized = ir.serialize(original)
    deserialized = ir.deserialize(serialized)
    assert isinstance(deserialized, ir.Function)

    # Check that type is preserved
    assert deserialized.name == "test_func"
    assert deserialized.func_type == ir.FunctionType.Orchestration

    # Check structural equality
    assert ir.structural_equal(original, deserialized)


def test_function_type_structural_comparison():
    """Test that function type is considered in structural equality."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    # Create two functions with same structure but different types
    with ib.function("func1", span=span, type=ir.FunctionType.Opaque) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_opaque = f.get_result()

    with ib.function("func1", span=span, type=ir.FunctionType.Orchestration) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_orch = f.get_result()

    assert not ir.structural_equal(func_opaque, func_orch)
    assert func_opaque.func_type != func_orch.func_type


def test_function_type_language_export():
    """Test that FunctionType is exported from language module."""

    assert hasattr(pl, "FunctionType")
    assert pl.FunctionType.Opaque
    assert pl.FunctionType.Orchestration
    assert pl.FunctionType.InCore


if __name__ == "__main__":
    pytest.main([__file__])
