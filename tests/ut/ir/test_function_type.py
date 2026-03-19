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
    """Test FunctionType enum values exist and are distinct."""
    all_types = [
        ir.FunctionType.Opaque,
        ir.FunctionType.Orchestration,
        ir.FunctionType.InCore,
        ir.FunctionType.AIC,
        ir.FunctionType.AIV,
        ir.FunctionType.Group,
    ]
    assert len(all_types) == len(set(all_types))


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

    # Create function with AIC type
    func_aic = ir.Function("test_aic", params, return_types, body, span, ir.FunctionType.AIC)
    assert func_aic.func_type == ir.FunctionType.AIC

    # Create function with AIV type
    func_aiv = ir.Function("test_aiv", params, return_types, body, span, ir.FunctionType.AIV)
    assert func_aiv.func_type == ir.FunctionType.AIV

    # Create function with Group type
    func_group = ir.Function("test_group", params, return_types, body, span, ir.FunctionType.Group)
    assert func_group.func_type == ir.FunctionType.Group


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
    printed = func_opaque.as_python("pl")
    assert "@pl.function\n" in printed
    assert "type=" not in printed  # Opaque should not print type parameter

    # Orchestration function should print type parameter
    with ib.function("orchestrator", span=span, type=ir.FunctionType.Orchestration) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_orch = f.get_result()
    printed_orch = func_orch.as_python("pl")
    assert "@pl.function(type=pl.FunctionType.Orchestration)" in printed_orch

    # InCore function should print type parameter
    with ib.function("kernel", span=span, type=ir.FunctionType.InCore) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_incore = f.get_result()
    printed_incore = func_incore.as_python("pl")
    assert "@pl.function(type=pl.FunctionType.InCore)" in printed_incore

    # AIC function should print type parameter
    with ib.function("aic_kernel", span=span, type=ir.FunctionType.AIC) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_aic = f.get_result()
    printed_aic = func_aic.as_python("pl")
    assert "@pl.function(type=pl.FunctionType.AIC)" in printed_aic

    # AIV function should print type parameter
    with ib.function("aiv_kernel", span=span, type=ir.FunctionType.AIV) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_aiv = f.get_result()
    printed_aiv = func_aiv.as_python("pl")
    assert "@pl.function(type=pl.FunctionType.AIV)" in printed_aiv

    # Group function should print type parameter
    with ib.function("group_func", span=span, type=ir.FunctionType.Group) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_group = f.get_result()
    printed_group = func_group.as_python("pl")
    assert "@pl.function(type=pl.FunctionType.Group)" in printed_group


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

    # Test AIC
    @pl.function(type=pl.FunctionType.AIC)
    def aic_kernel(x: pl.Tensor[[4], pl.INT64]) -> pl.Tensor[[4], pl.INT64]:
        return x

    assert aic_kernel.name == "aic_kernel"
    assert aic_kernel.func_type == ir.FunctionType.AIC

    # Test AIV
    @pl.function(type=pl.FunctionType.AIV)
    def aiv_kernel(x: pl.Tensor[[4], pl.INT64]) -> pl.Tensor[[4], pl.INT64]:
        return x

    assert aiv_kernel.name == "aiv_kernel"
    assert aiv_kernel.func_type == ir.FunctionType.AIV

    # Test Group
    @pl.function(type=pl.FunctionType.Group)
    def group_func(x: pl.Tensor[[4], pl.INT64]) -> pl.Tensor[[4], pl.INT64]:
        return x

    assert group_func.name == "group_func"
    assert group_func.func_type == ir.FunctionType.Group


def test_function_type_serialization():
    """Test that function type is correctly serialized and deserialized for all non-Opaque types."""
    ib = IRBuilder()
    span = ir.Span.unknown()
    dtype = DataType.INT64

    non_opaque_types = [
        ir.FunctionType.Orchestration,
        ir.FunctionType.InCore,
        ir.FunctionType.AIC,
        ir.FunctionType.AIV,
        ir.FunctionType.Group,
    ]

    for func_type in non_opaque_types:
        with ib.function("test_func", span=span, type=func_type) as f:
            x = f.param("x", ir.ScalarType(dtype), span=span)
            f.return_type(ir.ScalarType(dtype))
            ib.return_stmt(x, span=span)

        original = f.get_result()
        serialized = ir.serialize(original)
        deserialized = ir.deserialize(serialized)
        assert isinstance(deserialized, ir.Function)
        assert deserialized.name == "test_func"
        assert deserialized.func_type == func_type
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

    # AIC != AIV
    with ib.function("func1", span=span, type=ir.FunctionType.AIC) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_aic = f.get_result()

    with ib.function("func1", span=span, type=ir.FunctionType.AIV) as f:
        x = f.param("x", ir.ScalarType(dtype), span=span)
        f.return_type(ir.ScalarType(dtype))
        ib.return_stmt(x, span=span)

    func_aiv = f.get_result()

    assert not ir.structural_equal(func_aic, func_aiv)
    assert func_aic.func_type != func_aiv.func_type


def test_function_type_language_export():
    """Test that FunctionType is exported from language module."""
    assert hasattr(pl, "FunctionType")
    # Verify all enum values are accessible through the language module
    assert pl.FunctionType.Opaque is ir.FunctionType.Opaque
    assert pl.FunctionType.Orchestration is ir.FunctionType.Orchestration
    assert pl.FunctionType.InCore is ir.FunctionType.InCore
    assert pl.FunctionType.AIC is ir.FunctionType.AIC
    assert pl.FunctionType.AIV is ir.FunctionType.AIV
    assert pl.FunctionType.Group is ir.FunctionType.Group


def test_is_incore_type():
    """Test IsInCoreType helper returns correct results."""
    assert ir.is_incore_type(ir.FunctionType.InCore)
    assert ir.is_incore_type(ir.FunctionType.AIC)
    assert ir.is_incore_type(ir.FunctionType.AIV)
    assert not ir.is_incore_type(ir.FunctionType.Opaque)
    assert not ir.is_incore_type(ir.FunctionType.Orchestration)
    assert not ir.is_incore_type(ir.FunctionType.Group)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
