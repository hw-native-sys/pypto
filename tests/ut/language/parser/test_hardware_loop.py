# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Hardware-loop intent must survive frontend and printer roundtrips."""

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir, passes
from pypto.language.parser import ParserError


def _loop(function):
    class Find(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.loops = []

        def visit_for_stmt(self, op):
            self.loops.append(op)
            super().visit_for_stmt(op)

    finder = Find()
    finder.visit_stmt(function.body)
    return finder.loops[0]


@pytest.mark.parametrize("ascend_backend", [backend.BackendType.Ascend950], indirect=True)
def test_hardware_loop_frontend(ascend_backend):
    @pl.function(type=pl.FunctionType.AIV)
    def required(x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
        with pl.reg_scope():
            for i, (acc,) in pl.hardware_loop(8, init_values=(x,)):
                _result = pl.yield_(acc + i)
        return x

    loop = _loop(required)
    assert loop.hardware_loop_policy == ir.HardwareLoopPolicy.Require
    assert loop.start.value == 0
    assert loop.step.value == 1
    text = ir.python_print(required)
    assert "pl.hardware_loop(" in text
    restored = pl.parse(text)
    assert _loop(restored).hardware_loop_policy == ir.HardwareLoopPolicy.Require


@pytest.mark.parametrize("arguments", ["", "0, 8", "0, 8, 1", "0, 8, 1, 2", "8, attrs={}", "8, step=1"])
@pytest.mark.parametrize("ascend_backend", [backend.BackendType.Ascend950], indirect=True)
def test_hardware_loop_accepts_only_stop_and_init_values(arguments, ascend_backend):
    source = f"""
@pl.function(type=pl.FunctionType.AIV)
def invalid():
    with pl.reg_scope():
        for i in pl.hardware_loop({arguments}):
            pass
"""
    with pytest.raises(ParserError, match="hardware_loop"):
        pl.parse(source)


@pytest.mark.parametrize("start,step", [(1, 1), (0, 2), (0, -1), ("dynamic", 1), (0, "dynamic")])
def test_ir_contract_rejects_noncanonical_required_loop_bounds(start, step):
    span = ir.Span.unknown()
    typ = ir.ScalarType(DataType.INDEX)

    def bound(value):
        return (
            ir.Var(value, typ, span) if isinstance(value, str) else ir.ConstInt(value, DataType.INDEX, span)
        )

    loop = ir.ForStmt(
        ir.Var("i", typ, span),
        bound(start),
        bound(0),
        bound(step),
        [],
        ir.SeqStmts([], span),
        [],
        span,
        hardware_loop_policy=ir.HardwareLoopPolicy.Require,
    )
    function = ir.Function("invalid", [], [], ir.RegScopeStmt("", loop, span), span, type=ir.FunctionType.AIV)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.HardwareLoopContractValid)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, ir.Program([function], "Invalid", span))
    assert any("start=0 and step=1" in diagnostic.message for diagnostic in diagnostics)


@pytest.mark.parametrize("ascend_backend", [backend.BackendType.Ascend950], indirect=True)
def test_hardware_loop_keyword_stop(ascend_backend):
    @pl.function(type=pl.FunctionType.AIV)
    def valid():
        with pl.reg_scope():
            for i in pl.hardware_loop(stop=8):
                pass

    loop = _loop(valid)
    assert loop.start.value == 0 and loop.stop.value == 8 and loop.step.value == 1


@pytest.mark.parametrize("ascend_backend", [backend.BackendType.Ascend950], indirect=True)
def test_dynamic_hardware_loop_stop_roundtrip(ascend_backend):
    @pl.function(type=pl.FunctionType.AIV)
    def dynamic(n: pl.Scalar[pl.INDEX]):
        with pl.reg_scope():
            for i in pl.hardware_loop(n):
                pass

    required = _loop(dynamic)
    assert isinstance(required.stop, ir.Var)
    p = ir.HardwareLoopProfile()
    p.supported = True
    assert ir.analyze_hardware_loop(required, p).state == ir.HardwareLoopState.Eligible
    restored = pl.parse(ir.python_print(dynamic))
    assert isinstance(_loop(restored).stop, ir.Var)


@pytest.mark.parametrize("stop", ["1.5", "n"])
@pytest.mark.parametrize("ascend_backend", [backend.BackendType.Ascend950], indirect=True)
def test_float_hardware_loop_stop_is_rejected(stop, ascend_backend):
    source = f"""
@pl.function(type=pl.FunctionType.AIV)
def invalid(n: pl.Scalar[pl.FP32]):
    with pl.reg_scope():
        for i in pl.hardware_loop({stop}):
            pass
"""
    with pytest.raises(ParserError, match="integer scalar stop"):
        pl.parse(source)


@pytest.mark.parametrize("start,step", [(1, 1), (0, 2)])
@pytest.mark.parametrize("ascend_backend", [backend.BackendType.Ascend950], indirect=True)
def test_builder_rejects_noncanonical_required_loop_bounds(start, step, ascend_backend):
    builder = ir.IRBuilder()
    with pytest.raises(ValueError, match="start=0 and step=1"):
        with builder.function("invalid", type=ir.FunctionType.AIV):
            with builder.reg_scope():
                i = builder.var("i", ir.ScalarType(DataType.INDEX))
                with builder.for_loop(i, start, 0, step, hardware_loop_policy=ir.HardwareLoopPolicy.Require):
                    pass


@pytest.mark.parametrize("case", ["typed_bounds", "attrs"])
def test_stop_only_contract_preserves_printer_roundtrip(case):
    span = ir.Span.unknown()
    typ = ir.ScalarType(DataType.INDEX)
    bounds_dtype = DataType.INT64 if case == "typed_bounds" else DataType.INDEX
    loop = ir.ForStmt(
        ir.Var("i", typ, span),
        ir.ConstInt(0, bounds_dtype, span),
        ir.ConstInt(8, DataType.INDEX, span),
        ir.ConstInt(1, bounds_dtype, span),
        [],
        ir.SeqStmts([], span),
        [],
        span,
        attrs={"hint": 1} if case == "attrs" else None,
        hardware_loop_policy=ir.HardwareLoopPolicy.Require,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.HardwareLoopContractValid)
    function = ir.Function("invalid", [], [], loop, span)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, ir.Program([function], "Invalid", span))
    assert diagnostics
    with pytest.raises(ValueError, match="HardwareLoopPolicy.Require"):
        ir.python_print(loop)


def test_hardware_loop_outside_reg_scope_is_rejected():
    with pytest.raises(ParserError, match="hardware_loop.*reg_scope"):

        @pl.function(type=pl.FunctionType.AIV)
        def invalid():
            for i in pl.hardware_loop(0):
                pass


def test_builder_hardware_loop_outside_reg_scope_is_rejected():
    builder = ir.IRBuilder()
    with pytest.raises(ValueError, match="HardwareLoopPolicy.Require.*RegScope"):
        with builder.function("invalid", type=ir.FunctionType.AIV):
            index = builder.var("i", ir.ScalarType(DataType.INDEX))
            with builder.for_loop(index, 0, 0, 1, hardware_loop_policy=ir.HardwareLoopPolicy.Require):
                pass


@pytest.mark.parametrize("kind", [ir.ForKind.Parallel, ir.ForKind.Unroll, ir.ForKind.Pipeline])
def test_require_rejects_nonsequential(kind):
    span = ir.Span.unknown()
    scalar = ir.ScalarType(DataType.INDEX)
    with pytest.raises(ValueError, match="Sequential"):
        ir.ForStmt(
            ir.Var("i", scalar, span),
            ir.ConstInt(0, DataType.INDEX, span),
            ir.ConstInt(4, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [],
            ir.SeqStmts([], span),
            [],
            span,
            kind=kind,
            hardware_loop_policy=ir.HardwareLoopPolicy.Require,
        )


@pytest.mark.parametrize("ascend_backend", [backend.BackendType.Ascend950], indirect=True)
def test_required_loop_inside_register_scope(ascend_backend):
    @pl.function(type=pl.FunctionType.AIV)
    def register_loop(src: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec]):
        with pl.reg_scope():
            for i in pl.hardware_loop(8):
                x = pl.reg.load(src, lanes=64)
                _y = pl.reg.add(x, x)

    required = _loop(register_loop)
    assert required.hardware_loop_policy == ir.HardwareLoopPolicy.Require
    p = ir.HardwareLoopProfile()
    p.supported = True
    assert ir.analyze_hardware_loop(required, p).state == ir.HardwareLoopState.Unknown
    p.allowed_ops = ["reg.load", "reg.add"]
    assert ir.analyze_hardware_loop(required, p).state == ir.HardwareLoopState.Eligible
    with pytest.raises(ValueError, match="evidence"):
        ir.verify_hardware_loop_requirement(required, p)


@pytest.mark.parametrize("dtype", ["INT8", "INT32", "INT64", "UINT32"])
@pytest.mark.parametrize("ascend_backend", [backend.BackendType.Ascend950], indirect=True)
def test_typed_hardware_loop_stop_preserves_index_induction_roundtrip(dtype, ascend_backend):
    # The stop keeps its explicit scalar dtype, but scf.for always binds an
    # INDEX induction variable. Printing must not change that distinction.
    function = pl.parse(f"""
@pl.function(type=pl.FunctionType.AIV)
def typed_stop():
    with pl.reg_scope():
        for i in pl.hardware_loop(pl.const(2, pl.{dtype})):
            pass
""")
    for candidate in (function, pl.parse(ir.python_print(function))):
        loop = _loop(candidate)
        assert loop.loop_var.type.dtype == DataType.INDEX
        assert loop.stop.dtype == getattr(DataType, dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
