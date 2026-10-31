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
from pypto import DataType, backend, ir
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
            for i, (acc,) in pl.hardware_loop(0, 8, 2, init_values=(x,)):
                _result = pl.yield_(acc + i)
        return x

    loop = _loop(required)
    assert loop.hardware_loop_policy == ir.HardwareLoopPolicy.Require
    assert loop.step.value == 2
    text = ir.python_print(required)
    assert "pl.hardware_loop(" in text
    restored = pl.parse(text)
    assert _loop(restored).hardware_loop_policy == ir.HardwareLoopPolicy.Require


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
    p.profile_id = "reg-test-v1"
    p.toolchain_id = "test-v1"
    assert ir.analyze_hardware_loop(required, p).state == ir.HardwareLoopState.Unknown
    p.allowed_ops = ["reg.load", "reg.add"]
    assert ir.analyze_hardware_loop(required, p).state == ir.HardwareLoopState.Eligible
    with pytest.raises(ValueError, match="evidence"):
        ir.verify_hardware_loop_requirement(required, p)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
