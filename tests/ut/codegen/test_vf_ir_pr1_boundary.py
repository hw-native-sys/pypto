# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PR1 descriptors must not silently acquire unsupported lowering semantics."""

import pytest
from pypto import DataType, ir, passes
from pypto.pypto_core import codegen

pytestmark = pytest.mark.usefixtures("ascend_backend")


def _loop(kind=ir.ForKind.Sequential):
    span = ir.Span.unknown()
    return ir.ForStmt(
        ir.Var("i", ir.ScalarType(DataType.INDEX), span),
        ir.ConstInt(0, DataType.INDEX, span),
        ir.ConstInt(8, DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
        [],
        ir.EvalStmt(ir.ConstInt(0, DataType.INDEX, span), span),
        [],
        span,
        kind=kind,
        attrs={"pipeline_stages": 2} if kind == ir.ForKind.Pipeline else None,
        hardware_loop_policy=ir.HardwareLoopPolicy.Require,
    )


def _program(body):
    span = ir.Span.unknown()
    function = ir.Function("kernel", [], [], body, span, type=ir.FunctionType.InCore)
    return ir.Program([function], "VFBoundary", span)


def test_pto_codegen_rejects_register_scope_until_lowering_exists():
    span = ir.Span.unknown()
    scope = ir.RegScopeStmt("vf", ir.SeqStmts([], span), span)
    with pytest.raises(ValueError, match="RegScopeStmt.*not supported"):
        codegen.PTOCodegen().generate(_program(scope))


@pytest.mark.parametrize("function_type", [ir.FunctionType.InCore, ir.FunctionType.AIC, ir.FunctionType.AIV])
@pytest.mark.parametrize(
    "dtype,kind", [(DataType.FP32, ir.RegKind.VecReg), (DataType.BOOL, ir.RegKind.MaskReg)]
)
def test_pto_codegen_rejects_register_typed_parameters_until_lowering_exists(function_type, dtype, kind):
    """A bare register parameter must fail before the scalar-pointer ABI fallback."""
    span = ir.Span.unknown()
    register = ir.Var("r", ir.RegType(64, dtype, kind), span)
    function = ir.Function(
        "kernel",
        [register],
        [],
        ir.SeqStmts([], span),
        span,
        type=function_type,
    )
    program = ir.Program([function], "VFBoundary", span)
    with pytest.raises(ValueError, match="RegType parameters.*not supported"):
        codegen.PTOCodegen().generate(program)


def test_pto_codegen_rejects_unimplemented_required_hardware_loop():
    with pytest.raises(ValueError, match="Require.*not supported"):
        codegen.PTOCodegen().generate(_program(_loop()))


def test_unroll_cannot_erase_required_hardware_loop():
    with passes.PassContext([]), pytest.raises(ValueError, match="Require"):
        passes.unroll_loops()(_program(_loop(ir.ForKind.Unroll)))


@pytest.mark.parametrize("transform", [passes.lower_pipeline_loops, passes.skew_cross_core_pipeline])
def test_pipeline_expansion_cannot_erase_required_hardware_loop(transform):
    with passes.PassContext([]), pytest.raises(ValueError, match="Require"):
        transform()(_program(_loop(ir.ForKind.Pipeline)))


@pytest.mark.parametrize("entrypoint", ["distributed", "orchestration"])
@pytest.mark.parametrize("register_scope", [False, True])
def test_host_codegen_rejects_vf_infrastructure(entrypoint, register_scope):
    span = ir.Span.unknown()
    body = ir.RegScopeStmt("vf", ir.SeqStmts([], span), span) if register_scope else _loop()
    program = _program(body)
    with pytest.raises(ValueError, match="RegScopeStmt|Require"):
        if entrypoint == "distributed":
            codegen.DistributedCodegen().generate(program)
        else:
            function = program.get_function("kernel")
            assert function is not None
            codegen.generate_orchestration(program, function)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
