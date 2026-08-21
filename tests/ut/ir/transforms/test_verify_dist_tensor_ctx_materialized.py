# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the DistTensorCtxMaterialized property verifier.

``MaterializeDistTensorCtx`` replaces every ``pld.system.get_comm_ctx`` outside
host orchestration with the explicit ``CommCtxType`` parameter, because device
and chip-orchestration codegen have no runtime representation for the query.
The pass enforces that for every function it rewrites; this verifier checks the
invariant independently, which also covers the Programs the pass returns
untouched (no function declares a DistributedTensor parameter).
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _props() -> passes.IRPropertySet:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.DistTensorCtxMaterialized)
    return props


def _errors(program: ir.Program) -> list:
    diagnostics = passes.PropertyVerifierRegistry.verify(_props(), program)
    return [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]


def _span() -> ir.Span:
    return ir.Span("test_verify_dist_tensor_ctx_materialized.py", 1, 1)


def _get_ctx_function(name: str, **func_kwargs) -> ir.Function:
    """A function whose body queries the context of its DistributedTensor param."""
    span = _span()
    data = ir.Var("data", ir.DistributedTensorType([4], pl.FP32), span)
    ctx = ir.Var("data_ctx", ir.CommCtxType.get(), span)
    return ir.Function(
        name,
        [(data, ir.ParamDirection.In)],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(ctx, ir.create_op_call("pld.system.get_comm_ctx", [data], {}, span), span),
                ir.ReturnStmt(span),
            ],
            span,
        ),
        span,
        **func_kwargs,
    )


def test_device_get_comm_ctx_is_flagged():
    kernel = _get_ctx_function("kernel", type=ir.FunctionType.InCore)
    errors = _errors(ir.Program([kernel], "device_get_ctx", _span()))

    assert len(errors) == 1
    assert "kernel" in errors[0].message
    assert "pld.system.get_comm_ctx" in errors[0].message


def test_chip_orchestration_get_comm_ctx_is_flagged():
    chip_orch = _get_ctx_function("chip_orch", type=ir.FunctionType.Orchestration)
    errors = _errors(ir.Program([chip_orch], "chip_orch_get_ctx", _span()))

    assert len(errors) == 1
    assert "chip_orch" in errors[0].message


def test_host_orchestration_get_comm_ctx_is_allowed():
    """Host codegen resolves the query from the window's per-rank context."""
    # `func_type=Orchestration` implies level=CHIP, so host orchestration is
    # spelled with the HOST level plus the Orchestrator role.
    host = _get_ctx_function("host_orch", level=ir.Level.HOST, role=ir.Role.Orchestrator)
    assert _errors(ir.Program([host], "host_get_ctx", _span())) == []


def test_program_without_get_comm_ctx_passes():
    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

    assert _errors(P) == []


def test_property_is_verified_on_the_default_path():
    """The pipeline only verifies a produced property that is in this set.

    `PassManager.run_passes` intersects each pass's produced properties with
    `get_verified_properties()`, so a property missing from it is checked only
    when a test installs a `VerificationInstrument` by hand — which is exactly
    the case this verifier exists to cover, since `MaterializeDistTensorCtx`
    returns early (leaving any query in place) when no function in the Program
    declares a DistributedTensor parameter.
    """
    assert passes.get_verified_properties().contains(passes.IRProperty.DistTensorCtxMaterialized)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
