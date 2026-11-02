# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Orchestration codegen for ``pld.system.nranks(ctx)``.

Computing ``B = CalAllToAllVBlocks(P, L)`` inside a CHIP orchestration needs the
rank count there. The count is reachable without a new operand or scalar: the
comm context already arrives as a materialized CommCtx scalar param, so the read
is one load at the offset ``comm_layout`` pins for the kernel side.
"""

import pytest
from _orchestration_codegen_common import _generate_orch_code
from pypto import backend
from pypto.backend import BackendType
from pypto.pypto_core import DataType, ir

S = ir.Span.unknown()


def _nranks_orchestration():
    """A chip orchestration whose body reads the rank count off its CommCtx.

    The signature mirrors what ``MaterializeDistTensorCtx`` produces: one
    DistributedTensor param and its materialized CommCtx scalar, in that order.
    """
    signal = ir.Var("signal", ir.DistributedTensorType([64], DataType.FP32), S)
    signal_ctx = ir.Var("signal_ctx", ir.CommCtxType.get(), S)
    ctx = ir.Var("ctx", ir.CommCtxType.get(), S)
    nranks = ir.Var("nranks", ir.ScalarType(DataType.INT32), S)
    bind_ctx = ir.AssignStmt(
        ctx, ir.Call(ir.get_op("pld.system.get_comm_ctx"), [signal], {}, None, ir.CommCtxType.get(), S), S
    )
    read = ir.AssignStmt(
        nranks,
        ir.Call(ir.get_op("pld.system.nranks"), [ctx], {}, None, ir.ScalarType(DataType.INT32), S),
        S,
    )
    body = ir.SeqStmts([bind_ctx, read, ir.ReturnStmt([], S)], S)
    orch = ir.Function("chip_orch", [signal, signal_ctx], [], body, S, type=ir.FunctionType.Orchestration)
    return ir.Program([orch], "nranks_orch", S)


def test_nranks_reads_the_context_the_orchestration_already_holds():
    """The read lands on the CommCtx the orchestration already holds.

    The context arrives as a materialized scalar param and ``get_comm_ctx``
    aliases it, so the count is one load at the ``rankNum`` offset. Pinning the
    literal keeps the emitted code and ``comm_layout.h`` from drifting apart.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)

    code = _generate_orch_code(_nranks_orchestration())

    assert "uint64_t signal_ctx = orch_args.scalar<uint64_t>(0);" in code, code
    assert "uint64_t ctx = signal_ctx;" in code, code
    assert "int32_t nranks = *reinterpret_cast<const int32_t*>(ctx + 20);" in code, code


def test_nranks_needs_no_extra_operand():
    """No tensor, scalar or launch-spec change rides along with the read."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)

    code = _generate_orch_code(_nranks_orchestration())

    assert "add_scalar" not in code, code
    assert "add_tensor" not in code, code
    assert "set_block_num" not in code, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
