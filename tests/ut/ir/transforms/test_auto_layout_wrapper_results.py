# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AutoLayout result types across Graph and SPMD calls."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import passes

_KERNEL = """
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def mm(self, x: pl.Tensor[[64, 64], pl.FP32], w: pl.Tensor[[64, 64], pl.FP32],
           dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        tx = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        tw = pl.load(w, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        dst = pl.store(pl.matmul(tx, tw), [0, 0], dst)
        return dst
"""

_EXPLICIT_WRAPPERS = """
    @pl.function(type=pl.FunctionType.Spmd)
    def z_spmd(self, x: pl.Tensor[[64, 64], pl.FP32], w: pl.Tensor[[64, 64], pl.FP32],
               dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        pl.func_attr({"core_num": 1})
        result = self.mm(x, w, dst)
        return result

    @pl.function(type=pl.FunctionType.Graph)
    def layer(self, x: pl.Tensor[[64, 64], pl.FP32], w: pl.Tensor[[64, 64], pl.FP32],
              dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        result = self.z_spmd(x, w, dst)
        return result
"""

_ENTRY = """
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64, 64], pl.FP32], w: pl.Tensor[[64, 64], pl.FP32],
             dst: pl.Out[pl.Tensor[[64, 64], pl.FP32]]) -> pl.Tensor[[64, 64], pl.FP32]:
        mid = pl.tensor.create([64, 64], dtype=pl.FP32)
{produce}
        for i in pl.range(64):
            dst = self.mm(mid, w, dst)
        return dst
"""


def _assert_blocked_nz(tensor_type: ir.Type) -> None:
    assert isinstance(tensor_type, ir.TensorType)
    assert tensor_type.tensor_view is not None
    assert tensor_type.tensor_view.layout == ir.TensorLayout.NZ
    assert tensor_type.tensor_view.nz_blocked
    assert len(tensor_type.shape) == 5
    assert [dim.value for dim in tensor_type.shape if isinstance(dim, ir.ConstInt)] == [1, 8, 4, 16, 8]


def _assert_call_result(
    program: ir.Program, caller: str, callee: str, *, returns_tensor: bool = True
) -> None:
    function = program.get_function(caller)
    assert function is not None
    body = function.body
    for stmt in body.stmts if isinstance(body, ir.SeqStmts) else [body]:
        if isinstance(stmt, ir.AssignStmt):
            call = stmt.value
        elif isinstance(stmt, ir.EvalStmt):
            call = stmt.expr
        else:
            continue
        if isinstance(call, ir.Call) and call.op.name == callee:
            break
    else:
        pytest.fail(f"missing {caller} → {callee} call")

    callee_function = program.get_function(callee)
    assert callee_function is not None
    if returns_tensor:
        assert isinstance(stmt, ir.AssignStmt)
        assert len(callee_function.return_types) == 1
        for tensor_type in (callee_function.return_types[0], call.type, stmt.var.type):
            _assert_blocked_nz(tensor_type)
    else:
        assert not callee_function.return_types
        assert isinstance(stmt, ir.EvalStmt)
        assert isinstance(call.type, ir.UnknownType)


@pytest.mark.parametrize("outlined", [True, False], ids=["scope", "explicit"])
def test_graph_spmd_call_returns_blocked_private_storage(
    tmp_path, ascend_backend, pass_verification_instruments, outlined
):
    wrappers = "" if outlined else _EXPLICIT_WRAPPERS
    produce = (
        '        with pl.graph("layer"):\n'
        "            with pl.spmd(1):\n"
        "                mid = self.mm(x, w, mid)"
        if outlined
        else "        mid = self.layer(x, w, mid)"
    )
    program = pl.parse_program(_KERNEL + wrappers + _ENTRY.format(produce=produce))
    captured = []

    def capture(pass_obj: passes.Pass, candidate: ir.Program) -> None:
        if pass_obj.get_name() == "DeriveCallDirections":
            captured.append(candidate)

    # The frequent reads favor NZ for the private allocation produced through
    # both wrappers. The explicit Graph forwards the SPMD result and needs the
    # later-named SPMD rewritten before it. The outlined SPMD itself is void.
    instruments: list[passes.PassInstrument] = [
        *pass_verification_instruments,
        passes.CallbackInstrument(after_pass=capture),
    ]
    with passes.PassContext(instruments, runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        ir.compile(
            program,
            strategy=ir.OptimizationStrategy.AutoLayout,
            backend_type=ascend_backend,
            skip_ptoas=True,
            output_dir=str(tmp_path),
        )

    (lowered,) = captured
    spmd = "layer_spmd_0" if outlined else "z_spmd"
    _assert_call_result(lowered, "main", "layer")
    _assert_call_result(lowered, "layer", spmd, returns_tensor=not outlined)
    _assert_call_result(lowered, spmd, "mm_site1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
