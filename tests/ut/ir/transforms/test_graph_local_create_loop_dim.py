# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Local allocation extent facts do not imply descriptor or data invariance."""

import pytest
from pypto import ir, passes
from pypto.pypto_core import DataType


@pytest.mark.parametrize(
    "case,accepted",
    [
        ("normalized", True),
        ("frontend_tuple", True),
        ("bare_alias", True),
        ("negative_axis", True),
        ("partial_fixed_axis", False),
        ("runtime_axis_extent", False),
        ("forged_tuple", False),
        ("dynamic_axis", False),
        ("unknown_callee", False),
        ("view", False),
        ("pipe_override", False),
        ("create_memref", True),
        ("alias_memref", True),
        ("create_shape_mismatch", False),
        ("alias_shape_mismatch", False),
        ("alias_rank_mismatch", False),
        ("boundary_alias_shape_mismatch", False),
    ],
)
def test_local_loop_allocation_axes_can_be_frozen(case, accepted):
    s = ir.Span.unknown()
    index = ir.ScalarType(DataType.INDEX)
    n = ir.Var("num_tokens", index, s)

    def ci(value):
        return ir.ConstInt(value, DataType.INDEX, s)

    out_type = ir.TensorType([80, 16], DataType.FP32)
    out = ir.Var("output", out_type, s)
    local_shape = [n, 16] if case in ("partial_fixed_axis", "runtime_axis_extent") else [80, 16]
    local_type = ir.TensorType(local_shape, DataType.FP32)
    assigned_type = local_type
    if case in ("create_memref", "alias_memref"):
        assigned_type = ir.TensorType(
            local_shape, DataType.FP32, memref=ir.MemRef("local_storage", 0, 5120, s)
        )
    if case == "create_shape_mismatch":
        assigned_type = ir.TensorType([79, 16], DataType.FP32, memref=ir.MemRef("local_storage", 0, 5120, s))
    local = ir.Var("gm_pipe_buffer_local" if case == "pipe_override" else "local", assigned_type, s)
    child_out = ir.Var("child_out", out_type, s)
    child_rows = ir.Var("rows", index, s)
    child = ir.Function(
        "consume",
        [(child_out, ir.ParamDirection.InOut), child_rows],
        [out_type],
        ir.ReturnStmt([child_out], s),
        s,
        ir.FunctionType.AIV,
    )
    shape_args = []
    if case in ("frontend_tuple", "forged_tuple"):
        shape_args = [ir.MakeTuple([n if case == "forged_tuple" else ci(80), ci(16)], s)]
    create = ir.Call(ir.Op("tensor.create"), shape_args, local_type, s)
    params = [(out, ir.ParamDirection.InOut), n]
    if case == "unknown_callee":
        producer_out = ir.Var("producer_out", local_type, s)
        producer = ir.Function(
            "producer",
            [],
            [local_type],
            ir.SeqStmts(
                [
                    ir.AssignStmt(producer_out, ir.Call(ir.Op("tensor.create"), [], local_type, s), s),
                    ir.ReturnStmt([producer_out], s),
                ],
                s,
            ),
            s,
            ir.FunctionType.AIV,
        )
        create = ir.Call(ir.GlobalVar("producer"), [], local_type, s)
    stmts: list[ir.Stmt]
    if case == "boundary_alias_shape_mismatch":
        # This is a deliberately malformed late-IR alias. A reannotation
        # cannot turn the runtime extent into fixed boundary metadata.
        params.append(local)
        stmts = []
    else:
        stmts = [ir.AssignStmt(local, create, s)]
    source = local
    alias_shapes = {
        "bare_alias": local_shape,
        "alias_memref": local_shape,
        "alias_shape_mismatch": [79, 16],
        "alias_rank_mismatch": [80],
        "boundary_alias_shape_mismatch": [n, 16],
    }
    if case in alias_shapes:
        alias_type = ir.TensorType(
            alias_shapes[case], DataType.FP32, memref=ir.MemRef("alias_storage", 0, 5120, s)
        )
        source = ir.Var("alias", alias_type, s)
        stmts.append(ir.AssignStmt(source, local, s))
    if case == "view":
        source = ir.Var("view", local_type, s)
        view = ir.Call(ir.Op("tensor.reshape"), [local, ir.MakeTuple([ci(80), ci(16)], s)], local_type, s)
        stmts.append(ir.AssignStmt(source, view, s))
    axis = (
        n
        if case == "dynamic_axis"
        else ci(-1 if case == "negative_axis" else 1 if case == "partial_fixed_axis" else 0)
    )
    dim = ir.Call(ir.Op("tensor.dim"), [source, axis], index, s)
    rows = ir.Var("capacity", index, s)
    result = ir.Var("result", out_type, s)
    stmts.extend(
        [
            ir.AssignStmt(rows, dim, s),
            ir.AssignStmt(result, ir.Call(ir.GlobalVar("consume"), [out, rows], out_type, s), s),
            ir.ReturnStmt([out], s),
        ]
    )
    # Actual40 shape: fresh allocation inside the layer loop, so Step C cannot
    # hoist it into a boundary tensor and accidentally bypass the new proof.
    i = ir.Var("layer", index, s)
    loop = ir.ForStmt(i, ci(0), ci(4), ci(1), [], ir.SeqStmts(stmts[:-1], s), [], s)
    graph = ir.Function(
        "graph", params, [out_type], ir.SeqStmts([loop, stmts[-1]], s), s, ir.FunctionType.Graph
    )
    functions = [child, graph] + ([producer] if case == "unknown_callee" else [])
    program = ir.Program(functions, "local_extent", s)
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        if accepted:
            passes.legalize_graph_boundary()(program)
        else:
            with pytest.raises(ValueError):
                passes.legalize_graph_boundary()(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
