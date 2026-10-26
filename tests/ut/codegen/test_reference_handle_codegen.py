# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Selected Mat references are native SCF handles under every planner."""

import re

import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import BackendType


def _reference_program(kind, *, direct_buffer, addressed):
    """Build already legalized IR so codegen cannot conceal a missing handle phi.

    These storage-only tests use create/alloc rather than payload computation.
    Both owners are distinct and live before the region; selecting their handle
    must preserve the branch edge and allocate no third Mat payload buffer.
    """
    span = ir.Span.unknown()
    index = ir.ScalarType(DataType.INDEX)
    flag = ir.Var("flag", ir.ScalarType(DataType.BOOL), span)
    view = ir.TileView(
        valid_shape=[16, 16],
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
    )
    if direct_buffer:
        reference_type = ir.BufferType(
            [16, 16],
            DataType.FP16,
            ir.MemorySpace.Mat,
            blayout=ir.TileLayout.col_major,
            slayout=ir.TileLayout.row_major,
        )
    else:
        reference_type = ir.TileType([16, 16], DataType.FP16, tile_view=view, memory_space=ir.MemorySpace.Mat)

    def owner(name, address):
        if direct_buffer:
            value = ir.Var(name, reference_type, span)
            arguments: list[ir.Expr] = [ir.MakeTuple([], span)]
            if addressed:
                arguments.append(ir.ConstInt(address, DataType.INT64, span))
            call = ir.Call(ir.get_op("buffer.alloc"), arguments, reference_type, span)
        else:
            base = ir.Var(f"{name}_base", ir.PtrType(), span)
            memory = ir.MemRef(base, address, 512, span)
            owner_type = ir.TileType([16, 16], DataType.FP16, memory, view, ir.MemorySpace.Mat)
            value = ir.Var(name, owner_type, span)
            call = ir.Call(ir.get_op("tile.create"), [], owner_type, span)
        return value, ir.AssignStmt(value, call, span)

    a, define_a = owner("a", 0)
    b, define_b = owner("b", 512)
    result = ir.Var("selected", reference_type, span)

    def branch(left, selected):
        return ir.IfStmt(
            flag,
            ir.YieldStmt([left], span),
            ir.YieldStmt([b], span),
            [selected],
            span,
        )

    if kind == "if":
        region = branch(a, result)
    else:
        carried = ir.IterArg("carried", reference_type, a, span)
        updated = ir.Var("updated", reference_type, span)
        body = ir.SeqStmts([branch(carried, updated), ir.YieldStmt([updated], span)], span)
        if kind == "for":
            region = ir.ForStmt(
                ir.Var("iteration", index, span),
                ir.ConstInt(0, DataType.INDEX, span),
                ir.ConstInt(2, DataType.INDEX, span),
                ir.ConstInt(1, DataType.INDEX, span),
                [carried],
                body,
                [result],
                span,
            )
        else:
            region = ir.WhileStmt(flag, [carried], body, [result], span)
    # A bare alias must retain the current selected SSA even in addressed
    # legacy codegen; it has no allocation address from which to recover it.
    alias = ir.Var("alias", reference_type, span)
    observed = ir.Var("observed", reference_type, span)
    # Explicit Buffer IR names aliases with an operation, rather than a bare
    # handle assignment; legacy Tile SSA still uses the ordinary rename form.
    alias_value = (
        ir.Call(ir.get_op("buffer.reshape"), [result], reference_type, span) if direct_buffer else result
    )
    body = ir.SeqStmts(
        [
            define_a,
            define_b,
            region,
            ir.AssignStmt(alias, alias_value, span),
            branch(alias, observed),
            ir.ReturnStmt([], span),
        ],
        span,
    )
    function = ir.Function("kernel", [flag], [], body, span, type=ir.FunctionType.InCore)
    return ir.Program([function], "ReferenceHandles", span)


@pytest.mark.parametrize("planner", list(passes.MemoryPlanner.__members__.values()))
@pytest.mark.parametrize("direct_buffer", [False, True], ids=["legacy-tile", "buffer"])
@pytest.mark.parametrize("kind", ["if", "for", "while"])
def test_reference_regions_preserve_mat_handles(kind, direct_buffer, planner):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    addressed = planner != passes.MemoryPlanner.PTOAS
    program = _reference_program(kind, direct_buffer=direct_buffer, addressed=addressed)
    try:
        with passes.PassContext([], memory_planner=planner):
            mlir = codegen.PTOCodegen().generate(program, emit_tile_addr=addressed)
    finally:
        backend.reset_for_testing()
    assert "pto.tmov" not in mlir, mlir
    allocations = [line for line in mlir.splitlines() if " = pto.alloc_tile" in line]
    assert len(allocations) == 2, mlir
    selected = [line for line in mlir.splitlines() if " = scf.if " in line]
    assert len(selected) == 2 and all("!pto.tile_buf<" in line for line in selected), mlir
    yields = [line for line in mlir.splitlines() if "scf.yield " in line]
    assert len(yields) == (4 if kind == "if" else 5), mlir
    assert all("!pto.tile_buf<" in line for line in yields), mlir
    assert len(set(re.findall(r"scf.yield (%[A-Za-z0-9_]+)", mlir))) >= 2, mlir
    if kind != "if":
        loop = next(line for line in mlir.splitlines() if f" = scf.{kind} " in line)
        assert "!pto.tile_buf<" in loop, mlir
        if kind == "for":
            assert "iter_args(" in loop, mlir
        else:
            condition = next(line for line in mlir.splitlines() if "scf.condition(" in line)
            assert "!pto.tile_buf<" in condition, mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
