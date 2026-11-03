# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared construction, inspection and comparison helpers for storage regressions."""

from textwrap import dedent, indent

import pypto.language as pl
from pypto import ir, passes
from pypto.pypto_core.ir import _without_type_memref


def without_call_memrefs(program: ir.Program) -> ir.Program:
    """Remove inferred Call result allocations, retaining all variable storage.

    Storage passes allocate variables, while DSL operator inference can copy
    input MemRefs into Call result types. These tests specify the variables'
    storage contract, not those redundant inferred copies. Exact comparison
    still checks every variable's allocation identity, offset, size and slots,
    as well as every other Call type field. Roundtrip instruments compare the
    original IR without this normalization.
    """

    class NormalizeCalls(ir.IRMutator):
        def visit_call(self, op: ir.Call) -> ir.Expr:
            call = super().visit_call(op)
            assert isinstance(call, ir.Call)
            return ir.Call(
                call.op, call.args, call.kwargs, call.attrs, _without_type_memref(call.type), call.span
            )

    return NormalizeCalls().visit_program(program)


def _program(body, *, input_shape="[8, 8]", output_shape="[4, 8]", rows=False):
    """Share the compile-only vector signature; keep each regression body visible."""
    extra = "rows_a: pl.Scalar[pl.INDEX]," if rows else ""
    return pl.parse_program(f"""
@pl.program
class NativeSelection:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[{input_shape}, pl.FP32], flag: pl.Scalar[pl.INT32],
               count: pl.Scalar[pl.INDEX], {extra}
               out: pl.Out[pl.Tensor[{output_shape}, pl.FP32]]) -> pl.Tensor[{output_shape}, pl.FP32]:
{indent(dedent(body).strip(), "        ")}
""")


def _place(program, planner, instruments=(), *, flatten=False):
    with passes.PassContext(list(instruments), memory_planner=planner, enable_buffer_ir=False):
        if flatten:
            program = passes.flatten_tile_nd_to_2d()(program)
        program = passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))
        if planner == passes.MemoryPlanner.PYPTO:
            program = passes.memory_reuse()(program)
        return passes.allocate_memory_addr()(program)


class _Nodes(ir.IRVisitor):
    def __init__(self, program):
        super().__init__()
        self.assignments = []
        self.branches = []
        self.loops = []
        self.visit_program(program)

    def visit_assign_stmt(self, op):
        self.assignments.append(op)
        super().visit_assign_stmt(op)

    def visit_if_stmt(self, op):
        self.branches.append(op)
        super().visit_if_stmt(op)

    def visit_for_stmt(self, op):
        self.loops.append(op)
        super().visit_for_stmt(op)

    def visit_while_stmt(self, op):
        self.loops.append(op)
        super().visit_while_stmt(op)


def _assert_dynamic_if_results(program):
    for branch in _Nodes(program).branches:
        for result in branch.return_vars:
            if isinstance(result.type, ir.TileType):
                assert result.type.memref is None, "If results must not inherit an arm's allocation"
                assert result.type.memory_space is not None


def _assert_descriptor_carries(program, expected_slots):
    checked = 0
    for loop in _Nodes(program).loops:
        tiles = [i for i, arg in enumerate(loop.iter_args) if isinstance(arg.type, ir.TileType)]
        if tiles and all(loop.iter_args[i].type.memref is None for i in tiles):
            assert len(tiles) == expected_slots
            assert all(loop.return_vars[i].type.memref is None for i in tiles)
            checked += 1
    assert checked == 1
    assert "_carry_spill_" not in program.as_python()
    assert "carry_input_" not in program.as_python()
