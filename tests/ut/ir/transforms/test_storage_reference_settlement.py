# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Storage references preserve fixed destinations and mixed loop entry transfers."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.op import tile as tile_ops


@pytest.mark.parametrize("parameter_kind", ["pointer", "tile"])
@pytest.mark.parametrize("buffer_ir", [False, True])
def test_common_window_does_not_replace_caller_result_binding(parameter_kind, buffer_ir):
    """Equal incoming handles cannot discharge a different caller destination."""
    span = ir.Span.unknown()
    destination_base = ir.Var("caller_destination", ir.PtrType(), span)
    source_base = ir.Var("source_storage", ir.PtrType(), span)
    destination_type = ir.TileType(
        [16, 16],
        ir.DataType.FP16,
        ir.MemRef(destination_base, 0, 512, span, is_pinned=True),
        memory_space=ir.MemorySpace.Mat,
    )
    source_type = ir.TileType(
        [16, 16],
        ir.DataType.FP16,
        ir.MemRef(source_base, 0, 512, span),
        memory_space=ir.MemorySpace.Mat,
    )
    parameter = (
        destination_base if parameter_kind == "pointer" else ir.Var("caller_tile", destination_type, span)
    )
    pool = ir.Var("pool", ir.TensorType([16, 16], ir.DataType.FP16), span)
    flag = ir.Var("flag", ir.ScalarType(ir.DataType.BOOL), span)
    source = ir.Var("source", source_type, span)
    result = ir.Var("result", destination_type, span)
    allocation = ir.Call(
        ir.get_op("tile.alloc"),
        [
            ir.ConstInt(ir.MemorySpace.Mat.value, ir.DataType.INDEX, span),
            ir.ConstInt(512, ir.DataType.INDEX, span),
        ],
        source_base.type,
        span,
    )
    load = tile_ops.load(pool, [0, 0], [16, 16], target_memory=ir.MemorySpace.Mat)
    typed_load = ir.Call(load.op, list(load.args), dict(load.kwargs), dict(load.attrs), source_type, span)
    branch = ir.IfStmt(
        flag,
        ir.YieldStmt([source], span),
        ir.YieldStmt([source], span),
        [result],
        span,
    )
    function = ir.Function(
        "kernel",
        [parameter, pool, flag],
        [ir.TileType([16, 16], ir.DataType.FP16, memory_space=ir.MemorySpace.Mat)],
        ir.SeqStmts(
            [
                ir.AssignStmt(source_base, allocation, span),
                ir.AssignStmt(source, typed_load, span),
                branch,
                ir.ReturnStmt([result], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.InCore,
    )
    program = ir.Program([function], "CallerResultBinding", span)
    with passes.PassContext([], enable_buffer_ir=buffer_ir):
        with pytest.raises(ValueError, match="unsupported tile.move memory-space pair"):
            passes.materialize_semantic_aliases()(program)


@pytest.mark.parametrize("bounds", ["count", "0", "2"], ids=["runtime", "zero_trip", "two_trips"])
@pytest.mark.parametrize("buffer_ir", [False, True])
def test_reference_carry_preserves_copyable_peer_entry_isolation(bounds, buffer_ir):
    """A Mat reference rewrite cannot discard its Vec peer's planned entry copy."""
    program = pl.parse_program(f"""
@pl.program
class MixedCarryIsolation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[32, 16], pl.FP16], count: pl.Scalar[pl.INDEX]
               ) -> tuple[pl.Tile[[16, 16], pl.FP16, pl.Mem.Left],
                          pl.Tile[[16, 16], pl.FP16, pl.Mem.Left],
                          pl.Tile[[16, 16], pl.FP16, pl.Mem.Vec],
                          pl.Tile[[16, 16], pl.FP16, pl.Mem.Vec]]:
        mat_seed = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
        vec_seed = pl.tile.load(pool, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        for index, (mat_current, vec_current) in pl.range({bounds}, init_values=(mat_seed, vec_seed)):
            mat_next = pl.tile.load(pool, [16, 0], [16, 16], target_memory=pl.Mem.Mat)
            vec_next = pl.tile.add(vec_current, vec_current)
            mat_result, vec_result = pl.yield_(mat_next, vec_next)
        old_left = pl.tile.move(mat_seed, target_memory=pl.Mem.Left)
        result_left = pl.tile.move(mat_result, target_memory=pl.Mem.Left)
        return old_left, result_left, vec_seed, vec_result
""")
    with passes.PassContext([], enable_buffer_ir=buffer_ir):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))
    assignments = {}
    loops = []

    class Collector(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            assignments[op.var.unique_id] = op
            super().visit_assign_stmt(op)

        def visit_for_stmt(self, op: ir.ForStmt) -> None:
            loops.append(op)
            super().visit_for_stmt(op)

    Collector().visit_stmt(next(iter(after.functions.values())).body)
    assert len(loops) == 1
    loop = loops[0]
    mat_argument, vec_argument = loop.iter_args
    initial = vec_argument.initValue
    assert isinstance(initial, ir.Var)
    definition = assignments[initial.unique_id]
    assert isinstance(definition.value, ir.Call)
    if bounds == "0":
        assert definition.value.op.name == ir.get_op("tile.load").name
        return
    assert isinstance(mat_argument.type, ir.TileType) and mat_argument.type.memref is None
    assert isinstance(loop.return_vars[0].type, ir.TileType) and loop.return_vars[0].type.memref is None
    assert definition.value.op.name == ir.get_op("tile.move").name
    source = definition.value.args[0]
    assert isinstance(source, ir.Var) and source.name_hint == "vec_seed"
    assert isinstance(initial.type, ir.TileType) and initial.type.memref is not None
    assert isinstance(source.type, ir.TileType) and source.type.memref is not None
    assert not ir.MemRef.may_alias(initial.type.memref, source.type.memref)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
