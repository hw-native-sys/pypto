# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the staged logical-Tile to explicit-PTO-handle bridge pass."""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    # This experimental pass is intentionally outside the default pipeline.
    # Disable the outer test fixture's pipeline-property checks and verify the
    # new produced property directly below.
    with passes.PassContext(
        [],
        passes.VerificationLevel.NONE,
        memory_planner=planner,
    ):
        return passes.materialize_pto_tile_handles()(program)


def _function(program: ir.Program) -> ir.Function:
    return next(iter(program.functions.values()))


def _statements(program: ir.Program) -> list[ir.Stmt]:
    body = _function(program).body
    return list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]


def _with_statements(program: ir.Program, statements: list[ir.Stmt]) -> ir.Program:
    old_func = _function(program)
    new_func = ir.Function(
        old_func.name,
        list(zip(old_func.params, old_func.param_directions)),
        old_func.return_types,
        ir.SeqStmts(statements, old_func.body.span),
        old_func.span,
        type=old_func.func_type,
        attrs=dict(old_func.attrs),
    )
    return ir.Program([new_func], program.name, program.span)


def _verify_handle_property(program: ir.Program) -> None:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.PTOHandlesMaterialized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
    assert diagnostics == []


def _make_straight_line_program() -> ir.Program:
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            input_a: pl.Tensor[[16, 16], pl.FP32],
            input_b: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a: pl.Tile[[16, 16], pl.FP32, pl.MemRef("mem_a", 64, 1024), pl.Mem.Vec] = pl.load(
                input_a, [0, 0], [16, 16]
            )
            b: pl.Tile[[16, 16], pl.FP32, pl.MemRef("mem_b", 128, 1024), pl.Mem.Vec] = pl.load(
                input_b, [0, 0], [16, 16]
            )
            root: pl.Tile[[16, 16], pl.FP32, pl.MemRef("mem_root", 192, 1024), pl.Mem.Vec] = pl.sqrt(a)
            added: pl.Tile[[16, 16], pl.FP32, pl.MemRef("mem_added", 256, 1024), pl.Mem.Vec] = pl.add(root, b)
            multiplied: pl.Tile[[16, 16], pl.FP32, pl.MemRef("mem_mul", 320, 1024), pl.Mem.Vec] = pl.mul(
                added, b
            )
            result: pl.Tensor[[16, 16], pl.FP32] = pl.store(multiplied, [0, 0], output)
            return result

    return Program


@pytest.mark.parametrize(
    ("planner", "alloc_arg_count"),
    [
        (passes.MemoryPlanner.PYPTO, 3),
        (passes.MemoryPlanner.PTOAS, 2),
    ],
)
def test_materializes_unique_explicit_handle_plan(
    planner: passes.MemoryPlanner,
    alloc_arg_count: int,
) -> None:
    before = _make_straight_line_program()
    after = _run(before, planner)
    stmts = _statements(after)

    alloc_name = ir.get_op("pto.alloc_tile").name
    supported_names = {
        ir.get_op("tile.load").name,
        ir.get_op("tile.sqrt").name,
        ir.get_op("tile.add").name,
        ir.get_op("tile.mul").name,
        ir.get_op("tile.store").name,
    }
    allocations: list[ir.AssignStmt] = []
    logical_to_handle: dict[int, ir.Var] = {}

    for index, stmt in enumerate(stmts):
        if not isinstance(stmt, ir.AssignStmt) or not isinstance(stmt.value, ir.Call):
            continue
        call = stmt.value
        if call.op.name == alloc_name:
            allocations.append(stmt)
            assert isinstance(stmt.var.type, ir.PTOTileBufType)
            assert len(call.args) == alloc_arg_count
            valid_rows = call.args[-2]
            valid_cols = call.args[-1]
            assert isinstance(valid_rows, ir.ConstInt)
            assert isinstance(valid_cols, ir.ConstInt)
            assert valid_rows.value == 16
            assert valid_cols.value == 16
            continue
        if call.op.name not in supported_names:
            continue

        input_handles = list(call.attrs["pto_input_handles"])
        expected_inputs: list[ir.Var] = []
        for arg in call.args:
            if not isinstance(arg.type, ir.TileType):
                continue
            assert isinstance(arg, ir.Var)
            expected_inputs.append(logical_to_handle[id(arg)])
        assert input_handles == expected_inputs

        if isinstance(stmt.var.type, ir.TileType):
            handle = call.attrs["pto_output_handle"]
            assert isinstance(handle, ir.Var)
            preceding_alloc = stmts[index - 1]
            assert isinstance(preceding_alloc, ir.AssignStmt)
            assert preceding_alloc.var is handle
            logical_to_handle[id(stmt.var)] = handle
        else:
            assert "pto_output_handle" not in call.attrs

    assert len(allocations) == 5
    assert len({id(alloc.var) for alloc in allocations}) == 5

    if planner is passes.MemoryPlanner.PYPTO:
        addresses: list[int] = []
        for alloc in allocations:
            assert isinstance(alloc.value, ir.Call)
            address = alloc.value.args[0]
            assert isinstance(address, ir.ConstInt)
            addresses.append(address.value)
        assert addresses == [64, 128, 192, 256, 320]

    store_stmt = next(
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tile.store").name
    )
    store_call = store_stmt.value
    assert isinstance(store_call, ir.Call)
    stored_tile = store_call.args[0]
    assert isinstance(stored_tile, ir.Var)
    assert list(store_call.attrs["pto_input_handles"]) == [logical_to_handle[id(stored_tile)]]
    _verify_handle_property(after)


def test_is_idempotent() -> None:
    once = _run(_make_straight_line_program(), passes.MemoryPlanner.PYPTO)
    twice = _run(once, passes.MemoryPlanner.PYPTO)
    ir.assert_structural_equal(once, twice)
    _verify_handle_property(twice)


def test_handle_plan_survives_binary_serialization() -> None:
    after = _run(_make_straight_line_program(), passes.MemoryPlanner.PTOAS)
    restored = ir.deserialize(ir.serialize(after))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(after, restored)
    _verify_handle_property(restored)


def test_property_verifier_rejects_non_dominating_output_handle() -> None:
    after = _run(_make_straight_line_program(), passes.MemoryPlanner.PTOAS)
    stmts = _statements(after)
    allocations = [
        stmt
        for stmt in stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("pto.alloc_tile").name
    ]
    first_producer_index = next(
        index
        for index, stmt in enumerate(stmts)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tile.load").name
    )
    producer = stmts[first_producer_index]
    assert isinstance(producer, ir.AssignStmt)
    assert isinstance(producer.value, ir.Call)
    corrupted_call = ir.set_call_attrs(
        producer.value,
        {
            "pto_input_handles": [],
            # The second allocation appears after this producer, so it does
            # not dominate the binding.
            "pto_output_handle": allocations[1].var,
        },
    )
    stmts[first_producer_index] = ir.AssignStmt(producer.var, corrupted_call, producer.span)

    corrupted = _with_statements(after, stmts)

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.PTOHandlesMaterialized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, corrupted)
    assert any("does not name a dominating" in diagnostic.message for diagnostic in diagnostics)


def test_rejects_existing_allocation_from_different_planner() -> None:
    ptoas = _run(_make_straight_line_program(), passes.MemoryPlanner.PTOAS)

    with pytest.raises(ValueError, match="PYPTO planning requires 3"):
        _run(ptoas, passes.MemoryPlanner.PYPTO)


def test_property_verifier_rejects_malformed_allocation_metadata() -> None:
    after = _run(_make_straight_line_program(), passes.MemoryPlanner.PTOAS)
    stmts = _statements(after)
    alloc_index = next(
        index
        for index, stmt in enumerate(stmts)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("pto.alloc_tile").name
    )
    alloc = stmts[alloc_index]
    assert isinstance(alloc, ir.AssignStmt)
    assert isinstance(alloc.value, ir.Call)
    malformed_call = ir.Call(alloc.value.op, alloc.value.args[:1], alloc.value.type, alloc.value.span)
    stmts[alloc_index] = ir.AssignStmt(alloc.var, malformed_call, alloc.span)

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.PTOHandlesMaterialized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, _with_statements(after, stmts))
    assert any("must have two or three metadata operands" in diagnostic.message for diagnostic in diagnostics)


def test_rejects_view_operation_explicitly() -> None:
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]):
            tile: pl.Tile[[16, 16], pl.FP32, pl.MemRef("mem_tile", 64, 1024), pl.Mem.Vec] = pl.load(
                x, [0, 0], [16, 16]
            )
            _view: pl.Tile[[8, 32], pl.FP32, pl.MemRef("mem_tile", 64, 1024), pl.Mem.Vec] = pl.reshape(
                tile, [8, 32]
            )

    with pytest.raises(ValueError, match="does not yet support Tile operation 'tile.reshape'"):
        _run(Program, passes.MemoryPlanner.PYPTO)


def test_rejects_dynamic_valid_shape_until_step_5() -> None:
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, x: pl.Tensor[[16, 16], pl.FP32], valid_rows: pl.Scalar[pl.INDEX]):
            _tile: pl.Tile[
                [16, 16],
                pl.FP32,
                pl.MemRef("mem_tile", 64, 1024),
                pl.Mem.Vec,
                pl.TileView(valid_shape=[valid_rows, 16]),
            ] = pl.load(x, [0, 0], [16, 16], [valid_rows, 16])

    with pytest.raises(ValueError, match="does not support dynamic valid shape"):
        _run(Program, passes.MemoryPlanner.PYPTO)


def test_rejects_control_flow_explicitly() -> None:
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]):
            for _ in pl.range(1):
                _tile: pl.Tile[[16, 16], pl.FP32, pl.MemRef("mem_tile", 64, 1024), pl.Mem.Vec] = pl.load(
                    x, [0, 0], [16, 16]
                )

    with pytest.raises(ValueError, match="supports straight-line functions only"):
        _run(Program, passes.MemoryPlanner.PYPTO)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
