# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the staged logical-Tile to explicit-PTO-handle bridge pass."""

import re

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.pypto_core import codegen


def _run(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    # Isolate Step 3 from its normal default-pipeline prerequisites; the tests
    # construct the required Tile/MemRef state directly and verify the produced
    # target property below.
    with passes.PassContext(
        [],
        passes.VerificationLevel.NONE,
        memory_planner=planner,
    ):
        return passes.materialize_pto_tile_handles()(program)


def _lower(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    materialized = _run(program, planner)
    with passes.PassContext(
        [],
        passes.VerificationLevel.NONE,
        memory_planner=planner,
    ):
        return passes.lower_tile_to_pto_ir()(materialized)


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


def _verify_bufferized_property(program: ir.Program) -> None:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.PTOBufferized)
    diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
    assert diagnostics == []


def _normalized_codegen_contract(text: str) -> list[str]:
    contract_ops = (
        "pto.make_tensor_view",
        "pto.alloc_tile",
        "pto.partition_view",
        "pto.tload",
        "pto.tsqrt",
        "pto.tadd",
        "pto.tmul",
        "pto.tstore",
    )
    normalized: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not any(op in stripped for op in contract_ops):
            continue
        normalized.append(re.sub(r"%[A-Za-z0-9_]+", "%ssa", stripped))
    return normalized


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
            assert isinstance(address.type, ir.ScalarType)
            assert address.type.dtype == pl.INT64
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


@pytest.mark.parametrize(
    ("planner", "emit_tile_addr"),
    [
        (passes.MemoryPlanner.PYPTO, True),
        (passes.MemoryPlanner.PTOAS, False),
    ],
)
def test_step4_eliminates_logical_tiles_and_matches_legacy_codegen(
    planner: passes.MemoryPlanner,
    emit_tile_addr: bool,
) -> None:
    logical = _make_straight_line_program()
    lowered = _lower(logical, planner)
    statements = _statements(lowered)

    target_calls = [
        stmt.expr for stmt in statements if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    ]
    assert [call.op.name for call in target_calls] == [
        "pto.tload",
        "pto.tload",
        "pto.tsqrt",
        "pto.tadd",
        "pto.tmul",
        "pto.tstore",
    ]

    for call in target_calls:
        assert isinstance(call.type, ir.UnknownType)
        assert all(not isinstance(arg.type, ir.TileType) for arg in call.args)
        assert "pto_input_handles" not in call.attrs
        assert "pto_output_handle" not in call.attrs

    load = target_calls[0]
    assert len(load.args) == 4
    assert load.kwargs == {}
    assert isinstance(load.args[1], ir.MakeTuple)
    assert isinstance(load.args[2], ir.MakeTuple)
    assert isinstance(load.args[3].type, ir.PTOTileBufType)

    store = target_calls[-1]
    assert len(store.args) == 4
    assert isinstance(store.args[0].type, ir.PTOTileBufType)
    assert isinstance(store.args[1], ir.MakeTuple)
    assert isinstance(store.args[2], ir.MakeTuple)

    tile_definitions = [
        stmt
        for stmt in statements
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType)
    ]
    assert tile_definitions == []
    _verify_bufferized_property(lowered)

    legacy_mlir = codegen.PTOCodegen().generate(logical, emit_tile_addr=emit_tile_addr)
    target_mlir = codegen.PTOCodegen().generate(lowered, emit_tile_addr=emit_tile_addr)
    assert _normalized_codegen_contract(target_mlir) == _normalized_codegen_contract(legacy_mlir)


def test_codegen_dispatches_target_and_legacy_functions_independently() -> None:
    """A deferred/legacy function must not force target functions back to inference."""

    target = _function(_lower(_make_straight_line_program(), passes.MemoryPlanner.PYPTO))

    @pl.program
    class LegacyProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def scalar_io(
            self,
            source: pl.Tensor[[1], pl.INT32],
            output: pl.Out[pl.Tensor[[1], pl.INT32]],
        ) -> pl.Tensor[[1], pl.INT32]:
            value: pl.Scalar[pl.INT32] = pl.tensor.read(source, [0])
            result: pl.Tensor[[1], pl.INT32] = pl.tensor.write(output, [0], value)
            return result

    legacy = _function(LegacyProgram)
    mixed = ir.Program([target, legacy], "mixed_target_legacy", ir.Span.unknown())

    mlir = codegen.PTOCodegen().generate(mixed, emit_tile_addr=True)
    assert "func.func @main" in mlir
    assert "func.func @scalar_io" in mlir
    assert "pto.tadd" in mlir
    assert "pto.make_tensor_view" in mlir


def test_step4_target_ir_survives_binary_serialization() -> None:
    lowered = _lower(_make_straight_line_program(), passes.MemoryPlanner.PTOAS)
    restored = ir.deserialize(ir.serialize(lowered))

    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored)
    _verify_bufferized_property(restored)


def test_step4_materializes_spmd_identity_as_explicit_target_parameters() -> None:
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self):
            _block_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            _block_num: pl.Scalar[pl.INDEX] = pl.tile.get_block_num()
            _subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()

    lowered = _lower(Program, passes.MemoryPlanner.PTOAS)
    func = _function(lowered)

    assert [param.name_hint for param in func.params] == [
        "__pypto_spmd_block_idx",
        "__pypto_spmd_block_num",
        "__pypto_spmd_subblock_idx",
    ]
    assert all(isinstance(param.type, ir.ScalarType) for param in func.params)
    assert all(
        isinstance(param.type, ir.ScalarType) and param.type.dtype == pl.INT32 for param in func.params
    )
    assert func.attrs["pto.uses_spmd_block_params"] is True
    assert func.attrs["pto.uses_subblock_param"] is True

    assignments = [stmt for stmt in _statements(lowered) if isinstance(stmt, ir.AssignStmt)]
    assert len(assignments) == 3
    assert all(isinstance(stmt.value, ir.Cast) for stmt in assignments)
    assert all(
        isinstance(stmt.value, ir.Cast)
        and isinstance(stmt.value.type, ir.ScalarType)
        and stmt.value.type.dtype == pl.INDEX
        for stmt in assignments
    )
    assert all(not isinstance(stmt.value, ir.Call) for stmt in assignments)
    _verify_bufferized_property(lowered)


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


def test_materializes_reshape_as_a_distinct_typed_handle() -> None:
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

    after = _run(Program, passes.MemoryPlanner.PYPTO)
    allocs = [
        stmt
        for stmt in _statements(after)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pto.alloc_tile"
    ]
    assert len(allocs) == 2
    assert allocs[0].var.type != allocs[1].var.type
    _verify_handle_property(after)

    lowered = _lower(Program, passes.MemoryPlanner.PYPTO)
    assert all(
        not (
            isinstance(stmt, ir.AssignStmt)
            and isinstance(stmt.value, ir.Call)
            and stmt.value.op.name == "tile.reshape"
        )
        for stmt in _statements(lowered)
    )
    _verify_bufferized_property(lowered)


def test_materializes_dynamic_valid_shape_as_allocation_operands() -> None:
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

    after = _run(Program, passes.MemoryPlanner.PYPTO)
    func = _function(after)
    alloc = next(
        stmt
        for stmt in _statements(after)
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pto.alloc_tile"
    )
    assert isinstance(alloc.value, ir.Call)
    assert alloc.value.args[-2] is func.params[1]
    assert isinstance(alloc.value.args[-1], ir.ConstInt)
    assert alloc.value.args[-1].value == 16
    _verify_handle_property(after)


def test_materializes_handles_inside_structured_control_flow() -> None:
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]):
            for _ in pl.range(1):
                _tile: pl.Tile[[16, 16], pl.FP32, pl.MemRef("mem_tile", 64, 1024), pl.Mem.Vec] = pl.load(
                    x, [0, 0], [16, 16]
                )

    after = _run(Program, passes.MemoryPlanner.PYPTO)
    loop = next(stmt for stmt in _statements(after) if isinstance(stmt, ir.ForStmt))
    loop_stmts = list(loop.body.stmts) if isinstance(loop.body, ir.SeqStmts) else [loop.body]
    assert any(
        isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "pto.alloc_tile"
        for stmt in loop_stmts
    )
    _verify_handle_property(after)

    lowered = _lower(Program, passes.MemoryPlanner.PYPTO)
    lowered_loop = next(stmt for stmt in _statements(lowered) if isinstance(stmt, ir.ForStmt))
    lowered_stmts = (
        list(lowered_loop.body.stmts) if isinstance(lowered_loop.body, ir.SeqStmts) else [lowered_loop.body]
    )
    assert any(
        isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call) and stmt.expr.op.name == "pto.tload"
        for stmt in lowered_stmts
    )
    _verify_bufferized_property(lowered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
