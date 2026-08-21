# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes, testing
from pypto.backend import BackendType


def _plan_with_dsa_rp(program):
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP):
        initialized = passes.init_mem_ref()(program)
        return passes.allocate_memory_addr()(initialized)


def _recognized_edges(program) -> set[tuple[str, str, int]]:
    """Inspect recognizer output before placement or solver tie-breaking."""
    initialized = passes.init_mem_ref()(program)
    function = next(iter(initialized.functions.values()))
    return {
        (edge["first_name"], edge["second_name"], edge["cost"])
        for edge in testing.recognize_dsa_reuse_penalties(function)
    }


def _recognized_pairs(program) -> set[frozenset[str]]:
    return {frozenset((first, second)) for first, second, _cost in _recognized_edges(program)}


def _find_call(program, op_name: str) -> ir.Call:
    function = next(iter(program.functions.values()))
    calls: list[ir.Call] = []

    class _CallCollector(ir.IRVisitor):
        def visit_call(self, call):  # type: ignore[override]
            if call.op.name == op_name:
                calls.append(call)
            super().visit_call(call)

    _CallCollector().visit_stmt(function.body)
    assert len(calls) == 1
    return calls[0]


def _tile_ranges(program) -> dict[str, tuple[int, int]]:
    function = next(iter(program.functions.values()))
    result: dict[str, tuple[int, int]] = {}

    def record(var):
        tile_type = var.type
        if isinstance(tile_type, ir.TileType) and tile_type.memref is not None:
            offset = tile_type.memref.byte_offset_
            assert isinstance(offset, ir.ConstInt)
            result[var.name_hint] = (offset.value, tile_type.memref.size_)

    def visit(stmt):
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                visit(child)
        elif isinstance(stmt, ir.AssignStmt):
            record(stmt.var)
        elif isinstance(stmt, (ir.ForStmt, ir.WhileStmt)):
            visit(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            visit(stmt.then_body)
            if stmt.else_body is not None:
                visit(stmt.else_body)

    for param in function.params:
        record(param)
    visit(function.body)
    return result


def _overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    first_offset, first_size = first
    second_offset, second_size = second
    return first_offset < second_offset + second_size and second_offset < first_offset + first_size


def test_dsa_rp_recognizes_cross_pipe_war(ascend_backend):
    """A terminal Vector read followed by an inbound-DMA write is penalized."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            _consumed = pl.add(prior, prior)
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    assert _recognized_edges(Before) == {
        ("prior", "_consumed", 1),
        ("prior", "next_value", 1),
        ("_consumed", "next_value", 1),
    }
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["prior"], ranges["next_value"])


def test_dsa_rp_recognizes_cross_pipe_waw(ascend_backend):
    """A Vector write followed by an inbound-DMA write is penalized."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            _prior = pl.tile.full([64, 64], dtype=pl.FP32, value=0.0)
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    assert _recognized_edges(Before) == {("_prior", "next_value", 1)}
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["_prior"], ranges["next_value"])


def test_dsa_rp_does_not_penalize_same_pipe_handoff(ascend_backend):
    """Two Vector writes remain free to share one lifetime-compatible slot."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            _prior = pl.tile.full([64, 64], dtype=pl.FP32, value=0.0)
            next_value = pl.tile.full([64, 64], dtype=pl.FP32, value=1.0)
            return pl.store(next_value, [0, 0], output)

    assert _recognized_edges(Before) == set()
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert ranges["_prior"] == ranges["next_value"]


def test_dsa_rp_preserves_not_inplace_safe_semantic_separation():
    """Correctness separations remain hard even at a read-before-write boundary."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            source = pl.load(input_a, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            result = pl.recip(source)
            return pl.store(result, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["source"], ranges["result"])


def test_dsa_rp_preserves_tile_move_semantic_separation():
    """tile.move source and destination remain physically disjoint."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            source = pl.load(input_a, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            moved = pl.tile.move(source, target_memory=pl.Mem.Vec)
            return pl.store(moved, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["source"], ranges["moved"])


def test_dsa_rp_preserves_tuple_result_not_inplace_safe_separation():
    """Every physical result of a tuple op inherits its semantic no-alias inputs."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            source_tensor: pl.Tensor[[16, 64], pl.INT32],
            kvalue: pl.Scalar[pl.INT32],
            output: pl.Tensor[[16, 8], pl.INT32],
        ) -> pl.Tensor[[16, 8], pl.INT32]:
            source = pl.load(source_tensor, [0, 0], [16, 64], target_memory=pl.Mem.Vec)
            temporary = pl.tile.create([16, 64], pl.UINT8, target_memory=pl.Mem.Vec)
            destination, count = pl.tile.gather_compare(
                source,
                kvalue,
                temporary,
                cmp_mode="eq",
                out_cols=8,
            )
            _count_copy = pl.add(count, count)
            return pl.store(destination, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    for result in ("destination", "count"):
        for forbidden_input in ("source", "temporary"):
            assert not _overlap(ranges[result], ranges[forbidden_input])


def test_dsa_rp_does_not_promote_partial_handoff_endpoint(ascend_backend):
    """A consumer of only half an allocation does not create a whole-buffer edge."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            upper = prior[0:32, 0:64]
            _consumed = pl.add(upper, upper)
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    assert frozenset(("prior", "next_value")) not in _recognized_pairs(Before)
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert _overlap(ranges["prior"], ranges["next_value"])


def test_dsa_rp_vec_to_vec_tile_move_uses_vector_resource(ascend_backend):
    """A materialized Vec-to-Vec move is a Vector access, not an unknown op."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[32, 32], pl.FP32],
            input_b: pl.Tensor[[32, 32], pl.FP32],
            output: pl.Tensor[[32, 32], pl.FP32],
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            _moved = pl.tile.move(prior, target_memory=pl.Mem.Vec)
            next_value = pl.load(input_b, [0, 0], [32, 32], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    assert frozenset(("_moved", "next_value")) in _recognized_pairs(Before)
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["_moved"], ranges["next_value"])


def test_dsa_rp_acc_to_acc_tile_move_uses_vector_resource(ascend_backend):
    """PTOAS executes a materialized same-L0 tmov on the Vector pipe."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            lhs_input: pl.Tensor[[16, 64], pl.BF16],
            rhs_input: pl.Tensor[[64, 64], pl.BF16],
            output: pl.Tensor[[16, 64], pl.FP32],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            lhs_l1 = pl.load(lhs_input, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
            rhs_l1 = pl.load(rhs_input, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            lhs_l0 = pl.tile.move(lhs_l1, target_memory=pl.Mem.Left)
            rhs_l0 = pl.tile.move(rhs_l1, target_memory=pl.Mem.Right)
            prior = pl.matmul(lhs_l0, rhs_l0)
            _moved: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.move(prior, target_memory=pl.Mem.Acc)
            next_value = pl.matmul(lhs_l0, rhs_l0)
            return pl.store(next_value, [0, 0], output)

    assert frozenset(("_moved", "next_value")) in _recognized_pairs(Before)
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["_moved"], ranges["next_value"])


def test_dsa_rp_same_address_tile_move_remains_execution_access(ascend_backend):
    """A same-address tile.move is functional, never NoAccess."""
    span = ir.Span.unknown()
    shared_base = ir.Var("shared", ir.PtrType(), span)
    next_base = ir.Var("next", ir.PtrType(), span)
    input_a_base = ir.Var("input_a_base", ir.PtrType(), span)
    input_b_base = ir.Var("input_b_base", ir.PtrType(), span)

    input_a = ir.Var(
        "input_a",
        ir.TensorType([32, 32], DataType.FP32, ir.MemRef(input_a_base, 0, 4096, span)),
        span,
    )
    input_b = ir.Var(
        "input_b",
        ir.TensorType([32, 32], DataType.FP32, ir.MemRef(input_b_base, 0, 4096, span)),
        span,
    )
    source_type = ir.TileType(
        [32, 32],
        DataType.FP32,
        ir.MemRef(shared_base, 0, 4096, span),
        None,
        ir.MemorySpace.Vec,
    )
    moved_type = ir.TileType(
        [32, 32],
        DataType.FP32,
        ir.MemRef(shared_base, 0, 4096, span),
        None,
        ir.MemorySpace.Vec,
    )
    next_type = ir.TileType(
        [32, 32],
        DataType.FP32,
        ir.MemRef(next_base, 0, 4096, span),
        None,
        ir.MemorySpace.Vec,
    )
    source = ir.Var("source", source_type, span)
    moved = ir.Var("moved", moved_type, span)
    next_value = ir.Var("next_value", next_type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(source, ir.Call(ir.Op("tile.load"), [input_a], source_type, span), span),
            ir.AssignStmt(
                moved,
                ir.Call(
                    ir.Op("tile.move"),
                    [source],
                    {"target_memory": ir.MemorySpace.Vec},
                    moved_type,
                    span,
                ),
                span,
            ),
            ir.AssignStmt(next_value, ir.Call(ir.Op("tile.load"), [input_b], next_type, span), span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    function = ir.Function("main", [input_a, input_b], [], body, span, type=ir.FunctionType.InCore)

    edges = {
        (edge["first_name"], edge["second_name"], edge["cost"])
        for edge in testing.recognize_dsa_reuse_penalties(function)
    }
    assert edges == {("source", "next_value", 1)}


def test_dsa_rp_same_base_different_offset_tile_move_is_not_elided():
    """Subview offsets differ, so the move remains an execution access."""
    span = ir.Span.unknown()
    shared_base = ir.Var("shared", ir.PtrType(), span)
    next_base = ir.Var("next", ir.PtrType(), span)
    input_base = ir.Var("input_base", ir.PtrType(), span)

    input_var = ir.Var(
        "input",
        ir.TensorType([32, 32], DataType.FP32, ir.MemRef(input_base, 0, 4096, span)),
        span,
    )
    source_type = ir.TileType(
        [32, 32],
        DataType.FP32,
        ir.MemRef(shared_base, 0, 4096, span),
        None,
        ir.MemorySpace.Vec,
    )
    moved_type = ir.TileType(
        [32, 32],
        DataType.FP32,
        ir.MemRef(shared_base, 32, 4096, span),
        None,
        ir.MemorySpace.Vec,
    )
    next_type = ir.TileType(
        [32, 32],
        DataType.FP32,
        ir.MemRef(next_base, 0, 4096, span),
        None,
        ir.MemorySpace.Vec,
    )
    source = ir.Var("source", source_type, span)
    moved = ir.Var("moved", moved_type, span)
    next_value = ir.Var("next_value", next_type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(source, ir.Call(ir.Op("tile.load"), [input_var], source_type, span), span),
            ir.AssignStmt(
                moved,
                ir.Call(
                    ir.Op("tile.move"),
                    [source],
                    {"target_memory": ir.MemorySpace.Vec},
                    moved_type,
                    span,
                ),
                span,
            ),
            ir.AssignStmt(next_value, ir.Call(ir.Op("tile.full"), [], next_type, span), span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    program = ir.Program(
        [ir.Function("main", [input_var], [], body, span, type=ir.FunctionType.InCore)],
        "same_base_different_offset",
        span,
    )

    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP):
        allocated = passes.allocate_memory_addr()(program)
    ranges = _tile_ranges(allocated)

    assert ranges["source"][0] != ranges["moved"][0]
    assert _overlap(ranges["source"], ranges["next_value"])


def test_dsa_rp_tile_create_is_not_an_execution_write():
    """A storage declaration alone does not create a WAW penalty."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            _declaration = pl.tile.create([64, 64], pl.FP32, target_memory=pl.Mem.Vec)
            actual = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(actual, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert ranges["_declaration"] == ranges["actual"]


def test_dsa_rp_tile_assemble_is_an_execution_write():
    """A whole-allocation assemble write is visible to hazard recognition."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            source_input: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            target = pl.tile.create([64, 64], pl.FP32, target_memory=pl.Mem.Vec)
            source = pl.load(source_input, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            _assembled = pl.tile.assemble(target, source, [0, 0])
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["_assembled"], ranges["next_value"])


def test_dsa_rp_partial_assemble_poisons_touched_allocations():
    """A subrange assemble stays Unknown and cannot produce a whole-buffer edge."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            target_input: pl.Tensor[[64, 64], pl.FP32],
            source_input: pl.Tensor[[32, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            target = pl.load(target_input, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            source = pl.load(source_input, [0, 0], [32, 64], target_memory=pl.Mem.Vec)
            _assembled = pl.tile.assemble(target, source, [0, 0])
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert _overlap(ranges["_assembled"], ranges["next_value"])


def test_dsa_rp_unknown_tuple_operation_poisons_all_results():
    """Unknown tuple operations conservatively poison every physical result."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            source_tensor: pl.Tensor[[16, 64], pl.INT32],
            next_destination_tensor: pl.Tensor[[16, 8], pl.INT32],
            next_count_tensor: pl.Tensor[[1, 16], pl.INT32],
            kvalue: pl.Scalar[pl.INT32],
            output: pl.Tensor[[16, 8], pl.INT32],
        ) -> pl.Tensor[[16, 8], pl.INT32]:
            source = pl.load(source_tensor, [0, 0], [16, 64], target_memory=pl.Mem.Vec)
            temporary = pl.tile.create([16, 64], pl.UINT8, target_memory=pl.Mem.Vec)
            destination, _count = pl.tile.gather_compare(
                source,
                kvalue,
                temporary,
                cmp_mode="eq",
                out_cols=8,
            )
            _next_count = pl.load(next_count_tensor, [0, 0], [1, 16], target_memory=pl.Mem.Vec)
            next_destination = pl.load(next_destination_tensor, [0, 0], [16, 8], target_memory=pl.Mem.Vec)
            return pl.store(next_destination, [0, 0], output)

    recognized = _recognized_pairs(Before)
    assert frozenset(("destination", "next_destination")) not in recognized
    assert frozenset(("_count", "_next_count")) not in recognized


def test_dsa_rp_unknown_mutating_operand_stays_unpenalized():
    """A destination-passing write remains Unknown until its effects are modeled."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            value: pl.Scalar[pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            _written = pl.tile.write(prior, [0, 0], value)
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert _overlap(ranges["prior"], ranges["next_value"])


def test_dsa_rp_scalar_subrange_read_stays_unpenalized():
    """A scalar tile.read does not masquerade as a whole-allocation terminal read."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            _scalar = pl.tile.read(prior, [0, 0])
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert _overlap(ranges["prior"], ranges["next_value"])


def test_dsa_rp_recognizes_nested_cross_pipe_handoff(ascend_backend):
    """Nested loop accesses participate in distance-zero recognition."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            for _i in pl.range(1):
                prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
                _consumed = pl.add(prior, prior)
                _next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            result = pl.tile.full([64, 64], dtype=pl.FP32, value=2.0)
            return pl.store(result, [0, 0], output)

    assert frozenset(("prior", "_next_value")) in _recognized_pairs(Before)
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["prior"], ranges["_next_value"])


def test_dsa_rp_recognizes_true_distance_one_handoff():
    """A later outbound read can gate an earlier Vector write next iteration."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            scratch: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            for _i in pl.range(2):
                _first = pl.tile.full([64, 64], dtype=pl.FP32, value=0.0)
                second = pl.tile.full([64, 64], dtype=pl.FP32, value=1.0)
                _stored = pl.store(second, [0, 0], scratch)
            result = pl.tile.full([64, 64], dtype=pl.FP32, value=2.0)
            return pl.store(result, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["_first"], ranges["second"])


def test_dsa_rp_rejects_mutually_exclusive_branch_handoff():
    """Opposite branches are not a realizable distance-zero handoff."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            condition: pl.Scalar[pl.INT64],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            if condition < 1:
                branch_prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
                _branch_consumed = pl.add(branch_prior, branch_prior)
            else:
                _branch_next = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            result = pl.tile.full([64, 64], dtype=pl.FP32, value=2.0)
            return pl.store(result, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert _overlap(ranges["branch_prior"], ranges["_branch_next"])


def test_dsa_rp_keeps_logically_ordered_cross_pipe_handoff():
    """SSA reachability does not prove completion order across pipes."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            scratch: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            updated = pl.store(prior, [0, 0], scratch)
            next_value = pl.load(updated, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["prior"], ranges["next_value"])


def test_no_access_operation_has_no_exact_pipe(ascend_backend):
    """Declarations cannot acquire a synthetic execution resource."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            output: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            created = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            return pl.store(created, [0, 0], output)

    call = _find_call(Before, "tile.create")
    assert testing.get_execution_memory_access_evidence("tile.create") == "no_access"
    assert testing.try_infer_pipe(call) is None


@pytest.mark.parametrize("op_name", ["tile.matmul_acc", "tile.gemv_acc", "tile.batch_matmul_acc"])
def test_accumulate_ops_declare_functional_access(op_name):
    """Accumulator ops expose their complete read/write contract to the recognizer."""
    assert testing.get_execution_memory_access_evidence(op_name) == "functional"


def test_matmul_acc_functional_access_does_not_poison_allocation(ascend_backend):
    """A full-access accumulator op preserves later edge recognition for its Acc allocation."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            target_input: pl.Tensor[[16, 16], pl.FP32],
            lhs_input: pl.Tensor[[16, 16], pl.BF16],
            rhs_input: pl.Tensor[[16, 16], pl.BF16],
            output: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            target = pl.load(target_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            lhs_l1 = pl.load(lhs_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            rhs_l1 = pl.load(rhs_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            lhs_l0 = pl.tile.move(lhs_l1, target_memory=pl.Mem.Left)
            rhs_l0 = pl.tile.move(rhs_l1, target_memory=pl.Mem.Right)
            acc = pl.matmul(lhs_l0, rhs_l0)
            acc_updated = pl.matmul_acc(acc, lhs_l0, rhs_l0)
            _assembled = pl.tile.assemble(target, acc_updated, [0, 0])
            later_acc = pl.matmul(lhs_l0, rhs_l0)
            return pl.store(later_acc, [0, 0], output)

    assert frozenset(("acc", "later_acc")) in _recognized_pairs(Before)


def test_dsa_rp_places_on_chip_tile_parameter(ascend_backend):
    """Tile parameters are entry-live allocation identities, not body definitions."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            tile_param: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec],
        ) -> pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec]:
            result = pl.add(tile_param, tile_param)
            return result

    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert set(ranges) >= {"tile_param", "result"}
    assert ranges["tile_param"][0] >= 0


def test_dsa_rp_recognizes_l1_to_l0_route(ascend_backend):
    """A Mat buffer drained to L0 is separated from a later inbound load."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            input_a: pl.Tensor[[16, 16], pl.BF16],
            input_b: pl.Tensor[[16, 16], pl.BF16],
            rhs_input: pl.Tensor[[16, 16], pl.BF16],
            output: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            prior_l0 = pl.tile.move(prior, target_memory=pl.Mem.Left)
            _later = pl.load(input_b, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            rhs_l1 = pl.load(rhs_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            rhs_l0 = pl.tile.move(rhs_l1, target_memory=pl.Mem.Right)
            result = pl.matmul(prior_l0, rhs_l0)
            return pl.store(result, [0, 0], output)

    assert frozenset(("prior", "_later")) in _recognized_pairs(Before)
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["prior"], ranges["_later"])


def test_dsa_rp_recognizes_l0_to_l1_route(ascend_backend):
    """A full Acc-to-Mat assemble is separated from a later inbound load."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            target_input: pl.Tensor[[16, 16], pl.FP32],
            lhs_input: pl.Tensor[[16, 16], pl.BF16],
            rhs_input: pl.Tensor[[16, 16], pl.BF16],
            later_input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            target = pl.load(target_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            lhs_l1 = pl.load(lhs_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            rhs_l1 = pl.load(rhs_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            lhs_l0 = pl.tile.move(lhs_l1, target_memory=pl.Mem.Left)
            rhs_l0 = pl.tile.move(rhs_l1, target_memory=pl.Mem.Right)
            source = pl.matmul(lhs_l0, rhs_l0)
            _assembled = pl.tile.assemble(target, source, [0, 0])
            _later = pl.load(later_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            return pl.store(source, [0, 0], output)

    # Assemble writes the target allocation in place, so the edge is keyed by
    # that allocation's representative rather than the SSA result name.
    assert frozenset(("target", "_later")) in _recognized_pairs(Before)
    ranges = _tile_ranges(_plan_with_dsa_rp(Before))
    assert not _overlap(ranges["_assembled"], ranges["_later"])


@pytest.mark.parametrize(
    ("ascend_backend", "expect_edge"),
    [(BackendType.Ascend910B, False), (BackendType.Ascend950, True)],
    indirect=["ascend_backend"],
)
def test_dsa_rp_uses_backend_memory_graph_for_acc_to_vec_route(ascend_backend, expect_edge):
    """Acc-to-Vec is an A5 FIX route; A2/A3 must conservatively skip it."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            lhs_input: pl.Tensor[[16, 16], pl.BF16],
            rhs_input: pl.Tensor[[16, 16], pl.BF16],
            later_input: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            lhs_l1 = pl.load(lhs_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            rhs_l1 = pl.load(rhs_input, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            lhs_l0 = pl.tile.move(lhs_l1, target_memory=pl.Mem.Left)
            rhs_l0 = pl.tile.move(rhs_l1, target_memory=pl.Mem.Right)
            product = pl.matmul(lhs_l0, rhs_l0)
            _moved = pl.tile.move(product, target_memory=pl.Mem.Vec)
            _consumed = pl.add(_moved, _moved)
            later = pl.load(later_input, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            return pl.store(later, [0, 0], output)

    pair = frozenset(("_moved", "_consumed"))
    assert (pair in _recognized_pairs(Before)) is expect_edge


@pytest.mark.parametrize("consume_view", [False, True])
def test_dsa_rp_reinterpret_view_is_metadata_only(consume_view):
    """The view adds no access; an actual consumer still records the aliased base."""

    @pl.program
    class ConsumeView:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            viewed: pl.Tile[[64, 64], pl.INT32] = pl.tile.reinterpret_view(prior, dtype=pl.INT32)
            _consumed = pl.add(viewed, viewed)
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    @pl.program
    class IgnoreView:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            prior = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            _viewed: pl.Tile[[64, 64], pl.INT32] = pl.tile.reinterpret_view(prior, dtype=pl.INT32)
            next_value = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(next_value, [0, 0], output)

    program = ConsumeView if consume_view else IgnoreView
    ranges = _tile_ranges(_plan_with_dsa_rp(program))
    assert ("_consumed" in ranges) is consume_view
    assert _overlap(ranges["prior"], ranges["next_value"]) is not consume_view


def test_dsa_rp_unit_edges_and_placement_are_deterministic():
    """Equivalent recognized relations receive equal priority deterministically."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            input_c: pl.Tensor[[64, 64], pl.FP32],
            input_d: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Tensor[[64, 64], pl.FP32],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            prior_a = pl.load(input_a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            _used_a = pl.add(prior_a, prior_a)
            prior_b = pl.load(input_b, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            _used_b = pl.add(prior_b, prior_b)
            _later_a = pl.load(input_c, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            later_b = pl.load(input_d, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
            return pl.store(later_b, [0, 0], output)

    first = _plan_with_dsa_rp(Before)
    second = _plan_with_dsa_rp(Before)
    ir.assert_structural_equal(first, second)

    ranges = _tile_ranges(first)
    for prior in ("prior_a", "prior_b"):
        for later in ("_later_a", "later_b"):
            assert not _overlap(ranges[prior], ranges[later])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
