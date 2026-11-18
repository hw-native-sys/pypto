# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Logical SSA completion dependencies do not consume hardware-style flags."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.op import sync
from pypto.language.parser.diagnostics import ParserTypeError


@pytest.mark.parametrize("operation", [sync.record, sync.completed])
def test_completion_type_deduction(operation):
    call = operation(ir.PipeType.MTE2)
    assert isinstance(call.type, ir.SyncTokenType)
    assert call.type.pipe == ir.PipeType.MTE2
    assert call.kwargs == {"pipe": int(ir.PipeType.MTE2)}
    assert ir.get_op_output_arity(call.op.name) == 1


def test_wait_is_void_and_nonconsuming_for_multiple_consumers():
    recorded = sync.record(ir.PipeType.MTE2)
    token = ir.Var("token", recorded.type, ir.Span.unknown())
    for destination in (ir.PipeType.MTE2, ir.PipeType.V, ir.PipeType.MTE3, ir.PipeType.V):
        call = sync.wait(token, destination)
        assert isinstance(call.type, ir.VoidType)
        assert call.args[0].same_as(token)
    assert ir.get_op_output_arity("sync.wait") == 0
    with pytest.raises(ValueError, match="VoidType"):
        ir.AssignStmt(token, sync.wait(token, ir.PipeType.V), ir.Span.unknown())


def test_wait_validates_source_type_and_destination_pipe():
    span = ir.Span.unknown()
    with pytest.raises(ValueError, match="SyncTokenType"):
        sync.wait(ir.ConstInt(0, DataType.INDEX, span), ir.PipeType.V)
    token = sync.record(ir.PipeType.MTE2)
    with pytest.raises(ValueError, match="concrete PipeType"):
        sync.wait(token, ir.PipeType.ALL)
    with pytest.raises(ValueError, match="requires dst_pipe"):
        ir.create_op_call("sync.wait", [token], {}, span)
    with pytest.raises(ValueError, match="exactly one completion token"):
        ir.create_op_call("sync.wait", [token, token], {"dst_pipe": int(ir.PipeType.V)}, span)


@pytest.mark.parametrize("operation", [sync.record, sync.completed])
def test_completion_rejects_hardware_ids_and_invalid_protocol(operation):
    span = ir.Span.unknown()
    name = operation(ir.PipeType.V).op.name
    with pytest.raises(ValueError, match="Unknown kwarg"):
        ir.create_op_call(name, [], {"pipe": int(ir.PipeType.V), "event_id": 3}, span)
    with pytest.raises(ValueError, match="Unknown kwarg"):
        ir.create_op_call(name, [], {"pipe": int(ir.PipeType.V), "core_type": "aiv"}, span)
    with pytest.raises(ValueError, match="concrete PipeType"):
        operation(ir.PipeType.ALL)
    with pytest.raises(TypeError, match="PipeType"):
        operation(cast(ir.PipeType, 1))


@pytest.mark.parametrize("name", ["sync.record", "sync.completed"])
def test_low_level_completion_requires_pipeline_attribute(name):
    with pytest.raises(ValueError, match="requires only the pipe attribute"):
        ir.create_op_call(name, [], {}, ir.Span.unknown())


@pytest.mark.parametrize("name", ["sync.record", "sync.completed"])
@pytest.mark.parametrize("pipe", [-1, int(ir.PipeType.ALL), 99])
def test_low_level_completion_rejects_integer_outside_pipe_enum(name, pipe):
    with pytest.raises(ValueError, match="concrete PipeType"):
        ir.create_op_call(name, [], {"pipe": pipe}, ir.Span.unknown())


def _assert_roundtrips(program: ir.Program) -> None:
    """Check text/binary persistence and preserve the complete SSA use graph."""
    ir.assert_structural_equal(program, pl.parse_program(str(program)))
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))


def test_join_types_and_same_pipeline_contract():
    span = ir.Span.unknown()
    first = sync.record(ir.PipeType.MTE2)
    second = sync.record(ir.PipeType.MTE2)
    combined = sync.join(first, second)
    ir.assert_structural_equal(combined.type, first.type)
    assert combined.args[0].same_as(first)
    assert combined.args[1].same_as(second)
    with pytest.raises(ValueError, match="SyncTokenType"):
        sync.join(first, ir.ConstInt(0, DataType.INDEX, span))
    with pytest.raises(ValueError, match="same source pipe"):
        sync.join(first, sync.record(ir.PipeType.MTE3))
    with pytest.raises(ValueError, match="exactly two"):
        ir.create_op_call("sync.join", [first], {}, span)
    with pytest.raises(ValueError, match="no attributes"):
        ir.create_op_call("sync.join", [first, second], {"unused": 1}, span)


def test_join_roundtrips_and_preserves_shared_input_completions():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self):
            first = pl.sync.record(pl.PipeType.MTE2)
            second = pl.sync.record(pl.PipeType.MTE2)
            combined = pl.sync.join(first, second)
            pl.sync.wait(combined, pl.PipeType.V)
            pl.sync.wait(first, pl.PipeType.MTE3)

    _assert_roundtrips(Program)
    optimized = passes.simplify()(Program)
    assert str(optimized).count("pl.sync.join(") == 1
    assert str(optimized).count("pl.sync.wait(") == 2


def test_completion_program_roundtrips_and_preserves_fanout():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self):
            ready = pl.sync.record(pl.PipeType.MTE2)
            pl.sync.wait(ready, pl.PipeType.V)
            pl.sync.wait(ready, pl.PipeType.MTE3)
            pl.sync.wait(ready, pl.PipeType.MTE2)

    text = str(Program)
    assert "pl.SyncToken[" in text
    assert text.count("pl.sync.record(") == 1
    assert text.count("pl.sync.wait(") == 3
    assert "event_id" not in text
    _assert_roundtrips(Program)


def test_completion_record_and_wait_survive_ordinary_simplify():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self):
            incoming = pl.sync.record(pl.PipeType.V)
            first = pl.sync.record(pl.PipeType.MTE2)
            # The second capture includes an imported V completion; treating
            # both MTE2 records as identical would erase that dependency.
            pl.sync.wait(incoming, pl.PipeType.MTE2)
            second = pl.sync.record(pl.PipeType.MTE2)
            pl.sync.wait(first, pl.PipeType.V)
            pl.sync.wait(second, pl.PipeType.V)
            pl.sync.wait(first, pl.PipeType.MTE3)

    # Recorded completion points cannot be treated as pure identical calls.
    result = passes.simplify()(Program)
    text = str(result)
    assert text.count("pl.sync.record(") == 3
    assert text.count("pl.sync.wait(") == 4


def test_unused_record_does_not_allow_removing_observable_waits():
    span = ir.Span.unknown()
    record = sync.record(ir.PipeType.MTE2)
    unused = ir.Var("unused", record.type, span)
    completed = sync.completed(ir.PipeType.V)
    ready = ir.Var("ready", completed.type, span)
    remaining: list[ir.Stmt] = [
        ir.AssignStmt(ready, completed, span),
        ir.EvalStmt(sync.wait(ready, ir.PipeType.MTE3), span),
    ]
    before_body = ir.SeqStmts([ir.AssignStmt(unused, record, span), *remaining], span)
    expected_body = ir.SeqStmts(remaining, span)
    Before = ir.Program(
        [ir.Function("main", [], [], before_body, span, type=ir.FunctionType.AIV)], "Program", span
    )
    Expected = ir.Program(
        [ir.Function("main", [], [], expected_body, span, type=ir.FunctionType.AIV)], "Program", span
    )
    # An unused capture may be deleted, but Simplify is not required to run
    # general DCE. Both outcomes preserve the observable wait and its input.
    result = passes.simplify()(Before)
    assert ir.structural_equal(result, Before) or ir.structural_equal(result, Expected)


def test_completion_tokens_roundtrip_through_if_yields():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def main(self, flag: pl.Scalar[pl.BOOL]):
            if flag:
                produced = pl.sync.record(pl.PipeType.MTE2)
                selected = pl.yield_(produced)
            else:
                empty = pl.sync.completed(pl.PipeType.MTE2)
                selected = pl.yield_(empty)
            pl.sync.wait(selected, pl.PipeType.V)
            pl.sync.wait(selected, pl.PipeType.MTE3)

    _assert_roundtrips(Program)


@pytest.mark.parametrize("trips", [0, 3])
def test_completion_tokens_roundtrip_through_loop_carries(trips):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def main(self):
            initial = pl.sync.completed(pl.PipeType.V)
            for i, (previous,) in pl.range(trips, init_values=(initial,)):
                pl.sync.wait(previous, pl.PipeType.MTE2)
                current = pl.sync.record(pl.PipeType.V)
                final = pl.yield_(current)
            pl.sync.wait(final, pl.PipeType.MTE3)

    _assert_roundtrips(Program)


def test_completion_tokens_roundtrip_through_while_carries():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def main(self, trips: pl.Scalar[pl.INDEX]):
            initial = pl.sync.completed(pl.PipeType.V)
            for remaining, previous in pl.while_(init_values=(trips, initial)):
                pl.cond(remaining > 0)
                pl.sync.wait(previous, pl.PipeType.MTE2)
                current = pl.sync.record(pl.PipeType.V)
                next_remaining, final = pl.yield_(remaining - 1, current)
            pl.sync.wait(final, pl.PipeType.MTE3)

    _assert_roundtrips(Program)


@pytest.mark.parametrize("kind", [pl.FunctionType.AIC, pl.FunctionType.AIV])
def test_core_ownership_is_supplied_by_the_enclosing_function(kind):
    # Build the function directly: the decorator parser requires a literal
    # FunctionType member rather than the parametrized Python ``kind`` value.
    span = ir.Span.unknown()
    recorded = sync.record(ir.PipeType.MTE2)
    ready = ir.Var("ready", recorded.type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(ready, recorded, span), ir.EvalStmt(sync.wait(ready, ir.PipeType.MTE3), span)], span
    )
    Program = ir.Program([ir.Function("main", [], [], body, span, type=kind)], "Program", span)
    _assert_roundtrips(Program)
    assert "core_type" not in str(Program)


def test_sync_token_annotation_must_match_recorded_pipeline():
    with pytest.raises(ParserTypeError, match="disagrees with the source pipeline"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.AIV)
            def main(self):
                token: pl.SyncToken[pl.PipeType.MTE3] = pl.sync.record(pl.PipeType.MTE2)
                pl.sync.wait(token, pl.PipeType.V)


def test_vf_barrier_roundtrip_and_void_type():
    assert isinstance(sync.vf_barrier().type, ir.VoidType)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self):
            with pl.reg_scope():
                pl.sync.vf_barrier()

    _assert_roundtrips(Program)
    with pytest.raises(ValueError, match="takes no operands"):
        ir.create_op_call(
            "sync.vf_barrier", [ir.ConstInt(0, DataType.INDEX, ir.Span.unknown())], {}, ir.Span.unknown()
        )


@pytest.mark.parametrize("kind", ["tensor", "distributed_tensor", "tile", "buffer"])
@pytest.mark.parametrize("begin,end", [(0, 0), (-1024, 1024), (1 << 40, (1 << 40) + 64)])
def test_memory_overlaps_accepts_runtime_handles_and_signed_byte_bounds(kind, begin, end):
    span = ir.Span.unknown()
    if kind == "tensor":
        handle_type = ir.TensorType([16], DataType.FP32)
    elif kind == "distributed_tensor":
        handle_type = ir.DistributedTensorType([16], DataType.FP32)
    elif kind == "tile":
        handle_type = ir.TileType([16, 16], DataType.FP32, memory_space=ir.MemorySpace.Vec)
    else:
        handle_type = ir.BufferType([16, 16], DataType.FP32, ir.MemorySpace.Vec)
    first = ir.Var("first", handle_type, span)
    second = ir.Var("second", handle_type, span)
    call = sync.memory_overlaps(
        first, second, first_begin=begin, first_end=end, second_begin=-64, second_end=64
    )
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.BOOL
    assert call.kwargs == {}
    assert len(call.args) == 6
    assert call.args[0].same_as(first)
    assert call.args[1].same_as(second)
    for bound, expected in zip(call.args[2:], (begin, end, -64, 64), strict=True):
        assert isinstance(bound, ir.ConstInt)
        assert bound.dtype == DataType.INDEX
        assert bound.value == expected
    restored = ir.deserialize(ir.serialize(call))
    assert isinstance(restored, ir.Call)
    ir.assert_structural_equal(call, restored, enable_auto_mapping=True)
    assert restored.kwargs == call.kwargs
    # Rebuilding retains SSA operands; no interval state is hidden in attrs.
    rebuilt = ir.Call(restored.op, list(restored.args), restored.kwargs, restored.type, span)
    ir.assert_structural_equal(call, rebuilt, enable_auto_mapping=True)


@pytest.mark.parametrize("kind", ["scalar", "token", "memref", "array", "ptr"])
def test_memory_overlaps_rejects_non_address_values_and_memref_metadata(kind):
    span = ir.Span.unknown()
    types = {
        "scalar": ir.ScalarType(DataType.INT64),
        "token": ir.SyncTokenType(ir.PipeType.V),
        "array": ir.ArrayType(DataType.INT64, 4),
        "ptr": ir.PtrType(),
    }
    pointer = ir.Var("pointer", ir.PtrType(), span)
    invalid = ir.MemRef(pointer, 0, 64, span) if kind == "memref" else ir.Var("invalid", types[kind], span)
    storage = ir.Var("storage", ir.TensorType([16], DataType.FP32), span)
    with pytest.raises(ValueError, match="Tensor, Tile, or Buffer storage handles"):
        sync.memory_overlaps(invalid, storage, first_begin=0, first_end=64, second_begin=0, second_end=64)


def test_memory_overlaps_only_compares_handles_in_one_memory_space():
    span = ir.Span.unknown()
    tensor = ir.Var("gm", ir.TensorType([16], DataType.FP32), span)
    tile = ir.Var("ub", ir.TileType([16, 16], DataType.FP32, memory_space=ir.MemorySpace.Vec), span)
    buffer = ir.Var("ub_buffer", ir.BufferType([16, 16], DataType.FP32, ir.MemorySpace.Vec), span)
    with pytest.raises(ValueError, match="same known memory space"):
        sync.memory_overlaps(tensor, tile, first_begin=0, first_end=64, second_begin=0, second_end=64)
    predicate = sync.memory_overlaps(tile, buffer, first_begin=0, first_end=64, second_begin=0, second_end=64)
    assert predicate.type == ir.ScalarType(DataType.BOOL)


def test_memory_overlaps_rejects_unknown_tile_address_space():
    span = ir.Span.unknown()
    unknown = ir.Var("unknown", ir.TileType([16], DataType.FP32), span)
    with pytest.raises(ValueError, match="same known memory space"):
        sync.memory_overlaps(unknown, unknown, first_begin=0, first_end=64, second_begin=0, second_end=64)


def test_memory_overlaps_requires_well_formed_index_intervals():
    span = ir.Span.unknown()
    storage = ir.Var("storage", ir.TensorType([16], DataType.FP32), span)
    with pytest.raises(ValueError, match="begin <= end"):
        sync.memory_overlaps(storage, storage, first_begin=65, first_end=64, second_begin=-64, second_end=64)
    with pytest.raises(ValueError, match="signed 64-bit INDEX"):
        sync.memory_overlaps(
            storage, storage, first_begin=0, first_end=1 << 63, second_begin=-64, second_end=64
        )
    for invalid in (True, 1.5):
        with pytest.raises(TypeError, match="int or INDEX expression"):
            sync.memory_overlaps(
                storage, storage, first_begin=0, first_end=cast(int, invalid), second_begin=-64, second_end=64
            )
    operands = [storage, storage, *(ir.ConstInt(v, DataType.INDEX, span) for v in (0, 64, -64, 64))]
    for missing in range(6):
        with pytest.raises(ValueError, match="two storage handles and four INDEX bounds"):
            ir.create_op_call("sync.memory_overlaps", operands[:missing] + operands[missing + 1 :], {}, span)
    with pytest.raises(ValueError, match="interval bounds as operands, not attributes"):
        ir.create_op_call("sync.memory_overlaps", operands, {"first_begin": 0}, span)


@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.INT64, DataType.FP32, DataType.BOOL])
@pytest.mark.parametrize("bound_index", [0, 1, 2, 3])
def test_memory_overlaps_rejects_non_index_bound_expressions(dtype, bound_index):
    span = ir.Span.unknown()
    storage = ir.Var("storage", ir.TensorType([16], DataType.FP32), span)
    bounds: list[int | ir.Expr] = [0, 64, 0, 64]
    bounds[bound_index] = ir.Var("invalid", ir.ScalarType(dtype), span)
    with pytest.raises(ValueError, match=r"ScalarType\(INDEX\)"):
        sync.memory_overlaps(
            storage,
            storage,
            first_begin=bounds[0],
            first_end=bounds[1],
            second_begin=bounds[2],
            second_end=bounds[3],
        )


@pytest.mark.parametrize("bound", [-(1 << 63), (1 << 63) - 1])
def test_memory_overlaps_preserves_extreme_index_constants(bound):
    span = ir.Span.unknown()
    storage = ir.Var("storage", ir.TensorType([16], DataType.FP32), span)
    call = sync.memory_overlaps(
        storage, storage, first_begin=bound, first_end=bound, second_begin=bound, second_end=bound
    )
    assert all(isinstance(arg, ir.ConstInt) and arg.value == bound for arg in call.args[2:])
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)


@pytest.mark.parametrize("bound", [-(1 << 63) - 1, 1 << 63])
def test_memory_overlaps_rejects_python_integer_overflow(bound):
    span = ir.Span.unknown()
    storage = ir.Var("storage", ir.TensorType([16], DataType.FP32), span)
    with pytest.raises(ValueError, match="signed 64-bit INDEX"):
        sync.memory_overlaps(
            storage, storage, first_begin=bound, first_end=bound, second_begin=0, second_end=0
        )


def test_memory_overlaps_preserves_dynamic_bounds_as_operands():
    span = ir.Span.unknown()
    storage = ir.Var("storage", ir.TensorType([16], DataType.FP32), span)
    begin = ir.Var("begin", ir.ScalarType(DataType.INDEX), span)
    end = ir.Var("end", ir.ScalarType(DataType.INDEX), span)
    call = sync.memory_overlaps(
        storage, storage, first_begin=begin, first_end=end, second_begin=0, second_end=64
    )
    assert call.args[2].same_as(begin)
    assert call.args[3].same_as(end)
    assert call.kwargs == {}
    ir.assert_structural_equal(call, ir.deserialize(ir.serialize(call)), enable_auto_mapping=True)

    # Changing an interval operand changes the predicate even when the storage
    # handles and other bounds are shared; serialization must retain the use.
    other = sync.memory_overlaps(
        storage, storage, first_begin=begin, first_end=end, second_begin=0, second_end=65
    )
    assert not ir.structural_equal(call, other)
    assert ir.structural_hash(call) != ir.structural_hash(other)


def test_memory_overlaps_condition_and_token_fanout_roundtrip():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def main(self, first: pl.Tensor[[16], pl.FP32], second: pl.Tensor[[16], pl.FP32]):
            ready = pl.sync.record(pl.PipeType.MTE2)
            overlap = pl.sync.memory_overlaps(
                first,
                second,
                first_begin=-64,
                first_end=0,
                second_begin=1099511627776,
                second_end=1099511627840,
            )
            if overlap:
                pl.sync.wait(ready, pl.PipeType.MTE3)
            pl.sync.wait(ready, pl.PipeType.V)

    _assert_roundtrips(Program)
    text = str(Program)
    assert "first_begin=-64" in text
    assert "second_begin=1099511627776" in text


def test_memory_overlaps_dynamic_index_bounds_roundtrip():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def main(
            self,
            first: pl.Tensor[[64], pl.FP32],
            second: pl.Tensor[[64], pl.FP32],
            start: pl.Scalar[pl.INDEX],
            size: pl.Scalar[pl.INDEX],
        ):
            ready = pl.sync.record(pl.PipeType.MTE2)
            end = start + size
            overlap = pl.sync.memory_overlaps(
                first, second, first_begin=start, first_end=end, second_begin=0, second_end=size
            )
            if overlap:
                pl.sync.wait(ready, pl.PipeType.MTE3)
            # A later consumer retains the same completion even when the
            # conditional wait did not execute, or already executed once.
            pl.sync.wait(ready, pl.PipeType.V)

    _assert_roundtrips(Program)
    text = str(Program)
    assert "first_begin=start" in text
    assert "first_end=end" in text
    assert "second_end=size" in text


def test_memory_overlaps_dsl_accepts_scalar_wrappers_and_raw_buffer_handles():
    span = ir.Span.unknown()
    storage = ir.Var("storage", ir.BufferType([16], DataType.FP32, ir.MemorySpace.Vec), span)
    begin = ir.Var("begin", ir.ScalarType(DataType.INDEX), span)
    end = ir.Var("end", ir.ScalarType(DataType.INDEX), span)
    predicate = pl.sync.memory_overlaps(
        storage,
        storage,
        first_begin=pl.Scalar(expr=begin),
        first_end=end,
        second_begin=0,
        second_end=64,
    ).unwrap()
    assert isinstance(predicate, ir.Call)
    assert predicate.args[2].same_as(begin)
    assert predicate.args[3].same_as(end)
    assert predicate.type == ir.ScalarType(DataType.BOOL)


def test_memory_overlaps_caller_owned_tiles_roundtrip():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
        def main(
            self,
            first: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec],
            second: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec],
        ):
            ready = pl.sync.record(pl.PipeType.V)
            overlap = pl.sync.memory_overlaps(
                first, second, first_begin=-64, first_end=0, second_begin=0, second_end=64
            )
            if overlap:
                pl.sync.wait(ready, pl.PipeType.MTE3)

    _assert_roundtrips(Program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
