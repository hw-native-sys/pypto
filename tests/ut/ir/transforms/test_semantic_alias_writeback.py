# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Required storage aliases for every registered output-reuses-input operator.

Each call is type-deduced normally, then its result descriptor models stale
post-lowering storage. The expected program follows the designated destination
through a rename and a view, retaining the result's independently inferred
layout, padding and valid extent. This is semantic aliasing, before lifetime
reuse or address placement.
"""

from math import prod

import pytest
from pypto import DataType, ir, passes

from .storage_test_utils import without_call_memrefs

_WRITEBACK_CASES = [
    pytest.param("tile.matmul_acc", "matmul", False, id="matmul_acc"),
    pytest.param("tile.gemv_acc", "gemv", False, id="gemv_acc"),
    pytest.param("tile.batch_matmul_acc", "batch_matmul", False, id="batch_matmul_acc"),
    pytest.param("tile.matmul_mx_acc", "matmul_mx", False, id="matmul_mx_acc"),
    pytest.param("tile.fillpad_inplace", "fillpad", False, id="fillpad_inplace"),
    pytest.param("tile.tget_scale_addr", "scale_addr", False, id="tget_scale_addr"),
    pytest.param("tile.gather_row", "gather", False, id="gather_row"),
    pytest.param("tile.scatter", "scatter", False, id="scatter"),
    pytest.param("tile.scatter_mask", "scatter_mask", False, id="scatter_mask"),
    pytest.param("tile.scatter_update", "scatter_update", False, id="scatter_update"),
    pytest.param("tile.set_validshape", "tile_validshape", False, id="tile_set_validshape"),
    pytest.param("tile.store", "store", False, id="store"),
    pytest.param("tile.mscatter", "mscatter", False, id="mscatter"),
    pytest.param("tile.store", "store", True, id="store_distributed"),
    pytest.param("tile.mscatter", "mscatter", True, id="mscatter_distributed"),
]

# These tensor registrations are audited as isolated storage contracts. The
# normal device pipeline converts tensor.set_validshape to tile.set_validshape;
# tensor.assemble selects tile.assemble/tile.store for tile operands, while its
# tensor-to-tensor fallback is rejected by IncoreTileOps. Neither unlowered
# TensorOp is stage-valid input to MaterializeSemanticAliases.
_TENSOR_WRITEBACK_CONTRACT_CASES = [
    pytest.param("tensor.assemble", "assemble", False, id="tensor_assemble"),
    pytest.param("tensor.set_validshape", "tensor_validshape", False, id="tensor_set_validshape"),
    pytest.param("tensor.assemble", "assemble", True, id="tensor_assemble_distributed"),
]


@pytest.fixture(autouse=True)
def legacy_storage_context(pass_verification_instruments):
    """Exercise the common storage normalizer without Buffer IR isolation."""
    with passes.PassContext(pass_verification_instruments, enable_buffer_ir=False):
        yield


def _with_memref(type_: ir.Type, memref: ir.MemRef, memory_space: ir.MemorySpace | None = None) -> ir.Type:
    if isinstance(type_, ir.TileType):
        return ir.TileType(type_.shape, type_.dtype, memref, type_.tile_view, memory_space)
    if isinstance(type_, ir.DistributedTensorType):
        return ir.DistributedTensorType(
            type_.shape, type_.dtype, memref, type_.tensor_view, type_.window_buffer
        )
    assert isinstance(type_, ir.TensorType)
    return ir.TensorType(type_.shape, type_.dtype, memref, type_.tensor_view)


def _writeback_call(
    op_name: str, kind: str, distributed: bool
) -> tuple[ir.Call, ir.Var, list[ir.Var], ir.MemRef]:
    """Construct valid operand geometry without relying on the normalizer's op predicates."""
    span = ir.Span.unknown()
    params: list[ir.Var] = []

    def memref(name: str, size: int) -> ir.MemRef:
        base = ir.Var(f"{name}_base", ir.PtrType(), span)
        params.append(base)
        return ir.MemRef(base, 256, size, span)

    def tile(
        name: str,
        shape: list[int],
        dtype: DataType = DataType.FP32,
        memory: ir.MemorySpace = ir.MemorySpace.Vec,
        view: ir.TileView | None = None,
    ) -> ir.Var:
        element_bytes = {
            DataType.FP8E4M3FN: 1,
            DataType.FP8E8M0: 1,
            DataType.FP16: 2,
            DataType.BF16: 2,
        }.get(dtype, 4)
        type_ = ir.TileType(shape, dtype, memref(name, prod(shape) * element_bytes), view, memory)
        var = ir.Var(name, type_, span)
        params.append(var)
        return var

    def tensor(name: str, shape: list[int], *, partial: bool = False, window: bool = False) -> ir.Var:
        view = ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[0, 0]) if partial else None
        type_class = ir.DistributedTensorType if window else ir.TensorType
        var = ir.Var(name, type_class(shape, DataType.FP32, memref(name, prod(shape) * 4), view), span)
        params.append(var)
        return var

    def dims(values: list[int]) -> ir.MakeTuple:
        return ir.MakeTuple([ir.ConstInt(value, DataType.INDEX, span) for value in values], span)

    args: list[ir.Expr]
    kwargs: dict[str, int | ir.PadValue] = {}
    if kind in {"matmul", "gemv", "batch_matmul"}:
        rows = 1 if kind == "gemv" else 16
        # The batch op's deducer accepts a batch-free 2D form. Actual ND batches
        # are expanded to tile.matmul_acc by FlattenTileNdTo2D before this pass.
        accumulator_view = (
            ir.TileView(
                valid_shape=[1, 32],
                blayout=ir.TileLayout.col_major,
                slayout=ir.TileLayout.row_major,
                fractal=1024,
            )
            if kind == "gemv"
            else None
        )
        destination = tile("destination", [16, 32], memory=ir.MemorySpace.Acc, view=accumulator_view)
        lhs = tile("lhs", [rows, 64], DataType.FP16, ir.MemorySpace.Left)
        rhs = tile("rhs", [64, 32], DataType.FP16, ir.MemorySpace.Right)
        args = [destination, lhs, rhs]
    elif kind == "matmul_mx":
        destination = tile("destination", [16, 32], memory=ir.MemorySpace.Acc)
        lhs = tile("lhs", [16, 64], DataType.FP8E4M3FN, ir.MemorySpace.Left)
        lhs_scale = tile("lhs_scale", [16, 2], DataType.FP8E8M0, ir.MemorySpace.LeftScale)
        rhs = tile("rhs", [64, 32], DataType.FP8E4M3FN, ir.MemorySpace.Right)
        rhs_scale = tile("rhs_scale", [2, 32], DataType.FP8E8M0, ir.MemorySpace.RightScale)
        args = [destination, lhs, lhs_scale, rhs, rhs_scale]
    elif kind == "scale_addr":
        destination = tile("destination", [2, 32], DataType.FP8E8M0, ir.MemorySpace.RightScale)
        source = tile("source", [64, 32], DataType.FP8E4M3FN, ir.MemorySpace.Right)
        args = [destination, source]
    elif kind == "fillpad":
        destination = tile("destination", [16, 32], view=ir.TileView(valid_shape=[8, 32]))
        args = [destination]
        kwargs = {"pad_value": ir.PadValue.min}
    elif kind == "gather":
        destination = tile("destination", [16, 32], memory=ir.MemorySpace.Mat)
        source = tensor("source", [16, 32])
        args = [destination, source, dims([8, 0]), dims([0, 0]), dims([8, 32])]
    elif kind in {"scatter", "scatter_mask", "scatter_update"}:
        destination = tile("destination", [16, 32])
        source = tile("source", [16, 32])
        if kind == "scatter_mask":
            args = [destination, source]
            kwargs = {"mask_pattern": 7}
        else:
            index_shape = [16, 32] if kind == "scatter" else [16, 1]
            indices = tile("indices", index_shape, DataType.INT32)
            args = [destination, source, indices] if kind == "scatter" else [destination, indices, source]
    elif kind == "tile_validshape":
        destination = tile("destination", [16, 32])
        args = [destination, ir.ConstInt(8, DataType.INDEX, span), ir.ConstInt(32, DataType.INDEX, span)]
    elif kind == "tile_assemble":
        destination = tile("destination", [16, 32], view=ir.TileView(valid_shape=[8, 32]))
        source = tile("source", [8, 32])
        args = [destination, source, dims([8, 0])]
    elif kind in {"store", "mscatter"}:
        destination = tensor("destination", [16, 32], partial=kind == "store", window=distributed)
        source = tile("source", [8, 32])
        offsets = dims([0, 0]) if kind == "store" else tile("indices", [8, 32], DataType.INT32)
        args = [source, offsets, destination]
    elif kind == "assemble":
        destination = tensor("destination", [16, 32], partial=True, window=distributed)
        source = tensor("source", [8, 32])
        args = [destination, source, dims([0, 0])]
    else:
        assert kind == "tensor_validshape"
        destination = tensor("destination", [16, 32])
        args = [destination, ir.ConstInt(8, DataType.INDEX, span), ir.ConstInt(32, DataType.INDEX, span)]

    call = ir.create_op_call(op_name, args, kwargs, span)
    destination_type = destination.type
    assert isinstance(destination_type, ir.ShapedType) and destination_type.memref is not None
    stale = memref("stale", destination_type.memref.size_)
    return call, destination, params, stale


def _program(call: ir.Call, destination: ir.Var, params: list[ir.Var], storage: ir.MemRef) -> ir.Program:
    span = ir.Span.unknown()
    destination_type = destination.type
    memory = destination_type.memory_space if isinstance(destination_type, ir.TileType) else None
    result = ir.Var("result", _with_memref(call.type, storage, memory), span)
    renamed = ir.Var("renamed", result.type, span)
    assert isinstance(result.type, ir.ShapedType)
    shape = ir.MakeTuple(list(result.type.shape), span)
    if isinstance(result.type, ir.TileType):
        view_call = ir.create_op_call("tile.reshape", [renamed, shape], span)
    else:
        # A layout-only view preserves partial valid extents; specifying shape
        # requests a reinterpretation that requires a fully valid source.
        view_call = ir.op.tensor.view(renamed, layout=ir.TensorLayout.ND, span=span)
    viewed = ir.Var("viewed", _with_memref(view_call.type, storage, memory), span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(result, call, span),
            ir.AssignStmt(renamed, result, span),
            ir.AssignStmt(viewed, view_call, span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    function = ir.Function("kernel", params, [], body, span, type=ir.FunctionType.InCore)
    return ir.Program([function], "WritebackAliases", span)


def _assert_writeback_result_and_views_follow_destination(
    op_name: str, kind: str, distributed: bool, stale_result: bool
) -> None:
    call, destination, params, stale = _writeback_call(op_name, kind, distributed)
    destination_type = destination.type
    assert isinstance(destination_type, ir.ShapedType) and destination_type.memref is not None
    current = destination_type.memref
    Before = _program(call, destination, params, stale if stale_result else current)
    Expected = _program(call, destination, params, current)

    After = passes.materialize_semantic_aliases()(Before)

    ir.assert_structural_equal(without_call_memrefs(After), without_call_memrefs(Expected))
    ir.assert_structural_equal(passes.materialize_semantic_aliases()(After), After)
    if not stale_result:
        ir.assert_structural_equal(After, Before)


@pytest.mark.parametrize("op_name,kind,distributed", _WRITEBACK_CASES)
@pytest.mark.parametrize("stale_result", [True, False], ids=["stale", "canonical"])
def test_registered_writeback_result_and_views_follow_destination(
    op_name: str, kind: str, distributed: bool, stale_result: bool
):
    """Stage-valid writebacks retain required aliases under full property verification."""
    _assert_writeback_result_and_views_follow_destination(op_name, kind, distributed, stale_result)


@pytest.mark.parametrize("op_name,kind,distributed", _TENSOR_WRITEBACK_CONTRACT_CASES)
@pytest.mark.parametrize("stale_result", [True, False], ids=["stale", "canonical"])
def test_unlowered_tensor_writeback_storage_contract(
    op_name: str, kind: str, distributed: bool, stale_result: bool
):
    """Audit pre-conversion registry aliases separately from device-pipeline validity.

    These two TensorOp registrations are not valid IncoreTileOps-stage input.
    The isolated normalization contract still checks exact expected IR and
    print/parse structural roundtrips; pipeline cases above retain both property
    verification and roundtrips.
    """
    with passes.PassContext([ir.make_roundtrip_instrument()], enable_buffer_ir=False):
        _assert_writeback_result_and_views_follow_destination(op_name, kind, distributed, stale_result)


@pytest.mark.parametrize("stale_result", [True, False], ids=["stale", "canonical"])
def test_first_input_assemble_alias_retains_full_destination_window(stale_result: bool):
    """A mutating alias without a writeback slot preserves its expanded valid region."""
    _assert_writeback_result_and_views_follow_destination(
        "tile.assemble", "tile_assemble", False, stale_result
    )


def _tensor_loop_program(
    call: ir.Call,
    destination: ir.Var,
    params: list[ir.Var],
    storage: ir.MemRef,
    loop_kind: str,
    trip_count: int,
) -> ir.Program:
    """Model a destination rename corrected before binding a tensor loop carry."""
    span = ir.Span.unknown()
    renamed = ir.Var("renamed_destination", _with_memref(destination.type, storage), span)
    carried = ir.IterArg("carried", renamed.type, renamed, span)
    store = ir.create_op_call("tile.store", [call.args[0], call.args[1], carried], span)
    written = ir.Var("written", _with_memref(store.type, storage), span)
    returned = ir.Var("returned_destination", _with_memref(destination.type, storage), span)
    zero = ir.ConstInt(0, DataType.INDEX, span)
    one = ir.ConstInt(1, DataType.INDEX, span)
    stop = ir.ConstInt(trip_count, DataType.INDEX, span)
    loop: ir.Stmt
    if loop_kind == "for":
        index = ir.Var("index", ir.ScalarType(DataType.INDEX), span)
        body = ir.SeqStmts([ir.AssignStmt(written, store, span), ir.YieldStmt([written], span)], span)
        loop = ir.ForStmt(index, zero, stop, one, [carried], body, [returned], span)
    else:
        assert loop_kind in {"while", "false_while"}
        counter = ir.IterArg("counter", ir.ScalarType(DataType.INDEX), zero, span)
        advanced = ir.Var("advanced", counter.type, span)
        counter_result = ir.Var("counter_result", counter.type, span)
        body = ir.SeqStmts(
            [
                ir.AssignStmt(written, store, span),
                ir.AssignStmt(advanced, ir.Add(counter, one, DataType.INDEX, span), span),
                ir.YieldStmt([written, advanced], span),
            ],
            span,
        )
        loop = ir.WhileStmt(
            ir.ConstBool(False, span)
            if loop_kind == "false_while"
            else ir.Lt(counter, stop, DataType.BOOL, span),
            [carried, counter],
            body,
            [returned, counter_result],
            span,
        )
    observed = ir.Var("observed_destination", returned.type, span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(renamed, destination, span),
            loop,
            ir.AssignStmt(observed, returned, span),
            ir.ReturnStmt(span),
        ],
        span,
    )
    function = ir.Function("kernel", params, [], body, span, type=ir.FunctionType.InCore)
    return ir.Program([function], "TensorLoopAliases", span)


@pytest.mark.parametrize("distributed", [False, True], ids=["tensor", "distributed_tensor"])
@pytest.mark.parametrize(
    "loop_kind,trip_count",
    [("for", 0), ("for", 2), ("while", 0), ("while", 2), ("false_while", 0)],
    ids=["for_zero", "for_two", "while_zero", "while_two", "while_false"],
)
@pytest.mark.parametrize("buffer_ir", [False, True], ids=["legacy", "buffer_ir"])
def test_tensor_loop_bindings_follow_reanchored_initializer(
    distributed: bool,
    loop_kind: str,
    trip_count: int,
    buffer_ir: bool,
    pass_verification_instruments,
):
    """Tensor carry/return storage follows its initializer, including zero-trip loops.

    PTO codegen binds tensor-like carries and loop results directly to the
    initializer's mutable view. Correcting a bare initializer rename must update
    those boundary descriptors before a store follows the carried destination.
    This does not retarget independent yielded tensor allocations.
    """
    call, old_destination, params, stale = _writeback_call("tile.store", "store", distributed)
    old_type = old_destination.type
    assert isinstance(old_type, ir.TensorType) and old_type.memref is not None
    # The same partial extent is valid before and after the in-bounds store,
    # including when the loop executes zero times.
    view = ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[8, 32])
    if isinstance(old_type, ir.DistributedTensorType):
        current_type = ir.DistributedTensorType(
            old_type.shape, old_type.dtype, old_type.memref, view, old_type.window_buffer
        )
    else:
        current_type = ir.TensorType(old_type.shape, old_type.dtype, old_type.memref, view)
    destination = ir.Var(old_destination.name_hint, current_type, old_destination.span)
    params = [destination if param is old_destination else param for param in params]
    current = old_type.memref
    Before = _tensor_loop_program(call, destination, params, stale, loop_kind, trip_count)
    Expected = _tensor_loop_program(call, destination, params, current, loop_kind, trip_count)

    with passes.PassContext(pass_verification_instruments, enable_buffer_ir=buffer_ir):
        After = passes.materialize_semantic_aliases()(Before)
        ir.assert_structural_equal(without_call_memrefs(After), without_call_memrefs(Expected))
        ir.assert_structural_equal(passes.materialize_semantic_aliases()(After), After)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
