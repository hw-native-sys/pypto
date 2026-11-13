# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SSA pipeline completion operations before hardware synchronization lowering."""

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import Call, Expr, PipeType, Span

from ..utils import _get_span_or_capture


def _check_pipe(pipe: object, operation: str) -> None:
    """Reject Python integers before they lose the explicit pipeline identity."""
    if not isinstance(pipe, PipeType):
        raise TypeError(f"{operation} pipe must be a PipeType value, got {pipe!r}")


def _completion(name: str, pipe: PipeType, span: Span | None) -> Call:
    """Build a completion value whose execution owner is the enclosing function."""
    _check_pipe(pipe, name)
    return _ir.create_op_call(name, [], {"pipe": int(pipe)}, _get_span_or_capture(span, frame_offset=2))


def record(pipe: PipeType, *, span: Span | None = None) -> Call:
    """Capture completion of a pipeline's command prefix at this statement.

    The result is an immutable completion fact. Waiting does not consume it,
    so multiple consumers can depend on the same recorded prefix. The prefix
    includes preceding waits imported into the pipeline, excludes later work,
    and belongs to this execution of the enclosing specialized device function.

    Args:
        pipe: Pipeline whose preceding work must complete.
        span: Optional source location.

    Returns:
        A SyncTokenType SSA completion token without an allocated hardware ID.
    """
    return _completion("sync.record", pipe, span)


def completed(pipe: PipeType, *, span: Span | None = None) -> Call:
    """Create an already-completed token for an initial value or empty path.

    This token imposes no dependency on preceding work. It provides a typed
    initial value for loop carries and the empty side of a branch merge.

    Args:
        pipe: Pipeline associated with the completion token.
        span: Optional source location.

    Returns:
        An immediately completed SyncTokenType SSA value.
    """
    return _completion("sync.completed", pipe, span)


def join(first: Expr, second: Expr, span: Span | None = None) -> Call:
    """Complete after both same-pipeline completion tokens have completed.

    Join is a pure, non-consuming conjunction. Its operands must describe the
    same source pipeline and function execution. Join captures no new commands
    and allocates no physical synchronization resource.

    Args:
        first: First SSA completion token.
        second: Second SSA completion token with the same source pipeline.
        span: Optional source location.

    Returns:
        A SyncTokenType SSA value that requires both input completions.
    """
    return _ir.create_op_call("sync.join", [first, second], {}, _get_span_or_capture(span))


def _index_bound(value: int | Expr, name: str, span: Span) -> Expr:
    """Normalize a byte bound without silently coercing bool or float values."""
    if isinstance(value, Expr):
        # The registered schema validates that expressions are INDEX scalars.
        return value
    if type(value) is not int:
        raise TypeError(f"sync.memory_overlaps {name} must be an int or INDEX expression, got {value!r}")
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError(f"sync.memory_overlaps {name} must fit a signed 64-bit INDEX value, got {value}")
    return _ir.ConstInt(value, DataType.INDEX, span)


def memory_overlaps(
    first: Expr,
    second: Expr,
    *,
    first_begin: int | Expr,
    first_end: int | Expr,
    second_begin: int | Expr,
    second_end: int | Expr,
    span: Span | None = None,
) -> Call:
    """Test whether two runtime storage byte intervals overlap.

    Each half-open interval is relative to its handle's actual
    runtime data pointer. Empty intervals never overlap. This pure predicate
    only inspects addresses; it neither accesses payload nor waits for work.
    It is a logical relation, not an allocation-bounds validator. LowerSync
    must compare addresses without arithmetic overflow or underflow. Bounds are
    INDEX SSA operands, must satisfy begin <= end at runtime, and must already
    be correctly evaluated: this predicate cannot recover a wrapped bound.

    Args:
        first: First Tensor, Tile, or Buffer runtime storage handle.
        second: Second handle in the same known memory space.
        first_begin: INDEX expression or int giving the first signed byte begin.
        first_end: INDEX expression or int giving the first signed byte end.
        second_begin: INDEX expression or int giving the second signed byte begin.
        second_end: INDEX expression or int giving the second signed byte end.
        span: Optional source location.

    Returns:
        A ScalarType(BOOL) logical predicate for conditional synchronization.
    """
    actual_span = _get_span_or_capture(span)
    bounds = (
        ("first_begin", first_begin),
        ("first_end", first_end),
        ("second_begin", second_begin),
        ("second_end", second_end),
    )
    operands = [first, second, *(_index_bound(value, name, actual_span) for name, value in bounds)]
    return _ir.create_op_call("sync.memory_overlaps", operands, {}, actual_span)


def wait(token: Expr, dst_pipe: PipeType, span: Span | None = None) -> Call:
    """Order subsequent destination-pipeline work after a completion token.

    Waiting is non-consuming: the same token can be waited on repeatedly or
    by multiple pipelines, including its source pipeline.

    Args:
        token: An SSA pipeline completion token.
        dst_pipe: Pipeline whose subsequent work must wait for this completion.
        span: Optional source location.

    Returns:
        A VoidType call executed as an EvalStmt.
    """
    _check_pipe(dst_pipe, "sync.wait dst_pipe")
    return _ir.create_op_call("sync.wait", [token], {"dst_pipe": int(dst_pipe)}, _get_span_or_capture(span))


def vf_barrier(span: Span | None = None) -> Call:
    """Complete preceding VF work before subsequent VF instructions.

    This operation is legal only inside an AIV function's register scope. It
    does not replace the outer pipeline's entry or completion dependencies.

    Args:
        span: Optional source location.

    Returns:
        A VoidType logical barrier call; hardware lowering is deferred.
    """
    return _ir.create_op_call("sync.vf_barrier", [], {}, _get_span_or_capture(span))


__all__ = ["completed", "join", "memory_overlaps", "record", "vf_barrier", "wait"]
