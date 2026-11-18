# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit SSA completion dependencies before hardware synchronization lowering."""

from pypto.ir.op import sync_ops as _sync
from pypto.pypto_core.ir import Call, Expr, PipeType

from ..typing import IntLike, Scalar, SyncToken, Tensor, Tile


def record(pipe: PipeType) -> SyncToken:
    """Capture completion of a pipeline's command prefix at this statement.

    The result is an immutable completion fact. Waiting does not consume it,
    so multiple consumers can depend on the same recorded prefix. The prefix
    includes earlier waits on this pipeline and excludes commands submitted
    after this record. The enclosing AIC/AIV function supplies the core identity.

    Args:
        pipe: Pipeline whose preceding work must complete.

    Returns:
        A SyncToken SSA completion token without an allocated hardware ID.
    """
    return SyncToken(_sync.record(pipe))


def completed(pipe: PipeType) -> SyncToken:
    """Create an already-completed token for an initial value or empty path.

    This token imposes no dependency on preceding work. It provides a typed
    initial value for loop carries and the empty side of a branch merge.

    Args:
        pipe: Pipeline associated with the completion token.

    Returns:
        An immediately completed SyncToken SSA value.
    """
    return SyncToken(_sync.completed(pipe))


def join(first: SyncToken, second: SyncToken) -> SyncToken:
    """Complete after both same-pipeline completion tokens have completed.

    Join is a pure, non-consuming conjunction. Its operands must describe the
    same source pipeline and function execution. It captures no new commands
    and allocates no physical synchronization resource.

    Args:
        first: First SSA completion token.
        second: Second SSA completion token with the same source pipeline.

    Returns:
        A SyncToken SSA value that requires both input completions.
    """
    return SyncToken(_sync.join(first.unwrap(), second.unwrap()))


def memory_overlaps(
    first: Tensor | Tile | Expr,
    second: Tensor | Tile | Expr,
    *,
    first_begin: IntLike,
    first_end: IntLike,
    second_begin: IntLike,
    second_end: IntLike,
) -> Scalar:
    """Test whether two runtime storage byte intervals overlap.

    Each half-open interval is relative to its handle's actual
    runtime data pointer. Empty intervals never overlap. This pure predicate
    only inspects addresses; it neither accesses payload nor waits for work.
    It is a logical relation, not an allocation-bounds validator. LowerSync
    must compare addresses without arithmetic overflow or underflow. Bounds are
    INDEX values and must satisfy begin <= end without prior arithmetic overflow.

    Args:
        first: First Tensor, Tile, or raw Buffer expression.
        second: Second handle in the same known memory space.
        first_begin: INDEX scalar, expression, or int giving the first byte begin.
        first_end: INDEX scalar, expression, or int giving the first byte end.
        second_begin: INDEX scalar, expression, or int giving the second byte begin.
        second_end: INDEX scalar, expression, or int giving the second byte end.

    Returns:
        A Scalar BOOL logical predicate for conditional synchronization.
    """
    first_expr = first.unwrap() if isinstance(first, (Tensor, Tile)) else first
    second_expr = second.unwrap() if isinstance(second, (Tensor, Tile)) else second
    return Scalar(
        expr=_sync.memory_overlaps(
            first_expr,
            second_expr,
            first_begin=first_begin.unwrap() if isinstance(first_begin, Scalar) else first_begin,
            first_end=first_end.unwrap() if isinstance(first_end, Scalar) else first_end,
            second_begin=second_begin.unwrap() if isinstance(second_begin, Scalar) else second_begin,
            second_end=second_end.unwrap() if isinstance(second_end, Scalar) else second_end,
        )
    )


def wait(token: SyncToken, dst_pipe: PipeType) -> Call:
    """Order subsequent destination-pipeline work after a completion token.

    Waiting is non-consuming: the same token can be waited on repeatedly or
    by multiple pipelines, including its source pipeline.

    Args:
        token: An SSA pipeline completion token.
        dst_pipe: Pipeline whose subsequent work must wait for this completion.

    Returns:
        A VoidType call executed as a statement by the parser.
    """
    return _sync.wait(token.unwrap(), dst_pipe)


def vf_barrier() -> Call:
    """Complete preceding VF work before subsequent VF instructions.

    Legal only inside an AIV function's register scope. Outer entry and
    completion dependencies remain ordinary pipeline record/wait operations.

    Returns:
        A VoidType logical barrier call; hardware lowering is deferred.
    """
    return _sync.vf_barrier()


__all__ = ["completed", "join", "memory_overlaps", "record", "vf_barrier", "wait"]
