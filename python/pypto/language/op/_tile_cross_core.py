# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile block queries and cross-core transfers."""

from typing import TypeVar

from pypto.ir.op import tile_ops as _ir_ops
from pypto.pypto_core.ir import (
    Span,
)

from ..typing import Scalar, Tensor, Tile

# Constrained TypeVar for the split-axis reshape wrappers (aiv_shard / aic_gather):
# the operand is either a Tile (legacy @pl.program form) or a Tensor (@pl.jit /
# pl.spmd form), and the result is the SAME kind as the input.
#
# Constrained, NOT bound, and deliberately unlike the loop-carry TypeVars in
# dsl_api: a bound ``Tensor | Tile`` would propagate a DistributedTensor operand
# into the result type, but AIV/AIC split does not accept one. The deducer
# matches the operand with ``As<TensorType>``, which is precise-match and returns
# null for DistributedTensorType by design, so ``tensor.aiv_shard(dist)`` is
# rejected ("requires argument to be a (non-distributed) TensorType", see
# DeduceSplitReshapeTensor in src/ir/op/tile_ops/cross_core.cpp). Collapsing the
# subclass here keeps the annotation from advertising a result the op cannot
# produce; widening it needs both operation contracts updated first.
_SplitOperandT = TypeVar("_SplitOperandT", Tensor, Tile)


def aiv_shard(x: _SplitOperandT, span: Span | None = None) -> _SplitOperandT:
    """Bring a cube-produced operand onto the AIV lane (AIC -> AIV crossing).

    In a data-parallel region (``UP_DOWN`` / ``LEFT_RIGHT``) the crossing also
    **halves** the operand along the split axis, so each lane gets one half; in a
    task-parallel ``mode=NONE`` region there is no split axis, so it crosses and
    **preserves the shape**. Either way, writing it is how a C->V crossing into a
    region is named — the ``AivSplitValid`` verifier rejects an unnamed one.

    The split mode is **inherited** from the enclosing
    ``for aiv_id in pl.split_aiv(mode=...)`` scope — it is not passed here.
    This wrapper therefore only resolves inside a parsed kernel, where the
    parser intercepts the call and fills the inherited mode. Calling it eagerly
    (outside a parsed program) raises, since there is no scope to read the mode
    from.

    The operand may be a ``Tile`` (legacy ``@pl.program`` form -> ``tile.aiv_shard``)
    or a high-level ``Tensor`` (``@pl.jit`` / ``pl.spmd`` form -> ``tensor.aiv_shard``,
    lowered 1:1 to ``tile.aiv_shard`` at ConvertTensorToTileOps). The Tensor form
    is region-only. Distributed tensors are not supported.

    Args:
        x: Input operand (2D Tile or Tensor)
        span: Optional source span

    Returns:
        Operand of the same kind: the split axis halved in a data-parallel region,
        the shape unchanged in a ``mode=NONE`` one.
    """
    raise RuntimeError(
        "pl.aiv_shard must be used inside a 'for aiv_id in pl.split_aiv(...)' "
        "loop, which supplies the split mode"
    )


def aic_gather(x: _SplitOperandT, span: Span | None = None) -> _SplitOperandT:
    """Hand a vector-produced operand to the cube (AIV -> AIC crossing).

    Inverse of [`aiv_shard`][pypto.language.tile.aiv_shard]: in a data-parallel region it **rejoins** the two
    lanes' halves along the split axis, and in a task-parallel ``mode=NONE`` region
    it crosses and **preserves the shape**. It is how a V->C crossing out of a
    region is named; an unnamed one is rejected by the ``AivSplitValid`` verifier.

    Out of a ``mode=NONE`` region the two lanes share one destination slot with no
    per-lane offset and nothing arbitrates between them: both push, so when they
    hold different values the cube receives an **unspecified** one of the two.
    Guarding the *production* of the value does not help — lane 1 still reaches
    the push and still sends its own tile. Gather only a value both lanes agree
    on; if they must contribute different data, use a data-parallel region.

    Like [`aiv_shard`][pypto.language.tile.aiv_shard], the split mode is
    **inherited** from the enclosing ``for aiv_id in pl.split_aiv(mode=...)``
    scope and must not be passed here. Calling it eagerly (outside a parsed
    program) raises, since there is no scope to read the mode from.

    The operand may be a ``Tile`` (legacy ``@pl.program`` form -> ``tile.aic_gather``)
    or a high-level ``Tensor`` (``@pl.jit`` / ``pl.spmd`` form -> ``tensor.aic_gather``,
    lowered 1:1 to ``tile.aic_gather`` at ConvertTensorToTileOps). The Tensor form
    is region-only. Distributed tensors are not supported.

    Args:
        x: Input operand (2D Tile or Tensor)
        span: Optional source span

    Returns:
        Operand of the same kind: the split axis doubled in a data-parallel region,
        the shape unchanged in a ``mode=NONE`` one.
    """
    raise RuntimeError(
        "pl.aic_gather must be used inside a 'for aiv_id in pl.split_aiv(...)' "
        "loop, which supplies the split mode"
    )


def get_block_idx() -> Scalar:
    """Get the current block index.

    This operation returns the index of the current compute tile. It is typically
    used in tile-level programming to identify which block of data is being processed.

    Returns:
        Scalar wrapping the get_block_idx operation (INDEX type)

    Example:
        >>> block_idx = pl.tile.get_block_idx()
        >>> if block_idx < 10:
        >>>     # Process first 10 blocks differently
        >>>     ...
    """
    call_expr = _ir_ops.get_block_idx()
    return Scalar(expr=call_expr)


def get_subblock_idx() -> Scalar:
    """Get the current sub-block (vector core) index.

    Returns the index of the current vector core within a split execution.
    Core 0 returns 0, core 1 returns 1.

    Returns:
        Scalar wrapping the get_subblock_idx operation (INDEX type)
    """
    call_expr = _ir_ops.get_subblock_idx()
    return Scalar(expr=call_expr)


def get_block_num() -> Scalar:
    """Get the total number of blocks in the current SPMD task.

    This operation returns the total count of blocks dispatched for the current
    task. Used with get_block_idx() for SPMD work partitioning.

    Returns:
        Scalar wrapping the get_block_num operation (INDEX type)

    Example:
        >>> block_idx = pl.tile.get_block_idx()
        >>> block_num = pl.tile.get_block_num()
    """
    call_expr = _ir_ops.get_block_num()
    return Scalar(expr=call_expr)
