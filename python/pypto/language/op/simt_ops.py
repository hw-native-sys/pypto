# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scalar operations inside an explicit SIMT execution region."""

from collections.abc import Sequence
from typing import Any

from pypto.ir.op import simt as _simt
from pypto.language.typing import Scalar, Tensor, Tile
from pypto.pypto_core import ir

Axis = ir.SimtAxis


def _unwrap(value: Scalar | Tensor | Tile | ir.Expr) -> ir.Expr:
    return value if isinstance(value, ir.Expr) else value.unwrap()


def thread_idx(axis: ir.SimtAxis) -> Scalar:
    """Thread coordinate along the required typed axis."""
    return Scalar(expr=_simt.thread_idx(axis))


def block_idx(axis: ir.SimtAxis) -> Scalar:
    """Block coordinate along the required typed axis."""
    return Scalar(expr=_simt.block_idx(axis))


def block_dim(axis: ir.SimtAxis) -> Scalar:
    """Thread-block extent along the required typed axis."""
    return Scalar(expr=_simt.block_dim(axis))


def grid_dim(axis: ir.SimtAxis) -> Scalar:
    """Grid extent along the required typed axis."""
    return Scalar(expr=_simt.grid_dim(axis))


def lane_id() -> Scalar:
    """Lane identifier within the current warp."""
    return Scalar(expr=_simt.lane_id())


def load(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    *,
    l1cache: str | None = None,
    l2cache: str | None = None,
) -> Scalar:
    """Read a scalar from existing GM or Vec/UB storage."""
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return Scalar(expr=_simt.load(_unwrap(storage), coordinates, l1cache=l1cache, l2cache=l2cache))


def store(
    storage: Tensor | Tile | ir.Expr,
    indices: Sequence[int | Scalar | ir.Expr],
    value: Scalar | ir.Expr,
    *,
    l1cache: str | None = None,
    l2cache: str | None = None,
) -> ir.Call:
    """Write a scalar; untouched elements retain their pre-region contents."""
    coordinates = [index if isinstance(index, int) else _unwrap(index) for index in indices]
    return _simt.store(_unwrap(storage), coordinates, _unwrap(value), l1cache=l1cache, l2cache=l2cache)


def launch(
    helper: object,
    *args: object,
    block_dims: tuple[int, int, int],
    result_param_indices: tuple[int, ...] = (),
) -> Any:
    """Explicit canonical helper launch, interpreted by the DSL parser."""
    raise RuntimeError("pl.simt.launch is only valid inside a parsed PyPTO function")
