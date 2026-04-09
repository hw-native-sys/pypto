# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO Language operations module.

This module organizes language-level operations by category:
- tensor: High-level tensor operations (TensorType)
- tile: Tile-level operations (TileType)
- system: Hardware synchronization primitives

A unified namespace (``pl.add``, ``pl.exp``, ...) auto-dispatches
between tensor and tile paths based on the input type (Tensor vs Tile).
The explicit ``pl.tensor.*``, ``pl.tile.*``, and ``pl.system.*``
namespaces remain available for cases where the caller wants to be explicit.

"""

from . import system_ops as system
from . import tensor_ops as tensor
from . import tile_ops as tile

# Promoted tensor-only ops (accessible as pl.create_tensor, etc.)
from .tensor_ops import assemble, dim, expand_clone, scatter_update
from .tensor_ops import create as create_tensor

# Promoted tile-only ops (accessible as pl.load, etc.)
from .tile_ops import (
    abs,
    addc,
    addsc,
    and_,
    ands,
    cmp,
    cmps,
    gemv,
    gemv_acc,
    gemv_bias,
    load,
    log,
    lrelu,
    matmul_bias,
    max,
    maxs,
    min,
    minimum,
    mins,
    move,
    not_,
    or_,
    ors,
    prelu,
    relu,
    rem,
    rems,
    sel,
    sels,
    shl,
    shls,
    shr,
    shrs,
    store,
    subc,
    subsc,
    sum,
    xor,
    xors,
)
from .tile_ops import (
    create as create_tile,
)

# Unified dispatch (overlapping ops)
from .unified_ops import (
    add,
    cast,
    col_expand,
    col_expand_div,
    col_expand_mul,
    col_expand_sub,
    div,
    exp,
    expands,
    fillpad,
    matmul,
    matmul_acc,
    maximum,
    mul,
    neg,
    recip,
    reshape,
    row_expand,
    row_expand_add,
    row_expand_div,
    row_expand_mul,
    row_expand_sub,
    row_max,
    row_min,
    row_sum,
    rsqrt,
    slice,
    sqrt,
    sub,
    transpose,
)

__all__ = [
    "tile",
    "system",
    "tensor",
    # Unified dispatch
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "min",
    "sum",
    "max",
    "exp",
    "sqrt",
    "rsqrt",
    "cast",
    "reshape",
    "transpose",
    "slice",
    "matmul",
    "matmul_acc",
    "row_max",
    "row_sum",
    "row_min",
    "row_expand",
    "row_expand_add",
    "row_expand_sub",
    "row_expand_mul",
    "row_expand_div",
    "col_expand",
    "col_expand_mul",
    "col_expand_div",
    "col_expand_sub",
    "expand_clone",
    "expands",
    "neg",
    "recip",
    # Promoted tile-only
    "create_tile",
    "fillpad",
    "load",
    "store",
    "move",
    "log",
    "abs",
    "relu",
    "matmul_bias",
    "gemv",
    "gemv_acc",
    "gemv_bias",
    "minimum",
    "cmp",
    "cmps",
    "rem",
    "rems",
    "and_",
    "ands",
    "or_",
    "ors",
    "xor",
    "xors",
    "shl",
    "shls",
    "shr",
    "shrs",
    "maxs",
    "mins",
    "prelu",
    "not_",
    "addc",
    "subc",
    "addsc",
    "subsc",
    "lrelu",
    "sel",
    "sels",
    # Promoted tensor-only
    "create_tensor",
    "assemble",
    "dim",
    "scatter_update",
]
