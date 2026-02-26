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
- block: Block-level tile operations (TileType)

A unified namespace (``pl.add``, ``pl.exp``, ...) auto-dispatches
between tensor and block paths based on the input type (Tensor vs Tile).
The explicit ``pl.tensor.*`` and ``pl.block.*`` namespaces remain
available for cases where the caller wants to be explicit.
"""

from . import block_ops as block
from . import tensor_ops as tensor

# Promoted block-only ops (accessible as pl.load, etc.)
from .block_ops import (
    abs,
    addc,
    addsc,
    and_,
    ands,
    cmp,
    cmps,
    col_expand,
    col_expand_div,
    col_expand_mul,
    col_expand_sub,
    create_tile,
    expands,
    gemv,
    gemv_acc,
    gemv_bias,
    l0c_store,
    load,
    log,
    lrelu,
    matmul_acc,
    matmul_bias,
    max,
    maxs,
    min,
    minimum,
    mins,
    move,
    neg,
    not_,
    or_,
    ors,
    prelu,
    recip,
    relu,
    rem,
    rems,
    row_expand,
    row_expand_add,
    row_expand_div,
    row_expand_mul,
    row_expand_sub,
    row_min,
    rsqrt,
    sel,
    sels,
    shl,
    shls,
    shr,
    shrs,
    sqrt,
    store,
    subc,
    subsc,
    sum,
    vec_move,
    xor,
    xors,
)

# Promoted tensor-only ops (accessible as pl.create_tensor, etc.)
from .tensor_ops import assemble, create_tensor, dim

# Unified dispatch (overlapping ops)
from .unified_ops import (
    add,
    cast,
    div,
    exp,
    matmul,
    maximum,
    mul,
    reshape,
    row_max,
    row_sum,
    sub,
    transpose,
    view,
)

__all__ = [
    "block",
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
    "cast",
    "reshape",
    "transpose",
    "view",
    "matmul",
    "row_max",
    "row_sum",
    # Promoted block-only
    "create_tile",
    "load",
    "store",
    "l0c_store",
    "move",
    "vec_move",
    "neg",
    "sqrt",
    "rsqrt",
    "recip",
    "log",
    "abs",
    "relu",
    "matmul_acc",
    "matmul_bias",
    "gemv",
    "gemv_acc",
    "gemv_bias",
    "minimum",
    "cmp",
    "cmps",
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
    "expands",
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
]
