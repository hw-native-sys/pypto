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

A unified namespace (``pl.op.add``, ``pl.op.exp``, ...) auto-dispatches
between tensor and block paths based on the input type (Tensor vs Tile).
The explicit ``pl.op.tensor.*`` and ``pl.op.block.*`` namespaces remain
available for cases where the caller wants to be explicit.
"""

from . import block_ops as block
from . import tensor_ops as tensor

# Promoted block-only ops (accessible as pl.op.load, etc.)
from .block_ops import (
    abs,
    cmp,
    cmps,
    col_expand,
    col_expand_div,
    col_expand_mul,
    col_expand_sub,
    expands,
    l0c_store,
    load,
    log,
    matmul_acc,
    max,
    min,
    minimum,
    move,
    neg,
    recip,
    relu,
    row_expand_add,
    row_expand_div,
    row_expand_mul,
    row_expand_sub,
    row_min,
    rsqrt,
    sqrt,
    store,
    sum,
)

# Promoted tensor-only ops (accessible as pl.op.create, etc.)
from .tensor_ops import assemble, create

# Unified dispatch (overlapping ops)
from .unified_ops import (
    add,
    cast,
    create_tile,
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
    "neg",
    "sqrt",
    "rsqrt",
    "recip",
    "log",
    "abs",
    "relu",
    "matmul_acc",
    "minimum",
    "cmp",
    "cmps",
    "row_min",
    "row_expand_add",
    "row_expand_sub",
    "row_expand_mul",
    "row_expand_div",
    "col_expand",
    "col_expand_mul",
    "col_expand_div",
    "col_expand_sub",
    "expands",
    # Promoted tensor-only
    "create",
    "assemble",
]
