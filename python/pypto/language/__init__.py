# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO Language module - Type-safe DSL API for writing IR functions.

This module provides:
- function decorator for parsing DSL functions to IR
- Tensor type for tensor annotations and runtime wrapping
- Tile type for tile/block annotations and runtime wrapping
- Type-safe operation wrappers (tensor.*, block.*, and unified ops)
- DSL helpers (range, yield_)
- DataType constants

Typical usage:
    import pypto.language as pl

    @pl.function
    def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        result: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
        return result

    @pl.function
    def block_func(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        tile: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
        return pl.store(result, [0, 0], [64, 64], x)

    @pl.function
    def scalar_func(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
        return x
"""

# Import decorators and parsing functions from local parser module
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ForKind, FunctionType, MemorySpace

from . import parser
from .dsl_api import cond, const, incore, parallel, range, while_, yield_
from .op import block_ops as block
from .op import tensor_ops as tensor
from .op.block_ops import (
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
from .op.tensor_ops import assemble, create_tensor, dim
from .op.unified_ops import (
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
from .parser.decorator import function, program
from .parser.text_parser import loads, loads_program, parse, parse_program
from .typing import DynVar, IntLike, Scalar, Tensor, Tile, dynamic

# Re-export DataType constants for convenience
FP4 = DataType.FP4
FP8E4M3FN = DataType.FP8E4M3FN
FP8E5M2 = DataType.FP8E5M2
FP16 = DataType.FP16
FP32 = DataType.FP32
BF16 = DataType.BF16
HF4 = DataType.HF4
HF8 = DataType.HF8
INT4 = DataType.INT4
INT8 = DataType.INT8
INT16 = DataType.INT16
INT32 = DataType.INT32
INT64 = DataType.INT64
UINT4 = DataType.UINT4
UINT8 = DataType.UINT8
UINT16 = DataType.UINT16
UINT32 = DataType.UINT32
UINT64 = DataType.UINT64
BOOL = DataType.BOOL
INDEX = DataType.INDEX

__all__ = [
    "function",
    "program",
    "parse",
    "parser",
    "loads",
    "parse_program",
    "loads_program",
    "Tensor",
    "Tile",
    "Scalar",
    "DynVar",
    "IntLike",
    "dynamic",
    "const",
    "range",
    "parallel",
    "while_",
    "yield_",
    "cond",
    "incore",
    "block",
    "tensor",
    # Unified dispatch
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
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
    "min",
    "sum",
    "max",
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
    "FunctionType",
    "ForKind",
    "MemorySpace",
    "FP4",
    "FP8E4M3FN",
    "FP8E5M2",
    "FP16",
    "FP32",
    "BF16",
    "HF4",
    "HF8",
    "INT4",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT4",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "BOOL",
    "INDEX",
]
