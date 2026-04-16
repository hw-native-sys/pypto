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
- Tile type for tile annotations and runtime wrapping
- Type-safe operation wrappers (tensor.*, tile.*, system.*, and unified ops)
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
        return pl.store(result, [0, 0], x)

    @pl.function
    def scalar_func(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
        return x
"""

from pypto.ir import TensorView, TileView
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    ChunkConfig,
    ChunkPolicy,
    ForKind,
    FunctionType,
    Level,
    LoopOrigin,
    MemorySpace,
    PadValue,
    PipeType,
    PtrType,
    Role,
    SplitMode,
    TensorLayout,
    TileLayout,
)

from . import optimizations, parser
from .dsl_api import (
    at,
    auto_incore,
    chunked_loop_optimizer,
    cluster,
    cond,
    const,
    incore,
    parallel,
    range,
    spmd,
    static_assert,
    static_print,
    unroll,
    while_,
    yield_,
)
from .op import system_ops as system
from .op import tensor_ops as tensor
from .op import tile_ops as tile
from .op.system_ops import (
    AUTO,
    aic_initialize_pipe,
    aiv_initialize_pipe,
    import_peer_buffer,
    reserve_buffer,
    tfree_to_aic,
    tfree_to_aiv,
    tpop_from_aic,
    tpop_from_aiv,
    tpush_to_aic,
    tpush_to_aiv,
)
from .op.tensor_ops import assemble, create_tensor, dim, expand_clone, full, scatter_update
from .op.tile_ops import (
    MemRefType,
    abs,
    addc,
    addsc,
    and_,
    ands,
    cmp,
    cmps,
    create_tile,
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
from .op.unified_ops import (
    add,
    cast,
    col_expand,
    col_expand_div,
    col_expand_mul,
    col_expand_sub,
    concat,
    div,
    exp,
    expands,
    fillpad,
    matmul,
    matmul_acc,
    maximum,
    mul,
    neg,
    read,
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
    write,
)
from .optimizations import auto_chunk, split
from .parser.decorator import InlineFunction, function, inline, program
from .parser.text_parser import loads, loads_program, parse, parse_program
from .typing import DynVar, InOut, IntLike, MemRef, Out, Scalar, Tensor, Tile, Tuple, dynamic

# Short alias for MemorySpace (pl.Mem.Vec instead of pl.MemorySpace.Vec)
Mem = MemorySpace

# Alias for PtrType — used in printed IR as type annotation for alloc LHS
Ptr = PtrType

# Re-export TensorLayout constants for convenience
ND = TensorLayout.ND
DN = TensorLayout.DN
NZ = TensorLayout.NZ

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
    "inline",
    "program",
    "InlineFunction",
    "parse",
    "parser",
    "loads",
    "parse_program",
    "loads_program",
    "Tensor",
    "Tile",
    "Scalar",
    "Tuple",
    "DynVar",
    "InOut",
    "IntLike",
    "Out",
    "dynamic",
    "const",
    "range",
    "parallel",
    "unroll",
    "while_",
    "yield_",
    "cond",
    "static_print",
    "static_assert",
    "at",
    "incore",
    "auto_incore",
    "cluster",
    "spmd",
    "chunked_loop_optimizer",
    "optimizations",
    "split",
    "auto_chunk",
    "tile",
    "system",
    "tensor",
    # Unified dispatch
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "exp",
    "cast",
    "concat",
    "reshape",
    "transpose",
    "slice",
    "matmul",
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
    "read",
    "write",
    # Promoted tile-only
    "create_tile",
    "fillpad",
    "load",
    "store",
    "move",
    "sqrt",
    "rsqrt",
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
    # Promoted system ops (cross-core)
    "AUTO",
    "tpush_to_aiv",
    "tpush_to_aic",
    "tpop_from_aic",
    "tpop_from_aiv",
    "aic_initialize_pipe",
    "aiv_initialize_pipe",
    "reserve_buffer",
    "import_peer_buffer",
    "tfree_to_aic",
    "tfree_to_aiv",
    # Promoted tensor-only
    "create_tensor",
    "assemble",
    "dim",
    "full",
    "scatter_update",
    "ChunkConfig",
    "ChunkPolicy",
    "FunctionType",
    "ForKind",
    "Level",
    "LoopOrigin",
    "MemRef",
    "Role",
    "SplitMode",
    "Mem",
    "MemRefType",
    "MemorySpace",
    "Ptr",
    "PtrType",
    "PipeType",
    "TensorLayout",
    "TensorView",
    "TileLayout",
    "PadValue",
    "TileView",
    "ND",
    "DN",
    "NZ",
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
