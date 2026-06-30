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
from pypto.jit import JITFunction, jit
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    AtomicType,
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

from . import arg_direction as adir
from . import optimizations, parser
from .dsl_api import (
    at,
    cluster,
    cond,
    const,
    parallel,
    pipeline,
    range,
    split_aiv,
    spmd,
    static_assert,
    static_print,
    unroll,
    while_,
    yield_,
)
from .op import array_ops as array
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
from .op.tensor_ops import (
    assemble,
    cos,
    create_l1,
    create_tensor,
    dim,
    dump_tag,
    expand_clone,
    full,
    gather,
    gather_row,
    get_block_idx,
    get_block_num,
    get_subblock_idx,
    mrgsort,
    no_dep,
    paged_gather,
    scatter,
    scatter_update,
    sin,
    sort32,
)
from .op.tensor_ops import ci as arange
from .op.tile_ops import (
    MemRefType,
    addc,
    addsc,
    aic_gather,
    aiv_shard,
    and_,
    ands,
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
    maximums,
    min,
    minimums,
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
from .op.tile_ops import (
    mscatter as mscatter,
)
from .op.unified_ops import (
    abs,
    add,
    batch_matmul,
    cast,
    cmp,
    col_argmax,
    col_argmin,
    col_expand,
    col_expand_add,
    col_expand_div,
    col_expand_expdif,
    col_expand_max,
    col_expand_min,
    col_expand_mul,
    col_expand_sub,
    col_max,
    col_min,
    col_prod,
    col_sum,
    concat,
    div,
    exp,
    expands,
    fillpad,
    fillpad_expand,
    fmod,
    fmods,
    matmul,
    matmul_acc,
    maximum,
    minimum,
    mul,
    neg,
    part_add,
    part_max,
    part_min,
    part_mul,
    read,
    recip,
    reshape,
    row_argmax,
    row_argmin,
    row_expand,
    row_expand_add,
    row_expand_div,
    row_expand_expdif,
    row_expand_max,
    row_expand_min,
    row_expand_mul,
    row_expand_sub,
    row_max,
    row_min,
    row_prod,
    row_sum,
    rsqrt,
    set_validshape,
    slice,
    sqrt,
    sub,
    transpose,
    write,
)
from .optimizations import split
from .parser.decorator import InlineFunction, function, inline, program
from .parser.text_parser import loads, loads_program, parse, parse_program
from .scope import ScopeMode, manual_scope, scope, spmd_submit, submit
from .typing import Array, DynVar, InOut, IntLike, MemRef, Out, Scalar, Tensor, Tile, Tuple, dynamic

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
TASK_ID = DataType.TASK_ID

# Convenience alias for the TaskId scalar annotation (manual_scope deps).
# ``pl.TaskId`` is equivalent to ``pl.Scalar[pl.TASK_ID]``.
TaskId = Scalar[TASK_ID]

__all__ = [
    "jit",
    "JITFunction",
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
    "Array",
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
    "pipeline",
    "while_",
    "yield_",
    "cond",
    "static_print",
    "static_assert",
    "at",
    "cluster",
    "spmd",
    "split_aiv",
    "optimizations",
    "split",
    "adir",
    "array",
    "tile",
    "system",
    "tensor",
    # Unified dispatch
    "add",
    "sub",
    "mul",
    "div",
    "part_add",
    "part_mul",
    "part_max",
    "part_min",
    "maximum",
    "exp",
    "cast",
    "concat",
    "reshape",
    "transpose",
    "slice",
    "matmul",
    "batch_matmul",
    "row_max",
    "row_sum",
    "row_min",
    "row_prod",
    "col_sum",
    "col_max",
    "col_min",
    "col_prod",
    "row_argmax",
    "row_argmin",
    "col_argmax",
    "col_argmin",
    "row_expand",
    "row_expand_add",
    "row_expand_sub",
    "row_expand_mul",
    "row_expand_div",
    "row_expand_max",
    "row_expand_min",
    "row_expand_expdif",
    "col_expand",
    "col_expand_mul",
    "col_expand_div",
    "col_expand_sub",
    "col_expand_add",
    "col_expand_max",
    "col_expand_min",
    "col_expand_expdif",
    "expand_clone",
    "expands",
    "neg",
    "abs",
    "recip",
    "read",
    "write",
    # Promoted tile-only
    "create_tile",
    "fillpad",
    "fillpad_expand",
    "load",
    "store",
    "move",
    "mscatter",
    "sqrt",
    "rsqrt",
    "log",
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
    "set_validshape",
    "rem",
    "rems",
    "fmod",
    "fmods",
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
    "maximums",
    "minimums",
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
    "aiv_shard",
    "aic_gather",
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
    "cos",
    "dim",
    "full",
    "ScopeMode",
    "scope",
    "manual_scope",
    "submit",
    "spmd_submit",
    "no_dep",
    "dump_tag",
    "scatter",
    "scatter_update",
    "sin",
    "arange",
    "gather",
    "paged_gather",
    "create_l1",
    "gather_row",
    "get_block_idx",
    "get_block_num",
    "get_subblock_idx",
    "mrgsort",
    "sort32",
    "FunctionType",
    "ForKind",
    "AtomicType",
    "Level",
    "LoopOrigin",
    "MemRef",
    "Role",
    "SplitMode",
    "Mem",
    "MemRefType",
    "MemorySpace",
    "PipeType",
    "Ptr",
    "PtrType",
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
    "TASK_ID",
    "TaskId",
]
