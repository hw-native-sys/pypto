# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile operations for PyPTO Language DSL.

This module provides type-safe wrappers around pypto.ir.op.tile operations
that accept and return Tile types instead of raw Expr/Call objects.

Accessed as ``pl.tile.*``

Implementations live in the private ``_tile_*`` modules grouped by operator
family. Re-export the functions directly to preserve signatures and avoid
adding stack frames to source-span capture.
"""

# Explicit same-name aliases mark compatibility re-exports for type checkers,
# including names outside the DSL __all__. Ruff's redundant-alias rule conflicts
# with that export convention; this facade contains no implementation imports.
# ruff: noqa: PLC0414
import warnings as warnings
from collections.abc import Sequence as Sequence
from typing import Any as Any
from typing import Literal as Literal
from typing import TypeVar as TypeVar
from typing import overload as overload

from pypto.ir.utils import _get_span_or_capture as _get_span_or_capture
from pypto.ir.utils import _normalize_expr as _normalize_expr
from pypto.ir.utils import caller_warning_stacklevel as caller_warning_stacklevel
from pypto.ir.utils import has_partial_valid_region as has_partial_valid_region
from pypto.pypto_core import DataType as DataType
from pypto.pypto_core.ir import AccPhase as AccPhase
from pypto.pypto_core.ir import AtomicType as AtomicType
from pypto.pypto_core.ir import CachePolicy as CachePolicy
from pypto.pypto_core.ir import Expr as Expr
from pypto.pypto_core.ir import MemorySpace as MemorySpace
from pypto.pypto_core.ir import PadValue as PadValue
from pypto.pypto_core.ir import Span as Span
from pypto.pypto_core.ir import STPhase as STPhase
from pypto.pypto_core.ir import TileLayout as TileLayout

from ..typing import BoolLike as BoolLike
from ..typing import IntLike as IntLike
from ..typing import Ptr as Ptr
from ..typing import Scalar as Scalar
from ..typing import Tensor as Tensor
from ..typing import Tile as Tile
from ..typing import predicate_to_expr as predicate_to_expr
from ._tile_common import _TensorT as _TensorT
from ._tile_cross_core import _SplitOperandT as _SplitOperandT
from ._tile_cross_core import aic_gather as aic_gather
from ._tile_cross_core import aiv_shard as aiv_shard
from ._tile_cross_core import get_block_idx as get_block_idx
from ._tile_cross_core import get_block_num as get_block_num
from ._tile_cross_core import get_subblock_idx as get_subblock_idx
from ._tile_elementwise import _ir_core as _ir_core
from ._tile_elementwise import _scalar_operand_to_expr as _scalar_operand_to_expr
from ._tile_elementwise import _unwrap_rhs as _unwrap_rhs
from ._tile_elementwise import abs as abs
from ._tile_elementwise import add as add
from ._tile_elementwise import addc as addc
from ._tile_elementwise import adds as adds
from ._tile_elementwise import addsc as addsc
from ._tile_elementwise import and_ as and_
from ._tile_elementwise import ands as ands
from ._tile_elementwise import cast as cast
from ._tile_elementwise import cmp as cmp
from ._tile_elementwise import cmps as cmps
from ._tile_elementwise import cos as cos
from ._tile_elementwise import div as div
from ._tile_elementwise import divs as divs
from ._tile_elementwise import exp as exp
from ._tile_elementwise import fmod as fmod
from ._tile_elementwise import fmods as fmods
from ._tile_elementwise import log as log
from ._tile_elementwise import lrelu as lrelu
from ._tile_elementwise import max as max
from ._tile_elementwise import maximum as maximum
from ._tile_elementwise import maximums as maximums
from ._tile_elementwise import min as min
from ._tile_elementwise import minimum as minimum
from ._tile_elementwise import minimums as minimums
from ._tile_elementwise import mul as mul
from ._tile_elementwise import muls as muls
from ._tile_elementwise import neg as neg
from ._tile_elementwise import not_ as not_
from ._tile_elementwise import or_ as or_
from ._tile_elementwise import ors as ors
from ._tile_elementwise import part_add as part_add
from ._tile_elementwise import part_max as part_max
from ._tile_elementwise import part_min as part_min
from ._tile_elementwise import part_mul as part_mul
from ._tile_elementwise import prelu as prelu
from ._tile_elementwise import recip as recip
from ._tile_elementwise import relu as relu
from ._tile_elementwise import rem as rem
from ._tile_elementwise import rems as rems
from ._tile_elementwise import rsqrt as rsqrt
from ._tile_elementwise import sel as sel
from ._tile_elementwise import select as select
from ._tile_elementwise import sels as sels
from ._tile_elementwise import shl as shl
from ._tile_elementwise import shls as shls
from ._tile_elementwise import shr as shr
from ._tile_elementwise import shrs as shrs
from ._tile_elementwise import sin as sin
from ._tile_elementwise import sqrt as sqrt
from ._tile_elementwise import sub as sub
from ._tile_elementwise import subc as subc
from ._tile_elementwise import subs as subs
from ._tile_elementwise import subsc as subsc
from ._tile_elementwise import xor as xor
from ._tile_elementwise import xors as xors
from ._tile_indexing import gather as gather
from ._tile_indexing import gather_compare as gather_compare
from ._tile_indexing import gather_mask as gather_mask
from ._tile_indexing import gatherb as gatherb
from ._tile_indexing import mgather as mgather
from ._tile_indexing import mrgsort as mrgsort
from ._tile_indexing import mscatter as mscatter
from ._tile_indexing import scatter as scatter
from ._tile_indexing import scatter_mask as scatter_mask
from ._tile_indexing import scatter_update as scatter_update
from ._tile_indexing import sort32 as sort32
from ._tile_linalg import batch_matmul as batch_matmul
from ._tile_linalg import batch_matmul_acc as batch_matmul_acc
from ._tile_linalg import gemv as gemv
from ._tile_linalg import gemv_acc as gemv_acc
from ._tile_linalg import gemv_bias as gemv_bias
from ._tile_linalg import matmul as matmul
from ._tile_linalg import matmul_acc as matmul_acc
from ._tile_linalg import matmul_bias as matmul_bias
from ._tile_linalg import matmul_mx as matmul_mx
from ._tile_linalg import matmul_mx_acc as matmul_mx_acc
from ._tile_linalg import matmul_mx_bias as matmul_mx_bias
from ._tile_linalg import quant_mx as quant_mx
from ._tile_linalg import quant_mx_raw as quant_mx_raw
from ._tile_linalg import tmov_x2zz as tmov_x2zz
from ._tile_memory import _ir_ops as _ir_ops
from ._tile_memory import alloc as alloc
from ._tile_memory import arange as arange
from ._tile_memory import assemble as assemble
from ._tile_memory import ci as ci
from ._tile_memory import concat as concat
from ._tile_memory import create as create
from ._tile_memory import create_tile as create_tile
from ._tile_memory import extract as extract
from ._tile_memory import fillpad as fillpad
from ._tile_memory import fillpad_expand as fillpad_expand
from ._tile_memory import fillpad_inplace as fillpad_inplace
from ._tile_memory import full as full
from ._tile_memory import gather_row as gather_row
from ._tile_memory import load as load
from ._tile_memory import move as move
from ._tile_memory import random as random
from ._tile_memory import read as read
from ._tile_memory import reinterpret_view as reinterpret_view
from ._tile_memory import reshape as reshape
from ._tile_memory import set_validshape as set_validshape
from ._tile_memory import slice as slice
from ._tile_memory import store as store
from ._tile_memory import transpose as transpose
from ._tile_memory import transpose_view as transpose_view
from ._tile_memory import tri as tri
from ._tile_memory import write as write
from ._tile_reduction import col_argmax as col_argmax
from ._tile_reduction import col_argmin as col_argmin
from ._tile_reduction import col_expand as col_expand
from ._tile_reduction import col_expand_add as col_expand_add
from ._tile_reduction import col_expand_div as col_expand_div
from ._tile_reduction import col_expand_expdif as col_expand_expdif
from ._tile_reduction import col_expand_max as col_expand_max
from ._tile_reduction import col_expand_min as col_expand_min
from ._tile_reduction import col_expand_mul as col_expand_mul
from ._tile_reduction import col_expand_sub as col_expand_sub
from ._tile_reduction import col_max as col_max
from ._tile_reduction import col_min as col_min
from ._tile_reduction import col_prod as col_prod
from ._tile_reduction import col_sum as col_sum
from ._tile_reduction import expands as expands
from ._tile_reduction import row_argmax as row_argmax
from ._tile_reduction import row_argmin as row_argmin
from ._tile_reduction import row_expand as row_expand
from ._tile_reduction import row_expand_add as row_expand_add
from ._tile_reduction import row_expand_div as row_expand_div
from ._tile_reduction import row_expand_expdif as row_expand_expdif
from ._tile_reduction import row_expand_max as row_expand_max
from ._tile_reduction import row_expand_min as row_expand_min
from ._tile_reduction import row_expand_mul as row_expand_mul
from ._tile_reduction import row_expand_sub as row_expand_sub
from ._tile_reduction import row_max as row_max
from ._tile_reduction import row_min as row_min
from ._tile_reduction import row_prod as row_prod
from ._tile_reduction import row_sum as row_sum
from ._utils import _normalize_intlike as _normalize_intlike
from .system_ops import tpop_from_aic as tpop_from_aic
from .system_ops import tpop_from_aiv as tpop_from_aiv
from .system_ops import tpush_to_aic as tpush_to_aic
from .system_ops import tpush_to_aiv as tpush_to_aiv

__all__ = [
    "MemRefType",
    "alloc",
    "create_tile",
    "create",
    "read",
    "write",
    "load",
    "store",
    "assemble",
    "gather_row",
    "extract",
    "scatter_update",
    "concat",
    "move",
    "aiv_shard",
    "aic_gather",
    "full",
    "ci",
    "arange",
    "tri",
    "random",
    "fillpad",
    "fillpad_inplace",
    "fillpad_expand",
    "get_block_idx",
    "get_subblock_idx",
    "get_block_num",
    "add",
    "sub",
    "mul",
    "div",
    "adds",
    "subs",
    "muls",
    "divs",
    "neg",
    "exp",
    "sin",
    "cos",
    "sqrt",
    "rsqrt",
    "recip",
    "log",
    "abs",
    "relu",
    "cast",
    "quant_mx",
    "quant_mx_raw",
    "tmov_x2zz",
    "matmul",
    "batch_matmul",
    "matmul_acc",
    "batch_matmul_acc",
    "matmul_bias",
    "matmul_mx",
    "matmul_mx_acc",
    "matmul_mx_bias",
    "gemv",
    "gemv_acc",
    "gemv_bias",
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
    "maximum",
    "row_expand",
    "row_expand_sub",
    "row_expand_div",
    "row_expand_mul",
    "row_expand_add",
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
    "expands",
    "minimum",
    "cmp",
    "cmps",
    "max",
    "min",
    "slice",
    "reshape",
    "reinterpret_view",
    "transpose",
    "transpose_view",
    "set_validshape",
    "rem",
    "rems",
    "part_add",
    "part_mul",
    "part_max",
    "part_min",
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
    "select",
    "tpush_to_aiv",
    "tpush_to_aic",
    "tpop_from_aic",
    "tpop_from_aiv",
    "sort32",
    "gather",
    "gatherb",
    "gather_mask",
    "gather_compare",
    "scatter",
    "scatter_mask",
    "mscatter",
    "mgather",
    "MaskPattern",
    "mrgsort",
]


class MemRefType:
    """Opaque sentinel type for ``MemRef``-typed variables in printed IR.

    The C++ printer emits ``pl.MemRefType`` as the annotation for a bare
    ``MemRef`` variable (the SSA-edge type of a ``MemRef`` expression).  This
    class exists solely so that the printed Python source is valid Python that
    the text-parser can ``exec()``.

    Note: this is *not* the type of a ``tile.alloc`` / ``tensor.alloc`` result —
    those produce a base ``Ptr`` (``PtrType``); see [`alloc`][pypto.language.tile.alloc].
    """


class MaskPattern:
    """Hardware mask pattern selectors for tile.gather_mask.

    Bit patterns are read right-to-left; lower bits correspond to lower indices.
    """

    P0101 = 1  # stride-2: select positions 0, 2, 4, ...
    P1010 = 2  # stride-2: select positions 1, 3, 5, ...
    P0001 = 3  # stride-4: select positions 0, 4, 8, ...
    P0010 = 4  # stride-4: select positions 1, 5, 9, ...
    P0100 = 5  # stride-4: select positions 2, 6, 10, ...
    P1000 = 6  # stride-4: select positions 3, 7, 11, ...
    P1111 = 7  # select all positions
