# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile operations for PyPTO IR.

Tile operations work on TileType (unified buffer) and support tile-level programming.
These operations include memory operations (load, store), element-wise operations,
unary operations, and reduction operations.

Implementations live in the private ``_tile_*`` modules grouped by operator
family. Re-export the functions directly to preserve signatures and avoid
adding stack frames to source-span capture.
"""

# Explicit same-name aliases mark compatibility re-exports for type checkers,
# including names outside the DSL __all__. Ruff's redundant-alias rule conflicts
# with that export convention; this facade contains no implementation imports.
# ruff: noqa: PLC0414
from collections.abc import Sequence as Sequence
from typing import Any as Any

from pypto.pypto_core import DataType as DataType
from pypto.pypto_core.ir import AccPhase as AccPhase
from pypto.pypto_core.ir import Call as Call
from pypto.pypto_core.ir import ConstInt as ConstInt
from pypto.pypto_core.ir import Expr as Expr
from pypto.pypto_core.ir import MemorySpace as MemorySpace
from pypto.pypto_core.ir import PadValue as PadValue
from pypto.pypto_core.ir import ScalarType as ScalarType
from pypto.pypto_core.ir import Span as Span
from pypto.pypto_core.ir import STPhase as STPhase
from pypto.pypto_core.ir import TensorLayout as TensorLayout
from pypto.pypto_core.ir import TileLayout as TileLayout

from ..utils import _get_span_or_capture as _get_span_or_capture
from ..utils import _normalize_const_to_dtype as _normalize_const_to_dtype
from ..utils import _normalize_expr as _normalize_expr
from ..utils import _normalize_scalar_operand as _normalize_scalar_operand
from ..utils import (
    _normalize_signless_same_width_scalar_operand as _normalize_signless_same_width_scalar_operand,
)
from ..utils import _to_int32_scalar as _to_int32_scalar
from ..utils import _to_make_tuple as _to_make_tuple
from ..utils import resolve_cast_mode as resolve_cast_mode
from ..utils import resolve_fixpipe_epilogue as resolve_fixpipe_epilogue
from ..utils import resolve_saturation_deviation as resolve_saturation_deviation
from ._pad_value import normalize_pad_value as normalize_pad_value
from ._tile_cross_core import _resolve_tpop_type as _resolve_tpop_type
from ._tile_cross_core import aic_gather as aic_gather
from ._tile_cross_core import aiv_shard as aiv_shard
from ._tile_cross_core import get_block_idx as get_block_idx
from ._tile_cross_core import get_block_num as get_block_num
from ._tile_cross_core import get_subblock_idx as get_subblock_idx
from ._tile_cross_core import tpop_from_aic as tpop_from_aic
from ._tile_cross_core import tpop_from_aiv as tpop_from_aiv
from ._tile_cross_core import tpush_to_aic as tpush_to_aic
from ._tile_cross_core import tpush_to_aiv as tpush_to_aiv
from ._tile_elementwise import _create_tile_binary_call as _create_tile_binary_call
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
from ._tile_elementwise import maximum as maximum
from ._tile_elementwise import maximums as maximums
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
from ._tile_indexing import _GATHER_COMPARE_CMP_MODES as _GATHER_COMPARE_CMP_MODES
from ._tile_indexing import _MGATHER_COALESCE as _MGATHER_COALESCE
from ._tile_indexing import _MGATHER_GATHER_OOB as _MGATHER_GATHER_OOB
from ._tile_indexing import _resolve_mgather_coalesce as _resolve_mgather_coalesce
from ._tile_indexing import _resolve_mgather_gather_oob as _resolve_mgather_gather_oob
from ._tile_indexing import gather as gather
from ._tile_indexing import gather_compare as gather_compare
from ._tile_indexing import gather_mask as gather_mask
from ._tile_indexing import gatherb as gatherb
from ._tile_indexing import mgather as mgather
from ._tile_indexing import mrgsort as mrgsort
from ._tile_indexing import mrgsort_format1 as mrgsort_format1
from ._tile_indexing import mrgsort_format2 as mrgsort_format2
from ._tile_indexing import mscatter as mscatter
from ._tile_indexing import resolve_gather_compare_cmp_mode as resolve_gather_compare_cmp_mode
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
from ._tile_linalg import tget_scale_addr as tget_scale_addr
from ._tile_linalg import tmov_x2zz as tmov_x2zz
from ._tile_linalg import tquant_mx as tquant_mx
from ._tile_linalg import tquant_mx_raw as tquant_mx_raw
from ._tile_memory import _fixpipe_epilogue_kwargs as _fixpipe_epilogue_kwargs
from ._tile_memory import _ir_core as _ir_core
from ._tile_memory import _normalize_axis_const as _normalize_axis_const
from ._tile_memory import _validate_offsets_shapes as _validate_offsets_shapes
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
