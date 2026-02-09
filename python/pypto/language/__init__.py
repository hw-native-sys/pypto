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
- Type-safe operation wrappers (op.tensor.*, op.block.*)
- DSL helpers (range, yield_)
- DataType constants

Typical usage:
    import pypto.language as pl

    @pl.function
    def my_func(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
        result: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.create([64, 128], dtype=pl.FP32)
        return result

    @pl.function
    def block_func(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        tile: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(x, 0, 0, 64, 64)
        result: pl.Tile[[64, 64], pl.FP32] = pl.op.block.add(tile, tile)
        return pl.op.block.store(result, 0, 0, 64, 64, x)

    @pl.function
    def scalar_func(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
        return x
"""

# Import decorators and parsing functions from local parser module
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ForKind, FunctionType

from . import op, parser
from .dsl_api import parallel, range, yield_
from .parser.decorator import function, program
from .parser.text_parser import load, load_program, parse, parse_program
from .scalar import Scalar
from .tensor import Tensor
from .tile import Tile

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

__all__ = [
    "function",
    "program",
    "parse",
    "parser",
    "load",
    "parse_program",
    "load_program",
    "Tensor",
    "Tile",
    "Scalar",
    "range",
    "parallel",
    "yield_",
    "op",
    "FunctionType",
    "ForKind",
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
]
