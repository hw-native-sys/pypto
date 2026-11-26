# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Runtime tests for tile-based elementwise operations using the @pl.jit frontend.

Verifies that the migrated tile_add_64/tile_add_128/tile_mul_64/tile_mul_128 kernels
from ``examples.beginner.elementwise`` produce results matching torch references.
"""

import pypto.language as pl
import pytest
import torch
from examples.beginner.elementwise import (
    tile_add_64,
    tile_add_128,
    tile_mul_64,
    tile_mul_128,
)
from harness import st

_ADD_KERNELS = {64: tile_add_64, 128: tile_add_128}
_MUL_KERNELS = {64: tile_mul_64, 128: tile_mul_128}


def _add_case(size):
    """Tile addition: c = a + b at the given square size."""
    a = torch.full((size, size), 2.0, dtype=torch.float32)
    b = torch.full((size, size), 3.0, dtype=torch.float32)
    c = torch.zeros((size, size), dtype=torch.float32)
    return st.case(_ADD_KERNELS[size], a, b, c, name=f"tile_add_{size}", golden=lambda _: a + b)


def _mul_case(size):
    """Tile multiplication: c = a * b at the given square size."""
    torch.manual_seed(0)
    a = torch.randn(size, size, dtype=torch.float32)
    b = torch.full((size, size), 3.0, dtype=torch.float32)
    c = torch.zeros((size, size), dtype=torch.float32)
    return st.case(_MUL_KERNELS[size], a, b, c, name=f"tile_mul_{size}", golden=lambda _: a * b)


@st.cases(_add_case(64), _add_case(128))
def test_tile_add(case_run):
    """Elementwise add matches the torch reference at both sizes."""
    case_run.assert_passed()


@st.cases(_mul_case(64), _mul_case(128))
def test_tile_mul(case_run):
    """Elementwise multiply matches the torch reference at both sizes."""
    case_run.assert_passed()


@pl.jit
def _mul_column_coefficients(
    x: pl.Tensor[[8, 32], pl.FP32],
    c: pl.Tensor[[1, 32], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 32], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.assemble(out, pl.mul(x, c), [0, 0])
    return out


def _column_broadcast_case():
    # Every column has a distinct coefficient and every input row is different.
    x = torch.arange(1, 257, dtype=torch.float32).reshape(8, 32)
    c = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32)
    return st.case(
        _mul_column_coefficients, x, c, torch.zeros_like(x), golden=lambda _: x * c, rtol=0, atol=0
    )


@st.cases(_column_broadcast_case())
def test_tensor_mul_column_broadcast(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
