# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test fillpad_inplace operation with different pad values (zero, max, min).

Each test verifies that fillpad_inplace correctly fills the padding region in place:
1. Load 48x64 data into 64x64 tile (rows 48-63 are padding region)
2. fillpad_inplace with specified pad_value fills rows 48-63 in place (shared UB buffer)
3. Store the full 64x64 tile to output
4. Verify: rows 0-47 = input data, rows 48-63 = expected fill value
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto.pypto_core.passes import MemoryPlanner
from pypto.runtime.tensor_spec import ScalarSpec


@pl.jit
def fillpad_inplace_zero(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Fill the invalid rows with zero and store the full tile."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
        pl.store(padded, [0, 0], out)
    return out


@pl.jit
def fillpad_inplace_max(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Fill the invalid rows with positive infinity and store the full tile."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.max)
        pl.store(padded, [0, 0], out)
    return out


@pl.jit
def fillpad_inplace_min(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Fill the invalid rows with negative infinity and store the full tile."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.min)
        pl.store(padded, [0, 0], out)
    return out


def _fillpad_cases():
    for kernel, mode, value in (
        (fillpad_inplace_zero, "zero", 0.0),
        (fillpad_inplace_max, "max", float("inf")),
        (fillpad_inplace_min, "min", float("-inf")),
    ):
        x = torch.arange(1, 48 * 64 + 1, dtype=torch.float32).reshape(48, 64)
        expected = torch.full((64, 64), value, dtype=torch.float32)
        expected[:48, :] = x
        yield st.case(
            kernel,
            x,
            torch.empty(64, 64, dtype=torch.float32),
            name=f"fillpad_inplace_{mode}",
            golden=lambda _tensors, expected=expected: expected,
        )


@st.cases(*_fillpad_cases())
def test_fillpad_inplace(case_run):
    """Check both preserved input rows and padding for every fill mode."""
    case_run.assert_passed()


@pl.jit
def fillpad_then_narrow_static(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """A post-padding validity update must mutate the original padded handle."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
        _narrowed = pl.tile.set_validshape(padded, 32, 48)
        repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.max)
        pl.store(repadded, [0, 0], out)
    return out


@pl.jit
def fillpad_then_narrow_dynamic(
    x: pl.Tensor, out: pl.Out[pl.Tensor], rows: pl.Scalar[pl.INDEX], cols: pl.Scalar[pl.INDEX]
):
    """Runtime bounds also preserve alias mutation, including previously padded rows."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
        _narrowed = pl.tile.set_validshape(padded, rows, cols)
        repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.max)
        pl.store(repadded, [0, 0], out)
    return out


@pl.jit
def fillpad_then_narrow_same_mode(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Repeating zero padding must restore validity without widening the source."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
        _narrowed = pl.tile.set_validshape(padded, 32, 48)
        repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.zero)
        pl.store(repadded, [0, 0], out)
    return out


@pl.jit
def fillpad_branch_then_narrow(
    x: pl.Tensor, out: pl.Out[pl.Tensor], rows: pl.Scalar[pl.INDEX], cols: pl.Scalar[pl.INDEX]
):
    """Exercise both outcomes of native Tile selection before padding."""
    with pl.at(level=pl.Level.CORE_GROUP):
        if rows < 32:
            src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        else:
            src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
        padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
        _narrowed = pl.tile.set_validshape(padded, rows, cols)
        repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.max)
        pl.store(repadded, [0, 0], out)
    return out


def _fillpad_narrow_cases():
    for kernel, bounds in (
        (fillpad_then_narrow_static, (32, 48)),
        (fillpad_then_narrow_dynamic, (32, 48)),
        (fillpad_then_narrow_dynamic, (56, 48)),
        (fillpad_then_narrow_same_mode, (32, 48)),
        (fillpad_branch_then_narrow, (16, 48)),
        (fillpad_branch_then_narrow, (56, 48)),
    ):
        rows, cols = bounds
        # Nonzero deterministic input distinguishes preserved data from padding.
        x = torch.arange(1, 48 * 64 + 1, dtype=torch.float32).reshape(48, 64)
        padded = torch.zeros(64, 64, dtype=torch.float32)
        padded[:48, :] = x
        fill_value = 0.0 if kernel is fillpad_then_narrow_same_mode else float("inf")
        expected = torch.full((64, 64), fill_value, dtype=torch.float32)
        expected[:rows, :cols] = padded[:rows, :cols]
        dynamic = kernel in (fillpad_then_narrow_dynamic, fillpad_branch_then_narrow)
        if kernel is fillpad_then_narrow_same_mode:
            kind = "same_mode"
        else:
            kind = "branch" if kernel is fillpad_branch_then_narrow else "dynamic" if dynamic else "static"
        scalar_args = bounds if dynamic else ()
        case = st.case(
            kernel,
            x,
            torch.empty(64, 64, dtype=torch.float32),
            *scalar_args,
            name=f"fillpad_after_{kind}_narrow_{rows}x{cols}",
            scalars=[ScalarSpec("rows", rows), ScalarSpec("cols", cols)] if dynamic else [],
            memory_planner=MemoryPlanner.PTOAS,
            golden=lambda _tensors, expected=expected: expected,
        )
        # Keep ordinary allocation-backed validity tests enabled. Only the
        # native If descriptor composition awaits PTOAS backend support.
        if kernel is fillpad_branch_then_narrow:
            yield pytest.param(
                case,
                marks=pytest.mark.skip(
                    reason="PTOAS set_validshape on native scf.if Tile results: "
                    "https://github.com/hw-native-sys/PTOAS/issues/1708"
                ),
            )
        else:
            yield case


@st.cases(*_fillpad_narrow_cases())
def test_fillpad_inplace_after_set_validshape(case_run):
    """The second fill must observe the narrowed metadata through the old alias."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
