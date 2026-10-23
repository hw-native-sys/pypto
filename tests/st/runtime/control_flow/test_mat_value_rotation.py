# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical and overwrite protection for Mat current/next prefetch rotation."""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import passes

_ROWS, _K, _N, _BLOCKS = 64, 64, 16, 8
_COUNTS = torch.tensor([0, 1, 3, 4, 7, 8], dtype=torch.int64)
_CASES = len(_COUNTS)
_BAND = _ROWS * _BLOCKS


@pl.jit.incore
def _rotate_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    count: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    current = pl.load(a, [0, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [_K, _N], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(count):
        if block + 1 < count:
            following = pl.load(a, [(block + 1) * _ROWS, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        out = pl.store(result, [base + block * _ROWS, 0], out)
        current = following
    return out


@pl.jit
def _rotate_entry(a: pl.Tensor, b: pl.Tensor, counts: pl.Tensor, out: pl.InOut[pl.Tensor]):
    for case_index in pl.range(_CASES):
        count = pl.tensor.read(counts, [case_index])
        base = case_index * _BAND
        with pl.spmd(1):
            out = _rotate_kernel(a, b, out, count, base)
    return out


@pl.jit.incore
def _stalled_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    count: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    current = pl.load(a, [0, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [_K, _N], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(count):
        if block + 1 < count and block % 3 != 1:
            following = pl.load(a, [(block + 1) * _ROWS, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        out = pl.store(result, [base + block * _ROWS, 0], out)
        current = following
    return out


@pl.jit
def _stalled_entry(a: pl.Tensor, b: pl.Tensor, counts: pl.Tensor, out: pl.InOut[pl.Tensor]):
    for case_index in pl.range(_CASES):
        count = pl.tensor.read(counts, [case_index])
        base = case_index * _BAND
        with pl.spmd(1):
            out = _stalled_kernel(a, b, out, count, base)
    return out


@pl.jit.incore
def _repeated_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    count: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    seed = pl.load(a, [0, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [_K, _N], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for repeat in pl.range(2):
        current = seed
        for block in pl.range(count):
            if block + 1 < count:
                following = pl.load(
                    a, [(block + 1) * _ROWS, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat
                )
            else:
                following = current
            left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
            result = pl.tile.matmul(left, right)
            out = pl.store(result, [base + repeat * _BAND + block * _ROWS, 0], out)
            current = following
    return out


@pl.jit
def _repeated_entry(a: pl.Tensor, b: pl.Tensor, counts: pl.Tensor, out: pl.InOut[pl.Tensor]):
    for case_index in pl.range(_CASES):
        count = pl.tensor.read(counts, [case_index])
        base = case_index * 2 * _BAND
        with pl.spmd(1):
            out = _repeated_kernel(a, b, out, count, base)
    return out


@pl.jit.incore
def _preserved_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    observed: pl.InOut[pl.Tensor],
    count: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
    observed_base: pl.Scalar[pl.INT64],
):
    current = pl.load(a, [0, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
    saved = current
    weight = pl.load(b, [0, 0], [_K, _N], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(count):
        if block + 1 < count:
            following = pl.load(a, [(block + 1) * _ROWS, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        out = pl.store(result, [base + block * _ROWS, 0], out)
        current = following
    # Both observations matter: keeping the seed intact must not replace the
    # final selector with the initializer, including when the loop takes no trips.
    saved_left = pl.tile.move(saved, target_memory=pl.MemorySpace.Left)
    saved_result = pl.tile.matmul(saved_left, right)
    observed = pl.store(saved_result, [observed_base, 0], observed)
    final_left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
    final_result = pl.tile.matmul(final_left, right)
    observed = pl.store(final_result, [observed_base + _ROWS, 0], observed)
    return out, observed


@pl.jit
def _preserved_entry(
    a: pl.Tensor,
    b: pl.Tensor,
    counts: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    observed: pl.InOut[pl.Tensor],
):
    for case_index in pl.range(_CASES):
        count = pl.tensor.read(counts, [case_index])
        base = case_index * _BAND
        observed_base = case_index * 2 * _ROWS
        with pl.spmd(1):
            out, observed = _preserved_kernel(a, b, out, observed, count, base, observed_base)
    return out, observed


@pl.jit.incore
def _dual_kernel(
    a: pl.Tensor,
    c: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    out2: pl.InOut[pl.Tensor],
    count: pl.Scalar[pl.INT64],
    base: pl.Scalar[pl.INT64],
):
    current = pl.load(a, [0, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
    current2 = pl.load(c, [0, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [_K, _N], target_memory=pl.MemorySpace.Mat)
    right = pl.tile.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(count):
        if block + 1 < count:
            following = pl.load(a, [(block + 1) * _ROWS, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
            following2 = pl.load(c, [(block + 1) * _ROWS, 0], [_ROWS, _K], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
            following2 = current2
        left = pl.tile.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        out = pl.store(result, [base + block * _ROWS, 0], out)
        left2 = pl.tile.move(current2, target_memory=pl.MemorySpace.Left)
        result2 = pl.tile.matmul(left2, right)
        out2 = pl.store(result2, [base + block * _ROWS, 0], out2)
        current = following
        current2 = following2
    return out, out2


@pl.jit
def _dual_entry(
    a: pl.Tensor,
    c: pl.Tensor,
    b: pl.Tensor,
    counts: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    out2: pl.InOut[pl.Tensor],
):
    for case_index in pl.range(_CASES):
        count = pl.tensor.read(counts, [case_index])
        base = case_index * _BAND
        with pl.spmd(1):
            out, out2 = _dual_kernel(a, c, b, out, out2, count, base)
    return out, out2


def _golden(kind: str, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = tensors["out"].clone()
    product = tensors["a"].float() @ tensors["b"].float()
    extra = {}
    if kind == "preserved":
        extra["observed"] = tensors["observed"].clone()
    elif kind == "dual":
        extra["out2"] = tensors["out2"].clone()
        product2 = tensors["c"].float() @ tensors["b"].float()
    for case_index, count in enumerate(tensors["counts"].tolist()):
        base = case_index * _BAND * (2 if kind == "repeated" else 1)
        active_rows = count * _ROWS
        if kind == "stalled":
            current_block = 0
            for block in range(count):
                row = base + block * _ROWS
                source = current_block * _ROWS
                out[row : row + _ROWS] = product[source : source + _ROWS]
                if block + 1 < count and block % 3 != 1:
                    current_block = block + 1
        else:
            out[base : base + active_rows] = product[:active_rows]
            if kind == "repeated":
                out[base + _BAND : base + _BAND + active_rows] = product[:active_rows]
        if kind == "preserved":
            observed_base = case_index * 2 * _ROWS
            last = max(0, count - 1) * _ROWS
            extra["observed"][observed_base : observed_base + _ROWS] = product[:_ROWS]
            extra["observed"][observed_base + _ROWS : observed_base + 2 * _ROWS] = product[
                last : last + _ROWS
            ]
        elif kind == "dual":
            extra["out2"][base : base + active_rows] = product2[:active_rows]
    return {"out": out, **extra}


# Dyadic inputs keep the FP16 values exact while varying every source block.
# A constant panel or identical streams would hide a missed rotation/overwrite.
_GRID = torch.arange(_BLOCKS * _ROWS * _K, dtype=torch.float32).reshape(_BLOCKS, _ROWS, _K)
_BLOCK_BIAS = torch.arange(_BLOCKS, dtype=torch.float32).reshape(_BLOCKS, 1, 1) / 4
_A = ((_GRID.remainder(37) - 18) / 16 + _BLOCK_BIAS).reshape(_BAND, _K).to(torch.float16)
_C = (_A.flip(0) * -0.5 + 0.75).to(torch.float16)
_B = ((torch.arange(_K * _N).reshape(_K, _N).remainder(23) - 11) / 32).to(torch.float16)


def _case(kind: str, enable_buffer_ir: bool):
    bands = 2 if kind == "repeated" else 1
    out = torch.full((_CASES * _BAND * bands, _N), -777.0)
    if kind in {"rotation", "stalled", "repeated"}:
        program = {"rotation": _rotate_entry, "stalled": _stalled_entry, "repeated": _repeated_entry}[kind]
        inputs = [_A, _B, _COUNTS, out]
    elif kind == "preserved":
        program = _preserved_entry
        inputs = [_A, _B, _COUNTS, out, torch.full((_CASES * 2 * _ROWS, _N), -555.0)]
    else:
        program = _dual_entry
        inputs = [_A, _C, _B, _COUNTS, out, torch.full_like(out, -333.0)]
    return st.case(
        program,
        *inputs,
        name=f"mat_value_{kind}_{'buffer' if enable_buffer_ir else 'tile'}",
        golden=lambda tensors: _golden(kind, tensors),
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=enable_buffer_ir,
        rtol=1e-4,
        atol=1e-4,
    )


@st.cases(
    *[
        _case(kind, enable_buffer_ir)
        for kind in ("rotation", "preserved", "dual", "stalled", "repeated")
        for enable_buffer_ir in (False, True)
    ]
)
@pytest.mark.platforms("a2a3", "a2a3sim", reason="Mat prefetch rotation regression for A2/A3.")
def test_mat_prefetch_value_rotation(case_run):
    """Every runtime trip count preserves values, inactive rows, and seed observations."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
