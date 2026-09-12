# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for an ND x NZ matmul: an ND activation against a GM weight
annotated ``pl.NZ``.

``pl.NZ`` is an *assertion about the bytes already in GM* — that they are in
pto-isa's NZ fractal order — not a request to convert anything. So the host has
to produce those bytes (``_pack_nz``) and the golden has to read them back
logically (``_unpack_nz``). The DSL keeps the logical ``[N, K]`` shape and
logical slicing throughout; ``BlockNzTensorViews`` supplies the physical rank-5
description the backend needs.

What this guards is that the weight's ``TLOAD`` is NZ->NZ rather than ND->NZ,
so no online fractal conversion runs on the weight load.

Three kernels cover the two coordinate axes the pass has to map, because a
whole-tensor load leaves every offset zero and so exercises none of them:

| Kernel | Weight window | Offset under test |
| ------ | ------------- | ----------------- |
| ``nz_matmul`` | all of ``[N, K]`` | none — the baseline |
| ``nz_matmul_n_sliced`` | two ``[N/2, K]`` halves | row fractal, ``n0 // 16`` |
| ``nz_matmul_k_sliced`` | ``[N, K/2]`` upper half | C0 column block, ``k0 // c0`` |

Inputs are small integers so the FP32 accumulation is exact and the comparison
runs at ``rtol=atol=0``: a mis-addressed fractal reads entirely different
elements, which must not be able to hide under a numeric tolerance.

Requires PTOAS >= 0.61, the first release that treats an explicit ``layout=nz``
annotation on ``pto.make_tensor_view`` as authoritative instead of re-inferring
it structurally (see ``docs/en/dev/passes/15-block_nz_tensor_views.md``).
"""

import pypto.language as pl
import pytest
import torch
from harness import st

M, K, N = 64, 256, 128
N_TILE = N // 2
K_TILE = K // 2

# One 32-byte C0 line and a 16-row fractal are the two constants that define
# pto-isa's NZ blocking; `c0` is the number of *elements* in that line, so it
# follows from the dtype width.
_C0_BYTES = 32
_FRACTAL_ROWS = 16


def _c0_elems(dtype: torch.dtype) -> int:
    """Number of elements in one 32-byte C0 line for *dtype*."""
    return _C0_BYTES // torch.empty((), dtype=dtype).element_size()


def _pack_nz(logical: torch.Tensor) -> torch.Tensor:
    """Reorder a logical row-major ``[R, C]`` matrix into NZ fractal order.

    NZ is "column blocks outside, row fractals inside": ``c0`` contiguous
    elements form one C0 line, 16 rows form one ``16 x c0`` fractal, ``R/16``
    fractals walk down the row axis, and ``C/c0`` steps between column blocks —
    the blocked shape ``[C/c0, R/16, 16, c0]`` that ``BlockNzTensorViews`` gives
    the GM view.

    The result is reshaped back to ``[R, C]`` so the buffer the harness
    allocates matches the *logical* shape the kernel is annotated with. Only the
    byte order differs; the element count is identical.
    """
    rows, cols = logical.shape
    c0 = _c0_elems(logical.dtype)
    assert rows % _FRACTAL_ROWS == 0, f"NZ needs {_FRACTAL_ROWS}-row fractals, got {rows} rows"
    assert cols % c0 == 0, f"NZ needs whole C0 lines of {c0} elements, got {cols} cols"
    return (
        logical.reshape(rows // _FRACTAL_ROWS, _FRACTAL_ROWS, cols // c0, c0)
        .permute(2, 0, 1, 3)
        .contiguous()
        .reshape(rows, cols)
    )


def _unpack_nz(packed: torch.Tensor) -> torch.Tensor:
    """Restore NZ-ordered bytes to the logical row-major ``[R, C]`` matrix."""
    rows, cols = packed.shape
    c0 = _c0_elems(packed.dtype)
    return (
        packed.reshape(cols // c0, rows // _FRACTAL_ROWS, _FRACTAL_ROWS, c0)
        .permute(1, 2, 0, 3)
        .contiguous()
        .reshape(rows, cols)
    )


# ============================================================================
# Kernels
#
# `target_memory=pl.Mem.Mat` is required on an NZ load, not merely preferred:
# NZ->NZ is the cube operand path, and BlockNzTensorViews rejects any other
# target rather than mis-addressing the tensor.
# ============================================================================


@pl.jit.incore
def _nz_matmul_kernel(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16, pl.NZ],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """out = x @ w^T, with w read straight out of its NZ fractal layout."""
    xt = pl.load(x, [0, 0], [M, K], target_memory=pl.Mem.Mat)
    wt = pl.load(w, [0, 0], [N, K], target_memory=pl.Mem.Mat)
    return pl.store(pl.matmul(xt, pl.tile.transpose_view(wt)), [0, 0], out)


@pl.jit
def nz_matmul(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16, pl.NZ],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    return _nz_matmul_kernel(x, w, out)


@pl.jit.incore
def _nz_matmul_n_sliced_kernel(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16, pl.NZ],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """The same product, but the weight arrives as two N halves.

    The second load's logical row offset ``N_TILE`` must become the fractal
    offset ``N_TILE // 16``. Drop the division and the load reads 64 fractals
    past where it should; drop the offset and both halves read the same rows —
    either way the right half of ``out`` is wrong while the left half still
    matches, so the two-half shape localises the failure.
    """
    xt = pl.load(x, [0, 0], [M, K], target_memory=pl.Mem.Mat)
    lo = pl.load(w, [0, 0], [N_TILE, K], target_memory=pl.Mem.Mat)
    hi = pl.load(w, [N_TILE, 0], [N_TILE, K], target_memory=pl.Mem.Mat)
    out = pl.store(pl.matmul(xt, pl.tile.transpose_view(lo)), [0, 0], out)
    return pl.store(pl.matmul(xt, pl.tile.transpose_view(hi)), [0, N_TILE], out)


@pl.jit
def nz_matmul_n_sliced(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16, pl.NZ],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    return _nz_matmul_n_sliced_kernel(x, w, out)


@pl.jit.incore
def _nz_matmul_k_sliced_kernel(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16, pl.NZ],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """A partial-K product over the upper half of the reduction axis.

    This is the other offset axis: the logical column offset ``K_TILE`` maps to
    the C0 block offset ``K_TILE // c0``, a different slot of the blocked view
    than the row fractal above.
    """
    xt = pl.load(x, [0, K_TILE], [M, K_TILE], target_memory=pl.Mem.Mat)
    wt = pl.load(w, [0, K_TILE], [N, K_TILE], target_memory=pl.Mem.Mat)
    return pl.store(pl.matmul(xt, pl.tile.transpose_view(wt)), [0, 0], out)


@pl.jit
def nz_matmul_k_sliced(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16, pl.NZ],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    return _nz_matmul_k_sliced_kernel(x, w, out)


# ============================================================================
# Cases
# ============================================================================


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small integers, so the FP32 accumulation is exact at rtol=atol=0."""
    generator = torch.Generator().manual_seed(17)
    x = torch.randint(-4, 5, (M, K), generator=generator).to(torch.float16)
    w_logical = torch.randint(-4, 5, (N, K), generator=generator).to(torch.float16)
    return x, _pack_nz(w_logical), torch.zeros((M, N), dtype=torch.float32)


def _full_golden(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    x = tensors["x"].to(torch.float32)
    w = _unpack_nz(tensors["w"]).to(torch.float32)
    return torch.matmul(x, w.T)


def _k_sliced_golden(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    x = tensors["x"].to(torch.float32)[:, K_TILE:]
    w = _unpack_nz(tensors["w"]).to(torch.float32)[:, K_TILE:]
    return torch.matmul(x, w.T)


def _nz_cases():
    for kernel, label, golden in (
        (nz_matmul, "full", _full_golden),
        (nz_matmul_n_sliced, "n_sliced", _full_golden),
        (nz_matmul_k_sliced, "k_sliced", _k_sliced_golden),
    ):
        yield st.case(
            kernel,
            *_inputs(),
            name=f"matmul_nz_{label}_{M}x{K}x{N}",
            golden=golden,
            rtol=0.0,
            atol=0.0,
        )


def test_pack_places_each_element_at_its_blocked_offset():
    """The host packer is this test's own premise, so pin it independently.

    A roundtrip alone would accept any self-inverse permutation (a plain
    transpose, say), so check the absolute destination: logical ``(r, c)`` with
    ``r = i*16 + ii`` and ``c = j*c0 + jj`` must land at flat offset
    ``j*(R*c0) + i*(16*c0) + ii*c0 + jj``.
    """
    rows, cols = 64, 128
    # int16 shares FP16's 2-byte width, hence its c0, and indexes every element
    # of this shape exactly — which FP16 itself cannot do past 2048.
    logical = torch.arange(rows * cols, dtype=torch.int16).reshape(rows, cols)
    c0 = _c0_elems(logical.dtype)
    assert c0 == 16

    flat = _pack_nz(logical).reshape(-1)
    for r, c in ((0, 0), (0, c0), (1, 0), (17, 33), (rows - 1, cols - 1)):
        i, ii = divmod(r, _FRACTAL_ROWS)
        j, jj = divmod(c, c0)
        offset = j * (rows * c0) + i * (_FRACTAL_ROWS * c0) + ii * c0 + jj
        assert flat[offset] == logical[r, c], f"({r}, {c}) is not at blocked offset {offset}"

    assert torch.equal(_unpack_nz(_pack_nz(logical)), logical)


@st.cases(*_nz_cases())
def test_matmul_nz(case_run):
    """An NZ weight is consumed from its fractal layout with no conversion."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
