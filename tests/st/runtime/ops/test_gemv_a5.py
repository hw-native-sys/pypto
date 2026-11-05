# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 device coverage for the ordinary GEMV family: `C[1,N] = A[1,K] @ B[K,N]`.

This is the **first** device coverage of `tile.gemv` / `_acc` / `_bias` on A5,
for every dtype and not only the FP8 pairs this change adds. Until the
static-valid accumulator bridge (`EmitStaticValidGemvAccView`,
src/backend/common/pto_ops_elementwise.cpp) landed, PTOAS v0.72 rejected every
ordinary `pto.tgemv*` for a5 at every dtype: `verifyGemvTileOperandsA5` chains
into the matmul operand verifier, whose "static matmul tile shapes lhs[M,K],
rhs[K,N], and dst[M,N]" rule reads the *physical* extent when the type's valid
is dynamic, and a GEMV dst is the mandatory 16-row Acc box. So the FP16 / BF16 /
FP32 / INT8 cases below are the regression guard for that bridge, and the FP8
cases are the new operand pairs riding on it.

Physical K is a function of the operand dtype, not a constant. The Mat-to-Left
extract of the single-row lhs goes through `TExtractToAVector` (pto-isa
`npu/a5/TExtractCommon.hpp:182-183`), which `static_assert`s
`Cols % (512 / sizeof(DType)) == 0` on both the Mat source and the Left
destination -- i.e. the lhs needs a whole 512-byte block of physical K. So a
1-byte operand needs K = 512, a 2-byte one 256, a 4-byte one 128, and `_ALIGNED_K`
below is that rule rather than a tuning choice.

Why the operands look the way they do:

- A **dyadic** table (multiples of 2**-3, magnitudes up to 6, both signs) drives
  every case. Each product is a multiple of 2**-6 bounded by 36, so even the
  longest contraction here (K = 1024, the split-K total) needs at most
  1024 * 36 = 36864 of them, i.e. 2359296 units of 2**-6 -- far under 2**24,
  hence exactly representable in FP32 whatever order the Cube accumulates in.
  Every case therefore asserts **byte** equality, not a tolerance.
- Byte equality rather than `torch.isclose`: `isclose(-0.0, +0.0)` is True even at
  rtol=atol=0, so a tolerance-based check is blind to a sign-bit error. The golden
  additionally asserts no expected element is zero, which keeps that blind spot
  out of the comparison by construction rather than by luck.

Two gaps, both deliberate and both recorded in unit tests rather than skipped here:

- `HF8 x HF8` is a table pair but has no ST carrier (torch has no `hif8` dtype),
  so it is covered by operand-pair inference and by real-PTOAS assembly only --
  the same treatment ordinary matmul gives it.
- A **partially-valid N** cannot be assembled for a5 even with the bridge: making
  the dst's logical N static turns on the rule's `rhs logical N == dst logical N`
  comparison, and the rhs type's valid stays dynamic so that rule reads its
  physical N. The bridge therefore declines to fire for a narrowed N (see
  `test_a5_gemv_skips_the_bridge_for_a_narrowed_n` in
  tests/ut/codegen/test_pto_codegen_ops.py), leaving that case exactly as
  unassemblable as it was before. Fixing it needs every GEMV operand re-typed,
  not just the accumulator.
"""

import pypto.language as pl
import pytest
import torch
from harness import st

if not hasattr(torch, "float8_e5m2"):
    pytest.skip("torch.float8_e5m2 required", allow_module_level=True)

N = 64
SPLIT_CHUNKS = 2

# Physical K per operand byte width, from the TExtractToAVector static_assert
# quoted in the module docstring: the lhs needs a whole 512-byte K block.
_ALIGNED_K = {1: 512, 2: 256, 4: 128}

# Multiples of 2**-3, exact in every float dtype under test. Both signs, so a
# sign-bit error in either operand changes the result.
_DYADIC = [0.25, -0.375, 1.0, -1.5, 3.0, -6.0, 0.5, -2.0, 4.0, -0.125, 6.0, -3.0, 2.0, -1.0]
# The integral subset, for INT8 operands.
_INTEGRAL = [1.0, -2.0, 3.0, -1.0, 2.0, -3.0, 4.0, -6.0, 6.0, -4.0, 1.0, -3.0, 2.0, -1.0]


def _aligned_k(dtype: torch.dtype) -> int:
    return _ALIGNED_K[torch.empty(0, dtype=dtype).element_size()]


def _fill(rows: int, cols: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Tile the dyadic table over a [rows, cols] operand, visiting every entry.

    Stride 5 is coprime with the 14-entry table, so the walk covers it whole. An
    even stride would hold the sign parity fixed and silently yield a
    single-signed operand, which no sign-bit error could show up in.
    """
    values = _INTEGRAL if dtype == torch.int8 else _DYADIC
    assert len(values) % 5 != 0, "stride 5 must be coprime with the table size"
    index = (torch.arange(rows * cols) * 5 + seed) % len(values)
    exact = torch.tensor(values, dtype=torch.float32)[index].reshape(rows, cols)
    data = exact.to(dtype)
    # The table must survive the round trip, or the case would be about rounding
    # rather than about the GEMV operand pair.
    torch.testing.assert_close(data.float(), exact, rtol=0, atol=0)
    assert (data.float() > 0).any() and (data.float() < 0).any(), "operand must cover both signs"
    return data


def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.int32 if dtype == torch.int8 else torch.float32


def _reference(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """The GEMV the Cube computes: widen both operands, accumulate in FP32 / INT32.

    Torch has no matmul kernel for 1-byte floats and refuses a mixed float pair,
    and an int8 matmul would wrap rather than widen, so both operands widen first
    -- which is what the hardware accumulates in anyway.
    """
    if lhs.dtype == torch.int8:
        return (lhs.to(torch.int32) @ rhs.to(torch.int32)).to(torch.int32)
    return (lhs.float() @ rhs.float()).to(torch.float32)


def _bytes_equal(actual: dict, expected: dict) -> None:
    """Byte equality on every output.

    `torch.isclose` -- the harness default -- treats `-0.0` and `+0.0` as equal at
    any tolerance, so it cannot see a sign-bit error. Comparing the raw bytes can.
    """
    for name, got in actual.items():
        ref = expected[name]
        assert got.dtype == ref.dtype, f"{name}: dtype {got.dtype} != {ref.dtype}"
        assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), (
            f"{name}: bytes differ\n  actual   {got}\n  expected {ref}"
        )


def _assert_no_zeros(tensor: torch.Tensor, what: str) -> None:
    """Keep the signed-zero blind spot out of the comparison by construction.

    A zero expected element is the one value whose byte comparison could fail on a
    correct result (+0.0 vs -0.0) -- and the one whose tolerance comparison proves
    nothing. If the operand tables ever drift into producing one, this fails loudly
    at golden time rather than turning the case into a coin flip.
    """
    assert (tensor != 0).all(), f"{what} must have no zero element; operand tables need adjusting"


# ===========================================================================
# gemv / gemv_acc / gemv_bias, one kernel per operand pair
# ===========================================================================


def _make_forms_kernel(lhs_pl, rhs_pl, out_pl, k: int):
    """All three GEMV forms over one operand pair, in one kernel.

    Built here rather than written out per pair because only the dtypes and the
    dtype-dependent K vary; they are closure constants the DSL parser resolves
    while tracing the annotations.
    """

    @pl.jit.incore
    def gemv_forms_kernel(
        a: pl.Tensor[[1, k], lhs_pl],
        b: pl.Tensor[[k, N], rhs_pl],
        bias: pl.Tensor[[1, N], out_pl],
        out_base: pl.Out[pl.Tensor[[1, N], out_pl]],
        out_acc: pl.Out[pl.Tensor[[1, N], out_pl]],
        out_bias: pl.Out[pl.Tensor[[1, N], out_pl]],
    ):
        # Operands land in Mat (L1); the layout passes insert the L0 Left/Right
        # extracts, and the single-row lhs takes the Mat-to-Left vector path.
        lhs = pl.load(a, [0, 0], [1, k], target_memory=pl.MemorySpace.Mat)
        rhs = pl.load(b, [0, 0], [k, N], target_memory=pl.MemorySpace.Mat)
        bias_tile = pl.load(bias, [0, 0], [1, N], target_memory=pl.MemorySpace.Mat)
        base = pl.tile.gemv(lhs, rhs)
        out_base = pl.store(base, [0, 0], out_base)
        # acc seeds from the accumulator product, so the expected result is exactly 2x base.
        accumulated = pl.tile.gemv_acc(base, lhs, rhs)
        out_acc = pl.store(accumulated, [0, 0], out_acc)
        biased = pl.tile.gemv_bias(lhs, rhs, bias_tile)
        out_bias = pl.store(biased, [0, 0], out_bias)
        return out_base, out_acc, out_bias

    @pl.jit
    def gemv_forms(
        a: pl.Tensor[[1, k], lhs_pl],
        b: pl.Tensor[[k, N], rhs_pl],
        bias: pl.Tensor[[1, N], out_pl],
        out_base: pl.Out[pl.Tensor[[1, N], out_pl]],
        out_acc: pl.Out[pl.Tensor[[1, N], out_pl]],
        out_bias: pl.Out[pl.Tensor[[1, N], out_pl]],
    ):
        return gemv_forms_kernel(a, b, bias, out_base, out_acc, out_bias)

    return gemv_forms


def _forms_golden(tensors: dict) -> dict:
    product = _reference(tensors["a"], tensors["b"])
    bias = tensors["bias"]
    _assert_no_zeros(product, "gemv product")
    _assert_no_zeros(product + bias, "gemv_bias result")
    return {"out_base": product, "out_acc": 2 * product, "out_bias": product + bias}


def _forms_case(label: str, lhs_pl, rhs_pl, lhs_torch, rhs_torch, seeds=(3, 11)) -> st.Case:
    out_torch = _acc_dtype(lhs_torch)
    out_pl = pl.INT32 if out_torch == torch.int32 else pl.FP32
    k = _aligned_k(lhs_torch)
    a = _fill(1, k, lhs_torch, seeds[0])
    b = _fill(k, N, rhs_torch, seeds[1])
    # Small odd integers: exact in both accumulator types, and the products are
    # even multiples of 2**-6, so no sum can be cancelled to zero by the bias.
    bias = ((torch.arange(N) % 5) * 2 + 1).reshape(1, N).to(out_torch)
    zero = torch.zeros(1, N, dtype=out_torch)
    return st.case(
        _make_forms_kernel(lhs_pl, rhs_pl, out_pl, k),
        a,
        b,
        bias,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        name=f"gemv_a5_forms_{label}",
        platform="a5",
        golden=_forms_golden,
        compare=_bytes_equal,
    )


# The four FP8 pairs this change adds.
_FP8_CASES = [
    _forms_case("e4m3_x_e4m3", pl.FP8E4M3FN, pl.FP8E4M3FN, torch.float8_e4m3fn, torch.float8_e4m3fn),
    _forms_case("e4m3_x_e5m2", pl.FP8E4M3FN, pl.FP8E5M2, torch.float8_e4m3fn, torch.float8_e5m2),
    _forms_case("e5m2_x_e4m3", pl.FP8E5M2, pl.FP8E4M3FN, torch.float8_e5m2, torch.float8_e4m3fn),
    _forms_case("e5m2_x_e5m2", pl.FP8E5M2, pl.FP8E5M2, torch.float8_e5m2, torch.float8_e5m2),
]

# The dtypes A5 always had on paper. These are the regression guard for the
# static-valid Acc bridge, which is what makes any of them assemble for a5.
_WIDE_CASES = [
    _forms_case("fp16", pl.FP16, pl.FP16, torch.float16, torch.float16),
    _forms_case("bf16", pl.BF16, pl.BF16, torch.bfloat16, torch.bfloat16),
    _forms_case("fp32", pl.FP32, pl.FP32, torch.float32, torch.float32),
    # INT8 needs its own seed pair. Its table is integral, so the K-long sums land
    # on a coarse lattice and the default seeds put a zero in one output column --
    # the one value the byte comparison cannot judge (see `_assert_no_zeros`).
    _forms_case("int8", pl.INT8, pl.INT8, torch.int8, torch.int8, seeds=(1, 0)),
]


# ===========================================================================
# Segmented-K accumulation and init_cond
# ===========================================================================

# Both split-K kernels are FP8 E4M3FN x E5M2, so the chunk is the 1-byte aligned K.
_KC = _ALIGNED_K[1]
_KT = _KC * SPLIT_CHUNKS


@pl.jit.incore
def gemv_split_k_kernel(
    a: pl.Tensor[[1, _KT], pl.FP8E4M3FN],
    b: pl.Tensor[[_KT, N], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
):
    """Two K chunks: a fresh `gemv` then a `gemv_acc`, with the phase bits set.

    `AccPhase.Partial` on the first chunk and `AccPhase.Final` on the last, with a
    matching `STPhase.Final` on the store -- the producer/consumer unit-flag
    contract this change must leave alone. The second chunk also reads the lhs at
    a non-zero GM column offset, so the Mat-to-Left movement is exercised off
    origin as well as at it.
    """
    a0 = pl.load(a, [0, 0], [1, _KC], target_memory=pl.MemorySpace.Mat)
    b0 = pl.load(b, [0, 0], [_KC, N], target_memory=pl.MemorySpace.Mat)
    acc = pl.tile.gemv(a0, b0, acc_phase=pl.AccPhase.Partial)

    a1 = pl.load(a, [0, _KC], [1, _KC], target_memory=pl.MemorySpace.Mat)
    b1 = pl.load(b, [_KC, 0], [_KC, N], target_memory=pl.MemorySpace.Mat)
    acc = pl.tile.gemv_acc(acc, a1, b1, acc_phase=pl.AccPhase.Final)
    return pl.store(acc, [0, 0], out, st_phase=pl.STPhase.Final)


@pl.jit
def gemv_split_k(
    a: pl.Tensor[[1, _KT], pl.FP8E4M3FN],
    b: pl.Tensor[[_KT, N], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
):
    return gemv_split_k_kernel(a, b, out)


@pl.jit.incore
def gemv_init_cond_kernel(
    a: pl.Tensor[[1, _KT], pl.FP8E4M3FN],
    b: pl.Tensor[[_KT, N], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
):
    """Split-K driven by `init_cond` instead of a peeled first step.

    The accumulator is minted by `tile.create` and never zeroed, so if `init_cond`
    failed to select the overwriting form on `k0 == 0` the result would carry
    whatever L0C held and the byte comparison would fail. It is created at the
    *padded* physical shape -- a `[1, N]` GEMV result occupies 16 Acc rows -- then
    narrowed to its valid `[1, N]` rectangle, which is the type the peeled
    `tile.gemv` produced implicitly.
    """
    acc_raw = pl.tile.create([16, N], pl.FP32, target_memory=pl.MemorySpace.Acc)
    acc = pl.tile.set_validshape(acc_raw, 1, N)
    for k0 in pl.range(0, _KT, _KC):
        a_l1 = pl.load(a, [0, k0], [1, _KC], target_memory=pl.MemorySpace.Mat)
        b_l1 = pl.load(b, [k0, 0], [_KC, N], target_memory=pl.MemorySpace.Mat)
        acc = pl.tile.gemv_acc(acc, a_l1, b_l1, init_cond=(k0 == 0))
    return pl.store(acc, [0, 0], out)


@pl.jit
def gemv_init_cond(
    a: pl.Tensor[[1, _KT], pl.FP8E4M3FN],
    b: pl.Tensor[[_KT, N], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
):
    return gemv_init_cond_kernel(a, b, out)


def _whole_k_golden(tensors: dict) -> torch.Tensor:
    product = _reference(tensors["a"], tensors["b"])
    _assert_no_zeros(product, "gemv product")
    return product


_SPLIT_CASES = [
    st.case(
        gemv_split_k,
        _fill(1, _KT, torch.float8_e4m3fn, seed=7),
        _fill(_KT, N, torch.float8_e5m2, seed=17),
        torch.zeros(1, N, dtype=torch.float32),
        name="gemv_a5_split_k_partial_final",
        platform="a5",
        golden=_whole_k_golden,
        compare=_bytes_equal,
    ),
    st.case(
        gemv_init_cond,
        _fill(1, _KT, torch.float8_e4m3fn, seed=19),
        _fill(_KT, N, torch.float8_e5m2, seed=23),
        torch.zeros(1, N, dtype=torch.float32),
        name="gemv_a5_init_cond_split_k",
        platform="a5",
        golden=_whole_k_golden,
        compare=_bytes_equal,
    ),
]


@pytest.mark.platforms("a5")
class TestGemvA5:
    """Ordinary GEMV on an A5 device: the new FP8 pairs and every pre-existing dtype."""

    @st.cases(*_FP8_CASES)
    def test_fp8_operand_pairs(self, case_run):
        case_run.assert_passed()

    @st.cases(*_WIDE_CASES)
    def test_pre_existing_dtypes_still_run(self, case_run):
        case_run.assert_passed()

    @st.cases(*_SPLIT_CASES)
    def test_segmented_k_accumulation_and_init_cond(self, case_run):
        case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
