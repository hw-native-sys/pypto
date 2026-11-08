# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 device coverage for the MX GEMV family: `C[1,N] = (A[1,K]*SA) @ (B[K,N]*SB)`.

`tile.gemv_mx` / `_acc` / `_bias` take FP8E4M3FN or FP8E5M2 on each operand
independently, so all four pairs run here in all three forms against an FP32
torch golden.

There is deliberately **no segmented-K case here**, and the omission is a
toolchain limit rather than an oversight. A multi-chunk MX reduction needs a
fresh `tile.tget_scale_addr` binding per chunk, because each chunk carries its
own scale tiles; `InsertMxScaleAddr` can only avoid re-binding after an MX
consumer when the binding is *unchanged*, which segmented K makes impossible. The
resulting `tget_scale_addr / consumer / tget_scale_addr / consumer` sequence is
the shape PTOAS hoists back to the original scale fill (PTOAS #1691), so chunk 1
reads chunk 0's scales for part of its output. On device that is
nondeterministic: a two-chunk K=1024 case passed only 3 of 6 runs, and failed
identically with the phase bits removed, so it is not an `acc_phase` problem.
That shape is now rejected at compile time rather than emitted, and the rejection
is pinned in `tests/ut/codegen/test_mx_ops_codegen.py`; re-add a device case here
when #1691 is fixed. This file therefore has NO segmented-K device coverage, and
no other document should claim it does.

Three shape facts drive the constants below, none of them a tuning choice:

- **Physical K = 512.** The Mat-to-Left extract of the single-row lhs is a vector
  load whose `static_assert` requires `Cols % (512 / sizeof(DType)) == 0`
  (pto-isa `npu/a5/TExtractCommon.hpp:182-183`), and these operands are 1 byte.
- **Valid K == physical K.** A tile MX program with a narrower valid K compiles
  but faults the card for every FP8 pair, so no such shape is constructed here.
  The compile-side contract for it lives in the codegen tests instead.
- **Valid N == physical N.** A narrowed N cannot be assembled for a5: making the
  accumulator's logical N static turns on PTOAS's `rhs logical N == dst logical N`
  rule, which then reads the rhs type's *physical* N. The accumulator bridge
  therefore declines for a narrowed N, which the codegen tests pin.

The A-side scale is a **plain ND** FP8E8M0 tensor, not `pl.MX_A_ZZ`: that layout
interleaves 16 rows x 2 groups into each 32-byte fractal, which is meaningless
for the one row a GEMV multiplies, and the hardware reads a one-row scale as
`ceil(K/32)` contiguous bytes. The B-side keeps `pl.MX_B_NN`, so both scale
transports are exercised. Data and scales are read at non-zero GM offsets, so a
dropped offset term cannot pass.

Exactness (`rtol=atol=0`) is deliberate and constrains the inputs: every decoded
product is a multiple of 2**-4 and every dot product stays well under 2**24, so
the whole FP32 accumulation is exact and any difference is a real bug rather than
rounding. Comparison is on **bytes**, not `torch.isclose`, which treats `-0.0`
and `+0.0` as equal at any tolerance and so cannot see a sign-bit error; the
golden additionally asserts no expected element is zero, keeping that blind spot
out of the comparison by construction rather than by luck.
"""

import pypto.language as pl
import pytest
import torch
from harness import st

_REQUIRED_TORCH_DTYPES = ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch MXFP8 / E8M0 dtypes required", allow_module_level=True)

K, N = 512, 96
MX_GROUP_SIZE = 32
G = K // MX_GROUP_SIZE
# MX_B_NN physical SFractal: 16 columns x 2 groups.
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2

# Windows are read out of larger GM tensors. MX_B_NN constrains its offsets to
# group % 2 and column % 16; the plain-ND A scale has no such rule but still
# reads off origin.
A_COLS = 2 * K
B_ROWS, B_COLS = 2 * K, 2 * N
A_OFF = K
A_SCALE_OFF = G
B_OFF = (K, N)
B_SCALE_OFF = (G, N)

_E8M0_BIAS = 127
# Ordinary groups: 2**-1 .. 2**2, varying per group and per column.
_ORDINARY_SCALE_CODES = (_E8M0_BIAS - 1, _E8M0_BIAS + 2)

# Exactly representable in both FP8 formats and in FP32; all multiples of 0.5.
_ORDINARY_VALUES = (0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0)
# The rhs stays ordinary so the magnitude budget holds whatever the lhs carries.
_RHS_VALUES = (0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0)

# Per format: values past the *other* format's range, that format's subnormals,
# and the E8M0 code that pulls each group back into the exact-sum budget.
# E5M2 has 2 mantissa bits and min normal 2**-14; E4M3FN has 3 and min normal 2**-6.
_FORMAT_EXTREMES = {
    torch.float8_e5m2: {
        # Above E4M3FN's 448 maximum; scaled by 2**-10 to land in [0.5, 1].
        "large": (512.0, -512.0, 768.0, -1024.0, 640.0, -896.0),
        "large_scale_code": _E8M0_BIAS - 10,
        # Subnormals 2**-16 * {1, 2, 3}; scaled by 2**16 to land on {1, 2, 3}.
        "subnormal": (2.0**-16, -(2.0**-16), 2.0**-15, -(2.0**-15), 3 * 2.0**-16),
        "subnormal_scale_code": _E8M0_BIAS + 16,
    },
    torch.float8_e4m3fn: {
        # Beyond FP16-free territory for an 8-bit format; scaled by 2**-8.
        "large": (256.0, -256.0, 384.0, -384.0, 448.0, -320.0),
        "large_scale_code": _E8M0_BIAS - 8,
        # Subnormals 2**-9 * {1..7}; scaled by 2**9 to land on {1..7}.
        "subnormal": (2.0**-9, -(2.0**-9), 3 * 2.0**-9, -(5 * 2.0**-9), 7 * 2.0**-9),
        "subnormal_scale_code": _E8M0_BIAS + 9,
    },
}

_PL_DTYPES = {torch.float8_e4m3fn: pl.FP8E4M3FN, torch.float8_e5m2: pl.FP8E5M2}
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _pack_b_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical B scales `[G, N]` into the MX_B_NN physical layout."""
    k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    return (
        scale_codes.reshape(
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(k_groups, n)
    )


def _exact_fp8(values: tuple[float, ...], dtype: torch.dtype) -> torch.Tensor:
    """Cast *values* to *dtype*, asserting every one survives the round trip.

    A value the format cannot hold would silently become a neighbour and make the
    exact golden unreachable for a reason unrelated to the kernel.
    """
    reference = torch.tensor(values, dtype=torch.float64)
    encoded = reference.to(dtype)
    assert torch.equal(encoded.to(torch.float64), reference), (
        f"{values} is not exactly representable in {dtype}"
    )
    return encoded


def _tile_choices(values: torch.Tensor, shape: tuple[int, int], generator: torch.Generator):
    """Pick from *values* into *shape*, deterministically."""
    picks = torch.randint(0, values.numel(), shape, generator=generator)
    return values[picks]


def _assert_no_zeros(tensor: torch.Tensor, what: str) -> None:
    """Keep the signed-zero blind spot out of the comparison by construction.

    A zero expected element is the one value whose byte comparison could fail on a
    correct result (+0.0 vs -0.0) -- and the one whose tolerance comparison proves
    nothing. If the operand tables ever drift into producing one, this fails loudly
    at golden time rather than turning the case into a coin flip.
    """
    assert (tensor != 0).all(), f"{what} must have no zero element; operand tables need adjusting"


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


def _gemv_mx_golden(
    a: torch.Tensor,
    a_scale_codes: torch.Tensor,
    b: torch.Tensor,
    b_scale_codes: torch.Tensor,
) -> torch.Tensor:
    """MX GEMV from decoded values and logical per-group E8M0 scales.

    Decoding in float64 and only then rounding to FP32 keeps the reference
    independent of the Cube's accumulation order, which the exactness budget
    makes irrelevant anyway.
    """
    k_group = torch.arange(a.shape[1]) // MX_GROUP_SIZE
    a_scale = torch.pow(2.0, a_scale_codes.to(torch.float64) - _E8M0_BIAS)
    b_scale = torch.pow(2.0, b_scale_codes.to(torch.float64) - _E8M0_BIAS)
    a_scaled = a.to(torch.float64) * a_scale[:, k_group]
    b_scaled = b.to(torch.float64) * b_scale[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def _make_inputs(lhs_dtype: torch.dtype, rhs_dtype: torch.dtype):
    """Build deterministic host-prequantized inputs for one dtype pair."""
    generator = torch.Generator().manual_seed(
        29 + 7 * _FP8_DTYPES.index(lhs_dtype) + 3 * _FP8_DTYPES.index(rhs_dtype)
    )
    extremes = _FORMAT_EXTREMES[lhs_dtype]

    a = _tile_choices(_exact_fp8(_ORDINARY_VALUES, lhs_dtype), (1, A_COLS), generator)
    b = _tile_choices(_exact_fp8(_RHS_VALUES, rhs_dtype), (B_ROWS, B_COLS), generator)
    a_scale_codes = torch.randint(
        _ORDINARY_SCALE_CODES[0],
        _ORDINARY_SCALE_CODES[1] + 1,
        (1, A_COLS // MX_GROUP_SIZE),
        generator=generator,
    ).to(torch.uint8)
    b_scale_codes = torch.randint(
        _ORDINARY_SCALE_CODES[0],
        _ORDINARY_SCALE_CODES[1] + 1,
        (B_ROWS // MX_GROUP_SIZE, B_COLS),
        generator=generator,
    ).to(torch.uint8)

    # Groups 0 and 1 of the *window* carry the lhs format's extremes, each with
    # the compensating scale that keeps its contribution inside the exact-sum
    # budget -- which also makes the scales vary per group, as the ISA requires.
    for offset, key, code_key in (
        (0, "large", "large_scale_code"),
        (1, "subnormal", "subnormal_scale_code"),
    ):
        k_from = A_OFF + offset * MX_GROUP_SIZE
        a[:, k_from : k_from + MX_GROUP_SIZE] = _tile_choices(
            _exact_fp8(extremes[key], lhs_dtype), (1, MX_GROUP_SIZE), generator
        )
        a_scale_codes[:, A_SCALE_OFF + offset] = extremes[code_key]

    assert torch.unique(a_scale_codes).numel() > 1
    assert torch.unique(b_scale_codes).numel() > 1
    # Left and right scales must not be mistakable for one another.
    window_a = a_scale_codes[:, A_SCALE_OFF : A_SCALE_OFF + G].flatten()
    window_b = b_scale_codes[B_SCALE_OFF[0] : B_SCALE_OFF[0] + G, B_SCALE_OFF[1] : B_SCALE_OFF[1] + N]
    assert not torch.equal(window_a, window_b.flatten()[: window_a.numel()])

    # Multiples of 0.25 keep the bias add exact alongside the product sum.
    bias = torch.randint(-256, 257, (1, N), generator=generator).to(torch.float32) * 0.25
    return a, a_scale_codes, b, b_scale_codes, bias


def _forms_kernel(lhs_dtype: torch.dtype, rhs_dtype: torch.dtype):
    """All three MX GEMV forms over one operand pair, in one kernel.

    Built here rather than written out per pair because only the operand dtypes
    vary; they are closure constants the DSL parser resolves while tracing the
    annotations.
    """
    lhs_pl = _PL_DTYPES[lhs_dtype]
    rhs_pl = _PL_DTYPES[rhs_dtype]

    @pl.jit.incore
    def gemv_mx_forms_kernel(
        a: pl.Tensor[[1, A_COLS], lhs_pl],
        a_scale: pl.Tensor[[1, A_COLS // MX_GROUP_SIZE], pl.FP8E8M0],
        b: pl.Tensor[[B_ROWS, B_COLS], rhs_pl],
        b_scale: pl.Tensor[[B_ROWS // MX_GROUP_SIZE, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, N], pl.FP32],
        out_base: pl.Out[pl.Tensor[[1, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[1, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    ):
        # Data operands land in Mat (L1); the layout passes insert the L0
        # Left/Right extracts, and the single-row lhs takes the Mat-to-Left
        # vector path. The A scale is a plain ND one-row FP8E8M0 window, the B
        # scale an MX_B_NN one -- both reach LeftScale/RightScale through the
        # ordinary Mat staging move.
        lhs = pl.load(a, [0, A_OFF], [1, K], target_memory=pl.MemorySpace.Mat)
        lhs_scale = pl.load(a_scale, [0, A_SCALE_OFF], [1, G], target_memory=pl.MemorySpace.Mat)
        rhs = pl.load(b, [B_OFF[0], B_OFF[1]], [K, N], target_memory=pl.MemorySpace.Mat)
        rhs_scale = pl.load(b_scale, [B_SCALE_OFF[0], B_SCALE_OFF[1]], [G, N])
        bias_tile = pl.load(bias, [0, 0], [1, N], target_memory=pl.MemorySpace.Mat)

        base = pl.tile.gemv_mx(lhs, lhs_scale, rhs, rhs_scale)
        out_base = pl.store(base, [0, 0], out_base)
        # acc seeds from the accumulator product, so the expected result is exactly 2x base.
        accumulated = pl.tile.gemv_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
        out_acc = pl.store(accumulated, [0, 0], out_acc)
        biased = pl.tile.gemv_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile)
        out_bias = pl.store(biased, [0, 0], out_bias)
        return out_base, out_acc, out_bias

    @pl.jit
    def gemv_mx_forms(
        a: pl.Tensor[[1, A_COLS], lhs_pl],
        a_scale: pl.Tensor[[1, A_COLS // MX_GROUP_SIZE], pl.FP8E8M0],
        b: pl.Tensor[[B_ROWS, B_COLS], rhs_pl],
        b_scale: pl.Tensor[[B_ROWS // MX_GROUP_SIZE, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, N], pl.FP32],
        out_base: pl.Out[pl.Tensor[[1, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[1, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[1, N], pl.FP32]],
    ):
        return gemv_mx_forms_kernel(a, a_scale, b, b_scale, bias, out_base, out_acc, out_bias)

    return gemv_mx_forms


def _window(a, a_scale_codes, b, b_scale_codes):
    """The logical windows the kernel reads, in logical (unpacked) scale order."""
    return (
        a[:, A_OFF : A_OFF + K],
        a_scale_codes[:, A_SCALE_OFF : A_SCALE_OFF + G],
        b[B_OFF[0] : B_OFF[0] + K, B_OFF[1] : B_OFF[1] + N],
        b_scale_codes[B_SCALE_OFF[0] : B_SCALE_OFF[0] + G, B_SCALE_OFF[1] : B_SCALE_OFF[1] + N],
    )


def _forms_case(lhs_dtype: torch.dtype, rhs_dtype: torch.dtype, label: str) -> st.Case:
    a, a_scale_codes, b, b_scale_codes, bias = _make_inputs(lhs_dtype, rhs_dtype)
    base = _gemv_mx_golden(*_window(a, a_scale_codes, b, b_scale_codes))
    _assert_no_zeros(base, f"gemv_mx product ({label})")
    _assert_no_zeros(base + bias, f"gemv_mx_bias result ({label})")
    expected = {"out_base": base, "out_acc": 2 * base, "out_bias": base + bias}
    zero = torch.zeros(1, N, dtype=torch.float32)
    return st.case(
        _forms_kernel(lhs_dtype, rhs_dtype),
        a,
        # Plain ND: a one-row A scale has no SFractal packing, so the logical
        # codes *are* the physical bytes. Only the B scale is packed.
        a_scale_codes.view(torch.float8_e8m0fnu),
        b,
        _pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu),
        bias,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        name=f"gemv_mx_a5_forms_{label}",
        platform="a5",
        golden=lambda _tensors: expected,
        compare=_bytes_equal,
    )


_FORM_CASES = [
    _forms_case(torch.float8_e4m3fn, torch.float8_e4m3fn, "e4m3_x_e4m3"),
    _forms_case(torch.float8_e4m3fn, torch.float8_e5m2, "e4m3_x_e5m2"),
    _forms_case(torch.float8_e5m2, torch.float8_e4m3fn, "e5m2_x_e4m3"),
    _forms_case(torch.float8_e5m2, torch.float8_e5m2, "e5m2_x_e5m2"),
]


@pytest.mark.platforms("a5")
class TestGemvMxA5:
    """MX GEMV on an A5 device: all four FP8 pairs in the base, acc and bias forms."""

    @st.cases(*_FORM_CASES)
    def test_fp8_pairs_in_every_form(self, case_run):
        case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
