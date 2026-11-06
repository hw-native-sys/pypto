# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 runtime tests for host-prequantized MX matmul.

A5 MX matmul takes FP8E4M3FN or FP8E5M2 on each operand independently, so all
four FP8 pairs run here against an FP32 torch golden, in the base, accumulating
and bias forms. Native packed-FP4 matmul is intentionally outside the supported
surface; the FP4xFP8 case uses packed FP4E2M1X2 carrier dimensions and
explicitly casts its lhs to FP8E4M3FN.

Exactness (``rtol=atol=0``) is deliberate and constrains the inputs: every
decoded product is a multiple of 2^-4 and every dot product stays below 2^15, so
the whole FP32 accumulation is exact and any numerical difference is a real bug
rather than rounding. Operand values beyond the other format's range (E5M2
reaches 57344, E4M3 stops at 448) and subnormals of both formats would break
that budget if they shared a dot product with ordinary values, so each lives in
its own K group paired with a compensating E8M0 scale -- which also makes the
scales vary per group, as the ISA contract requires.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_REQUIRED_TORCH_DTYPES = ("float4_e2m1fn_x2", "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch MXFP4/MXFP8/E8M0 dtypes required", allow_module_level=True)

M, K, N = 32, 256, 96
MX_GROUP_SIZE = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2
K_GROUPS = K // MX_GROUP_SIZE

# The data and scale windows are read at non-zero offsets out of larger GM
# tensors, so a dropped offset term cannot pass. MX offsets are constrained:
# MX_A_ZZ needs row % 16 and group % 2, MX_B_NN needs group % 2 and col % 16.
A_ROWS, A_COLS = 2 * M, 2 * K
B_ROWS, B_COLS = 2 * K, 2 * N
A_OFF = (M, K)
B_OFF = (K, N)
A_SCALE_OFF = (M, K_GROUPS)
B_SCALE_OFF = (K_GROUPS, N)

_E8M0_BIAS = 127
# Ordinary groups: scale 2^-1 .. 2^2, varying per row/column and group.
_ORDINARY_SCALE_CODES = (_E8M0_BIAS - 1, _E8M0_BIAS + 2)

_FP8_DTYPES = (DataType.FP8E4M3FN, DataType.FP8E5M2)
_MX_FP8_PAIRS = [
    pytest.param(DataType.FP8E4M3FN, DataType.FP8E4M3FN, id="e4m3-x-e4m3"),
    pytest.param(DataType.FP8E5M2, DataType.FP8E5M2, id="e5m2-x-e5m2"),
    pytest.param(DataType.FP8E4M3FN, DataType.FP8E5M2, id="e4m3-x-e5m2"),
    pytest.param(DataType.FP8E5M2, DataType.FP8E4M3FN, id="e5m2-x-e4m3"),
]

# Per format: values past the *other* format's range, that format's subnormals,
# and the E8M0 code that pulls each group back into the exact-sum budget.
# E5M2 has 2 mantissa bits and min normal 2^-14; E4M3FN has 3 and min normal 2^-6.
_FORMAT_EXTREMES = {
    DataType.FP8E5M2: {
        # Above E4M3FN's 448 maximum; scaled by 2^-10 to land in [0.5, 1].
        "large": (512.0, -512.0, 768.0, -1024.0, 640.0, -896.0),
        "large_scale_code": _E8M0_BIAS - 10,
        # Subnormals 2^-16 * {1, 2, 3}; scaled by 2^16 to land on {1, 2, 3}.
        "subnormal": (2.0**-16, -(2.0**-16), 2.0**-15, -(2.0**-15), 3 * 2.0**-16),
        "subnormal_scale_code": _E8M0_BIAS + 16,
    },
    DataType.FP8E4M3FN: {
        # Beyond FP16-free territory for an 8-bit format; scaled by 2^-8.
        "large": (256.0, -256.0, 384.0, -384.0, 448.0, -320.0),
        "large_scale_code": _E8M0_BIAS - 8,
        # Subnormals 2^-9 * {1..7}; scaled by 2^9 to land on {1..7}.
        "subnormal": (2.0**-9, -(2.0**-9), 3 * 2.0**-9, -(5 * 2.0**-9), 7 * 2.0**-9),
        "subnormal_scale_code": _E8M0_BIAS + 9,
    },
}

# Harness dtype -> the DSL dtype its kernel annotation needs.
_PL_DTYPES = {DataType.FP8E4M3FN: pl.FP8E4M3FN, DataType.FP8E5M2: pl.FP8E5M2}

# Exactly representable in both FP8 formats and in FP32; all multiples of 0.5.
_ORDINARY_VALUES = (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0)
# The rhs stays ordinary so the magnitude budget holds whatever the lhs carries.
_RHS_VALUES = (0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0)


def _pack_a_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical A scales into the MX_A_ZZ physical layout."""
    m, k_groups = scale_codes.shape
    assert m % SCALE_BLOCK_SIZE == 0
    assert k_groups % SCALE_C0_SIZE == 0
    return (
        scale_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _unpack_a_scale(packed_codes: torch.Tensor) -> torch.Tensor:
    """Restore MX_A_ZZ physical scale bytes to logical [M, K/32]."""
    m, k_groups = packed_codes.shape
    return (
        packed_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _pack_b_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical B scales into the MX_B_NN physical layout."""
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


def _unpack_b_scale(packed_codes: torch.Tensor) -> torch.Tensor:
    """Restore MX_B_NN physical scale bytes to logical [K/32, N]."""
    k_groups, n = packed_codes.shape
    return (
        packed_codes.reshape(
            n // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(1, 3, 0, 2)
        .contiguous()
        .reshape(k_groups, n)
    )


def _decode_fp4_data(data: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Decode a physical x2 FP4 tensor into its logical float64 matrix."""
    packed = data.contiguous().view(torch.uint8).reshape(rows, cols // 2)
    codes = torch.empty((rows, cols), dtype=torch.long)
    codes[:, 0::2] = (packed & 0x0F).to(torch.long)
    codes[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float64,
    )
    return values[codes]


def _exact_fp8(values: tuple[float, ...], dtype: DataType) -> torch.Tensor:
    """Cast *values* to *dtype*, asserting every one survives the round trip.

    A value the format cannot hold would silently become a neighbour and make the
    exact golden unreachable for a reason unrelated to the kernel.
    """
    reference = torch.tensor(values, dtype=torch.float64)
    encoded = reference.to(dtype.torch_dtype)
    assert torch.equal(encoded.to(torch.float64), reference), (
        f"{values} is not exactly representable in {dtype.value}"
    )
    return encoded


def _tile_choices(values: torch.Tensor, shape: tuple[int, int], generator: torch.Generator):
    """Pick from *values* into *shape*, deterministically."""
    picks = torch.randint(0, values.numel(), shape, generator=generator)
    return values[picks]


class MatmulMxTestCase(PTOTestCase):
    """One host-prequantized MX FP8 pair, covering base, acc and bias outputs."""

    __test__ = False

    def __init__(self, lhs_dtype: DataType, rhs_dtype: DataType):
        if lhs_dtype not in _FP8_DTYPES or rhs_dtype not in _FP8_DTYPES:
            raise ValueError(
                f"A5 MX matmul pairs FP8E4M3FN/FP8E5M2 on each side; got {lhs_dtype}, {rhs_dtype}"
            )
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")
        self._lhs_dtype = lhs_dtype
        self._rhs_dtype = rhs_dtype

    def get_name(self) -> str:
        return f"matmul_mx_{self._lhs_dtype.value}_x_{self._rhs_dtype.value}_base_acc_bias"

    def _make_inputs(self) -> tuple[torch.Tensor, ...]:
        """Build deterministic host-prequantized inputs for this dtype pair."""
        generator = torch.Generator().manual_seed(
            23 + 7 * _FP8_DTYPES.index(self._lhs_dtype) + 3 * _FP8_DTYPES.index(self._rhs_dtype)
        )
        extremes = _FORMAT_EXTREMES[self._lhs_dtype]

        a = _tile_choices(_exact_fp8(_ORDINARY_VALUES, self._lhs_dtype), (A_ROWS, A_COLS), generator)
        b = _tile_choices(_exact_fp8(_RHS_VALUES, self._rhs_dtype), (B_ROWS, B_COLS), generator)

        a_scale_codes = torch.randint(
            _ORDINARY_SCALE_CODES[0],
            _ORDINARY_SCALE_CODES[1] + 1,
            (A_ROWS, A_COLS // MX_GROUP_SIZE),
            generator=generator,
        ).to(torch.uint8)
        b_scale_codes = torch.randint(
            _ORDINARY_SCALE_CODES[0],
            _ORDINARY_SCALE_CODES[1] + 1,
            (B_ROWS // MX_GROUP_SIZE, B_COLS),
            generator=generator,
        ).to(torch.uint8)

        # Groups 0 and 1 of the *window* carry the format's extremes, each with the
        # compensating scale that keeps its contribution inside the exact-sum budget.
        row_lo, k_lo = A_OFF[0], A_OFF[1]
        group_lo = A_SCALE_OFF[1]
        for offset, key, code_key in (
            (0, "large", "large_scale_code"),
            (1, "subnormal", "subnormal_scale_code"),
        ):
            k_from = k_lo + offset * MX_GROUP_SIZE
            a[row_lo : row_lo + M, k_from : k_from + MX_GROUP_SIZE] = _tile_choices(
                _exact_fp8(extremes[key], self._lhs_dtype), (M, MX_GROUP_SIZE), generator
            )
            a_scale_codes[row_lo : row_lo + M, group_lo + offset] = extremes[code_key]

        assert torch.unique(a_scale_codes).numel() > 1
        assert torch.unique(b_scale_codes).numel() > 1
        # Left and right scales must not be mistakable for one another.
        window_a_codes = a_scale_codes[row_lo : row_lo + M, group_lo : group_lo + K_GROUPS]
        window_b_codes = b_scale_codes[
            B_SCALE_OFF[0] : B_SCALE_OFF[0] + K_GROUPS, B_SCALE_OFF[1] : B_SCALE_OFF[1] + N
        ]
        assert not torch.equal(
            window_a_codes.flatten()[: min(window_a_codes.numel(), window_b_codes.numel())],
            window_b_codes.flatten()[: min(window_a_codes.numel(), window_b_codes.numel())],
        )

        a_scale = _pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu)
        b_scale = _pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu)
        # Multiples of 0.25 keep the bias add exact alongside the product sum.
        bias = torch.randint(-256, 257, (1, N), generator=generator).to(torch.float32) * 0.25
        return a, a_scale, b, b_scale, bias

    def define_tensors(self) -> list[TensorSpec]:
        a, a_scale, b, b_scale, bias = self._make_inputs()
        return [
            TensorSpec("a", [A_ROWS, A_COLS], self._lhs_dtype, init_value=a),
            TensorSpec("a_scale", list(a_scale.shape), DataType.FP8E8M0, init_value=a_scale),
            TensorSpec("b", [B_ROWS, B_COLS], self._rhs_dtype, init_value=b),
            TensorSpec("b_scale", list(b_scale.shape), DataType.FP8E8M0, init_value=b_scale),
            TensorSpec("bias", [1, N], DataType.FP32, init_value=bias),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_bias", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _mx_matmul_entry(self._lhs_dtype, self._rhs_dtype).specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        base = _matmul_mx_golden(
            tensors["a"][A_OFF[0] : A_OFF[0] + M, A_OFF[1] : A_OFF[1] + K],
            _unpack_a_scale(tensors["a_scale"].view(torch.uint8))[
                A_SCALE_OFF[0] : A_SCALE_OFF[0] + M, A_SCALE_OFF[1] : A_SCALE_OFF[1] + K_GROUPS
            ],
            tensors["b"][B_OFF[0] : B_OFF[0] + K, B_OFF[1] : B_OFF[1] + N],
            _unpack_b_scale(tensors["b_scale"].view(torch.uint8))[
                B_SCALE_OFF[0] : B_SCALE_OFF[0] + K_GROUPS, B_SCALE_OFF[1] : B_SCALE_OFF[1] + N
            ],
            k_valid=K,
        )
        tensors["out"][:] = base
        tensors["out_acc"][:] = 2 * base
        tensors["out_bias"][:] = base + tensors["bias"].to(torch.float64).to(torch.float32)


def _matmul_mx_golden(
    a: torch.Tensor,
    a_scale_codes: torch.Tensor,
    b: torch.Tensor,
    b_scale_codes: torch.Tensor,
    k_valid: int,
) -> torch.Tensor:
    """MX matmul from decoded values and logical per-group E8M0 scales.

    Only the first *k_valid* columns of ``a`` / rows of ``b`` contribute: PTOAS
    reduces over the operands' valid K, not their physical K.
    """
    a_values = a if a.dtype == torch.float64 else a.to(torch.float64)
    k_group = torch.arange(k_valid) // MX_GROUP_SIZE
    a_scale = torch.pow(2.0, a_scale_codes.to(torch.float64) - _E8M0_BIAS)
    b_scale = torch.pow(2.0, b_scale_codes.to(torch.float64) - _E8M0_BIAS)
    a_scaled = a_values[:, :k_valid] * a_scale[:, k_group]
    b_scaled = b.to(torch.float64)[:k_valid, :] * b_scale[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def _mx_matmul_entry(lhs_dtype: DataType, rhs_dtype: DataType):
    """Build the jitted entry for one FP8 pair.

    The operand dtypes appear in the kernel's annotations, so one entry is
    specialized per pair rather than branching inside a single kernel.
    """
    lhs_pl = _PL_DTYPES[lhs_dtype]
    rhs_pl = _PL_DTYPES[rhs_dtype]

    @pl.jit.incore
    def mx_matmul_kernel(
        a: pl.Tensor[[A_ROWS, A_COLS], lhs_pl],
        a_scale: pl.Tensor[[A_ROWS, A_COLS // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[B_ROWS, B_COLS], rhs_pl],
        b_scale: pl.Tensor[[B_ROWS // 32, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        lhs = pl.load(a, [A_OFF[0], A_OFF[1]], [M, K])
        lhs_scale = pl.load(a_scale, [A_SCALE_OFF[0], A_SCALE_OFF[1]], [M, K_GROUPS])
        rhs = pl.load(b, [B_OFF[0], B_OFF[1]], [K, N])
        rhs_scale = pl.load(b_scale, [B_SCALE_OFF[0], B_SCALE_OFF[1]], [K_GROUPS, N])
        bias_tile = pl.load(bias, [0, 0], [1, N])
        base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(base, [0, 0], out)
        out_acc = pl.store(pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale), [0, 0], out_acc)
        out_bias = pl.store(pl.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile), [0, 0], out_bias)
        return out, out_acc, out_bias

    @pl.jit
    def mx_matmul(
        a: pl.Tensor[[A_ROWS, A_COLS], lhs_pl],
        a_scale: pl.Tensor[[A_ROWS, A_COLS // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[B_ROWS, B_COLS], rhs_pl],
        b_scale: pl.Tensor[[B_ROWS // 32, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        return mx_matmul_kernel(a, a_scale, b, b_scale, bias, out, out_acc, out_bias)

    return mx_matmul


# ---------------------------------------------------------------------------
# Packed FP4 lhs, cast to FP8 before the matmul
# ---------------------------------------------------------------------------
# Native FP4 x FP4 is unsupported: PTOAS emits inconsistent packed operand and
# accumulator dimensions for it. The supported form casts the packed lhs to FP8
# first, so physical K, valid K and the ceil(K/32) group count are all measured
# on the post-cast FP8 tile rather than the x2 carrier.


def _fp4_matmul_entry():
    """Jitted entry for a packed-FP4 lhs explicitly cast to FP8E4M3FN."""

    @pl.jit.incore
    def mxfp4_fp8_kernel(
        a: pl.Tensor[[A_ROWS, A_COLS // 2], pl.FP4E2M1X2],
        a_scale: pl.Tensor[[A_ROWS, A_COLS // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[B_ROWS, B_COLS], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[B_ROWS // 32, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
        # Two FP4 values per carrier byte, so the load window is half as wide
        # along K; the cast restores the logical K the matmul reduces over.
        lhs = pl.cast(pl.load(a, [A_OFF[0], A_OFF[1] // 2], [M, K // 2]), pl.FP8E4M3FN)
        lhs_scale = pl.load(a_scale, [A_SCALE_OFF[0], A_SCALE_OFF[1]], [M, K_GROUPS])
        rhs = pl.load(b, [B_OFF[0], B_OFF[1]], [K, N])
        rhs_scale = pl.load(b_scale, [B_SCALE_OFF[0], B_SCALE_OFF[1]], [K_GROUPS, N])
        base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(base, [0, 0], out)
        out_acc = pl.store(pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale), [0, 0], out_acc)
        return out, out_acc

    @pl.jit
    def mxfp4_fp8_matmul(
        a: pl.Tensor[[A_ROWS, A_COLS // 2], pl.FP4E2M1X2],
        a_scale: pl.Tensor[[A_ROWS, A_COLS // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[B_ROWS, B_COLS], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[B_ROWS // 32, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
        return mxfp4_fp8_kernel(a, a_scale, b, b_scale, out, out_acc)

    return mxfp4_fp8_matmul


class MatmulMxFp4TestCase(PTOTestCase):
    """Packed FP4 lhs x FP8 rhs, with the explicit lhs cast."""

    __test__ = False

    def __init__(self):
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")

    def get_name(self) -> str:
        return "matmul_mx_fp4e2m1x2_x_fp8e4m3fn_base_acc"

    def define_tensors(self) -> list[TensorSpec]:
        generator = torch.Generator().manual_seed(101)
        # Every FP4 code appears, so a decode error cannot hide; all decoded
        # values are multiples of 0.5 and at most 6, keeping the sum exact.
        codes = torch.randint(0, 16, (A_ROWS, A_COLS), generator=generator).to(torch.uint8)
        codes.reshape(-1)[:16] = torch.arange(16, dtype=torch.uint8)
        packed = ((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)
        a = packed.contiguous().view(torch.float4_e2m1fn_x2)
        b = _tile_choices(_exact_fp8(_RHS_VALUES, DataType.FP8E4M3FN), (B_ROWS, B_COLS), generator)
        a_scale_codes = torch.randint(
            _ORDINARY_SCALE_CODES[0],
            _ORDINARY_SCALE_CODES[1] + 1,
            (A_ROWS, A_COLS // MX_GROUP_SIZE),
            generator=generator,
        ).to(torch.uint8)
        b_scale_codes = torch.randint(
            _ORDINARY_SCALE_CODES[0],
            _ORDINARY_SCALE_CODES[1] + 1,
            (B_ROWS // MX_GROUP_SIZE, B_COLS),
            generator=generator,
        ).to(torch.uint8)
        assert torch.unique(a_scale_codes).numel() > 1
        assert torch.unique(b_scale_codes).numel() > 1
        return [
            TensorSpec("a", [A_ROWS, A_COLS // 2], DataType.FP4E2M1X2, init_value=a),
            TensorSpec(
                "a_scale",
                [A_ROWS, A_COLS // MX_GROUP_SIZE],
                DataType.FP8E8M0,
                init_value=_pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("b", [B_ROWS, B_COLS], DataType.FP8E4M3FN, init_value=b),
            TensorSpec(
                "b_scale",
                [B_ROWS // MX_GROUP_SIZE, B_COLS],
                DataType.FP8E8M0,
                init_value=_pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _fp4_matmul_entry().specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a_logical = _decode_fp4_data(tensors["a"], A_ROWS, A_COLS)
        base = _matmul_mx_golden(
            a_logical[A_OFF[0] : A_OFF[0] + M, A_OFF[1] : A_OFF[1] + K],
            _unpack_a_scale(tensors["a_scale"].view(torch.uint8))[
                A_SCALE_OFF[0] : A_SCALE_OFF[0] + M, A_SCALE_OFF[1] : A_SCALE_OFF[1] + K_GROUPS
            ],
            tensors["b"][B_OFF[0] : B_OFF[0] + K, B_OFF[1] : B_OFF[1] + N],
            _unpack_b_scale(tensors["b_scale"].view(torch.uint8))[
                B_SCALE_OFF[0] : B_SCALE_OFF[0] + K_GROUPS, B_SCALE_OFF[1] : B_SCALE_OFF[1] + N
            ],
            k_valid=K,
        )
        tensors["out"][:] = base
        tensors["out_acc"][:] = 2 * base


@pytest.mark.platforms("a5")
class TestMatmulMx:
    """Numerical execution coverage for the four A5 MX FP8 pairs."""

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MX_FP8_PAIRS)
    def test_matmul_mx_base_acc_and_bias(self, test_runner, lhs_dtype, rhs_dtype):
        result = test_runner.run(MatmulMxTestCase(lhs_dtype, rhs_dtype))
        assert result.passed, f"Test failed: {result.error}"

    def test_mxfp4_lhs_cast_to_fp8(self, test_runner):
        result = test_runner.run(MatmulMxFp4TestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
