# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared host-side data helpers for the A5 MX matmul STs.

Not a test module (leading underscore, no ``test_`` prefix), so pytest does not
collect it. It exists because the Tile-surface cases
(``test_matmul_mx.py``) and the Tensor-surface cases
(``test_matmul_mx_tensor_a5.py``) must run on **the same bytes**: the Tensor
dispatch takes whole tensors rather than windows, so its operands are the exact
``[M, K]`` / ``[K, N]`` window the Tile cases read through their offsets. Both
modules import ``_make_inputs`` from here, which makes "same inputs" literal
rather than two generators that happen to agree today.

The two modules live in separate pytest sessions on CI (separate
``onboard-tests-a5`` steps); sharing the data layer is what keeps that split
from turning into two drifting copies.
"""

import pypto.language as pl
import torch
from harness.core.harness import DataType

REQUIRED_TORCH_DTYPES = ("float4_e2m1fn_x2", "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")

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
# (lhs, rhs, short id) -- the four pairs, shared by the Tile and Tensor cases.
_MX_FP8_PAIR_VALUES = (
    (DataType.FP8E4M3FN, DataType.FP8E4M3FN, "e4m3-x-e4m3"),
    (DataType.FP8E5M2, DataType.FP8E5M2, "e5m2-x-e5m2"),
    (DataType.FP8E4M3FN, DataType.FP8E5M2, "e4m3-x-e5m2"),
    (DataType.FP8E5M2, DataType.FP8E4M3FN, "e5m2-x-e4m3"),
)

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


def _make_inputs(lhs_dtype: DataType, rhs_dtype: DataType) -> tuple[torch.Tensor, ...]:
    """Build deterministic host-prequantized inputs for one dtype pair.

    Shared by the Tile-surface and the Tensor-surface cases so both run on the
    same bytes; the Tensor case reads the window out of these (see
    ``_window_operands``).
    """
    generator = torch.Generator().manual_seed(
        23 + 7 * _FP8_DTYPES.index(lhs_dtype) + 3 * _FP8_DTYPES.index(rhs_dtype)
    )
    extremes = _FORMAT_EXTREMES[lhs_dtype]

    a = _tile_choices(_exact_fp8(_ORDINARY_VALUES, lhs_dtype), (A_ROWS, A_COLS), generator)
    b = _tile_choices(_exact_fp8(_RHS_VALUES, rhs_dtype), (B_ROWS, B_COLS), generator)
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
            _exact_fp8(extremes[key], lhs_dtype), (M, MX_GROUP_SIZE), generator
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
