# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical validation of the FP32 sin/cos recipe used by LowerCompositeOps.

This test file mirrors the exact mathematical recipe implemented in
``src/ir/transforms/lower_composite_ops_pass.cpp`` (Cody-Waite range reduction +
degree-9 odd Horner polynomial) in pure NumPy, then compares the result
against ``numpy.sin`` / ``numpy.cos`` over the validated input range
``|x| <= 2*pi*1024``.

The PyPTO unit test layer has no IR interpreter, so we cannot run the
lowered IR end-to-end without hardware. Instead, this test verifies that
the constants and operation sequence baked into the C++ pass are
numerically correct. A regression here means a transcription error in
either this file or the C++ pass.
"""

import numpy as np
import pytest

# ============================================================================
# FP32 constants. These MUST match the values in
# src/ir/transforms/lower_composite_ops_pass.cpp digit-for-digit.
# ============================================================================
PI_INV = np.float32(0.31830988732818603515625)
PI_V2 = np.float32(3.140625)
PI_C1 = np.float32(0.0009670257568359375)
PI_C2 = np.float32(6.2771141529083251953125e-7)
PI_C3 = np.float32(1.21644916362129151821e-10)
PI_C4 = np.float32(-1.0290623200529979163e-13)
PI_HALF_HEAD = np.float32(1.57079637050628662109375)
PI_HALF_TAIL = np.float32(-4.371139000189375e-8)
HALF = np.float32(0.5)
M4 = np.float32(4.0)
NEG2 = np.float32(-2.0)
ONE = np.float32(1.0)
R0 = np.float32(2.604926501e-6)
R1 = np.float32(-1.980894471e-4)
R2 = np.float32(8.333049340e-3)
R3 = np.float32(-1.666665792e-1)


def lowered_sin(x: np.ndarray) -> np.ndarray:
    """Pure-Python mirror of the sin lowering in lower_composite_ops_pass.cpp."""
    x = x.astype(np.float32)

    # Range reduction: k = round(x / pi)
    k_f = (x * PI_INV).astype(np.float32)
    # CAST_ROUND in PTO is round-half-away-from-zero (ISO C lround), not
    # banker's rounding. ``np.round`` follows IEEE round-half-to-even, so it
    # would diverge from the hardware on tie inputs (e.g. 0.5, 1.5).
    k_i = (np.sign(k_f) * np.floor(np.abs(k_f) + 0.5)).astype(np.int32)  # CAST_ROUND
    k_f = k_i.astype(np.float32)

    # 4-part Cody-Waite subtraction: t = x - k * pi
    t = x.astype(np.float32)
    for c in (PI_V2, PI_C1, PI_C2, PI_C3, PI_C4):
        t = (t - (k_f * c).astype(np.float32)).astype(np.float32)

    # sign = floor(k/2) * 4 + k * (-2) + 1
    half_k = (k_f * HALF).astype(np.float32)
    floor_hk_i = np.floor(half_k).astype(np.int32)  # CAST_FLOOR
    floor_hk_f = floor_hk_i.astype(np.float32)
    sign_pre = ((floor_hk_f * M4).astype(np.float32) + (k_f * NEG2).astype(np.float32)).astype(np.float32)
    sign = (sign_pre + ONE).astype(np.float32)

    # Horner: P(t^2) = (((R0*t^2 + R1)*t^2 + R2)*t^2 + R3)*t^2 + 1
    t2 = (t * t).astype(np.float32)
    p = ((t2 * R0).astype(np.float32) + R1).astype(np.float32)
    p = ((p * t2).astype(np.float32) + R2).astype(np.float32)
    p = ((p * t2).astype(np.float32) + R3).astype(np.float32)
    p = ((p * t2).astype(np.float32) + ONE).astype(np.float32)

    # out = sign * (t * P(t^2))
    t_p = (t * p).astype(np.float32)
    return (sign * t_p).astype(np.float32)


def lowered_cos(x: np.ndarray) -> np.ndarray:
    """Pure-Python mirror of the cos lowering in lower_composite_ops_pass.cpp.

    Differs from sin in three places:
      1. k = rint(x * PI_INV + 0.5) (RINT mode = banker's rounding to even)
      2. After the PI_C1 subtraction, +PI_HALF_HEAD is added.
      3. After the PI_C4 subtraction, +PI_HALF_TAIL is added.
    """
    x = x.astype(np.float32)

    # Range reduction: k = rint(x / pi + 0.5). The +0.5 is applied to the
    # FP32 product BEFORE the cast, then RINT (banker's rounding to even).
    pi_inv_x = (x * PI_INV).astype(np.float32)
    k_pre = (pi_inv_x + HALF).astype(np.float32)
    k_i = np.rint(k_pre).astype(np.int32)  # CAST_RINT
    k_f = k_i.astype(np.float32)

    # 4-part Cody-Waite subtraction with PI_HALF_HEAD interleaved between
    # PI_C1 and PI_C2, and PI_HALF_TAIL after PI_C4.
    t = x.astype(np.float32)
    t = (t - (k_f * PI_V2).astype(np.float32)).astype(np.float32)
    t = (t - (k_f * PI_C1).astype(np.float32)).astype(np.float32)
    t = (t + PI_HALF_HEAD).astype(np.float32)
    t = (t - (k_f * PI_C2).astype(np.float32)).astype(np.float32)
    t = (t - (k_f * PI_C3).astype(np.float32)).astype(np.float32)
    t = (t - (k_f * PI_C4).astype(np.float32)).astype(np.float32)
    t = (t + PI_HALF_TAIL).astype(np.float32)

    # sign = floor(k/2) * 4 + k * (-2) + 1
    half_k = (k_f * HALF).astype(np.float32)
    floor_hk_i = np.floor(half_k).astype(np.int32)
    floor_hk_f = floor_hk_i.astype(np.float32)
    sign_pre = ((floor_hk_f * M4).astype(np.float32) + (k_f * NEG2).astype(np.float32)).astype(np.float32)
    sign = (sign_pre + ONE).astype(np.float32)

    # Horner: same as sin
    t2 = (t * t).astype(np.float32)
    p = ((t2 * R0).astype(np.float32) + R1).astype(np.float32)
    p = ((p * t2).astype(np.float32) + R2).astype(np.float32)
    p = ((p * t2).astype(np.float32) + R3).astype(np.float32)
    p = ((p * t2).astype(np.float32) + ONE).astype(np.float32)

    t_p = (t * p).astype(np.float32)
    return (sign * t_p).astype(np.float32)


# ============================================================================
# Tests
# ============================================================================
# Validated input range: |x| <= 2*pi*1024 (~6435.0). Beyond this the
# range-reduction error grows because k_f loses too many integer bits.
_RANGE = 2.0 * float(np.pi) * 1024.0


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_recipe_matches_numpy_sin(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-_RANGE, _RANGE, size=(2048,)).astype(np.float32)
    expected = np.sin(x).astype(np.float32)
    actual = lowered_sin(x)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_recipe_matches_numpy_cos(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-_RANGE, _RANGE, size=(2048,)).astype(np.float32)
    expected = np.cos(x).astype(np.float32)
    actual = lowered_cos(x)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_recipe_special_points_sin():
    """Verify the recipe matches at well-known points (within FP32 noise)."""
    cases = [
        (0.0, 0.0),
        (float(np.pi) / 2, 1.0),
        (float(np.pi), 0.0),
        (3 * float(np.pi) / 2, -1.0),
        (2 * float(np.pi), 0.0),
    ]
    for x_val, expected in cases:
        actual = lowered_sin(np.array([x_val], dtype=np.float32))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_recipe_special_points_cos():
    cases = [
        (0.0, 1.0),
        (float(np.pi) / 2, 0.0),
        (float(np.pi), -1.0),
        (3 * float(np.pi) / 2, 0.0),
        (2 * float(np.pi), 1.0),
    ]
    for x_val, expected in cases:
        actual = lowered_cos(np.array([x_val], dtype=np.float32))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
