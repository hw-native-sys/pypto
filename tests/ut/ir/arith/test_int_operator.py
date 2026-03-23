# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for arith integer operator utilities (floordiv, floormod, GCD, LCM, ExtEuclid)."""

import pytest
from pypto.arith import extended_euclidean, floordiv, floormod, gcd, lcm


class TestFloorDiv:
    def test_positive(self):
        assert floordiv(7, 3) == 2

    def test_negative_numerator(self):
        assert floordiv(-7, 3) == -3  # floor, not trunc (-2)

    def test_negative_denominator(self):
        assert floordiv(7, -3) == -3

    def test_both_negative(self):
        assert floordiv(-7, -3) == 2

    def test_exact(self):
        assert floordiv(6, 3) == 2

    def test_zero_numerator(self):
        assert floordiv(0, 5) == 0

    def test_by_zero_raises(self):
        with pytest.raises(Exception):
            floordiv(5, 0)

    def test_int64_min_div_neg1_raises(self):
        with pytest.raises(Exception):
            floordiv(-(2**63), -1)


class TestFloorMod:
    def test_positive(self):
        assert floormod(7, 3) == 1

    def test_negative_numerator(self):
        assert floormod(-7, 3) == 2  # floor mod, not trunc (-1)

    def test_negative_denominator(self):
        assert floormod(7, -3) == -2

    def test_both_negative(self):
        assert floormod(-7, -3) == -1

    def test_exact(self):
        assert floormod(6, 3) == 0

    def test_by_zero_raises(self):
        with pytest.raises(Exception):
            floormod(5, 0)


class TestGCD:
    def test_basic(self):
        assert gcd(12, 8) == 4

    def test_coprime(self):
        assert gcd(7, 13) == 1

    def test_one_zero(self):
        assert gcd(0, 5) == 5
        assert gcd(5, 0) == 5

    def test_both_zero(self):
        assert gcd(0, 0) == 0

    def test_negative(self):
        assert gcd(-12, 8) == 4
        assert gcd(12, -8) == 4
        assert gcd(-12, -8) == 4

    def test_same(self):
        assert gcd(7, 7) == 7


class TestLCM:
    def test_basic(self):
        assert lcm(4, 6) == 12

    def test_coprime(self):
        assert lcm(3, 5) == 15

    def test_one_zero(self):
        assert lcm(0, 5) == 0
        assert lcm(5, 0) == 0

    def test_negative(self):
        assert lcm(-6, 4) == 12  # always non-negative
        assert lcm(6, -4) == 12

    def test_same(self):
        assert lcm(7, 7) == 7


class TestExtendedEuclidean:
    def test_basic(self):
        g, x, y = extended_euclidean(12, 8)
        assert g == 4
        assert 12 * x + 8 * y == g

    def test_coprime(self):
        g, x, y = extended_euclidean(7, 13)
        assert g == 1
        assert 7 * x + 13 * y == g

    def test_zero_b(self):
        g, x, y = extended_euclidean(5, 0)
        assert g == 5

    def test_negative_a(self):
        g, x, y = extended_euclidean(-12, 8)
        assert g == 4
        assert -12 * x + 8 * y == g

    def test_negative_b(self):
        g, x, y = extended_euclidean(12, -8)
        assert g == 4
        assert 12 * x + (-8) * y == g

    def test_bezout_identity_large(self):
        """Verify Bezout identity for larger values."""
        g, x, y = extended_euclidean(35, 15)
        assert g == 5
        assert 35 * x + 15 * y == g


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
