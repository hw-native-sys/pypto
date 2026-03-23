/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*
 * The arithmetic simplification module takes reference from:
 * - Apache TVM (https://github.com/apache/tvm), Apache License 2.0
 * - MLC-Python (https://github.com/mlc-ai/mlc-python), Apache License 2.0
 */

#ifndef PYPTO_IR_ARITH_INT_OPERATOR_H_
#define PYPTO_IR_ARITH_INT_OPERATOR_H_

#include <cstdint>
#include <limits>
#include <utility>

#include "pypto/core/logging.h"

namespace pypto {
namespace ir {
namespace arith {

/// Safe absolute value for int64_t. Returns uint64_t to avoid
/// overflow when the input is INT64_MIN.
inline uint64_t SafeAbs(int64_t x) {
  if (x >= 0) return static_cast<uint64_t>(x);
  // -(INT64_MIN) overflows int64_t, but converting to unsigned first is safe.
  return static_cast<uint64_t>(-(x + 1)) + 1U;
}

/// Floor division: rounds toward negative infinity.
/// Corrects C++'s truncation-toward-zero behavior for negative quotients.
/// Precondition: y != 0 and not (x == INT64_MIN && y == -1).
inline int64_t floordiv(int64_t x, int64_t y) {
  INTERNAL_CHECK(y != 0) << "floordiv: division by zero";
  INTERNAL_CHECK(!(x == std::numeric_limits<int64_t>::min() && y == -1))
      << "floordiv: INT64_MIN / -1 overflows int64_t";
  int64_t rdiv = x / y;
  int64_t rmod = x % y;
  bool is_floor = (y >= 0 && rmod >= 0) || (y < 0 && rmod <= 0);
  return is_floor ? rdiv : (rdiv - 1);
}

/// Floor modulo: result has the same sign as the divisor.
/// Precondition: y != 0 and not (x == INT64_MIN && y == -1).
inline int64_t floormod(int64_t x, int64_t y) {
  INTERNAL_CHECK(y != 0) << "floormod: division by zero";
  INTERNAL_CHECK(!(x == std::numeric_limits<int64_t>::min() && y == -1))
      << "floormod: INT64_MIN % -1 overflows int64_t";
  int64_t rmod = x % y;
  bool is_floor = (y >= 0 && rmod >= 0) || (y < 0 && rmod <= 0);
  return is_floor ? rmod : rmod + y;
}

/// Extended Euclidean algorithm: solve a*x + b*y = gcd(a, b).
/// Returns gcd (always non-negative), sets *px and *py.
inline int64_t ExtendedEuclidean(int64_t a, int64_t b, int64_t* px, int64_t* py) {
  int64_t s = 0, old_s = 1;
  // Work on non-negative magnitudes. SafeAbs handles INT64_MIN.
  uint64_t r = SafeAbs(b);
  uint64_t old_r = SafeAbs(a);
  while (r != 0) {
    uint64_t q = old_r / r;
    uint64_t tmp_r = old_r - q * r;
    int64_t tmp_s = old_s - static_cast<int64_t>(q) * s;
    old_r = r;
    r = tmp_r;
    old_s = s;
    s = tmp_s;
  }
  *px = a >= 0 ? old_s : -old_s;
  INTERNAL_CHECK(old_r <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      << "ExtendedEuclidean: GCD exceeds INT64_MAX";
  int64_t gcd = static_cast<int64_t>(old_r);
  if (b != 0) {
    *py = (gcd - (*px) * a) / b;
  } else {
    *py = 1;
  }
  return gcd;
}

/// GCD that treats 0 as +infinity (identity element for GCD).
/// Always returns a non-negative value. Safe for INT64_MIN inputs
/// (the result is always <= max(|a|, |b|); the only case where it
/// exceeds INT64_MAX is gcd(INT64_MIN, 0) = 2^63, which cannot be
/// represented as int64_t and triggers INTERNAL_CHECK).
inline int64_t ZeroAwareGCD(int64_t a, int64_t b) {
  uint64_t ua = SafeAbs(a);
  uint64_t ub = SafeAbs(b);
  if (ua < ub) std::swap(ua, ub);
  if (ub == 0) {
    INTERNAL_CHECK(ua <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        << "GCD result exceeds INT64_MAX";
    return static_cast<int64_t>(ua);
  }
  while (ua % ub != 0) {
    ua = ua % ub;
    std::swap(ua, ub);
  }
  return static_cast<int64_t>(ub);
}

/// Least common multiple. Always returns a non-negative value.
/// Returns 0 if either input is 0.
inline int64_t LeastCommonMultiple(int64_t a, int64_t b) {
  if (a == 0 || b == 0) return 0;
  int64_t x, y;
  int64_t g = ExtendedEuclidean(a, b, &x, &y);
  int64_t lcm = (a / g) * b;
  return lcm < 0 ? -lcm : lcm;
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_ARITH_INT_OPERATOR_H_
