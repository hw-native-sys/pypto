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

#include <algorithm>
#include <cstdint>
#include <utility>

namespace pypto {
namespace ir {
namespace arith {

/// Floor division: rounds toward negative infinity.
/// Corrects C++'s truncation-toward-zero behavior for negative quotients.
inline int64_t floordiv(int64_t x, int64_t y) {
  int64_t rdiv = x / y;
  int64_t rmod = x % y;
  bool is_floor = (y >= 0 && rmod >= 0) || (y < 0 && rmod <= 0);
  return is_floor ? rdiv : (rdiv - 1);
}

/// Floor modulo: result has the same sign as the divisor.
inline int64_t floormod(int64_t x, int64_t y) {
  int64_t rmod = x % y;
  bool is_floor = (y >= 0 && rmod >= 0) || (y < 0 && rmod <= 0);
  return is_floor ? rmod : rmod + y;
}

/// Extended Euclidean algorithm: solve a*x + b*y = gcd(a, b).
/// Returns gcd, sets *px and *py.
inline int64_t ExtendedEuclidean(int64_t a, int64_t b, int64_t* px, int64_t* py) {
  int64_t s = 0, old_s = 1;
  int64_t r = b, old_r = a >= 0 ? a : -a;
  while (r != 0) {
    int64_t q = old_r / r;
    int64_t tmp = old_r;
    old_r = r;
    r = tmp - q * r;
    tmp = old_s;
    old_s = s;
    s = tmp - q * s;
  }
  *px = a >= 0 ? old_s : -old_s;
  if (b != 0) {
    *py = (old_r - (*px) * a) / b;
  } else {
    *py = 1;
  }
  return old_r;
}

/// GCD that treats 0 as +infinity (identity element for GCD).
inline int64_t ZeroAwareGCD(int64_t a, int64_t b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  if (a < b) std::swap(a, b);
  if (b == 0) return a;
  while (a % b != 0) {
    a = a % b;
    std::swap(a, b);
  }
  return b;
}

/// Least common multiple via Extended Euclidean GCD.
/// Precondition: at least one of a, b must be non-zero.
inline int64_t LeastCommonMultiple(int64_t a, int64_t b) {
  int64_t x, y;
  return (a / ExtendedEuclidean(a, b, &x, &y)) * b;
}

}  // namespace arith
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_ARITH_INT_OPERATOR_H_
