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

/**
 * @file arith.cpp
 * @brief Python bindings for arithmetic simplification utilities.
 *
 * Exposes constant folding for testing.
 * The full Analyzer API will be added in a later PR.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <tuple>

#include "../module.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/arith/int_operator.h"

namespace nb = nanobind;

namespace pypto {
namespace python {

void BindArith(nb::module_& m) {
  nb::module_ arith = m.def_submodule("arith", "Arithmetic simplification utilities");

  arith.def(
      "fold_const", [](const ir::ExprPtr& expr) -> ir::ExprPtr { return ir::arith::TryConstFold(expr); },
      nb::arg("expr"),
      "Try to constant-fold an expression. Accepts any BinaryExpr or UnaryExpr.\n"
      "Returns the folded result, or None if folding is not possible.");

  // Integer operator utilities (exposed for testing)
  arith.def("floordiv", &ir::arith::floordiv, nb::arg("x"), nb::arg("y"), "Floor division.");
  arith.def("floormod", &ir::arith::floormod, nb::arg("x"), nb::arg("y"), "Floor modulo.");
  arith.def("gcd", &ir::arith::ZeroAwareGCD, nb::arg("a"), nb::arg("b"), "GCD (treats 0 as identity).");
  arith.def("lcm", &ir::arith::LeastCommonMultiple, nb::arg("a"), nb::arg("b"), "Least common multiple.");
  arith.def(
      "extended_euclidean",
      [](int64_t a, int64_t b) -> std::tuple<int64_t, int64_t, int64_t> {
        int64_t x, y;
        int64_t g = ir::arith::ExtendedEuclidean(a, b, &x, &y);
        return {g, x, y};
      },
      nb::arg("a"), nb::arg("b"), "Extended Euclidean: returns (gcd, x, y) where a*x + b*y = gcd.");
}

}  // namespace python
}  // namespace pypto
