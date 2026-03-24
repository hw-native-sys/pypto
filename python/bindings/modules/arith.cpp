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
 * Exposes constant folding and ConstIntBoundAnalyzer.
 * The full Analyzer API will be added in a later PR.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "../module.h"
#include "pypto/ir/arith/analyzer.h"
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

  // ConstIntBound
  nb::class_<ir::arith::ConstIntBound>(arith, "ConstIntBound",
                                       "Inclusive integer bounds [min_value, max_value] for an expression.")
      .def(nb::init<int64_t, int64_t>(), nb::arg("min_value"), nb::arg("max_value"),
           "Create inclusive integer bounds [min_value, max_value].")
      .def_ro("min_value", &ir::arith::ConstIntBound::min_value, "Inclusive lower bound.")
      .def_ro("max_value", &ir::arith::ConstIntBound::max_value, "Inclusive upper bound.")
      .def_ro_static("kPosInf", &ir::arith::ConstIntBound::kPosInf, "Sentinel for positive infinity.")
      .def_ro_static("kNegInf", &ir::arith::ConstIntBound::kNegInf, "Sentinel for negative infinity.")
      .def("is_const", nb::overload_cast<>(&ir::arith::ConstIntBound::is_const, nb::const_),
           "Check if min == max (constant).")
      .def("is_non_negative", &ir::arith::ConstIntBound::is_non_negative, "Check if min >= 0.")
      .def("is_positive", &ir::arith::ConstIntBound::is_positive, "Check if min > 0.")
      .def("is_everything", &ir::arith::ConstIntBound::is_everything,
           "Check if bounds are [-inf, +inf] (no information).")
      .def("__repr__", [](const ir::arith::ConstIntBound& b) {
        auto fmt = [](int64_t v) -> std::string {
          if (v == ir::arith::ConstIntBound::kPosInf) return "+inf";
          if (v == ir::arith::ConstIntBound::kNegInf) return "-inf";
          return std::to_string(v);
        };
        return "ConstIntBound[" + fmt(b.min_value) + ", " + fmt(b.max_value) + "]";
      });

  // RewriteSimplifier
  nb::class_<ir::arith::RewriteSimplifier>(
      arith, "RewriteSimplifier",
      "Simplifies integer/index expressions by applying algebraic rewrite rules.\n\n"
      "Float-typed expressions are returned unchanged.")
      .def(nb::init<>(), "Create a standalone RewriteSimplifier.")
      .def("__call__", &ir::arith::RewriteSimplifier::operator(), nb::arg("expr"),
           "Simplify an expression by applying rewrite rules.")
      .def("update", &ir::arith::RewriteSimplifier::Update, nb::arg("var"), nb::arg("new_expr").none(),
           "Register a variable substitution. Pass None to remove a previous substitution.")
      .def("enter_constraint", &ir::arith::RewriteSimplifier::EnterConstraint, nb::arg("constraint"),
           "Enter a constraint scope. Returns a recovery function that restores original state.");

  // ConstIntBoundAnalyzer
  nb::class_<ir::arith::ConstIntBoundAnalyzer>(arith, "ConstIntBoundAnalyzer",
                                               "Propagates constant integer bounds through expression trees.")
      .def(nb::init<>(), "Create a standalone ConstIntBoundAnalyzer.")
      .def("__call__", &ir::arith::ConstIntBoundAnalyzer::operator(), nb::arg("expr"),
           "Compute bounds for an expression.")
      .def("bind", &ir::arith::ConstIntBoundAnalyzer::Bind, nb::arg("var"), nb::arg("min_val"),
           nb::arg("max_val_exclusive"),
           "Bind a variable to the half-open range [min_val, max_val_exclusive).")
      .def("update", &ir::arith::ConstIntBoundAnalyzer::Update, nb::arg("var"), nb::arg("bound"),
           "Update a variable's bound (inclusive on both ends).");

  // ModularSet
  nb::class_<ir::arith::ModularSet>(arith, "ModularSet",
                                    "Modular arithmetic properties: value = coeff * k + base.")
      .def(nb::init<int64_t, int64_t>(), nb::arg("coeff"), nb::arg("base"),
           "Create a modular set with given coeff and base.")
      .def_ro("coeff", &ir::arith::ModularSet::coeff, "Coefficient (>= 0). 0 means exact value known.")
      .def_ro("base", &ir::arith::ModularSet::base, "Base value. Normalized to [0, coeff) when coeff > 0.")
      .def("is_exact", &ir::arith::ModularSet::is_exact, "Check if exact value is known (coeff == 0).")
      .def("is_everything", &ir::arith::ModularSet::is_everything,
           "Check if no useful modular info (coeff == 1, base == 0).")
      .def("__repr__", [](const ir::arith::ModularSet& s) {
        return "ModularSet(coeff=" + std::to_string(s.coeff) + ", base=" + std::to_string(s.base) + ")";
      });

  // ModularSetAnalyzer
  nb::class_<ir::arith::ModularSetAnalyzer>(arith, "ModularSetAnalyzer",
                                            "Tracks modular arithmetic properties through expression trees.")
      .def(nb::init<>(), "Create a standalone ModularSetAnalyzer.")
      .def("__call__", &ir::arith::ModularSetAnalyzer::operator(), nb::arg("expr"),
           "Compute modular set for an expression.")
      .def("update", &ir::arith::ModularSetAnalyzer::Update, nb::arg("var"), nb::arg("info"),
           "Update a variable's modular set information.")
      .def("enter_constraint", &ir::arith::ModularSetAnalyzer::EnterConstraint, nb::arg("constraint"),
           "Enter a constraint scope. Returns a recovery function, or None if constraint is not useful.");
}

}  // namespace python
}  // namespace pypto
