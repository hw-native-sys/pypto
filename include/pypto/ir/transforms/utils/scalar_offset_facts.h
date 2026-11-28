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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_SCALAR_OFFSET_FACTS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_SCALAR_OFFSET_FACTS_H_

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/program.h"

namespace pypto::ir::scalar_offset_facts {

// Shared across the entire proof, including all caller paths through a parameter.
constexpr int kScalarOffsetProofStepBudget = 256;

struct ScalarOffsetFacts {
  // All callbacks are optional. Missing bindings (including loop endpoints) return null.
  std::function<ExprPtr(const VarPtr&)> definition;
  std::function<std::pair<ExprPtr, ExprPtr>(const VarPtr&)> loop_range;
  std::function<bool(const VarPtr&)> is_non_negative;
  // A nonempty list covers every known caller. A missing source invalidates the whole list.
  std::function<const std::vector<ExprPtr>*(const VarPtr&)> parameter_sources;
};

class ScalarOffsetFactStore {
 public:
  explicit ScalarOffsetFactStore(ProgramPtr program);
  // The callbacks borrow this store, which must outlive the rewrite using them.
  [[nodiscard]] ScalarOffsetFacts Facts() const;

 private:
  class Collector;
  std::unordered_map<VarPtr, ExprPtr> definitions_;
  std::unordered_map<VarPtr, std::pair<ExprPtr, ExprPtr>> loop_bindings_;
  std::unordered_set<VarPtr> non_negative_vars_;
  std::unordered_map<VarPtr, std::vector<ExprPtr>> parameter_sources_;
  std::unordered_set<VarPtr> unknown_params_;
};

// Check alignment without reassociating fixed-width arithmetic. The consumer divides
// the original offset as a whole to preserve its evaluation before division.
bool IsProvableMultipleOf(const ExprPtr& expr, int64_t divisor, const ScalarOffsetFacts& facts, int* budget);
bool IsProvableNonNegative(const ExprPtr& expr, const ScalarOffsetFacts& facts, int* budget);

}  // namespace pypto::ir::scalar_offset_facts

#endif  // PYPTO_IR_TRANSFORMS_UTILS_SCALAR_OFFSET_FACTS_H_
