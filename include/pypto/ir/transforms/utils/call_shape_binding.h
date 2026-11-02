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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_CALL_SHAPE_BINDING_H_
#define PYPTO_IR_TRANSFORMS_UTILS_CALL_SHAPE_BINDING_H_

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

inline constexpr size_t kCallShapeBudget = 65536;

using CallShapeMap = std::unordered_map<const Var*, ExprPtr>;

enum class CallShapeStatus { Success, Unsupported, Conflict, MissingSource, BudgetExhausted };

// First-stage contract: scalar parameters and ordinary, unallocated Tensor
// signatures with physical shapes and optional layout-only views. Richer views,
// distributed storage and output-only symbols keep the existing dedup guard until their lowering contracts
// are tested.
class CallShapeBindingPlan {
 public:
  explicit CallShapeBindingPlan(FunctionPtr function);
  [[nodiscard]] bool supported() const { return supported_; }
  CallShapeStatus Bind(const std::vector<ExprPtr>& args, CallShapeMap& bindings, size_t& budget) const;
  bool MatchesResult(const ExprPtr& call, const CallShapeMap& bindings, size_t& budget) const;

 private:
  FunctionPtr function_;
  bool supported_ = true;
  std::vector<VarPtr> required_;
};

// Actual expressions are already in caller coordinates: never follow a second
// callee mapping inside a replacement. Original argument bindings stay fixed;
// newly materialized output parameters extend the environment in order.
// One instance per call keeps memoized DAGs isolated between callers.
class CallShapeInstantiation : public IRMutator {
 public:
  explicit CallShapeInstantiation(CallShapeMap bindings) : bindings_(std::move(bindings)) {}
  TypePtr InstantiateType(const TypePtr& type);
  void BindOutput(const VarPtr& output, const ExprPtr& storage);

 protected:
  ExprPtr VisitExpr_(const VarPtr& var) override;
  ExprPtr VisitExpr_(const IterArgPtr& var) override;
  ExprPtr VisitBinaryExpr_(const BinaryExprPtr& expr) override;
  ExprPtr VisitUnaryExpr_(const UnaryExprPtr& expr) override;

 private:
  ExprPtr Replace(const VarPtr& var) const;
  CallShapeMap bindings_;
  std::unordered_map<const Expr*, ExprPtr> expressions_;
  std::unordered_map<const Type*, TypePtr> types_;
};

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_CALL_SHAPE_BINDING_H_
