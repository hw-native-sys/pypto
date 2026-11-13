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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_TUPLE_RESULTS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_TUPLE_RESULTS_H_

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {

/// Index canonical multi-result bindings once, before planning or lowering.
/// Every native output must have one direct projection in the producer's
/// sequence, including unread outputs. Tuple escapes and duplicate projections
/// are deliberately outside this first rollout. No output allocation is invented.
class BufferTupleResults : public IRVisitor {
 public:
  struct Result {
    CallPtr call;
    const SeqStmts* region = nullptr;
    std::vector<VarPtr> outputs;
    bool valid = true;
  };
  BufferTupleResults() = default;
  explicit BufferTupleResults(const StmtPtr& body) {
    if (body) VisitStmt(body);
  }

  [[nodiscard]] const Result* Get(const VarPtr& tuple) const {
    const auto found = results_.find(tuple.get());
    if (found == results_.end() || !found->second.valid) return nullptr;
    for (const auto& output : found->second.outputs) {
      if (!output) return nullptr;
    }
    return &found->second;
  }

  [[nodiscard]] bool IsProjection(const AssignStmtPtr& statement) const {
    const auto item = As<TupleGetItemExpr>(statement->value_);
    const auto result = item ? Get(AsVarLike(item->tuple_)) : nullptr;
    return result && item->index_ >= 0 && static_cast<size_t>(item->index_) < result->outputs.size() &&
           result->outputs[item->index_] == statement->var_;
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitStmt_(const SeqStmtsPtr& sequence) override {
    const auto previous = region_;
    region_ = sequence.get();
    IRVisitor::VisitStmt_(sequence);
    region_ = previous;
  }
  void VisitStmt_(const AssignStmtPtr& statement) override {
    const auto call = As<Call>(statement->value_);
    if (IsOp(call, "tile.gather_compare")) {
      const auto type = As<TupleType>(call->GetType());
      const auto arity = OpRegistry::GetInstance().GetEntry(call->op_->name_).GetOutputArity();
      if (call->args_.size() == 3 && type && type->types_.size() == arity) {
        results_.emplace(statement->var_.get(), Result{call, region_, std::vector<VarPtr>(arity), true});
      }
    } else if (const auto item = As<TupleGetItemExpr>(statement->value_)) {
      const auto found = results_.find(AsVarLike(item->tuple_).get());
      if (found != results_.end()) {
        auto& result = found->second;
        const auto type = As<TupleType>(result.call->GetType());
        if (item->index_ < 0 || static_cast<size_t>(item->index_) >= result.outputs.size() ||
            region_ != result.region) {
          result.valid = false;
        } else {
          auto& output = result.outputs[item->index_];
          result.valid &= !output && As<TileType>(statement->var_->GetType()) &&
                          structural_equal(CloneTypeWithoutMemRef(statement->var_->GetType()),
                                           CloneTypeWithoutMemRef(type->types_[item->index_]));
          output = statement->var_;
        }
      }
    }
    IRVisitor::VisitStmt_(statement);
  }

 private:
  const SeqStmts* region_ = nullptr;
  std::unordered_map<const Var*, Result> results_;
};

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_TUPLE_RESULTS_H_
