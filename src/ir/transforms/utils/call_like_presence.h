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

#ifndef SRC_IR_TRANSFORMS_UTILS_CALL_LIKE_PRESENCE_H_
#define SRC_IR_TRANSFORMS_UTILS_CALL_LIKE_PRESENCE_H_

#include <unordered_map>

#include "pypto/ir/expr.h"
#include "pypto/ir/transforms/base/visitor.h"

namespace pypto::ir::transform_utils {

// Calls have no purity contract. Cache occurrence information over immutable
// expression nodes so guarding each nested arithmetic node remains O(N), rather
// than rescanning its entire subtree. Owning keys prevent address reuse after
// a rewritten temporary expression is destroyed. The cache lives for one pass
// invocation; Var/IterArg uses do not re-evaluate their types or initializers.
class CallLikePresence : public IRVisitor {
 public:
  bool Contains(const ExprPtr& expr) {
    found_ = false;
    VisitExpr(expr);
    return found_;
  }

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr) return;
    if (auto it = cache_.find(expr); it != cache_.end()) {
      found_ |= it->second;
      return;
    }
    const bool enclosing = found_;
    found_ = false;
    IRVisitor::VisitExpr(expr);
    cache_.emplace(expr, found_);
    found_ |= enclosing;
  }

 protected:
  void VisitExpr_(const CallPtr&) override { found_ = true; }
  void VisitExpr_(const SubmitPtr&) override { found_ = true; }
  void VisitVarLike_(const VarPtr&) override {}

 private:
  bool found_ = false;
  std::unordered_map<ExprPtr, bool> cache_;
};

}  // namespace pypto::ir::transform_utils
#endif  // SRC_IR_TRANSFORMS_UTILS_CALL_LIKE_PRESENCE_H_
