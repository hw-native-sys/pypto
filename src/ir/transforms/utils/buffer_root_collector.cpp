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

#include "pypto/ir/transforms/utils/buffer_root_collector.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {
namespace {

// Builtin ops (tile.* / tensor.* / system.* / array.*) are never user
// functions, so they carry no callee Out/InOut params to trace. Mirrors the
// canonical IsBuiltinOp predicate; kept file-local to avoid a cross-layer
// dependency from this IR util onto the codegen module.
bool IsBuiltinOpName(const std::string& op_name) {
  return op_name.find("tile.") == 0 || op_name.find("tensor.") == 0 || op_name.find("system.") == 0 ||
         op_name.find("array.") == 0;
}

}  // namespace

BufferRootCollector::BufferRootCollector(ProgramPtr program) : program_(std::move(program)) {}

void BufferRootCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    buffer_roots[param.get()] = param.get();
  }
}

void BufferRootCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = for_stmt->iter_args_[i];
    const Var* root = ResolveExpr(iter_arg->initValue_);
    if (root) {
      buffer_roots[iter_arg.get()] = root;
      if (i < for_stmt->return_vars_.size()) {
        buffer_roots[for_stmt->return_vars_[i].get()] = root;
      }
    }
  }
  IRVisitor::VisitStmt_(for_stmt);
}

void BufferRootCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = while_stmt->iter_args_[i];
    const Var* root = ResolveExpr(iter_arg->initValue_);
    if (root) {
      buffer_roots[iter_arg.get()] = root;
      if (i < while_stmt->return_vars_.size()) {
        buffer_roots[while_stmt->return_vars_[i].get()] = root;
      }
    }
  }
  IRVisitor::VisitStmt_(while_stmt);
}

void BufferRootCollector::VisitStmt_(const AssignStmtPtr& assign) {
  // Submit (pl.submit inside a pl.manual_scope) is a sibling call-like kind;
  // route it through the Call-shaped view so a task launch's Out/InOut roots are
  // tracked identically to a plain Call (results keyed on the stable binding
  // Var). The view preserves args_ and the TASK_ID-augmented return type, so the
  // tuple path maps the submit-result projections to the callee's Out roots and
  // leaves the trailing TASK_ID element unmapped
  // (see .claude/rules/pass-submit-awareness.md).
  if (auto call = transform_utils::AsCallOrSubmitView(assign->value_)) {
    const std::string& op_name = call->op_->name_;
    if (IsOp(call, "tensor.create") || IsOp(call, "tensor.slice")) {
      buffer_roots[assign->var_.get()] = assign->var_.get();
    } else if (IsOp(call, "tensor.assemble")) {
      if (call->args_.size() == 3) {
        if (const Var* target_root = ResolveExpr(call->args_[0])) {
          buffer_roots[assign->var_.get()] = target_root;
        }
      }
    } else if (!IsBuiltinOpName(op_name)) {
      auto out_roots = CollectCallOutputRoots(call);
      if (As<TupleType>(call->GetType())) {
        std::vector<const Var*> roots;
        roots.reserve(out_roots.size());
        for (const auto& entry : out_roots) roots.push_back(entry.root);
        tuple_output_roots_[assign->var_.get()] = std::move(roots);
      } else if (const Var* root = SelectReturnRoot(out_roots, call->GetType())) {
        buffer_roots[assign->var_.get()] = root;
      }
    }
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_output_roots_.find(tuple_var.get());
      if (it != tuple_output_roots_.end() && tuple_get->index_ < static_cast<int>(it->second.size()) &&
          it->second[tuple_get->index_]) {
        buffer_roots[assign->var_.get()] = it->second[tuple_get->index_];
      }
    }
  } else if (auto src_var = AsVarLike(assign->value_)) {
    if (const Var* root = ResolveVar(src_var.get())) {
      buffer_roots[assign->var_.get()] = root;
    }
  }
  IRVisitor::VisitStmt_(assign);
}

const Var* BufferRootCollector::ResolveVar(const Var* var) const {
  auto it = buffer_roots.find(var);
  return it != buffer_roots.end() ? it->second : nullptr;
}

const Var* BufferRootCollector::ResolveExpr(const ExprPtr& expr) const {
  if (auto var = AsVarLike(expr)) {
    return ResolveVar(var.get());
  }
  return nullptr;
}

std::vector<BufferRootCollector::OutputRoot> BufferRootCollector::CollectCallOutputRoots(
    const CallPtr& call) const {
  auto callee = program_->GetFunction(call->op_->name_);
  if (!callee) return {};

  std::vector<OutputRoot> roots;
  for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
    if (callee->param_directions_[i] != ParamDirection::Out &&
        callee->param_directions_[i] != ParamDirection::InOut) {
      continue;
    }
    const Var* root = nullptr;
    if (auto arg_var = AsVarLike(call->args_[i])) {
      root = ResolveVar(arg_var.get());
    }
    roots.push_back(OutputRoot{root, call->args_[i]->GetType()});
  }
  return roots;
}

const Var* BufferRootCollector::SelectReturnRoot(const std::vector<OutputRoot>& out_roots,
                                                 const TypePtr& return_type) const {
  if (out_roots.empty()) return nullptr;
  if (out_roots.size() == 1) return out_roots[0].root;

  const Var* match = nullptr;
  bool ambiguous = false;
  for (const auto& candidate : out_roots) {
    if (candidate.root && TypesMatchShapeDtype(candidate.type, return_type)) {
      if (match == nullptr) {
        match = candidate.root;
      } else if (match != candidate.root) {
        ambiguous = true;
      }
    }
  }
  if (match && !ambiguous) return match;
  // Ambiguous (>1 distinct match) or no provable match: do not guess. Fusion /
  // aliasing is an optimization, so skipping it (no root -> no aliasing) is
  // always safe, whereas guessing out_roots[0] could re-alias a scratch onto
  // the output.
  return nullptr;
}

bool BufferRootCollector::TypesMatchShapeDtype(const TypePtr& a, const TypePtr& b) {
  auto ta = As<TensorType>(a);
  auto tb = As<TensorType>(b);
  if (!ta || !tb) return false;
  if (ta->dtype_ != tb->dtype_) return false;
  if (ta->shape_.size() != tb->shape_.size()) return false;
  for (size_t i = 0; i < ta->shape_.size(); ++i) {
    if (!AreExprsEqual(ta->shape_[i], tb->shape_[i])) return false;
  }
  return true;
}

}  // namespace buffer_root
}  // namespace ir
}  // namespace pypto
