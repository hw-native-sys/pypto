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

/**
 * @file inject_gm_pipe_buffer_pass.cpp
 * @brief Inject the __gm_pipe_buffer workspace parameter for cross-core pipes on backends
 *        that route slot data through GM, and per-dispatch A5 URMA Scatter descriptor scratch.
 *
 * On 910B, cross-core tpush/tpop rides through a shared GM buffer instead of a
 * direct inter-core fabric. This pass runs after ExpandMixedKernel has split
 * mixed InCore functions into AIC/AIV pairs, and:
 *
 *   1. Finds every function that issues initialize_pipe ops.
 *   2. Adds a fresh __gm_pipe_buffer Out-tensor parameter to each, propagating
 *      the parameter upward through callers except Orchestration functions,
 *      which instead get a per-call-site placeholder tensor.create. Codegen
 *      owns the real hardware workspace size and per-pipe offsets.
 *
 * For A5 Scatter GET, the backend supplies the exact size of the sender slots.
 * Orchestration allocates that tensor before each dispatch. MIX members share
 * the same argument layout and select their slot by physical sender ID.
 */

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

/// Well-known parameter name for the GM slot buffer.
constexpr const char* kGMPipeBufferName = "__gm_pipe_buffer";

struct ScratchSpec {
  const char* param;
  const char* prefix;
  int64_t elements;
  DataType dtype;
  bool (*matches)(const CallPtr&);
  // New scratch arguments precede the user prefix so a Submit can still omit
  // its trailing Out parameters. Pipe callers keep their existing trailing ABI.
  bool prepend = false;
};
const ScratchSpec kPipeScratch{kGMPipeBufferName, "gm_pipe_buffer_", 1, DataType::FP32,
                               op_predicates::IsInitializePipe};

class ScratchUseFinder : public IRVisitor {
 public:
  explicit ScratchUseFinder(const ScratchSpec& scratch) : scratch_(scratch) {}
  bool found = false;

 private:
  void VisitExpr_(const CallPtr& call) override {
    found |= scratch_.matches(call);
    if (!found) IRVisitor::VisitExpr_(call);
  }
  const ScratchSpec& scratch_;
};

bool HasScratchParam(const FunctionPtr& func, const ScratchSpec& scratch) {
  for (const auto& param : func->params_) {
    if (param->name_hint_ == scratch.param) return true;
  }
  return false;
}

OpPtr GetCallLikeOpFromStmt(const StmtPtr& stmt) {
  ExprPtr expr;
  if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
    expr = assign->value_;
  } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
    expr = eval->expr_;
  }
  if (auto call = As<Call>(expr)) return call->op_;
  if (auto submit = As<Submit>(expr)) return submit->op_;
  return nullptr;
}

StmtPtr CloneScopeWithBody(const ScopeStmtPtr& scope, const StmtPtr& body) {
  if (auto incore = As<InCoreScopeStmt>(scope)) {
    return std::make_shared<InCoreScopeStmt>(incore->split_, incore->name_hint_, body, incore->span_,
                                             incore->leading_comments_, incore->attrs_);
  }
  if (auto cluster = As<ClusterScopeStmt>(scope)) {
    return std::make_shared<ClusterScopeStmt>(cluster->name_hint_, body, cluster->span_,
                                              cluster->leading_comments_, cluster->attrs_);
  }
  if (auto hierarchy = As<HierarchyScopeStmt>(scope)) {
    return std::make_shared<HierarchyScopeStmt>(hierarchy->level_, hierarchy->role_, hierarchy->name_hint_,
                                                body, hierarchy->span_, hierarchy->leading_comments_,
                                                hierarchy->attrs_);
  }
  if (auto spmd = As<SpmdScopeStmt>(scope)) {
    return std::make_shared<SpmdScopeStmt>(spmd->core_num_, spmd->sync_start_, spmd->name_hint_, body,
                                           spmd->span_, spmd->leading_comments_, spmd->attrs_);
  }
  if (auto runtime = As<RuntimeScopeStmt>(scope)) {
    return std::make_shared<RuntimeScopeStmt>(runtime->manual_, runtime->name_hint_, body, runtime->span_,
                                              runtime->leading_comments_, runtime->attrs_);
  }
  INTERNAL_CHECK_SPAN(false, scope->span_) << "Unsupported ScopeStmt kind in InjectGMPipeBuffer";
  return scope;
}

void BuildCallGraphFromFunctions(const std::vector<FunctionPtr>& functions,
                                 std::unordered_map<std::string, std::unordered_set<std::string>>& callers,
                                 std::unordered_map<std::string, std::unordered_set<std::string>>& callees) {
  std::unordered_set<std::string> func_names;
  for (const auto& func : functions) func_names.insert(func->name_);
  for (const auto& func : functions) {
    std::function<void(const std::vector<StmtPtr>&)> walk = [&](const std::vector<StmtPtr>& stmts) {
      for (const auto& stmt : stmts) {
        if (auto op = GetCallLikeOpFromStmt(stmt)) {
          auto gv = std::dynamic_pointer_cast<const GlobalVar>(op);
          if (gv && func_names.count(gv->name_)) {
            callees[func->name_].insert(gv->name_);
            callers[gv->name_].insert(func->name_);
          }
        }
        if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
          walk(FlattenBody(for_stmt->body_));
        } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
          walk(FlattenBody(if_stmt->then_body_));
          const auto& else_body = if_stmt->else_body_;
          if (else_body.has_value()) walk(FlattenBody(*else_body));
        } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
          walk(FlattenBody(while_stmt->body_));
        } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
          walk(FlattenBody(scope->body_));
        }
      }
    };
    if (func->body_) walk(FlattenBody(func->body_));
  }
}

FunctionPtr AddScratchParam(const FunctionPtr& func, const ScratchSpec& scratch) {
  auto gm_type = std::make_shared<TensorType>(std::vector<int64_t>{scratch.elements}, scratch.dtype,
                                              std::nullopt, std::nullopt);
  auto gm_var = std::make_shared<Var>(scratch.param, gm_type, func->span_);
  auto new_params = func->params_;
  new_params.insert(scratch.prepend ? new_params.begin() : new_params.end(), gm_var);
  auto new_directions = func->param_directions_;
  new_directions.insert(scratch.prepend ? new_directions.begin() : new_directions.end(), ParamDirection::Out);
  auto result = MutableCopy(func);
  result->params_ = new_params;
  result->param_directions_ = new_directions;
  return result;
}

template <typename CallLike>
ExprPtr WithScratchOperand(const std::shared_ptr<const CallLike>& call, const VarPtr& gm_param,
                           bool prepend) {
  auto updated = MutableCopy(call);
  auto& args = updated->args_;
  args.insert(prepend ? args.begin() : args.end(), gm_param);
  for (auto& [key, value] : updated->attrs_) {
    if (key == kAttrArgDirections) {
      auto& directions = std::any_cast<std::vector<ArgDirection>&>(value);
      directions.insert(prepend ? directions.begin() : directions.end(), ArgDirection::OutputExisting);
    } else if (prepend && key == kAttrArgDirectionOverrides) {
      for (auto& index : std::any_cast<std::vector<int32_t>&>(value)) ++index;
    }
  }
  return updated;
}

/// Forward @p gm_param to every call/submit of a function in @p modified_funcs.
///
/// Walks one statement level and recurses only into bodies that can hold a
/// call. Rewritten nodes are mutable copies; a subtree that produces no change
/// is returned by pointer so unchanged statements stay shared.
StmtPtr RewriteCallsForGMBuffer(const StmtPtr& body, const std::unordered_set<std::string>& modified_funcs,
                                const VarPtr& gm_param, bool prepend) {
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any_changed = false;
  // A call site is rewritten only when its callee took the injected GM pipe parameter.
  auto should_rewrite = [&](const OpPtr& op) -> bool {
    auto gv = std::dynamic_pointer_cast<const GlobalVar>(op);
    return gv && modified_funcs.count(gv->name_);
  };
  // Copy-on-write for the two call-like kinds; Submit is not a Call subclass
  // and must keep its kind and deps.
  auto try_rewrite_expr = [&](const ExprPtr& expr) -> ExprPtr {
    if (!expr) return nullptr;
    if (auto call = std::dynamic_pointer_cast<const Call>(expr)) {
      if (!should_rewrite(call->op_)) return nullptr;
      return WithScratchOperand(call, gm_param, prepend);
    }
    if (auto submit = std::dynamic_pointer_cast<const Submit>(expr)) {
      if (!should_rewrite(submit->op_)) return nullptr;
      return WithScratchOperand(submit, gm_param, prepend);
    }
    return nullptr;
  };
  // Dispatch on statement kind, recursing into control-flow and scope bodies.
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto rw = try_rewrite_expr(assign->value_)) {
        auto new_assign = MutableCopy(assign);
        new_assign->value_ = rw;
        new_stmts.push_back(std::move(new_assign));
        any_changed = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      if (auto rw = try_rewrite_expr(eval->expr_)) {
        auto new_eval = MutableCopy(eval);
        new_eval->expr_ = rw;
        new_stmts.push_back(std::move(new_eval));
        any_changed = true;
        continue;
      }
    }
    // Control-flow and scope statements: recurse into the body, and rebuild
    // the node only when that body changed.
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteCallsForGMBuffer(for_stmt->body_, modified_funcs, gm_param, prepend);
      if (nb != for_stmt->body_) {
        auto new_for = MutableCopy(for_stmt);
        new_for->body_ = nb;
        new_stmts.push_back(new_for);
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto nt = RewriteCallsForGMBuffer(if_stmt->then_body_, modified_funcs, gm_param, prepend);
      std::optional<StmtPtr> ne;
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        ne = RewriteCallsForGMBuffer(*else_body, modified_funcs, gm_param, prepend);
      }
      bool body_changed = (nt != if_stmt->then_body_);
      if (!body_changed && ne.has_value() && else_body.has_value()) {
        body_changed = (*ne != *else_body);
      }
      if (body_changed) {
        auto new_if = MutableCopy(if_stmt);
        new_if->then_body_ = nt;
        new_if->else_body_ = ne;
        new_stmts.push_back(new_if);
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto nb = RewriteCallsForGMBuffer(while_stmt->body_, modified_funcs, gm_param, prepend);
      if (nb != while_stmt->body_) {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = nb;
        new_stmts.push_back(new_while);
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      auto nb = RewriteCallsForGMBuffer(scope->body_, modified_funcs, gm_param, prepend);
      if (nb != scope->body_) {
        new_stmts.push_back(CloneScopeWithBody(scope, nb));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else {
      new_stmts.push_back(stmt);
    }
  }
  // Nothing changed: keep the original node so callers can compare by pointer.
  if (!any_changed) return body;
  return SeqStmts::Flatten(std::move(new_stmts), body->span_);
}

CallPtr CreateScratchTensor(const ScratchSpec& scratch, const Span& span) {
  auto shape_elem = std::make_shared<ConstInt>(scratch.elements, DataType::INDEX, span);
  auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{shape_elem}, span);
  return OpRegistry::GetInstance().Create("tensor.create", {shape_tuple},
                                          {{"dtype", std::any(scratch.dtype)},
                                           {"layout", std::any(TensorLayout::ND)},
                                           {"manual_dep", std::any(true)}},
                                          span);
}

/// Allocate a private GM buffer for each call/submit of a function in
/// @p modified_funcs.
///
/// Unlike :func:`RewriteCallsForGMBuffer`, each rewritten call gets its own
/// allocation (``gm_pipe_buffer_<counter>``), returned alongside the rewritten
/// expression so the caller can place it next to the call it serves.
StmtPtr RewriteCallsWithPerCallGMBuffer(const StmtPtr& body,
                                        const std::unordered_set<std::string>& modified_funcs,
                                        const ScratchSpec& scratch, const Span& span, int& counter) {
  auto gm_type = std::make_shared<TensorType>(std::vector<int64_t>{scratch.elements}, scratch.dtype,
                                              std::nullopt, std::nullopt);
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any_changed = false;

  auto should_rewrite = [&](const OpPtr& op) -> bool {
    auto gv = std::dynamic_pointer_cast<const GlobalVar>(op);
    return gv && modified_funcs.count(gv->name_);
  };
  auto make_gm_create = [&]() -> std::pair<StmtPtr, VarPtr> {
    std::string var_name = std::string(scratch.prefix) + std::to_string(counter++);
    auto gm_var = std::make_shared<Var>(var_name, gm_type, span);
    auto create_call = CreateScratchTensor(scratch, span);
    auto create_stmt = std::make_shared<AssignStmt>(gm_var, create_call, span);
    return std::make_pair(StmtPtr(create_stmt), gm_var);
  };
  auto try_rewrite = [&](const ExprPtr& expr) -> std::pair<StmtPtr, ExprPtr> {
    if (!expr) return std::make_pair(StmtPtr{}, ExprPtr{});

    if (auto call = std::dynamic_pointer_cast<const Call>(expr)) {
      if (!should_rewrite(call->op_)) return std::make_pair(StmtPtr{}, ExprPtr{});
      auto [create_stmt, gm_var] = make_gm_create();

      // Preserve the original Call's kwargs_ / attrs_ (compiler metadata such as
      // Call::arg_directions_) by copying rather than reconstructing.
      return std::make_pair(create_stmt, WithScratchOperand(call, gm_var, scratch.prepend));
    }
    if (auto submit = std::dynamic_pointer_cast<const Submit>(expr)) {
      if (!should_rewrite(submit->op_)) return std::make_pair(StmtPtr{}, ExprPtr{});
      auto [create_stmt, gm_var] = make_gm_create();
      return std::make_pair(create_stmt, WithScratchOperand(submit, gm_var, scratch.prepend));
    }
    return std::make_pair(StmtPtr{}, ExprPtr{});
  };

  // Dispatch on statement kind, threading each call's allocation in front of
  // the statement that uses it.
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto [create, rw] = try_rewrite(assign->value_);
      if (rw) {
        new_stmts.push_back(create);
        auto new_assign = MutableCopy(assign);
        new_assign->value_ = rw;
        new_stmts.push_back(std::move(new_assign));
        any_changed = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      auto [create, rw] = try_rewrite(eval->expr_);
      if (rw) {
        new_stmts.push_back(create);
        auto new_eval = MutableCopy(eval);
        new_eval->expr_ = rw;
        new_stmts.push_back(std::move(new_eval));
        any_changed = true;
        continue;
      }
    }
    // Control-flow and scope statements: recurse into the body, and rebuild
    // the node only when that body changed.
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteCallsWithPerCallGMBuffer(for_stmt->body_, modified_funcs, scratch, span, counter);
      if (nb != for_stmt->body_) {
        auto new_for = MutableCopy(for_stmt);
        new_for->body_ = nb;
        new_stmts.push_back(std::move(new_for));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto nt = RewriteCallsWithPerCallGMBuffer(if_stmt->then_body_, modified_funcs, scratch, span, counter);
      std::optional<StmtPtr> ne;
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        ne = RewriteCallsWithPerCallGMBuffer(*else_body, modified_funcs, scratch, span, counter);
      }
      bool body_changed = (nt != if_stmt->then_body_);
      if (!body_changed && ne.has_value() && else_body.has_value()) {
        body_changed = (*ne != *else_body);
      }
      if (body_changed) {
        auto new_if = MutableCopy(if_stmt);
        new_if->then_body_ = nt;
        new_if->else_body_ = ne;
        new_stmts.push_back(std::move(new_if));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto nb = RewriteCallsWithPerCallGMBuffer(while_stmt->body_, modified_funcs, scratch, span, counter);
      if (nb != while_stmt->body_) {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = nb;
        new_stmts.push_back(std::move(new_while));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      auto nb = RewriteCallsWithPerCallGMBuffer(scope->body_, modified_funcs, scratch, span, counter);
      if (nb != scope->body_) {
        new_stmts.push_back(CloneScopeWithBody(scope, nb));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else {
      new_stmts.push_back(stmt);
    }
  }
  if (!any_changed) return body;
  return SeqStmts::Flatten(std::move(new_stmts), body->span_);
}

bool InjectScratchInPlace(std::vector<FunctionPtr>& functions, const ScratchSpec& scratch) {
  std::unordered_map<std::string, FunctionPtr*> func_by_name;
  for (auto& func : functions) func_by_name[func->name_] = &func;

  std::unordered_map<std::string, std::unordered_set<std::string>> callers, callees;

  std::unordered_set<std::string> scratch_funcs;
  for (const auto& func : functions) {
    ScratchUseFinder finder(scratch);
    if (func->body_ && !HasScratchParam(func, scratch)) finder.VisitStmt(func->body_);
    if (finder.found) {
      scratch_funcs.insert(func->name_);
    }
  }
  if (scratch_funcs.empty()) return false;
  BuildCallGraphFromFunctions(functions, callers, callees);

  // Propagate upward, stopping at Orchestration boundaries (they materialize
  // the buffer locally instead of taking it as a parameter).
  std::unordered_set<std::string> needs_gm_param = scratch_funcs;
  std::unordered_set<std::string> orch_needs_tensor_create;
  std::vector<std::string> worklist(scratch_funcs.begin(), scratch_funcs.end());
  while (!worklist.empty()) {
    std::string name = worklist.back();
    worklist.pop_back();
    // MIX members unpack one shared tensor/scalar payload. Adding a tensor to
    // just the submitting AIV would shift the scalar slots seen by its siblings.
    if ((*func_by_name.at(name))->func_type_ == FunctionType::Group) {
      for (const auto& member : callees[name]) {
        if (needs_gm_param.insert(member).second) worklist.push_back(member);
      }
    }
    auto it = callers.find(name);
    if (it == callers.end()) continue;
    for (const auto& caller_name : it->second) {
      auto fit = func_by_name.find(caller_name);
      if (fit == func_by_name.end()) continue;
      if (IsOrchestrationLike((*fit->second)->func_type_)) {
        orch_needs_tensor_create.insert(caller_name);
      } else {
        if (needs_gm_param.insert(caller_name).second) worklist.push_back(caller_name);
      }
    }
  }

  for (auto& func : functions) {
    if (needs_gm_param.count(func->name_) && !HasScratchParam(func, scratch)) {
      func = AddScratchParam(func, scratch);
    }
  }

  for (auto& func : functions) {
    if (!needs_gm_param.count(func->name_)) continue;

    VarPtr gm_param;
    for (const auto& p : func->params_) {
      if (p->name_hint_ == scratch.param) {
        gm_param = p;
        break;
      }
    }
    INTERNAL_CHECK_SPAN(gm_param, func->span_)
        << "Internal error: " << func->name_ << " should have " << scratch.param;

    std::unordered_set<std::string> mod_callees;
    auto ci = callees.find(func->name_);
    if (ci != callees.end()) {
      for (const auto& c : ci->second) {
        if (needs_gm_param.count(c)) mod_callees.insert(c);
      }
    }

    if (!mod_callees.empty()) {
      auto nb = RewriteCallsForGMBuffer(func->body_, mod_callees, gm_param, scratch.prepend);
      auto updated = MutableCopy(func);
      updated->body_ = nb;
      func = updated;
    }
  }

  for (auto& func : functions) {
    // Must stay in lockstep with the boundary marking above: widening only one
    // side leaves a caller passing the old arity to a rewritten ``__gm_pipe_buffer``.
    if (!orch_needs_tensor_create.count(func->name_)) continue;

    const std::unordered_set<std::string>& mod_callees = needs_gm_param;
    if (mod_callees.empty()) continue;

    int counter = 0;
    auto new_body = RewriteCallsWithPerCallGMBuffer(func->body_, mod_callees, scratch, func->span_, counter);
    auto updated = MutableCopy(func);
    updated->body_ = new_body;
    func = updated;
  }
  return true;
}

}  // namespace

namespace pass {

Pass InjectGMPipeBuffer() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!backend::BackendConfig::IsConfigured()) {
      return program;
    }
    std::vector<FunctionPtr> functions;
    functions.reserve(program->functions_.size());
    for (const auto& [gvar, func] : program->functions_) {
      functions.push_back(func);
    }
    const auto& backend = PassContext::Current()->GetBackendHandler();
    bool changed = backend->RequiresGMPipeBuffer() && InjectScratchInPlace(functions, kPipeScratch);
    const auto bytes = backend->GetUrmaScatterScratchBytes();
    if (bytes > 0) {
      changed |= InjectScratchInPlace(
          functions, {backend::kUrmaScatterScratchParam, "urma_scatter_scratch_", bytes, DataType::UINT8,
                      [](const CallPtr& op) {
                        // One shared scratch serves both directions: the slot is per
                        // physical AIV and a Gather pair fits the same 64-byte line.
                        return IsOp(op, "pld.tile.get_async") || IsOp(op, "pld.tile.put_async");
                      },
                      true});
    }
    if (!changed) return program;
    return std::make_shared<Program>(functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "InjectGMPipeBuffer", kInjectGMPipeBufferProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
