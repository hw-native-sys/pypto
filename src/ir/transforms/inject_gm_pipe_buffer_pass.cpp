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
 *        that route slot data through GM (currently Ascend910B).
 *
 * On 910B, cross-core tpush/tpop rides through a shared GM buffer instead of a
 * direct inter-core fabric. This pass runs after ExpandMixedKernel has split
 * mixed InCore functions into AIC/AIV pairs, and:
 *
 *   1. Finds every function that issues initialize_pipe ops.
 *   2. Adds a fresh __gm_pipe_buffer Out-tensor parameter to each, propagating
 *      the parameter upward through callers — except Orchestration functions,
 *      which instead get a per-call-site tensor.create that materializes the
 *      workspace locally.
 *
 * The pass is gated on BackendHandler::RequiresGMPipeBuffer(); other backends
 * see it as a no-op. Nothing else in the pipeline depends on its output, so it
 * produces the same IR properties it requires (MixedKernelExpanded).
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
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
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

/// Check if a statement list contains aic_initialize_pipe or aiv_initialize_pipe.
bool HasInitializePipeOps(const std::vector<StmtPtr>& stmts) {
  for (const auto& stmt : stmts) {
    if (op_predicates::IsInitializePipe(transform_utils::GetCallFromStmt(stmt))) return true;
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      if (HasInitializePipeOps(FlattenBody(for_stmt->body_))) return true;
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      if (HasInitializePipeOps(FlattenBody(if_stmt->then_body_))) return true;
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        if (HasInitializePipeOps(FlattenBody(*else_body))) return true;
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      if (HasInitializePipeOps(FlattenBody(while_stmt->body_))) return true;
    }
  }
  return false;
}

bool HasGMPipeBufferParam(const FunctionPtr& func) {
  for (const auto& param : func->params_) {
    if (param->name_hint_ == kGMPipeBufferName) return true;
  }
  return false;
}

void BuildCallGraphFromFunctions(const std::vector<FunctionPtr>& functions,
                                 std::unordered_map<std::string, std::unordered_set<std::string>>& callers,
                                 std::unordered_map<std::string, std::unordered_set<std::string>>& callees) {
  std::unordered_set<std::string> func_names;
  for (const auto& func : functions) func_names.insert(func->name_);
  for (const auto& func : functions) {
    std::function<void(const std::vector<StmtPtr>&)> walk = [&](const std::vector<StmtPtr>& stmts) {
      for (const auto& stmt : stmts) {
        if (auto call = transform_utils::GetCallFromStmt(stmt)) {
          auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
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
        }
      }
    };
    if (func->body_) walk(FlattenBody(func->body_));
  }
}

FunctionPtr AddGMSlotBufferParam(const FunctionPtr& func, int64_t gm_buffer_elems) {
  auto gm_type = std::make_shared<TensorType>(std::vector<int64_t>{gm_buffer_elems}, DataType::FP32,
                                              std::nullopt, std::nullopt);
  auto gm_var = std::make_shared<Var>(kGMPipeBufferName, gm_type, func->span_);
  auto new_params = func->params_;
  new_params.push_back(gm_var);
  auto new_directions = func->param_directions_;
  new_directions.push_back(ParamDirection::Out);
  auto result = MutableCopy(func);
  result->params_ = new_params;
  result->param_directions_ = new_directions;
  return result;
}

StmtPtr RewriteCallsForGMBuffer(const StmtPtr& body, const std::unordered_set<std::string>& modified_funcs,
                                const VarPtr& gm_param) {
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any_changed = false;
  for (const auto& stmt : stmts) {
    auto try_rewrite = [&](const CallPtr& call) -> CallPtr {
      if (!call) return nullptr;
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
      if (!gv || !modified_funcs.count(gv->name_)) return nullptr;
      // Copy the original Call so kwargs_ / attrs_ (e.g. Call::arg_directions_)
      // survive the rewrite — only args_ needs to grow.
      auto new_call = MutableCopy(call);
      new_call->args_.push_back(gm_param);
      return new_call;
    };
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto rw = try_rewrite(std::dynamic_pointer_cast<const Call>(assign->value_))) {
        auto new_assign = MutableCopy(assign);
        new_assign->value_ = rw;
        new_stmts.push_back(std::move(new_assign));
        any_changed = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      if (auto rw = try_rewrite(std::dynamic_pointer_cast<const Call>(eval->expr_))) {
        auto new_eval = MutableCopy(eval);
        new_eval->expr_ = rw;
        new_stmts.push_back(std::move(new_eval));
        any_changed = true;
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteCallsForGMBuffer(for_stmt->body_, modified_funcs, gm_param);
      if (nb != for_stmt->body_) {
        auto new_for = MutableCopy(for_stmt);
        new_for->body_ = nb;
        new_stmts.push_back(new_for);
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto nt = RewriteCallsForGMBuffer(if_stmt->then_body_, modified_funcs, gm_param);
      std::optional<StmtPtr> ne;
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        ne = RewriteCallsForGMBuffer(*else_body, modified_funcs, gm_param);
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
      auto nb = RewriteCallsForGMBuffer(while_stmt->body_, modified_funcs, gm_param);
      if (nb != while_stmt->body_) {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = nb;
        new_stmts.push_back(new_while);
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

CallPtr CreateGMPipeBufferTensorCreate(int64_t buffer_size_bytes, const Span& span) {
  int64_t shape_dim = (buffer_size_bytes + 3) / 4;  // FP32 elements (ceil)
  auto shape_elem = std::make_shared<ConstInt>(shape_dim, DataType::INDEX, span);
  auto shape_tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{shape_elem}, span);
  return OpRegistry::GetInstance().Create("tensor.create", {shape_tuple},
                                          {{"dtype", std::any(DataType::FP32)},
                                           {"layout", std::any(TensorLayout::ND)},
                                           {"manual_dep", std::any(true)}},
                                          span);
}

StmtPtr RewriteCallsWithPerCallGMBuffer(const StmtPtr& body,
                                        const std::unordered_set<std::string>& modified_funcs,
                                        int64_t gm_buffer_bytes, int64_t gm_buffer_elems, const Span& span,
                                        int& counter) {
  auto gm_type = std::make_shared<TensorType>(std::vector<int64_t>{gm_buffer_elems}, DataType::FP32,
                                              std::nullopt, std::nullopt);
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any_changed = false;

  auto try_rewrite = [&](const CallPtr& call) -> std::pair<StmtPtr, CallPtr> {
    if (!call) return std::make_pair(StmtPtr{}, CallPtr{});
    auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!gv || !modified_funcs.count(gv->name_)) return std::make_pair(StmtPtr{}, CallPtr{});

    std::string var_name = std::string("gm_pipe_buffer_") + std::to_string(counter++);
    auto gm_var = std::make_shared<Var>(var_name, gm_type, span);
    auto create_call = CreateGMPipeBufferTensorCreate(gm_buffer_bytes, span);
    auto create_stmt = std::make_shared<AssignStmt>(gm_var, create_call, span);

    // Preserve the original Call's kwargs_ / attrs_ (compiler metadata such as
    // Call::arg_directions_) by copying rather than reconstructing.
    auto new_call = MutableCopy(call);
    new_call->args_.push_back(gm_var);
    StmtPtr create_stmt_ptr = create_stmt;
    return std::make_pair(create_stmt_ptr, CallPtr(new_call));
  };

  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto [create, rw] = try_rewrite(std::dynamic_pointer_cast<const Call>(assign->value_));
      if (rw) {
        new_stmts.push_back(create);
        auto new_assign = MutableCopy(assign);
        new_assign->value_ = rw;
        new_stmts.push_back(std::move(new_assign));
        any_changed = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      auto [create, rw] = try_rewrite(std::dynamic_pointer_cast<const Call>(eval->expr_));
      if (rw) {
        new_stmts.push_back(create);
        auto new_eval = MutableCopy(eval);
        new_eval->expr_ = rw;
        new_stmts.push_back(std::move(new_eval));
        any_changed = true;
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteCallsWithPerCallGMBuffer(for_stmt->body_, modified_funcs, gm_buffer_bytes,
                                                gm_buffer_elems, span, counter);
      if (nb != for_stmt->body_) {
        auto new_for = MutableCopy(for_stmt);
        new_for->body_ = nb;
        new_stmts.push_back(std::move(new_for));
        any_changed = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto nt = RewriteCallsWithPerCallGMBuffer(if_stmt->then_body_, modified_funcs, gm_buffer_bytes,
                                                gm_buffer_elems, span, counter);
      std::optional<StmtPtr> ne;
      const auto& else_body = if_stmt->else_body_;
      if (else_body.has_value()) {
        ne = RewriteCallsWithPerCallGMBuffer(*else_body, modified_funcs, gm_buffer_bytes, gm_buffer_elems,
                                             span, counter);
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
      auto nb = RewriteCallsWithPerCallGMBuffer(while_stmt->body_, modified_funcs, gm_buffer_bytes,
                                                gm_buffer_elems, span, counter);
      if (nb != while_stmt->body_) {
        auto new_while = MutableCopy(while_stmt);
        new_while->body_ = nb;
        new_stmts.push_back(std::move(new_while));
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

int64_t ComputeGMBufferSizeFromPipeOps(const std::vector<FunctionPtr>& functions) {
  int64_t max_bytes = 0;
  std::function<void(const std::vector<StmtPtr>&)> scan_stmts;
  scan_stmts = [&](const std::vector<StmtPtr>& stmts) {
    for (const auto& stmt : stmts) {
      auto call = transform_utils::GetCallFromStmt(stmt);
      if (op_predicates::IsInitializePipe(call)) {
        int ss = call->GetKwarg<int>("slot_size", 0);
        int dm = call->GetKwarg<int>("dir_mask", 0);
        if (ss > 0 && dm != 0) {
          // Mirror the per-direction `buffer_size` formula from
          // cross_core_pipe.cpp (buffer_size = slot_size * GetSlotNumForDirMask).
          // For bidirectional pipes the two reserve_buffers share this workspace
          // at runtime via the AllocateMemoryAddr layout; the AIC/AIV-side a2a3
          // tests pin both the per-direction reserve sizes and the workspace
          // shape and would catch any drift here.
          int64_t bytes =
              static_cast<int64_t>(ss) * static_cast<int64_t>(cross_core_pipe::GetSlotNumForDirMask(dm));
          if (bytes > max_bytes) {
            max_bytes = bytes;
          }
        }
      }
      if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
        scan_stmts(FlattenBody(for_stmt->body_));
      } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
        scan_stmts(FlattenBody(if_stmt->then_body_));
        if (if_stmt->else_body_.has_value()) {
          scan_stmts(FlattenBody(if_stmt->else_body_.value()));
        }
      } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
        scan_stmts(FlattenBody(while_stmt->body_));
      }
    }
  };
  for (const auto& func : functions) {
    if (!func->body_) continue;
    scan_stmts(FlattenBody(func->body_));
  }
  return max_bytes;
}

void InjectGMSlotBufferInPlace(std::vector<FunctionPtr>& functions) {
  std::unordered_map<std::string, FunctionPtr*> func_by_name;
  for (auto& func : functions) func_by_name[func->name_] = &func;

  std::unordered_map<std::string, std::unordered_set<std::string>> callers, callees;
  BuildCallGraphFromFunctions(functions, callers, callees);

  std::unordered_set<std::string> pipe_funcs;
  for (const auto& func : functions) {
    if (!HasGMPipeBufferParam(func) && func->body_ && HasInitializePipeOps(FlattenBody(func->body_))) {
      pipe_funcs.insert(func->name_);
    }
  }
  if (pipe_funcs.empty()) return;

  int64_t gm_buffer_bytes = ComputeGMBufferSizeFromPipeOps(functions);
  INTERNAL_CHECK(gm_buffer_bytes > 0) << "Internal error: cross-core pipe functions found but no "
                                         "initialize_pipe ops to determine buffer size";

  // Propagate upward, stopping at Orchestration boundaries (they materialize
  // the buffer locally instead of taking it as a parameter).
  std::unordered_set<std::string> needs_gm_param = pipe_funcs;
  std::unordered_set<std::string> orch_needs_tensor_create;
  std::vector<std::string> worklist(pipe_funcs.begin(), pipe_funcs.end());
  while (!worklist.empty()) {
    std::string name = worklist.back();
    worklist.pop_back();
    auto it = callers.find(name);
    if (it == callers.end()) continue;
    for (const auto& caller_name : it->second) {
      auto fit = func_by_name.find(caller_name);
      if (fit == func_by_name.end()) continue;
      if ((*fit->second)->func_type_ == FunctionType::Orchestration) {
        orch_needs_tensor_create.insert(caller_name);
      } else {
        if (needs_gm_param.insert(caller_name).second) worklist.push_back(caller_name);
      }
    }
  }

  int64_t gm_buffer_elems = (gm_buffer_bytes + 3) / 4;

  for (auto& func : functions) {
    if (needs_gm_param.count(func->name_) && !HasGMPipeBufferParam(func)) {
      func = AddGMSlotBufferParam(func, gm_buffer_elems);
    }
  }

  for (auto& func : functions) {
    if (!needs_gm_param.count(func->name_)) continue;

    VarPtr gm_param;
    for (const auto& p : func->params_) {
      if (p->name_hint_ == kGMPipeBufferName) {
        gm_param = p;
        break;
      }
    }
    INTERNAL_CHECK_SPAN(gm_param, func->span_)
        << "Internal error: " << func->name_ << " should have " << kGMPipeBufferName;

    std::unordered_set<std::string> mod_callees;
    auto ci = callees.find(func->name_);
    if (ci != callees.end()) {
      for (const auto& c : ci->second) {
        if (needs_gm_param.count(c)) mod_callees.insert(c);
      }
    }

    if (!mod_callees.empty()) {
      auto nb = RewriteCallsForGMBuffer(func->body_, mod_callees, gm_param);
      auto updated = MutableCopy(func);
      updated->body_ = nb;
      func = updated;
    }
  }

  if (orch_needs_tensor_create.empty()) return;

  for (auto& func : functions) {
    if (!orch_needs_tensor_create.count(func->name_)) continue;

    std::unordered_set<std::string> mod_callees;
    auto ci = callees.find(func->name_);
    if (ci != callees.end()) {
      for (const auto& c : ci->second) {
        if (needs_gm_param.count(c)) mod_callees.insert(c);
      }
    }
    if (mod_callees.empty()) continue;

    int counter = 0;
    auto new_body = RewriteCallsWithPerCallGMBuffer(func->body_, mod_callees, gm_buffer_bytes,
                                                    gm_buffer_elems, func->span_, counter);
    auto updated = MutableCopy(func);
    updated->body_ = new_body;
    func = updated;
  }
}

}  // namespace

namespace pass {

Pass InjectGMPipeBuffer() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!backend::BackendConfig::IsConfigured() ||
        !PassContext::Current()->GetBackendHandler()->RequiresGMPipeBuffer()) {
      return program;
    }
    std::vector<FunctionPtr> functions;
    functions.reserve(program->functions_.size());
    for (const auto& [gvar, func] : program->functions_) {
      functions.push_back(func);
    }
    InjectGMSlotBufferInPlace(functions);
    return std::make_shared<Program>(functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "InjectGMPipeBuffer", kInjectGMPipeBufferProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
