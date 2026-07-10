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

#include "pypto/codegen/orchestration/orchestration_analysis.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

std::string GetSSABaseName(const std::string& name) { return auto_name::GetCompatibleBaseName(name); }

bool IsBuiltinOp(const std::string& op_name) {
  return op_name.find("tile.") == 0 || op_name.find("tensor.") == 0 || op_name.find("system.") == 0 ||
         op_name.find("array.") == 0;
}

bool IsTensorOp(const std::string& op_name) { return op_name.find("tensor.") == 0; }

bool IsArrayOp(const std::string& op_name) { return op_name.find("array.") == 0; }

std::string FormatConstIntValue(const ConstIntPtr& c, const std::string& cpp_type) {
  int64_t v = c->value_;
  if (cpp_type != "int64_t") {
    return "static_cast<" + cpp_type + ">(" + std::to_string(v) + ")";
  }
  return std::to_string(v);
}

std::string FormatConstFloatValue(const ConstFloatPtr& c, const std::string& cpp_type) {
  double v = c->value_;
  if (cpp_type == "float") {
    return std::to_string(static_cast<float>(v));
  }
  return std::to_string(v);  // double
}

int GetOrCreateFuncId(const std::string& func_name, std::map<std::string, int>* func_name_to_id,
                      int* next_func_id) {
  if (func_name_to_id->find(func_name) == func_name_to_id->end()) {
    (*func_name_to_id)[func_name] = (*next_func_id)++;
  }
  return (*func_name_to_id)[func_name];
}

namespace {

int GetGMPipeSlotCount(int dir_mask) {
  const int bidirectional = core_affinity::kDirMaskC2V | core_affinity::kDirMaskV2C;
  if (dir_mask == bidirectional) {
    return 4;
  }
  if (dir_mask == core_affinity::kDirMaskC2V || dir_mask == core_affinity::kDirMaskV2C) {
    return 8;
  }
  return 0;
}

}  // namespace

int64_t ComputeGMPipeWorkspaceElements(const ProgramPtr& program, const FunctionPtr& root_func) {
  std::map<std::pair<int, int>, int> slot_size_by_pipe;

  std::unordered_set<std::string> visited_funcs;
  std::function<void(const std::vector<StmtPtr>&)> scan_stmts;
  std::function<void(const FunctionPtr&)> scan_func;
  scan_stmts = [&](const std::vector<StmtPtr>& stmts) {
    for (const auto& stmt : stmts) {
      auto call = transform_utils::GetCallFromStmt(stmt);
      if (op_predicates::IsInitializePipe(call)) {
        const int pipe_id = call->GetKwarg<int>("id", 0);
        const int dir_mask = call->GetKwarg<int>("dir_mask", 0);
        const int slot_size = call->GetKwarg<int>("slot_size", 0);
        if (dir_mask > 0 && slot_size > 0) {
          const auto key = std::make_pair(pipe_id, dir_mask);
          auto [it, inserted] = slot_size_by_pipe.emplace(key, slot_size);
          CHECK(inserted || it->second == slot_size)
              << "initialize_pipe for frontend pipe id " << pipe_id << " and dir_mask " << dir_mask
              << " uses inconsistent slot_size values: " << it->second << " and " << slot_size;
        }
      } else if (call) {
        auto gv = As<GlobalVar>(call->op_);
        if (gv) {
          scan_func(program->GetFunction(gv->name_));
        }
      }

      if (auto for_stmt = As<ForStmt>(stmt)) {
        scan_stmts(transform_utils::FlattenToStmts(for_stmt->body_));
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        scan_stmts(transform_utils::FlattenToStmts(if_stmt->then_body_));
        const auto& else_body = if_stmt->else_body_;
        if (else_body) {
          scan_stmts(transform_utils::FlattenToStmts(*else_body));
        }
      } else if (auto while_stmt = As<WhileStmt>(stmt)) {
        scan_stmts(transform_utils::FlattenToStmts(while_stmt->body_));
      } else if (auto scope = As<RuntimeScopeStmt>(stmt)) {
        scan_stmts(transform_utils::FlattenToStmts(scope->body_));
      }
    }
  };

  scan_func = [&](const FunctionPtr& func) {
    if (!func || !visited_funcs.insert(func->name_).second) {
      return;
    }
    if (func->body_) {
      scan_stmts(transform_utils::FlattenToStmts(func->body_));
    }
  };

  scan_func(root_func);

  int64_t total_bytes = 0;
  for (const auto& [key, slot_size] : slot_size_by_pipe) {
    const int dir_mask = key.second;
    const int slot_count = GetGMPipeSlotCount(dir_mask);
    CHECK(slot_count > 0) << "initialize_pipe has invalid dir_mask for GM slot buffer: " << dir_mask;
    CHECK(total_bytes <= std::numeric_limits<int64_t>::max() -
                             static_cast<int64_t>(slot_count) * static_cast<int64_t>(slot_size))
        << "GM slot buffer size overflow while sizing frontend pipe id " << key.first;
    total_bytes += static_cast<int64_t>(slot_count) * static_cast<int64_t>(slot_size);
  }

  if (total_bytes == 0) {
    return 0;
  }
  return (total_bytes + static_cast<int64_t>(sizeof(float)) - 1) / static_cast<int64_t>(sizeof(float));
}

// ---------------------------------------------------------------------------
// OrchestrationInfoCollector
// ---------------------------------------------------------------------------

void OrchestrationInfoCollector::VisitStmt_(const AssignStmtPtr& assign) {
  // The tuple key is registered against the *binding Var* (stable identity),
  // never the call pointer — a SubmitToCallView is transient, so keying on it
  // would desynchronise this analysis from the codegen lookup (which now also
  // keys on the binding Var).
  if (auto call = AsCallOrSubmitView(assign->value_)) {
    if (!IsBuiltinOp(call->op_->name_) && !IsOp(call, "tensor.create")) {
      if (As<TupleType>(call->GetType())) {
        std::string unique_key = "_tc_" + std::to_string(tuple_call_counter_++);
        tuple_var_to_key[assign->var_.get()] = unique_key;
      }
    }
  } else if (As<MakeTuple>(assign->value_)) {
    std::string unique_key = "_tc_" + std::to_string(tuple_call_counter_++);
    tuple_var_to_key[assign->var_.get()] = unique_key;
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_ref = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_var_to_key.find(tuple_ref.get());
      if (it != tuple_var_to_key.end()) {
        call_tuple_elements[it->second].push_back({tuple_get->index_, assign->var_.get()});
      }
    }
  }
  IRVisitor::VisitStmt_(assign);
}

// ---------------------------------------------------------------------------
// VarLineageCollector
// ---------------------------------------------------------------------------

VarLineageCollector::VarLineageCollector(ProgramPtr program) : program_(std::move(program)) {}

void VarLineageCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    var_to_param[param.get()] = param.get();
  }
}

void VarLineageCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = for_stmt->iter_args_[i];
    const Var* param = ResolveExpr(iter_arg->initValue_);
    if (param) {
      var_to_param[iter_arg.get()] = param;
      // Only propagate buffer lineage to the return_var for Tensor-type carries.
      // Scalar carries (e.g. a loop counter ``idx = batch_base + inner``) are
      // value-typed: the body may overwrite them with a freshly computed value
      // that has no relationship to the init param.  Propagating param lineage
      // to a Scalar return_var makes FindReturnedParamIndices incorrectly map
      // a Scalar return element to a param index, causing EmitTensorAlias to
      // emit ``const Tensor&`` for an int64_t variable (issue #1580).
      if (i < for_stmt->return_vars_.size() && AsTensorTypeLike(for_stmt->return_vars_[i]->GetType())) {
        var_to_param[for_stmt->return_vars_[i].get()] = param;
      }
    }
  }
  IRVisitor::VisitStmt_(for_stmt);
}

void VarLineageCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
    const auto& iter_arg = while_stmt->iter_args_[i];
    const Var* param = ResolveExpr(iter_arg->initValue_);
    if (param) {
      var_to_param[iter_arg.get()] = param;
      // Same guard as ForStmt: only propagate to Tensor return_vars.
      if (i < while_stmt->return_vars_.size() && AsTensorTypeLike(while_stmt->return_vars_[i]->GetType())) {
        var_to_param[while_stmt->return_vars_[i].get()] = param;
      }
    }
  }
  IRVisitor::VisitStmt_(while_stmt);
}

void VarLineageCollector::VisitStmt_(const AssignStmtPtr& assign) {
  if (auto src_var = AsVarLike(assign->value_)) {
    const Var* param = ResolveVar(src_var.get());
    if (param) {
      var_to_param[assign->var_.get()] = param;
    }
  } else if (auto call = AsCallOrSubmitView(assign->value_)) {
    // Propagate lineage through function calls: the result inherits lineage
    // from the Out/InOut argument. This covers sequential SPMD submissions
    // like: out = self.kernel(a, b, out) where `out` is the output buffer.
    //
    // Group/Spmd wrappers carry their effective directions in the signature —
    // DeriveCallDirections materialized them — so a plain field read covers
    // every callee kind.
    if (!IsBuiltinOp(call->op_->name_)) {
      auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
      if (callee) {
        const auto& effective_dirs = callee->param_directions_;
        // Prefer tracing through the Out/InOut arg the callee actually
        // returns (multi-Out kernels would otherwise be mis-traced to the
        // first Out, leaking scratch-buffer lineage onto the result Var).
        std::optional<size_t> returned_idx = FindReturnedParamIndex(callee, program_);
        for (size_t i = 0; i < effective_dirs.size() && i < call->args_.size(); ++i) {
          if (effective_dirs[i] != ParamDirection::Out && effective_dirs[i] != ParamDirection::InOut) {
            continue;
          }
          if (returned_idx.has_value() && i != *returned_idx) {
            continue;
          }
          if (auto arg_var = AsVarLike(call->args_[i])) {
            const Var* param = ResolveVar(arg_var.get());
            if (param) {
              var_to_param[assign->var_.get()] = param;
              break;
            }
          }
        }
      }
    }
  }
  IRVisitor::VisitStmt_(assign);
}

const Var* VarLineageCollector::ResolveVar(const Var* var) const {
  auto it = var_to_param.find(var);
  return it != var_to_param.end() ? it->second : nullptr;
}

const Var* VarLineageCollector::ResolveExpr(const ExprPtr& expr) const {
  if (auto var = AsVarLike(expr)) {
    return ResolveVar(var.get());
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// FindReturnedParamIndex
// ---------------------------------------------------------------------------

// Both functions delegate to the shared IR-level return-lineage utility,
// which traces through SSA rebinds, loop carries, builtin writeback ops,
// TupleGetItem of user calls, and Group/Spmd wrapper inner calls — with
// per-function memoization and cycle protection. See return_lineage_utils.h.

std::optional<size_t> FindReturnedParamIndex(const FunctionPtr& callee, const ProgramPtr& program) {
  return ir::return_lineage::ReturnedParamIndex(callee, program);
}

std::vector<std::optional<size_t>> FindReturnedParamIndices(const FunctionPtr& callee,
                                                            const ProgramPtr& program) {
  return ir::return_lineage::ReturnedParamIndices(callee, program);
}

}  // namespace codegen
}  // namespace pypto
