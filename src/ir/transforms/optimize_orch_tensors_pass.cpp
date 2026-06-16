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

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;

namespace {

// ============================================================================
// Shared helpers
// ============================================================================

/// Find a function by name in a program.
FunctionPtr FindFunction(const ProgramPtr& program, const std::string& name) {
  for (const auto& [gvar, func] : program->functions_) {
    if (func->name_ == name) return func;
  }
  return nullptr;
}

/// Get the GlobalVar name from a Call, or empty string.
std::string GetCallFuncName(const CallPtr& call) {
  auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
  return gvar ? gvar->name_ : "";
}

/// Compute row-major strides from a shape: [D1*D2*...*Dn, D2*...*Dn, ..., 1].
/// Returns empty vector if any dimension is not a ConstInt.
std::vector<ExprPtr> ComputeRowMajorStrides(const std::vector<ExprPtr>& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (const auto& dim : shape) {
    auto ci = As<ConstInt>(dim);
    if (!ci) return {};
    dims.push_back(ci->value_);
  }

  size_t ndim = dims.size();
  std::vector<ExprPtr> strides(ndim);
  int64_t product = 1;
  for (size_t i = ndim; i > 0; --i) {
    strides[i - 1] = std::make_shared<ConstInt>(product, DataType::INDEX, Span::unknown());
    product *= dims[i - 1];
  }
  return strides;
}

std::string MakeUniqueFunctionName(const ProgramPtr& program, const std::string& base_name) {
  if (!program || !program->GetFunction(base_name)) return base_name;
  for (size_t suffix = 1;; ++suffix) {
    auto candidate = base_name + "_" + std::to_string(suffix);
    if (!program->GetFunction(candidate)) return candidate;
  }
}

/// Count Var/IterArg references to `target` inside a statement subtree.
size_t CountVarRefsInStmt(const StmtPtr& stmt, const Var* target) {
  class Counter : public IRVisitor {
   public:
    explicit Counter(const Var* target) : target_(target) {}

    [[nodiscard]] size_t count() const { return count_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (op.get() == target_) ++count_;
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (op.get() == target_) ++count_;
      IRVisitor::VisitExpr_(op);
    }

   private:
    const Var* target_;
    size_t count_ = 0;
  };

  Counter counter(target);
  counter.VisitStmt(stmt);
  return counter.count();
}

size_t CountVarRefsInExpr(const ExprPtr& expr, const Var* target) {
  class Counter : public IRVisitor {
   public:
    explicit Counter(const Var* target) : target_(target) {}

    [[nodiscard]] size_t count() const { return count_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (op.get() == target_) ++count_;
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (op.get() == target_) ++count_;
      IRVisitor::VisitExpr_(op);
    }

   private:
    const Var* target_;
    size_t count_ = 0;
  };

  Counter counter(target);
  counter.VisitExpr(expr);
  return counter.count();
}

bool ExprReferencesOnlyVarsIn(const ExprPtr& expr, const std::unordered_set<const Var*>& allowed) {
  class Checker : public IRVisitor {
   public:
    explicit Checker(const std::unordered_set<const Var*>& allowed) : allowed_(allowed) {}

    [[nodiscard]] bool ok() const { return ok_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      if (!allowed_.count(op.get())) ok_ = false;
    }

    void VisitExpr_(const IterArgPtr& op) override {
      if (!allowed_.count(op.get())) ok_ = false;
    }

   private:
    const std::unordered_set<const Var*>& allowed_;
    bool ok_ = true;
  };

  Checker checker(allowed);
  checker.VisitExpr(expr);
  return checker.ok();
}

std::unordered_set<const Var*> CollectAllowedVars(const std::vector<VarPtr>& vars,
                                                  const Var* extra_allowed = nullptr) {
  std::unordered_set<const Var*> allowed;
  allowed.reserve(vars.size() + (extra_allowed ? 1 : 0));
  for (const auto& var : vars) {
    if (var) allowed.insert(var.get());
  }
  if (extra_allowed) allowed.insert(extra_allowed);
  return allowed;
}

bool ExprsReferenceOnlyVarsIn(const std::vector<ExprPtr>& exprs,
                              const std::unordered_set<const Var*>& allowed) {
  for (const auto& expr : exprs) {
    if (!ExprReferencesOnlyVarsIn(expr, allowed)) return false;
  }
  return true;
}

std::unordered_map<std::string, FunctionPtr> BuildFunctionLookup(const ProgramPtr& program) {
  std::unordered_map<std::string, FunctionPtr> lookup;
  if (!program) return lookup;
  lookup.reserve(program->functions_.size());
  for (const auto& [gvar, func] : program->functions_) {
    if (func) lookup.emplace(func->name_, func);
  }
  return lookup;
}

using LoopIterInitSubstMap = std::unordered_map<const Var*, ExprPtr>;

class ScopedLoopIterInitSubst {
 public:
  ScopedLoopIterInitSubst(LoopIterInitSubstMap* subst, const std::vector<IterArgPtr>& iter_args)
      : subst_(subst), saved_(*subst) {
    for (const auto& iter_arg : iter_args) {
      if (iter_arg && iter_arg->initValue_) {
        (*subst_)[iter_arg.get()] = iter_arg->initValue_;
      }
    }
  }

  ~ScopedLoopIterInitSubst() { *subst_ = std::move(saved_); }

 private:
  LoopIterInitSubstMap* subst_;
  LoopIterInitSubstMap saved_;
};

bool IsAllZeroOffsets(const std::vector<ExprPtr>& offsets) {
  for (const auto& offset : offsets) {
    auto ci = As<ConstInt>(offset);
    if (!ci || ci->value_ != 0) return false;
  }
  return true;
}

bool IsTensorAllocationOp(const CallPtr& call) {
  if (!call || std::dynamic_pointer_cast<const GlobalVar>(call->op_)) return false;
  return call->op_->name_ == "tensor.create" || call->op_->name_ == "tensor.full";
}

bool IsOutputDirection(ParamDirection direction, bool include_inout) {
  return direction == ParamDirection::Out || (include_inout && direction == ParamDirection::InOut);
}

std::unordered_set<const Var*> CollectLoopLocalTensorAllocs(const ForStmtPtr& loop) {
  class Collector : public IRVisitor {
   public:
    [[nodiscard]] const std::unordered_set<const Var*>& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (IsTensorAllocationOp(call) && As<TensorType>(op->var_->GetType())) {
        result_.insert(op->var_.get());
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    std::unordered_set<const Var*> result_;
  };

  if (!loop) return {};
  Collector collector;
  collector.VisitStmt(loop->body_);
  return collector.result();
}

std::vector<size_t> CollectOutParamIndices(const FunctionPtr& func) {
  std::vector<size_t> result;
  if (!func) return result;
  for (size_t i = 0; i < func->param_directions_.size() && i < func->params_.size(); ++i) {
    if (IsOutputDirection(func->param_directions_[i], /*include_inout=*/true)) {
      result.push_back(i);
    }
  }
  return result;
}

bool IsTensorTypedArg(const ExprPtr& arg) {
  TypePtr ty = arg ? arg->GetType() : TypePtr{};
  if (!ty) return false;
  return AsTensorTypeLike(ty) || As<TupleType>(ty);
}

/// Info about an InCore function's Out params and their return mappings.
struct OutParamReturnMapping {
  size_t param_index;   ///< Position in param list
  size_t return_index;  ///< Which return value stores to this Out param
  VarPtr param_var;     ///< The Out param variable
};

/// Build the mapping from Out params to return indices for an InCore function.
/// Scans tile.store calls before the ReturnStmt to find which Out param
/// each return value stores to.
std::vector<OutParamReturnMapping> BuildOutParamReturnMappings(const FunctionPtr& func,
                                                               bool include_inout = false) {
  // Collect output param vars and their indices.
  std::unordered_map<const Var*, size_t> out_var_to_param_idx;
  for (size_t i = 0; i < func->params_.size(); ++i) {
    if (i < func->param_directions_.size() && IsOutputDirection(func->param_directions_[i], include_inout)) {
      out_var_to_param_idx[func->params_[i].get()] = i;
    }
  }
  if (out_var_to_param_idx.empty()) return {};

  auto body_stmts = FlattenToStmts(func->body_);

  // Build var->assign map for quick lookup
  std::unordered_map<const Var*, AssignStmtPtr> var_def;
  for (const auto& stmt : body_stmts) {
    if (auto assign = As<AssignStmt>(stmt)) {
      var_def[assign->var_.get()] = assign;
    }
  }

  std::unordered_map<const Var*, ExprPtr> loop_return_to_init;
  for (const auto& stmt : body_stmts) {
    if (auto loop = As<ForStmt>(stmt)) {
      for (size_t i = 0; i < loop->return_vars_.size() && i < loop->iter_args_.size(); ++i) {
        loop_return_to_init[loop->return_vars_[i].get()] = loop->iter_args_[i]->initValue_;
      }
    } else if (auto loop = As<WhileStmt>(stmt)) {
      for (size_t i = 0; i < loop->return_vars_.size() && i < loop->iter_args_.size(); ++i) {
        loop_return_to_init[loop->return_vars_[i].get()] = loop->iter_args_[i]->initValue_;
      }
    }
  }

  // Find return statement
  ReturnStmtPtr return_stmt;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      return_stmt = ret;
      break;
    }
  }
  if (!return_stmt) return {};

  std::vector<OutParamReturnMapping> result;

  for (size_t ret_i = 0; ret_i < return_stmt->value_.size(); ++ret_i) {
    auto ret_var = As<Var>(return_stmt->value_[ret_i]);
    if (!ret_var) continue;

    auto def_it = var_def.find(ret_var.get());
    if (def_it == var_def.end()) {
      auto loop_it = loop_return_to_init.find(ret_var.get());
      if (loop_it == loop_return_to_init.end()) continue;
      auto init_var = AsVarLike(loop_it->second);
      if (!init_var) continue;
      auto param_it = out_var_to_param_idx.find(init_var.get());
      if (param_it == out_var_to_param_idx.end()) continue;
      result.push_back({param_it->second, ret_i, func->params_[param_it->second]});
      continue;
    }

    auto call = As<Call>(def_it->second->value_);
    if (!call || call->op_->name_ != "tile.store") continue;

    if (call->args_.size() < 3) continue;
    auto out_tensor = As<Var>(call->args_[2]);
    if (!out_tensor) continue;

    auto param_it = out_var_to_param_idx.find(out_tensor.get());
    if (param_it == out_var_to_param_idx.end()) continue;

    result.push_back({param_it->second, ret_i, func->params_[param_it->second]});
  }

  return result;
}

// ============================================================================
// Pattern 1: IterArgReuseOptimizer
//
// Detects when a tensor.create for an InCore Out param is inside a
// ForStmt/WhileStmt loop where the InCore result feeds back as an iter-arg,
// and the corresponding In param receives the iter-arg value.
//
// Optimization: remove the tensor.create, remove the Out param from the InCore
// function, promote the In param to InOut, redirect tile.store to the In param.
// ============================================================================

class IterArgReuseOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto reuse_results = Analyze(program, incore_names);

    // Rewrite InCore functions
    std::unordered_map<std::string, FunctionPtr> rewritten_incores;
    for (auto& [fname, reuse] : reuse_results) {
      auto func = FindFunction(program, fname);
      if (!func) continue;
      rewritten_incores[fname] = RewriteIncore(func, reuse.merges);
    }

    // Build the new function list
    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (rewritten_incores.count(func->name_)) {
        new_functions.push_back(rewritten_incores[func->name_]);
      } else if (!incore_names.count(func->name_)) {
        // Orchestration function: rewrite call sites
        DeadCreateScanner scanner(reuse_results);
        scanner.VisitStmt(func->body_);

        CallSiteRewriter rewriter(reuse_results, rewritten_incores, scanner.dead_creates());
        auto new_body = rewriter.VisitStmt(func->body_);
        if (new_body.get() != func->body_.get()) {
          auto new_func = MutableCopy(func);
          new_func->body_ = new_body;
          new_functions.push_back(new_func);
        } else {
          new_functions.push_back(func);
        }
      } else {
        new_functions.push_back(func);
      }
    }

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  /// A single Out->In merge for an InCore function.
  struct OutToInMerge {
    size_t out_param_index;
    size_t in_param_index;
  };

  /// Per-InCore-function analysis result.
  struct AnalysisResult {
    std::string func_name;
    std::vector<OutToInMerge> merges;
  };

  // -- Analysis: IRVisitor that finds ForStmt/WhileStmt with iter-arg reuse --

  class LoopAnalyzer : public IRVisitor {
   public:
    LoopAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names,
                 const std::unordered_map<std::string, std::vector<OutParamReturnMapping>>& out_mappings)
        : program_(program), incore_names_(incore_names), out_mappings_(out_mappings) {}

    const std::unordered_map<std::string, AnalysisResult>& results() const { return results_; }

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);  // Recurse into body first
      if (!op->iter_args_.empty()) {
        AnalyzeLoop(op->iter_args_, op->body_);
      }
    }

    void VisitStmt_(const WhileStmtPtr& op) override {
      IRVisitor::VisitStmt_(op);
      if (!op->iter_args_.empty()) {
        AnalyzeLoop(op->iter_args_, op->body_);
      }
    }

   private:
    void AnalyzeLoop(const std::vector<IterArgPtr>& iter_args, const StmtPtr& body) {
      auto loop_body_stmts = FlattenToStmts(body);

      // Collect tensor.create vars and InCore call assignments
      std::unordered_set<const Var*> tensor_create_vars;
      std::vector<AssignStmtPtr> incore_calls;

      for (const auto& stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(stmt);
        if (!assign) continue;
        auto call = As<Call>(assign->value_);
        if (!call) continue;

        if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.create") {
          tensor_create_vars.insert(assign->var_.get());
        } else {
          auto fname = GetCallFuncName(call);
          if (incore_names_.count(fname)) {
            incore_calls.push_back(assign);
          }
        }
      }

      // Find yield statement
      auto yield = transform_utils::FindYieldStmt(body);
      if (!yield) return;

      // Build tuple extract map: call_result_var -> {tuple_index -> dest_var}
      std::unordered_map<const Var*, std::unordered_map<size_t, const Var*>> tuple_extracts;
      for (const auto& stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(stmt);
        if (!assign) continue;
        auto tgi = As<TupleGetItemExpr>(assign->value_);
        if (!tgi) continue;
        if (auto src_var = As<Var>(tgi->tuple_)) {
          tuple_extracts[src_var.get()][static_cast<size_t>(tgi->index_)] = assign->var_.get();
        }
      }

      // Analyze each InCore call
      for (const auto& call_assign : incore_calls) {
        auto call = As<Call>(call_assign->value_);
        auto fname = GetCallFuncName(call);
        auto mapping_it = out_mappings_.find(fname);
        if (mapping_it == out_mappings_.end()) continue;
        const auto& out_param_mappings = mapping_it->second;

        auto incore_func = FindFunction(program_, fname);
        if (!incore_func) continue;

        AnalysisResult reuse_result;
        reuse_result.func_name = fname;

        for (const auto& opm : out_param_mappings) {
          if (opm.param_index >= call->args_.size()) continue;
          auto out_arg_var = As<Var>(call->args_[opm.param_index]);
          if (!out_arg_var || !tensor_create_vars.count(out_arg_var.get())) continue;

          // Trace return[ret_i] -> yield[y_i]
          size_t yield_index = SIZE_MAX;

          if (out_param_mappings.size() == 1 && incore_func->return_types_.size() == 1) {
            for (size_t yi = 0; yi < yield->value_.size(); ++yi) {
              auto yv = As<Var>(yield->value_[yi]);
              if (yv && yv.get() == call_assign->var_.get()) {
                yield_index = yi;
                break;
              }
            }
          } else {
            auto extract_it = tuple_extracts.find(call_assign->var_.get());
            if (extract_it != tuple_extracts.end()) {
              auto ret_it = extract_it->second.find(opm.return_index);
              if (ret_it != extract_it->second.end()) {
                for (size_t yi = 0; yi < yield->value_.size(); ++yi) {
                  auto yv = As<Var>(yield->value_[yi]);
                  if (yv && yv.get() == ret_it->second) {
                    yield_index = yi;
                    break;
                  }
                }
              }
            }
          }
          if (yield_index == SIZE_MAX) continue;

          // Check iter_arg[yield_index] maps to an In param of the same call
          if (yield_index >= iter_args.size()) continue;
          const auto* iter_arg_ptr = iter_args[yield_index].get();

          for (size_t arg_i = 0; arg_i < call->args_.size(); ++arg_i) {
            const Var* raw_ptr = nullptr;
            if (auto var = As<Var>(call->args_[arg_i])) {
              raw_ptr = var.get();
            } else if (auto ia = As<IterArg>(call->args_[arg_i])) {
              raw_ptr = ia.get();
            }
            if (raw_ptr != iter_arg_ptr) continue;

            if (arg_i < incore_func->param_directions_.size() &&
                incore_func->param_directions_[arg_i] == ParamDirection::In) {
              reuse_result.merges.push_back({opm.param_index, arg_i});
            }
            break;
          }
        }

        if (!reuse_result.merges.empty()) {
          if (results_.find(fname) == results_.end()) {
            results_[fname] = std::move(reuse_result);
          }
        }
      }
    }

    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    const std::unordered_map<std::string, std::vector<OutParamReturnMapping>>& out_mappings_;
    std::unordered_map<std::string, AnalysisResult> results_;
  };

  // -- Analysis: InCore internal In↔Out param pairings ----------------------
  //
  // A callee's In param and Out param are aliasing-compatible when the callee
  // reads the In fully via `tile.load` and writes the Out fully via `tile.store`
  // of a value that flows through at least one loop iter_arg chain starting
  // from that tile.load. The loop hop is what signals the In/Out are meant to
  // share storage (an accumulator); a plain load→store pair does not.

  /// Check that a tile.load call reads the full tensor — all offsets zero and
  /// both `shapes` and `valid_shapes` match the tensor shape dimension-by-
  /// dimension. `valid_shapes` differs from `shapes` for masked/padded loads,
  /// which must NOT be treated as full loads.
  static bool IsFullTensorLoad(const CallPtr& load_call, const TensorTypePtr& tensor_type) {
    if (!load_call || load_call->args_.size() < 4 || !tensor_type) return false;
    auto offsets = As<MakeTuple>(load_call->args_[1]);
    auto load_shape = As<MakeTuple>(load_call->args_[2]);
    auto valid_shape = As<MakeTuple>(load_call->args_[3]);
    if (!offsets || !load_shape || !valid_shape) return false;
    const size_t ndim = tensor_type->shape_.size();
    if (offsets->elements_.size() != ndim || load_shape->elements_.size() != ndim ||
        valid_shape->elements_.size() != ndim) {
      return false;
    }
    for (size_t i = 0; i < ndim; ++i) {
      auto want = std::dynamic_pointer_cast<const ConstInt>(tensor_type->shape_[i]);
      auto got_load = std::dynamic_pointer_cast<const ConstInt>(load_shape->elements_[i]);
      auto got_valid = std::dynamic_pointer_cast<const ConstInt>(valid_shape->elements_[i]);
      if (!want || !got_load || !got_valid) return false;
      if (want->value_ != got_load->value_ || want->value_ != got_valid->value_) return false;
      if (!IsConstValue(offsets->elements_[i], 0)) return false;
    }
    return true;
  }

  /// Compare two TensorTypes for compatible constant shape + dtype.
  static bool TensorTypesMatch(const TypePtr& a, const TypePtr& b) {
    auto ta = As<TensorType>(a);
    auto tb = As<TensorType>(b);
    if (!ta || !tb || ta->dtype_ != tb->dtype_) return false;
    if (ta->shape_.size() != tb->shape_.size()) return false;
    for (size_t i = 0; i < ta->shape_.size(); ++i) {
      auto ca = std::dynamic_pointer_cast<const ConstInt>(ta->shape_[i]);
      auto cb = std::dynamic_pointer_cast<const ConstInt>(tb->shape_[i]);
      if (!ca || !cb || ca->value_ != cb->value_) return false;
    }
    return true;
  }

  /// Walk body collecting: AssignStmt var_def map, ForStmt/WhileStmt
  /// return_var → iter_arg init value map, and the top-level ReturnStmt.
  class IterChainCollector : public IRVisitor {
   public:
    std::unordered_map<const Var*, AssignStmtPtr> var_def;
    std::unordered_map<const Var*, ExprPtr> return_var_to_init;
    ReturnStmtPtr return_stmt;

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      var_def[op->var_.get()] = op;
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const ReturnStmtPtr& op) override {
      if (!return_stmt) return_stmt = op;
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const ForStmtPtr& op) override {
      for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
        return_var_to_init[op->return_vars_[i].get()] = op->iter_args_[i]->initValue_;
      }
      IRVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const WhileStmtPtr& op) override {
      for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
        return_var_to_init[op->return_vars_[i].get()] = op->iter_args_[i]->initValue_;
      }
      IRVisitor::VisitStmt_(op);
    }
  };

  /// For each Out param, trace `tile.store` source back through loop iter_arg
  /// chains to a `tile.load` of an In param. Returns (in_idx, out_idx) pairs
  /// where the chain exists, types match, and the load covers the full tensor.
  static std::vector<std::pair<size_t, size_t>> BuildInOutParamPairings(const FunctionPtr& func) {
    std::vector<std::pair<size_t, size_t>> pairings;

    std::unordered_map<const Var*, size_t> in_param_idx;
    for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i) {
      if (func->param_directions_[i] != ParamDirection::In) continue;
      if (!As<TensorType>(func->params_[i]->GetType())) continue;
      in_param_idx[func->params_[i].get()] = i;
    }
    auto out_mappings = BuildOutParamReturnMappings(func);
    if (in_param_idx.empty() || out_mappings.empty()) return pairings;

    IterChainCollector collector;
    collector.VisitStmt(func->body_);
    if (!collector.return_stmt) return pairings;

    std::unordered_set<size_t> used_in_indices;
    for (const auto& opm : out_mappings) {
      if (opm.return_index >= collector.return_stmt->value_.size()) continue;
      auto ret_var = As<Var>(collector.return_stmt->value_[opm.return_index]);
      if (!ret_var) continue;
      auto ret_def = collector.var_def.find(ret_var.get());
      if (ret_def == collector.var_def.end()) continue;
      auto store_call = As<Call>(ret_def->second->value_);
      if (!store_call || store_call->op_->name_ != "tile.store" || store_call->args_.empty()) continue;

      // Trace backward through iter_arg chains. Require at least one loop hop:
      // a bare tile.load → tile.store without an accumulator has no semantic
      // indication that In and Out were intended to alias.
      const Var* current = nullptr;
      if (auto src_var = As<Var>(store_call->args_[0])) current = src_var.get();
      int loop_hops = 0;
      for (int hops = 0; hops < 16 && current; ++hops) {
        auto it = collector.return_var_to_init.find(current);
        if (it == collector.return_var_to_init.end()) break;
        auto init_var = As<Var>(it->second);
        if (!init_var) {
          current = nullptr;
          break;
        }
        current = init_var.get();
        ++loop_hops;
      }
      if (!current || loop_hops == 0) continue;

      auto load_def = collector.var_def.find(current);
      if (load_def == collector.var_def.end()) continue;
      auto load_call = As<Call>(load_def->second->value_);
      if (!load_call || load_call->op_->name_ != "tile.load" || load_call->args_.empty()) continue;
      auto load_src = As<Var>(load_call->args_[0]);
      if (!load_src) continue;
      auto in_it = in_param_idx.find(load_src.get());
      if (in_it == in_param_idx.end()) continue;

      auto tensor_type = As<TensorType>(func->params_[in_it->second]->GetType());
      if (!TensorTypesMatch(tensor_type, opm.param_var->GetType())) continue;
      if (!IsFullTensorLoad(load_call, tensor_type)) continue;

      if (!used_in_indices.insert(in_it->second).second) continue;
      pairings.emplace_back(in_it->second, opm.param_index);
    }
    return pairings;
  }

  // -- Analysis: standalone (non-looped) InCore calls whose In/Out can merge -

  /// One-shot visitor that collects everything the standalone analyzer needs
  /// from an orchestration function body: per-Var use counts, the set of Vars
  /// assigned by `tensor.create`, and the expression AST of each AssignStmt.
  /// Keeping it all in a single walk keeps per-function analysis O(N).
  ///
  /// Counts exclude definitional occurrences (AssignStmt LHS, loop_var,
  /// return_vars, iter_arg self-refs) so `use_count[v]` is the number of
  /// real reads of `v` in expressions.
  class FunctionBodyIndex : public IRVisitor {
   public:
    std::unordered_map<const Var*, size_t> use_count;
    std::unordered_set<const Var*> local_creates;

   protected:
    void VisitExpr_(const VarPtr& op) override { ++use_count[op.get()]; }
    void VisitExpr_(const IterArgPtr& op) override { ++use_count[op.get()]; }

    void VisitStmt_(const AssignStmtPtr& op) override {
      // Skip LHS (a def); visit only the RHS value.
      VisitExpr(op->value_);
      if (auto call = As<Call>(op->value_); call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_) &&
                                            call->op_->name_ == "tensor.create") {
        local_creates.insert(op->var_.get());
      }
    }

    void VisitStmt_(const ForStmtPtr& op) override {
      VisitExpr(op->start_);
      VisitExpr(op->stop_);
      VisitExpr(op->step_);
      for (const auto& ia : op->iter_args_) {
        if (ia->initValue_) VisitExpr(ia->initValue_);
      }
      VisitStmt(op->body_);
    }

    void VisitStmt_(const WhileStmtPtr& op) override {
      VisitExpr(op->condition_);
      for (const auto& ia : op->iter_args_) {
        if (ia->initValue_) VisitExpr(ia->initValue_);
      }
      VisitStmt(op->body_);
    }
  };

  /// Count references to `target` within a single expression tree.
  static size_t CountVarRefs(const ExprPtr& expr, const Var* target) {
    class Counter : public IRVisitor {
     public:
      const Var* target;
      size_t count = 0;
      void VisitExpr_(const VarPtr& op) override {
        if (op.get() == target) ++count;
      }
      void VisitExpr_(const IterArgPtr& op) override {
        if (op.get() == target) ++count;
      }
    } c;
    c.target = target;
    c.VisitExpr(expr);
    return c.count;
  }

  /// Record of a standalone InCore call site. Owns references to the call's
  /// orchestration-function context so that we can test each candidate merge
  /// against every call site of the same callee before recording it.
  struct StandaloneCallSite {
    const FunctionBodyIndex* body_index;
    AssignStmtPtr assign_stmt;
    CallPtr call;
  };

  /// Collects standalone InCore calls (those outside any iter-arg-carrying
  /// loop) in an orchestration function body, keyed by callee name.
  class StandaloneCallCollector : public IRVisitor {
   public:
    StandaloneCallCollector(const std::unordered_set<std::string>& incore_names,
                            const FunctionBodyIndex& body_index,
                            std::unordered_map<std::string, std::vector<StandaloneCallSite>>& out)
        : incore_names_(incore_names), body_index_(body_index), out_(out) {}

   protected:
    void VisitStmt_(const ForStmtPtr& op) override {
      bool prev = inside_iter_loop_;
      if (!op->iter_args_.empty()) inside_iter_loop_ = true;
      IRVisitor::VisitStmt_(op);
      inside_iter_loop_ = prev;
    }
    void VisitStmt_(const WhileStmtPtr& op) override {
      bool prev = inside_iter_loop_;
      if (!op->iter_args_.empty()) inside_iter_loop_ = true;
      IRVisitor::VisitStmt_(op);
      inside_iter_loop_ = prev;
    }
    void VisitStmt_(const AssignStmtPtr& op) override {
      if (!inside_iter_loop_) {
        if (auto call = As<Call>(op->value_)) {
          auto fname = GetCallFuncName(call);
          if (!fname.empty() && incore_names_.count(fname)) {
            out_[fname].push_back({&body_index_, op, call});
          }
        }
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    const std::unordered_set<std::string>& incore_names_;
    const FunctionBodyIndex& body_index_;
    std::unordered_map<std::string, std::vector<StandaloneCallSite>>& out_;
    bool inside_iter_loop_ = false;
  };

  /// Check whether a (in_idx, out_idx) pairing is safe to apply at `site`:
  /// the Out arg is a locally-allocated `tensor.create`, and the In arg's
  /// sole use in the enclosing orch function is this call.
  static bool IsPairingSafeAtCallSite(const StandaloneCallSite& site, size_t in_idx, size_t out_idx) {
    const auto& call = site.call;
    if (out_idx >= call->args_.size() || in_idx >= call->args_.size()) return false;
    auto out_var = As<Var>(call->args_[out_idx]);
    auto in_var = As<Var>(call->args_[in_idx]);
    if (!out_var || !in_var) return false;
    if (!site.body_index->local_creates.count(out_var.get())) return false;

    auto use_it = site.body_index->use_count.find(in_var.get());
    size_t total_refs = use_it == site.body_index->use_count.end() ? 0 : use_it->second;
    size_t self_refs = CountVarRefs(site.assign_stmt->value_, in_var.get());
    return total_refs == self_refs;
  }

  /// Analyze orchestration functions for iter-arg reuse opportunities.
  std::unordered_map<std::string, AnalysisResult> Analyze(
      const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    std::unordered_map<std::string, std::vector<OutParamReturnMapping>> out_mappings;
    std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>> in_out_pairings;
    for (const auto& [gvar, func] : program->functions_) {
      if (!incore_names.count(func->name_)) continue;
      out_mappings[func->name_] = BuildOutParamReturnMappings(func);
      in_out_pairings[func->name_] = BuildInOutParamPairings(func);
    }

    LoopAnalyzer analyzer(program, incore_names, out_mappings);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    auto results = analyzer.results();

    // Collect all standalone call sites (preserving body_index references per
    // orchestration function).
    std::vector<FunctionBodyIndex> body_indices;
    body_indices.reserve(program->functions_.size());
    std::unordered_map<std::string, std::vector<StandaloneCallSite>> standalone_sites;
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      body_indices.emplace_back();
      auto& body_index = body_indices.back();
      body_index.VisitStmt(func->body_);
      StandaloneCallCollector collector(incore_names, body_index, standalone_sites);
      collector.VisitStmt(func->body_);
    }

    // For each callee with standalone call sites, only record a merge if
    // EVERY standalone call site satisfies the pairing's safety preconditions.
    // Caller-dependent safety cannot be cached per-callee otherwise: the
    // rewrite applies globally to every call of that function.
    for (const auto& [fname, sites] : standalone_sites) {
      if (results.count(fname)) continue;  // LoopAnalyzer already handled
      auto pair_it = in_out_pairings.find(fname);
      if (pair_it == in_out_pairings.end() || pair_it->second.empty()) continue;

      AnalysisResult partial;
      partial.func_name = fname;
      std::unordered_set<size_t> seen_out;
      std::unordered_set<size_t> seen_in;

      for (const auto& [in_idx, out_idx] : pair_it->second) {
        bool all_safe = true;
        for (const auto& site : sites) {
          if (!IsPairingSafeAtCallSite(site, in_idx, out_idx)) {
            all_safe = false;
            break;
          }
        }
        if (!all_safe) continue;
        if (!seen_out.insert(out_idx).second) continue;
        if (!seen_in.insert(in_idx).second) continue;
        partial.merges.push_back({out_idx, in_idx});
      }

      if (!partial.merges.empty()) results[fname] = std::move(partial);
    }
    return results;
  }

  // -- Pre-scan: IRVisitor that identifies dead tensor.create vars -----------

  class DeadCreateScanner : public IRVisitor {
   public:
    explicit DeadCreateScanner(const std::unordered_map<std::string, AnalysisResult>& reuse_results)
        : reuse_results_(reuse_results) {}

    const std::unordered_set<const Var*>& dead_creates() const { return dead_creates_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return;
      auto fname = GetCallFuncName(call);
      auto reuse_it = reuse_results_.find(fname);
      if (reuse_it == reuse_results_.end()) return;
      for (const auto& merge : reuse_it->second.merges) {
        if (merge.out_param_index < call->args_.size()) {
          if (auto create_var = As<Var>(call->args_[merge.out_param_index])) {
            dead_creates_.insert(create_var.get());
          }
        }
      }
    }

   private:
    const std::unordered_map<std::string, AnalysisResult>& reuse_results_;
    std::unordered_set<const Var*> dead_creates_;
  };

  // -- Mutation: IRMutator that substitutes Var references --------------------

  class VarSubstitutionMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) { subs_[old_ptr] = new_var; }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      return op;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
  };

  /// Rewrite an InCore function to merge Out params into In params.
  FunctionPtr RewriteIncore(const FunctionPtr& func, const std::vector<OutToInMerge>& merges) {
    std::unordered_set<size_t> out_indices_to_remove;
    VarSubstitutionMutator mutator;
    std::unordered_set<size_t> in_indices_to_promote;

    for (const auto& merge : merges) {
      out_indices_to_remove.insert(merge.out_param_index);
      in_indices_to_promote.insert(merge.in_param_index);
      mutator.AddSubstitution(func->params_[merge.out_param_index].get(),
                              func->params_[merge.in_param_index]);
    }

    std::vector<VarPtr> new_params;
    std::vector<ParamDirection> new_directions;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (out_indices_to_remove.count(i)) continue;
      new_params.push_back(func->params_[i]);
      if (in_indices_to_promote.count(i)) {
        new_directions.push_back(ParamDirection::InOut);
      } else {
        new_directions.push_back(i < func->param_directions_.size() ? func->param_directions_[i]
                                                                    : ParamDirection::In);
      }
    }

    auto new_body = mutator.VisitStmt(func->body_);

    return std::make_shared<Function>(func->name_, new_params, new_directions, func->return_types_, new_body,
                                      func->span_, func->func_type_, func->level_, func->role_, func->attrs_);
  }

  // -- Mutation: IRMutator that rewrites orch call sites ---------------------

  class CallSiteRewriter : public IRMutator {
   public:
    CallSiteRewriter(const std::unordered_map<std::string, AnalysisResult>& reuse_results,
                     const std::unordered_map<std::string, FunctionPtr>& rewritten_funcs,
                     const std::unordered_set<const Var*>& dead_creates)
        : reuse_results_(reuse_results), rewritten_funcs_(rewritten_funcs), dead_creates_(dead_creates) {}

   protected:
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return IRMutator::VisitStmt_(op);

      // Remove dead tensor.create
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.create") {
        if (dead_creates_.count(op->var_.get())) {
          return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
        }
        return IRMutator::VisitStmt_(op);
      }

      // Rewrite calls to rewritten InCore functions
      auto fname = GetCallFuncName(call);
      auto reuse_it = reuse_results_.find(fname);
      if (reuse_it == reuse_results_.end()) return IRMutator::VisitStmt_(op);

      auto func_it = rewritten_funcs_.find(fname);
      if (func_it == rewritten_funcs_.end()) return IRMutator::VisitStmt_(op);

      const auto& merges = reuse_it->second.merges;
      const auto& new_func = func_it->second;

      std::unordered_set<size_t> remove_indices;
      for (const auto& merge : merges) {
        remove_indices.insert(merge.out_param_index);
      }

      std::vector<ExprPtr> new_args;
      for (size_t i = 0; i < call->args_.size(); ++i) {
        if (remove_indices.count(i)) continue;
        new_args.push_back(VisitExpr(call->args_[i]));
      }

      TypePtr new_return_type;
      if (new_func->return_types_.empty()) {
        new_return_type = nullptr;
      } else if (new_func->return_types_.size() == 1) {
        new_return_type = new_func->return_types_[0];
      } else {
        new_return_type = std::make_shared<TupleType>(new_func->return_types_);
      }

      // Preserve attrs_ (e.g. kAttrDumpVars) through the arg-subset rewrite —
      // mirrors the base IRMutator. Only a merged Out param is removed; any
      // dump/dep Var the tag references is a surviving In arg, so a verbatim
      // attr copy stays valid. Fall back to UnknownType for a void-return callee
      // so the rewritten Call's type_ matches the prior 4-arg ctor path.
      auto new_call =
          std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_,
                                 new_return_type ? new_return_type : GetUnknownType(), call->span_);

      auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_return_type, op->var_->span_);
      var_remap_[op->var_.get()] = new_var;
      auto result = MutableCopy(op);
      result->var_ = new_var;
      result->value_ = new_call;
      return result;
    }

    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = var_remap_.find(op.get());
      if (it != var_remap_.end()) return it->second;
      return op;
    }

   private:
    const std::unordered_map<std::string, AnalysisResult>& reuse_results_;
    const std::unordered_map<std::string, FunctionPtr>& rewritten_funcs_;
    std::unordered_set<const Var*> dead_creates_;
    std::unordered_map<const Var*, VarPtr> var_remap_;
  };
};

// ============================================================================
// Pattern 2: AssembleParentStridesOptimizer
//
// Cross-function analysis: scans orchestration for
//   tensor.assemble(parent, incore_result, offset)
// where incore_result comes from an InCore call. Records the parent
// tensor's shape. Then updates the InCore function's Out param
// TensorType to carry parent-derived strides via TensorView.
// ============================================================================

class AssembleParentStridesOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto parent_shapes = Analyze(program, incore_names);
    if (parent_shapes.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
    }

    Apply(new_functions, incore_names, parent_shapes);

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  using ParentShapeMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<ExprPtr>>>;

  // -- Analysis: IRVisitor that tracks InCore call results and finds assemble patterns --

  class AssembleAnalyzer : public IRVisitor {
   public:
    AssembleAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names)
        : program_(program), incore_names_(incore_names) {}

    const ParentShapeMap& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) {
        // Check for TupleGetItem extracting from an InCore call result
        auto tgi = As<TupleGetItemExpr>(op->value_);
        if (tgi) {
          auto src_var = AsVarLike(tgi->tuple_);
          if (src_var) {
            auto it = var_to_incore_return_.find(src_var.get());
            if (it != var_to_incore_return_.end()) {
              var_to_incore_return_[op->var_.get()] = {it->second.func_name,
                                                       static_cast<size_t>(tgi->index_)};
            }
          }
        }
        return;
      }

      // Check if this is an InCore call
      auto fname = GetCallFuncName(call);
      if (incore_names_.count(fname)) {
        auto incore_func = FindFunction(program_, fname);
        if (incore_func && incore_func->return_types_.size() == 1) {
          var_to_incore_return_[op->var_.get()] = {fname, 0};
        } else if (incore_func && incore_func->return_types_.size() > 1) {
          var_to_incore_return_[op->var_.get()] = {fname, SIZE_MAX};
        }
        return;
      }

      // Check if this is a tensor.assemble(parent, source, offset)
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.assemble" &&
          call->args_.size() == 3) {
        auto parent_var = AsVarLike(call->args_[0]);
        auto source_var = AsVarLike(call->args_[1]);
        if (!parent_var || !source_var) return;

        auto src_it = var_to_incore_return_.find(source_var.get());
        if (src_it == var_to_incore_return_.end()) return;
        if (src_it->second.return_index == SIZE_MAX) return;

        auto parent_tensor_type = As<TensorType>(parent_var->GetType());
        if (!parent_tensor_type) return;

        result_[src_it->second.func_name][src_it->second.return_index] = parent_tensor_type->shape_;
      }
    }

   private:
    struct IncoreReturnInfo {
      std::string func_name;
      size_t return_index;
    };

    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    std::unordered_map<const Var*, IncoreReturnInfo> var_to_incore_return_;
    ParentShapeMap result_;
  };

  /// Analyze orchestration functions for tensor.assemble patterns.
  ParentShapeMap Analyze(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    AssembleAnalyzer analyzer(program, incore_names);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    return analyzer.result();
  }

  static std::vector<ExprPtr> ComputeStrides(const std::vector<ExprPtr>& shape) {
    return ComputeRowMajorStrides(shape);
  }

  // -- Mutation: IRMutator that propagates updated param types through tile.store --

  class ParamStrideUpdateMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) {
      subs_[old_ptr] = new_var;
      new_param_ptrs_.insert(new_var.get());
    }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      auto remap_it = var_remap_.find(op.get());
      if (remap_it != var_remap_.end()) return remap_it->second;
      return op;
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto visited = IRMutator::VisitStmt_(op);
      auto assign = As<AssignStmt>(visited);
      if (!assign) return visited;

      auto call = As<Call>(assign->value_);
      if (!call || call->op_->name_ != "tile.store" || call->args_.size() < 3) return assign;

      auto out_var = AsVarLike(call->args_[2]);
      if (!out_var || !new_param_ptrs_.count(out_var.get())) return assign;

      auto out_type = out_var->GetType();
      auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, out_type, call->span_);
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, out_type, assign->var_->span_);
      var_remap_[op->var_.get()] = new_var;
      auto result = MutableCopy(assign);
      result->var_ = new_var;
      result->value_ = new_call;
      return result;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
    std::unordered_set<const Var*> new_param_ptrs_;
    std::unordered_map<const Var*, VarPtr> var_remap_;
  };

  /// Apply assemble parent strides to InCore functions.
  void Apply(std::vector<FunctionPtr>& functions, const std::unordered_set<std::string>& incore_names,
             const ParentShapeMap& parent_shapes) {
    for (auto& func : functions) {
      if (!incore_names.count(func->name_)) continue;

      auto ps_it = parent_shapes.find(func->name_);
      if (ps_it == parent_shapes.end()) continue;
      const auto& return_idx_to_shape = ps_it->second;

      auto out_mappings = BuildOutParamReturnMappings(func);
      if (out_mappings.empty()) continue;

      bool changed = false;
      std::vector<VarPtr> new_params = func->params_;

      for (const auto& opm : out_mappings) {
        auto shape_it = return_idx_to_shape.find(opm.return_index);
        if (shape_it == return_idx_to_shape.end()) continue;

        auto full_strides = ComputeStrides(shape_it->second);
        if (full_strides.empty()) continue;

        auto tensor_type = As<TensorType>(func->params_[opm.param_index]->GetType());
        if (!tensor_type) continue;

        // Extract trailing strides matching the output tensor's rank.
        // For a 3D parent [B, M, N] with strides [M*N, N, 1] and a 2D output [M', N'],
        // we need the last 2 strides: [N, 1].
        size_t out_rank = tensor_type->shape_.size();
        if (out_rank > full_strides.size()) continue;
        std::vector<ExprPtr> strides(full_strides.end() - static_cast<std::ptrdiff_t>(out_rank),
                                     full_strides.end());

        TensorView view(std::move(strides), TensorLayout::ND);
        auto new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_,
                                                     tensor_type->memref_, std::move(view));
        auto new_param = std::make_shared<Var>(func->params_[opm.param_index]->name_hint_, new_type,
                                               func->params_[opm.param_index]->span_);

        changed = true;
        new_params[opm.param_index] = new_param;
      }

      if (!changed) continue;

      ParamStrideUpdateMutator mutator;
      for (size_t i = 0; i < func->params_.size(); ++i) {
        if (new_params[i].get() != func->params_[i].get()) {
          mutator.AddSubstitution(func->params_[i].get(), new_params[i]);
        }
      }
      auto new_body = mutator.VisitStmt(func->body_);

      func = std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                        new_body, func->span_, func->func_type_, func->level_, func->role_,
                                        func->attrs_);
    }
  }
};

// ============================================================================
// Pattern 3: AssembleLoopRewriter
//
// InCore-local optimization: when an InCore function body has a ForStmt
// that does tile.assemble accumulation yielding back to iter-arg, and the
// ForStmt result feeds the final tile.store -> return, rewrite the loop
// to use tile.store instead of tile.assemble, with the iter-arg initialized
// from the Out param.
// ============================================================================

class AssembleLoopRewriter {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    std::vector<FunctionPtr> new_functions;
    bool changed = false;

    for (const auto& [gvar, func] : program->functions_) {
      if (!incore_names.count(func->name_)) {
        new_functions.push_back(func);
        continue;
      }
      bool has_out = false;
      for (const auto& dir : func->param_directions_) {
        if (dir == ParamDirection::Out) {
          has_out = true;
          break;
        }
      }
      if (!has_out) {
        new_functions.push_back(func);
        continue;
      }
      auto rewritten = RewriteFunction(func);
      if (rewritten.get() != func.get()) changed = true;
      new_functions.push_back(rewritten);
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  /// Check if a statement subtree uses a given Var (by raw pointer).
  /// Uses VarDefUseCollector which handles all statement/expression types.
  static bool StmtUsesVar(const StmtPtr& stmt, const Var* var) {
    if (!stmt || !var) return false;
    var_collectors::VarDefUseCollector collector;
    collector.VisitStmt(stmt);
    return collector.var_uses.count(var) > 0;
  }

  // -- Pre-scan: IRVisitor that collects var definitions and return statement --

  class BodyScanner : public IRVisitor {
   public:
    const std::unordered_map<const Var*, AssignStmtPtr>& var_def() const { return var_def_; }
    const ReturnStmtPtr& return_stmt() const { return return_stmt_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override { var_def_[op->var_.get()] = op; }
    void VisitStmt_(const ReturnStmtPtr& op) override { return_stmt_ = op; }

   private:
    std::unordered_map<const Var*, AssignStmtPtr> var_def_;
    ReturnStmtPtr return_stmt_;
  };

  // -- Mutation: IRMutator that rewrites matching ForStmt patterns -----------

  class LoopRewriteMutator : public IRMutator {
   public:
    struct StoreReturnInfo {
      const Var* store_var;
      const Var* out_param;
      size_t return_index;
    };

    LoopRewriteMutator(const std::unordered_map<const Var*, size_t>& out_var_to_param_idx,
                       const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store,
                       const std::unordered_set<const Var*>& dead_create_vars,
                       const std::unordered_set<const Var*>& dead_store_vars,
                       const std::unordered_map<const Var*, VarPtr>& return_var_remap,
                       const FunctionPtr& func)
        : out_var_to_param_idx_(out_var_to_param_idx),
          for_return_to_store_(for_return_to_store),
          dead_create_vars_(dead_create_vars),
          dead_store_vars_(dead_store_vars),
          return_var_remap_(return_var_remap),
          func_(func) {}

   protected:
    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      if (op->iter_args_.size() != 1 || op->return_vars_.size() != 1) {
        return IRMutator::VisitStmt_(op);
      }

      auto fret_it = for_return_to_store_.find(op->return_vars_[0].get());
      if (fret_it == for_return_to_store_.end()) {
        return IRMutator::VisitStmt_(op);
      }
      const auto& store_info = *fret_it->second;

      auto loop_body_stmts = FlattenToStmts(op->body_);
      auto yield = transform_utils::FindYieldStmt(op->body_);
      if (!yield || yield->value_.size() != 1) {
        return IRMutator::VisitStmt_(op);
      }

      const IterArg* iter_arg = op->iter_args_[0].get();

      // Find the tile.assemble call
      AssignStmtPtr assemble_assign;
      for (const auto& body_stmt : loop_body_stmts) {
        auto assign = As<AssignStmt>(body_stmt);
        if (!assign) continue;
        auto call = As<Call>(assign->value_);
        if (!call || call->op_->name_ != "tile.assemble") continue;
        if (call->args_.size() < 3) continue;
        const Var* arg0_raw = nullptr;
        if (auto v = As<Var>(call->args_[0])) arg0_raw = v.get();
        if (auto ia = As<IterArg>(call->args_[0])) arg0_raw = ia.get();
        if (arg0_raw != iter_arg) continue;
        assemble_assign = assign;
        break;
      }

      if (!assemble_assign) return IRMutator::VisitStmt_(op);

      // --- Rewrite: tile.assemble -> tile.store ---

      auto out_param_var = func_->params_[out_var_to_param_idx_.at(store_info.out_param)];
      auto out_tensor_type = As<TensorType>(out_param_var->GetType());
      INTERNAL_CHECK_SPAN(out_tensor_type, out_param_var->span_)
          << "Internal error: Out param should be TensorType";

      auto new_iter_arg = std::make_shared<IterArg>(op->iter_args_[0]->name_hint_, out_tensor_type,
                                                    out_param_var, op->iter_args_[0]->span_);

      auto assemble_call = As<Call>(assemble_assign->value_);
      auto& op_registry = OpRegistry::GetInstance();
      auto store_call =
          op_registry.Create("tile.store", {assemble_call->args_[1], assemble_call->args_[2], new_iter_arg},
                             assemble_assign->value_->span_);
      auto store_result_var = std::make_shared<Var>(assemble_assign->var_->name_hint_, store_call->GetType(),
                                                    assemble_assign->span_);

      std::vector<StmtPtr> new_loop_stmts;
      for (const auto& body_stmt : loop_body_stmts) {
        if (body_stmt.get() == assemble_assign.get()) {
          auto store_assign = MutableCopy(assemble_assign);
          store_assign->var_ = store_result_var;
          store_assign->value_ = store_call;
          new_loop_stmts.push_back(std::move(store_assign));
        } else if (auto y = As<YieldStmt>(body_stmt)) {
          auto new_yield = MutableCopy(y);
          new_yield->value_ = std::vector<ExprPtr>{store_result_var};
          new_loop_stmts.push_back(std::move(new_yield));
        } else {
          new_loop_stmts.push_back(body_stmt);
        }
      }

      auto new_loop_body = SeqStmts::Flatten(std::move(new_loop_stmts), op->body_->span_);
      auto new_return_var = return_var_remap_.at(store_info.store_var);

      auto result = MutableCopy(op);
      result->iter_args_ = std::vector<IterArgPtr>{new_iter_arg};
      result->body_ = new_loop_body;
      result->return_vars_ = std::vector<VarPtr>{new_return_var};
      return result;
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      if (dead_create_vars_.count(op->var_.get()) || dead_store_vars_.count(op->var_.get())) {
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
      }
      return IRMutator::VisitStmt_(op);
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      auto visited = IRMutator::VisitStmt_(op);
      auto seq = As<SeqStmts>(visited);
      if (!seq) return visited;
      // Filter out empty SeqStmts children (from deleted statements)
      std::vector<StmtPtr> filtered;
      for (const auto& s : seq->stmts_) {
        if (auto child_seq = As<SeqStmts>(s)) {
          if (child_seq->stmts_.empty()) continue;
        }
        filtered.push_back(s);
      }
      if (filtered.size() == seq->stmts_.size()) return seq;
      return SeqStmts::Flatten(std::move(filtered), seq->span_);
    }

    StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
      if (return_var_remap_.empty()) return op;
      std::vector<ExprPtr> new_ret_values;
      bool remapped = false;
      for (const auto& v : op->value_) {
        auto var = As<Var>(v);
        if (var) {
          auto remap_it = return_var_remap_.find(var.get());
          if (remap_it != return_var_remap_.end()) {
            new_ret_values.push_back(remap_it->second);
            remapped = true;
            continue;
          }
        }
        new_ret_values.push_back(v);
      }
      if (!remapped) return op;
      auto result = MutableCopy(op);
      result->value_ = std::move(new_ret_values);
      return result;
    }

   private:
    const std::unordered_map<const Var*, size_t>& out_var_to_param_idx_;
    const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store_;
    const std::unordered_set<const Var*>& dead_create_vars_;
    const std::unordered_set<const Var*>& dead_store_vars_;
    const std::unordered_map<const Var*, VarPtr>& return_var_remap_;
    const FunctionPtr& func_;
  };

  /// Rewrite assemble-loop pattern in an InCore function.
  FunctionPtr RewriteFunction(const FunctionPtr& func) {
    // Pre-scan: collect var definitions and return statement
    BodyScanner scanner;
    scanner.VisitStmt(func->body_);
    const auto& var_def = scanner.var_def();
    const auto& return_stmt = scanner.return_stmt();
    if (!return_stmt) return func;

    // Identify Out params
    std::unordered_map<const Var*, size_t> out_var_to_param_idx;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (i < func->param_directions_.size() && func->param_directions_[i] == ParamDirection::Out) {
        out_var_to_param_idx[func->params_[i].get()] = i;
      }
    }
    if (out_var_to_param_idx.empty()) return func;

    // Map return values -> tile.store -> Out params
    using StoreReturnInfo = LoopRewriteMutator::StoreReturnInfo;
    std::vector<StoreReturnInfo> store_returns;

    for (size_t ret_i = 0; ret_i < return_stmt->value_.size(); ++ret_i) {
      auto ret_var = As<Var>(return_stmt->value_[ret_i]);
      if (!ret_var) continue;
      auto def_it = var_def.find(ret_var.get());
      if (def_it == var_def.end()) continue;
      auto call = As<Call>(def_it->second->value_);
      if (!call || call->op_->name_ != "tile.store") continue;
      if (call->args_.size() < 3) continue;
      auto out_tensor = As<Var>(call->args_[2]);
      if (!out_tensor || !out_var_to_param_idx.count(out_tensor.get())) continue;
      store_returns.push_back({ret_var.get(), out_tensor.get(), ret_i});
    }
    if (store_returns.empty()) return func;

    // Map ForStmt return_var -> which store_return it feeds
    std::unordered_map<const Var*, const StoreReturnInfo*> for_return_to_store;
    for (const auto& sr : store_returns) {
      auto def_it = var_def.find(sr.store_var);
      if (def_it == var_def.end()) continue;
      auto call = As<Call>(def_it->second->value_);
      if (!call || call->op_->name_ != "tile.store") continue;
      auto tile_data_var = As<Var>(call->args_[0]);
      if (!tile_data_var) continue;
      for_return_to_store[tile_data_var.get()] = &sr;
    }

    // Pre-compute dead sets by scanning ForStmts for pattern matches.
    // This must happen before the IRMutator pass because dead tile.create
    // statements may appear before the ForStmt they correspond to.
    std::unordered_set<const Var*> dead_create_vars;
    std::unordered_set<const Var*> dead_store_vars;
    std::unordered_map<const Var*, VarPtr> return_var_remap;

    class ForStmtMatchScanner : public IRVisitor {
     public:
      ForStmtMatchScanner(const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store,
                          const std::unordered_map<const Var*, size_t>& out_var_to_param_idx,
                          const FunctionPtr& func, std::unordered_set<const Var*>& dead_create_vars,
                          std::unordered_set<const Var*>& dead_store_vars,
                          std::unordered_map<const Var*, VarPtr>& return_var_remap)
          : for_return_to_store_(for_return_to_store),
            out_var_to_param_idx_(out_var_to_param_idx),
            func_(func),
            dead_create_vars_(dead_create_vars),
            dead_store_vars_(dead_store_vars),
            return_var_remap_(return_var_remap) {}

      [[nodiscard]] bool matched() const { return matched_; }

     protected:
      void VisitStmt_(const ForStmtPtr& op) override {
        IRVisitor::VisitStmt_(op);
        if (op->iter_args_.size() != 1 || op->return_vars_.size() != 1) return;

        auto fret_it = for_return_to_store_.find(op->return_vars_[0].get());
        if (fret_it == for_return_to_store_.end()) return;
        const auto& store_info = *fret_it->second;

        auto loop_body_stmts = FlattenToStmts(op->body_);
        auto yield = transform_utils::FindYieldStmt(op->body_);
        if (!yield || yield->value_.size() != 1) return;

        const IterArg* iter_arg = op->iter_args_[0].get();

        AssignStmtPtr assemble_assign;
        for (const auto& body_stmt : loop_body_stmts) {
          auto assign = As<AssignStmt>(body_stmt);
          if (!assign) continue;
          auto call = As<Call>(assign->value_);
          if (!call || call->op_->name_ != "tile.assemble") continue;
          if (call->args_.size() < 3) continue;
          const Var* arg0_raw = nullptr;
          if (auto v = As<Var>(call->args_[0])) arg0_raw = v.get();
          if (auto ia = As<IterArg>(call->args_[0])) arg0_raw = ia.get();
          if (arg0_raw != iter_arg) continue;
          assemble_assign = assign;
          break;
        }
        if (!assemble_assign) return;

        auto yield_var = As<Var>(yield->value_[0]);
        if (!yield_var || yield_var.get() != assemble_assign->var_.get()) return;

        bool iter_arg_used_elsewhere = false;
        for (const auto& body_stmt : loop_body_stmts) {
          if (body_stmt.get() == assemble_assign.get()) continue;
          if (As<YieldStmt>(body_stmt)) continue;
          if (StmtUsesVar(body_stmt, iter_arg)) {
            iter_arg_used_elsewhere = true;
            break;
          }
        }
        if (iter_arg_used_elsewhere) return;

        // Pattern matched — record dead sets
        matched_ = true;
        auto init_var = As<Var>(op->iter_args_[0]->initValue_);
        if (init_var) dead_create_vars_.insert(init_var.get());
        dead_store_vars_.insert(store_info.store_var);

        auto out_param_var = func_->params_[out_var_to_param_idx_.at(store_info.out_param)];
        auto out_tensor_type = As<TensorType>(out_param_var->GetType());
        INTERNAL_CHECK_SPAN(out_tensor_type, out_param_var->span_)
            << "Internal error: Out param should be TensorType";
        auto new_return_var = std::make_shared<Var>(op->return_vars_[0]->name_hint_, out_tensor_type,
                                                    op->return_vars_[0]->span_);
        return_var_remap_[store_info.store_var] = new_return_var;
      }

     private:
      const std::unordered_map<const Var*, const StoreReturnInfo*>& for_return_to_store_;
      const std::unordered_map<const Var*, size_t>& out_var_to_param_idx_;
      const FunctionPtr& func_;
      std::unordered_set<const Var*>& dead_create_vars_;
      std::unordered_set<const Var*>& dead_store_vars_;
      std::unordered_map<const Var*, VarPtr>& return_var_remap_;
      bool matched_ = false;
    };

    ForStmtMatchScanner match_scanner(for_return_to_store, out_var_to_param_idx, func, dead_create_vars,
                                      dead_store_vars, return_var_remap);
    match_scanner.VisitStmt(func->body_);
    if (!match_scanner.matched()) return func;

    // Apply the IRMutator using pre-computed dead sets
    LoopRewriteMutator mutator(out_var_to_param_idx, for_return_to_store, dead_create_vars, dead_store_vars,
                               return_var_remap, func);
    auto new_body = mutator.VisitStmt(func->body_);

    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  }
};

// ============================================================================
// Pattern 4: SliceInputStridesOptimizer
//
// Cross-function analysis: scans orchestration for
//   tensor.slice(parent, size, offset)
// where the slice result is passed as an In argument to an InCore call.
// Records the parent tensor's shape. Then updates the InCore function's
// In param TensorType to carry parent-derived strides via TensorView.
// ============================================================================

class SliceInputStridesOptimizer {
 public:
  ProgramPtr Run(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    auto input_shapes = Analyze(program, incore_names);
    if (input_shapes.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
    }

    Apply(new_functions, incore_names, input_shapes);

    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  // func_name -> { param_index -> parent_shape }
  using InputShapeMap = std::unordered_map<std::string, std::unordered_map<size_t, std::vector<ExprPtr>>>;

  static bool ShapesMatch(const std::vector<ExprPtr>& a, const std::vector<ExprPtr>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      auto ca = As<ConstInt>(a[i]);
      auto cb = As<ConstInt>(b[i]);
      if (!ca || !cb || ca->value_ != cb->value_) return false;
    }
    return true;
  }

  class SliceAnalyzer : public IRVisitor {
   public:
    SliceAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names)
        : program_(program), incore_names_(incore_names) {}

    const InputShapeMap& result() const { return result_; }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      auto call = As<Call>(op->value_);
      if (!call) return;

      // Track tensor.slice(parent, size, offset) results
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && call->op_->name_ == "tensor.slice" &&
          call->args_.size() >= 3) {
        auto parent_var = AsVarLike(call->args_[0]);
        if (parent_var) {
          auto parent_tensor_type = As<TensorType>(parent_var->GetType());
          if (parent_tensor_type) {
            var_to_parent_shape_[op->var_.get()] = parent_tensor_type->shape_;
          }
        }
        return;
      }

      // Check InCore calls: map sliced In arguments to parent shapes
      auto fname = GetCallFuncName(call);
      if (!incore_names_.count(fname)) return;

      auto incore_func = FindFunction(program_, fname);
      if (!incore_func) return;

      for (size_t i = 0; i < call->args_.size() && i < incore_func->param_directions_.size(); ++i) {
        if (incore_func->param_directions_[i] != ParamDirection::In) continue;

        auto arg_var = AsVarLike(call->args_[i]);
        if (!arg_var) continue;

        auto it = var_to_parent_shape_.find(arg_var.get());
        if (it == var_to_parent_shape_.end()) continue;

        auto conflict_key = fname + ":" + std::to_string(i);
        if (conflicted_.count(conflict_key)) continue;

        auto& entry = result_[fname][i];
        if (entry.empty()) {
          entry = it->second;
        } else if (!ShapesMatch(entry, it->second)) {
          conflicted_.insert(conflict_key);
          entry.clear();
        }
      }
    }

   private:
    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    std::unordered_map<const Var*, std::vector<ExprPtr>> var_to_parent_shape_;
    std::unordered_set<std::string> conflicted_;
    InputShapeMap result_;
  };

  InputShapeMap Analyze(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    SliceAnalyzer analyzer(program, incore_names);
    for (const auto& [gvar, func] : program->functions_) {
      if (incore_names.count(func->name_)) continue;
      analyzer.VisitStmt(func->body_);
    }
    return analyzer.result();
  }

  class InParamSubstitutionMutator : public IRMutator {
   public:
    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) { subs_[old_ptr] = new_var; }

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      return op;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
  };

  void Apply(std::vector<FunctionPtr>& functions, const std::unordered_set<std::string>& incore_names,
             const InputShapeMap& input_shapes) {
    for (auto& func : functions) {
      if (!incore_names.count(func->name_)) continue;

      auto is_it = input_shapes.find(func->name_);
      if (is_it == input_shapes.end()) continue;
      const auto& param_idx_to_shape = is_it->second;

      bool changed = false;
      std::vector<VarPtr> new_params = func->params_;

      for (const auto& [param_idx, parent_shape] : param_idx_to_shape) {
        if (parent_shape.empty()) continue;  // conflicted or not from slice
        if (param_idx >= func->params_.size()) continue;

        auto full_strides = ComputeRowMajorStrides(parent_shape);
        if (full_strides.empty()) continue;

        auto tensor_type = As<TensorType>(func->params_[param_idx]->GetType());
        if (!tensor_type) continue;

        // Skip params that already have explicit strides
        if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty()) continue;

        size_t in_rank = tensor_type->shape_.size();
        if (in_rank > full_strides.size()) continue;
        std::vector<ExprPtr> strides(full_strides.end() - static_cast<std::ptrdiff_t>(in_rank),
                                     full_strides.end());

        TensorView view(std::move(strides), TensorLayout::ND);
        auto new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_,
                                                     tensor_type->memref_, std::move(view));
        auto new_param = std::make_shared<Var>(func->params_[param_idx]->name_hint_, new_type,
                                               func->params_[param_idx]->span_);

        changed = true;
        new_params[param_idx] = new_param;
      }

      if (!changed) continue;

      InParamSubstitutionMutator mutator;
      for (size_t i = 0; i < func->params_.size(); ++i) {
        if (new_params[i].get() != func->params_[i].get()) {
          mutator.AddSubstitution(func->params_[i].get(), new_params[i]);
        }
      }
      auto new_body = mutator.VisitStmt(func->body_);

      func = std::make_shared<Function>(func->name_, new_params, func->param_directions_, func->return_types_,
                                        new_body, func->span_, func->func_type_, func->level_, func->role_,
                                        func->attrs_);
    }
  }
};

// ============================================================================
// Pattern 5: Static Out-window externalization
//
// Rewrites statically provable local-window writes into explicit
// slice -> windowed callee -> assemble structure at the orchestration callsite.
//
// Supported shapes:
// - FinalStore: single call writes one final local window of an Out param
// - AggregateWindowLoop: an outlined non-builtin callee writes a loop-carried
//   aggregate window into one or more Out params.
//
// Multi-Out policy is per-output and conservative: each Out param is rewritten
// only when its own read/write footprint can be proven as one or more dense
// pieces representable with the existing tensor.slice/tensor.assemble runtime
// views. Unproven Out params stay as baseline full-tensor args/results.
// ============================================================================

class OutWindowExternalizer {
 public:
  enum class OutputWindowPolicy {
    ExactPieces,
    CoalescePieces,
  };

  enum class WindowRewritePolicy {
    Auto,
    All,
    InputsOnly,
    OutputsOnly,
    NoMultiPieceOutputs,
    None,
  };

  explicit OutWindowExternalizer(OutputWindowPolicy output_window_policy = OutputWindowPolicy::CoalescePieces,
                                 WindowRewritePolicy window_rewrite_policy = WindowRewritePolicy::Auto)
      : output_window_policy_(output_window_policy), window_rewrite_policy_(window_rewrite_policy) {}

  ProgramPtr Run(const ProgramPtr& program) {
    auto analyses = Analyze(program);
    ApplyDebugNameFilters(program, &analyses);
    if (output_window_policy_ == OutputWindowPolicy::CoalescePieces) {
      for (auto& [_, analysis] : analyses) {
        CoalesceOutputPieces(&analysis);
      }
    }
    ApplyWindowRewritePolicy(program, &analyses);
    if (analyses.empty()) return program;

    auto function_lookup = BuildFunctionLookup(program);

    std::unordered_map<std::string, FunctionPtr> cloned_funcs;
    for (const auto& [func_name, analysis] : analyses) {
      auto callee_it = function_lookup.find(func_name);
      if (callee_it == function_lookup.end()) continue;
      auto callee = callee_it->second;
      auto cloned = RewriteCallee(program, callee, analysis);
      if (!cloned) {
        if (WindowExternalizeLogEnabled()) {
          LOG_INFO << "[window-clone] failed func=" << func_name << " outputs=" << analysis.outputs.size()
                   << " inputs=" << analysis.inputs.size()
                   << " dynamic_inputs=" << HasDynamicIndexedInputWindow(analysis.inputs);
        }
        continue;
      }
      if (WindowExternalizeLogEnabled()) {
        LOG_INFO << "[window-clone] created func=" << func_name << " outputs=" << analysis.outputs.size()
                 << " inputs=" << analysis.inputs.size()
                 << " dynamic_inputs=" << HasDynamicIndexedInputWindow(analysis.inputs);
      }
      cloned_funcs.emplace(func_name, cloned);
    }
    if (cloned_funcs.empty()) return program;

    std::unordered_map<const Function*, FunctionPtr> rewritten_orch_funcs;
    std::unordered_set<std::string> used_clone_names;
    bool changed = false;
    for (const auto& [_, func] : program->functions_) {
      if (!func || func->func_type_ != FunctionType::Orchestration) continue;
      OrchRewriter rewriter(program, analyses, cloned_funcs, function_lookup,
                            output_dynamic_extent_dims_by_func_, window_rewrite_policy_);
      auto new_body = rewriter.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;
      changed = true;
      for (const auto& clone_name : rewriter.used_clone_names()) used_clone_names.insert(clone_name);
      rewritten_orch_funcs.emplace(
          func.get(), std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                                 func->return_types_, new_body, func->span_, func->func_type_,
                                                 func->level_, func->role_, func->attrs_));
    }

    if (!changed) return program;
    std::vector<FunctionPtr> new_functions;
    new_functions.reserve(program->functions_.size() + used_clone_names.size());
    for (const auto& [_, func] : program->functions_) {
      auto rewritten_it = rewritten_orch_funcs.find(func.get());
      new_functions.push_back(rewritten_it == rewritten_orch_funcs.end() ? func : rewritten_it->second);
      auto clone_it = cloned_funcs.find(func->name_);
      if (clone_it != cloned_funcs.end() && used_clone_names.count(func->name_) != 0) {
        new_functions.push_back(clone_it->second);
      }
    }
    return std::make_shared<Program>(new_functions, program->name_, program->span_);
  }

 private:
  enum class RewriteKind {
    FinalStore,
    AggregateWindowLoop,
  };

  struct DenseRegionPiece {
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> callsite_offsets;
    std::vector<ExprPtr> local_offsets;
  };

  struct AccessRegion {
    // Internal proof result. Today every region lowers to one or more dense
    // tensor.slice views; unsupported access sets stay baseline.
    std::vector<DenseRegionPiece> dense_pieces;
  };

  struct DynamicIndexedInputWindow {
    size_t index_tensor_param_index = SIZE_MAX;
    VarPtr loop_var;
    ExprPtr loop_start;
    ExprPtr loop_stop;
    ExprPtr loop_step;
    ExprPtr guard_condition;
    std::vector<ExprPtr> index_offsets;
    const Var* dynamic_index_value = nullptr;
    size_t dynamic_dim = 0;
  };

  struct OutputRewriteInfo {
    size_t out_param_index;
    size_t return_index;
    std::vector<ExprPtr> parent_shape;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> callsite_offsets;
    std::vector<ExprPtr> local_store_offsets;
    AccessRegion region;
    AccessRegion assemble_region;
    std::vector<size_t> piece_return_indices;
    size_t iter_arg_index = SIZE_MAX;
  };

  struct InputRewriteInfo {
    size_t in_param_index;
    std::vector<ExprPtr> parent_shape;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> callsite_offsets;
    std::vector<ExprPtr> local_read_offsets;
    AccessRegion region;
    std::optional<DynamicIndexedInputWindow> dynamic_indexed_window = std::nullopt;
    VarPtr dynamic_base_param;
    VarPtr dynamic_extent_param;
  };

  struct CalleeRewriteAnalysis {
    RewriteKind kind = RewriteKind::FinalStore;
    std::vector<OutputRewriteInfo> outputs;
    std::vector<InputRewriteInfo> inputs;
  };

  struct WindowRewriteCost {
    int original_tensor_args = 0;
    int rewritten_tensor_args = 0;
    int abi_delta = 0;
    size_t dense_piece_count = 0;
    size_t assemble_piece_count = 0;
    bool is_output = false;
    bool is_multi_piece_output = false;
    bool has_callsite_assemble = false;
    std::string reason;
  };

  using AnalysisMap = std::unordered_map<std::string, CalleeRewriteAnalysis>;

  static std::vector<std::string> SplitDebugNameList(const char* value) {
    std::vector<std::string> result;
    if (!value) return result;
    std::string text(value);
    size_t start = 0;
    while (start <= text.size()) {
      size_t end = text.find(',', start);
      if (end == std::string::npos) end = text.size();
      auto token = text.substr(start, end - start);
      auto first = token.find_first_not_of(" \t\n\r");
      auto last = token.find_last_not_of(" \t\n\r");
      if (first != std::string::npos) result.push_back(token.substr(first, last - first + 1));
      if (end == text.size()) break;
      start = end + 1;
    }
    return result;
  }

  static bool DebugNameTokenMatches(const std::string& name, const std::string& token) {
    if (name.empty() || token.empty()) return false;
    if (token.size() >= 2 && token.front() == '*' && token.back() == '*') {
      return name.find(token.substr(1, token.size() - 2)) != std::string::npos;
    }
    return name == token || name == token + "__windowed";
  }

  static bool DebugNameMatches(const std::string& callee_name, const FunctionPtr& func, size_t param_index,
                               const std::vector<std::string>& tokens) {
    if (tokens.empty()) return false;
    std::string param_name;
    if (func && param_index < func->params_.size() && func->params_[param_index]) {
      param_name = func->params_[param_index]->name_hint_;
    }
    for (const auto& token : tokens) {
      if (DebugNameTokenMatches(callee_name, token)) return true;
      if (DebugNameTokenMatches(param_name, token)) return true;
    }
    return false;
  }

  static bool WindowExternalizeLogEnabled() {
    auto value = std::getenv("PYPTO_WINDOW_EXTERNALIZE_LOG");
    if (!value) return false;
    std::string text(value);
    return !text.empty() && text != "0" && text != "false" && text != "False";
  }

  static std::string DebugParamName(const std::string& callee_name, const FunctionPtr& func,
                                    size_t param_index) {
    if (func && param_index < func->params_.size() && func->params_[param_index]) {
      return callee_name + "." + func->params_[param_index]->name_hint_;
    }
    return callee_name + ".param" + std::to_string(param_index);
  }

  static void LogAutoDecision(bool accepted, const std::string& target, const WindowRewriteCost& cost) {
    if (!WindowExternalizeLogEnabled()) return;
    std::ostringstream os;
    os << "[window-auto] " << (accepted ? "accept " : "reject ") << target << ": abi_delta=" << cost.abi_delta
       << " pieces=" << cost.dense_piece_count << " assemble_pieces=" << cost.assemble_piece_count
       << " reason=" << cost.reason;
    LOG_INFO << os.str();
  }

  static std::string ExprVectorDebugString(const std::vector<ExprPtr>& exprs) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < exprs.size(); ++i) {
      if (i != 0) os << ", ";
      os << PythonPrint(exprs[i], "pl", /*concise=*/true);
    }
    os << "]";
    return os.str();
  }

  static void ApplyDebugNameFilters(const ProgramPtr& program, AnalysisMap* analyses) {
    if (!analyses) return;
    auto include = SplitDebugNameList(std::getenv("PYPTO_WINDOW_EXTERNALIZE_INCLUDE"));
    auto exclude = SplitDebugNameList(std::getenv("PYPTO_WINDOW_EXTERNALIZE_EXCLUDE"));
    if (include.empty() && exclude.empty()) return;
    auto function_lookup = BuildFunctionLookup(program);

    for (auto it = analyses->begin(); it != analyses->end();) {
      const auto& callee_name = it->first;
      auto func_it = function_lookup.find(callee_name);
      auto func = func_it == function_lookup.end() ? nullptr : func_it->second;
      auto& analysis = it->second;

      analysis.outputs.erase(
          std::remove_if(analysis.outputs.begin(), analysis.outputs.end(),
                         [&](const OutputRewriteInfo& output) {
                           bool matched =
                               DebugNameMatches(callee_name, func, output.out_param_index, include);
                           if (!include.empty() && !matched) return true;
                           return DebugNameMatches(callee_name, func, output.out_param_index, exclude);
                         }),
          analysis.outputs.end());
      analysis.inputs.erase(
          std::remove_if(analysis.inputs.begin(), analysis.inputs.end(),
                         [&](const InputRewriteInfo& input) {
                           bool matched = DebugNameMatches(callee_name, func, input.in_param_index, include);
                           if (!include.empty() && !matched) return true;
                           return DebugNameMatches(callee_name, func, input.in_param_index, exclude);
                         }),
          analysis.inputs.end());

      if (analysis.outputs.empty() && analysis.inputs.empty()) {
        it = analyses->erase(it);
      } else {
        ++it;
      }
    }
  }

  static DenseRegionPiece MakeDensePiece(std::vector<ExprPtr> window_shape,
                                         std::vector<ExprPtr> callsite_offsets,
                                         std::vector<ExprPtr> local_offsets) {
    return DenseRegionPiece{std::move(window_shape), std::move(callsite_offsets), std::move(local_offsets)};
  }

  static AccessRegion MakeDenseRegion(std::vector<DenseRegionPiece> pieces) {
    return AccessRegion{std::move(pieces)};
  }

  static bool DenseRectsAreDisjoint(const std::vector<ExprPtr>& lhs_offsets,
                                    const std::vector<ExprPtr>& lhs_shape,
                                    const std::vector<ExprPtr>& rhs_offsets,
                                    const std::vector<ExprPtr>& rhs_shape) {
    if (lhs_offsets.size() != rhs_offsets.size() || lhs_shape.size() != rhs_shape.size() ||
        lhs_offsets.size() != lhs_shape.size()) {
      return false;
    }
    arith::Analyzer analyzer;
    for (size_t dim = 0; dim < lhs_offsets.size(); ++dim) {
      auto lhs_dim = As<ConstInt>(lhs_shape[dim]);
      auto rhs_dim = As<ConstInt>(rhs_shape[dim]);
      if (!lhs_dim || !rhs_dim) return false;
      auto diff = analyzer.Simplify(MakeSub(rhs_offsets[dim], lhs_offsets[dim], rhs_offsets[dim]->span_));
      auto diff_ci = As<ConstInt>(diff);
      if (!diff_ci) continue;
      // Tensor windows are half-open intervals, so touching boundaries are disjoint.
      if (diff_ci->value_ >= lhs_dim->value_ || -diff_ci->value_ >= rhs_dim->value_) {
        return true;
      }
    }
    return false;
  }

  static const std::vector<DenseRegionPiece>& DensePieces(const OutputRewriteInfo& info) {
    return info.region.dense_pieces;
  }

  static const std::vector<DenseRegionPiece>& AssemblePieces(const OutputRewriteInfo& info) {
    return info.assemble_region.dense_pieces.empty() ? info.region.dense_pieces
                                                     : info.assemble_region.dense_pieces;
  }

  static bool HasFineAssemblePieces(const OutputRewriteInfo& info) {
    return !info.assemble_region.dense_pieces.empty();
  }

  static const std::vector<DenseRegionPiece>& DensePieces(const InputRewriteInfo& info) {
    return info.region.dense_pieces;
  }

  static bool HasMultiPieceOutput(const CalleeRewriteAnalysis& analysis) {
    return std::any_of(analysis.outputs.begin(), analysis.outputs.end(),
                       [](const OutputRewriteInfo& output) { return DensePieces(output).size() > 1; });
  }

  static bool HasCoalescedOutput(const CalleeRewriteAnalysis& analysis) {
    return std::any_of(analysis.outputs.begin(), analysis.outputs.end(), HasFineAssemblePieces);
  }

  static WindowRewriteCost EstimateOutputCost(const OutputRewriteInfo& output) {
    WindowRewriteCost cost;
    cost.is_output = true;
    cost.dense_piece_count = DensePieces(output).size();
    cost.assemble_piece_count = AssemblePieces(output).size();
    cost.is_multi_piece_output = cost.dense_piece_count > 1 || cost.assemble_piece_count > 1;
    cost.has_callsite_assemble = HasFineAssemblePieces(output);
    cost.original_tensor_args = 1;
    cost.rewritten_tensor_args = static_cast<int>(cost.dense_piece_count);
    cost.abi_delta = cost.rewritten_tensor_args - cost.original_tensor_args;
    return cost;
  }

  static WindowRewriteCost EstimateInputCost(const InputRewriteInfo& input) {
    WindowRewriteCost cost;
    cost.is_output = false;
    cost.dense_piece_count = DensePieces(input).size();
    cost.assemble_piece_count = cost.dense_piece_count;
    cost.original_tensor_args = 1;
    cost.rewritten_tensor_args = static_cast<int>(cost.dense_piece_count);
    cost.abi_delta = cost.rewritten_tensor_args - cost.original_tensor_args;
    return cost;
  }

  static bool IsSingleCoalescedMultiPieceOutput(const OutputRewriteInfo& output,
                                                const CalleeRewriteAnalysis& analysis,
                                                const WindowRewriteCost& cost) {
    if (!cost.is_multi_piece_output) return false;
    if (analysis.outputs.size() != 1) return false;
    // Exact multi-piece output expands the runtime ABI.  The only multi-piece
    // shape auto keeps is a single coarse carrier with fine callsite assembles.
    return cost.dense_piece_count == 1 && cost.assemble_piece_count > 1 && cost.has_callsite_assemble &&
           cost.abi_delta == 0;
  }

  static bool ShouldKeepOutputInAuto(const OutputRewriteInfo& output, const CalleeRewriteAnalysis& analysis,
                                     WindowRewriteCost* out_cost) {
    auto cost = EstimateOutputCost(output);
    if (cost.abi_delta > 0) {
      cost.reason = "abi_expanding";
      if (out_cost) *out_cost = std::move(cost);
      return false;
    }
    if (cost.is_multi_piece_output) {
      if (!IsSingleCoalescedMultiPieceOutput(output, analysis, cost)) {
        cost.reason = analysis.outputs.size() == 1 ? "multi_piece_output" : "multi_output_coalescing";
        if (out_cost) *out_cost = std::move(cost);
        return false;
      }
      cost.reason = "single_coalesced_multi_piece_output";
      if (out_cost) *out_cost = std::move(cost);
      return true;
    }
    if (cost.has_callsite_assemble && cost.abi_delta >= 0) {
      cost.reason = "assemble_without_abi_reduction";
      if (out_cost) *out_cost = std::move(cost);
      return false;
    }

    cost.reason = cost.abi_delta < 0 ? "abi_reducing" : "abi_neutral_single_piece";
    if (out_cost) *out_cost = std::move(cost);
    return true;
  }

  static bool ShouldKeepInputInAuto(const InputRewriteInfo& input, WindowRewriteCost* out_cost) {
    auto cost = EstimateInputCost(input);
    if (cost.abi_delta > 0) {
      cost.reason = "abi_expanding";
      if (out_cost) *out_cost = std::move(cost);
      return false;
    }

    cost.reason = cost.abi_delta < 0 ? "abi_reducing" : "abi_neutral";
    if (out_cost) *out_cost = std::move(cost);
    return true;
  }

  static bool CanUseRuntimeViewDisjointness(const CalleeRewriteAnalysis& analysis) {
    return analysis.kind == RewriteKind::AggregateWindowLoop &&
           (HasMultiPieceOutput(analysis) || HasCoalescedOutput(analysis));
  }

  static bool HasDynamicIndexedInputWindow(const std::vector<InputRewriteInfo>& inputs) {
    return std::any_of(inputs.begin(), inputs.end(), [](const InputRewriteInfo& input) {
      return input.dynamic_indexed_window.has_value();
    });
  }

  static std::optional<DenseRegionPiece> BuildBoundingDensePiece(const std::vector<DenseRegionPiece>& pieces,
                                                                 const std::vector<ExprPtr>& parent_shape,
                                                                 const Span& span) {
    if (pieces.size() <= 1 || pieces.front().callsite_offsets.empty()) return std::nullopt;
    const size_t rank = pieces.front().callsite_offsets.size();
    if (pieces.front().window_shape.size() != rank) return std::nullopt;

    std::vector<ExprPtr> min_offsets(rank);
    std::vector<ExprPtr> max_extents(rank);
    arith::Analyzer analyzer;
    for (const auto& piece : pieces) {
      if (piece.callsite_offsets.size() != rank || piece.window_shape.size() != rank) return std::nullopt;
      for (size_t dim = 0; dim < rank; ++dim) {
        auto min_offset = SelectMinExpr(min_offsets[dim], piece.callsite_offsets[dim], span);
        if (!min_offset.has_value()) return std::nullopt;
        min_offsets[dim] = *min_offset;

        auto extent = analyzer.Simplify(MakeAdd(piece.callsite_offsets[dim], piece.window_shape[dim], span));
        auto max_extent = SelectMaxExpr(max_extents[dim], extent, span);
        if (!max_extent.has_value()) return std::nullopt;
        max_extents[dim] = *max_extent;
      }
    }

    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> local_offsets;
    window_shape.reserve(rank);
    local_offsets.reserve(rank);
    for (size_t dim = 0; dim < rank; ++dim) {
      auto span_value = GetConstantSpanValue(max_extents[dim], min_offsets[dim], span);
      if (!span_value.has_value() || *span_value <= 0) return std::nullopt;
      window_shape.push_back(std::make_shared<ConstInt>(*span_value, DataType::INDEX, span));
      local_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, span));
    }

    if (AreExprVectorsEqual(window_shape, parent_shape) && IsAllZeroOffsets(min_offsets)) {
      return std::nullopt;
    }
    return MakeDensePiece(std::move(window_shape), std::move(min_offsets), std::move(local_offsets));
  }

  static void CoalesceOutputPieces(CalleeRewriteAnalysis* analysis) {
    if (!analysis) return;
    for (auto& output : analysis->outputs) {
      const auto original_pieces = output.region.dense_pieces;
      auto carrier = BuildBoundingDensePiece(original_pieces, output.parent_shape, Span::unknown());
      if (!carrier.has_value()) continue;

      output.assemble_region = MakeDenseRegion(original_pieces);
      output.window_shape = carrier->window_shape;
      output.callsite_offsets = carrier->callsite_offsets;
      output.local_store_offsets = carrier->local_offsets;
      output.region = MakeDenseRegion({std::move(*carrier)});
    }
  }

  void ApplyWindowRewritePolicy(const ProgramPtr& program, AnalysisMap* analyses) const {
    if (!analyses) return;
    auto function_lookup = BuildFunctionLookup(program);
    for (auto it = analyses->begin(); it != analyses->end();) {
      const auto& callee_name = it->first;
      auto& analysis = it->second;
      switch (window_rewrite_policy_) {
        case WindowRewritePolicy::Auto: {
          auto func_it = function_lookup.find(callee_name);
          auto func = func_it == function_lookup.end() ? nullptr : func_it->second;
          analysis.outputs.erase(
              std::remove_if(analysis.outputs.begin(), analysis.outputs.end(),
                             [&](const OutputRewriteInfo& output) {
                               WindowRewriteCost cost;
                               bool keep = ShouldKeepOutputInAuto(output, analysis, &cost);
                               LogAutoDecision(
                                   keep, DebugParamName(callee_name, func, output.out_param_index), cost);
                               return !keep;
                             }),
              analysis.outputs.end());
          analysis.inputs.erase(
              std::remove_if(analysis.inputs.begin(), analysis.inputs.end(),
                             [&](const InputRewriteInfo& input) {
                               WindowRewriteCost cost;
                               bool keep = ShouldKeepInputInAuto(input, &cost);
                               LogAutoDecision(keep, DebugParamName(callee_name, func, input.in_param_index),
                                               cost);
                               return !keep;
                             }),
              analysis.inputs.end());
          break;
        }
        case WindowRewritePolicy::All:
          break;
        case WindowRewritePolicy::InputsOnly:
          analysis.outputs.clear();
          break;
        case WindowRewritePolicy::OutputsOnly:
          analysis.inputs.clear();
          break;
        case WindowRewritePolicy::NoMultiPieceOutputs:
          analysis.outputs.erase(std::remove_if(analysis.outputs.begin(), analysis.outputs.end(),
                                                [](const OutputRewriteInfo& output) {
                                                  return DensePieces(output).size() > 1 ||
                                                         AssemblePieces(output).size() > 1;
                                                }),
                                 analysis.outputs.end());
          break;
        case WindowRewritePolicy::None:
          analysis.outputs.clear();
          analysis.inputs.clear();
          break;
      }
      if (analysis.outputs.empty() && analysis.inputs.empty()) {
        it = analyses->erase(it);
      } else {
        ++it;
      }
    }
  }

  static std::optional<TensorView> MakeWindowTensorView(const std::shared_ptr<const TensorType>& tensor_type,
                                                        const std::vector<ExprPtr>& parent_shape,
                                                        const std::vector<ExprPtr>& window_shape) {
    if (!tensor_type) return std::nullopt;
    if (tensor_type->tensor_view_.has_value()) {
      auto new_view = tensor_type->tensor_view_;
      if (new_view->stride.empty()) {
        if (new_view->layout == TensorLayout::NZ) return std::nullopt;
        new_view->stride =
            tensor_view_semantics::BuildLogicalStridesFromLayout(tensor_type->shape_, new_view->layout);
      }
      if (!new_view->valid_shape.empty()) new_view->valid_shape = window_shape;
      return new_view;
    }

    auto parent_strides =
        tensor_view_semantics::BuildLogicalStridesFromLayout(parent_shape, TensorLayout::ND);
    if (parent_strides.size() != window_shape.size()) return std::nullopt;
    return TensorView(std::move(parent_strides), TensorLayout::ND);
  }

  static TypePtr MakeWindowTensorType(const std::shared_ptr<const TensorType>& tensor_type,
                                      const std::vector<ExprPtr>& parent_shape,
                                      const std::vector<ExprPtr>& window_shape) {
    auto new_view = MakeWindowTensorView(tensor_type, parent_shape, window_shape);
    if (!new_view.has_value()) return nullptr;
    return std::make_shared<TensorType>(window_shape, tensor_type->dtype_, tensor_type->memref_, new_view);
  }

  static std::vector<ExprPtr> SubstituteExprVector(const std::vector<ExprPtr>& exprs,
                                                   const std::unordered_map<const Var*, ExprPtr>& subst) {
    std::vector<ExprPtr> result;
    result.reserve(exprs.size());
    for (const auto& expr : exprs) {
      result.push_back(transform_utils::Substitute(expr, subst));
    }
    return result;
  }

  static TypePtr SubstituteTypeExprs(const TypePtr& type,
                                     const std::unordered_map<const Var*, ExprPtr>& subst) {
    if (!type || subst.empty()) return type;
    if (auto tuple_type = As<TupleType>(type)) {
      std::vector<TypePtr> new_types;
      new_types.reserve(tuple_type->types_.size());
      bool changed = false;
      for (const auto& elem_type : tuple_type->types_) {
        auto new_type = SubstituteTypeExprs(elem_type, subst);
        changed = changed || new_type.get() != elem_type.get();
        new_types.push_back(std::move(new_type));
      }
      if (!changed) return type;
      return std::make_shared<TupleType>(std::move(new_types));
    }
    if (auto tensor_type = As<TensorType>(type)) {
      auto new_shape = SubstituteExprVector(tensor_type->shape_, subst);
      auto new_view = tensor_type->tensor_view_;
      if (new_view.has_value()) {
        new_view->stride = SubstituteExprVector(new_view->stride, subst);
        new_view->valid_shape = SubstituteExprVector(new_view->valid_shape, subst);
      }
      return std::make_shared<TensorType>(std::move(new_shape), tensor_type->dtype_, tensor_type->memref_,
                                          std::move(new_view));
    }
    return type;
  }

  struct AffineForm {
    int64_t coeff = 0;
    ExprPtr base;
  };

  struct OrderedLoopOffsets {
    ExprPtr min;
    ExprPtr max;
  };

  struct LinearIndexExpr {
    std::unordered_map<const Var*, int64_t> coeffs;
    int64_t constant = 0;
  };

  static std::optional<int64_t> CheckedAdd(int64_t lhs, int64_t rhs) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
      return std::nullopt;
    }
    return lhs + rhs;
  }

  static std::optional<int64_t> CheckedSub(int64_t lhs, int64_t rhs) {
    if (rhs == std::numeric_limits<int64_t>::min()) {
      return std::nullopt;
    }
    return CheckedAdd(lhs, -rhs);
  }

  static std::optional<int64_t> CheckedMul(int64_t lhs, int64_t rhs) {
    if (lhs == 0 || rhs == 0) return int64_t{0};
    if (lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) return std::nullopt;
    if (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()) return std::nullopt;
    if (lhs > 0) {
      if (rhs > 0) {
        if (lhs > std::numeric_limits<int64_t>::max() / rhs) return std::nullopt;
      } else if (rhs < std::numeric_limits<int64_t>::min() / lhs) {
        return std::nullopt;
      }
    } else {
      if (rhs > 0) {
        if (lhs < std::numeric_limits<int64_t>::min() / rhs) return std::nullopt;
      } else if (lhs < std::numeric_limits<int64_t>::max() / rhs) {
        return std::nullopt;
      }
    }
    return lhs * rhs;
  }

  static bool AddLinearCoeff(LinearIndexExpr* expr, const Var* var, int64_t coeff) {
    if (!expr || !var || coeff == 0) return true;
    auto& slot = expr->coeffs[var];
    auto sum = CheckedAdd(slot, coeff);
    if (!sum.has_value()) return false;
    slot = *sum;
    if (slot == 0) expr->coeffs.erase(var);
    return true;
  }

  static std::optional<LinearIndexExpr> ParseLinearIndexExpr(const ExprPtr& expr) {
    if (!expr) return std::nullopt;
    if (auto ci = As<ConstInt>(expr)) {
      return LinearIndexExpr{{}, ci->value_};
    }
    if (auto var = AsVarLike(expr)) {
      LinearIndexExpr result;
      AddLinearCoeff(&result, var.get(), 1);
      return result;
    }
    if (auto add = As<Add>(expr)) {
      auto lhs = ParseLinearIndexExpr(add->left_);
      auto rhs = ParseLinearIndexExpr(add->right_);
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      auto constant = CheckedAdd(lhs->constant, rhs->constant);
      if (!constant.has_value()) return std::nullopt;
      lhs->constant = *constant;
      for (const auto& [var, coeff] : rhs->coeffs) {
        if (!AddLinearCoeff(&*lhs, var, coeff)) return std::nullopt;
      }
      return lhs;
    }
    if (auto sub = As<Sub>(expr)) {
      auto lhs = ParseLinearIndexExpr(sub->left_);
      auto rhs = ParseLinearIndexExpr(sub->right_);
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      auto constant = CheckedSub(lhs->constant, rhs->constant);
      if (!constant.has_value()) return std::nullopt;
      lhs->constant = *constant;
      for (const auto& [var, coeff] : rhs->coeffs) {
        auto neg_coeff = CheckedSub(0, coeff);
        if (!neg_coeff.has_value()) return std::nullopt;
        if (!AddLinearCoeff(&*lhs, var, *neg_coeff)) return std::nullopt;
      }
      return lhs;
    }
    if (auto mul = As<Mul>(expr)) {
      auto lhs_ci = As<ConstInt>(mul->left_);
      auto rhs_ci = As<ConstInt>(mul->right_);
      ExprPtr scaled_expr;
      int64_t scale = 0;
      if (lhs_ci) {
        scaled_expr = mul->right_;
        scale = lhs_ci->value_;
      } else if (rhs_ci) {
        scaled_expr = mul->left_;
        scale = rhs_ci->value_;
      } else {
        return std::nullopt;
      }
      auto parsed = ParseLinearIndexExpr(scaled_expr);
      if (!parsed.has_value()) return std::nullopt;
      auto constant = CheckedMul(parsed->constant, scale);
      if (!constant.has_value()) return std::nullopt;
      parsed->constant = *constant;
      std::vector<const Var*> zero_coeff_vars;
      for (auto& [var, coeff] : parsed->coeffs) {
        auto scaled_coeff = CheckedMul(coeff, scale);
        if (!scaled_coeff.has_value()) return std::nullopt;
        coeff = *scaled_coeff;
        if (coeff == 0) zero_coeff_vars.push_back(var);
      }
      for (const auto* var : zero_coeff_vars) parsed->coeffs.erase(var);
      return parsed;
    }
    return std::nullopt;
  }

  static std::optional<int64_t> ConstantDiffIfSameLinearBase(const ExprPtr& lhs, const ExprPtr& rhs) {
    auto lhs_linear = ParseLinearIndexExpr(lhs);
    auto rhs_linear = ParseLinearIndexExpr(rhs);
    if (!lhs_linear.has_value() || !rhs_linear.has_value()) return std::nullopt;
    if (lhs_linear->coeffs != rhs_linear->coeffs) return std::nullopt;
    return CheckedSub(lhs_linear->constant, rhs_linear->constant);
  }

  static std::optional<int64_t> GetConstantSpanValue(const ExprPtr& max_extent, const ExprPtr& min_offset,
                                                     const Span& span) {
    arith::Analyzer analyzer;
    auto span_expr = analyzer.Simplify(MakeSub(max_extent, min_offset, span));
    if (auto span_ci = As<ConstInt>(span_expr)) return span_ci->value_;
    return ConstantDiffIfSameLinearBase(max_extent, min_offset);
  }

  static std::optional<ExprPtr> SelectMinExpr(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
    if (!lhs) return rhs;
    if (!rhs) return lhs;
    if (AreExprsEqual(lhs, rhs)) return lhs;

    arith::Analyzer analyzer;
    auto diff = analyzer.Simplify(MakeSub(lhs, rhs, span));
    auto diff_ci = As<ConstInt>(diff);
    if (diff_ci) return diff_ci->value_ <= 0 ? lhs : rhs;
    auto linear_diff = ConstantDiffIfSameLinearBase(lhs, rhs);
    if (!linear_diff.has_value()) return std::nullopt;
    return *linear_diff <= 0 ? lhs : rhs;
  }

  static std::optional<ExprPtr> SelectMaxExpr(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
    if (!lhs) return rhs;
    if (!rhs) return lhs;
    if (AreExprsEqual(lhs, rhs)) return lhs;

    arith::Analyzer analyzer;
    auto diff = analyzer.Simplify(MakeSub(lhs, rhs, span));
    auto diff_ci = As<ConstInt>(diff);
    if (diff_ci) return diff_ci->value_ >= 0 ? lhs : rhs;
    auto linear_diff = ConstantDiffIfSameLinearBase(lhs, rhs);
    if (!linear_diff.has_value()) return std::nullopt;
    return *linear_diff >= 0 ? lhs : rhs;
  }

  static std::optional<AffineForm> ParseAffineInLoop(const ExprPtr& expr, const Var* loop_var) {
    if (!expr) return std::nullopt;
    if (CountVarRefsInExpr(expr, loop_var) == 0) {
      return AffineForm{0, expr};
    }
    if (auto ci = As<ConstInt>(expr)) {
      return AffineForm{0, expr};
    }
    if (auto var = AsVarLike(expr)) {
      if (var.get() == loop_var) {
        auto zero = std::make_shared<ConstInt>(0, DataType::INDEX, expr->span_);
        return AffineForm{1, zero};
      }
      return AffineForm{0, expr};
    }
    if (auto add = As<Add>(expr)) {
      auto lhs = ParseAffineInLoop(add->left_, loop_var);
      auto rhs = ParseAffineInLoop(add->right_, loop_var);
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      return AffineForm{lhs->coeff + rhs->coeff, MakeAdd(lhs->base, rhs->base, expr->span_)};
    }
    if (auto sub = As<Sub>(expr)) {
      auto lhs = ParseAffineInLoop(sub->left_, loop_var);
      auto rhs = ParseAffineInLoop(sub->right_, loop_var);
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      return AffineForm{lhs->coeff - rhs->coeff, MakeSub(lhs->base, rhs->base, expr->span_)};
    }
    if (auto mul = As<Mul>(expr)) {
      auto lhs_ci = As<ConstInt>(mul->left_);
      auto rhs_ci = As<ConstInt>(mul->right_);
      if (lhs_ci) {
        auto rhs = ParseAffineInLoop(mul->right_, loop_var);
        if (!rhs.has_value()) return std::nullopt;
        return AffineForm{lhs_ci->value_ * rhs->coeff,
                          MakeMul(std::make_shared<ConstInt>(lhs_ci->value_, lhs_ci->dtype(), lhs_ci->span_),
                                  rhs->base, expr->span_)};
      }
      if (rhs_ci) {
        auto lhs = ParseAffineInLoop(mul->left_, loop_var);
        if (!lhs.has_value()) return std::nullopt;
        return AffineForm{
            rhs_ci->value_ * lhs->coeff,
            MakeMul(lhs->base, std::make_shared<ConstInt>(rhs_ci->value_, rhs_ci->dtype(), rhs_ci->span_),
                    expr->span_)};
      }
    }
    return std::nullopt;
  }

  class WindowWriteLocalizer : public IRMutator {
   public:
    WindowWriteLocalizer(const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var,
                         const std::unordered_map<const Var*, ExprPtr>& new_out_vars)
        : out_info_by_var_(out_info_by_var), new_out_vars_(new_out_vars) {}

   protected:
    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto remap_it = result_var_remap_.find(op.get());
      if (remap_it != result_var_remap_.end()) return remap_it->second;
      auto out_it = new_out_vars_.find(op.get());
      if (out_it != new_out_vars_.end()) return out_it->second;
      return IRMutator::VisitExpr_(op);
    }

    ExprPtr VisitExpr_(const IterArgPtr& op) override {
      auto out_it = new_out_vars_.find(op.get());
      if (out_it != new_out_vars_.end()) return out_it->second;
      return IRMutator::VisitExpr_(op);
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto visited_value = VisitExpr(op->value_);
      auto assign = MutableCopy(op);
      assign->value_ = visited_value;
      auto call = As<Call>(assign->value_);
      if (!call) return assign;

      ExprPtr rewritten_target_expr;
      const Var* target_var = nullptr;
      MakeTuplePtr offsets;
      size_t offset_arg_index = SIZE_MAX;
      size_t target_arg_index = SIZE_MAX;

      if (call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
        rewritten_target_expr = call->args_[2];
        auto out_var = AsVarLike(rewritten_target_expr);
        if (!out_var) return assign;
        target_var = out_var.get();
        offsets = As<MakeTuple>(call->args_[1]);
        offset_arg_index = 1;
        target_arg_index = 2;
      } else if (call->op_->name_ == "tensor.assemble" && call->args_.size() >= 3) {
        rewritten_target_expr = call->args_[0];
        auto parent_var = AsVarLike(rewritten_target_expr);
        if (!parent_var) return assign;
        target_var = parent_var.get();
        offsets = As<MakeTuple>(call->args_[2]);
        offset_arg_index = 2;
        target_arg_index = 0;
      } else if (call->op_->name_ == "tile.load" && call->args_.size() >= 3) {
        rewritten_target_expr = call->args_[0];
        auto parent_var = AsVarLike(rewritten_target_expr);
        if (!parent_var) return assign;
        target_var = parent_var.get();
        offsets = As<MakeTuple>(call->args_[1]);
        offset_arg_index = 1;
        target_arg_index = 0;
      } else if (call->op_->name_ == "tensor.slice" && call->args_.size() >= 3) {
        rewritten_target_expr = call->args_[0];
        auto parent_var = AsVarLike(rewritten_target_expr);
        if (!parent_var) return assign;
        target_var = parent_var.get();
        offsets = As<MakeTuple>(call->args_[2]);
        offset_arg_index = 2;
        target_arg_index = 0;
      } else {
        return assign;
      }

      const OutputRewriteInfo* info = nullptr;
      auto info_it = out_info_by_var_.find(target_var);
      if (info_it != out_info_by_var_.end()) {
        info = &info_it->second;
      } else {
        auto result_info_it = result_var_output_info_.find(target_var);
        if (result_info_it != result_var_output_info_.end()) info = result_info_it->second;
      }
      if (!info) return assign;
      if (!offsets) return assign;
      if (offsets->elements_.size() != info->callsite_offsets.size()) return assign;

      arith::Analyzer analyzer;
      std::vector<ExprPtr> local_offsets;
      local_offsets.reserve(offsets->elements_.size());
      for (size_t i = 0; i < offsets->elements_.size(); ++i) {
        local_offsets.push_back(analyzer.Simplify(
            MakeSub(offsets->elements_[i], info->callsite_offsets[i], offsets->elements_[i]->span_)));
      }
      auto new_offset_tuple = std::make_shared<MakeTuple>(std::move(local_offsets), offsets->span_);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[offset_arg_index] = new_offset_tuple;
      auto new_out_it = new_out_vars_.find(target_var);
      if (new_out_it != new_out_vars_.end()) new_args[target_arg_index] = new_out_it->second;
      auto new_type = (call->op_->name_ == "tile.store" || call->op_->name_ == "tensor.assemble")
                          ? new_args[target_arg_index]->GetType()
                          : call->GetType();
      auto new_call =
          std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, new_type, call->span_);

      auto new_result_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
      result_var_remap_[assign->var_.get()] = new_result_var;
      result_var_output_info_[new_result_var.get()] = info;
      assign->var_ = new_result_var;
      assign->value_ = new_call;
      return assign;
    }

    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      auto new_loop = MutableCopy(op);
      new_loop->start_ = VisitExpr(op->start_);
      new_loop->stop_ = VisitExpr(op->stop_);
      new_loop->step_ = VisitExpr(op->step_);

      std::unordered_map<const Var*, OutputRewriteInfo> nested_out_info(out_info_by_var_.begin(),
                                                                        out_info_by_var_.end());
      std::unordered_map<const Var*, ExprPtr> nested_new_out_vars(new_out_vars_.begin(), new_out_vars_.end());
      bool changed = false;

      for (size_t i = 0; i < new_loop->iter_args_.size() && i < new_loop->return_vars_.size(); ++i) {
        auto old_iter_arg = new_loop->iter_args_[i];
        auto old_return_var = new_loop->return_vars_[i];
        auto init_expr = VisitExpr(old_iter_arg->initValue_);
        auto init_var = AsVarLike(init_expr);
        if (!init_var) {
          if (init_expr.get() != old_iter_arg->initValue_.get()) {
            auto new_iter_arg = std::make_shared<IterArg>(old_iter_arg->name_hint_, old_iter_arg->GetType(),
                                                          init_expr, old_iter_arg->span_);
            new_loop->iter_args_[i] = new_iter_arg;
            changed = true;
          }
          continue;
        }

        const OutputRewriteInfo* info = nullptr;
        auto direct_info_it = out_info_by_var_.find(init_var.get());
        if (direct_info_it != out_info_by_var_.end()) {
          info = &direct_info_it->second;
        } else {
          auto result_info_it = result_var_output_info_.find(init_var.get());
          if (result_info_it != result_var_output_info_.end()) info = result_info_it->second;
        }

        if (!info) {
          if (init_expr.get() != old_iter_arg->initValue_.get()) {
            auto new_iter_arg = std::make_shared<IterArg>(old_iter_arg->name_hint_, old_iter_arg->GetType(),
                                                          init_expr, old_iter_arg->span_);
            new_loop->iter_args_[i] = new_iter_arg;
            changed = true;
          }
          continue;
        }

        auto narrowed_type = init_expr->GetType();
        auto new_iter_arg = std::make_shared<IterArg>(old_iter_arg->name_hint_, narrowed_type, init_expr,
                                                      old_iter_arg->span_);
        auto new_return_var =
            std::make_shared<Var>(old_return_var->name_hint_, narrowed_type, old_return_var->span_);

        nested_out_info[old_iter_arg.get()] = *info;
        nested_out_info[new_iter_arg.get()] = *info;
        nested_new_out_vars[old_iter_arg.get()] = new_iter_arg;
        nested_new_out_vars[new_iter_arg.get()] = new_iter_arg;
        result_var_remap_[old_return_var.get()] = new_return_var;
        result_var_output_info_[new_return_var.get()] = info;

        new_loop->iter_args_[i] = new_iter_arg;
        new_loop->return_vars_[i] = new_return_var;
        changed = true;
      }

      if (!changed) return IRMutator::VisitStmt_(op);

      WindowWriteLocalizer nested_localizer(nested_out_info, nested_new_out_vars);
      new_loop->body_ = nested_localizer.VisitStmt(new_loop->body_);
      return new_loop;
    }

   private:
    const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var_;
    const std::unordered_map<const Var*, ExprPtr>& new_out_vars_;
    std::unordered_map<const Var*, VarPtr> result_var_remap_;
    std::unordered_map<const Var*, const OutputRewriteInfo*> result_var_output_info_;
  };

  class WindowReadLocalizer : public IRMutator {
   public:
    explicit WindowReadLocalizer(const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var)
        : in_info_by_var_(in_info_by_var) {}

   protected:
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      auto visited_value = VisitExpr(op->value_);
      auto assign = MutableCopy(op);
      assign->value_ = visited_value;

      auto call = As<Call>(assign->value_);
      if (!call || call->args_.empty()) return assign;

      size_t offset_arg_index = SIZE_MAX;
      if (call->op_->name_ == "tile.load" && call->args_.size() >= 3) {
        offset_arg_index = 1;
      } else if (call->op_->name_ == "tensor.slice" && call->args_.size() >= 3) {
        // Keep the localizer aligned with AnalyzeInputWindows(): only window
        // reads that are already proven as a fixed tile.load/tensor.slice are
        // rewritten, and tensor.slice only localizes the matched offset.
        offset_arg_index = 2;
      } else {
        return assign;
      }

      auto parent = AsVarLike(call->args_[0]);
      auto info_it = parent ? in_info_by_var_.find(parent.get()) : in_info_by_var_.end();
      if (info_it == in_info_by_var_.end()) return assign;

      auto old_offsets = As<MakeTuple>(call->args_[offset_arg_index]);
      if (!old_offsets) return assign;
      if (old_offsets->elements_.size() != info_it->second.callsite_offsets.size()) return assign;

      arith::Analyzer analyzer;
      std::vector<ExprPtr> local_offsets;
      local_offsets.reserve(old_offsets->elements_.size());
      for (size_t i = 0; i < old_offsets->elements_.size(); ++i) {
        ExprPtr base_offset = info_it->second.callsite_offsets[i];
        if (info_it->second.dynamic_indexed_window.has_value() &&
            i == info_it->second.dynamic_indexed_window->dynamic_dim) {
          if (!info_it->second.dynamic_base_param) return assign;
          base_offset = info_it->second.dynamic_base_param;
        }
        local_offsets.push_back(analyzer.Simplify(
            MakeSub(old_offsets->elements_[i], base_offset, old_offsets->elements_[i]->span_)));
      }

      std::vector<ExprPtr> new_args = call->args_;
      new_args[offset_arg_index] = std::make_shared<MakeTuple>(std::move(local_offsets), old_offsets->span_);
      assign->value_ = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_,
                                              call->GetType(), call->span_);
      return assign;
    }

   private:
    const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var_;
  };

  class OrchRewriter : public IRMutator {
   public:
    OrchRewriter(ProgramPtr program, const AnalysisMap& analyses,
                 const std::unordered_map<std::string, FunctionPtr>& cloned_funcs,
                 const std::unordered_map<std::string, FunctionPtr>& function_lookup,
                 const std::unordered_map<std::string, std::unordered_map<size_t, VarPtr>>&
                     output_dynamic_extent_dims_by_func,
                 WindowRewritePolicy window_rewrite_policy)
        : program_(std::move(program)),
          analyses_(analyses),
          cloned_funcs_(cloned_funcs),
          function_lookup_(function_lookup),
          output_dynamic_extent_dims_by_func_(output_dynamic_extent_dims_by_func),
          window_rewrite_policy_(window_rewrite_policy) {}

    const std::unordered_set<std::string>& used_clone_names() const { return used_clone_names_; }

   protected:
    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      bool is_sequential = op->kind_ != ForKind::Parallel;
      StmtPtr result;
      {
        ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&loop_iter_init_subst_, op->iter_args_);

        loop_context_.push_back(LoopContext{op, op->loop_var_, op->start_, op->stop_, op->step_});
        if (is_sequential) {
          sequential_loops_.push_back(op);
          loop_local_allocs_.emplace_back(CollectLoopLocalTensorAllocs(op));
        }
        result = IRMutator::VisitStmt_(op);
        if (is_sequential) {
          loop_local_allocs_.pop_back();
          sequential_loops_.pop_back();
        }
        loop_context_.pop_back();
      }
      RecordLoopReturnInitAliases(op);
      return result;
    }

    StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
      StmtPtr result;
      {
        ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&loop_iter_init_subst_, op->iter_args_);
        ++while_depth_;
        result = IRMutator::VisitStmt_(op);
        --while_depth_;
      }
      auto visited_loop = As<WhileStmt>(result);
      RecordLoopReturnInitAliases(visited_loop ? visited_loop : op);
      return result;
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      std::vector<StmtPtr> new_stmts;
      new_stmts.reserve(op->stmts_.size());
      bool changed = false;
      auto saved_scalar_defs = scalar_defs_;
      auto saved_tuple_result_subst = tuple_result_subst_;
      auto saved_window_parent_subst = window_parent_subst_;
      auto saved_sibling_output_alias_roots = sibling_output_alias_roots_;
      auto saved_sibling_unwindowable_output_roots = sibling_unwindowable_output_roots_;
      auto saved_sequence_carrier_plan = sequence_carrier_plan_;
      auto later_assemble_source_indices = CollectAssembleSourceIndices(op->stmts_);
      sibling_output_alias_roots_.clear();
      sibling_unwindowable_output_roots_.clear();
      sequence_carrier_plan_ = SequenceCarrierPlan();
      CollectSiblingOutputAliases(op->stmts_);
      sequence_carrier_plan_ =
          MergeSequenceCarrierPlans(saved_sequence_carrier_plan, BuildSequenceCarrierPlan(op->stmts_));
      if (!sequence_carrier_plan_.prelude_stmts.empty()) {
        changed = true;
        for (const auto& prelude_stmt : sequence_carrier_plan_.prelude_stmts) {
          if (auto prelude_assign = As<AssignStmt>(prelude_stmt)) {
            if (As<ScalarType>(prelude_assign->var_->GetType())) {
              scalar_defs_[prelude_assign->var_.get()] = prelude_assign->value_;
            }
          }
          new_stmts.push_back(prelude_stmt);
        }
      }

      for (size_t stmt_index = 0; stmt_index < op->stmts_.size(); ++stmt_index) {
        const auto& stmt = op->stmts_[stmt_index];
        auto call_assign = As<AssignStmt>(stmt);
        auto bundle = call_assign ? TryRewriteCall(call_assign, later_assemble_source_indices, stmt_index)
                                  : std::nullopt;
        if (bundle.has_value()) {
          changed = true;
          for (const auto& new_stmt : bundle->stmts) {
            auto visited = VisitStmt(new_stmt);
            if (auto visited_assign = As<AssignStmt>(visited)) {
              if (As<ScalarType>(visited_assign->var_->GetType())) {
                scalar_defs_[visited_assign->var_.get()] = visited_assign->value_;
              }
            }
            new_stmts.push_back(visited);
          }
          for (const auto& [parent, replacement] : bundle->parent_substs) {
            window_parent_subst_[parent] = replacement;
          }
          continue;
        }

        auto visited = VisitStmt(stmt);
        changed = changed || visited.get() != stmt.get();
        new_stmts.push_back(visited);

        auto visited_assign = As<AssignStmt>(visited);
        if (visited_assign && As<ScalarType>(visited_assign->var_->GetType())) {
          scalar_defs_[visited_assign->var_.get()] = visited_assign->value_;
        }
      }

      scalar_defs_ = std::move(saved_scalar_defs);
      tuple_result_subst_ = std::move(saved_tuple_result_subst);
      window_parent_subst_ = std::move(saved_window_parent_subst);
      sibling_output_alias_roots_ = std::move(saved_sibling_output_alias_roots);
      sibling_unwindowable_output_roots_ = std::move(saved_sibling_unwindowable_output_roots);
      sequence_carrier_plan_ = std::move(saved_sequence_carrier_plan);
      if (!changed) return op;
      return SeqStmts::Flatten(std::move(new_stmts), op->span_);
    }

   private:
    struct SliceBundle {
      VarPtr slice_var;
      ExprPtr parent_expr;
      MakeTuplePtr offset_tuple;
    };

    struct RewriteBundle {
      std::vector<StmtPtr> stmts;
      std::vector<std::pair<const Var*, ExprPtr>> parent_substs;
    };

    struct RuntimeObservableCarrier {
      const Var* parent_root = nullptr;
      ExprPtr parent_expr;
      std::vector<ExprPtr> offsets;
      std::vector<ExprPtr> shape;
      VarPtr initial_var;
      VarPtr current_var;
      bool has_dynamic_reader = false;
      size_t dynamic_dim = SIZE_MAX;
      InputRewriteInfo reader_input;
      DenseRegionPiece reader_piece;
      CallPtr reader_call;
      AssignStmtPtr reader_assign;
      VarPtr dynamic_base_arg;
      VarPtr dynamic_extent_arg;
    };

    struct LoopContext {
      ForStmtPtr loop;
      VarPtr loop_var;
      ExprPtr start;
      ExprPtr stop;
      ExprPtr step;
    };

    struct CallsiteAccessCandidate {
      size_t stmt_index = 0;
      AssignStmtPtr assign;
      CallPtr call;
      size_t param_index = SIZE_MAX;
      ParamDirection direction = ParamDirection::In;
      const Var* parent_root = nullptr;
      ExprPtr parent_expr;
      std::vector<ExprPtr> offsets;
      std::vector<ExprPtr> shape;
      bool dynamic_indexed_reader = false;
      bool output_writer = false;
      bool output_has_fine_assemble = false;
      InputRewriteInfo input_info;
      std::vector<LoopContext> enclosing_loops;
      std::unordered_map<const Var*, ExprPtr> callsite_subst;
    };

    struct SequenceCarrierPlan {
      std::unordered_map<const Var*, RuntimeObservableCarrier> carriers_by_parent;
      std::unordered_set<const Var*> dynamic_reader_roots;
      std::unordered_set<const Var*> fallback_parent_roots;
      std::vector<StmtPtr> prelude_stmts;
    };

    struct CarrierUse {
      const Var* parent_root = nullptr;
      RuntimeObservableCarrier* carrier = nullptr;
    };

    struct DynamicWindowScanResult {
      std::unordered_map<const Var*, ExprPtr> min_subst;
      std::unordered_map<const Var*, ExprPtr> max_subst;
    };

    struct FullOffsetScanResult {
      VarPtr min_offset;
      VarPtr max_offset;
    };

    bool CanProveDynamicWindowScanHasUpdate(
        const DynamicIndexedInputWindow& dynamic,
        const std::unordered_map<const Var*, ExprPtr>& callsite_subst) const {
      arith::Analyzer analyzer;
      for (const auto& loop : loop_context_) {
        if (!loop.loop_var || !loop.start || !loop.stop) continue;
        analyzer.int_set.Bind(loop.loop_var, loop.start, loop.stop);
      }

      if (!dynamic.loop_var || !dynamic.loop_start || !dynamic.loop_stop || !dynamic.loop_step) {
        return false;
      }
      auto start = analyzer.Simplify(transform_utils::Substitute(dynamic.loop_start, callsite_subst));
      auto stop = analyzer.Simplify(transform_utils::Substitute(dynamic.loop_stop, callsite_subst));
      auto step = analyzer.Simplify(transform_utils::Substitute(dynamic.loop_step, callsite_subst));
      auto step_const = GetConstIntValue(step);
      if (!step_const.has_value() || *step_const == 0) return false;

      auto non_empty =
          *step_const > 0 ? MakeLt(start, stop, Span::unknown()) : MakeGt(start, stop, Span::unknown());
      if (!analyzer.CanProve(non_empty)) return false;

      if (!dynamic.guard_condition) return true;

      auto guard_subst = callsite_subst;
      guard_subst[dynamic.loop_var.get()] = start;
      auto guard = analyzer.Simplify(transform_utils::Substitute(dynamic.guard_condition, guard_subst));
      return analyzer.CanProve(guard);
    }

    static ExprPtr SubstituteCallsiteAndLoops(const ExprPtr& expr,
                                              const std::unordered_map<const Var*, ExprPtr>& callsite_subst,
                                              const std::unordered_map<const Var*, ExprPtr>& loop_subst) {
      return arith::Analyzer().Simplify(
          transform_utils::Substitute(transform_utils::Substitute(expr, callsite_subst), loop_subst));
    }

    static bool IsSafeInlineScalarSubstitution(const ExprPtr& expr) {
      class Checker : public IRVisitor {
       public:
        [[nodiscard]] bool ok() const { return ok_; }

       protected:
        void VisitExpr_(const CallPtr& op) override {
          ok_ = false;
          IRVisitor::VisitExpr_(op);
        }

        void VisitExpr_(const SubmitPtr& op) override {
          ok_ = false;
          IRVisitor::VisitExpr_(op);
        }

       private:
        bool ok_ = true;
      };

      if (!expr) return false;
      Checker checker;
      checker.VisitExpr(expr);
      return checker.ok();
    }

    ExprPtr FlattenGeneratedScalarExpr(const ExprPtr& expr, const std::string& name_prefix, const Span& span,
                                       std::vector<StmtPtr>* stmts) {
      class LocalFlattener : public IRMutator {
       public:
        LocalFlattener(std::string name_prefix, size_t* counter, std::vector<StmtPtr>* stmts, Span span)
            : name_prefix_(std::move(name_prefix)),
              counter_(counter),
              stmts_(stmts),
              span_(std::move(span)) {}

       protected:
        ExprPtr VisitExpr_(const CallPtr& op) override {
          std::vector<ExprPtr> new_args;
          new_args.reserve(op->args_.size());
          bool changed = false;
          for (const auto& arg : op->args_) {
            auto visited = VisitExpr(arg);
            if (As<Call>(visited) || As<Submit>(visited)) {
              visited = ExtractCallToTemp(visited);
              changed = true;
            } else if (visited.get() != arg.get()) {
              changed = true;
            }
            new_args.push_back(visited);
          }
          if (!changed) return op;
          return std::make_shared<Call>(op->op_, new_args, op->kwargs_, op->attrs_, op->GetType(), op->span_);
        }

        ExprPtr VisitExpr_(const SubmitPtr& op) override {
          std::vector<ExprPtr> new_args;
          new_args.reserve(op->args_.size());
          bool changed = false;
          for (const auto& arg : op->args_) {
            auto visited = VisitExpr(arg);
            if (As<Call>(visited) || As<Submit>(visited)) {
              visited = ExtractCallToTemp(visited);
              changed = true;
            } else if (visited.get() != arg.get()) {
              changed = true;
            }
            new_args.push_back(visited);
          }
          if (!changed) return op;
          return std::make_shared<Submit>(op->op_, new_args, op->deps_, op->kwargs_, op->attrs_,
                                          op->GetType(), op->span_, op->core_num_, op->sync_start_);
        }

        ExprPtr VisitExpr_(const AddPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const SubPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const MulPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const FloorDivPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const FloorModPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const FloatDivPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const MinPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const MaxPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const PowPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const EqPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const NePtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const LtPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const LePtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const GtPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const GePtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const AndPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const OrPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const XorPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const BitAndPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const BitOrPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const BitXorPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const BitShiftLeftPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const BitShiftRightPtr& op) override { return ProcessBinaryExpr(op); }
        ExprPtr VisitExpr_(const AbsPtr& op) override { return ProcessUnaryExpr(op); }
        ExprPtr VisitExpr_(const NegPtr& op) override { return ProcessUnaryExpr(op); }
        ExprPtr VisitExpr_(const NotPtr& op) override { return ProcessUnaryExpr(op); }
        ExprPtr VisitExpr_(const BitNotPtr& op) override { return ProcessUnaryExpr(op); }
        ExprPtr VisitExpr_(const CastPtr& op) override { return ProcessUnaryExpr(op); }

       private:
        ExprPtr ExtractCallToTemp(const ExprPtr& expr) {
          if (!As<Call>(expr) && !As<Submit>(expr)) return expr;
          auto temp_var = std::make_shared<Var>(name_prefix_ + "__expr_tmp_" + std::to_string((*counter_)++),
                                                expr->GetType(), expr->span_);
          stmts_->push_back(std::make_shared<AssignStmt>(temp_var, expr, temp_var->span_));
          return temp_var;
        }

#define PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(OpName)                                                    \
  ExprPtr ProcessBinaryExpr(const OpName##Ptr& op) {                                                    \
    auto new_left = VisitExpr(op->left_);                                                               \
    auto new_right = VisitExpr(op->right_);                                                             \
    if (As<Call>(new_left) || As<Submit>(new_left)) new_left = ExtractCallToTemp(new_left);             \
    if (As<Call>(new_right) || As<Submit>(new_right)) new_right = ExtractCallToTemp(new_right);         \
    if (new_left.get() == op->left_.get() && new_right.get() == op->right_.get()) return op;            \
    auto scalar_type = As<ScalarType>(op->GetType());                                                   \
    INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "Generated scalar binary expression must be scalar"; \
    return std::make_shared<const OpName>(new_left, new_right, scalar_type->dtype_, op->span_);         \
  }

        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Add)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Sub)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Mul)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(FloorDiv)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(FloorMod)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(FloatDiv)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Min)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Max)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Pow)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Eq)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Ne)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Lt)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Le)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Gt)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Ge)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(And)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Or)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(Xor)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(BitAnd)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(BitOr)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(BitXor)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(BitShiftLeft)
        PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD(BitShiftRight)
#undef PYPTO_WINDOW_PROCESS_BINARY_OVERLOAD

#define PYPTO_WINDOW_PROCESS_UNARY_OVERLOAD(OpName)                                                    \
  ExprPtr ProcessUnaryExpr(const OpName##Ptr& op) {                                                    \
    auto new_operand = VisitExpr(op->operand_);                                                        \
    if (As<Call>(new_operand) || As<Submit>(new_operand)) {                                            \
      new_operand = ExtractCallToTemp(new_operand);                                                    \
    }                                                                                                  \
    if (new_operand.get() == op->operand_.get()) return op;                                            \
    auto scalar_type = As<ScalarType>(op->GetType());                                                  \
    INTERNAL_CHECK_SPAN(scalar_type, op->span_) << "Generated scalar unary expression must be scalar"; \
    return std::make_shared<const OpName>(new_operand, scalar_type->dtype_, op->span_);                \
  }

        PYPTO_WINDOW_PROCESS_UNARY_OVERLOAD(Abs)
        PYPTO_WINDOW_PROCESS_UNARY_OVERLOAD(Neg)
        PYPTO_WINDOW_PROCESS_UNARY_OVERLOAD(Not)
        PYPTO_WINDOW_PROCESS_UNARY_OVERLOAD(BitNot)
        PYPTO_WINDOW_PROCESS_UNARY_OVERLOAD(Cast)
#undef PYPTO_WINDOW_PROCESS_UNARY_OVERLOAD

        std::string name_prefix_;
        size_t* counter_;
        std::vector<StmtPtr>* stmts_;
        Span span_;
      };

      if (!expr || !stmts) return expr;
      LocalFlattener flattener(name_prefix, &generated_scalar_temp_counter_, stmts, span);
      return flattener.VisitExpr(expr);
    }

    static std::optional<std::vector<ExprPtr>> SubstituteSingletonLoopStarts(
        const std::vector<ExprPtr>& exprs, const std::vector<LoopContext>& loops) {
      std::unordered_map<const Var*, ExprPtr> subst;
      for (const auto& loop : loops) {
        if (!loop.loop || !loop.loop_var) continue;
        bool referenced = false;
        for (const auto& expr : exprs) {
          if (CountVarRefsInExpr(expr, loop.loop_var.get()) != 0) {
            referenced = true;
            break;
          }
        }
        if (!referenced) continue;
        auto trip_count = GetKnownPositiveTripCount(loop.loop);
        if (!trip_count.has_value() || *trip_count != 1) return std::nullopt;
        subst[loop.loop_var.get()] = loop.start;
      }
      return SubstituteExprVector(exprs, subst);
    }

    static SequenceCarrierPlan MergeSequenceCarrierPlans(SequenceCarrierPlan parent,
                                                         SequenceCarrierPlan child) {
      for (auto& [root, carrier] : child.carriers_by_parent) {
        parent.carriers_by_parent[root] = std::move(carrier);
      }
      parent.dynamic_reader_roots.insert(child.dynamic_reader_roots.begin(),
                                         child.dynamic_reader_roots.end());
      parent.fallback_parent_roots.insert(child.fallback_parent_roots.begin(),
                                          child.fallback_parent_roots.end());
      parent.prelude_stmts.insert(parent.prelude_stmts.end(), child.prelude_stmts.begin(),
                                  child.prelude_stmts.end());
      for (const auto* root : parent.fallback_parent_roots) {
        parent.carriers_by_parent.erase(root);
      }
      return parent;
    }

    struct LoopDisjointnessCandidate {
      ForStmtPtr loop;
      const std::unordered_set<const Var*>* loop_local_allocs = nullptr;
    };

    enum class LoopRegionRole {
      Partition,
      Reduction,
      Unknown,
    };

    template <typename LoopPtr>
    void RecordLoopReturnInitAliases(const LoopPtr& loop) {
      if (!loop) return;
      size_t n = std::min(loop->iter_args_.size(), loop->return_vars_.size());
      for (size_t i = 0; i < n; ++i) {
        const auto& iter_arg = loop->iter_args_[i];
        const auto& return_var = loop->return_vars_[i];
        if (!iter_arg || !iter_arg->initValue_ || !return_var) continue;
        if (return_var.get() == iter_arg.get()) continue;
        if (!AsTensorTypeLike(return_var->GetType())) continue;
        auto parent_expr = ResolveLoopInitExpr(iter_arg->initValue_);
        if (!AsVarLike(parent_expr)) continue;
        loop_iter_init_subst_[return_var.get()] = parent_expr;
        loop_return_init_subst_[return_var.get()] = parent_expr;
      }
    }

    const std::vector<OutParamReturnMapping>& GetOutParamReturnMappings(const FunctionPtr& func,
                                                                        bool include_inout) {
      static const std::vector<OutParamReturnMapping> kEmpty;
      if (!func) return kEmpty;
      auto key = func->name_ + (include_inout ? "#inout" : "#out");
      auto it = out_param_return_mappings_cache_.find(key);
      if (it != out_param_return_mappings_cache_.end()) return it->second;
      auto [inserted_it, _] = out_param_return_mappings_cache_.emplace(
          std::move(key), BuildOutParamReturnMappings(func, include_inout));
      return inserted_it->second;
    }

    static std::unordered_map<const Var*, size_t> CollectAssembleSourceIndices(
        const std::vector<StmtPtr>& sibling_stmts) {
      std::unordered_map<const Var*, size_t> result;
      for (size_t i = 0; i < sibling_stmts.size(); ++i) {
        auto assign = As<AssignStmt>(sibling_stmts[i]);
        auto call = assign ? As<Call>(assign->value_) : nullptr;
        if (!call || call->op_->name_ != "tensor.assemble" || call->args_.size() < 2) continue;
        auto source = AsVarLike(call->args_[1]);
        if (source) result[source.get()] = i;
      }
      return result;
    }

    static bool IsCallResultAssembledLater(
        const VarPtr& result_var, const std::unordered_map<const Var*, size_t>& assemble_source_indices,
        size_t stmt_index) {
      if (!result_var) return false;
      auto it = assemble_source_indices.find(result_var.get());
      return it != assemble_source_indices.end() && it->second > stmt_index;
    }

    std::optional<DynamicWindowScanResult> EmitDynamicWindowScan(
        const InputRewriteInfo& input, const DenseRegionPiece& piece, const CallPtr& call,
        const AssignStmtPtr& call_assign, const VarPtr& in_arg,
        const std::unordered_map<const Var*, ExprPtr>& callsite_subst, arith::Analyzer* analyzer,
        std::vector<StmtPtr>* stmts) {
      if (!input.dynamic_indexed_window.has_value() || !call || !call_assign || !in_arg || !analyzer ||
          !stmts) {
        return std::nullopt;
      }

      const auto& dynamic = *input.dynamic_indexed_window;
      if (dynamic.index_tensor_param_index >= call->args_.size() ||
          dynamic.dynamic_dim >= piece.window_shape.size() ||
          dynamic.dynamic_dim >= piece.callsite_offsets.size() || !dynamic.dynamic_index_value) {
        return std::nullopt;
      }

      auto index_tensor_arg = MaterializeWindowParentExpr(call->args_[dynamic.index_tensor_param_index]);
      auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
      auto int32_type = std::make_shared<ScalarType>(DataType::INT32);
      auto scan_loop_var =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_scan_i", index_type, call_assign->span_);
      auto scan_min_init =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_min_init", index_type, call_assign->span_);
      auto scan_max_init =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_max_init", index_type, call_assign->span_);
      auto scan_min_iter = std::make_shared<IterArg>(in_arg->name_hint_ + "__window_min_iter", index_type,
                                                     scan_min_init, call_assign->span_);
      auto scan_max_iter = std::make_shared<IterArg>(in_arg->name_hint_ + "__window_max_iter", index_type,
                                                     scan_max_init, call_assign->span_);
      auto scan_min_phi =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_min_phi", index_type, call_assign->span_);
      auto scan_max_phi =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_max_phi", index_type, call_assign->span_);
      auto dynamic_min_var =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_min", index_type, call_assign->span_);
      auto dynamic_max_var =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_max", index_type, call_assign->span_);

      std::unordered_map<const Var*, ExprPtr> scan_subst = callsite_subst;
      scan_subst[dynamic.loop_var.get()] = scan_loop_var;

      std::vector<ExprPtr> index_offsets;
      index_offsets.reserve(dynamic.index_offsets.size());
      for (const auto& index_offset : dynamic.index_offsets) {
        index_offsets.push_back(analyzer->Simplify(transform_utils::Substitute(index_offset, scan_subst)));
      }
      std::vector<StmtPtr> then_stmts;
      std::vector<ExprPtr> index_offset_vars;
      index_offset_vars.reserve(index_offsets.size());
      for (size_t i = 0; i < index_offsets.size(); ++i) {
        auto index_var = std::make_shared<Var>(in_arg->name_hint_ + "__window_index_" + std::to_string(i),
                                               index_type, call_assign->span_);
        auto index_expr =
            FlattenGeneratedScalarExpr(index_offsets[i], in_arg->name_hint_, call_assign->span_, &then_stmts);
        then_stmts.push_back(std::make_shared<AssignStmt>(index_var, index_expr, call_assign->span_));
        index_offset_vars.push_back(index_var);
      }
      auto index_tuple = std::make_shared<MakeTuple>(index_offset_vars, call_assign->span_);
      auto read_call = OpRegistry::GetInstance().Create("tensor.read", {index_tensor_arg, index_tuple},
                                                        call_assign->span_);
      auto read_var =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_pbid_i32", int32_type, call_assign->span_);
      auto cast_expr = MakeCast(read_var, DataType::INDEX, call_assign->span_);
      auto cast_var =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_pbid", index_type, call_assign->span_);
      auto next_min =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_min_next", index_type, call_assign->span_);
      auto next_max =
          std::make_shared<Var>(in_arg->name_hint_ + "__window_max_next", index_type, call_assign->span_);

      then_stmts.push_back(std::make_shared<AssignStmt>(read_var, read_call, call_assign->span_));
      then_stmts.push_back(std::make_shared<AssignStmt>(cast_var, cast_expr, call_assign->span_));
      auto next_min_expr = FlattenGeneratedScalarExpr(MakeMin(scan_min_iter, cast_var, call_assign->span_),
                                                      in_arg->name_hint_, call_assign->span_, &then_stmts);
      auto next_max_expr = FlattenGeneratedScalarExpr(MakeMax(scan_max_iter, cast_var, call_assign->span_),
                                                      in_arg->name_hint_, call_assign->span_, &then_stmts);
      then_stmts.push_back(std::make_shared<AssignStmt>(next_min, next_min_expr, call_assign->span_));
      then_stmts.push_back(std::make_shared<AssignStmt>(next_max, next_max_expr, call_assign->span_));
      then_stmts.push_back(
          std::make_shared<YieldStmt>(std::vector<ExprPtr>{next_min, next_max}, call_assign->span_));
      auto then_body = std::make_shared<SeqStmts>(then_stmts, call_assign->span_);
      auto else_body =
          std::make_shared<YieldStmt>(std::vector<ExprPtr>{scan_min_iter, scan_max_iter}, call_assign->span_);

      StmtPtr loop_body;
      if (dynamic.guard_condition) {
        auto guard = analyzer->Simplify(transform_utils::Substitute(dynamic.guard_condition, scan_subst));
        auto if_stmt =
            std::make_shared<IfStmt>(guard, then_body, StmtPtr(else_body),
                                     std::vector<VarPtr>{scan_min_phi, scan_max_phi}, call_assign->span_);
        loop_body = std::make_shared<SeqStmts>(
            std::vector<StmtPtr>{if_stmt,
                                 std::make_shared<YieldStmt>(std::vector<ExprPtr>{scan_min_phi, scan_max_phi},
                                                             call_assign->span_)},
            call_assign->span_);
      } else {
        loop_body = then_body;
      }

      auto start = analyzer->Simplify(transform_utils::Substitute(dynamic.loop_start, callsite_subst));
      auto stop = analyzer->Simplify(transform_utils::Substitute(dynamic.loop_stop, callsite_subst));
      auto step = analyzer->Simplify(transform_utils::Substitute(dynamic.loop_step, callsite_subst));
      stmts->push_back(std::make_shared<AssignStmt>(
          scan_min_init, std::make_shared<ConstInt>(2147483647, DataType::INDEX, call_assign->span_),
          call_assign->span_));
      stmts->push_back(std::make_shared<AssignStmt>(
          scan_max_init, std::make_shared<ConstInt>(0, DataType::INDEX, call_assign->span_),
          call_assign->span_));
      stmts->push_back(std::make_shared<ForStmt>(
          scan_loop_var, start, stop, step, std::vector<IterArgPtr>{scan_min_iter, scan_max_iter}, loop_body,
          std::vector<VarPtr>{dynamic_min_var, dynamic_max_var}, call_assign->span_, ForKind::Sequential));

      DynamicWindowScanResult result;
      result.min_subst[dynamic.dynamic_index_value] = dynamic_min_var;
      result.max_subst[dynamic.dynamic_index_value] = dynamic_max_var;
      return result;
    }

    std::optional<FullOffsetScanResult> EmitFullOffsetScanPrelude(const CallsiteAccessCandidate& reader,
                                                                  arith::Analyzer* analyzer,
                                                                  std::vector<StmtPtr>* stmts) {
      if (!reader.dynamic_indexed_reader || !reader.input_info.dynamic_indexed_window.has_value() ||
          !reader.call || !reader.assign || !analyzer || !stmts) {
        return std::nullopt;
      }
      const auto& input = reader.input_info;
      const auto& dynamic = *input.dynamic_indexed_window;
      if (dynamic.index_tensor_param_index >= reader.call->args_.size() ||
          dynamic.dynamic_dim >= reader.shape.size() || dynamic.dynamic_dim >= reader.offsets.size() ||
          !dynamic.dynamic_index_value) {
        return std::nullopt;
      }
      if (dynamic.guard_condition &&
          CountVarRefsInExpr(dynamic.guard_condition, dynamic.dynamic_index_value) != 0) {
        return std::nullopt;
      }

      struct ScanLoopSpec {
        VarPtr original_var;
        ExprPtr start;
        ExprPtr stop;
        ExprPtr step;
      };

      std::vector<ScanLoopSpec> loops;
      loops.reserve(reader.enclosing_loops.size() + 1);
      for (const auto& loop : reader.enclosing_loops) {
        if (!loop.loop_var || !loop.start || !loop.stop || !loop.step) return std::nullopt;
        loops.push_back(ScanLoopSpec{loop.loop_var, loop.start, loop.stop, loop.step});
      }
      loops.push_back(
          ScanLoopSpec{dynamic.loop_var, dynamic.loop_start, dynamic.loop_stop, dynamic.loop_step});

      auto index_tensor_arg =
          MaterializeWindowParentExpr(reader.call->args_[dynamic.index_tensor_param_index]);
      auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
      auto int32_type = std::make_shared<ScalarType>(DataType::INT32);
      const auto& span = reader.assign->span_;
      const std::string base_name =
          reader.parent_root ? reader.parent_root->name_hint_ : std::string("carrier");

      auto scan_min_init = std::make_shared<Var>(base_name + "__carrier_min_init", index_type, span);
      auto scan_max_init = std::make_shared<Var>(base_name + "__carrier_max_init", index_type, span);
      auto scan_min_result = std::make_shared<Var>(base_name + "__carrier_min", index_type, span);
      auto scan_max_result = std::make_shared<Var>(base_name + "__carrier_max", index_type, span);

      auto apply_subst = [&](const ExprPtr& expr, const std::unordered_map<const Var*, ExprPtr>& loop_subst) {
        return SubstituteCallsiteAndLoops(expr, reader.callsite_subst, loop_subst);
      };

      std::function<StmtPtr(size_t, ExprPtr, ExprPtr, std::unordered_map<const Var*, ExprPtr>)> build_body =
          [&](size_t loop_index, ExprPtr current_min, ExprPtr current_max,
              std::unordered_map<const Var*, ExprPtr> loop_subst) -> StmtPtr {
        if (loop_index == loops.size()) {
          std::vector<ExprPtr> index_offsets;
          index_offsets.reserve(dynamic.index_offsets.size());
          for (const auto& index_offset : dynamic.index_offsets) {
            index_offsets.push_back(analyzer->Simplify(apply_subst(index_offset, loop_subst)));
          }

          std::vector<StmtPtr> then_stmts;
          std::vector<ExprPtr> index_offset_vars;
          index_offset_vars.reserve(index_offsets.size());
          for (size_t i = 0; i < index_offsets.size(); ++i) {
            auto index_var =
                std::make_shared<Var>(base_name + "__carrier_index_" + std::to_string(i), index_type, span);
            auto index_expr = FlattenGeneratedScalarExpr(index_offsets[i], base_name, span, &then_stmts);
            then_stmts.push_back(std::make_shared<AssignStmt>(index_var, index_expr, span));
            index_offset_vars.push_back(index_var);
          }
          auto read_call = OpRegistry::GetInstance().Create(
              "tensor.read", {index_tensor_arg, std::make_shared<MakeTuple>(index_offset_vars, span)}, span);
          auto read_var = std::make_shared<Var>(base_name + "__carrier_pbid_i32", int32_type, span);
          auto cast_expr = MakeCast(read_var, DataType::INDEX, span);
          auto cast_var = std::make_shared<Var>(base_name + "__carrier_pbid", index_type, span);
          auto next_min = std::make_shared<Var>(base_name + "__carrier_min_next", index_type, span);
          auto next_max = std::make_shared<Var>(base_name + "__carrier_max_next", index_type, span);

          auto offset_subst = loop_subst;
          offset_subst[dynamic.dynamic_index_value] = cast_var;
          auto full_offset =
              analyzer->Simplify(apply_subst(reader.offsets[dynamic.dynamic_dim], offset_subst));

          then_stmts.push_back(std::make_shared<AssignStmt>(read_var, read_call, span));
          then_stmts.push_back(std::make_shared<AssignStmt>(cast_var, cast_expr, span));
          auto next_min_expr = FlattenGeneratedScalarExpr(MakeMin(current_min, full_offset, span), base_name,
                                                          span, &then_stmts);
          auto next_max_expr = FlattenGeneratedScalarExpr(MakeMax(current_max, full_offset, span), base_name,
                                                          span, &then_stmts);
          then_stmts.push_back(std::make_shared<AssignStmt>(next_min, next_min_expr, span));
          then_stmts.push_back(std::make_shared<AssignStmt>(next_max, next_max_expr, span));
          then_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{next_min, next_max}, span));
          auto then_body = std::make_shared<SeqStmts>(then_stmts, span);
          if (!dynamic.guard_condition) return then_body;

          auto guard = analyzer->Simplify(apply_subst(dynamic.guard_condition, loop_subst));
          auto min_phi = std::make_shared<Var>(base_name + "__carrier_min_phi", index_type, span);
          auto max_phi = std::make_shared<Var>(base_name + "__carrier_max_phi", index_type, span);
          auto else_body = std::make_shared<YieldStmt>(std::vector<ExprPtr>{current_min, current_max}, span);
          auto if_stmt = std::make_shared<IfStmt>(guard, then_body, StmtPtr(else_body),
                                                  std::vector<VarPtr>{min_phi, max_phi}, span);
          return std::make_shared<SeqStmts>(
              std::vector<StmtPtr>{if_stmt,
                                   std::make_shared<YieldStmt>(std::vector<ExprPtr>{min_phi, max_phi}, span)},
              span);
        }

        const auto& loop = loops[loop_index];
        auto scan_loop_var = std::make_shared<Var>(base_name + "__carrier_scan_" + std::to_string(loop_index),
                                                   index_type, span);
        auto min_iter = std::make_shared<IterArg>(
            base_name + "__carrier_min_iter_" + std::to_string(loop_index), index_type, current_min, span);
        auto max_iter = std::make_shared<IterArg>(
            base_name + "__carrier_max_iter_" + std::to_string(loop_index), index_type, current_max, span);
        auto min_out = std::make_shared<Var>(base_name + "__carrier_min_out_" + std::to_string(loop_index),
                                             index_type, span);
        auto max_out = std::make_shared<Var>(base_name + "__carrier_max_out_" + std::to_string(loop_index),
                                             index_type, span);

        auto nested_subst = loop_subst;
        nested_subst[loop.original_var.get()] = scan_loop_var;
        auto body = build_body(loop_index + 1, min_iter, max_iter, nested_subst);
        auto start = analyzer->Simplify(apply_subst(loop.start, loop_subst));
        auto stop = analyzer->Simplify(apply_subst(loop.stop, loop_subst));
        auto step = analyzer->Simplify(apply_subst(loop.step, loop_subst));
        auto for_stmt = std::make_shared<ForStmt>(
            scan_loop_var, start, stop, step, std::vector<IterArgPtr>{min_iter, max_iter}, body,
            std::vector<VarPtr>{min_out, max_out}, span, ForKind::Sequential);
        return std::make_shared<SeqStmts>(
            std::vector<StmtPtr>{for_stmt,
                                 std::make_shared<YieldStmt>(std::vector<ExprPtr>{min_out, max_out}, span)},
            span);
      };

      stmts->push_back(std::make_shared<AssignStmt>(
          scan_min_init, std::make_shared<ConstInt>(2147483647, DataType::INDEX, span), span));
      stmts->push_back(std::make_shared<AssignStmt>(
          scan_max_init, std::make_shared<ConstInt>(0, DataType::INDEX, span), span));

      if (loops.empty()) return std::nullopt;
      const auto& first_loop = loops.front();
      auto scan_loop_var = std::make_shared<Var>(base_name + "__carrier_scan_0", index_type, span);
      auto min_iter =
          std::make_shared<IterArg>(base_name + "__carrier_min_iter_0", index_type, scan_min_init, span);
      auto max_iter =
          std::make_shared<IterArg>(base_name + "__carrier_max_iter_0", index_type, scan_max_init, span);
      std::unordered_map<const Var*, ExprPtr> loop_subst;
      loop_subst[first_loop.original_var.get()] = scan_loop_var;
      auto body = build_body(1, min_iter, max_iter, loop_subst);
      stmts->push_back(std::make_shared<ForStmt>(
          scan_loop_var, analyzer->Simplify(apply_subst(first_loop.start, {})),
          analyzer->Simplify(apply_subst(first_loop.stop, {})),
          analyzer->Simplify(apply_subst(first_loop.step, {})), std::vector<IterArgPtr>{min_iter, max_iter},
          body, std::vector<VarPtr>{scan_min_result, scan_max_result}, span, ForKind::Sequential));

      return FullOffsetScanResult{scan_min_result, scan_max_result};
    }

    std::optional<RewriteBundle> TryRewriteCall(
        const AssignStmtPtr& call_assign,
        const std::unordered_map<const Var*, size_t>& assemble_source_indices, size_t stmt_index) {
      // Submit (pl.submit inside pl.manual_scope) is a sibling call-like kind;
      // run the windowing analysis/rewrite on its augmented-Call view, then
      // rebuild as a Submit to preserve task-launch semantics + deps_
      // (.claude/rules/pass-submit-awareness.md). The per-callee analysis and
      // windowed clone are callee-body-driven (Analyze() over all functions),
      // so they exist regardless of the call-site kind.
      auto submit = As<Submit>(call_assign->value_);
      auto call = submit ? SubmitToCallView(submit) : As<Call>(call_assign->value_);
      if (!call) return std::nullopt;

      auto callee_name = GetCallFuncName(call);
      auto analysis_it = analyses_.find(callee_name);
      if (analysis_it == analyses_.end()) return std::nullopt;
      auto clone_it = cloned_funcs_.find(callee_name);
      if (clone_it == cloned_funcs_.end()) return std::nullopt;
      auto original_func = LookupFunction(callee_name);
      if (!original_func) return std::nullopt;

      const auto& analysis = analysis_it->second;
      auto cloned_func = clone_it->second;
      if (WindowExternalizeLogEnabled() && HasDynamicIndexedInputWindow(analysis.inputs)) {
        LOG_INFO << "[window-call] try func=" << callee_name << " outputs=" << analysis.outputs.size()
                 << " inputs=" << analysis.inputs.size() << " stmt_index=" << stmt_index;
      }

      if (analysis.outputs.empty() && analysis.inputs.empty()) return std::nullopt;
      if (submit && analysis.outputs.empty() && !HasDynamicIndexedInputWindow(analysis.inputs)) {
        return std::nullopt;
      }
      std::unordered_map<const Var*, ExprPtr> callsite_subst;
      for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
        callsite_subst[original_func->params_[i].get()] = call->args_[i];
      }
      if (analysis.outputs.empty() &&
          IsCallResultAssembledLater(call_assign->var_, assemble_source_indices, stmt_index)) {
        bool has_runtime_observable_reader_carrier = false;
        for (const auto& input : analysis.inputs) {
          if (!input.dynamic_indexed_window.has_value()) continue;
          if (WindowExternalizeLogEnabled()) {
            const Var* parent_root = input.in_param_index < call->args_.size()
                                         ? ResolveCarrierParentRoot(call->args_[input.in_param_index])
                                         : nullptr;
            LOG_INFO << "[window-call] assembled-later carrier check func=" << callee_name
                     << " input=" << input.in_param_index
                     << " root=" << (parent_root ? parent_root->name_hint_ : std::string("<none>"));
          }
          const auto& pieces = DensePieces(input);
          if (pieces.size() == 1 &&
              FindDynamicReaderCarrierUse(call, input.in_param_index, &input, &pieces.front()).has_value()) {
            has_runtime_observable_reader_carrier = true;
            break;
          }
        }
        if (!has_runtime_observable_reader_carrier) {
          if (WindowExternalizeLogEnabled()) {
            LOG_INFO << "[window-call] reject assembled-later without carrier func=" << callee_name;
          }
          return std::nullopt;
        }
      }
      if (!analysis.outputs.empty() && !ProveCallsiteDisjointness(call_assign, call, analysis) &&
          !CanUseRuntimeViewDisjointness(analysis)) {
        return std::nullopt;
      }
      if (HasUnwindowableSiblingOutputWriter(call, analysis)) return std::nullopt;
      if (HasDuplicateExternalizedOutputParent(call, analysis)) return std::nullopt;
      if (HasManualDepsToMultiPieceOutput(call, analysis)) return std::nullopt;
      if (window_rewrite_policy_ != WindowRewritePolicy::All && HasDynamicReaderCarrierRisk(call, analysis)) {
        return std::nullopt;
      }

      std::unordered_map<size_t, VarPtr> slices_by_in_index;
      std::unordered_map<size_t, std::vector<VarPtr>> slices_by_in_index_multi;
      std::unordered_map<size_t, std::vector<ExprPtr>> extra_args_by_in_index;
      std::unordered_map<size_t, std::vector<ExprPtr>> extra_args_by_out_index;
      std::unordered_map<size_t, ExprPtr> output_extent_arg_by_out_index;
      std::unordered_map<size_t, std::vector<SliceBundle>> slices_by_out_index;
      std::vector<StmtPtr> stmts;
      stmts.reserve((analysis.inputs.size() + analysis.outputs.size()) * 2 + 2);

      arith::Analyzer input_offset_analyzer;
      for (const auto& input : analysis.inputs) {
        if (input.in_param_index >= call->args_.size()) return std::nullopt;
        auto in_arg = AsVarLike(call->args_[input.in_param_index]);
        if (!in_arg) return std::nullopt;
        const auto& pieces = DensePieces(input);
        if (pieces.empty()) return std::nullopt;

        std::vector<VarPtr> input_slices;
        input_slices.reserve(pieces.size());
        for (size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
          const auto& piece = pieces[piece_index];
          auto carrier_use = input.dynamic_indexed_window.has_value()
                                 ? FindDynamicReaderCarrierUse(call, input.in_param_index, &input, &piece)
                                 : FindCarrierUse(call, input.in_param_index, piece);
          if (WindowExternalizeLogEnabled() && input.dynamic_indexed_window.has_value()) {
            LOG_INFO << "[window-call] dynamic input func=" << callee_name
                     << " input=" << input.in_param_index << " carrier=" << carrier_use.has_value()
                     << " current="
                     << (carrier_use.has_value() && carrier_use->carrier &&
                         carrier_use->carrier->current_var);
          }
          if (carrier_use.has_value()) {
            if (!carrier_use->carrier) return std::nullopt;
            if (input.dynamic_indexed_window.has_value()) {
              if (!carrier_use->carrier->dynamic_base_arg || !carrier_use->carrier->dynamic_extent_arg) {
                return std::nullopt;
              }
              if (!carrier_use->carrier->current_var) {
                return std::nullopt;
              }
              extra_args_by_in_index.emplace(input.in_param_index,
                                             std::vector<ExprPtr>{carrier_use->carrier->dynamic_base_arg,
                                                                  carrier_use->carrier->dynamic_extent_arg});
            }
            if (carrier_use->carrier->current_var) {
              input_slices.push_back(carrier_use->carrier->current_var);
            } else {
              return std::nullopt;
            }
            continue;
          }

          if (input.dynamic_indexed_window.has_value() &&
              window_rewrite_policy_ != WindowRewritePolicy::All) {
            return std::nullopt;
          }

          std::unordered_map<const Var*, ExprPtr> dynamic_min_subst;
          std::unordered_map<const Var*, ExprPtr> dynamic_max_subst;
          std::vector<ExprPtr> input_extra_args;

          if (input.dynamic_indexed_window.has_value()) {
            if (pieces.size() != 1) return std::nullopt;
            if (!CanProveDynamicWindowScanHasUpdate(*input.dynamic_indexed_window, callsite_subst)) {
              return std::nullopt;
            }
            auto scan = EmitDynamicWindowScan(input, piece, call, call_assign, in_arg, callsite_subst,
                                              &input_offset_analyzer, &stmts);
            if (!scan.has_value()) return std::nullopt;
            dynamic_min_subst = std::move(scan->min_subst);
            dynamic_max_subst = std::move(scan->max_subst);
          }

          std::vector<ExprPtr> shape_exprs;
          shape_exprs.reserve(piece.window_shape.size());
          for (size_t dim_i = 0; dim_i < piece.window_shape.size(); ++dim_i) {
            const auto& dim = piece.window_shape[dim_i];
            if (input.dynamic_indexed_window.has_value() &&
                dim_i == input.dynamic_indexed_window->dynamic_dim) {
              auto max_offset_subst = callsite_subst;
              for (const auto& [var, expr] : dynamic_max_subst) max_offset_subst[var] = expr;
              auto min_offset_subst = callsite_subst;
              for (const auto& [var, expr] : dynamic_min_subst) min_offset_subst[var] = expr;
              auto max_offset = input_offset_analyzer.Simplify(
                  transform_utils::Substitute(piece.callsite_offsets[dim_i], max_offset_subst));
              auto min_offset = input_offset_analyzer.Simplify(
                  transform_utils::Substitute(piece.callsite_offsets[dim_i], min_offset_subst));
              auto access_extent =
                  input_offset_analyzer.Simplify(transform_utils::Substitute(dim, callsite_subst));
              shape_exprs.push_back(input_offset_analyzer.Simplify(MakeSub(
                  MakeAdd(max_offset, access_extent, call_assign->span_), min_offset, call_assign->span_)));
            } else {
              shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
            }
          }
          std::vector<ExprPtr> offset_exprs;
          offset_exprs.reserve(piece.callsite_offsets.size());
          for (size_t offset_i = 0; offset_i < piece.callsite_offsets.size(); ++offset_i) {
            const auto& offset = piece.callsite_offsets[offset_i];
            auto offset_subst = callsite_subst;
            if (input.dynamic_indexed_window.has_value() &&
                offset_i == input.dynamic_indexed_window->dynamic_dim) {
              for (const auto& [var, expr] : dynamic_min_subst) offset_subst[var] = expr;
            }
            offset_exprs.push_back(
                input_offset_analyzer.Simplify(transform_utils::Substitute(offset, offset_subst)));
          }
          if (input.dynamic_indexed_window.has_value()) {
            const size_t dynamic_dim = input.dynamic_indexed_window->dynamic_dim;
            auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
            auto base_arg = std::make_shared<Var>(in_arg->name_hint_ + "__window_base_arg", index_type,
                                                  call_assign->span_);
            auto extent_arg = std::make_shared<Var>(in_arg->name_hint_ + "__window_extent_arg", index_type,
                                                    call_assign->span_);
            stmts.push_back(
                std::make_shared<AssignStmt>(base_arg, offset_exprs[dynamic_dim], call_assign->span_));
            stmts.push_back(
                std::make_shared<AssignStmt>(extent_arg, shape_exprs[dynamic_dim], call_assign->span_));
            offset_exprs[dynamic_dim] = base_arg;
            shape_exprs[dynamic_dim] = extent_arg;
            input_extra_args.push_back(base_arg);
            input_extra_args.push_back(extent_arg);
          }
          auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);
          auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

          ExprPtr parent_expr = MaterializeWindowParentExpr(call->args_[input.in_param_index]);
          auto slice_call = OpRegistry::GetInstance().Create(
              "tensor.slice", {parent_expr, shape_tuple, offset_tuple}, call_assign->span_);
          auto suffix =
              pieces.size() == 1 ? std::string("__window") : "__window_" + std::to_string(piece_index);
          auto slice_var =
              std::make_shared<Var>(in_arg->name_hint_ + suffix, slice_call->GetType(), in_arg->span_);
          stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_));
          input_slices.push_back(slice_var);
          if (!input_extra_args.empty()) {
            extra_args_by_in_index.emplace(input.in_param_index, std::move(input_extra_args));
          }
        }
        if (input_slices.size() == 1) slices_by_in_index.emplace(input.in_param_index, input_slices[0]);
        slices_by_in_index_multi.emplace(input.in_param_index, std::move(input_slices));
      }

      arith::Analyzer output_offset_analyzer;
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return std::nullopt;
        auto out_arg = AsVarLike(call->args_[output.out_param_index]);
        if (!out_arg) return std::nullopt;
        const auto& pieces = DensePieces(output);
        if (pieces.empty()) return std::nullopt;

        std::vector<SliceBundle> output_slices;
        output_slices.reserve(pieces.size());
        ExprPtr parent_expr = MaterializeWindowParentExpr(call->args_[output.out_param_index]);
        for (size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
          const auto& piece = pieces[piece_index];
          auto carrier_use = FindCarrierUse(call, output.out_param_index, piece);
          std::vector<ExprPtr> output_extra_args;
          std::vector<ExprPtr> shape_exprs;
          shape_exprs.reserve(piece.window_shape.size());
          for (const auto& dim : piece.window_shape) {
            shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
          }

          std::vector<ExprPtr> offset_exprs;
          offset_exprs.reserve(piece.callsite_offsets.size());
          for (const auto& offset : piece.callsite_offsets) {
            offset_exprs.push_back(
                output_offset_analyzer.Simplify(transform_utils::Substitute(offset, callsite_subst)));
          }
          if (carrier_use.has_value() && carrier_use->carrier && carrier_use->carrier->has_dynamic_reader) {
            if (piece_index != 0 || !HasFineAssemblePieces(output) ||
                carrier_use->carrier->dynamic_dim != 0) {
              return std::nullopt;
            }
            if (carrier_use->carrier->dynamic_base_arg && carrier_use->carrier->dynamic_extent_arg) {
              offset_exprs[0] = carrier_use->carrier->dynamic_base_arg;
              shape_exprs[0] = carrier_use->carrier->dynamic_extent_arg;
            } else {
              if (!carrier_use->carrier->reader_call || !carrier_use->carrier->reader_assign) {
                return std::nullopt;
              }
              auto reader_in_arg = AsVarLike(carrier_use->carrier->reader_call
                                                 ->args_[carrier_use->carrier->reader_input.in_param_index]);
              auto reader_func = LookupFunction(GetCallFuncName(carrier_use->carrier->reader_call));
              if (!reader_in_arg || !reader_func) return std::nullopt;
              std::unordered_map<const Var*, ExprPtr> reader_callsite_subst;
              for (size_t i = 0;
                   i < reader_func->params_.size() && i < carrier_use->carrier->reader_call->args_.size();
                   ++i) {
                reader_callsite_subst[reader_func->params_[i].get()] =
                    carrier_use->carrier->reader_call->args_[i];
              }
              if (!CanProveDynamicWindowScanHasUpdate(
                      *carrier_use->carrier->reader_input.dynamic_indexed_window, reader_callsite_subst)) {
                return std::nullopt;
              }
              auto scan = EmitDynamicWindowScan(carrier_use->carrier->reader_input,
                                                carrier_use->carrier->reader_piece,
                                                carrier_use->carrier->reader_call, call_assign, reader_in_arg,
                                                reader_callsite_subst, &output_offset_analyzer, &stmts);
              if (!scan.has_value()) return std::nullopt;

              auto max_offset_subst = reader_callsite_subst;
              for (const auto& [var, expr] : scan->max_subst) max_offset_subst[var] = expr;
              auto min_offset_subst = reader_callsite_subst;
              for (const auto& [var, expr] : scan->min_subst) min_offset_subst[var] = expr;
              const auto& reader_piece = carrier_use->carrier->reader_piece;
              auto reader_min = output_offset_analyzer.Simplify(
                  transform_utils::Substitute(reader_piece.callsite_offsets[0], min_offset_subst));
              auto reader_max = output_offset_analyzer.Simplify(
                  transform_utils::Substitute(reader_piece.callsite_offsets[0], max_offset_subst));
              auto reader_access_extent = output_offset_analyzer.Simplify(
                  transform_utils::Substitute(reader_piece.window_shape[0], reader_callsite_subst));
              auto reader_end = output_offset_analyzer.Simplify(
                  MakeAdd(reader_max, reader_access_extent, call_assign->span_));
              auto writer_end = output_offset_analyzer.Simplify(
                  MakeAdd(offset_exprs[0], shape_exprs[0], call_assign->span_));
              auto carrier_base =
                  output_offset_analyzer.Simplify(MakeMin(offset_exprs[0], reader_min, call_assign->span_));
              auto carrier_end =
                  output_offset_analyzer.Simplify(MakeMax(writer_end, reader_end, call_assign->span_));
              auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
              auto carrier_base_var = std::make_shared<Var>(out_arg->name_hint_ + "__window_carrier_base",
                                                            index_type, call_assign->span_);
              auto carrier_end_var = std::make_shared<Var>(out_arg->name_hint_ + "__window_carrier_end",
                                                           index_type, call_assign->span_);
              auto carrier_extent_var = std::make_shared<Var>(out_arg->name_hint_ + "__window_carrier_extent",
                                                              index_type, call_assign->span_);
              carrier_base =
                  FlattenGeneratedScalarExpr(carrier_base, out_arg->name_hint_, call_assign->span_, &stmts);
              carrier_end =
                  FlattenGeneratedScalarExpr(carrier_end, out_arg->name_hint_, call_assign->span_, &stmts);
              stmts.push_back(
                  std::make_shared<AssignStmt>(carrier_base_var, carrier_base, call_assign->span_));
              stmts.push_back(std::make_shared<AssignStmt>(carrier_end_var, carrier_end, call_assign->span_));
              stmts.push_back(
                  std::make_shared<AssignStmt>(carrier_extent_var,
                                               output_offset_analyzer.Simplify(MakeSub(
                                                   carrier_end_var, carrier_base_var, call_assign->span_)),
                                               call_assign->span_));
              offset_exprs[0] = carrier_base_var;
              shape_exprs[0] = carrier_extent_var;
            }
          }
          if (piece_index == 0 && HasFineAssemblePieces(output)) {
            if (shape_exprs.empty() || offset_exprs.empty()) return std::nullopt;
            auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
            VarPtr base_arg;
            VarPtr extent_arg;
            const bool use_existing_carrier_args =
                carrier_use.has_value() && carrier_use->carrier && carrier_use->carrier->has_dynamic_reader &&
                carrier_use->carrier->dynamic_base_arg && carrier_use->carrier->dynamic_extent_arg &&
                offset_exprs[0].get() == carrier_use->carrier->dynamic_base_arg.get() &&
                shape_exprs[0].get() == carrier_use->carrier->dynamic_extent_arg.get();
            if (use_existing_carrier_args) {
              base_arg = carrier_use->carrier->dynamic_base_arg;
              extent_arg = carrier_use->carrier->dynamic_extent_arg;
            } else {
              base_arg = std::make_shared<Var>(out_arg->name_hint_ + "__window_base_arg", index_type,
                                               call_assign->span_);
              extent_arg = std::make_shared<Var>(out_arg->name_hint_ + "__window_extent_arg", index_type,
                                                 call_assign->span_);
              stmts.push_back(std::make_shared<AssignStmt>(base_arg, offset_exprs[0], call_assign->span_));
              stmts.push_back(std::make_shared<AssignStmt>(extent_arg, shape_exprs[0], call_assign->span_));
            }
            offset_exprs[0] = base_arg;
            shape_exprs[0] = extent_arg;
            output_extra_args.push_back(base_arg);
            output_extra_args.push_back(extent_arg);
            output_extent_arg_by_out_index.emplace(output.out_param_index, extent_arg);
            if (!use_existing_carrier_args && carrier_use.has_value() && carrier_use->carrier &&
                carrier_use->carrier->has_dynamic_reader) {
              carrier_use->carrier->dynamic_base_arg = base_arg;
              carrier_use->carrier->dynamic_extent_arg = extent_arg;
            }
          }
          auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);
          auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

          auto slice_call = OpRegistry::GetInstance().Create(
              "tensor.slice", {parent_expr, shape_tuple, offset_tuple}, call_assign->span_);
          auto suffix =
              pieces.size() == 1 ? std::string("__window") : "__window_" + std::to_string(piece_index);
          auto slice_var =
              std::make_shared<Var>(out_arg->name_hint_ + suffix, slice_call->GetType(), out_arg->span_);
          stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_));
          if (carrier_use.has_value()) {
            if (!carrier_use->carrier || carrier_use->carrier->current_var) return std::nullopt;
            carrier_use->carrier->initial_var = slice_var;
            carrier_use->carrier->current_var = slice_var;
          }
          output_slices.push_back(SliceBundle{slice_var, parent_expr, offset_tuple});
          if (!output_extra_args.empty()) {
            extra_args_by_out_index.emplace(output.out_param_index, std::move(output_extra_args));
          }
        }
        slices_by_out_index.emplace(output.out_param_index, std::move(output_slices));
      }

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size());
      for (size_t i = 0; i < call->args_.size(); ++i) {
        auto input_slice_it = slices_by_in_index_multi.find(i);
        if (input_slice_it != slices_by_in_index_multi.end()) {
          auto extra_it = extra_args_by_in_index.find(i);
          if (extra_it != extra_args_by_in_index.end()) {
            for (const auto& arg : extra_it->second) new_args.push_back(arg);
          }
          for (const auto& slice : input_slice_it->second) new_args.push_back(slice);
          continue;
        }
        auto slice_it = slices_by_out_index.find(i);
        if (slice_it != slices_by_out_index.end()) {
          auto extra_it = extra_args_by_out_index.find(i);
          if (extra_it != extra_args_by_out_index.end()) {
            for (const auto& arg : extra_it->second) new_args.push_back(arg);
          }
          for (const auto& slice : slice_it->second) new_args.push_back(slice.slice_var);
        } else {
          new_args.push_back(VisitExpr(call->args_[i]));
        }
      }

      auto cloned_gvar = std::make_shared<GlobalVar>(cloned_func->name_);
      const bool is_submit_call = IsSubmitCall(call);
      std::vector<TypePtr> result_types = cloned_func->return_types_;
      std::unordered_map<const Var*, ExprPtr> cloned_param_callsite_subst;
      for (size_t i = 0; i < cloned_func->params_.size() && i < new_args.size(); ++i) {
        cloned_param_callsite_subst[cloned_func->params_[i].get()] = new_args[i];
      }
      auto dynamic_dim_it = output_dynamic_extent_dims_by_func_.find(callee_name);
      if (dynamic_dim_it != output_dynamic_extent_dims_by_func_.end()) {
        for (const auto& [out_param_index, extent_dim] : dynamic_dim_it->second) {
          auto extent_arg_it = output_extent_arg_by_out_index.find(out_param_index);
          if (extent_dim && extent_arg_it != output_extent_arg_by_out_index.end()) {
            cloned_param_callsite_subst[extent_dim.get()] = extent_arg_it->second;
          }
        }
      }
      for (auto& result_type : result_types) {
        result_type = SubstituteTypeExprs(result_type, cloned_param_callsite_subst);
      }
      std::unordered_map<size_t, std::vector<size_t>> piece_return_indices_by_out_param;
      size_t next_extra_return_index = original_func->return_types_.size();
      for (const auto& output : analysis.outputs) {
        const auto& pieces = DensePieces(output);
        if (pieces.empty()) return std::nullopt;
        std::vector<size_t> piece_return_indices;
        piece_return_indices.reserve(pieces.size());
        piece_return_indices.push_back(output.return_index);
        for (size_t piece_index = 1; piece_index < pieces.size(); ++piece_index) {
          piece_return_indices.push_back(next_extra_return_index++);
        }
        piece_return_indices_by_out_param.emplace(output.out_param_index, std::move(piece_return_indices));
      }
      if (next_extra_return_index != cloned_func->return_types_.size()) return std::nullopt;
      if (is_submit_call) {
        auto tuple_ty = As<TupleType>(call->GetType());
        if (!tuple_ty || tuple_ty->types_.size() != result_types.size() + 1) return std::nullopt;
        result_types.push_back(tuple_ty->types_.back());
      }
      auto finish_bundle = [&](RewriteBundle bundle) -> RewriteBundle {
        used_clone_names_.insert(callee_name);
        return bundle;
      };
      TypePtr new_return_type =
          result_types.size() == 1 ? result_types[0] : std::make_shared<TupleType>(result_types);

      auto new_attrs = RewriteCallAttrs(call, analysis, slices_by_out_index);
      ExprPtr new_call;
      if (submit) {
        // Preserve Submit-ness and deps_ (the canonical encoding); drop the
        // view's synthesised manual_dep_edges attr so deps aren't duplicated.
        // new_return_type already carries the trailing TASK_ID (is_submit_call).
        std::vector<std::pair<std::string, std::any>> submit_attrs;
        submit_attrs.reserve(new_attrs.size());
        for (const auto& [k, v] : new_attrs) {
          if (k != kAttrManualDepEdges) submit_attrs.emplace_back(k, v);
        }
        new_call = std::make_shared<Submit>(cloned_gvar, new_args, submit->deps_, submit->kwargs_,
                                            std::move(submit_attrs), new_return_type, submit->span_,
                                            submit->core_num_, submit->sync_start_);
      } else {
        new_call = std::make_shared<Call>(cloned_gvar, new_args, call->kwargs_, new_attrs, new_return_type,
                                          call->span_);
      }
      if (analysis.outputs.empty()) {
        if (WindowExternalizeLogEnabled() && HasDynamicIndexedInputWindow(analysis.inputs)) {
          LOG_INFO << "[window-call] rewrite input-only func=" << callee_name
                   << " new_args=" << new_args.size() << " stmts=" << stmts.size();
        }
        stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, new_call, call_assign->span_));
        RewriteBundle bundle;
        bundle.stmts = std::move(stmts);
        return finish_bundle(std::move(bundle));
      }
      auto tmp_result_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__windowed",
                                                  new_return_type, call_assign->var_->span_);
      stmts.push_back(std::make_shared<AssignStmt>(tmp_result_var, new_call, call_assign->span_));

      size_t total_output_pieces = 0;
      bool has_fine_assemble_output = false;
      for (const auto& output : analysis.outputs) {
        total_output_pieces += AssemblePieces(output).size();
        has_fine_assemble_output = has_fine_assemble_output || HasFineAssemblePieces(output);
      }
      if (!is_submit_call && analysis.outputs.size() == 1 && total_output_pieces == 1 &&
          result_types.size() == 1 && !has_fine_assemble_output) {
        const auto& output = analysis.outputs[0];
        auto carrier_use = FindCarrierUse(call, output.out_param_index, DensePieces(output).front());
        if (carrier_use.has_value()) {
          if (!carrier_use->carrier || !carrier_use->carrier->current_var) return std::nullopt;
          carrier_use->carrier->current_var = tmp_result_var;
        }
        const auto& slice_bundle = slices_by_out_index.at(output.out_param_index).front();
        auto assemble_call = OpRegistry::GetInstance().Create(
            "tensor.assemble", {slice_bundle.parent_expr, ExprPtr(tmp_result_var), slice_bundle.offset_tuple},
            call_assign->span_);
        stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, assemble_call, call_assign->span_));

        RewriteBundle bundle;
        bundle.stmts = std::move(stmts);
        if (auto parent_var = AsVarLike(slice_bundle.parent_expr)) {
          bundle.parent_substs.emplace_back(parent_var.get(), call_assign->var_);
        }
        return finish_bundle(std::move(bundle));
      }

      const size_t visible_result_count = original_func->return_types_.size() + (is_submit_call ? 1 : 0);
      std::vector<ExprPtr> assembled_result_exprs(visible_result_count);
      std::vector<StmtPtr> tail_stmts;
      tail_stmts.reserve(total_output_pieces * 3 + result_types.size() + 1);
      std::vector<std::pair<const Var*, ExprPtr>> bundle_parent_substs;

      std::unordered_map<size_t, VarPtr> tuple_items;
      for (const auto& output : analysis.outputs) {
        const auto& piece_return_indices = piece_return_indices_by_out_param.at(output.out_param_index);
        const auto& slice_bundles = slices_by_out_index.at(output.out_param_index);
        const auto& assemble_pieces = AssemblePieces(output);
        const bool uses_fine_assemble = HasFineAssemblePieces(output);
        if (piece_return_indices.size() != slice_bundles.size()) return std::nullopt;
        if (uses_fine_assemble && piece_return_indices.empty()) return std::nullopt;
        if (!uses_fine_assemble && piece_return_indices.size() != assemble_pieces.size()) return std::nullopt;

        ExprPtr current_parent_expr = slice_bundles.front().parent_expr;
        for (size_t piece_index = 0; piece_index < assemble_pieces.size(); ++piece_index) {
          const size_t piece_return_index =
              uses_fine_assemble ? piece_return_indices.front() : piece_return_indices[piece_index];
          ExprPtr item_expr;
          if (result_types.size() == 1) {
            item_expr = tmp_result_var;
          } else {
            auto item_it = tuple_items.find(piece_return_index);
            if (item_it == tuple_items.end()) {
              auto get_item = std::make_shared<TupleGetItemExpr>(
                  tmp_result_var, static_cast<int>(piece_return_index), call_assign->span_);
              auto item_var = std::make_shared<Var>(
                  call_assign->var_->name_hint_ + "__windowed_" + std::to_string(piece_return_index),
                  result_types[piece_return_index], call_assign->var_->span_);
              tail_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));
              item_it = tuple_items.emplace(piece_return_index, item_var).first;
            }
            item_expr = item_it->second;
          }
          if (piece_index == 0) {
            auto carrier_use = FindCarrierUse(call, output.out_param_index, DensePieces(output).front());
            if (carrier_use.has_value()) {
              auto item_var = AsVarLike(item_expr);
              if (!carrier_use->carrier || !carrier_use->carrier->current_var || !item_var) {
                return std::nullopt;
              }
              carrier_use->carrier->current_var = item_var;
            }
          }

          const SliceBundle& slice_bundle =
              uses_fine_assemble ? slice_bundles.front() : slice_bundles[piece_index];
          const auto& assemble_piece = assemble_pieces[piece_index];
          auto assemble_item_expr = item_expr;
          auto assemble_offset_tuple = slice_bundle.offset_tuple;
          if (uses_fine_assemble) {
            if (assemble_piece.callsite_offsets.size() != slice_bundle.offset_tuple->elements_.size()) {
              return std::nullopt;
            }
            arith::Analyzer fine_offset_analyzer;

            std::vector<ExprPtr> fine_shape_exprs;
            fine_shape_exprs.reserve(assemble_piece.window_shape.size());
            for (const auto& dim : assemble_piece.window_shape) {
              fine_shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
            }
            auto fine_shape_tuple = std::make_shared<MakeTuple>(fine_shape_exprs, call_assign->span_);

            std::vector<ExprPtr> fine_local_offsets;
            fine_local_offsets.reserve(assemble_piece.callsite_offsets.size());
            for (size_t dim = 0; dim < assemble_piece.callsite_offsets.size(); ++dim) {
              auto fine_offset = fine_offset_analyzer.Simplify(
                  transform_utils::Substitute(assemble_piece.callsite_offsets[dim], callsite_subst));
              auto carrier_offset = slice_bundle.offset_tuple->elements_[dim];
              fine_local_offsets.push_back(
                  fine_offset_analyzer.Simplify(MakeSub(fine_offset, carrier_offset, call_assign->span_)));
            }
            auto fine_local_offset_tuple =
                std::make_shared<MakeTuple>(fine_local_offsets, call_assign->span_);
            auto fine_slice_call = OpRegistry::GetInstance().Create(
                "tensor.slice", {item_expr, fine_shape_tuple, fine_local_offset_tuple}, call_assign->span_);
            auto fine_slice_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__fine_" +
                                                            std::to_string(output.return_index) + "_" +
                                                            std::to_string(piece_index),
                                                        fine_slice_call->GetType(), call_assign->var_->span_);
            tail_stmts.push_back(
                std::make_shared<AssignStmt>(fine_slice_var, fine_slice_call, call_assign->span_));
            assemble_item_expr = fine_slice_var;

            std::vector<ExprPtr> fine_global_offsets;
            fine_global_offsets.reserve(assemble_piece.callsite_offsets.size());
            for (const auto& offset : assemble_piece.callsite_offsets) {
              fine_global_offsets.push_back(
                  fine_offset_analyzer.Simplify(transform_utils::Substitute(offset, callsite_subst)));
            }
            assemble_offset_tuple = std::make_shared<MakeTuple>(fine_global_offsets, call_assign->span_);
          }
          auto assemble_call = OpRegistry::GetInstance().Create(
              "tensor.assemble", {current_parent_expr, assemble_item_expr, assemble_offset_tuple},
              call_assign->span_);
          auto parent_type = current_parent_expr->GetType();
          auto assembled_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__assembled_" +
                                                         std::to_string(output.return_index) + "_" +
                                                         std::to_string(piece_index),
                                                     parent_type, call_assign->var_->span_);
          tail_stmts.push_back(
              std::make_shared<AssignStmt>(assembled_var, assemble_call, call_assign->span_));
          current_parent_expr = assembled_var;
        }

        assembled_result_exprs[output.return_index] = current_parent_expr;
        if (auto parent_var = AsVarLike(slice_bundles.front().parent_expr)) {
          bundle_parent_substs.emplace_back(parent_var.get(), current_parent_expr);
        }
      }

      for (size_t i = 0; i < assembled_result_exprs.size(); ++i) {
        if (!assembled_result_exprs[i]) {
          const size_t source_index =
              is_submit_call && i == assembled_result_exprs.size() - 1 ? result_types.size() - 1 : i;
          if (result_types.size() == 1) {
            assembled_result_exprs[i] = tmp_result_var;
          } else {
            auto get_item = std::make_shared<TupleGetItemExpr>(tmp_result_var, static_cast<int>(source_index),
                                                               call_assign->span_);
            auto item_var =
                std::make_shared<Var>(call_assign->var_->name_hint_ + "__pass_" + std::to_string(i),
                                      result_types[source_index], call_assign->var_->span_);
            tail_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));
            assembled_result_exprs[i] = item_var;
          }
        }
      }

      if (visible_result_count == 1) {
        stmts.insert(stmts.end(), tail_stmts.begin(), tail_stmts.end());
        stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, assembled_result_exprs.front(),
                                                     call_assign->span_));
        RewriteBundle bundle;
        bundle.stmts = std::move(stmts);
        bundle.parent_substs = std::move(bundle_parent_substs);
        return finish_bundle(std::move(bundle));
      }

      tuple_result_subst_[call_assign->var_.get()] = std::move(assembled_result_exprs);
      stmts.insert(stmts.end(), tail_stmts.begin(), tail_stmts.end());
      auto rebuilt_tuple =
          std::make_shared<MakeTuple>(tuple_result_subst_.at(call_assign->var_.get()), call_assign->span_);
      stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, rebuilt_tuple, call_assign->span_));

      RewriteBundle bundle;
      bundle.stmts = std::move(stmts);
      bundle.parent_substs = std::move(bundle_parent_substs);
      return finish_bundle(std::move(bundle));
    }

    static bool IsSubmitCall(const CallPtr& call) {
      auto tuple_ty = As<TupleType>(call->GetType());
      if (!tuple_ty || tuple_ty->types_.empty()) return false;
      auto last = As<ScalarType>(tuple_ty->types_.back());
      return last != nullptr && last->dtype_ == DataType::TASK_ID;
    }

    std::vector<std::pair<std::string, std::any>> RewriteCallAttrs(
        const CallPtr& call, const CalleeRewriteAnalysis& analysis,
        const std::unordered_map<size_t, std::vector<SliceBundle>>& slices_by_out_index) const {
      std::vector<std::pair<std::string, std::any>> attrs;
      attrs.reserve(call->attrs_.size());
      for (const auto& [k, v] : call->attrs_) {
        if (k == kAttrArgDirections) continue;
        attrs.emplace_back(k, v);
      }
      for (auto& [k, v] : attrs) {
        if (k != kAttrManualDepEdges) continue;
        const auto* user_deps = std::any_cast<std::vector<VarPtr>>(&v);
        if (!user_deps) break;
        std::vector<VarPtr> rewritten;
        rewritten.reserve(user_deps->size());
        bool changed = false;
        for (const auto& dep : *user_deps) {
          bool replaced = false;
          for (const auto& output : analysis.outputs) {
            auto out_arg = AsVarLike(call->args_[output.out_param_index]);
            if (dep && out_arg && dep.get() == out_arg.get()) {
              const auto& slices = slices_by_out_index.at(output.out_param_index);
              if (slices.empty()) return attrs;
              rewritten.push_back(slices.front().slice_var);
              changed = true;
              replaced = true;
              break;
            }
          }
          if (!replaced) rewritten.push_back(dep);
        }
        if (changed) {
          return WithManualDepEdgesAttr(std::move(attrs), std::move(rewritten));
        }
        break;
      }
      return attrs;
    }

    bool HasManualDepsToMultiPieceOutput(const CallPtr& call, const CalleeRewriteAnalysis& analysis) const {
      for (const auto& [k, v] : call->attrs_) {
        if (k != kAttrManualDepEdges) continue;
        const auto* user_deps = std::any_cast<std::vector<VarPtr>>(&v);
        if (!user_deps) return false;
        for (const auto& dep : *user_deps) {
          for (const auto& output : analysis.outputs) {
            if (DensePieces(output).size() <= 1) continue;
            if (output.out_param_index >= call->args_.size()) return true;
            auto out_arg = AsVarLike(call->args_[output.out_param_index]);
            if (dep && out_arg && dep.get() == out_arg.get()) return true;
          }
        }
        return false;
      }
      return false;
    }

    const Var* ResolveOutputParentRoot(const CallPtr& call, size_t arg_index) const {
      if (!call || arg_index >= call->args_.size()) return nullptr;
      return ResolveCarrierParentRoot(call->args_[arg_index]);
    }

    const Var* ResolveOutputRootExpr(const ExprPtr& expr) const { return ResolveCarrierParentRoot(expr); }

    const Var* CanonicalizeCarrierParentRoot(const Var* root) const {
      if (!root) return nullptr;
      std::unordered_set<const Var*> seen;
      const Var* current = root;
      while (seen.insert(current).second) {
        auto it = sibling_output_alias_roots_.find(current);
        if (it == sibling_output_alias_roots_.end()) break;
        current = it->second;
        if (!current) return nullptr;
      }
      return current;
    }

    const Var* ResolveCarrierParentRoot(const ExprPtr& expr) const {
      auto parent = AsVarLike(ResolveLoopInitExpr(ResolveLoopReturnInitExpr(expr)));
      if (!parent) return nullptr;
      const Var* root = CanonicalizeCarrierParentRoot(parent.get());
      if (!root) return nullptr;
      return root;
    }

    void CollectSiblingOutputAliases(const std::vector<StmtPtr>& sibling_stmts) {
      std::unordered_map<const Var*, std::vector<const Var*>> sibling_tuple_output_roots;

      class SiblingWriterCollector : public IRVisitor {
       public:
        SiblingWriterCollector(OrchRewriter* rewriter,
                               std::unordered_map<const Var*, std::vector<const Var*>>* tuple_output_roots)
            : rewriter_(rewriter), tuple_output_roots_(tuple_output_roots) {}

       protected:
        void VisitStmt_(const AssignStmtPtr& op) override {
          if (!op) return;
          CallPtr call;
          if (auto submit = As<Submit>(op->value_)) {
            call = SubmitToCallView(submit);
          } else {
            call = As<Call>(op->value_);
          }

          if (auto tuple_get = As<TupleGetItemExpr>(op->value_)) {
            auto tuple_var = AsVarLike(tuple_get->tuple_);
            auto tuple_it =
                tuple_var ? tuple_output_roots_->find(tuple_var.get()) : tuple_output_roots_->end();
            if (tuple_it != tuple_output_roots_->end() && tuple_get->index_ >= 0 &&
                static_cast<size_t>(tuple_get->index_) < tuple_it->second.size()) {
              if (const Var* root = tuple_it->second[static_cast<size_t>(tuple_get->index_)]) {
                rewriter_->sibling_output_alias_roots_[op->var_.get()] = root;
              }
            }
          }

          if (!call || pypto::codegen::IsBuiltinOp(call->op_->name_)) {
            IRVisitor::VisitStmt_(op);
            return;
          }

          auto callee = rewriter_->LookupFunction(call->op_->name_);
          if (!callee) {
            IRVisitor::VisitStmt_(op);
            return;
          }

          const Var* single_output_root = nullptr;
          size_t output_root_count = 0;
          auto arg_directions = call->GetArgDirections();
          bool has_callsite_directions = arg_directions.size() == call->args_.size();
          for (size_t i = 0; i < call->args_.size() && i < callee->param_directions_.size(); ++i) {
            bool is_writer = false;
            if (has_callsite_directions) {
              is_writer = IsWriterArgDirection(arg_directions[i]);
            } else {
              is_writer = callee->param_directions_[i] == ParamDirection::Out ||
                          callee->param_directions_[i] == ParamDirection::InOut;
            }
            if (!is_writer) {
              continue;
            }
            if (const Var* parent_root = rewriter_->ResolveOutputParentRoot(call, i)) {
              if (!rewriter_->HasOutputWindowAnalysis(call->op_->name_, i)) {
                rewriter_->sibling_unwindowable_output_roots_.insert(parent_root);
              }
              single_output_root = parent_root;
              ++output_root_count;
            }
          }
          if (output_root_count == 1 && AsTensorTypeLike(op->var_->GetType())) {
            rewriter_->sibling_output_alias_roots_[op->var_.get()] = single_output_root;
          }
          if (output_root_count > 0 && As<TupleType>(op->var_->GetType())) {
            std::vector<const Var*> tuple_roots(callee->return_types_.size(), nullptr);
            for (const auto& mapping : rewriter_->GetOutParamReturnMappings(callee, /*include_inout=*/true)) {
              if (mapping.return_index >= tuple_roots.size() || mapping.param_index >= call->args_.size()) {
                continue;
              }
              tuple_roots[mapping.return_index] =
                  rewriter_->ResolveOutputParentRoot(call, mapping.param_index);
            }
            (*tuple_output_roots_)[op->var_.get()] = std::move(tuple_roots);
          }

          IRVisitor::VisitStmt_(op);
        }

        void VisitStmt_(const ForStmtPtr& op) override {
          {
            ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&rewriter_->loop_iter_init_subst_,
                                                                op->iter_args_);
            IRVisitor::VisitStmt_(op);
          }
          rewriter_->RecordLoopReturnInitAliases(op);
        }

        void VisitStmt_(const WhileStmtPtr& op) override {
          {
            ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&rewriter_->loop_iter_init_subst_,
                                                                op->iter_args_);
            IRVisitor::VisitStmt_(op);
          }
          rewriter_->RecordLoopReturnInitAliases(op);
        }

        void VisitStmt_(const IfStmtPtr& op) override { IRVisitor::VisitStmt_(op); }

       private:
        OrchRewriter* rewriter_;
        std::unordered_map<const Var*, std::vector<const Var*>>* tuple_output_roots_;
      };

      SiblingWriterCollector collector(this, &sibling_tuple_output_roots);
      for (const auto& sibling_stmt : sibling_stmts) {
        collector.VisitStmt(sibling_stmt);
      }
    }

    void AddCallsiteAccessCandidates(const AssignStmtPtr& assign, size_t stmt_index,
                                     const std::vector<LoopContext>& enclosing_loops,
                                     const std::unordered_map<const Var*, ExprPtr>& scalar_subst,
                                     std::vector<CallsiteAccessCandidate>* candidates) {
      if (!assign || !candidates) return;
      CallPtr call;
      if (auto submit = As<Submit>(assign->value_)) {
        call = SubmitToCallView(submit);
      } else {
        call = As<Call>(assign->value_);
      }
      if (!call || pypto::codegen::IsBuiltinOp(call->op_->name_)) return;
      auto analysis_it = analyses_.find(call->op_->name_);
      if (analysis_it == analyses_.end()) return;
      auto original_func = LookupFunction(call->op_->name_);
      std::unordered_map<const Var*, ExprPtr> resolved_callsite_subst;
      if (original_func) {
        for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
          resolved_callsite_subst[original_func->params_[i].get()] =
              arith::Analyzer().Simplify(transform_utils::Substitute(call->args_[i], scalar_subst));
        }
      }

      const auto& analysis = analysis_it->second;
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) continue;
        auto root = ResolveOutputRootExpr(call->args_[output.out_param_index]);
        if (!root) continue;
        CallsiteAccessCandidate candidate;
        candidate.stmt_index = stmt_index;
        candidate.assign = assign;
        candidate.call = call;
        candidate.param_index = output.out_param_index;
        candidate.direction = ParamDirection::InOut;
        candidate.parent_root = root;
        candidate.parent_expr = call->args_[output.out_param_index];
        candidate.output_writer = true;
        candidate.output_has_fine_assemble = HasFineAssemblePieces(output);
        candidate.enclosing_loops = enclosing_loops;
        candidate.callsite_subst = resolved_callsite_subst;
        if (!output.callsite_offsets.empty()) candidate.offsets = output.callsite_offsets;
        if (!output.window_shape.empty()) candidate.shape = output.window_shape;
        candidates->push_back(std::move(candidate));
      }

      for (const auto& input : analysis_it->second.inputs) {
        if (input.in_param_index >= call->args_.size()) continue;
        if (const Var* root = ResolveOutputRootExpr(call->args_[input.in_param_index])) {
          CallsiteAccessCandidate candidate;
          candidate.stmt_index = stmt_index;
          candidate.assign = assign;
          candidate.call = call;
          candidate.param_index = input.in_param_index;
          candidate.direction = ParamDirection::In;
          candidate.parent_root = root;
          candidate.parent_expr = call->args_[input.in_param_index];
          candidate.dynamic_indexed_reader = input.dynamic_indexed_window.has_value();
          candidate.input_info = input;
          candidate.enclosing_loops = enclosing_loops;
          candidate.callsite_subst = resolved_callsite_subst;
          if (!input.callsite_offsets.empty()) candidate.offsets = input.callsite_offsets;
          if (!input.window_shape.empty()) candidate.shape = input.window_shape;
          candidates->push_back(std::move(candidate));
        }
      }
    }

    void CollectCallsiteAccessCandidates(const StmtPtr& stmt, size_t stmt_index,
                                         const std::vector<LoopContext>& enclosing_loops,
                                         const std::unordered_map<const Var*, ExprPtr>& scalar_subst,
                                         std::vector<CallsiteAccessCandidate>* candidates) {
      if (!stmt || !candidates) return;
      if (auto assign = As<AssignStmt>(stmt)) {
        AddCallsiteAccessCandidates(assign, stmt_index, enclosing_loops, scalar_subst, candidates);
        return;
      }
      if (auto seq = As<SeqStmts>(stmt)) {
        auto scoped_scalar_subst = scalar_subst;
        for (const auto& child : seq->stmts_) {
          CollectCallsiteAccessCandidates(child, stmt_index, enclosing_loops, scoped_scalar_subst,
                                          candidates);
          auto child_assign = As<AssignStmt>(child);
          if (child_assign && As<ScalarType>(child_assign->var_->GetType())) {
            auto substituted = arith::Analyzer().Simplify(
                transform_utils::Substitute(child_assign->value_, scoped_scalar_subst));
            if (IsSafeInlineScalarSubstitution(substituted)) {
              scoped_scalar_subst[child_assign->var_.get()] = substituted;
            }
          }
        }
        return;
      }
      if (auto loop = As<ForStmt>(stmt)) {
        ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&loop_iter_init_subst_, loop->iter_args_);
        auto nested_loops = enclosing_loops;
        nested_loops.push_back(LoopContext{loop, loop->loop_var_, loop->start_, loop->stop_, loop->step_});
        CollectCallsiteAccessCandidates(loop->body_, stmt_index, nested_loops, scalar_subst, candidates);
        return;
      }
      if (auto while_stmt = As<WhileStmt>(stmt)) {
        ScopedLoopIterInitSubst scoped_loop_iter_init_subst(&loop_iter_init_subst_, while_stmt->iter_args_);
        CollectCallsiteAccessCandidates(while_stmt->body_, stmt_index, enclosing_loops, scalar_subst,
                                        candidates);
        return;
      }
      if (As<IfStmt>(stmt)) {
        // A carrier current produced in one branch is not guaranteed to dominate
        // the other branch or the post-if continuation.  Until this pass has a
        // real dominance/phi proof for carrier currents, do not build
        // sequence-level carrier classes from branch-local accesses.
        return;
      }
    }

    SequenceCarrierPlan BuildSequenceCarrierPlan(const std::vector<StmtPtr>& sibling_stmts) {
      SequenceCarrierPlan plan;
      std::vector<CallsiteAccessCandidate> candidates;
      candidates.reserve(sibling_stmts.size());

      for (size_t stmt_index = 0; stmt_index < sibling_stmts.size(); ++stmt_index) {
        CollectCallsiteAccessCandidates(sibling_stmts[stmt_index], stmt_index, {}, scalar_defs_, &candidates);
      }

      std::unordered_map<const Var*, std::vector<const CallsiteAccessCandidate*>> candidates_by_root;
      for (const auto& candidate : candidates) {
        if (candidate.parent_root) candidates_by_root[candidate.parent_root].push_back(&candidate);
      }

      for (const auto& [root, root_candidates] : candidates_by_root) {
        if (!root) continue;

        const CallsiteAccessCandidate* writer = nullptr;
        const CallsiteAccessCandidate* reader = nullptr;
        const CallsiteAccessCandidate* dynamic_reader = nullptr;
        bool unsupported = false;
        for (const auto* candidate : root_candidates) {
          if (!candidate) continue;
          if (candidate->output_writer) {
            if (writer) {
              unsupported = true;
              break;
            }
            writer = candidate;
          } else if (candidate->dynamic_indexed_reader) {
            if (dynamic_reader) {
              unsupported = true;
              break;
            }
            dynamic_reader = candidate;
          } else {
            if (reader) {
              unsupported = true;
              break;
            }
            reader = candidate;
          }
        }
        if (dynamic_reader) {
          if (window_rewrite_policy_ == WindowRewritePolicy::All) {
            continue;
          }
          plan.dynamic_reader_roots.insert(root);
          size_t dynamic_dim = SIZE_MAX;
          bool can_make_dynamic_carrier = !unsupported && writer && writer->output_has_fine_assemble &&
                                          writer->stmt_index < dynamic_reader->stmt_index &&
                                          !writer->shape.empty() && !writer->offsets.empty() &&
                                          !dynamic_reader->shape.empty() &&
                                          !dynamic_reader->offsets.empty() &&
                                          dynamic_reader->input_info.dynamic_indexed_window.has_value();
          if (can_make_dynamic_carrier) {
            dynamic_dim = dynamic_reader->input_info.dynamic_indexed_window->dynamic_dim;
            can_make_dynamic_carrier = dynamic_dim == 0 &&
                                       writer->shape.size() == dynamic_reader->shape.size() &&
                                       writer->offsets.size() == dynamic_reader->offsets.size() &&
                                       dynamic_dim < writer->shape.size();
            for (size_t dim = 0; can_make_dynamic_carrier && dim < writer->shape.size(); ++dim) {
              if (dim == dynamic_dim) continue;
              can_make_dynamic_carrier = AreExprsEqual(writer->shape[dim], dynamic_reader->shape[dim]) &&
                                         AreExprsEqual(writer->offsets[dim], dynamic_reader->offsets[dim]);
            }
          }
          if (can_make_dynamic_carrier) {
            arith::Analyzer carrier_analyzer;
            auto writer_offsets = SubstituteSingletonLoopStarts(
                SubstituteExprVector(writer->offsets, writer->callsite_subst), writer->enclosing_loops);
            auto writer_shape = SubstituteSingletonLoopStarts(
                SubstituteExprVector(writer->shape, writer->callsite_subst), writer->enclosing_loops);
            auto reader_scan =
                writer_offsets.has_value() && writer_shape.has_value()
                    ? EmitFullOffsetScanPrelude(*dynamic_reader, &carrier_analyzer, &plan.prelude_stmts)
                    : std::nullopt;
            if (!writer_offsets.has_value() || !writer_shape.has_value() || !reader_scan.has_value()) {
              plan.fallback_parent_roots.insert(root);
              continue;
            }
            auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
            auto carrier_base_var = std::make_shared<Var>(
                root->name_hint_ + "__carrier_base", index_type,
                dynamic_reader->assign ? dynamic_reader->assign->span_ : Span::unknown());
            auto carrier_extent_var = std::make_shared<Var>(
                root->name_hint_ + "__carrier_extent", index_type,
                dynamic_reader->assign ? dynamic_reader->assign->span_ : Span::unknown());
            auto writer_end = carrier_analyzer.Simplify(
                MakeAdd((*writer_offsets)[0], (*writer_shape)[0], carrier_base_var->span_));
            auto reader_access_extent = carrier_analyzer.Simplify(SubstituteCallsiteAndLoops(
                dynamic_reader->shape[dynamic_dim], dynamic_reader->callsite_subst, {}));
            auto reader_end = carrier_analyzer.Simplify(
                MakeAdd(reader_scan->max_offset, reader_access_extent, carrier_base_var->span_));
            auto carrier_base = carrier_analyzer.Simplify(
                MakeMin((*writer_offsets)[0], reader_scan->min_offset, carrier_base_var->span_));
            auto carrier_end =
                carrier_analyzer.Simplify(MakeMax(writer_end, reader_end, carrier_base_var->span_));
            auto carrier_end_var = std::make_shared<Var>(
                root->name_hint_ + "__carrier_end", index_type,
                dynamic_reader->assign ? dynamic_reader->assign->span_ : Span::unknown());
            carrier_base = FlattenGeneratedScalarExpr(
                carrier_base, root->name_hint_,
                dynamic_reader->assign ? dynamic_reader->assign->span_ : Span::unknown(),
                &plan.prelude_stmts);
            carrier_end = FlattenGeneratedScalarExpr(
                carrier_end, root->name_hint_,
                dynamic_reader->assign ? dynamic_reader->assign->span_ : Span::unknown(),
                &plan.prelude_stmts);
            plan.prelude_stmts.push_back(
                std::make_shared<AssignStmt>(carrier_base_var, carrier_base, carrier_base_var->span_));
            plan.prelude_stmts.push_back(
                std::make_shared<AssignStmt>(carrier_end_var, carrier_end, carrier_end_var->span_));
            plan.prelude_stmts.push_back(std::make_shared<AssignStmt>(
                carrier_extent_var,
                carrier_analyzer.Simplify(
                    MakeSub(carrier_end_var, carrier_base_var, carrier_extent_var->span_)),
                carrier_extent_var->span_));

            RuntimeObservableCarrier carrier;
            carrier.parent_root = root;
            carrier.parent_expr = writer->parent_expr;
            carrier.offsets = *writer_offsets;
            carrier.shape = *writer_shape;
            carrier.offsets[dynamic_dim] = carrier_base_var;
            carrier.shape[dynamic_dim] = carrier_extent_var;
            carrier.has_dynamic_reader = true;
            carrier.dynamic_dim = dynamic_reader->input_info.dynamic_indexed_window->dynamic_dim;
            carrier.reader_input = dynamic_reader->input_info;
            carrier.reader_piece = MakeDensePiece(dynamic_reader->shape, dynamic_reader->offsets, {});
            carrier.reader_call = dynamic_reader->call;
            carrier.reader_assign = dynamic_reader->assign;
            carrier.dynamic_base_arg = carrier_base_var;
            carrier.dynamic_extent_arg = carrier_extent_var;
            plan.carriers_by_parent.emplace(root, std::move(carrier));
            if (WindowExternalizeLogEnabled()) {
              LOG_INFO << "[window-carrier] create dynamic carrier root=" << root->name_hint_
                       << " writer_shape=" << ExprVectorDebugString(writer->shape)
                       << " reader_shape=" << ExprVectorDebugString(dynamic_reader->shape);
            }
          } else {
            plan.fallback_parent_roots.insert(root);
          }
          continue;
        }

        if (plan.fallback_parent_roots.count(root)) continue;
        if (unsupported || !writer || !reader) continue;
        if (writer->stmt_index >= reader->stmt_index) continue;
        if (writer->shape.empty() || writer->offsets.empty()) continue;
        if (!AreExprVectorsEqual(writer->shape, reader->shape) ||
            !AreExprVectorsEqual(writer->offsets, reader->offsets)) {
          if (WindowExternalizeLogEnabled()) {
            LOG_INFO << "[window-carrier] reject static carrier: writer shape="
                     << ExprVectorDebugString(writer->shape)
                     << " offsets=" << ExprVectorDebugString(writer->offsets)
                     << " reader shape=" << ExprVectorDebugString(reader->shape)
                     << " offsets=" << ExprVectorDebugString(reader->offsets);
          }
          continue;
        }

        RuntimeObservableCarrier carrier;
        carrier.parent_root = root;
        carrier.parent_expr = writer->parent_expr;
        carrier.offsets = writer->offsets;
        carrier.shape = writer->shape;
        plan.carriers_by_parent.emplace(root, std::move(carrier));
        if (WindowExternalizeLogEnabled()) {
          LOG_INFO << "[window-carrier] create static carrier root=" << root->name_hint_
                   << " shape=" << ExprVectorDebugString(writer->shape)
                   << " offsets=" << ExprVectorDebugString(writer->offsets);
        }
      }
      return plan;
    }

    std::optional<CarrierUse> FindCarrierUse(const CallPtr& call, size_t arg_index,
                                             const DenseRegionPiece& piece) {
      if (!call || arg_index >= call->args_.size()) return std::nullopt;
      const Var* parent_root = ResolveOutputRootExpr(call->args_[arg_index]);
      if (!parent_root) return std::nullopt;
      auto carrier_it = sequence_carrier_plan_.carriers_by_parent.find(parent_root);
      if (carrier_it == sequence_carrier_plan_.carriers_by_parent.end()) return std::nullopt;
      auto& carrier = carrier_it->second;
      if (carrier.has_dynamic_reader) {
        if (carrier.dynamic_dim >= piece.window_shape.size() ||
            carrier.dynamic_dim >= piece.callsite_offsets.size() ||
            carrier.shape.size() != piece.window_shape.size() ||
            carrier.offsets.size() != piece.callsite_offsets.size()) {
          return std::nullopt;
        }
        for (size_t dim = 0; dim < piece.window_shape.size(); ++dim) {
          if (dim == carrier.dynamic_dim) continue;
          if (!AreExprsEqual(carrier.shape[dim], piece.window_shape[dim]) ||
              !AreExprsEqual(carrier.offsets[dim], piece.callsite_offsets[dim])) {
            return std::nullopt;
          }
        }
        return CarrierUse{parent_root, &carrier};
      }
      if (!AreExprVectorsEqual(carrier.shape, piece.window_shape) ||
          !AreExprVectorsEqual(carrier.offsets, piece.callsite_offsets)) {
        return std::nullopt;
      }
      return CarrierUse{parent_root, &carrier};
    }

    std::optional<CarrierUse> FindDynamicReaderCarrierUse(const CallPtr& call, size_t arg_index,
                                                          const InputRewriteInfo* input = nullptr,
                                                          const DenseRegionPiece* piece = nullptr) {
      if (!call || arg_index >= call->args_.size()) return std::nullopt;
      const Var* parent_root = ResolveCarrierParentRoot(call->args_[arg_index]);
      if (!parent_root) return std::nullopt;
      auto carrier_it = sequence_carrier_plan_.carriers_by_parent.find(parent_root);
      if (carrier_it == sequence_carrier_plan_.carriers_by_parent.end()) return std::nullopt;
      auto& carrier = carrier_it->second;
      if (!carrier.has_dynamic_reader || carrier.dynamic_dim == SIZE_MAX ||
          carrier.reader_input.in_param_index != arg_index) {
        return std::nullopt;
      }
      if (input) {
        if (!input->dynamic_indexed_window.has_value() ||
            input->dynamic_indexed_window->dynamic_dim != carrier.dynamic_dim) {
          return std::nullopt;
        }
      }
      if (piece) {
        if (piece->window_shape.size() != carrier.reader_piece.window_shape.size() ||
            piece->callsite_offsets.size() != carrier.reader_piece.callsite_offsets.size()) {
          return std::nullopt;
        }
        for (size_t dim = 0; dim < piece->window_shape.size(); ++dim) {
          if (dim == carrier.dynamic_dim) continue;
          if (!AreExprsEqual(piece->window_shape[dim], carrier.reader_piece.window_shape[dim])) {
            return std::nullopt;
          }
        }
        for (size_t dim = 0; dim < piece->callsite_offsets.size(); ++dim) {
          if (dim == carrier.dynamic_dim) continue;
          if (!AreExprsEqual(piece->callsite_offsets[dim], carrier.reader_piece.callsite_offsets[dim])) {
            return std::nullopt;
          }
        }
      }
      return CarrierUse{parent_root, &carrier};
    }

    static bool IsWriterArgDirection(ArgDirection direction) {
      return direction == ArgDirection::Output || direction == ArgDirection::OutputExisting ||
             direction == ArgDirection::InOut;
    }

    bool HasOutputWindowAnalysis(const std::string& callee_name, size_t out_param_index) const {
      auto analysis_it = analyses_.find(callee_name);
      if (analysis_it == analyses_.end()) return false;
      const auto& outputs = analysis_it->second.outputs;
      return std::any_of(outputs.begin(), outputs.end(), [out_param_index](const OutputRewriteInfo& output) {
        return output.out_param_index == out_param_index;
      });
    }

    bool HasUnwindowableSiblingOutputWriter(const CallPtr& call,
                                            const CalleeRewriteAnalysis& analysis) const {
      for (const auto& output : analysis.outputs) {
        const Var* parent_root = ResolveOutputParentRoot(call, output.out_param_index);
        if (!parent_root) return true;
        if (sibling_unwindowable_output_roots_.count(parent_root)) return true;
        if (sequence_carrier_plan_.fallback_parent_roots.count(parent_root)) return true;
      }
      return false;
    }

    bool HasDuplicateExternalizedOutputParent(const CallPtr& call,
                                              const CalleeRewriteAnalysis& analysis) const {
      std::unordered_set<const Var*> seen_roots;
      for (const auto& output : analysis.outputs) {
        const Var* parent_root = ResolveOutputParentRoot(call, output.out_param_index);
        if (!parent_root) return true;
        if (!seen_roots.insert(parent_root).second) return true;
      }
      return false;
    }

    bool ProveCallsiteDisjointness(const AssignStmtPtr& call_assign, const CallPtr& call,
                                   const CalleeRewriteAnalysis& analysis) const {
      if (while_depth_ > 0) return false;
      std::vector<LoopDisjointnessCandidate> candidate_loops;
      candidate_loops.reserve(sequential_loops_.size());
      for (size_t i = 0; i < sequential_loops_.size(); ++i) {
        const auto& loop = sequential_loops_[i];
        if (!loop) continue;
        const auto* local_allocs = i < loop_local_allocs_.size() ? &loop_local_allocs_[i] : nullptr;
        candidate_loops.push_back(LoopDisjointnessCandidate{loop, local_allocs});
      }
      if (candidate_loops.empty()) return true;

      auto original_func = LookupFunction(call->op_->name_);
      if (!original_func) return false;

      std::unordered_map<const Var*, ExprPtr> callsite_subst;
      for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
        callsite_subst[original_func->params_[i].get()] = call->args_[i];
      }

      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= original_func->params_.size()) return false;
        if (!ProveOutputDisjoint(candidate_loops, output,
                                 original_func->params_[output.out_param_index].get(), callsite_subst)) {
          return false;
        }
      }
      return true;
    }

    bool ProveOutputDisjoint(const std::vector<LoopDisjointnessCandidate>& loops,
                             const OutputRewriteInfo& output, const Var* output_param,
                             const std::unordered_map<const Var*, ExprPtr>& callsite_subst) const {
      std::unordered_set<size_t> varying_dims_used;
      for (const auto& candidate : loops) {
        const auto role =
            ClassifyOutputLoopRole(candidate, output, output_param, callsite_subst, &varying_dims_used);
        if (role == LoopRegionRole::Unknown) {
          return false;
        }
      }
      return true;
    }

    LoopRegionRole ClassifyOutputLoopRole(const LoopDisjointnessCandidate& candidate,
                                          const OutputRewriteInfo& output, const Var* output_param,
                                          const std::unordered_map<const Var*, ExprPtr>& callsite_subst,
                                          std::unordered_set<size_t>* varying_dims_used) const {
      auto loop = candidate.loop;
      if (!loop) return LoopRegionRole::Unknown;
      if (IsOutputParentLocalToLoop(output_param, callsite_subst, candidate.loop_local_allocs)) {
        return LoopRegionRole::Reduction;
      }

      auto trip_count = GetStaticTripCount(loop);
      if (trip_count.has_value() && *trip_count <= 1) {
        return LoopRegionRole::Reduction;
      }

      std::optional<size_t> varying_dim;
      for (size_t i = 0; i < output.callsite_offsets.size(); ++i) {
        auto rewritten = transform_utils::Substitute(output.callsite_offsets[i], callsite_subst);
        rewritten = transform_utils::Substitute(rewritten, scalar_defs_);
        auto affine = ParseAffineInLoop(rewritten, loop->loop_var_.get());
        if (!affine.has_value()) return LoopRegionRole::Unknown;
        if (affine->coeff == 0) {
          continue;
        }

        auto extent_ci = As<ConstInt>(output.window_shape[i]);
        auto loop_step = GetConstIntValue(loop->step_);
        if (!extent_ci || !loop_step.has_value()) return LoopRegionRole::Unknown;
        if (varying_dim.has_value()) return LoopRegionRole::Unknown;
        if (varying_dims_used && varying_dims_used->count(i)) return LoopRegionRole::Unknown;
        if (std::abs(affine->coeff * *loop_step) < extent_ci->value_) return LoopRegionRole::Unknown;
        varying_dim = i;
      }
      if (!varying_dim.has_value()) {
        return LoopRegionRole::Reduction;
      }
      if (varying_dims_used) varying_dims_used->insert(*varying_dim);
      return LoopRegionRole::Partition;
    }

    bool IsOutputParentLocalToLoop(const Var* output_param,
                                   const std::unordered_map<const Var*, ExprPtr>& callsite_subst,
                                   const std::unordered_set<const Var*>* loop_local_allocs) const {
      if (!loop_local_allocs || loop_local_allocs->empty()) return false;

      auto subst_it = callsite_subst.find(output_param);
      if (subst_it == callsite_subst.end()) return false;

      auto parent_expr = ResolveLoopInitExpr(subst_it->second);
      auto parent_var = AsVarLike(parent_expr);
      if (parent_var) {
        const Var* root = parent_var.get();
        std::unordered_set<const Var*> seen;
        while (seen.insert(root).second) {
          auto alias_it = sibling_output_alias_roots_.find(root);
          if (alias_it == sibling_output_alias_roots_.end()) break;
          root = alias_it->second;
        }
        return loop_local_allocs->count(root);
      }
      return parent_var && loop_local_allocs->count(parent_var.get());
    }

    ExprPtr ResolveLoopInitExpr(const ExprPtr& expr) const {
      ExprPtr current = expr;
      std::unordered_set<const Var*> seen;
      while (auto var = AsVarLike(current)) {
        if (!seen.insert(var.get()).second) break;
        auto it = loop_iter_init_subst_.find(var.get());
        if (it == loop_iter_init_subst_.end()) break;
        current = it->second;
      }
      return current;
    }

    ExprPtr ResolveLoopReturnInitExpr(const ExprPtr& expr) const {
      ExprPtr current = expr;
      std::unordered_set<const Var*> seen;
      while (auto var = AsVarLike(current)) {
        if (!seen.insert(var.get()).second) break;
        auto it = loop_return_init_subst_.find(var.get());
        if (it == loop_return_init_subst_.end()) break;
        current = it->second;
      }
      return current;
    }

    ExprPtr MaterializeWindowParentExpr(const ExprPtr& expr) {
      return VisitExpr(ResolveLoopReturnInitExpr(expr));
    }

    bool HasDynamicReaderCarrierRisk(const CallPtr& call, const CalleeRewriteAnalysis& analysis) const {
      for (const auto& input : analysis.inputs) {
        if (!input.dynamic_indexed_window.has_value()) continue;
        if (input.in_param_index >= call->args_.size()) return true;
        const Var* parent_root = ResolveCarrierParentRoot(call->args_[input.in_param_index]);
        if (WindowExternalizeLogEnabled()) {
          LOG_INFO << "[window-call] dynamic risk func=" << GetCallFuncName(call)
                   << " input=" << input.in_param_index
                   << " root=" << (parent_root ? parent_root->name_hint_ : std::string("<none>"))
                   << " has_carrier="
                   << (parent_root && sequence_carrier_plan_.carriers_by_parent.count(parent_root) != 0);
        }
        if (!parent_root) return true;
        if (sequence_carrier_plan_.carriers_by_parent.count(parent_root) != 0) continue;
        // Dynamic indexed reader windows must share an exact runtime-observable
        // TensorKey carrier with every may-overlap writer.  A standalone slice
        // from either an older loop-init parent or a freshly assembled full
        // parent creates a different key from the producer window, so keep the
        // baseline until sequence-level carrier lowering is available.
        return true;
      }
      return false;
    }

    FunctionPtr LookupFunction(const std::string& name) const {
      auto it = function_lookup_.find(name);
      if (it == function_lookup_.end()) return nullptr;
      return it->second;
    }

    ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
      auto tuple_var = AsVarLike(op->tuple_);
      if (tuple_var) {
        auto subst_it = tuple_result_subst_.find(tuple_var.get());
        if (subst_it != tuple_result_subst_.end() && op->index_ >= 0 &&
            static_cast<size_t>(op->index_) < subst_it->second.size()) {
          return VisitExpr(subst_it->second[static_cast<size_t>(op->index_)]);
        }
      }
      return IRMutator::VisitExpr_(op);
    }

    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = window_parent_subst_.find(op.get());
      if (it != window_parent_subst_.end()) return VisitExpr(it->second);
      return IRMutator::VisitExpr_(op);
    }

    ProgramPtr program_;
    const AnalysisMap& analyses_;
    const std::unordered_map<std::string, FunctionPtr>& cloned_funcs_;
    const std::unordered_map<std::string, FunctionPtr>& function_lookup_;
    const std::unordered_map<std::string, std::unordered_map<size_t, VarPtr>>&
        output_dynamic_extent_dims_by_func_;
    WindowRewritePolicy window_rewrite_policy_ = WindowRewritePolicy::Auto;
    std::unordered_set<std::string> used_clone_names_;
    std::vector<ForStmtPtr> sequential_loops_;
    std::vector<LoopContext> loop_context_;
    std::vector<std::unordered_set<const Var*>> loop_local_allocs_;
    std::unordered_map<const Var*, ExprPtr> loop_iter_init_subst_;
    std::unordered_map<const Var*, ExprPtr> loop_return_init_subst_;
    std::unordered_map<const Var*, ExprPtr> scalar_defs_;
    std::unordered_map<const Var*, std::vector<ExprPtr>> tuple_result_subst_;
    std::unordered_map<const Var*, ExprPtr> window_parent_subst_;
    std::unordered_map<const Var*, const Var*> sibling_output_alias_roots_;
    std::unordered_set<const Var*> sibling_unwindowable_output_roots_;
    SequenceCarrierPlan sequence_carrier_plan_;
    std::unordered_map<std::string, std::vector<OutParamReturnMapping>> out_param_return_mappings_cache_;
    size_t generated_scalar_temp_counter_ = 0;
    int while_depth_ = 0;
  };

  struct FinalStoreInfo {
    size_t return_index;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> offsets;
  };

  struct AggregateWindowInfo {
    size_t return_index;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> base_offsets;
    std::vector<ExprPtr> local_offsets;
    size_t iter_arg_index;
  };

  struct InputWindowUse {
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> offsets;
    size_t param_refs_in_stmt = 0;
  };

  struct InputParamUseSummary {
    size_t total_refs = 0;
    bool unsupported_ref = false;
    std::vector<InputWindowUse> uses;
  };

  static bool CanMaterializeWindowParamType(const std::shared_ptr<const TensorType>& tensor_type,
                                            const std::vector<ExprPtr>& window_shape) {
    if (!tensor_type) return false;
    if (tensor_type->tensor_view_.has_value()) {
      if (tensor_type->tensor_view_->stride.empty() &&
          tensor_type->tensor_view_->layout == TensorLayout::NZ) {
        return false;
      }
      return true;
    }
    auto parent_strides =
        tensor_view_semantics::BuildLogicalStridesFromLayout(tensor_type->shape_, TensorLayout::ND);
    return parent_strides.size() == window_shape.size();
  }

  static std::optional<size_t> FindReturnIndexForOutParam(const FunctionPtr& func, size_t out_param_index) {
    if (!func || out_param_index >= func->params_.size()) return std::nullopt;
    auto body_stmts = FlattenToStmts(func->body_);
    ReturnStmtPtr ret_stmt;
    for (const auto& stmt : body_stmts) {
      if (auto ret = As<ReturnStmt>(stmt)) {
        ret_stmt = ret;
        break;
      }
    }
    if (!ret_stmt) return std::nullopt;

    const auto* out_param = func->params_[out_param_index].get();
    for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
      auto ret_var = AsVarLike(ret_stmt->value_[ret_i]);
      if (!ret_var) continue;
      if (ret_var.get() == out_param) return ret_i;
    }
    return std::nullopt;
  }

  static std::optional<int64_t> GetConstIntValue(const ExprPtr& expr) {
    auto ci = As<ConstInt>(expr);
    if (!ci) return std::nullopt;
    return ci->value_;
  }

  static std::optional<int64_t> GetStaticTripCount(const ForStmtPtr& loop) {
    if (!loop) return std::nullopt;
    auto start = GetConstIntValue(loop->start_);
    auto stop = GetConstIntValue(loop->stop_);
    auto step = GetConstIntValue(loop->step_);
    if (!start.has_value() || !stop.has_value() || !step.has_value() || *step == 0) return std::nullopt;
    if ((*step > 0 && *stop <= *start) || (*step < 0 && *stop >= *start)) return int64_t{0};
    int64_t distance = *stop - *start;
    int64_t step_abs = *step > 0 ? *step : -*step;
    int64_t distance_abs = distance > 0 ? distance : -distance;
    return (distance_abs + step_abs - 1) / step_abs;
  }

  static std::optional<int64_t> GetKnownPositiveTripCount(const ForStmtPtr& loop) {
    auto static_trip_count = GetStaticTripCount(loop);
    if (static_trip_count.has_value()) return static_trip_count;
    if (!loop) return std::nullopt;
    auto step = GetConstIntValue(loop->step_);
    if (!step.has_value() || *step == 0) return std::nullopt;

    auto distance_expr = *step > 0 ? MakeSub(loop->stop_, loop->start_, loop->span_)
                                   : MakeSub(loop->start_, loop->stop_, loop->span_);
    distance_expr = arith::Analyzer().Simplify(distance_expr);
    auto distance = As<ConstInt>(distance_expr);
    int64_t distance_value = 0;
    if (distance) {
      distance_value = distance->value_;
    } else {
      auto linear_distance = *step > 0 ? ConstantDiffIfSameLinearBase(loop->stop_, loop->start_)
                                       : ConstantDiffIfSameLinearBase(loop->start_, loop->stop_);
      if (!linear_distance.has_value()) return std::nullopt;
      distance_value = *linear_distance;
    }
    if (distance_value <= 0) return int64_t{0};
    int64_t step_abs = *step > 0 ? *step : -*step;
    return (distance_value + step_abs - 1) / step_abs;
  }

  static std::optional<ExprPtr> SimplifyWithLoopBound(const ExprPtr& expr, const VarPtr& loop_var,
                                                      int64_t value) {
    if (!expr) return std::nullopt;
    arith::Analyzer analyzer;
    analyzer.Bind(loop_var, value, value + 1);
    return analyzer.Simplify(expr);
  }

  static std::optional<ExprPtr> SimplifyWithLoopValue(const ExprPtr& expr, const VarPtr& loop_var,
                                                      const ExprPtr& value) {
    if (!expr || !value) return std::nullopt;
    arith::Analyzer analyzer;
    analyzer.Bind(loop_var, value);
    return analyzer.Simplify(expr);
  }

  static std::optional<ExprPtr> GetLoopValueAtTrip(const ForStmtPtr& loop, int64_t trip_index) {
    if (!loop || trip_index < 0) return std::nullopt;
    auto step = GetConstIntValue(loop->step_);
    if (!step.has_value()) return std::nullopt;
    int64_t delta = trip_index * *step;
    if (delta == 0) return loop->start_;
    auto delta_expr = std::make_shared<ConstInt>(delta, DataType::INDEX, loop->span_);
    return arith::Analyzer().Simplify(MakeAdd(loop->start_, delta_expr, loop->span_));
  }

  static std::optional<OrderedLoopOffsets> GetOrderedLoopOffsets(const ExprPtr& expr, const ForStmtPtr& loop,
                                                                 const ExprPtr& first_loop_value,
                                                                 const ExprPtr& last_loop_value) {
    if (!expr || !loop || !first_loop_value || !last_loop_value) return std::nullopt;
    auto first_offset = SimplifyWithLoopValue(expr, loop->loop_var_, first_loop_value);
    auto last_offset = SimplifyWithLoopValue(expr, loop->loop_var_, last_loop_value);
    if (!first_offset.has_value() || !last_offset.has_value()) return std::nullopt;

    auto affine = ParseAffineInLoop(expr, loop->loop_var_.get());
    auto loop_step = GetConstIntValue(loop->step_);
    if (!affine.has_value() || !loop_step.has_value()) return std::nullopt;
    if (affine->coeff * *loop_step >= 0) {
      return OrderedLoopOffsets{*first_offset, *last_offset};
    }
    return OrderedLoopOffsets{*last_offset, *first_offset};
  }

  static std::optional<ExprPtr> ExpandLoopLocalExpr(
      const ExprPtr& expr, const std::unordered_map<const Var*, ExprPtr>& scalar_defs) {
    if (!expr) return std::nullopt;
    return transform_utils::Substitute(expr, scalar_defs);
  }

  static std::optional<InputWindowUse> MatchDirectTensorWindowAccess(const AssignStmtPtr& assign,
                                                                     const Var* param) {
    if (!assign || !param) return std::nullopt;
    auto call = As<Call>(assign->value_);
    if (!call || call->args_.empty()) return std::nullopt;

    std::vector<ExprPtr> window_shape;
    MakeTuplePtr offsets;
    if (call->op_->name_ == "tile.load" && call->args_.size() >= 3) {
      auto parent = AsVarLike(call->args_[0]);
      offsets = As<MakeTuple>(call->args_[1]);
      auto tile_type = As<TileType>(call->GetType());
      if (!parent || parent.get() != param || !offsets || !tile_type) return std::nullopt;
      auto read_shape = As<MakeTuple>(call->args_[2]);
      if (!read_shape) return std::nullopt;
      if (call->GetKwarg<bool>("transpose", false)) {
        if (read_shape->elements_.size() != 2) return std::nullopt;
        window_shape = read_shape->elements_;
      } else {
        window_shape = tile_type->shape_;
        if (!AreExprVectorsEqual(window_shape, read_shape->elements_)) return std::nullopt;
      }
    } else if (call->op_->name_ == "tensor.slice" && call->args_.size() >= 3) {
      auto parent = AsVarLike(call->args_[0]);
      offsets = As<MakeTuple>(call->args_[2]);
      auto tensor_type = As<TensorType>(call->GetType());
      if (!parent || parent.get() != param || !offsets || !tensor_type) return std::nullopt;
      window_shape = tensor_type->shape_;
    } else {
      return std::nullopt;
    }

    if (window_shape.size() != offsets->elements_.size()) return std::nullopt;
    size_t refs = CountVarRefsInStmt(assign, param);
    if (refs == 0) return std::nullopt;
    return InputWindowUse{std::move(window_shape), offsets->elements_, refs};
  }

  static bool IsProvenSameRegionInOutAccess(const FunctionPtr& func, size_t out_param_index,
                                            const AssignStmtPtr& store_assign,
                                            const std::vector<ExprPtr>& store_shape,
                                            const std::vector<ExprPtr>& store_offsets,
                                            const ReturnStmtPtr& ret_stmt = nullptr) {
    if (!func || out_param_index >= func->params_.size() ||
        out_param_index >= func->param_directions_.size() ||
        func->param_directions_[out_param_index] != ParamDirection::InOut) {
      return false;
    }
    const auto* param = func->params_[out_param_index].get();
    size_t total_refs = CountVarRefsInStmt(func->body_, param);
    size_t matched_refs = store_assign ? CountVarRefsInStmt(store_assign, param) : 0;
    if (total_refs == 0 || matched_refs == 0 || matched_refs > total_refs) return false;

    auto body_stmts = FlattenToStmts(func->body_);
    for (const auto& stmt : body_stmts) {
      auto assign = As<AssignStmt>(stmt);
      size_t refs = CountVarRefsInStmt(stmt, param);
      if (refs == 0) continue;
      if (assign && store_assign && assign.get() == store_assign.get()) continue;
      if (ret_stmt && stmt.get() == ret_stmt.get()) {
        matched_refs += refs;
        continue;
      }

      auto use = MatchDirectTensorWindowAccess(assign, param);
      if (!use.has_value()) return false;
      if (!AreExprVectorsEqual(use->window_shape, store_shape) ||
          !AreExprVectorsEqual(use->offsets, store_offsets)) {
        return false;
      }
      matched_refs += use->param_refs_in_stmt;
    }
    return matched_refs == total_refs;
  }

  static bool IsProvenSideEffectStoreWithDirectReturn(const FunctionPtr& func, size_t out_param_index,
                                                      const AssignStmtPtr& store_assign,
                                                      const std::vector<ExprPtr>& store_shape,
                                                      const std::vector<ExprPtr>& store_offsets,
                                                      const ReturnStmtPtr& ret_stmt) {
    if (!func || !store_assign || !ret_stmt || out_param_index >= func->params_.size() ||
        out_param_index >= func->param_directions_.size()) {
      return false;
    }
    const auto direction = func->param_directions_[out_param_index];
    const auto* param = func->params_[out_param_index].get();
    size_t total_refs = CountVarRefsInStmt(func->body_, param);
    size_t matched_refs = CountVarRefsInStmt(store_assign, param) + CountVarRefsInStmt(ret_stmt, param);
    if (total_refs == 0 || matched_refs == 0 || matched_refs > total_refs) return false;

    auto body_stmts = FlattenToStmts(func->body_);
    for (const auto& stmt : body_stmts) {
      size_t refs = CountVarRefsInStmt(stmt, param);
      if (refs == 0 || stmt.get() == store_assign.get() || stmt.get() == ret_stmt.get()) continue;
      if (direction != ParamDirection::InOut) return false;
      auto use = MatchDirectTensorWindowAccess(As<AssignStmt>(stmt), param);
      if (!use.has_value()) return false;
      if (!AreExprVectorsEqual(use->window_shape, store_shape) ||
          !AreExprVectorsEqual(use->offsets, store_offsets)) {
        return false;
      }
      matched_refs += use->param_refs_in_stmt;
    }
    return matched_refs == total_refs;
  }

  static std::optional<FinalStoreInfo> AnalyzeFinalStore(const FunctionPtr& func, size_t out_param_index) {
    if (!func || out_param_index >= func->params_.size()) return std::nullopt;

    auto body_stmts = FlattenToStmts(func->body_);
    std::unordered_map<const Var*, AssignStmtPtr> var_defs;
    for (const auto& stmt : body_stmts) {
      if (auto assign = As<AssignStmt>(stmt)) var_defs[assign->var_.get()] = assign;
    }

    ReturnStmtPtr ret_stmt;
    for (const auto& stmt : body_stmts) {
      if (auto ret = As<ReturnStmt>(stmt)) {
        ret_stmt = ret;
        break;
      }
    }
    if (!ret_stmt) return std::nullopt;

    size_t total_out_refs = CountVarRefsInStmt(func->body_, func->params_[out_param_index].get());
    std::optional<FinalStoreInfo> result;
    size_t matched_refs = 0;
    for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
      auto ret_var = AsVarLike(ret_stmt->value_[ret_i]);
      if (!ret_var) continue;
      auto def_it = var_defs.find(ret_var.get());
      if (def_it == var_defs.end()) continue;
      auto store_call = As<Call>(def_it->second->value_);
      if (!store_call || store_call->op_->name_ != "tile.store" || store_call->args_.size() < 3) continue;

      auto out_target = AsVarLike(store_call->args_[2]);
      if (!out_target || out_target.get() != func->params_[out_param_index].get()) continue;
      auto offset_tuple = As<MakeTuple>(store_call->args_[1]);
      auto tile_type = As<TileType>(store_call->args_[0]->GetType());
      if (!offset_tuple || !tile_type) return std::nullopt;

      matched_refs = CountVarRefsInStmt(def_it->second, func->params_[out_param_index].get());
      if (total_out_refs != matched_refs &&
          !IsProvenSameRegionInOutAccess(func, out_param_index, def_it->second, tile_type->shape_,
                                         offset_tuple->elements_)) {
        return std::nullopt;
      }

      result = FinalStoreInfo{ret_i, tile_type->shape_, offset_tuple->elements_};
      break;
    }
    if (result.has_value()) return result;

    auto direct_return_index = FindReturnIndexForOutParam(func, out_param_index);
    if (!direct_return_index.has_value()) return std::nullopt;
    for (const auto& stmt : body_stmts) {
      auto assign = As<AssignStmt>(stmt);
      if (!assign) continue;
      auto store_call = As<Call>(assign->value_);
      if (!store_call || store_call->op_->name_ != "tile.store" || store_call->args_.size() < 3) continue;
      auto out_target = AsVarLike(store_call->args_[2]);
      if (!out_target || out_target.get() != func->params_[out_param_index].get()) continue;
      auto offset_tuple = As<MakeTuple>(store_call->args_[1]);
      auto tile_type = As<TileType>(store_call->args_[0]->GetType());
      if (!offset_tuple || !tile_type) return std::nullopt;
      if (!IsProvenSideEffectStoreWithDirectReturn(func, out_param_index, assign, tile_type->shape_,
                                                   offset_tuple->elements_, ret_stmt)) {
        continue;
      }
      result = FinalStoreInfo{*direct_return_index, tile_type->shape_, offset_tuple->elements_};
      break;
    }
    return result;
  }

  static bool HasOnlyFullShapeZeroOffsetReturnOutputs(const FunctionPtr& func,
                                                      const std::vector<size_t>& out_indices) {
    if (!func) return false;
    for (const auto& out_index : out_indices) {
      auto out_tensor_type = As<TensorType>(func->params_[out_index]->GetType());
      if (!out_tensor_type) return false;
      auto info = AnalyzeFinalStore(func, out_index);
      if (!info.has_value()) return false;
      if (!AreExprVectorsEqual(info->window_shape, out_tensor_type->shape_) ||
          !IsAllZeroOffsets(info->offsets)) {
        return false;
      }
    }
    return true;
  }

  static std::optional<InputWindowUse> MatchInputWindowUse(const AssignStmtPtr& assign, const Var* param,
                                                           size_t refs_in_stmt) {
    if (!assign || !param) return std::nullopt;
    auto call = As<Call>(assign->value_);
    if (!call || call->args_.empty()) return std::nullopt;

    std::vector<ExprPtr> window_shape;
    MakeTuplePtr offsets;
    if (call->op_->name_ == "tile.load" && call->args_.size() >= 3) {
      auto parent = AsVarLike(call->args_[0]);
      offsets = As<MakeTuple>(call->args_[1]);
      auto tile_type = As<TileType>(call->GetType());
      if (!parent || parent.get() != param || !offsets || !tile_type) return std::nullopt;
      auto read_shape = As<MakeTuple>(call->args_[2]);
      if (!read_shape) return std::nullopt;
      if (call->GetKwarg<bool>("transpose", false)) {
        if (read_shape->elements_.size() != 2) return std::nullopt;
        window_shape = read_shape->elements_;
      } else {
        window_shape = tile_type->shape_;
        if (!AreExprVectorsEqual(window_shape, read_shape->elements_)) return std::nullopt;
      }
    } else if (call->op_->name_ == "tensor.slice" && call->args_.size() >= 3) {
      auto parent = AsVarLike(call->args_[0]);
      offsets = As<MakeTuple>(call->args_[2]);
      auto tensor_type = As<TensorType>(call->GetType());
      if (!parent || parent.get() != param || !offsets || !tensor_type) return std::nullopt;
      // The slice op is itself the complete access to the parent region. Any
      // later use must reference the slice value, so total_refs accounting below
      // rejects extra reads from the original full input.
      window_shape = tensor_type->shape_;
    } else {
      return std::nullopt;
    }

    if (window_shape.size() != offsets->elements_.size()) return std::nullopt;
    if (refs_in_stmt == 0) return std::nullopt;
    return InputWindowUse{std::move(window_shape), offsets->elements_, refs_in_stmt};
  }

  static std::optional<InputWindowUse> MatchExpandedInputWindowUse(
      const AssignStmtPtr& assign, const Var* param, size_t refs_in_stmt,
      const std::unordered_map<const Var*, ExprPtr>& subst) {
    auto use = MatchInputWindowUse(assign, param, refs_in_stmt);
    if (!use.has_value()) return std::nullopt;

    arith::Analyzer analyzer;
    for (auto& dim : use->window_shape) {
      dim = analyzer.Simplify(transform_utils::Substitute(dim, subst));
    }
    for (auto& offset : use->offsets) {
      offset = analyzer.Simplify(transform_utils::Substitute(offset, subst));
    }
    return use;
  }

  static bool IsSafeDynamicIndexedWindowExprs(const FunctionPtr& func, const InputWindowUse& use,
                                              size_t dynamic_dim, const Var* dynamic_value) {
    if (!func || !dynamic_value || dynamic_dim >= use.offsets.size() ||
        use.offsets.size() != use.window_shape.size()) {
      return false;
    }

    auto allowed_params = CollectAllowedVars(func->params_);
    auto allowed_dynamic_offset = allowed_params;
    allowed_dynamic_offset.insert(dynamic_value);

    if (!ExprsReferenceOnlyVarsIn(use.window_shape, allowed_params)) {
      return false;
    }
    for (size_t dim = 0; dim < use.offsets.size(); ++dim) {
      const auto& offset = use.offsets[dim];
      if (dim == dynamic_dim) {
        if (!ExprReferencesOnlyVarsIn(offset, allowed_dynamic_offset)) return false;
        if (CountVarRefsInExpr(offset, dynamic_value) != 1) return false;
      } else {
        if (!ExprReferencesOnlyVarsIn(offset, allowed_params)) return false;
        if (CountVarRefsInExpr(offset, dynamic_value) != 0) return false;
      }
    }
    return true;
  }

  struct DynamicIndexSource {
    size_t index_tensor_param_index = SIZE_MAX;
    std::vector<ExprPtr> index_offsets;
  };

  static std::optional<DynamicIndexSource> MatchTensorReadIndexSource(
      const ExprPtr& expr, const FunctionPtr& func,
      const std::unordered_map<const Var*, size_t>& tensor_param_indices,
      const std::unordered_map<const Var*, DynamicIndexSource>& scalar_index_sources,
      const std::unordered_map<const Var*, ExprPtr>& scalar_defs) {
    if (!expr || !func) return std::nullopt;

    if (auto call = As<Call>(expr)) {
      if (call->op_->name_ != "tensor.read" || call->args_.size() != 2) return std::nullopt;
      auto parent = AsVarLike(call->args_[0]);
      auto indices = As<MakeTuple>(call->args_[1]);
      if (!parent || !indices) return std::nullopt;
      auto param_it = tensor_param_indices.find(parent.get());
      if (param_it == tensor_param_indices.end()) return std::nullopt;

      DynamicIndexSource source;
      source.index_tensor_param_index = param_it->second;
      arith::Analyzer analyzer;
      source.index_offsets.reserve(indices->elements_.size());
      for (const auto& index : indices->elements_) {
        source.index_offsets.push_back(analyzer.Simplify(transform_utils::Substitute(index, scalar_defs)));
      }
      return source;
    }

    if (auto cast = As<Cast>(expr)) {
      auto operand = AsVarLike(cast->operand_);
      if (!operand)
        return MatchTensorReadIndexSource(cast->operand_, func, tensor_param_indices, scalar_index_sources,
                                          scalar_defs);
      auto source_it = scalar_index_sources.find(operand.get());
      if (source_it == scalar_index_sources.end()) return std::nullopt;
      return source_it->second;
    }

    auto var = AsVarLike(expr);
    if (var) {
      auto source_it = scalar_index_sources.find(var.get());
      if (source_it != scalar_index_sources.end()) return source_it->second;
    }
    return std::nullopt;
  }

  static std::optional<InputRewriteInfo> AnalyzeDynamicIndexedInputWindowInLoop(const FunctionPtr& func,
                                                                                size_t param_index,
                                                                                const ForStmtPtr& loop) {
    if (!func || param_index >= func->params_.size() || !loop) return std::nullopt;
    auto tensor_type = As<TensorType>(func->params_[param_index]->GetType());
    if (!tensor_type) return std::nullopt;
    auto trip_count = GetKnownPositiveTripCount(loop);
    if (!trip_count.has_value() || *trip_count <= 0 || *trip_count > 256) return std::nullopt;

    std::unordered_map<const Var*, size_t> tensor_param_indices;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (i >= func->param_directions_.size()) continue;
      if (func->param_directions_[i] != ParamDirection::In) continue;
      if (!As<TensorType>(func->params_[i]->GetType())) continue;
      tensor_param_indices.emplace(func->params_[i].get(), i);
    }

    struct CandidateUse {
      InputWindowUse use;
      ExprPtr guard_condition;
    };

    const Var* param = func->params_[param_index].get();
    std::vector<CandidateUse> uses;
    size_t total_refs = 0;
    bool unsupported = false;

    std::function<void(const StmtPtr&, std::unordered_map<const Var*, ExprPtr>,
                       std::unordered_map<const Var*, DynamicIndexSource>, ExprPtr)>
        visit_stmt = [&](const StmtPtr& stmt, std::unordered_map<const Var*, ExprPtr> scalar_defs,
                         std::unordered_map<const Var*, DynamicIndexSource> scalar_index_sources,
                         ExprPtr guard_condition) {
          if (!stmt || unsupported) return;

          if (auto seq = As<SeqStmts>(stmt)) {
            for (const auto& child : seq->stmts_) {
              visit_stmt(child, scalar_defs, scalar_index_sources, guard_condition);
              if (unsupported) return;
              if (auto assign = As<AssignStmt>(child)) {
                if (!As<ScalarType>(assign->var_->GetType())) continue;
                auto source = MatchTensorReadIndexSource(assign->value_, func, tensor_param_indices,
                                                         scalar_index_sources, scalar_defs);
                if (source.has_value()) {
                  scalar_index_sources[assign->var_.get()] = *source;
                  continue;
                }
                scalar_defs[assign->var_.get()] =
                    arith::Analyzer().Simplify(transform_utils::Substitute(assign->value_, scalar_defs));
              }
            }
            return;
          }

          if (auto if_stmt = As<IfStmt>(stmt)) {
            if (if_stmt->else_body_.has_value()) {
              if (CountVarRefsInStmt(*if_stmt->else_body_, param) != 0) {
                unsupported = true;
                return;
              }
            }
            auto expanded_guard =
                arith::Analyzer().Simplify(transform_utils::Substitute(if_stmt->condition_, scalar_defs));
            visit_stmt(
                if_stmt->then_body_, scalar_defs, scalar_index_sources,
                guard_condition ? MakeAnd(guard_condition, expanded_guard, if_stmt->span_) : expanded_guard);
            return;
          }

          if (As<ForStmt>(stmt) || As<WhileStmt>(stmt)) {
            if (CountVarRefsInStmt(stmt, param) != 0) unsupported = true;
            return;
          }

          auto assign = As<AssignStmt>(stmt);
          size_t refs = CountVarRefsInStmt(stmt, param);
          if (refs != 0) {
            auto use = MatchExpandedInputWindowUse(assign, param, refs, scalar_defs);
            if (!use.has_value()) {
              unsupported = true;
              return;
            }
            total_refs += refs;
            uses.push_back(CandidateUse{std::move(*use), guard_condition});
          }
        };

    visit_stmt(loop->body_, {}, {}, nullptr);
    if (unsupported || total_refs == 0 || uses.empty()) return std::nullopt;

    const auto& first = uses.front();
    for (const auto& use : uses) {
      if (!AreExprVectorsEqual(use.use.window_shape, first.use.window_shape) ||
          use.use.offsets.size() != first.use.offsets.size() ||
          !AreExprsEqual(use.use.offsets[0], first.use.offsets[0]) ||
          !AreExprVectorsEqual(use.use.offsets, first.use.offsets)) {
        return std::nullopt;
      }
      if ((use.guard_condition && !first.guard_condition) ||
          (!use.guard_condition && first.guard_condition) ||
          (use.guard_condition && first.guard_condition &&
           !AreExprsEqual(use.guard_condition, first.guard_condition))) {
        return std::nullopt;
      }
    }

    if (first.use.offsets.empty() || first.use.window_shape.size() != tensor_type->shape_.size()) {
      return std::nullopt;
    }

    std::unordered_map<const Var*, ExprPtr> scalar_defs;
    std::unordered_map<const Var*, DynamicIndexSource> scalar_index_sources;
    std::function<void(const StmtPtr&)> collect_defs = [&](const StmtPtr& stmt) {
      if (!stmt || unsupported) return;
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) collect_defs(child);
        return;
      }
      if (auto if_stmt = As<IfStmt>(stmt)) {
        collect_defs(if_stmt->then_body_);
        return;
      }
      auto assign = As<AssignStmt>(stmt);
      if (!assign || !As<ScalarType>(assign->var_->GetType())) return;
      auto source = MatchTensorReadIndexSource(assign->value_, func, tensor_param_indices,
                                               scalar_index_sources, scalar_defs);
      if (source.has_value()) {
        scalar_index_sources[assign->var_.get()] = *source;
        return;
      }
      scalar_defs[assign->var_.get()] =
          arith::Analyzer().Simplify(transform_utils::Substitute(assign->value_, scalar_defs));
    };
    collect_defs(loop->body_);

    std::optional<DynamicIndexSource> dynamic_source;
    const Var* dynamic_value = nullptr;
    for (const auto& [candidate_var, source] : scalar_index_sources) {
      size_t refs = CountVarRefsInExpr(first.use.offsets[0], candidate_var);
      if (refs == 0) continue;
      if (refs != 1 || dynamic_source.has_value()) return std::nullopt;
      dynamic_source = source;
      dynamic_value = candidate_var;
    }
    if (!dynamic_source.has_value() || !dynamic_value) return std::nullopt;

    auto linear = ParseLinearIndexExpr(first.use.offsets[0]);
    if (!linear.has_value()) return std::nullopt;
    auto coeff_it = linear->coeffs.find(dynamic_value);
    if (coeff_it == linear->coeffs.end() || coeff_it->second <= 0) return std::nullopt;

    auto allowed_index_exprs = CollectAllowedVars(func->params_, loop->loop_var_.get());
    if (!ExprsReferenceOnlyVarsIn(dynamic_source->index_offsets, allowed_index_exprs)) {
      return std::nullopt;
    }
    if (!IsSafeDynamicIndexedWindowExprs(func, first.use, /*dynamic_dim=*/0, dynamic_value)) {
      return std::nullopt;
    }

    if (first.guard_condition) {
      auto allowed_guard_vars = allowed_index_exprs;
      allowed_guard_vars.insert(dynamic_value);
      if (!ExprReferencesOnlyVarsIn(first.guard_condition, allowed_guard_vars)) return std::nullopt;
    }

    std::vector<ExprPtr> local_zero_offsets;
    local_zero_offsets.reserve(first.use.offsets.size());
    for (size_t i = 0; i < first.use.offsets.size(); ++i) {
      local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
    }

    auto piece = MakeDensePiece(first.use.window_shape, first.use.offsets, local_zero_offsets);
    InputRewriteInfo info{param_index,       tensor_type->shape_, first.use.window_shape,
                          first.use.offsets, local_zero_offsets,  MakeDenseRegion({std::move(piece)})};
    info.dynamic_indexed_window = DynamicIndexedInputWindow{dynamic_source->index_tensor_param_index,
                                                            loop->loop_var_,
                                                            loop->start_,
                                                            loop->stop_,
                                                            loop->step_,
                                                            first.guard_condition,
                                                            dynamic_source->index_offsets,
                                                            dynamic_value,
                                                            0};
    return info;
  }

  static std::vector<InputRewriteInfo> AnalyzeDynamicIndexedInputWindows(const FunctionPtr& func) {
    std::vector<InputRewriteInfo> inputs;
    if (!func || func->return_types_.empty()) return inputs;

    std::vector<ForStmtPtr> loops;
    std::function<void(const StmtPtr&)> collect_loops = [&](const StmtPtr& stmt) {
      if (!stmt) return;
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) collect_loops(child);
        return;
      }
      if (auto loop = As<ForStmt>(stmt)) {
        loops.push_back(loop);
        return;
      }
      if (auto if_stmt = As<IfStmt>(stmt)) {
        collect_loops(if_stmt->then_body_);
        if (if_stmt->else_body_.has_value()) collect_loops(*if_stmt->else_body_);
      }
    };
    collect_loops(func->body_);

    for (size_t param_index = 0; param_index < func->params_.size(); ++param_index) {
      if (param_index >= func->param_directions_.size()) continue;
      if (func->param_directions_[param_index] != ParamDirection::In) continue;
      if (!As<TensorType>(func->params_[param_index]->GetType())) continue;
      for (const auto& loop : loops) {
        if (CountVarRefsInStmt(loop, func->params_[param_index].get()) == 0) continue;
        auto info = AnalyzeDynamicIndexedInputWindowInLoop(func, param_index, loop);
        if (info.has_value()) {
          inputs.push_back(std::move(*info));
          break;
        }
      }
    }
    return inputs;
  }

  struct ExtractedInputAccessSet {
    size_t total_refs = 0;
    bool unsupported_ref = false;
    std::vector<InputWindowUse> uses;
  };

  static ExtractedInputAccessSet ExtractInputAccessSet(const StmtPtr& root, const Var* param,
                                                       std::unordered_map<const Var*, ExprPtr> subst = {}) {
    ExtractedInputAccessSet result;
    if (!root || !param) return result;

    // Keep recursive input access extraction bounded; larger dynamic patterns
    // stay on the baseline path instead of expanding compile-time work.
    constexpr int64_t kMaxEnumeratedLoopTripCount = 256;
    constexpr size_t kMaxEnumeratedInputUses = 512;
    arith::Analyzer analyzer;

    auto simplify_with_subst = [&](const ExprPtr& expr) -> ExprPtr {
      return analyzer.Simplify(transform_utils::Substitute(expr, subst));
    };

    std::function<void(const StmtPtr&)> visit_stmt = [&](const StmtPtr& stmt) {
      if (!stmt || result.unsupported_ref) return;

      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) visit_stmt(child);
        return;
      }

      if (auto assign = As<AssignStmt>(stmt)) {
        size_t refs = CountVarRefsInStmt(assign, param);
        if (refs != 0) {
          auto use = MatchExpandedInputWindowUse(assign, param, refs, subst);
          if (!use.has_value()) {
            result.unsupported_ref = true;
            return;
          }
          result.total_refs += refs;
          result.uses.push_back(std::move(*use));
          if (result.uses.size() > kMaxEnumeratedInputUses) {
            result.unsupported_ref = true;
            return;
          }
        }
        if (As<ScalarType>(assign->var_->GetType())) {
          subst[assign->var_.get()] = simplify_with_subst(assign->value_);
        }
        return;
      }

      if (auto loop = As<ForStmt>(stmt)) {
        if (CountVarRefsInExpr(loop->start_, param) != 0 || CountVarRefsInExpr(loop->stop_, param) != 0 ||
            CountVarRefsInExpr(loop->step_, param) != 0) {
          result.unsupported_ref = true;
          return;
        }

        auto trip_count = GetKnownPositiveTripCount(loop);
        if (!trip_count.has_value() || *trip_count < 0 || *trip_count > kMaxEnumeratedLoopTripCount) {
          if (CountVarRefsInStmt(loop->body_, param) != 0) result.unsupported_ref = true;
          return;
        }

        auto saved_subst = subst;
        for (int64_t trip = 0; trip < *trip_count; ++trip) {
          auto loop_value = GetLoopValueAtTrip(loop, trip);
          if (!loop_value.has_value()) {
            result.unsupported_ref = true;
            break;
          }
          subst = saved_subst;
          subst[loop->loop_var_.get()] = analyzer.Simplify(transform_utils::Substitute(*loop_value, subst));
          visit_stmt(loop->body_);
          if (result.unsupported_ref) break;
        }
        subst = std::move(saved_subst);
        return;
      }

      size_t refs = CountVarRefsInStmt(stmt, param);
      if (refs != 0) result.unsupported_ref = true;
    };

    visit_stmt(root);
    return result;
  }

  static std::optional<InputRewriteInfo> BuildDenseInputWindowFromAccessSet(
      const FunctionPtr& func, size_t param_index, const ExtractedInputAccessSet& access_set) {
    if (!func || param_index >= func->params_.size()) return std::nullopt;
    auto tensor_type = As<TensorType>(func->params_[param_index]->GetType());
    if (!tensor_type || access_set.unsupported_ref || access_set.uses.empty() || access_set.total_refs == 0) {
      return std::nullopt;
    }

    std::vector<ExprPtr> base_offsets(tensor_type->shape_.size());
    std::vector<ExprPtr> max_extents(tensor_type->shape_.size());
    arith::Analyzer analyzer;
    bool expands_beyond_single_access = access_set.uses.size() > 1;
    for (const auto& use : access_set.uses) {
      if (use.offsets.size() != use.window_shape.size() || use.offsets.size() != tensor_type->shape_.size()) {
        return std::nullopt;
      }
      for (size_t dim = 0; dim < use.offsets.size(); ++dim) {
        auto min_expr = SelectMinExpr(base_offsets[dim], use.offsets[dim], func->span_);
        if (!min_expr.has_value()) return std::nullopt;
        base_offsets[dim] = *min_expr;

        auto extent = analyzer.Simplify(MakeAdd(use.offsets[dim], use.window_shape[dim], func->span_));
        auto max_expr = SelectMaxExpr(max_extents[dim], extent, func->span_);
        if (!max_expr.has_value()) return std::nullopt;
        max_extents[dim] = *max_expr;
      }
    }

    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> local_zero_offsets;
    window_shape.reserve(tensor_type->shape_.size());
    local_zero_offsets.reserve(tensor_type->shape_.size());
    for (size_t dim = 0; dim < tensor_type->shape_.size(); ++dim) {
      if (!base_offsets[dim] || !max_extents[dim]) return std::nullopt;
      auto span = GetConstantSpanValue(max_extents[dim], base_offsets[dim], func->span_);
      if (!span.has_value()) return std::nullopt;
      int64_t span_value = *span;
      if (span_value <= 0) return std::nullopt;
      window_shape.push_back(std::make_shared<ConstInt>(span_value, DataType::INDEX, func->span_));
      local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
    }

    if (!expands_beyond_single_access && access_set.uses.size() == 1) {
      expands_beyond_single_access =
          !AreExprVectorsEqual(access_set.uses.front().window_shape, window_shape) ||
          !AreExprVectorsEqual(access_set.uses.front().offsets, base_offsets);
    }
    if (!expands_beyond_single_access) return std::nullopt;

    if (AreExprVectorsEqual(window_shape, tensor_type->shape_) && IsAllZeroOffsets(base_offsets)) {
      return std::nullopt;
    }
    if (!CanMaterializeWindowParamType(tensor_type, window_shape)) return std::nullopt;

    auto allowed_params = CollectAllowedVars(func->params_);
    if (!ExprsReferenceOnlyVarsIn(window_shape, allowed_params) ||
        !ExprsReferenceOnlyVarsIn(base_offsets, allowed_params)) {
      return std::nullopt;
    }

    auto piece = MakeDensePiece(window_shape, base_offsets, local_zero_offsets);
    return InputRewriteInfo{param_index,
                            tensor_type->shape_,
                            std::move(window_shape),
                            std::move(base_offsets),
                            std::move(local_zero_offsets),
                            MakeDenseRegion({std::move(piece)})};
  }

  static std::unordered_map<const Var*, InputParamUseSummary> CollectInputParamUsesInStmt(
      const StmtPtr& root, const std::unordered_map<const Var*, size_t>& candidate_indices) {
    std::unordered_map<const Var*, InputParamUseSummary> summaries;
    if (!root || candidate_indices.empty()) return summaries;

    auto body_stmts = FlattenToStmts(root);
    class CandidateRefCollector : public IRVisitor {
     public:
      explicit CandidateRefCollector(const std::unordered_map<const Var*, size_t>& candidate_indices)
          : candidate_indices_(candidate_indices) {}

      [[nodiscard]] const std::unordered_map<const Var*, size_t>& refs() const { return refs_; }

     protected:
      void VisitExpr_(const VarPtr& op) override {
        if (candidate_indices_.count(op.get())) ++refs_[op.get()];
        IRVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const IterArgPtr& op) override {
        if (candidate_indices_.count(op.get())) ++refs_[op.get()];
        IRVisitor::VisitExpr_(op);
      }

     private:
      const std::unordered_map<const Var*, size_t>& candidate_indices_;
      std::unordered_map<const Var*, size_t> refs_;
    };

    for (const auto& stmt : body_stmts) {
      CandidateRefCollector collector(candidate_indices);
      collector.VisitStmt(stmt);

      for (const auto& [param, refs_in_stmt] : collector.refs()) {
        auto& summary = summaries[param];
        summary.total_refs += refs_in_stmt;

        auto use = MatchInputWindowUse(As<AssignStmt>(stmt), param, refs_in_stmt);
        if (!use.has_value()) {
          summary.unsupported_ref = true;
          continue;
        }
        summary.uses.push_back(std::move(*use));
      }
    }

    return summaries;
  }

  static std::unordered_map<const Var*, InputParamUseSummary> CollectInputParamUses(
      const FunctionPtr& func, const std::unordered_map<const Var*, size_t>& candidate_indices) {
    if (!func) return {};
    return CollectInputParamUsesInStmt(func->body_, candidate_indices);
  }

  static std::vector<InputRewriteInfo> AnalyzeInputWindows(const FunctionPtr& func) {
    std::vector<InputRewriteInfo> inputs;
    if (!func) return inputs;
    if (func->return_types_.empty()) return inputs;

    auto allowed_params = CollectAllowedVars(func->params_);

    std::unordered_map<const Var*, size_t> candidate_indices;
    std::vector<std::pair<const Var*, size_t>> ordered_candidates;
    for (size_t param_index = 0; param_index < func->params_.size(); ++param_index) {
      if (param_index >= func->param_directions_.size()) continue;
      if (func->param_directions_[param_index] != ParamDirection::In) continue;
      if (!As<TensorType>(func->params_[param_index]->GetType())) continue;
      candidate_indices.emplace(func->params_[param_index].get(), param_index);
      ordered_candidates.emplace_back(func->params_[param_index].get(), param_index);
    }

    auto summaries = CollectInputParamUses(func, candidate_indices);
    for (const auto& [param_ptr, param_index] : ordered_candidates) {
      const auto& param = func->params_[param_index];
      auto summary_it = summaries.find(param_ptr);
      if (summary_it == summaries.end() || summary_it->second.total_refs == 0) continue;

      auto tensor_type = As<TensorType>(param->GetType());
      if (!tensor_type) continue;

      std::optional<InputRewriteInfo> input_info;
      std::optional<InputWindowUse> matched;
      size_t matched_refs = 0;
      bool unsupported_ref = summary_it->second.unsupported_ref;
      for (const auto& use : summary_it->second.uses) {
        if (!AreExprVectorsEqual(use.window_shape, matched ? matched->window_shape : use.window_shape) ||
            !AreExprVectorsEqual(use.offsets, matched ? matched->offsets : use.offsets)) {
          unsupported_ref = true;
          break;
        }
        matched = use;
        matched_refs += use.param_refs_in_stmt;
      }
      if (!unsupported_ref && matched.has_value() && matched_refs == summary_it->second.total_refs &&
          !(AreExprVectorsEqual(matched->window_shape, tensor_type->shape_) &&
            IsAllZeroOffsets(matched->offsets)) &&
          CanMaterializeWindowParamType(tensor_type, matched->window_shape) &&
          ExprsReferenceOnlyVarsIn(matched->window_shape, allowed_params) &&
          ExprsReferenceOnlyVarsIn(matched->offsets, allowed_params)) {
        std::vector<ExprPtr> local_zero_offsets;
        local_zero_offsets.reserve(matched->offsets.size());
        for (size_t i = 0; i < matched->offsets.size(); ++i) {
          local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
        }
        auto piece = MakeDensePiece(matched->window_shape, matched->offsets, local_zero_offsets);
        input_info = InputRewriteInfo{
            param_index,      tensor_type->shape_,           matched->window_shape,
            matched->offsets, std::move(local_zero_offsets), MakeDenseRegion({std::move(piece)})};
      }

      if (!input_info.has_value()) {
        auto access_set = ExtractInputAccessSet(func->body_, param_ptr);
        input_info = BuildDenseInputWindowFromAccessSet(func, param_index, access_set);
      }
      if (input_info.has_value()) inputs.push_back(std::move(*input_info));
    }

    return inputs;
  }

  static std::optional<InputRewriteInfo> AnalyzeAggregateInputWindowInLoop(
      const FunctionPtr& func, size_t param_index, const ForStmtPtr& loop, size_t total_refs,
      const InputParamUseSummary& loop_summary) {
    if (!func || param_index >= func->params_.size() || !loop) return std::nullopt;
    auto tensor_type = As<TensorType>(func->params_[param_index]->GetType());
    if (!tensor_type) return std::nullopt;

    auto trip_count = GetKnownPositiveTripCount(loop);
    if (!trip_count.has_value() || *trip_count <= 0) return std::nullopt;
    auto first_loop_value = GetLoopValueAtTrip(loop, 0);
    auto last_loop_value = GetLoopValueAtTrip(loop, *trip_count - 1);
    if (!first_loop_value.has_value() || !last_loop_value.has_value()) return std::nullopt;

    if (total_refs == 0 || total_refs != loop_summary.total_refs || loop_summary.unsupported_ref) {
      return std::nullopt;
    }

    auto loop_body_stmts = FlattenToStmts(loop->body_);
    std::unordered_map<const Var*, ExprPtr> scalar_defs;
    for (const auto& stmt : loop_body_stmts) {
      if (auto assign = As<AssignStmt>(stmt)) {
        if (As<ScalarType>(assign->var_->GetType())) {
          scalar_defs[assign->var_.get()] = assign->value_;
        }
      }
    }

    const auto& uses = loop_summary.uses;
    size_t matched_refs = 0;
    for (const auto& use : uses) matched_refs += use.param_refs_in_stmt;
    if (uses.empty() || matched_refs != total_refs) return std::nullopt;

    auto allowed = CollectAllowedVars(func->params_, loop->loop_var_.get());

    std::optional<InputRewriteInfo> result;
    for (const auto& use : uses) {
      if (use.offsets.size() != use.window_shape.size() || use.offsets.size() != tensor_type->shape_.size()) {
        return std::nullopt;
      }

      std::vector<ExprPtr> base_offsets;
      std::vector<ExprPtr> local_offsets;
      std::vector<ExprPtr> window_shape;
      bool expands_across_loop = false;
      arith::Analyzer analyzer;
      for (size_t i = 0; i < use.offsets.size(); ++i) {
        auto expanded = ExpandLoopLocalExpr(use.offsets[i], scalar_defs);
        if (!expanded.has_value()) return std::nullopt;
        if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) return std::nullopt;

        auto ordered_offsets = GetOrderedLoopOffsets(*expanded, loop, *first_loop_value, *last_loop_value);
        if (!ordered_offsets.has_value()) return std::nullopt;

        auto max_extent = analyzer.Simplify(MakeAdd(ordered_offsets->max, use.window_shape[i], func->span_));
        auto span_value = GetConstantSpanValue(max_extent, ordered_offsets->min, func->span_);
        if (!span_value.has_value() || *span_value <= 0) return std::nullopt;

        if (!AreExprsEqual(ordered_offsets->min, ordered_offsets->max)) {
          expands_across_loop = true;
        }
        base_offsets.push_back(ordered_offsets->min);
        local_offsets.push_back(
            analyzer.Simplify(MakeSub(use.offsets[i], ordered_offsets->min, use.offsets[i]->span_)));
        window_shape.push_back(std::make_shared<ConstInt>(*span_value, DataType::INDEX, func->span_));
      }
      if (!expands_across_loop) return std::nullopt;

      auto current_window_shape = std::move(window_shape);
      auto current_base_offsets = std::move(base_offsets);
      auto current_local_offsets = std::move(local_offsets);
      auto current_piece = MakeDensePiece(current_window_shape, current_base_offsets, current_local_offsets);
      InputRewriteInfo current{param_index,
                               tensor_type->shape_,
                               std::move(current_window_shape),
                               std::move(current_base_offsets),
                               std::move(current_local_offsets),
                               MakeDenseRegion({std::move(current_piece)})};
      if (!CanMaterializeWindowParamType(tensor_type, current.window_shape)) return std::nullopt;

      if (!result.has_value()) {
        result = std::move(current);
        continue;
      }
      if (!AreExprVectorsEqual(result->window_shape, current.window_shape) ||
          !AreExprVectorsEqual(result->callsite_offsets, current.callsite_offsets) ||
          !AreExprVectorsEqual(result->local_read_offsets, current.local_read_offsets)) {
        return std::nullopt;
      }
    }

    if (!result.has_value()) return std::nullopt;
    auto allowed_params = CollectAllowedVars(func->params_);
    if (!ExprsReferenceOnlyVarsIn(result->window_shape, allowed_params) ||
        !ExprsReferenceOnlyVarsIn(result->callsite_offsets, allowed_params)) {
      return std::nullopt;
    }
    return result;
  }

  static std::vector<InputRewriteInfo> AnalyzeAggregateInputWindows(
      const FunctionPtr& func, const std::vector<InputRewriteInfo>& existing_inputs, const ForStmtPtr& loop) {
    std::vector<InputRewriteInfo> inputs;
    if (!func || !loop) return inputs;

    std::unordered_set<size_t> existing_indices;
    for (const auto& input : existing_inputs) existing_indices.insert(input.in_param_index);

    std::unordered_map<const Var*, size_t> candidate_indices;
    std::vector<std::pair<const Var*, size_t>> ordered_candidates;
    for (size_t param_index = 0; param_index < func->params_.size(); ++param_index) {
      if (existing_indices.count(param_index)) continue;
      if (param_index >= func->param_directions_.size()) continue;
      if (func->param_directions_[param_index] != ParamDirection::In) continue;
      if (!As<TensorType>(func->params_[param_index]->GetType())) continue;
      candidate_indices.emplace(func->params_[param_index].get(), param_index);
      ordered_candidates.emplace_back(func->params_[param_index].get(), param_index);
    }
    if (candidate_indices.empty()) return inputs;

    auto total_summaries = CollectInputParamUsesInStmt(func->body_, candidate_indices);
    auto loop_summaries = CollectInputParamUsesInStmt(loop->body_, candidate_indices);
    for (const auto& [param_ptr, param_index] : ordered_candidates) {
      auto total_it = total_summaries.find(param_ptr);
      auto loop_it = loop_summaries.find(param_ptr);
      if (total_it == total_summaries.end() || loop_it == loop_summaries.end()) continue;

      auto matched = AnalyzeAggregateInputWindowInLoop(func, param_index, loop, total_it->second.total_refs,
                                                       loop_it->second);
      if (!matched.has_value()) {
        auto access_set = ExtractInputAccessSet(func->body_, param_ptr);
        matched = BuildDenseInputWindowFromAccessSet(func, param_index, access_set);
      }
      if (matched.has_value()) inputs.push_back(std::move(*matched));
    }
    return inputs;
  }

  static std::optional<CalleeRewriteAnalysis> AnalyzeAggregateWindowLoop(
      const FunctionPtr& func, const std::vector<size_t>& out_indices,
      const std::vector<InputRewriteInfo>& existing_inputs, bool include_full_shape_zero_outputs = false) {
    if (!func || out_indices.empty()) return std::nullopt;

    auto body_stmts = FlattenToStmts(func->body_);
    if (body_stmts.empty()) return std::nullopt;

    ReturnStmtPtr ret_stmt = As<ReturnStmt>(body_stmts.back());
    if (!ret_stmt) return std::nullopt;

    struct AggregateLoopOutputMatch {
      size_t out_param_index;
      size_t return_index;
      size_t iter_arg_index;
    };

    ForStmtPtr loop;
    std::vector<AggregateLoopOutputMatch> loop_matches;
    for (const auto& stmt : body_stmts) {
      auto candidate = As<ForStmt>(stmt);
      if (!candidate || candidate->iter_args_.empty()) continue;
      std::vector<AggregateLoopOutputMatch> candidate_matches;
      std::unordered_set<size_t> matched_iter_arg_indices;

      for (const auto& out_param_index : out_indices) {
        std::optional<size_t> direct_return_index = FindReturnIndexForOutParam(func, out_param_index);
        VarPtr direct_returned;
        if (direct_return_index.has_value() && *direct_return_index < ret_stmt->value_.size()) {
          direct_returned = AsVarLike(ret_stmt->value_[*direct_return_index]);
        }

        for (size_t i = 0; i < candidate->iter_args_.size() && i < candidate->return_vars_.size(); ++i) {
          auto init_var = AsVarLike(candidate->iter_args_[i]->initValue_);
          if (!init_var || init_var.get() != func->params_[out_param_index].get()) continue;

          std::optional<size_t> return_index = direct_return_index;
          if (direct_returned && direct_returned.get() != candidate->return_vars_[i].get() &&
              direct_returned.get() != func->params_[out_param_index].get()) {
            return_index = std::nullopt;
          }
          for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
            if (return_index.has_value()) break;
            auto returned = AsVarLike(ret_stmt->value_[ret_i]);
            if (returned && returned.get() == candidate->return_vars_[i].get()) {
              return_index = ret_i;
              break;
            }
          }
          if (!return_index.has_value()) continue;

          if (!matched_iter_arg_indices.insert(i).second) return std::nullopt;
          candidate_matches.push_back(AggregateLoopOutputMatch{out_param_index, *return_index, i});
          break;
        }
      }

      if (candidate_matches.empty()) continue;
      if (candidate->iter_args_.size() != candidate->return_vars_.size()) return std::nullopt;

      if (loop) return std::nullopt;
      loop = candidate;
      loop_matches = std::move(candidate_matches);
    }
    if (!loop) return std::nullopt;

    auto stop = GetConstIntValue(loop->stop_);
    auto step = GetConstIntValue(loop->step_);
    if (!stop.has_value() || !step.has_value()) {
      auto known_trip_count = GetKnownPositiveTripCount(loop);
      if (!known_trip_count.has_value() || *known_trip_count <= 0) return std::nullopt;
    } else if (*step <= 0) {
      return std::nullopt;
    }
    auto trip_count = GetKnownPositiveTripCount(loop);
    if (!trip_count.has_value() || *trip_count <= 0) return std::nullopt;
    auto first_loop_value = GetLoopValueAtTrip(loop, 0);
    auto last_loop_value = GetLoopValueAtTrip(loop, *trip_count - 1);
    if (!first_loop_value.has_value() || !last_loop_value.has_value()) return std::nullopt;

    auto loop_body_stmts = FlattenToStmts(loop->body_);
    YieldStmtPtr yield_stmt;
    struct AggregateUpdate {
      AssignStmtPtr assign;
      std::vector<ExprPtr> window_shape;
      std::vector<ExprPtr> offsets;
    };

    std::unordered_map<size_t, std::vector<AggregateUpdate>> updates_by_iter_arg_index;
    std::unordered_map<size_t, std::vector<AggregateUpdate>> reads_by_iter_arg_index;
    std::unordered_map<size_t, const Var*> update_tail_by_iter_arg_index;
    std::unordered_map<size_t, std::unordered_set<const Var*>> carrier_vars_by_iter_arg_index;
    for (const auto& match : loop_matches) {
      if (match.iter_arg_index >= loop->iter_args_.size()) return std::nullopt;
      update_tail_by_iter_arg_index[match.iter_arg_index] = loop->iter_args_[match.iter_arg_index].get();
      carrier_vars_by_iter_arg_index[match.iter_arg_index].insert(
          loop->iter_args_[match.iter_arg_index].get());
    }
    std::unordered_map<const Var*, ExprPtr> scalar_defs;
    constexpr int64_t kMaxNestedAccessTripCount = 32;

    auto substitute_local_scalars = [](const ExprPtr& expr,
                                       const std::unordered_map<const Var*, ExprPtr>& local_defs) -> ExprPtr {
      return transform_utils::Substitute(expr, local_defs);
    };

    std::function<bool(const StmtPtr&, std::unordered_map<size_t, const Var*>*,
                       std::unordered_map<const Var*, ExprPtr>*, YieldStmtPtr*)>
        collect_accesses;

    collect_accesses = [&](const StmtPtr& stmt, std::unordered_map<size_t, const Var*>* tails,
                           std::unordered_map<const Var*, ExprPtr>* local_scalar_defs,
                           YieldStmtPtr* seen_yield) -> bool {
      if (!stmt || !tails || !local_scalar_defs || !seen_yield) return false;
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) {
          if (!collect_accesses(child, tails, local_scalar_defs, seen_yield)) return false;
        }
        return true;
      }

      if (auto assign = As<AssignStmt>(stmt)) {
        auto call = As<Call>(assign->value_);
        if (call) {
          const Var* updated_tail = nullptr;
          const Var* read_tail = nullptr;
          std::vector<ExprPtr> window_shape;
          std::vector<ExprPtr> offsets;
          if (call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
            auto out_arg = AsVarLike(call->args_[2]);
            auto offset_tuple = As<MakeTuple>(call->args_[1]);
            auto tile_type = As<TileType>(call->args_[0]->GetType());
            if (out_arg && offset_tuple && tile_type) {
              updated_tail = out_arg.get();
              window_shape = tile_type->shape_;
              offsets = offset_tuple->elements_;
            }
          } else if (call->op_->name_ == "tensor.assemble" && call->args_.size() >= 3) {
            auto parent_arg = AsVarLike(call->args_[0]);
            auto offset_tuple = As<MakeTuple>(call->args_[2]);
            auto source_type = As<TensorType>(call->args_[1]->GetType());
            if (parent_arg && offset_tuple && source_type) {
              updated_tail = parent_arg.get();
              window_shape = source_type->shape_;
              offsets = offset_tuple->elements_;
            }
          } else if (call->op_->name_ == "tile.load" && call->args_.size() >= 3) {
            auto parent_arg = AsVarLike(call->args_[0]);
            auto offset_tuple = As<MakeTuple>(call->args_[1]);
            auto tile_type = As<TileType>(call->GetType());
            if (parent_arg && offset_tuple && tile_type) {
              read_tail = parent_arg.get();
              window_shape = tile_type->shape_;
              offsets = offset_tuple->elements_;
            }
          } else if (call->op_->name_ == "tensor.slice" && call->args_.size() >= 3) {
            auto parent_arg = AsVarLike(call->args_[0]);
            auto offset_tuple = As<MakeTuple>(call->args_[2]);
            auto source_type = As<TensorType>(call->GetType());
            if (parent_arg && offset_tuple && source_type) {
              read_tail = parent_arg.get();
              window_shape = source_type->shape_;
              offsets = offset_tuple->elements_;
            }
          }

          if (updated_tail) {
            for (auto& [iter_arg_index, tail] : *tails) {
              if (updated_tail != tail) continue;
              for (auto& offset : offsets) {
                offset = substitute_local_scalars(offset, *local_scalar_defs);
              }
              updates_by_iter_arg_index[iter_arg_index].push_back(
                  AggregateUpdate{assign, std::move(window_shape), std::move(offsets)});
              tail = assign->var_.get();
              carrier_vars_by_iter_arg_index[iter_arg_index].insert(assign->var_.get());
              return true;
            }
          }
          if (read_tail) {
            for (auto& [iter_arg_index, tail] : *tails) {
              if (read_tail != tail) continue;
              for (auto& offset : offsets) {
                offset = substitute_local_scalars(offset, *local_scalar_defs);
              }
              reads_by_iter_arg_index[iter_arg_index].push_back(
                  AggregateUpdate{assign, std::move(window_shape), std::move(offsets)});
              return true;
            }
          }
        }

        for (const auto& [iter_arg_index, carrier_vars] : carrier_vars_by_iter_arg_index) {
          (void)iter_arg_index;
          for (const auto* carrier_var : carrier_vars) {
            if (CountVarRefsInStmt(assign, carrier_var) != 0) return false;
          }
        }
        if (As<ScalarType>(assign->var_->GetType())) {
          (*local_scalar_defs)[assign->var_.get()] =
              substitute_local_scalars(assign->value_, *local_scalar_defs);
        }
        return true;
      }

      if (auto nested_loop = As<ForStmt>(stmt)) {
        std::unordered_map<size_t, size_t> nested_iter_by_outer_iter;
        for (auto& [outer_iter_index, tail] : *tails) {
          for (size_t nested_i = 0; nested_i < nested_loop->iter_args_.size(); ++nested_i) {
            auto init_var = AsVarLike(nested_loop->iter_args_[nested_i]->initValue_);
            if (init_var && init_var.get() == tail) {
              nested_iter_by_outer_iter.emplace(outer_iter_index, nested_i);
              break;
            }
          }
        }
        if (nested_iter_by_outer_iter.empty()) return true;

        auto nested_trip_count = GetKnownPositiveTripCount(nested_loop);
        if (!nested_trip_count.has_value() || *nested_trip_count < 0 ||
            *nested_trip_count > kMaxNestedAccessTripCount) {
          return false;
        }
        if (nested_loop->iter_args_.size() != nested_loop->return_vars_.size()) return false;

        for (int64_t trip = 0; trip < *nested_trip_count; ++trip) {
          auto loop_value = GetLoopValueAtTrip(nested_loop, trip);
          if (!loop_value.has_value()) return false;

          auto trip_scalar_defs = *local_scalar_defs;
          trip_scalar_defs[nested_loop->loop_var_.get()] =
              substitute_local_scalars(*loop_value, trip_scalar_defs);

          std::unordered_map<size_t, const Var*> trip_tails;
          for (const auto& [outer_iter_index, nested_i] : nested_iter_by_outer_iter) {
            trip_tails[outer_iter_index] = nested_loop->iter_args_[nested_i].get();
            carrier_vars_by_iter_arg_index[outer_iter_index].insert(nested_loop->iter_args_[nested_i].get());
          }

          YieldStmtPtr nested_yield;
          if (!collect_accesses(nested_loop->body_, &trip_tails, &trip_scalar_defs, &nested_yield)) {
            return false;
          }
          if (!nested_yield || nested_yield->value_.size() != nested_loop->return_vars_.size()) {
            return false;
          }
          for (const auto& [outer_iter_index, nested_i] : nested_iter_by_outer_iter) {
            auto yielded = AsVarLike(nested_yield->value_[nested_i]);
            auto tail_it = trip_tails.find(outer_iter_index);
            if (!yielded || tail_it == trip_tails.end() || yielded.get() != tail_it->second) {
              return false;
            }
          }
        }

        for (const auto& [outer_iter_index, nested_i] : nested_iter_by_outer_iter) {
          (*tails)[outer_iter_index] = nested_loop->return_vars_[nested_i].get();
          carrier_vars_by_iter_arg_index[outer_iter_index].insert(nested_loop->return_vars_[nested_i].get());
        }
        return true;
      }

      if (auto yield = As<YieldStmt>(stmt)) {
        if (*seen_yield) return false;
        *seen_yield = yield;
        return true;
      }

      return true;
    };

    if (!collect_accesses(loop->body_, &update_tail_by_iter_arg_index, &scalar_defs, &yield_stmt)) {
      return std::nullopt;
    }

    if (!yield_stmt) return std::nullopt;

    auto allowed = CollectAllowedVars(func->params_, loop->loop_var_.get());

    CalleeRewriteAnalysis analysis;
    analysis.kind = RewriteKind::AggregateWindowLoop;

    for (const auto& match : loop_matches) {
      auto update_it = updates_by_iter_arg_index.find(match.iter_arg_index);
      if (update_it == updates_by_iter_arg_index.end()) continue;
      const auto& updates = update_it->second;
      if (updates.empty()) continue;
      const auto* tail = update_tail_by_iter_arg_index[match.iter_arg_index];

      auto yielded = AsVarLike(yield_stmt->value_[match.iter_arg_index]);
      if (!yielded || yielded.get() != tail) continue;

      if (!As<TensorType>(loop->iter_args_[match.iter_arg_index]->GetType()) ||
          !As<TensorType>(loop->return_vars_[match.iter_arg_index]->GetType())) {
        continue;
      }

      auto out_param = func->params_[match.out_param_index].get();
      size_t total_out_refs = CountVarRefsInStmt(func->body_, out_param);
      size_t carrier_out_refs = CountVarRefsInStmt(loop, out_param) + CountVarRefsInStmt(ret_stmt, out_param);
      if (total_out_refs == 0 || total_out_refs != carrier_out_refs) {
        continue;
      }

      bool update_chain_is_linear = true;
      for (size_t update_idx = 0; update_idx < updates.size(); ++update_idx) {
        const auto& update = updates[update_idx];
        const size_t result_refs = CountVarRefsInStmt(loop->body_, update.assign->var_.get());
        if (result_refs == 0 || result_refs > 2) {
          update_chain_is_linear = false;
          break;
        }
      }
      if (!update_chain_is_linear) continue;

      const auto read_it = reads_by_iter_arg_index.find(match.iter_arg_index);
      if (read_it != reads_by_iter_arg_index.end()) {
        bool reads_match_writes = true;
        for (const auto& read : read_it->second) {
          bool matched_write = false;
          for (const auto& update : updates) {
            if (AreExprVectorsEqual(read.window_shape, update.window_shape) &&
                AreExprVectorsEqual(read.offsets, update.offsets)) {
              matched_write = true;
              break;
            }
          }
          if (!matched_write) {
            reads_match_writes = false;
            break;
          }
        }
        if (!reads_match_writes) continue;
      }

      auto out_tensor_type = As<TensorType>(func->params_[match.out_param_index]->GetType());
      if (!out_tensor_type) continue;

      auto const_volume = [](const std::vector<ExprPtr>& shape) -> std::optional<int64_t> {
        int64_t volume = 1;
        for (const auto& dim : shape) {
          auto ci = As<ConstInt>(dim);
          if (!ci || ci->value_ <= 0) return std::nullopt;
          volume *= ci->value_;
        }
        return volume;
      };

      struct DenseRect {
        std::vector<ExprPtr> offsets;
        std::vector<ExprPtr> shape;
        int64_t volume = 0;
      };

      auto rects_exactly_tile_bounds = [&](const std::vector<DenseRect>& rects,
                                           const std::vector<ExprPtr>& bounds_offsets,
                                           const std::vector<ExprPtr>& bounds_shape) -> bool {
        if (rects.empty()) return false;
        auto bounds_volume = const_volume(bounds_shape);
        if (!bounds_volume.has_value()) return false;

        int64_t rect_volume_sum = 0;
        for (size_t i = 0; i < rects.size(); ++i) {
          rect_volume_sum += rects[i].volume;
          for (size_t j = 0; j < i; ++j) {
            if (!DenseRectsAreDisjoint(rects[j].offsets, rects[j].shape, rects[i].offsets, rects[i].shape)) {
              return false;
            }
          }
        }
        return rect_volume_sum == *bounds_volume;
      };

      auto try_build_static_pieces = [&]() -> std::vector<DenseRegionPiece> {
        // Static pieces are for small, exactly tiled loop nests; larger loops
        // would bloat signatures and orchestration code, so they stay baseline.
        constexpr int64_t kMaxStaticPieces = 32;
        if (*trip_count <= 0 || *trip_count > kMaxStaticPieces) return {};

        std::vector<DenseRegionPiece> pieces;
        pieces.reserve(static_cast<size_t>(*trip_count));
        arith::Analyzer analyzer;
        for (int64_t trip = 0; trip < *trip_count; ++trip) {
          auto loop_value = GetLoopValueAtTrip(loop, trip);
          if (!loop_value.has_value()) return {};

          std::vector<ExprPtr> piece_offsets(out_tensor_type->shape_.size());
          std::vector<ExprPtr> piece_extents(out_tensor_type->shape_.size());
          std::vector<DenseRect> update_rects;
          update_rects.reserve(updates.size());
          for (const auto& update : updates) {
            if (update.offsets.size() != update.window_shape.size() ||
                update.offsets.size() != out_tensor_type->shape_.size()) {
              return {};
            }
            auto update_volume = const_volume(update.window_shape);
            if (!update_volume.has_value()) return {};
            DenseRect update_rect;
            update_rect.shape = update.window_shape;
            update_rect.volume = *update_volume;
            update_rect.offsets.resize(update.offsets.size());

            for (size_t dim = 0; dim < update.offsets.size(); ++dim) {
              auto expanded = ExpandLoopLocalExpr(update.offsets[dim], scalar_defs);
              if (!expanded.has_value()) return {};
              if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) return {};
              auto offset_at_trip = SimplifyWithLoopValue(*expanded, loop->loop_var_, *loop_value);
              if (!offset_at_trip.has_value()) return {};
              update_rect.offsets[dim] = *offset_at_trip;
              auto min_expr = SelectMinExpr(piece_offsets[dim], *offset_at_trip, func->span_);
              if (!min_expr.has_value()) return {};
              piece_offsets[dim] = *min_expr;
              auto extent =
                  analyzer.Simplify(MakeAdd(*offset_at_trip, update.window_shape[dim], func->span_));
              auto max_expr = SelectMaxExpr(piece_extents[dim], extent, func->span_);
              if (!max_expr.has_value()) return {};
              piece_extents[dim] = *max_expr;
            }
            update_rects.push_back(std::move(update_rect));
          }

          std::vector<ExprPtr> piece_shape;
          std::vector<ExprPtr> local_zero_offsets;
          piece_shape.reserve(out_tensor_type->shape_.size());
          local_zero_offsets.reserve(out_tensor_type->shape_.size());
          for (size_t dim = 0; dim < out_tensor_type->shape_.size(); ++dim) {
            if (!piece_offsets[dim] || !piece_extents[dim]) return {};
            auto span_value = GetConstantSpanValue(piece_extents[dim], piece_offsets[dim], func->span_);
            if (!span_value.has_value() || *span_value <= 0) return {};
            piece_shape.push_back(std::make_shared<ConstInt>(*span_value, DataType::INDEX, func->span_));
            local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
          }

          if (!rects_exactly_tile_bounds(update_rects, piece_offsets, piece_shape)) return {};
          DenseRegionPiece piece =
              MakeDensePiece(std::move(piece_shape), std::move(piece_offsets), std::move(local_zero_offsets));
          for (const auto& existing : pieces) {
            if (!DenseRectsAreDisjoint(existing.callsite_offsets, existing.window_shape,
                                       piece.callsite_offsets, piece.window_shape)) {
              return {};
            }
          }
          pieces.push_back(std::move(piece));
        }
        return pieces;
      };

      std::vector<ExprPtr> base_offsets;
      std::vector<ExprPtr> window_shape;
      std::vector<ExprPtr> max_extents;
      std::vector<ExprPtr> first_iter_base_offsets;
      std::vector<ExprPtr> first_iter_max_extents;
      std::vector<bool> dim_varies;
      base_offsets.resize(out_tensor_type->shape_.size());
      max_extents.resize(out_tensor_type->shape_.size());
      first_iter_base_offsets.resize(out_tensor_type->shape_.size());
      first_iter_max_extents.resize(out_tensor_type->shape_.size());
      dim_varies.resize(out_tensor_type->shape_.size(), false);
      arith::Analyzer analyzer;
      bool output_window_is_proven = true;
      std::vector<DenseRect> first_iter_update_rects;
      first_iter_update_rects.reserve(updates.size());
      for (const auto& update : updates) {
        if (update.offsets.size() != update.window_shape.size() ||
            update.offsets.size() != out_tensor_type->shape_.size()) {
          output_window_is_proven = false;
          break;
        }
        auto update_volume = const_volume(update.window_shape);
        if (!update_volume.has_value()) {
          output_window_is_proven = false;
          break;
        }
        DenseRect first_iter_update_rect;
        first_iter_update_rect.shape = update.window_shape;
        first_iter_update_rect.volume = *update_volume;
        first_iter_update_rect.offsets.resize(update.offsets.size());
        for (size_t i = 0; i < update.offsets.size(); ++i) {
          auto expanded = ExpandLoopLocalExpr(update.offsets[i], scalar_defs);
          if (!expanded.has_value()) {
            output_window_is_proven = false;
            break;
          }
          if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) {
            output_window_is_proven = false;
            break;
          }

          auto ordered_offsets = GetOrderedLoopOffsets(*expanded, loop, *first_loop_value, *last_loop_value);
          if (!ordered_offsets.has_value()) {
            output_window_is_proven = false;
            break;
          }
          if (!AreExprsEqual(ordered_offsets->min, ordered_offsets->max)) dim_varies[i] = true;

          auto min_expr = SelectMinExpr(base_offsets[i], ordered_offsets->min, func->span_);
          if (!min_expr.has_value()) {
            output_window_is_proven = false;
            break;
          }
          base_offsets[i] = *min_expr;

          auto extent = analyzer.Simplify(MakeAdd(ordered_offsets->max, update.window_shape[i], func->span_));
          auto max_expr = SelectMaxExpr(max_extents[i], extent, func->span_);
          if (!max_expr.has_value()) {
            output_window_is_proven = false;
            break;
          }
          max_extents[i] = *max_expr;

          auto first_offset = SimplifyWithLoopValue(*expanded, loop->loop_var_, *first_loop_value);
          if (!first_offset.has_value()) {
            output_window_is_proven = false;
            break;
          }
          first_iter_update_rect.offsets[i] = *first_offset;
          auto first_min_expr = SelectMinExpr(first_iter_base_offsets[i], *first_offset, func->span_);
          if (!first_min_expr.has_value()) {
            output_window_is_proven = false;
            break;
          }
          first_iter_base_offsets[i] = *first_min_expr;
          auto first_extent = analyzer.Simplify(MakeAdd(*first_offset, update.window_shape[i], func->span_));
          auto first_max_expr = SelectMaxExpr(first_iter_max_extents[i], first_extent, func->span_);
          if (!first_max_expr.has_value()) {
            output_window_is_proven = false;
            break;
          }
          first_iter_max_extents[i] = *first_max_expr;
        }
        if (!output_window_is_proven) break;
        first_iter_update_rects.push_back(std::move(first_iter_update_rect));
      }
      if (!output_window_is_proven) {
        auto pieces = try_build_static_pieces();
        if (pieces.empty()) continue;
        analysis.outputs.push_back(OutputRewriteInfo{match.out_param_index,
                                                     match.return_index,
                                                     out_tensor_type->shape_,
                                                     pieces.front().window_shape,
                                                     pieces.front().callsite_offsets,
                                                     pieces.front().local_offsets,
                                                     MakeDenseRegion(std::move(pieces)),
                                                     {},
                                                     {},
                                                     match.iter_arg_index});
        continue;
      }

      std::vector<ExprPtr> local_zero_offsets;
      std::vector<ExprPtr> first_iter_window_shape;
      local_zero_offsets.reserve(out_tensor_type->shape_.size());
      window_shape.reserve(out_tensor_type->shape_.size());
      first_iter_window_shape.reserve(out_tensor_type->shape_.size());
      size_t varying_dim_count = 0;
      bool allow_static_fallback = true;
      for (size_t i = 0; i < out_tensor_type->shape_.size(); ++i) {
        if (!base_offsets[i] || !max_extents[i]) {
          output_window_is_proven = false;
          break;
        }
        auto span_value = GetConstantSpanValue(max_extents[i], base_offsets[i], func->span_);
        if (!span_value.has_value() || *span_value <= 0) {
          output_window_is_proven = false;
          break;
        }

        if (dim_varies[i]) {
          ++varying_dim_count;
          if (!first_iter_base_offsets[i] || !first_iter_max_extents[i]) {
            output_window_is_proven = false;
            break;
          }
          auto first_iter_span_value =
              GetConstantSpanValue(first_iter_max_extents[i], first_iter_base_offsets[i], func->span_);
          if (!first_iter_span_value.has_value() || *first_iter_span_value <= 0) {
            output_window_is_proven = false;
            break;
          }
          int64_t expected_dense_span = *first_iter_span_value * *trip_count;
          if (*span_value != expected_dense_span) {
            output_window_is_proven = false;
            break;
          }
        }

        if (!first_iter_base_offsets[i] || !first_iter_max_extents[i]) {
          output_window_is_proven = false;
          break;
        }
        auto first_iter_span_value =
            GetConstantSpanValue(first_iter_max_extents[i], first_iter_base_offsets[i], func->span_);
        if (!first_iter_span_value.has_value() || *first_iter_span_value <= 0) {
          output_window_is_proven = false;
          break;
        }
        first_iter_window_shape.push_back(
            std::make_shared<ConstInt>(*first_iter_span_value, DataType::INDEX, func->span_));
        window_shape.push_back(std::make_shared<ConstInt>(*span_value, DataType::INDEX, func->span_));
        local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
      }
      if (varying_dim_count > 1) {
        output_window_is_proven = false;
        allow_static_fallback = false;
      } else if (!rects_exactly_tile_bounds(first_iter_update_rects, first_iter_base_offsets,
                                            first_iter_window_shape)) {
        output_window_is_proven = false;
      }
      if (!output_window_is_proven) {
        if (!allow_static_fallback) continue;
        auto pieces = try_build_static_pieces();
        if (pieces.empty()) continue;
        analysis.outputs.push_back(OutputRewriteInfo{match.out_param_index,
                                                     match.return_index,
                                                     out_tensor_type->shape_,
                                                     pieces.front().window_shape,
                                                     pieces.front().callsite_offsets,
                                                     pieces.front().local_offsets,
                                                     MakeDenseRegion(std::move(pieces)),
                                                     {},
                                                     {},
                                                     match.iter_arg_index});
        continue;
      }

      if (AreExprVectorsEqual(window_shape, out_tensor_type->shape_) && IsAllZeroOffsets(base_offsets) &&
          !include_full_shape_zero_outputs) {
        continue;
      }

      auto output_window_shape = std::move(window_shape);
      auto output_base_offsets = std::move(base_offsets);
      auto output_local_offsets = std::move(local_zero_offsets);
      auto output_piece = MakeDensePiece(output_window_shape, output_base_offsets, output_local_offsets);
      analysis.outputs.push_back(OutputRewriteInfo{match.out_param_index,
                                                   match.return_index,
                                                   out_tensor_type->shape_,
                                                   std::move(output_window_shape),
                                                   std::move(output_base_offsets),
                                                   std::move(output_local_offsets),
                                                   MakeDenseRegion({std::move(output_piece)}),
                                                   {},
                                                   {},
                                                   match.iter_arg_index});
    }

    if (analysis.outputs.empty()) return std::nullopt;

    analysis.inputs = existing_inputs;
    auto aggregate_inputs = AnalyzeAggregateInputWindows(func, existing_inputs, loop);
    analysis.inputs.insert(analysis.inputs.end(), std::make_move_iterator(aggregate_inputs.begin()),
                           std::make_move_iterator(aggregate_inputs.end()));
    return analysis;
  }

  static bool HasAggregateFullShapeZeroOffsetReturnOutputs(
      const FunctionPtr& func, const std::vector<size_t>& out_indices,
      const std::vector<InputRewriteInfo>& existing_inputs) {
    auto analysis = AnalyzeAggregateWindowLoop(func, out_indices, existing_inputs,
                                               /*include_full_shape_zero_outputs=*/true);
    if (!analysis.has_value() || analysis->outputs.size() != out_indices.size()) return false;

    for (const auto& out_index : out_indices) {
      auto out_tensor_type = As<TensorType>(func->params_[out_index]->GetType());
      if (!out_tensor_type) return false;
      auto it = std::find_if(
          analysis->outputs.begin(), analysis->outputs.end(),
          [out_index](const OutputRewriteInfo& info) { return info.out_param_index == out_index; });
      if (it == analysis->outputs.end()) return false;
      if (!AreExprVectorsEqual(it->window_shape, out_tensor_type->shape_) ||
          !IsAllZeroOffsets(it->callsite_offsets)) {
        return false;
      }
    }
    return true;
  }

  AnalysisMap Analyze(const ProgramPtr& program) {
    AnalysisMap analyses;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || pypto::codegen::IsBuiltinOp(func->name_) ||
          func->func_type_ == FunctionType::Orchestration || func->func_type_ == FunctionType::Inline) {
        continue;
      }

      auto out_indices = CollectOutParamIndices(func);
      auto input_windows = AnalyzeInputWindows(func);
      if (window_rewrite_policy_ == WindowRewritePolicy::All) {
        auto dynamic_input_windows = AnalyzeDynamicIndexedInputWindows(func);
        for (auto& dynamic_input : dynamic_input_windows) {
          auto duplicate =
              std::find_if(input_windows.begin(), input_windows.end(), [&](const InputRewriteInfo& existing) {
                return existing.in_param_index == dynamic_input.in_param_index;
              });
          if (duplicate == input_windows.end()) input_windows.push_back(std::move(dynamic_input));
        }
      }
      if (out_indices.empty()) {
        if (!input_windows.empty()) {
          CalleeRewriteAnalysis analysis;
          analysis.kind = RewriteKind::FinalStore;
          analysis.inputs = std::move(input_windows);
          analyses.emplace(func->name_, std::move(analysis));
        }
        continue;
      }

      CalleeRewriteAnalysis analysis;
      for (const auto& out_index : out_indices) {
        auto info = AnalyzeFinalStore(func, out_index);
        if (!info.has_value()) {
          continue;
        }

        auto out_tensor_type = As<TensorType>(func->params_[out_index]->GetType());
        if (!out_tensor_type) {
          continue;
        }
        if (AreExprVectorsEqual(info->window_shape, out_tensor_type->shape_) &&
            IsAllZeroOffsets(info->offsets)) {
          continue;
        }

        auto allowed_params = CollectAllowedVars(func->params_);
        if (!ExprsReferenceOnlyVarsIn(info->window_shape, allowed_params) ||
            !ExprsReferenceOnlyVarsIn(info->offsets, allowed_params)) {
          continue;
        }

        std::vector<ExprPtr> local_zero_offsets;
        local_zero_offsets.reserve(info->offsets.size());
        for (size_t i = 0; i < info->offsets.size(); ++i) {
          local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
        }
        auto output_piece = MakeDensePiece(info->window_shape, info->offsets, local_zero_offsets);
        analysis.outputs.push_back(OutputRewriteInfo{out_index,
                                                     info->return_index,
                                                     out_tensor_type->shape_,
                                                     info->window_shape,
                                                     info->offsets,
                                                     local_zero_offsets,
                                                     MakeDenseRegion({std::move(output_piece)}),
                                                     {},
                                                     {},
                                                     SIZE_MAX});
      }
      if (!analysis.outputs.empty()) {
        analysis.kind = RewriteKind::FinalStore;
        analysis.inputs = std::move(input_windows);
        analyses.emplace(func->name_, std::move(analysis));
        continue;
      }

      auto aggregate_analysis = AnalyzeAggregateWindowLoop(func, out_indices, input_windows);
      if (aggregate_analysis.has_value() && !aggregate_analysis->outputs.empty()) {
        analyses.emplace(func->name_, std::move(*aggregate_analysis));
        continue;
      }

      if (!input_windows.empty() &&
          (HasOnlyFullShapeZeroOffsetReturnOutputs(func, out_indices) ||
           HasAggregateFullShapeZeroOffsetReturnOutputs(func, out_indices, input_windows))) {
        CalleeRewriteAnalysis input_only_analysis;
        input_only_analysis.kind = RewriteKind::FinalStore;
        input_only_analysis.inputs = std::move(input_windows);
        analyses.emplace(func->name_, std::move(input_only_analysis));
        continue;
      }

      if (!input_windows.empty() && window_rewrite_policy_ == WindowRewritePolicy::All &&
          HasDynamicIndexedInputWindow(input_windows)) {
        CalleeRewriteAnalysis input_only_analysis;
        input_only_analysis.kind = RewriteKind::FinalStore;
        input_only_analysis.inputs = std::move(input_windows);
        analyses.emplace(func->name_, std::move(input_only_analysis));
      }
    }
    return analyses;
  }

  FunctionPtr RewriteCallee(const ProgramPtr& program, const FunctionPtr& func,
                            const CalleeRewriteAnalysis& analysis) {
    if (!func) return nullptr;

    std::vector<VarPtr> new_params;
    new_params.reserve(func->params_.size());
    std::vector<TypePtr> new_return_types = func->return_types_;
    std::vector<ParamDirection> new_param_directions;
    new_param_directions.reserve(func->param_directions_.size());
    std::vector<VarPtr> primary_new_param_by_old_index(func->params_.size());
    std::unordered_map<size_t, std::vector<VarPtr>> output_piece_params_by_old_index;
    std::unordered_map<size_t, std::vector<size_t>> output_piece_return_indices_by_old_index;
    std::unordered_map<size_t, VarPtr> output_dynamic_base_params_by_old_index;
    std::unordered_map<size_t, VarPtr> output_dynamic_extent_params_by_old_index;
    std::unordered_map<size_t, VarPtr> output_dynamic_extent_dims_by_old_index;
    std::unordered_map<size_t, VarPtr> dynamic_base_params_by_old_index;
    std::unordered_map<size_t, VarPtr> dynamic_extent_params_by_old_index;
    std::unordered_map<size_t, VarPtr> dynamic_extent_dims_by_old_index;

    std::unordered_map<const Var*, ExprPtr> seed;
    for (size_t i = 0; i < func->params_.size(); ++i) {
      auto param_type = func->params_[i]->GetType();
      auto rewrite_it =
          std::find_if(analysis.outputs.begin(), analysis.outputs.end(),
                       [i](const OutputRewriteInfo& info) { return info.out_param_index == i; });
      if (rewrite_it != analysis.outputs.end()) {
        auto out_tensor_type = As<TensorType>(func->params_[i]->GetType());
        if (!out_tensor_type) return nullptr;
        const auto& pieces = DensePieces(*rewrite_it);
        if (pieces.empty()) return nullptr;

        std::vector<VarPtr> piece_params;
        std::vector<size_t> piece_return_indices;
        piece_params.reserve(pieces.size());
        piece_return_indices.reserve(pieces.size());
        for (size_t piece_index = 0; piece_index < pieces.size(); ++piece_index) {
          const auto& piece = pieces[piece_index];
          std::vector<ExprPtr> piece_window_shape = piece.window_shape;
          if (piece_index == 0 && HasFineAssemblePieces(*rewrite_it)) {
            auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
            auto base_param = std::make_shared<Var>(func->params_[i]->name_hint_ + "__window_base",
                                                    index_type, func->params_[i]->span_);
            auto extent_param = std::make_shared<Var>(func->params_[i]->name_hint_ + "__window_extent",
                                                      index_type, func->params_[i]->span_);
            auto extent_dim = std::make_shared<Var>(func->params_[i]->name_hint_ + "__window_extent_dyn",
                                                    index_type, func->params_[i]->span_);
            new_params.push_back(base_param);
            new_param_directions.push_back(ParamDirection::In);
            new_params.push_back(extent_param);
            new_param_directions.push_back(ParamDirection::In);
            output_dynamic_base_params_by_old_index.emplace(i, base_param);
            output_dynamic_extent_params_by_old_index.emplace(i, extent_param);
            output_dynamic_extent_dims_by_old_index.emplace(i, extent_dim);
            if (piece_window_shape.empty()) return nullptr;
            piece_window_shape[0] = extent_dim;
          }
          auto piece_type =
              MakeWindowTensorType(out_tensor_type, rewrite_it->parent_shape, piece_window_shape);
          if (!piece_type) return nullptr;
          auto name_hint = func->params_[i]->name_hint_;
          if (piece_index > 0) name_hint += "_piece" + std::to_string(piece_index);
          auto new_param = std::make_shared<Var>(name_hint, piece_type, func->params_[i]->span_);
          new_params.push_back(new_param);
          new_param_directions.push_back(func->param_directions_[i]);
          piece_params.push_back(new_param);

          size_t piece_return_index = rewrite_it->return_index;
          if (piece_index == 0) {
            new_return_types[piece_return_index] = piece_type;
          } else {
            piece_return_index = new_return_types.size();
            new_return_types.push_back(piece_type);
          }
          piece_return_indices.push_back(piece_return_index);
        }

        primary_new_param_by_old_index[i] = piece_params.front();
        output_piece_params_by_old_index.emplace(i, std::move(piece_params));
        output_piece_return_indices_by_old_index.emplace(i, std::move(piece_return_indices));
        seed[func->params_[i].get()] = primary_new_param_by_old_index[i];
        continue;
      }
      auto input_rewrite_it =
          std::find_if(analysis.inputs.begin(), analysis.inputs.end(),
                       [i](const InputRewriteInfo& info) { return info.in_param_index == i; });
      if (input_rewrite_it != analysis.inputs.end()) {
        auto in_tensor_type = As<TensorType>(func->params_[i]->GetType());
        if (!in_tensor_type) return nullptr;
        const auto& pieces = DensePieces(*input_rewrite_it);
        if (pieces.size() != 1) return nullptr;
        std::vector<ExprPtr> window_shape = pieces.front().window_shape;
        if (input_rewrite_it->dynamic_indexed_window.has_value()) {
          auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
          auto base_param = std::make_shared<Var>(func->params_[i]->name_hint_ + "__window_base", index_type,
                                                  func->params_[i]->span_);
          auto extent_param = std::make_shared<Var>(func->params_[i]->name_hint_ + "__window_extent",
                                                    index_type, func->params_[i]->span_);
          auto extent_dim = std::make_shared<Var>(func->params_[i]->name_hint_ + "__window_extent_dyn",
                                                  index_type, func->params_[i]->span_);
          new_params.push_back(base_param);
          new_param_directions.push_back(ParamDirection::In);
          new_params.push_back(extent_param);
          new_param_directions.push_back(ParamDirection::In);
          dynamic_base_params_by_old_index.emplace(i, base_param);
          dynamic_extent_params_by_old_index.emplace(i, extent_param);
          dynamic_extent_dims_by_old_index.emplace(i, extent_dim);
          const size_t dynamic_dim = input_rewrite_it->dynamic_indexed_window->dynamic_dim;
          if (dynamic_dim >= window_shape.size()) return nullptr;
          window_shape[dynamic_dim] = extent_dim;
        }
        param_type = MakeWindowTensorType(in_tensor_type, input_rewrite_it->parent_shape, window_shape);
        if (!param_type) return nullptr;
      }

      auto new_param =
          std::make_shared<Var>(func->params_[i]->name_hint_, param_type, func->params_[i]->span_);
      new_params.push_back(new_param);
      new_param_directions.push_back(func->param_directions_[i]);
      primary_new_param_by_old_index[i] = new_param;
      seed[func->params_[i].get()] = new_param;
    }
    if (!output_dynamic_extent_dims_by_old_index.empty()) {
      output_dynamic_extent_dims_by_func_[func->name_] = output_dynamic_extent_dims_by_old_index;
    } else {
      output_dynamic_extent_dims_by_func_.erase(func->name_);
    }

    auto cloned_name = MakeUniqueFunctionName(program, func->name_ + "__windowed");
    auto cloned = DeepClone(func->body_, seed);
    std::unordered_map<const Var*, ExprPtr> body_subst = seed;
    for (const auto& [old_var, new_var] : cloned.var_map) {
      body_subst[old_var] = new_var;
    }

    std::vector<OutputRewriteInfo> localized_outputs = analysis.outputs;
    for (auto& output : localized_outputs) {
      auto return_it = output_piece_return_indices_by_old_index.find(output.out_param_index);
      if (return_it != output_piece_return_indices_by_old_index.end()) {
        output.piece_return_indices = return_it->second;
      }
      auto output_base_it = output_dynamic_base_params_by_old_index.find(output.out_param_index);
      auto output_extent_it = output_dynamic_extent_params_by_old_index.find(output.out_param_index);
      if (output_base_it != output_dynamic_base_params_by_old_index.end() ||
          output_extent_it != output_dynamic_extent_params_by_old_index.end()) {
        if (output_base_it == output_dynamic_base_params_by_old_index.end() ||
            output_extent_it == output_dynamic_extent_params_by_old_index.end() ||
            output.window_shape.empty() || output.callsite_offsets.empty() ||
            output.region.dense_pieces.empty() || output.region.dense_pieces.front().window_shape.empty() ||
            output.region.dense_pieces.front().callsite_offsets.empty()) {
          return nullptr;
        }
        output.window_shape[0] = output_extent_it->second;
        output.callsite_offsets[0] = output_base_it->second;
        output.region.dense_pieces.front().window_shape[0] = output_extent_it->second;
        output.region.dense_pieces.front().callsite_offsets[0] = output_base_it->second;
      }
      for (auto& offset : output.callsite_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& offset : output.local_store_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& piece : output.region.dense_pieces) {
        for (auto& dim : piece.window_shape) {
          dim = transform_utils::Substitute(dim, body_subst);
        }
        for (auto& offset : piece.callsite_offsets) {
          offset = transform_utils::Substitute(offset, body_subst);
        }
        for (auto& offset : piece.local_offsets) {
          offset = transform_utils::Substitute(offset, body_subst);
        }
      }
      for (auto& piece : output.assemble_region.dense_pieces) {
        for (auto& dim : piece.window_shape) {
          dim = transform_utils::Substitute(dim, body_subst);
        }
        for (auto& offset : piece.callsite_offsets) {
          offset = transform_utils::Substitute(offset, body_subst);
        }
        for (auto& offset : piece.local_offsets) {
          offset = transform_utils::Substitute(offset, body_subst);
        }
      }
    }
    std::vector<InputRewriteInfo> localized_inputs = analysis.inputs;
    for (auto& input : localized_inputs) {
      if (input.dynamic_indexed_window.has_value()) {
        auto base_it = dynamic_base_params_by_old_index.find(input.in_param_index);
        auto extent_it = dynamic_extent_params_by_old_index.find(input.in_param_index);
        if (base_it == dynamic_base_params_by_old_index.end() ||
            extent_it == dynamic_extent_params_by_old_index.end()) {
          return nullptr;
        }
        input.dynamic_base_param = base_it->second;
        input.dynamic_extent_param = extent_it->second;
        const size_t dynamic_dim = input.dynamic_indexed_window->dynamic_dim;
        if (dynamic_dim >= input.window_shape.size()) return nullptr;
        input.window_shape[dynamic_dim] = input.dynamic_extent_param;
        for (auto& piece : input.region.dense_pieces) {
          if (dynamic_dim >= piece.window_shape.size()) return nullptr;
          piece.window_shape[dynamic_dim] = input.dynamic_extent_param;
        }
      }
      for (auto& offset : input.callsite_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& offset : input.local_read_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& piece : input.region.dense_pieces) {
        for (auto& dim : piece.window_shape) {
          dim = transform_utils::Substitute(dim, body_subst);
        }
        for (auto& offset : piece.callsite_offsets) {
          offset = transform_utils::Substitute(offset, body_subst);
        }
        for (auto& offset : piece.local_offsets) {
          offset = transform_utils::Substitute(offset, body_subst);
        }
      }
    }
    StmtPtr new_body = cloned.cloned_body;

    if (analysis.kind == RewriteKind::AggregateWindowLoop) {
      auto find_aggregate_loop = [&](const StmtPtr& body) -> ForStmtPtr {
        auto body_stmts = FlattenToStmts(body);
        auto ret_stmt = body_stmts.empty() ? nullptr : As<ReturnStmt>(body_stmts.back());
        if (!ret_stmt) return nullptr;

        ForStmtPtr matched_loop;
        for (const auto& stmt : body_stmts) {
          auto candidate = As<ForStmt>(stmt);
          if (!candidate) continue;

          bool matches_outputs = true;
          for (const auto& output : analysis.outputs) {
            if (output.iter_arg_index >= candidate->iter_args_.size() ||
                output.iter_arg_index >= candidate->return_vars_.size() ||
                output.return_index >= ret_stmt->value_.size()) {
              matches_outputs = false;
              break;
            }
            auto init_var = AsVarLike(candidate->iter_args_[output.iter_arg_index]->initValue_);
            auto returned = AsVarLike(ret_stmt->value_[output.return_index]);
            if (!init_var || !returned) {
              matches_outputs = false;
              break;
            }
            auto direct_return_param = output.out_param_index < primary_new_param_by_old_index.size()
                                           ? primary_new_param_by_old_index[output.out_param_index]
                                           : nullptr;
            if (output.out_param_index >= primary_new_param_by_old_index.size() ||
                init_var.get() != primary_new_param_by_old_index[output.out_param_index].get() ||
                (returned.get() != candidate->return_vars_[output.iter_arg_index].get() &&
                 (!direct_return_param || returned.get() != direct_return_param.get()))) {
              matches_outputs = false;
              break;
            }
          }
          if (!matches_outputs) continue;
          if (matched_loop) return nullptr;
          matched_loop = candidate;
        }
        return matched_loop;
      };

      auto cloned_loop = find_aggregate_loop(new_body);
      if (!cloned_loop) return nullptr;

      std::unordered_map<const Var*, TypePtr> narrowed_return_vars;
      for (const auto& output : analysis.outputs) {
        if (output.iter_arg_index >= cloned_loop->return_vars_.size()) {
          return nullptr;
        }
        narrowed_return_vars.emplace(cloned_loop->return_vars_[output.iter_arg_index].get(),
                                     new_return_types[output.return_index]);
      }

      class AggregateLoopTypeLocalizer : public IRMutator {
       public:
        explicit AggregateLoopTypeLocalizer(
            const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars)
            : narrowed_return_vars_(narrowed_return_vars) {}

       protected:
        StmtPtr VisitStmt_(const ForStmtPtr& op) override {
          std::vector<const Var*> old_iter_args_to_erase;
          bool changed = false;
          for (size_t i = 0; i < op->return_vars_.size() && i < op->iter_args_.size(); ++i) {
            auto it = narrowed_return_vars_.find(op->return_vars_[i].get());
            if (it == narrowed_return_vars_.end()) continue;
            auto old_iter = op->iter_args_[i];
            auto old_ret = op->return_vars_[i];
            auto new_iter = std::make_shared<IterArg>(old_iter->name_hint_, it->second, old_iter->initValue_,
                                                      old_iter->span_);
            auto new_ret = std::make_shared<Var>(old_ret->name_hint_, it->second, old_ret->span_);
            var_remap_[old_iter.get()] = new_iter;
            var_remap_[old_ret.get()] = new_ret;
            old_iter_args_to_erase.push_back(old_iter.get());
            changed = true;
          }
          auto new_stmt = IRMutator::VisitStmt_(op);
          for (const auto* old_iter : old_iter_args_to_erase) {
            var_remap_.erase(old_iter);
          }
          return new_stmt;
        }

       private:
        const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars_;
      };

      AggregateLoopTypeLocalizer type_localizer(narrowed_return_vars);
      new_body = type_localizer.VisitStmt(new_body);

      auto typed_loop = find_aggregate_loop(new_body);
      if (!typed_loop) return nullptr;

      if (std::any_of(localized_outputs.begin(), localized_outputs.end(),
                      [](const OutputRewriteInfo& output) { return DensePieces(output).size() > 1; })) {
        class StaticPieceLoopExternalizer : public IRMutator {
         public:
          StaticPieceLoopExternalizer(
              ForStmtPtr target_loop, std::vector<OutputRewriteInfo> outputs,
              std::unordered_map<size_t, std::vector<VarPtr>> piece_params_by_old_index)
              : target_loop_(std::move(target_loop)),
                outputs_(std::move(outputs)),
                piece_params_by_old_index_(std::move(piece_params_by_old_index)) {
            for (size_t output_index = 0; output_index < outputs_.size(); ++output_index) {
              const auto& output = outputs_[output_index];
              output_by_iter_arg_index_[output.iter_arg_index] = output_index;
              if (DensePieces(output).size() > 1) {
                multi_output_by_return_index_[output.return_index] = output_index;
              }
            }
          }

          bool failed() const { return failed_; }
          bool rewrote_loop() const { return rewrote_loop_; }

         protected:
          ExprPtr VisitExpr_(const VarPtr& op) override {
            auto it = return_var_remap_.find(op.get());
            if (it != return_var_remap_.end()) return it->second;
            return IRMutator::VisitExpr_(op);
          }

          StmtPtr VisitStmt_(const ForStmtPtr& op) override {
            if (op.get() != target_loop_.get()) return IRMutator::VisitStmt_(op);
            rewrote_loop_ = true;

            auto trip_count = GetKnownPositiveTripCount(op);
            if (!trip_count.has_value() || *trip_count <= 0) return MarkFailed(op);

            std::vector<ExprPtr> current_values;
            current_values.reserve(op->iter_args_.size());
            for (const auto& iter_arg : op->iter_args_) {
              current_values.push_back(iter_arg->initValue_);
            }

            std::vector<StmtPtr> unrolled_stmts;
            for (int64_t trip = 0; trip < *trip_count; ++trip) {
              auto loop_value = GetLoopValueAtTrip(op, trip);
              if (!loop_value.has_value()) return MarkFailed(op);

              for (const auto& [iter_arg_index, output_index] : output_by_iter_arg_index_) {
                if (iter_arg_index >= current_values.size()) return MarkFailed(op);
                const auto& output = outputs_[output_index];
                if (DensePieces(output).size() <= 1) continue;
                auto params_it = piece_params_by_old_index_.find(output.out_param_index);
                if (params_it == piece_params_by_old_index_.end() ||
                    static_cast<size_t>(trip) >= params_it->second.size()) {
                  return MarkFailed(op);
                }
                current_values[iter_arg_index] = params_it->second[static_cast<size_t>(trip)];
              }

              std::unordered_map<const Var*, ExprPtr> sub_map;
              sub_map[op->loop_var_.get()] = *loop_value;
              for (size_t i = 0; i < op->iter_args_.size(); ++i) {
                sub_map[op->iter_args_[i].get()] = current_values[i];
              }

              auto cloned = DeepClone(op->body_, sub_map);
              auto force_sub_map = sub_map;
              for (size_t i = 0; i < op->iter_args_.size() && i < current_values.size(); ++i) {
                auto cloned_iter_it = cloned.var_map.find(op->iter_args_[i].get());
                if (cloned_iter_it != cloned.var_map.end()) {
                  force_sub_map[cloned_iter_it->second.get()] = current_values[i];
                }
              }
              auto iteration_body = ForceSubstituteExprRefs(cloned.cloned_body, force_sub_map);
              auto localized_body =
                  LocalizeIteration(iteration_body, current_values, static_cast<size_t>(trip));
              if (!localized_body.has_value()) return MarkFailed(op);
              auto body_stmts = FlattenToStmts(*localized_body);
              if (body_stmts.empty()) return MarkFailed(op);
              auto yield = As<YieldStmt>(body_stmts.back());
              if (!yield || yield->value_.size() != op->iter_args_.size()) return MarkFailed(op);

              for (size_t i = 0; i + 1 < body_stmts.size(); ++i) {
                unrolled_stmts.push_back(body_stmts[i]);
              }

              for (size_t i = 0; i < yield->value_.size(); ++i) {
                auto output_it = output_by_iter_arg_index_.find(i);
                if (output_it != output_by_iter_arg_index_.end() &&
                    DensePieces(outputs_[output_it->second]).size() > 1) {
                  final_piece_values_[outputs_[output_it->second].return_index].push_back(yield->value_[i]);
                  continue;
                }
                current_values[i] = yield->value_[i];
              }
            }

            for (size_t i = 0; i < op->return_vars_.size() && i < current_values.size(); ++i) {
              return_var_remap_[op->return_vars_[i].get()] = current_values[i];
            }
            return std::make_shared<SeqStmts>(std::move(unrolled_stmts), op->span_);
          }

          StmtPtr VisitStmt_(const ReturnStmtPtr& op) override {
            std::vector<ExprPtr> new_values;
            std::vector<std::pair<size_t, ExprPtr>> extra_piece_values;
            bool changed = false;
            for (size_t i = 0; i < op->value_.size(); ++i) {
              auto multi_it = multi_output_by_return_index_.find(i);
              if (multi_it != multi_output_by_return_index_.end()) {
                const auto& output = outputs_[multi_it->second];
                auto final_it = final_piece_values_.find(output.return_index);
                if (final_it == final_piece_values_.end() ||
                    final_it->second.size() != DensePieces(output).size() ||
                    output.piece_return_indices.size() != DensePieces(output).size()) {
                  return MarkFailed(op);
                }
                new_values.push_back(final_it->second.front());
                for (size_t piece_index = 1; piece_index < final_it->second.size(); ++piece_index) {
                  extra_piece_values.emplace_back(output.piece_return_indices[piece_index],
                                                  final_it->second[piece_index]);
                }
                changed = true;
                continue;
              }
              auto new_value = VisitExpr(op->value_[i]);
              if (new_value.get() != op->value_[i].get()) changed = true;
              new_values.push_back(new_value);
            }
            std::sort(extra_piece_values.begin(), extra_piece_values.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            for (const auto& [_, value] : extra_piece_values) {
              new_values.push_back(value);
            }
            if (!changed) return op;
            auto result = MutableCopy(op);
            result->value_ = std::move(new_values);
            return result;
          }

         private:
          StmtPtr MarkFailed(const StmtPtr& fallback) {
            failed_ = true;
            return fallback;
          }

          static StmtPtr ForceSubstituteExprRefs(
              const StmtPtr& stmt, const std::unordered_map<const Var*, ExprPtr>& replacements) {
            class Replacer : public IRMutator {
             public:
              explicit Replacer(const std::unordered_map<const Var*, ExprPtr>& replacements)
                  : replacements_(replacements) {}

             protected:
              ExprPtr VisitExpr_(const VarPtr& op) override {
                auto it = replacements_.find(op.get());
                if (it != replacements_.end()) return it->second;
                return IRMutator::VisitExpr_(op);
              }

              ExprPtr VisitExpr_(const IterArgPtr& op) override {
                auto it = replacements_.find(op.get());
                if (it != replacements_.end()) return it->second;
                return IRMutator::VisitExpr_(op);
              }

             private:
              const std::unordered_map<const Var*, ExprPtr>& replacements_;
            };

            Replacer replacer(replacements);
            return replacer.VisitStmt(stmt);
          }

          std::optional<StmtPtr> LocalizeIteration(const StmtPtr& body,
                                                   const std::vector<ExprPtr>& current_values,
                                                   size_t trip) const {
            std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
            std::unordered_map<const Var*, ExprPtr> new_out_vars;
            for (const auto& [iter_arg_index, output_index] : output_by_iter_arg_index_) {
              if (iter_arg_index >= current_values.size()) return std::nullopt;
              const auto& output = outputs_[output_index];
              const auto& pieces = DensePieces(output);
              const size_t piece_index = pieces.size() > 1 ? trip : 0;
              if (piece_index >= pieces.size()) return std::nullopt;
              auto target_var = AsVarLike(current_values[iter_arg_index]);
              if (!target_var) return std::nullopt;

              OutputRewriteInfo piece_info = output;
              piece_info.window_shape = pieces[piece_index].window_shape;
              piece_info.callsite_offsets = pieces[piece_index].callsite_offsets;
              piece_info.local_store_offsets = pieces[piece_index].local_offsets;
              piece_info.region = MakeDenseRegion({pieces[piece_index]});
              out_info_by_var.emplace(target_var.get(), std::move(piece_info));
              new_out_vars.emplace(target_var.get(), target_var);
            }

            WindowWriteLocalizer localizer(out_info_by_var, new_out_vars);
            return localizer.VisitStmt(body);
          }

          ForStmtPtr target_loop_;
          std::vector<OutputRewriteInfo> outputs_;
          std::unordered_map<size_t, std::vector<VarPtr>> piece_params_by_old_index_;
          std::unordered_map<size_t, size_t> output_by_iter_arg_index_;
          std::unordered_map<size_t, size_t> multi_output_by_return_index_;
          std::unordered_map<const Var*, ExprPtr> return_var_remap_;
          std::unordered_map<size_t, std::vector<ExprPtr>> final_piece_values_;
          bool failed_ = false;
          bool rewrote_loop_ = false;
        };

        StaticPieceLoopExternalizer static_piece_externalizer(typed_loop, localized_outputs,
                                                              output_piece_params_by_old_index);
        new_body = static_piece_externalizer.VisitStmt(new_body);
        if (static_piece_externalizer.failed() || !static_piece_externalizer.rewrote_loop()) {
          return nullptr;
        }
      } else {
        std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
        std::unordered_map<const Var*, ExprPtr> new_out_vars;
        for (const auto& output : localized_outputs) {
          if (output.iter_arg_index >= typed_loop->iter_args_.size()) {
            return nullptr;
          }
          auto iter_arg = typed_loop->iter_args_[output.iter_arg_index];
          out_info_by_var.emplace(iter_arg.get(), output);
          new_out_vars.emplace(iter_arg.get(), iter_arg);
        }

        WindowWriteLocalizer localizer(out_info_by_var, new_out_vars);
        new_body = localizer.VisitStmt(new_body);
      }
    } else {
      std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
      std::unordered_map<const Var*, ExprPtr> new_out_vars;
      for (const auto& output : localized_outputs) {
        if (output.out_param_index >= primary_new_param_by_old_index.size()) {
          return nullptr;
        }
        auto new_out = primary_new_param_by_old_index[output.out_param_index];
        out_info_by_var.emplace(new_out.get(), output);
        new_out_vars.emplace(new_out.get(), new_out);
      }
      WindowWriteLocalizer localizer(out_info_by_var, new_out_vars);
      new_body = localizer.VisitStmt(new_body);
    }

    std::unordered_map<const Var*, InputRewriteInfo> in_info_by_var;
    for (const auto& input : localized_inputs) {
      if (input.in_param_index >= primary_new_param_by_old_index.size()) {
        return nullptr;
      }
      in_info_by_var.emplace(primary_new_param_by_old_index[input.in_param_index].get(), input);
    }
    if (!in_info_by_var.empty()) {
      WindowReadLocalizer read_localizer(in_info_by_var);
      new_body = read_localizer.VisitStmt(new_body);
    }

    return std::make_shared<Function>(cloned_name, new_params, new_param_directions, new_return_types,
                                      new_body, func->span_, func->func_type_, func->level_, func->role_,
                                      func->attrs_);
  }

  OutputWindowPolicy output_window_policy_ = OutputWindowPolicy::CoalescePieces;
  WindowRewritePolicy window_rewrite_policy_ = WindowRewritePolicy::Auto;
  std::unordered_map<std::string, std::unordered_map<size_t, VarPtr>> output_dynamic_extent_dims_by_func_;
};

}  // namespace

// ============================================================================
// Pass entry point
// ============================================================================

namespace pass {

Pass OptimizeOrchTensors(std::string output_window_policy, std::string window_rewrite_policy) {
  auto parsed_output_window_policy = OutWindowExternalizer::OutputWindowPolicy::CoalescePieces;
  if (output_window_policy == "exact_pieces") {
    parsed_output_window_policy = OutWindowExternalizer::OutputWindowPolicy::ExactPieces;
  } else if (output_window_policy == "coalesce_pieces") {
    parsed_output_window_policy = OutWindowExternalizer::OutputWindowPolicy::CoalescePieces;
  } else {
    CHECK(false) << "Invalid output_window_policy '" << output_window_policy
                 << "': expected 'exact_pieces' or 'coalesce_pieces'";
  }

  auto parsed_window_rewrite_policy = OutWindowExternalizer::WindowRewritePolicy::Auto;
  if (window_rewrite_policy == "auto") {
    parsed_window_rewrite_policy = OutWindowExternalizer::WindowRewritePolicy::Auto;
  } else if (window_rewrite_policy == "all") {
    parsed_window_rewrite_policy = OutWindowExternalizer::WindowRewritePolicy::All;
  } else if (window_rewrite_policy == "inputs_only" || window_rewrite_policy == "no_outputs") {
    parsed_window_rewrite_policy = OutWindowExternalizer::WindowRewritePolicy::InputsOnly;
  } else if (window_rewrite_policy == "outputs_only" || window_rewrite_policy == "no_inputs") {
    parsed_window_rewrite_policy = OutWindowExternalizer::WindowRewritePolicy::OutputsOnly;
  } else if (window_rewrite_policy == "no_multi_piece_outputs") {
    parsed_window_rewrite_policy = OutWindowExternalizer::WindowRewritePolicy::NoMultiPieceOutputs;
  } else if (window_rewrite_policy == "none" || window_rewrite_policy == "disabled") {
    parsed_window_rewrite_policy = OutWindowExternalizer::WindowRewritePolicy::None;
  } else {
    CHECK(false) << "Invalid window_rewrite_policy '" << window_rewrite_policy
                 << "': expected 'auto', 'all', 'inputs_only', 'outputs_only', 'no_inputs', "
                    "'no_outputs', 'no_multi_piece_outputs', or 'none'";
  }

  auto pass_func = [parsed_output_window_policy,
                    parsed_window_rewrite_policy](const ProgramPtr& program) -> ProgramPtr {
    // Collect InCore function names
    std::unordered_set<std::string> incore_names;
    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        incore_names.insert(func->name_);
      }
    }

    // Pattern 1: Iter-arg reuse (may remove Out params)
    auto p1 = IterArgReuseOptimizer().Run(program, incore_names);

    // Pattern 2: Assemble parent strides (sees Pattern 1 results)
    auto p2 = AssembleParentStridesOptimizer().Run(p1, incore_names);

    // Pattern 3: Assemble-loop rewrite (sees Pattern 2 results)
    auto p3 = AssembleLoopRewriter().Run(p2, incore_names);

    // Pattern 4: Slice input strides (propagate parent strides to In params)
    auto p4 = SliceInputStridesOptimizer().Run(p3, incore_names);

    // Pattern 5: Static out-window externalization for statically provable
    // local-window writes in outlined callees.
    return OutWindowExternalizer(parsed_output_window_policy, parsed_window_rewrite_policy).Run(p4);
  };

  return CreateProgramPass(pass_func, "OptimizeOrchTensors", kOptimizeOrchTensorsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
