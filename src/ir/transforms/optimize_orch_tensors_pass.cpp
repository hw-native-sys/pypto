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
#include <memory>
#include <optional>
#include <string>
#include <tuple>
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

std::unordered_map<const Var*, size_t> CountAllVarRefsInStmt(const StmtPtr& stmt) {
  class Counter : public IRVisitor {
   public:
    [[nodiscard]] const std::unordered_map<const Var*, size_t>& counts() const { return counts_; }

   protected:
    void VisitExpr_(const VarPtr& op) override {
      ++counts_[op.get()];
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const IterArgPtr& op) override {
      ++counts_[op.get()];
      IRVisitor::VisitExpr_(op);
    }

   private:
    std::unordered_map<const Var*, size_t> counts_;
  };

  Counter counter;
  counter.VisitStmt(stmt);
  return counter.counts();
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

std::vector<size_t> CollectOutParamIndices(const FunctionPtr& func, bool include_inout = false) {
  std::vector<size_t> result;
  if (!func) return result;
  for (size_t i = 0; i < func->param_directions_.size() && i < func->params_.size(); ++i) {
    if (IsOutputDirection(func->param_directions_[i], include_inout)) {
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

bool IsDirectParamForwardingCall(const FunctionPtr& wrapper, const CallPtr& call, const FunctionPtr& inner) {
  if (!wrapper || !call || !inner) return false;
  if (inner->params_.size() != wrapper->params_.size() ||
      inner->param_directions_.size() != wrapper->param_directions_.size() ||
      call->args_.size() != wrapper->params_.size()) {
    return false;
  }
  for (size_t i = 0; i < wrapper->params_.size(); ++i) {
    auto arg = AsVarLike(call->args_[i]);
    if (!arg || arg.get() != wrapper->params_[i].get()) return false;
  }
  return true;
}

template <typename ResolveArgRoot>
std::vector<const Var*> BuildCallOutputRoots(const ProgramPtr& program, const CallPtr& call,
                                             ResolveArgRoot resolve_arg_root) {
  auto callee = program && call ? program->GetFunction(call->op_->name_) : nullptr;
  if (!callee) return {};

  std::vector<const Var*> roots(callee->return_types_.size(), nullptr);
  auto apply_mapping = [&](const FunctionPtr& mapped_func, const CallPtr& mapped_call) {
    for (const auto& mapping : BuildOutParamReturnMappings(mapped_func, /*include_inout=*/true)) {
      if (mapping.return_index >= roots.size() || mapping.param_index >= mapped_call->args_.size()) continue;
      roots[mapping.return_index] = resolve_arg_root(mapped_call->args_[mapping.param_index]);
    }
  };

  apply_mapping(callee, call);
  if (callee->func_type_ != FunctionType::Spmd) return roots;

  auto body_stmts = FlattenToStmts(callee->body_);
  if (body_stmts.size() != 2) return roots;
  auto wrapper_assign = As<AssignStmt>(body_stmts[0]);
  auto wrapper_return = As<ReturnStmt>(body_stmts[1]);
  auto inner_call = wrapper_assign ? As<Call>(wrapper_assign->value_) : nullptr;
  if (!wrapper_assign || !wrapper_return || !inner_call || wrapper_return->value_.size() != 1) return roots;
  auto returned = AsVarLike(wrapper_return->value_[0]);
  if (!returned || returned.get() != wrapper_assign->var_.get()) return roots;
  auto inner = program->GetFunction(GetCallFuncName(inner_call));
  if (!IsDirectParamForwardingCall(callee, inner_call, inner)) return roots;

  std::unordered_map<const Var*, const Var*> wrapper_param_roots;
  for (size_t i = 0; i < callee->params_.size() && i < call->args_.size(); ++i) {
    wrapper_param_roots[callee->params_[i].get()] = resolve_arg_root(call->args_[i]);
  }
  for (const auto& mapping : BuildOutParamReturnMappings(inner, /*include_inout=*/true)) {
    if (mapping.return_index >= roots.size() || mapping.param_index >= inner_call->args_.size()) continue;
    auto wrapper_arg = AsVarLike(inner_call->args_[mapping.param_index]);
    if (!wrapper_arg) continue;
    auto root_it = wrapper_param_roots.find(wrapper_arg.get());
    if (root_it != wrapper_param_roots.end()) roots[mapping.return_index] = root_it->second;
  }
  return roots;
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

      std::shared_ptr<Call> new_call;
      if (new_return_type) {
        new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, new_return_type, call->span_);
      } else {
        new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->span_);
      }

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
//   aggregate window into one or more Out params, and every rewritten Out can
//   be proven disjoint across sequential sibling callsites.
//
// Multi-Out policy is all-or-nothing: either every Out param is rewritten, or
// the callee stays baseline.
// ============================================================================

class OutWindowExternalizer {
 public:
  ProgramPtr Run(const ProgramPtr& program) {
    auto analyses = Analyze(program);
    if (analyses.empty()) return program;

    std::unordered_map<std::string, FunctionPtr> cloned_funcs;
    for (const auto& [func_name, analysis] : analyses) {
      auto callee = program->GetFunction(func_name);
      if (!callee) continue;
      auto cloned = RewriteCallee(program, callee, analysis);
      if (!cloned) continue;
      cloned_funcs.emplace(func_name, cloned);
    }
    if (cloned_funcs.empty()) return program;

    std::vector<FunctionPtr> new_functions;
    new_functions.reserve(program->functions_.size() + cloned_funcs.size());
    for (const auto& [gvar, func] : program->functions_) {
      new_functions.push_back(func);
      auto clone_it = cloned_funcs.find(func->name_);
      if (clone_it != cloned_funcs.end()) {
        new_functions.push_back(clone_it->second);
      }
    }

    bool changed = false;
    for (auto& func : new_functions) {
      if (!func || func->func_type_ != FunctionType::Orchestration) continue;
      OrchRewriter rewriter(program, analyses, cloned_funcs, func);
      auto new_body = rewriter.VisitStmt(func->body_);
      if (new_body.get() == func->body_.get()) continue;
      changed = true;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }

    if (!changed) return program;
    return std::make_shared<Program>(new_functions, program->comm_groups_, program->name_, program->span_);
  }

 private:
  enum class RewriteKind {
    FinalStore,
    AggregateWindowLoop,
    SpmdRowBlock,
  };

  struct OutputRewriteInfo {
    size_t out_param_index;
    size_t return_index;
    std::vector<ExprPtr> parent_shape;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> callsite_offsets;
    std::vector<ExprPtr> local_store_offsets;
    size_t iter_arg_index = SIZE_MAX;
  };

  struct InputRewriteInfo {
    size_t in_param_index;
    std::vector<ExprPtr> parent_shape;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> callsite_offsets;
    std::vector<ExprPtr> local_slice_offsets;
  };

  struct CalleeRewriteAnalysis {
    RewriteKind kind = RewriteKind::FinalStore;
    std::vector<OutputRewriteInfo> outputs;
    std::vector<InputRewriteInfo> inputs;
    std::optional<int64_t> spmd_core_num;
    FunctionPtr spmd_inner_func;
    VarPtr spmd_block_idx_var;
  };

  struct DirectSpmdCall {
    FunctionPtr inner_func;
    CallPtr call;
    bool returns_value = false;
  };

  using AnalysisMap = std::unordered_map<std::string, CalleeRewriteAnalysis>;

  struct AffineForm {
    int64_t coeff = 0;
    ExprPtr base;
  };

  struct OrderedLoopOffsets {
    ExprPtr min;
    ExprPtr max;
  };

  static std::optional<AffineForm> ParseAffineInLoop(const ExprPtr& expr, const Var* loop_var) {
    if (!expr) return std::nullopt;
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
                         const std::unordered_map<const Var*, ExprPtr>& new_out_vars,
                         const std::unordered_map<const Var*, TypePtr>& new_store_types)
        : out_info_by_var_(out_info_by_var), new_out_vars_(new_out_vars), new_store_types_(new_store_types) {}

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

    StmtPtr VisitStmt_(const IfStmtPtr& op) override {
      auto new_condition = VisitExpr(op->condition_);
      auto incoming_remap = result_var_remap_;
      auto incoming_info = result_var_info_;

      result_var_remap_ = incoming_remap;
      result_var_info_ = incoming_info;
      auto new_then_body = VisitStmt(op->then_body_);
      auto then_info = result_var_info_;

      std::optional<StmtPtr> new_else_body;
      std::unordered_map<const Var*, const OutputRewriteInfo*> else_info;
      if (op->else_body_.has_value()) {
        result_var_remap_ = incoming_remap;
        result_var_info_ = incoming_info;
        new_else_body = VisitStmt(op->else_body_.value());
        else_info = result_var_info_;
      }

      result_var_remap_ = incoming_remap;
      result_var_info_ = incoming_info;

      std::vector<VarPtr> new_return_vars = op->return_vars_;
      bool return_vars_changed = false;
      auto then_yield = transform_utils::GetLastYieldStmt(new_then_body);
      auto else_yield =
          new_else_body.has_value() ? transform_utils::GetLastYieldStmt(*new_else_body) : nullptr;
      if (then_yield && else_yield && then_yield->value_.size() == op->return_vars_.size() &&
          else_yield->value_.size() == op->return_vars_.size()) {
        for (size_t i = 0; i < op->return_vars_.size(); ++i) {
          auto then_var = AsVarLike(then_yield->value_[i]);
          auto else_var = AsVarLike(else_yield->value_[i]);
          if (!then_var || !else_var) continue;
          auto then_it = then_info.find(then_var.get());
          auto else_it = else_info.find(else_var.get());
          if (then_it == then_info.end() || else_it == else_info.end() ||
              then_it->second != else_it->second) {
            continue;
          }
          auto new_ret = std::make_shared<Var>(op->return_vars_[i]->name_hint_, then_var->GetType(),
                                               op->return_vars_[i]->span_);
          result_var_remap_[op->return_vars_[i].get()] = new_ret;
          result_var_info_[op->return_vars_[i].get()] = then_it->second;
          result_var_info_[new_ret.get()] = then_it->second;
          new_return_vars[i] = new_ret;
          return_vars_changed = true;
        }
      }

      if (new_condition.get() != op->condition_.get() || new_then_body.get() != op->then_body_.get() ||
          (new_else_body.has_value() &&
           (!op->else_body_.has_value() || new_else_body->get() != op->else_body_->get())) ||
          return_vars_changed) {
        auto result = MutableCopy(op);
        result->condition_ = std::move(new_condition);
        result->then_body_ = std::move(new_then_body);
        result->else_body_ = new_else_body;
        result->return_vars_ = std::move(new_return_vars);
        return result;
      }
      return op;
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
      } else {
        return assign;
      }

      auto info_it = out_info_by_var_.find(target_var);
      auto chained_info_it = result_var_info_.find(target_var);
      if (info_it == out_info_by_var_.end() && chained_info_it == result_var_info_.end()) return assign;
      if (!offsets) return assign;
      const auto& output_info =
          info_it != out_info_by_var_.end() ? info_it->second : *chained_info_it->second;

      auto new_offset_tuple = std::make_shared<MakeTuple>(output_info.local_store_offsets, offsets->span_);
      std::vector<ExprPtr> new_args = call->args_;
      new_args[offset_arg_index] = new_offset_tuple;
      ExprPtr new_target;
      auto new_target_it = new_out_vars_.find(target_var);
      if (new_target_it != new_out_vars_.end()) {
        new_target = new_target_it->second;
      } else {
        auto remap_it = result_var_remap_.find(target_var);
        if (remap_it != result_var_remap_.end()) new_target = remap_it->second;
      }
      if (!new_target && chained_info_it != result_var_info_.end()) {
        new_target = rewritten_target_expr;
      }
      if (!new_target) return assign;
      new_args[target_arg_index] = new_target;
      auto new_type_it = new_store_types_.find(target_var);
      auto new_type = new_type_it != new_store_types_.end() ? new_type_it->second : new_target->GetType();
      auto new_call =
          std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_, new_type, call->span_);

      auto new_result_var = std::make_shared<Var>(assign->var_->name_hint_, new_type, assign->var_->span_);
      result_var_remap_[assign->var_.get()] = new_result_var;
      result_var_info_[assign->var_.get()] = &output_info;
      result_var_info_[new_result_var.get()] = &output_info;
      assign->var_ = new_result_var;
      assign->value_ = new_call;
      return assign;
    }

   private:
    const std::unordered_map<const Var*, OutputRewriteInfo>& out_info_by_var_;
    const std::unordered_map<const Var*, ExprPtr>& new_out_vars_;
    const std::unordered_map<const Var*, TypePtr>& new_store_types_;
    std::unordered_map<const Var*, VarPtr> result_var_remap_;
    std::unordered_map<const Var*, const OutputRewriteInfo*> result_var_info_;
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
      if (!call) return assign;

      size_t offset_arg_index = SIZE_MAX;
      if (call->op_->name_ == "tensor.slice" && call->args_.size() >= 3) {
        offset_arg_index = 2;
      } else if (call->op_->name_ == "tile.load" && call->args_.size() >= 3) {
        offset_arg_index = 1;
      } else {
        return assign;
      }

      auto parent_var = AsVarLike(call->args_[0]);
      auto info_it = parent_var ? in_info_by_var_.find(parent_var.get()) : in_info_by_var_.end();
      if (info_it == in_info_by_var_.end()) return assign;

      auto offsets = As<MakeTuple>(call->args_[offset_arg_index]);
      if (!offsets) return assign;

      std::vector<ExprPtr> new_args = call->args_;
      new_args[offset_arg_index] =
          std::make_shared<MakeTuple>(info_it->second.local_slice_offsets, offsets->span_);
      assign->value_ = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_,
                                              call->GetType(), call->span_);
      return assign;
    }

   private:
    const std::unordered_map<const Var*, InputRewriteInfo>& in_info_by_var_;
  };

  class OrchRewriter : public IRMutator {
   public:
    using RootSet = std::unordered_set<const Var*>;

    OrchRewriter(ProgramPtr program, const AnalysisMap& analyses,
                 const std::unordered_map<std::string, FunctionPtr>& cloned_funcs,
                 const FunctionPtr& current_func)
        : program_(std::move(program)), analyses_(analyses), cloned_funcs_(cloned_funcs) {
      if (current_func) {
        for (size_t i = 0; i < current_func->params_.size(); ++i) {
          const auto& param = current_func->params_[i];
          if (param && AsTensorTypeLike(param->GetType())) {
            full_buffer_roots_[param.get()] = param.get();
            param_roots_.insert(param.get());
            if (i < current_func->param_directions_.size() &&
                IsOutputDirection(current_func->param_directions_[i], /*include_inout=*/true)) {
              writable_buffer_roots_.insert(param.get());
            }
          }
        }
        PrecomputeFullBufferRoots(current_func->body_);
      }
    }

   protected:
    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      auto saved_loop_iter_init_subst = loop_iter_init_subst_;
      for (const auto& iter_arg : op->iter_args_) {
        if (iter_arg && iter_arg->initValue_) loop_iter_init_subst_[iter_arg.get()] = iter_arg->initValue_;
        if (iter_arg && iter_arg->initValue_) {
          if (const Var* root = ResolveBufferRoot(iter_arg->initValue_)) {
            full_buffer_roots_[iter_arg.get()] = root;
          }
        }
      }

      bool is_sequential = op->kind_ != ForKind::Parallel;
      if (is_sequential) {
        sequential_loops_.push_back(op);
        loop_local_allocs_.emplace_back(CollectLoopLocalTensorAllocs(op));
      }
      auto result = IRMutator::VisitStmt_(op);
      if (is_sequential) {
        loop_local_allocs_.pop_back();
        sequential_loops_.pop_back();
      }
      auto visited_loop = As<ForStmt>(result);
      AddLoopReturnRoots(visited_loop ? visited_loop : op);
      loop_iter_init_subst_ = std::move(saved_loop_iter_init_subst);
      return result;
    }

    StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
      auto saved_loop_iter_init_subst = loop_iter_init_subst_;
      for (const auto& iter_arg : op->iter_args_) {
        if (iter_arg && iter_arg->initValue_) {
          loop_iter_init_subst_[iter_arg.get()] = iter_arg->initValue_;
          if (const Var* root = ResolveBufferRoot(iter_arg->initValue_)) {
            full_buffer_roots_[iter_arg.get()] = root;
          }
        }
      }
      ++while_depth_;
      auto result = IRMutator::VisitStmt_(op);
      --while_depth_;
      auto visited_loop = As<WhileStmt>(result);
      AddLoopReturnRoots(visited_loop ? visited_loop : op);
      loop_iter_init_subst_ = std::move(saved_loop_iter_init_subst);
      return result;
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
      std::vector<StmtPtr> new_stmts;
      new_stmts.reserve(op->stmts_.size());
      bool changed = false;
      auto saved_scalar_defs = scalar_defs_;
      auto saved_tuple_result_subst = tuple_result_subst_;

      RootSet later_reads = enclosing_later_full_parent_reads_;
      std::unordered_map<const Stmt*, RootSet> later_reads_by_stmt;
      later_reads_by_stmt.reserve(op->stmts_.size());
      for (auto stmt_it = op->stmts_.rbegin(); stmt_it != op->stmts_.rend(); ++stmt_it) {
        later_reads_by_stmt.emplace(stmt_it->get(), later_reads);
        AddFullRootReadsFromStmt(*stmt_it, later_reads, /*allow_windowed_call_skip=*/true);
      }

      for (const auto& stmt : op->stmts_) {
        auto saved_enclosing_reads = enclosing_later_full_parent_reads_;
        auto later_it = later_reads_by_stmt.find(stmt.get());
        enclosing_later_full_parent_reads_ =
            later_it != later_reads_by_stmt.end() ? later_it->second : saved_enclosing_reads;

        auto call_assign = As<AssignStmt>(stmt);
        auto bundle = call_assign ? TryRewriteCall(call_assign) : std::nullopt;
        if (!bundle.has_value()) {
          if (auto eval = As<EvalStmt>(stmt)) {
            bundle = TryRewriteEvalCall(eval);
          }
        }
        if (bundle.has_value()) {
          changed = true;
          for (const auto& new_stmt : bundle->stmts) {
            auto visited = VisitStmt(new_stmt);
            if (auto visited_assign = As<AssignStmt>(visited)) {
              AddFullBufferRootForAssign(visited_assign);
              if (As<ScalarType>(visited_assign->var_->GetType())) {
                scalar_defs_[visited_assign->var_.get()] = visited_assign->value_;
              }
            }
            new_stmts.push_back(visited);
          }
          enclosing_later_full_parent_reads_ = std::move(saved_enclosing_reads);
          continue;
        }

        auto visited = VisitStmt(stmt);
        changed = changed || visited.get() != stmt.get();
        new_stmts.push_back(visited);

        auto visited_assign = As<AssignStmt>(visited);
        if (visited_assign) AddFullBufferRootForAssign(visited_assign);
        if (visited_assign && As<ScalarType>(visited_assign->var_->GetType())) {
          scalar_defs_[visited_assign->var_.get()] = visited_assign->value_;
        }
        enclosing_later_full_parent_reads_ = std::move(saved_enclosing_reads);
      }

      scalar_defs_ = std::move(saved_scalar_defs);
      tuple_result_subst_ = std::move(saved_tuple_result_subst);
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
    };

    struct LoopDisjointnessCandidate {
      ForStmtPtr loop;
      const std::unordered_set<const Var*>* loop_local_allocs = nullptr;
    };

    struct RewriteCallsiteCandidate {
      SubmitPtr submit;
      CallPtr call;
      FunctionPtr original_func;
      const CalleeRewriteAnalysis* analysis = nullptr;
      FunctionPtr cloned_func;
    };

    static bool IsTensorTypedExpr(const ExprPtr& expr) {
      return expr && AsTensorTypeLike(expr->GetType()) != nullptr;
    }

    const Var* ResolveBufferRoot(const ExprPtr& expr) const {
      auto current = ResolveLoopInitExpr(expr);
      auto var = AsVarLike(current);
      if (!var) return nullptr;
      return ResolveBufferRoot(var.get());
    }

    const Var* ResolveBufferRoot(const Var* var) const {
      const Var* current = var;
      std::unordered_set<const Var*> seen;
      while (current && seen.insert(current).second) {
        auto it = full_buffer_roots_.find(current);
        if (it == full_buffer_roots_.end()) return nullptr;
        if (it->second == current) return current;
        current = it->second;
      }
      return current;
    }

    bool IsFullRootExpr(const ExprPtr& expr) const {
      auto current = ResolveLoopInitExpr(expr);
      auto var = AsVarLike(current);
      return var && full_buffer_roots_.count(var.get()) > 0 && ResolveBufferRoot(var.get()) != nullptr;
    }

    bool IsWritableRootExpr(const ExprPtr& expr) const {
      auto root = ResolveBufferRoot(expr);
      if (!root) return false;
      if (param_roots_.count(root) == 0) return true;
      return writable_buffer_roots_.count(root) > 0;
    }

    static bool IsReadDirection(ParamDirection direction) {
      return direction == ParamDirection::In || direction == ParamDirection::InOut;
    }

    void AddFullBufferRootForAssign(const AssignStmtPtr& assign) {
      if (!assign) return;
      if (auto call = As<Call>(assign->value_)) {
        const auto& op_name = call->op_->name_;
        if ((op_name == "tensor.reshape" || op_name == "tensor.assemble") && !call->args_.empty()) {
          if (const Var* root = ResolveBufferRoot(call->args_[0])) {
            full_buffer_roots_[assign->var_.get()] = root;
          }
        } else if (IsTensorAllocationOp(call)) {
          full_buffer_roots_[assign->var_.get()] = assign->var_.get();
        } else if (!codegen::IsBuiltinOp(op_name)) {
          AddCallOutputRoots(assign, call);
        }
      } else if (auto submit = As<Submit>(assign->value_)) {
        auto call = SubmitToCallView(submit);
        if (!codegen::IsBuiltinOp(call->op_->name_)) {
          AddCallOutputRoots(assign, call);
        }
      } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
        AddTupleGetItemRoot(assign, tuple_get);
      } else if (auto src_var = AsVarLike(assign->value_)) {
        if (const Var* root = ResolveBufferRoot(src_var.get())) {
          full_buffer_roots_[assign->var_.get()] = root;
        }
      } else if (auto tuple = As<MakeTuple>(assign->value_)) {
        std::vector<const Var*> roots;
        roots.reserve(tuple->elements_.size());
        for (const auto& element : tuple->elements_) {
          roots.push_back(IsTensorTypedArg(element) ? ResolveBufferRoot(element) : nullptr);
        }
        tuple_output_roots_[assign->var_.get()] = std::move(roots);
      }
    }

    void AddCallOutputRoots(const AssignStmtPtr& assign, const CallPtr& call) {
      auto roots =
          BuildCallOutputRoots(program_, call, [this](const ExprPtr& arg) { return ResolveBufferRoot(arg); });
      if (roots.empty()) return;

      if (As<TupleType>(call->GetType())) {
        tuple_output_roots_[assign->var_.get()] = std::move(roots);
      } else if (!roots.empty() && roots[0]) {
        full_buffer_roots_[assign->var_.get()] = roots[0];
      }
    }

    void PrecomputeFullBufferRoots(const StmtPtr& stmt) {
      if (!stmt) return;
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) {
          PrecomputeFullBufferRoots(child);
        }
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        for (const auto& iter_arg : for_stmt->iter_args_) {
          if (iter_arg && iter_arg->initValue_) {
            if (const Var* root = ResolveBufferRoot(iter_arg->initValue_)) {
              full_buffer_roots_[iter_arg.get()] = root;
            }
          }
        }
        PrecomputeFullBufferRoots(for_stmt->body_);
        AddLoopReturnRoots(for_stmt);
      } else if (auto while_stmt = As<WhileStmt>(stmt)) {
        for (const auto& iter_arg : while_stmt->iter_args_) {
          if (iter_arg && iter_arg->initValue_) {
            if (const Var* root = ResolveBufferRoot(iter_arg->initValue_)) {
              full_buffer_roots_[iter_arg.get()] = root;
            }
          }
        }
        PrecomputeFullBufferRoots(while_stmt->body_);
        AddLoopReturnRoots(while_stmt);
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        PrecomputeFullBufferRoots(if_stmt->then_body_);
        if (if_stmt->else_body_.has_value()) {
          PrecomputeFullBufferRoots(if_stmt->else_body_.value());
        }
      } else if (auto scope = As<ScopeStmt>(stmt)) {
        PrecomputeFullBufferRoots(scope->body_);
      } else if (auto assign = As<AssignStmt>(stmt)) {
        PrecomputeFullBufferRootsInExpr(assign->value_);
        AddFullBufferRootForAssign(assign);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        PrecomputeFullBufferRootsInExpr(eval->expr_);
      }
    }

    void PrecomputeFullBufferRootsInExpr(const ExprPtr& expr) {
      if (auto call = As<Call>(expr)) {
        for (const auto& arg : call->args_) {
          PrecomputeFullBufferRootsInExpr(arg);
        }
      } else if (auto tuple = As<MakeTuple>(expr)) {
        for (const auto& element : tuple->elements_) {
          PrecomputeFullBufferRootsInExpr(element);
        }
      } else if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
        PrecomputeFullBufferRootsInExpr(tuple_get->tuple_);
      }
    }

    void AddTupleGetItemRoot(const AssignStmtPtr& assign, const TupleGetItemExprPtr& tuple_get) {
      auto tuple_var = AsVarLike(tuple_get->tuple_);
      if (!tuple_var) return;

      auto subst_it = tuple_result_subst_.find(tuple_var.get());
      if (subst_it != tuple_result_subst_.end() && tuple_get->index_ >= 0 &&
          static_cast<size_t>(tuple_get->index_) < subst_it->second.size()) {
        if (const Var* root = ResolveBufferRoot(subst_it->second[static_cast<size_t>(tuple_get->index_)])) {
          full_buffer_roots_[assign->var_.get()] = root;
        }
        return;
      }

      auto roots_it = tuple_output_roots_.find(tuple_var.get());
      if (roots_it == tuple_output_roots_.end()) return;
      if (tuple_get->index_ < 0 || static_cast<size_t>(tuple_get->index_) >= roots_it->second.size()) return;
      if (const Var* root = roots_it->second[static_cast<size_t>(tuple_get->index_)]) {
        full_buffer_roots_[assign->var_.get()] = root;
      }
    }

    void AddLoopReturnRoots(const ForStmtPtr& loop) {
      if (!loop) return;
      for (size_t i = 0; i < loop->return_vars_.size(); ++i) {
        const Var* root = nullptr;
        if (i < loop->iter_args_.size()) {
          root = ResolveBufferRoot(loop->iter_args_[i].get());
        }
        auto yield = transform_utils::GetLastYieldStmt(loop->body_);
        if (!root && yield && i < yield->value_.size()) {
          root = ResolveBufferRoot(yield->value_[i]);
        }
        if (root) full_buffer_roots_[loop->return_vars_[i].get()] = root;
      }
    }

    void AddLoopReturnRoots(const WhileStmtPtr& loop) {
      if (!loop) return;
      for (size_t i = 0; i < loop->return_vars_.size(); ++i) {
        const Var* root = nullptr;
        if (i < loop->iter_args_.size()) {
          root = ResolveBufferRoot(loop->iter_args_[i].get());
        }
        auto yield = transform_utils::GetLastYieldStmt(loop->body_);
        if (!root && yield && i < yield->value_.size()) {
          root = ResolveBufferRoot(yield->value_[i]);
        }
        if (root) full_buffer_roots_[loop->return_vars_[i].get()] = root;
      }
    }

    bool HasLaterFullParentReadOfRewrittenOutput(const CallPtr& call, const CalleeRewriteAnalysis& analysis,
                                                 const RootSet& reads) const {
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return true;
        const Var* root = ResolveBufferRoot(call->args_[output.out_param_index]);
        if (root && reads.count(root) > 0) {
          return true;
        }
      }
      return false;
    }

    std::optional<RewriteCallsiteCandidate> GetRewriteCallsiteCandidate(
        const AssignStmtPtr& call_assign) const {
      if (!call_assign) return std::nullopt;
      auto submit = As<Submit>(call_assign->value_);
      auto call = submit ? SubmitToCallView(submit) : As<Call>(call_assign->value_);
      if (!call) return std::nullopt;

      auto callee_name = GetCallFuncName(call);
      auto analysis_it = analyses_.find(callee_name);
      if (analysis_it == analyses_.end()) return std::nullopt;
      auto clone_it = cloned_funcs_.find(callee_name);
      if (clone_it == cloned_funcs_.end()) return std::nullopt;
      auto original_func = program_ ? program_->GetFunction(callee_name) : nullptr;
      if (!original_func) return std::nullopt;

      const auto& analysis = analysis_it->second;
      if (analysis.outputs.empty()) return std::nullopt;
      for (const auto& input : analysis.inputs) {
        if (input.in_param_index >= call->args_.size()) return std::nullopt;
        if (!AsVarLike(call->args_[input.in_param_index])) return std::nullopt;
        if (!IsWritableRootExpr(call->args_[input.in_param_index])) return std::nullopt;
      }
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return std::nullopt;
        if (!AsVarLike(call->args_[output.out_param_index])) return std::nullopt;
      }
      if (IsSubmitCall(call)) {
        auto tuple_ty = As<TupleType>(call->GetType());
        if (!tuple_ty || tuple_ty->types_.size() != clone_it->second->return_types_.size() + 1) {
          return std::nullopt;
        }
      }
      if (!ProveCallsiteDisjointness(call_assign, call, analysis)) return std::nullopt;
      return RewriteCallsiteCandidate{submit, call, original_func, &analysis, clone_it->second};
    }

    std::optional<RewriteCallsiteCandidate> GetRewriteCallsiteCandidate(const EvalStmtPtr& eval) const {
      if (!eval) return std::nullopt;
      auto call = As<Call>(eval->expr_);
      if (!call) return std::nullopt;

      auto callee_name = GetCallFuncName(call);
      auto analysis_it = analyses_.find(callee_name);
      if (analysis_it == analyses_.end()) return std::nullopt;
      auto clone_it = cloned_funcs_.find(callee_name);
      if (clone_it == cloned_funcs_.end()) return std::nullopt;
      auto original_func = program_ ? program_->GetFunction(callee_name) : nullptr;
      if (!original_func) return std::nullopt;

      const auto& analysis = analysis_it->second;
      if (analysis.kind != RewriteKind::SpmdRowBlock || analysis.outputs.empty() ||
          !clone_it->second->return_types_.empty()) {
        return std::nullopt;
      }
      for (const auto& input : analysis.inputs) {
        if (input.in_param_index >= call->args_.size()) return std::nullopt;
        if (!AsVarLike(call->args_[input.in_param_index])) return std::nullopt;
        if (!IsWritableRootExpr(call->args_[input.in_param_index])) return std::nullopt;
      }
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return std::nullopt;
        if (!AsVarLike(call->args_[output.out_param_index])) return std::nullopt;
      }
      return RewriteCallsiteCandidate{nullptr, call, original_func, &analysis, clone_it->second};
    }

    bool CanSkipFullRootReadForWindowedCallsite(const AssignStmtPtr& call_assign,
                                                const RootSet& reads) const {
      auto candidate = GetRewriteCallsiteCandidate(call_assign);
      if (!candidate.has_value()) return false;
      return !HasLaterFullParentReadOfRewrittenOutput(candidate->call, *candidate->analysis, reads);
    }

    bool CanSkipFullRootReadForWindowedCallsite(const EvalStmtPtr& eval, const RootSet& reads) const {
      auto candidate = GetRewriteCallsiteCandidate(eval);
      if (!candidate.has_value()) return false;
      return !HasLaterFullParentReadOfRewrittenOutput(candidate->call, *candidate->analysis, reads);
    }

    void AddFullRootReadsFromCall(const AssignStmtPtr& call_assign, const CallPtr& call,
                                  RootSet& reads) const {
      if (!call || !program_ || codegen::IsBuiltinOp(call->op_->name_)) return;
      auto callee_name = call->op_->name_;
      auto callee = program_->GetFunction(callee_name);
      if (!callee) return;
      auto analysis_it = analyses_.find(callee_name);
      const CalleeRewriteAnalysis* analysis =
          analysis_it == analyses_.end() || !cloned_funcs_.count(callee_name) ? nullptr
                                                                              : &analysis_it->second;
      const bool can_skip_windowed_reads =
          analysis && CanSkipFullRootReadForWindowedCallsite(call_assign, reads);
      for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
        if (!IsReadDirection(callee->param_directions_[i])) continue;
        if (can_skip_windowed_reads && HasAnalyzedInputWindow(*analysis, i) &&
            IsWritableRootExpr(call->args_[i])) {
          continue;
        }
        if (!IsFullRootExpr(call->args_[i])) continue;
        if (const Var* root = ResolveBufferRoot(call->args_[i])) {
          reads.insert(root);
        }
      }
    }

    void AddFullRootReadsFromEval(const EvalStmtPtr& eval, RootSet& reads,
                                  bool allow_windowed_call_skip) const {
      if (!eval) return;
      auto call = As<Call>(eval->expr_);
      if (!call) {
        if (auto submit = As<Submit>(eval->expr_)) {
          AddFullRootReadsFromCall(nullptr, SubmitToCallView(submit), reads);
        }
        return;
      }
      if (!call || !program_ || codegen::IsBuiltinOp(call->op_->name_)) return;
      auto callee_name = call->op_->name_;
      auto callee = program_->GetFunction(callee_name);
      if (!callee) return;
      auto analysis_it = analyses_.find(callee_name);
      const CalleeRewriteAnalysis* analysis =
          analysis_it == analyses_.end() || !cloned_funcs_.count(callee_name) ? nullptr
                                                                              : &analysis_it->second;
      const bool can_skip_windowed_reads =
          allow_windowed_call_skip && analysis && CanSkipFullRootReadForWindowedCallsite(eval, reads);
      for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
        if (!IsReadDirection(callee->param_directions_[i])) continue;
        if (can_skip_windowed_reads && HasAnalyzedInputWindow(*analysis, i) &&
            IsWritableRootExpr(call->args_[i])) {
          continue;
        }
        if (!IsFullRootExpr(call->args_[i])) continue;
        if (const Var* root = ResolveBufferRoot(call->args_[i])) {
          reads.insert(root);
        }
      }
    }

    static bool HasAnalyzedInputWindow(const CalleeRewriteAnalysis& analysis, size_t param_index) {
      const auto& inputs = analysis.inputs;
      return std::any_of(inputs.begin(), inputs.end(), [param_index](const InputRewriteInfo& input) {
        return input.in_param_index == param_index;
      });
    }

    // A later reader can be a Call OR a Submit (pl.submit in a manual_scope).
    // Route the Submit through its augmented-Call view so its In/InOut full-root
    // reads are counted by the later-read safety guard — otherwise a windowed
    // submit could be externalized even though a subsequent submit reads the
    // full output (.claude/rules/pass-submit-awareness.md).
    void AddFullRootReadsFromCallLike(const ExprPtr& value, RootSet& reads) {
      if (auto call = As<Call>(value)) {
        AddFullRootReadsFromCall(nullptr, call, reads);
      } else if (auto submit = As<Submit>(value)) {
        AddFullRootReadsFromCall(nullptr, SubmitToCallView(submit), reads);
      }
    }

    void AddFullRootReadsFromStmt(const StmtPtr& stmt, RootSet& reads, bool allow_windowed_call_skip) {
      if (!stmt) return;
      if (auto assign = As<AssignStmt>(stmt)) {
        if (auto call = As<Call>(assign->value_)) {
          AddFullRootReadsFromCall(allow_windowed_call_skip ? assign : nullptr, call, reads);
        } else if (auto submit = As<Submit>(assign->value_)) {
          AddFullRootReadsFromCall(allow_windowed_call_skip ? assign : nullptr, SubmitToCallView(submit),
                                   reads);
        }
      } else if (auto eval = As<EvalStmt>(stmt)) {
        AddFullRootReadsFromEval(eval, reads, allow_windowed_call_skip);
      } else if (auto seq = As<SeqStmts>(stmt)) {
        for (auto it = seq->stmts_.rbegin(); it != seq->stmts_.rend(); ++it) {
          AddFullRootReadsFromStmt(*it, reads, allow_windowed_call_skip);
        }
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        auto saved_loop_iter_init_subst = loop_iter_init_subst_;
        for (const auto& iter_arg : for_stmt->iter_args_) {
          if (!iter_arg || !iter_arg->initValue_) continue;
          loop_iter_init_subst_[iter_arg.get()] = iter_arg->initValue_;
          if (const Var* root = ResolveBufferRoot(iter_arg->initValue_)) {
            full_buffer_roots_[iter_arg.get()] = root;
          }
        }
        bool is_sequential = for_stmt->kind_ != ForKind::Parallel;
        if (is_sequential) {
          sequential_loops_.push_back(for_stmt);
          loop_local_allocs_.emplace_back(CollectLoopLocalTensorAllocs(for_stmt));
        }
        AddFullRootReadsFromStmt(for_stmt->body_, reads, allow_windowed_call_skip);
        if (is_sequential) {
          loop_local_allocs_.pop_back();
          sequential_loops_.pop_back();
        }
        AddLoopReturnRoots(for_stmt);
        loop_iter_init_subst_ = std::move(saved_loop_iter_init_subst);
      } else if (auto while_stmt = As<WhileStmt>(stmt)) {
        auto saved_loop_iter_init_subst = loop_iter_init_subst_;
        for (const auto& iter_arg : while_stmt->iter_args_) {
          if (!iter_arg || !iter_arg->initValue_) continue;
          loop_iter_init_subst_[iter_arg.get()] = iter_arg->initValue_;
          if (const Var* root = ResolveBufferRoot(iter_arg->initValue_)) {
            full_buffer_roots_[iter_arg.get()] = root;
          }
        }
        ++while_depth_;
        AddFullRootReadsFromStmt(while_stmt->body_, reads, allow_windowed_call_skip);
        --while_depth_;
        AddLoopReturnRoots(while_stmt);
        loop_iter_init_subst_ = std::move(saved_loop_iter_init_subst);
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        AddFullRootReadsFromStmt(if_stmt->then_body_, reads, allow_windowed_call_skip);
        if (if_stmt->else_body_.has_value()) {
          AddFullRootReadsFromStmt(if_stmt->else_body_.value(), reads, allow_windowed_call_skip);
        }
      } else if (auto scope = As<ScopeStmt>(stmt)) {
        AddFullRootReadsFromStmt(scope->body_, reads, allow_windowed_call_skip);
      } else if (auto rscope = As<RuntimeScopeStmt>(stmt)) {
        AddFullRootReadsFromStmt(rscope->body_, reads, allow_windowed_call_skip);
      }
    }

    bool HasLaterFullParentReadOfRewrittenOutput(const CallPtr& call,
                                                 const CalleeRewriteAnalysis& analysis) const {
      return HasLaterFullParentReadOfRewrittenOutput(call, analysis, enclosing_later_full_parent_reads_);
    }

    std::optional<RewriteBundle> TryRewriteCall(const AssignStmtPtr& call_assign) {
      // Submit (pl.submit inside pl.manual_scope) is a sibling call-like kind;
      // run the windowing analysis/rewrite on its augmented-Call view, then
      // rebuild as a Submit to preserve task-launch semantics + deps_
      // (.claude/rules/pass-submit-awareness.md). The per-callee analysis and
      // windowed clone are callee-body-driven (Analyze() over all functions),
      // so they exist regardless of the call-site kind.
      auto candidate = GetRewriteCallsiteCandidate(call_assign);
      if (!candidate.has_value()) return std::nullopt;
      auto submit = candidate->submit;
      auto call = candidate->call;
      auto original_func = candidate->original_func;
      const auto& analysis = *candidate->analysis;
      auto cloned_func = candidate->cloned_func;

      if (HasLaterFullParentReadOfRewrittenOutput(call, analysis)) return std::nullopt;
      if (analysis.kind == RewriteKind::SpmdRowBlock) {
        return TryRewriteSpmdRowBlock(call_assign, *candidate);
      }

      std::unordered_map<const Var*, ExprPtr> callsite_subst;
      for (size_t i = 0; i < original_func->params_.size() && i < call->args_.size(); ++i) {
        callsite_subst[original_func->params_[i].get()] = call->args_[i];
      }

      std::unordered_map<size_t, SliceBundle> slices_by_out_index;
      std::unordered_map<size_t, VarPtr> slices_by_in_index;
      std::vector<StmtPtr> stmts;
      stmts.reserve((analysis.outputs.size() + analysis.inputs.size()) * 2 + 2);

      for (const auto& input : analysis.inputs) {
        if (input.in_param_index >= call->args_.size()) return std::nullopt;
        auto in_arg = AsVarLike(call->args_[input.in_param_index]);
        if (!in_arg) return std::nullopt;
        INTERNAL_CHECK_SPAN(IsWritableRootExpr(call->args_[input.in_param_index]), call_assign->span_)
            << "Internal error: input window parameter must be backed by a writable root at callsite";

        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(input.window_shape.size());
        for (const auto& dim : input.window_shape) {
          shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);

        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(input.callsite_offsets.size());
        for (const auto& offset : input.callsite_offsets) {
          offset_exprs.push_back(
              arith::Analyzer().Simplify(transform_utils::Substitute(offset, callsite_subst)));
        }
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

        ExprPtr parent_expr = VisitExpr(call->args_[input.in_param_index]);
        auto slice_call = OpRegistry::GetInstance().Create(
            "tensor.slice", {parent_expr, shape_tuple, offset_tuple}, call_assign->span_);
        auto slice_var =
            std::make_shared<Var>(in_arg->name_hint_ + "__window", slice_call->GetType(), in_arg->span_);
        stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_));
        slices_by_in_index.emplace(input.in_param_index, slice_var);
      }

      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return std::nullopt;
        auto out_arg = AsVarLike(call->args_[output.out_param_index]);
        if (!out_arg) return std::nullopt;

        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(output.window_shape.size());
        for (const auto& dim : output.window_shape) {
          shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);

        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(output.callsite_offsets.size());
        for (const auto& offset : output.callsite_offsets) {
          offset_exprs.push_back(
              arith::Analyzer().Simplify(transform_utils::Substitute(offset, callsite_subst)));
        }
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

        ExprPtr parent_expr = VisitExpr(call->args_[output.out_param_index]);
        auto slice_call = OpRegistry::GetInstance().Create(
            "tensor.slice", {parent_expr, shape_tuple, offset_tuple}, call_assign->span_);
        auto slice_var =
            std::make_shared<Var>(out_arg->name_hint_ + "__window", slice_call->GetType(), out_arg->span_);
        stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_));
        slices_by_out_index.emplace(output.out_param_index,
                                    SliceBundle{slice_var, parent_expr, offset_tuple});
      }

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size());
      for (size_t i = 0; i < call->args_.size(); ++i) {
        auto input_slice_it = slices_by_in_index.find(i);
        if (input_slice_it != slices_by_in_index.end()) {
          new_args.push_back(input_slice_it->second);
          continue;
        }
        auto slice_it = slices_by_out_index.find(i);
        if (slice_it != slices_by_out_index.end()) {
          new_args.push_back(slice_it->second.slice_var);
        } else {
          new_args.push_back(VisitExpr(call->args_[i]));
        }
      }

      auto cloned_gvar = std::make_shared<GlobalVar>(cloned_func->name_);
      const bool is_submit_call = IsSubmitCall(call);
      std::vector<TypePtr> result_types = cloned_func->return_types_;
      if (is_submit_call) {
        auto tuple_ty = As<TupleType>(call->GetType());
        if (!tuple_ty || tuple_ty->types_.size() != result_types.size() + 1) return std::nullopt;
        result_types.push_back(tuple_ty->types_.back());
      }
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
      auto tmp_result_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__windowed",
                                                  new_return_type, call_assign->var_->span_);
      stmts.push_back(std::make_shared<AssignStmt>(tmp_result_var, new_call, call_assign->span_));

      if (!is_submit_call && analysis.outputs.size() == 1 && result_types.size() == 1) {
        const auto& output = analysis.outputs[0];
        const auto& slice_bundle = slices_by_out_index.at(output.out_param_index);
        auto assemble_call = OpRegistry::GetInstance().Create(
            "tensor.assemble", {slice_bundle.parent_expr, ExprPtr(tmp_result_var), slice_bundle.offset_tuple},
            call_assign->span_);
        stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, assemble_call, call_assign->span_));

        RewriteBundle bundle;
        bundle.stmts = std::move(stmts);
        return bundle;
      }

      std::vector<ExprPtr> assembled_result_exprs(result_types.size());
      std::vector<StmtPtr> tail_stmts;
      tail_stmts.reserve(analysis.outputs.size() * 2 + 1);

      std::unordered_map<size_t, VarPtr> tuple_items;
      for (const auto& output : analysis.outputs) {
        auto get_item = std::make_shared<TupleGetItemExpr>(
            tmp_result_var, static_cast<int>(output.return_index), call_assign->span_);
        auto item_var = std::make_shared<Var>(
            call_assign->var_->name_hint_ + "__windowed_" + std::to_string(output.return_index),
            result_types[output.return_index], call_assign->var_->span_);
        tail_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));

        const auto& slice_bundle = slices_by_out_index.at(output.out_param_index);
        auto assemble_call = OpRegistry::GetInstance().Create(
            "tensor.assemble", {slice_bundle.parent_expr, ExprPtr(item_var), slice_bundle.offset_tuple},
            call_assign->span_);
        auto parent_type = slice_bundle.parent_expr->GetType();
        auto assembled_var = std::make_shared<Var>(
            call_assign->var_->name_hint_ + "__assembled_" + std::to_string(output.return_index), parent_type,
            call_assign->var_->span_);
        tail_stmts.push_back(std::make_shared<AssignStmt>(assembled_var, assemble_call, call_assign->span_));
        assembled_result_exprs[output.return_index] = assembled_var;
      }

      for (size_t i = 0; i < assembled_result_exprs.size(); ++i) {
        if (!assembled_result_exprs[i]) {
          auto get_item =
              std::make_shared<TupleGetItemExpr>(tmp_result_var, static_cast<int>(i), call_assign->span_);
          auto item_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__pass_" + std::to_string(i),
                                                result_types[i], call_assign->var_->span_);
          tail_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));
          assembled_result_exprs[i] = item_var;
        }
      }

      tuple_result_subst_[call_assign->var_.get()] = std::move(assembled_result_exprs);
      stmts.insert(stmts.end(), tail_stmts.begin(), tail_stmts.end());
      auto rebuilt_tuple =
          std::make_shared<MakeTuple>(tuple_result_subst_.at(call_assign->var_.get()), call_assign->span_);
      stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, rebuilt_tuple, call_assign->span_));

      RewriteBundle bundle;
      bundle.stmts = std::move(stmts);
      return bundle;
    }

    std::optional<RewriteBundle> TryRewriteEvalCall(const EvalStmtPtr& eval) {
      auto candidate = GetRewriteCallsiteCandidate(eval);
      if (!candidate.has_value()) return std::nullopt;
      return TryRewriteSpmdRowBlockEval(eval, *candidate);
    }

    std::optional<RewriteBundle> TryRewriteSpmdRowBlock(const AssignStmtPtr& call_assign,
                                                        const RewriteCallsiteCandidate& candidate) {
      auto call = candidate.call;
      const auto& analysis = *candidate.analysis;
      auto cloned_func = candidate.cloned_func;
      if (!call_assign || !call || candidate.submit || !cloned_func || !analysis.spmd_core_num.has_value()) {
        return std::nullopt;
      }
      if (analysis.outputs.empty() || cloned_func->return_types_.empty()) return std::nullopt;
      if (!SpmdRewriteCoversAllReturns(analysis, cloned_func->return_types_.size())) return std::nullopt;

      std::unordered_map<const Var*, ExprPtr> outer_subst;
      for (size_t i = 0; i < candidate.original_func->params_.size() && i < call->args_.size(); ++i) {
        outer_subst[candidate.original_func->params_[i].get()] = call->args_[i];
      }

      auto loop_var = std::make_shared<Var>("spmd_idx", std::make_shared<ScalarType>(DataType::INDEX),
                                            call_assign->span_);
      auto start = std::make_shared<ConstInt>(0, DataType::INDEX, call_assign->span_);
      auto stop = std::make_shared<ConstInt>(*analysis.spmd_core_num, DataType::INDEX, call_assign->span_);
      auto step = std::make_shared<ConstInt>(1, DataType::INDEX, call_assign->span_);

      std::vector<IterArgPtr> iter_args;
      std::vector<VarPtr> return_vars;
      std::unordered_map<size_t, VarPtr> output_iters;
      std::unordered_map<size_t, SliceBundle> slices_by_out_index;
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return std::nullopt;
        auto out_arg = AsVarLike(call->args_[output.out_param_index]);
        if (!out_arg) return std::nullopt;
        auto iter_arg =
            std::make_shared<IterArg>(out_arg->name_hint_ + "_iter", out_arg->GetType(),
                                      VisitExpr(call->args_[output.out_param_index]), out_arg->span_);
        auto return_var =
            std::make_shared<Var>(out_arg->name_hint_ + "_rv", out_arg->GetType(), out_arg->span_);
        output_iters.emplace(output.out_param_index, iter_arg);
        iter_args.push_back(iter_arg);
        return_vars.push_back(return_var);
      }

      std::unordered_map<const Var*, ExprPtr> callsite_subst = outer_subst;
      if (analysis.spmd_block_idx_var) callsite_subst[analysis.spmd_block_idx_var.get()] = loop_var;

      std::vector<StmtPtr> loop_stmts;
      loop_stmts.reserve((analysis.outputs.size() + analysis.inputs.size()) * 3 + 4);

      std::unordered_map<size_t, VarPtr> slices_by_in_index;
      for (const auto& input : analysis.inputs) {
        if (input.in_param_index >= call->args_.size()) return std::nullopt;
        auto in_arg = AsVarLike(call->args_[input.in_param_index]);
        if (!in_arg) return std::nullopt;
        if (!IsWritableRootExpr(call->args_[input.in_param_index])) return std::nullopt;

        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(input.window_shape.size());
        for (const auto& dim : input.window_shape) {
          shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);

        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(input.callsite_offsets.size());
        for (const auto& offset : input.callsite_offsets) {
          offset_exprs.push_back(
              arith::Analyzer().Simplify(transform_utils::Substitute(offset, callsite_subst)));
        }
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

        auto slice_call = OpRegistry::GetInstance().Create(
            "tensor.slice", {VisitExpr(call->args_[input.in_param_index]), shape_tuple, offset_tuple},
            call_assign->span_);
        auto slice_var =
            std::make_shared<Var>(in_arg->name_hint_ + "__window", slice_call->GetType(), in_arg->span_);
        loop_stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_));
        slices_by_in_index.emplace(input.in_param_index, slice_var);
      }

      for (const auto& output : analysis.outputs) {
        auto out_iter_it = output_iters.find(output.out_param_index);
        if (out_iter_it == output_iters.end()) return std::nullopt;
        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(output.window_shape.size());
        for (const auto& dim : output.window_shape) {
          shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, call_assign->span_);

        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(output.callsite_offsets.size());
        for (const auto& offset : output.callsite_offsets) {
          offset_exprs.push_back(
              arith::Analyzer().Simplify(transform_utils::Substitute(offset, callsite_subst)));
        }
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, call_assign->span_);

        auto slice_call = OpRegistry::GetInstance().Create(
            "tensor.slice", {ExprPtr(out_iter_it->second), shape_tuple, offset_tuple}, call_assign->span_);
        auto slice_var = std::make_shared<Var>(out_iter_it->second->name_hint_ + "__window",
                                               slice_call->GetType(), out_iter_it->second->span_);
        loop_stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, call_assign->span_));
        slices_by_out_index.emplace(output.out_param_index,
                                    SliceBundle{slice_var, out_iter_it->second, offset_tuple});
      }

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size() + 1);
      new_args.push_back(loop_var);
      for (size_t i = 0; i < call->args_.size(); ++i) {
        auto input_slice_it = slices_by_in_index.find(i);
        if (input_slice_it != slices_by_in_index.end()) {
          new_args.push_back(input_slice_it->second);
          continue;
        }
        auto output_slice_it = slices_by_out_index.find(i);
        if (output_slice_it != slices_by_out_index.end()) {
          new_args.push_back(output_slice_it->second.slice_var);
        } else {
          new_args.push_back(VisitExpr(call->args_[i]));
        }
      }

      auto cloned_gvar = std::make_shared<GlobalVar>(cloned_func->name_);
      TypePtr new_return_type = cloned_func->return_types_.size() == 1
                                    ? cloned_func->return_types_[0]
                                    : std::make_shared<TupleType>(cloned_func->return_types_);
      auto new_call = std::make_shared<Call>(cloned_gvar, std::move(new_args), call->kwargs_,
                                             RewriteCallAttrs(call, analysis, slices_by_out_index),
                                             new_return_type, call->span_);
      auto tmp_result_var = std::make_shared<Var>(call_assign->var_->name_hint_ + "__windowed",
                                                  new_return_type, call_assign->var_->span_);
      loop_stmts.push_back(std::make_shared<AssignStmt>(tmp_result_var, new_call, call_assign->span_));

      std::vector<ExprPtr> yield_values;
      yield_values.reserve(iter_args.size());
      for (size_t i = 0; i < analysis.outputs.size(); ++i) {
        const auto& output = analysis.outputs[i];
        if (output.return_index >= cloned_func->return_types_.size()) return std::nullopt;
        const auto& slice_bundle = slices_by_out_index.at(output.out_param_index);
        ExprPtr source_expr = tmp_result_var;
        if (cloned_func->return_types_.size() != 1) {
          auto get_item = std::make_shared<TupleGetItemExpr>(
              tmp_result_var, static_cast<int>(output.return_index), call_assign->span_);
          auto item_var = std::make_shared<Var>(
              call_assign->var_->name_hint_ + "__windowed_" + std::to_string(output.return_index),
              cloned_func->return_types_[output.return_index], call_assign->var_->span_);
          loop_stmts.push_back(std::make_shared<AssignStmt>(item_var, get_item, call_assign->span_));
          source_expr = item_var;
        }
        auto assemble_call = OpRegistry::GetInstance().Create(
            "tensor.assemble", {slice_bundle.parent_expr, source_expr, slice_bundle.offset_tuple},
            call_assign->span_);
        auto assembled_var =
            std::make_shared<Var>(call_assign->var_->name_hint_ + "__assembled_" + std::to_string(i),
                                  slice_bundle.parent_expr->GetType(), call_assign->var_->span_);
        loop_stmts.push_back(std::make_shared<AssignStmt>(assembled_var, assemble_call, call_assign->span_));
        yield_values.push_back(assembled_var);
      }
      loop_stmts.push_back(std::make_shared<YieldStmt>(yield_values, call_assign->span_));

      auto loop_body = SeqStmts::Flatten(std::move(loop_stmts), call_assign->span_);
      auto loop_stmt = std::make_shared<ForStmt>(loop_var, start, stop, step, iter_args, loop_body,
                                                 return_vars, call_assign->span_, ForKind::Sequential);

      RewriteBundle bundle;
      bundle.stmts.push_back(loop_stmt);
      if (return_vars.size() == 1) {
        bundle.stmts.push_back(
            std::make_shared<AssignStmt>(call_assign->var_, return_vars[0], call_assign->span_));
      } else {
        std::vector<ExprPtr> tuple_values(return_vars.begin(), return_vars.end());
        auto tuple = std::make_shared<MakeTuple>(std::move(tuple_values), call_assign->span_);
        bundle.stmts.push_back(std::make_shared<AssignStmt>(call_assign->var_, tuple, call_assign->span_));
      }
      return bundle;
    }

    std::optional<RewriteBundle> TryRewriteSpmdRowBlockEval(const EvalStmtPtr& eval,
                                                            const RewriteCallsiteCandidate& candidate) {
      auto call = candidate.call;
      const auto& analysis = *candidate.analysis;
      auto cloned_func = candidate.cloned_func;
      if (!eval || !call || !cloned_func || !analysis.spmd_core_num.has_value() ||
          !cloned_func->return_types_.empty()) {
        return std::nullopt;
      }

      std::unordered_map<const Var*, ExprPtr> outer_subst;
      for (size_t i = 0; i < candidate.original_func->params_.size() && i < call->args_.size(); ++i) {
        outer_subst[candidate.original_func->params_[i].get()] = call->args_[i];
      }

      auto loop_var =
          std::make_shared<Var>("spmd_idx", std::make_shared<ScalarType>(DataType::INDEX), eval->span_);
      auto start = std::make_shared<ConstInt>(0, DataType::INDEX, eval->span_);
      auto stop = std::make_shared<ConstInt>(*analysis.spmd_core_num, DataType::INDEX, eval->span_);
      auto step = std::make_shared<ConstInt>(1, DataType::INDEX, eval->span_);

      std::unordered_map<const Var*, ExprPtr> callsite_subst = outer_subst;
      if (analysis.spmd_block_idx_var) callsite_subst[analysis.spmd_block_idx_var.get()] = loop_var;

      std::vector<StmtPtr> loop_stmts;
      loop_stmts.reserve((analysis.outputs.size() + analysis.inputs.size()) * 2 + 1);

      std::unordered_map<size_t, VarPtr> slices_by_in_index;
      for (const auto& input : analysis.inputs) {
        if (input.in_param_index >= call->args_.size()) return std::nullopt;
        auto in_arg = AsVarLike(call->args_[input.in_param_index]);
        if (!in_arg || !IsWritableRootExpr(call->args_[input.in_param_index])) return std::nullopt;

        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(input.window_shape.size());
        for (const auto& dim : input.window_shape) {
          shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, eval->span_);

        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(input.callsite_offsets.size());
        for (const auto& offset : input.callsite_offsets) {
          offset_exprs.push_back(
              arith::Analyzer().Simplify(transform_utils::Substitute(offset, callsite_subst)));
        }
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, eval->span_);

        auto slice_call = OpRegistry::GetInstance().Create(
            "tensor.slice", {VisitExpr(call->args_[input.in_param_index]), shape_tuple, offset_tuple},
            eval->span_);
        auto slice_var =
            std::make_shared<Var>(in_arg->name_hint_ + "__window", slice_call->GetType(), in_arg->span_);
        loop_stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, eval->span_));
        slices_by_in_index.emplace(input.in_param_index, slice_var);
      }

      std::unordered_map<size_t, SliceBundle> slices_by_out_index;
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return std::nullopt;
        auto out_arg = AsVarLike(call->args_[output.out_param_index]);
        if (!out_arg) return std::nullopt;

        std::vector<ExprPtr> shape_exprs;
        shape_exprs.reserve(output.window_shape.size());
        for (const auto& dim : output.window_shape) {
          shape_exprs.push_back(transform_utils::Substitute(dim, callsite_subst));
        }
        auto shape_tuple = std::make_shared<MakeTuple>(shape_exprs, eval->span_);

        std::vector<ExprPtr> offset_exprs;
        offset_exprs.reserve(output.callsite_offsets.size());
        for (const auto& offset : output.callsite_offsets) {
          offset_exprs.push_back(
              arith::Analyzer().Simplify(transform_utils::Substitute(offset, callsite_subst)));
        }
        auto offset_tuple = std::make_shared<MakeTuple>(offset_exprs, eval->span_);

        auto parent_expr = VisitExpr(call->args_[output.out_param_index]);
        auto slice_call = OpRegistry::GetInstance().Create(
            "tensor.slice", {parent_expr, shape_tuple, offset_tuple}, eval->span_);
        auto slice_var =
            std::make_shared<Var>(out_arg->name_hint_ + "__window", slice_call->GetType(), out_arg->span_);
        loop_stmts.push_back(std::make_shared<AssignStmt>(slice_var, slice_call, eval->span_));
        slices_by_out_index.emplace(output.out_param_index,
                                    SliceBundle{slice_var, parent_expr, offset_tuple});
      }

      std::vector<ExprPtr> new_args;
      new_args.reserve(call->args_.size() + 1);
      new_args.push_back(loop_var);
      for (size_t i = 0; i < call->args_.size(); ++i) {
        auto input_slice_it = slices_by_in_index.find(i);
        if (input_slice_it != slices_by_in_index.end()) {
          new_args.push_back(input_slice_it->second);
          continue;
        }
        auto output_slice_it = slices_by_out_index.find(i);
        if (output_slice_it != slices_by_out_index.end()) {
          new_args.push_back(output_slice_it->second.slice_var);
        } else {
          new_args.push_back(VisitExpr(call->args_[i]));
        }
      }

      auto cloned_gvar = std::make_shared<GlobalVar>(cloned_func->name_);
      auto new_call = std::make_shared<Call>(cloned_gvar, std::move(new_args), call->kwargs_,
                                             RewriteCallAttrs(call, analysis, slices_by_out_index),
                                             call->GetType(), call->span_);
      loop_stmts.push_back(std::make_shared<EvalStmt>(new_call, eval->span_));

      auto loop_body = SeqStmts::Flatten(std::move(loop_stmts), eval->span_);
      auto loop = std::make_shared<ForStmt>(loop_var, start, stop, step, std::vector<IterArgPtr>{}, loop_body,
                                            std::vector<VarPtr>{}, eval->span_, ForKind::Sequential);
      RewriteBundle bundle;
      bundle.stmts.push_back(loop);
      return bundle;
    }

    static bool IsSubmitCall(const CallPtr& call) {
      auto tuple_ty = As<TupleType>(call->GetType());
      if (!tuple_ty || tuple_ty->types_.empty()) return false;
      auto last = As<ScalarType>(tuple_ty->types_.back());
      return last != nullptr && last->dtype_ == DataType::TASK_ID;
    }

    static bool SpmdRewriteCoversAllReturns(const CalleeRewriteAnalysis& analysis, size_t return_count) {
      std::vector<bool> covered(return_count, false);
      for (const auto& output : analysis.outputs) {
        if (output.return_index == SIZE_MAX) continue;
        if (output.return_index >= return_count) return false;
        covered[output.return_index] = true;
      }
      return std::all_of(covered.begin(), covered.end(), [](bool is_covered) { return is_covered; });
    }

    std::vector<std::pair<std::string, std::any>> RewriteCallAttrs(
        const CallPtr& call, const CalleeRewriteAnalysis& analysis,
        const std::unordered_map<size_t, SliceBundle>& slices_by_out_index) const {
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
              rewritten.push_back(slices_by_out_index.at(output.out_param_index).slice_var);
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

      auto original_func = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
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
        auto loop = candidate.loop;
        if (IsOutputParentLocalToLoop(output_param, callsite_subst, candidate.loop_local_allocs)) {
          continue;
        }

        auto trip_count = GetStaticTripCount(loop);
        if (!trip_count.has_value()) return false;
        if (*trip_count <= 1) continue;

        std::optional<size_t> varying_dim;
        for (size_t i = 0; i < output.callsite_offsets.size(); ++i) {
          auto rewritten = transform_utils::Substitute(output.callsite_offsets[i], callsite_subst);
          rewritten = transform_utils::Substitute(rewritten, scalar_defs_);
          auto affine = ParseAffineInLoop(rewritten, loop->loop_var_.get());
          if (!affine.has_value()) return false;
          if (affine->coeff == 0) continue;

          auto extent_ci = As<ConstInt>(output.window_shape[i]);
          auto loop_step = GetConstIntValue(loop->step_);
          if (!extent_ci || !loop_step.has_value()) return false;
          if (varying_dim.has_value()) return false;
          if (varying_dims_used.count(i)) return false;
          if (std::abs(affine->coeff * *loop_step) < extent_ci->value_) return false;
          varying_dim = i;
        }
        if (!varying_dim.has_value()) return false;
        varying_dims_used.insert(*varying_dim);
      }
      return true;
    }

    bool IsOutputParentLocalToLoop(const Var* output_param,
                                   const std::unordered_map<const Var*, ExprPtr>& callsite_subst,
                                   const std::unordered_set<const Var*>* loop_local_allocs) const {
      if (!loop_local_allocs || loop_local_allocs->empty()) return false;

      auto subst_it = callsite_subst.find(output_param);
      if (subst_it == callsite_subst.end()) return false;

      auto parent_expr = ResolveLoopInitExpr(subst_it->second);
      auto parent_var = AsVarLike(parent_expr);
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

    ProgramPtr program_;
    const AnalysisMap& analyses_;
    const std::unordered_map<std::string, FunctionPtr>& cloned_funcs_;
    std::vector<ForStmtPtr> sequential_loops_;
    std::vector<std::unordered_set<const Var*>> loop_local_allocs_;
    std::unordered_map<const Var*, ExprPtr> loop_iter_init_subst_;
    std::unordered_map<const Var*, ExprPtr> scalar_defs_;
    std::unordered_map<const Var*, const Var*> full_buffer_roots_;
    std::unordered_set<const Var*> param_roots_;
    std::unordered_set<const Var*> writable_buffer_roots_;
    std::unordered_map<const Var*, std::vector<const Var*>> tuple_output_roots_;
    std::unordered_map<const Var*, std::vector<ExprPtr>> tuple_result_subst_;
    RootSet enclosing_later_full_parent_reads_;
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
    if (!distance) return std::nullopt;
    if (distance->value_ <= 0) return int64_t{0};
    int64_t step_abs = *step > 0 ? *step : -*step;
    return (distance->value_ + step_abs - 1) / step_abs;
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
    auto parse_store = [](const AssignStmtPtr& assign)
        -> std::optional<std::tuple<const Var*, std::vector<ExprPtr>, std::vector<ExprPtr>>> {
      auto store_call = assign ? As<Call>(assign->value_) : nullptr;
      if (!store_call) return std::nullopt;
      if (store_call->op_->name_ == "tile.store" && store_call->args_.size() >= 3) {
        auto out_target = AsVarLike(store_call->args_[2]);
        auto offset_tuple = As<MakeTuple>(store_call->args_[1]);
        auto tile_type = As<TileType>(store_call->args_[0]->GetType());
        if (!out_target || !offset_tuple || !tile_type) return std::nullopt;
        return std::make_tuple(out_target.get(), tile_type->shape_, offset_tuple->elements_);
      }
      if (store_call->op_->name_ == "tensor.assemble" && store_call->args_.size() >= 3) {
        auto out_target = AsVarLike(store_call->args_[0]);
        auto offset_tuple = As<MakeTuple>(store_call->args_[2]);
        auto tensor_type = As<TensorType>(store_call->args_[1]->GetType());
        if (!out_target || !offset_tuple || !tensor_type) return std::nullopt;
        return std::make_tuple(out_target.get(), tensor_type->shape_, offset_tuple->elements_);
      }
      return std::nullopt;
    };

    for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
      auto ret_var = AsVarLike(ret_stmt->value_[ret_i]);
      if (!ret_var) continue;
      std::unordered_set<const Var*> seen;
      const Var* current = ret_var.get();
      while (current && seen.insert(current).second) {
        auto def_it = var_defs.find(current);
        if (def_it == var_defs.end()) break;
        auto parsed = parse_store(def_it->second);
        if (!parsed.has_value()) break;

        auto [out_target, window_shape, offsets] = *parsed;
        if (out_target == func->params_[out_param_index].get()) {
          size_t matched_refs = CountVarRefsInStmt(def_it->second, func->params_[out_param_index].get());
          if (total_out_refs != matched_refs) return std::nullopt;
          result = FinalStoreInfo{ret_i, std::move(window_shape), std::move(offsets)};
          break;
        }
        current = out_target;
      }
      if (result.has_value()) break;
    }
    return result;
  }

  static std::vector<InputRewriteInfo> AnalyzeInputWindows(const FunctionPtr& func,
                                                           const std::vector<OutputRewriteInfo>& outputs) {
    std::vector<InputRewriteInfo> inputs;
    if (!func || outputs.empty()) return inputs;

    auto body_stmts = FlattenToStmts(func->body_);
    std::unordered_set<const Var*> output_params;
    for (const auto& output : outputs) {
      if (output.out_param_index < func->params_.size())
        output_params.insert(func->params_[output.out_param_index].get());
    }

    std::unordered_map<size_t, InputRewriteInfo> by_index;
    std::unordered_map<size_t, size_t> matched_refs_by_index;
    std::unordered_set<size_t> conflicted;
    auto total_refs_by_var = CountAllVarRefsInStmt(func->body_);
    for (const auto& stmt : body_stmts) {
      auto assign = As<AssignStmt>(stmt);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!call) continue;

      size_t shape_arg_index = SIZE_MAX;
      size_t offset_arg_index = SIZE_MAX;
      if (call->op_->name_ == "tensor.slice" && call->args_.size() >= 3) {
        shape_arg_index = 1;
        offset_arg_index = 2;
      } else if (call->op_->name_ == "tile.load" && call->args_.size() >= 3) {
        offset_arg_index = 1;
        shape_arg_index = 2;
      } else {
        continue;
      }

      auto parent = AsVarLike(call->args_[0]);
      auto shape_tuple = As<MakeTuple>(call->args_[shape_arg_index]);
      auto offset_tuple = As<MakeTuple>(call->args_[offset_arg_index]);
      if (!parent || !shape_tuple || !offset_tuple) continue;

      size_t param_index = SIZE_MAX;
      for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i) {
        if (func->params_[i].get() == parent.get()) {
          param_index = i;
          break;
        }
      }
      if (param_index == SIZE_MAX) continue;
      if (func->param_directions_[param_index] != ParamDirection::In) continue;
      if (output_params.count(parent.get()) > 0) continue;

      auto tensor_type = As<TensorType>(func->params_[param_index]->GetType());
      if (!tensor_type) continue;
      if (shape_tuple->elements_.size() != offset_tuple->elements_.size() ||
          shape_tuple->elements_.size() != tensor_type->shape_.size()) {
        conflicted.insert(param_index);
        by_index.erase(param_index);
        continue;
      }
      if (AreExprVectorsEqual(shape_tuple->elements_, tensor_type->shape_) &&
          IsAllZeroOffsets(offset_tuple->elements_)) {
        continue;
      }

      std::unordered_set<const Var*> allowed_params;
      for (const auto& param : func->params_) allowed_params.insert(param.get());
      bool exprs_ok = true;
      for (const auto& expr : shape_tuple->elements_) {
        if (!ExprReferencesOnlyVarsIn(expr, allowed_params)) {
          exprs_ok = false;
          break;
        }
      }
      for (const auto& expr : offset_tuple->elements_) {
        if (!ExprReferencesOnlyVarsIn(expr, allowed_params)) {
          exprs_ok = false;
          break;
        }
      }
      if (!exprs_ok) {
        conflicted.insert(param_index);
        by_index.erase(param_index);
        matched_refs_by_index.erase(param_index);
        continue;
      }

      std::vector<ExprPtr> local_zero_offsets;
      local_zero_offsets.reserve(offset_tuple->elements_.size());
      for (size_t i = 0; i < offset_tuple->elements_.size(); ++i) {
        local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
      }

      InputRewriteInfo info{param_index, tensor_type->shape_, shape_tuple->elements_, offset_tuple->elements_,
                            std::move(local_zero_offsets)};
      auto existing = by_index.find(param_index);
      if (existing == by_index.end()) {
        by_index.emplace(param_index, std::move(info));
        matched_refs_by_index[param_index] = CountVarRefsInStmt(stmt, parent.get());
        continue;
      }
      if (!AreExprVectorsEqual(existing->second.window_shape, info.window_shape) ||
          !AreExprVectorsEqual(existing->second.callsite_offsets, info.callsite_offsets)) {
        conflicted.insert(param_index);
        by_index.erase(param_index);
        matched_refs_by_index.erase(param_index);
        continue;
      }
      matched_refs_by_index[param_index] += CountVarRefsInStmt(stmt, parent.get());
    }

    inputs.reserve(by_index.size());
    for (auto& [param_index, info] : by_index) {
      if (conflicted.count(param_index) != 0 || param_index >= func->params_.size()) continue;
      auto total_refs_it = total_refs_by_var.find(func->params_[param_index].get());
      auto total_refs = total_refs_it == total_refs_by_var.end() ? 0 : total_refs_it->second;
      auto matched_it = matched_refs_by_index.find(param_index);
      auto matched_refs = matched_it == matched_refs_by_index.end() ? 0 : matched_it->second;
      if (total_refs == matched_refs) inputs.push_back(std::move(info));
    }
    std::sort(inputs.begin(), inputs.end(), [](const InputRewriteInfo& lhs, const InputRewriteInfo& rhs) {
      return lhs.in_param_index < rhs.in_param_index;
    });
    return inputs;
  }

  class InputWindowCallsiteFilter {
   public:
    InputWindowCallsiteFilter(ProgramPtr program, const AnalysisMap& analyses)
        : program_(std::move(program)), analyses_(analyses) {}

    std::unordered_map<std::string, std::unordered_set<size_t>> Run() {
      if (!program_) return {};
      for (const auto& [gvar, func] : program_->functions_) {
        if (!func || func->func_type_ != FunctionType::Orchestration) continue;
        ScanFunction(func);
      }
      std::unordered_map<std::string, std::unordered_set<size_t>> eligible_inputs;
      for (const auto& [func_name, candidates] : input_callsites_) {
        for (const auto& [param_index, callsites] : candidates) {
          if (callsites.seen && callsites.all_writable) eligible_inputs[func_name].insert(param_index);
        }
      }
      return eligible_inputs;
    }

   private:
    struct InputCallsites {
      bool seen = false;
      bool all_writable = true;
    };

    void ScanFunction(const FunctionPtr& func) {
      full_buffer_roots_.clear();
      param_roots_.clear();
      writable_buffer_roots_.clear();
      tuple_output_roots_.clear();
      loop_iter_init_subst_.clear();
      for (size_t i = 0; i < func->params_.size(); ++i) {
        const auto& param = func->params_[i];
        if (!param || !AsTensorTypeLike(param->GetType())) continue;
        full_buffer_roots_[param.get()] = param.get();
        param_roots_.insert(param.get());
        if (i < func->param_directions_.size() &&
            IsOutputDirection(func->param_directions_[i], /*include_inout=*/true)) {
          writable_buffer_roots_.insert(param.get());
        }
      }
      VisitStmt(func->body_);
    }

    void VisitStmt(const StmtPtr& stmt) {
      if (!stmt) return;
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) VisitStmt(child);
      } else if (auto loop = As<ForStmt>(stmt)) {
        auto saved_loop_iter_init_subst = loop_iter_init_subst_;
        for (const auto& iter_arg : loop->iter_args_) {
          if (!iter_arg || !iter_arg->initValue_) continue;
          loop_iter_init_subst_[iter_arg.get()] = iter_arg->initValue_;
          if (const Var* root = ResolveBufferRoot(iter_arg->initValue_)) {
            full_buffer_roots_[iter_arg.get()] = root;
          }
        }
        VisitStmt(loop->body_);
        AddLoopReturnRoots(loop);
        loop_iter_init_subst_ = std::move(saved_loop_iter_init_subst);
      } else if (auto loop = As<WhileStmt>(stmt)) {
        auto saved_loop_iter_init_subst = loop_iter_init_subst_;
        for (const auto& iter_arg : loop->iter_args_) {
          if (!iter_arg || !iter_arg->initValue_) continue;
          loop_iter_init_subst_[iter_arg.get()] = iter_arg->initValue_;
          if (const Var* root = ResolveBufferRoot(iter_arg->initValue_)) {
            full_buffer_roots_[iter_arg.get()] = root;
          }
        }
        VisitStmt(loop->body_);
        AddLoopReturnRoots(loop);
        loop_iter_init_subst_ = std::move(saved_loop_iter_init_subst);
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        VisitStmt(if_stmt->then_body_);
        if (if_stmt->else_body_.has_value()) VisitStmt(if_stmt->else_body_.value());
      } else if (auto scope = As<ScopeStmt>(stmt)) {
        VisitStmt(scope->body_);
      } else if (auto rscope = As<RuntimeScopeStmt>(stmt)) {
        VisitStmt(rscope->body_);
      } else if (auto assign = As<AssignStmt>(stmt)) {
        VisitCallsite(assign->value_);
        AddFullBufferRootForAssign(assign);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        VisitCallsite(eval->expr_);
      }
    }

    void VisitCallsite(const ExprPtr& expr) {
      auto submit = As<Submit>(expr);
      auto call = submit ? SubmitToCallView(submit) : As<Call>(expr);
      if (!call || codegen::IsBuiltinOp(call->op_->name_)) return;
      auto analysis_it = analyses_.find(call->op_->name_);
      if (analysis_it == analyses_.end()) return;
      for (const auto& input : analysis_it->second.inputs) {
        if (input.in_param_index >= call->args_.size()) continue;
        auto& callsites = input_callsites_[call->op_->name_][input.in_param_index];
        callsites.seen = true;
        if (!IsWritableRootExpr(call->args_[input.in_param_index])) {
          callsites.all_writable = false;
        }
      }
    }

    const Var* ResolveBufferRoot(const ExprPtr& expr) const {
      auto current = ResolveLoopInitExpr(expr);
      auto var = AsVarLike(current);
      if (!var) return nullptr;
      return ResolveBufferRoot(var.get());
    }

    const Var* ResolveBufferRoot(const Var* var) const {
      const Var* current = var;
      std::unordered_set<const Var*> seen;
      while (current && seen.insert(current).second) {
        auto it = full_buffer_roots_.find(current);
        if (it == full_buffer_roots_.end()) return nullptr;
        if (it->second == current) return current;
        current = it->second;
      }
      return current;
    }

    bool IsWritableRootExpr(const ExprPtr& expr) const {
      auto root = ResolveBufferRoot(expr);
      if (!root) return false;
      if (param_roots_.count(root) == 0) return true;
      return writable_buffer_roots_.count(root) > 0;
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

    void AddFullBufferRootForAssign(const AssignStmtPtr& assign) {
      if (!assign) return;
      if (auto call = As<Call>(assign->value_)) {
        const auto& op_name = call->op_->name_;
        if ((op_name == "tensor.reshape" || op_name == "tensor.assemble") && !call->args_.empty()) {
          if (const Var* root = ResolveBufferRoot(call->args_[0])) {
            full_buffer_roots_[assign->var_.get()] = root;
          }
        } else if (IsTensorAllocationOp(call)) {
          full_buffer_roots_[assign->var_.get()] = assign->var_.get();
        } else if (!codegen::IsBuiltinOp(op_name)) {
          AddCallOutputRoots(assign, call);
        }
      } else if (auto submit = As<Submit>(assign->value_)) {
        auto call = SubmitToCallView(submit);
        if (!codegen::IsBuiltinOp(call->op_->name_)) {
          AddCallOutputRoots(assign, call);
        }
      } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
        AddTupleGetItemRoot(assign, tuple_get);
      } else if (auto src_var = AsVarLike(assign->value_)) {
        if (const Var* root = ResolveBufferRoot(src_var.get())) {
          full_buffer_roots_[assign->var_.get()] = root;
        }
      } else if (auto tuple = As<MakeTuple>(assign->value_)) {
        std::vector<const Var*> roots;
        roots.reserve(tuple->elements_.size());
        for (const auto& element : tuple->elements_) {
          roots.push_back(IsTensorTypedArg(element) ? ResolveBufferRoot(element) : nullptr);
        }
        tuple_output_roots_[assign->var_.get()] = std::move(roots);
      }
    }

    void AddCallOutputRoots(const AssignStmtPtr& assign, const CallPtr& call) {
      auto roots =
          BuildCallOutputRoots(program_, call, [this](const ExprPtr& arg) { return ResolveBufferRoot(arg); });
      if (roots.empty()) return;

      if (As<TupleType>(call->GetType())) {
        tuple_output_roots_[assign->var_.get()] = std::move(roots);
      } else if (!roots.empty() && roots[0]) {
        full_buffer_roots_[assign->var_.get()] = roots[0];
      }
    }

    void AddTupleGetItemRoot(const AssignStmtPtr& assign, const TupleGetItemExprPtr& tuple_get) {
      auto tuple_var = AsVarLike(tuple_get->tuple_);
      if (!tuple_var) return;
      auto roots_it = tuple_output_roots_.find(tuple_var.get());
      if (roots_it == tuple_output_roots_.end()) return;
      if (tuple_get->index_ < 0 || static_cast<size_t>(tuple_get->index_) >= roots_it->second.size()) return;
      if (const Var* root = roots_it->second[static_cast<size_t>(tuple_get->index_)]) {
        full_buffer_roots_[assign->var_.get()] = root;
      }
    }

    void AddLoopReturnRoots(const ForStmtPtr& loop) {
      if (!loop) return;
      auto yield = transform_utils::GetLastYieldStmt(loop->body_);
      for (size_t i = 0; i < loop->return_vars_.size(); ++i) {
        const Var* root = nullptr;
        if (i < loop->iter_args_.size()) root = ResolveBufferRoot(loop->iter_args_[i].get());
        if (!root && yield && i < yield->value_.size()) root = ResolveBufferRoot(yield->value_[i]);
        if (root) full_buffer_roots_[loop->return_vars_[i].get()] = root;
      }
    }

    void AddLoopReturnRoots(const WhileStmtPtr& loop) {
      if (!loop) return;
      auto yield = transform_utils::GetLastYieldStmt(loop->body_);
      for (size_t i = 0; i < loop->return_vars_.size(); ++i) {
        const Var* root = nullptr;
        if (i < loop->iter_args_.size()) root = ResolveBufferRoot(loop->iter_args_[i].get());
        if (!root && yield && i < yield->value_.size()) root = ResolveBufferRoot(yield->value_[i]);
        if (root) full_buffer_roots_[loop->return_vars_[i].get()] = root;
      }
    }

    ProgramPtr program_;
    const AnalysisMap& analyses_;
    std::unordered_map<std::string, std::unordered_map<size_t, InputCallsites>> input_callsites_;
    std::unordered_map<const Var*, const Var*> full_buffer_roots_;
    std::unordered_set<const Var*> param_roots_;
    std::unordered_set<const Var*> writable_buffer_roots_;
    std::unordered_map<const Var*, ExprPtr> loop_iter_init_subst_;
    std::unordered_map<const Var*, std::vector<const Var*>> tuple_output_roots_;
  };

  static void FilterInputWindowsByCallsites(const ProgramPtr& program, AnalysisMap* analyses) {
    if (!program || !analyses) return;
    auto eligible = InputWindowCallsiteFilter(program, *analyses).Run();
    for (auto& [func_name, analysis] : *analyses) {
      auto eligible_it = eligible.find(func_name);
      if (eligible_it == eligible.end()) {
        analysis.inputs.clear();
        continue;
      }
      const auto& eligible_indices = eligible_it->second;
      analysis.inputs.erase(std::remove_if(analysis.inputs.begin(), analysis.inputs.end(),
                                           [&eligible_indices](const InputRewriteInfo& input) {
                                             return eligible_indices.count(input.in_param_index) == 0;
                                           }),
                            analysis.inputs.end());
    }
  }

  static std::optional<CalleeRewriteAnalysis> AnalyzeAggregateWindowLoop(
      const FunctionPtr& func, const std::vector<size_t>& out_indices,
      std::unordered_map<const Var*, ExprPtr> scalar_defs = {},
      std::unordered_set<const Var*> extra_allowed_vars = {}) {
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
      bool matches_all_outputs = true;

      for (const auto& out_param_index : out_indices) {
        std::optional<size_t> direct_return_index = FindReturnIndexForOutParam(func, out_param_index);
        VarPtr direct_returned;
        if (direct_return_index.has_value() && *direct_return_index < ret_stmt->value_.size()) {
          direct_returned = AsVarLike(ret_stmt->value_[*direct_return_index]);
        }

        bool matched_output = false;
        for (size_t i = 0; i < candidate->iter_args_.size(); ++i) {
          auto init_var = AsVarLike(candidate->iter_args_[i]->initValue_);
          if (!init_var || init_var.get() != func->params_[out_param_index].get()) continue;

          std::optional<size_t> return_index = direct_return_index;
          if (i < candidate->return_vars_.size() && direct_returned &&
              direct_returned.get() != candidate->return_vars_[i].get()) {
            return_index = std::nullopt;
          }
          if (i < candidate->return_vars_.size()) {
            for (size_t ret_i = 0; ret_i < ret_stmt->value_.size(); ++ret_i) {
              if (return_index.has_value()) break;
              auto returned = AsVarLike(ret_stmt->value_[ret_i]);
              if (returned && returned.get() == candidate->return_vars_[i].get()) {
                return_index = ret_i;
                break;
              }
            }
          }
          if (!return_index.has_value() && ret_stmt->value_.empty()) {
            return_index = SIZE_MAX;
          }
          if (!return_index.has_value()) continue;

          if (!matched_iter_arg_indices.insert(i).second) return std::nullopt;
          candidate_matches.push_back(AggregateLoopOutputMatch{out_param_index, *return_index, i});
          matched_output = true;
          break;
        }
        if (!matched_output) {
          matches_all_outputs = false;
          break;
        }
      }

      if (!matches_all_outputs) continue;
      for (size_t i = 0; i < candidate->iter_args_.size(); ++i) {
        if (matched_iter_arg_indices.count(i) == 0 && AsTensorTypeLike(candidate->iter_args_[i]->GetType())) {
          return std::nullopt;
        }
      }
      if (candidate->return_vars_.size() > candidate->iter_args_.size()) return std::nullopt;

      if (loop) return std::nullopt;
      loop = candidate;
      loop_matches = std::move(candidate_matches);
    }
    if (!loop) return std::nullopt;
    if (loop_matches.size() != out_indices.size()) return std::nullopt;

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
      size_t iter_arg_refs = 0;
    };

    std::unordered_map<size_t, AggregateUpdate> updates_by_iter_arg_index;
    std::unordered_map<const Var*, size_t> store_result_to_iter_arg_index;
    std::unordered_map<const Var*, size_t> store_result_to_iter_arg_refs;
    const bool has_seeded_scalar_defs = !scalar_defs.empty();
    auto resolve_update_index = [&](const Var* updated_iter_arg) -> std::optional<size_t> {
      if (!updated_iter_arg) return std::nullopt;
      for (const auto& match : loop_matches) {
        if (updated_iter_arg == loop->iter_args_[match.iter_arg_index].get()) {
          return match.iter_arg_index;
        }
      }
      auto chained_it = store_result_to_iter_arg_index.find(updated_iter_arg);
      if (chained_it != store_result_to_iter_arg_index.end()) return chained_it->second;
      return std::nullopt;
    };
    auto process_update_assign = [&](const AssignStmtPtr& assign) -> bool {
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!call) return false;

      const Var* updated_iter_arg = nullptr;
      std::vector<ExprPtr> window_shape;
      std::vector<ExprPtr> offsets;
      if (call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
        auto out_arg = AsVarLike(call->args_[2]);
        auto offset_tuple = As<MakeTuple>(call->args_[1]);
        auto tile_type = As<TileType>(call->args_[0]->GetType());
        if (out_arg && offset_tuple && tile_type) {
          updated_iter_arg = out_arg.get();
          window_shape = tile_type->shape_;
          offsets = offset_tuple->elements_;
        }
      } else if (call->op_->name_ == "tensor.assemble" && call->args_.size() >= 3) {
        auto parent_arg = AsVarLike(call->args_[0]);
        auto offset_tuple = As<MakeTuple>(call->args_[2]);
        auto source_type = As<TensorType>(call->args_[1]->GetType());
        if (parent_arg && offset_tuple && source_type) {
          updated_iter_arg = parent_arg.get();
          window_shape = source_type->shape_;
          offsets = offset_tuple->elements_;
        }
      }

      auto updated_iter_arg_index = resolve_update_index(updated_iter_arg);
      if (!updated_iter_arg_index.has_value()) return false;

      auto iter_arg = loop->iter_args_[*updated_iter_arg_index].get();
      const bool direct_iter_update = updated_iter_arg == iter_arg;
      size_t iter_refs = CountVarRefsInStmt(assign, iter_arg);
      auto chained_refs_it = store_result_to_iter_arg_refs.find(updated_iter_arg);
      if (chained_refs_it != store_result_to_iter_arg_refs.end()) {
        iter_refs += chained_refs_it->second;
      }
      if (direct_iter_update ||
          updates_by_iter_arg_index.find(*updated_iter_arg_index) == updates_by_iter_arg_index.end()) {
        updates_by_iter_arg_index[*updated_iter_arg_index] =
            AggregateUpdate{assign, std::move(window_shape), std::move(offsets), iter_refs};
      }
      store_result_to_iter_arg_index[assign->var_.get()] = *updated_iter_arg_index;
      store_result_to_iter_arg_refs[assign->var_.get()] = iter_refs;
      return true;
    };
    std::function<void(const StmtPtr&)> visit_nested_control_flow;
    visit_nested_control_flow = [&](const StmtPtr& stmt) {
      if (!stmt) return;
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) visit_nested_control_flow(child);
        return;
      }
      if (auto assign = As<AssignStmt>(stmt)) {
        process_update_assign(assign);
        return;
      }
      if (auto if_stmt = As<IfStmt>(stmt)) {
        visit_nested_control_flow(if_stmt->then_body_);
        if (if_stmt->else_body_.has_value()) visit_nested_control_flow(if_stmt->else_body_.value());
        auto then_yield = transform_utils::GetLastYieldStmt(if_stmt->then_body_);
        auto else_yield = if_stmt->else_body_.has_value()
                              ? transform_utils::GetLastYieldStmt(if_stmt->else_body_.value())
                              : nullptr;
        if (!then_yield || !else_yield || then_yield->value_.size() != if_stmt->return_vars_.size() ||
            else_yield->value_.size() != if_stmt->return_vars_.size()) {
          return;
        }
        for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
          auto then_var = AsVarLike(then_yield->value_[i]);
          auto else_var = AsVarLike(else_yield->value_[i]);
          if (!then_var || !else_var) continue;
          auto then_index = resolve_update_index(then_var.get());
          auto else_index = resolve_update_index(else_var.get());
          if (!then_index.has_value() || !else_index.has_value() || *then_index != *else_index) continue;
          store_result_to_iter_arg_index[if_stmt->return_vars_[i].get()] = *then_index;
          auto then_refs_it = store_result_to_iter_arg_refs.find(then_var.get());
          auto else_refs_it = store_result_to_iter_arg_refs.find(else_var.get());
          size_t refs = 0;
          if (then_refs_it != store_result_to_iter_arg_refs.end())
            refs = std::max(refs, then_refs_it->second);
          if (else_refs_it != store_result_to_iter_arg_refs.end())
            refs = std::max(refs, else_refs_it->second);
          store_result_to_iter_arg_refs[if_stmt->return_vars_[i].get()] = refs;
        }
      }
    };
    for (const auto& stmt : loop_body_stmts) {
      if (auto assign = As<AssignStmt>(stmt)) {
        if (process_update_assign(assign)) continue;
        if (As<ScalarType>(assign->var_->GetType())) {
          auto value = assign->value_;
          if (has_seeded_scalar_defs) {
            value = arith::Analyzer().Simplify(transform_utils::Substitute(value, scalar_defs));
          }
          scalar_defs[assign->var_.get()] = value;
        }
        continue;
      }
      if (auto yield = As<YieldStmt>(stmt)) {
        if (yield_stmt || yield->value_.size() != loop->return_vars_.size()) return std::nullopt;
        yield_stmt = yield;
        continue;
      }
      visit_nested_control_flow(stmt);
    }

    if (!yield_stmt || updates_by_iter_arg_index.size() != loop_matches.size()) return std::nullopt;

    std::unordered_set<const Var*> allowed;
    for (const auto& param : func->params_) allowed.insert(param.get());
    allowed.insert(loop->loop_var_.get());
    allowed.insert(extra_allowed_vars.begin(), extra_allowed_vars.end());

    CalleeRewriteAnalysis analysis;
    analysis.kind = RewriteKind::AggregateWindowLoop;

    for (const auto& match : loop_matches) {
      auto update_it = updates_by_iter_arg_index.find(match.iter_arg_index);
      if (update_it == updates_by_iter_arg_index.end()) return std::nullopt;
      const auto& update = update_it->second;
      auto store_assign = update.assign;

      auto yielded = AsVarLike(yield_stmt->value_[match.iter_arg_index]);
      if (!yielded) return std::nullopt;
      if (yielded.get() != store_assign->var_.get()) {
        auto yielded_it = store_result_to_iter_arg_index.find(yielded.get());
        if (yielded_it == store_result_to_iter_arg_index.end() ||
            yielded_it->second != match.iter_arg_index) {
          return std::nullopt;
        }
      }

      if (!As<TensorType>(loop->iter_args_[match.iter_arg_index]->GetType())) {
        return std::nullopt;
      }
      if (match.return_index != SIZE_MAX &&
          (match.iter_arg_index >= loop->return_vars_.size() ||
           !As<TensorType>(loop->return_vars_[match.iter_arg_index]->GetType()))) {
        return std::nullopt;
      }

      size_t total_out_refs = CountVarRefsInStmt(func->body_, func->params_[match.out_param_index].get());
      size_t store_out_refs = CountVarRefsInStmt(store_assign, func->params_[match.out_param_index].get());
      if (total_out_refs != store_out_refs + 1) return std::nullopt;

      size_t total_iter_refs = CountVarRefsInStmt(loop->body_, loop->iter_args_[match.iter_arg_index].get());
      if (total_iter_refs != update.iter_arg_refs) return std::nullopt;

      auto out_tensor_type = As<TensorType>(func->params_[match.out_param_index]->GetType());
      if (!out_tensor_type) return std::nullopt;
      if (update.offsets.size() != update.window_shape.size() ||
          update.offsets.size() != out_tensor_type->shape_.size()) {
        return std::nullopt;
      }

      std::vector<ExprPtr> base_offsets;
      std::vector<ExprPtr> local_offsets;
      std::vector<ExprPtr> window_shape;
      for (size_t i = 0; i < update.offsets.size(); ++i) {
        auto expanded = ExpandLoopLocalExpr(update.offsets[i], scalar_defs);
        if (!expanded.has_value()) return std::nullopt;
        if (!ExprReferencesOnlyVarsIn(*expanded, allowed)) return std::nullopt;

        auto ordered_offsets = GetOrderedLoopOffsets(*expanded, loop, *first_loop_value, *last_loop_value);
        if (!ordered_offsets.has_value()) return std::nullopt;

        auto span_expr = arith::Analyzer().Simplify(
            MakeAdd(MakeSub(ordered_offsets->max, ordered_offsets->min, func->span_), update.window_shape[i],
                    func->span_));
        auto span_ci = As<ConstInt>(span_expr);
        if (!span_ci || span_ci->value_ <= 0) return std::nullopt;

        base_offsets.push_back(ordered_offsets->min);
        local_offsets.push_back(arith::Analyzer().Simplify(
            MakeSub(update.offsets[i], ordered_offsets->min, update.offsets[i]->span_)));
        window_shape.push_back(std::make_shared<ConstInt>(span_ci->value_, DataType::INDEX, func->span_));
      }

      if (AreExprVectorsEqual(window_shape, out_tensor_type->shape_) && IsAllZeroOffsets(base_offsets)) {
        return std::nullopt;
      }

      analysis.outputs.push_back(OutputRewriteInfo{
          match.out_param_index, match.return_index, out_tensor_type->shape_, std::move(window_shape),
          std::move(base_offsets), std::move(local_offsets), match.iter_arg_index});
    }

    return analysis;
  }

  static std::optional<int64_t> GetFunctionCoreNum(const FunctionPtr& func) {
    if (!func) return std::nullopt;
    auto core_num_expr = func->GetAttr<ExprPtr>("core_num", nullptr);
    if (auto core_num_ci = As<ConstInt>(core_num_expr)) {
      if (core_num_ci->value_ > 0) return core_num_ci->value_;
    }
    auto core_num = func->GetAttr<int64_t>("core_num", 0);
    if (core_num <= 0) {
      core_num = static_cast<int64_t>(func->GetAttr<int>("core_num", 0));
    }
    if (core_num <= 0) return std::nullopt;
    return core_num;
  }

  static std::optional<DirectSpmdCall> GetDirectSpmdInnerCall(const ProgramPtr& program,
                                                              const FunctionPtr& spmd_func) {
    if (!program || !spmd_func || spmd_func->func_type_ != FunctionType::Spmd) return std::nullopt;
    auto body_stmts = FlattenToStmts(spmd_func->body_);
    CallPtr call;
    bool returns_value = false;
    if (body_stmts.size() >= 2) {
      auto call_assign = As<AssignStmt>(body_stmts[0]);
      auto ret_stmt = As<ReturnStmt>(body_stmts.back());
      call = call_assign ? As<Call>(call_assign->value_) : nullptr;
      if (!call || !ret_stmt || ret_stmt->value_.empty()) return std::nullopt;
      if (ret_stmt->value_.size() == 1) {
        auto ret_var = AsVarLike(ret_stmt->value_[0]);
        if (!ret_var || ret_var.get() != call_assign->var_.get()) return std::nullopt;
      } else {
        std::unordered_map<const Var*, int> tuple_item_indices;
        for (size_t i = 1; i + 1 < body_stmts.size(); ++i) {
          auto item_assign = As<AssignStmt>(body_stmts[i]);
          auto tuple_get = item_assign ? As<TupleGetItemExpr>(item_assign->value_) : nullptr;
          auto tuple_var = tuple_get ? AsVarLike(tuple_get->tuple_) : nullptr;
          if (!item_assign || !tuple_get || !tuple_var || tuple_var.get() != call_assign->var_.get()) {
            return std::nullopt;
          }
          tuple_item_indices[item_assign->var_.get()] = tuple_get->index_;
        }
        for (size_t i = 0; i < ret_stmt->value_.size(); ++i) {
          auto ret_var = AsVarLike(ret_stmt->value_[i]);
          if (!ret_var) return std::nullopt;
          auto item_it = tuple_item_indices.find(ret_var.get());
          if (item_it == tuple_item_indices.end() || item_it->second != static_cast<int>(i)) {
            return std::nullopt;
          }
        }
      }
      returns_value = true;
    } else if (body_stmts.size() == 1) {
      auto eval = As<EvalStmt>(body_stmts[0]);
      call = eval ? As<Call>(eval->expr_) : nullptr;
      if (!call) return std::nullopt;
    } else {
      return std::nullopt;
    }

    auto inner = program->GetFunction(GetCallFuncName(call));
    if (!inner || inner->func_type_ == FunctionType::Spmd ||
        inner->func_type_ == FunctionType::Orchestration || inner->func_type_ == FunctionType::Inline) {
      return std::nullopt;
    }
    if (!IsDirectParamForwardingCall(spmd_func, call, inner)) {
      return std::nullopt;
    }
    return DirectSpmdCall{inner, call, returns_value};
  }

  static std::optional<FunctionPtr> GetDirectSpmdInnerFunction(const ProgramPtr& program,
                                                               const FunctionPtr& spmd_func) {
    auto inner_call = GetDirectSpmdInnerCall(program, spmd_func);
    if (!inner_call.has_value()) return std::nullopt;
    return inner_call->inner_func;
  }

  static std::optional<VarPtr> FindSingleBlockIdxVar(const FunctionPtr& func) {
    if (!func) return std::nullopt;
    VarPtr block_idx;
    auto body_stmts = FlattenToStmts(func->body_);
    for (const auto& stmt : body_stmts) {
      auto assign = As<AssignStmt>(stmt);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!call || call->op_->name_ != "tile.get_block_idx") continue;
      if (block_idx) return std::nullopt;
      block_idx = assign->var_;
    }
    if (!block_idx) return std::nullopt;
    return block_idx;
  }

  static std::optional<ExprPtr> TryRemoveBaseOffset(const ExprPtr& expr, const ExprPtr& base) {
    if (!expr || !base) return std::nullopt;
    auto local = arith::Analyzer().Simplify(MakeSub(expr, base, expr->span_));
    if (!local || local.get() == expr.get()) return std::nullopt;
    return local;
  }

  static std::unordered_map<const Var*, ExprPtr> CollectTopLevelScalarDefsBeforeLoop(
      const FunctionPtr& func, const ForStmtPtr& loop,
      const std::unordered_set<const Var*>& preserved_vars = {}) {
    std::unordered_map<const Var*, ExprPtr> defs;
    if (!func || !loop) return defs;
    for (const auto& stmt : FlattenToStmts(func->body_)) {
      if (stmt.get() == loop.get()) break;
      auto assign = As<AssignStmt>(stmt);
      if (!assign || !As<ScalarType>(assign->var_->GetType())) continue;
      if (preserved_vars.count(assign->var_.get()) != 0) continue;
      defs[assign->var_.get()] =
          arith::Analyzer().Simplify(transform_utils::Substitute(assign->value_, defs));
    }
    return defs;
  }

  static std::optional<ForStmtPtr> FindSingleTopLevelAggregateLoop(const FunctionPtr& func) {
    if (!func) return std::nullopt;
    ForStmtPtr result;
    for (const auto& stmt : FlattenToStmts(func->body_)) {
      auto loop = As<ForStmt>(stmt);
      if (!loop) continue;
      if (result) return std::nullopt;
      result = loop;
    }
    if (!result) return std::nullopt;
    return result;
  }

  static std::optional<InputRewriteInfo> AnalyzeSpmdInputWindow(const FunctionPtr& func, size_t param_index,
                                                                const Var* block_idx_var,
                                                                const ExprPtr& base_row_offset,
                                                                const ExprPtr& row_block_shape) {
    if (!func || param_index >= func->params_.size() || !block_idx_var || !base_row_offset ||
        !row_block_shape) {
      return std::nullopt;
    }
    if (param_index >= func->param_directions_.size() ||
        func->param_directions_[param_index] != ParamDirection::In) {
      return std::nullopt;
    }
    auto tensor_type = As<TensorType>(func->params_[param_index]->GetType());
    if (!tensor_type || tensor_type->shape_.size() != 2) return std::nullopt;

    auto total_refs_by_var = CountAllVarRefsInStmt(func->body_);
    auto total_refs_it = total_refs_by_var.find(func->params_[param_index].get());
    const size_t total_refs = total_refs_it == total_refs_by_var.end() ? 0 : total_refs_it->second;

    std::optional<InputRewriteInfo> result;
    size_t matched_refs = 0;
    auto aggregate_loop = FindSingleTopLevelAggregateLoop(func);
    if (!aggregate_loop.has_value()) return std::nullopt;
    auto scalar_defs = CollectTopLevelScalarDefsBeforeLoop(func, *aggregate_loop, {block_idx_var});
    bool invalid = false;
    std::function<void(const StmtPtr&, std::unordered_map<const Var*, ExprPtr>&)> scan_stmt;
    scan_stmt = [&](const StmtPtr& stmt, std::unordered_map<const Var*, ExprPtr>& defs) {
      if (!stmt || invalid) return;
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) scan_stmt(child, defs);
        return;
      }
      if (auto if_stmt = As<IfStmt>(stmt)) {
        auto then_defs = defs;
        scan_stmt(if_stmt->then_body_, then_defs);
        if (if_stmt->else_body_.has_value()) {
          auto else_defs = defs;
          scan_stmt(if_stmt->else_body_.value(), else_defs);
        }
        return;
      }
      if (auto loop_stmt = As<ForStmt>(stmt)) {
        scan_stmt(loop_stmt->body_, defs);
        return;
      }
      if (auto loop_stmt = As<WhileStmt>(stmt)) {
        scan_stmt(loop_stmt->body_, defs);
        return;
      }
      auto assign = As<AssignStmt>(stmt);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (assign && As<ScalarType>(assign->var_->GetType())) {
        defs[assign->var_.get()] =
            arith::Analyzer().Simplify(transform_utils::Substitute(assign->value_, defs));
      }
      if (!call || call->op_->name_ != "tile.load" || call->args_.size() < 3) return;

      auto parent = AsVarLike(call->args_[0]);
      auto offset_tuple = As<MakeTuple>(call->args_[1]);
      auto shape_tuple = As<MakeTuple>(call->args_[2]);
      if (!parent || parent.get() != func->params_[param_index].get() || !offset_tuple || !shape_tuple) {
        return;
      }
      if (offset_tuple->elements_.size() != 2 || shape_tuple->elements_.size() != 2) {
        invalid = true;
        return;
      }
      if (!AreExprVectorsEqual(
              shape_tuple->elements_,
              {std::make_shared<ConstInt>(1, DataType::INDEX, func->span_), tensor_type->shape_[1]}) ||
          !IsConstValue(offset_tuple->elements_[1], 0)) {
        invalid = true;
        return;
      }

      auto expanded_row_offset =
          arith::Analyzer().Simplify(transform_utils::Substitute(offset_tuple->elements_[0], defs));
      auto local_row = TryRemoveBaseOffset(expanded_row_offset, base_row_offset);
      if (!local_row.has_value()) {
        invalid = true;
        return;
      }

      std::vector<ExprPtr> window_shape{row_block_shape, tensor_type->shape_[1]};
      std::vector<ExprPtr> callsite_offsets{base_row_offset,
                                            std::make_shared<ConstInt>(0, DataType::INDEX, func->span_)};
      std::vector<ExprPtr> local_offsets{*local_row,
                                         std::make_shared<ConstInt>(0, DataType::INDEX, func->span_)};
      InputRewriteInfo info{param_index, tensor_type->shape_, std::move(window_shape),
                            std::move(callsite_offsets), std::move(local_offsets)};
      if (result.has_value() && (!AreExprVectorsEqual(result->window_shape, info.window_shape) ||
                                 !AreExprVectorsEqual(result->callsite_offsets, info.callsite_offsets))) {
        invalid = true;
        return;
      }
      result = std::move(info);
      matched_refs += CountVarRefsInStmt(stmt, parent.get());
    };
    scan_stmt((*aggregate_loop)->body_, scalar_defs);
    if (invalid) return std::nullopt;
    if (!result.has_value() || total_refs != matched_refs) return std::nullopt;
    return result;
  }

  static std::optional<InputRewriteInfo> AnalyzeDirectSpmdInputWindow(const FunctionPtr& func,
                                                                      size_t param_index,
                                                                      const Var* block_idx_var) {
    if (!func || param_index >= func->params_.size() || !block_idx_var) return std::nullopt;
    if (param_index >= func->param_directions_.size() ||
        func->param_directions_[param_index] != ParamDirection::In) {
      return std::nullopt;
    }

    auto tensor_type = As<TensorType>(func->params_[param_index]->GetType());
    if (!tensor_type) return std::nullopt;

    auto total_refs_by_var = CountAllVarRefsInStmt(func->body_);
    auto total_refs_it = total_refs_by_var.find(func->params_[param_index].get());
    const size_t total_refs = total_refs_it == total_refs_by_var.end() ? 0 : total_refs_it->second;

    auto body_stmts = FlattenToStmts(func->body_);
    std::unordered_map<const Var*, ExprPtr> scalar_defs;
    std::optional<InputRewriteInfo> result;
    size_t matched_refs = 0;
    std::unordered_set<const Var*> allowed;
    for (const auto& param : func->params_) allowed.insert(param.get());
    allowed.insert(block_idx_var);

    for (const auto& stmt : body_stmts) {
      auto assign = As<AssignStmt>(stmt);
      if (assign && As<ScalarType>(assign->var_->GetType()) && assign->var_.get() != block_idx_var) {
        scalar_defs[assign->var_.get()] =
            arith::Analyzer().Simplify(transform_utils::Substitute(assign->value_, scalar_defs));
      }

      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!call) continue;

      size_t shape_arg_index = SIZE_MAX;
      size_t offset_arg_index = SIZE_MAX;
      if (call->op_->name_ == "tensor.slice" && call->args_.size() >= 3) {
        shape_arg_index = 1;
        offset_arg_index = 2;
      } else if (call->op_->name_ == "tile.load" && call->args_.size() >= 3) {
        offset_arg_index = 1;
        shape_arg_index = 2;
      } else {
        continue;
      }

      auto parent = AsVarLike(call->args_[0]);
      if (!parent || parent.get() != func->params_[param_index].get()) continue;

      auto shape_tuple = As<MakeTuple>(call->args_[shape_arg_index]);
      auto offset_tuple = As<MakeTuple>(call->args_[offset_arg_index]);
      if (!shape_tuple || !offset_tuple || shape_tuple->elements_.size() != tensor_type->shape_.size() ||
          offset_tuple->elements_.size() != tensor_type->shape_.size()) {
        return std::nullopt;
      }

      std::vector<ExprPtr> window_shape;
      window_shape.reserve(shape_tuple->elements_.size());
      for (const auto& dim : shape_tuple->elements_) {
        auto expanded = arith::Analyzer().Simplify(transform_utils::Substitute(dim, scalar_defs));
        if (!ExprReferencesOnlyVarsIn(expanded, allowed)) return std::nullopt;
        window_shape.push_back(std::move(expanded));
      }

      std::vector<ExprPtr> callsite_offsets;
      callsite_offsets.reserve(offset_tuple->elements_.size());
      std::vector<ExprPtr> local_zero_offsets;
      local_zero_offsets.reserve(offset_tuple->elements_.size());
      for (const auto& offset : offset_tuple->elements_) {
        auto expanded = arith::Analyzer().Simplify(transform_utils::Substitute(offset, scalar_defs));
        if (!ExprReferencesOnlyVarsIn(expanded, allowed)) return std::nullopt;
        callsite_offsets.push_back(std::move(expanded));
        local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
      }

      if (AreExprVectorsEqual(window_shape, tensor_type->shape_) && IsAllZeroOffsets(callsite_offsets)) {
        return std::nullopt;
      }

      InputRewriteInfo info{param_index, tensor_type->shape_, std::move(window_shape),
                            std::move(callsite_offsets), std::move(local_zero_offsets)};
      if (result.has_value() && (!AreExprVectorsEqual(result->window_shape, info.window_shape) ||
                                 !AreExprVectorsEqual(result->callsite_offsets, info.callsite_offsets))) {
        return std::nullopt;
      }
      result = std::move(info);
      matched_refs += CountVarRefsInStmt(stmt, parent.get());
    }

    if (!result.has_value() || total_refs != matched_refs) return std::nullopt;
    return result;
  }

  static std::optional<CalleeRewriteAnalysis> AnalyzeSpmdOutputs(const FunctionPtr& inner_func,
                                                                 const std::vector<size_t>& out_indices,
                                                                 const Var* block_idx_var) {
    if (!inner_func || out_indices.empty() || !block_idx_var) return std::nullopt;

    auto aggregate_loop = FindSingleTopLevelAggregateLoop(inner_func);
    if (aggregate_loop.has_value()) {
      auto scalar_defs = CollectTopLevelScalarDefsBeforeLoop(inner_func, *aggregate_loop, {block_idx_var});
      auto analysis =
          AnalyzeAggregateWindowLoop(inner_func, out_indices, std::move(scalar_defs), {block_idx_var});
      if (analysis.has_value() && !analysis->outputs.empty()) return analysis;
    }

    CalleeRewriteAnalysis analysis;
    analysis.kind = RewriteKind::SpmdRowBlock;
    bool matched_any = false;
    std::unordered_map<const Var*, ExprPtr> scalar_defs;
    for (const auto& stmt : FlattenToStmts(inner_func->body_)) {
      auto assign = As<AssignStmt>(stmt);
      if (!assign || !As<ScalarType>(assign->var_->GetType()) || assign->var_.get() == block_idx_var) {
        continue;
      }
      scalar_defs[assign->var_.get()] =
          arith::Analyzer().Simplify(transform_utils::Substitute(assign->value_, scalar_defs));
    }
    for (const auto& out_index : out_indices) {
      auto info = AnalyzeFinalStore(inner_func, out_index);
      if (!info.has_value()) return std::nullopt;
      auto out_tensor_type = As<TensorType>(inner_func->params_[out_index]->GetType());
      if (!out_tensor_type) return std::nullopt;
      if (AreExprVectorsEqual(info->window_shape, out_tensor_type->shape_) &&
          IsAllZeroOffsets(info->offsets)) {
        return std::nullopt;
      }

      std::unordered_set<const Var*> allowed;
      for (const auto& param : inner_func->params_) allowed.insert(param.get());
      allowed.insert(block_idx_var);
      std::vector<ExprPtr> window_shape;
      window_shape.reserve(info->window_shape.size());
      for (const auto& expr : info->window_shape) {
        auto expanded = arith::Analyzer().Simplify(transform_utils::Substitute(expr, scalar_defs));
        if (!ExprReferencesOnlyVarsIn(expanded, allowed)) return std::nullopt;
        window_shape.push_back(std::move(expanded));
      }
      std::vector<ExprPtr> offsets;
      offsets.reserve(info->offsets.size());
      for (const auto& expr : info->offsets) {
        auto expanded = arith::Analyzer().Simplify(transform_utils::Substitute(expr, scalar_defs));
        if (!ExprReferencesOnlyVarsIn(expanded, allowed)) return std::nullopt;
        offsets.push_back(std::move(expanded));
      }

      std::vector<ExprPtr> local_zero_offsets;
      local_zero_offsets.reserve(offsets.size());
      for (size_t i = 0; i < offsets.size(); ++i) {
        local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, inner_func->span_));
      }
      analysis.outputs.push_back(OutputRewriteInfo{out_index, info->return_index, out_tensor_type->shape_,
                                                   std::move(window_shape), std::move(offsets),
                                                   std::move(local_zero_offsets), SIZE_MAX});
      matched_any = true;
    }
    if (!matched_any) return std::nullopt;
    return analysis;
  }

  struct NestedStoreAccess {
    size_t out_param_index;
    std::vector<ExprPtr> window_shape;
    std::vector<ExprPtr> offsets;
    std::vector<ForStmtPtr> loops;
  };

  static std::optional<std::pair<std::vector<ExprPtr>, std::vector<ExprPtr>>> ParseWindowStore(
      const AssignStmtPtr& assign) {
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    if (!call) return std::nullopt;
    if (call->op_->name_ == "tile.store" && call->args_.size() >= 3) {
      auto shape = As<TileType>(call->args_[0]->GetType());
      auto offsets = As<MakeTuple>(call->args_[1]);
      if (!shape || !offsets) return std::nullopt;
      return std::make_pair(shape->shape_, offsets->elements_);
    }
    if (call->op_->name_ == "tensor.assemble" && call->args_.size() >= 3) {
      auto shape = As<TensorType>(call->args_[1]->GetType());
      auto offsets = As<MakeTuple>(call->args_[2]);
      if (!shape || !offsets) return std::nullopt;
      return std::make_pair(shape->shape_, offsets->elements_);
    }
    return std::nullopt;
  }

  static ExprPtr GetWindowStoreTarget(const AssignStmtPtr& assign) {
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    if (!call) return nullptr;
    if (call->op_->name_ == "tile.store" && call->args_.size() >= 3) return call->args_[2];
    if (call->op_->name_ == "tensor.assemble" && call->args_.size() >= 3) return call->args_[0];
    return nullptr;
  }

  static bool IsRootedExpr(const ExprPtr& expr, const std::unordered_set<const Var*>& roots) {
    auto var = AsVarLike(expr);
    return var && roots.count(var.get()) != 0;
  }

  static std::optional<CalleeRewriteAnalysis> AnalyzeNestedSpmdOutputWindows(
      const FunctionPtr& inner_func, const std::vector<size_t>& out_indices, const Var* block_idx_var) {
    if (!inner_func || out_indices.empty() || !block_idx_var) return std::nullopt;

    std::unordered_map<const Var*, size_t> root_to_out_index;
    std::unordered_set<const Var*> roots;
    for (auto out_index : out_indices) {
      if (out_index >= inner_func->params_.size()) return std::nullopt;
      root_to_out_index[inner_func->params_[out_index].get()] = out_index;
      roots.insert(inner_func->params_[out_index].get());
    }

    std::vector<NestedStoreAccess> accesses;
    std::function<void(const StmtPtr&, std::unordered_set<const Var*>&,
                       std::unordered_map<const Var*, ExprPtr>&, std::vector<ForStmtPtr>&)>
        scan_stmt;
    scan_stmt = [&](const StmtPtr& stmt, std::unordered_set<const Var*>& active_roots,
                    std::unordered_map<const Var*, ExprPtr>& scalar_defs,
                    std::vector<ForStmtPtr>& active_loops) {
      if (!stmt) return;
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) scan_stmt(child, active_roots, scalar_defs, active_loops);
        return;
      }
      if (auto loop = As<ForStmt>(stmt)) {
        auto loop_roots = active_roots;
        auto loop_defs = scalar_defs;
        for (const auto& iter_arg : loop->iter_args_) {
          if (iter_arg && IsRootedExpr(iter_arg->initValue_, active_roots)) {
            loop_roots.insert(iter_arg.get());
          }
        }
        active_loops.push_back(loop);
        scan_stmt(loop->body_, loop_roots, loop_defs, active_loops);
        active_loops.pop_back();

        auto yield = transform_utils::GetLastYieldStmt(loop->body_);
        for (size_t i = 0; i < loop->return_vars_.size(); ++i) {
          if (yield && i < yield->value_.size() && IsRootedExpr(yield->value_[i], loop_roots)) {
            active_roots.insert(loop->return_vars_[i].get());
          }
        }
        return;
      }
      if (auto if_stmt = As<IfStmt>(stmt)) {
        auto then_roots = active_roots;
        auto then_defs = scalar_defs;
        scan_stmt(if_stmt->then_body_, then_roots, then_defs, active_loops);
        if (if_stmt->else_body_.has_value()) {
          auto else_roots = active_roots;
          auto else_defs = scalar_defs;
          scan_stmt(if_stmt->else_body_.value(), else_roots, else_defs, active_loops);
          for (const auto* root : else_roots) {
            if (then_roots.count(root) != 0) active_roots.insert(root);
          }
        } else {
          active_roots.insert(then_roots.begin(), then_roots.end());
        }
        return;
      }
      auto assign = As<AssignStmt>(stmt);
      if (!assign) return;
      if (As<ScalarType>(assign->var_->GetType()) && assign->var_.get() != block_idx_var) {
        scalar_defs[assign->var_.get()] =
            arith::Analyzer().Simplify(transform_utils::Substitute(assign->value_, scalar_defs));
      }

      auto target = GetWindowStoreTarget(assign);
      auto target_var = AsVarLike(target);
      if (!target_var || active_roots.count(target_var.get()) == 0) return;

      auto parsed = ParseWindowStore(assign);
      if (!parsed.has_value()) return;
      auto root_it = root_to_out_index.find(target_var.get());
      size_t out_index = SIZE_MAX;
      if (root_it != root_to_out_index.end()) {
        out_index = root_it->second;
      } else {
        for (const auto& [root, candidate_out_index] : root_to_out_index) {
          if (active_roots.count(root) != 0) {
            out_index = candidate_out_index;
            break;
          }
        }
      }
      if (out_index == SIZE_MAX) return;

      std::vector<ExprPtr> offsets;
      offsets.reserve(parsed->second.size());
      for (const auto& offset : parsed->second) {
        offsets.push_back(arith::Analyzer().Simplify(transform_utils::Substitute(offset, scalar_defs)));
      }
      accesses.push_back(NestedStoreAccess{out_index, parsed->first, std::move(offsets), active_loops});
      active_roots.insert(assign->var_.get());
      root_to_out_index[assign->var_.get()] = out_index;
    };

    std::unordered_map<const Var*, ExprPtr> scalar_defs;
    std::vector<ForStmtPtr> active_loops;
    scan_stmt(inner_func->body_, roots, scalar_defs, active_loops);
    if (accesses.empty()) return std::nullopt;

    ReturnStmtPtr ret_stmt;
    for (const auto& stmt : FlattenToStmts(inner_func->body_)) {
      if (auto ret = As<ReturnStmt>(stmt)) {
        ret_stmt = ret;
        break;
      }
    }
    if (!ret_stmt) return std::nullopt;

    CalleeRewriteAnalysis analysis;
    analysis.kind = RewriteKind::SpmdRowBlock;
    for (auto out_index : out_indices) {
      auto out_tensor_type = As<TensorType>(inner_func->params_[out_index]->GetType());
      if (!out_tensor_type) return std::nullopt;

      std::vector<const NestedStoreAccess*> output_accesses;
      for (const auto& access : accesses) {
        if (access.out_param_index == out_index) output_accesses.push_back(&access);
      }
      if (output_accesses.empty()) return std::nullopt;

      std::vector<ExprPtr> base_offsets;
      std::vector<ExprPtr> local_offsets;
      std::vector<ExprPtr> window_shape;
      for (size_t dim = 0; dim < out_tensor_type->shape_.size(); ++dim) {
        bool full_dim = false;
        ExprPtr dim_min;
        ExprPtr dim_max;
        ExprPtr store_shape;
        for (const auto* access : output_accesses) {
          if (dim >= access->offsets.size() || dim >= access->window_shape.size()) return std::nullopt;
          auto cur_min = access->offsets[dim];
          auto cur_max = access->offsets[dim];
          for (const auto& loop : access->loops) {
            auto trip_count = GetKnownPositiveTripCount(loop);
            if (!trip_count.has_value() || *trip_count <= 0) continue;
            auto first = GetLoopValueAtTrip(loop, 0);
            auto last = GetLoopValueAtTrip(loop, *trip_count - 1);
            if (!first.has_value() || !last.has_value()) continue;
            auto min_offsets = GetOrderedLoopOffsets(cur_min, loop, *first, *last);
            auto max_offsets = GetOrderedLoopOffsets(cur_max, loop, *first, *last);
            if (!min_offsets.has_value() || !max_offsets.has_value()) continue;
            cur_min = min_offsets->min;
            cur_max = max_offsets->max;
          }

          std::unordered_set<const Var*> allowed{block_idx_var};
          if (!ExprReferencesOnlyVarsIn(cur_min, allowed) || !ExprReferencesOnlyVarsIn(cur_max, allowed)) {
            full_dim = true;
            break;
          }
          if (!dim_min) {
            dim_min = cur_min;
            dim_max = cur_max;
            store_shape = access->window_shape[dim];
          } else if (!AreExprVectorsEqual({dim_min}, {cur_min}) ||
                     !AreExprVectorsEqual({dim_max}, {cur_max}) ||
                     !AreExprVectorsEqual({store_shape}, {access->window_shape[dim]})) {
            full_dim = true;
            break;
          }
        }

        if (full_dim) {
          base_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, inner_func->span_));
          local_offsets.push_back(output_accesses.front()->offsets[dim]);
          window_shape.push_back(out_tensor_type->shape_[dim]);
          continue;
        }
        if (!dim_min || !dim_max || !store_shape) return std::nullopt;
        base_offsets.push_back(dim_min);
        local_offsets.push_back(arith::Analyzer().Simplify(MakeSub(
            output_accesses.front()->offsets[dim], dim_min, output_accesses.front()->offsets[dim]->span_)));
        window_shape.push_back(arith::Analyzer().Simplify(
            MakeAdd(MakeSub(dim_max, dim_min, inner_func->span_), store_shape, inner_func->span_)));
      }

      if (AreExprVectorsEqual(window_shape, out_tensor_type->shape_) && IsAllZeroOffsets(base_offsets)) {
        return std::nullopt;
      }
      std::optional<size_t> return_index;
      for (const auto& mapping : BuildOutParamReturnMappings(inner_func, /*include_inout=*/true)) {
        if (mapping.param_index == out_index) {
          return_index = mapping.return_index;
          break;
        }
      }
      if (!return_index.has_value()) return std::nullopt;
      analysis.outputs.push_back(OutputRewriteInfo{out_index, *return_index, out_tensor_type->shape_,
                                                   std::move(window_shape), std::move(base_offsets),
                                                   std::move(local_offsets), SIZE_MAX});
    }
    return analysis;
  }

  static std::optional<CalleeRewriteAnalysis> AnalyzeSpmdRowBlock(const ProgramPtr& program,
                                                                  const FunctionPtr& spmd_func,
                                                                  const std::vector<size_t>& out_indices) {
    auto core_num = GetFunctionCoreNum(spmd_func);
    auto inner_func = GetDirectSpmdInnerFunction(program, spmd_func);
    if (!core_num.has_value() || !inner_func.has_value() || out_indices.empty()) return std::nullopt;

    auto block_idx = FindSingleBlockIdxVar(*inner_func);
    if (!block_idx.has_value()) return std::nullopt;

    auto analysis = AnalyzeSpmdOutputs(*inner_func, out_indices, block_idx->get());
    if (!analysis.has_value()) {
      analysis = AnalyzeNestedSpmdOutputWindows(*inner_func, out_indices, block_idx->get());
    }
    if (!analysis.has_value() || analysis->outputs.empty()) return std::nullopt;

    std::optional<ExprPtr> base_row_offset;
    std::optional<ExprPtr> row_block_shape;
    bool can_try_row_block_inputs = true;
    for (const auto& output : analysis->outputs) {
      if (output.window_shape.size() != output.callsite_offsets.size() ||
          output.window_shape.size() != output.local_store_offsets.size()) {
        return std::nullopt;
      }
      if (!ExprReferencesOnlyVarsIn(output.callsite_offsets[0], {block_idx->get()})) return std::nullopt;

      auto tensor_type = As<TensorType>((*inner_func)->params_[output.out_param_index]->GetType());
      if (!tensor_type || tensor_type->shape_.size() != output.window_shape.size()) {
        return std::nullopt;
      }
      const bool row_block_like =
          output.window_shape.size() == 2 && output.callsite_offsets.size() == 2 &&
          output.local_store_offsets.size() == 2 && IsConstValue(output.callsite_offsets[1], 0) &&
          IsConstValue(output.local_store_offsets[1], 0) &&
          AreExprVectorsEqual(output.window_shape, {output.window_shape[0], tensor_type->shape_[1]});
      if (!row_block_like) {
        can_try_row_block_inputs = false;
        continue;
      }
      if (!base_row_offset.has_value()) {
        base_row_offset = output.callsite_offsets[0];
        row_block_shape = output.window_shape[0];
      } else if (!AreExprVectorsEqual({*base_row_offset}, {output.callsite_offsets[0]}) ||
                 !AreExprVectorsEqual({*row_block_shape}, {output.window_shape[0]})) {
        can_try_row_block_inputs = false;
      }
    }

    analysis->kind = RewriteKind::SpmdRowBlock;
    analysis->spmd_core_num = *core_num;
    analysis->spmd_inner_func = *inner_func;
    analysis->spmd_block_idx_var = *block_idx;
    analysis->inputs.clear();
    for (size_t i = 0; i < (*inner_func)->params_.size() && i < (*inner_func)->param_directions_.size();
         ++i) {
      std::optional<InputRewriteInfo> input;
      if (can_try_row_block_inputs && base_row_offset.has_value() && row_block_shape.has_value()) {
        input = AnalyzeSpmdInputWindow(*inner_func, i, block_idx->get(), *base_row_offset, *row_block_shape);
      }
      if (!input.has_value()) {
        input = AnalyzeDirectSpmdInputWindow(*inner_func, i, block_idx->get());
      }
      if (input.has_value()) analysis->inputs.push_back(std::move(*input));
    }
    return analysis;
  }

  static FunctionPtr GetCalleeRewriteSource(const FunctionPtr& func, const CalleeRewriteAnalysis& analysis) {
    if (analysis.kind == RewriteKind::SpmdRowBlock) return analysis.spmd_inner_func;
    return func;
  }

  static std::optional<StmtPtr> RemoveBlockIdxDefinition(const StmtPtr& body, const Var* block_idx_var) {
    if (!body || !block_idx_var) return std::nullopt;
    auto stmts = FlattenToStmts(body);
    std::vector<StmtPtr> filtered;
    filtered.reserve(stmts.size());
    bool removed = false;
    for (const auto& stmt : stmts) {
      auto assign = As<AssignStmt>(stmt);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (assign && assign->var_.get() == block_idx_var && call && call->op_->name_ == "tile.get_block_idx") {
        if (removed) return std::nullopt;
        removed = true;
        continue;
      }
      filtered.push_back(stmt);
    }
    if (!removed) return std::nullopt;
    return SeqStmts::Flatten(std::move(filtered), body->span_);
  }

  static ForStmtPtr FindAggregateLoopInBody(const StmtPtr& body,
                                            const std::vector<OutputRewriteInfo>& outputs,
                                            const std::vector<VarPtr>& new_params,
                                            size_t param_index_offset = 0) {
    auto body_stmts = FlattenToStmts(body);
    auto ret_stmt = body_stmts.empty() ? nullptr : As<ReturnStmt>(body_stmts.back());
    if (!ret_stmt) return nullptr;

    ForStmtPtr matched_loop;
    for (const auto& stmt : body_stmts) {
      auto candidate = As<ForStmt>(stmt);
      if (!candidate) continue;

      bool matches_outputs = true;
      for (const auto& output : outputs) {
        size_t new_out_param_index = output.out_param_index + param_index_offset;
        if (output.iter_arg_index >= candidate->iter_args_.size() ||
            new_out_param_index >= new_params.size()) {
          matches_outputs = false;
          break;
        }
        auto init_var = AsVarLike(candidate->iter_args_[output.iter_arg_index]->initValue_);
        if (!init_var) {
          matches_outputs = false;
          break;
        }
        if (init_var.get() != new_params[new_out_param_index].get()) {
          matches_outputs = false;
          break;
        }
        if (output.return_index != SIZE_MAX) {
          if (output.iter_arg_index >= candidate->return_vars_.size() ||
              output.return_index >= ret_stmt->value_.size()) {
            matches_outputs = false;
            break;
          }
          auto returned = AsVarLike(ret_stmt->value_[output.return_index]);
          if (!returned || returned.get() != candidate->return_vars_[output.iter_arg_index].get()) {
            matches_outputs = false;
            break;
          }
        }
      }
      if (!matches_outputs) continue;
      if (matched_loop) return nullptr;
      matched_loop = candidate;
    }
    return matched_loop;
  }

  static StmtPtr LocalizeAggregateLoopTypes(
      const StmtPtr& body, const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars) {
    class AggregateLoopTypeLocalizer : public IRMutator {
     public:
      explicit AggregateLoopTypeLocalizer(const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars)
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
        return changed ? new_stmt : op;
      }

     private:
      const std::unordered_map<const Var*, TypePtr>& narrowed_return_vars_;
    };

    AggregateLoopTypeLocalizer type_localizer(narrowed_return_vars);
    return type_localizer.VisitStmt(body);
  }

  AnalysisMap Analyze(const ProgramPtr& program) {
    AnalysisMap analyses;
    for (const auto& [gvar, func] : program->functions_) {
      if (!func || pypto::codegen::IsBuiltinOp(func->name_) ||
          func->func_type_ == FunctionType::Orchestration || func->func_type_ == FunctionType::Inline) {
        continue;
      }

      if (func->func_type_ == FunctionType::Spmd) {
        auto out_indices = CollectOutParamIndices(func, /*include_inout=*/true);
        if (out_indices.empty()) continue;
        auto spmd_analysis = AnalyzeSpmdRowBlock(program, func, out_indices);
        if (spmd_analysis.has_value() && !spmd_analysis->outputs.empty()) {
          analyses.emplace(func->name_, std::move(*spmd_analysis));
        }
        continue;
      }

      auto out_indices = CollectOutParamIndices(func);
      if (out_indices.empty()) continue;

      CalleeRewriteAnalysis analysis;
      bool all_final = true;
      for (const auto& out_index : out_indices) {
        auto info = AnalyzeFinalStore(func, out_index);
        if (!info.has_value()) {
          all_final = false;
          break;
        }

        auto out_tensor_type = As<TensorType>(func->params_[out_index]->GetType());
        if (!out_tensor_type) {
          all_final = false;
          break;
        }
        if (AreExprVectorsEqual(info->window_shape, out_tensor_type->shape_) &&
            IsAllZeroOffsets(info->offsets)) {
          all_final = false;
          break;
        }

        std::unordered_set<const Var*> allowed_params;
        for (const auto& param : func->params_) allowed_params.insert(param.get());
        bool exprs_ok = true;
        for (const auto& expr : info->window_shape) {
          if (!ExprReferencesOnlyVarsIn(expr, allowed_params)) {
            exprs_ok = false;
            break;
          }
        }
        for (const auto& expr : info->offsets) {
          if (!ExprReferencesOnlyVarsIn(expr, allowed_params)) {
            exprs_ok = false;
            break;
          }
        }
        if (!exprs_ok) {
          all_final = false;
          break;
        }

        std::vector<ExprPtr> local_zero_offsets;
        local_zero_offsets.reserve(info->offsets.size());
        for (size_t i = 0; i < info->offsets.size(); ++i) {
          local_zero_offsets.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, func->span_));
        }
        analysis.outputs.push_back(OutputRewriteInfo{out_index, info->return_index, out_tensor_type->shape_,
                                                     info->window_shape, info->offsets, local_zero_offsets,
                                                     SIZE_MAX});
      }
      if (all_final && !analysis.outputs.empty()) {
        analysis.kind = RewriteKind::FinalStore;
        analysis.inputs = AnalyzeInputWindows(func, analysis.outputs);
        analyses.emplace(func->name_, std::move(analysis));
        continue;
      }

      auto aggregate_analysis = AnalyzeAggregateWindowLoop(func, out_indices);
      if (aggregate_analysis.has_value() && !aggregate_analysis->outputs.empty()) {
        aggregate_analysis->inputs = AnalyzeInputWindows(func, aggregate_analysis->outputs);
        analyses.emplace(func->name_, std::move(*aggregate_analysis));
      }
    }
    FilterInputWindowsByCallsites(program, &analyses);
    return analyses;
  }

  FunctionPtr RewriteCallee(const ProgramPtr& program, const FunctionPtr& func,
                            const CalleeRewriteAnalysis& analysis) {
    auto source_func = GetCalleeRewriteSource(func, analysis);
    if (!source_func) return nullptr;

    std::vector<VarPtr> new_params;
    new_params.reserve(source_func->params_.size() + (analysis.kind == RewriteKind::SpmdRowBlock ? 1 : 0));
    std::vector<TypePtr> new_return_types = source_func->return_types_;
    auto new_param_directions = source_func->param_directions_;

    std::unordered_map<const Var*, ExprPtr> seed;
    const size_t param_index_offset = analysis.kind == RewriteKind::SpmdRowBlock ? 1 : 0;
    if (analysis.kind == RewriteKind::SpmdRowBlock) {
      if (!analysis.spmd_block_idx_var) return nullptr;
      auto block_idx_param = std::make_shared<Var>("block_idx", analysis.spmd_block_idx_var->GetType(),
                                                   analysis.spmd_block_idx_var->span_);
      new_params.push_back(block_idx_param);
      new_param_directions.insert(new_param_directions.begin(), ParamDirection::In);
      seed[analysis.spmd_block_idx_var.get()] = block_idx_param;
    }
    for (size_t i = 0; i < source_func->params_.size(); ++i) {
      auto param_type = source_func->params_[i]->GetType();
      auto rewrite_it =
          std::find_if(analysis.outputs.begin(), analysis.outputs.end(),
                       [i](const OutputRewriteInfo& info) { return info.out_param_index == i; });
      if (rewrite_it != analysis.outputs.end()) {
        auto out_tensor_type = As<TensorType>(source_func->params_[i]->GetType());
        if (!out_tensor_type) return nullptr;

        std::optional<TensorView> new_view = std::nullopt;
        if (out_tensor_type->tensor_view_.has_value()) {
          new_view = out_tensor_type->tensor_view_;
          if (new_view->stride.empty()) {
            if (new_view->layout == TensorLayout::NZ) return nullptr;
            new_view->stride = tensor_view_semantics::BuildLogicalStridesFromLayout(out_tensor_type->shape_,
                                                                                    new_view->layout);
          }
          if (!new_view->valid_shape.empty()) new_view->valid_shape = rewrite_it->window_shape;
        } else {
          auto parent_strides = ComputeRowMajorStrides(rewrite_it->parent_shape);
          if (parent_strides.empty() || parent_strides.size() != rewrite_it->window_shape.size()) {
            return nullptr;
          }
          new_view = TensorView(std::move(parent_strides), TensorLayout::ND);
        }

        param_type = std::make_shared<TensorType>(rewrite_it->window_shape, out_tensor_type->dtype_,
                                                  out_tensor_type->memref_, new_view);
        if (rewrite_it->return_index != SIZE_MAX) {
          if (rewrite_it->return_index >= new_return_types.size()) return nullptr;
          new_return_types[rewrite_it->return_index] = param_type;
        }
      }
      auto input_rewrite_it =
          std::find_if(analysis.inputs.begin(), analysis.inputs.end(),
                       [i](const InputRewriteInfo& info) { return info.in_param_index == i; });
      if (input_rewrite_it != analysis.inputs.end()) {
        auto in_tensor_type = As<TensorType>(source_func->params_[i]->GetType());
        if (!in_tensor_type) return nullptr;

        std::optional<TensorView> new_view = std::nullopt;
        if (in_tensor_type->tensor_view_.has_value()) {
          new_view = in_tensor_type->tensor_view_;
          if (new_view->stride.empty()) {
            if (new_view->layout == TensorLayout::NZ) return nullptr;
            new_view->stride = tensor_view_semantics::BuildLogicalStridesFromLayout(in_tensor_type->shape_,
                                                                                    new_view->layout);
          }
          if (!new_view->valid_shape.empty()) new_view->valid_shape = input_rewrite_it->window_shape;
        } else {
          auto parent_strides = ComputeRowMajorStrides(input_rewrite_it->parent_shape);
          if (parent_strides.empty() || parent_strides.size() != input_rewrite_it->window_shape.size()) {
            return nullptr;
          }
          new_view = TensorView(std::move(parent_strides), TensorLayout::ND);
        }

        param_type = std::make_shared<TensorType>(input_rewrite_it->window_shape, in_tensor_type->dtype_,
                                                  in_tensor_type->memref_, new_view);
      }

      auto new_param = std::make_shared<Var>(source_func->params_[i]->name_hint_, param_type,
                                             source_func->params_[i]->span_);
      new_params.push_back(new_param);
      seed[source_func->params_[i].get()] = new_param;
    }

    auto cloned_name = MakeUniqueFunctionName(program, func->name_ + "__windowed");
    auto cloned = DeepClone(source_func->body_, seed);
    std::unordered_map<const Var*, ExprPtr> body_subst = seed;
    for (const auto& [old_var, new_var] : cloned.var_map) {
      body_subst[old_var] = new_var;
    }

    std::vector<OutputRewriteInfo> localized_outputs = analysis.outputs;
    for (auto& output : localized_outputs) {
      for (auto& offset : output.callsite_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& offset : output.local_store_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
    }
    StmtPtr new_body = cloned.cloned_body;

    std::unordered_map<const Var*, InputRewriteInfo> in_info_by_var;
    for (auto input : analysis.inputs) {
      if (input.in_param_index >= source_func->params_.size()) return nullptr;
      size_t new_param_index = input.in_param_index + param_index_offset;
      if (new_param_index >= new_params.size()) return nullptr;
      for (auto& offset : input.callsite_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      for (auto& offset : input.local_slice_offsets) {
        offset = transform_utils::Substitute(offset, body_subst);
      }
      in_info_by_var.emplace(new_params[new_param_index].get(), std::move(input));
    }
    if (!in_info_by_var.empty()) {
      WindowReadLocalizer read_localizer(in_info_by_var);
      new_body = read_localizer.VisitStmt(new_body);
    }

    if (analysis.kind == RewriteKind::SpmdRowBlock) {
      if (!analysis.spmd_block_idx_var) return nullptr;
      auto cloned_block_idx = AsVarLike(transform_utils::Substitute(analysis.spmd_block_idx_var, body_subst));
      if (!cloned_block_idx) return nullptr;
      auto stripped_body = RemoveBlockIdxDefinition(new_body, cloned_block_idx.get());
      if (!stripped_body.has_value()) return nullptr;
      new_body = *stripped_body;
    }

    const bool uses_aggregate_loop =
        std::any_of(analysis.outputs.begin(), analysis.outputs.end(),
                    [](const OutputRewriteInfo& output) { return output.iter_arg_index != SIZE_MAX; });
    if (analysis.kind == RewriteKind::AggregateWindowLoop || uses_aggregate_loop) {
      auto cloned_loop = FindAggregateLoopInBody(new_body, analysis.outputs, new_params, param_index_offset);
      if (!cloned_loop) return nullptr;

      std::unordered_map<const Var*, TypePtr> narrowed_return_vars;
      for (const auto& output : analysis.outputs) {
        if (output.iter_arg_index >= cloned_loop->return_vars_.size()) return nullptr;
        TypePtr narrowed_type;
        if (output.return_index == SIZE_MAX) {
          auto param_index = output.out_param_index + param_index_offset;
          if (param_index >= new_params.size()) return nullptr;
          narrowed_type = new_params[param_index]->GetType();
        } else {
          if (output.return_index >= new_return_types.size()) return nullptr;
          narrowed_type = new_return_types[output.return_index];
        }
        narrowed_return_vars.emplace(cloned_loop->return_vars_[output.iter_arg_index].get(), narrowed_type);
      }

      new_body = LocalizeAggregateLoopTypes(new_body, narrowed_return_vars);

      auto typed_loop = FindAggregateLoopInBody(new_body, analysis.outputs, new_params, param_index_offset);
      if (!typed_loop) return nullptr;

      std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
      std::unordered_map<const Var*, TypePtr> new_store_types;
      std::unordered_map<const Var*, ExprPtr> new_out_vars;
      for (const auto& output : localized_outputs) {
        if (output.iter_arg_index >= typed_loop->iter_args_.size()) return nullptr;
        auto iter_arg = typed_loop->iter_args_[output.iter_arg_index];
        out_info_by_var.emplace(iter_arg.get(), output);
        new_out_vars.emplace(iter_arg.get(), iter_arg);
        TypePtr store_type = iter_arg->GetType();
        if (output.return_index != SIZE_MAX) {
          if (output.return_index >= new_return_types.size()) return nullptr;
          store_type = new_return_types[output.return_index];
        }
        new_store_types.emplace(iter_arg.get(), store_type);
      }

      WindowWriteLocalizer localizer(out_info_by_var, new_out_vars, new_store_types);
      new_body = localizer.VisitStmt(new_body);
    } else {
      std::unordered_map<const Var*, OutputRewriteInfo> out_info_by_var;
      std::unordered_map<const Var*, TypePtr> new_store_types;
      std::unordered_map<const Var*, ExprPtr> new_out_vars;
      for (const auto& output : localized_outputs) {
        auto new_out = new_params[output.out_param_index + param_index_offset];
        out_info_by_var.emplace(new_out.get(), output);
        new_store_types.emplace(new_out.get(), new_out->GetType());
        new_out_vars.emplace(new_out.get(), new_out);
      }
      WindowWriteLocalizer localizer(out_info_by_var, new_out_vars, new_store_types);
      new_body = localizer.VisitStmt(new_body);
    }

    return std::make_shared<Function>(cloned_name, new_params, new_param_directions, new_return_types,
                                      new_body, source_func->span_, source_func->func_type_,
                                      source_func->level_, source_func->role_, source_func->attrs_);
  }
};

}  // namespace

// ============================================================================
// Pass entry point
// ============================================================================

namespace pass {

Pass OptimizeOrchTensors() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
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
    return OutWindowExternalizer().Run(p4);
  };

  return CreateProgramPass(pass_func, "OptimizeOrchTensors", kOptimizeOrchTensorsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
