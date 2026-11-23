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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/transforms/utils/window_externalization.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;

namespace {

using window_externalization::BuildOutParamReturnMappings;
using window_externalization::GetCallFuncName;
using window_externalization::OutParamReturnMapping;

/// Find a function by name in a program.
FunctionPtr FindFunction(const ProgramPtr& program, const std::string& name) {
  for (const auto& [gvar, func] : program->functions_) {
    if (func->name_ == name) return func;
  }
  return nullptr;
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

/// Preserve assignment symmetry when input/output stride refinement rewrites
/// a value. Tuple and projection constructors infer their types from operands;
/// remapping the LHS makes subsequent uses follow those refined types too.
AssignStmtPtr RetypeStrideRefinedAssignment(const AssignStmtPtr& original, const AssignStmtPtr& rewritten,
                                            std::unordered_map<const Var*, VarPtr>& remap) {
  const auto& value_type = rewritten->value_->GetType();
  if (rewritten->value_.get() == original->value_.get() ||
      structural_equal(rewritten->var_->GetType(), value_type) ||
      !structural_equal(original->var_->GetType(), original->value_->GetType())) {
    return rewritten;
  }
  auto fresh = std::make_shared<Var>(original->var_->name_hint_, value_type, original->var_->span_);
  remap[original->var_.get()] = fresh;
  auto assign = MutableCopy(rewritten);
  assign->var_ = fresh;
  return assign;
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
    if (reuse_results.empty()) return program;

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

        if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && IsOp(call, "tensor.create")) {
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
  /// both `shapes` and `valid_shape` match the tensor shape dimension-by-
  /// dimension. `valid_shape` differs from `shapes` for masked/padded loads,
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
      if (!store_call || !IsOp(store_call, "tile.store") || store_call->args_.empty()) continue;

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
      if (!load_call || !IsOp(load_call, "tile.load") || load_call->args_.empty()) continue;
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
      if (auto call = As<Call>(op->value_);
          call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_) && IsOp(call, "tensor.create")) {
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
                                      func->span_, func->func_type_, func->level_, func->role_, func->attrs_,
                                      func->requires_runtime_binding_, func->ir_stage_);
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
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && IsOp(call, "tensor.create")) {
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
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && IsOp(call, "tensor.assemble") &&
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
      refined_vars_.insert(new_var.get());
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
      auto assign = RetypeStrideRefinedAssignment(op, As<AssignStmt>(IRMutator::VisitStmt_(op)), var_remap_);
      if (assign->var_.get() != op->var_.get()) refined_vars_.insert(assign->var_.get());

      auto call = As<Call>(assign->value_);
      if (!call || !IsOp(call, "tile.store") || call->args_.size() < 3) return assign;

      auto out_var = AsVarLike(call->args_[2]);
      if (!out_var || !refined_vars_.count(out_var.get())) return assign;

      auto out_type = out_var->GetType();
      auto new_call = std::make_shared<Call>(call->op_, call->args_, call->kwargs_, out_type, call->span_);
      auto new_var = std::make_shared<Var>(assign->var_->name_hint_, out_type, assign->var_->span_);
      var_remap_[op->var_.get()] = new_var;
      refined_vars_.insert(new_var.get());
      auto result = MutableCopy(assign);
      result->var_ = new_var;
      result->value_ = new_call;
      return result;
    }

   private:
    std::unordered_map<const Var*, VarPtr> subs_;
    std::unordered_set<const Var*> refined_vars_;
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

        // Materialising the parent's strides says nothing about byte order, so
        // keep whatever layout the parameter already declared. Hard-coding ND
        // silently retyped an NZ or MX parameter and left the call site's
        // argument disagreeing with it.
        const TensorLayout param_layout =
            tensor_type->tensor_view_ ? tensor_type->tensor_view_->layout : TensorLayout::ND;
        TensorView view(std::move(strides), param_layout);
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
                                        func->attrs_, func->requires_runtime_binding_, func->ir_stage_);
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

/// True when a `tile.assemble` / `tile.store` carries a FIXPIPE epilogue.
///
/// This rewrite turns an in-loop assemble into a store, which changes both the
/// destination *class* (an on-chip tile becomes a GM tensor, and the legal
/// scale-bearing dtype pairs are keyed on exactly that -- see
/// `BackendHandler::FixpipeDest`) and the destination *dtype* (the accumulator
/// tile's, versus the output tensor's, which is what
/// `codegen::EncodeFixpipePreQuant` reads for the signedness field). Carrying
/// the kwargs across unchanged would therefore re-target an epilogue validated
/// for one writeback onto a different one, and re-deriving legality here would
/// duplicate the backend tables inside an optimization.
///
/// So the pattern is declined instead. This costs a loop-fusion opportunity and
/// nothing else; dropping the kwargs silently costs a ReLU or a dequantization
/// scale, and `FixpipeEpilogueValid` cannot catch that afterwards because by
/// then there is no epilogue left to reject.
[[nodiscard]] inline bool HasFixpipeEpilogue(const CallPtr& call) {
  if (!call) return false;
  return GetOptionalDoubleKwarg(call->kwargs_, "pre_quant").has_value() ||
         call->GetKwarg<bool>("pre_relu", false);
}

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
        if (!call || !IsOp(call, "tile.assemble")) continue;
        // Declined rather than rewritten -- see HasFixpipeEpilogue.
        if (HasFixpipeEpilogue(call)) continue;
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
      if (!call || !IsOp(call, "tile.store")) continue;
      // The rewrite deletes this store; an epilogue on it would go with it.
      if (HasFixpipeEpilogue(call)) continue;
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
      if (!call || !IsOp(call, "tile.store")) continue;
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
          if (!call || !IsOp(call, "tile.assemble")) continue;
          // Must decline on exactly the same condition as the mutator above, or
          // the store below would be marked dead for a loop that is never rewritten.
          if (HasFixpipeEpilogue(call)) continue;
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
                                      func->level_, func->role_, func->attrs_,
                                      func->requires_runtime_binding_, func->ir_stage_);
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
  // func_name -> { param_index -> metadata }. Legacy InCore entries carry
  // parent shapes; explicit AIV entries carry resolved physical strides.
  struct InputMetadata {
    std::vector<ExprPtr> dimensions;
    TensorLayout layout = TensorLayout::ND;
  };
  using InputShapeMap = std::unordered_map<std::string, std::unordered_map<size_t, InputMetadata>>;

  static bool ShapesMatch(const std::vector<ExprPtr>& a, const std::vector<ExprPtr>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      auto ca = As<ConstInt>(a[i]);
      auto cb = As<ConstInt>(b[i]);
      if (!ca || !cb || ca->value_ != cb->value_) return false;
    }
    return true;
  }

  static TypePtr WithView(const TensorTypePtr& tensor, TensorView view) {
    if (auto distributed = As<DistributedTensorType>(tensor)) {
      return std::make_shared<DistributedTensorType>(tensor->shape_, tensor->dtype_, tensor->memref_,
                                                     std::move(view), distributed->window_buffer_);
    }
    return std::make_shared<TensorType>(tensor->shape_, tensor->dtype_, tensor->memref_, std::move(view));
  }

  static bool IsTensorMetadataView(const CallPtr& call) {
    return IsOp(call, "tensor.transpose") || IsOp(call, "tensor.view") || IsOp(call, "tensor.reshape") ||
           IsOp(call, "tensor.reinterpret_view") || IsOp(call, "tensor.set_validshape");
  }

  class SliceAnalyzer : public IRVisitor {
   public:
    SliceAnalyzer(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names)
        : program_(program), incore_names_(incore_names) {
      for (const auto& [gvar, func] : program->functions_) functions_.emplace(func->name_, func);
    }

    void AnalyzeFunctions() {
      // Index each body once. Analyze a helper on its first call so returned
      // views are available immediately; later callers constrain its bindings
      // without rescanning the body, just like loop backedges.
      class CalleeCollector : public IRVisitor {
       public:
        std::vector<std::string> names;
        void VisitExpr_(const CallPtr& call) override {
          if (As<GlobalVar>(call->op_)) names.push_back(call->op_->name_);
          IRVisitor::VisitExpr_(call);
        }
        void VisitExpr_(const SubmitPtr& call) override {
          if (As<GlobalVar>(call->op_)) names.push_back(call->op_->name_);
          IRVisitor::VisitExpr_(call);
        }
      };
      std::unordered_map<std::string, size_t> incoming;
      std::unordered_map<std::string, std::vector<std::string>> outgoing;
      for (const auto& [name, func] : functions_) {
        if (!incore_names_.count(name)) incoming.emplace(name, 0);
      }
      for (const auto& [name, count] : incoming) {
        CalleeCollector collector;
        collector.VisitStmt(functions_.at(name)->body_);
        for (const auto& callee : collector.names) {
          if (auto edge = incoming.find(callee); edge != incoming.end()) {
            ++edge->second;
            outgoing[name].push_back(callee);
          }
        }
      }
      std::vector<std::string> ready;
      for (const auto& [name, count] : incoming) {
        if (count == 0) ready.push_back(name);
      }
      size_t visited = 0;
      while (!ready.empty()) {
        auto name = std::move(ready.back());
        ready.pop_back();
        AnalyzeFunction(functions_.at(name));
        ++visited;
        for (const auto& callee : outgoing[name]) {
          if (--incoming.at(callee) == 0) ready.push_back(callee);
        }
      }
      CHECK_SPAN(visited == incoming.size(), program_->span_)
          << "Tensor stride propagation requires an acyclic orchestration call graph";
    }

    void AnalyzeFunction(const FunctionPtr& func) {
      if (analyzed_.count(func->name_)) return;
      CHECK_SPAN(analyzing_.insert(func->name_).second, func->span_)
          << "Tensor stride propagation requires an acyclic orchestration call graph";
      // Parameters exist before any branch, even without caller-derived storage.
      // Seed them so non-SSA parameter writes remain visible at branch joins.
      for (const auto& param : func->params_) BindStorage(param.get(), TypeStorage(param->GetType()));
      if (auto bindings = caller_bindings_.find(func->name_); bindings != caller_bindings_.end()) {
        for (const auto& [index, value] : bindings->second) BindStorage(func->params_[index].get(), value);
      }
      auto previous = current_function_;
      current_function_ = func->name_;
      VisitStmt(func->body_);
      current_function_ = std::move(previous);
      analyzing_.erase(func->name_);
      analyzed_.insert(func->name_);
    }

    InputShapeMap result() {
      // A later caller or backedge can introduce a slice or disprove packed ND
      // storage. Each fact propagates once per edge, including around cycles.
      std::vector<std::vector<size_t>> dependents(constraints_.size());
      std::vector<bool> pitch_conflict(constraints_.size());
      std::vector<size_t> pending;
      for (size_t i = 0; i < constraints_.size(); ++i) {
        for (const auto parent : constraints_[i].parents) dependents[parent].push_back(i);
        pitch_conflict[i] = !constraints_[i].consistent;
        if (constraints_[i].sliced || !constraints_[i].packed_nd || pitch_conflict[i]) pending.push_back(i);
      }
      while (!pending.empty()) {
        auto index = pending.back();
        pending.pop_back();
        for (const auto dependent : dependents[index]) {
          auto& child = constraints_[dependent];
          const bool sliced = child.sliced || constraints_[index].sliced;
          const bool conflict = pitch_conflict[dependent] || pitch_conflict[index];
          const bool packed_nd = child.packed_nd && constraints_[index].packed_nd &&
                                 !(child.uses_inherited_pitch && pitch_conflict[index]);
          const bool mixed_layouts = child.mixed_layouts || constraints_[index].mixed_layouts;
          if (sliced == child.sliced && packed_nd == child.packed_nd &&
              conflict == pitch_conflict[dependent] && mixed_layouts == child.mixed_layouts) {
            continue;
          }
          child.sliced = sliced;
          child.packed_nd = packed_nd;
          child.mixed_layouts = mixed_layouts;
          pitch_conflict[dependent] = conflict;
          pending.push_back(dependent);
        }
      }
      std::vector<size_t> descriptor_roots;
      std::vector<size_t> packed_roots;
      for (const auto& root : aiv_roots_) {
        const auto& binding = aiv_bindings_.at(root.function->name_).at(root.param);
        if (UsesRuntimeDescriptor(*root.function, root.param, binding) ||
            UsesPackedRuntimeStrides(*root.function, root.param, binding, pitch_conflict)) {
          descriptor_roots.push_back(root.constraint);
        } else if (binding.packed_nd && !binding.sliced && binding.constraint &&
                   constraints_[*binding.constraint].packed_nd && !constraints_[*binding.constraint].sliced &&
                   root.accepts_packed_shape) {
          packed_roots.push_back(root.constraint);
        } else {
          constraint_roots_.push_back(root.constraint);
        }
      }
      // Loop backedges are known only after visiting their bodies. Validate the
      // constraints reachable from AIV operands once, including operands used
      // inside the loop. Indexed edges permit cycles without a fixed-point walk.
      // Extern operands can keep different pitches after a join, but a
      // zero-copy reshape still needs packed input on every incoming path.
      ValidateConstraints(std::move(descriptor_roots), StrideRequirement::Runtime);
      ValidateConstraints(std::move(packed_roots), StrideRequirement::Packed);
      ValidateConstraints(std::move(constraint_roots_), StrideRequirement::Physical);

      for (const auto& [name, params] : aiv_bindings_) {
        for (const auto& [index, binding] : params) {
          if (!binding.sliced && (!binding.constraint || !constraints_[*binding.constraint].sliced)) continue;
          const auto& function = *functions_.at(name);
          if (UsesPackedRuntimeStrides(function, index, binding, pitch_conflict)) continue;
          if (UsesRuntimeDescriptor(function, index, binding)) {
            if (binding.consistent && !binding.strides.empty() &&
                (!binding.constraint || !pitch_conflict[*binding.constraint]) &&
                std::all_of(binding.strides.begin(), binding.strides.end(),
                            [](const ExprPtr& stride) { return As<ConstInt>(stride) != nullptr; })) {
              result_[name][index] = {binding.strides, binding.layout};
            }
            continue;
          }
          CHECK_SPAN(binding.consistent && !binding.strides.empty(), *binding.span)
              << "AIV sliced tensor argument requires one consistent static stride per parameter across "
                 "all call sites; pass the full parent tensor and use parent-relative indices instead";
          for (const auto& stride : binding.strides) {
            CHECK_SPAN(As<ConstInt>(stride), *binding.span)
                << "AIV sliced tensor argument requires static physical strides; pass the full parent "
                   "tensor and use parent-relative indices instead";
          }
          result_[name][index] = {binding.strides, binding.layout};
        }
      }
      return result_;
    }

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      // Snapshot operand metadata at each definition, including before SSA
      // conversion, when an assignment may reuse a variable identity.
      auto value = ResolveStorage(op->value_);
      IRVisitor::VisitStmt_(op);
      BindStorage(op->var_.get(), std::move(value));
      auto call = As<Call>(op->value_);
      if (!call) return;

      // Track tensor.slice(parent, size, offset) results
      if (!std::dynamic_pointer_cast<const GlobalVar>(call->op_) && IsOp(call, "tensor.slice") &&
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

      const auto found = functions_.find(fname);
      if (found == functions_.end()) return;
      const auto& incore_func = found->second;
      if (incore_func->func_type_ == FunctionType::AIV) return;  // handled for Call and Submit below

      for (size_t i = 0; i < call->args_.size() && i < incore_func->param_directions_.size(); ++i) {
        if (incore_func->param_directions_[i] != ParamDirection::In) continue;

        auto arg_var = AsVarLike(call->args_[i]);
        if (!arg_var) continue;

        auto it = var_to_parent_shape_.find(arg_var.get());
        if (it == var_to_parent_shape_.end()) continue;

        auto conflict_key = fname + ":" + std::to_string(i);
        if (conflicted_.count(conflict_key)) continue;

        auto& entry = result_[fname][i].dimensions;
        if (entry.empty()) {
          entry = it->second;
        } else if (!ShapesMatch(entry, it->second)) {
          conflicted_.insert(conflict_key);
          entry.clear();
        }
      }
    }

    void VisitExpr_(const CallPtr& op) override {
      AnalyzeCall(op->op_, op->args_, op->span_, false);
      IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const SubmitPtr& op) override {
      AnalyzeCall(op->op_, op->args_, op->span_, true);
      IRVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const ReturnStmtPtr& op) override {
      ValueGroups values;
      for (const auto& value : op->value_) values.push_back(ResolveStorage(value));
      const auto& signature = functions_.at(current_function_)->return_types_;
      if (values.size() == 1 && signature.size() > 1 && TupleSize(values.front()) == signature.size()) {
        // A forwarded tuple may spell several flat return positions. A single
        // declared TupleType remains one nested value, as required by Submit.
        auto tuple = values.front();
        values.clear();
        for (size_t i = 0; i < signature.size(); ++i) values.push_back(TupleElement(tuple, i));
      }
      auto [found, inserted] = returned_storage_.emplace(current_function_, values);
      if (!inserted) {
        for (size_t i = 0; i < values.size() && i < found->second.size(); ++i) {
          found->second[i] = MergeStorage(found->second[i], values[i], op->span_);
        }
      }
      IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const ForStmtPtr& op) override {
      auto initial = SeedLoop(op->iter_args_, op->span_);
      IRVisitor::VisitStmt_(op);
      FinishLoop(initial, op->body_, op->return_vars_);
    }

    void VisitStmt_(const WhileStmtPtr& op) override {
      auto initial = SeedLoop(op->iter_args_, op->span_);
      IRVisitor::VisitStmt_(op);
      FinishLoop(initial, op->body_, op->return_vars_);
    }

    void VisitStmt_(const IfStmtPtr& op) override {
      VisitExpr(op->condition_);
      auto then_result = AnalyzeBranch(op->then_body_);
      auto else_result = op->else_body_ ? AnalyzeBranch(*op->else_body_) : BranchResult{};
      // Both arms start from the incoming bindings. Join only written keys;
      // unrelated storage entries are neither copied nor scanned.
      for (const auto& [key, value] : then_result.bindings) {
        auto other = else_result.bindings.find(key);
        BindStorage(key, MergeStorage(
                             value, other == else_result.bindings.end() ? CurrentStorage(key) : other->second,
                             op->span_));
      }
      for (const auto& [key, value] : else_result.bindings) {
        if (!then_result.bindings.count(key)) {
          BindStorage(key, MergeStorage(CurrentStorage(key), value, op->span_));
        }
      }
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        auto value = i < then_result.yielded.size() ? then_result.yielded[i]
                                                    : TypeStorage(op->return_vars_[i]->GetType());
        const auto other = i < else_result.yielded.size() ? else_result.yielded[i] : Storage{};
        BindStorage(op->return_vars_[i].get(), MergeStorage(value, other, op->span_));
      }
    }

   private:
    struct LazyTuple;
    struct Storage {
      std::vector<ExprPtr> strides;
      TensorLayout layout = TensorLayout::ND;
      bool sliced = false;
      bool packed_nd = false;
      bool consistent = true;
      const Span* span = nullptr;
      std::optional<size_t> constraint;
      std::shared_ptr<const std::vector<Storage>> elements;
      LazyTuple* lazy = nullptr;
    };
    struct LazyTuple {
      Storage initial;
      size_t size;
      const Span* span;
      std::vector<Storage> alternatives;
      std::unordered_map<size_t, Storage> projected;
    };

    struct StrideConstraint {
      bool consistent = true;
      bool sliced = false;
      bool packed_nd = false;
      const Span* span;
      std::vector<size_t> parents;
      bool uses_inherited_pitch = false;
      bool requires_packed_nd = false;
      bool mixed_layouts = false;  // A pitch-only descriptor cannot reconcile different layouts.
      // Call sites agree on the graph formal's concrete extents and allow packed ND.
      // Only a consumer of the complete packed tensor may use that equivalence.
      bool packed_graph_formal = false;
    };
    struct AivRoot {
      const Function* function;
      size_t param;
      size_t constraint;
      bool accepts_packed_shape;
    };

    // An opaque kernel receives a runtime descriptor. Its row pitch can vary
    // when every reaching allocation/view has the packed strides of its own
    // runtime shape. An explicit strided formal still requires specialization.
    bool UsesPackedRuntimeStrides(const Function& function, size_t param, const Storage& value,
                                  const std::vector<bool>& pitch_conflict) const {
      if (!function.HasAttr(kAttrExternalSource) || !value.packed_nd ||
          (value.constraint && !constraints_[*value.constraint].packed_nd)) {
        return false;
      }
      const auto formal = AsTensorTypeLike(function.params_[param]->GetType());
      if (!formal || formal->shape_.size() != value.strides.size()) return false;
      const auto expected =
          tensor_view_semantics::BuildLogicalStridesFromLayout(formal->shape_, TensorLayout::ND);
      const auto& view = formal->tensor_view_;
      if (view && (view->layout != TensorLayout::ND ||
                   (!view->stride.empty() && !AreExprVectorsEqual(view->stride, expected)))) {
        return false;
      }
      bool dynamic = false;
      for (size_t i = 0; i < value.strides.size(); ++i) {
        if (As<ConstInt>(expected[i])) {
          if (!AreExprsEqual(expected[i], value.strides[i])) return false;
          // Only the innermost unit stride follows from packed ND alone. A
          // fixed outer stride must remain true for all incoming layouts.
          if (i + 1 < expected.size() && value.constraint && pitch_conflict[*value.constraint]) return false;
        }
        dynamic |= !As<ConstInt>(value.strides[i]);
      }
      return dynamic;
    }
    using ValueGroups = std::vector<Storage>;
    using Bindings = std::unordered_map<const Expr*, Storage>;
    using UndoBindings = std::unordered_map<const Expr*, std::optional<Storage>>;
    struct BranchResult {
      ValueGroups yielded;
      Bindings bindings;
    };

    // Symbolic extents can acquire different values on each call. Concrete
    // extents and explicit layout contracts still describe the callee's storage.
    static bool AcceptsPackedShape(const TypePtr& formal_type, const TypePtr& actual_type) {
      const auto formal = AsTensorTypeLike(formal_type);
      const auto actual = AsTensorTypeLike(actual_type);
      if (!formal || !actual || formal->shape_.size() != actual->shape_.size() ||
          (formal->tensor_view_ &&
           (formal->tensor_view_->layout != TensorLayout::ND || !formal->tensor_view_->stride.empty()))) {
        return false;
      }
      for (size_t axis = 0; axis < formal->shape_.size(); ++axis) {
        if (As<ConstInt>(formal->shape_[axis]) &&
            !AreExprsEqual(formal->shape_[axis], actual->shape_[axis])) {
          return false;
        }
      }
      return true;
    }

    bool UsesRuntimeDescriptor(const Function& func, size_t index, const Storage& binding) const {
      if (!func.HasAttr(kAttrExternalSource) ||
          (binding.constraint && constraints_[*binding.constraint].mixed_layouts)) {
        return false;
      }
      auto tensor = AsTensorTypeLike(func.params_[index]->GetType());
      // With dynamic inner extents, packed runtime pitch also depends on the
      // operand's runtime shape. Those operands need the packed-storage proof.
      return tensor && !tensor->tensor_view_.has_value() && !tensor->shape_.empty() &&
             std::all_of(tensor->shape_.begin() + 1, tensor->shape_.end(),
                         [](const ExprPtr& extent) { return As<ConstInt>(extent) != nullptr; });
    }

    enum class StrideRequirement { Runtime, Packed, Physical };

    void ValidateConstraints(std::vector<size_t> pending, StrideRequirement requirement) {
      if (pending.empty()) return;
      std::vector<bool> visited(constraints_.size(), false);
      while (!pending.empty()) {
        const auto index = pending.back();
        pending.pop_back();
        if (visited[index]) continue;
        visited[index] = true;
        const auto& constraint = constraints_[index];
        if (requirement != StrideRequirement::Physical && constraint.requires_packed_nd) {
          constraint_roots_.push_back(index);
          continue;
        }
        if (requirement != StrideRequirement::Runtime) {
          // A packed graph formal describes each caller in its own shape
          // variables. A sliced consumer still needs the actual physical pitch.
          const bool packed_formal = requirement == StrideRequirement::Packed &&
                                     constraint.packed_graph_formal && constraint.packed_nd &&
                                     !constraint.sliced;
          CHECK_SPAN(constraint.consistent || packed_formal, *constraint.span)
              << "AIV tensor argument requires consistent physical strides across loop carries and branches, "
                 "and graph calls; tensor.reshape requires packed ND storage, as does shaped ND tensor.view; "
                 "pass the full parent tensor and use parent-relative indices instead";
        }
        pending.insert(pending.end(), constraint.parents.begin(), constraint.parents.end());
      }
    }

    Storage CurrentStorage(const Expr* expr) {
      if (auto found = storage_.find(expr); found != storage_.end()) {
        return found->second;
      }
      return TypeStorage(expr->GetType());
    }

    void BindStorage(const Expr* expr, Storage value) {
      if (!branch_undo_.empty() && !branch_undo_.back().count(expr)) {
        auto previous = storage_.find(expr);
        branch_undo_.back().emplace(
            expr, previous == storage_.end() ? std::nullopt : std::optional<Storage>(previous->second));
      }
      storage_[expr] = std::move(value);
    }

    BranchResult AnalyzeBranch(const StmtPtr& body) {
      branch_undo_.emplace_back();
      VisitStmt(body);
      BranchResult result{YieldValues(body), {}};
      auto undo = std::move(branch_undo_.back());
      branch_undo_.pop_back();
      for (const auto& [key, previous] : undo) {
        if (previous) {
          result.bindings.emplace(key, storage_.at(key));
          storage_[key] = *previous;
        } else {
          // Local definitions escape only through explicit yields. Exporting
          // them here would re-merge each one at every enclosing branch.
          storage_.erase(key);
        }
      }
      return result;
    }

    size_t AddConstraint(Storage& info, const Span& span) {
      StrideConstraint constraint{info.consistent, info.sliced, info.packed_nd, &span, {}};
      if (info.constraint) constraint.parents.push_back(*info.constraint);
      const auto index = constraints_.size();
      info.constraint = index;
      constraints_.push_back(std::move(constraint));
      return index;
    }

    // The proof for a derived view compares the strides inherited from the
    // first incoming binding. Later callers/backedges may supply packed parents
    // with a different pitch. A whole parent remains packed, but a window
    // inferred from that first pitch is no longer known to be packed.
    void RecordInheritedPitch(Storage& info, const Span& span) {
      if (info.packed_nd && info.constraint) {
        constraints_[AddConstraint(info, span)].uses_inherited_pitch = true;
      }
    }

    void Constrain(size_t index, const Storage& initial, const Storage& yielded) {
      auto& constraint = constraints_[index];
      constraint.consistent &= yielded.consistent && initial.layout == yielded.layout &&
                               AreExprVectorsEqual(initial.strides, yielded.strides);
      constraint.mixed_layouts |= initial.layout != yielded.layout;
      constraint.sliced |= initial.sliced || yielded.sliced;
      constraint.packed_nd &= initial.packed_nd && yielded.packed_nd;
      if (yielded.constraint) constraint.parents.push_back(*yielded.constraint);
    }

    static size_t TupleSize(const Storage& value) {
      return value.lazy ? value.lazy->size : (value.elements ? value.elements->size() : 0);
    }

    Storage TupleElement(const Storage& value, size_t index) {
      if (index >= TupleSize(value)) return {};
      if (!value.lazy) return (*value.elements)[index];
      auto& node = *value.lazy;
      if (auto found = node.projected.find(index); found != node.projected.end()) {
        return found->second;
      }
      // Publish the seeded element before following backedges, which can refer
      // to this tuple. Only projected paths acquire constraints; unused wide
      // carries cost O(1) per loop instead of copying every tensor leaf.
      auto seeded = SeedStorage(TupleElement(node.initial, index), *node.span);
      node.projected.emplace(index, seeded);
      for (const auto& other : node.alternatives) {
        seeded = MergeStorage(seeded, TupleElement(other, index), *node.span, true);
      }
      node.projected[index] = seeded;
      return seeded;
    }

    Storage SeedStorage(Storage value, const Span& span) {
      if (value.elements || value.lazy) {
        auto node = std::make_unique<LazyTuple>();
        node->initial = value;
        node->size = TupleSize(value);
        node->span = &span;
        value.elements.reset();
        value.lazy = node.get();
        lazy_tuples_.push_back(std::move(node));
      } else if (!value.strides.empty()) {
        AddConstraint(value, span);
      }
      return value;
    }

    Storage MergeStorage(Storage value, const Storage& other, const Span& span, bool backedge = false) {
      if (value.elements || value.lazy) {
        if ((value.lazy && value.lazy == other.lazy) ||
            (value.elements && value.elements == other.elements)) {
          return value;
        }
        if (!backedge || !value.lazy) value = SeedStorage(value, span);
        auto& node = *value.lazy;
        node.alternatives.push_back(other);
        // Snapshot projected keys: following an alternative may lazily insert
        // another element into this same node during a cyclic backedge.
        std::vector<size_t> keys;
        keys.reserve(node.projected.size());
        for (const auto& [index, element] : node.projected) keys.push_back(index);
        for (auto index : keys) {
          auto element = node.projected.at(index);
          auto alternate = TupleElement(other, index);
          node.projected[index] = MergeStorage(element, alternate, span, true);
        }
      } else if (!value.strides.empty()) {
        if (!backedge) AddConstraint(value, span);
        if (value.constraint) Constrain(*value.constraint, value, other);
        value.sliced |= other.sliced;
        value.packed_nd &= other.packed_nd;
      }
      return value;
    }

    ValueGroups YieldValues(const StmtPtr& body) {
      ValueGroups values;
      if (auto yield = transform_utils::GetLastYieldStmt(body)) {
        for (const auto& value : yield->value_) values.push_back(ResolveStorage(value));
      }
      return values;
    }

    ValueGroups SeedLoop(const std::vector<IterArgPtr>& iter_args, const Span& span) {
      ValueGroups initial;
      for (const auto& iter : iter_args) {
        initial.push_back(SeedStorage(ResolveStorage(iter->initValue_), span));
      }
      // Resolve every initializer before binding any loop-local definitions.
      for (size_t i = 0; i < iter_args.size(); ++i) BindStorage(iter_args[i].get(), initial[i]);
      return initial;
    }

    void FinishLoop(ValueGroups& initial, const StmtPtr& body, const std::vector<VarPtr>& return_vars) {
      auto yielded = YieldValues(body);
      for (size_t i = 0; i < initial.size(); ++i) {
        const auto other = i < yielded.size() ? yielded[i] : Storage{};
        initial[i] = MergeStorage(initial[i], other, body->span_, true);
        if (i < return_vars.size()) BindStorage(return_vars[i].get(), initial[i]);
      }
    }

    Storage TypeStorage(const TypePtr& type) {
      if (auto found = type_storage_.find(type.get()); found != type_storage_.end()) {
        return found->second;
      }
      Storage info;
      if (auto tuple = As<TupleType>(type)) {
        auto elements = std::make_shared<std::vector<Storage>>();
        for (const auto& element : tuple->types_) elements->push_back(TypeStorage(element));
        info.elements = std::move(elements);
        type_storage_.emplace(type.get(), info);
        return info;
      }
      auto tensor = AsTensorTypeLike(type);
      if (!tensor) return info;
      const auto layout = tensor->tensor_view_ ? tensor->tensor_view_->layout : TensorLayout::ND;
      info.layout = layout;
      if (layout == TensorLayout::ND || layout == TensorLayout::DN) {
        info.strides = tensor->tensor_view_ && !tensor->tensor_view_->stride.empty()
                           ? tensor->tensor_view_->stride
                           : tensor_view_semantics::BuildLogicalStridesFromLayout(tensor->shape_, layout);
        info.packed_nd =
            layout == TensorLayout::ND &&
            (!tensor->tensor_view_ || tensor->tensor_view_->stride.empty() ||
             AreExprVectorsEqual(info.strides, tensor_view_semantics::BuildLogicalStridesFromLayout(
                                                   tensor->shape_, TensorLayout::ND)));
      }
      return info;
    }

    std::vector<Storage> ResolveCallResults(const ExprPtr& expr) {
      auto call = transform_utils::AsCallOrSubmitView(expr);
      if (!call || !As<GlobalVar>(call->op_)) return {};
      auto found = functions_.find(call->op_->name_);
      if (found == functions_.end()) return {};
      const auto& callee = found->second;
      if (!incore_names_.count(callee->name_)) {
        AnalyzeCall(call->op_, call->args_, expr->span_, As<Submit>(expr) != nullptr);
      }
      auto returned = returned_storage_.find(callee->name_);
      const auto return_values = returned == returned_storage_.end() ? ValueGroups{} : returned->second;
      auto [mapping, inserted] = return_params_.try_emplace(callee.get());
      if (inserted) mapping->second = return_lineage::ReturnedParamIndices(callee, program_);
      const auto param_indices = mapping->second;
      // CallerSuppliedArg's prefix contract, computed once for all results.
      // Submit may leave runtime-allocated Out slots before the CommCtx suffix.
      const auto ctx = transform_utils::TrailingCommCtxParamCount(callee);
      const auto supplied = call->args_.size() >= ctx && call->args_.size() <= callee->params_.size()
                                ? call->args_.size() - ctx
                                : 0;
      std::vector<Storage> results;
      for (size_t i = 0; i < callee->return_types_.size(); ++i) {
        auto value = i < return_values.size() ? return_values[i] : TypeStorage(callee->return_types_[i]);
        if (i < param_indices.size()) {
          const auto& param = param_indices[i];
          if (param) {
            if (auto storage =
                    ArgumentStorage(callee, call->args_, supplied, *param, As<Submit>(expr) != nullptr)) {
              value = *storage;
            }
          }
        }
        results.push_back(std::move(value));
      }
      if (As<Submit>(expr)) results.emplace_back();  // TASK_ID has no tensor storage.
      return results;
    }

    Storage ResolveStorage(const ExprPtr& expr) {
      if (auto found = storage_.find(expr.get()); found != storage_.end()) {
        return found->second;
      }
      if (auto item = As<TupleGetItemExpr>(expr)) {
        auto tuple = ResolveStorage(item->tuple_);
        if (item->index_ >= 0 && static_cast<size_t>(item->index_) < TupleSize(tuple)) {
          return TupleElement(tuple, static_cast<size_t>(item->index_));
        }
      }
      if (auto tuple = As<MakeTuple>(expr)) {
        Storage info;
        auto elements = std::make_shared<std::vector<Storage>>();
        for (const auto& element : tuple->elements_) elements->push_back(ResolveStorage(element));
        info.elements = std::move(elements);
        return info;
      }
      if (auto iter = As<IterArg>(expr)) return ResolveStorage(iter->initValue_);
      if (auto results = ResolveCallResults(expr); !results.empty()) {
        const auto call = As<Call>(expr);
        const auto found = call ? functions_.find(call->op_->name_) : functions_.end();
        if (found != functions_.end() && found->second->return_types_.size() == 1) return results.front();
        Storage info;
        info.elements = std::make_shared<const std::vector<Storage>>(std::move(results));
        return info;
      }
      auto info = TypeStorage(expr->GetType());
      if (auto call = As<Call>(expr); call && IsOp(call, "tensor.slice")) {
        info = ResolveStorage(call->args_[0]);
        info.sliced = true;
        const auto shape_type = As<TupleType>(call->args_[1]->GetType());
        INTERNAL_CHECK_SPAN(shape_type, call->span_) << "tensor.slice shape must have tuple type";
        const auto shape = ExtractTupleElements(call->args_[1], shape_type->types_.size());
        const auto dropped =
            ParseSliceDropDims(call->args_.size() > 4 ? call->args_[4] : nullptr, shape, "tensor.slice");
        const auto source = AsTensorTypeLike(call->args_[0]->GetType());
        INTERNAL_CHECK_SPAN(source, call->span_) << "tensor.slice source must have tensor type";
        const auto source_rank = source->shape_.size();
        // Match DeduceTensorSliceType: DN describes the trailing plane and
        // survives only a same-rank window that drops leading axes alone.
        bool trailing_plane_intact = shape.size() == source_rank && shape.size() - dropped.size() >= 2;
        for (auto axis : dropped) {
          if (axis < 0 || static_cast<size_t>(axis) + 2 >= source_rank) trailing_plane_intact = false;
        }
        if (!trailing_plane_intact) info.layout = TensorLayout::ND;
        // Rank-reinterpreting windows use the parent's trailing axes, matching
        // the existing InCore contract. Explicit drop_dims removes those axes.
        if (shape.size() <= info.strides.size()) {
          info.strides.erase(info.strides.begin(),
                             info.strides.end() - static_cast<std::ptrdiff_t>(shape.size()));
          info.strides = ApplyDropDims(info.strides, dropped);
        } else {
          info.strides.clear();
        }
        // Codegen clamps shapes to the source at runtime. A shifted or ragged
        // inner axis could therefore shrink while still inheriting the original
        // pitch. Multi-row slabs must retain every inner extent at offset zero.
        const auto result = AsTensorTypeLike(expr->GetType());
        const auto offsets = ExtractTupleElements(call->args_[2], shape.size());
        bool seen_leading = false;
        bool full_inner_axes = shape.size() == source_rank && offsets.size() == source_rank;
        for (size_t axis = 0; full_inner_axes && axis < source_rank; ++axis) {
          if (!seen_leading) {
            seen_leading =
                std::find(dropped.begin(), dropped.end(), static_cast<int64_t>(axis)) == dropped.end();
            continue;
          }
          const auto offset = As<ConstInt>(offsets[axis]);
          full_inner_axes = offset && offset->value_ == 0 && AreExprsEqual(shape[axis], source->shape_[axis]);
        }
        // A one-row window has no transition between rows. Cropping its
        // contiguous inner axis therefore preserves packed addressing even
        // when the parent row pitch differs from the window's dynamic width.
        const bool single_row = result && result->shape_.size() == 2 && info.strides.size() == 2 &&
                                IsConstValue(result->shape_[0], 1) && IsConstValue(info.strides[1], 1);
        const bool full_slab =
            full_inner_axes && result &&
            AreExprVectorsEqual(info.strides, tensor_view_semantics::BuildLogicalStridesFromLayout(
                                                  result->shape_, TensorLayout::ND));
        info.packed_nd &= info.layout == TensorLayout::ND && (single_row || full_slab);
        // Every packed ND parent has unit pitch on its innermost axis, even
        // when another caller supplies a different outer pitch.
        const auto inner_axis = static_cast<int64_t>(shape.size()) - 1;
        if (!single_row || std::find(dropped.begin(), dropped.end(), inner_axis) != dropped.end()) {
          RecordInheritedPitch(info, call->span_);
        }
      } else {
        if (auto call = As<Call>(expr)) {
          if (IsTensorMetadataView(call)) {
            auto source = ResolveStorage(call->args_[0]);
            auto tensor = AsTensorTypeLike(call->args_[0]->GetType());
            if (tensor && !source.strides.empty()) {
              auto view = tensor->tensor_view_.value_or(TensorView(std::vector<ExprPtr>{}, source.layout));
              view.layout = source.layout;
              view.stride = source.strides;
              // Shape-bearing ND views also lower to Tensor::reshape. Record
              // their constraints on the result for reachable AIV operands.
              const bool requires_packed_nd =
                  IsOp(call, "tensor.reshape") ||
                  (IsOp(call, "tensor.view") && call->args_.size() >= 2 && info.layout == TensorLayout::ND &&
                   (!tensor->tensor_view_ || tensor->tensor_view_->layout == TensorLayout::ND));
              if (requires_packed_nd &&
                  (source.layout != TensorLayout::ND ||
                   !AreExprVectorsEqual(source.strides, tensor_view_semantics::BuildLogicalStridesFromLayout(
                                                            tensor->shape_, TensorLayout::ND)))) {
                info.sliced = source.sliced;
                info.consistent = false;
                info.packed_nd = false;
                info.constraint = source.constraint;
                constraints_[AddConstraint(info, call->span_)].requires_packed_nd = true;
                return info;
              }
              auto args = call->args_;
              args[0] = std::make_shared<Var>("view_source", WithView(tensor, std::move(view)),
                                              call->args_[0]->span_);
              auto inferred =
                  OpRegistry::GetInstance().Create(call->op_->name_, args, call->kwargs_, call->span_);
              info = TypeStorage(inferred->GetType());
              info.sliced = source.sliced;
              info.packed_nd &= source.packed_nd;
              info.consistent = source.consistent;
              info.constraint = source.constraint;
              if (requires_packed_nd && info.constraint) {
                auto& constraint = constraints_[AddConstraint(info, call->span_)];
                constraint.requires_packed_nd = true;
                constraint.uses_inherited_pitch = info.packed_nd;
              } else {
                RecordInheritedPitch(info, call->span_);
              }
              return info;
            }
          }
          if (auto index = ResultAliasedArgIndex(call)) {
            return ResolveStorage(call->args_[*index]);
          }
        }
      }
      return info;
    }

    std::optional<Storage> ArgumentStorage(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                           size_t supplied, size_t index, bool submit) {
      if (index < supplied) return ResolveStorage(args[index]);
      if (!submit || callee->param_directions_[index] != ParamDirection::Out) return std::nullopt;
      auto tensor = AsTensorTypeLike(callee->params_[index]->GetType());
      if (!tensor) return TypeStorage(callee->params_[index]->GetType());
      // Synthesized outputs allocate shape/dtype only, hence packed row-major
      // storage regardless of an explicit view on the formal parameter.
      Storage value;
      value.packed_nd = true;
      value.strides = tensor_view_semantics::BuildLogicalStridesFromLayout(tensor->shape_, TensorLayout::ND);
      value.sliced = tensor->tensor_view_ && (!tensor->tensor_view_->stride.empty() ||
                                              tensor->tensor_view_->layout != TensorLayout::ND);
      return value;
    }

    void AnalyzeCall(const OpPtr& op, const std::vector<ExprPtr>& args, const Span& span, bool submit) {
      if (!As<GlobalVar>(op)) return;
      auto found = functions_.find(op->name_);
      if (found == functions_.end()) return;
      const auto& callee = found->second;
      const auto& params = callee->params_;
      const auto ctx = transform_utils::TrailingCommCtxParamCount(callee);
      const auto supplied = args.size() >= ctx && args.size() <= params.size() ? args.size() - ctx : 0;
      if (!incore_names_.count(callee->name_)) {
        auto& bindings = caller_bindings_[callee->name_];
        for (size_t i = 0; i < params.size() - ctx; ++i) {
          auto value = ArgumentStorage(callee, args, supplied, i, submit);
          if (!value) continue;
          auto [binding, inserted] = bindings.emplace(i, *value);
          if (inserted) {
            binding->second = SeedStorage(*value, span);
          } else {
            binding->second = MergeStorage(binding->second, *value, span, true);
          }
          if (callee->func_type_ == FunctionType::Graph && binding->second.constraint) {
            const bool matches = i < supplied && AcceptsPackedShape(params[i]->GetType(), args[i]->GetType());
            auto& constraint = constraints_[*binding->second.constraint];
            constraint.packed_graph_formal = matches && (inserted || constraint.packed_graph_formal);
          }
        }
        AnalyzeFunction(callee);
        return;
      }
      if (callee->func_type_ != FunctionType::AIV) return;
      // Include omitted Out slots before the CommCtx suffix in consensus.
      for (size_t i = 0; i < params.size() - ctx; ++i) {
        auto tensor = AsTensorTypeLike(i < supplied ? args[i]->GetType() : params[i]->GetType());
        if (!tensor) continue;
        // Blocked layouts have their own legalization; their logical strides
        // are not affine physical strides at this point in the pipeline.
        const auto layout = tensor->tensor_view_ ? tensor->tensor_view_->layout : TensorLayout::ND;
        if (layout != TensorLayout::ND && layout != TensorLayout::DN) continue;
        auto value = ArgumentStorage(callee, args, supplied, i, submit);
        if (!value) continue;
        auto info = *value;
        if (info.constraint) {
          aiv_roots_.push_back(
              {callee.get(), i, *info.constraint, AcceptsPackedShape(params[i]->GetType(), tensor)});
        }
        info.span = &span;
        auto [binding, inserted] = aiv_bindings_[op->name_].emplace(i, info);
        if (!inserted) {
          binding->second.consistent &=
              binding->second.layout == info.layout && ShapesMatch(binding->second.strides, info.strides);
          binding->second = MergeStorage(binding->second, info, span);
        }
      }
    }

    const ProgramPtr& program_;
    const std::unordered_set<std::string>& incore_names_;
    std::unordered_map<const Var*, std::vector<ExprPtr>> var_to_parent_shape_;
    std::unordered_set<std::string> conflicted_;
    InputShapeMap result_;
    std::unordered_map<std::string, FunctionPtr> functions_;
    Bindings storage_;
    std::vector<UndoBindings> branch_undo_;
    std::unordered_map<const Type*, Storage> type_storage_;
    std::unordered_map<std::string, std::unordered_map<size_t, Storage>> caller_bindings_;
    std::unordered_map<std::string, ValueGroups> returned_storage_;
    std::unordered_set<std::string> analyzing_;
    std::unordered_set<std::string> analyzed_;
    std::string current_function_;
    std::unordered_map<std::string, std::unordered_map<size_t, Storage>> aiv_bindings_;
    std::unordered_map<const Function*, std::vector<std::optional<size_t>>> return_params_;
    std::vector<std::unique_ptr<LazyTuple>> lazy_tuples_;
    std::vector<StrideConstraint> constraints_;
    std::vector<AivRoot> aiv_roots_;
    std::vector<size_t> constraint_roots_;
  };

  InputShapeMap Analyze(const ProgramPtr& program, const std::unordered_set<std::string>& incore_names) {
    return_lineage::ScopedTraceCache trace_cache;
    SliceAnalyzer analyzer(program, incore_names);
    analyzer.AnalyzeFunctions();
    return analyzer.result();
  }

  class InParamSubstitutionMutator : public IRMutator {
   public:
    explicit InParamSubstitutionMutator(bool refine_metadata) : refine_metadata_(refine_metadata) {}

    void AddSubstitution(const Var* old_ptr, const VarPtr& new_var) { subs_[old_ptr] = new_var; }

   protected:
    ExprPtr VisitExpr_(const IterArgPtr& op) override {
      if (auto it = subs_.find(op.get()); it != subs_.end()) {
        return it->second;
      }
      auto rewritten = As<IterArg>(IRMutator::VisitExpr_(op));
      if (structural_equal(op->GetType(), op->initValue_->GetType()) &&
          !structural_equal(rewritten->GetType(), rewritten->initValue_->GetType())) {
        rewritten = std::make_shared<IterArg>(op->name_hint_, rewritten->initValue_->GetType(),
                                              rewritten->initValue_, op->span_);
        subs_[op.get()] = rewritten;
      }
      return rewritten;
    }

    StmtPtr VisitStmt_(const ForStmtPtr& op) override {
      return RefineLoopResults(op, As<ForStmt>(IRMutator::VisitStmt_(op)));
    }

    StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
      return RefineLoopResults(op, As<WhileStmt>(IRMutator::VisitStmt_(op)));
    }

    StmtPtr VisitStmt_(const IfStmtPtr& op) override {
      auto rewritten = As<IfStmt>(IRMutator::VisitStmt_(op));
      if (!op->else_body_ || !rewritten->else_body_ || op->return_vars_.empty()) return rewritten;
      auto old_then = transform_utils::GetLastYieldStmt(op->then_body_);
      auto old_else = transform_utils::GetLastYieldStmt(*op->else_body_);
      auto new_then = transform_utils::GetLastYieldStmt(rewritten->then_body_);
      auto new_else = transform_utils::GetLastYieldStmt(*rewritten->else_body_);
      if (!old_then || !old_else || !new_then || !new_else) return rewritten;
      auto result = MutableCopy(rewritten);
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        if (i >= old_then->value_.size() || i >= old_else->value_.size() || i >= new_then->value_.size() ||
            i >= new_else->value_.size()) {
          break;
        }
        const auto& old_type = op->return_vars_[i]->GetType();
        if (!structural_equal(old_type, old_then->value_[i]->GetType()) ||
            !structural_equal(old_type, old_else->value_[i]->GetType())) {
          continue;
        }
        const auto& new_type = new_then->value_[i]->GetType();
        CHECK_SPAN(StorageTypesMatch(new_type, new_else->value_[i]->GetType()), op->span_)
            << "AIV branch yields disagree after physical stride refinement";
        result->return_vars_[i] = RefineResult(op->return_vars_[i], result->return_vars_[i], new_type);
      }
      return result;
    }

    ExprPtr VisitExpr_(const CallPtr& op) override {
      auto call = As<Call>(IRMutator::VisitExpr_(op));
      if (call.get() != op.get() &&
          (ResultAliasedArgIndex(call) ||
           (refine_metadata_ && (IsTensorMetadataView(call) || IsOp(call, "tensor.slice"))))) {
        // Metadata views change shape/layout without being whole-value aliases.
        // Both kinds must infer from the refined operands, preserving attributes.
        if (refine_metadata_ && IsOp(call, "tensor.reshape")) {
          auto tensor = AsTensorTypeLike(call->args_[0]->GetType());
          INTERNAL_CHECK_SPAN(tensor, call->span_) << "tensor.reshape source must have tensor type";
          const auto& view = tensor->tensor_view_;
          CHECK_SPAN(
              !view ||
                  (view->layout == TensorLayout::ND &&
                   (view->stride.empty() ||
                    AreExprVectorsEqual(view->stride, tensor_view_semantics::BuildLogicalStridesFromLayout(
                                                          tensor->shape_, TensorLayout::ND)))),
              call->span_)
              << "tensor.reshape requires packed ND storage after AIV parameter refinement";
        }
        auto inferred =
            OpRegistry::GetInstance().Create(call->op_->name_, call->args_, call->kwargs_, call->span_);
        return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_,
                                      inferred->GetType(), call->span_);
      }
      return call;
    }

    ExprPtr VisitExpr_(const VarPtr& op) override {
      auto it = subs_.find(op.get());
      if (it != subs_.end()) return it->second;
      // The base mutator also remaps SSA results defined by a SIMT launch
      // when a rewritten storage argument changes their view type.
      return IRMutator::VisitExpr_(op);
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
      return RetypeStrideRefinedAssignment(op, As<AssignStmt>(IRMutator::VisitStmt_(op)), subs_);
    }

   private:
    TypePtr EffectiveStorageType(const TypePtr& type) {
      if (auto found = effective_types_.find(type.get()); found != effective_types_.end()) {
        return found->second;
      }
      TypePtr result = type;
      if (auto tensor = AsTensorTypeLike(type)) {
        auto view = tensor->tensor_view_.value_or(TensorView(std::vector<ExprPtr>{}, TensorLayout::ND));
        if (view.stride.empty() && (view.layout == TensorLayout::ND || view.layout == TensorLayout::DN)) {
          view.stride = tensor_view_semantics::BuildLogicalStridesFromLayout(tensor->shape_, view.layout);
          result = WithView(tensor, std::move(view));
        }
      } else if (auto tuple = As<TupleType>(type)) {
        std::vector<TypePtr> elements;
        for (const auto& element : tuple->types_) elements.push_back(EffectiveStorageType(element));
        result = std::make_shared<TupleType>(std::move(elements));
      }
      effective_types_.emplace(type.get(), result);
      return result;
    }

    bool StorageTypesMatch(const TypePtr& a, const TypePtr& b) {
      return structural_equal(EffectiveStorageType(a), EffectiveStorageType(b));
    }

    VarPtr RefineResult(const VarPtr& original, const VarPtr& rewritten, const TypePtr& type) {
      if (structural_equal(rewritten->GetType(), type)) return rewritten;
      auto result = std::make_shared<Var>(original->name_hint_, type, original->span_);
      subs_[original.get()] = result;
      return result;
    }

    template <typename Loop>
    StmtPtr RefineLoopResults(const std::shared_ptr<const Loop>& original,
                              const std::shared_ptr<const Loop>& rewritten) {
      // The base visitor rewrites the body once using the refined IterArgs.
      // Only inspect its trailing yield; never rewalk a nested loop body.
      auto yield = transform_utils::GetLastYieldStmt(rewritten->body_);
      auto result = MutableCopy(rewritten);
      for (size_t i = 0; i < original->iter_args_.size(); ++i) {
        const auto& old_type = original->iter_args_[i]->GetType();
        const auto& new_type = rewritten->iter_args_[i]->GetType();
        if (structural_equal(old_type, new_type)) continue;
        if (yield && i < yield->value_.size()) {
          CHECK_SPAN(StorageTypesMatch(new_type, yield->value_[i]->GetType()), yield->span_)
              << "AIV loop yield disagrees with its initializer after physical stride refinement";
        }
        if (i < original->return_vars_.size() &&
            structural_equal(original->return_vars_[i]->GetType(), old_type)) {
          result->return_vars_[i] =
              RefineResult(original->return_vars_[i], result->return_vars_[i], new_type);
        }
      }
      return result;
    }

    bool refine_metadata_;
    std::unordered_map<const Type*, TypePtr> effective_types_;
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

      for (const auto& [param_idx, metadata] : param_idx_to_shape) {
        const auto& parent_shape = metadata.dimensions;
        if (parent_shape.empty()) continue;  // conflicted or not from slice
        if (param_idx >= func->params_.size()) continue;

        auto full_strides =
            func->func_type_ == FunctionType::AIV ? parent_shape : ComputeRowMajorStrides(parent_shape);
        if (full_strides.empty()) continue;

        auto tensor_type = AsTensorTypeLike(func->params_[param_idx]->GetType());
        if (!tensor_type) continue;

        // An explicit AIV declaration must agree with the caller's storage.
        if (func->func_type_ == FunctionType::AIV && tensor_type->tensor_view_) {
          CHECK_SPAN(tensor_type->tensor_view_->layout == metadata.layout, func->params_[param_idx]->span_)
              << "AIV tensor argument disagrees with the parameter's explicit layout; "
                 "declare the actual storage layout or pass a matching tensor";
        }
        if (tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty()) {
          CHECK_SPAN(func->func_type_ != FunctionType::AIV ||
                         (ShapesMatch(tensor_type->tensor_view_->stride, full_strides) &&
                          tensor_type->tensor_view_->layout == metadata.layout),
                     func->params_[param_idx]->span_)
              << "AIV sliced tensor argument disagrees with the parameter's explicit physical strides";
          continue;
        }

        // NZ bytes are fractal-blocked, so the parent's logical row-major strides
        // do not describe them. BlockNzTensorViews derives the real strides from
        // the blocked shape and refuses an explicit one; stamping them here made
        // it blame the user for a stride they never wrote.
        const auto& param_view = tensor_type->tensor_view_;
        if (param_view.has_value() && param_view->layout == TensorLayout::NZ) continue;

        size_t in_rank = tensor_type->shape_.size();
        if (in_rank > full_strides.size()) continue;
        std::vector<ExprPtr> strides(full_strides.end() - static_cast<std::ptrdiff_t>(in_rank),
                                     full_strides.end());

        // AIV metadata resolves the physical layout as well as its strides.
        // Legacy InCore metadata contains only a parent shape, so that path
        // retains the layout already declared on the parameter.
        const TensorLayout param_layout =
            func->func_type_ == FunctionType::AIV
                ? metadata.layout
                : (tensor_type->tensor_view_ ? tensor_type->tensor_view_->layout : TensorLayout::ND);
        TensorView view =
            tensor_type->tensor_view_.value_or(TensorView(std::vector<ExprPtr>{}, param_layout));
        view.layout = param_layout;
        view.stride = std::move(strides);
        TypePtr new_type;
        if (auto distributed = As<DistributedTensorType>(tensor_type)) {
          new_type = std::make_shared<DistributedTensorType>(tensor_type->shape_, tensor_type->dtype_,
                                                             tensor_type->memref_, std::move(view),
                                                             distributed->window_buffer_);
        } else {
          new_type = std::make_shared<TensorType>(tensor_type->shape_, tensor_type->dtype_,
                                                  tensor_type->memref_, std::move(view));
        }
        auto new_param = std::make_shared<Var>(func->params_[param_idx]->name_hint_, new_type,
                                               func->params_[param_idx]->span_);

        changed = true;
        new_params[param_idx] = new_param;
      }

      if (!changed) continue;

      InParamSubstitutionMutator mutator(func->func_type_ == FunctionType::AIV);
      for (size_t i = 0; i < func->params_.size(); ++i) {
        if (new_params[i].get() != func->params_[i].get()) {
          mutator.AddSubstitution(func->params_[i].get(), new_params[i]);
        }
      }
      // The common function visitor also remaps reference-valued attributes,
      // including external_return_params, through the same substitutions.
      func = mutator.VisitFunction(func);
    }
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

    // Explicit AIV functions bypass the InCore lowering patterns above, but
    // still need sliced storage metadata at their call boundary.
    auto stride_targets = incore_names;
    for (const auto& [gvar, func] : p3->functions_) {
      if (func->func_type_ == FunctionType::AIV) stride_targets.insert(func->name_);
    }
    auto p4 = SliceInputStridesOptimizer().Run(p3, stride_targets);

    // Optional Pattern 5 module: default off unless a kernel explicitly opts in.
    if (!window_externalization::HasWindowizeEnabledFunction(p4)) return p4;
    return window_externalization::ApplyWindowExternalization(p4);
  };

  return CreateProgramPass(pass_func, "OptimizeOrchTensors", kOptimizeOrchTensorsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
