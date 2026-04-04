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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
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
#include "pypto/ir/transforms/op_conversion_registry.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using transform_utils::FlattenToStmts;

namespace {

std::string MakeTileValueName(const std::string& source_name) {
  return auto_name::BuildName(auto_name::GetBaseName(source_name), "", "tile");
}

std::string MakeOutParamName(size_t index) {
  return auto_name::BuildName("ret" + std::to_string(index), "", "out");
}

std::string MakeStoreResultName(size_t index) {
  return auto_name::BuildName("ret" + std::to_string(index), "", "store");
}

/**
 * @brief Visitor that collects tensor-typed variable names used directly by converted ops.
 *
 * Traverses the IR tree via IRVisitor and records the name of every Var/IterArg argument
 * whose type is TensorType and that appears in a call to an op registered in
 * OpConversionRegistry (i.e. an op that will be converted from tensor.* to tile.*).
 *
 * Used by TransformIncoreFunction to decide which tensor parameters require a synthesised
 * default Vec-space tile.load in Phase 1.  Parameters that are only referenced by
 * non-converted ops (e.g. tile.load, tile.move) already manage their own tile
 * representation and must NOT get an extra load inserted.
 *
 * Also excludes parameters used by tensor.slice and tensor.matmul since those conversions
 * create their own block.load with proper offsets/memory spaces.
 */
class TensorArgsInConvertedOpsCollector : public IRVisitor {
 public:
  explicit TensorArgsInConvertedOpsCollector(const OpConversionRegistry& conv_registry)
      : conv_registry_(conv_registry) {}

  [[nodiscard]] const std::unordered_set<const Var*>& GetUsed() const { return used_; }

  /**
   * @brief Trace from collected IterArgs to their ForStmt/WhileStmt initValue_ expressions.
   *
   * When an IterArg is in used_ (consumed by a converted op), its initValue_ may be a
   * function parameter that also needs a Phase-1 tile.load.  This fixpoint loop propagates
   * through chains of IterArgs (e.g. nested loops) until no new entries are added.
   */
  void TraceIterArgInitValues() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& [iter_arg_ptr, init_expr] : iter_arg_to_init_) {
        if (used_.count(iter_arg_ptr) == 0) continue;
        if (auto var = As<Var>(init_expr)) {
          if (As<TensorType>(var->GetType()) && used_.insert(var.get()).second) {
            changed = true;
          }
        }
      }
    }
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    auto call = As<Call>(op->value_);
    if (call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_) &&
        conv_registry_.Lookup(call->op_->name_)) {
      // Skip ops that manage their own data loading (they create block.load
      // with specific offsets/memory-spaces during conversion, so an extra
      // Phase-1 default Vec load would be redundant or wrong).
      static const std::unordered_set<std::string> kSelfLoadingOps = {"tensor.slice",      "tensor.matmul",
                                                                      "tensor.matmul_acc", "tensor.assemble",
                                                                      "tensor.read",       "tensor.write"};
      if (kSelfLoadingOps.count(call->op_->name_)) {
        IRVisitor::VisitStmt_(op);
        return;
      }
      for (const auto& arg : call->args_) {
        if (auto iter_arg = As<IterArg>(arg)) {
          if (As<TensorType>(iter_arg->GetType())) used_.insert(iter_arg.get());
        } else if (auto var = As<Var>(arg)) {
          if (As<TensorType>(var->GetType())) used_.insert(var.get());
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    for (const auto& iter_arg : op->iter_args_) {
      iter_arg_to_init_[iter_arg.get()] = iter_arg->initValue_;
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    for (const auto& iter_arg : op->iter_args_) {
      iter_arg_to_init_[iter_arg.get()] = iter_arg->initValue_;
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  const OpConversionRegistry& conv_registry_;
  std::unordered_set<const Var*> used_;
  std::unordered_map<const Var*, ExprPtr> iter_arg_to_init_;
};

/**
 * @brief Build a MakeTuple of zeros for load/store offsets (INT64).
 */
ExprPtr MakeZeroOffsets(size_t ndim, const Span& span) {
  std::vector<ExprPtr> zeros;
  zeros.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    zeros.push_back(std::make_shared<ConstInt>(0, DataType::INDEX, span));
  }
  return std::make_shared<MakeTuple>(zeros, span);
}

/**
 * @brief Build a MakeTuple from a shape vector.
 */
ExprPtr MakeShapeTuple(const std::vector<ExprPtr>& shape, const Span& span) {
  return std::make_shared<MakeTuple>(shape, span);
}

/**
 * @brief Find the YieldStmt in a list of statements and return its value types.
 *
 * Recurses into SeqStmts and ScopeStmt to find yields in nested containers.
 */
std::vector<TypePtr> FindYieldTypes(const std::vector<StmtPtr>& stmts) {
  for (const auto& stmt : stmts) {
    if (auto yield = As<YieldStmt>(stmt)) {
      std::vector<TypePtr> types;
      types.reserve(yield->value_.size());
      for (const auto& val : yield->value_) {
        types.push_back(val->GetType());
      }
      return types;
    }
    if (auto seq = As<SeqStmts>(stmt)) {
      auto found = FindYieldTypes(seq->stmts_);
      if (!found.empty()) return found;
    }
    if (auto scope = As<ScopeStmt>(stmt)) {
      auto body_stmts = FlattenToStmts(scope->body_);
      auto found = FindYieldTypes(body_stmts);
      if (!found.empty()) return found;
    }
  }
  return {};
}

// ============================================================================
// Iter-arg mapping: return_index → call_arg_index for InCore functions
// called inside ForStmt loops.
// ============================================================================

/**
 * @brief Per-InCore function mapping: which return indices correspond to which
 * call argument indices (iter-args that get yielded back).
 *
 * For example, if return[0] feeds back to iter_arg whose init is call arg[2],
 * then return_to_arg[0] = 2.
 */
using IterArgMapping = std::unordered_map<size_t, size_t>;  // return_index → call_arg_index

/**
 * @brief Scan orchestration functions to build iter-arg mappings for InCore calls.
 *
 * For each ForStmt containing an InCore call, determines which call arguments are
 * iter-args and which return values feed back to those iter-args via yield.
 *
 * Pattern recognized:
 *   for idx, (a, b, c) in pl.range(N, init_values=(a0, b0, c0)):
 *       result = incore_func(..., a, b, c, ...)
 *       new_a = result[0]; new_b = result[1]; new_c = result[2]
 *       yield_(new_a, new_b, new_c)
 *
 * Here result[0] → a (arg position of a), result[1] → b, result[2] → c.
 */
std::unordered_map<std::string, IterArgMapping> AnalyzeIterArgMappings(
    const std::vector<FunctionPtr>& functions) {
  std::unordered_map<std::string, IterArgMapping> result;

  // Pre-build set of InCore function names for O(1) lookups
  std::unordered_set<std::string> incore_func_names;
  for (const auto& func : functions) {
    if (func->func_type_ == FunctionType::InCore) {
      incore_func_names.insert(func->name_);
    }
  }

  for (const auto& func : functions) {
    if (func->func_type_ == FunctionType::InCore) continue;

    // Walk all ForStmts recursively using a stack of statement lists
    std::vector<std::vector<StmtPtr>> worklist;
    worklist.push_back(FlattenToStmts(func->body_));

    while (!worklist.empty()) {
      auto stmts = std::move(worklist.back());
      worklist.pop_back();

      for (const auto& stmt : stmts) {
        // Recurse into nested control flow
        if (auto seq = As<SeqStmts>(stmt)) {
          worklist.push_back(seq->stmts_);
          continue;
        }
        if (auto scope = As<ScopeStmt>(stmt)) {
          worklist.push_back(FlattenToStmts(scope->body_));
          continue;
        }
        if (auto if_stmt = As<IfStmt>(stmt)) {
          worklist.push_back(FlattenToStmts(if_stmt->then_body_));
          if (if_stmt->else_body_.has_value()) {
            worklist.push_back(FlattenToStmts(*if_stmt->else_body_));
          }
          continue;
        }

        auto for_stmt = As<ForStmt>(stmt);
        if (!for_stmt) {
          if (auto while_stmt = As<WhileStmt>(stmt)) {
            worklist.push_back(FlattenToStmts(while_stmt->body_));
          }
          continue;
        }

        // Flatten the ForStmt body once, reuse for both worklist and analysis
        auto body_stmts = FlattenToStmts(for_stmt->body_);

        // Always recurse into this ForStmt's body for nested loops
        worklist.push_back(body_stmts);

        if (for_stmt->iter_args_.empty()) continue;

        // Collect ALL InCore call assignments in this ForStmt body.
        // A loop body may contain multiple InCore calls (e.g. incore_2, _3, _4, _5
        // in flash_attention's sb loop), and only the one whose returns feed back
        // through yield to iter-args should get the mapping.
        std::vector<AssignStmtPtr> incore_call_assigns;
        for (const auto& body_stmt : body_stmts) {
          auto assign = As<AssignStmt>(body_stmt);
          if (!assign) continue;
          auto call = As<Call>(assign->value_);
          if (!call) continue;
          auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
          if (!gvar) continue;
          if (incore_func_names.count(gvar->name_) > 0) {
            incore_call_assigns.push_back(assign);
          }
        }
        if (incore_call_assigns.empty()) continue;

        // Build map: iter_arg pointer → iter_arg index
        std::unordered_map<const Var*, size_t> iter_arg_index;
        for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
          iter_arg_index[for_stmt->iter_args_[i].get()] = i;
        }

        // Find the yield in the loop body
        auto yield = transform_utils::FindYieldStmt(for_stmt->body_);
        if (!yield) continue;

        // Build map: call result tuple var → set of (tuple_index, dest_var)
        std::unordered_map<const Var*, std::unordered_map<size_t, const Var*>> tuple_extracts;
        for (const auto& body_stmt : body_stmts) {
          auto assign = As<AssignStmt>(body_stmt);
          if (!assign) continue;
          auto tgi = As<TupleGetItemExpr>(assign->value_);
          if (!tgi) continue;
          if (auto src_var = As<Var>(tgi->tuple_)) {
            tuple_extracts[src_var.get()][static_cast<size_t>(tgi->index_)] = assign->var_.get();
          }
        }

        // Try each InCore call to find one whose returns feed back as iter-args
        for (const auto& call_assign : incore_call_assigns) {
          auto call = As<Call>(call_assign->value_);
          auto gvar = std::dynamic_pointer_cast<const GlobalVar>(call->op_);

          // Find which call arg positions correspond to iter_args
          std::unordered_map<size_t, size_t> iter_idx_to_arg;  // iter_arg_index → call_arg_index
          for (size_t arg_i = 0; arg_i < call->args_.size(); ++arg_i) {
            const Var* raw_ptr = nullptr;
            if (auto var = As<Var>(call->args_[arg_i])) {
              raw_ptr = var.get();
            } else if (auto ia = As<IterArg>(call->args_[arg_i])) {
              raw_ptr = ia.get();
            }
            if (raw_ptr) {
              auto it = iter_arg_index.find(raw_ptr);
              if (it != iter_arg_index.end()) {
                iter_idx_to_arg[it->second] = arg_i;
              }
            }
          }

          // Check yield values: yield[j] should be result[i] from this call
          IterArgMapping mapping;
          auto call_result_var = call_assign->var_.get();

          for (size_t yield_i = 0; yield_i < yield->value_.size(); ++yield_i) {
            auto yielded = As<Var>(yield->value_[yield_i]);
            if (!yielded) continue;

            auto extract_it = tuple_extracts.find(call_result_var);
            if (extract_it == tuple_extracts.end()) continue;

            for (const auto& [ret_idx, dest_var] : extract_it->second) {
              if (dest_var == yielded.get()) {
                auto arg_it = iter_idx_to_arg.find(yield_i);
                if (arg_it != iter_idx_to_arg.end()) {
                  mapping[ret_idx] = arg_it->second;
                }
                break;
              }
            }
          }

          if (!mapping.empty()) {
            result[gvar->name_] = std::move(mapping);
            break;  // Found the right InCore call for this ForStmt
          }
        }
      }
    }
  }

  return result;
}

// ============================================================================
// IfStmt store-sinking helpers
// ============================================================================

// Tile alias chains can form when IfStmt branches yield a var that is just an
// alias for another tile var (e.g. from an assign `a = b` where both are tiles).
// We cap the chain length to avoid infinite loops on malformed IR.
constexpr int kMaxTileAliasChainDepth = 10;

using TileAliasMap = std::unordered_map<const Var*, ExprPtr>;

/// Build a tile alias map from a flat statement list.
/// Scans for AssignStmts where the value is another Var with TileType.
TileAliasMap BuildTileAliasMap(const std::vector<StmtPtr>& stmts) {
  TileAliasMap alias_map;
  for (const auto& stmt : stmts) {
    auto assign = As<AssignStmt>(stmt);
    if (!assign) continue;
    auto val_var = As<Var>(assign->value_);
    if (val_var && As<TileType>(val_var->GetType())) {
      alias_map[assign->var_.get()] = assign->value_;
    }
  }
  return alias_map;
}

/// Resolve a tile alias chain using a pre-built alias map.
/// Follows the chain to the original tile to avoid storing from an alias
/// that has no tmov to populate it.
ExprPtr ResolveTileAlias(const ExprPtr& expr, const TileAliasMap& alias_map) {
  auto var = As<Var>(expr);
  if (!var || !As<TileType>(var->GetType())) return expr;

  ExprPtr current = expr;
  for (int depth = 0; depth < kMaxTileAliasChainDepth; ++depth) {
    auto cur_var = As<Var>(current);
    if (!cur_var) break;
    auto it = alias_map.find(cur_var.get());
    if (it == alias_map.end()) break;
    current = it->second;
  }
  return current;
}

/// Find the index of the YieldStmt in a flat statement list (backward search).
/// Returns stmts.size() if not found.
size_t FindYieldIndex(const std::vector<StmtPtr>& stmts) {
  for (size_t si = stmts.size(); si > 0; --si) {
    if (As<YieldStmt>(stmts[si - 1])) return si - 1;
  }
  return stmts.size();
}

/// Insert a statement before the YieldStmt at yield_index.
void InsertBeforeYield(std::vector<StmtPtr>& stmts, size_t yield_index, const StmtPtr& to_insert) {
  stmts.insert(stmts.begin() + static_cast<std::ptrdiff_t>(yield_index), to_insert);
}

/**
 * @brief Info about a tensor.slice result that feeds into a tensor.matmul/tensor.matmul_acc operand.
 *
 * When a tensor.slice result is consumed by tensor.matmul or tensor.matmul_acc, the slice conversion
 * should produce tile.load(Mat, transpose=...) instead of tile.load(Vec) so that
 * the matmul conversion can skip the load and directly use the Mat-space tile.
 */
struct MatmulSliceInfo {
  bool is_rhs;     ///< true if the slice result is the rhs operand of matmul
  bool transpose;  ///< transpose flag from matmul (b_trans for rhs, a_trans for lhs)
};

/**
 * @brief Visitor that collects tensor.slice results consumed by tensor.matmul/tensor.matmul_acc.
 *
 * Scans the full function body to build a map from slice result variable pointers
 * to their matmul usage info (which side and transpose flag).
 */
class MatmulSlicePatternCollector : public IRVisitor {
 public:
  [[nodiscard]] const std::unordered_map<const Var*, MatmulSliceInfo>& GetTargets() const { return targets_; }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    auto call = As<Call>(op->value_);
    if (!call) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    const auto& op_name = call->op_->name_;
    if (op_name == "tensor.slice") {
      slice_results_.insert(op->var_.get());
    } else if (op_name == "tensor.matmul" || op_name == "tensor.matmul_acc") {
      CollectMatmulOperands(call, op_name == "tensor.matmul_acc");
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CollectMatmulOperands(const CallPtr& call, bool is_acc) {
    const size_t lhs_idx = is_acc ? 1 : 0;
    const size_t rhs_idx = is_acc ? 2 : 1;
    if (call->args_.size() <= rhs_idx) return;

    bool a_trans = false;
    bool b_trans = false;
    for (const auto& [k, v] : call->kwargs_) {
      if (k == "a_trans") a_trans = std::any_cast<bool>(v);
      if (k == "b_trans") b_trans = std::any_cast<bool>(v);
    }
    if (auto lhs_var = As<Var>(call->args_[lhs_idx])) {
      if (slice_results_.count(lhs_var.get())) targets_[lhs_var.get()] = {false, a_trans};
    }
    if (auto rhs_var = As<Var>(call->args_[rhs_idx])) {
      if (slice_results_.count(rhs_var.get())) targets_[rhs_var.get()] = {true, b_trans};
    }
  }

  std::unordered_set<const Var*> slice_results_;
  std::unordered_map<const Var*, MatmulSliceInfo> targets_;
};

// ============================================================================
// TypePropagatingMutator: base class that extends IRMutator with type
// propagation through control flow (IterArg types, ForStmt/WhileStmt
// return_vars, IfStmt return_vars from yield types).
//
// Subclasses override VisitStmt_(AssignStmtPtr) for domain-specific logic
// (op conversion, call-site updates, etc.) and call HandlePassThroughAssign
// for non-converted assignments.
// ============================================================================

class TypePropagatingMutator : public IRMutator {
 public:
  /// Add a mapping from an old variable to a new one (populates var_remap_).
  void AddMapping(const Expr* old_ptr, const ExprPtr& new_expr) { var_remap_[old_ptr] = new_expr; }

 protected:
  /// Override IterArg to propagate type from initValue_ when it changes.
  /// The base IRMutator preserves the original type; we want the new type
  /// so that downstream references see the correct (e.g. TileType) type.
  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = var_remap_.find(op.get());
    if (it != var_remap_.end()) return it->second;
    auto new_init = VisitExpr(op->initValue_);
    if (new_init.get() == op->initValue_.get()) return op;
    return std::make_shared<IterArg>(op->name_hint_, new_init->GetType(), new_init, op->span_);
  }

  /// Override ForStmt to update return_vars types to match iter_arg types.
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    auto new_for = As<ForStmt>(result);
    if (!new_for) return result;
    return UpdateLoopReturnVars(
        new_for->iter_args_, new_for->return_vars_, op->return_vars_,
        [&](auto new_rv) {
          return std::make_shared<ForStmt>(
              new_for->loop_var_, new_for->start_, new_for->stop_, new_for->step_, new_for->iter_args_,
              new_for->body_, std::move(new_rv), new_for->span_, new_for->kind_, new_for->chunk_size_,
              new_for->chunk_policy_, new_for->loop_origin_);
        },
        result);
  }

  /// Override WhileStmt to update return_vars types to match iter_arg types.
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    auto new_while = As<WhileStmt>(result);
    if (!new_while) return result;
    return UpdateLoopReturnVars(
        new_while->iter_args_, new_while->return_vars_, op->return_vars_,
        [&](auto new_rv) {
          return std::make_shared<WhileStmt>(new_while->condition_, new_while->iter_args_, new_while->body_,
                                             std::move(new_rv), new_while->span_);
        },
        result);
  }

  /// Override IfStmt to (a) isolate var_remap_ per branch and
  /// (b) update return_vars types from yield types.
  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_condition = VisitExpr(op->condition_);

    // Save var_remap_ and visit each branch in isolation
    auto saved_remap = var_remap_;
    auto new_then_body = VisitStmt(op->then_body_);

    var_remap_ = saved_remap;
    std::optional<StmtPtr> new_else_body;
    if (op->else_body_.has_value()) {
      new_else_body = VisitStmt(*op->else_body_);
    }
    var_remap_ = saved_remap;

    // Determine yield types from branches to update return_var types
    auto yield_types = FindYieldTypes(FlattenToStmts(new_then_body));
    if (yield_types.empty() && new_else_body.has_value()) {
      yield_types = FindYieldTypes(FlattenToStmts(*new_else_body));
    }

    std::vector<VarPtr> new_return_vars;
    new_return_vars.reserve(op->return_vars_.size());
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      if (i < yield_types.size() && yield_types[i] != rv->GetType()) {
        auto new_rv = std::make_shared<Var>(rv->name_hint_, yield_types[i], rv->span_);
        var_remap_[rv.get()] = new_rv;
        new_return_vars.push_back(new_rv);
      } else {
        new_return_vars.push_back(rv);
      }
    }

    // Copy-on-write: return original when nothing changed
    bool rv_changed = (new_return_vars != op->return_vars_);
    bool else_changed = new_else_body.has_value() != op->else_body_.has_value() ||
                        (new_else_body.has_value() && new_else_body->get() != op->else_body_->get());
    if (!rv_changed && new_condition.get() == op->condition_.get() &&
        new_then_body.get() == op->then_body_.get() && !else_changed) {
      return op;
    }

    return std::make_shared<IfStmt>(new_condition, new_then_body, new_else_body, std::move(new_return_vars),
                                    op->span_);
  }

  /// Handle a non-converted assignment: propagate type change if value type changed.
  StmtPtr HandlePassThroughAssign(const AssignStmtPtr& op, const ExprPtr& new_value) {
    if (new_value.get() == op->value_.get()) {
      // Assignment is unchanged — clear any stale remap so future uses of this Var*
      // are not rewritten to an older replacement.
      var_remap_.erase(op->var_.get());
      return op;
    }
    if (new_value->GetType() != op->value_->GetType()) {
      auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_value->GetType(), op->var_->span_);
      var_remap_[op->var_.get()] = new_var;
      return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
    }
    // Value changed but type did not — keep original Var, clear any stale remap.
    var_remap_.erase(op->var_.get());
    return std::make_shared<AssignStmt>(op->var_, new_value, op->span_);
  }

 private:
  /// Shared logic for ForStmt/WhileStmt: update return_vars types to match iter_arg types.
  template <typename ReconstructFn>
  StmtPtr UpdateLoopReturnVars(const std::vector<IterArgPtr>& new_iter_args,
                               const std::vector<VarPtr>& new_return_vars,
                               const std::vector<VarPtr>& orig_return_vars, ReconstructFn reconstruct,
                               const StmtPtr& original) {
    bool rv_changed = false;
    std::vector<VarPtr> updated_rv;
    updated_rv.reserve(new_return_vars.size());
    for (size_t i = 0; i < new_return_vars.size(); ++i) {
      const auto& rv = new_return_vars[i];
      if (i < new_iter_args.size() && new_iter_args[i]->GetType() != rv->GetType()) {
        auto updated = std::make_shared<Var>(rv->name_hint_, new_iter_args[i]->GetType(), rv->span_);
        // Register mapping for both original and current return_var pointers
        var_remap_[orig_return_vars[i].get()] = updated;
        if (rv.get() != orig_return_vars[i].get()) var_remap_[rv.get()] = updated;
        updated_rv.push_back(updated);
        rv_changed = true;
      } else {
        updated_rv.push_back(rv);
      }
    }
    if (!rv_changed) return original;
    return reconstruct(std::move(updated_rv));
  }
};

// ============================================================================
// TensorToTileMutator: converts tensor ops to tile ops in InCore function
// bodies.  Overrides AssignStmt/EvalStmt to run converters from
// OpConversionRegistry; everything else (control flow recursion, variable
// substitution, IterArg/return_var type propagation) comes from the base.
// ============================================================================

class TensorToTileMutator : public TypePropagatingMutator {
 public:
  TensorToTileMutator(const OpConversionRegistry& conv_registry, const OpRegistry& op_registry,
                      const std::unordered_map<const Var*, MatmulSliceInfo>& matmul_targets)
      : conv_registry_(conv_registry), op_registry_(op_registry), matmul_targets_(matmul_targets) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = VisitExpr(op->value_);
    auto call = As<Call>(new_value);

    // Non-call values: propagate type change
    if (!call) return HandlePassThroughAssign(op, new_value);

    // Function calls (GlobalVar) pass through — only process op calls
    if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
      LOG_DEBUG << "[TensorToTileMutator] Skipping GlobalVar call: " << call->op_->name_;
      return HandlePassThroughAssign(op, new_value);
    }

    const auto* converter = conv_registry_.Lookup(call->op_->name_);
    if (!converter) {
      // Verify unregistered TensorOps are expected passthroughs
      if (op_registry_.IsRegistered(call->op_->name_)) {
        const auto& entry = op_registry_.GetEntry(call->op_->name_);
        static const std::unordered_set<std::string> kPassthroughTensorOps = {"tensor.dim"};
        INTERNAL_CHECK(entry.GetOpCategory() != "TensorOp" || kPassthroughTensorOps.count(call->op_->name_))
            << "TensorOp \"" << call->op_->name_ << "\" has no registered tile conversion. "
            << "Add a conversion in src/ir/transforms/op_conversion_registry.cpp.";
      }
      return HandlePassThroughAssign(op, new_value);
    }

    // Special: tensor.slice feeding into tensor.matmul → tile.load(Mat)
    if (call->op_->name_ == "tensor.slice" && matmul_targets_.count(op->var_.get())) {
      auto mat_load = HandleMatmulSlice(op, call);
      if (mat_load) return mat_load;
    }

    // Run the converter
    auto conv_result = (*converter)(call->args_, call->kwargs_, call->span_);

    // Prologue statements may themselves contain tensor ops — recurse
    std::vector<StmtPtr> stmts;
    stmts.reserve(conv_result.prologue.size() + 1);
    for (auto& prologue_stmt : conv_result.prologue) {
      stmts.push_back(VisitStmt(prologue_stmt));
    }

    // Revisit result after mutating prologue — prologue conversions may have
    // remapped vars that the result expression references.
    auto new_result = VisitExpr(conv_result.result);

    auto tile_name = MakeTileValueName(op->var_->name_hint_);
    auto tile_var = std::make_shared<Var>(tile_name, new_result->GetType(), op->var_->span_);
    stmts.push_back(std::make_shared<AssignStmt>(tile_var, new_result, op->span_));
    var_remap_[op->var_.get()] = tile_var;

    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto new_expr = VisitExpr(op->expr_);
    // Helper: return updated EvalStmt only when the expression actually changed.
    auto maybe_update = [&]() -> StmtPtr {
      return (new_expr.get() != op->expr_.get()) ? std::make_shared<EvalStmt>(new_expr, op->span_) : op;
    };

    auto call = As<Call>(new_expr);
    if (!call || std::dynamic_pointer_cast<const GlobalVar>(call->op_)) return maybe_update();

    const auto* converter = conv_registry_.Lookup(call->op_->name_);
    if (!converter) return maybe_update();

    auto conv_result = (*converter)(call->args_, call->kwargs_, call->span_);
    std::vector<StmtPtr> stmts;
    stmts.reserve(conv_result.prologue.size() + 1);
    for (auto& prologue_stmt : conv_result.prologue) {
      stmts.push_back(VisitStmt(prologue_stmt));
    }
    auto new_result = VisitExpr(conv_result.result);
    stmts.push_back(std::make_shared<EvalStmt>(new_result, op->span_));
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

 private:
  /// Handle tensor.slice that feeds into tensor.matmul — produce tile.load(Mat).
  StmtPtr HandleMatmulSlice(const AssignStmtPtr& op, const CallPtr& call) {
    const auto& info = matmul_targets_.at(op->var_.get());
    const auto& input = call->args_[0];
    auto tensor_type = As<TensorType>(input->GetType());
    if (!tensor_type) return nullptr;

    const auto& shape_arg = call->args_[1];
    const auto& offset_arg = call->args_[2];
    ExprPtr valid_shapes = (call->args_.size() == 4) ? call->args_[3] : shape_arg;

    std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Mat},
                                                                 {"transpose", info.transpose}};
    auto load_call = op_registry_.Create("tile.load", {input, offset_arg, shape_arg, valid_shapes},
                                         load_kwargs, call->span_);

    auto tile_name = MakeTileValueName(op->var_->name_hint_);
    auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), op->var_->span_);
    var_remap_[op->var_.get()] = tile_var;
    return std::make_shared<AssignStmt>(tile_var, load_call, op->span_);
  }

  const OpConversionRegistry& conv_registry_;
  const OpRegistry& op_registry_;
  const std::unordered_map<const Var*, MatmulSliceInfo>&
      matmul_targets_;  // owned by MatmulSlicePatternCollector — must outlive this mutator
};

class VarUseVisitor : public IRVisitor {
 public:
  explicit VarUseVisitor(const Var* target) : target_(target) {}

  [[nodiscard]] bool Found() const { return found_; }
  void CheckExpr(const ExprPtr& expr) { VisitExpr(expr); }
  void CheckStmt(const StmtPtr& stmt) { VisitStmt(stmt); }

 protected:
  void VisitExpr(const ExprPtr& expr) override {
    if (found_ || !expr) return;
    IRVisitor::VisitExpr(expr);
  }

  void VisitStmt(const StmtPtr& stmt) override {
    if (found_ || !stmt) return;
    IRVisitor::VisitStmt(stmt);
  }

  void VisitVarLike_(const VarPtr& op) override {
    if (op.get() == target_) {
      found_ = true;
      return;
    }
  }

  void VisitExpr_(const IterArgPtr& op) override { VisitVarLike_(op); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->value_);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->expr_);
  }

  void VisitStmt_(const YieldStmtPtr& op) override {
    if (!op) return;
    for (const auto& value : op->value_) {
      VisitExpr(value);
      if (found_) return;
    }
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    if (!op) return;
    for (const auto& value : op->value_) {
      VisitExpr(value);
      if (found_) return;
    }
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    if (!op) return;
    for (const auto& stmt : op->stmts_) {
      VisitStmt(stmt);
      if (found_) return;
    }
  }

  void VisitStmt_(const ScopeStmtPtr& op) override {
    if (!op) return;
    VisitStmt(op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->condition_);
    if (found_) return;
    VisitStmt(op->then_body_);
    if (found_ || !op->else_body_.has_value()) return;
    VisitStmt(*op->else_body_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->start_);
    if (found_) return;
    VisitExpr(op->stop_);
    if (found_) return;
    VisitExpr(op->step_);
    if (found_) return;
    if (op->chunk_size_.has_value()) {
      VisitExpr(*op->chunk_size_);
      if (found_) return;
    }
    for (const auto& iter_arg : op->iter_args_) {
      VisitExpr(iter_arg->initValue_);
      if (found_) return;
    }
    VisitStmt(op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    VisitExpr(op->condition_);
    if (found_) return;
    for (const auto& iter_arg : op->iter_args_) {
      VisitExpr(iter_arg->initValue_);
      if (found_) return;
    }
    VisitStmt(op->body_);
  }

 private:
  const Var* target_;
  bool found_ = false;
};

bool ExprUsesVar(const ExprPtr& expr, const Var* target) {
  if (!expr || !target) return false;
  VarUseVisitor visitor(target);
  visitor.CheckExpr(expr);
  return visitor.Found();
}

bool StmtUsesVar(const StmtPtr& stmt, const Var* target) {
  if (!stmt || !target) return false;
  VarUseVisitor visitor(target);
  visitor.CheckStmt(stmt);
  return visitor.Found();
}

using ParamOrigins = std::vector<size_t>;
using AliasOriginMap = std::unordered_map<const Var*, ParamOrigins>;

struct YieldAliasInfo {
  bool has_yield = false;
  std::vector<ParamOrigins> origins;
};

void AddOrigin(ParamOrigins& origins, size_t index) {
  if (std::find(origins.begin(), origins.end(), index) == origins.end()) {
    origins.push_back(index);
  }
}

void MergeOrigins(ParamOrigins& dst, const ParamOrigins& src) {
  for (size_t index : src) {
    AddOrigin(dst, index);
  }
}

void MarkAccess(const ParamOrigins& origins, std::vector<bool>& flags) {
  for (size_t index : origins) {
    if (index < flags.size()) {
      flags[index] = true;
    }
  }
}

ParamOrigins LookupOrigins(const Var* var, const AliasOriginMap& origin_map) {
  if (!var) return {};
  auto it = origin_map.find(var);
  if (it == origin_map.end()) return {};
  return it->second;
}

ParamOrigins CollectReferencedOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map);

ExprPtr GetCallKwargExpr(const CallPtr& call, const std::string& key) {
  if (!call || !call->HasKwarg(key)) return nullptr;
  return call->GetKwarg<ExprPtr>(key, ExprPtr{});
}

ExprPtr GetWriteTargetExpr(const CallPtr& call) {
  if (!call) return nullptr;

  const std::string& op_name = call->op_->name_;
  if (op_name == "tensor.write" && !call->args_.empty()) {
    return call->args_[0];
  }
  if (op_name == "tile.store") {
    if (call->args_.size() >= 3) {
      return call->args_[2];
    }
    return GetCallKwargExpr(call, "output_tensor");
  }
  if (op_name == "tensor.assemble" && !call->args_.empty()) {
    return call->args_[0];
  }
  return nullptr;
}

void UpdateTensorAliasOrigin(const VarPtr& var, const ParamOrigins& origins, AliasOriginMap& origin_map) {
  if (As<TensorType>(var->GetType()) && !origins.empty()) {
    origin_map[var.get()] = origins;
  } else {
    origin_map.erase(var.get());
  }
}

ParamOrigins GetAliasOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map) {
  if (!expr) return {};

  if (auto var = AsVarLike(expr)) {
    return LookupOrigins(var.get(), origin_map);
  }

  if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
    if (auto tuple = As<MakeTuple>(tuple_get->tuple_)) {
      if (tuple_get->index_ >= 0 && static_cast<size_t>(tuple_get->index_) < tuple->elements_.size()) {
        return GetAliasOrigins(tuple->elements_[static_cast<size_t>(tuple_get->index_)], origin_map);
      }
    }
    return {};
  }

  auto call = As<Call>(expr);
  if (!call) return {};

  const std::string& op_name = call->op_->name_;
  if (auto write_target = GetWriteTargetExpr(call)) {
    return GetAliasOrigins(write_target, origin_map);
  }
  if (op_name == "tensor.slice" && !call->args_.empty()) {
    return GetAliasOrigins(call->args_[0], origin_map);
  }
  return {};
}

ParamOrigins CollectReferencedOrigins(const ExprPtr& expr, const AliasOriginMap& origin_map) {
  if (!expr) return {};

  if (auto var = AsVarLike(expr)) {
    return LookupOrigins(var.get(), origin_map);
  }

  if (auto tuple = As<MakeTuple>(expr)) {
    ParamOrigins origins;
    for (const auto& element : tuple->elements_) {
      MergeOrigins(origins, CollectReferencedOrigins(element, origin_map));
    }
    return origins;
  }

  if (auto tuple_get = As<TupleGetItemExpr>(expr)) {
    if (auto tuple = As<MakeTuple>(tuple_get->tuple_)) {
      if (tuple_get->index_ >= 0 && static_cast<size_t>(tuple_get->index_) < tuple->elements_.size()) {
        return CollectReferencedOrigins(tuple->elements_[static_cast<size_t>(tuple_get->index_)], origin_map);
      }
    }
    return CollectReferencedOrigins(tuple_get->tuple_, origin_map);
  }

  auto call = As<Call>(expr);
  if (!call) return {};

  ParamOrigins origins;
  for (const auto& arg : call->args_) {
    MergeOrigins(origins, CollectReferencedOrigins(arg, origin_map));
  }
  return origins;
}

void AnalyzeCallAccess(const CallPtr& call, const AliasOriginMap& origin_map, std::vector<bool>& has_read,
                       std::vector<bool>& has_write) {
  if (!call) return;

  const std::string& op_name = call->op_->name_;
  if (op_name == "tile.load" || op_name == "tensor.read") {
    if (!call->args_.empty()) {
      MarkAccess(GetAliasOrigins(call->args_[0], origin_map), has_read);
    }
    for (size_t i = 1; i < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    return;
  }

  if (op_name == "tile.store") {
    if (!call->args_.empty()) {
      MarkAccess(CollectReferencedOrigins(call->args_[0], origin_map), has_read);
    }
    for (size_t i = 1; i + 1 < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    if (call->args_.size() < 3) {
      MarkAccess(CollectReferencedOrigins(GetCallKwargExpr(call, "offsets"), origin_map), has_read);
    }
    if (auto write_target = GetWriteTargetExpr(call)) {
      MarkAccess(GetAliasOrigins(write_target, origin_map), has_write);
    }
    return;
  }

  if (op_name == "tensor.write") {
    for (size_t i = 1; i < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    if (auto write_target = GetWriteTargetExpr(call)) {
      MarkAccess(GetAliasOrigins(write_target, origin_map), has_write);
    }
    return;
  }

  if (op_name == "tensor.assemble") {
    for (size_t i = 1; i < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    if (!call->args_.empty()) {
      MarkAccess(GetAliasOrigins(call->args_[0], origin_map), has_write);
    }
    return;
  }

  if (op_name == "tensor.slice" || op_name == "tensor.create" || op_name == "tensor.full") {
    for (size_t i = 1; i < call->args_.size(); ++i) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    return;
  }

  for (const auto& arg : call->args_) {
    MarkAccess(CollectReferencedOrigins(arg, origin_map), has_read);
  }
}

YieldAliasInfo MergeYieldInfos(const YieldAliasInfo& lhs, const YieldAliasInfo& rhs) {
  if (!lhs.has_yield) return rhs;
  if (!rhs.has_yield) return lhs;

  YieldAliasInfo merged;
  merged.has_yield = true;
  size_t count = std::max(lhs.origins.size(), rhs.origins.size());
  merged.origins.resize(count);
  for (size_t i = 0; i < lhs.origins.size(); ++i) {
    MergeOrigins(merged.origins[i], lhs.origins[i]);
  }
  for (size_t i = 0; i < rhs.origins.size(); ++i) {
    MergeOrigins(merged.origins[i], rhs.origins[i]);
  }
  return merged;
}

YieldAliasInfo AnalyzeStmtAliases(const StmtPtr& stmt, AliasOriginMap& origin_map,
                                  std::vector<bool>& has_read, std::vector<bool>& has_write);

YieldAliasInfo AnalyzeStmtSequenceAliases(const std::vector<StmtPtr>& stmts, AliasOriginMap& origin_map,
                                          std::vector<bool>& has_read, std::vector<bool>& has_write) {
  YieldAliasInfo last_yield;
  for (const auto& stmt : stmts) {
    auto yield_info = AnalyzeStmtAliases(stmt, origin_map, has_read, has_write);
    if (yield_info.has_yield) {
      last_yield = yield_info;
    }
  }
  return last_yield;
}

YieldAliasInfo AnalyzeStmtAliases(const StmtPtr& stmt, AliasOriginMap& origin_map,
                                  std::vector<bool>& has_read, std::vector<bool>& has_write) {
  if (!stmt) return {};

  if (auto assign = As<AssignStmt>(stmt)) {
    if (auto call = As<Call>(assign->value_)) {
      AnalyzeCallAccess(call, origin_map, has_read, has_write);
    }

    if (As<TensorType>(assign->var_->GetType())) {
      auto origins = GetAliasOrigins(assign->value_, origin_map);
      UpdateTensorAliasOrigin(assign->var_, origins, origin_map);
    }
    return {};
  }

  if (auto eval = As<EvalStmt>(stmt)) {
    if (auto call = As<Call>(eval->expr_)) {
      AnalyzeCallAccess(call, origin_map, has_read, has_write);
    }
    return {};
  }

  if (auto seq = As<SeqStmts>(stmt)) {
    return AnalyzeStmtSequenceAliases(seq->stmts_, origin_map, has_read, has_write);
  }

  if (auto scope = As<ScopeStmt>(stmt)) {
    return AnalyzeStmtAliases(scope->body_, origin_map, has_read, has_write);
  }

  if (auto if_stmt = As<IfStmt>(stmt)) {
    if (auto cond_call = As<Call>(if_stmt->condition_)) {
      AnalyzeCallAccess(cond_call, origin_map, has_read, has_write);
    }
    auto then_map = origin_map;
    auto then_yield = AnalyzeStmtAliases(if_stmt->then_body_, then_map, has_read, has_write);
    YieldAliasInfo else_yield;
    if (if_stmt->else_body_.has_value()) {
      auto else_map = origin_map;
      else_yield = AnalyzeStmtAliases(*if_stmt->else_body_, else_map, has_read, has_write);
    }
    auto merged_yield = MergeYieldInfos(then_yield, else_yield);
    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins;
      if (merged_yield.has_yield && i < merged_yield.origins.size()) {
        origins = merged_yield.origins[i];
      }
      UpdateTensorAliasOrigin(if_stmt->return_vars_[i], origins, origin_map);
    }
    return merged_yield;
  }

  if (auto for_stmt = As<ForStmt>(stmt)) {
    if (auto start_call = As<Call>(for_stmt->start_)) {
      AnalyzeCallAccess(start_call, origin_map, has_read, has_write);
    }
    if (auto stop_call = As<Call>(for_stmt->stop_)) {
      AnalyzeCallAccess(stop_call, origin_map, has_read, has_write);
    }
    if (auto step_call = As<Call>(for_stmt->step_)) {
      AnalyzeCallAccess(step_call, origin_map, has_read, has_write);
    }
    if (for_stmt->chunk_size_.has_value()) {
      if (auto chunk_call = As<Call>(*for_stmt->chunk_size_)) {
        AnalyzeCallAccess(chunk_call, origin_map, has_read, has_write);
      }
    }

    auto body_map = origin_map;
    std::vector<ParamOrigins> init_origins(for_stmt->iter_args_.size());
    for (size_t i = 0; i < for_stmt->iter_args_.size(); ++i) {
      init_origins[i] = GetAliasOrigins(for_stmt->iter_args_[i]->initValue_, origin_map);
      if (!init_origins[i].empty()) {
        body_map[for_stmt->iter_args_[i].get()] = init_origins[i];
      } else {
        body_map.erase(for_stmt->iter_args_[i].get());
      }
    }

    auto yield_info = AnalyzeStmtAliases(for_stmt->body_, body_map, has_read, has_write);
    for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins;
      if (yield_info.has_yield && i < yield_info.origins.size()) {
        origins = yield_info.origins[i];
      }
      if (origins.empty() && i < init_origins.size()) {
        origins = init_origins[i];
      }
      UpdateTensorAliasOrigin(for_stmt->return_vars_[i], origins, origin_map);
    }
    return {};
  }

  if (auto while_stmt = As<WhileStmt>(stmt)) {
    if (auto cond_call = As<Call>(while_stmt->condition_)) {
      AnalyzeCallAccess(cond_call, origin_map, has_read, has_write);
    }

    auto body_map = origin_map;
    std::vector<ParamOrigins> init_origins(while_stmt->iter_args_.size());
    for (size_t i = 0; i < while_stmt->iter_args_.size(); ++i) {
      init_origins[i] = GetAliasOrigins(while_stmt->iter_args_[i]->initValue_, origin_map);
      if (!init_origins[i].empty()) {
        body_map[while_stmt->iter_args_[i].get()] = init_origins[i];
      } else {
        body_map.erase(while_stmt->iter_args_[i].get());
      }
    }

    auto yield_info = AnalyzeStmtAliases(while_stmt->body_, body_map, has_read, has_write);
    for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins;
      if (yield_info.has_yield && i < yield_info.origins.size()) {
        origins = yield_info.origins[i];
      }
      if (origins.empty() && i < init_origins.size()) {
        origins = init_origins[i];
      }
      if (As<TensorType>(while_stmt->return_vars_[i]->GetType()) && !origins.empty()) {
        origin_map[while_stmt->return_vars_[i].get()] = origins;
      } else {
        origin_map.erase(while_stmt->return_vars_[i].get());
      }
    }
    return {};
  }

  if (auto yield = As<YieldStmt>(stmt)) {
    YieldAliasInfo info;
    info.has_yield = true;
    info.origins.reserve(yield->value_.size());
    for (const auto& value : yield->value_) {
      info.origins.push_back(GetAliasOrigins(value, origin_map));
    }
    return info;
  }

  return {};
}

void UpgradeWrittenTensorParamDirections(const std::vector<StmtPtr>& stmts, const std::vector<VarPtr>& params,
                                         std::vector<ParamDirection>& param_directions) {
  std::vector<bool> has_read(params.size(), false);
  std::vector<bool> has_write(params.size(), false);
  AliasOriginMap origin_map;

  for (size_t i = 0; i < params.size() && i < param_directions.size(); ++i) {
    if (!As<TensorType>(params[i]->GetType())) {
      continue;
    }
    origin_map[params[i].get()] = ParamOrigins{i};
  }

  auto analysis_map = origin_map;
  AnalyzeStmtSequenceAliases(stmts, analysis_map, has_read, has_write);

  for (size_t i = 0; i < params.size() && i < param_directions.size(); ++i) {
    if (param_directions[i] != ParamDirection::In || !has_write[i]) {
      continue;
    }
    param_directions[i] = has_read[i] ? ParamDirection::InOut : ParamDirection::Out;
  }
}

struct ReturnedAssembleLoopRewrite {
  size_t stmt_index;
  std::optional<size_t> dead_init_stmt_index;
  ForStmtPtr new_for_stmt;
  VarPtr new_return_var;
};

std::optional<ReturnedAssembleLoopRewrite> RewriteReturnedAssembleLoopToStore(
    const std::vector<StmtPtr>& stmts, const ExprPtr& ret_expr, const VarPtr& out_param,
    const TensorTypePtr& out_tensor_type, const OpRegistry& op_registry) {
  auto ret_var = As<Var>(ret_expr);
  if (!ret_var) return std::nullopt;

  for (size_t stmt_index = 0; stmt_index < stmts.size(); ++stmt_index) {
    auto for_stmt = As<ForStmt>(stmts[stmt_index]);
    if (!for_stmt || for_stmt->iter_args_.size() != 1 || for_stmt->return_vars_.size() != 1 ||
        for_stmt->return_vars_[0].get() != ret_var.get()) {
      continue;
    }

    const auto& old_iter_arg = for_stmt->iter_args_[0];
    auto body_stmts = FlattenToStmts(for_stmt->body_);

    AssignStmtPtr assemble_assign;
    YieldStmtPtr yield_stmt;
    for (const auto& body_stmt : body_stmts) {
      auto assign = As<AssignStmt>(body_stmt);
      if (assign) {
        auto call = As<Call>(assign->value_);
        bool is_target_assemble = false;
        if (call && call->op_->name_ == "tile.assemble" && call->args_.size() == 3) {
          if (auto iter = As<IterArg>(call->args_[0])) {
            is_target_assemble = iter.get() == old_iter_arg.get();
          } else if (auto var = As<Var>(call->args_[0])) {
            is_target_assemble = var.get() == old_iter_arg.get();
          }
        }
        if (is_target_assemble) {
          if (assemble_assign) return std::nullopt;
          auto assemble_call = As<Call>(assign->value_);
          CHECK(assemble_call) << "Internal error: expected tile.assemble call in assemble loop rewrite";
          if (ExprUsesVar(assemble_call->args_[1], old_iter_arg.get()) ||
              ExprUsesVar(assemble_call->args_[2], old_iter_arg.get())) {
            return std::nullopt;
          }
          assemble_assign = assign;
          continue;
        }
      }

      if (auto yield = As<YieldStmt>(body_stmt)) {
        if (yield->value_.size() != 1 || yield_stmt) return std::nullopt;
        yield_stmt = yield;
        continue;
      }

      if (StmtUsesVar(body_stmt, old_iter_arg.get())) {
        return std::nullopt;
      }
    }

    if (!assemble_assign || !yield_stmt) return std::nullopt;

    auto yielded_var = As<Var>(yield_stmt->value_[0]);
    if (!yielded_var || yielded_var.get() != assemble_assign->var_.get()) {
      return std::nullopt;
    }

    auto assemble_call = As<Call>(assemble_assign->value_);
    CHECK(assemble_call) << "Internal error: expected tile.assemble call in assemble loop rewrite";

    auto new_iter_arg =
        std::make_shared<IterArg>(old_iter_arg->name_hint_, out_tensor_type, out_param, old_iter_arg->span_);
    auto store_call = op_registry.Create(
        "tile.store", {assemble_call->args_[1], assemble_call->args_[2], new_iter_arg}, assemble_call->span_);
    auto store_var = std::make_shared<Var>(assemble_assign->var_->name_hint_, store_call->GetType(),
                                           assemble_assign->var_->span_);

    std::vector<StmtPtr> new_body_stmts;
    new_body_stmts.reserve(body_stmts.size());
    for (const auto& body_stmt : body_stmts) {
      if (body_stmt == assemble_assign) {
        new_body_stmts.push_back(std::make_shared<AssignStmt>(store_var, store_call, assemble_assign->span_));
        continue;
      }
      if (body_stmt == yield_stmt) {
        new_body_stmts.push_back(
            std::make_shared<YieldStmt>(std::vector<ExprPtr>{store_var}, yield_stmt->span_));
        continue;
      }
      new_body_stmts.push_back(body_stmt);
    }

    auto new_return_var = std::make_shared<Var>(for_stmt->return_vars_[0]->name_hint_, out_tensor_type,
                                                for_stmt->return_vars_[0]->span_);
    auto new_for_stmt =
        std::make_shared<ForStmt>(for_stmt->loop_var_, for_stmt->start_, for_stmt->stop_, for_stmt->step_,
                                  std::vector<IterArgPtr>{new_iter_arg},
                                  SeqStmts::Flatten(std::move(new_body_stmts), for_stmt->body_->span_),
                                  std::vector<VarPtr>{new_return_var}, for_stmt->span_, for_stmt->kind_,
                                  for_stmt->chunk_size_, for_stmt->chunk_policy_, for_stmt->loop_origin_);

    std::optional<size_t> dead_init_stmt_index;
    if (auto init_var = As<Var>(old_iter_arg->initValue_)) {
      bool has_other_uses = false;
      for (size_t other_index = 0; other_index < stmts.size(); ++other_index) {
        const StmtPtr& stmt_to_check = other_index == stmt_index ? new_for_stmt : stmts[other_index];
        if (StmtUsesVar(stmt_to_check, init_var.get())) {
          has_other_uses = true;
          break;
        }
      }
      if (!has_other_uses) {
        for (size_t other_index = 0; other_index < stmts.size(); ++other_index) {
          auto init_assign = As<AssignStmt>(stmts[other_index]);
          if (init_assign && init_assign->var_.get() == init_var.get()) {
            dead_init_stmt_index = other_index;
            break;
          }
        }
      }
    }

    return ReturnedAssembleLoopRewrite{
        stmt_index,
        dead_init_stmt_index,
        new_for_stmt,
        new_return_var,
    };
  }

  return std::nullopt;
}

/**
 * @brief Transform an InCore function: insert loads, convert ops, insert stores
 *
 * @param func The InCore function to transform
 * @return Transformed function with tile ops, plus the number of added output params
 */
struct IncoreTransformResult {
  FunctionPtr func;
  size_t num_added_outputs;
};

IncoreTransformResult TransformIncoreFunction(const FunctionPtr& func,
                                              const IterArgMapping& iter_arg_mapping) {
  auto& conv_registry = OpConversionRegistry::GetInstance();
  auto& op_registry = OpRegistry::GetInstance();
  const auto& span = func->span_;

  // Pre-scan for tensor.slice → tensor.matmul patterns (need Mat-space loads).
  MatmulSlicePatternCollector matmul_collector;
  matmul_collector.VisitStmt(func->body_);

  // Create the body mutator
  TensorToTileMutator mutator(conv_registry, op_registry, matmul_collector.GetTargets());

  // New body statements (prefix tile.loads + mutated body)
  std::vector<StmtPtr> new_stmts;

  // Phase 1: Insert tile.load for each TensorType parameter that is directly consumed
  // by a converted tensor op.  Parameters that are only referenced by non-converted ops
  // (e.g. tile.load, tile.move) already manage their own tile representation and must
  // NOT get an additional Vec-space load inserted here.
  TensorArgsInConvertedOpsCollector collector(conv_registry);
  collector.VisitStmt(func->body_);
  collector.TraceIterArgInitValues();
  const auto& params_used_by_converted_ops = collector.GetUsed();

  for (const auto& var : func->params_) {
    auto tensor_type = As<TensorType>(var->GetType());
    if (!tensor_type) continue;

    if (params_used_by_converted_ops.find(var.get()) == params_used_by_converted_ops.end()) continue;

    auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), span);
    auto shapes = MakeShapeTuple(tensor_type->shape_, span);
    std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", MemorySpace::Vec},
                                                                 {"transpose", false}};
    auto load_call = op_registry.Create("tile.load", {var, offsets, shapes, shapes}, load_kwargs, span);

    std::string tile_name = MakeTileValueName(var->name_hint_);
    auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), span);

    new_stmts.push_back(std::make_shared<AssignStmt>(tile_var, load_call, span));
    mutator.AddMapping(var.get(), tile_var);
  }

  // Phase 2: Transform body via mutator (handles control flow recursion + op conversion)
  auto body_stmts = FlattenToStmts(func->body_);

  // Separate return statement from body (will be replaced in Phase 3)
  ReturnStmtPtr return_stmt;
  std::vector<StmtPtr> non_return_stmts;
  for (const auto& stmt : body_stmts) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      return_stmt = ret;
    } else {
      non_return_stmts.push_back(stmt);
    }
  }

  auto body_to_transform = SeqStmts::Flatten(std::move(non_return_stmts), span);
  auto mutated = mutator.VisitStmt(body_to_transform);
  auto transformed = FlattenToStmts(mutated);
  new_stmts.insert(new_stmts.end(), transformed.begin(), transformed.end());

  // Phase 3: Add output params + tile.store for return values
  std::vector<VarPtr> new_params = func->params_;
  std::vector<ParamDirection> new_param_directions = func->param_directions_;
  std::vector<TypePtr> new_return_types;
  size_t num_added_outputs = 0;
  std::unordered_set<size_t> merged_return_indices;

  if (return_stmt) {
    std::vector<ExprPtr> new_return_exprs;

    // Phase 3a: Sink tile.store into IfStmt branches for return values that
    // map to iter-arg In params. This avoids phi-variable-mediated stores that
    // are problematic on some backends (e.g. A5).
    //
    // Find the last IfStmt in new_stmts — candidate for store sinking.
    IfStmtPtr last_if_stmt;
    size_t last_if_index = 0;
    for (size_t si = new_stmts.size(); si > 0; --si) {
      if (auto ifs = As<IfStmt>(new_stmts[si - 1])) {
        last_if_stmt = ifs;
        last_if_index = si - 1;
        break;
      }
    }

    // Collect return values that are IfStmt return_vars AND have iter-arg mappings.
    struct IfStmtSinkCandidate {
      size_t return_index;
      size_t ifstmt_rv_index;
      size_t in_param_index;
      TensorTypePtr orig_tensor_type;
    };
    std::vector<IfStmtSinkCandidate> sink_candidates;

    if (last_if_stmt && last_if_stmt->else_body_.has_value() && !last_if_stmt->return_vars_.empty()) {
      // Pre-build map from IfStmt return var pointers to their indices.
      std::unordered_map<const Var*, size_t> if_return_var_to_index;
      for (size_t rv_i = 0; rv_i < last_if_stmt->return_vars_.size(); ++rv_i) {
        if_return_var_to_index[last_if_stmt->return_vars_[rv_i].get()] = rv_i;
      }

      for (size_t i = 0; i < return_stmt->value_.size(); ++i) {
        auto ret_expr = mutator.VisitExpr(return_stmt->value_[i]);
        auto ret_var = As<Var>(ret_expr);
        if (!ret_var || !As<TileType>(ret_var->GetType())) continue;

        auto if_rv_it = if_return_var_to_index.find(ret_var.get());
        if (if_rv_it == if_return_var_to_index.end()) continue;
        size_t rv_i = if_rv_it->second;

        auto map_it = iter_arg_mapping.find(i);
        if (map_it != iter_arg_mapping.end()) {
          size_t arg_idx = map_it->second;
          if (arg_idx < func->params_.size()) {
            auto orig_tensor_type = As<TensorType>(func->return_types_[i]);
            if (orig_tensor_type) {
              sink_candidates.push_back({i, rv_i, arg_idx, orig_tensor_type});
            }
          }
        }
      }
    }

    // Apply store sinking into both branches.
    if (!sink_candidates.empty()) {
      INTERNAL_CHECK(last_if_stmt->else_body_.has_value())
          << "Internal error: sink candidates require IfStmt with else branch";

      auto then_stmts = FlattenToStmts(last_if_stmt->then_body_);
      auto else_stmts = FlattenToStmts(*last_if_stmt->else_body_);

      auto then_yield = transform_utils::FindYieldStmt(last_if_stmt->then_body_);
      auto else_yield = transform_utils::FindYieldStmt(*last_if_stmt->else_body_);

      if (then_yield && else_yield) {
        auto then_alias_map = BuildTileAliasMap(then_stmts);
        auto else_alias_map = BuildTileAliasMap(else_stmts);

        size_t then_yield_idx = FindYieldIndex(then_stmts);
        size_t else_yield_idx = FindYieldIndex(else_stmts);
        INTERNAL_CHECK(then_yield_idx < then_stmts.size())
            << "Internal error: yield not found in then branch";
        INTERNAL_CHECK(else_yield_idx < else_stmts.size())
            << "Internal error: yield not found in else branch";

        auto new_then_yield_values = then_yield->value_;
        auto new_else_yield_values = else_yield->value_;
        std::vector<VarPtr> new_rv = last_if_stmt->return_vars_;

        for (const auto& cand : sink_candidates) {
          auto in_param = new_params[cand.in_param_index];
          auto offsets = MakeZeroOffsets(cand.orig_tensor_type->shape_.size(), span);

          // Resolve aliases so we store from the actual computed tile, not an uninitialized alias.
          auto then_tile = ResolveTileAlias(then_yield->value_[cand.ifstmt_rv_index], then_alias_map);
          auto then_store = op_registry.Create("tile.store", {then_tile, offsets, in_param}, span);
          auto then_store_var =
              std::make_shared<Var>(MakeStoreResultName(cand.return_index), then_store->GetType(), span);

          auto else_tile = ResolveTileAlias(else_yield->value_[cand.ifstmt_rv_index], else_alias_map);
          auto else_store = op_registry.Create("tile.store", {else_tile, offsets, in_param}, span);
          auto else_store_var =
              std::make_shared<Var>(MakeStoreResultName(cand.return_index), else_store->GetType(), span);

          new_then_yield_values[cand.ifstmt_rv_index] = then_store_var;
          new_else_yield_values[cand.ifstmt_rv_index] = else_store_var;
          new_rv[cand.ifstmt_rv_index] = std::make_shared<Var>(
              last_if_stmt->return_vars_[cand.ifstmt_rv_index]->name_hint_, then_store->GetType(), span);

          InsertBeforeYield(then_stmts, then_yield_idx,
                            std::make_shared<AssignStmt>(then_store_var, then_store, span));
          ++then_yield_idx;
          InsertBeforeYield(else_stmts, else_yield_idx,
                            std::make_shared<AssignStmt>(else_store_var, else_store, span));
          ++else_yield_idx;

          merged_return_indices.insert(cand.return_index);
        }

        then_stmts[then_yield_idx] = std::make_shared<YieldStmt>(new_then_yield_values, then_yield->span_);
        else_stmts[else_yield_idx] = std::make_shared<YieldStmt>(new_else_yield_values, else_yield->span_);

        // Remove alias assignments for yield values replaced by store sinking,
        // but only if the alias var has no remaining uses in the branch.
        auto remove_dead_aliases = [&sink_candidates](std::vector<StmtPtr>& stmts, const YieldStmtPtr& yield,
                                                      const TileAliasMap& alias_map) {
          std::unordered_set<const Var*> candidates;
          for (const auto& cand : sink_candidates) {
            auto old_var = As<Var>(yield->value_[cand.ifstmt_rv_index]);
            if (old_var && alias_map.count(old_var.get())) {
              candidates.insert(old_var.get());
            }
          }
          if (candidates.empty()) return;

          // Full IR walk to collect all var uses (handles nested control flow).
          // Only remove aliases whose var is truly unused.
          var_collectors::VarDefUseCollector collector;
          for (const auto& s : stmts) collector.VisitStmt(s);
          const auto& used = collector.var_uses;
          stmts.erase(std::remove_if(stmts.begin(), stmts.end(),
                                     [&candidates, &used](const StmtPtr& s) {
                                       auto assign = As<AssignStmt>(s);
                                       return assign && candidates.count(assign->var_.get()) > 0 &&
                                              used.count(assign->var_.get()) == 0;
                                     }),
                      stmts.end());
        };
        remove_dead_aliases(then_stmts, then_yield, then_alias_map);
        remove_dead_aliases(else_stmts, else_yield, else_alias_map);

        auto new_then_body = SeqStmts::Flatten(std::move(then_stmts), last_if_stmt->then_body_->span_);
        auto new_else_body = SeqStmts::Flatten(std::move(else_stmts), (*last_if_stmt->else_body_)->span_);
        auto new_if_stmt =
            std::make_shared<IfStmt>(last_if_stmt->condition_, new_then_body, new_else_body, new_rv, span);
        new_stmts[last_if_index] = new_if_stmt;

        // Update mutator mappings so Phase 3b sees the new TensorType return vars.
        for (const auto& cand : sink_candidates) {
          const auto& new_var = new_rv[cand.ifstmt_rv_index];
          mutator.AddMapping(last_if_stmt->return_vars_[cand.ifstmt_rv_index].get(), new_var);
          auto orig_ret_var = As<Var>(return_stmt->value_[cand.return_index]);
          if (orig_ret_var) mutator.AddMapping(orig_ret_var.get(), new_var);
        }
      }
    }

    // Phase 3b: Process each return value
    for (size_t i = 0; i < return_stmt->value_.size(); ++i) {
      auto ret_expr = mutator.VisitExpr(return_stmt->value_[i]);

      // If this return was merged into an In param via IfStmt store sinking,
      // no new Out param needed — just pass through the TensorType result.
      if (merged_return_indices.count(i) > 0) {
        new_return_types.push_back(ret_expr->GetType());
        new_return_exprs.push_back(ret_expr);
        continue;
      }

      // Check if the return value is a tile (was converted from tensor)
      auto tile_type = As<TileType>(ret_expr->GetType());
      if (tile_type) {
        // Find the original tensor type from the function's return types
        auto orig_tensor_type = As<TensorType>(func->return_types_[i]);
        INTERNAL_CHECK(orig_tensor_type)
            << "Internal error: return type " << i << " should be TensorType but got "
            << func->return_types_[i]->TypeName();

        // Check if this return value has an iter-arg mapping: store to the
        // existing In param (auto-promoted to InOut) instead of adding a new Out param.
        auto map_it = iter_arg_mapping.find(i);
        if (map_it != iter_arg_mapping.end()) {
          size_t arg_idx = map_it->second;
          INTERNAL_CHECK(arg_idx < new_params.size())
              << "Internal error: iter-arg mapping arg_idx " << arg_idx << " exceeds param count "
              << new_params.size();

          auto in_param = new_params[arg_idx];
          auto offsets = MakeZeroOffsets(orig_tensor_type->shape_.size(), span);
          auto store_call = op_registry.Create("tile.store", {ret_expr, offsets, in_param}, span);
          auto store_var = std::make_shared<Var>(MakeStoreResultName(i), store_call->GetType(), span);
          new_stmts.push_back(std::make_shared<AssignStmt>(store_var, store_call, span));
          new_return_types.push_back(store_call->GetType());
          new_return_exprs.push_back(store_var);
          continue;
        }

        // Add output tensor parameter
        std::string out_name = MakeOutParamName(num_added_outputs);
        auto out_param = std::make_shared<Var>(out_name, orig_tensor_type, span);
        new_params.push_back(out_param);
        new_param_directions.push_back(ParamDirection::Out);

        if (auto loop_rewrite = RewriteReturnedAssembleLoopToStore(new_stmts, ret_expr, out_param,
                                                                   orig_tensor_type, op_registry)) {
          new_stmts[loop_rewrite->stmt_index] = loop_rewrite->new_for_stmt;
          if (loop_rewrite->dead_init_stmt_index.has_value()) {
            new_stmts.erase(new_stmts.begin() +
                            static_cast<std::ptrdiff_t>(*loop_rewrite->dead_init_stmt_index));
          }
          new_return_types.push_back(orig_tensor_type);
          new_return_exprs.push_back(loop_rewrite->new_return_var);
          ++num_added_outputs;
          continue;
        }

        // Insert tile.store(tile, zeros, out_param)
        auto offsets = MakeZeroOffsets(tile_type->shape_.size(), span);
        auto store_call = op_registry.Create("tile.store", {ret_expr, offsets, out_param}, span);

        auto store_var =
            std::make_shared<Var>(MakeStoreResultName(num_added_outputs), store_call->GetType(), span);
        new_stmts.push_back(std::make_shared<AssignStmt>(store_var, store_call, span));

        new_return_types.push_back(store_call->GetType());
        new_return_exprs.push_back(store_var);
        ++num_added_outputs;
      } else {
        // Non-tile return values pass through
        new_return_types.push_back(ret_expr->GetType());
        new_return_exprs.push_back(ret_expr);
      }
    }

    // Build new return statement
    new_stmts.push_back(std::make_shared<ReturnStmt>(new_return_exprs, return_stmt->span_));
  } else {
    // Void function (e.g. cross-core producer): add empty return
    INTERNAL_CHECK(func->return_types_.empty())
        << "Internal error: function '" << func->name_ << "' has no ReturnStmt but declares "
        << func->return_types_.size() << " return type(s) — possible malformed IR";
    new_stmts.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, span));
  }

  UpgradeWrittenTensorParamDirections(new_stmts, new_params, new_param_directions);

  auto new_body = SeqStmts::Flatten(std::move(new_stmts), span);
  auto new_func =
      std::make_shared<Function>(func->name_, new_params, new_param_directions, new_return_types, new_body,
                                 span, FunctionType::InCore, func->level_, func->role_, func->attrs_);

  return {new_func, num_added_outputs};
}

// ============================================================================
// CallSiteUpdateMutator: updates call sites in orchestration/opaque functions.
// For each call to a transformed InCore function, inserts tensor.create for
// output params and adds them as extra arguments.
// ============================================================================

class CallSiteUpdateMutator : public TypePropagatingMutator {
 public:
  CallSiteUpdateMutator(const std::unordered_map<std::string, size_t>& incore_added_outputs,
                        const std::unordered_map<std::string, FunctionPtr>& transformed_incore_funcs,
                        const OpRegistry& op_registry)
      : incore_added_outputs_(incore_added_outputs),
        transformed_incore_funcs_(transformed_incore_funcs),
        op_registry_(op_registry) {}

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = VisitExpr(op->value_);
    auto call = As<Call>(new_value);

    // Non-call or non-GlobalVar: propagate type change
    if (!call) return HandlePassThroughAssign(op, new_value);
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!global_var) return HandlePassThroughAssign(op, new_value);

    // Not a transformed InCore function: propagate type change
    auto it = incore_added_outputs_.find(global_var->name_);
    if (it == incore_added_outputs_.end() || it->second == 0) {
      return HandlePassThroughAssign(op, new_value);
    }

    // This call targets a transformed InCore function — add output tensor args
    size_t num_outputs = it->second;
    auto incore_func_it = transformed_incore_funcs_.find(global_var->name_);
    INTERNAL_CHECK(incore_func_it != transformed_incore_funcs_.end())
        << "Internal error: transformed InCore function not found: " << global_var->name_;
    const auto& incore_func = incore_func_it->second;

    std::vector<StmtPtr> stmts;
    std::vector<ExprPtr> extra_args;
    size_t orig_param_count = incore_func->params_.size() - num_outputs;

    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& out_param = incore_func->params_[orig_param_count + i];
      auto out_tensor_type = As<TensorType>(out_param->GetType());
      INTERNAL_CHECK(out_tensor_type) << "Internal error: output param is not TensorType";

      auto shape_tuple = MakeShapeTuple(out_tensor_type->shape_, call->span_);
      TensorLayout layout = out_tensor_type->tensor_view_.has_value() ? out_tensor_type->tensor_view_->layout
                                                                      : TensorLayout::ND;
      std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", out_tensor_type->dtype_},
                                                                     {"layout", layout}};
      auto create_call = op_registry_.Create("tensor.create", {shape_tuple}, create_kwargs, call->span_);

      auto out_var = std::make_shared<Var>(MakeOutParamName(i), create_call->GetType(), call->span_);
      stmts.push_back(std::make_shared<AssignStmt>(out_var, create_call, op->span_));
      extra_args.push_back(out_var);
    }

    std::vector<ExprPtr> new_args = call->args_;
    new_args.insert(new_args.end(), extra_args.begin(), extra_args.end());

    TypePtr new_return_type;
    if (incore_func->return_types_.empty()) {
      new_return_type = nullptr;
    } else if (incore_func->return_types_.size() == 1) {
      new_return_type = incore_func->return_types_[0];
    } else {
      new_return_type = std::make_shared<TupleType>(incore_func->return_types_);
    }

    std::shared_ptr<Call> new_call;
    if (new_return_type) {
      new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, new_return_type, call->span_);
    } else {
      new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->span_);
    }

    auto new_assign_var = std::make_shared<Var>(op->var_->name_hint_, new_return_type, op->var_->span_);
    stmts.push_back(std::make_shared<AssignStmt>(new_assign_var, new_call, op->span_));
    var_remap_[op->var_.get()] = new_assign_var;

    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

 private:
  const std::unordered_map<std::string, size_t>& incore_added_outputs_;
  const std::unordered_map<std::string, FunctionPtr>& transformed_incore_funcs_;
  const OpRegistry& op_registry_;
};

/**
 * @brief Update call sites in orchestration/opaque functions.
 */
FunctionPtr UpdateCallSites(const FunctionPtr& func,
                            const std::unordered_map<std::string, size_t>& incore_added_outputs,
                            const std::unordered_map<std::string, FunctionPtr>& transformed_incore_funcs) {
  CallSiteUpdateMutator mutator(incore_added_outputs, transformed_incore_funcs, OpRegistry::GetInstance());
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_, func->return_types_,
                                    new_body, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_);
}

}  // namespace

namespace pass {

Pass ConvertTensorToTileOps() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Phase 0: Analyze iter-arg mappings from orchestration call sites
    std::vector<FunctionPtr> all_funcs;
    all_funcs.reserve(program->functions_.size());
    for (const auto& [gvar, func] : program->functions_) {
      all_funcs.push_back(func);
    }
    auto iter_arg_mappings = AnalyzeIterArgMappings(all_funcs);

    // Phase 1: Transform InCore functions
    std::unordered_map<std::string, size_t> incore_added_outputs;
    std::unordered_map<std::string, FunctionPtr> transformed_incore_funcs;
    std::vector<FunctionPtr> functions_phase1;
    const IterArgMapping empty_mapping;

    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        auto mapping_it = iter_arg_mappings.find(func->name_);
        const auto& mapping = (mapping_it != iter_arg_mappings.end()) ? mapping_it->second : empty_mapping;
        auto result = TransformIncoreFunction(func, mapping);
        incore_added_outputs[func->name_] = result.num_added_outputs;
        transformed_incore_funcs[func->name_] = result.func;
        functions_phase1.push_back(result.func);
      } else {
        functions_phase1.push_back(func);
      }
    }

    // Phase 2: Update call sites in non-InCore functions
    std::vector<FunctionPtr> functions_phase2;
    for (const auto& func : functions_phase1) {
      if (func->func_type_ != FunctionType::InCore) {
        functions_phase2.push_back(UpdateCallSites(func, incore_added_outputs, transformed_incore_funcs));
      } else {
        functions_phase2.push_back(func);
      }
    }

    return std::make_shared<Program>(functions_phase2, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "ConvertTensorToTileOps", kConvertTensorToTileOpsProperties);
}

}  // namespace pass

// ============================================================================
// IncoreTileOps property verifier
// ============================================================================

namespace {

/**
 * @brief Checks that InCore functions have no TensorType ops (only tile ops).
 */
class IncoreTileOpsVerifier : public IRVisitor {
 public:
  explicit IncoreTileOpsVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->value_)) {
      CheckTensorOp(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (!op) return;
    if (auto call = As<Call>(op->expr_)) {
      CheckTensorOp(call, op->span_);
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void CheckTensorOp(const std::shared_ptr<const Call>& call, const Span& span) {
    // Op calls use plain Op (not GlobalVar); GlobalVar is for function calls
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (global_var) return;

    // Use op category from OpRegistry instead of brittle string prefix check
    auto& op_registry = OpRegistry::GetInstance();
    if (!op_registry.IsRegistered(call->op_->name_)) return;

    const auto& entry = op_registry.GetEntry(call->op_->name_);
    if (entry.GetOpCategory() == "TensorOp" &&
        OpConversionRegistry::GetInstance().HasConversion(call->op_->name_)) {
      // tensor.read/tensor.write on a gm_tensor (TensorType input) intentionally stays unconverted
      if ((call->op_->name_ == "tensor.read" || call->op_->name_ == "tensor.write") && !call->args_.empty() &&
          As<TensorType>(call->args_[0]->GetType())) {
        return;
      }

      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "IncoreTileOps", 0,
          "Tensor op '" + call->op_->name_ + "' found in InCore function (should have been converted)", span);
    }
  }

  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class IncoreTileOpsPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "IncoreTileOps"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      if (func->func_type_ != FunctionType::InCore) continue;
      IncoreTileOpsVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateIncoreTileOpsPropertyVerifier() {
  return std::make_shared<IncoreTileOpsPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
