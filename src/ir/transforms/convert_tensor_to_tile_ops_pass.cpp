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
#include <cctype>
#include <cstddef>
#include <cstdint>
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
#include "pypto/ir/memory_space.h"
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
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/buffer_root_collector.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/narrow_loop_carry.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/tile_conversion_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

using tile_conversion_utils::MakeShapeTuple;
using tile_conversion_utils::MakeZeroOffsets;
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

CallPtr MarkCompilerMatBridge(const CallPtr& call, MemorySpace space) {
  if (!call || space != MemorySpace::Mat) return call;
  auto marked = MutableCopy(call);
  marked->attrs_.emplace_back(kCompilerTensorToTileMatBridgeAttr, true);
  return marked;
}

bool IsPassthroughTensorOp(const CallPtr& call) {
  return IsOp(call, "tensor.dim") || IsOp(call, "tensor.view");
}

/// Declared GM cache policy per source tensor, keyed by the param Var the
/// declaration resolved to. ``CachePolicy`` stored as ``int``, the type the
/// ``tile.load`` ``cache`` kwarg is registered with.
using CachePolicyByParam = std::unordered_map<const Var*, int>;

/**
 * @brief Resolve an InCore function's ``cache_policy`` attr to its param Vars.
 *
 * ``OutlineIncoreScopes`` (pass 8) records ``pl.set_cache_policy`` declarations
 * as (param index, policy) pairs, because at that point the param Vars are
 * freshly minted. Here the indices are turned back into Var identities — the
 * form every load site below matches its source arg against — and the attr is
 * erased on the way out (see ``EraseCachePolicyAttr``): param indices are only
 * valid across passes 8..10, since later passes both append to and prepend onto
 * param lists.
 */
CachePolicyByParam BuildCachePolicyByParam(const FunctionPtr& func) {
  CachePolicyByParam policies;
  auto indices = func->GetAttr<std::vector<std::pair<int32_t, int>>>(kAttrCachePolicyParams);
  policies.reserve(indices.size());
  for (const auto& [idx, policy] : indices) {
    INTERNAL_CHECK_SPAN(idx >= 0 && static_cast<size_t>(idx) < func->params_.size(), func->span_)
        << "Internal error: cache_policy param index " << idx << " out of range for function '" << func->name_
        << "' with " << func->params_.size() << " param(s)";
    // insert_or_assign, not emplace: a tensor declared twice takes its last
    // declaration, deterministically (the attr is sorted by index).
    policies.insert_or_assign(func->params_[static_cast<size_t>(idx)].get(), policy);
  }
  return policies;
}

/// Drop the consumed ``cache_policy`` attr. Nothing downstream may see it —
/// its param indices go stale the moment a later pass grows the param list.
std::vector<std::pair<std::string, std::any>> EraseCachePolicyAttr(
    const std::vector<std::pair<std::string, std::any>>& attrs) {
  std::vector<std::pair<std::string, std::any>> kept;
  kept.reserve(attrs.size());
  for (const auto& kv : attrs) {
    if (kv.first == kAttrCachePolicyParams) continue;
    kept.push_back(kv);
  }
  return kept;
}

/// Longest ``IterArg`` init chain followed when resolving a load source. Loop
/// nesting is the real bound (single digits); this only stops malformed IR from
/// spinning.
constexpr int kMaxIterArgInitChain = 64;

/**
 * @brief Resolve a load source to the function parameter it ultimately reads.
 *
 * A loop-carried tensor reaches its load as an ``IterArg``, not as the param
 * ``Var`` the declaration named -- ``IterArg`` is its own ``ObjectKind``, so
 * ``AsVarLike`` hands it back as itself and a param-keyed lookup misses. Follow
 * ``initValue_`` back to the root first, mirroring how
 * ``PTOCodegen::TryGetTensorView`` resolves the same shape. Returns the
 * innermost resolvable Var-like, which for a non-carried source is the source
 * itself.
 */
VarPtr ResolveToRootParam(const ExprPtr& src) {
  auto var = AsVarLike(src);
  for (int depth = 0; var && depth < kMaxIterArgInitChain; ++depth) {
    auto iter_arg = As<IterArg>(var);
    if (!iter_arg) return var;
    auto init = AsVarLike(iter_arg->initValue_);
    if (!init) return var;
    var = init;
  }
  return var;
}

/**
 * @brief Add the declared ``cache`` kwarg for a synthesised ``tile.load``.
 *
 * No-op unless ``src`` is a param carrying a declaration. An explicit
 * ``cache=`` already in ``kwargs`` always wins (precedence: per-access kwarg,
 * then the scope declaration, then ``CachePolicy::kDefault``).
 */
void AppendCachePolicyKwarg(const ExprPtr& src, const CachePolicyByParam& policies,
                            std::vector<std::pair<std::string, std::any>>* kwargs) {
  if (policies.empty()) return;
  auto var = ResolveToRootParam(src);
  if (!var) return;
  auto it = policies.find(var.get());
  if (it == policies.end()) return;
  const bool stated_explicitly =
      std::any_of(kwargs->begin(), kwargs->end(), [](const auto& kv) { return kv.first == "cache"; });
  if (stated_explicitly) return;
  kwargs->emplace_back("cache", it->second);
}

/**
 * @brief Stamp the declared policy onto a ``tile.load`` already in the body.
 *
 * A hand-written (or earlier-pass) load of a declared tensor must honour the
 * declaration exactly as a synthesised one does, unless it states its own
 * ``cache=``. Returns nullptr when nothing changes, so callers keep their
 * copy-on-write short-circuit. Only ``Call`` is considered: a ``Submit``
 * launches a task through a ``GlobalVar`` callee and can never carry a tile op.
 */
CallPtr StampCachePolicyOnLoad(const CallPtr& call, const CachePolicyByParam& policies) {
  if (policies.empty() || !IsOp(call, "tile.load") || call->args_.empty()) return nullptr;
  auto kwargs = call->kwargs_;
  AppendCachePolicyKwarg(call->args_[0], policies, &kwargs);
  // Unchanged when the source carries no declaration, or when the load already
  // states its own ``cache=``.
  if (kwargs.size() == call->kwargs_.size()) return nullptr;
  auto stamped = MutableCopy(call);
  stamped->kwargs_ = std::move(kwargs);
  return stamped;
}

void CheckReinterpretViewIncoreLayout(const CallPtr& call) {
  if (!IsOp(call, "tensor.reinterpret_view")) return;

  INTERNAL_CHECK_SPAN(!call->args_.empty(), call->span_)
      << "Internal error: tensor.reinterpret_view reached conversion without a data argument";
  auto source_type = As<TensorType>(call->args_[0]->GetType());
  INTERNAL_CHECK_SPAN(source_type, call->span_)
      << "Internal error: tensor.reinterpret_view source must be TensorType before tile conversion";
  const TensorLayout source_layout =
      source_type->tensor_view_.has_value() ? source_type->tensor_view_->layout : TensorLayout::ND;
  CHECK_SPAN(source_layout == TensorLayout::ND, call->span_)
      << "tensor.reinterpret_view in an InCore function currently supports only packed ND tensors; "
         "DN layout changes which logical axis is physically contiguous, and that information cannot "
         "yet be preserved by tensor-to-tile lowering";
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
    const auto* conv_entry = (call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_))
                                 ? conv_registry_.Lookup(call->op_->name_)
                                 : nullptr;
    if (conv_entry) {
      // Skip ops whose inputs are handled by their own converter (self-loading):
      // they create loads with specific offsets/spaces, so Phase-1 default Vec loads
      // would be redundant or wrong.
      static const std::unordered_set<std::string> kSelfLoadingOps = {
          "tensor.slice",        "tensor.assemble",     "tensor.read",
          "tensor.write",        "tensor.expand_clone", "tensor.gather",
          "tensor.paged_gather", "tensor.create_l1",    "tensor.gather_row"};
      if (kSelfLoadingOps.count(call->op_->name_)) {
        IRVisitor::VisitStmt_(op);
        return;
      }
      // Per-arg exclusion: args covered by input_reqs are handled by framework auto-bridging.
      // Other args (e.g. matmul_acc's acc, which has no input_req) still need Phase-1 loads
      // so they reach the converter as TileType.
      for (size_t i = 0; i < call->args_.size(); ++i) {
        if (conv_entry->input_reqs.count(i)) continue;
        const auto& arg = call->args_[i];
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
// Consumer-driven memory space collection.
//
// Pre-scans the function body to build a map from variables to the memory
// space their downstream consumers require (as declared via InputSpaceReq
// in OpConversionRegistry).  This lets load-like ops (tensor.slice on
// TensorType) produce the right space directly, avoiding a redundant
// load(Vec) + move(Mat) sequence.
// ============================================================================

/**
 * @brief Resolved consumer memory space requirement for a variable.
 */
struct ConsumerSpaceReq {
  MemorySpace space;  ///< Required memory space. The consumer-driven load is always
                      ///< natural; a transposed (b_trans/a_trans) operand is realised
                      ///< by a zero-copy tile.transpose_view in BridgeInputSpaces.
};

/**
 * @brief Visitor that collects consumer memory space requirements for variables.
 *
 * For each op with declared InputSpaceReq, records which variables need which
 * memory space.  Replaces the special-purpose MatmulSlicePatternCollector with
 * a general mechanism driven entirely by registered converter metadata.
 */
class ConsumerSpaceCollector : public IRVisitor {
 public:
  explicit ConsumerSpaceCollector(const OpConversionRegistry& registry) : registry_(registry) {}

  [[nodiscard]] std::optional<ConsumerSpaceReq> GetConsumerReq(const Var* var) const {
    auto it = consumer_reqs_.find(var);
    return it != consumer_reqs_.end() ? std::optional{it->second} : std::nullopt;
  }

  /// Second phase: propagate collected requirements backward through
  ///   (a) ops registered with `set_output_memory_inherit_input()` — output
  ///       memory equals the first tile/tensor-typed input's, so a demand on
  ///       the output is equivalently a demand on that input, and
  ///   (b) plain SSA aliases `y = x` where both sides are shaped Vars (the
  ///       parser elides no-op `tensor.fillpad(pad=zero)` into this form when
  ///       the input's valid_shape already zeroes the pad region).
  ///
  /// Edges are recorded in program order during the forward visit. Since the
  /// inherit-input and alias relations are acyclic and flow strictly backward
  /// (output/dst defined after input/src), a single reverse-order sweep
  /// reaches the fixed point in O(N). Total pass cost stays O(N log N).
  void PropagateThroughInheritInputOps() {
    for (auto it = propagation_edges_.rbegin(); it != propagation_edges_.rend(); ++it) {
      const auto& [dst, src] = *it;
      auto out_it = consumer_reqs_.find(dst);
      if (out_it == consumer_reqs_.end()) continue;
      const auto& req = out_it->second;
      auto [ins_it, inserted] = consumer_reqs_.try_emplace(src, req);
      if (!inserted && ins_it->second.space == MemorySpace::Vec && req.space != MemorySpace::Vec) {
        ins_it->second = req;
      }
    }
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op) return;
    auto is_shaped = [](const TypePtr& t) { return As<TensorType>(t) || As<TileType>(t); };

    // Record a propagation edge `dst -> src` in program order when the RHS is
    // either a plain SSA alias (both sides shaped) or an inherit-input Call
    // (first shaped input carries the memory-space relation). The reverse walk
    // in phase 2 then resolves all back-propagation in a single pass.
    if (op->var_ && is_shaped(op->var_->GetType())) {
      if (auto src_var = As<Var>(op->value_); src_var && is_shaped(src_var->GetType())) {
        propagation_edges_.emplace_back(op->var_.get(), src_var.get());
      } else if (auto call = As<Call>(op->value_);
                 call && !std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
        auto& op_reg = OpRegistry::GetInstance();
        if (op_reg.IsRegistered(call->op_->name_) &&
            op_reg.GetEntry(call->op_->name_).OutputMemoryInheritsInput()) {
          for (const auto& arg : call->args_) {
            if (auto arg_var = As<Var>(arg); arg_var && is_shaped(arg_var->GetType())) {
              propagation_edges_.emplace_back(op->var_.get(), arg_var.get());
              break;
            }
          }
        }
      }
    }

    auto call = As<Call>(op->value_);
    if (!call || std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    const auto* entry = registry_.Lookup(call->op_->name_);
    if (!entry || entry->input_reqs.empty()) {
      IRVisitor::VisitStmt_(op);
      return;
    }

    // Both 2D tile.matmul and ND tile.batch_matmul realise a transposed operand
    // as a zero-copy tile.transpose_view added in BridgeInputSpaces (issues #1776
    // / ND extension): the consumer-driven load is ALWAYS natural, and the view
    // supplies the transpose. (No more transpose-at-load baking.)
    for (const auto& [idx, req] : entry->input_reqs) {
      if (idx >= call->args_.size()) continue;
      if (auto var = As<Var>(call->args_[idx])) {
        // Prioritize non-Vec spaces: if an existing requirement is the default Vec but this
        // consumer needs a specialized space (Mat/Left/Right/Acc/Bias), override it so the
        // load-like producer can emit the specialized space directly.
        auto [it, inserted] = consumer_reqs_.try_emplace(var.get(), ConsumerSpaceReq{req.space});
        if (!inserted && it->second.space == MemorySpace::Vec && req.space != MemorySpace::Vec) {
          it->second = ConsumerSpaceReq{req.space};
        }
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  const OpConversionRegistry& registry_;
  std::unordered_map<const Var*, ConsumerSpaceReq> consumer_reqs_;
  // `dst -> src` edges captured in program order — covers both Call-valued
  // inherit-input ops and plain SSA aliases. A single reverse-order walk in
  // PropagateThroughInheritInputOps reaches the fixed point.
  std::vector<std::pair<const Var*, const Var*>> propagation_edges_;
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
          auto copy = MutableCopy(new_for);
          copy->return_vars_ = std::move(new_rv);
          return copy;
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
          auto result = MutableCopy(new_while);
          result->return_vars_ = std::move(new_rv);
          return StmtPtr(result);
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

    auto new_if = MutableCopy(op);
    new_if->condition_ = new_condition;
    new_if->then_body_ = new_then_body;
    new_if->else_body_ = new_else_body;
    new_if->return_vars_ = std::move(new_return_vars);
    return new_if;
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
      auto result = MutableCopy(op);
      result->var_ = new_var;
      result->value_ = new_value;
      return result;
    }
    // Value changed but type did not — keep original Var, clear any stale remap.
    var_remap_.erase(op->var_.get());
    auto result = MutableCopy(op);
    result->value_ = new_value;
    return result;
  }

  /// Keep a Var shared_ptr alive for the lifetime of this mutator.
  ///
  /// ``var_remap_`` is keyed by raw ``const Expr*`` pointers (inherited from
  /// IRMutator). Converters create temporary Vars (e.g. ``paged_gather`` builds
  /// per-row scalars like ``pg_idx`` in its loop body); when a converter's
  /// ``AssignStmt`` is replaced during conversion, the old Var is freed and the
  /// allocator can hand its address to a *later* Var. A stale ``var_remap_``
  /// entry keyed on the freed address would then mis-resolve the new Var,
  /// silently rewriting an unrelated value (observed: a matmul result resolving
  /// to a freed ``pg_idx`` scalar). Retaining every mapped-from Var prevents the
  /// address reuse that triggers the collision.
  void RetainVar(const ExprPtr& v) {
    if (v) retained_vars_.push_back(v);
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

  /// Vars kept alive for the pass lifetime — see RetainVar.
  std::vector<ExprPtr> retained_vars_;
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
                      const ConsumerSpaceCollector& consumer_collector, CachePolicyByParam cache_policies)
      : conv_registry_(conv_registry),
        op_registry_(op_registry),
        consumer_collector_(consumer_collector),
        cache_policies_(std::move(cache_policies)) {}

 protected:
  /// Honour a ``pl.set_cache_policy`` declaration on a ``tile.load`` that was
  /// already in the body (user-written, or produced by an earlier pass) rather
  /// than synthesised here. Hooked on the generic Call visit so the loads a
  /// converter emits in its own prologue are covered too.
  ExprPtr VisitExpr_(const CallPtr& op) override {
    // Qualified with IRMutator: TypePropagatingMutator declares only the
    // IterArg overload, which hides the base Call one from name lookup.
    auto visited = IRMutator::VisitExpr_(op);
    auto call = As<Call>(visited);
    if (!call) return visited;
    auto stamped = StampCachePolicyOnLoad(call, cache_policies_);
    return stamped ? ExprPtr(stamped) : visited;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Pin this Var's address for the pass so a freed-then-reused address cannot
    // alias a stale var_remap_ entry (see TypePropagatingMutator::RetainVar).
    RetainVar(op->var_);
    CheckReinterpretViewIncoreLayout(As<Call>(op->value_));
    auto new_value = VisitExpr(op->value_);
    auto call = As<Call>(new_value);

    // Non-call values: propagate type change
    if (!call) return HandlePassThroughAssign(op, new_value);

    // Function calls (GlobalVar) pass through — only process op calls
    if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
      LOG_DEBUG << "[TensorToTileMutator] Skipping GlobalVar call: " << call->op_->name_;
      return HandlePassThroughAssign(op, new_value);
    }

    const auto* entry = conv_registry_.Lookup(call->op_->name_);
    if (!entry) {
      if (IsOp(call, "tensor.view")) {
        CHECK_SPAN(!call->args_.empty() && AsTensorTypeLike(call->args_[0]->GetType()), call->span_)
            << "tensor.view in an InCore function requires a GM Tensor input that remains tensor-like "
               "through ConvertTensorToTileOps; viewing the result of an op lowered to Tile is not supported";
      }
      // Verify unregistered TensorOps are expected passthroughs
      if (op_registry_.IsRegistered(call->op_->name_)) {
        const auto& op_entry = op_registry_.GetEntry(call->op_->name_);
        INTERNAL_CHECK_SPAN(op_entry.GetOpCategory() != "TensorOp" || IsPassthroughTensorOp(call),
                            call->span_)
            << "TensorOp \"" << call->op_->name_ << "\" has no registered tile conversion. "
            << "Add a conversion in src/ir/transforms/op_conversion_registry.cpp.";
      }
      return HandlePassThroughAssign(op, new_value);
    }

    // Consumer-driven space override for load-like ops (e.g. tensor.slice
    // feeding into tensor.matmul → load to Mat instead of default Vec).
    if (IsOp(call, "tensor.slice")) {
      auto consumer_req = consumer_collector_.GetConsumerReq(op->var_.get());
      if (consumer_req) {
        auto override_load = HandleConsumerDrivenLoad(op, call, *consumer_req);
        if (override_load) return override_load;
      }
    }

    // Auto-bridge: load TensorType args to the memory space required by input_reqs
    auto [bridged_args, bridge_stmts] = BridgeInputSpaces(call, entry->input_reqs);

    // Run the converter with bridged args
    auto conv_result = entry->func(bridged_args, call->kwargs_, call->span_);

    // Collect all statements: bridge prologue + converter prologue + final assignment
    std::vector<StmtPtr> stmts;
    stmts.reserve(bridge_stmts.size() + conv_result.prologue.size() + 1);

    // Bridge statements are fully resolved — no recursive visit needed
    for (auto& s : bridge_stmts) stmts.push_back(std::move(s));

    // Converter prologue may contain nested tensor ops — recurse
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
    CheckReinterpretViewIncoreLayout(As<Call>(op->expr_));
    auto new_expr = VisitExpr(op->expr_);
    // Helper: return updated EvalStmt only when the expression actually changed.
    auto maybe_update = [&]() -> StmtPtr {
      if (new_expr.get() == op->expr_.get()) return StmtPtr(op);
      auto result = MutableCopy(op);
      result->expr_ = new_expr;
      return result;
    };

    auto call = As<Call>(new_expr);
    if (!call || std::dynamic_pointer_cast<const GlobalVar>(call->op_)) return maybe_update();

    const auto* entry = conv_registry_.Lookup(call->op_->name_);
    if (!entry) return maybe_update();

    auto [bridged_args, bridge_stmts] = BridgeInputSpaces(call, entry->input_reqs);
    auto conv_result = entry->func(bridged_args, call->kwargs_, call->span_);

    std::vector<StmtPtr> stmts;
    stmts.reserve(bridge_stmts.size() + conv_result.prologue.size() + 1);
    for (auto& s : bridge_stmts) stmts.push_back(std::move(s));
    for (auto& prologue_stmt : conv_result.prologue) {
      stmts.push_back(VisitStmt(prologue_stmt));
    }
    auto new_result = VisitExpr(conv_result.result);
    stmts.push_back(std::make_shared<EvalStmt>(new_result, op->span_));
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

 private:
  /// Handle tensor.slice whose consumer needs a specific memory space — produce tile.load with that space.
  StmtPtr HandleConsumerDrivenLoad(const AssignStmtPtr& op, const CallPtr& call,
                                   const ConsumerSpaceReq& req) {
    const auto& input = call->args_[0];
    auto tensor_type = AsTensorTypeLike(input->GetType());
    if (!tensor_type) return nullptr;

    const auto& shape_arg = call->args_[1];
    const auto& offset_arg = call->args_[2];
    auto shape_type = As<TupleType>(shape_arg->GetType());
    auto offset_type = As<TupleType>(offset_arg->GetType());
    INTERNAL_CHECK_SPAN(shape_type && offset_type, call->span_)
        << "Internal error: tensor.slice shape and offset must be tuples during conversion";
    auto full_shape = ExtractTupleElements(shape_arg, shape_type->types_.size());
    auto offsets = ExtractTupleElements(offset_arg, offset_type->types_.size());
    std::vector<ExprPtr> requested_valid;
    if (call->args_.size() >= 4) {
      auto valid_shape_tuple = As<MakeTuple>(call->args_[3]);
      INTERNAL_CHECK_SPAN(valid_shape_tuple, call->span_)
          << "Internal error: tensor.slice valid_shape must be a MakeTuple during conversion";
      requested_valid = valid_shape_tuple->elements_;
    }
    auto valid_shape = MakeShapeTuple(
        InferTensorSliceFullValidShape(*tensor_type, full_shape, offsets, requested_valid,
                                       GetKwargOr<bool>(call->kwargs_, "clamp", false), call->span_),
        call->span_);
    auto drop_dims = ParseSliceDropDims(call->args_.size() == 5 ? call->args_[4] : nullptr, full_shape,
                                        "tensor.slice conversion");
    CHECK_SPAN(drop_dims.size() < full_shape.size(), call->span_)
        << "tensor.slice conversion does not support rank-0 tiles; keep one unit axis or use tensor.read "
           "for a scalar result";

    // The consumer-driven load is always natural; a transposed (b_trans/a_trans)
    // operand gets a zero-copy tile.transpose_view at the matmul site instead.
    std::vector<std::pair<std::string, std::any>> load_kwargs = {{"target_memory", req.space}};
    AppendCachePolicyKwarg(input, cache_policies_, &load_kwargs);
    auto load_call =
        MarkCompilerMatBridge(op_registry_.Create("tile.load", {input, offset_arg, shape_arg, valid_shape},
                                                  load_kwargs, call->span_),
                              req.space);

    auto tile_name = MakeTileValueName(op->var_->name_hint_);
    if (drop_dims.empty()) {
      auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), op->var_->span_);
      var_remap_[op->var_.get()] = tile_var;
      auto result = MutableCopy(op);
      result->var_ = tile_var;
      result->value_ = load_call;
      return result;
    }

    auto loaded_var = std::make_shared<Var>(tile_name + "_full_rank", load_call->GetType(), op->var_->span_);
    auto reduced_shape = MakeShapeTuple(ApplyDropDims(full_shape, drop_dims), call->span_);
    auto reshape_call = op_registry_.Create("tile.reshape", {loaded_var, reduced_shape}, {}, call->span_);
    auto tile_var = std::make_shared<Var>(tile_name, reshape_call->GetType(), op->var_->span_);
    var_remap_[op->var_.get()] = tile_var;
    std::vector<StmtPtr> stmts = {
        std::make_shared<AssignStmt>(loaded_var, load_call, op->span_),
        std::make_shared<AssignStmt>(tile_var, reshape_call, op->span_),
    };
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  /// Auto-bridge TensorType args to the memory space required by input_reqs.
  /// Returns the (possibly modified) args and any load statements to prepend.
  std::pair<std::vector<ExprPtr>, std::vector<StmtPtr>> BridgeInputSpaces(
      const CallPtr& call, const std::unordered_map<size_t, InputSpaceReq>& input_reqs) {
    if (input_reqs.empty()) return {call->args_, {}};

    auto args = call->args_;
    std::vector<StmtPtr> stmts;

    // An operand of rank > 2 means this matmul lowers to tile.batch_matmul, not
    // tile.matmul (see the rank dispatch in op_conversion_registry.cpp).
    //
    // Both 2D tile.matmul AND ND tile.batch_matmul realise a transposed operand
    // as a zero-copy tile.transpose_view over ONE natural load/move (issues #1776
    // / ND extension). FlattenTileNdTo2D slices the whole transposed view per
    // batch; the tile-level (batch_)matmul carries no transpose semantic.

    // Emit a `tile.load` of `arg` (TensorType) into `space`, append its AssignStmt,
    // and return the bound load Var. The load is always natural; a transposed
    // operand is realised by a zero-copy tile.transpose_view on the result.
    auto emit_load = [&](const ExprPtr& arg, const TensorTypePtr& tensor_type, MemorySpace space,
                         size_t idx) -> VarPtr {
      auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), call->span_);
      auto shapes = MakeShapeTuple(tensor_type->shape_, call->span_);
      std::vector<std::pair<std::string, std::any>> load_kw = {{"target_memory", space}};
      AppendCachePolicyKwarg(arg, cache_policies_, &load_kw);
      auto load = MarkCompilerMatBridge(
          op_registry_.Create("tile.load", {arg, offsets, shapes, shapes}, load_kw, call->span_), space);
      std::string var_name;
      if (auto var = As<Var>(arg)) {
        auto space_str = MemorySpaceToString(space);
        std::transform(space_str.begin(), space_str.end(), space_str.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        var_name = var->name_hint_ + "_" + space_str;
      } else {
        var_name = "bridged_" + std::to_string(idx);
      }
      auto load_var = std::make_shared<Var>(var_name, load->GetType(), call->span_);
      stmts.push_back(std::make_shared<AssignStmt>(load_var, load, call->span_));
      return load_var;
    };

    // Iterate in sorted index order to produce deterministic statement ordering.
    std::vector<size_t> sorted_indices;
    sorted_indices.reserve(input_reqs.size());
    for (const auto& [idx, _] : input_reqs) sorted_indices.push_back(idx);
    std::sort(sorted_indices.begin(), sorted_indices.end());

    // Zero-copy reinterpret a Mat-resident tile as its transpose (NZ<->ZN),
    // aliasing the SAME L1 buffer (issue #1776). Valid only for Mat tiles.
    auto emit_view = [&](const VarPtr& v) -> VarPtr {
      auto view = op_registry_.Create("tile.transpose_view", {v}, {}, call->span_);
      auto view_var = std::make_shared<Var>(v->name_hint_ + "_t", view->GetType(), call->span_);
      stmts.push_back(std::make_shared<AssignStmt>(view_var, view, call->span_));
      return view_var;
    };

    // Bridge a non-Mat (e.g. Vec) tile to Mat in its NATURAL orientation via a
    // V2C move. transpose_view needs a Mat-resident tile (a col_major Vec tile
    // cannot be pushed V2C — a2a3 TPUSH transfers only ND/NZ tiles), so a Vec
    // compute result feeding a b_trans matmul (mixed kernel) is moved to Mat with
    // its original shape first, then reinterpreted as its transpose on the Mat side.
    auto emit_move_to_mat = [&](const VarPtr& v) -> VarPtr {
      std::vector<std::pair<std::string, std::any>> move_kw = {{"target_memory", MemorySpace::Mat}};
      auto mv = op_registry_.Create("tile.move", {v}, move_kw, call->span_);
      auto mv_var = std::make_shared<Var>(v->name_hint_ + "_mat", mv->GetType(), call->span_);
      stmts.push_back(std::make_shared<AssignStmt>(mv_var, mv, call->span_));
      return mv_var;
    };

    for (size_t idx : sorted_indices) {
      const auto& req = input_reqs.at(idx);
      if (idx >= args.size()) continue;
      const bool use_view = req.trans_kwarg ? call->GetKwarg<bool>(*req.trans_kwarg, false) : false;
      auto tensor_type = As<TensorType>(args[idx]->GetType());

      if (tensor_type) {
        // GM operand: load NATURAL (2D and ND alike), then reinterpret as its
        // transpose with a zero-copy view when b_trans/a_trans.
        auto loaded = emit_load(args[idx], tensor_type, req.space, idx);
        args[idx] = use_view ? emit_view(loaded) : loaded;
        continue;
      }

      // Tile operand. Only a transposed operand needs rewriting; a non-transposed
      // tile passes through.
      if (!use_view) continue;
      auto var = As<Var>(args[idx]);
      if (!var) continue;  // non-Var tile expr: leave as-is
      auto tile_ty = As<TileType>(var->GetType());
      const bool mat_resident =
          tile_ty && tile_ty->memory_space_.value_or(MemorySpace::Vec) == MemorySpace::Mat;
      // Mat-resident operand (e.g. a consumer-driven Mat load): zero-copy view.
      // Non-Mat operand (Vec compute result, mixed kernel): move to Mat in its
      // natural shape first, then view the Mat tile.
      args[idx] = mat_resident ? emit_view(var) : emit_view(emit_move_to_mat(var));
    }

    return {std::move(args), std::move(stmts)};
  }

  const OpConversionRegistry& conv_registry_;
  const OpRegistry& op_registry_;
  const ConsumerSpaceCollector& consumer_collector_;
  /// Declared GM cache policies of this function's params (empty when the
  /// function carries no ``pl.set_cache_policy`` declaration).
  const CachePolicyByParam cache_policies_;
};

bool ExprUsesVar(const ExprPtr& expr, const Var* target) {
  if (!expr || !target) return false;
  var_collectors::VarDefUseCollector collector;
  collector.VisitExpr(expr);
  return collector.var_uses.count(target) > 0;
}

bool StmtUsesVar(const StmtPtr& stmt, const Var* target) {
  if (!stmt || !target) return false;
  var_collectors::VarDefUseCollector collector;
  collector.VisitStmt(stmt);
  return collector.var_uses.count(target) > 0;
}

struct FullTensorScalarUpdateCandidate {
  size_t assemble_index;
  VarPtr target;
  VarPtr staging;
  AssignStmtPtr full_stmt;
  AssignStmtPtr assemble_stmt;
  bool safe = true;
  size_t scalar_write_count = 0;
};

bool IsZeroOffsetTuple(const ExprPtr& expr, size_t rank) {
  auto offsets = As<MakeTuple>(expr);
  if (!offsets || offsets->elements_.size() != rank) return false;
  return std::all_of(offsets->elements_.begin(), offsets->elements_.end(), [](const ExprPtr& offset) {
    auto value = As<ConstInt>(offset);
    return value && value->value_ == 0;
  });
}

/** @brief Analyze all full-init candidates in one O(N) traversal. */
class StagedScalarUseAnalyzer : public IRVisitor {
 public:
  explicit StagedScalarUseAnalyzer(std::vector<FullTensorScalarUpdateCandidate>* candidates)
      : candidates_(candidates) {
    for (size_t i = 0; i < candidates_->size(); ++i) {
      const auto& candidate = (*candidates_)[i];
      candidate_by_target_[candidate.target.get()] = i;
      candidate_by_staging_[candidate.staging.get()] = i;
      definition_stmts_.insert(candidate.full_stmt.get());
      definition_stmts_.insert(candidate.assemble_stmt.get());
    }
  }

 protected:
  void VisitVarLike_(const VarPtr& op) override {
    auto target_it = candidate_by_target_.find(op.get());
    if (target_it != candidate_by_target_.end()) {
      (*candidates_)[target_it->second].safe = false;
    }
    auto staging_it = candidate_by_staging_.find(op.get());
    if (staging_it != candidate_by_staging_.end()) {
      (*candidates_)[staging_it->second].safe = false;
    }
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (definition_stmts_.count(op.get()) > 0) return;
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallPtr& op) override {
    if (IsOp(op, "tensor.write") && op->args_.size() == 3) {
      auto target = AsVarLike(op->args_[0]);
      auto it = target ? candidate_by_target_.find(target.get()) : candidate_by_target_.end();
      if (it != candidate_by_target_.end()) {
        ++(*candidates_)[it->second].scalar_write_count;
        VisitExpr(op->args_[1]);
        VisitExpr(op->args_[2]);
        return;
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    for (const auto& value : op->value_) {
      auto var = AsVarLike(value);
      if (var && candidate_by_target_.count(var.get()) > 0) continue;
      VisitExpr(value);
    }
  }

 private:
  std::vector<FullTensorScalarUpdateCandidate>* candidates_;
  std::unordered_map<const Var*, size_t> candidate_by_target_;
  std::unordered_map<const Var*, size_t> candidate_by_staging_;
  std::unordered_set<const Stmt*> definition_stmts_;
};

class StagedScalarWriteRewriter : public IRMutator {
 public:
  explicit StagedScalarWriteRewriter(std::unordered_map<const Var*, VarPtr> staging_by_target)
      : staging_by_target_(std::move(staging_by_target)) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = As<Call>(IRMutator::VisitExpr_(op));
    if (!visited || !IsOp(visited, "tensor.write") || visited->args_.empty()) return visited;

    auto target = AsVarLike(visited->args_[0]);
    auto it = target ? staging_by_target_.find(target.get()) : staging_by_target_.end();
    if (it == staging_by_target_.end()) return visited;

    auto rewritten = MutableCopy(visited);
    rewritten->args_[0] = it->second;
    return rewritten;
  }

 private:
  std::unordered_map<const Var*, VarPtr> staging_by_target_;
};

/**
 * @brief Stage full-tensor initialization plus scalar overrides in one local tensor.
 *
 * Rewrites this safe subset:
 *
 *   staging = tensor.full(full_shape, value)
 *   updated = tensor.assemble(gm, staging, zeros)
 *   ... tensor.write(updated, dynamic_indices, value) ...
 *
 * into local writes followed by one final tensor.assemble. This preserves the
 * source semantics while ensuring all GM traffic uses MTE3.
 */
StmtPtr CanonicalizeFullTensorScalarUpdates(const StmtPtr& body, const std::vector<VarPtr>& params) {
  auto stmts = FlattenToStmts(body);
  if (stmts.size() < 3 || !As<ReturnStmt>(stmts.back())) return body;

  std::unordered_set<const Var*> param_set;
  for (const auto& param : params) {
    param_set.insert(param.get());
  }

  std::vector<FullTensorScalarUpdateCandidate> candidates;
  std::unordered_set<const Var*> used_targets;
  std::unordered_set<const Var*> used_staging;
  for (size_t i = 1; i + 1 < stmts.size(); ++i) {
    auto staging_assign = As<AssignStmt>(stmts[i - 1]);
    auto staging_call = staging_assign ? As<Call>(staging_assign->value_) : nullptr;
    auto assemble_stmt = As<AssignStmt>(stmts[i]);
    auto assemble = assemble_stmt ? As<Call>(assemble_stmt->value_) : nullptr;
    if (!staging_assign || !staging_call || !IsOp(staging_call, "tensor.full") || !assemble ||
        !IsOp(assemble, "tensor.assemble") || assemble->args_.size() != 3) {
      continue;
    }

    auto destination = AsVarLike(assemble->args_[0]);
    auto target = assemble_stmt->var_;
    auto staging = AsVarLike(assemble->args_[1]);
    auto target_type = target ? As<TensorType>(target->GetType()) : nullptr;
    auto destination_type = destination ? As<TensorType>(destination->GetType()) : nullptr;
    auto staging_type = staging ? As<TensorType>(staging->GetType()) : nullptr;
    if (!destination || !target || !staging || staging.get() != staging_assign->var_.get() ||
        param_set.count(destination.get()) == 0 || target.get() == staging.get() || !target_type ||
        !destination_type || !staging_type || target_type->dtype_ != destination_type->dtype_ ||
        target_type->dtype_ != staging_type->dtype_ || target_type->tensor_view_.has_value() ||
        destination_type->tensor_view_.has_value() || staging_type->tensor_view_.has_value() ||
        !AreExprVectorsEqual(target_type->shape_, destination_type->shape_) ||
        !AreExprVectorsEqual(target_type->shape_, staging_type->shape_) ||
        !IsZeroOffsetTuple(assemble->args_[2], target_type->shape_.size()) ||
        used_targets.count(target.get()) > 0 || used_staging.count(staging.get()) > 0) {
      continue;
    }

    candidates.push_back({i, target, staging, staging_assign, assemble_stmt});
    used_targets.insert(target.get());
    used_staging.insert(staging.get());
  }
  if (candidates.empty()) return body;

  StagedScalarUseAnalyzer analyzer(&candidates);
  analyzer.VisitStmt(body);

  std::vector<FullTensorScalarUpdateCandidate> accepted;
  for (const auto& candidate : candidates) {
    if (candidate.safe && candidate.scalar_write_count > 0) accepted.push_back(candidate);
  }
  if (accepted.empty()) return body;

  std::unordered_map<const Var*, VarPtr> staging_by_target;
  std::unordered_set<size_t> removed_assemble_indices;
  for (const auto& candidate : accepted) {
    staging_by_target[candidate.target.get()] = candidate.staging;
    removed_assemble_indices.insert(candidate.assemble_index);
  }
  StagedScalarWriteRewriter rewriter(std::move(staging_by_target));

  std::vector<StmtPtr> rewritten;
  rewritten.reserve(stmts.size());
  for (size_t i = 0; i + 1 < stmts.size(); ++i) {
    if (removed_assemble_indices.count(i) == 0) {
      rewritten.push_back(rewriter.VisitStmt(stmts[i]));
    }
  }
  for (const auto& candidate : accepted) {
    rewritten.push_back(candidate.assemble_stmt);
  }
  rewritten.push_back(rewriter.VisitStmt(stmts.back()));
  return SeqStmts::Flatten(std::move(rewritten), body->span_);
}

/**
 * @brief Rewrite a constant contiguous scalar-fill loop into one tensor.full + tensor.assemble.
 *
 * A scalar loop that writes every element of one contiguous tensor region with
 * the same constant does not need the D-cache path. Canonicalizing it before
 * tensor-to-tile conversion makes the normal tensor.full/tensor.assemble
 * lowering emit tile.full/tile.store instead. This is both faster and avoids a
 * false mixed-store rejection when another control-flow path stores the same GM
 * tensor through MTE3.
 */
class ConstantScalarFillLoopCanonicalizer : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto visited = As<ForStmt>(IRMutator::VisitStmt_(op));
    if (!visited) return op;

    auto rewrite = TryRewrite(visited);
    return rewrite ? rewrite : visited;
  }

 private:
  static bool IsVar(const ExprPtr& expr, const Var* expected) {
    auto var = AsVarLike(expr);
    return var && var.get() == expected;
  }

  static ExprPtr MatchUnitStrideIndex(const ExprPtr& index, const Var* loop_var, const Span& span) {
    if (IsVar(index, loop_var)) {
      return std::make_shared<ConstInt>(0, DataType::INDEX, span);
    }

    auto add = As<Add>(index);
    if (!add) return nullptr;
    if (IsVar(add->left_, loop_var) && !ExprUsesVar(add->right_, loop_var)) {
      return add->right_;
    }
    if (IsVar(add->right_, loop_var) && !ExprUsesVar(add->left_, loop_var)) {
      return add->left_;
    }
    return nullptr;
  }

  static StmtPtr TryRewrite(const ForStmtPtr& loop) {
    auto start = As<ConstInt>(loop->start_);
    auto stop = As<ConstInt>(loop->stop_);
    auto step = As<ConstInt>(loop->step_);
    if (!start || start->value_ != 0 || !stop || stop->value_ <= 0 || !step || step->value_ != 1 ||
        loop->kind_ != ForKind::Sequential || !loop->iter_args_.empty() || !loop->return_vars_.empty() ||
        !loop->attrs_.empty()) {
      return nullptr;
    }

    auto body_stmts = FlattenToStmts(loop->body_);
    if (body_stmts.empty()) return nullptr;

    std::vector<StmtPtr> rewritten;
    rewritten.reserve(body_stmts.size() * 3);
    std::vector<ExprPtr> reference_indices;
    std::vector<ExprPtr> reference_shape;
    std::optional<DataType> reference_dtype;
    bool reference_has_tensor_view = false;
    for (size_t stmt_index = 0; stmt_index < body_stmts.size(); ++stmt_index) {
      auto eval = As<EvalStmt>(body_stmts[stmt_index]);
      auto write = eval ? As<Call>(eval->expr_) : nullptr;
      if (!write || !IsOp(write, "tensor.write") || write->args_.size() != 3 ||
          (!As<ConstInt>(write->args_[2]) && !As<ConstFloat>(write->args_[2]))) {
        return nullptr;
      }

      auto target_type = As<TensorType>(write->args_[0]->GetType());
      auto indices = As<MakeTuple>(write->args_[1]);
      if (!target_type || !indices || indices->elements_.size() != target_type->shape_.size()) {
        return nullptr;
      }

      // Hoisting multiple scalar writes changes the iteration order from
      // (element, statement) to (statement, element). Keep that rewrite only
      // when every statement has the same contiguous access pattern and type;
      // then even aliased targets observe the same per-element statement order.
      if (stmt_index == 0) {
        reference_indices = indices->elements_;
        reference_shape = target_type->shape_;
        reference_dtype = target_type->dtype_;
        reference_has_tensor_view = target_type->tensor_view_.has_value();
      } else if (reference_has_tensor_view || target_type->tensor_view_.has_value() ||
                 target_type->dtype_ != *reference_dtype ||
                 !AreExprVectorsEqual(target_type->shape_, reference_shape) ||
                 !AreExprVectorsEqual(indices->elements_, reference_indices)) {
        return nullptr;
      }

      std::optional<size_t> varying_axis;
      std::vector<ExprPtr> offsets;
      offsets.reserve(indices->elements_.size());
      for (size_t i = 0; i < indices->elements_.size(); ++i) {
        const auto& index = indices->elements_[i];
        if (!ExprUsesVar(index, loop->loop_var_.get())) {
          offsets.push_back(index);
          continue;
        }
        if (varying_axis.has_value()) return nullptr;
        auto base = MatchUnitStrideIndex(index, loop->loop_var_.get(), write->span_);
        if (!base) return nullptr;
        varying_axis = i;
        offsets.push_back(base);
      }
      if (!varying_axis.has_value()) return nullptr;

      int64_t fill_bits = 0;
      const int64_t element_bits = static_cast<int64_t>(target_type->dtype_.GetBit());
      if (__builtin_mul_overflow(stop->value_, element_bits, &fill_bits) || fill_bits % 256 != 0) {
        return nullptr;
      }

      // A rectangular [1, ..., trip_count, ..., 1] tensor is contiguous only
      // when every physical dimension after the varying axis is a singleton.
      // Restrict the rewrite to that case rather than changing strided-write
      // semantics for a general tensor.
      for (size_t i = *varying_axis + 1; i < target_type->shape_.size(); ++i) {
        auto extent = As<ConstInt>(target_type->shape_[i]);
        auto offset = As<ConstInt>(offsets[i]);
        if (!extent || extent->value_ != 1 || !offset || offset->value_ != 0) {
          return nullptr;
        }
      }

      std::vector<ExprPtr> logical_shape;
      logical_shape.reserve(target_type->shape_.size());
      for (size_t i = 0; i < target_type->shape_.size(); ++i) {
        const int64_t extent = i == *varying_axis ? stop->value_ : 1;
        logical_shape.push_back(std::make_shared<ConstInt>(extent, DataType::INDEX, write->span_));
      }
      std::vector<ExprPtr> physical_shape = {
          std::make_shared<ConstInt>(1, DataType::INDEX, write->span_),
          std::make_shared<ConstInt>(stop->value_, DataType::INDEX, write->span_),
      };

      auto& op_registry = OpRegistry::GetInstance();
      std::vector<std::pair<std::string, std::any>> full_kwargs = {{"dtype", target_type->dtype_}};
      auto full =
          op_registry.Create("tensor.full", {MakeShapeTuple(physical_shape, write->span_), write->args_[2]},
                             full_kwargs, write->span_);

      auto target = AsVarLike(write->args_[0]);
      const std::string base_name =
          auto_name::GetBaseName(target ? target->name_hint_ : loop->loop_var_->name_hint_);
      auto full_var = std::make_shared<Var>(auto_name::BuildName(base_name, "", "scalar_fill_storage"),
                                            full->GetType(), write->span_);
      auto comments = eval->leading_comments_;
      if (stmt_index == 0) {
        comments.insert(comments.begin(), loop->leading_comments_.begin(), loop->leading_comments_.end());
      }
      rewritten.push_back(std::make_shared<AssignStmt>(full_var, full, write->span_, std::move(comments)));

      auto reshape = op_registry.Create(
          "tensor.reshape", {full_var, MakeShapeTuple(logical_shape, write->span_)}, write->span_);
      auto reshape_var = std::make_shared<Var>(auto_name::BuildName(base_name, "", "scalar_fill"),
                                               reshape->GetType(), write->span_);
      rewritten.push_back(std::make_shared<AssignStmt>(reshape_var, reshape, write->span_));

      auto assemble = op_registry.Create(
          "tensor.assemble",
          {write->args_[0], reshape_var, std::make_shared<MakeTuple>(offsets, write->span_)}, write->span_);
      rewritten.push_back(std::make_shared<EvalStmt>(assemble, write->span_));
    }
    return SeqStmts::Flatten(std::move(rewritten), loop->span_);
  }
};

// ============================================================================
// Param direction inference: analyze read/write patterns to upgrade In→Out/InOut
//
// Traces tensor alias chains through ForStmt/WhileStmt iter-args, IfStmt
// branches, tile.store, tensor.write, and tensor.assemble to determine
// which function parameters are written to.  Parameters that are written
// but not read become Out; those that are both read and written become InOut.
// ============================================================================

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

void RecordFirstStoreSpan(const ParamOrigins& origins, std::vector<std::optional<Span>>& spans,
                          const Span& span) {
  for (size_t index : origins) {
    if (index < spans.size() && !spans[index].has_value()) {
      spans[index].emplace(span);
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

/// The argument whose buffer this call's SSA result names, or null when the
/// result is a fresh value.
///
/// This is the destination-rebind idiom — `c2 = tile.store(t, off, c)`, where
/// reading `c2` reads `c` — and it is a *narrower* question than "which argument
/// does this operator write", which the registry now answers (`set_arg_effect`).
/// The two differ: `tile.mgather` clobbers a GM scratch operand yet returns a
/// fresh tile, so it writes an argument it does not alias.
///
/// An operator declaring `set_output_reuses_input(N)` states exactly this
/// relation, so that declaration is consulted first. The remaining entries are
/// the tensor-level and cross-rank operators whose result rebinds a destination
/// without reusing an on-chip MemRef; unifying them onto one declaration is
/// follow-up work, and the two sources are kept consistent by the write-effect
/// check below.
ExprPtr ResultAliasedDestination(const CallPtr& call) {
  if (auto index = ResultAliasedArgIndex(call)) return call->args_[*index];
  return nullptr;
}

void UpdateTensorAliasOrigin(const VarPtr& var, const ParamOrigins& origins, AliasOriginMap& origin_map) {
  if (AsTensorTypeLike(var->GetType()) && !origins.empty()) {
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

  if (auto write_target = ResultAliasedDestination(call)) {
    return GetAliasOrigins(write_target, origin_map);
  }
  if ((IsOp(call, "tensor.slice") || IsOp(call, "tensor.view")) && !call->args_.empty()) {
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

/// Mark the parameter origins each argument of @p call reads and writes.
///
/// Which argument a call writes is a property of the operator, declared once on
/// the registry (`set_arg_effect`) and read here — this pass used to carry its
/// own table of twenty operators, whose default arm counted every argument of an
/// operator it did not recognise as a read. That default is why a GM tensor
/// written only by `tile.mscatter` kept direction `In`.
///
/// Reads resolve through `CollectReferencedOrigins`, since an operand may merely
/// *mention* a buffer (an offsets tuple built from `tensor.dim`). Writes resolve
/// through `GetAliasOrigins`, since a destination operand names the buffer
/// itself. A pure `Write` argument is not marked as a read: a store that lands
/// on a sub-region never reads the untouched remainder.
void AnalyzeCallAccess(const CallPtr& call, const AliasOriginMap& origin_map, std::vector<bool>& has_read,
                       std::vector<bool>& has_write, std::vector<std::optional<Span>>& dma_store_spans,
                       std::vector<std::optional<Span>>& scalar_store_spans) {
  if (!call) return;

  // A call to a user function reaches here with a GlobalVar callee and no
  // registry entry. Its writes are propagated separately, from the callee's
  // declared param directions; every operand counts as a read here.
  const auto* entry = LookupOpEntry(call->op_);

  for (size_t i = 0; i < call->args_.size(); ++i) {
    const auto effect = entry ? entry->GetArgEffect(i, call->kwargs_) : ArgEffect::Read;

    if (ArgEffectReads(effect)) {
      MarkAccess(CollectReferencedOrigins(call->args_[i], origin_map), has_read);
    }
    if (!ArgEffectWrites(effect)) continue;

    auto origins = GetAliasOrigins(call->args_[i], origin_map);
    MarkAccess(origins, has_write);
    // Only a GM store participates in the mixed-channel diagnostic below; an
    // operator that writes a tile, an array or a signal slot declares no
    // channel and is skipped.
    if (auto channel = entry->GetWriteChannel()) {
      RecordFirstStoreSpan(origins, *channel == WriteChannel::Scalar ? scalar_store_spans : dma_store_spans,
                           call->span_);
    }
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
                                  std::vector<bool>& has_read, std::vector<bool>& has_write,
                                  std::vector<std::optional<Span>>& dma_store_spans,
                                  std::vector<std::optional<Span>>& scalar_store_spans);

void BindLoopOrigins(const std::vector<IterArgPtr>& iter_args, const std::vector<ParamOrigins>& origins,
                     AliasOriginMap& origin_map) {
  for (size_t i = 0; i < iter_args.size(); ++i) {
    if (i < origins.size() && !origins[i].empty()) {
      origin_map[iter_args[i].get()] = origins[i];
    } else {
      origin_map.erase(iter_args[i].get());
    }
  }
}

std::vector<ParamOrigins> AnalyzeLoopCarriedOrigins(const StmtPtr& body,
                                                    const std::vector<IterArgPtr>& iter_args,
                                                    const AliasOriginMap& origin_map,
                                                    std::vector<bool>& has_read, std::vector<bool>& has_write,
                                                    std::vector<std::optional<Span>>& dma_store_spans,
                                                    std::vector<std::optional<Span>>& scalar_store_spans) {
  std::vector<ParamOrigins> carried_origins(iter_args.size());
  for (size_t i = 0; i < iter_args.size(); ++i) {
    carried_origins[i] = GetAliasOrigins(iter_args[i]->initValue_, origin_map);
  }

  auto body_map = origin_map;
  BindLoopOrigins(iter_args, carried_origins, body_map);
  auto yield_info =
      AnalyzeStmtAliases(body, body_map, has_read, has_write, dma_store_spans, scalar_store_spans);

  bool origins_widened = false;
  if (yield_info.has_yield) {
    for (size_t i = 0; i < carried_origins.size() && i < yield_info.origins.size(); ++i) {
      size_t previous_size = carried_origins[i].size();
      MergeOrigins(carried_origins[i], yield_info.origins[i]);
      origins_widened |= carried_origins[i].size() != previous_size;
    }
  }

  // Alias transfers only preserve or union parameter origins. One widened
  // rescan therefore covers stores in later iterations without an unbounded
  // fixed-point traversal of the loop body.
  if (origins_widened) {
    body_map = origin_map;
    BindLoopOrigins(iter_args, carried_origins, body_map);
    yield_info = AnalyzeStmtAliases(body, body_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    if (yield_info.has_yield) {
      for (size_t i = 0; i < carried_origins.size() && i < yield_info.origins.size(); ++i) {
        MergeOrigins(carried_origins[i], yield_info.origins[i]);
      }
    }
  }
  return carried_origins;
}

YieldAliasInfo AnalyzeStmtSequenceAliases(const std::vector<StmtPtr>& stmts, AliasOriginMap& origin_map,
                                          std::vector<bool>& has_read, std::vector<bool>& has_write,
                                          std::vector<std::optional<Span>>& dma_store_spans,
                                          std::vector<std::optional<Span>>& scalar_store_spans) {
  YieldAliasInfo last_yield;
  for (const auto& stmt : stmts) {
    auto yield_info =
        AnalyzeStmtAliases(stmt, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    if (yield_info.has_yield) {
      last_yield = yield_info;
    }
  }
  return last_yield;
}

YieldAliasInfo AnalyzeStmtAliases(const StmtPtr& stmt, AliasOriginMap& origin_map,
                                  std::vector<bool>& has_read, std::vector<bool>& has_write,
                                  std::vector<std::optional<Span>>& dma_store_spans,
                                  std::vector<std::optional<Span>>& scalar_store_spans) {
  if (!stmt) return {};

  if (auto assign = As<AssignStmt>(stmt)) {
    if (auto call = As<Call>(assign->value_)) {
      AnalyzeCallAccess(call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }

    if (AsTensorTypeLike(assign->var_->GetType())) {
      auto origins = GetAliasOrigins(assign->value_, origin_map);
      UpdateTensorAliasOrigin(assign->var_, origins, origin_map);
    }
    return {};
  }

  if (auto eval = As<EvalStmt>(stmt)) {
    if (auto call = As<Call>(eval->expr_)) {
      AnalyzeCallAccess(call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    return {};
  }

  if (auto seq = As<SeqStmts>(stmt)) {
    return AnalyzeStmtSequenceAliases(seq->stmts_, origin_map, has_read, has_write, dma_store_spans,
                                      scalar_store_spans);
  }

  if (auto scope = As<ScopeStmt>(stmt)) {
    return AnalyzeStmtAliases(scope->body_, origin_map, has_read, has_write, dma_store_spans,
                              scalar_store_spans);
  }

  if (auto if_stmt = As<IfStmt>(stmt)) {
    if (auto cond_call = As<Call>(if_stmt->condition_)) {
      AnalyzeCallAccess(cond_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    auto then_map = origin_map;
    auto then_yield = AnalyzeStmtAliases(if_stmt->then_body_, then_map, has_read, has_write, dma_store_spans,
                                         scalar_store_spans);
    YieldAliasInfo else_yield;
    if (auto else_body = if_stmt->else_body_.value_or(nullptr)) {
      auto else_map = origin_map;
      else_yield =
          AnalyzeStmtAliases(else_body, else_map, has_read, has_write, dma_store_spans, scalar_store_spans);
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
      AnalyzeCallAccess(start_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    if (auto stop_call = As<Call>(for_stmt->stop_)) {
      AnalyzeCallAccess(stop_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }
    if (auto step_call = As<Call>(for_stmt->step_)) {
      AnalyzeCallAccess(step_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }

    auto carried_origins =
        AnalyzeLoopCarriedOrigins(for_stmt->body_, for_stmt->iter_args_, origin_map, has_read, has_write,
                                  dma_store_spans, scalar_store_spans);
    for (size_t i = 0; i < for_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins = i < carried_origins.size() ? carried_origins[i] : ParamOrigins{};
      UpdateTensorAliasOrigin(for_stmt->return_vars_[i], origins, origin_map);
    }
    return {};
  }

  if (auto while_stmt = As<WhileStmt>(stmt)) {
    if (auto cond_call = As<Call>(while_stmt->condition_)) {
      AnalyzeCallAccess(cond_call, origin_map, has_read, has_write, dma_store_spans, scalar_store_spans);
    }

    auto carried_origins =
        AnalyzeLoopCarriedOrigins(while_stmt->body_, while_stmt->iter_args_, origin_map, has_read, has_write,
                                  dma_store_spans, scalar_store_spans);
    for (size_t i = 0; i < while_stmt->return_vars_.size(); ++i) {
      ParamOrigins origins = i < carried_origins.size() ? carried_origins[i] : ParamOrigins{};
      UpdateTensorAliasOrigin(while_stmt->return_vars_[i], origins, origin_map);
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

/// Upgrade In params to Out/InOut based on tile.store/tensor.write usage analysis.
void UpgradeWrittenTensorParamDirections(const std::vector<StmtPtr>& stmts, const std::vector<VarPtr>& params,
                                         std::vector<ParamDirection>& param_directions) {
  std::vector<bool> has_read(params.size(), false);
  std::vector<bool> has_write(params.size(), false);
  std::vector<std::optional<Span>> dma_store_spans(params.size());
  std::vector<std::optional<Span>> scalar_store_spans(params.size());
  AliasOriginMap origin_map;

  for (size_t i = 0; i < params.size() && i < param_directions.size(); ++i) {
    // AsTensorTypeLike also seeds DistributedTensorType window params, so a
    // pld.tile.remote_store into such a param is attributed as a write below.
    if (!AsTensorTypeLike(params[i]->GetType())) {
      continue;
    }
    origin_map[params[i].get()] = ParamOrigins{i};
  }

  auto analysis_map = origin_map;
  AnalyzeStmtSequenceAliases(stmts, analysis_map, has_read, has_write, dma_store_spans, scalar_store_spans);

  for (size_t i = 0; i < params.size() && i < param_directions.size(); ++i) {
    if (dma_store_spans[i].has_value() && scalar_store_spans[i].has_value()) {
      CHECK_SPAN(false, scalar_store_spans[i].value_or(Span::unknown()))
          << "GM tensor '" << params[i]->name_hint_
          << "' mixes MTE3 and scalar stores in one InCore function. tile.store/tensor.assemble "
             "uses the MTE3 path while tensor.write uses the scalar D-cache path; PyPTO cannot "
             "guarantee ordering or cache-line coherence between them. Rewrite the updates through "
             "one UB tile (use tile.write, then one tile.store), or use tensor.write for every GM "
             "element written to this tensor.";
    }
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
        if (call && IsOp(call, "tile.assemble") && call->args_.size() == 3) {
          if (auto iter = As<IterArg>(call->args_[0])) {
            is_target_assemble = iter.get() == old_iter_arg.get();
          } else if (auto var = As<Var>(call->args_[0])) {
            is_target_assemble = var.get() == old_iter_arg.get();
          }
        }
        if (is_target_assemble) {
          if (assemble_assign) return std::nullopt;
          auto assemble_call = As<Call>(assign->value_);
          INTERNAL_CHECK_SPAN(assemble_call, assign->span_)
              << "Internal error: expected tile.assemble call in assemble loop rewrite";
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
    INTERNAL_CHECK_SPAN(assemble_call, assemble_assign->span_)
        << "Internal error: expected tile.assemble call in assemble loop rewrite";

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
        auto new_assign = MutableCopy(assemble_assign);
        new_assign->var_ = store_var;
        new_assign->value_ = store_call;
        new_body_stmts.push_back(std::move(new_assign));
        continue;
      }
      if (body_stmt == yield_stmt) {
        auto new_yield = MutableCopy(yield_stmt);
        new_yield->value_ = std::vector<ExprPtr>{store_var};
        new_body_stmts.push_back(std::move(new_yield));
        continue;
      }
      new_body_stmts.push_back(body_stmt);
    }

    auto new_return_var = std::make_shared<Var>(for_stmt->return_vars_[0]->name_hint_, out_tensor_type,
                                                for_stmt->return_vars_[0]->span_);
    auto new_for_stmt = MutableCopy(for_stmt);
    new_for_stmt->iter_args_ = std::vector<IterArgPtr>{new_iter_arg};
    new_for_stmt->body_ = SeqStmts::Flatten(std::move(new_body_stmts), for_stmt->body_->span_);
    new_for_stmt->return_vars_ = std::vector<VarPtr>{new_return_var};

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

IncoreTransformResult TransformIncoreFunction(const FunctionPtr& func) {
  auto& conv_registry = OpConversionRegistry::GetInstance();
  auto& op_registry = OpRegistry::GetInstance();
  // Structural nodes (body SeqStmts, the rebuilt Function) belong to the whole
  // function, so they carry its span.  Synthesized *ops* must not: an entry load
  // is attributed to the parameter that motivated it and an exit store to the
  // `return` that motivated it, so post-pass diagnostics point at real source
  // lines instead of the `def` line.
  const auto& span = func->span_;

  auto staged_body = CanonicalizeFullTensorScalarUpdates(func->body_, func->params_);
  ConstantScalarFillLoopCanonicalizer scalar_fill_canonicalizer;
  auto canonical_body = scalar_fill_canonicalizer.VisitStmt(staged_body);

  // Pre-scan: collect consumer memory space requirements (e.g. tensor.slice → tensor.matmul
  // needs Mat-space loads).  Driven by InputSpaceReq metadata in OpConversionRegistry.
  // Then propagate demands backward through pass-through ops (tensor.fillpad etc.) so a
  // chain like `slice → fillpad → matmul` routes the slice's load directly into Mat.
  ConsumerSpaceCollector consumer_collector(conv_registry);
  consumer_collector.VisitStmt(canonical_body);
  consumer_collector.PropagateThroughInheritInputOps();

  // Resolve the scope-declared GM cache policies onto this function's params.
  // Every load this pass synthesises below, and every load already in the body,
  // carries the declaration onward as a ``cache`` kwarg; the function attr is
  // erased when the transformed function is rebuilt.
  auto cache_policies = BuildCachePolicyByParam(func);

  // Create the body mutator
  TensorToTileMutator mutator(conv_registry, op_registry, consumer_collector, cache_policies);

  // New body statements (prefix tile.loads + mutated body)
  std::vector<StmtPtr> new_stmts;

  // Phase 1: Insert tile.load for each TensorType parameter that is directly consumed
  // by a converted tensor op.  Parameters that are only referenced by non-converted ops
  // (e.g. tile.load, tile.move) already manage their own tile representation and must
  // NOT get an additional load inserted here.
  TensorArgsInConvertedOpsCollector collector(conv_registry);
  collector.VisitStmt(canonical_body);
  collector.TraceIterArgInitValues();
  const auto& params_used_by_converted_ops = collector.GetUsed();

  for (const auto& var : func->params_) {
    auto tensor_type = As<TensorType>(var->GetType());
    if (!tensor_type) continue;

    if (params_used_by_converted_ops.find(var.get()) == params_used_by_converted_ops.end()) continue;

    // Attribute the entry load to the parameter declaration it loads, not to `def`.
    const auto& load_span = var->span_;
    auto offsets = MakeZeroOffsets(tensor_type->shape_.size(), load_span);
    auto shapes = MakeShapeTuple(tensor_type->shape_, load_span);

    // Honour the same consumer demand the mutator honours for the loads it
    // creates (see BridgeInputSpaces): a parameter feeding a matmul goes
    // straight to Mat rather than landing in Vec and being moved out again.
    //
    // With no demand recorded, leave `target_memory` *absent*. It used to be
    // hard-coded to Vec, which is a guess this pass is not equipped to make --
    // it sees only the ops it converts, while InferTileMemorySpace (pass 17)
    // sees the whole function and places the tile from actual consumer demand.
    // An unset space is the IR's "not decided yet", so stating Vec here would
    // overwrite a real answer with a default and make pass 17 honour it (it
    // never overrides a present kwarg).
    auto entry_req = consumer_collector.GetConsumerReq(var.get());
    std::vector<std::pair<std::string, std::any>> load_kwargs;
    if (entry_req.has_value()) {
      load_kwargs.emplace_back("target_memory", entry_req->space);
    }
    AppendCachePolicyKwarg(var, cache_policies, &load_kwargs);
    auto load_call = MarkCompilerMatBridge(
        op_registry.Create("tile.load", {var, offsets, shapes, shapes}, load_kwargs, load_span),
        entry_req.has_value() ? entry_req->space : MemorySpace::Vec);

    std::string tile_name = MakeTileValueName(var->name_hint_);
    auto tile_var = std::make_shared<Var>(tile_name, load_call->GetType(), load_span);

    new_stmts.push_back(std::make_shared<AssignStmt>(tile_var, load_call, load_span));
    mutator.AddMapping(var.get(), tile_var);
  }

  // Phase 2: Transform body via mutator (handles control flow recursion + op conversion)
  auto body_stmts = FlattenToStmts(canonical_body);

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

  if (return_stmt) {
    std::vector<ExprPtr> new_return_exprs;
    // The exit stores and the Out params they write exist because of this
    // `return`, so they are attributed to it rather than to the `def` line.
    const auto& ret_span = return_stmt->span_;

    // Process each return value
    for (size_t i = 0; i < return_stmt->value_.size(); ++i) {
      auto ret_expr = mutator.VisitExpr(return_stmt->value_[i]);

      // Check if the return value is a tile (was converted from tensor)
      auto tile_type = As<TileType>(ret_expr->GetType());
      if (tile_type) {
        // Find the original tensor type from the function's return types
        auto orig_tensor_type = As<TensorType>(func->return_types_[i]);
        INTERNAL_CHECK_SPAN(orig_tensor_type, func->span_)
            << "Internal error: return type " << i << " should be TensorType but got "
            << func->return_types_[i]->TypeName();

        // Add output tensor parameter
        std::string out_name = MakeOutParamName(num_added_outputs);

        auto out_type = orig_tensor_type;
        auto out_param = std::make_shared<Var>(out_name, out_type, ret_span);
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
        auto offsets = MakeZeroOffsets(tile_type->shape_.size(), ret_span);
        auto store_call = op_registry.Create("tile.store", {ret_expr, offsets, out_param}, ret_span);

        auto store_var =
            std::make_shared<Var>(MakeStoreResultName(num_added_outputs), store_call->GetType(), ret_span);
        new_stmts.push_back(std::make_shared<AssignStmt>(store_var, store_call, ret_span));

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
    auto new_return = MutableCopy(return_stmt);
    new_return->value_ = std::move(new_return_exprs);
    new_stmts.push_back(std::move(new_return));
  } else {
    // Void function (e.g. cross-core producer): add empty return
    INTERNAL_CHECK_SPAN(func->return_types_.empty(), func->span_)
        << "Internal error: function '" << func->name_ << "' has no ReturnStmt but declares "
        << func->return_types_.size() << " return type(s) — possible malformed IR";
    new_stmts.push_back(std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, span));
  }

  UpgradeWrittenTensorParamDirections(new_stmts, new_params, new_param_directions);

  auto new_body = SeqStmts::Flatten(std::move(new_stmts), span);
  auto new_func = std::make_shared<Function>(func->name_, new_params, new_param_directions, new_return_types,
                                             new_body, span, FunctionType::InCore, func->level_, func->role_,
                                             EraseCachePolicyAttr(func->attrs_));

  return {new_func, num_added_outputs};
}

// ============================================================================
// Wrapper forward propagation: Spmd/Group wrappers produced by
// OutlineClusterScopes are transparent 1:1 forwarders from their params to a
// single inner InCore call. When the InCore callee gains output params
// (Phase 1), the wrapper must mirror those params on its own signature and
// forward them to the inner call, instead of synthesising tensor.create in
// the wrapper body. Orchestration codegen's BuildWrapperReorderedParams
// relies on every inner-call Var arg resolving to a wrapper param.
// ============================================================================

/// Find the first Call in a stmt tree whose callee is a transformed InCore
/// function (listed in `incore_added_outputs` with >0 added outputs). Used to
/// pre-size the wrapper's new Out params before the mutator runs.
class ForwardedCallFinder : public IRVisitor {
 public:
  explicit ForwardedCallFinder(const std::unordered_map<std::string, size_t>& incore_added_outputs)
      : incore_added_outputs_(incore_added_outputs) {}

  [[nodiscard]] const CallPtr& GetFound() const { return found_; }

  void VisitExpr_(const CallPtr& op) override {
    if (!found_) {
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(op->op_);
      if (gv) {
        auto it = incore_added_outputs_.find(gv->name_);
        if (it != incore_added_outputs_.end() && it->second > 0) {
          found_ = op;
          return;
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  const std::unordered_map<std::string, size_t>& incore_added_outputs_;
  CallPtr found_;
};

/// Mutator that rewrites a wrapper's forwarding call: append the
/// pre-allocated `new_output_vars_` (wrapper-level Out params) to the call's
/// arg list and update the call's return type to match the transformed
/// InCore callee. Does NOT insert tensor.create — the allocation is the
/// responsibility of the wrapper's caller. Recurses through nested control
/// flow via IRMutator's base behavior, so forwarded calls inside ForStmt /
/// IfStmt / WhileStmt bodies are handled correctly.
class WrapperForwardMutator : public TypePropagatingMutator {
 public:
  WrapperForwardMutator(const std::unordered_map<std::string, size_t>& incore_added_outputs,
                        const std::unordered_map<std::string, FunctionPtr>& transformed_incore_funcs,
                        std::vector<VarPtr> new_output_vars)
      : incore_added_outputs_(incore_added_outputs),
        transformed_incore_funcs_(transformed_incore_funcs),
        new_output_vars_(std::move(new_output_vars)) {}

  [[nodiscard]] bool applied() const { return applied_; }

 protected:
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Pin this Var's address for the pass so a freed-then-reused address cannot
    // alias a stale var_remap_ entry (see TypePropagatingMutator::RetainVar).
    RetainVar(op->var_);
    auto new_value = VisitExpr(op->value_);
    auto call = As<Call>(new_value);
    if (!call) return HandlePassThroughAssign(op, new_value);
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
    if (!global_var) return HandlePassThroughAssign(op, new_value);

    auto it = incore_added_outputs_.find(global_var->name_);
    if (it == incore_added_outputs_.end() || it->second == 0) {
      return HandlePassThroughAssign(op, new_value);
    }

    INTERNAL_CHECK_SPAN(!applied_, call->span_)
        << "Wrapper forward propagation saw more than one forwarded call; outlining invariant violated";
    INTERNAL_CHECK_SPAN(it->second == new_output_vars_.size(), call->span_)
        << "Wrapper new-output count mismatch: callee added " << it->second << ", wrapper prepared "
        << new_output_vars_.size();

    auto incore_func_it = transformed_incore_funcs_.find(global_var->name_);
    INTERNAL_CHECK_SPAN(incore_func_it != transformed_incore_funcs_.end(), call->span_)
        << "Internal error: transformed InCore function not found: " << global_var->name_;
    const auto& incore_func = incore_func_it->second;

    std::vector<ExprPtr> new_args = call->args_;
    for (const auto& v : new_output_vars_) {
      new_args.push_back(v);
    }

    TypePtr new_return_type;
    if (incore_func->return_types_.empty()) {
      new_return_type = nullptr;
    } else if (incore_func->return_types_.size() == 1) {
      new_return_type = incore_func->return_types_[0];
    } else {
      new_return_type = std::make_shared<TupleType>(incore_func->return_types_);
    }

    // Preserve the original call's attrs_ (e.g. kAttrDumpVars from
    // pl.dump_tag / dumps=) through the arg-appending rewrite — mirrors the
    // base IRMutator and the Submit path below. The appended outputs do not
    // rename existing arg Vars, so a verbatim attr copy keeps any dump/dep Var
    // references valid. Fall back to UnknownType for a void-return callee so the
    // rewritten Call's type_ stays identical to the prior 4-arg ctor path.
    auto new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_,
                                           new_return_type ? new_return_type : GetUnknownType(), call->span_);

    auto new_assign_var = std::make_shared<Var>(op->var_->name_hint_, new_return_type, op->var_->span_);
    std::shared_ptr<AssignStmt> new_assign = MutableCopy(op);
    new_assign->var_ = new_assign_var;
    new_assign->value_ = new_call;
    var_remap_[op->var_.get()] = new_assign_var;
    applied_ = true;
    StmtPtr result = new_assign;
    return result;
  }

 private:
  const std::unordered_map<std::string, size_t>& incore_added_outputs_;
  const std::unordered_map<std::string, FunctionPtr>& transformed_incore_funcs_;
  std::vector<VarPtr> new_output_vars_;
  bool applied_ = false;
};

struct WrapperTransformResult {
  FunctionPtr func;
  size_t num_added_outputs;
};

/// Propagate a transformed InCore's added output params through a Spmd/Group
/// wrapper: mirror them on the wrapper's signature and forward them to the
/// inner call. Returns {func, 0} if the wrapper does not forward to any
/// transformed InCore callee.
WrapperTransformResult PropagateOutputsThroughWrapper(
    const FunctionPtr& func, const std::unordered_map<std::string, size_t>& incore_added_outputs,
    const std::unordered_map<std::string, FunctionPtr>& transformed_incore_funcs) {
  ForwardedCallFinder finder(incore_added_outputs);
  finder.VisitStmt(func->body_);
  const auto& target_call = finder.GetFound();
  if (!target_call) return {func, 0};

  auto gv = std::dynamic_pointer_cast<const GlobalVar>(target_call->op_);
  INTERNAL_CHECK_SPAN(gv != nullptr, target_call->span_)
      << "Internal error: forwarded call op is not a GlobalVar";
  auto added_outputs_it = incore_added_outputs.find(gv->name_);
  INTERNAL_CHECK_SPAN(added_outputs_it != incore_added_outputs.end(), target_call->span_)
      << "Internal error: missing added-output metadata for forwarded callee " << gv->name_;
  auto transformed_incore_func_it = transformed_incore_funcs.find(gv->name_);
  INTERNAL_CHECK_SPAN(transformed_incore_func_it != transformed_incore_funcs.end(), target_call->span_)
      << "Internal error: missing transformed InCore function for forwarded callee " << gv->name_;
  size_t num_added = added_outputs_it->second;
  const auto& incore_func = transformed_incore_func_it->second;

  // Mirror the InCore's appended Out params on the wrapper: same type, Out
  // direction. Names are scoped to the wrapper so the clone is safe.
  std::vector<VarPtr> new_params = func->params_;
  std::vector<ParamDirection> new_dirs = func->param_directions_;
  std::vector<VarPtr> new_output_vars;
  new_output_vars.reserve(num_added);
  size_t orig_incore_param_count = incore_func->params_.size() - num_added;
  for (size_t i = 0; i < num_added; ++i) {
    const auto& out_param = incore_func->params_[orig_incore_param_count + i];
    auto new_var = std::make_shared<Var>(out_param->name_hint_, out_param->GetType(), func->span_);
    new_params.push_back(new_var);
    new_dirs.push_back(ParamDirection::Out);
    new_output_vars.push_back(new_var);
  }

  WrapperForwardMutator mutator(incore_added_outputs, transformed_incore_funcs, new_output_vars);
  auto new_body = mutator.VisitStmt(func->body_);
  // Outlined Spmd/Group wrappers always forward via an `out = self.kernel(x, ...);
  // return out` AssignStmt — WrapperForwardMutator rewrites that shape. If
  // ForwardedCallFinder found a target call but the mutator failed to apply,
  // the wrapper's signature has been mirrored but its inner call was not
  // updated — a silent mis-rewrite. Fail fast instead.
  INTERNAL_CHECK_SPAN(mutator.applied(), target_call->span_)
      << "Wrapper forward propagation identified a forwarded call in " << func->name_
      << " but could not rewrite it (call not in AssignStmt RHS form expected by outlining invariant)";

  // Keep wrapper's declared returns. The forwarded inner call may be rewritten
  // to pass extra output tensors, but non-transparent wrappers can still
  // construct additional return values (e.g. scalar + tensor tuples). Forcing
  // wrapper return_types_ to match the inner callee can invalidate existing
  // TupleGetItem users at call-sites.
  std::vector<TypePtr> new_return_types = func->return_types_;
  auto new_func =
      std::make_shared<Function>(func->name_, new_params, new_dirs, new_return_types, new_body, func->span_,
                                 func->func_type_, func->level_, func->role_, func->attrs_);
  return {new_func, num_added};
}

// ============================================================================
// CallSiteUpdateMutator: updates call sites in orchestration/opaque functions.
// For each call to a transformed InCore function or a wrapper that has
// absorbed output params, inserts tensor.create for each output param and
// appends them as extra arguments.
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
    // Pin this Var's address for the pass so a freed-then-reused address cannot
    // alias a stale var_remap_ entry (see TypePropagatingMutator::RetainVar).
    RetainVar(op->var_);
    auto new_value = VisitExpr(op->value_);
    auto call = As<Call>(new_value);

    // Submit (pl.submit inside pl.manual_scope) is a sibling call-like kind;
    // route it through the same appended-Out allocation as Call, preserving
    // Submit-ness and its TASK_ID-augmented return type
    // (.claude/rules/pass-submit-awareness.md).
    if (!call) {
      if (auto submit = As<Submit>(new_value)) return HandleSubmitCallSite(op, submit);
      // Non-call or non-GlobalVar: propagate type change
      return HandlePassThroughAssign(op, new_value);
    }
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
    INTERNAL_CHECK_SPAN(incore_func_it != transformed_incore_funcs_.end(), call->span_)
        << "Internal error: transformed InCore function not found: " << global_var->name_;
    const auto& incore_func = incore_func_it->second;

    std::vector<StmtPtr> stmts;
    std::vector<ExprPtr> extra_args;
    size_t orig_param_count = incore_func->params_.size() - num_outputs;

    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& out_param = incore_func->params_[orig_param_count + i];
      auto out_tensor_type = As<TensorType>(out_param->GetType());
      INTERNAL_CHECK_SPAN(out_tensor_type, call->span_) << "Internal error: output param is not TensorType";

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

    // Preserve attrs_ (e.g. kAttrDumpVars) through the arg-appending rewrite —
    // mirrors the base IRMutator and the Submit path below. Appended outputs do
    // not rename existing arg Vars, so a verbatim attr copy stays valid. Fall
    // back to UnknownType for a void-return callee so the rewritten Call's type_
    // stays identical to the prior 4-arg ctor path.
    auto new_call = std::make_shared<Call>(call->op_, new_args, call->kwargs_, call->attrs_,
                                           new_return_type ? new_return_type : GetUnknownType(), call->span_);

    auto new_assign_var = std::make_shared<Var>(op->var_->name_hint_, new_return_type, op->var_->span_);
    auto new_assign = MutableCopy(op);
    new_assign->var_ = new_assign_var;
    new_assign->value_ = new_call;
    stmts.push_back(std::move(new_assign));
    var_remap_[op->var_.get()] = new_assign_var;

    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }

  // Submit variant of the call-site update. The appended outputs are Out
  // *params* (inputs the callee writes) mirroring existing declared returns,
  // not additional returns, so the Submit's own result tuple —
  // Tuple[<callee returns>..., TASK_ID] — is unchanged; only args_ grows and
  // Submit-ness / deps_ / kwargs_ / attrs_ are preserved. The result var keeps
  // its type, so no var remap.
  //
  // Because this forwards an arg for every param it appends, the Submit leaves
  // here with coverage still *exact* — this pass does not open the
  // args_.size() < params_.size() gap that Submit::args_ (include/pypto/ir/expr.h)
  // permits. Forwarding is deliberate: a caller-allocated Out arg lets
  // orchestration codegen alias the return-tuple element straight to the arg
  // instead of synthesising a runtime add_output.
  StmtPtr HandleSubmitCallSite(const AssignStmtPtr& op, const SubmitPtr& submit) {
    auto global_var = std::dynamic_pointer_cast<const GlobalVar>(submit->op_);
    if (!global_var) return HandlePassThroughAssign(op, submit);

    auto it = incore_added_outputs_.find(global_var->name_);
    if (it == incore_added_outputs_.end() || it->second == 0) {
      return HandlePassThroughAssign(op, submit);
    }

    size_t num_outputs = it->second;
    auto incore_func_it = transformed_incore_funcs_.find(global_var->name_);
    INTERNAL_CHECK_SPAN(incore_func_it != transformed_incore_funcs_.end(), submit->span_)
        << "Internal error: transformed InCore function not found: " << global_var->name_;
    const auto& incore_func = incore_func_it->second;

    std::vector<StmtPtr> stmts;
    std::vector<ExprPtr> extra_args;
    size_t orig_param_count = incore_func->params_.size() - num_outputs;
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto& out_param = incore_func->params_[orig_param_count + i];
      auto out_tensor_type = As<TensorType>(out_param->GetType());
      INTERNAL_CHECK_SPAN(out_tensor_type, submit->span_) << "Internal error: output param is not TensorType";
      auto shape_tuple = MakeShapeTuple(out_tensor_type->shape_, submit->span_);
      TensorLayout layout = out_tensor_type->tensor_view_.has_value() ? out_tensor_type->tensor_view_->layout
                                                                      : TensorLayout::ND;
      std::vector<std::pair<std::string, std::any>> create_kwargs = {{"dtype", out_tensor_type->dtype_},
                                                                     {"layout", layout}};
      auto create_call = op_registry_.Create("tensor.create", {shape_tuple}, create_kwargs, submit->span_);
      auto out_var = std::make_shared<Var>(MakeOutParamName(i), create_call->GetType(), submit->span_);
      stmts.push_back(std::make_shared<AssignStmt>(out_var, create_call, op->span_));
      extra_args.push_back(out_var);
    }

    std::vector<ExprPtr> new_args = submit->args_;
    new_args.insert(new_args.end(), extra_args.begin(), extra_args.end());

    // Note: 7-arg Submit ctor order is (op, args, deps, kwargs, attrs, type, span).
    auto new_submit =
        std::make_shared<Submit>(submit->op_, std::move(new_args), submit->deps_, submit->kwargs_,
                                 submit->attrs_, submit->GetType(), submit->span_, submit->core_num_,
                                 submit->sync_start_, submit->allow_early_resolve_, submit->predicate_);
    auto new_assign = MutableCopy(op);
    new_assign->value_ = new_submit;
    stmts.push_back(std::move(new_assign));
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
  auto new_func = MutableCopy(func);
  new_func->body_ = new_body;
  return new_func;
}

}  // namespace

namespace pass {

Pass ConvertTensorToTileOps() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Phase 1: Transform InCore functions
    std::unordered_map<std::string, size_t> incore_added_outputs;
    std::unordered_map<std::string, FunctionPtr> transformed_incore_funcs;
    std::vector<FunctionPtr> functions_phase1;

    for (const auto& [gvar, func] : program->functions_) {
      if (func->func_type_ == FunctionType::InCore) {
        auto result = TransformIncoreFunction(func);
        incore_added_outputs[func->name_] = result.num_added_outputs;
        transformed_incore_funcs[func->name_] = result.func;
        functions_phase1.push_back(result.func);
      } else {
        functions_phase1.push_back(func);
      }
    }

    // Phase 2a: Propagate added output params through Spmd/Group wrappers so
    // they remain transparent 1:1 forwarders of their params to the inner
    // call (an invariant relied on by orchestration codegen).
    std::unordered_map<std::string, size_t> wrapper_added_outputs;
    std::unordered_map<std::string, FunctionPtr> transformed_wrapper_funcs;
    std::vector<FunctionPtr> functions_phase2a;
    functions_phase2a.reserve(functions_phase1.size());
    for (const auto& func : functions_phase1) {
      if (IsWrapperType(func->func_type_)) {
        auto result = PropagateOutputsThroughWrapper(func, incore_added_outputs, transformed_incore_funcs);
        functions_phase2a.push_back(result.func);
        if (result.num_added_outputs > 0) {
          wrapper_added_outputs[func->name_] = result.num_added_outputs;
          transformed_wrapper_funcs[func->name_] = result.func;
        }
      } else {
        functions_phase2a.push_back(func);
      }
    }

    // Phase 2b: Update call sites in orchestration/opaque functions. The
    // callee map covers both transformed InCore functions and wrappers that
    // absorbed their output params.
    std::unordered_map<std::string, size_t> all_added_outputs = incore_added_outputs;
    all_added_outputs.insert(wrapper_added_outputs.begin(), wrapper_added_outputs.end());
    std::unordered_map<std::string, FunctionPtr> all_transformed_funcs = transformed_incore_funcs;
    all_transformed_funcs.insert(transformed_wrapper_funcs.begin(), transformed_wrapper_funcs.end());

    std::vector<FunctionPtr> functions_phase2b;
    functions_phase2b.reserve(functions_phase2a.size());
    for (const auto& func : functions_phase2a) {
      // Skip InCore (rewritten in Phase 1) and every Spmd/Group (rewritten in
      // Phase 2a when forwarding a transformed InCore; otherwise nothing to
      // forward because ForwardedCallFinder rejects callees that gained zero
      // Out params). The postcondition check in PropagateOutputsThroughWrapper
      // turns any finder/mutator mismatch into a hard INTERNAL_CHECK rather
      // than a silent mis-rewrite.
      if (func->func_type_ == FunctionType::InCore || func->func_type_ == FunctionType::Spmd ||
          func->func_type_ == FunctionType::Group) {
        functions_phase2b.push_back(func);
      } else {
        functions_phase2b.push_back(UpdateCallSites(func, all_added_outputs, all_transformed_funcs));
      }
    }

    // Phase 3: Propagate Function::param_directions_ along the call chain.
    //
    // When the user writes inline `pl.at(...)` blocks, OutlineHierarchyScopes
    // extracts them into a host_orch → chip_orch → incore chain. The outlined
    // chip_orch has no direction info on its own parameters yet. Phase 1 has
    // already marked the InCore's tile-written params as Out/InOut; if
    // chip_orch(a, b, f) forwards its own `f` to that InCore, chip_orch's
    // own `f` must be upgraded to Out so the signature matches the data flow.
    //
    // This phase mutates Function::param_directions_ (function signature)
    // only. Per-call-site arg directions (Call::attrs_["arg_directions"]) are
    // owned by the later DeriveCallDirections pass and are not touched here.
    //
    // Fixed-point iteration handles multi-level chains.
    {
      std::unordered_map<std::string, FunctionPtr> func_map;
      for (const auto& func : functions_phase2b) {
        func_map[func->name_] = func;
      }

      // Everything derived from a *body* is computed once, before the fixed
      // point. The loop below only rewrites `param_directions_` — it replaces a
      // function with a `MutableCopy` whose body is the same node — so the
      // parameter index, the buffer lineage and the call list are all invariant
      // across rounds. Rebuilding them per round made the added analysis cost
      // O(rounds x program), over the O(N log N) bound
      // `.claude/rules/pass-complexity.md` sets; hoisting leaves the loop
      // proportional to the call arguments it actually re-reads.
      struct CallerFacts {
        std::unordered_map<const Var*, size_t> param_idx;
        std::vector<CallPtr> calls;
        std::unordered_map<const Var*, const Var*> buffer_roots;
        std::unordered_map<const Var*, std::vector<const Var*>> candidates;
      };
      std::unordered_map<std::string, CallerFacts> facts_by_name;
      for (const auto& func : functions_phase2b) {
        if (func->func_type_ == FunctionType::InCore) continue;

        CallerFacts f;
        for (size_t i = 0; i < func->params_.size(); ++i) {
          f.param_idx[func->params_[i].get()] = i;
        }

        // An argument rarely *is* the parameter. `for acc in ...: acc =
        // kernel(x, acc)` forwards a loop-carried `IterArg` whose value is the
        // parameter's buffer, and looking the IterArg up in `param_idx` finds
        // nothing — so the enclosing signature kept declaring `In` for a buffer
        // the call chain writes. Resolving the argument to its owning buffer
        // first is what makes the propagation reach through a carry, and it is
        // the same resolution the InParamWritten warning uses to decide which
        // parameter a write lands on.
        buffer_root::BufferRootCollector roots(program, buffer_root::AmbiguousRootPolicy::kSkip);
        roots.Initialize(func->params_);
        roots.VisitStmt(func->body_);
        f.buffer_roots = roots.buffer_roots;
        // An ambiguous var is exactly the case the single-root map cannot
        // answer, so keep the candidates it does have.
        for (const Var* var : roots.ambiguous_buffer_vars) {
          f.candidates[var] = roots.RootCandidatesOf(var);
        }

        class CallScanner : public IRVisitor {
         public:
          std::vector<CallPtr> calls;
          void VisitExpr_(const CallPtr& call) override {
            Record(call);
            IRVisitor::VisitExpr_(call);
          }

          /// A task launch forwards its arguments exactly as a plain call does,
          /// and the base visitor does not route `Submit` through the `Call`
          /// handler (`.claude/rules/pass-submit-awareness.md`). Without this a
          /// parameter handed to an `Out` callee through `pl.submit` never
          /// reached the propagation below, so an orchestration function that
          /// only ever submits kept declaring `In` for a buffer its tasks write.
          /// The view is transient — the loop reads only `op_` and `args_` — and
          /// `args_` is a positional prefix of the callee's params, which the
          /// loop's dual bound already respects.
          void VisitExpr_(const SubmitPtr& submit) override {
            Record(SubmitToCallView(submit));
            IRVisitor::VisitExpr_(submit);
          }

         private:
          void Record(const CallPtr& call) {
            if (std::dynamic_pointer_cast<const GlobalVar>(call->op_)) {
              calls.push_back(call);
            }
          }
        };
        CallScanner scanner;
        scanner.VisitStmt(func->body_);
        f.calls = std::move(scanner.calls);

        facts_by_name[func->name_] = std::move(f);
      }

      bool changed = true;
      while (changed) {
        changed = false;
        for (auto& func : functions_phase2b) {
          if (func->func_type_ == FunctionType::InCore) continue;
          auto facts_it = facts_by_name.find(func->name_);
          if (facts_it == facts_by_name.end()) continue;
          const CallerFacts& facts = facts_it->second;

          auto new_dirs = func->param_directions_;
          for (const auto& call : facts.calls) {
            auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
            if (!gv) continue;
            auto callee_it = func_map.find(gv->name_);
            if (callee_it == func_map.end()) continue;
            const auto& callee = callee_it->second;
            for (size_t ai = 0; ai < call->args_.size() && ai < callee->param_directions_.size(); ++ai) {
              // AsVarLike, not As<Var>: an IterArg has its own ObjectKind and
              // does not match As<Var> (.claude/rules/ir-kind-traits.md).
              auto arg_var = AsVarLike(call->args_[ai]);
              if (!arg_var) continue;
              const ParamDirection callee_dir = callee->param_directions_[ai];
              if (callee_dir != ParamDirection::Out && callee_dir != ParamDirection::InOut) continue;

              // Control flow can leave a value naming more than one buffer:
              //
              //     t = a if cond else b
              //     self.writer(src, t)      # writer declares its slot Out
              //
              // The callee writes whichever `t` turned out to be, so *both* `a`
              // and `b` may be written and both must be upgraded. Skipping the
              // ambiguous var — which is what this did — drops the dependency
              // for every candidate at once, and that is the direction that
              // fails silently: an under-declared `In` loses the RAW edge and
              // races on device, where an over-declared `Out` only over-orders.
              // Under `kSkip` such a var has no `buffer_roots` entry by
              // construction, so the candidate list is the only place the answer
              // exists.
              auto cand_it = facts.candidates.find(arg_var.get());
              std::vector<const Var*> arg_roots;
              if (cand_it != facts.candidates.end()) {
                arg_roots = cand_it->second;
              } else {
                auto root_it = facts.buffer_roots.find(arg_var.get());
                arg_roots.push_back(root_it == facts.buffer_roots.end() ? arg_var.get() : root_it->second);
              }

              for (const Var* arg_root : arg_roots) {
                auto pi = facts.param_idx.find(arg_root);
                if (pi == facts.param_idx.end()) continue;
                ParamDirection& caller_dir = new_dirs[pi->second];
                if (callee_dir == ParamDirection::Out && caller_dir == ParamDirection::In) {
                  caller_dir = ParamDirection::Out;
                } else if (callee_dir == ParamDirection::InOut && caller_dir != ParamDirection::InOut) {
                  caller_dir = ParamDirection::InOut;
                }
              }
            }
          }

          if (new_dirs != func->param_directions_) {
            changed = true;
            auto new_func = MutableCopy(func);
            new_func->param_directions_ = std::move(new_dirs);
            func = new_func;
            func_map[func->name_] = func;
          }
        }
      }
    }

    // A `tensor.matmul` drops its operands' valid_shape, so an accumulator only
    // becomes narrower than the seed it is carried from once this pass turns it into
    // a `tile.matmul` -- which re-types the yields but not the carry those yields
    // flow through. Repair it here rather than leave a carry the TypeCheck and
    // AccCompactValid verifiers reject (issue #2470).
    for (auto& func : functions_phase2b) {
      func = narrow_loop_carry::NarrowAccCarries(func);
    }

    return std::make_shared<Program>(functions_phase2b, program->name_, program->span_);
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
      // tensor.read/tensor.write on a gm_tensor (TensorType input) intentionally stays unconverted.
      // ``AsTensorTypeLike`` also whitelists ``DistributedTensorType``, which the
      // conversion registry above keeps as ``tensor.read`` / ``tensor.write`` so the
      // PTO codegen can lower it as a local-rank ``pto.load_scalar`` / ``pto.store_scalar``.
      if ((IsOp(call, "tensor.read") || IsOp(call, "tensor.write")) && !call->args_.empty() &&
          AsTensorTypeLike(call->args_[0]->GetType())) {
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
