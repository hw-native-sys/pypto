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

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/codegen/orchestration/orchestration_analysis.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

using ::pypto::codegen::BufferRootCollector;
using ::pypto::codegen::ComputeGroupEffectiveDirections;
using ::pypto::codegen::IsBuiltinOp;

/// Decide whether an argument expression refers to a tensor (not a scalar/index).
bool IsTensorTypedArg(const ExprPtr& arg) {
  TypePtr ty = arg ? arg->GetType() : TypePtr{};
  if (!ty) return false;
  if (As<TensorType>(ty)) return true;
  if (As<TupleType>(ty)) return true;
  return false;
}

/// Compute the per-position ParamDirection vector for a callee, expanding Group/Spmd
/// callees whose effective directions depend on inner-task call sites.
std::vector<ParamDirection> ResolveCalleeDirections(const ProgramPtr& program, const CallPtr& call,
                                                    const FunctionPtr& callee) {
  if (callee->func_type_ == FunctionType::Group || callee->func_type_ == FunctionType::Spmd) {
    return ComputeGroupEffectiveDirections(callee, program);
  }
  return callee->param_directions_;
}

/// Resolve the buffer root for an argument expression, regardless of whether
/// the root is locally allocated or rooted at an enclosing function parameter.
/// Returns nullptr only when the arg is not a var or has no known buffer root.
const Var* ResolveAnyRoot(const ExprPtr& arg,
                          const std::unordered_map<const Var*, const Var*>& buffer_roots) {
  auto var = AsVarLike(arg);
  if (!var) return nullptr;
  auto it = buffer_roots.find(var.get());
  if (it == buffer_roots.end()) return nullptr;
  return it->second;
}

/// Pre-pass that decides, per (Call, root), whether the call is the "first
/// writer" of that root within its enclosing scope, treating ForStmt/WhileStmt/
/// IfStmt as opaque writer-units. ScopeStmt and SeqStmts are transparent.
/// Tracks both locally-allocated roots and roots that trace back to enclosing
/// function parameters; either kind needs WAW chaining when a prior sibling
/// already wrote to the same root.
///
/// Two phases:
///   1. PrecomputeWrittenRoots: bottom-up cache of the union of local roots
///      written by any non-builtin call inside each subtree.
///   2. AnalyzeScope: top-down scan that maintains a `seen_roots` set of roots
///      already written by prior siblings; for each Call, every Out-param arg
///      whose root is *not* in `seen_roots` is recorded as "first writer".
class PriorWriterCollector {
 public:
  PriorWriterCollector(ProgramPtr program, const std::unordered_map<const Var*, const Var*>& buffer_roots)
      : program_(std::move(program)), buffer_roots_(buffer_roots) {}

  void Run(const StmtPtr& body) {
    if (!body) return;
    PrecomputeWrittenRoots(body);
    std::unordered_set<const Var*> seen;
    AnalyzeScope(body, seen);
  }

  /// Per-Call set of roots for which the call is the first writer in its scope.
  /// Roots not in the set (or Calls absent from the map) are by definition
  /// preceded by another writer-unit and therefore subject to R-prior promotion.
  /// Includes both locally-allocated roots and enclosing-param-rooted ones.
  std::unordered_map<const Call*, std::unordered_set<const Var*>> first_writer_roots;

 private:
  /// Compute (and cache) the set of local roots written by any non-builtin Call
  /// inside the subtree rooted at `stmt`. The result is treated as the "writer
  /// footprint" of the stmt when it appears as a sibling in an outer scope.
  const std::unordered_set<const Var*>& PrecomputeWrittenRoots(const StmtPtr& stmt) {
    auto cached = written_roots_.find(stmt.get());
    if (cached != written_roots_.end()) return cached->second;
    auto& result = written_roots_[stmt.get()];
    if (!stmt) return result;

    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) {
        const auto& child = PrecomputeWrittenRoots(s);
        result.insert(child.begin(), child.end());
      }
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      const auto& body_roots = PrecomputeWrittenRoots(for_stmt->body_);
      result.insert(body_roots.begin(), body_roots.end());
    } else if (auto while_stmt = As<WhileStmt>(stmt)) {
      const auto& body_roots = PrecomputeWrittenRoots(while_stmt->body_);
      result.insert(body_roots.begin(), body_roots.end());
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      const auto& then_roots = PrecomputeWrittenRoots(if_stmt->then_body_);
      result.insert(then_roots.begin(), then_roots.end());
      if (if_stmt->else_body_.has_value() && if_stmt->else_body_.value()) {
        const auto& else_roots = PrecomputeWrittenRoots(if_stmt->else_body_.value());
        result.insert(else_roots.begin(), else_roots.end());
      }
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      const auto& body_roots = PrecomputeWrittenRoots(scope->body_);
      result.insert(body_roots.begin(), body_roots.end());
    } else if (auto assign = As<AssignStmt>(stmt)) {
      CollectCallWrittenRoots(assign->value_, result);
    } else if (auto eval = As<EvalStmt>(stmt)) {
      CollectCallWrittenRoots(eval->expr_, result);
    }
    // YieldStmt / ReturnStmt / BreakStmt / ContinueStmt: no writes.
    return result;
  }

  /// If `expr` is a non-builtin Call, add every Out/InOut root it writes
  /// (local or enclosing-param-rooted) into `out`.
  void CollectCallWrittenRoots(const ExprPtr& expr, std::unordered_set<const Var*>& out) {
    auto call = As<Call>(expr);
    if (!call) return;
    if (IsBuiltinOp(call->op_->name_)) return;
    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) return;

    auto dirs = ResolveCalleeDirections(program_, call, callee);
    for (size_t i = 0; i < dirs.size() && i < call->args_.size(); ++i) {
      if (dirs[i] != ParamDirection::Out && dirs[i] != ParamDirection::InOut) continue;
      if (const Var* root = ResolveAnyRoot(call->args_[i], buffer_roots_)) {
        out.insert(root);
      }
    }
  }

  /// Top-down analysis. `seen` carries the set of local roots already written by
  /// prior siblings (or ancestors' prior siblings) in the surrounding scope.
  /// For/While/If subtrees are entered with a *snapshot copy* of `seen`, so that
  /// writes within the subtree do not leak into the outer scope's sibling tracking.
  /// The unit's pre-computed `written_roots` is then merged into the outer `seen`.
  /// ScopeStmt and SeqStmts are transparent and share the same `seen`.
  void AnalyzeScope(const StmtPtr& stmt, std::unordered_set<const Var*>& seen) {
    if (!stmt) return;
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& s : seq->stmts_) {
        AnalyzeScope(s, seen);
      }
    } else if (auto for_stmt = As<ForStmt>(stmt)) {
      auto inner = seen;
      AnalyzeScope(for_stmt->body_, inner);
      const auto& written = PrecomputeWrittenRoots(for_stmt->body_);
      seen.insert(written.begin(), written.end());
    } else if (auto while_stmt = As<WhileStmt>(stmt)) {
      auto inner = seen;
      AnalyzeScope(while_stmt->body_, inner);
      const auto& written = PrecomputeWrittenRoots(while_stmt->body_);
      seen.insert(written.begin(), written.end());
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      auto then_seen = seen;
      AnalyzeScope(if_stmt->then_body_, then_seen);
      if (if_stmt->else_body_.has_value() && if_stmt->else_body_.value()) {
        auto else_seen = seen;
        AnalyzeScope(if_stmt->else_body_.value(), else_seen);
      }
      const auto& written_then = PrecomputeWrittenRoots(if_stmt->then_body_);
      seen.insert(written_then.begin(), written_then.end());
      if (if_stmt->else_body_.has_value() && if_stmt->else_body_.value()) {
        const auto& written_else = PrecomputeWrittenRoots(if_stmt->else_body_.value());
        seen.insert(written_else.begin(), written_else.end());
      }
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      AnalyzeScope(scope->body_, seen);
    } else if (auto assign = As<AssignStmt>(stmt)) {
      AnalyzeCall(assign->value_, seen);
    } else if (auto eval = As<EvalStmt>(stmt)) {
      AnalyzeCall(eval->expr_, seen);
    }
    // Other stmts (Yield/Return/Break/Continue): no Calls to analyze.
  }

  /// For a single Call expression, mark "first writer" roots and update `seen`.
  void AnalyzeCall(const ExprPtr& expr, std::unordered_set<const Var*>& seen) {
    auto call = As<Call>(expr);
    if (!call) return;
    if (IsBuiltinOp(call->op_->name_)) return;
    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) return;

    auto dirs = ResolveCalleeDirections(program_, call, callee);
    std::unordered_set<const Var*> roots_this_call;
    for (size_t i = 0; i < dirs.size() && i < call->args_.size(); ++i) {
      // Only Out is decision-relevant for promotion (InOut is already InOut).
      // We still register InOut roots into `roots_this_call` so subsequent
      // siblings see them as prior writers.
      if (dirs[i] != ParamDirection::Out && dirs[i] != ParamDirection::InOut) continue;
      const Var* root = ResolveAnyRoot(call->args_[i], buffer_roots_);
      if (!root) continue;
      if (dirs[i] == ParamDirection::Out && seen.count(root) == 0) {
        first_writer_roots[call.get()].insert(root);
      }
      roots_this_call.insert(root);
    }
    seen.insert(roots_this_call.begin(), roots_this_call.end());
  }

  ProgramPtr program_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  std::unordered_map<const Stmt*, std::unordered_set<const Var*>> written_roots_;
};

/// IRMutator that rewrites every non-builtin Call in a function body and writes
/// the per-argument ArgDirection vector based on callee param directions, the
/// pre-computed buffer-root map, and prior-writer / sequential-context analysis.
///
/// Promotion rules for callee Out (apply uniformly to local and enclosing-param roots):
///   - R-seq:        any sequential ancestor (For{Sequential,Unroll,Pipeline} or While) → InOut
///   - R-prior:      a prior writer-unit in the same scope wrote to the same root      → InOut
///   - R-enclosing:  the root is the enclosing function's param and that param is
///                   declared InOut by the user                                         → InOut
///   - default:      OutputExisting (write into a pre-allocated buffer that the
///                   runtime treats as an output slot, no extra dependency edge
///                   introduced).
///
/// R-seq is applied unconditionally: any callee Out under a sequential ancestor
/// is promoted to InOut. A prior "disjoint variable-offset store" exception was
/// removed — proving cross-iteration writes are disjoint requires a sound
/// dependence analysis (affine offset extraction, stride-vs-tile-extent, offset
/// injectivity, cross-procedural composition) that the cheap syntactic check did
/// not perform, so it could silently drop real WAW edges.
class CallDirectionMutator : public IRMutator {
 public:
  CallDirectionMutator(
      ProgramPtr program, const std::unordered_map<const Var*, const Var*>& buffer_roots,
      const std::unordered_map<const Call*, std::unordered_set<const Var*>>& first_writer_roots,
      const std::unordered_map<const Var*, ParamDirection>& enclosing_param_dir_by_root)
      : program_(std::move(program)),
        buffer_roots_(buffer_roots),
        first_writer_roots_(first_writer_roots),
        enclosing_param_dir_by_root_(enclosing_param_dir_by_root) {}

 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    bool is_sequential = op->kind_ != ForKind::Parallel;
    if (is_sequential) {
      ++sequential_depth_;
    }
    auto out = IRMutator::VisitStmt_(op);
    if (is_sequential) {
      --sequential_depth_;
    }
    return out;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    ++sequential_depth_;
    auto out = IRMutator::VisitStmt_(op);
    --sequential_depth_;
    return out;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    // First descend so nested Calls also get arg_directions assigned.
    auto base = IRMutator::VisitExpr_(op);
    auto call = As<Call>(base);
    if (!call) return base;

    if (IsBuiltinOp(call->op_->name_)) {
      return call;
    }

    auto callee = program_ ? program_->GetFunction(call->op_->name_) : nullptr;
    if (!callee) {
      // Unknown op (e.g. Opaque function not in program). Leave directions empty.
      return call;
    }

    auto effective = ResolveCalleeDirections(program_, call, callee);
    if (effective.size() != call->args_.size()) {
      // Safety: if the length disagrees we can't produce a sound mapping.
      // Leave directions empty so the verify pass surfaces a clear error.
      return call;
    }

    // Respect explicit call-site directions. The Call constructor's
    // ValidateArgDirectionsAttr already enforces size == args_.size(), and
    // some directions (e.g. NoDep) are not derivable here, so a populated
    // attrs['arg_directions'] is treated as authoritative and left as-is.
    if (call->HasArgDirections()) {
      return call;
    }

    // Look up first-writer info computed against the *original* Call object.
    // The mutator may produce a new shared_ptr above (when nested Calls are
    // rewritten), but the prior-writer collector keyed on the original op.
    const Call* original_call = op.get();
    auto fw_it = first_writer_roots_.find(original_call);
    const std::unordered_set<const Var*>* first_writer_set =
        fw_it != first_writer_roots_.end() ? &fw_it->second : nullptr;

    std::vector<ArgDirection> dirs;
    dirs.reserve(call->args_.size());
    for (size_t i = 0; i < call->args_.size(); ++i) {
      const auto& arg = call->args_[i];
      bool is_tensor = IsTensorTypedArg(arg);
      if (!is_tensor) {
        dirs.push_back(ArgDirection::Scalar);
        continue;
      }

      ParamDirection cd = effective[i];
      if (cd == ParamDirection::In) {
        dirs.push_back(ArgDirection::Input);
      } else if (cd == ParamDirection::InOut) {
        dirs.push_back(ArgDirection::InOut);
      } else {
        // ParamDirection::Out — apply the promotion rules uniformly to both
        // locally-allocated roots and roots that trace back to an enclosing
        // function parameter.
        const Var* root = ResolveAnyRoot(arg, buffer_roots_);

        // R-seq: any sequential ancestor forces InOut to keep cross-iteration
        // WAW chains correct. Applied unconditionally — see the class comment
        // for why the prior "disjoint variable-offset store" exception was
        // removed.
        if (sequential_depth_ > 0) {
          dirs.push_back(ArgDirection::InOut);
          continue;
        }
        // R-prior: a prior writer-unit in this scope already wrote to this root → InOut.
        if (root) {
          bool is_first_writer = first_writer_set != nullptr && first_writer_set->count(root) > 0;
          if (!is_first_writer) {
            dirs.push_back(ArgDirection::InOut);
            continue;
          }
        }
        // R-enclosing: if the root is an enclosing function param that the user
        // declared InOut, honor that declaration — the function effectively reads
        // the prior-call value and writes a new one back into the same buffer.
        if (root) {
          auto it = enclosing_param_dir_by_root_.find(root);
          if (it != enclosing_param_dir_by_root_.end() && it->second == ParamDirection::InOut) {
            dirs.push_back(ArgDirection::InOut);
            continue;
          }
        }
        // Default: first writer, no sequential ancestor, no InOut declaration → OutputExisting.
        dirs.push_back(ArgDirection::OutputExisting);
      }
    }

    // Apply user-specified per-arg overrides (e.g. pl.no_dep(...) at call site).
    // Stored as a vector<int32_t> of arg indices that should resolve to NoDep,
    // overriding the auto-derived direction at those slots.
    for (const auto& [k, v] : call->attrs_) {
      if (k != kAttrArgDirectionOverrides) continue;
      const auto* indices = std::any_cast<std::vector<int32_t>>(&v);
      if (!indices) {
        INTERNAL_CHECK_SPAN(false, call->span_)
            << "Internal error: " << kAttrArgDirectionOverrides << " attr must hold std::vector<int32_t>";
      }
      for (int32_t idx : *indices) {
        INTERNAL_CHECK_SPAN(idx >= 0 && static_cast<size_t>(idx) < dirs.size(), call->span_)
            << "Internal error: arg_direction_overrides index " << idx << " out of range for call to '"
            << call->op_->name_ << "' (args size " << call->args_.size() << ")";
        dirs[static_cast<size_t>(idx)] = ArgDirection::NoDep;
      }
      break;
    }

    // Skip rewriting if directions are unchanged.
    if (call->GetArgDirections() == dirs) {
      return call;
    }

    auto new_attrs = WithArgDirectionsAttr(call->attrs_, std::move(dirs));
    return std::make_shared<const Call>(call->op_, call->args_, call->kwargs_, std::move(new_attrs),
                                        call->GetType(), call->span_);
  }

 private:
  ProgramPtr program_;
  const std::unordered_map<const Var*, const Var*>& buffer_roots_;
  const std::unordered_map<const Call*, std::unordered_set<const Var*>>& first_writer_roots_;
  const std::unordered_map<const Var*, ParamDirection>& enclosing_param_dir_by_root_;
  int sequential_depth_ = 0;
};

}  // namespace

namespace pass {

Pass DeriveCallDirections() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    if (!program) return program;

    // We need a non-const handle to rewrite functions with new bodies.
    auto new_functions = program->functions_;

    for (auto& [gvar, func] : new_functions) {
      if (!func || !func->body_) continue;

      BufferRootCollector br_collector(program);
      br_collector.Initialize(func->params_);
      br_collector.VisitStmt(func->body_);

      // Build a Var* → ParamDirection map for the enclosing function's params,
      // so call sites can honor an explicit ``pl.InOut`` declaration when the
      // arg traces back to such a param via the buffer-root map.
      std::unordered_map<const Var*, ParamDirection> enclosing_param_dir_by_root;
      enclosing_param_dir_by_root.reserve(func->params_.size());
      for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i) {
        enclosing_param_dir_by_root.emplace(func->params_[i].get(), func->param_directions_[i]);
      }

      PriorWriterCollector pw_collector(program, br_collector.buffer_roots);
      pw_collector.Run(func->body_);

      CallDirectionMutator mutator(program, br_collector.buffer_roots, pw_collector.first_writer_roots,
                                   enclosing_param_dir_by_root);
      auto new_body = mutator.VisitStmt(func->body_);

      if (new_body.get() == func->body_.get()) continue;
      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                        func->return_types_, new_body, func->span_, func->func_type_,
                                        func->level_, func->role_, func->attrs_);
    }

    if (new_functions == program->functions_) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "DeriveCallDirections", kDeriveCallDirectionsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
