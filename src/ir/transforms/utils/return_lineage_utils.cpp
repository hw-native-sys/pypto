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

#include "pypto/ir/transforms/utils/return_lineage_utils.h"

#include <algorithm>
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
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace return_lineage {

namespace {

using op_predicates::IsBuiltinOp;

// Borrow the original operands without allocating a Call view for each tuple
// projection. Submit keeps its original compact argument list and result type.
struct CallLikeView {
  OpPtr op;
  const std::vector<ExprPtr>* args = nullptr;
};

CallLikeView AsCallLike(const ExprPtr& expr) {
  if (auto call = As<Call>(expr)) return {call->op_, &call->args_};
  if (auto submit = As<Submit>(expr)) return {submit->op_, &submit->args_};
  return {};
}

// Return normalization is a value-replacement policy, not storage lineage.
// Preserve its established destination-returning builtins explicitly: adding a
// storage alias (especially a slice or reshape) must not authorize `return p`.
std::optional<size_t> ReturnedBuiltinArgIndex(const OpPtr& op, size_t arg_count) {
  auto index = op_predicates::StorageAliasArgIndex(op, arg_count);
  if (!index) return std::nullopt;
  if (IsOp(op, "tile.store") || IsOp(op, "tile.mscatter")) return 2;
  if (IsOp(op, "tile.set_validshape") || IsOp(op, "tensor.assemble") || IsOp(op, "tensor.set_validshape") ||
      IsOp(op, "tile.batch_matmul_acc") || IsOp(op, "tile.fillpad_inplace") || IsOp(op, "tile.gather_row") ||
      IsOp(op, "tile.gemv_acc") || IsOp(op, "tile.matmul_acc") || IsOp(op, "tile.matmul_mx_acc") ||
      IsOp(op, "tile.scatter") || IsOp(op, "tile.scatter_mask") || IsOp(op, "tile.scatter_update") ||
      IsOp(op, "tile.tget_scale_addr")) {
    return 0;
  }
  return std::nullopt;
}

using ParamIndexMap = std::unordered_map<const Var*, size_t>;

ParamIndexMap IndexParams(const std::vector<VarPtr>& params) {
  ParamIndexMap result;
  result.reserve(params.size());
  for (size_t i = 0; i < params.size(); ++i) result.emplace(params[i].get(), i);
  return result;
}

// Per-body index: topmost ReturnStmt, per-var defining AssignStmt, loop-carry
// edges (iter_arg / tensor return_var -> init value var), and optionally tensor
// IfStmt merge edges. Return queries and scope outlining share branch consensus.
class BodyIndexCollector : public IRVisitor {
 public:
  explicit BodyIndexCollector(bool collect_if_merges = false) : collect_if_merges_(collect_if_merges) {}

  ReturnStmtPtr first_return;
  std::unordered_map<const Var*, AssignStmtPtr> var_def;
  std::unordered_map<const Var*, const Var*> carry_src;
  std::unordered_map<const Var*, std::vector<const Var*>> merge_srcs;

 protected:
  void VisitStmt_(const ReturnStmtPtr& ret) override {
    if (!first_return) first_return = ret;
  }
  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (assign->var_) var_def.emplace(assign->var_.get(), assign);
    IRVisitor::VisitStmt_(assign);
  }
  void VisitStmt_(const ForStmtPtr& for_stmt) override {
    RecordCarries(for_stmt->iter_args_, for_stmt->return_vars_);
    IRVisitor::VisitStmt_(for_stmt);
  }
  void VisitStmt_(const WhileStmtPtr& while_stmt) override {
    RecordCarries(while_stmt->iter_args_, while_stmt->return_vars_);
    IRVisitor::VisitStmt_(while_stmt);
  }
  void VisitStmt_(const IfStmtPtr& if_stmt) override {
    if (collect_if_merges_) RecordMergeSources(if_stmt);
    IRVisitor::VisitStmt_(if_stmt);
  }

 private:
  void RecordCarries(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars) {
    for (size_t i = 0; i < iter_args.size(); ++i) {
      auto init_var = AsVarLike(iter_args[i]->initValue_);
      if (!init_var) continue;
      carry_src[iter_args[i].get()] = init_var.get();
      // Scalar carries are value-typed: the body may overwrite them with a
      // value unrelated to the init param, so only Tensor carries propagate.
      if (i < return_vars.size() && AsTensorTypeLike(return_vars[i]->GetType())) {
        carry_src[return_vars[i].get()] = init_var.get();
      }
    }
  }

  void RecordMergeSources(const IfStmtPtr& if_stmt) {
    if (!if_stmt->else_body_.has_value()) return;
    auto then_yield = transform_utils::GetLastYieldStmt(if_stmt->then_body_);
    auto else_yield = transform_utils::GetLastYieldStmt(*if_stmt->else_body_);
    if (!then_yield || !else_yield) return;

    size_t count =
        std::min({if_stmt->return_vars_.size(), then_yield->value_.size(), else_yield->value_.size()});
    for (size_t i = 0; i < count; ++i) {
      const auto& return_var = if_stmt->return_vars_[i];
      if (!return_var || !AsTensorTypeLike(return_var->GetType())) continue;
      auto then_var = AsVarLike(then_yield->value_[i]);
      auto else_var = AsVarLike(else_yield->value_[i]);
      if (!then_var || !else_var) continue;
      merge_srcs.emplace(return_var.get(), std::vector<const Var*>{then_var.get(), else_var.get()});
    }
  }

  bool collect_if_merges_;
};

// Cross-function memo + cycle guard, shared by the recursive tracer. The
// Results depend on the callee and its callees in an unchanged Program. The
// thread_local cache bounds work to one body walk per distinct function reached
// from the outermost function, batched query, or explicit cache scope.
struct TraceContext {
  std::vector<const Function*> call_stack;
  std::unordered_set<const Function*> active_functions;
  std::unordered_map<const Function*, std::vector<std::optional<size_t>>> memo;
  std::unordered_map<const Function*, size_t> trailing_ctx_counts;
  size_t query_depth = 0;

  void ClearCache() {
    memo.clear();
    trailing_ctx_counts.clear();
  }
};

TraceContext& Ctx() {
  thread_local TraceContext ctx;
  return ctx;
}

// Keep callee maps alive while a batch traces several independent Vars. A
// nested function query must not clear maps still needed by the outer batch.
using TraceQueryGuard = ScopedTraceCache;

// Map a callee parameter back to its actual argument using Submit::args_'s
// prefix / omitted-Out gap / CommCtx suffix contract. Call's full coverage is
// the gap==0 case; wrapper Call views use this same compact-list mapping.
ExprPtr ParamArgument(const CallLikeView& call, const FunctionPtr& callee, size_t param_index) {
  if (!call.args || param_index >= callee->params_.size()) return nullptr;
  const auto& args = *call.args;
  auto& counts = Ctx().trailing_ctx_counts;
  auto [it, inserted] = counts.try_emplace(callee.get(), 0);
  if (inserted) it->second = transform_utils::TrailingCommCtxParamCount(callee);
  const size_t ctx_count = it->second;
  if (args.size() < ctx_count || args.size() > callee->params_.size()) return nullptr;
  const size_t prefix = args.size() - ctx_count;
  if (param_index < prefix) return args[param_index];
  if (param_index < callee->params_.size() - ctx_count) return nullptr;
  return args[param_index - (callee->params_.size() - args.size())];
}

std::vector<std::optional<size_t>> ReturnedParamIndicesImpl(const FunctionPtr& func,
                                                            const ProgramPtr& program);

// Avoid copying an entire tuple return map for each TupleGetItem in a batch.
// TraceQueryGuard or the enclosing function query owns the reference lifetime.
const std::vector<std::optional<size_t>>& CachedReturnedParamIndices(const FunctionPtr& func,
                                                                     const ProgramPtr& program) {
  static const std::vector<std::optional<size_t>> empty;
  if (!func || !func->body_) return empty;
  auto& ctx = Ctx();
  if (ctx.active_functions.count(func.get()) != 0) {
    return empty;
  }
  if (auto it = ctx.memo.find(func.get()); it != ctx.memo.end()) {
    return it->second;
  }
  auto result = ReturnedParamIndicesImpl(func, program);
  return ctx.memo.emplace(func.get(), std::move(result)).first->second;
}

struct VarLineage {
  const Var* root = nullptr;
  TypePtr returned_type;
};
using VarRootMap = std::unordered_map<const Var*, VarLineage>;
using TupleSourceMap = std::unordered_map<const Expr*, ExprPtr>;

struct LineageMemo {
  VarRootMap var_roots;
  std::unordered_map<const Expr*, VarLineage> expr_roots;
  TupleSourceMap tuple_sources;
};

ExprPtr ResolveTupleSource(ExprPtr expr, const BodyIndexCollector& index, const ProgramPtr& program,
                           TupleSourceMap& memo);

// Select an original expression, rather than inventing an SSA binding. This
// lets direct projections and assigned projections share the same lineage.
ExprPtr ResolveTupleElement(const ExprPtr& tuple, int64_t position, const BodyIndexCollector& index,
                            const ProgramPtr& program, TupleSourceMap& memo) {
  if (position < 0) return nullptr;
  auto source = ResolveTupleSource(tuple, index, program, memo);
  if (!source) return nullptr;
  size_t pos = static_cast<size_t>(position);
  if (auto make_tuple = As<MakeTuple>(source)) {
    return pos < make_tuple->elements_.size() ? make_tuple->elements_[pos] : nullptr;
  }

  auto call = AsCallLike(source);
  if (!call.args || IsBuiltinOp(call.op->name_)) return nullptr;
  auto callee = program ? program->GetFunction(call.op->name_) : nullptr;
  if (!callee) return nullptr;
  // A single TupleType return has one return position, not one position per
  // tuple element. Its nested element ABI is outside the flat return map.
  if (callee->return_types_.size() == 1 && As<TupleType>(callee->return_types_[0])) return nullptr;
  const auto& ret_map = CachedReturnedParamIndices(callee, program);
  if (pos >= ret_map.size() || !ret_map[pos]) return nullptr;
  size_t mapped = ret_map[pos].value();  // NOLINT(bugprone-unchecked-optional-access)
  return ParamArgument(call, callee, mapped);
}

// Tuple aliases and nested projections can share arbitrarily long prefixes.
// Compress each source chain once, including failures and provisional cycle
// entries, so querying many projections never rescans their common prefix.
ExprPtr ResolveTupleSource(ExprPtr expr, const BodyIndexCollector& index, const ProgramPtr& program,
                           TupleSourceMap& memo) {
  std::vector<const Expr*> path;
  auto finish = [&](const ExprPtr& source) {
    for (const auto* item : path) memo[item] = source;
    return source;
  };
  while (expr) {
    if (auto it = memo.find(expr.get()); it != memo.end()) {
      return finish(it->second);
    }
    memo.emplace(expr.get(), nullptr);
    path.push_back(expr.get());
    if (auto var = AsVarLike(expr)) {
      auto def = index.var_def.find(var.get());
      if (def == index.var_def.end()) return finish(nullptr);
      expr = def->second->value_;
    } else if (auto projection = As<TupleGetItemExpr>(expr)) {
      expr = ResolveTupleElement(projection->tuple_, projection->index_, index, program, memo);
    } else {
      return finish(expr);
    }
  }
  return finish(nullptr);
}

VarLineage TraceExpr(const ExprPtr& expr, const BodyIndexCollector& index, const ParamIndexMap& param_indices,
                     const ProgramPtr& program, LineageMemo& memo);

// Trace each body var once, caching both roots and failures across branch arms
// and return values. A provisional empty lineage stops cycles without copying
// the active path at every merge. Alias chains remain iterative.
VarLineage TraceVar(const Var* var, const BodyIndexCollector& index, const ParamIndexMap& param_indices,
                    const ProgramPtr& program, LineageMemo& memo) {
  // Aliases and validated user calls forward their source type. Builtin
  // writebacks retain their own final type, allowing intermediate narrowing
  // followed by restoration without poisoning the cached storage lineage.
  auto& roots = memo.var_roots;
  std::vector<std::pair<const Var*, bool>> path;
  auto finish = [&](const Var* root, TypePtr returned_type = nullptr) {
    VarLineage lineage{root, std::move(returned_type)};
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      if (!it->second) lineage.returned_type = it->first->GetType();
      roots[it->first] = lineage;
    }
    return lineage;
  };
  while (var) {
    if (auto it = roots.find(var); it != roots.end()) {
      return finish(it->second.root, it->second.returned_type);
    }
    roots.emplace(var, VarLineage{});
    path.emplace_back(var, false);
    if (param_indices.count(var)) return finish(var);

    if (auto merge_it = index.merge_srcs.find(var); merge_it != index.merge_srcs.end()) {
      const Var* root = nullptr;
      TypePtr returned_type;
      for (const Var* source : merge_it->second) {
        auto branch = TraceVar(source, index, param_indices, program, memo);
        if (!branch.root || (root && branch.root != root)) return finish(nullptr);
        if (!root) {
          returned_type = branch.returned_type;
        } else if (!returned_type || !branch.returned_type ||
                   !structural_equal(returned_type, branch.returned_type)) {
          returned_type = nullptr;
        }
        root = branch.root;
      }
      // Keep storage consensus through narrowed merges; an outer writeback may
      // restore the final type. Inconsistent branch metadata stays unknown.
      path.back().second = true;
      return finish(root, std::move(returned_type));
    }

    if (auto carry_it = index.carry_src.find(var); carry_it != index.carry_src.end()) {
      var = carry_it->second;
      continue;
    }

    auto def_it = index.var_def.find(var);
    if (def_it == index.var_def.end()) return finish(nullptr);
    const auto& value = def_it->second->value_;

    if (auto rhs_var = AsVarLike(value)) {
      path.back().second = true;
      var = rhs_var.get();
      continue;
    }

    if (auto projection = As<TupleGetItemExpr>(value)) {
      path.back().second = true;
      auto selected =
          ResolveTupleElement(projection->tuple_, projection->index_, index, program, memo.tuple_sources);
      if (auto source_var = AsVarLike(selected)) {
        var = source_var.get();
        continue;
      }
      auto lineage = TraceExpr(selected, index, param_indices, program, memo);
      return finish(lineage.root, lineage.returned_type);
    }

    if (auto call = AsCallLike(value); call.args) {
      const auto& args = *call.args;
      const std::string& op_name = call.op->name_;
      // Builtin output-side ops bind a fresh SSA var to an existing buffer, so
      // the result inherits that argument's param lineage.
      if (auto aliased = ReturnedBuiltinArgIndex(call.op, args.size())) {
        auto arg_var = AsVarLike(args[*aliased]);
        if (!arg_var) {
          auto lineage = TraceExpr(args[*aliased], index, param_indices, program, memo);
          return finish(lineage.root, lineage.returned_type);
        }
        var = arg_var.get();
        continue;
      }
      // Any other builtin stops the walk — including buffer-aliasing views
      // (`tensor.view`, `tile.slice`, ...). A view lands in its source's buffer
      // but carries a different shape / tensor_view, so `NormalizeReturnOrder`
      // cannot use it: rewriting `return view(p, [1, 17])` to `return p` would
      // return the wrong shape.
      //
      // Consumers must not read that stop as "this value is unrelated to any
      // param". Two of them care about different things than the rewrite does,
      // and views are imprecise for both:
      //   - `ReturnParamsExplicit` treats a nullopt position as a fresh
      //     kernel-allocated tensor. A view-shaped writeback is not fresh — it
      //     already lives in the param's buffer.
      //   - identity lineage (which comm domain a DistributedTensor belongs to)
      //     *does* follow views, because the view propagates `window_buffer_`
      //     verbatim; `MaterializeDistTensorCtx` layers that on top of this walk.
      // Splitting writeback lineage from identity lineage would fix both; until
      // then, downstream buffer identity is recovered by the MemRef layer.
      if (!IsBuiltinOp(op_name)) {
        // Single-result user call: continue from the arg the callee returns.
        auto callee = program ? program->GetFunction(op_name) : nullptr;
        if (!callee) return finish(nullptr);
        const auto& ret_map = CachedReturnedParamIndices(callee, program);
        if (ret_map.size() != 1 || !ret_map[0]) return finish(nullptr);
        size_t mapped = ret_map[0].value();  // NOLINT(bugprone-unchecked-optional-access)
        auto source = ParamArgument(call, callee, mapped);
        auto arg_var = AsVarLike(source);
        path.back().second = true;
        if (!arg_var) {
          auto lineage = TraceExpr(source, index, param_indices, program, memo);
          return finish(lineage.root, lineage.returned_type);
        }
        var = arg_var.get();
        continue;
      }
      return finish(nullptr);
    }
    return finish(nullptr);
  }
  return finish(nullptr);
}

VarLineage TraceExpr(const ExprPtr& expr, const BodyIndexCollector& index, const ParamIndexMap& param_indices,
                     const ProgramPtr& program, LineageMemo& memo) {
  if (!expr) return {};
  if (auto var = AsVarLike(expr)) return TraceVar(var.get(), index, param_indices, program, memo);
  if (auto it = memo.expr_roots.find(expr.get()); it != memo.expr_roots.end()) {
    return it->second;
  }
  memo.expr_roots.emplace(expr.get(), VarLineage{});
  VarLineage lineage;
  if (auto projection = As<TupleGetItemExpr>(expr)) {
    auto selected =
        ResolveTupleElement(projection->tuple_, projection->index_, index, program, memo.tuple_sources);
    lineage = TraceExpr(selected, index, param_indices, program, memo);
  }
  memo.expr_roots[expr.get()] = lineage;
  return lineage;
}

std::vector<std::optional<size_t>> TraceIndexedExprsToParamIndices(const std::vector<ExprPtr>& exprs,
                                                                   const BodyIndexCollector& index,
                                                                   const ParamIndexMap& param_indices,
                                                                   const ProgramPtr& program) {
  TraceQueryGuard query_guard;
  std::vector<std::optional<size_t>> result;
  result.reserve(exprs.size());
  LineageMemo memo;
  for (const auto& expr : exprs) {
    std::optional<size_t> idx;
    if (expr) {
      auto lineage = TraceExpr(expr, index, param_indices, program, memo);
      if (lineage.root &&
          (!AsTensorTypeLike(expr->GetType()) ||
           (lineage.returned_type && structural_equal(lineage.returned_type, lineage.root->GetType())))) {
        auto param_it = param_indices.find(lineage.root);
        if (param_it != param_indices.end()) idx = param_it->second;
      }
    }
    result.push_back(idx);
  }
  return result;
}

// Expand a return that forwards a multi-result call's *whole tuple*:
//
//     result = self.inner(...)   # TupleType, N results
//     return result              # ONE return expr, but N return positions
//
// This is the shape a Group/Spmd wrapper takes when the inner kernel has more
// than one result (the outliner emits the call, binds its tuple to one SSA var,
// and returns that var). ``TraceVar`` cannot describe it: it resolves a var to a
// *single* param root, and its user-call branch deliberately bails on a
// multi-result callee (``ret_map.size() != 1``). Without this expansion the
// wrapper's map comes back as a bogus ``[nullopt]``, every caller that
// destructures the wrapper sees an imprecise map, and they all fall back to the
// legacy tail-alignment heuristic -- which shifts each returned element onto the
// wrong Out/InOut param whenever the callee writes an Out/InOut param it does
// not return (accumulators, ``__gm_pipe_buffer``, an in-place-written KV cache).
// That is issue #1573 resurfacing on exactly the Group/Spmd wrappers its fix
// left on the heuristic.
//
// So resolve the tuple var's defining call, take the callee's own returned-param
// map, and re-map each position through the call's args back onto ``params``.
// Returns {} when the shape is not a recognisable forward; callers then keep
// their previous behaviour.
std::vector<std::optional<size_t>> ExpandForwardedTupleVar(const Var* tuple_var,
                                                           const BodyIndexCollector& index,
                                                           const ParamIndexMap& param_indices,
                                                           const ProgramPtr& program) {
  // Walk SSA var-to-var aliases down to the statement that defines the tuple.
  const Var* var = tuple_var;
  std::unordered_set<const Var*> seen;
  CallLikeView call;
  while (var) {
    if (!seen.insert(var).second) return {};
    auto def_it = index.var_def.find(var);
    if (def_it == index.var_def.end()) return {};
    const auto& value = def_it->second->value_;
    if (auto rhs_var = AsVarLike(value)) {
      var = rhs_var.get();
      continue;
    }
    // A literal tuple construction: each element traces on its own.
    if (auto make_tuple = As<MakeTuple>(value)) {
      return TraceIndexedExprsToParamIndices(make_tuple->elements_, index, param_indices, program);
    }
    call = AsCallLike(value);
    break;
  }
  if (!call.args || IsBuiltinOp(call.op->name_)) return {};

  auto callee = program ? program->GetFunction(call.op->name_) : nullptr;
  if (!callee) return {};
  auto inner_map = ReturnedParamIndicesImpl(callee, program);
  if (inner_map.empty()) return {};

  std::vector<ExprPtr> forwarded_args(inner_map.size());
  for (size_t pos = 0; pos < inner_map.size(); ++pos) {
    if (!inner_map[pos]) continue;
    size_t mapped = inner_map[pos].value();  // NOLINT(bugprone-unchecked-optional-access)
    forwarded_args[pos] = ParamArgument(call, callee, mapped);
  }
  // Args may themselves be assemble / loop-carry chains, so trace them through
  // the same batch path as ordinary returns.
  return TraceIndexedExprsToParamIndices(forwarded_args, index, param_indices, program);
}

// Group/Spmd wrappers may end in the inner kernel call with no top-level
// ReturnStmt; their return values are the inner call's by construction. Map
// each inner return position to the wrapper param passed as the matching arg.
std::vector<std::optional<size_t>> MapWrapperReturnToParams(const FunctionPtr& wrapper,
                                                            const ProgramPtr& program) {
  if (!program || wrapper->return_types_.empty()) return {};
  CallPtr returning_call;
  FunctionPtr returning_callee;
  for (const auto& info : CollectInnerCalls(wrapper, program)) {
    if (info.inner_callee->return_types_.size() != wrapper->return_types_.size()) continue;
    if (returning_call) return {};  // ambiguous
    returning_call = info.inner_call;
    returning_callee = info.inner_callee;
  }
  if (!returning_call) return {};

  auto inner_map = ReturnedParamIndicesImpl(returning_callee, program);
  if (inner_map.size() != wrapper->return_types_.size()) return {};

  std::unordered_map<const Var*, size_t> param_to_index;
  for (size_t i = 0; i < wrapper->params_.size(); ++i) param_to_index[wrapper->params_[i].get()] = i;

  std::vector<std::optional<size_t>> result(inner_map.size());
  for (size_t pos = 0; pos < inner_map.size(); ++pos) {
    if (!inner_map[pos]) continue;
    size_t mapped = inner_map[pos].value();  // NOLINT(bugprone-unchecked-optional-access)
    if (auto arg_var = AsVarLike(ParamArgument(AsCallLike(returning_call), returning_callee, mapped))) {
      if (auto it = param_to_index.find(arg_var.get()); it != param_to_index.end()) {
        result[pos] = it->second;
      }
    }
  }
  return result;
}

std::vector<std::optional<size_t>> ReturnedParamIndicesImpl(const FunctionPtr& func,
                                                            const ProgramPtr& program) {
  if (!func || !func->body_) return {};

  auto& ctx = Ctx();
  if (ctx.active_functions.count(func.get()) != 0) {
    return {};
  }
  if (auto it = ctx.memo.find(func.get()); it != ctx.memo.end()) {
    return it->second;
  }

  ctx.call_stack.push_back(func.get());
  ctx.active_functions.insert(func.get());
  bool is_top_level = ctx.call_stack.size() == 1;
  struct Guard {
    TraceContext& ctx;
    const Function* function;
    bool top_level;
    ~Guard() {
      ctx.active_functions.erase(function);
      ctx.call_stack.pop_back();
      if (top_level && ctx.query_depth == 0) ctx.ClearCache();
    }
  } guard{ctx, func.get(), is_top_level};

  auto record = [&](std::vector<std::optional<size_t>> result) {
    if (!is_top_level || ctx.query_depth != 0) ctx.memo.emplace(func.get(), result);
    return result;
  };

  // Graph boundary analyses do not propagate tensor roots through If merges yet.
  BodyIndexCollector index(/*collect_if_merges=*/func->func_type_ != FunctionType::Graph);
  index.VisitStmt(func->body_);
  if (!index.first_return || index.first_return->value_.empty()) {
    if (IsWrapperType(func->func_type_)) {
      return record(MapWrapperReturnToParams(func, program));
    }
    return record({});
  }
  // A wrapper whose single return value is the forwarded tuple of a multi-result
  // inner call carries N return positions in ONE return expr. Expand it so the
  // map stays precise; fall through to the per-expr tracer when it is not that
  // shape, so this can only ever *add* precision.
  //
  // The expansion is only well-formed when the function really does declare N
  // flat return positions (``return_types_.size() == N``). A function that
  // declares a single ``pl.Tuple[T1, ..., TN]`` return has ONE return type -- a
  // TupleType -- and returns the tuple as one value. Expanding that would hand
  // every consumer an N-entry map for a 1-arity return, and CanonicalizeReturnValues
  // would rewrite ``return t`` into N param returns that the one-entry
  // ``return_types_`` cannot describe.
  const auto& rets = index.first_return->value_;
  const size_t declared_arity = func->return_types_.size();
  auto param_indices = IndexParams(func->params_);
  if (rets.size() == 1 && declared_arity > 1) {
    if (auto tuple_var = AsVarLike(rets[0])) {
      if (auto tuple_ty = As<TupleType>(tuple_var->GetType());
          tuple_ty && tuple_ty->types_.size() == declared_arity) {
        auto expanded = ExpandForwardedTupleVar(tuple_var.get(), index, param_indices, program);
        if (expanded.size() == declared_arity) return record(expanded);
      }
    }
  }
  return record(TraceIndexedExprsToParamIndices(rets, index, param_indices, program));
}

std::vector<std::optional<size_t>> TraceBodyExprsToParamIndices(const std::vector<ExprPtr>& exprs,
                                                                const StmtPtr& body,
                                                                const std::vector<VarPtr>& params,
                                                                const ProgramPtr& program,
                                                                bool trace_if_merges) {
  if (!body) return std::vector<std::optional<size_t>>(exprs.size());

  BodyIndexCollector index(trace_if_merges);
  index.VisitStmt(body);
  return TraceIndexedExprsToParamIndices(exprs, index, IndexParams(params), program);
}

}  // namespace

ScopedTraceCache::ScopedTraceCache() { ++Ctx().query_depth; }

ScopedTraceCache::~ScopedTraceCache() {
  auto& ctx = Ctx();
  --ctx.query_depth;
  if (ctx.query_depth == 0 && ctx.call_stack.empty()) ctx.ClearCache();
}

// Locate the topmost ReturnStmt. Fast path: a function body is normally a
// SeqStmts whose last statement is the return. Split AIV kernels keep theirs
// nested inside the split body, so fall back to a walk.
ReturnStmtPtr FindFirstReturn(const StmtPtr& body) {
  if (auto seq = As<SeqStmts>(body); seq && !seq->stmts_.empty()) {
    if (auto ret = As<ReturnStmt>(seq->stmts_.back())) return ret;
  }
  class FirstReturnFinder : public IRVisitor {
   public:
    ReturnStmtPtr first_return;

   protected:
    void VisitStmt_(const ReturnStmtPtr& ret) override {
      if (!first_return) first_return = ret;
    }
  };
  FirstReturnFinder finder;
  finder.VisitStmt(body);
  return finder.first_return;
}

std::vector<std::optional<size_t>> ExplicitReturnedParamIndices(const FunctionPtr& func) {
  if (!func || !func->body_) return {};
  auto ret = FindFirstReturn(func->body_);
  if (!ret || ret->value_.empty()) return {};

  std::unordered_map<const Var*, size_t> param_to_index;
  for (size_t i = 0; i < func->params_.size(); ++i) param_to_index.emplace(func->params_[i].get(), i);

  std::vector<std::optional<size_t>> result;
  result.reserve(ret->value_.size());
  for (const auto& value : ret->value_) {
    std::optional<size_t> idx;
    if (auto var = AsVarLike(value)) {
      auto it = param_to_index.find(var.get());
      if (it != param_to_index.end()) idx = it->second;
    }
    result.push_back(idx);
  }
  return result;
}

std::optional<size_t> ExplicitReturnedParamIndex(const FunctionPtr& func) {
  auto indices = ExplicitReturnedParamIndices(func);
  return indices.empty() ? std::nullopt : indices[0];
}

std::optional<size_t> ReturnedParamIndex(const FunctionPtr& func, const ProgramPtr& program) {
  auto indices = ReturnedParamIndicesImpl(func, program);
  return indices.empty() ? std::nullopt : indices[0];
}

std::vector<std::optional<size_t>> ReturnedParamIndices(const FunctionPtr& func, const ProgramPtr& program) {
  return ReturnedParamIndicesImpl(func, program);
}

std::vector<ReturnedValueBinding> AnalyzeReturnedValues(const FunctionPtr& func, const ProgramPtr& program) {
  if (!func) return {};
  std::vector<ReturnedValueBinding> result;
  result.reserve(func->return_types_.size());
  for (const auto& type : func->return_types_) result.push_back({nullptr, type, std::nullopt});
  if (result.empty()) return result;

  TraceQueryGuard query_guard;
  auto ret = func->body_ ? FindFirstReturn(func->body_) : nullptr;
  if (ret && !ret->value_.empty()) {
    const auto& values = ret->value_;
    auto tuple_type = values.size() == 1 ? As<TupleType>(values[0]->GetType()) : nullptr;
    if (result.size() > 1 && tuple_type && tuple_type->types_.size() == result.size()) {
      // A forwarded tuple describes several flat return positions, but its
      // original expression remains the source of every analysis projection.
      for (size_t i = 0; i < result.size(); ++i) {
        result[i].value =
            std::make_shared<TupleGetItemExpr>(values[0], static_cast<int>(i), values[0]->span_);
      }
    } else {
      for (size_t i = 0; i < std::min(values.size(), result.size()); ++i) result[i].value = values[i];
    }
  } else if (func->body_ && program && IsWrapperType(func->func_type_)) {
    CallPtr returning_call;
    FunctionPtr returning_callee;
    bool ambiguous = false;
    for (const auto& info : CollectInnerCalls(func, program)) {
      if (info.inner_callee->return_types_.size() != result.size()) continue;
      if (returning_call) {
        ambiguous = true;
        break;
      }
      returning_call = info.inner_call;
      returning_callee = info.inner_callee;
    }
    if (returning_call && !ambiguous) {
      // A Submit's transient Call view also includes the trailing TaskId.
      // Keep only the callee's declared return positions, without fabricating
      // an SSA variable or treating that launch handle as a kernel output.
      auto tuple_type = As<TupleType>(returning_call->GetType());
      if (result.size() == 1 &&
          structural_equal(returning_call->GetType(), returning_callee->return_types_[0])) {
        result[0].value = returning_call;
      } else if (result.size() > 1 && tuple_type && tuple_type->types_.size() >= result.size()) {
        for (size_t i = 0; i < result.size(); ++i) {
          result[i].value =
              std::make_shared<TupleGetItemExpr>(returning_call, static_cast<int>(i), returning_call->span_);
        }
      } else if (result.size() == 1 && tuple_type && tuple_type->types_.size() == 2) {
        auto task_id_type = As<ScalarType>(tuple_type->types_[1]);
        if (task_id_type && task_id_type->dtype_ == DataType::TASK_ID &&
            structural_equal(tuple_type->types_[0], returning_callee->return_types_[0])) {
          result[0].value = std::make_shared<TupleGetItemExpr>(returning_call, 0, returning_call->span_);
        }
      }
    }
  }

  // Unknown lineage is storage information, not a missing return position.
  // Preserve the callee-local value and type even for fresh or partial buffers.
  auto param_indices = ReturnedParamIndicesImpl(func, program);
  for (size_t i = 0; i < std::min(param_indices.size(), result.size()); ++i) {
    result[i].param_index = param_indices[i];
  }
  return result;
}

std::vector<std::optional<size_t>> TraceToParamIndices(const std::vector<VarPtr>& vars, const StmtPtr& body,
                                                       const std::vector<VarPtr>& params,
                                                       const ProgramPtr& program) {
  return TraceBodyExprsToParamIndices(std::vector<ExprPtr>(vars.begin(), vars.end()), body, params, program,
                                      /*trace_if_merges=*/false);
}

std::vector<std::optional<size_t>> TraceExprsToParamIndices(const std::vector<ExprPtr>& exprs,
                                                            const StmtPtr& body,
                                                            const std::vector<VarPtr>& params,
                                                            const ProgramPtr& program) {
  return TraceBodyExprsToParamIndices(exprs, body, params, program, /*trace_if_merges=*/false);
}

std::vector<std::optional<size_t>> TraceToParamIndicesForOutlining(const std::vector<VarPtr>& vars,
                                                                   const StmtPtr& body,
                                                                   const std::vector<VarPtr>& params,
                                                                   const ProgramPtr& program,
                                                                   bool trace_if_merges) {
  return TraceBodyExprsToParamIndices(std::vector<ExprPtr>(vars.begin(), vars.end()), body, params, program,
                                      trace_if_merges);
}

VarPtr TraceToParam(const VarPtr& var, const StmtPtr& body, const std::vector<VarPtr>& params,
                    const ProgramPtr& program) {
  if (!var || !body) return nullptr;
  TraceQueryGuard query_guard;
  BodyIndexCollector index;
  index.VisitStmt(body);
  auto param_indices = IndexParams(params);
  LineageMemo memo;
  const Var* root = TraceVar(var.get(), index, param_indices, program, memo).root;
  if (!root) return nullptr;
  auto param_it = param_indices.find(root);
  return param_it == param_indices.end() ? nullptr : params[param_it->second];
}

}  // namespace return_lineage
}  // namespace ir
}  // namespace pypto
