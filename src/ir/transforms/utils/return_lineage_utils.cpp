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
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/wrapper_call_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace return_lineage {

namespace {

bool IsBuiltinOp(const std::string& op_name) {
  return op_name.find("tile.") == 0 || op_name.find("tensor.") == 0 || op_name.find("system.") == 0 ||
         op_name.find("array.") == 0;
}

CallPtr AsCallOrSubmitView(const ExprPtr& expr) {
  if (auto call = As<Call>(expr)) return call;
  if (auto submit = As<Submit>(expr)) return SubmitToCallView(submit);
  return nullptr;
}

// Per-body index: topmost ReturnStmt, per-var defining AssignStmt, and
// loop-carry edges (iter_arg / tensor return_var -> init value var).
class BodyIndexCollector : public IRVisitor {
 public:
  ReturnStmtPtr first_return;
  std::unordered_map<const Var*, AssignStmtPtr> var_def;
  std::unordered_map<const Var*, const Var*> carry_src;

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
};

// Cross-function memo + cycle guard, shared by the recursive tracer. The
// per-function result depends only on the callee body, so a thread_local
// cache bounds total work to one body walk per distinct function reached
// from the outermost query (cleared when it unwinds).
struct TraceContext {
  std::vector<const Function*> call_stack;
  std::unordered_map<const Function*, std::vector<std::optional<size_t>>> memo;
};

TraceContext& Ctx() {
  thread_local TraceContext ctx;
  return ctx;
}

std::vector<std::optional<size_t>> ReturnedParamIndicesImpl(const FunctionPtr& func,
                                                            const ProgramPtr& program);

// Trace a body var back to a param of `params`; nullptr when untraceable.
const Var* TraceVar(const Var* var, const BodyIndexCollector& index,
                    const std::unordered_set<const Var*>& params, const ProgramPtr& program,
                    std::unordered_set<const Var*>& visited) {
  while (var) {
    if (!visited.insert(var).second) return nullptr;
    if (params.count(var)) return var;

    if (auto carry_it = index.carry_src.find(var); carry_it != index.carry_src.end()) {
      var = carry_it->second;
      continue;
    }

    auto def_it = index.var_def.find(var);
    if (def_it == index.var_def.end()) return nullptr;
    const auto& value = def_it->second->value_;

    if (auto rhs_var = AsVarLike(value)) {
      var = rhs_var.get();
      continue;
    }

    if (auto tuple_get = As<TupleGetItemExpr>(value)) {
      // Tuple destructuring of a user call: resolve via the callee's own
      // returned-param map, then continue from the corresponding arg var.
      auto tuple_var = AsVarLike(tuple_get->tuple_);
      if (!tuple_var) return nullptr;
      auto tuple_def = index.var_def.find(tuple_var.get());
      if (tuple_def == index.var_def.end()) return nullptr;
      auto call = AsCallOrSubmitView(tuple_def->second->value_);
      if (!call || IsBuiltinOp(call->op_->name_)) return nullptr;
      auto callee = program ? program->GetFunction(call->op_->name_) : nullptr;
      if (!callee) return nullptr;
      auto ret_map = ReturnedParamIndicesImpl(callee, program);
      size_t pos = static_cast<size_t>(tuple_get->index_);
      if (tuple_get->index_ < 0 || pos >= ret_map.size() || !ret_map[pos]) return nullptr;
      size_t mapped = ret_map[pos].value();  // NOLINT(bugprone-unchecked-optional-access)
      if (mapped >= call->args_.size()) return nullptr;
      auto arg_var = AsVarLike(call->args_[mapped]);
      if (!arg_var) return nullptr;
      var = arg_var.get();
      continue;
    }

    if (auto call = AsCallOrSubmitView(value)) {
      const std::string& op_name = call->op_->name_;
      // Builtin output-side ops bind a fresh SSA var to an existing buffer.
      // tensor.assemble(target, tile, offset) / tensor.set_validshape(target, ...)
      // alias args[0]; tile.store(value, indices, target) aliases args[2].
      if (IsOp(call, "tensor.assemble") || IsOp(call, "tensor.set_validshape")) {
        auto arg_var = !call->args_.empty() ? AsVarLike(call->args_[0]) : nullptr;
        if (!arg_var) return nullptr;
        var = arg_var.get();
        continue;
      }
      if (IsOp(call, "tile.store")) {
        auto arg_var = call->args_.size() >= 3 ? AsVarLike(call->args_[2]) : nullptr;
        if (!arg_var) return nullptr;
        var = arg_var.get();
        continue;
      }
      if (!IsBuiltinOp(op_name)) {
        // Single-result user call: continue from the arg the callee returns.
        auto callee = program ? program->GetFunction(op_name) : nullptr;
        if (!callee) return nullptr;
        auto ret_map = ReturnedParamIndicesImpl(callee, program);
        if (ret_map.size() != 1 || !ret_map[0]) return nullptr;
        size_t mapped = ret_map[0].value();  // NOLINT(bugprone-unchecked-optional-access)
        if (mapped >= call->args_.size()) return nullptr;
        auto arg_var = AsVarLike(call->args_[mapped]);
        if (!arg_var) return nullptr;
        var = arg_var.get();
        continue;
      }
      return nullptr;
    }
    return nullptr;
  }
  return nullptr;
}

std::vector<std::optional<size_t>> TraceExprsToParamIndices(const std::vector<ExprPtr>& exprs,
                                                            const BodyIndexCollector& index,
                                                            const std::vector<VarPtr>& params,
                                                            const ProgramPtr& program) {
  std::unordered_set<const Var*> param_set;
  for (const auto& p : params) param_set.insert(p.get());

  std::vector<std::optional<size_t>> result;
  result.reserve(exprs.size());
  for (const auto& expr : exprs) {
    std::optional<size_t> idx;
    if (auto var = AsVarLike(expr)) {
      std::unordered_set<const Var*> visited;
      if (const Var* root = TraceVar(var.get(), index, param_set, program, visited)) {
        for (size_t i = 0; i < params.size(); ++i) {
          if (params[i].get() == root) {
            idx = i;
            break;
          }
        }
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
                                                           const std::vector<VarPtr>& params,
                                                           const ProgramPtr& program) {
  std::unordered_set<const Var*> param_set;
  for (const auto& p : params) param_set.insert(p.get());

  // Walk SSA var-to-var aliases down to the statement that defines the tuple.
  const Var* var = tuple_var;
  std::unordered_set<const Var*> seen;
  CallPtr call;
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
      return TraceExprsToParamIndices(make_tuple->elements_, index, params, program);
    }
    call = AsCallOrSubmitView(value);
    break;
  }
  if (!call || IsBuiltinOp(call->op_->name_)) return {};

  auto callee = program ? program->GetFunction(call->op_->name_) : nullptr;
  if (!callee) return {};
  auto inner_map = ReturnedParamIndicesImpl(callee, program);
  if (inner_map.empty()) return {};

  std::vector<std::optional<size_t>> result(inner_map.size());
  for (size_t pos = 0; pos < inner_map.size(); ++pos) {
    if (!inner_map[pos]) continue;
    size_t mapped = inner_map[pos].value();  // NOLINT(bugprone-unchecked-optional-access)
    if (mapped >= call->args_.size()) continue;
    auto arg_var = AsVarLike(call->args_[mapped]);
    if (!arg_var) continue;
    // The arg need not be a param outright -- it may itself be an assemble /
    // loop-carry chain rooted at one, so run it through the normal tracer.
    std::unordered_set<const Var*> visited;
    const Var* root = TraceVar(arg_var.get(), index, param_set, program, visited);
    if (!root) continue;
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i].get() == root) {
        result[pos] = i;
        break;
      }
    }
  }
  return result;
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
    if (mapped >= returning_call->args_.size()) continue;
    if (auto arg_var = AsVarLike(returning_call->args_[mapped])) {
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
  if (std::find(ctx.call_stack.begin(), ctx.call_stack.end(), func.get()) != ctx.call_stack.end()) {
    return {};
  }
  if (auto it = ctx.memo.find(func.get()); it != ctx.memo.end()) {
    return it->second;
  }

  ctx.call_stack.push_back(func.get());
  bool is_top_level = ctx.call_stack.size() == 1;
  struct Guard {
    TraceContext& ctx;
    bool top_level;
    ~Guard() {
      ctx.call_stack.pop_back();
      if (top_level) ctx.memo.clear();
    }
  } guard{ctx, is_top_level};

  auto record = [&](std::vector<std::optional<size_t>> result) {
    if (!is_top_level) ctx.memo.emplace(func.get(), result);
    return result;
  };

  BodyIndexCollector index;
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
  if (rets.size() == 1 && declared_arity > 1) {
    if (auto tuple_var = AsVarLike(rets[0])) {
      if (auto tuple_ty = As<TupleType>(tuple_var->GetType());
          tuple_ty && tuple_ty->types_.size() == declared_arity) {
        auto expanded = ExpandForwardedTupleVar(tuple_var.get(), index, func->params_, program);
        if (expanded.size() == declared_arity) return record(expanded);
      }
    }
  }
  return record(TraceExprsToParamIndices(rets, index, func->params_, program));
}

}  // namespace

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

VarPtr TraceToParam(const VarPtr& var, const StmtPtr& body, const std::vector<VarPtr>& params,
                    const ProgramPtr& program) {
  if (!var || !body) return nullptr;
  BodyIndexCollector index;
  index.VisitStmt(body);
  std::unordered_set<const Var*> param_set;
  for (const auto& p : params) param_set.insert(p.get());
  std::unordered_set<const Var*> visited;
  const Var* root = TraceVar(var.get(), index, param_set, program, visited);
  if (!root) return nullptr;
  for (const auto& p : params) {
    if (p.get() == root) return p;
  }
  return nullptr;
}

}  // namespace return_lineage
}  // namespace ir
}  // namespace pypto
