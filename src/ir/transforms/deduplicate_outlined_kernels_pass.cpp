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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"
#include "utils/bounded_kernel_index.h"
#include "utils/structural_inspection.h"

namespace pypto::ir {
namespace {
using structural_comparison_detail::Inspection;
using Names = std::map<std::string, std::string>;

bool IsPeerImport(const OpPtr& op) { return As<Op>(op) && IsOp(op, "system.import_peer_buffer"); }

// Callee replacement does not change arguments or caller-coordinate result types.
// Memoization keeps shared expression DAGs shared during the single rewrite.
class CalleeRewriter : public IRMutator {
 public:
  explicit CalleeRewriter(const Names& aliases) : aliases_(aliases) {}
  FunctionPtr VisitFunction(const FunctionPtr& func) override {
    auto body = VisitStmt(func->body_);
    if (body == func->body_) return func;
    // Metadata references were pinned during admission; only body callees and
    // the recognized peer field can refer to functions being removed.
    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, body, func->span_, func->func_type_, func->level_,
                                      func->role_, func->attrs_, func->requires_runtime_binding_,
                                      func->ir_stage_);
  }

 protected:
  // IRMutator's recursion dispatches directly through ExprFunctor, bypassing
  // VisitExpr overrides. Memoize at the concrete handlers instead.
  ExprPtr VisitExpr_(const VarPtr& op) override { return op; }
  ExprPtr VisitExpr_(const MemRefPtr& op) override { return op; }
  ExprPtr VisitExpr_(const WindowBufferPtr& op) override { return op; }
  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    return Memo(op, [&] { return IRMutator::VisitExpr_(op); });
  }
  ExprPtr VisitBinaryExpr_(const BinaryExprPtr& op) override {
    return Memo(op, [&] { return IRMutator::VisitBinaryExpr_(op); });
  }
  ExprPtr VisitUnaryExpr_(const UnaryExprPtr& op) override {
    return Memo(op, [&] { return IRMutator::VisitUnaryExpr_(op); });
  }
  ExprPtr VisitExpr_(const MakeTuplePtr& op) override {
    return Memo(op, [&] { return IRMutator::VisitExpr_(op); });
  }
  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    return Memo(op, [&] { return IRMutator::VisitExpr_(op); });
  }
  ExprPtr VisitExpr_(const CallPtr& call) override {
    return Memo(call, [&]() -> ExprPtr {
      auto args = RewriteArgs(call->args_);
      auto callee = Rename(call->op_);
      auto kwargs = call->kwargs_;
      bool changed = RewritePeer(call->op_, kwargs);
      if (callee == call->op_ && !changed && args == call->args_) return call;
      return std::make_shared<Call>(callee, args, kwargs, call->attrs_, call->GetType(), call->span_);
    });
  }
  ExprPtr VisitExpr_(const SubmitPtr& submit) override {
    return Memo(submit, [&]() -> ExprPtr {
      auto args = RewriteArgs(submit->args_);
      auto deps = RewriteArgs(submit->deps_);
      auto core_num = submit->core_num_;
      auto predicate = submit->predicate_;
      if (core_num) core_num = VisitExpr(*core_num);
      if (predicate) predicate = VisitExpr(*predicate);
      auto callee = Rename(submit->op_);
      auto kwargs = submit->kwargs_;
      bool changed = RewritePeer(submit->op_, kwargs);
      if (callee == submit->op_ && !changed && args == submit->args_ && deps == submit->deps_ &&
          core_num == submit->core_num_ && predicate == submit->predicate_) {
        return submit;
      }
      return std::make_shared<Submit>(callee, args, deps, kwargs, submit->attrs_, submit->GetType(),
                                      submit->span_, core_num, submit->sync_start_,
                                      submit->allow_early_resolve_, predicate);
    });
  }

 private:
  template <typename Rewrite>
  ExprPtr Memo(const ExprPtr& expr, Rewrite rewrite) {
    auto found = cache_.find(expr.get());
    if (found != cache_.end()) return found->second;
    auto result = rewrite();
    cache_.emplace(expr.get(), result);
    return result;
  }
  std::vector<ExprPtr> RewriteArgs(const std::vector<ExprPtr>& args) {
    std::vector<ExprPtr> result;
    result.reserve(args.size());
    for (const auto& arg : args) result.push_back(VisitExpr(arg));
    return result;
  }
  OpPtr Rename(const OpPtr& op) const {
    if (!As<GlobalVar>(op)) return op;
    auto found = aliases_.find(op->name_);
    return found == aliases_.end() ? op : std::make_shared<GlobalVar>(found->second);
  }
  bool RewritePeer(const OpPtr& op, std::vector<std::pair<std::string, std::any>>& kwargs) const {
    if (!IsPeerImport(op)) return false;
    for (auto& [key, value] : kwargs) {
      if (key != "peer_func") continue;
      auto peer = std::any_cast<std::string>(&value);
      if (!peer) return false;
      auto found = aliases_.find(*peer);
      if (found == aliases_.end()) return false;
      value = found->second;
      return true;
    }
    return false;
  }
  const Names& aliases_;
  std::unordered_map<const Expr*, ExprPtr> cache_;
};

struct FunctionInfo {
  FunctionPtr function;
  Inspection inspection;
  std::vector<std::string> callees;
  std::set<std::string> peers;
  bool eligible = true;
};

FunctionInfo InspectFunction(const FunctionPtr& function) {
  FunctionInfo info{function, structural_comparison_detail::Inspect(function), {}, {}, true};
  info.eligible = info.inspection.free_variables.empty() && !function->requires_runtime_binding_ &&
                  !function->HasAttr(kAttrExternalSource) && !function->HasAttr(kAttrBuiltinTemplateDir);
  for (const auto& type : info.inspection.types) {
    // Explicit storage/communication/async identities are outside the early
    // functional-kernel contract. Ordinary tensor loads/stores remain eligible.
    if (As<UnknownType>(type) || As<MemRefType>(type) || As<WindowBufferType>(type) ||
        As<CommCtxType>(type) || As<PtrType>(type) || As<BufferType>(type) || As<MultiBufferType>(type) ||
        As<PrefetchAsyncContextType>(type) || As<AsyncEventType>(type) || As<AsyncSessionType>(type)) {
      info.eligible = false;
    }
  }
  for (const auto& node : info.inspection.nodes) {
    if (As<MemRef>(node) || As<WindowBuffer>(node)) info.eligible = false;
    OpPtr op;
    const std::vector<std::pair<std::string, std::any>>* kwargs = nullptr;
    if (auto call = As<Call>(node)) {
      op = call->op_;
      kwargs = &call->kwargs_;
    }
    if (auto submit = As<Submit>(node)) {
      op = submit->op_;
      kwargs = &submit->kwargs_;
    }
    if (!op || !IsPeerImport(op)) continue;
    bool found = false;
    for (const auto& [key, value] : *kwargs) {
      if (key != "peer_func") continue;
      if (auto peer = std::any_cast<std::string>(&value)) {
        info.peers.insert(*peer);
        found = true;
      }
    }
    if (!found) info.eligible = false;
  }
  for (const auto& [op, rewritable] : info.inspection.operators) {
    if (IsPeerImport(op) && !rewritable) info.eligible = false;
    if (As<GlobalVar>(op)) {
      info.callees.push_back(op->name_);
    } else if (!OpRegistry::GetInstance().IsRegistered(op->name_)) {
      info.eligible = false;
    }
  }
  return info;
}

struct Unit {
  std::vector<std::string> names;  // root, then members in first-use order
  IRNodePtr normalized;
  size_t weight = 1;
  TypePtr signature = nullptr;
};

FunctionPtr WithName(const FunctionPtr& func, const std::string& name) {
  return std::make_shared<Function>(name, func->params_, func->param_directions_, func->return_types_,
                                    func->body_, func->span_, func->func_type_, func->level_, func->role_,
                                    func->attrs_, func->requires_runtime_binding_, func->ir_stage_);
}

ProgramPtr Deduplicate(const ProgramPtr& program) {
  std::map<std::string, FunctionInfo> info;
  std::map<std::string, std::set<std::string>> callers, peer_users;
  std::set<std::string> pinned, group_members;
  // Complete the reference index before considering any removal. An unknown
  // payload can hide a reference, so an uninspectable program stays unchanged.
  try {
    for (const auto& [gv, func] : program->functions_) {
      // The map constructor permits symbol aliases, which are outside the
      // outlined-kernel naming contract used by the reference index below.
      if (gv->name_ != func->name_) return program;
      auto item = InspectFunction(func);
      for (const auto& name : item.callees) callers[name].insert(func->name_);
      for (const auto& name : item.peers) peer_users[name].insert(func->name_);
      for (const auto& [op, rewritable] : item.inspection.operators) {
        if (As<GlobalVar>(op) && !rewritable) pinned.insert(op->name_);
      }
      if (func->func_type_ == FunctionType::Group) {
        group_members.insert(item.callees.begin(), item.callees.end());
      }
      info.emplace(func->name_, std::move(item));
    }
  } catch (const StructuralComparisonUnsupported& error) {
    LOG_DEBUG << "DeduplicateOutlinedKernels: cannot index references: " << error.what();
    return program;
  }

  std::map<std::string, bool> admission_cache;
  auto admitted = [&](const std::string& name) {
    auto [cached, inserted] = admission_cache.emplace(name, false);
    if (!inserted) return cached->second;
    auto found = info.find(name);
    if (found == info.end() || !found->second.eligible || pinned.count(name)) return false;
    try {
      verify_structural_bindings(found->second.function);
    } catch (const StructuralComparisonUnsupported& error) {
      LOG_DEBUG << "DeduplicateOutlinedKernels: skip " << name << ": " << error.what();
      return false;
    }
    cached->second = true;
    return true;
  };

  std::vector<Unit> units;
  for (const auto& [name, item] : info) {
    // Unreferenced kernels are potentially exposed entries. Preserve all of
    // them, independent of Program ordering or the presence of another entry.
    if (callers[name].empty() || !admitted(name)) continue;
    if (item.function->func_type_ == FunctionType::InCore) {
      if (group_members.count(name) || !item.callees.empty() || !item.peers.empty() ||
          !peer_users[name].empty()) {
        continue;
      }
      units.push_back({{name}, item.function, item.inspection.weight});
      continue;
    }
    if (item.function->func_type_ != FunctionType::Group || group_members.count(name) ||
        !item.peers.empty() || !peer_users[name].empty()) {
      continue;
    }
    Unit unit{{name}, nullptr, item.inspection.weight};
    std::set<std::string> members;
    bool valid = true;
    for (const auto& callee : item.callees) {
      if (!members.insert(callee).second) continue;
      if (callers[callee] != std::set<std::string>{name} || !admitted(callee)) {
        valid = false;
        break;
      }
      const auto& member = info.at(callee);
      if (!IsInCoreType(member.function->func_type_) || !member.callees.empty()) {
        valid = false;
        break;
      }
      unit.names.push_back(callee);
      unit.weight += member.inspection.weight;
    }
    if (!valid || members.empty()) continue;
    for (const auto& member : members) {
      for (const auto& peer : info.at(member).peers) {
        if (!members.count(peer)) valid = false;
      }
      for (const auto& user : peer_users[member]) {
        if (!members.count(user)) valid = false;
      }
    }
    if (!valid) continue;
    Names roles{{name, "unit_root"}};
    for (size_t i = 1; i < unit.names.size(); ++i) {
      roles.emplace(unit.names[i], "member_" + std::to_string(i - 1));
    }
    std::vector<FunctionPtr> normalized;
    unit.weight = 0;
    for (const auto& original : unit.names) {
      CalleeRewriter rewriter(roles);
      auto func = WithName(rewriter.VisitFunction(info.at(original).function), roles.at(original));
      unit.weight += structural_comparison_detail::Inspect(func).weight;
      normalized.push_back(func);
    }
    unit.normalized = std::make_shared<Program>(normalized, "unit", program->span_);
    // Include the temporary Program container and normalized symbol strings.
    unit.weight += 128 * unit.names.size();
    units.push_back(std::move(unit));
  }

  deduplication_detail::BoundedIndex<Unit> index;
  Names aliases;
  for (auto& unit : units) {
    // Tensor-to-tile lowering allocates outputs from the callee's signature,
    // without substituting symbols into each caller's coordinates. Function
    // equality alpha-maps those symbols, so it alone cannot authorize aliasing.
    // Project all signatures as standalone types: their variables remain free
    // and must have identical identities, including in views and Group members.
    // One tuple projection memoizes shared type/shape DAGs across the whole unit.
    std::vector<TypePtr> signatures;
    for (const auto& name : unit.names) {
      const auto& func = info.at(name).function;
      std::vector<TypePtr> types;
      for (const auto& param : func->params_) types.push_back(param->GetType());
      types.insert(types.end(), func->return_types_.begin(), func->return_types_.end());
      unit.weight += 128 * types.size();
      signatures.push_back(std::make_shared<TupleType>(types));
    }
    unit.signature = std::make_shared<TupleType>(signatures);
    const auto hash = structural_hash(unit.normalized, false) ^ structural_hash(unit.signature, false);
    auto canonical = index.FindOrInsert(hash, unit, [](const Unit& left, const Unit& right) {
      return structural_equal(left.signature, right.signature, false) &&
             structural_equal(left.normalized, right.normalized, false);
    });
    if (canonical == &unit) continue;
    INTERNAL_CHECK_SPAN(unit.names.size() == canonical->names.size(), program->span_)
        << "Equal units must have equal member counts";
    for (size_t i = 0; i < unit.names.size(); ++i) aliases.emplace(unit.names[i], canonical->names[i]);
  }
  if (aliases.empty()) return program;
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions;
  for (const auto& [gv, func] : program->functions_) {
    if (aliases.count(func->name_)) continue;
    CalleeRewriter rewriter(aliases);
    auto rewritten = rewriter.VisitFunction(func);
    const auto checked = InspectFunction(rewritten);
    for (const auto& callee : checked.callees) {
      INTERNAL_CHECK_SPAN(!aliases.count(callee), rewritten->span_)
          << "Deduplication left a dangling callee: " << callee;
    }
    for (const auto& peer : checked.peers) {
      INTERNAL_CHECK_SPAN(!aliases.count(peer), rewritten->span_)
          << "Deduplication left a dangling peer: " << peer;
    }
    functions.emplace(gv, rewritten);
  }
  return std::make_shared<Program>(functions, program->name_, program->span_);
}
}  // namespace

namespace pass {
Pass DeduplicateOutlinedKernels() {
  return CreateProgramPass(Deduplicate, "DeduplicateOutlinedKernels", kDeduplicateOutlinedKernelsProperties);
}
}  // namespace pass
}  // namespace pypto::ir
