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
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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
#include "pypto/ir/transforms/utils/ccu_collective_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto::ir {
namespace {

// Index each body once. Memoized provenance follows aliases, views and returned
// parameters without persisting a second description of uses on Function attrs.
class Origins : public IRVisitor {
 public:
  explicit Origins(ProgramPtr program) : program_(std::move(program)) {}
  std::unordered_map<const Var*, ExprPtr> defs;
  void VisitStmt_(const AssignStmtPtr& op) override {
    defs[op->var_.get()] = op->value_;
    IRVisitor::VisitStmt_(op);
  }
  ExprPtr Root(const ExprPtr& expr) {
    if (auto var = AsVarLike(expr)) {
      auto cached = roots_.find(var.get());
      if (cached != roots_.end()) return cached->second;
      roots_[var.get()] = nullptr;  // Cycle guard; unresolved loops are rejected.
      auto it = defs.find(var.get());
      auto root = it == defs.end() ? expr : Root(it->second);
      if (auto call = it == defs.end() ? nullptr : As<Call>(it->second);
          IsOp(call, "pld.tensor.alloc_window_buffer")) {
        root = expr;
      }
      roots_[var.get()] = root;
      return root;
    }
    if (auto item = As<TupleGetItemExpr>(expr)) {
      ExprPtr tuple = item->tuple_;
      if (auto var = AsVarLike(tuple); var && defs.count(var.get())) {
        tuple = defs.at(var.get());
      }
      if (auto values = As<MakeTuple>(tuple)) return Root(values->elements_.at(item->index_));
      return CallResult(tuple, item->index_, /*projection=*/true);
    }
    if (auto call = As<Call>(expr)) {
      if (IsOp(call, "pld.tensor.window") || IsOp(call, "tensor.slice") || IsOp(call, "tensor.reshape")) {
        return Root(call->args_[0]);
      }
    }
    if (As<Call>(expr) || As<Submit>(expr)) return CallResult(expr, 0);
    return expr;
  }

 private:
  ExprPtr CallResult(const ExprPtr& expr, size_t index, bool projection = false) {
    OpPtr op;
    std::vector<ExprPtr> args;
    if (auto call = As<Call>(expr)) {
      op = call->op_;
      args = call->args_;
    }
    if (auto call = As<Submit>(expr)) {
      op = call->op_;
      args = call->args_;
    }
    auto callee = As<GlobalVar>(op) ? program_->GetFunction(op->name_) : nullptr;
    if (!callee) return nullptr;
    // A direct call with one tuple-valued return exposes its elements. A
    // Submit wraps the declared return positions in a tuple with a task ID.
    const bool tuple_elements = projection && As<Call>(expr) && return_lineage::HasSingleTupleReturn(callee);
    auto& cache = tuple_elements ? tuple_returns_ : returns_;
    auto [it, inserted] = cache.try_emplace(callee.get());
    if (inserted) {
      it->second = tuple_elements ? return_lineage::ReturnedTupleElementParamIndices(callee, program_)
                                  : return_lineage::ReturnedParamIndices(callee, program_);
    }
    if (index >= it->second.size()) return nullptr;
    // Keep the presence check and dereference on the same optional value.
    const auto& returned_param = it->second[index];
    if (!returned_param.has_value()) return nullptr;
    auto param = *returned_param;
    return param < args.size() ? Root(args[param]) : nullptr;
  }
  ProgramPtr program_;
  std::unordered_map<const Var*, ExprPtr> roots_;
  std::unordered_map<const Function*, std::vector<std::optional<size_t>>> returns_, tuple_returns_;
};

struct Use {
  std::string definition;
  std::vector<size_t> params;
  int64_t ranks;
};
struct CallableUses {
  std::vector<Use> requirements;
  // Codegen can package several shape-specific definition symbols as one module.
  std::map<std::string, std::string> representative;
};
using Uses = std::map<std::string, CallableUses>;

class Scan : public IRVisitor {
 public:
  Scan(ProgramPtr program, const std::map<std::string, CallPtr>& wrappers, Origins& origins,
       const std::unordered_map<const Var*, size_t>& params, CallableUses& uses)
      : program_(std::move(program)), wrappers_(wrappers), origins_(origins), params_(params), uses_(uses) {}
  void VisitExpr_(const CallPtr& call) override {
    Record(call);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& call) override {
    Record(call);
    IRVisitor::VisitExpr_(call);
  }
  void VisitStmt_(const ForStmtPtr& op) override { Nested(op); }
  void VisitStmt_(const WhileStmtPtr& op) override { Nested(op); }
  void VisitStmt_(const IfStmtPtr& op) override { Nested(op); }

 private:
  template <typename T>
  void Nested(const T& op) {
    ++depth_;
    IRVisitor::VisitStmt_(op);
    --depth_;
  }
  template <typename T>
  void Record(const T& call) {
    auto it = wrappers_.find(call->op_->name_);
    if (it == wrappers_.end()) return;
    CHECK_SPAN(depth_ == 0, call->span_) << "CCU calls require straight-line CHIP control flow";
    auto definition = program_->GetFunction(it->second->op_->name_);
    CHECK_SPAN(IsCcuAllGatherBuiltin(definition) && definition->params_.size() >= 2, call->span_)
        << "CCU binding currently supports native allgather declarations only";
    auto target = As<DistributedTensorType>(definition->params_[1]->GetType());
    CHECK_SPAN(target && !target->shape_.empty(), call->span_) << "CCU declaration requires a target window";
    auto ranks = As<ConstInt>(target->shape_[0]);
    CHECK_SPAN(ranks && ranks->value_ >= 2 && ranks->value_ <= 64, call->span_)
        << "CCU declaration requires a static rank count in 2..64";
    auto [symbol, new_symbol] = uses_.representative.try_emplace(definition->name_);
    if (new_symbol) {
      symbol->second =
          modules_.try_emplace(GetCcuModuleIdentity(definition), definition->name_).first->second;
    }
    Use use{symbol->second, {}, ranks->value_};
    // Includes signal even though it is not packed into native arguments.
    for (const auto& arg : call->args_) {
      if (!As<DistributedTensorType>(arg->GetType())) continue;
      auto root = AsVarLike(origins_.Root(arg));
      auto param = params_.find(root.get());
      CHECK_SPAN(param != params_.end(), call->span_)
          << "Cannot prove CCU DistributedTensor parameter provenance";
      use.params.push_back(param->second);
    }
    if (seen_.emplace(use.definition, use.params).second) uses_.requirements.push_back(std::move(use));
  }
  ProgramPtr program_;
  const std::map<std::string, CallPtr>& wrappers_;
  Origins& origins_;
  const std::unordered_map<const Var*, size_t>& params_;
  CallableUses& uses_;
  std::map<CcuModuleIdentity, std::string> modules_;
  std::set<std::pair<std::string, std::vector<size_t>>> seen_;
  int depth_ = 0;
};
Uses AnalyzeUses(const ProgramPtr& program) {
  Uses result;
  std::map<std::string, CallPtr> wrappers;
  for (const auto& [gv, func] : program->functions_) {
    if (auto call = CcuWrapperCall(func, program)) wrappers.emplace(func->name_, call);
  }
  for (const auto& [gv, func] : program->functions_) {
    if (!IsOrchestrationLike(func->func_type_) || func->level_ != Level::CHIP) continue;
    Origins origins(program);
    origins.VisitStmt(func->body_);
    std::unordered_map<const Var*, size_t> params;
    for (size_t i = 0; i < func->params_.size(); ++i) params.emplace(func->params_[i].get(), i);
    Scan scan(program, wrappers, origins, params, result[func->name_]);
    scan.VisitStmt(func->body_);
    if (result[func->name_].requirements.empty()) result.erase(func->name_);
  }
  class RejectNested : public IRVisitor {
   public:
    explicit RejectNested(const Uses& uses) : uses_(uses) {}
    void VisitExpr_(const CallPtr& call) override {
      Check(call->op_, call->span_);
      IRVisitor::VisitExpr_(call);
    }
    void VisitExpr_(const SubmitPtr& call) override {
      Check(call->op_, call->span_);
      IRVisitor::VisitExpr_(call);
    }

   private:
    void Check(const OpPtr& op, const Span& span) {
      CHECK_SPAN(!uses_.count(op->name_), span)
          << "Nested CCU CHIP calls require interprocedural provenance and are not supported yet";
    }
    const Uses& uses_;
  } reject_nested(result);
  for (const auto& [gv, func] : program->functions_) {
    if (func->level_ != Level::HOST && func->body_) reject_nested.VisitStmt(func->body_);
  }
  return result;
}

// Each registration is local to a constant worker, including rank-loop uses.
using RegistrationKey = std::tuple<const CommDomainScopeStmt*, std::string, std::string, int64_t>;
struct DeviceSelection {
  ExprPtr expr;
  int64_t constant = 0;
};

class HostBindings : public IRVisitor {
 public:
  HostBindings(const ProgramPtr& program, const Uses& uses, bool verify = false)
      : origins(program), program_(program), uses_(uses), verify_(verify) {}
  Origins origins;
  std::map<const Expr*, std::map<int64_t, std::vector<VarPtr>>> bindings;
  std::map<const Var*, std::vector<StmtPtr>> after_alloc;
  std::set<const CommDomainScopeStmt*> used_domains;
  std::set<std::string> names;

  void VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    for (const auto& slot : op->slots_) {
      auto base = AsVarLike(slot->base_);
      INTERNAL_CHECK_SPAN(base, op->span_) << "CCU domain slot must have an allocation Var";
      domains_[base.get()] = op;
    }
    IRVisitor::VisitStmt_(op);
    auto registrations = scope_registrations_.find(op.get());
    if (registrations != scope_registrations_.end()) {
      for (const auto* var : registrations->second) live_registrations_.erase(var);
      scope_registrations_.erase(registrations);
    }
    for (const auto& slot : op->slots_) domains_.erase(AsVarLike(slot->base_).get());
  }
  void VisitStmt_(const AssignStmtPtr& op) override {
    names.insert(op->var_->name_hint_);
    if (auto call = As<Call>(op->value_); IsOp(call, "pld.system.register_ccu")) {
      auto root = AsVarLike(origins.Root(call->args_.at(0)));
      auto domain = domains_.find(root.get());
      CHECK_SPAN(domain != domains_.end() && allocations_.count(root.get()), op->span_)
          << "register_ccu requires a dominating allocation token in a live domain";
      auto definition = program_->GetFunction(call->GetKwarg<std::string>("callee"));
      CHECK_SPAN(definition && definition->func_type_ == FunctionType::CCU, op->span_)
          << "register_ccu callee must resolve to a CCU function";
      CHECK_SPAN(IsA<CcuRegistrationType>(op->var_->GetType()), op->span_)
          << "register_ccu requires a HOST local registration result";
      CHECK_SPAN(!submitted_ && depth_ == 0, op->span_)
          << "CCU registrations must precede the first task submission";
      auto device = ResolveDevice(call->GetAttr<ExprPtr>(kAttrDevice), op->span_);
      CheckMembers(device, domain->second, op->span_);
      auto chip = program_->GetFunction(call->GetKwarg<std::string>("chip_callee"));
      CHECK_SPAN(chip && chip->level_ == Level::CHIP && IsOrchestrationLike(chip->func_type_), op->span_)
          << "register_ccu chip_callee must resolve to a CHIP callable";
      const auto uses = uses_.find(chip->name_);
      CHECK_SPAN(uses != uses_.end() && uses->second.representative.count(definition->name_), op->span_)
          << "CCU bindings must match the required callable, domain and definition on the same device";
      RegistrationKey key{domain->second.get(), chip->name_,
                          uses->second.representative.at(definition->name_), device.constant};
      live_registrations_[op->var_.get()] = key;
      scope_registrations_[domain->second.get()].push_back(op->var_.get());
    }
    if (IsA<CcuRegistrationType>(op->var_->GetType()) && AsVarLike(op->value_)) {
      auto alias = AsVarLike(op->value_);
      auto it = live_registrations_.find(alias.get());
      CHECK_SPAN(it != live_registrations_.end(), op->span_) << "Registration alias outlives its domain";
      live_registrations_[op->var_.get()] = it->second;
      scope_registrations_[std::get<0>(it->second)].push_back(op->var_.get());
    }
    if (auto call = As<Call>(op->value_); IsOp(call, "pld.tensor.alloc_window_buffer")) {
      // Only unconditional allocations can anchor a compiler-inserted registration.
      if (depth_ == 0 && unsupported_ == 0) allocations_.insert(op->var_.get());
      if (submitted_) late_allocations_.insert(op->var_.get());
    }
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForStmtPtr& op) override {
    auto start = As<ConstInt>(origins.Root(op->start_));
    auto stop = As<ConstInt>(origins.Root(op->stop_));
    auto step = As<ConstInt>(origins.Root(op->step_));
    if (depth_ == 0 && start && stop && step && step->value_ == 1 && start->value_ >= 0 &&
        stop->value_ > start->value_ && stop->value_ - start->value_ <= 64) {
      // Analyze each worker's requirements, but retain the original loop and
      // window construction. Only the final dispatch selects local handles.
      ++depth_;
      rank_var_ = op->loop_var_.get();
      for (rank_ = start->value_; rank_ < stop->value_; ++rank_) VisitStmt(op->body_);
      rank_var_ = nullptr;
      --depth_;
    } else {
      ++unsupported_;
      IRVisitor::VisitStmt_(op);
      --unsupported_;
    }
  }
  void VisitStmt_(const ReturnStmtPtr& op) override {
    for (const auto& value : op->value_) {
      CHECK_SPAN(!IsA<CcuRegistrationType>(value->GetType()), op->span_)
          << "CCU registration cannot escape its owning domain";
    }
    IRVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const IfStmtPtr& op) override {
    // Select the worker-local handle when the rank predicate is statically known.
    if (auto eq = As<Eq>(op->condition_); eq && rank_var_) {
      auto left = DeviceValue(eq->left_);
      auto right = DeviceValue(eq->right_);
      if (left && right) {
        auto selected = *left == *right ? op->then_body_ : op->else_body_.value_or(nullptr);
        if (selected) {
          VisitStmt(selected);
          // Registration selection is an ordinary SSA phi. Resolve the chosen
          // handle for each analyzed rank without making the handle a device ID.
          auto yield = transform_utils::GetLastYieldStmt(transform_utils::UnwrapAutoScope(selected));
          for (size_t i = 0; yield && i < op->return_vars_.size(); ++i) {
            const auto& var = op->return_vars_[i];
            if (!IsA<CcuRegistrationType>(var->GetType())) continue;
            auto source = AsVarLike(yield->value_.at(i));
            auto reg = live_registrations_.find(source.get());
            CHECK_SPAN(reg != live_registrations_.end(), op->span_)
                << "CCU binding selection must yield a live registration";
            live_registrations_[var.get()] = reg->second;
            scope_registrations_[std::get<0>(reg->second)].push_back(var.get());
          }
        }
        return;
      }
    }
    ++unsupported_;
    IRVisitor::VisitStmt_(op);
    --unsupported_;
  }
  void VisitStmt_(const WhileStmtPtr& op) override {
    ++unsupported_;
    IRVisitor::VisitStmt_(op);
    --unsupported_;
  }
  void VisitExpr_(const CallPtr& call) override {
    Record(call);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& call) override {
    Record(call);
    IRVisitor::VisitExpr_(call);
  }

 private:
  std::optional<int64_t> DeviceValue(const ExprPtr& expr) {
    auto root = origins.Root(expr);
    if (auto value = As<ConstInt>(root)) return value->value_;
    if (auto var = AsVarLike(root); var && var.get() == rank_var_) {
      return rank_;
    }
    return std::nullopt;
  }
  DeviceSelection ResolveDevice(const ExprPtr& expr, const Span& span) {
    CHECK_SPAN(expr, span) << "CCU register/dispatch requires an explicit device";
    auto type = As<ScalarType>(expr->GetType());
    CHECK_SPAN(type && (type->dtype_.IsInt() || type->dtype_ == DataType::INDEX), span)
        << "CCU device must be an integer scalar";
    CHECK_SPAN(unsupported_ == 0, span) << "Conditional CCU registration/dispatch is unsupported";
    DeviceSelection device;
    auto value = DeviceValue(expr);
    CHECK_SPAN(value && depth_ <= 1, span) << "CCU register/dispatch requires constant devices or one "
                                              "positive static rank loop of at most 64 ranks";
    device.constant = *value;
    device.expr = std::make_shared<ConstInt>(*value, type->dtype_, span);
    return device;
  }
  static void CheckMembers(const DeviceSelection& device, const CommDomainScopeStmtPtr& domain,
                           const Span& span) {
    CHECK_SPAN(std::find(domain->devices_.begin(), domain->devices_.end(), device.constant) !=
                   domain->devices_.end(),
               span)
        << "CCU device is outside the registration domain";
  }
  template <typename T>
  void Record(const T& call) {
    for (const auto& arg : call->args_) {
      CHECK_SPAN(!IsA<CcuRegistrationType>(arg->GetType()), call->span_)
          << "HOST CCU registration handles cannot be passed as device or algorithm arguments";
    }
    auto callee = As<GlobalVar>(call->op_) ? program_->GetFunction(call->op_->name_) : nullptr;
    if (callee && callee->level_ != Level::HOST) submitted_ = true;
    auto uses = uses_.find(call->op_->name_);
    if (uses == uses_.end()) return;
    auto device = ResolveDevice(call->template GetAttr<ExprPtr>(kAttrDevice), call->span_);
    std::vector<VarPtr> handles;
    const auto refs =
        verify_ ? call->template GetAttr<std::vector<VarPtr>>(kAttrCcuBindings) : std::vector<VarPtr>{};
    // Bindings are an unordered requirement set, not positional call-site IDs.
    std::set<RegistrationKey> provided, required;
    for (const auto& ref : refs) {
      auto reg = live_registrations_.find(ref.get());
      CHECK_SPAN(reg != live_registrations_.end(), call->span_)
          << "CCU binding must dominate dispatch in a live domain";
      provided.insert(reg->second);
    }
    for (const auto& use : uses->second.requirements) {
      CommDomainScopeStmtPtr domain;
      VarPtr allocation;
      for (auto param : use.params) {
        auto root = AsVarLike(origins.Root(call->args_.at(param)));
        auto it = domains_.find(root.get());
        CHECK_SPAN(it != domains_.end() && allocations_.count(root.get()), call->span_)
            << "CCU argument must originate from an unconditional allocation in a live communication domain";
        CHECK_SPAN(!domain || domain == it->second, call->span_)
            << "CCU call arguments (including signal) must belong to the same communication domain";
        domain = it->second;
        if (!allocation) allocation = root;
      }
      CHECK_SPAN(domain && domain->devices_.size() == static_cast<size_t>(use.ranks), call->span_)
          << "CCU collective Host domain size must match target rank count " << use.ranks;
      CheckMembers(device, domain, call->span_);
      used_domains.insert(domain.get());
      RegistrationKey key{domain.get(), call->op_->name_, use.definition, device.constant};
      if (!required.insert(key).second || verify_) continue;
      auto [it, inserted] = registrations_.try_emplace(key);
      if (inserted) {
        CHECK_SPAN(!late_allocations_.count(allocation.get()), call->span_)
            << "CCU registrations must precede the first task submission; allocation is too late";
        std::string name;
        do {
          name = "_ccu_registration_" + std::to_string(next_name_++);
        } while (names.count(name));
        names.insert(name);
        auto reg = std::make_shared<Var>(name, GetCcuRegistrationType(), call->span_);
        it->second = reg;
        auto invoke = MutableCopy(OpRegistry::GetInstance().Create(
            "pld.system.register_ccu", {allocation},
            {{"callee", use.definition}, {"chip_callee", call->op_->name_}}, call->span_));
        invoke->attrs_.emplace_back(kAttrDevice, device.expr);
        auto registration = std::make_shared<AssignStmt>(reg, invoke, call->span_);
        after_alloc[allocation.get()].push_back(registration);
      }
      handles.push_back(it->second);
    }
    if (!verify_) bindings[call.get()][device.constant] = std::move(handles);
    CHECK_SPAN(!verify_ || provided == required, call->span_)
        << "CCU bindings must match the required callable, domain and definition on the same device";
  }

  ProgramPtr program_;
  const Uses& uses_;
  bool verify_;
  std::map<RegistrationKey, VarPtr> registrations_;
  std::map<const Var*, RegistrationKey> live_registrations_;
  std::map<const CommDomainScopeStmt*, std::vector<const Var*>> scope_registrations_;
  std::unordered_map<const Var*, CommDomainScopeStmtPtr> domains_;
  std::set<const Var*> allocations_;
  std::set<const Var*> late_allocations_;
  bool submitted_ = false;
  const Var* rank_var_ = nullptr;
  int64_t rank_ = 0;
  size_t next_name_ = 0;
  int depth_ = 0;
  int unsupported_ = 0;
};

class Rewrite : public IRMutator {
 public:
  explicit Rewrite(const HostBindings& plan) : plan_(plan), names_(plan.names) {}
  StmtPtr VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    auto scope = As<CommDomainScopeStmt>(IRMutator::VisitStmt_(op));
    // Dispatch operands identify their owning domain after URMA/MTE grouping.
    // Only domains participating in a CCU request need the extra workspace.
    if (!plan_.used_domains.count(op.get()) ||
        std::find(scope->engines_.begin(), scope->engines_.end(), "ccu") != scope->engines_.end()) {
      return scope;
    }
    auto updated = MutableCopy(scope);
    updated->engines_.emplace_back("ccu");
    return updated;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    if (auto dispatch = Dispatch(op->value_, op->var_)) return dispatch;
    auto result = IRMutator::VisitStmt_(op);
    auto it = plan_.after_alloc.find(op->var_.get());
    if (it == plan_.after_alloc.end()) return result;
    std::vector<StmtPtr> stmts{result};
    stmts.insert(stmts.end(), it->second.begin(), it->second.end());
    return SeqStmts::Flatten(std::move(stmts), op->span_);
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    if (auto dispatch = Dispatch(op->expr_, nullptr)) return dispatch;
    return IRMutator::VisitStmt_(op);
  }

 private:
  VarPtr Fresh(const Span& span) {
    std::string name;
    do {
      name = "_ccu_binding_" + std::to_string(next_name_++);
    } while (!names_.insert(name).second);
    return std::make_shared<Var>(name, GetCcuRegistrationType(), span);
  }
  template <typename T>
  ExprPtr Bind(const T& op, const std::vector<VarPtr>& handles) {
    auto updated = MutableCopy(op);
    auto& attrs = updated->attrs_;
    attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                               [](const auto& attr) { return attr.first == kAttrCcuBindings; }),
                attrs.end());
    attrs.emplace_back(kAttrCcuBindings, handles);
    return updated;
  }
  using Choices = std::map<int64_t, std::vector<VarPtr>>;
  StmtPtr Select(const ExprPtr& device, Choices::const_iterator it, Choices::const_iterator end,
                 const std::vector<VarPtr>& results, const Span& span) {
    auto then_body =
        std::make_shared<YieldStmt>(std::vector<ExprPtr>(it->second.begin(), it->second.end()), span);
    auto next = std::next(it);
    StmtPtr else_body;
    if (std::next(next) == end) {
      else_body =
          std::make_shared<YieldStmt>(std::vector<ExprPtr>(next->second.begin(), next->second.end()), span);
    } else {
      std::vector<VarPtr> nested;
      nested.reserve(results.size());
      for (size_t i = 0; i < results.size(); ++i) nested.push_back(Fresh(span));
      else_body = SeqStmts::Flatten(
          {Select(device, next, end, nested, span),
           std::make_shared<YieldStmt>(std::vector<ExprPtr>(nested.begin(), nested.end()), span)},
          span);
    }
    auto condition = std::make_shared<Eq>(
        device, std::make_shared<ConstInt>(it->first, DataType::INDEX, span), DataType::BOOL, span);
    return std::make_shared<IfStmt>(condition, then_body, else_body, results, span);
  }
  StmtPtr Dispatch(const ExprPtr& expr, const VarPtr& result) {
    auto it = plan_.bindings.find(expr.get());
    if (it == plan_.bindings.end()) return nullptr;
    auto call = As<Call>(expr);
    auto device =
        call ? call->GetAttr<ExprPtr>(kAttrDevice) : As<Submit>(expr)->GetAttr<ExprPtr>(kAttrDevice);
    auto handles = it->second.begin()->second;
    StmtPtr select;
    if (it->second.size() > 1) {
      for (const auto& [rank, refs] : it->second) {
        CHECK_SPAN(refs.size() == handles.size(), expr->span_)
            << "CCU rank selection requires a fixed number of local registration handles";
      }
      for (auto& handle : handles) handle = Fresh(expr->span_);
      select = Select(device, it->second.begin(), it->second.end(), handles, expr->span_);
    }
    auto bound = call ? Bind(call, handles) : Bind(As<Submit>(expr), handles);
    StmtPtr dispatch = result ? StmtPtr(std::make_shared<AssignStmt>(result, bound, expr->span_))
                              : StmtPtr(std::make_shared<EvalStmt>(bound, expr->span_));
    // Only handles are selected. Keep the original dispatch, return aliases,
    // Submit dependencies and TaskId definitions outside the selector branches.
    return select ? SeqStmts::Flatten({select, dispatch}, expr->span_) : dispatch;
  }
  const HostBindings& plan_;
  std::set<std::string> names_;
  size_t next_name_ = 0;
};

ProgramPtr ProcessCcuBindings(const ProgramPtr& program, bool verify_only) {
  return_lineage::ScopedTraceCache cache;
  const bool has_host = std::any_of(program->functions_.begin(), program->functions_.end(),
                                    [](const auto& item) { return item.second->level_ == Level::HOST; });
  if (!has_host) return program;
  auto uses = AnalyzeUses(program);
  if (uses.empty()) return program;
  auto result = MutableCopy(program);
  for (const auto& [gv, func] : program->functions_) {
    if (func->level_ != Level::HOST || !func->body_) continue;
    // Binding lowering is idempotent; verification/codegen checks existing ops.
    class Bound : public IRVisitor {
     public:
      bool found = false;
      void VisitExpr_(const CallPtr& op) override {
        found |= IsOp(op, "pld.system.register_ccu");
        IRVisitor::VisitExpr_(op);
      }
    } bound;
    bound.VisitStmt(func->body_);
    HostBindings plan(program, uses, bound.found || verify_only);
    plan.origins.VisitStmt(func->body_);
    for (const auto& [var, value] : plan.origins.defs) plan.names.insert(var->name_hint_);
    plan.VisitStmt(func->body_);
    if (bound.found || verify_only) {
      for (const auto* domain : plan.used_domains) {
        CHECK_SPAN(
            std::find(domain->engines_.begin(), domain->engines_.end(), "ccu") != domain->engines_.end(),
            func->span_)
            << "CCU dispatch requires the owning domain to provision a CCU workspace";
      }
      continue;
    }
    auto updated = MutableCopy(func);
    updated->body_ = Rewrite(plan).VisitStmt(func->body_);
    result->functions_[gv] = updated;
  }
  return result;
}
}  // namespace

ProgramPtr MaterializeCcuBindings(const ProgramPtr& program) { return ProcessCcuBindings(program, false); }
void VerifyCcuBindings(const ProgramPtr& program) { ProcessCcuBindings(program, true); }
}  // namespace pypto::ir
