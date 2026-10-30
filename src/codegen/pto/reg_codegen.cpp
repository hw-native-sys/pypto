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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto::codegen {
namespace {
using namespace ir;  // NOLINT(build/namespaces)

// Tile storage keeps its ordinary signless element type. Register arithmetic
// needs explicit signedness: VMI interprets a signless integer as unsigned.
std::string RegisterElement(DataType dtype) {
  if (dtype.IsSignedInt()) return "si" + std::to_string(dtype.GetBit());
  if (dtype.IsUnsignedInt()) return "ui" + std::to_string(dtype.GetBit());
  if (dtype == DataType::FP16) return "f16";
  if (dtype == DataType::BF16) return "bf16";
  return "f32";
}

// Pre-scan one RegScope before emitting it: pointer extraction must dominate
// vecscope, while SSA bindings for values defined inside must not escape it.
class ScopeOperands : public IRVisitor {
 public:
  // Access sites whose Tile operands need an external UB pointer.
  std::vector<CallPtr> memory;
  // Store/control-flow result -> source view. Stores change bytes, not Tile
  // ownership; branch aliases additionally require both sources to agree.
  std::map<const Var*, VarPtr> aliases;
  // Both arms must resolve to the same external view before an if result can
  // use one pointer hoisted outside vecscope. Keep both sources for validation.
  std::vector<std::pair<VarPtr, VarPtr>> branch_sources;
  // Bind these results to the proven external handle before branch emission;
  // a fresh phi buffer would require an unsupported Tile copy inside vecscope.
  std::vector<VarPtr> tile_branch_results;
  // Assignment-defined scalar/register bindings to discard on scope exit.
  std::vector<VarPtr> locals;

 protected:
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (As<RegType>(stmt->var_->GetType()) || As<ScalarType>(stmt->var_->GetType())) {
      locals.push_back(stmt->var_);
    }
    if (auto call = As<Call>(stmt->value_); call && IsOp(call, "reg.store")) {
      aliases.emplace(stmt->var_.get(), AsVarLike(call->args_[0]));
    }
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "reg.load") || IsOp(call, "reg.store")) memory.push_back(call);
    IRVisitor::VisitExpr_(call);
  }
  void VisitStmt_(const ForStmtPtr& stmt) override {
    RecordLoopResults(stmt->iter_args_, stmt->return_vars_);
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const WhileStmtPtr& stmt) override {
    RecordLoopResults(stmt->iter_args_, stmt->return_vars_);
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitStmt_(const IfStmtPtr& stmt) override {
    auto then_yield = transform_utils::GetLastYieldStmt(stmt->then_body_);
    auto else_yield = stmt->else_body_ ? transform_utils::GetLastYieldStmt(*stmt->else_body_) : nullptr;
    for (size_t i = 0; i < stmt->return_vars_.size(); ++i) {
      if (!As<TileType>(stmt->return_vars_[i]->GetType())) continue;
      INTERNAL_CHECK_SPAN(
          then_yield && else_yield && i < then_yield->value_.size() && i < else_yield->value_.size(),
          stmt->span_)
          << "Internal error: register Tile branch result requires yields from both arms";
      auto then_value = AsVarLike(then_yield->value_[i]);
      auto else_value = AsVarLike(else_yield->value_[i]);
      INTERNAL_CHECK_SPAN(then_value && else_value, stmt->span_)
          << "Internal error: register Tile branch yields require variables";
      aliases.emplace(stmt->return_vars_[i].get(), then_value);
      branch_sources.emplace_back(then_value, else_value);
      tile_branch_results.push_back(stmt->return_vars_[i]);
    }
    IRVisitor::VisitStmt_(stmt);
  }

 private:
  // Tile loop carries preserve the initial backing view, including zero-trip
  // loops. Results have no emitted SSA handle during this pre-scan, so resolve
  // them through the corresponding initial value before address extraction.
  void RecordLoopResults(const std::vector<IterArgPtr>& args, const std::vector<VarPtr>& results) {
    for (size_t i = 0; i < args.size() && i < results.size(); ++i) {
      if (As<TileType>(results[i]->GetType())) aliases.emplace(results[i].get(), args[i]);
    }
  }
};
}  // namespace

std::string PTOCodegen::GetRegTypeString(const ir::RegTypePtr& type) const {
  CHECK(type && GetBackendHandler()->SupportsRegType(*type)) << "Unsupported A5 register type";
  if (type->kind_ == ir::RegKind::MaskReg) return "!pto.vmi.mask<" + std::to_string(type->shape_) + "xpred>";
  return "!pto.vmi.vreg<" + std::to_string(type->shape_) + "x" + RegisterElement(type->dtype_) + ">";
}

void PTOCodegen::VisitStmt_(const ir::RegScopeStmtPtr& scope) {
  CHECK_SPAN(!inside_reg_scope_, scope->span_) << "RegScope cannot nest";
  ScopeOperands operands;
  // A plain Tile alias may have been defined before this RegScope. Preserve
  // that handle provenance so equivalent external names compare identically.
  operands.aliases.insert(fs_.tile_aliases.begin(), fs_.tile_aliases.end());
  operands.VisitStmt(scope->body_);
  std::map<const ir::Var*, std::string> external_addresses;
  // Path-compressed provenance avoids retracing every prefix of N chained
  // stores. Each alias is resolved once with ordered-map/set lookup cost.
  std::map<const ir::Var*, ir::VarPtr> external_roots;
  // Resolve all control-flow/store aliases after the complete pre-scan, so a
  // branch yield may itself refer to a nested branch or loop result.
  auto resolve_external = [&](const ir::VarPtr& target) {
    auto external = target;
    std::set<const ir::Var*> seen;
    std::vector<ir::VarPtr> path;
    while (true) {
      if (auto cached = external_roots.find(external.get()); cached != external_roots.end()) {
        external = cached->second;
        break;
      }
      CHECK_SPAN(seen.insert(external.get()).second, scope->span_) << "Cyclic register Tile alias";
      path.push_back(external);
      auto alias = operands.aliases.find(external.get());
      if (alias != operands.aliases.end()) {
        external = alias->second;
        continue;
      }
      if (auto iter = ir::As<ir::IterArg>(external)) {
        // The supported Tile carry preserves backing storage. Follow the
        // initial handle to obtain an address available before the loop/scope;
        // this does not construct or rewrite the loop's SSA carries.
        external = ir::AsVarLike(iter->initValue_);
        CHECK_SPAN(external, scope->span_) << "Cannot prove register loop address";
        continue;
      }
      break;
    }
    for (const auto& var : path) external_roots.emplace(var.get(), external);
    return external;
  };
  for (const auto& [then_value, else_value] : operands.branch_sources) {
    // Choosing only one arm without checking the other could silently access
    // the wrong buffer. A branch-selected backing view needs a dynamic pointer
    // carry, which this external-address ABI does not yet represent.
    // TODO(chenyiyuan): carry a selected pointer when branch yields use different views.
    CHECK_SPAN(resolve_external(then_value) == resolve_external(else_value), scope->span_)
        << "Register Tile branch results require a shared external view";
  }
  for (const auto& result : operands.tile_branch_results) {
    BindVarToMlir(result, GetExprAsCode(resolve_external(result)));
  }
  // Hoist addresses, including the addresses used by chained store versions.
  // A store version is an alias, never a new allocation or region result.
  for (const auto& call : operands.memory) {
    auto target = ir::AsVarLike(call->args_[0]);
    CHECK_SPAN(target, call->span_) << "Register address requires a Tile variable";
    auto external = resolve_external(target);
    auto tile = ir::As<ir::TileType>(external->GetType());
    CHECK_SPAN(tile, call->span_) << "Register address requires a Vec Tile";
    auto [it, inserted] = external_addresses.emplace(external.get(), "");
    if (inserted) {
      it->second = NewNamedTemp(external->name_hint_ + "_reg_ptr");
      Emit(it->second + " = pto.tile_buf_addr " + GetExprAsCode(external) + " : " +
           GetExprTypeAnnotation(external) + " -> !pto.ptr<" + GetTypeString(tile->dtype_) + ", ub>");
    }
    // Tile storage uses signless integers, while VMI requires pointer and
    // register element types to agree, including explicit signedness.
    if (inserted && tile->dtype_.IsSignedInt()) {
      auto typed = NewNamedTemp(external->name_hint_ + "_signed_ptr");
      Emit(typed + " = pto.castptr " + it->second + " : !pto.ptr<" + GetTypeString(tile->dtype_) +
           ", ub> -> !pto.ptr<" + RegisterElement(tile->dtype_) + ", ub>");
      it->second = typed;
    }
    reg_addresses_[target.get()] = it->second;
  }
  inside_reg_scope_ = true;
  EmitStructural("pto.vecscope {");
  IncreaseIndent();
  VisitStmt(scope->body_);
  DecreaseIndent();
  EmitStructural("}");
  inside_reg_scope_ = false;
  reg_addresses_.clear();
  for (const auto& local : operands.locals) fs_.var_to_mlir.erase(local.get());
}

bool PTOCodegen::TryEmitRegCall(const ir::CallPtr& call, const ir::VarPtr& result) {
  // Assignment RHS expressions also include scalar arithmetic and plain aliases.
  // Let the normal emitter handle these instead of dereferencing a failed cast.
  if (!call) return false;
  auto& registry = ir::OpRegistry::GetInstance();
  if (!registry.IsRegistered(call->op_->name_)) return false;
  const auto& entry = registry.GetEntry(call->op_->name_);
  const auto& spec = entry.GetRegSpec();
  if (!spec) return false;
  CHECK_SPAN(inside_reg_scope_ && result, call->span_)
      << "Register operations require a named assignment inside RegScope";
  // Revalidate forged/deserialized calls as well as normal Builder output.
  auto validated = registry.Create(call->op_->name_, call->args_, call->kwargs_, call->span_);
  const auto& args = call->args_;
  auto operand_type = [this](const ir::ExprPtr& expr) {
    if (auto reg = ir::As<ir::RegType>(expr->GetType())) return GetRegTypeString(reg);
    return GetTypeString(ir::As<ir::ScalarType>(expr->GetType())->dtype_);
  };
  const bool store = ir::IsOp(call, "reg.store");
  if (store || ir::IsOp(call, "reg.load")) {
    // Dynamic offsets pass directly to VMI; initialization analysis is an
    // optional static check, not a precondition for emitting an address.
    auto target = ir::AsVarLike(args[0]);
    auto addr = reg_addresses_.find(target.get());
    CHECK_SPAN(addr != reg_addresses_.end(), call->span_) << "Unprepared external register address";
    auto tile = ir::As<ir::TileType>(args[0]->GetType());
    std::string ptr_type = "!pto.ptr<" + RegisterElement(tile->dtype_) + ", ub>";
    auto offset = GetExprAsCode(args[1]);
    // Reg memory analysis includes a TileView origin in the proven interval.
    // Add that same element origin to the emitted VMI access; tile_buf_addr
    // points at the MemRef base and does not encode TileView.start_offset.
    if (tile->tile_view_ && tile->tile_view_->start_offset) {
      auto origin = EmitArithOperand(tile->tile_view_->start_offset, "index");
      auto adjusted = NewNamedTemp("reg_view_offset");
      Emit(adjusted + " = arith.addi " + offset + ", " + origin + " : index");
      offset = adjusted;
    }
    if (store) {
      std::string line =
          spec->emitter + " " + GetExprAsCode(args[2]) + ", " + addr->second + "[" + offset + "]";
      if (args.size() == 4) line += ", " + GetExprAsCode(args[3]);
      // A masked store changes only active lanes; preserve the destination
      // in inactive lanes rather than accepting VMI's zero-mode default.
      if (args.size() == 4) line += " {pmode = \"merge\"}";
      line += " : " + operand_type(args[2]) + ", " + ptr_type;
      if (args.size() == 4) line += ", " + operand_type(args[3]);
      Emit(line);
      // VMI stores have no SSA result. The IR's updated Tile version denotes
      // the same mutable handle, so bind it to the destination's existing SSA.
      BindVarToMlir(result, GetExprAsCode(args[0]));
      auto ctx = GetCommCtxSSAFor(target.get());
      if (!ctx.empty()) RegisterCommCtxFor(result, ctx);
      return true;
    }
    auto name = NewNamedTemp(result->name_hint_);
    Emit(name + " = " + spec->emitter + " " + addr->second + "[" + offset + "] : " + ptr_type + " -> " +
         GetRegTypeString(ir::As<ir::RegType>(validated->GetType())));
    BindVarToMlir(result, name);
    return true;
  }
  std::string operands, types, attrs;
  for (const auto& arg : args) {
    if (!operands.empty()) {
      operands += ", ";
      types += ", ";
    }
    auto value = GetExprAsCode(arg);
    auto type = operand_type(arg);
    if (ir::IsOp(call, "reg.broadcast")) {
      auto scalar = ir::As<ir::ScalarType>(arg->GetType());
      if (scalar && scalar->dtype_.IsSignedInt()) {
        const auto signed_type = RegisterElement(scalar->dtype_);
        auto typed = NewNamedTemp("broadcast_scalar");
        Emit(typed + " = builtin.unrealized_conversion_cast " + value + " : " + type + " to " + signed_type);
        value = typed;
        type = signed_type;
      }
    }
    operands += value;
    types += type;
  }
  if (ir::IsOp(call, "reg.full_mask")) operands = "\"PAT_ALL\"";
  if (ir::IsOp(call, "reg.index_gen")) {
    const auto reg_type = ir::As<ir::RegType>(validated->GetType());
    CHECK_SPAN(reg_type && reg_type->dtype_ != DataType::BF16, call->span_)
        << "reg.index_gen does not support BF16 VCI in PTOAS; use an integer index dtype";
    attrs = " {order = \"ASC\"}";
  }
  if (ir::IsOp(call, "reg.compare")) {
    attrs = " {cmp = \"" + ir::GetStringKwarg(call->kwargs_, "predicate", "") + "\"}";
  }
  if (ir::IsOp(call, "reg.convert")) {
    // The schema validates pair-specific attributes. Absence matters: widening
    // conversions prohibit rounding/saturate, and integer narrowing prohibits
    // rounding. Do not invent floating conversion defaults here.
    std::vector<std::string> attributes;
    for (const auto& [key, value] : call->kwargs_) {
      if (key == "rounding") {
        attributes.push_back("rounding = \"" + std::any_cast<std::string>(value) + "\"");
      } else if (key == "saturate") {
        attributes.push_back(std::string("saturate = \"") + (std::any_cast<bool>(value) ? "SAT" : "NOSAT") +
                             "\"");
      }
    }
    for (const auto& attr : attributes) {
      attrs += attrs.empty() ? " {" : ", ";
      attrs += attr;
    }
    if (!attrs.empty()) attrs += "}";
  }

  if (args.size() == 3 && ir::As<ir::RegType>(args[0]->GetType()) && !ir::IsOp(call, "reg.compare") &&
      !ir::IsOp(call, "reg.select")) {
    attrs = " {pmode = \"zero\"}";
  }
  auto name = NewNamedTemp(result->name_hint_);
  auto type = GetRegTypeString(ir::As<ir::RegType>(validated->GetType()));
  Emit(name + " = " + spec->emitter + " " + operands + attrs + " : " +
       (types.empty() ? type : types + " -> " + type));
  BindVarToMlir(result, name);
  return true;
}
}  // namespace pypto::codegen
