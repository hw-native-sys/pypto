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

#include <map>
#include <set>
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/visitor.h"

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

class ScopeOperands : public IRVisitor {
 public:
  std::vector<CallPtr> memory;
  std::map<const Var*, VarPtr> aliases;
  std::vector<VarPtr> locals;

 protected:
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (As<RegType>(stmt->var_->GetType()) || As<ScalarType>(stmt->var_->GetType()))
      locals.push_back(stmt->var_);
    if (auto call = As<Call>(stmt->value_); call && IsOp(call, "reg.store"))
      aliases.emplace(stmt->var_.get(), AsVarLike(call->args_[0]));
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "reg.load") || IsOp(call, "reg.store")) memory.push_back(call);
    IRVisitor::VisitExpr_(call);
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
  operands.VisitStmt(scope->body_);
  std::map<const ir::Var*, std::string> external_addresses;
  // Hoist addresses, including the addresses used by chained store versions.
  // A store version is an alias, never a new allocation or region result.
  for (const auto& call : operands.memory) {
    auto target = ir::AsVarLike(call->args_[0]);
    CHECK_SPAN(target, call->span_) << "Register address requires a Tile variable";
    auto external = target;
    std::set<const ir::Var*> seen;
    while (seen.insert(external.get()).second) {
      auto alias = operands.aliases.find(external.get());
      if (alias != operands.aliases.end()) {
        external = alias->second;
        continue;
      }
      if (auto iter = ir::As<ir::IterArg>(external)) {
        external = ir::AsVarLike(iter->initValue_);
        CHECK_SPAN(external, call->span_) << "Cannot prove register loop address";
        continue;
      }
      break;
    }
    auto tile = ir::As<ir::TileType>(external->GetType());
    CHECK_SPAN(tile, call->span_) << "Register address requires a Vec Tile";
    auto [it, inserted] = external_addresses.emplace(external.get(), "");
    if (inserted) {
      it->second = NewNamedTemp(external->name_hint_ + "_reg_ptr");
      Emit(it->second + " = pto.tile_buf_addr " + GetExprAsCode(external) + " : " +
           GetTileBufTypeStringFromTileType(tile) + " -> !pto.ptr<" + GetTypeString(tile->dtype_) + ", ub>");
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
    auto effect = ir::GetRegMemoryEffect(call);
    CHECK_SPAN(effect.root && effect.begin && effect.end, call->span_)
        << "Cannot prove register offset bounds";
    auto target = ir::AsVarLike(args[0]);
    auto addr = reg_addresses_.find(target.get());
    CHECK_SPAN(addr != reg_addresses_.end(), call->span_) << "Unprepared external register address";
    auto tile = ir::As<ir::TileType>(args[0]->GetType());
    std::string ptr_type = "!pto.ptr<" + GetTypeString(tile->dtype_) + ", ub>";
    auto offset = GetExprAsCode(args[1]);
    if (store) {
      std::string line =
          spec->emitter + " " + GetExprAsCode(args[2]) + ", " + addr->second + "[" + offset + "]";
      if (args.size() == 4) line += ", " + GetExprAsCode(args[3]);
      line += " : " + operand_type(args[2]) + ", " + ptr_type;
      if (args.size() == 4) line += ", " + operand_type(args[3]);
      Emit(line);
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
    operands += GetExprAsCode(arg);
    types += operand_type(arg);
  }
  if (ir::IsOp(call, "reg.full_mask")) operands = "\"PAT_ALL\"";
  if (ir::IsOp(call, "reg.index_gen")) attrs = " {order = \"ASC\"}";
  if (ir::IsOp(call, "reg.compare"))
    attrs = " {cmp = \"" + ir::GetStringKwarg(call->kwargs_, "predicate", "") + "\"}";
  if (ir::IsOp(call, "reg.convert")) {
    attrs = " {rounding = \"" + ir::GetStringKwarg(call->kwargs_, "rounding", "R") + "\", saturate = \"" +
            std::string(ir::GetKwargOr<bool>(call->kwargs_, "saturate", false) ? "SAT" : "NOSAT") + "\"}";
  }
  if (args.size() == 3 && ir::As<ir::RegType>(args[0]->GetType()) && !ir::IsOp(call, "reg.compare") &&
      !ir::IsOp(call, "reg.select"))
    attrs = " {pmode = \"zero\"}";
  auto name = NewNamedTemp(result->name_hint_);
  auto type = GetRegTypeString(ir::As<ir::RegType>(validated->GetType()));
  Emit(name + " = " + spec->emitter + " " + operands + attrs + " : " +
       (types.empty() ? type : types + " -> " + type));
  BindVarToMlir(result, name);
  return true;
}
}  // namespace pypto::codegen
