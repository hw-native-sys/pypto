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
#include <map>
#include <memory>
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
  if (dtype == DataType::FP8E4M3FN) return "f8E4M3FN";
  if (dtype == DataType::FP4E2M1X2) return "!pto.f4E2M1x2";
  return "f32";
}

// Pre-scan one RegScope before emitting it: pointer extraction must dominate
// vecscope, while SSA bindings for values defined inside must not escape it.
class ScopeOperands : public IRVisitor {
 public:
  // Access sites whose Tile operands need an external UB pointer.
  std::vector<CallPtr> memory;
  bool micro = false;
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
    if (auto call = As<Call>(stmt->value_); call && (IsOp(call, "reg.store") || IsOp(call, "reg.scatter"))) {
      aliases.emplace(stmt->var_.get(), AsVarLike(call->args_[0]));
    }
    if (auto call = As<Call>(stmt->value_); call && IsOp(call, "reg.squeeze_store_flush")) {
      aliases.emplace(stmt->var_.get(), AsVarLike(call->args_[1]));
    }
    IRVisitor::VisitStmt_(stmt);
  }
  void VisitExpr_(const CallPtr& call) override {
    micro |= IsOp(call, "reg.squeeze_store_init") || GetKwargOr<bool>(call->kwargs_, "native", false);
    if (IsOp(call, "reg.load") || IsOp(call, "reg.store") || IsOp(call, "reg.gather") ||
        IsOp(call, "reg.scatter") || IsOp(call, "reg.squeeze_store_step") ||
        IsOp(call, "reg.squeeze_store_flush")) {
      memory.push_back(call);
    }
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
  if (type->GetRegKind() == ir::RegKind::UReg) {
    CHECK(micro_reg_scope_) << "UReg requires squeeze-store micro lowering";
    return "!pto.align";
  }
  if (micro_reg_scope_) {
    if (type->GetRegKind() == ir::RegKind::MaskReg) {
      CHECK(type->GetShape() == 64 || type->GetShape() == 128 || type->GetShape() == 256)
          << "Physical micro lowering requires a full hardware mask";
      return "!pto.mask<b" + std::to_string(2048 / type->GetShape()) + ">";
    }
    CHECK(type->GetShape() * type->GetDType().GetByte() == 256)
        << "Squeeze-store micro lowering requires full 256-byte registers";
    return "!pto.vreg<" + std::to_string(type->GetShape()) + "x" + RegisterElement(type->GetDType()) + ">";
  }
  if (type->GetRegKind() == ir::RegKind::MaskReg) {
    return "!pto.vmi.mask<" + std::to_string(type->GetShape()) + "xpred>";
  }
  return "!pto.vmi.vreg<" + std::to_string(type->GetShape()) + "x" + RegisterElement(type->GetDType()) + ">";
}

void PTOCodegen::VisitStmt_(const ir::RegScopeStmtPtr& scope) {
  CHECK_SPAN(!inside_reg_scope_, scope->span_) << "RegScope cannot nest";
  if (separate_vf_) {
    EmitVFHelperCall(scope);
    return;
  }
  CHECK_SPAN(scope->AutoSync(), scope->span_)
      << "Manual RegScope synchronization requires generate_bundle(); combined diagnostic output "
         "and skip_ptoas=True cannot encode per-helper compilation policy";
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
    auto target = ir::AsVarLike(ir::GetRegMemoryEffect(call).target);
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
  micro_reg_scope_ = operands.micro;
  EmitStructural("pto.vecscope {");
  IncreaseIndent();
  VisitStmt(scope->body_);
  DecreaseIndent();
  EmitStructural("}");
  CHECK_SPAN(!squeeze_destination_, scope->span_) << "Unflushed squeeze-store state";
  inside_reg_scope_ = false;
  micro_reg_scope_ = false;
  reg_addresses_.clear();
  reg_tuple_values_.clear();
  for (const auto& local : operands.locals) fs_.var_to_mlir.erase(local.get());
}

bool PTOCodegen::TryEmitMicroRegCall(const ir::CallPtr& call, const ir::VarPtr& result) {
  using namespace ir;  // NOLINT(build/namespaces)
  const auto& args = call->args_;
  auto type = [this](const ExprPtr& expr) {
    if (auto reg = As<RegType>(expr->GetType())) return GetRegTypeString(reg);
    return GetTypeString(As<ScalarType>(expr->GetType())->dtype_);
  };
  auto value = [this](const ExprPtr& expr) { return GetExprAsCode(expr); };
  const auto name = NewNamedTemp(result->name_hint_);
  auto bind = [&] { BindVarToMlir(result, name); };
  auto full_mask = [&](int64_t lanes) {
    const auto mask_type =
        GetRegTypeString(std::make_shared<RegType>(lanes, DataType::BOOL, RegKind::MaskReg));
    auto mask = NewNamedTemp("squeeze_all");
    Emit(mask + " = pto.pset_b" + std::to_string(2048 / lanes) + " \"PAT_ALL\" : " + mask_type);
    return mask;
  };
  auto pointer = [&](const ExprPtr& target, const ExprPtr& offset) {
    auto var = AsVarLike(target);
    auto address = reg_addresses_.find(var.get());
    INTERNAL_CHECK_SPAN(address != reg_addresses_.end(), call->span_)
        << "Internal error: missing squeeze-store memory capture";
    auto tile = As<TileType>(target->GetType());
    const auto element = RegisterElement(tile->dtype_);
    auto index = value(offset);
    if (tile->tile_view_ && tile->tile_view_->start_offset) {
      const auto origin = EmitArithOperand(tile->tile_view_->start_offset, "index");
      const auto adjusted = NewNamedTemp("squeeze_view_offset");
      Emit(adjusted + " = arith.addi " + index + ", " + origin + " : index");
      index = adjusted;
    }
    const auto base = NewNamedTemp("squeeze_base");
    Emit(base + " = pto.addptr " + address->second + ", " + index + " : <" + element + ", ub>, index -> <" +
         element + ", ub>");
    return base;
  };

  if (IsOp(call, "reg.squeeze_store_init")) {
    CHECK_SPAN(!squeeze_destination_, call->span_) << "Squeeze streams cannot nest";
    squeeze_cursor_ = pointer(args[0], args[1]);
    auto tile = As<TileType>(args[0]->GetType());
    squeeze_pointer_type_ = "!pto.ptr<" + RegisterElement(tile->dtype_) + ", ub>";
    squeeze_destination_ = args[0];
    Emit("pto.sprclr \"AR\"");
    Emit(name + " = pto.init_align : !pto.align");
    bind();
    return true;
  }
  if (IsOp(call, "reg.squeeze_store_step") || IsOp(call, "reg.squeeze_store_flush")) {
    CHECK_SPAN(squeeze_destination_ && squeeze_destination_ == args[1], call->span_)
        << "Squeeze-store state must use its original destination";
    if (IsOp(call, "reg.squeeze_store_step")) {
      // Keep vsqz directly connected to vstur: PTOAS chooses SQZ MODE_STORED
      // from this use, and pushes the SQZN count needed by the corresponding store.
      auto packed = NewNamedTemp("squeeze_packed");
      Emit(packed + " = pto.vsqz " + value(args[2]) + ", " + value(args[3]) + " : " + type(args[2]) + ", " +
           type(args[3]) + " -> " + type(args[2]));
      Emit(name + " = pto.vstur " + value(args[0]) + ", " + packed + ", " + squeeze_cursor_ +
           ", \"POST_UPDATE\" : !pto.align, " + type(args[2]) + ", " + squeeze_pointer_type_ +
           " -> !pto.align");
      bind();
    } else {
      Emit("pto.vstar " + value(args[0]) + ", " + squeeze_cursor_ + " : !pto.align, " +
           squeeze_pointer_type_);
      BindVarToMlir(result, value(args[1]));
      squeeze_destination_.reset();
      squeeze_cursor_.clear();
      squeeze_pointer_type_.clear();
    }
    return true;
  }
  if (IsOp(call, "reg.load") || IsOp(call, "reg.store")) {
    auto base = pointer(args[0], args[1]);
    auto tile = As<TileType>(args[0]->GetType());
    auto ptr_type = "!pto.ptr<" + RegisterElement(tile->dtype_) + ", ub>";
    auto zero = GetOrEmitConstant(int64_t{0}, DataType::INDEX);
    const bool native = GetKwargOr<bool>(call->kwargs_, "native", false);
    auto dist = native ? GetStringKwarg(call->kwargs_, "dist", "") : "";
    // PTOAS spells ONEPT_B32 as 1PT_B32; an omitted NORM distribution
    // is the natural element-width store (NORM is only a load token).
    if (IsOp(call, "reg.store")) {
      if (dist == "NORM") dist.clear();
      if (dist == "ONEPT_B32") dist = "1PT_B32";
    }
    const auto attr = dist.empty() ? "" : " {dist = \"" + dist + "\"}";
    if (IsOp(call, "reg.load")) {
      auto result_type = As<RegType>(call->GetType());
      if (native && result_type->GetDType() != tile->dtype_) {
        // Offset is in storage elements; only reinterpret the pointer after
        // addressing the original storage. Validation ensures equal byte widths.
        const auto typed = NewNamedTemp("native_load_ptr");
        const auto new_type = "!pto.ptr<" + RegisterElement(result_type->GetDType()) + ", ub>";
        Emit(typed + " = pto.castptr " + base + " : " + ptr_type + " -> " + new_type);
        base = typed;
        ptr_type = new_type;
      }
      Emit(name + " = pto.vlds " + base + "[" + zero + "]" + attr + " : " + ptr_type + ", index -> " +
           GetRegTypeString(result_type));
      bind();
    } else {
      auto lanes = As<RegType>(args[2]->GetType())->GetShape();
      auto mask = args.size() == 4 ? value(args[3]) : full_mask(lanes);
      auto mask_type =
          args.size() == 4
              ? type(args[3])
              : GetRegTypeString(std::make_shared<RegType>(lanes, DataType::BOOL, RegKind::MaskReg));
      const auto block_stride = GetKwargOr<int>(call->kwargs_, "block_stride", 0);
      if (block_stride > 0) {
        CHECK_SPAN(!native, call->span_) << "Native distribution and block_stride cannot be combined";
        const auto block = GetOrEmitConstant(static_cast<int64_t>(block_stride), DataType::INT16);
        const auto repeat = GetOrEmitConstant(int64_t{0}, DataType::INT16);
        Emit("pto.vsstb " + value(args[2]) + ", " + base + ", " + block + ", " + repeat + ", " + mask +
             " : " + type(args[2]) + ", " + ptr_type + ", i16, i16, " + mask_type);
      } else {
        Emit("pto.vsts " + value(args[2]) + ", " + base + "[" + zero + "], " + mask + attr + " : " +
             type(args[2]) + ", " + ptr_type + ", index, " + mask_type);
      }
      BindVarToMlir(result, value(args[0]));
    }
    return true;
  }
  if (IsOp(call, "reg.dintlv") || IsOp(call, "reg.intlv")) {
    CHECK_SPAN(args.size() == 2, call->span_) << "Physical native interleave does not take a mask";
    auto low = NewNamedTemp("native_interleave_low");
    auto high = NewNamedTemp("native_interleave_high");
    const auto opcode = IsOp(call, "reg.dintlv") ? "pto.vdintlv" : "pto.vintlv";
    Emit(low + ", " + high + " = " + opcode + " " + value(args[0]) + ", " + value(args[1]) + " : " +
         type(args[0]) + ", " + type(args[1]) + " -> " + type(args[0]) + ", " + type(args[1]));
    reg_tuple_values_[result.get()] = {low, high};
    return true;
  }
  if (IsOp(call, "reg.gather")) {
    auto zero_expr = std::make_shared<ConstInt>(0, DataType::INDEX, call->span_);
    auto base = pointer(args[0], args.size() > 3 ? args[3] : zero_expr);
    auto tile = As<TileType>(args[0]->GetType());
    auto ptr_type = "!pto.ptr<" + RegisterElement(tile->dtype_) + ", ub>";
    Emit(name + " = pto.vgather2 " + base + ", " + value(args[1]) + ", " + value(args[2]) + " : " + ptr_type +
         ", " + type(args[1]) + ", " + type(args[2]) + " -> " +
         GetRegTypeString(As<RegType>(call->GetType())));
    bind();
    return true;
  }
  auto result_reg = As<RegType>(call->GetType());
  CHECK_SPAN(result_reg, call->span_) << "Unsupported result in squeeze-store micro lowering";
  auto result_type = GetRegTypeString(result_reg);
  if (IsOp(call, "reg.convert") && GetKwargOr<bool>(call->kwargs_, "native", false)) {
    std::string attrs = "{part = \"" + GetStringKwarg(call->kwargs_, "part", "") + "\"";
    if (GetStringKwarg(call->kwargs_, "rounding", "") != "") attrs += ", rnd = \"R\", sat = \"RS_DISABLE\"";
    attrs += "}";
    Emit(name + " = pto.vcvt " + value(args[0]) + ", " + value(args[1]) + " " + attrs + " : " +
         type(args[0]) + ", " + type(args[1]) + " -> " + result_type);
  } else if (IsOp(call, "reg.full_mask")) {
    Emit(name + " = pto.pset_b" + std::to_string(2048 / result_reg->GetShape()) +
         " \"PAT_ALL\" : " + result_type);
  } else if (IsOp(call, "reg.active_mask")) {
    auto count = NewNamedTemp("squeeze_count");
    auto unused = NewNamedTemp("squeeze_unused_count");
    Emit(count + " = arith.index_cast " + value(args[0]) + " : index to i32");
    Emit(name + ", " + unused + " = pto.plt_b" + std::to_string(2048 / result_reg->GetShape()) + " " + count +
         " : i32 -> " + result_type + ", i32");
  } else if (IsOp(call, "reg.compare") || IsOp(call, "reg.compares")) {
    auto opcode = IsOp(call, "reg.compare") ? "pto.vcmp" : "pto.vcmps";
    Emit(name + " = " + opcode + " " + value(args[0]) + ", " + value(args[1]) + ", " + value(args[2]) +
         ", \"" + GetStringKwarg(call->kwargs_, "predicate", "") + "\" : " + type(args[0]) + ", " +
         type(args[1]) + ", " + type(args[2]) + " -> " + result_type);
  } else if (IsOp(call, "reg.squeeze")) {
    Emit(name + " = pto.vsqz " + value(args[0]) + ", " + value(args[1]) + " : " + type(args[0]) + ", " +
         type(args[1]) + " -> " + result_type);
  } else if (IsOp(call, "reg.select")) {
    Emit(name + " = pto.vsel " + value(args[1]) + ", " + value(args[2]) + ", " + value(args[0]) + " : " +
         type(args[1]) + ", " + type(args[2]) + ", " + type(args[0]) + " -> " + result_type);
  } else if (IsOp(call, "reg.interpret_cast")) {
    Emit(name + " = pto.vbitcast " + value(args[0]) + " : " + type(args[0]) + " -> " + result_type);
  } else if (IsOp(call, "reg.broadcast")) {
    CHECK_SPAN(As<ScalarType>(args[0]->GetType()), call->span_)
        << "Squeeze-store micro broadcast requires a scalar";
    Emit(name + " = pto.vbr " + value(args[0]) + " : " + type(args[0]) + " -> " + result_type);
  } else if (IsOp(call, "reg.index_gen")) {
    auto input = value(args[0]);
    auto scalar_type = type(args[0]);
    auto generated_type = result_type;
    // VCI uses a signless integer ABI. Preserve all unsigned index bits by
    // reinterpreting the result, rather than converting through floating point.
    if (result_reg->GetDType().IsUnsignedInt()) {
      scalar_type = "i" + std::to_string(result_reg->GetDType().GetBit());
      auto cast = NewNamedTemp("squeeze_index_base");
      Emit(cast + " = builtin.unrealized_conversion_cast " + input + " : " + type(args[0]) + " to " +
           scalar_type);
      input = cast;
      generated_type = "!pto.vreg<" + std::to_string(result_reg->GetShape()) + "x" + scalar_type + ">";
    }
    auto generated = generated_type == result_type ? name : NewNamedTemp("squeeze_indices");
    Emit(generated + " = pto.vci " + input + " {order = \"ASC\"} : " + scalar_type + " -> " + generated_type);
    if (generated != name) {
      Emit(name + " = pto.vbitcast " + generated + " : " + generated_type + " -> " + result_type);
    }
  } else if (IsOp(call, "reg.mask_and") || IsOp(call, "reg.mask_or") || IsOp(call, "reg.mask_xor") ||
             IsOp(call, "reg.mask_not")) {
    auto all = full_mask(result_reg->GetShape());
    auto opcode = "pto.p" + call->op_->name_.substr(std::string("reg.mask_").size());
    std::string operands = value(args[0]);
    std::string types = type(args[0]);
    if (args.size() == 2) {
      operands += ", " + value(args[1]);
      types += ", " + type(args[1]);
    }
    Emit(name + " = " + opcode + " " + operands + ", " + all + " : " + types + ", " + result_type + " -> " +
         result_type);
  } else if (IsOp(call, "reg.mul") || IsOp(call, "reg.add") || IsOp(call, "reg.sub") ||
             IsOp(call, "reg.maximum") || IsOp(call, "reg.minimum") || IsOp(call, "reg.div") ||
             IsOp(call, "reg.exp") || IsOp(call, "reg.muls") || IsOp(call, "reg.shift_rights") ||
             IsOp(call, "reg.reduce_max") || IsOp(call, "reg.reduce_sum") || IsOp(call, "reg.gather_lanes")) {
    const auto& spec = OpRegistry::GetInstance().GetEntry(call->op_->name_).GetRegSpec();
    auto opcode = std::string("pto.") + spec->emitter.substr(std::string("pto.vmi.").size());
    std::string operands, types;
    for (const auto& arg : args) {
      if (!operands.empty()) {
        operands += ", ";
        types += ", ";
      }
      operands += value(arg);
      types += type(arg);
    }
    CHECK_SPAN(IsOp(call, "reg.gather_lanes") ||
                   As<RegType>(args.back()->GetType())->GetRegKind() == RegKind::MaskReg,
               call->span_)
        << "Physical native math requires explicit predicate";
    Emit(name + " = " + opcode + " " + operands + " : " + types + " -> " + result_type);
  } else {
    CHECK_SPAN(false, call->span_) << call->op_->name_ << " is not supported in physical micro lowering";
  }
  bind();
  return true;
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
  if (micro_reg_scope_) return TryEmitMicroRegCall(validated, result);
  auto operand_type = [this](const ir::ExprPtr& expr) {
    if (auto reg = ir::As<ir::RegType>(expr->GetType())) return GetRegTypeString(reg);
    return GetTypeString(ir::As<ir::ScalarType>(expr->GetType())->dtype_);
  };
  auto emit_full_mask = [this](const ir::ExprPtr& vector, const std::string& name_hint) {
    auto lanes = ir::As<ir::RegType>(vector->GetType())->GetShape();
    auto type = GetRegTypeString(std::make_shared<ir::RegType>(lanes, DataType::BOOL, ir::RegKind::MaskReg));
    auto mask = NewNamedTemp(name_hint);
    Emit(mask + " = pto.vmi.pset \"PAT_ALL\" : " + type);
    return std::pair{mask, type};
  };
  const bool indexed_read = ir::IsOp(call, "reg.gather");
  if (indexed_read || ir::IsOp(call, "reg.scatter")) {
    auto target = ir::AsVarLike(args[0]);
    auto addr = reg_addresses_.find(target.get());
    INTERNAL_CHECK_SPAN(addr != reg_addresses_.end(), call->span_)
        << "Internal error: unprepared indexed register address";
    auto tile = ir::As<ir::TileType>(args[0]->GetType());
    const std::string element = RegisterElement(tile->dtype_);
    const std::string ptr_type = "!pto.ptr<" + element + ", ub>";
    auto pointer = addr->second;
    // Index vectors are element offsets from the view, whereas tile_buf_addr
    // denotes the allocation. Shift the base exactly once, without rewriting
    // the user's index vector (which may have a narrower integer width).
    if (tile->tile_view_ && tile->tile_view_->start_offset) {
      auto origin = EmitArithOperand(tile->tile_view_->start_offset, "index");
      auto adjusted = NewNamedTemp("reg_indexed_base");
      Emit(adjusted + " = pto.addptr " + pointer + ", " + origin + " : <" + element + ", ub>, index -> <" +
           element + ", ub>");
      pointer = adjusted;
    }
    const size_t base_index = indexed_read ? 3 : 4;
    if (args.size() > base_index) {
      // A scalar base shift is independent of the index element width. Keep
      // the original index register and predicate unchanged for gather/scatter.
      auto offset = GetExprAsCode(args[base_index]);
      auto adjusted = NewNamedTemp("reg_indexed_offset_base");
      Emit(adjusted + " = pto.addptr " + pointer + ", " + offset + " : <" + element + ", ub>, index -> <" +
           element + ", ub>");
      pointer = adjusted;
    }
    // PTOAS >= 0.72 conservatively synchronizes unknown indexed footprints,
    // including aliased views and surrounding contiguous UB accesses.
    if (indexed_read) {
      auto name = NewNamedTemp(result->name_hint_);
      Emit(name + " = " + spec->emitter + " " + pointer + ", " + GetExprAsCode(args[1]) + ", " +
           GetExprAsCode(args[2]) + " {pmode = \"zero\"} : " + ptr_type + ", " + operand_type(args[1]) +
           ", " + operand_type(args[2]) + " -> " +
           GetRegTypeString(ir::As<ir::RegType>(validated->GetType())));
      BindVarToMlir(result, name);
    } else {
      // Scatter is intrinsically mask-governed: inactive addresses retain their
      // contents. Its PTOAS schema uses zero pmode, not vstore's merge attribute.
      Emit(spec->emitter + " " + GetExprAsCode(args[2]) + ", " + pointer + ", " + GetExprAsCode(args[1]) +
           ", " + GetExprAsCode(args[3]) + " : " + operand_type(args[2]) + ", " + ptr_type + ", " +
           operand_type(args[1]) + ", " + operand_type(args[3]));
      BindVarToMlir(result, GetExprAsCode(args[0]));
      auto ctx = GetCommCtxSSAFor(target.get());
      if (!ctx.empty()) RegisterCommCtxFor(result, ctx);
    }
    return true;
  }
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
      const auto stride = ir::GetKwargOr<int>(call->kwargs_, "block_stride", 0);
      if (stride) {
        auto scalar = NewNamedTemp("reg_block_stride");
        Emit(scalar + " = arith.constant " + std::to_string(stride) + " : i16");
        line += ", " + scalar;
      }
      // VMI prints the offset type after the pointer. A block stride is an
      // implicit i16 and its predicate is explicit, even for an all-lane store.
      auto types = operand_type(args[2]) + ", " + ptr_type + ", index";
      if (args.size() == 4 || stride) {
        auto [mask, mask_type] = args.size() == 4 ? std::pair{GetExprAsCode(args[3]), operand_type(args[3])}
                                                  : emit_full_mask(args[2], "store_all");
        line += ", " + mask;
        if (!stride) line += " {pmode = \"merge\"}";
        types += ", " + mask_type;
      }
      Emit(line + " : " + types);
      // VMI stores have no SSA result. The IR's updated Tile version denotes
      // the same mutable handle, so bind it to the destination's existing SSA.
      BindVarToMlir(result, GetExprAsCode(args[0]));
      auto ctx = GetCommCtxSSAFor(target.get());
      if (!ctx.empty()) RegisterCommCtxFor(result, ctx);
      return true;
    }
    auto name = NewNamedTemp(result->name_hint_);
    Emit(name + " = " + spec->emitter + " " + addr->second + "[" + offset + "] : " + ptr_type +
         ", index -> " + GetRegTypeString(ir::As<ir::RegType>(validated->GetType())));
    BindVarToMlir(result, name);
    return true;
  }
  // PTOAS VMULL uniquely spells the signed form as signless i32 (and rejects
  // si32). Bitwise reinterpretation bridges that ABI without changing PyPTO's
  // signed register type or numerically converting either product half.
  const bool signed_mull =
      ir::IsOp(call, "reg.mull") && ir::As<ir::RegType>(args[0]->GetType())->GetDType() == DataType::INT32;
  auto mull_type = [](const ir::RegTypePtr& type) {
    return "!pto.vmi.vreg<" + std::to_string(type->GetShape()) + "xi32>";
  };
  const bool unmasked_interleave =
      (ir::IsOp(call, "reg.intlv") || ir::IsOp(call, "reg.dintlv")) && args.size() == 2;
  std::string operands, types, attrs;
  for (const auto& arg : args) {
    if (!operands.empty()) {
      operands += ", ";
      types += ", ";
    }
    auto value = GetExprAsCode(arg);
    auto type = operand_type(arg);
    if (signed_mull) {
      if (auto reg = ir::As<ir::RegType>(arg->GetType()); reg && reg->GetRegKind() == ir::RegKind::VecReg) {
        auto typed = NewNamedTemp("mull_signed_input");
        auto signless = mull_type(reg);
        Emit(typed + " = pto.vmi.vinterpret_cast " + value + " : " + type + " -> " + signless);
        value = typed;
        type = signless;
      }
    }
    if (ir::IsOp(call, "reg.broadcast") || ir::IsOp(call, "reg.adds") || ir::IsOp(call, "reg.muls") ||
        ir::IsOp(call, "reg.maximums") || ir::IsOp(call, "reg.minimums") || ir::IsOp(call, "reg.compares")) {
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
  if (unmasked_interleave) {
    // VMI requires a predicate even for an all-active public call. Generate it
    // locally so PTOAS can recognize the all-active load/rearrangement chain
    // and preserve its fusion opportunities.
    auto [mask, mask_type] = emit_full_mask(args[0], "interleave_all");
    operands += ", " + mask;
    types += ", " + mask_type;
  }
  if (ir::IsOp(call, "reg.full_mask")) operands = "\"PAT_ALL\"";
  if (ir::IsOp(call, "reg.create_group_mask")) {
    attrs = " {num_groups = " +
            std::to_string(ir::GetRequiredKwarg<int>(call->kwargs_, "num_groups", "reg.create_group_mask")) +
            ", group_size = " +
            std::to_string(ir::GetRequiredKwarg<int>(call->kwargs_, "group_size", "reg.create_group_mask")) +
            "}";
  }
  if (ir::IsOp(call, "reg.index_gen")) {
    attrs = " {order = \"ASC\"}";
  }
  if (ir::IsOp(call, "reg.compare") || ir::IsOp(call, "reg.compares")) {
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

  // A grouped reduction keeps the same VMI opcode as a full one and selects the
  // grouped form with `group`, which PTOAS reads as the group COUNT, not the
  // group width. `reassoc` can accompany it on reduce_sum, so build both here
  // instead of letting the generic attribute loop below overwrite one.
  if (ir::IsOp(call, "reg.reduce_sum") || ir::IsOp(call, "reg.reduce_max") ||
      ir::IsOp(call, "reg.reduce_min")) {
    std::vector<std::string> attributes;
    const auto group_size = call->GetKwarg<int>("group_size", 0);
    if (group_size > 0) {
      auto source = ir::As<ir::RegType>(args[0]->GetType());
      INTERNAL_CHECK_SPAN(source, call->span_) << "Internal error: reduction source must be a register";
      attributes.push_back("group = " + std::to_string(source->GetShape() / group_size));
    }
    if (call->GetKwarg<bool>("reassoc", false)) attributes.emplace_back("reassoc");
    for (const auto& attr : attributes) {
      attrs += attrs.empty() ? " {" : ", ";
      attrs += attr;
    }
    if (!attrs.empty()) attrs += "}";
  }
  if (ir::IsOp(call, "reg.group_broadcast")) {
    // num_groups is the source lane count; the schema derives it there too.
    auto source = ir::As<ir::RegType>(args[0]->GetType());
    INTERNAL_CHECK_SPAN(source, call->span_) << "Internal error: group_broadcast source must be a register";
    attrs = " {num_groups = " + std::to_string(source->GetShape()) + "}";
  }

  // Only schemas with a pmode attribute may emit it. Carry operations have
  // a mandatory governing mask but no pmode, unlike ordinary arithmetic.
  for (const auto& kwarg : call->kwargs_) {
    if (kwarg.first == "pmode") attrs = " {pmode = \"zero\"}";
  }
  if (auto tuple = ir::As<ir::TupleType>(validated->GetType())) {
    std::vector<std::string> names;
    std::string results, result_types;
    for (size_t i = 0; i < tuple->types_.size(); ++i) {
      auto type = ir::As<ir::RegType>(tuple->types_[i]);
      INTERNAL_CHECK_SPAN(type, call->span_) << "Internal error: RegOp tuple must contain registers";
      auto name = NewNamedTemp(result->name_hint_ + "_" + std::to_string(i));
      if (i) {
        results += ", ";
        result_types += ", ";
      }
      results += name;
      result_types += signed_mull ? mull_type(type) : GetRegTypeString(type);
      names.push_back(name);
    }
    Emit(results + " = " + spec->emitter + " " + operands + attrs + " : " + types + " -> " + result_types);
    if (signed_mull) {
      for (size_t i = 0; i < names.size(); ++i) {
        auto type = ir::As<ir::RegType>(tuple->types_[i]);
        auto typed = NewNamedTemp("mull_signed_result");
        Emit(typed + " = pto.vmi.vinterpret_cast " + names[i] + " : " + mull_type(type) + " -> " +
             GetRegTypeString(type));
        names[i] = typed;
      }
    }
    // Masked reg.intlv/reg.dintlv keep their predicate: PTOAS >= 0.74 owns
    // inactive-lane zeroing for both outputs without converting active BF16
    // payloads. Bind native results directly, without PyPTO vsel/bitcast
    // compensation. The independent small-VL placement issue is PTOAS #1659.
    reg_tuple_values_[result.get()] = std::move(names);
    return true;
  }
  auto name = NewNamedTemp(result->name_hint_);
  auto type = GetRegTypeString(ir::As<ir::RegType>(validated->GetType()));
  if (ir::IsOp(call, "reg.axpy") || ir::IsOp(call, "reg.lrelu")) {
    // These scalar-float VMI schemas provide only MLIR's generic assembly form.
    Emit(name + " = \"" + spec->emitter + "\"(" + operands + ")" + attrs + " : (" + types + ") -> " + type);
  } else {
    Emit(name + " = " + spec->emitter + " " + operands + attrs + " : " +
         (types.empty() ? type : types + " -> " + type));
  }
  BindVarToMlir(result, name);
  return true;
}
}  // namespace pypto::codegen
