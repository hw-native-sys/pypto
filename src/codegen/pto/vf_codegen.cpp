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

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"

namespace pypto::codegen {
namespace {
using namespace ir;  // NOLINT(build/namespaces)

class VFOperands : public var_collectors::VarDefUseCollector {
 public:
  std::vector<CallPtr> memory;
  std::map<const Var*, VarPtr> aliases;
  std::map<const Var*, VarPtr> references;
  bool has_control_flow = false;

  void VisitStmt(const StmtPtr& stmt) override {
    has_control_flow |= As<ForStmt>(stmt) || As<WhileStmt>(stmt) || As<IfStmt>(stmt);
    var_collectors::VarDefUseCollector::VisitStmt(stmt);
  }

 protected:
  void VisitExpr_(const VarPtr& var) override {
    references.emplace(var.get(), var);
    var_collectors::VarDefUseCollector::VisitExpr_(var);
  }
  void VisitExpr_(const IterArgPtr& var) override {
    references.emplace(var.get(), var);
    var_collectors::VarDefUseCollector::VisitExpr_(var);
  }
  void VisitStmt_(const AssignStmtPtr& stmt) override {
    if (auto call = As<Call>(stmt->value_); call && IsOp(call, "reg.store")) {
      aliases.emplace(stmt->var_.get(), AsVarLike(call->args_[0]));
    }
    references.emplace(stmt->var_.get(), stmt->var_);
    var_collectors::VarDefUseCollector::VisitStmt_(stmt);
  }
  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "reg.load") || IsOp(call, "reg.store")) memory.push_back(call);
    IRVisitor::VisitExpr_(call);
  }
};
}  // namespace

void PTOCodegen::EmitVFHelperCall(const ir::RegScopeStmtPtr& scope) {
  CHECK_SPAN(backend_->GetHandler()->GetPtoTargetArch() == "a5" &&
                 fs_.current_function->func_type_ == ir::FunctionType::AIV,
             scope->span_)
      << "VF execution requires an A5 AIV function";
  CHECK_SPAN(single_vf_function_, scope->span_) << "VF execution does not yet support grouped/peer functions";
  CHECK_SPAN(emit_tile_addr_ && !fs_.buffer_ir, scope->span_)
      << "VF execution requires the PyPTO Tile memory planner";
  VFOperands operands;
  operands.VisitStmt(scope->body_);
  // Keep the execution subset within the straight-line synchronization proof.
  // Diagnostic codegen still supports register control flow.
  CHECK_SPAN(!operands.has_control_flow, scope->span_)
      << "VF execution currently requires straight-line RegScope; unroll static loops before codegen";

  // Collapse SSA store versions to their existing caller view. Memoization
  // bounds chained stores to O(N log N); identity, never names, identifies views.
  std::map<const ir::Var*, ir::VarPtr> roots;
  auto resolve = [&](ir::VarPtr var) {
    std::vector<ir::VarPtr> path;
    std::set<const ir::Var*> seen;
    while (operands.aliases.count(var.get()) || fs_.tile_aliases.count(var.get())) {
      if (auto it = roots.find(var.get()); it != roots.end()) {
        var = it->second;
        break;
      }
      CHECK_SPAN(seen.insert(var.get()).second, scope->span_) << "Cyclic register Tile alias";
      path.push_back(var);
      var =
          operands.aliases.count(var.get()) ? operands.aliases.at(var.get()) : fs_.tile_aliases.at(var.get());
    }
    for (const auto& item : path) roots[item.get()] = var;
    return var;
  };

  PTOCodegen helper(backend_);
  helper.emit_source_loc_ = emit_source_loc_;
  helper.inside_reg_scope_ = true;
  helper.fs_.current_function = fs_.current_function;
  helper.fs_.constants_indent = "    ";
  helper.indent_level_ = 2;
  VFHelper artifact;
  artifact.symbol = "__pypto_vf_" + fs_.current_function->name_ + "_" + std::to_string(vf_helpers_.size());
  artifact.adapter_symbol = artifact.symbol + "_tile_adapter";
  std::vector<std::string> arguments;
  std::map<const ir::Var*, size_t> captures;
  std::vector<int64_t> origins;
  std::vector<ir::TileTypePtr> capture_tiles;
  size_t base_index = 0;
  // Memory captures precede scalar captures, both in deterministic first-use
  // order. Keep Tile operands at the call site so PTOAS can relate the helper's
  // effects to surrounding transfers and aliases. Only the adapter extracts UB
  // pointers, after sync analysis has modeled its annotated Tile call.
  for (const auto& call : operands.memory) {
    auto target = ir::AsVarLike(call->args_[0]);
    CHECK_SPAN(target, call->span_) << "VF memory capture requires an external Tile variable";
    auto root = resolve(target);
    auto tile = ir::As<ir::TileType>(root->GetType());
    INTERNAL_CHECK_SPAN(tile, call->span_) << "VF memory capture must have Tile type";
    // Incoming Tile pointers have independent runtime addresses. Their MemRef
    // offsets (including those of derived views) are not planned UB addresses.
    CHECK_SPAN(!IsParameterBackedTile(tile), call->span_)
        << "VF shared-base captures do not support parameter-backed Tiles or their views";
    // The helper reconstructs every capture from one pointer and static MemRef
    // differences. A selected descriptor can point at another candidate, so its
    // representative offset cannot establish that shared coordinate system.
    CHECK_SPAN(!IsSelectedTileName(GetExprAsCode(root)), call->span_)
        << "VF shared-base captures do not support runtime-selected Tiles or their views; "
           "materialize the selection into independent fixed storage before RegScope";
    CHECK_SPAN(!operands.var_defs.count(root.get()), call->span_)
        << "VF Tile storage must be defined outside RegScope";
    // Check every access, including later uses of an already captured Tile.
    // The diagnostic VMI path accepts dynamic offsets; executable helpers
    // require the static addresses understood by synchronization admission.
    auto effect = ir::GetRegMemoryEffect(call);
    CHECK_SPAN(effect.root && effect.begin, call->span_)
        << "Cannot prove register offset bounds or static VF capture address";
    auto [it, inserted] = captures.emplace(root.get(), arguments.size());
    if (inserted) {
      // These are the same physical UB addresses emitted by the PyPTO memory
      // planner in alloc_tile. tile_buf_addr returns the MemRef base; the
      // reused VMI emitter adds TileView.start_offset to each access itself.
      // Including that view origin here would apply it twice. Do not infer
      // disjointness from Var identity: reused allocations can also alias.
      auto origin = tile->memref_ ? ir::As<ir::ConstInt>((*tile->memref_)->byte_offset_) : nullptr;
      CHECK_SPAN(origin && origin->value_ >= 0, call->span_)
          << "Cannot prove register offset bounds or static VF capture address";
      CHECK_SPAN(capture_tiles.empty() || tile->dtype_ == capture_tiles.front()->dtype_, call->span_)
          << "VF shared-base captures currently require identical element types";
      origins.push_back(origin->value_);
      capture_tiles.push_back(tile);
      if (origins.back() < origins[base_index]) base_index = origins.size() - 1;
      arguments.push_back(GetExprAsCode(root));
      artifact.caller_parameter_types.push_back(GetExprTypeAnnotation(root));
      artifact.parameter_effects.emplace_back("read");
    }
    // A masked or partial store must not be treated as overwriting the whole
    // Tile. Readwrite preserves dependencies on the untouched contents without
    // claiming that masked-off lanes physically read memory. Store-version
    // aliases share this capture and accumulate its conservative effect.
    if (IsOp(call, "reg.store")) artifact.parameter_effects[it->second] = "readwrite";
  }
  std::vector<std::string> addresses;
  if (!capture_tiles.empty()) {
    const auto dtype = capture_tiles.front()->dtype_;
    const auto element_bytes = static_cast<int64_t>(dtype.GetByte());
    INTERNAL_CHECK_SPAN(element_bytes > 0, scope->span_)
        << "VF capture element must have a positive byte size";
    // VMI memory operands carry signedness, unlike ordinary signless Tiles.
    const auto element = dtype.IsSignedInt() ? "si" + std::to_string(dtype.GetBit()) : GetTypeString(dtype);
    artifact.parameter_types.push_back("!pto.ptr<" + element + ", ub>");
    artifact.parameter_indices.push_back(base_index);
    helper.fs_.used_ssa_names.insert("arg0");
    for (size_t i = 0; i < origins.size(); ++i) {
      const int64_t relative = origins[i] - origins[base_index];
      artifact.capture_byte_offsets.push_back(relative);
      std::string address = "%arg0";
      if (relative != 0) {
        address = helper.NewNamedTemp("capture" + std::to_string(i));
        const auto offset = helper.GetOrEmitConstant(relative / element_bytes, DataType::INDEX);
        helper.Emit(address + " = pto.addptr %arg0, " + offset + " : <" + element + ", ub> -> <" + element +
                    ", ub>");
      }
      addresses.push_back(address);
    }
    for (const auto& call : operands.memory) {
      const auto target = ir::AsVarLike(call->args_[0]);
      const auto& address = addresses[captures.at(resolve(target).get())];
      helper.reg_addresses_[target.get()] = address;
      helper.BindVarToMlir(target, address);
    }
  }
  for (const auto* raw : operands.var_uses_ordered) {
    if (operands.var_defs.count(raw) || ir::As<ir::TileType>(raw->GetType())) continue;
    auto var = operands.references.at(raw);
    auto scalar = ir::As<ir::ScalarType>(var->GetType());
    CHECK_SPAN(scalar, var->span_) << "VF captures must be UB views or supported scalars";
    const auto dtype = scalar->dtype_;
    CHECK_SPAN(dtype == DataType::INT32 || dtype == DataType::UINT32 || dtype == DataType::FP32 ||
                   dtype == DataType::INDEX,
               var->span_)
        << "VF scalar capture supports i32, u32, f32 and index only";
    // Reserve manually spelled parameters before body emission, just as the
    // normal function emitter does. Local DSL names may also be argN/captureN.
    const auto argument_name = "arg" + std::to_string(artifact.parameter_types.size());
    helper.fs_.used_ssa_names.insert(argument_name);
    helper.BindVarToMlir(var, "%" + argument_name);
    artifact.parameter_indices.push_back(arguments.size());
    artifact.capture_byte_offsets.push_back(0);
    arguments.push_back(GetExprAsCode(var));
    artifact.parameter_types.push_back(GetTypeString(dtype));
    artifact.caller_parameter_types.push_back(GetTypeString(dtype));
    artifact.parameter_effects.emplace_back("none");
  }

  helper.EmitStructural("pto.vecscope {");
  helper.IncreaseIndent();
  helper.VisitStmt(scope->body_);
  helper.DecreaseIndent();
  helper.EmitStructural("}");
  helper.Emit("return");
  std::string signature, types, values;
  for (size_t i = 0; i < artifact.parameter_types.size(); ++i) {
    if (i) signature += ", ";
    signature += "%arg" + std::to_string(i) + ": " + artifact.parameter_types[i];
  }
  for (size_t i = 0; i < arguments.size(); ++i) {
    if (i) {
      types += ", ";
      values += ", ";
    }
    types += artifact.caller_parameter_types[i];
    values += arguments[i];
  }
  artifact.source =
      "module attributes {pto.target_arch = \"a5\", pto.backend = \"vpto\"} {\n"
      "  func.func @" +
      artifact.symbol + "(" + signature + ") attributes {pto.aiv} {\n" + helper.fs_.constants_section.str() +
      helper.stream_.str() + "  }\n}\n";
  // The adapter's PIPE_V/effect contract lets PTOAS place dependencies here.
  // The A5 VF exits after its register instructions complete; this models the
  // caller boundary, not VF-internal UB hazards. Kernel-tail sync stays enabled.
  Emit("func.call @" + artifact.adapter_symbol + "(" + values + ") : (" + types + ") -> ()");
  for (const auto& [raw, source] : operands.aliases) {
    const auto root = resolve(source);
    const auto& result = operands.references.at(raw);
    BindVarToMlir(result, GetExprAsCode(root));
    const auto ctx = GetCommCtxSSAFor(root.get());
    if (!ctx.empty()) RegisterCommCtxFor(result, ctx);
  }
  vf_helpers_.push_back(std::move(artifact));
}

void PTOCodegen::EmitVFAdapter(const VFHelper& helper) {
  const auto count = helper.parameter_types.size();
  const auto caller_count = helper.caller_parameter_types.size();
  INTERNAL_CHECK(helper.parameter_indices.size() == count &&
                 helper.parameter_effects.size() == caller_count &&
                 helper.capture_byte_offsets.size() == caller_count)
      << "VF adapter signatures and effects must have matching arity";
  stream_ << "  func.func private @" << helper.symbol << "(";
  for (size_t i = 0; i < count; ++i) {
    if (i) stream_ << ", ";
    stream_ << helper.parameter_types[i];
  }
  stream_ << ")\n  func.func private @" << helper.adapter_symbol << "(";
  for (size_t i = 0; i < caller_count; ++i) {
    if (i) stream_ << ", ";
    stream_ << "%arg" << i << ": " << helper.caller_parameter_types[i];
  }
  stream_ << ") attributes {pto.ptodsl.subkernel_helper = \"simd\", pto.tileop.effects = [";
  for (size_t i = 0; i < caller_count; ++i) {
    if (i) stream_ << ", ";
    stream_ << '"' << helper.parameter_effects[i] << '"';
  }
  stream_ << "]} {\n";
  // Emit a real PTO adapter definition, not a postprocessed C++ declaration.
  // PTOAS analyzes the annotated call before lowering Tile.data() in EmitC.
  for (size_t i = 0; i < count; ++i) {
    const auto caller = helper.parameter_indices[i];
    INTERNAL_CHECK(caller < caller_count) << "Invalid VF adapter argument mapping";
    if (helper.caller_parameter_types[caller] != helper.parameter_types[i]) {
      stream_ << "    %ptr" << caller << " = pto.tile_buf_addr %arg" << caller << " : "
              << helper.caller_parameter_types[caller] << " -> " << helper.parameter_types[i] << "\n";
    }
  }
  stream_ << "    func.call @" << helper.symbol << "(";
  for (size_t i = 0; i < count; ++i) {
    if (i) stream_ << ", ";
    const auto caller = helper.parameter_indices[i];
    stream_ << (helper.caller_parameter_types[caller] == helper.parameter_types[i] ? "%arg" : "%ptr")
            << caller;
  }
  stream_ << ") : (";
  for (size_t i = 0; i < count; ++i) {
    if (i) stream_ << ", ";
    stream_ << helper.parameter_types[i];
  }
  stream_ << ") -> ()\n    return\n  }\n";
}

}  // namespace pypto::codegen
