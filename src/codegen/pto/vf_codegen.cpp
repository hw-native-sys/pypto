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
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "pypto/ir/transforms/utils/hardware_loop_analysis.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/transforms/utils/vf_helpers.h"
#include "pypto/ir/type.h"
#include "src/codegen/pto/pto_scalar_floor_div_mod_plan.h"

namespace pypto::codegen {
namespace {
using namespace ir;  // NOLINT(build/namespaces)

class VFOperands : public var_collectors::VarDefUseCollector {
 public:
  std::vector<CallPtr> memory;
  std::map<const Var*, VarPtr> aliases;
  std::map<const Var*, VarPtr> references;
  bool has_control_flow = false;
  bool micro = false;
  bool preserve_order = false;
  std::vector<ForStmtPtr> loops;

  void VisitStmt(const StmtPtr& stmt) override {
    has_control_flow |= As<WhileStmt>(stmt) || As<IfStmt>(stmt);
    if (auto loop = As<ForStmt>(stmt)) loops.push_back(loop);
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
    if (auto call = As<Call>(stmt->value_); call && (IsOp(call, "reg.store") || IsOp(call, "reg.scatter"))) {
      aliases.emplace(stmt->var_.get(), AsVarLike(call->args_[0]));
    }
    if (auto call = As<Call>(stmt->value_); call && IsOp(call, "reg.squeeze_store_flush")) {
      aliases.emplace(stmt->var_.get(), AsVarLike(call->args_[1]));
    }
    references.emplace(stmt->var_.get(), stmt->var_);
    var_collectors::VarDefUseCollector::VisitStmt_(stmt);
  }
  void VisitExpr_(const CallPtr& call) override {
    preserve_order |= IsOp(call, "reg.mask_bitcast") ||
                      (IsOp(call, "reg.load") && ir::GetStringKwarg(call->kwargs_, "dist", "") == "US_B8");
    micro |= preserve_order || IsOp(call, "reg.squeeze_store_init");
    if (IsOp(call, "reg.load") || IsOp(call, "reg.store") || IsOp(call, "reg.gather") ||
        IsOp(call, "reg.scatter") || IsOp(call, "reg.squeeze_store_step") ||
        IsOp(call, "reg.squeeze_store_flush")) {
      memory.push_back(call);
      // VMI adds the view origin when materializing an address. Its scalar
      // dependencies live in the Tile type, so ordinary operand traversal
      // would otherwise omit them from the helper's scalar captures.
      const auto tile = As<TileType>(GetRegMemoryEffect(call).target->GetType());
      if (tile && tile->tile_view_ && tile->tile_view_->start_offset) {
        VisitExpr(tile->tile_view_->start_offset);
      }
    }
    IRVisitor::VisitExpr_(call);
  }
};

}  // namespace

bool PTOCodegen::TryEmitVFCall(const ir::CallPtr& call, const ir::VarPtr& result) {
  auto function = ir::ResolveVFCall(call, vf_program_);
  if (!function) return false;
  std::vector<std::pair<ir::VarPtr, ir::ExprPtr>> borrowed_args;
  auto scope = ir::BindVFRegion(function, call, &borrowed_args);
  bound_vf_regions_.push_back(scope);
  for (const auto& [borrowed, actual] : borrowed_args) {
    BindVarToMlir(borrowed, GetExprAsCode(actual));
    if (auto var = ir::AsVarLike(actual)) fs_.tile_aliases[borrowed.get()] = var;
  }
  // Logical capture discovery belongs to OutlineRegScopes. The existing VMI
  // emitter now sees declaration arguments bound to the final caller storage.
  VisitStmt_(scope);
  // Unused Tile captures may have no owner in the retained region. Erase their
  // raw-pointer keys before borrowed_args dies and an allocator can reuse them.
  for (const auto& [borrowed, actual] : borrowed_args) {
    fs_.tile_aliases.erase(borrowed.get());
    fs_.var_to_mlir.erase(borrowed.get());
  }
  auto aliases = ir::VFReturnedArguments(function, call);
  if (result && aliases.size() == 1) {
    BindVarToMlir(result, GetExprAsCode(aliases[0]));
    if (auto source = ir::AsVarLike(aliases[0])) fs_.tile_aliases[result.get()] = source;
  } else if (result) {
    vf_return_aliases_[result.get()] = std::move(aliases);
  }
  return true;
}

void PTOCodegen::EmitVFHelperCall(const ir::RegScopeStmtPtr& scope) {
  CHECK_SPAN(backend_->GetHandler()->GetPtoTargetArch() == "a5" &&
                 fs_.current_function->func_type_ == ir::FunctionType::AIV,
             scope->span_)
      << "VF execution requires an A5 AIV function";
  CHECK_SPAN(supported_vf_entries_, scope->span_)
      << "VF execution does not support grouped/peer functions other than one AIC/AIV pair";
  CHECK_SPAN(emit_tile_addr_ && !fs_.buffer_ir, scope->span_)
      << "VF execution requires the PyPTO Tile memory planner";
  VFOperands operands;
  operands.VisitStmt(scope->body_);
  // Preserve canonical hardware loops in the generated PTO. Backend lowering
  // and synchronization belong to PTOAS; runtime branches are not admitted here.
  CHECK_SPAN(!operands.has_control_flow, scope->span_)
      << "VF execution supports straight-line RegScope or canonical hardware loops nested up to four levels";
  if (!operands.loops.empty()) {
    for (const auto& loop : operands.loops) {
      const auto stop_type = ir::As<ir::ScalarType>(loop->stop_->GetType());
      CHECK_SPAN((loop->hardware_loop_policy_ == ir::HardwareLoopPolicy::Require ||
                  (operands.micro && ir::HasCanonicalHardwareLoopBounds(loop))) &&
                     stop_type && (stop_type->dtype_.IsInt() || stop_type->dtype_ == DataType::INDEX),
                 loop->span_)
          << "VF hardware loop lowering requires canonical integer loops";
      // Register/mask carries remain inside the VF. Tile carries would need
      // capture alias/effect analysis across iterations and are not admitted.
      for (const auto& arg : loop->iter_args_) {
        CHECK_SPAN(ir::As<ir::RegType>(arg->GetType()), loop->span_)
            << "VF hardware loops support register or mask loop-carried values only";
      }
    }
  }

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
  helper.micro_reg_scope_ = operands.micro;
  helper.fs_.current_function = fs_.current_function;
  helper.fs_.truncating_floor_div_mod = PrepareScalarFloorDivMod(scope->body_);
  helper.fs_.constants_indent = "    ";
  helper.indent_level_ = 2;
  VFHelper artifact;
  artifact.symbol = "__pypto_vf_" + fs_.current_function->name_ + "_" + std::to_string(vf_helpers_.size());
  artifact.adapter_symbol = artifact.symbol + "_tile_adapter";
  std::vector<std::string> arguments;
  std::map<const ir::Var*, size_t> captures;
  struct Capture {
    ir::TileTypePtr tile;
    DataType base_dtype;
    // Empty for the planner's common base, otherwise the defining TPOP descriptor.
    std::string received;
    int64_t offset;
    uint64_t planned_bytes = 0;
  };
  std::vector<Capture> memory;
  std::map<int, std::pair<int64_t, int64_t>> reservations;
  std::map<std::string, size_t> base_operands;
  // Memory captures precede scalar captures, both in deterministic first-use
  // order. Keep Tile operands at the call site so PTOAS can relate the helper's
  // effects to surrounding transfers and aliases. Only the adapter extracts UB
  // pointers, after sync analysis has modeled its annotated Tile call.
  for (const auto& call : operands.memory) {
    auto target = ir::AsVarLike(ir::GetRegMemoryEffect(call).target);
    CHECK_SPAN(target, call->span_) << "VF memory capture requires an external Tile variable";
    auto root = resolve(target);
    auto tile = ir::As<ir::TileType>(root->GetType());
    INTERNAL_CHECK_SPAN(tile, call->span_) << "VF memory capture must have Tile type";
    // Incoming Tile pointers have independent runtime addresses. Their MemRef
    // offsets (including those of derived views) are not planned UB addresses.
    CHECK_SPAN(!IsParameterBackedTile(tile), call->span_)
        << "VF shared-base captures do not support parameter-backed Tiles or their views";
    CHECK_SPAN(!operands.var_defs.count(root.get()), call->span_)
        << "VF Tile storage must be defined outside RegScope";
    const auto handle = GetExprAsCode(root);
    const auto fifo = fs_.c2v_views.find(handle);
    const bool borrowed = fifo != fs_.c2v_views.end();
    const auto allocation = tile->memref_.value_or(nullptr);
    CHECK_SPAN(borrowed || allocation, call->span_)
        << "VF capture requires planned local UB or a contiguous view of a known AIV C2V receive";
    auto effect = ir::GetRegMemoryEffect(call);
    CHECK_SPAN(
        (borrowed || effect.root) && (effect.begin || !operands.loops.empty() || IsOp(call, "reg.gather") ||
                                      IsOp(call, "reg.scatter") || IsOp(call, "reg.squeeze_store_step") ||
                                      IsOp(call, "reg.squeeze_store_flush")),
        call->span_)
        << "Cannot prove register offset bounds or static VF capture address";
    if (borrowed) {
      const auto& view = fifo->second;
      const auto pipe = fs_.c2v_buffers.find(view.pipe);
      CHECK_SPAN(pipe != fs_.c2v_buffers.end(), call->span_)
          << "VF C2V capture requires an initialized pipe with a reserved local Vec buffer covering its ring";
      const auto storage_bytes =
          ir::cross_core_pipe::TryGetTileSlotSizeBytes(fs_.ssa_to_tile_type.at(view.received));
      const auto access = ir::As<ir::TileType>(target->GetType());
      const auto view_bytes = ir::cross_core_pipe::TryGetTileSlotSizeBytes(access);
      const auto element_bytes = static_cast<int64_t>(access->dtype_.GetByte());
      const auto start = access->tile_view_ ? access->tile_view_->start_offset : nullptr;
      const auto start_const = ir::As<ir::ConstInt>(start);
      // The pointer is runtime, but its maximum physical window is fixed by
      // this receive. Include any additional element origin used by VMI.
      CHECK_SPAN(
          storage_bytes && *storage_bytes <= pipe->second.slot_size && view_bytes && element_bytes > 0 &&
              *storage_bytes >= view.byte_offset && *view_bytes <= *storage_bytes - view.byte_offset &&
              (!start ||
               (start_const && start_const->value_ >= 0 &&
                start_const->value_ <= (*storage_bytes - view.byte_offset - *view_bytes) / element_bytes)),
          call->span_)
          << "VF C2V view must fit within the received tile";
    }
    auto [it, inserted] = captures.emplace(root.get(), arguments.size());
    if (inserted) {
      Capture capture{tile, tile->dtype_, {}, 0};
      if (borrowed) {
        const auto& view = fifo->second;
        reservations.emplace(view.pipe, fs_.c2v_buffers.at(view.pipe).reservation);
        capture.received = view.received;
        fs_.vf_received_tiles.insert(view.received);
        capture.offset = view.byte_offset;
        capture.base_dtype = fs_.ssa_to_tile_type.at(view.received)->dtype_;
        arguments.push_back(view.received);
        artifact.caller_parameter_types.push_back(GetSSATileBufType(view.received));
      } else {
        const auto origin = ir::As<ir::ConstInt>(allocation->byte_offset_);
        CHECK_SPAN(origin && origin->value_ >= 0, call->span_)
            << "Cannot prove register offset bounds or static VF capture address";
        capture.offset = origin->value_;
        capture.planned_bytes = allocation->size_;
        arguments.push_back(handle);
        artifact.caller_parameter_types.push_back(GetExprTypeAnnotation(root));
      }
      const size_t index = memory.size();
      auto [base, fresh] = base_operands.emplace(capture.received, index);
      if (!fresh && !borrowed && capture.offset < memory[base->second].offset) base->second = index;
      memory.push_back(std::move(capture));
      artifact.parameter_effects.emplace_back("read");
    }
    // A masked or partial store must not be treated as overwriting the whole
    // Tile. Readwrite preserves dependencies on the untouched contents without
    // claiming that masked-off lanes physically read memory. Store-version
    // aliases share this capture and accumulate its conservative effect.
    if (IsOp(call, "reg.store") || IsOp(call, "reg.scatter") || IsOp(call, "reg.squeeze_store_step") ||
        IsOp(call, "reg.squeeze_store_flush")) {
      artifact.parameter_effects[it->second] = "readwrite";
    }
  }
  // AllocateMemoryAddr places local Tiles outside reserved FIFO storage. Check
  // the actual ranges as generate_bundle also accepts already-addressed IR.
  for (auto it = reservations.begin(); it != reservations.end(); ++it) {
    const auto [begin, end] = it->second;
    for (auto other = reservations.begin(); other != it; ++other) {
      CHECK_SPAN(other->second.second <= begin || end <= other->second.first, scope->span_)
          << "VF captures from different C2V pipes require disjoint reservations";
    }
    for (const auto& capture : memory) {
      if (!capture.received.empty()) continue;
      CHECK_SPAN(
          capture.offset >= end || (capture.offset <= begin &&
                                    capture.planned_bytes <= static_cast<uint64_t>(begin - capture.offset)),
          scope->span_)
          << "VF planned UB capture overlaps a reserved C2V buffer";
    }
  }

  auto element_type = [&](DataType dtype) {
    return dtype.IsSignedInt() ? "si" + std::to_string(dtype.GetBit()) : GetTypeString(dtype);
  };
  std::map<std::string, size_t> base_parameters;
  std::vector<std::string> addresses;
  for (size_t i = 0; i < memory.size(); ++i) {
    const auto& capture = memory[i];
    const size_t caller = base_operands.at(capture.received);
    const auto& base = memory[caller];
    auto [param, fresh] = base_parameters.emplace(capture.received, artifact.parameter_types.size());
    const auto argument = "arg" + std::to_string(param->second);
    const auto element = element_type(base.base_dtype);
    if (fresh) {
      artifact.parameter_types.push_back("!pto.ptr<" + element + ", ub>");
      artifact.parameter_indices.push_back(caller);
      helper.fs_.used_ssa_names.insert(argument);
    }
    // A FIFO parameter is always the canonical receive handle, even when the
    // first use is a slice. The VMI emitter separately adds TileView.start_offset.
    const int64_t relative = capture.offset - (capture.received.empty() ? base.offset : 0);
    artifact.capture_base_indices.push_back(static_cast<int64_t>(param->second));
    artifact.capture_byte_offsets.push_back(relative);
    const auto dtype = capture.tile->dtype_;
    const auto target = element_type(dtype);
    const auto bytes = static_cast<int64_t>(dtype.GetByte());
    CHECK_SPAN(bytes > 0 && relative >= 0 && relative % bytes == 0, scope->span_)
        << "VF capture displacement must align to its element size";
    std::string address = "%" + argument;
    if (dtype != base.base_dtype) {
      auto cast = helper.NewNamedTemp("capture_typed" + std::to_string(i));
      helper.Emit(cast + " = pto.castptr " + address + " : !pto.ptr<" + element + ", ub> -> !pto.ptr<" +
                  target + ", ub>");
      address = cast;
    }
    if (relative != 0) {
      const auto offset = helper.GetOrEmitConstant(relative / bytes, DataType::INDEX);
      auto adjusted = helper.NewNamedTemp("capture" + std::to_string(i));
      helper.Emit(adjusted + " = pto.addptr " + address + ", " + offset + " : <" + target +
                  ", ub>, index -> <" + target + ", ub>");
      address = adjusted;
    }
    addresses.push_back(address);
  }
  for (const auto& call : operands.memory) {
    const auto target = ir::AsVarLike(ir::GetRegMemoryEffect(call).target);
    const auto& address = addresses[captures.at(resolve(target).get())];
    helper.reg_addresses_[target.get()] = address;
    helper.BindVarToMlir(target, address);
  }
  for (const auto* raw : operands.var_uses_ordered) {
    if (operands.var_defs.count(raw) || ir::As<ir::TileType>(raw->GetType())) continue;
    auto var = operands.references.at(raw);
    auto scalar = ir::As<ir::ScalarType>(var->GetType());
    CHECK_SPAN(scalar, var->span_) << "VF captures must be UB views or supported scalars";
    const auto dtype = scalar->dtype_;
    CHECK_SPAN(dtype.IsInt() || dtype == DataType::FP32 || dtype == DataType::INDEX, var->span_)
        << "VF scalar capture supports integers, f32 and index only";
    // Reserve manually spelled parameters before body emission, just as the
    // normal function emitter does. Local DSL names may also be argN/captureN.
    const auto argument_name = "arg" + std::to_string(artifact.parameter_types.size());
    helper.fs_.used_ssa_names.insert(argument_name);
    helper.BindVarToMlir(var, "%" + argument_name);
    artifact.parameter_indices.push_back(arguments.size());
    artifact.capture_base_indices.push_back(-1);
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
  // Runtime reads this header to select PTOAS flags during artifact replay.
  artifact.source = std::string(operands.preserve_order ? "// pypto.reg_physical_order: preserve\n" : "") +
                    "// pypto.reg_auto_sync: " + (scope->AutoSync() ? "true\n" : "false\n") +
                    "module attributes {pto.target_arch = \"a5\", pto.backend = \"vpto\"} {\n"
                    "  func.func @" +
                    artifact.symbol + "(" + signature + ") attributes {pto.aiv} {\n" +
                    helper.fs_.constants_section.str() + helper.stream_.str() + "  }\n}\n";
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
  RecordEntryHelperUse(artifact.symbol);
  vf_helpers_.push_back(std::move(artifact));
}

void PTOCodegen::EmitVFC2VRelease(const ir::ExprPtr& tile) {
  if (fs_.vf_received_tiles.empty()) return;
  const auto view = fs_.c2v_views.find(GetExprAsCode(tile));
  if (view == fs_.c2v_views.end() || !fs_.vf_received_tiles.count(view->second.received)) return;
  const auto& received = view->second.received;
  const auto type = GetSSATileBufType(received);
  auto [symbol, inserted] = c2v_release_symbols_.try_emplace(type);
  if (inserted) symbol->second = "__pypto_c2v_release_" + std::to_string(c2v_release_symbols_.size() - 1);
  // TFREE returns UB credit on PIPE_V, but its operandless PTO form does not
  // expose the clobber of the slot to InsertSync. This Tile effect lets PTOAS
  // order outstanding readers (including MTE3) before that credit is issued.
  Emit("func.call @" + symbol->second + "(" + received + ") : (" + type + ") -> ()");
}

void PTOCodegen::EmitVFAdapter(const VFHelper& helper) {
  const auto count = helper.parameter_types.size();
  const auto caller_count = helper.caller_parameter_types.size();
  INTERNAL_CHECK(helper.parameter_indices.size() == count &&
                 helper.parameter_effects.size() == caller_count &&
                 helper.capture_base_indices.size() == caller_count &&
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
      if (helper.caller_parameter_types[caller].rfind("!pto.ptr<", 0) == 0) {
        // The EmitC caller's default GM space and the native SIMT ABI's
        // explicit GM space denote the same address, including its view origin.
        stream_ << "    %ptr" << caller << " = pto.castptr %arg" << caller << " : "
                << helper.caller_parameter_types[caller] << " -> " << helper.parameter_types[i] << "\n";
        continue;
      }
      // Tile storage spells signed integers as signless iN. Extract that
      // storage pointer first, then bridge the explicitly signed VMI ABI.
      const auto& parameter_type = helper.parameter_types[i];
      auto storage_type = parameter_type;
      const bool signed_pointer = parameter_type.rfind("!pto.ptr<si", 0) == 0;
      if (signed_pointer) storage_type.erase(std::string("!pto.ptr<").size(), 1);
      stream_ << "    %" << (signed_pointer ? "storage" : "ptr") << caller << " = pto.tile_buf_addr %arg"
              << caller << " : " << helper.caller_parameter_types[caller] << " -> " << storage_type << "\n";
      if (signed_pointer) {
        stream_ << "    %ptr" << caller << " = pto.castptr %storage" << caller << " : " << storage_type
                << " -> " << parameter_type << "\n";
      }
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
