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
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/simt.h"
#include "pypto/ir/simt_analysis.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto::codegen {
namespace {
using namespace ir;  // NOLINT(build/namespaces)

std::string Join(const std::vector<std::string>& values) {
  std::string result;
  for (const auto& value : values) {
    if (!result.empty()) result += ", ";
    result += value;
  }
  return result;
}

// Keep the initial executable scalar subset explicit. In particular, accepting
// unsigned types here would reuse signed comparisons in the ordinary emitter.
// Storage support alone does not establish arithmetic/conversion semantics.
void CheckScalar(DataType dtype, const Span& span) {
  CHECK_SPAN(dtype == DataType::FP32 || dtype == DataType::INT32 || dtype == DataType::INDEX ||
                 dtype == DataType::BOOL,
             span)
      << "SIMT scalar lowering currently supports FP32, INT32, INDEX and BOOL; got " << dtype.ToString();
}

class ScalarSubsetChecker : public IRVisitor {
 public:
  explicit ScalarSubsetChecker(const FunctionPtr& function) {
    const auto sequence = As<SeqStmts>(function->body_);
    final_return_ = sequence && !sequence->stmts_.empty() ? sequence->stmts_.back() : function->body_;
  }
  void VisitExpr(const ExprPtr& expr) override {
    if (auto scalar = As<ScalarType>(expr->GetType())) {
      CheckScalar(scalar->dtype_, expr->span_);
      // The generic IR visitor only traverses Pow operands; without a PTO
      // emitter it would leave the exponent as the expression's result.
      CHECK_SPAN(!IsA<Pow>(expr), expr->span_)
          << "SIMT scalar exponentiation is not yet supported; use explicit multiplication";
      CHECK_SPAN(!scalar->dtype_.IsFloat() || (!IsA<FloorDiv>(expr) && !IsA<FloorMod>(expr)), expr->span_)
          << "SIMT floating-point floor division/modulo is not yet supported";
    }
    IRVisitor::VisitExpr(expr);
  }

 protected:
  void VisitStmt_(const ReturnStmtPtr& stmt) override {
    CHECK_SPAN(stmt == final_return_, stmt->span_)
        << "SIMT lowering does not yet support early return; predicate the body instead";
  }
  void VisitStmt_(const BreakStmtPtr& stmt) override {
    CHECK_SPAN(false, stmt->span_) << "SIMT lowering requires structured loops without break";
  }
  void VisitStmt_(const ContinueStmtPtr& stmt) override {
    CHECK_SPAN(false, stmt->span_) << "SIMT lowering requires structured loops without continue";
  }

 private:
  StmtPtr final_return_;
};
}  // namespace

void PTOCodegen::VisitExpr_(const ir::TupleGetItemExprPtr& op) {
  if (!inside_simt_function_) {
    ir::IRVisitor::VisitExpr_(op);
    return;
  }
  // FlattenCallExpr leaves SIMT while conditions in place. Evaluate the tuple
  // eagerly inside scf.while's before region, including the final false check.
  // Retain emitted SSA values so selecting an earlier slot never repeats its
  // effects after later slots. Each occurrence is visited once, left to right.
  struct EvaluatedValue {
    std::string scalar;
    std::vector<EvaluatedValue> elements;
  };
  auto evaluate = [this](auto&& self, const ir::ExprPtr& expr) -> EvaluatedValue {
    if (auto tuple = ir::As<ir::MakeTuple>(expr)) {
      EvaluatedValue value;
      value.elements.reserve(tuple->elements_.size());
      for (const auto& element : tuple->elements_) {
        value.elements.push_back(self(self, element));
      }
      return value;
    }
    if (auto item = ir::As<ir::TupleGetItemExpr>(expr)) {
      auto tuple = self(self, item->tuple_);
      INTERNAL_CHECK_SPAN(item->index_ >= 0 && static_cast<size_t>(item->index_) < tuple.elements.size(),
                          item->span_)
          << "SIMT tuple projection requires a resolved tuple with a valid index";
      return std::move(tuple.elements[static_cast<size_t>(item->index_)]);
    }
    INTERNAL_CHECK_SPAN(!ir::IsA<ir::TupleType>(expr->GetType()), expr->span_)
        << "SIMT tuple operands must be resolved before scalar codegen";
    VisitExpr(expr);
    return {fs_.current_expr_value, {}};
  };
  fs_.current_expr_value = evaluate(evaluate, op).scalar;
}

std::string PTOCodegen::SimtParameterType(const ir::TypePtr& type) {
  if (auto scalar = ir::As<ir::ScalarType>(type)) {
    CheckScalar(scalar->dtype_, ir::Span::unknown());
    return GetTypeString(scalar->dtype_);
  }
  const auto tile = ir::As<ir::TileType>(type);
  const auto tensor = ir::AsTensorTypeLike(type);
  INTERNAL_CHECK(tile || tensor) << "Internal error: SIMT ABI requires scalar or borrowed storage";
  const auto dtype = tile ? tile->dtype_ : tensor->dtype_;
  CHECK(dtype == DataType::FP32 || dtype == DataType::INT32)
      << "SIMT storage lowering currently supports FP32 and INT32; got " << dtype.ToString();
  if (tile) {
    CHECK(tile->memory_space_ == ir::MemorySpace::Vec) << "SIMT Tile lowering requires Vec (UB) storage";
    const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
    CHECK(view.slayout == ir::TileLayout::none_box)
        << "SIMT Tile lowering requires an affine, unboxed layout";
  }
  return "!pto.ptr<" + GetTypeString(dtype) + (tile ? ", ub>" : ", gm>");
}

std::string PTOCodegen::GenerateSimtFunction(const ir::FunctionPtr& function) {
  const auto threads = function->GetAttr<int>(ir::kSimtMaxThreadsAttr);
  const auto regs = function->GetAttr<int>(ir::kSimtMaxRegsAttr);
  CHECK_SPAN(threads && regs, function->span_) << "Run SpecializeSimtHelpers before SIMT codegen";
  ScalarSubsetChecker(function).VisitStmt(function->body_);
  PTOCodegen body(backend_);
  body.inside_simt_function_ = true;
  body.emit_source_loc_ = emit_source_loc_;
  body.fs_.current_function = function;
  body.IncreaseIndent();
  body.IncreaseIndent();
  body.fs_.constants_indent = body.GetIndent();
  std::vector<std::string> parameters;
  for (size_t i = 0; i < function->params_.size(); ++i) {
    const auto name = "arg" + std::to_string(i);
    body.fs_.used_ssa_names.insert(name);
    body.BindVarToMlir(function->params_[i], "%" + name);
    parameters.push_back("%" + name + ": " + SimtParameterType(function->params_[i]->GetType()));
  }
  body.VisitStmt(function->body_);
  body.Emit("return");
  return "  func.func @" + function->name_ + "(" + Join(parameters) +
         ") attributes {pto.simt_entry, pto.simt_max_threads = " + std::to_string(threads) +
         " : i32, pto.simt_max_regs = " + std::to_string(regs) + " : i32} {\n" +
         body.fs_.constants_section.str() + body.stream_.str() + "  }\n";
}

void PTOCodegen::VisitStmt_(const ir::SimtScopeStmtPtr& scope) {
  CHECK_SPAN(false, scope->span_) << "Run OutlineSimtScopes before code generation";
}

void PTOCodegen::VisitStmt_(const ir::SimtLaunchStmtPtr& launch) {
  CHECK_SPAN(GetBackendHandler()->GetPtoTargetArch() == "a5", launch->span_)
      << "SIMT lowering requires the A5 backend";
  const auto found = simt_functions_.find(launch->callee_->name_);
  INTERNAL_CHECK_SPAN(found != simt_functions_.end(), launch->span_)
      << "Internal error: SIMT helper missing from the codegen unit: " << launch->callee_->name_;
  const auto& function = found->second;
  INTERNAL_CHECK_SPAN(function->params_.size() == launch->args_.size(), launch->span_)
      << "Internal error: SIMT launch/helper arity mismatch";
  // Analyze each helper once, even if many launch sites reuse it. This cache
  // belongs to one Generate/GenerateBundle invocation, just like the artifacts.
  auto effects = simt_effects_.find(function->name_);
  if (effects == simt_effects_.end()) {
    std::vector<std::string> values;
    for (const auto& access : ir::AnalyzeSimtAccess(function)) {
      values.emplace_back(access.writes ? "readwrite" : access.reads ? "read" : "none");
    }
    effects = simt_effects_.emplace(function->name_, std::move(values)).first;
  }
  VFHelper artifact;
  artifact.kind = "simt";
  std::vector<std::string> arguments;
  for (size_t i = 0; i < launch->args_.size(); ++i) {
    const auto& arg = launch->args_[i];
    const auto native_type = SimtParameterType(function->params_[i]->GetType());
    artifact.parameter_types.push_back(native_type);
    artifact.parameter_indices.push_back(i);
    artifact.capture_byte_offsets.push_back(0);
    artifact.parameter_effects.push_back(effects->second[i]);
    if (auto tensor = ir::AsTensorTypeLike(arg->GetType())) {
      // Runtime Tensor pointers already include their descriptor view origin.
      // Only physical strides remain in the helper's type.
      arguments.push_back(GetTensorBasePtr(ir::AsVarLike(arg)));
      artifact.caller_parameter_types.push_back("!pto.ptr<" + GetTypeString(tensor->dtype_) + ">");
    } else {
      const auto argument = GetExprAsCode(arg);
      auto source = argument;
      if (ir::IsA<ir::TileType>(arg->GetType())) {
        auto var = ir::AsVarLike(arg);
        std::vector<const ir::Var*> aliases;
        // Address-based allocation can give a local alias its own SSA handle.
        // Follow IR provenance as well, compressing paths so repeated launches
        // over long alias chains do not repeatedly traverse the same suffix.
        while (var) {
          const auto alias = fs_.tile_aliases.find(var.get());
          if (alias == fs_.tile_aliases.end()) break;
          aliases.push_back(var.get());
          INTERNAL_CHECK_SPAN(aliases.size() <= fs_.tile_aliases.size(), launch->span_)
              << "Internal error: cyclic SIMT Tile alias";
          var = alias->second;
        }
        for (const auto* alias : aliases) fs_.tile_aliases[alias] = var;
        if (var) source = GetExprAsCode(var);
      }
      // A pto.subview retains its parent's pitch, but the specialized slice
      // type may have an empty stride and only the smaller logical shape.
      // Reinterpreting that handle as a dense pointer silently misaddresses
      // later rows. Reshapes of views are equally unsafe without provenance.
      // Require an independently materialized Tile until that provenance is
      // carried through specialization; never repack over the live parent.
      CHECK_SPAN(
          !ir::IsA<ir::TileType>(arg->GetType()) || (!IsTileViewName(argument) && !IsTileViewName(source)),
          launch->span_)
          << "SIMT cannot borrow a Tile view with unproven physical strides; "
             "materialize the window with tile.extract before launching";
      arguments.push_back(argument);
      artifact.caller_parameter_types.push_back(GetExprTypeAnnotation(arg));
    }
  }

  if (separate_vf_) {
    const auto key = std::make_pair(function->name_, artifact.caller_parameter_types);
    auto existing = simt_artifacts_.find(key);
    if (existing == simt_artifacts_.end()) {
      artifact.symbol =
          "__pypto_simt_" + fs_.current_function->name_ + "_" + std::to_string(vf_helpers_.size());
      artifact.adapter_symbol = artifact.symbol + "_adapter";
      std::vector<std::string> signature, params;
      for (size_t i = 0; i < artifact.parameter_types.size(); ++i) {
        params.push_back("%arg" + std::to_string(i));
        signature.push_back(params.back() + ": " + artifact.parameter_types[i]);
      }
      std::string dims;
      std::vector<std::string> dimension_names;
      for (size_t i = 0; i < launch->block_dims_.size(); ++i) {
        const auto name = "%dim" + std::to_string(i);
        dimension_names.push_back(name);
        dims += "    " + name + " = arith.constant " +
                std::to_string(ir::As<ir::ConstInt>(launch->block_dims_[i])->value_) + " : i32\n";
      }
      artifact.source = "module attributes {pto.target_arch = \"a5\", pto.backend = \"vpto\"} {\n" +
                        std::string("  func.func @") + artifact.symbol + "(" + Join(signature) +
                        ") attributes {pto.aiv} {\n" + dims + "    pto.simt_launch @" + function->name_ +
                        "<<<" + Join(dimension_names) + ">>>(" + Join(params) + ") : (" +
                        Join(artifact.parameter_types) + ") -> ()\n    return\n  }\n" +
                        GenerateSimtFunction(function) + "}\n";
      existing = simt_artifacts_.emplace(key, vf_helpers_.size()).first;
      vf_helpers_.push_back(std::move(artifact));
    }
    const auto& emitted = vf_helpers_[existing->second];
    // Artifact reuse across entries must still record every caller's dependency.
    RecordEntryHelperUse(emitted.symbol);
    Emit("func.call @" + emitted.adapter_symbol + "(" + Join(arguments) + ") : (" +
         Join(emitted.caller_parameter_types) + ") -> ()");
  } else {
    // Combined diagnostic output retains native launches and helper definitions.
    // Executable bundles keep Tile operands visible to caller InsertSync instead.
    for (size_t i = 0; i < arguments.size(); ++i) {
      if (artifact.caller_parameter_types[i] == artifact.parameter_types[i]) continue;
      const auto pointer = NewTemp();
      const bool tile = ir::IsA<ir::TileType>(launch->args_[i]->GetType());
      Emit(pointer + (tile ? " = pto.tile_buf_addr " : " = pto.castptr ") + arguments[i] + " : " +
           artifact.caller_parameter_types[i] + " -> " + artifact.parameter_types[i]);
      arguments[i] = pointer;
    }
    std::vector<std::string> dimensions;
    for (const auto& dim : launch->block_dims_) dimensions.push_back(EmitArithOperand(dim, "i32"));
    Emit("pto.simt_launch @" + function->name_ + "<<<" + Join(dimensions) + ">>>(" + Join(arguments) +
         ") : (" + Join(artifact.parameter_types) + ") -> ()");
    used_simt_functions_.insert(function->name_);
  }
  // Output materialization already selected the writable allocation. A launch
  // creates an IR storage version, not a new physical allocation or a copy.
  for (size_t i = 0; i < launch->results_.size(); ++i) {
    const auto arg = ir::AsVarLike(launch->args_[launch->result_param_indices_[i]]);
    const auto& result = launch->results_[i];
    BindVarToMlir(result, GetExprAsCode(arg));
    if (ir::AsTensorTypeLike(arg->GetType())) {
      RegisterBasePtr(result, GetTensorBasePtr(arg));
      const auto view = TryGetTensorView(arg);
      if (!view.empty()) RegisterTensorView(result, view);
    }
  }
}

std::string PTOCodegen::SimtElementOffset(const ir::ExprPtr& storage, const ir::MakeTuplePtr& indices) {
  const auto tile = ir::As<ir::TileType>(storage->GetType());
  const auto tensor = ir::AsTensorTypeLike(storage->GetType());
  INTERNAL_CHECK_SPAN(tile || tensor, storage->span_)
      << "Internal error: SIMT addressing requires borrowed storage";
  const auto& shape = tile ? tile->shape_ : tensor->shape_;
  INTERNAL_CHECK_SPAN(indices && indices->elements_.size() == shape.size(), storage->span_)
      << "Internal error: SIMT indices must match the storage rank after flattening";
  std::vector<ir::ExprPtr> strides;
  ir::ExprPtr origin;
  bool column_major = false;
  if (tile) {
    const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
    strides = view.stride;
    origin = view.start_offset;
    column_major = view.blayout == ir::TileLayout::col_major;
  } else if (tensor->tensor_view_) {
    strides = tensor->tensor_view_->stride;
    column_major = tensor->tensor_view_->layout == ir::TensorLayout::DN;
  }
  std::vector<std::string> physical_strides(shape.size());
  if (!strides.empty()) {
    INTERNAL_CHECK_SPAN(strides.size() == shape.size(), storage->span_)
        << "Internal error: SIMT stride rank mismatch";
    for (size_t i = 0; i < strides.size(); ++i) physical_strides[i] = EmitArithOperand(strides[i], "index");
  } else {
    // Dense fallback follows physical layout, not logical valid_shape. Explicit
    // view strides (including dynamically captured metadata) take precedence.
    // DN transposes only the trailing matrix axes; batch axes stay outermost.
    const auto layout = column_major ? ir::TensorLayout::DN : ir::TensorLayout::ND;
    strides = ir::tensor_view_semantics::BuildLogicalStridesFromLayout(shape, layout);
    for (size_t i = 0; i < shape.size(); ++i) {
      physical_strides[i] = EmitArithOperand(strides[i], "index");
    }
  }
  auto offset = origin ? EmitArithOperand(origin, "index") : GetOrEmitConstant(int64_t{0}, DataType::INDEX);
  for (size_t i = 0; i < shape.size(); ++i) {
    const auto index = EmitArithOperand(indices->elements_[i], "index");
    const auto term = EmitArithBinaryOp("arith.muli", index, physical_strides[i], "index");
    offset = EmitArithBinaryOp("arith.addi", offset, term, "index");
  }
  return offset;
}

bool PTOCodegen::TryEmitSimtCall(const ir::CallPtr& call) {
  const auto* entry = ir::LookupOpEntry(call->op_);
  if (!entry || entry->GetOpCategory() != "SimtOp") return false;
  INTERNAL_CHECK_SPAN(inside_simt_function_, call->span_) << "Internal error: SIMT op outside helper";
  const bool store = ir::IsOp(call, "simt.store");
  const bool atomic =
      entry->HasDeclaredArgEffect(0) && entry->GetArgEffect(0, call->kwargs_) == ir::ArgEffect::ReadWrite;
  if (store || atomic || ir::IsOp(call, "simt.load")) {
    const size_t arity = ir::IsOp(call, "simt.atomic_cas") ? 4 : (store || atomic ? 3 : 2);
    INTERNAL_CHECK_SPAN(call->args_.size() == arity, call->span_) << "Internal error: SIMT memory arity";
    const auto& storage = call->args_[0];
    const auto indices = ir::As<ir::MakeTuple>(call->args_[1]);
    INTERNAL_CHECK_SPAN(indices, call->span_) << "Internal error: SIMT indices must be a tuple";
    const auto offset = SimtElementOffset(storage, indices);
    const auto pointer = GetExprAsCode(storage);
    const auto type = SimtParameterType(storage->GetType());
    const auto shaped = ir::As<ir::ShapedType>(storage->GetType());
    const auto element = GetTypeString(shaped->dtype_);
    const bool gm = ir::AsTensorTypeLike(storage->GetType()) != nullptr;
    std::string policy;
    for (const auto* key : {"l1cache", "l2cache"}) {
      if (auto value = call->GetKwarg<std::string>(key); !value.empty()) {
        policy += " " + std::string(key) + "(" + value + ")";
      }
    }
    if (atomic) {
      // addptr uses element offsets, including the affine view's physical stride.
      const auto address = NewTemp();
      const auto ptr_body = "<" + element + (gm ? ", gm>" : ", ub>");
      Emit(address + " = pto.addptr " + pointer + ", " + offset + " : " + ptr_body + " -> " + ptr_body);
      auto operands = address;
      for (size_t i = 2; i < call->args_.size(); ++i) operands += ", " + GetExprAsCode(call->args_[i]);
      if ((ir::IsOp(call, "simt.atomic_min") || ir::IsOp(call, "simt.atomic_max")) &&
          shaped->dtype_ == DataType::INT32) {
        policy += " signed";
      }
      const auto result = NewTemp();
      Emit(result + " = pto." + call->op_->name_.substr(5) + " " + operands + policy + " : " + type + ", " +
           element + " -> " + element);
      fs_.current_expr_value = result;
    } else if (store) {
      const auto value = GetExprAsCode(call->args_[2]);
      Emit(std::string(gm ? "pto.stg " : "pto.store ") + value + ", " + pointer + "[" + offset + "]" +
           policy + " : " + type + ", " + element);
      fs_.current_expr_value.clear();
    } else {
      const auto result = NewTemp();
      Emit(result + (gm ? " = pto.ldg " : " = pto.load ") + pointer + "[" + offset + "]" + policy + " : " +
           type + " -> " + element);
      fs_.current_expr_value = result;
    }
    return true;
  }
  if (ir::IsOp(call, "simt.syncthreads") || ir::IsOp(call, "simt.threadfence") ||
      ir::IsOp(call, "simt.threadfence_block")) {
    Emit("pto." + call->op_->name_.substr(5));
    fs_.current_expr_value.clear();
    return true;
  }
  // Keep the collective at its original control-flow location: its implicit
  // active-lane mask changes across branches and loop iterations.
  const bool vote = ir::IsOp(call, "simt.vote_all") || ir::IsOp(call, "simt.vote_any") ||
                    ir::IsOp(call, "simt.vote_uni") || ir::IsOp(call, "simt.vote_ballot");
  const bool shuffle = ir::IsOp(call, "simt.shuffle_idx") || ir::IsOp(call, "simt.shuffle_up") ||
                       ir::IsOp(call, "simt.shuffle_down") || ir::IsOp(call, "simt.shuffle_bfly");
  const bool redux = ir::IsOp(call, "simt.redux_add") || ir::IsOp(call, "simt.redux_min") ||
                     ir::IsOp(call, "simt.redux_max");
  if (vote || shuffle || redux) {
    INTERNAL_CHECK_SPAN(call->args_.size() == (shuffle ? 2 : 1), call->span_)
        << "Internal error: SIMT collective arity";
    const auto input = GetExprAsCode(call->args_[0]);
    const auto input_type = GetExprTypeAnnotation(call->args_[0]);
    const auto output_type = GetExprTypeAnnotation(call);
    auto operation = "pto." + call->op_->name_.substr(5);
    auto operands = input;
    std::string attributes;
    auto types = input_type;
    if (shuffle) {
      operands += ", " + GetExprAsCode(call->args_[1]);
      types += ", i32";
      attributes = " {width = " + std::to_string(call->GetKwarg<int>("width", 32)) + " : i32}";
    } else if (redux) {
      const bool integer = ir::As<ir::ScalarType>(call->GetType())->dtype_ == DataType::INT32;
      operation += integer ? "i" : "f";
      if (integer && !ir::IsOp(call, "simt.redux_add")) attributes = " signed";
    }
    const auto result = NewTemp();
    Emit(result + " = " + operation + " " + operands + attributes + " : " + types + " -> " + output_type);
    fs_.current_expr_value = result;
    return true;
  }
  std::string operation;
  if (ir::IsOp(call, "simt.lane_id")) {
    operation = "pto.get_laneid";
  } else {
    const auto axis = call->GetKwarg<ir::SimtAxis>("axis", ir::SimtAxis::X);
    const std::string suffix = axis == ir::SimtAxis::X ? "x" : axis == ir::SimtAxis::Y ? "y" : "z";
    if (ir::IsOp(call, "simt.thread_idx")) {
      operation = "pto.get_tid_";
    } else if (ir::IsOp(call, "simt.block_idx")) {
      operation = "pto.get_block_idx_";
    } else if (ir::IsOp(call, "simt.block_dim")) {
      operation = "pto.get_block_dim_";
    } else if (ir::IsOp(call, "simt.grid_dim")) {
      operation = "pto.get_grid_dim_";
    }
    INTERNAL_CHECK_SPAN(!operation.empty(), call->span_) << "Internal error: unhandled SIMT query";
    operation += suffix;
  }
  const auto result = NewTemp();
  Emit(result + " = " + operation + " : i32");
  fs_.current_expr_value = result;
  return true;
}
}  // namespace pypto::codegen
