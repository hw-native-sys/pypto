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
#include <functional>
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
#include "src/codegen/pto/pto_scalar_floor_div_mod_plan.h"

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

// Scalar storage and arithmetic use the same unpacked carrier subset. Operator
// schemas impose their narrower restrictions (for example no BF16 shuffle).
void CheckScalar(DataType dtype, const Span& span) {
  CHECK_SPAN(dtype == DataType::FP32 || dtype == DataType::FP16 || dtype == DataType::BF16 ||
                 (dtype.IsInt() && dtype.GetBit() >= 8 && dtype.GetBit() <= 64) || dtype == DataType::INDEX ||
                 dtype == DataType::BOOL,
             span)
      << "SIMT scalar lowering supports FP16/BF16/FP32, 8/16/32/64-bit integers, INDEX and BOOL; got "
      << dtype.ToString();
}

// Bisheng's A5 SIMT backend cannot select LLVM frem (the lowering of pto.remf).
// Compute fmod using binary long division of the FP32 significands instead.
// Unlike x - trunc(x / y) * y, this is exact for large ratios and subnormals,
// needs no floating-point arithmetic/FTZ assumptions, and never overflows i32.
class SimtRemainderEmitter {
 public:
  SimtRemainderEmitter(std::function<void(const std::string&, bool)> emit,
                       std::function<std::string()> new_temp)
      : emit_(std::move(emit)), new_temp_(std::move(new_temp)) {}

  std::string Generate(const std::string& lhs, const std::string& rhs) {
    zero_ = Constant(0);
    one_ = Constant(1);
    hidden_ = Constant(0x800000);
    const auto fraction_mask = Constant(0x7fffff);
    const auto magnitude_mask = Constant(0x7fffffff);
    const auto sign_mask = Constant(-2147483647 - 1);
    const auto infinity = Constant(0x7f800000);
    const auto nan = Constant(0x7fc00000);
    const auto x = Value("arith.bitcast " + lhs + " : f32 to i32");
    const auto y = Value("arith.bitcast " + rhs + " : f32 to i32");
    const auto ax = Binary("andi", x, magnitude_mask);
    const auto ay = Binary("andi", y, magnitude_mask);
    const auto sign = Binary("andi", x, sign_mask);
    const auto invalid_x = Compare("uge", ax, infinity);
    const auto invalid_y = Binary("ori", Compare("ugt", ay, infinity), Compare("eq", ay, zero_), "i1");
    const auto invalid = Binary("ori", invalid_x, invalid_y, "i1");
    const auto result = new_temp_();
    Open(result + " = scf.if " + invalid + " -> (i32) {");
    Emit("scf.yield " + nan + " : i32");
    Else();
    // This also handles finite x / infinity and preserves the sign of zero.
    const auto small = Compare("ult", ax, ay);
    const auto finite = new_temp_();
    Open(finite + " = scf.if " + small + " -> (i32) {");
    Emit("scf.yield " + x + " : i32");
    Else();
    const auto equal = Compare("eq", ax, ay);
    const auto reduced = new_temp_();
    Open(reduced + " = scf.if " + equal + " -> (i32) {");
    Emit("scf.yield " + sign + " : i32");
    Else();
    // Here both operands are finite/nonzero and |x| > |y|. Normalize both
    // significands into [2^23, 2^24), using signed biased exponents for subnormals.
    const auto shift = Constant(23);
    auto unpack = [&](const std::string& magnitude) {
      const auto exponent = Binary("shrui", magnitude, shift);
      const auto subnormal = Compare("eq", exponent, zero_);
      const auto fraction = Binary("andi", magnitude, fraction_mask);
      const auto mantissa = Select(subnormal, fraction, Binary("ori", fraction, hidden_));
      return Normalize(mantissa, Select(subnormal, one_, exponent));
    };
    const auto [mx, ex] = unpack(ax);
    const auto [my, ey] = unpack(ay);
    const auto loop = new_temp_();
    const auto m = new_temp_();
    const auto e = new_temp_();
    Open(loop + ":2 = scf.while (" + m + " = " + mx + ", " + e + " = " + ex +
         ") : (i32, i32) -> (i32, i32) {");
    const auto more = Compare("sgt", e, ey);
    Emit("scf.condition(" + more + ") " + m + ", " + e + " : i32, i32");
    Close("} do {");
    ++indent_;
    const auto next_m = new_temp_();
    const auto next_e = new_temp_();
    Emit("^bb0(" + next_m + ": i32, " + next_e + ": i32):", true);
    // Each iteration starts below 2*my; one subtraction then a shift suffices.
    const auto remainder = SubtractIfGreater(next_m, my);
    const auto shifted = Binary("shli", remainder, one_);
    const auto decremented = Binary("subi", next_e, one_);
    Emit("scf.yield " + shifted + ", " + decremented + " : i32, i32");
    Close("}");
    const auto residual = SubtractIfGreater(loop + "#0", my);
    const auto [mantissa, exponent] = Normalize(residual, ey);
    const auto packed = new_temp_();
    Open(packed + " = scf.if " + Compare("sgt", exponent, zero_) + " -> (i32) {");
    const auto normal = Binary("ori", Binary("subi", mantissa, hidden_), Binary("shli", exponent, shift));
    Emit("scf.yield " + normal + " : i32");
    Else();
    // A nonzero residual is a representable FP32 value; this shift is at most
    // 23 and discards only zero bits. Normalize leaves zero's exponent unchanged.
    const auto subnormal = Binary("shrui", mantissa, Binary("subi", one_, exponent));
    Emit("scf.yield " + subnormal + " : i32");
    Close("}");
    const auto magnitude = Select(Compare("eq", mantissa, zero_), zero_, packed);
    const auto signed_result = Binary("ori", magnitude, sign);
    Emit("scf.yield " + signed_result + " : i32");
    Close("}");
    Emit("scf.yield " + reduced + " : i32");
    Close("}");
    Emit("scf.yield " + finite + " : i32");
    Close("}");
    return Value("arith.bitcast " + result + " : i32 to f32");
  }

 private:
  void Emit(const std::string& text, bool structural = false) {
    emit_(std::string(indent_ * 2, ' ') + text, structural);
  }
  void Open(const std::string& text) {
    Emit(text, true);
    ++indent_;
  }
  void Close(const std::string& text) {
    --indent_;
    Emit(text, true);
  }
  void Else() {
    Close("} else {");
    ++indent_;
  }
  std::string Value(const std::string& operation) {
    const auto result = new_temp_();
    Emit(result + " = " + operation);
    return result;
  }
  std::string Constant(int32_t value) { return Value("arith.constant " + std::to_string(value) + " : i32"); }
  std::string Binary(const std::string& op, const std::string& lhs, const std::string& rhs,
                     const std::string& type = "i32") {
    return Value("arith." + op + " " + lhs + ", " + rhs + " : " + type);
  }
  std::string Compare(const std::string& predicate, const std::string& lhs, const std::string& rhs) {
    return Value("arith.cmpi " + predicate + ", " + lhs + ", " + rhs + " : i32");
  }
  std::string Select(const std::string& condition, const std::string& yes, const std::string& no) {
    return Value("arith.select " + condition + ", " + yes + ", " + no + " : i32");
  }
  std::string SubtractIfGreater(const std::string& lhs, const std::string& rhs) {
    return Select(Compare("uge", lhs, rhs), Binary("subi", lhs, rhs), lhs);
  }
  std::pair<std::string, std::string> Normalize(const std::string& mantissa, const std::string& exponent) {
    const auto result = new_temp_();
    const auto m = new_temp_();
    const auto e = new_temp_();
    Open(result + ":2 = scf.while (" + m + " = " + mantissa + ", " + e + " = " + exponent +
         ") : (i32, i32) -> (i32, i32) {");
    const auto nonzero = Compare("ne", m, zero_);
    const auto small = Compare("ult", m, hidden_);
    const auto more = Binary("andi", nonzero, small, "i1");
    Emit("scf.condition(" + more + ") " + m + ", " + e + " : i32, i32");
    Close("} do {");
    ++indent_;
    const auto next_m = new_temp_();
    const auto next_e = new_temp_();
    Emit("^bb0(" + next_m + ": i32, " + next_e + ": i32):", true);
    const auto shifted = Binary("shli", next_m, one_);
    const auto decremented = Binary("subi", next_e, one_);
    Emit("scf.yield " + shifted + ", " + decremented + " : i32, i32");
    Close("}");
    return {result + "#0", result + "#1"};
  }

  std::function<void(const std::string&, bool)> emit_;
  std::function<std::string()> new_temp_;
  size_t indent_ = 0;
  std::string zero_;
  std::string one_;
  std::string hidden_;
};

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
  CheckScalar(dtype, ir::Span::unknown());
  CHECK(dtype != DataType::INDEX && dtype != DataType::BOOL)
      << "SIMT storage requires a byte-addressable scalar dtype";
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
  body.fs_.truncating_floor_div_mod = PrepareScalarFloorDivMod(function->body_, function->params_);
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
    artifact.capture_base_indices.push_back(ir::As<ir::ScalarType>(arg->GetType()) ? -1
                                                                                   : static_cast<int64_t>(i));
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
  // Public unsigned values retain uiN in the helper ABI. Native SIMT arithmetic
  // ops require signless iN, with signedness selected separately. These bridges
  // preserve bits and never perform a numerical cast.
  auto native_type = [this](const ir::ExprPtr& value) {
    auto type = GetExprTypeAnnotation(value);
    return ir::GetScalarDtype(value).IsUnsignedInt() ? type.substr(1) : type;
  };
  auto native_operand = [this, &native_type](const ir::ExprPtr& value) {
    auto source = GetExprAsCode(value);
    if (!ir::GetScalarDtype(value).IsUnsignedInt()) return source;
    const auto result = NewTemp();
    Emit(result + " = builtin.unrealized_conversion_cast " + source + " : " + GetExprTypeAnnotation(value) +
         " to " + native_type(value));
    return result;
  };
  auto bind_result = [this, &call, &native_type](const std::string& value) {
    fs_.current_expr_value = value;
    if (!ir::GetScalarDtype(call).IsUnsignedInt()) return;
    const auto result = NewTemp();
    Emit(result + " = builtin.unrealized_conversion_cast " + value + " : " + native_type(call) + " to " +
         GetExprTypeAnnotation(call));
    fs_.current_expr_value = result;
  };
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
      Emit(address + " = pto.addptr " + pointer + ", " + offset + " : " + ptr_body + ", index -> " +
           ptr_body);
      auto operands = address;
      auto atomic_type = type;
      const auto atomic_element = native_type(call);
      if (shaped->dtype_.IsUnsignedInt()) {
        atomic_type = "!pto.ptr<" + atomic_element + (gm ? ", gm>" : ", ub>");
        operands = NewTemp();
        Emit(operands + " = pto.castptr " + address + " : " + type + " -> " + atomic_type);
      }
      for (size_t i = 2; i < call->args_.size(); ++i) operands += ", " + native_operand(call->args_[i]);
      if ((ir::IsOp(call, "simt.atomic_min") || ir::IsOp(call, "simt.atomic_max")) &&
          shaped->dtype_.IsInt()) {
        policy += call->GetKwarg<bool>("unsigned", false) || shaped->dtype_.IsUnsignedInt() ? " unsigned"
                                                                                            : " signed";
      }
      const auto result = NewTemp();
      Emit(result + " = pto." + call->op_->name_.substr(5) + " " + operands + policy + " : " + atomic_type +
           ", " + atomic_element + " -> " + atomic_element);
      bind_result(result);
    } else if (store) {
      const auto value = GetExprAsCode(call->args_[2]);
      Emit(std::string(gm ? "pto.stg " : "pto.store ") + value + ", " + pointer + "[" + offset + "]" +
           policy + " : " + type + ", index, " + element);
      fs_.current_expr_value.clear();
    } else {
      const auto result = NewTemp();
      Emit(result + (gm ? " = pto.ldg " : " = pto.load ") + pointer + "[" + offset + "]" + policy + " : " +
           type + ", index -> " + element);
      fs_.current_expr_value = result;
    }
    return true;
  }
  if (ir::IsOp(call, "simt.syncthreads") || ir::IsOp(call, "simt.threadfence") ||
      ir::IsOp(call, "simt.threadfence_block") || ir::IsOp(call, "simt.trap")) {
    Emit("pto." + call->op_->name_.substr(5));
    fs_.current_expr_value.clear();
    return true;
  }
  if (ir::IsOp(call, "simt.assert_")) {
    const auto predicate = GetExprAsCode(call->args_[0]);
    Emit("pto.assert " + predicate + " : i1");
    fs_.current_expr_value.clear();
    return true;
  }
  if (ir::IsOp(call, "simt.rem")) {
    const auto lhs = GetExprAsCode(call->args_[0]);
    const auto rhs = GetExprAsCode(call->args_[1]);
    SimtRemainderEmitter emitter(
        [this](const std::string& text, bool structural) {
          // Region delimiters and block headers cannot carry operation locations.
          if (structural) {
            EmitStructural(text);
          } else {
            Emit(text);
          }
        },
        [this]() { return NewTemp(); });
    fs_.current_expr_value = emitter.Generate(lhs, rhs);
    return true;
  }
  // Explicit scalar math remains in the SIMT execution domain. Validate names
  // through the registry just like the existing collective dispatch.
  struct MathOp {
    const char* name;
    const char* native;
    bool same_type_syntax;
    size_t arity;
  };
  static const MathOp math_ops[] = {
      {"simt.abs", "absf", true, 1},         {"simt.min", "minf", true, 2},
      {"simt.max", "maxf", true, 2},         {"simt.exp", "exp", false, 1},
      {"simt.log", "log", false, 1},         {"simt.sqrt", "sqrt", false, 1},
      {"simt.sin", "sin", false, 1},         {"simt.cos", "cos", false, 1},
      {"simt.ceil", "ceil", false, 1},       {"simt.floor", "floor", false, 1},
      {"simt.rint", "rint", false, 1},       {"simt.round", "round", false, 1},
      {"simt.pow", "pow", false, 2},         {"simt.fma", "fma", false, 3},
      {"simt.prmt", "prmt", false, 3},       {"simt.mulhi", "mulhi", false, 2},
      {"simt.ftoi", "ftoi", false, 1},       {"simt.itof", "itof", false, 1},
      {"simt.bitcast", "bitcast", false, 1}, {"simt.ftof", "ftof", false, 1},
      {"simt.div", "divf", true, 2},         {"simt.mul_i32toi64", "mul_i32toi64", false, 2},
  };
  for (const auto& spec : math_ops) {
    if (!ir::IsOp(call, spec.name)) continue;
    INTERNAL_CHECK_SPAN(call->args_.size() == spec.arity, call->span_)
        << "Internal error: SIMT scalar math arity for " << spec.name;
    std::vector<std::string> operands;
    std::vector<std::string> types;
    for (const auto& arg : call->args_) {
      operands.push_back(native_operand(arg));
      types.push_back(native_type(arg));
    }
    std::string attributes;
    const bool ftoi = ir::IsOp(call, "simt.ftoi");
    const bool itof = ir::IsOp(call, "simt.itof");
    const bool ftof = ir::IsOp(call, "simt.ftof");
    if (ftoi || itof || ir::IsOp(call, "simt.mulhi") || ir::IsOp(call, "simt.mul_i32toi64")) {
      const auto integer_type = ir::GetScalarDtype(ftoi ? ir::ExprPtr(call) : call->args_[0]);
      attributes =
          call->GetKwarg<bool>("unsigned", false) || integer_type.IsUnsignedInt() ? " unsigned" : " signed";
    }
    if (ir::IsOp(call, "simt.div")) {
      attributes = " {precisionType = #pto<div_precision " +
                   call->GetKwarg<std::string>("precision", "intrinsic") + ">}";
    }
    if (ftoi || itof || ftof) {
      attributes += " round(" + call->GetKwarg<std::string>("rounding", ftoi ? "z" : "r") + ")";
      // PTOAS requires saturation for FP32 -> integer; ordinary pl.cast keeps
      // its existing contract and does not use this explicit conversion API.
      if (ftoi || (ftof && call->GetKwarg<bool>("saturate", false))) attributes += " sat";
    }
    const auto result = NewTemp();
    const auto signature = spec.same_type_syntax ? types.front() : Join(types) + " -> " + native_type(call);
    Emit(result + " = pto." + spec.native + " " + Join(operands) + attributes + " : " + signature);
    bind_result(result);
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
    const auto input = native_operand(call->args_[0]);
    const auto input_type = native_type(call->args_[0]);
    const auto output_type = native_type(call);
    auto operation = "pto." + call->op_->name_.substr(5);
    auto operands = input;
    std::string attributes;
    auto types = input_type;
    if (shuffle) {
      operands += ", " + GetExprAsCode(call->args_[1]);
      types += ", i32";
      const auto width = call->GetKwarg<int>("width", 32);
      attributes = " {width = " + std::to_string(width) + " : i32";
      // Omission preserves PTOAS's widest-window default (zero for shuffle_up).
      if (call->HasKwarg("bound")) {
        attributes += ", bound = " + std::to_string(call->GetKwarg<int>("bound")) + " : i32";
      }
      attributes += "}";
    } else if (redux) {
      const auto dtype = ir::GetScalarDtype(call);
      const bool integer = dtype.IsInt();
      operation += integer ? "i" : "f";
      if (integer && !ir::IsOp(call, "simt.redux_add")) {
        attributes =
            call->GetKwarg<bool>("unsigned", false) || dtype.IsUnsignedInt() ? " unsigned" : " signed";
      }
    }
    const auto result = NewTemp();
    Emit(result + " = " + operation + " " + operands + attributes + " : " + types + " -> " + output_type);
    bind_result(result);
    return true;
  }
  std::string operation;
  if (ir::IsOp(call, "simt.lane_id")) {
    operation = "pto.get_laneid";
  } else if (ir::IsOp(call, "simt.veccore_id")) {
    operation = "pto.get_veccoreid";
  } else if (ir::IsOp(call, "simt.lanemask_eq") || ir::IsOp(call, "simt.lanemask_le") ||
             ir::IsOp(call, "simt.lanemask_lt") || ir::IsOp(call, "simt.lanemask_ge") ||
             ir::IsOp(call, "simt.lanemask_gt")) {
    operation = "pto.get_" + call->op_->name_.substr(5);
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
