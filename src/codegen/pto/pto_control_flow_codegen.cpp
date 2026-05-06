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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::EvalStmtPtr;
using ir::ForStmtPtr;
using ir::IfStmtPtr;
using ir::ScalarType;
using ir::StmtPtr;
using ir::TensorType;
using ir::TileType;
using ir::WhileStmtPtr;
using ir::YieldStmtPtr;

/// Join a vector of strings with ", " separator
static std::string JoinCommaSep(const std::vector<std::string>& items) {
  std::ostringstream oss;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << items[i];
  }
  return oss.str();
}

/// Join pairs of strings as "a sep b" with ", " between pairs
static std::string JoinPairs(const std::vector<std::string>& lhs, const std::string& sep,
                             const std::vector<std::string>& rhs) {
  INTERNAL_CHECK(lhs.size() == rhs.size()) << "Internal error: JoinPairs size mismatch";
  std::ostringstream oss;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << lhs[i] << sep << rhs[i];
  }
  return oss.str();
}

// ========================================================================
// Statement visitors - Control flow
// ========================================================================

void PTOCodegen::VisitStmt_(const EvalStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op != nullptr, op->span_) << "Internal error: null EvalStmt";
  INTERNAL_CHECK_SPAN(op->expr_ != nullptr, op->span_) << "Internal error: EvalStmt has null expression";
  VisitExpr(op->expr_);
}

void PTOCodegen::VisitStmt_(const YieldStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op != nullptr, op->span_) << "Internal error: null YieldStmt";

  if (op->value_.empty()) {
    return;
  }

  std::vector<std::string> yielded_values;
  for (const auto& expr : op->value_) {
    VisitExpr(expr);
    yielded_values.push_back(fs_.current_expr_value);
    fs_.current_expr_value = "";
  }
  fs_.yield_buffer = yielded_values;
}

std::string PTOCodegen::GetScalarIterArgTypeString(
    const std::shared_ptr<const ScalarType>& scalar_type) const {
  CHECK(scalar_type) << "PTOCodegen requires a valid ScalarType for iter_arg/result emission";
  return GetTypeString(scalar_type->dtype_);
}

void PTOCodegen::VisitStmt_(const IfStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op != nullptr, op->span_) << "Internal error: null IfStmt";
  INTERNAL_CHECK_SPAN(op->condition_ != nullptr, op->span_) << "Internal error: IfStmt has null condition";
  INTERNAL_CHECK_SPAN(op->then_body_ != nullptr, op->span_) << "Internal error: IfStmt has null then_body";

  // Evaluate condition
  VisitExpr(op->condition_);
  std::string condition = fs_.current_expr_value;
  fs_.current_expr_value = "";

  if (op->return_vars_.empty()) {
    // Simple scf.if (no return values)
    Emit("scf.if " + condition + " {");
    indent_level_++;
    VisitStmt(op->then_body_);
    indent_level_--;

    const auto& else_body = op->else_body_;
    if (else_body) {
      Emit("} else {");
      indent_level_++;
      VisitStmt(*else_body);
      indent_level_--;
    }
    Emit("}");
  } else {
    // Like loops, keep tile return values out of scf.if results. Pre-declare
    // tile buffers for return_vars using the canonical MemRef address (assigned
    // by MemoryReuse), and only use scf.if results for scalar-like SSA values.
    // MemoryReuse's YieldFixupMutator ensures all branch yields already share
    // the return_var's canonical MemRef, so no codegen-level tmov is needed.
    std::vector<bool> returns_via_scf(op->return_vars_.size(), false);
    std::vector<std::string> scf_return_names;
    std::vector<std::string> scf_return_types;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& return_var = op->return_vars_[i];
      if (auto scalar_type = As<ScalarType>(return_var->GetType())) {
        std::string ret_name = NewNamedTemp(return_var->name_hint_);
        BindVarToMlir(return_var, ret_name);
        scf_return_names.push_back(ret_name);
        scf_return_types.push_back(GetScalarIterArgTypeString(scalar_type));
        returns_via_scf[i] = true;
      } else if (auto tensor_type = As<TensorType>(return_var->GetType())) {
        std::string ret_name = NewNamedTemp(return_var->name_hint_);
        BindVarToMlir(return_var, ret_name);
        BindTensorView(return_var, ret_name);
        scf_return_names.push_back(ret_name);
        scf_return_types.push_back(GetTensorViewTypeString(tensor_type.get()));
        returns_via_scf[i] = true;
      } else if (auto tile_type = As<TileType>(return_var->GetType())) {
        INTERNAL_CHECK_SPAN(tile_type->memref_.has_value(), op->span_)
            << "TileType return_var must have a MemRef at codegen stage for var: " << return_var->name_hint_;
        // Reuse the same alloc_tile rules as EmitAllocTileForVar so this
        // deferred alloc emits a dynamic-validShape `pto.alloc_tile` with
        // explicit valid_row / valid_col operands.
        AllocTileFields fields = ComputeAllocTileFields(tile_type);
        std::string ret_name = AllocNewTileBuf(fields.type_str, return_var->name_hint_, fields.addr_ssa,
                                               fields.valid_row_ssa, fields.valid_col_ssa);
        BindVarToMlir(return_var, ret_name);
      } else {
        INTERNAL_CHECK_SPAN(false, op->span_)
            << "Internal error: unsupported IfStmt return_var type for " << return_var->name_hint_;
      }
    }

    CHECK(op->else_body_.has_value()) << "IfStmt with return_vars requires else_body";

    if (!scf_return_names.empty()) {
      Emit(JoinCommaSep(scf_return_names) + " = scf.if " + condition + " -> (" +
           JoinCommaSep(scf_return_types) + ") {");
    } else {
      Emit("scf.if " + condition + " {");
    }
    indent_level_++;

    auto emit_branch = [&](const StmtPtr& body, const char* branch_name) {
      fs_.yield_buffer.clear();
      VisitStmt(body);
      auto branch_yields = fs_.yield_buffer;
      CHECK(branch_yields.size() == op->return_vars_.size())
          << "IfStmt " << branch_name << "-branch yield count (" << branch_yields.size()
          << ") must match return_vars (" << op->return_vars_.size() << ")";

      std::vector<std::string> scalar_yields;
      scalar_yields.reserve(scf_return_types.size());
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        if (returns_via_scf[i]) {
          scalar_yields.push_back(branch_yields[i]);
          continue;
        }
        // Tile return_vars: MemoryReuse ensures branch yields share the return_var's
        // canonical MemRef (same physical address). No codegen-level tmov needed —
        // the IR-level tile.move (from MemoryReuse's YieldFixupMutator) handles the copy.
      }

      if (!scf_return_types.empty()) {
        Emit("scf.yield " + JoinCommaSep(scalar_yields) + " : " + JoinCommaSep(scf_return_types));
      }
      CHECK(scalar_yields.size() == scf_return_types.size())
          << "IfStmt " << branch_name << "-branch scalar yield count (" << scalar_yields.size()
          << ") must match scalar return_vars (" << scf_return_types.size() << ")";
      fs_.yield_buffer.clear();
    };

    emit_branch(op->then_body_, "then");
    indent_level_--;

    Emit("} else {");
    indent_level_++;
    const auto& else_body = op->else_body_;
    INTERNAL_CHECK_SPAN(else_body.has_value(), op->span_)
        << "Internal error: IfStmt with return_vars has no else_body";
    emit_branch(*else_body, "else");
    indent_level_--;
    Emit("}");
  }
}

void PTOCodegen::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op != nullptr, op->span_) << "Internal error: null ForStmt";
  INTERNAL_CHECK_SPAN(op->loop_var_ != nullptr, op->span_) << "Internal error: ForStmt has null loop_var";
  INTERNAL_CHECK_SPAN(op->body_ != nullptr, op->span_) << "Internal error: ForStmt has null body";

  CHECK(op->iter_args_.size() == op->return_vars_.size())
      << "ForStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  if (op->kind_ == ir::ForKind::Unroll) {
    LOG_WARN << "ForKind::Unroll loop was not expanded before codegen; "
                "generating sequential loop as fallback";
  } else if (op->kind_ == ir::ForKind::Pipeline) {
    LOG_WARN << "ForKind::Pipeline loop reached codegen; CanonicalizeIOOrder "
                "should have demoted it to Sequential. Generating sequential loop as fallback.";
  }

  // Evaluate loop bounds
  VisitExpr(op->start_);
  std::string start = fs_.current_expr_value;
  fs_.current_expr_value = "";

  VisitExpr(op->stop_);
  std::string stop = fs_.current_expr_value;
  fs_.current_expr_value = "";

  VisitExpr(op->step_);
  std::string step = fs_.current_expr_value;
  fs_.current_expr_value = "";

  // Register loop variable
  std::string loop_var_name = NewNamedTemp(op->loop_var_->name_hint_);
  BindVarToMlir(op->loop_var_, loop_var_name);

  // In PTO, only scalar types (index, f32, bool, etc.) need iter_args/yield
  // for loop-carried value semantics. Non-scalar types (TileType, TensorType)
  // are mutable references written in-place via outs(), so they are mapped
  // directly to their init values and excluded from iter_args/yield.
  std::vector<bool> is_scalar(op->iter_args_.size(), false);
  bool has_scalar_iter_args = false;
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (As<ScalarType>(op->iter_args_[i]->GetType())) {
      is_scalar[i] = true;
      has_scalar_iter_args = true;
    }
  }

  // Map non-scalar iter_args/return_vars directly to their init values
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (is_scalar[i]) continue;

    const auto& iter_arg = op->iter_args_[i];
    const auto& return_var = op->return_vars_[i];

    std::string init_mlir_name;
    auto tensor_type = As<TensorType>(iter_arg->GetType());
    if (tensor_type) {
      auto init_var = std::dynamic_pointer_cast<const ir::Var>(iter_arg->initValue_);
      INTERNAL_CHECK_SPAN(init_var, op->span_) << "TensorType iter_arg init value must be a Var or IterArg";
      init_mlir_name = GetOrCreateTensorView(init_var);
    } else {
      VisitExpr(iter_arg->initValue_);
      init_mlir_name = fs_.current_expr_value;
      fs_.current_expr_value = "";
    }

    BindVarToMlir(iter_arg, init_mlir_name);
    BindVarToMlir(return_var, init_mlir_name);

    if (tensor_type) {
      BindTensorView(iter_arg, init_mlir_name);
      BindTensorView(return_var, init_mlir_name);
    } else if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      const auto memref = ir::GetDefinedMemRef(tile_type);
      BindVarToMemRef(iter_arg, memref->base_.get());
      BindVarToMemRef(return_var, memref->base_.get());
    }
  }

  if (!has_scalar_iter_args) {
    // Simple scf.for (no iter_args, or all iter_args are non-scalar)
    Emit("scf.for " + loop_var_name + " = " + start + " to " + stop + " step " + step + " {");
    indent_level_++;

    fs_.yield_buffer.clear();
    VisitStmt(op->body_);
    fs_.yield_buffer.clear();

    indent_level_--;
    Emit("}");
  } else {
    // scf.for with scalar iter_args only
    std::vector<std::string> init_values;
    std::vector<std::string> iter_arg_names;
    std::vector<std::string> iter_arg_types;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!is_scalar[i]) continue;

      const auto& iter_arg = op->iter_args_[i];

      VisitExpr(iter_arg->initValue_);
      init_values.push_back(fs_.current_expr_value);
      fs_.current_expr_value = "";

      std::string iter_name = NewNamedTemp(iter_arg->name_hint_);
      BindVarToMlir(iter_arg, iter_name);
      iter_arg_names.push_back(iter_name);

      iter_arg_types.push_back(GetScalarIterArgTypeString(As<ScalarType>(iter_arg->GetType())));
    }

    // Register return_vars SSA names (scalar only)
    std::vector<std::string> return_var_names;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!is_scalar[i]) continue;
      std::string ret_name = NewNamedTemp(op->return_vars_[i]->name_hint_);
      BindVarToMlir(op->return_vars_[i], ret_name);
      return_var_names.push_back(ret_name);
    }

    // Emit: %ret0 = scf.for %i = %start to %stop step %step
    //           iter_args(%acc = %init) -> (type) {
    Emit(JoinCommaSep(return_var_names) + " = scf.for " + loop_var_name + " = " + start + " to " + stop +
         " step " + step + " iter_args(" + JoinPairs(iter_arg_names, " = ", init_values) + ") -> (" +
         JoinCommaSep(iter_arg_types) + ") {");
    indent_level_++;

    fs_.yield_buffer.clear();
    VisitStmt(op->body_);

    // Filter yield_buffer to keep only scalar iter_arg entries
    std::vector<std::string> scalar_yields;
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (is_scalar[i] && i < fs_.yield_buffer.size()) {
        scalar_yields.push_back(fs_.yield_buffer[i]);
      }
    }

    // Emit scf.yield from filtered yield values
    if (!scalar_yields.empty()) {
      std::ostringstream yield_oss;
      yield_oss << "scf.yield ";
      for (size_t i = 0; i < scalar_yields.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << scalar_yields[i];
      }
      yield_oss << " : ";
      for (size_t i = 0; i < iter_arg_types.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << iter_arg_types[i];
      }
      Emit(yield_oss.str());
    }
    CHECK(scalar_yields.size() == iter_arg_types.size())
        << "ForStmt scalar yield count (" << scalar_yields.size() << ") must match scalar iter_args ("
        << iter_arg_types.size() << ")";
    fs_.yield_buffer.clear();

    indent_level_--;
    Emit("}");
  }
}

void PTOCodegen::VisitStmt_(const WhileStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op != nullptr, op->span_) << "Internal error: null WhileStmt";
  INTERNAL_CHECK_SPAN(op->condition_ != nullptr, op->span_) << "Internal error: WhileStmt has null condition";
  INTERNAL_CHECK_SPAN(op->body_ != nullptr, op->span_) << "Internal error: WhileStmt has null body";

  CHECK(op->iter_args_.size() == op->return_vars_.size())
      << "WhileStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  // In PTO, only scalar types (index, f32, bool, etc.) need iter_args/yield
  // for loop-carried value semantics. Non-scalar types (TileType, TensorType)
  // are mutable references written in-place via outs(), so they are mapped
  // directly to their init values and excluded from iter_args/yield.
  std::vector<bool> is_scalar(op->iter_args_.size(), false);
  bool has_scalar_iter_args = false;
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (As<ScalarType>(op->iter_args_[i]->GetType())) {
      is_scalar[i] = true;
      has_scalar_iter_args = true;
    }
  }

  // Map non-scalar iter_args/return_vars directly to their init values
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (is_scalar[i]) continue;

    const auto& iter_arg = op->iter_args_[i];
    const auto& return_var = op->return_vars_[i];

    std::string init_mlir_name;
    auto tensor_type = As<TensorType>(iter_arg->GetType());
    if (tensor_type) {
      auto init_var = std::dynamic_pointer_cast<const ir::Var>(iter_arg->initValue_);
      INTERNAL_CHECK_SPAN(init_var, op->span_) << "TensorType iter_arg init value must be a Var or IterArg";
      init_mlir_name = GetOrCreateTensorView(init_var);
    } else {
      VisitExpr(iter_arg->initValue_);
      init_mlir_name = fs_.current_expr_value;
      fs_.current_expr_value = "";
    }

    BindVarToMlir(iter_arg, init_mlir_name);
    BindVarToMlir(return_var, init_mlir_name);

    if (tensor_type) {
      BindTensorView(iter_arg, init_mlir_name);
      BindTensorView(return_var, init_mlir_name);
    } else if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      const auto memref = ir::GetDefinedMemRef(tile_type);
      BindVarToMemRef(iter_arg, memref->base_.get());
      BindVarToMemRef(return_var, memref->base_.get());
    }
  }

  if (!has_scalar_iter_args) {
    // Simple scf.while (no iter_args, or all iter_args are non-scalar)
    Emit("scf.while : () -> () {");
    indent_level_++;

    VisitExpr(op->condition_);
    std::string cond = fs_.current_expr_value;
    fs_.current_expr_value = "";
    Emit("scf.condition(" + cond + ")");

    indent_level_--;
    Emit("} do {");
    indent_level_++;

    fs_.yield_buffer.clear();
    VisitStmt(op->body_);

    Emit("scf.yield");
    fs_.yield_buffer.clear();

    indent_level_--;
    Emit("}");
  } else {
    // scf.while with scalar iter_args only
    std::vector<std::string> init_values;
    std::vector<std::string> before_arg_names;
    std::vector<std::string> after_arg_names;
    std::vector<std::string> iter_arg_types;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!is_scalar[i]) continue;

      const auto& iter_arg = op->iter_args_[i];

      VisitExpr(iter_arg->initValue_);
      init_values.push_back(fs_.current_expr_value);
      fs_.current_expr_value = "";

      before_arg_names.push_back(NewTemp());
      after_arg_names.push_back(NewTemp());

      iter_arg_types.push_back(GetScalarIterArgTypeString(As<ScalarType>(iter_arg->GetType())));
    }

    // Register return_vars SSA names (scalar only)
    std::vector<std::string> return_var_names;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!is_scalar[i]) continue;

      std::string ret_name = NewTemp();
      BindVarToMlir(op->return_vars_[i], ret_name);
      return_var_names.push_back(ret_name);
    }

    // Lambda to register scalar iter_args in fs_.var_to_mlir
    auto register_scalar_iter_args = [&](const std::vector<std::string>& ssa_names) {
      size_t scalar_idx = 0;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        if (!is_scalar[i]) continue;
        BindVarToMlir(op->iter_args_[i], ssa_names[scalar_idx]);
        scalar_idx++;
      }
    };

    std::string types_str = "(" + JoinCommaSep(iter_arg_types) + ")";

    // Emit: %ret0, %ret1 = scf.while (%before0 = %init0, ...) : (types) -> (types) {
    Emit(JoinCommaSep(return_var_names) + " = scf.while (" + JoinPairs(before_arg_names, " = ", init_values) +
         ") : " + types_str + " -> " + types_str + " {");
    indent_level_++;

    // Before region: register before-region args, evaluate condition
    register_scalar_iter_args(before_arg_names);

    VisitExpr(op->condition_);
    std::string cond = fs_.current_expr_value;
    fs_.current_expr_value = "";

    // Emit: scf.condition(%cond) %before0, %before1 : type0, type1
    Emit("scf.condition(" + cond + ") " + JoinCommaSep(before_arg_names) + " : " +
         JoinCommaSep(iter_arg_types));

    indent_level_--;
    Emit("} do {");

    // After region: emit ^bb0 block header with typed arguments
    Emit("^bb0(" + JoinPairs(after_arg_names, " : ", iter_arg_types) + "):");
    indent_level_++;

    // Re-register iter_args with after-region SSA names
    register_scalar_iter_args(after_arg_names);

    // Visit body
    fs_.yield_buffer.clear();
    VisitStmt(op->body_);

    // Filter yield_buffer to keep only scalar iter_arg entries
    std::vector<std::string> scalar_yields;
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (is_scalar[i] && i < fs_.yield_buffer.size()) {
        scalar_yields.push_back(fs_.yield_buffer[i]);
      }
    }

    // Emit scf.yield from filtered yield values
    if (!scalar_yields.empty()) {
      Emit("scf.yield " + JoinCommaSep(scalar_yields) + " : " + JoinCommaSep(iter_arg_types));
    }
    CHECK(scalar_yields.size() == iter_arg_types.size())
        << "WhileStmt scalar yield count (" << scalar_yields.size() << ") must match scalar iter_args ("
        << iter_arg_types.size() << ")";
    fs_.yield_buffer.clear();

    indent_level_--;
    Emit("}");
  }
}

}  // namespace codegen
}  // namespace pypto
