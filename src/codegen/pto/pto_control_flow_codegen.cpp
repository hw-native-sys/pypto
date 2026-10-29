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
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
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
  INTERNAL_CHECK(op != nullptr) << "Internal error: null EvalStmt";
  INTERNAL_CHECK_SPAN(op->expr_ != nullptr, op->span_) << "Internal error: EvalStmt has null expression";
  VisitExpr(op->expr_);
}

void PTOCodegen::VisitStmt_(const YieldStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null YieldStmt";

  if (op->value_.empty()) {
    return;
  }

  std::vector<std::string> yielded_values;
  for (const auto& expr : op->value_) {
    // Tensors are mutable references: a branch may yield either a freshly
    // stored-into tensor (bound to its tensor_view) or the unchanged input (a
    // param bound only to its base ptr). Normalize tensor yields to the
    // canonical tensor_view so consumers — notably the IfStmt in-place
    // return_var binding — see a consistent SSA across branches that aliases
    // the same concrete make_tensor_view (issue #1533). For/While loops keep
    // only scalar yields, so this does not affect their lowering.
    if (ir::AsTensorTypeLike(expr->GetType())) {
      if (auto tensor_var = ir::AsVarLike(expr)) {
        // Only normalize when a tensor_view is actually registered. Some
        // tensors (e.g. return-value / loop phis without a make_tensor_view)
        // have no view at yield time; for those fall through to the default
        // expr lowering rather than hard-failing in GetOrCreateTensorView.
        if (std::string view = TryGetTensorView(tensor_var); !view.empty()) {
          // Cache view → base ptr so an IfStmt rebinding its phi return_var to
          // this shared view can also restore the base ptr (else
          // GetTensorBasePtr would fall back to the view SSA). Both branches
          // yield the same backing, so the recorded base ptr is consistent.
          fs_.view_ssa_to_base_ptr[view] = GetTensorBasePtr(tensor_var);
          if (As<ir::DistributedTensorType>(expr->GetType())) {
            INTERNAL_CHECK_SPAN(!GetCommCtxSSAFor(tensor_var.get()).empty(), op->span_)
                << "Internal error: yielded DistributedTensor '" << tensor_var->name_hint_
                << "' has no CommContext binding";
          }
          yielded_values.push_back(view);
          continue;
        }
      }
    }
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

namespace {
// A ScopeStmt subtype (InCoreScopeStmt, SpmdScopeStmt, ...) has its own
// ObjectKind, so As<ScopeStmt> misses it; match the whole family by base type.
std::shared_ptr<const ir::ScopeStmt> AsScope(const StmtPtr& s) {
  return std::dynamic_pointer_cast<const ir::ScopeStmt>(s);
}

// Find the YieldStmt terminating a branch body (last yield, recursing through the
// trailing SeqStmts and any scope wrappers). Used to recover the per-slot yield
// source for in-place tensor context propagation.
YieldStmtPtr FindBranchYield(const StmtPtr& body) {
  if (!body) return nullptr;
  if (auto y = As<ir::YieldStmt>(body)) return y;
  if (auto seq = As<ir::SeqStmts>(body)) {
    for (auto it = seq->stmts_.rbegin(); it != seq->stmts_.rend(); ++it) {
      if (auto y = FindBranchYield(*it)) return y;
    }
  }
  if (auto scope = AsScope(body)) return FindBranchYield(scope->body_);
  return nullptr;
}

}  // namespace

std::string PTOCodegen::GetValueCarryTypeString(const ir::TypePtr& type) const {
  if (auto reg = As<ir::RegType>(type)) return GetRegTypeString(reg);
  return GetScalarIterArgTypeString(As<ScalarType>(type));
}

void PTOCodegen::VisitStmt_(const IfStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null IfStmt";
  INTERNAL_CHECK_SPAN(op->condition_ != nullptr, op->span_) << "Internal error: IfStmt has null condition";
  INTERNAL_CHECK_SPAN(op->then_body_ != nullptr, op->span_) << "Internal error: IfStmt has null then_body";

  // Evaluate condition
  VisitExpr(op->condition_);
  std::string condition = fs_.current_expr_value;
  fs_.current_expr_value = "";

  if (op->return_vars_.empty()) {
    // Simple scf.if (no return values)
    EmitStructural("scf.if " + condition + " {");
    IncreaseIndent();
    VisitStmt(op->then_body_);
    DecreaseIndent();

    const auto& else_body = op->else_body_;
    if (else_body) {
      EmitStructural("} else {");
      IncreaseIndent();
      VisitStmt(*else_body);
      DecreaseIndent();
    }
    EmitStructural("}");
  } else {
    // Scalars, registers and ordinary Tile selections are SSA results. RegScope
    // Tile aliases retain their prepared external handles. Other Tile results
    // select descriptors without copying branch producers into a phi buffer.
    std::vector<bool> returns_via_scf(op->return_vars_.size(), false);
    std::vector<std::string> scf_return_names;
    std::vector<std::string> scf_return_types;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& return_var = op->return_vars_[i];
      if (As<ScalarType>(return_var->GetType()) || As<ir::RegType>(return_var->GetType())) {
        std::string ret_name = NewNamedTemp(return_var->name_hint_);
        BindVarToMlir(return_var, ret_name);
        scf_return_names.push_back(ret_name);
        scf_return_types.push_back(GetValueCarryTypeString(return_var->GetType()));
        returns_via_scf[i] = true;
      } else if (auto tile_type = As<TileType>(return_var->GetType())) {
        if (inside_reg_scope_) {
          // The RegScope pre-scan proved both arms alias one external view and
          // bound this result to its dominating handle. Keep that alias rather
          // than creating a phi buffer and copying Tiles inside vecscope.
          INTERNAL_CHECK_SPAN(fs_.var_to_mlir.count(GetVarKey(return_var)), op->span_)
              << "Register Tile branch result lacks a prepared external handle";
          continue;
        }
        const std::string ret_name = NewNamedTemp(return_var->name_hint_);
        BindVarToMlir(return_var, ret_name);
        RegisterSelectedTileName(ret_name);
        scf_return_names.push_back(ret_name);
        scf_return_types.emplace_back();
        returns_via_scf[i] = true;
      } else if (ir::AsTensorTypeLike(return_var->GetType()) || As<ir::ArrayType>(return_var->GetType())) {
        // Tensors and on-core arrays are mutable references mutated in place
        // (pl.assemble lowers to a tile store into the backing memref; arrays
        // write the same backing `pto.declare_local_array`). Both branches yield
        // the SAME underlying SSA, so the merged value is NOT an scf.if result.
        // Routing a tensor through scf.if would retype it to a fully-dynamic
        // !pto.tensor_view<?x?> and drop the concrete memref dims that
        // pto.partition_view requires (issue #1533). returns_via_scf stays
        // false; the return var is bound to the shared branch-yield SSA after
        // both branches emit (see below).
      } else {
        INTERNAL_CHECK_SPAN(false, op->span_)
            << "Internal error: unsupported IfStmt return_var type for " << return_var->name_hint_;
      }
    }

    CHECK(op->else_body_.has_value()) << "IfStmt with return_vars requires else_body";

    // For in-place return vars (ArrayType and TensorType, both kept out of
    // scf.if results), capture the backing SSA that the branches yield so it can
    // be bound to the merged return var after both branches emit. Both branches
    // yield the same SSA because every array.update_element / pl.assemble
    // aliases the one backing array / tensor.
    std::vector<std::string> inplace_return_ssa(op->return_vars_.size());
    std::vector<std::string> inplace_return_comm_ctx(op->return_vars_.size());

    struct BranchEmission {
      std::string body;
      std::vector<std::string> yields;
      std::vector<std::string> types;
    };
    // Views can have a different physical SSA descriptor from their logical
    // TileType. Emit both bodies before declaring the result type so equal
    // actual descriptors, including subviews, pass through without adapters.
    auto emit_branch = [&](const StmtPtr& body, const char* branch_name) {
      auto outer_stream = std::move(stream_);
      stream_ = std::ostringstream{};
      IncreaseIndent();
      const YieldStmtPtr yield = FindBranchYield(body);
      fs_.yield_buffer.clear();
      VisitStmt(body);
      auto branch_yields = fs_.yield_buffer;
      CHECK(branch_yields.size() == op->return_vars_.size())
          << "IfStmt " << branch_name << "-branch yield count (" << branch_yields.size()
          << ") must match return_vars (" << op->return_vars_.size() << ")";

      std::vector<std::string> scf_yields;
      scf_yields.reserve(scf_return_types.size());
      std::vector<std::string> yield_types;
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        if (returns_via_scf[i]) {
          const std::string type = As<TileType>(op->return_vars_[i]->GetType())
                                       ? GetSSATileBufType(branch_yields[i])
                                       : scf_return_types[scf_yields.size()];
          INTERNAL_CHECK_SPAN(!type.empty(), op->span_)
              << "Internal error: missing SSA type for IfStmt yield " << branch_yields[i];
          yield_types.push_back(type);
          scf_yields.push_back(branch_yields[i]);
        } else if (As<ir::ArrayType>(op->return_vars_[i]->GetType()) ||
                   ir::AsTensorTypeLike(op->return_vars_[i]->GetType())) {
          if (As<ir::DistributedTensorType>(op->return_vars_[i]->GetType())) {
            // Resolve the context from the yielded Var, as For/While do for their
            // init values. A yield whose tensor has no registered tensor_view falls
            // back to the plain expr lowering above, so keying off the yielded SSA
            // would miss it.
            std::string comm_ctx;
            if (yield && i < yield->value_.size()) {
              if (auto src = ir::AsVarLike(yield->value_[i])) comm_ctx = GetCommCtxSSAFor(src.get());
            }
            INTERNAL_CHECK_SPAN(!comm_ctx.empty(), op->span_)
                << "Internal error: IfStmt " << branch_name << "-branch DistributedTensor return_var '"
                << op->return_vars_[i]->name_hint_ << "' has no CommContext binding";
            if (inplace_return_comm_ctx[i].empty()) {
              inplace_return_comm_ctx[i] = std::move(comm_ctx);
            } else {
              // This tensor result stays outside scf.if and aliases one backing
              // SSA. Choosing either branch's context here could pair the other
              // branch's data pointer with the wrong remote-window table.
              // Supporting that selection requires merging the base pointer and
              // CommContext together, then rebuilding the static tensor view;
              // tracked by GitHub issue #2027.
              CHECK_SPAN(inplace_return_comm_ctx[i] == comm_ctx, op->span_)
                  << "Assigning a different DistributedTensor in each branch of an `if` is not supported: '"
                  << op->return_vars_[i]->name_hint_
                  << "' would take its data pointer from one allocation and its communication context from "
                     "another. Assign a single DistributedTensor before the `if`, and branch on the data "
                     "read from it instead (see GitHub issue #2027).";
            }
          }
          // In-place backing SSA (array or tensor); bound to the return var
          // after the branches. Both branches must agree on the same storage SSA
          // (every array.update_element / pl.assemble aliases the one backing
          // array / tensor) — assert it so a future divergence can't silently
          // bind to the last-emitted branch.
          if (inplace_return_ssa[i].empty()) {
            inplace_return_ssa[i] = branch_yields[i];
          } else {
            INTERNAL_CHECK_SPAN(inplace_return_ssa[i] == branch_yields[i], op->span_)
                << "Internal error: IfStmt in-place return_var '" << op->return_vars_[i]->name_hint_
                << "' yields different backing SSAs across branches: " << inplace_return_ssa[i] << " vs "
                << branch_yields[i];
          }
        }
      }

      INTERNAL_CHECK_SPAN(scf_yields.size() == scf_return_types.size(), op->span_)
          << "Internal error: IfStmt " << branch_name << "-branch SSA yield count differs from results";
      fs_.yield_buffer.clear();
      DecreaseIndent();
      BranchEmission result{stream_.str(), std::move(scf_yields), std::move(yield_types)};
      stream_ = std::move(outer_stream);
      return result;
    };

    auto then_branch = emit_branch(op->then_body_, "then");
    auto else_branch = emit_branch(*op->else_body_, "else");
    // A shared destination already dominates the If. Returning that exact SSA
    // from both arms adds no selection, and would unnecessarily hide its local
    // allocation provenance from PTOAS (notably set_validshape verification).
    // Compare actual SSA values, never the representative MemRef on a phi.
    size_t shared_index = 0;
    std::vector<size_t> kept_indices;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!returns_via_scf[i]) continue;
      if (As<TileType>(op->return_vars_[i]->GetType()) &&
          then_branch.yields[shared_index] == else_branch.yields[shared_index]) {
        BindVarToMlir(op->return_vars_[i], then_branch.yields[shared_index]);
        returns_via_scf[i] = false;
      } else {
        kept_indices.push_back(shared_index);
      }
      ++shared_index;
    }
    // Compact once rather than erasing per result, keeping large tuple-valued
    // Ifs linear in their number of results.
    for (auto* values : {&scf_return_names, &scf_return_types, &then_branch.yields, &then_branch.types,
                         &else_branch.yields, &else_branch.types}) {
      for (size_t i = 0; i < kept_indices.size(); ++i) {
        if (i != kept_indices[i]) (*values)[i] = std::move((*values)[kept_indices[i]]);
      }
      values->resize(kept_indices.size());
    }
    size_t result_index = 0;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!returns_via_scf[i]) continue;
      if (auto tile_type = As<TileType>(op->return_vars_[i]->GetType())) {
        scf_return_types[result_index] = then_branch.types[result_index] == else_branch.types[result_index]
                                             ? then_branch.types[result_index]
                                             : GetTileBufTypeStringFromTileType(tile_type);
        RegisterTileBufType(scf_return_names[result_index], scf_return_types[result_index], tile_type);
      }
      ++result_index;
    }

    auto append_branch = [&](const BranchEmission& branch) {
      IncreaseIndent();
      stream_ << branch.body;
      auto yields = branch.yields;
      size_t index = 0;
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        if (!returns_via_scf[i]) continue;
        if (auto tile_type = As<TileType>(op->return_vars_[i]->GetType());
            tile_type && branch.types[index] != scf_return_types[index]) {
          // Different static/dynamic validity requires a descriptor adapter.
          // Read each arm's live extent; never copy its payload or substitute
          // the return variable's representative allocation address.
          const std::string rows = NewNamedTemp("if_valid_rows");
          const std::string cols = NewNamedTemp("if_valid_cols");
          Emit(rows + ", " + cols + " = pto.get_validshape " + yields[index] + " : " + branch.types[index]);
          yields[index] = EnsureTileView(yields[index], branch.types[index], tile_type, "if_yield",
                                         std::make_pair(rows, cols));
        }
        ++index;
      }
      if (!yields.empty()) Emit("scf.yield " + JoinCommaSep(yields) + " : " + JoinCommaSep(scf_return_types));
      DecreaseIndent();
    };
    if (!scf_return_names.empty()) {
      EmitStructural(JoinCommaSep(scf_return_names) + " = scf.if " + condition + " -> (" +
                     JoinCommaSep(scf_return_types) + ") {");
    } else {
      EmitStructural("scf.if " + condition + " {");
    }
    append_branch(then_branch);
    EmitStructural("} else {");
    append_branch(else_branch);
    EmitStructural("}");

    // Bind in-place return vars (array / tensor) to the shared backing SSA both
    // branches mutated in place. Reads after the IfStmt then resolve to that
    // backing array / tensor_view (the concrete make_tensor_view), so a later
    // pto.partition_view keeps its static dims instead of an scf.if-retyped
    // !pto.tensor_view<?x?> (issue #1533).
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& return_var = op->return_vars_[i];
      const bool is_array = As<ir::ArrayType>(return_var->GetType()) != nullptr;
      const bool is_tensor = ir::AsTensorTypeLike(return_var->GetType()) != nullptr;
      const bool is_distributed = As<ir::DistributedTensorType>(return_var->GetType()) != nullptr;
      if (!is_array && !is_tensor) continue;
      INTERNAL_CHECK_SPAN(!inplace_return_ssa[i].empty(), op->span_)
          << "Internal error: in-place IfStmt return_var '" << return_var->name_hint_
          << "' has no branch-yield SSA";
      BindVarToMlir(return_var, inplace_return_ssa[i]);
      if (is_tensor) {
        BindTensorView(return_var, inplace_return_ssa[i]);
        // Restore the base ptr too, so element-wise pl.read / pl.write on the
        // merged tensor resolve to the backing pointer rather than the view SSA.
        auto base_it = fs_.view_ssa_to_base_ptr.find(inplace_return_ssa[i]);
        if (base_it != fs_.view_ssa_to_base_ptr.end()) {
          RegisterBasePtr(return_var, base_it->second);
        }
        if (is_distributed) {
          INTERNAL_CHECK_SPAN(!inplace_return_comm_ctx[i].empty(), op->span_)
              << "Internal error: IfStmt DistributedTensor return_var '" << return_var->name_hint_
              << "' has no merged CommContext binding";
          RegisterCommCtxFor(return_var, inplace_return_comm_ctx[i]);
        }
      }
    }
  }
}

void PTOCodegen::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null ForStmt";
  INTERNAL_CHECK_SPAN(op->loop_var_ != nullptr, op->span_) << "Internal error: ForStmt has null loop_var";
  INTERNAL_CHECK_SPAN(op->body_ != nullptr, op->span_) << "Internal error: ForStmt has null body";

  INTERNAL_CHECK_SPAN(op->iter_args_.size() == op->return_vars_.size(), op->span_)
      << "ForStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  INTERNAL_CHECK_SPAN(op->kind_ != ir::ForKind::Unroll, op->span_)
      << "Internal error: ForKind::Unroll reached codegen — UnrollLoops "
      << "should have resolved it. The pipeline is incomplete.";
  INTERNAL_CHECK_SPAN(op->kind_ != ir::ForKind::Pipeline, op->span_)
      << "Internal error: ForKind::Pipeline reached codegen — LowerPipelineLoops "
      << "and CanonicalizeIOOrder should have demoted it to Sequential. "
      << "The pipeline is incomplete.";

  // Device loops lower to MLIR ``scf.for``, which iterates lower -> upper bound
  // and is defined for a positive step only. A descending ``ForStmt`` has no
  // faithful lowering: emitting it verbatim yields a zero-trip ``scf.for`` that
  // the assembler folds away, silently discarding the loop body. Surface it as
  // a user-facing limitation instead of miscompiling.
  //
  // This check is permanent, not a workaround pending an assembler fix. The
  // PTOAS team has confirmed they will not support a non-positive ``scf.for``
  // step in the foreseeable future; hw-native-sys/PTOAS#1288 will be closed by
  // adding the missing assertion only (today ptoas silently accepts such a
  // step — a negative one drops the body, a zero one emits ``i += 0``). So a
  // descending device loop stays unrepresentable, and this check is the only
  // thing standing between the user and a silently empty kernel.
  //
  // Keep it even once that assertion ships: it fires earlier, names the user's
  // loop through the ``Span``, and says how to rewrite it, whereas the
  // assembler can only report against generated ``.pto`` the user never wrote.
  //
  // Only a compile-time step is checked. A runtime step that turns out negative
  // still lowers to the same ill-defined ``scf.for``; proving its sign needs the
  // arith analyzer. Closing that gap would mean normalizing descending loops to
  // ascending form here (deriving the induction variable from an ascending
  // counter), which is the only route to supporting them at all now that the
  // assembler will not. Orchestration functions emit C++ directly and are
  // unaffected — they support descending loops natively.
  if (auto const_step = ir::transform_utils::EvalConstInt(op->step_)) {
    CHECK_SPAN(*const_step > 0, op->span_)
        << "loops in device functions must have a positive step, but this loop steps by " << *const_step
        << ". Device code lowers to MLIR 'scf.for', which only counts upward. "
           "Rewrite the loop in ascending form and invert the index in the body — "
           "replace 'for i in pl.range(64, 0, -1)' with 'for t in pl.range(0, 64)' plus "
           "'i = 64 - t'. Orchestration functions support descending loops directly.";
  }

  // Evaluate loop bounds and ensure they are index-typed for scf.for.
  // EmitCastToIndex is a no-op when the bound is already DataType::INDEX
  // (e.g. ConstInt literals from pl.range(8)); for i32 runtime values such as
  // pld.nranks(ctx) or pld.rank(ctx) it emits arith.index_cast, consistent
  // with how every other integer-at-MLIR-boundary site (tensor views, array
  // offsets) is handled in this codegen.
  VisitExpr(op->start_);
  std::string start = EmitCastToIndex(op->start_, fs_.current_expr_value);
  fs_.current_expr_value = "";

  VisitExpr(op->stop_);
  std::string stop = EmitCastToIndex(op->stop_, fs_.current_expr_value);
  fs_.current_expr_value = "";

  VisitExpr(op->step_);
  std::string step = EmitCastToIndex(op->step_, fs_.current_expr_value);
  fs_.current_expr_value = "";

  // Register loop variable
  std::string loop_var_name = NewNamedTemp(op->loop_var_->name_hint_);
  BindVarToMlir(op->loop_var_, loop_var_name);

  // Scalars and registers (including masks) are immutable SSA values and need
  // iter_args/yield to carry each iteration's result. TileType and TensorType
  // are mutable references written in-place via outs(), so they are mapped
  // directly to their init values and excluded from iter_args/yield.
  std::vector<bool> is_ssa(op->iter_args_.size(), false);
  bool has_ssa_iter_args = false;
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (As<ScalarType>(op->iter_args_[i]->GetType()) || As<ir::RegType>(op->iter_args_[i]->GetType())) {
      is_ssa[i] = true;
      has_ssa_iter_args = true;
    }
  }

  // Map non-scalar iter_args/return_vars directly to their init values
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (is_ssa[i]) continue;

    const auto& iter_arg = op->iter_args_[i];
    const auto& return_var = op->return_vars_[i];

    std::string init_mlir_name;
    auto tensor_type = ir::AsTensorTypeLike(iter_arg->GetType());
    auto init_var = AsVarLike(iter_arg->initValue_);
    if (tensor_type) {
      INTERNAL_CHECK_SPAN(init_var, op->span_) << "Tensor-like iter_arg init value must be a Var or IterArg";
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
      const auto base_ptr = GetTensorBasePtr(init_var);
      RegisterBasePtr(iter_arg, base_ptr);
      RegisterBasePtr(return_var, base_ptr);
      const auto comm_ctx = GetCommCtxSSAFor(init_var.get());
      if (As<ir::DistributedTensorType>(iter_arg->GetType())) {
        INTERNAL_CHECK_SPAN(!comm_ctx.empty(), op->span_)
            << "Internal error: ForStmt DistributedTensor iter_arg '" << iter_arg->name_hint_
            << "' initialization '" << init_var->name_hint_ << "' has no CommContext binding";
      }
      RegisterCommCtxFor(iter_arg, comm_ctx);
      RegisterCommCtxFor(return_var, comm_ctx);
    } else if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      const auto memref = ir::GetDefinedMemRef(tile_type);
      BindVarToMemRef(iter_arg, memref->base_.get());
      BindVarToMemRef(return_var, memref->base_.get());
    }
  }

  // Record the induction variable's constant lower bound for the body, so an
  // index expression derived from it can be proven non-negative and lower to a
  // truncating division/remainder instead of a flooring correction. The loop is
  // ascending -- a non-positive step is rejected above -- so the variable never
  // goes below its start. Scoped to the body, and skipped if an enclosing loop
  // already bound the same Var, so that loop keeps ownership of the entry.
  //
  // The start must fit the dtype it is *emitted* at, which is the literal's own
  // -- `EmitArithOperand` narrows an operand into its own type before the loop
  // header is formed. A stored literal need not fit: an INT32 start of
  // 4294967295 is emitted as -1, so recording the positive stored value would
  // claim a lower bound the hardware never has. Checking it against the loop
  // variable's dtype instead is not enough, because a wide induction variable
  // admits a narrow literal that still changes sign.
  const auto* loop_var_key = op->loop_var_.get();
  auto constant_start = As<ir::ConstInt>(op->start_);
  if (constant_start &&
      !ir::arith::detail::ConstIntFitsDtype(constant_start->value_, ir::GetScalarDtype(op->start_))) {
    constant_start = nullptr;
  }
  struct LowerBoundScope {
    std::map<const ir::Var*, int64_t>* bounds = nullptr;
    const ir::Var* key = nullptr;
    LowerBoundScope() = default;
    LowerBoundScope(const LowerBoundScope&) = delete;
    LowerBoundScope& operator=(const LowerBoundScope&) = delete;
    ~LowerBoundScope() {
      if (bounds != nullptr) bounds->erase(key);
    }
  } lower_bound_scope;
  if (constant_start && fs_.loop_var_lower_bound.count(loop_var_key) == 0) {
    fs_.loop_var_lower_bound[loop_var_key] = constant_start->value_;
    // Assign the members directly: constructing a temporary here would run its
    // destructor at the end of this statement and erase the entry at once.
    lower_bound_scope.bounds = &fs_.loop_var_lower_bound;
    lower_bound_scope.key = loop_var_key;
  }

  if (!has_ssa_iter_args) {
    // Simple scf.for (no iter_args, or all iter_args are non-scalar)
    EmitStructural("scf.for " + loop_var_name + " = " + start + " to " + stop + " step " + step + " {");
    IncreaseIndent();

    fs_.yield_buffer.clear();
    VisitStmt(op->body_);
    fs_.yield_buffer.clear();

    DecreaseIndent();
    EmitStructural("}");
  } else {
    // scf.for with SSA value iter_args only
    std::vector<std::string> init_values;
    std::vector<std::string> iter_arg_names;
    std::vector<std::string> iter_arg_types;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!is_ssa[i]) continue;

      const auto& iter_arg = op->iter_args_[i];

      VisitExpr(iter_arg->initValue_);
      init_values.push_back(fs_.current_expr_value);
      fs_.current_expr_value = "";

      std::string iter_name = NewNamedTemp(iter_arg->name_hint_);
      BindVarToMlir(iter_arg, iter_name);
      iter_arg_names.push_back(iter_name);

      iter_arg_types.push_back(GetValueCarryTypeString(iter_arg->GetType()));
    }

    // Register return_vars SSA names (SSA values only)
    std::vector<std::string> return_var_names;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!is_ssa[i]) continue;
      std::string ret_name = NewNamedTemp(op->return_vars_[i]->name_hint_);
      BindVarToMlir(op->return_vars_[i], ret_name);
      return_var_names.push_back(ret_name);
    }

    // Emit: %ret0 = scf.for %i = %start to %stop step %step
    //           iter_args(%acc = %init) -> (type) {
    EmitStructural(JoinCommaSep(return_var_names) + " = scf.for " + loop_var_name + " = " + start + " to " +
                   stop + " step " + step + " iter_args(" + JoinPairs(iter_arg_names, " = ", init_values) +
                   ") -> (" + JoinCommaSep(iter_arg_types) + ") {");
    IncreaseIndent();

    fs_.yield_buffer.clear();
    VisitStmt(op->body_);

    // Filter yield_buffer to keep only SSA value iter_arg entries
    std::vector<std::string> ssa_yields;
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (is_ssa[i] && i < fs_.yield_buffer.size()) {
        ssa_yields.push_back(fs_.yield_buffer[i]);
      }
    }

    // Emit scf.yield from filtered yield values
    if (!ssa_yields.empty()) {
      std::ostringstream yield_oss;
      yield_oss << "scf.yield ";
      for (size_t i = 0; i < ssa_yields.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << ssa_yields[i];
      }
      yield_oss << " : ";
      for (size_t i = 0; i < iter_arg_types.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << iter_arg_types[i];
      }
      Emit(yield_oss.str());
    }
    CHECK(ssa_yields.size() == iter_arg_types.size())
        << "ForStmt scalar yield count (" << ssa_yields.size() << ") must match scalar iter_args ("
        << iter_arg_types.size() << ")";
    fs_.yield_buffer.clear();

    DecreaseIndent();
    EmitStructural("}");
  }
}

void PTOCodegen::VisitStmt_(const WhileStmtPtr& op) {
  INTERNAL_CHECK(op != nullptr) << "Internal error: null WhileStmt";
  INTERNAL_CHECK_SPAN(op->condition_ != nullptr, op->span_) << "Internal error: WhileStmt has null condition";
  INTERNAL_CHECK_SPAN(op->body_ != nullptr, op->span_) << "Internal error: WhileStmt has null body";

  INTERNAL_CHECK_SPAN(op->iter_args_.size() == op->return_vars_.size(), op->span_)
      << "WhileStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  // Scalars and registers (including masks) are immutable SSA values and need
  // iter_args/yield to carry each iteration's result. TileType and TensorType
  // are mutable references written in-place via outs(), so they are mapped
  // directly to their init values and excluded from iter_args/yield.
  std::vector<bool> is_ssa(op->iter_args_.size(), false);
  bool has_ssa_iter_args = false;
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (As<ScalarType>(op->iter_args_[i]->GetType()) || As<ir::RegType>(op->iter_args_[i]->GetType())) {
      is_ssa[i] = true;
      has_ssa_iter_args = true;
    }
  }

  // Map non-scalar iter_args/return_vars directly to their init values
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (is_ssa[i]) continue;

    const auto& iter_arg = op->iter_args_[i];
    const auto& return_var = op->return_vars_[i];

    std::string init_mlir_name;
    auto tensor_type = ir::AsTensorTypeLike(iter_arg->GetType());
    auto init_var = AsVarLike(iter_arg->initValue_);
    if (tensor_type) {
      INTERNAL_CHECK_SPAN(init_var, op->span_) << "Tensor-like iter_arg init value must be a Var or IterArg";
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
      const auto base_ptr = GetTensorBasePtr(init_var);
      RegisterBasePtr(iter_arg, base_ptr);
      RegisterBasePtr(return_var, base_ptr);
      const auto comm_ctx = GetCommCtxSSAFor(init_var.get());
      if (As<ir::DistributedTensorType>(iter_arg->GetType())) {
        INTERNAL_CHECK_SPAN(!comm_ctx.empty(), op->span_)
            << "Internal error: WhileStmt DistributedTensor iter_arg '" << iter_arg->name_hint_
            << "' initialization '" << init_var->name_hint_ << "' has no CommContext binding";
      }
      RegisterCommCtxFor(iter_arg, comm_ctx);
      RegisterCommCtxFor(return_var, comm_ctx);
    } else if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      const auto memref = ir::GetDefinedMemRef(tile_type);
      BindVarToMemRef(iter_arg, memref->base_.get());
      BindVarToMemRef(return_var, memref->base_.get());
    }
  }

  if (!has_ssa_iter_args) {
    // Simple scf.while (no iter_args, or all iter_args are non-scalar)
    EmitStructural("scf.while : () -> () {");
    IncreaseIndent();

    VisitExpr(op->condition_);
    std::string cond = fs_.current_expr_value;
    fs_.current_expr_value = "";
    Emit("scf.condition(" + cond + ")");

    DecreaseIndent();
    EmitStructural("} do {");
    IncreaseIndent();

    fs_.yield_buffer.clear();
    VisitStmt(op->body_);

    Emit("scf.yield");
    fs_.yield_buffer.clear();

    DecreaseIndent();
    EmitStructural("}");
  } else {
    // scf.while with SSA value iter_args only
    std::vector<std::string> init_values;
    std::vector<std::string> before_arg_names;
    std::vector<std::string> after_arg_names;
    std::vector<std::string> iter_arg_types;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!is_ssa[i]) continue;

      const auto& iter_arg = op->iter_args_[i];

      VisitExpr(iter_arg->initValue_);
      init_values.push_back(fs_.current_expr_value);
      fs_.current_expr_value = "";

      before_arg_names.push_back(NewTemp());
      after_arg_names.push_back(NewTemp());

      iter_arg_types.push_back(GetValueCarryTypeString(iter_arg->GetType()));
    }

    // Register return_vars SSA names (SSA values only)
    std::vector<std::string> return_var_names;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!is_ssa[i]) continue;

      std::string ret_name = NewTemp();
      BindVarToMlir(op->return_vars_[i], ret_name);
      return_var_names.push_back(ret_name);
    }

    // Lambda to register SSA value iter_args in fs_.var_to_mlir
    auto register_ssa_iter_args = [&](const std::vector<std::string>& ssa_names) {
      size_t ssa_idx = 0;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        if (!is_ssa[i]) continue;
        BindVarToMlir(op->iter_args_[i], ssa_names[ssa_idx]);
        ssa_idx++;
      }
    };

    std::string types_str = "(" + JoinCommaSep(iter_arg_types) + ")";

    // Emit: %ret0, %ret1 = scf.while (%before0 = %init0, ...) : (types) -> (types) {
    EmitStructural(JoinCommaSep(return_var_names) + " = scf.while (" +
                   JoinPairs(before_arg_names, " = ", init_values) + ") : " + types_str + " -> " + types_str +
                   " {");
    IncreaseIndent();

    // Before region: register before-region args, evaluate condition
    register_ssa_iter_args(before_arg_names);

    VisitExpr(op->condition_);
    std::string cond = fs_.current_expr_value;
    fs_.current_expr_value = "";

    // Emit: scf.condition(%cond) %before0, %before1 : type0, type1
    Emit("scf.condition(" + cond + ") " + JoinCommaSep(before_arg_names) + " : " +
         JoinCommaSep(iter_arg_types));

    DecreaseIndent();
    EmitStructural("} do {");

    // After region: emit ^bb0 block header with typed arguments
    EmitStructural("^bb0(" + JoinPairs(after_arg_names, " : ", iter_arg_types) + "):");
    IncreaseIndent();

    // Re-register iter_args with after-region SSA names
    register_ssa_iter_args(after_arg_names);

    // Visit body
    fs_.yield_buffer.clear();
    VisitStmt(op->body_);

    // Filter yield_buffer to keep only SSA value iter_arg entries
    std::vector<std::string> ssa_yields;
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (is_ssa[i] && i < fs_.yield_buffer.size()) {
        ssa_yields.push_back(fs_.yield_buffer[i]);
      }
    }

    // Emit scf.yield from filtered yield values
    if (!ssa_yields.empty()) {
      Emit("scf.yield " + JoinCommaSep(ssa_yields) + " : " + JoinCommaSep(iter_arg_types));
    }
    CHECK(ssa_yields.size() == iter_arg_types.size())
        << "WhileStmt scalar yield count (" << ssa_yields.size() << ") must match scalar iter_args ("
        << iter_arg_types.size() << ")";
    fs_.yield_buffer.clear();

    DecreaseIndent();
    EmitStructural("}");
  }
}

}  // namespace codegen
}  // namespace pypto
