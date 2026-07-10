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
#include <utility>
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
    // Tensors are mutable references: a branch may yield either a freshly
    // stored-into tensor (bound to its tensor_view) or the unchanged input (a
    // param bound only to its base ptr). Normalize tensor yields to the
    // canonical tensor_view so consumers — notably the IfStmt in-place
    // return_var binding — see a consistent SSA across branches that aliases
    // the same concrete make_tensor_view (issue #1533). For/While loops keep
    // only scalar yields, so this does not affect their lowering.
    if (As<TensorType>(expr->GetType())) {
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
// source so a branch-local producer can be re-pointed at the phi handle.
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

// True when `var` is defined lexically inside `body` — an AssignStmt result, a
// loop iter-arg/return-var, or a nested control-flow return-var — i.e. a
// branch-local value whose handle can be safely re-pointed at the phi buffer. A
// yield source that is NOT branch-local (a function param / outer-scope tile
// threaded through the branch unchanged) must not be re-bound: its handle is read
// elsewhere, so re-pointing it would corrupt those reads.
bool IsDefinedInBranch(const ir::Var* var, const StmtPtr& body) {
  if (!body || !var) return false;
  if (auto assign = As<ir::AssignStmt>(body)) {
    // Only a real op producer (Call value) writes a fresh buffer we may re-point.
    // A bare-var alias (`r = a`) or a view writes nothing at its own handle, so
    // re-pointing it would leave the phi buffer unwritten — the tmov fallback in
    // emit_branch copies those into the phi handle instead.
    return assign->var_.get() == var && As<ir::Call>(assign->value_) != nullptr;
  }
  if (auto seq = As<ir::SeqStmts>(body)) {
    for (const auto& s : seq->stmts_)
      if (IsDefinedInBranch(var, s)) return true;
    return false;
  }
  if (auto for_stmt = As<ir::ForStmt>(body)) {
    for (const auto& ia : for_stmt->iter_args_)
      if (ia.get() == var) return true;
    for (const auto& rv : for_stmt->return_vars_)
      if (rv.get() == var) return true;
    return IsDefinedInBranch(var, for_stmt->body_);
  }
  if (auto while_stmt = As<ir::WhileStmt>(body)) {
    for (const auto& ia : while_stmt->iter_args_)
      if (ia.get() == var) return true;
    for (const auto& rv : while_stmt->return_vars_)
      if (rv.get() == var) return true;
    return IsDefinedInBranch(var, while_stmt->body_);
  }
  if (auto if_stmt = As<ir::IfStmt>(body)) {
    for (const auto& rv : if_stmt->return_vars_)
      if (rv.get() == var) return true;
    if (IsDefinedInBranch(var, if_stmt->then_body_)) return true;
    return if_stmt->else_body_.has_value() && IsDefinedInBranch(var, *if_stmt->else_body_);
  }
  if (auto scope = AsScope(body)) return IsDefinedInBranch(var, scope->body_);
  return false;
}
}  // namespace

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
        // This head-declared handle is the phi buffer. Under PTOAS the branch
        // producers are re-bound to it (see emit_branch, fix #1956); mark it
        // emitted so their EmitAllocTileForVar dedups instead of re-declaring it.
        if (!emit_tile_addr_) fs_.emitted_tile_alloc_names.insert(ret_name);
      } else if (As<TensorType>(return_var->GetType()) || As<ir::ArrayType>(return_var->GetType())) {
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

    if (!scf_return_names.empty()) {
      Emit(JoinCommaSep(scf_return_names) + " = scf.if " + condition + " -> (" +
           JoinCommaSep(scf_return_types) + ") {");
    } else {
      Emit("scf.if " + condition + " {");
    }
    indent_level_++;

    // For in-place return vars (ArrayType and TensorType, both kept out of
    // scf.if results), capture the backing SSA that the branches yield so it can
    // be bound to the merged return var after both branches emit. Both branches
    // yield the same SSA because every array.update_element / pl.assemble
    // aliases the one backing array / tensor.
    std::vector<std::string> inplace_return_ssa(op->return_vars_.size());

    auto emit_branch = [&](const StmtPtr& body, const char* branch_name) {
      // Fix #1956: under memory_planner=PTOAS, MemoryReuse (which would alias the
      // branch yields onto the phi return_var's canonical MemRef via
      // YieldFixupMutator) is skipped. Without it, each branch's tile producer
      // writes its own handle and the post-if read of the phi handle reads a
      // buffer no branch wrote. Re-bind each tile yield-source var to the phi
      // return_var's (head-declared) handle so this branch's producer writes it.
      // Under PYPTO (emit_tile_addr_) the IR-level aliasing already holds, so
      // leave the bindings untouched.
      if (!emit_tile_addr_) {
        if (auto yield = FindBranchYield(body)) {
          for (size_t i = 0; i < op->return_vars_.size(); ++i) {
            if (i >= yield->value_.size()) continue;
            if (!As<TileType>(op->return_vars_[i]->GetType())) continue;
            auto src = ir::AsVarLike(yield->value_[i]);
            if (!src) continue;
            // Only re-point a branch-local producer; an outer var yielded through
            // the branch is read elsewhere and must keep its own handle.
            if (!IsDefinedInBranch(src.get(), body)) continue;
            auto phi_it = fs_.var_to_mlir.find(GetVarKey(op->return_vars_[i]));
            if (phi_it != fs_.var_to_mlir.end()) BindVarToMlir(src, phi_it->second);
          }
        }
      }
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
        } else if (As<ir::ArrayType>(op->return_vars_[i]->GetType()) ||
                   As<TensorType>(op->return_vars_[i]->GetType())) {
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
        } else if (!emit_tile_addr_ && As<TileType>(op->return_vars_[i]->GetType())) {
          // Tile phi under PTOAS (fix #1956). A branch-local producer was already
          // re-pointed at the phi handle above, so its yield resolves to the phi
          // handle and needs nothing. Any other yield — an outer tile threaded
          // through the branch, or one the re-point conservatively skipped — has
          // NOT written the phi buffer, so copy it in: without MemoryReuse's
          // YieldFixupMutator (skipped under PTOAS) the post-if read would
          // otherwise see an uninitialised phi buffer.
          auto phi_it = fs_.var_to_mlir.find(GetVarKey(op->return_vars_[i]));
          if (phi_it != fs_.var_to_mlir.end() && !branch_yields[i].empty() &&
              branch_yields[i] != phi_it->second) {
            const std::string& phi = phi_it->second;
            std::string ty;
            if (auto tit = fs_.ssa_to_tile_buf_type.find(phi); tit != fs_.ssa_to_tile_buf_type.end()) {
              ty = tit->second;
            }
            std::ostringstream mov;
            mov << "pto.tmov ins(" << branch_yields[i];
            if (!ty.empty()) mov << " : " << ty;
            mov << ") outs(" << phi;
            if (!ty.empty()) mov << " : " << ty;
            mov << ")";
            Emit(mov.str());
          }
        }
        // Tile return_vars under PYPTO: MemoryReuse ensures branch yields share the
        // return_var's canonical MemRef (same physical address), so no codegen-level
        // tmov is needed — the IR-level tile.move (from YieldFixupMutator) handles it.
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

    // Bind in-place return vars (array / tensor) to the shared backing SSA both
    // branches mutated in place. Reads after the IfStmt then resolve to that
    // backing array / tensor_view (the concrete make_tensor_view), so a later
    // pto.partition_view keeps its static dims instead of an scf.if-retyped
    // !pto.tensor_view<?x?> (issue #1533).
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& return_var = op->return_vars_[i];
      const bool is_array = As<ir::ArrayType>(return_var->GetType()) != nullptr;
      const bool is_tensor = As<TensorType>(return_var->GetType()) != nullptr;
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
      }
    }
  }
}

void PTOCodegen::VisitStmt_(const ForStmtPtr& op) {
  INTERNAL_CHECK_SPAN(op != nullptr, op->span_) << "Internal error: null ForStmt";
  INTERNAL_CHECK_SPAN(op->loop_var_ != nullptr, op->span_) << "Internal error: ForStmt has null loop_var";
  INTERNAL_CHECK_SPAN(op->body_ != nullptr, op->span_) << "Internal error: ForStmt has null body";

  CHECK(op->iter_args_.size() == op->return_vars_.size())
      << "ForStmt iter_args size (" << op->iter_args_.size() << ") must equal return_vars size ("
      << op->return_vars_.size() << ")";

  INTERNAL_CHECK_SPAN(op->kind_ != ir::ForKind::Unroll, op->span_)
      << "Internal error: ForKind::Unroll reached codegen — UnrollLoops "
      << "should have resolved it. The pipeline is incomplete.";
  INTERNAL_CHECK_SPAN(op->kind_ != ir::ForKind::Pipeline, op->span_)
      << "Internal error: ForKind::Pipeline reached codegen — LowerPipelineLoops "
      << "and CanonicalizeIOOrder should have demoted it to Sequential. "
      << "The pipeline is incomplete.";

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

  // ptoas multi-buffer (route-2): the region `pto.alloc_multi_tile` is emitted
  // inline at its own `tile.multi_buffer_alloc` AssignStmt (placed before this
  // loop by ConvertToPtoasMultiBuffer), and the buffer-less slot views emit
  // their `pto.multi_tile_get` in the get_slot / load_slot op codegen — so no
  // per-loop hoist/rotation state is needed here.

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
