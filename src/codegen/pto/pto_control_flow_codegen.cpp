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

#include <algorithm>
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
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/structural_comparison.h"
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
bool IsRegionSSAValue(const ir::TypePtr& type) {
  if (As<ScalarType>(type) || As<ir::BufferType>(type)) return true;
  auto tile = As<TileType>(type);
  return tile && !tile->memref_.has_value();
}

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
    if (assign->var_.get() != var) return false;
    auto call = As<ir::Call>(assign->value_);
    if (!call || !call->op_) return false;
    // A zero-copy view (inherit-input: tile.reshape, tile.slice, ...) IS the
    // "view" the comment above excludes. Its codegen emits nothing when the target
    // handle already carries the result type, so re-pointing it at the phi handle
    // leaves the phi buffer unwritten. `[N, 1]` col-vector carries always hit this:
    // their elementwise ops run on a `[1, N]` row-major view and the branch yields
    // the reshape back, not the op result.
    auto& registry = ir::OpRegistry::GetInstance();
    if (registry.IsRegistered(call->op_->name_) &&
        registry.GetEntry(call->op_->name_).OutputMemoryInheritsInput()) {
      return false;
    }
    return true;
  }
  if (auto seq = As<ir::SeqStmts>(body)) {
    for (const auto& s : seq->stmts_) {
      if (IsDefinedInBranch(var, s)) return true;
    }
    return false;
  }
  if (auto for_stmt = As<ir::ForStmt>(body)) {
    for (const auto& ia : for_stmt->iter_args_) {
      if (ia.get() == var) return true;
    }
    for (const auto& rv : for_stmt->return_vars_) {
      if (rv.get() == var) return true;
    }
    return IsDefinedInBranch(var, for_stmt->body_);
  }
  if (auto while_stmt = As<ir::WhileStmt>(body)) {
    for (const auto& ia : while_stmt->iter_args_) {
      if (ia.get() == var) return true;
    }
    for (const auto& rv : while_stmt->return_vars_) {
      if (rv.get() == var) return true;
    }
    return IsDefinedInBranch(var, while_stmt->body_);
  }
  if (auto if_stmt = As<ir::IfStmt>(body)) {
    for (const auto& rv : if_stmt->return_vars_) {
      if (rv.get() == var) return true;
    }
    if (IsDefinedInBranch(var, if_stmt->then_body_)) return true;
    return if_stmt->else_body_.has_value() && IsDefinedInBranch(var, *if_stmt->else_body_);
  }
  if (auto scope = AsScope(body)) return IsDefinedInBranch(var, scope->body_);
  return false;
}
}  // namespace

std::string PTOCodegen::GetRegionValueTypeString(const ir::TypePtr& type, const ir::Span& span) {
  if (auto scalar = As<ScalarType>(type)) return GetScalarIterArgTypeString(scalar);
  if (auto buffer = As<ir::BufferType>(type)) return GetBufferTypeString(buffer, span);
  auto tile = As<TileType>(type);
  INTERNAL_CHECK_SPAN(tile && !tile->memref_.has_value(), span)
      << "Internal error: SCF reference result requires a buffer or selected Tile reference";
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  const bool static_valid =
      std::all_of(view.valid_shape.begin(), view.valid_shape.end(),
                  [](const ir::ExprPtr& extent) { return As<ir::ConstInt>(extent) != nullptr; });
  // A static descriptor view needs no metadata setter and is legal even when
  // its address comes from another SCF reference. Dynamic views retain the
  // explicit valid operands supplied by their concrete incoming handles.
  return static_valid ? GetViewTileBufTypeStringFromTileType(tile) : GetTileBufTypeStringFromTileType(tile);
}

void PTOCodegen::BindRegionValue(const ir::VarPtr& var, const std::string& name) {
  BindVarToMlir(var, name);
  if (As<ScalarType>(var->GetType())) return;
  RegisterTileBufType(name, GetRegionValueTypeString(var->GetType(), var->span_),
                      As<TileType>(var->GetType()));
  // A reference phi owns its descriptor, while its storage is selected by SCF.
  // It must never enter allocation deduplication or acquire an invented MemRef.
  RegisterMutableTileName(name);
}

std::string PTOCodegen::PrepareRegionYield(const ir::ExprPtr& value, const std::string& name,
                                           const ir::VarPtr& result) {
  const auto target = As<TileType>(result->GetType());
  if (!target) return name;
  const std::string target_type = GetRegionValueTypeString(result->GetType(), result->span_);
  const std::string source_type = GetExprTypeAnnotation(value);
  const auto expected = ir::tile_view_semantics::GetEffectiveTileView(*target);
  const auto same_metadata = [&expected](const ir::TileTypePtr& incoming) {
    if (!incoming) return false;
    const auto actual = ir::tile_view_semantics::GetEffectiveTileView(*incoming);
    if (actual.pad != expected.pad || actual.valid_shape.size() != expected.valid_shape.size()) return false;
    for (size_t axis = 0; axis < actual.valid_shape.size(); ++axis) {
      if (!ir::structural_equal(actual.valid_shape[axis], expected.valid_shape[axis])) return false;
    }
    return true;
  };
  const auto actual = fs_.ssa_to_tile_type.find(name);
  const bool matching_metadata = same_metadata(As<TileType>(value->GetType())) &&
                                 (actual == fs_.ssa_to_tile_type.end() || same_metadata(actual->second));
  if (source_type == target_type && matching_metadata) return name;
  INTERNAL_CHECK_SPAN(!source_type.empty(), value->span_)
      << "Internal error: selected Tile yield has no emitted descriptor";
  const bool static_valid =
      std::all_of(expected.valid_shape.begin(), expected.valid_shape.end(),
                  [](const ir::ExprPtr& extent) { return As<ir::ConstInt>(extent) != nullptr; });
  if (static_valid) return EnsureTileView(name, source_type, target, "region_yield");
  std::string origin = name;
  if (const auto view = fs_.tile_view_names.find(name); view != fs_.tile_view_names.end()) {
    origin = view->second;
  }
  CHECK_SPAN(fs_.emitted_tile_alloc_names.count(origin) && !IsTileViewName(origin), result->span_)
      << "PTOAS v0.67 cannot change runtime valid extents on a selected storage reference; "
         "keep the same descriptor or prepare each concrete incoming value's metadata before selecting";
  // Equal emitted '?' types do not establish equal logical validity. A region
  // result owns its declared metadata, including when its selected source has
  // wider valid extents. Initialize an independent view from the RESULT's
  // expressions; querying or updating the source would violate that contract
  // and could corrupt later observations of the original full handle.
  const auto fields = ComputeAllocTileFields(target);
  INTERNAL_CHECK_SPAN(!fields.valid_row_ssa.empty() && !fields.valid_col_ssa.empty(), result->span_)
      << "Internal error: selected Tile result requires both valid-shape operands";
  return EnsureTileView(name, source_type, target, "region_yield",
                        std::make_pair(fields.valid_row_ssa, fields.valid_col_ssa));
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
    // Concrete Tile results retain their canonical destination buffers. Scalars
    // and references without one MemRef become real SCF results: choosing a
    // handle does not allocate storage or require an otherwise illegal copy.
    std::vector<bool> returns_via_scf(op->return_vars_.size(), false);
    std::vector<std::string> scf_return_names;
    std::vector<std::string> scf_return_types;
    std::vector<std::pair<std::string, AllocTileFields>> deferred_tile_valid_shapes;
    const auto then_yield = FindBranchYield(op->then_body_);
    const auto else_yield = op->else_body_ ? FindBranchYield(*op->else_body_) : nullptr;

    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& return_var = op->return_vars_[i];
      if (IsRegionSSAValue(return_var->GetType())) {
        std::string ret_name = NewNamedTemp(return_var->name_hint_);
        BindRegionValue(return_var, ret_name);
        scf_return_names.push_back(ret_name);
        scf_return_types.push_back(GetRegionValueTypeString(return_var->GetType(), return_var->span_));
        returns_via_scf[i] = true;
      } else if (auto tile_type = As<TileType>(return_var->GetType())) {
        INTERNAL_CHECK_SPAN(tile_type->memref_.has_value(), op->span_)
            << "TileType return_var must have a MemRef at codegen stage for var: " << return_var->name_hint_;
        // A tile phi handle is declared in the function head so it dominates
        // both branches and every post-if read. A dynamic valid_shape may be
        // computed only in the surrounding loop/body, however, and therefore
        // cannot be an operand of that head declaration. Declare such a handle
        // with its physical box and restore the logical valid shape here,
        // immediately before the if uses it.
        AllocTileFields logical_fields = ComputeAllocTileFields(tile_type);
        bool has_dynamic_valid_shape = false;
        if (const auto& tile_view = tile_type->tile_view_; tile_view.has_value()) {
          for (const auto& dim : tile_view->valid_shape) {
            if (dim && !As<ir::ConstInt>(dim)) {
              has_dynamic_valid_shape = true;
              break;
            }
          }
        }
        AllocTileFields alloc_fields =
            has_dynamic_valid_shape ? ComputeAllocTileFields(tile_type, true) : logical_fields;
        // Under PTOAS no `addr` is baked, so variables denoting the same buffer
        // must share ONE tile_buf handle — two addr-less allocs are two
        // independent buffers to ptoas PlanMemory. When the phi's MemRef is
        // already bound to a handle (a loop-carried accumulator: the `pl.range`
        // init, the iter_arg and the loop result all share the phi's MemRef),
        // reuse it. Minting a second handle here would strand that buffer: the
        // branch producers write the phi handle (they are re-bound to it below,
        // fix #1956) while the loop result and every post-if read still resolve
        // to the shared one, which no branch ever wrote.
        const auto then_value =
            then_yield && i < then_yield->value_.size() ? ir::AsVarLike(then_yield->value_[i]) : nullptr;
        const auto else_value =
            else_yield && i < else_yield->value_.size() ? ir::AsVarLike(else_yield->value_[i]) : nullptr;
        std::string ret_name =
            TryGetSharedTileBufHandle(tile_type, alloc_fields.type_str, then_value, else_value);
        if (!ret_name.empty()) {
          // The shared handle must dominate both branches and the post-if read.
          // Hoist its declaration to the function head unless the body already
          // emitted it before this region.
          DeclareTileBufAtHead(ret_name, alloc_fields);
        } else {
          ret_name = AllocNewTileBuf(alloc_fields.type_str, return_var->name_hint_, alloc_fields.addr_ssa,
                                     alloc_fields.valid_row_ssa, alloc_fields.valid_col_ssa);
          RegisterTileBufType(ret_name, alloc_fields.type_str, tile_type);
          // This head-declared handle is the phi buffer. Under PTOAS the branch
          // producers are re-bound to it (see emit_branch, fix #1956); mark it
          // emitted so their EmitAllocTileForVar dedups instead of re-declaring it.
          if (!emit_tile_addr_) fs_.emitted_tile_alloc_names.insert(ret_name);
        }
        // Prepare before entering either branch: a phi destination view must
        // dominate branch writes as well as reads after this scf.if. Branch
        // producers and the synthetic tmov below share this exact target.
        ret_name = PrepareTileDestination(return_var, ret_name, "if result", true).name;
        bool restore_valid_shape = has_dynamic_valid_shape;
        if (!emit_tile_addr_) {
          const auto actual = fs_.ssa_to_tile_type.find(ret_name);
          INTERNAL_CHECK_SPAN(actual != fs_.ssa_to_tile_type.end(), return_var->span_)
              << "Internal error: missing tile descriptor for if destination " << ret_name;
          const auto actual_view = ir::tile_view_semantics::GetEffectiveTileView(*actual->second);
          const auto result_view = ir::tile_view_semantics::GetEffectiveTileView(*tile_type);
          bool same_valid_shape = actual_view.valid_shape.size() == result_view.valid_shape.size();
          for (size_t dim = 0; same_valid_shape && dim < result_view.valid_shape.size(); ++dim) {
            same_valid_shape =
                ir::structural_equal(actual_view.valid_shape[dim], result_view.valid_shape[dim]);
          }
          if (!same_valid_shape || actual_view.pad != result_view.pad) {
            CHECK_SPAN(!tile_type->dtype_.IsPackedFp4(), return_var->span_)
                << "FP4E2M1X2 if-result metadata aliases require PYPTO or DSA_RP addressed memory; "
                   "PTOAS does not support packed-FP4 tile treshape";
            // Physical storage reuse does not transfer logical validity/padding.
            // Keep r's metadata independent from the canonical allocation and
            // every older alias, even when both dynamic types print as '?'.
            ret_name = EnsureTileView(
                ret_name, GetSSATileBufType(ret_name), tile_type, "if_result_view",
                std::make_pair(logical_fields.valid_row_ssa, logical_fields.valid_col_ssa), true);
            restore_valid_shape = true;
          }
        }
        BindVarToMlir(return_var, ret_name);
        if (restore_valid_shape) {
          deferred_tile_valid_shapes.emplace_back(ret_name, std::move(logical_fields));
        }
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

    for (const auto& [tile_buf, fields] : deferred_tile_valid_shapes) {
      INTERNAL_CHECK_SPAN(!fields.valid_row_ssa.empty() && !fields.valid_col_ssa.empty(), op->span_)
          << "Internal error: dynamic IfStmt tile return_var requires both valid_shape operands";
      Emit("pto.set_validshape " + tile_buf + ", " + fields.valid_row_ssa + ", " + fields.valid_col_ssa +
           " : " + fields.type_str);
    }

    if (!scf_return_names.empty()) {
      EmitStructural(JoinCommaSep(scf_return_names) + " = scf.if " + condition + " -> (" +
                     JoinCommaSep(scf_return_types) + ") {");
    } else {
      EmitStructural("scf.if " + condition + " {");
    }
    IncreaseIndent();

    // For in-place return vars (ArrayType and TensorType, both kept out of
    // scf.if results), capture the backing SSA that the branches yield so it can
    // be bound to the merged return var after both branches emit. Both branches
    // yield the same SSA because every array.update_element / pl.assemble
    // aliases the one backing array / tensor.
    std::vector<std::string> inplace_return_ssa(op->return_vars_.size());
    std::vector<std::string> inplace_return_comm_ctx(op->return_vars_.size());

    auto emit_branch = [&](const StmtPtr& body, const char* branch_name) {
      const YieldStmtPtr yield = FindBranchYield(body);
      // Fix #1956: under memory_planner=PTOAS, MemoryReuse (which would alias the
      // branch yields onto the phi return_var's canonical MemRef via
      // YieldFixupMutator) is skipped. Without it, each branch's tile producer
      // writes its own handle and the post-if read of the phi handle reads a
      // buffer no branch wrote. Re-bind each tile yield-source var to the phi
      // return_var's (head-declared) handle so this branch's producer writes it.
      // Under PYPTO (emit_tile_addr_) the IR-level aliasing already holds, so
      // leave the bindings untouched.
      if (!emit_tile_addr_ && yield) {
        for (size_t i = 0; i < op->return_vars_.size(); ++i) {
          if (i >= yield->value_.size()) continue;
          if (returns_via_scf[i]) continue;
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
      fs_.yield_buffer.clear();
      VisitStmt(body);
      auto branch_yields = fs_.yield_buffer;
      CHECK(branch_yields.size() == op->return_vars_.size())
          << "IfStmt " << branch_name << "-branch yield count (" << branch_yields.size()
          << ") must match return_vars (" << op->return_vars_.size() << ")";

      std::vector<std::string> scf_yields;
      scf_yields.reserve(scf_return_types.size());
      for (size_t i = 0; i < op->return_vars_.size(); ++i) {
        if (returns_via_scf[i]) {
          INTERNAL_CHECK_SPAN(yield && i < yield->value_.size(), op->span_)
              << "Internal error: SCF result has no corresponding branch yield";
          scf_yields.push_back(PrepareRegionYield(yield->value_[i], branch_yields[i], op->return_vars_[i]));
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
        } else if (!emit_tile_addr_ && As<TileType>(op->return_vars_[i]->GetType())) {
          // Tile phi under PTOAS (fix #1956). A branch-local producer was already
          // re-pointed at the phi handle above, so its yield resolves to the phi
          // handle and needs nothing. Any other yield — an outer tile threaded
          // through the branch, or one the re-point conservatively skipped — has
          // NOT written the phi buffer, so copy it in: without MemoryReuse's
          // YieldFixupMutator (skipped under PTOAS) the post-if read would
          // otherwise see an uninitialised phi buffer.
          auto phi_it = fs_.var_to_mlir.find(GetVarKey(op->return_vars_[i]));
          auto full_span_origin = [&](const std::string& name) -> const std::string& {
            const auto view = fs_.tile_view_names.find(name);
            return view == fs_.tile_view_names.end() ? name : view->second;
          };
          // Different metadata handles over one full-span reshape origin
          // already contain the same bytes. Subviews retain distinct origins,
          // so a window of a larger allocation still needs its real transfer.
          if (phi_it != fs_.var_to_mlir.end() && !branch_yields[i].empty() &&
              full_span_origin(branch_yields[i]) != full_span_origin(phi_it->second)) {
            const std::string& phi = phi_it->second;
            // Annotate each operand with the type its own SSA value was defined
            // with. They can differ: a `pto.treshape` view carries static valid
            // dims (the op takes no valid operands) while an `alloc_tile` handle
            // is always dynamic-valid.
            auto type_of = [&](const std::string& ssa) -> std::string {
              auto it = fs_.ssa_to_tile_buf_type.find(ssa);
              return it != fs_.ssa_to_tile_buf_type.end() ? it->second : std::string{};
            };
            std::string src_ty = type_of(branch_yields[i]);
            std::string dst_ty = type_of(phi);
            if (src_ty.empty()) src_ty = dst_ty;
            if (dst_ty.empty()) dst_ty = src_ty;
            std::ostringstream mov;
            mov << "pto.tmov ins(" << branch_yields[i];
            if (!src_ty.empty()) mov << " : " << src_ty;
            mov << ") outs(" << phi;
            if (!dst_ty.empty()) mov << " : " << dst_ty;
            mov << ")";
            Emit(mov.str());
          }
        }
        // Tile return_vars under PYPTO: MemoryReuse ensures branch yields share the
        // return_var's canonical MemRef (same physical address), so no codegen-level
        // tmov is needed — the IR-level tile.move (from YieldFixupMutator) handles it.
      }

      if (!scf_return_types.empty()) {
        Emit("scf.yield " + JoinCommaSep(scf_yields) + " : " + JoinCommaSep(scf_return_types));
      }
      INTERNAL_CHECK_SPAN(scf_yields.size() == scf_return_types.size(), op->span_)
          << "Internal error: IfStmt " << branch_name << "-branch SCF yield count (" << scf_yields.size()
          << ") must match SCF results (" << scf_return_types.size() << ")";
      fs_.yield_buffer.clear();
    };

    emit_branch(op->then_body_, "then");
    DecreaseIndent();

    EmitStructural("} else {");
    IncreaseIndent();
    const auto& else_body = op->else_body_;
    INTERNAL_CHECK_SPAN(else_body.has_value(), op->span_)
        << "Internal error: IfStmt with return_vars has no else_body";
    emit_branch(*else_body, "else");
    DecreaseIndent();
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

  // Scalars and selected buffer references use real loop-carried SSA values.
  // Concrete Tile storage and normalized GM tensors retain the in-place path.
  // Carrying a selected reference also makes a zero-trip loop return its init
  // handle, without copying payloads or choosing one candidate allocation.
  std::vector<bool> returns_via_scf(op->iter_args_.size(), false);
  bool has_scf_iter_args = false;
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (IsRegionSSAValue(op->iter_args_[i]->GetType())) {
      returns_via_scf[i] = true;
      has_scf_iter_args = true;
    }
  }

  // Map only concrete storage aliases directly to their init handles.
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (returns_via_scf[i]) continue;

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

  if (!has_scf_iter_args) {
    // Simple scf.for (all carried storage is statically shared).
    EmitStructural("scf.for " + loop_var_name + " = " + start + " to " + stop + " step " + step + " {");
    IncreaseIndent();

    fs_.yield_buffer.clear();
    VisitStmt(op->body_);
    fs_.yield_buffer.clear();

    DecreaseIndent();
    EmitStructural("}");
  } else {
    // scf.for carrying scalar values and selected storage references.
    std::vector<std::string> init_values;
    std::vector<std::string> iter_arg_names;
    std::vector<std::string> iter_arg_types;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!returns_via_scf[i]) continue;

      const auto& iter_arg = op->iter_args_[i];

      VisitExpr(iter_arg->initValue_);
      init_values.push_back(PrepareRegionYield(iter_arg->initValue_, fs_.current_expr_value, iter_arg));
      fs_.current_expr_value = "";

      std::string iter_name = NewNamedTemp(iter_arg->name_hint_);
      BindRegionValue(iter_arg, iter_name);
      iter_arg_names.push_back(iter_name);

      iter_arg_types.push_back(GetRegionValueTypeString(iter_arg->GetType(), iter_arg->span_));
    }

    // Register every real SCF result, including selected references.
    std::vector<std::string> return_var_names;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!returns_via_scf[i]) continue;
      std::string ret_name = NewNamedTemp(op->return_vars_[i]->name_hint_);
      BindRegionValue(op->return_vars_[i], ret_name);
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

    // Keep the original region positions for scalars and selected references.
    std::vector<std::string> scf_yields;
    const auto yield = FindBranchYield(op->body_);
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (returns_via_scf[i] && i < fs_.yield_buffer.size()) {
        INTERNAL_CHECK_SPAN(yield && i < yield->value_.size(), op->span_)
            << "Internal error: selected loop reference has no backedge value";
        scf_yields.push_back(PrepareRegionYield(yield->value_[i], fs_.yield_buffer[i], op->iter_args_[i]));
      }
    }

    // Emit scf.yield from filtered yield values
    if (!scf_yields.empty()) {
      std::ostringstream yield_oss;
      yield_oss << "scf.yield ";
      for (size_t i = 0; i < scf_yields.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << scf_yields[i];
      }
      yield_oss << " : ";
      for (size_t i = 0; i < iter_arg_types.size(); ++i) {
        if (i > 0) yield_oss << ", ";
        yield_oss << iter_arg_types[i];
      }
      Emit(yield_oss.str());
    }
    CHECK(scf_yields.size() == iter_arg_types.size())
        << "ForStmt SCF yield count (" << scf_yields.size() << ") must match SCF iter_args ("
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

  // Scalars and selected buffer references use real loop-carried SSA values.
  // Concrete Tile storage and normalized GM tensors retain the in-place path.
  // Carrying a selected reference also makes a zero-trip loop return its init
  // handle, without copying payloads or choosing one candidate allocation.
  std::vector<bool> returns_via_scf(op->iter_args_.size(), false);
  bool has_scf_iter_args = false;
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (IsRegionSSAValue(op->iter_args_[i]->GetType())) {
      returns_via_scf[i] = true;
      has_scf_iter_args = true;
    }
  }

  // Map only concrete storage aliases directly to their init handles.
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    if (returns_via_scf[i]) continue;

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

  if (!has_scf_iter_args) {
    // Simple scf.while (all carried storage is statically shared).
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
    // scf.while carrying scalar values and selected storage references.
    std::vector<std::string> init_values;
    std::vector<std::string> before_arg_names;
    std::vector<std::string> after_arg_names;
    std::vector<std::string> iter_arg_types;

    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (!returns_via_scf[i]) continue;

      const auto& iter_arg = op->iter_args_[i];

      VisitExpr(iter_arg->initValue_);
      init_values.push_back(PrepareRegionYield(iter_arg->initValue_, fs_.current_expr_value, iter_arg));
      fs_.current_expr_value = "";

      before_arg_names.push_back(NewTemp());
      after_arg_names.push_back(NewTemp());

      iter_arg_types.push_back(GetRegionValueTypeString(iter_arg->GetType(), iter_arg->span_));
    }

    // Register every real SCF result, including selected references.
    std::vector<std::string> return_var_names;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      if (!returns_via_scf[i]) continue;

      std::string ret_name = NewTemp();
      BindRegionValue(op->return_vars_[i], ret_name);
      return_var_names.push_back(ret_name);
    }

    // Each while region binds the same IR arguments to its own block arguments.
    auto register_region_iter_args = [&](const std::vector<std::string>& ssa_names) {
      size_t region_idx = 0;
      for (size_t i = 0; i < op->iter_args_.size(); ++i) {
        if (!returns_via_scf[i]) continue;
        BindRegionValue(op->iter_args_[i], ssa_names[region_idx]);
        region_idx++;
      }
    };

    std::string types_str = "(" + JoinCommaSep(iter_arg_types) + ")";

    // Emit: %ret0, %ret1 = scf.while (%before0 = %init0, ...) : (types) -> (types) {
    EmitStructural(JoinCommaSep(return_var_names) + " = scf.while (" +
                   JoinPairs(before_arg_names, " = ", init_values) + ") : " + types_str + " -> " + types_str +
                   " {");
    IncreaseIndent();

    // Before region: register before-region args, evaluate condition
    register_region_iter_args(before_arg_names);

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
    register_region_iter_args(after_arg_names);

    // Visit body
    fs_.yield_buffer.clear();
    VisitStmt(op->body_);

    // Keep the original region positions for scalars and selected references.
    std::vector<std::string> scf_yields;
    const auto yield = FindBranchYield(op->body_);
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (returns_via_scf[i] && i < fs_.yield_buffer.size()) {
        INTERNAL_CHECK_SPAN(yield && i < yield->value_.size(), op->span_)
            << "Internal error: selected loop reference has no backedge value";
        scf_yields.push_back(PrepareRegionYield(yield->value_[i], fs_.yield_buffer[i], op->iter_args_[i]));
      }
    }

    // Emit scf.yield from filtered yield values
    if (!scf_yields.empty()) {
      Emit("scf.yield " + JoinCommaSep(scf_yields) + " : " + JoinCommaSep(iter_arg_types));
    }
    CHECK(scf_yields.size() == iter_arg_types.size())
        << "WhileStmt SCF yield count (" << scf_yields.size() << ") must match SCF iter_args ("
        << iter_arg_types.size() << ")";
    fs_.yield_buffer.clear();

    DecreaseIndent();
    EmitStructural("}");
  }
}

}  // namespace codegen
}  // namespace pypto
