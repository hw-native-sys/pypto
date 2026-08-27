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

#include "pypto/codegen/pto/pto_codegen.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/codegen/distributed/comm_layout.h"
#include "pypto/codegen/gm_pipe_layout.h"
#include "pypto/codegen/pto/pto_type_utils.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/tile_buf_signature.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace codegen {

using ir::As;
using ir::AssignStmtPtr;
using ir::BinaryExprPtr;
using ir::CallPtr;
using ir::CommCtxType;
using ir::EvalStmtPtr;
using ir::ExprPtr;
using ir::ForStmtPtr;
using ir::FunctionPtr;
using ir::IfStmtPtr;
using ir::MemRefPtr;
using ir::ProgramPtr;
using ir::ScalarType;
using ir::StmtPtr;
using ir::TensorType;
using ir::TileType;
using ir::VarPtr;
using ir::WhileStmtPtr;
using ir::YieldStmtPtr;

namespace transform_utils = ir::transform_utils;

namespace {

// Implemented by the generated kernel wrapper rather than by PTO MLIR. Keep
// this name reserved in modules that use deferred completion so a user kernel
// cannot silently collide with the private adapter declaration.
constexpr const char* kDeferredCompletionAdapterName = "pypto_register_counter_completion";

// Escape a source path for an MLIR `"..."` string literal. A path is an
// arbitrary OS byte string — on POSIX every byte except `/` and NUL is legal, so
// a quote, a backslash, or a raw control character in one must not be able to
// break the module. MLIR's lexer accepts `\\`, `\"`, `\n`, `\t` and `\xx` hex
// escapes, and rejects an unescaped control character inside a literal, so
// anything outside printable ASCII is emitted as a hex escape.
std::string EscapeMlirString(const std::string& str) {
  static constexpr char kHexDigits[] = "0123456789ABCDEF";
  std::string escaped;
  escaped.reserve(str.size());
  for (char c : str) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '\\' || c == '"') {
      escaped.push_back('\\');
      escaped.push_back(c);
    } else if (c == '\n') {
      escaped += "\\n";
    } else if (c == '\t') {
      escaped += "\\t";
    } else if (byte < 0x20 || byte == 0x7F) {
      escaped += "\\";
      escaped.push_back(kHexDigits[byte >> 4]);
      escaped.push_back(kHexDigits[byte & 0x0F]);
    } else {
      escaped.push_back(c);
    }
  }
  return escaped;
}

// True when `inner` is a source range nested inside `outer` — the invariant a
// sub-expression's span satisfies with respect to its enclosing statement's span.
//
// This is what rejects a Call span that a pass overwrote with a coarser one.
// ConvertTensorToTileOps rebuilds every tile op it synthesizes with the enclosing
// *function*'s span (convert_tensor_to_tile_ops_pass.cpp), which begins before the
// statement and therefore fails containment — so the statement's own span, which
// that rewrite preserved, is kept instead.
bool SpanContains(const ir::Span* outer, const ir::Span& inner) {
  if (!inner.is_valid() || inner.filename_.empty()) return false;
  // No usable enclosing span means nothing contradicts the inner one.
  if (outer == nullptr || !outer->is_valid() || outer->filename_.empty()) return true;
  if (outer->filename_ != inner.filename_) return false;
  // An unknown end line (-1) degenerates to a single-line range at the start.
  const int outer_end = outer->end_line_ > 0 ? outer->end_line_ : outer->begin_line_;
  const int inner_end = inner.end_line_ > 0 ? inner.end_line_ : inner.begin_line_;
  // Both ends must sit inside the statement. Checking only the start would accept
  // a rebuilt span that begins within the statement but runs past its last line,
  // which is exactly the untrustworthy case this predicate exists to reject.
  return inner.begin_line_ >= outer->begin_line_ && inner_end <= outer_end;
}

// Full-MemRef-identity key used by PTOAS memory-planner codegen to decide when
// two tile variables denote the *same* buffer (and must share one tile_buf
// handle so the op writes in place). Same base + byte_offset + size = same
// buffer (loop-carried accumulator, in-place op result). A view shares the
// base but differs in offset and/or size, so it gets a distinct key.
std::string MemRefIdentityKey(const ir::MemRefPtr& memref) {
  std::ostringstream key;
  key << static_cast<const void*>(memref->base_.get()) << '|';
  if (auto off = As<ir::ConstInt>(memref->byte_offset_)) {
    key << "off" << off->value_;
  } else {
    key << "off@" << static_cast<const void*>(memref->byte_offset_.get());
  }
  key << "|sz" << memref->size_;
  return key.str();
}

// Base Ptrs of tile phis — an `IfStmt` / `ForStmt` / `WhileStmt` return var, or a
// loop-carried iter_arg. Under the PTOAS planner those take a handle declared in
// the function head (see pto_control_flow_codegen.cpp), which a per-use
// `pto.multi_tile_get` cannot supply: a runtime slot index is not in scope there.
// An allocation whose slots feed one is therefore rejected, not degraded.
class TilePhiBaseCollector : public ir::IRVisitor {
 public:
  std::set<const ir::Var*> bases;

  void VisitStmt_(const ir::IfStmtPtr& op) override {
    Record(op->return_vars_);
    ir::IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ir::ForStmtPtr& op) override {
    Record(op->return_vars_);
    Record(op->iter_args_);
    ir::IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ir::WhileStmtPtr& op) override {
    Record(op->return_vars_);
    Record(op->iter_args_);
    ir::IRVisitor::VisitStmt_(op);
  }

 private:
  template <typename VarLikePtr>
  void Record(const std::vector<VarLikePtr>& vars) {
    for (const auto& var : vars) {
      if (!var) continue;
      auto tile_type = ir::GetTileTypeWithMemRef(var->GetType());
      if (!tile_type) continue;
      bases.insert(ir::GetDefinedMemRef(tile_type)->base_.get());
    }
  }
};

// Allocations with two or more slots selected inside one loop body.
//
// ptoas derives the per-slot WAR guard from the slot expression, but only for the
// FIRST `multi_tile_get` of a region in an iteration: given two co-live slots it
// emits the dynamic `wait_flag`/`set_flag` pair for one and leaves the other load
// unguarded, so the next iteration's write into that slot races the current
// iteration's read of it. Measured on ptoas 0.54 (`--enable-insert-sync`,
// `--pto-level=level2`, a3) — the kernel is silently wrong on device, not slow.
// Filed as hw-native-sys/PTOAS#1118.
//
// The ping-pong the region form exists for takes ONE slot per iteration, and that
// shape is guarded correctly. So the co-live shape is rejected rather than
// miscompiled, and the author is pointed at the PyPTO planner, whose baked
// addresses and PyPTO-emitted sync handle it. Straight-line code is untouched:
// with no loop there is no cross-iteration reuse to guard.
class CoLiveSlotCollector : public ir::IRVisitor {
 public:
  std::set<const ir::Var*> bases;  ///< Allocations with >= 2 slots live in one loop body

  void VisitStmt_(const ir::ForStmtPtr& op) override { VisitLoop(op); }
  void VisitStmt_(const ir::WhileStmtPtr& op) override { VisitLoop(op); }

  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    if (loop_depth_ > 0) {
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        const auto memref = ir::GetDefinedMemRef(tile_type);
        if (memref->slot_count_ > 1 && memref->slot_index_.has_value() && *memref->slot_index_) {
          // Second slot-selecting tile on this allocation in the same loop body.
          if (!per_loop_seen_.insert(memref->base_.get()).second) bases.insert(memref->base_.get());
        }
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

 private:
  template <typename LoopPtr>
  void VisitLoop(const LoopPtr& op) {
    // Each loop body counts on its own: two slots in *sibling* loops are never
    // live together, and a nested loop's own body is the iteration that matters.
    auto saved = std::move(per_loop_seen_);
    per_loop_seen_.clear();
    ++loop_depth_;
    ir::IRVisitor::VisitStmt_(op);
    --loop_depth_;
    per_loop_seen_ = std::move(saved);
  }

  int loop_depth_ = 0;
  std::set<const ir::Var*> per_loop_seen_;
};

// The (valid_row, valid_col) extents a tile declares, when both are compile-time.
//
// Same source of truth as ComputeAllocTileFields — the author's valid_shape when
// there is one, the physical shape otherwise — but as values rather than emitted
// SSA, because a multi-buffer region states ONE valid extent for all its slots and
// therefore has to compare them before declaring it. The tile_buf type string
// cannot stand in for this comparison: it deliberately renders `v_row=?, v_col=?`,
// so two slots differing only in valid_shape print identically.
std::optional<std::pair<int64_t, int64_t>> StaticValidExtents(
    const std::shared_ptr<const ir::TileType>& tile_type) {
  const std::vector<ir::ExprPtr>* dims = nullptr;
  if (const auto& tile_view = tile_type->tile_view_;
      tile_view.has_value() && !tile_view->valid_shape.empty()) {
    dims = &tile_view->valid_shape;
  } else if (!tile_type->shape_.empty()) {
    dims = &tile_type->shape_;
  }
  if (dims == nullptr || dims->empty()) return std::nullopt;

  std::vector<int64_t> extents;
  for (size_t i = 0; i < dims->size() && i < 2; ++i) {
    auto const_dim = As<ir::ConstInt>((*dims)[i]);
    if (!const_dim) return std::nullopt;
    extents.push_back(const_dim->value_);
  }
  // Match ExtractTileTypeInfo: a 1-D tile is rows=1, cols=shape[0].
  if (dims->size() == 1) return std::make_pair(static_cast<int64_t>(1), extents[0]);
  return std::make_pair(extents[0], extents[1]);
}

// Memory spaces ptoas accepts for a `!pto.multi_tile_buf` slot. The multi-buffer
// design ships with local vec / mat support; `acc` compiles as well (verified
// against ptoas 0.54). A slotted allocation in any other space — gm above all —
// is rejected here rather than left to fail in the ptoas verifier.
bool IsMultiBufferMemorySpace(std::optional<ir::MemorySpace> space) {
  return space.has_value() &&
         (*space == ir::MemorySpace::Vec || *space == ir::MemorySpace::Mat || *space == ir::MemorySpace::Acc);
}

bool IsSameDimExpr(const ExprPtr& lhs, const ExprPtr& rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto lhs_const = As<ir::ConstInt>(lhs);
  auto rhs_const = As<ir::ConstInt>(rhs);
  return lhs_const && rhs_const && lhs_const->value_ == rhs_const->value_;
}

// Extract the (row, col) valid_shape expressions from a TileType's tile_view.
// Returns nullptr for a dimension when it is missing or is a ConstInt (static).
// Non-ConstInt expressions (Var, Call, BinaryOp, ...) flow through as dynamic
// and must be lowered to MLIR via GetExprAsCode at the call site.
std::pair<ExprPtr, ExprPtr> GetTileValidShapeExprs(const std::shared_ptr<const ir::TileType>& tile_type) {
  ExprPtr valid_row_expr;
  ExprPtr valid_col_expr;
  if (!tile_type) {
    return {valid_row_expr, valid_col_expr};
  }

  const auto& optional_tile_view = tile_type->tile_view_;
  if (!optional_tile_view) {
    return {valid_row_expr, valid_col_expr};
  }

  const auto& tile_view = *optional_tile_view;
  if (tile_view.valid_shape.size() >= 1 && tile_view.valid_shape[0] &&
      !As<ir::ConstInt>(tile_view.valid_shape[0])) {
    valid_row_expr = tile_view.valid_shape[0];
  }
  if (tile_view.valid_shape.size() >= 2 && tile_view.valid_shape[1] &&
      !As<ir::ConstInt>(tile_view.valid_shape[1])) {
    valid_col_expr = tile_view.valid_shape[1];
  }
  return {valid_row_expr, valid_col_expr};
}

bool HasDynamicTileValidShape(const std::shared_ptr<const ir::TileType>& tile_type) {
  auto [valid_row_expr, valid_col_expr] = GetTileValidShapeExprs(tile_type);
  return valid_row_expr || valid_col_expr;
}

// Collect Vars referenced by a shape expression in first-seen order (for trailing
// %argN: index in MLIR). Single source of truth: both the in-translation-unit
// caller ``CollectTensorShapeDynVars`` (driving the trailing index params on the
// emitted ``func.func`` signature) and the Python kernel-wrapper codegen
// (recovering a Var from runtime ``tensor->shapes[]`` inside
// ``_generate_arg_unpacking`` in python/pypto/backend/pto_backend.py) go through
// this walker. The Python side reaches it via the public ``CollectVarsFromShapeExpr``
// wrapper exposed through the codegen nanobind binding
// ``collect_vars_from_shape_expr``. There is no Python-side mirror to keep in sync.
//
// Dedup key: raw ``Var*`` is sound here because the IR holds the canonical
// shared_ptr graph (each Var has exactly one address).
//
// Unknown node kinds fail loudly: silently skipping them would recreate the
// very bug this function exists to fix (lost dynamic-dim params in the kernel
// signature) the next time a new Expr subclass is introduced in shapes.
void CollectVarsFromShapeExprImpl(const ExprPtr& expr, std::set<const ir::Var*>& seen,
                                  std::vector<VarPtr>& out) {
  if (!expr) {
    return;
  }
  if (auto var = As<ir::Var>(expr)) {
    if (seen.insert(var.get()).second) {
      out.push_back(var);
    }
    return;
  }
  if (auto binary = As<ir::BinaryExpr>(expr)) {
    CollectVarsFromShapeExprImpl(binary->left_, seen, out);
    CollectVarsFromShapeExprImpl(binary->right_, seen, out);
    return;
  }
  if (auto unary = As<ir::UnaryExpr>(expr)) {
    CollectVarsFromShapeExprImpl(unary->operand_, seen, out);
    return;
  }
  if (auto call = As<ir::Call>(expr)) {
    for (const auto& arg : call->args_) {
      CollectVarsFromShapeExprImpl(arg, seen, out);
    }
    return;
  }
  if (auto tget = As<ir::TupleGetItemExpr>(expr)) {
    CollectVarsFromShapeExprImpl(tget->tuple_, seen, out);
    return;
  }
  if (As<ir::ConstInt>(expr) || As<ir::ConstFloat>(expr) || As<ir::ConstBool>(expr)) {
    return;
  }
  INTERNAL_UNREACHABLE_SPAN(expr->span_) << "CollectVarsFromShapeExpr: unsupported shape expression node";
}

// Collect tensor-shape dyn Vars across a function's tensor params.
// Used both to reserve %argN names upfront (so NewNamedTemp does not collide)
// and to emit the trailing index params on the MLIR func.func signature.
std::vector<VarPtr> CollectTensorShapeDynVars(const FunctionPtr& func) {
  std::vector<VarPtr> dyn_vars;
  std::set<const ir::Var*> seen;
  for (const auto& param : func->params_) {
    if (auto tensor_type = ir::AsTensorTypeLike(param->GetType())) {
      for (const auto& dim : tensor_type->shape_) {
        CollectVarsFromShapeExprImpl(dim, seen, dyn_vars);
      }
    }
  }
  return dyn_vars;
}

// In-place DPS ops that write into input 0 rather than a freshly-allocated
// result tile:
//   * scatter family (`set_output_reuses_input(0)`): a tscatter into a fresh
//     uninitialized tile would lose the rows it does not write;
//   * `tile.assemble` (`set_output_memory_inherit_input()`): the result is the
//     target with one window overwritten — written in place so the out-of-window
//     data is preserved (and the Acc->Mat pto.tmov stays a clean converting move,
//     not an unsupported Mat->Mat preservation copy);
//   * `tile.tget_scale_addr` (`set_output_reuses_input(0)`): rebinds the scale
//     tile address in place (ISA GetScaleAddr); outs() must alias dst_scale.
// The aliasing is gated below on the result and input actually sharing a base
// memref, so it only triggers when memory reuse merged them in place.
bool IsInPlaceInput0DpsOp(const ir::OpPtr& op) {
  return ir::IsOp(op, "tile.scatter") || ir::IsOp(op, "tile.scatter_mask") || ir::IsOp(op, "tile.assemble") ||
         ir::IsOp(op, "tile.tget_scale_addr");
}

bool ShareOneMemRefWindow(const std::shared_ptr<const TileType>& lhs,
                          const std::shared_ptr<const TileType>& rhs) {
  auto lhs_memref = ir::GetDefinedMemRef(lhs);
  auto rhs_memref = ir::GetDefinedMemRef(rhs);
  if (!lhs_memref || !rhs_memref) return false;
  if (lhs_memref->base_.get() != rhs_memref->base_.get()) return false;
  // Same base is not enough: two *different* windows of one allocation share it,
  // and aliasing those would silently redirect the write. Require the same byte
  // offset and extent too. The offset is an expression (a slice of a loop-carried
  // tile carries a loop-dependent one), so compare it structurally.
  if (lhs_memref->size_ != rhs_memref->size_) return false;
  const auto& lhs_offset = lhs_memref->byte_offset_;
  const auto& rhs_offset = rhs_memref->byte_offset_;
  if (!lhs_offset || !rhs_offset) return lhs_offset == rhs_offset;
  return ir::structural_equal(lhs_offset, rhs_offset);
}

// Whether `stmt`'s result Var should be bound to the SSA of the operand the call
// writes in place, instead of getting its own `pto.alloc_tile`.
//
// Two arms, deliberately kept apart:
//
//   * `IsInPlaceInput0DpsOp` — ops whose in-place-ness is a codegen-lowering fact.
//     Gated on a shared base memref only, which is the long-standing behaviour.
//
//   * the registry (`set_output_reuses_input`) — the declared, op-level truth.
//     This is what lets `tile.matmul_acc` accumulate directly into its
//     accumulator operand: when that operand is a `tile.slice` of a larger Acc
//     tile its SSA is a `pto.subview`, so the MAD writes straight into the
//     destination window instead of into a private L0C buffer that would then
//     need an acc->acc `tmov` the ISA cannot express.
//
//     This arm additionally requires an identical `TileBufSignature`. One MLIR
//     SSA value has exactly one type, so aliasing two vars whose tile configs
//     differ would silently drop one of them — `tile.fillpad_inplace` reuses its
//     input's buffer but its result carries `pad`, which the input does not.
//
// A declared index naming a non-tile argument (`tile.store` / `tile.write`
// declare index 2, a TensorType) drops out: GetTileTypeWithMemRef returns null.
bool ShouldAliasResultToInPlaceInput(const AssignStmtPtr& stmt) {
  auto call = As<ir::Call>(stmt->value_);
  if (!call || !call->op_) return false;

  auto result_tile_type = ir::GetTileTypeWithMemRef(stmt->var_->GetType());
  if (!result_tile_type) return false;

  auto input_tile_type_at = [&](size_t index) -> std::shared_ptr<const TileType> {
    if (index >= call->args_.size()) return nullptr;
    return ir::GetTileTypeWithMemRef(call->args_[index]->GetType());
  };

  // Legacy arm: shared base memref only, exactly as before.
  if (IsInPlaceInput0DpsOp(call->op_)) {
    auto input_tile_type = input_tile_type_at(0);
    if (!input_tile_type) return false;
    auto result_memref = ir::GetDefinedMemRef(result_tile_type);
    auto input_memref = ir::GetDefinedMemRef(input_tile_type);
    return result_memref && input_memref && result_memref->base_.get() == input_memref->base_.get();
  }

  auto& registry = ir::OpRegistry::GetInstance();
  if (!registry.IsRegistered(call->op_->name_)) return false;
  auto declared = registry.GetEntry(call->op_->name_).GetOutputReusesInputArg();
  if (!declared.has_value()) return false;
  auto input_tile_type = input_tile_type_at(*declared);
  if (!input_tile_type) return false;
  return ShareOneMemRefWindow(result_tile_type, input_tile_type) &&
         ir::TileBufSignature::FromTileType(*result_tile_type) ==
             ir::TileBufSignature::FromTileType(*input_tile_type);
}

// `array.update_element` is SSA-functional in the IR (returns a fresh
// ArrayType value), but on-core arrays lower to a single `pto.declare_local_array`
// that is mutated in place. Aliasing the result Var to the input array's SSA
// name lets the emitted `pto.local_array_set` write the same storage — no copy.
bool ShouldAliasArrayUpdateResultToInput(const AssignStmtPtr& stmt) {
  auto call = As<ir::Call>(stmt->value_);
  return call && ir::IsOp(call, "array.update_element") && !call->args_.empty() &&
         As<ir::ArrayType>(stmt->var_->GetType());
}

const auto& FlattenBody = transform_utils::FlattenToStmts;

// Collects `<var> = TupleGetItemExpr(tuple_var, i)` AssignStmts. IRVisitor
// auto-recurses through all statement kinds (Seq/For/If/While/Scope/Inline/...),
// so this stays correct regardless of where the tuple-returning call is nested.
class TupleConsumerCollector : public ir::IRVisitor {
 public:
  explicit TupleConsumerCollector(const ir::Var* tuple_var, size_t arity)
      : tuple_var_(tuple_var), elements_(arity, nullptr) {}

  [[nodiscard]] const std::vector<ir::VarPtr>& elements() const { return elements_; }

 protected:
  void VisitStmt_(const ir::AssignStmtPtr& op) override {
    if (auto tge = As<ir::TupleGetItemExpr>(op->value_)) {
      if (auto base = As<ir::Var>(tge->tuple_)) {
        if (base.get() == tuple_var_ && tge->index_ >= 0 &&
            static_cast<size_t>(tge->index_) < elements_.size()) {
          elements_[tge->index_] = op->var_;
        }
      }
    }
    ir::IRVisitor::VisitStmt_(op);
  }

 private:
  const ir::Var* tuple_var_;
  std::vector<ir::VarPtr> elements_;
};

}  // namespace

std::vector<VarPtr> CollectVarsFromShapeExpr(const ExprPtr& expr) {
  std::vector<VarPtr> out;
  std::set<const ir::Var*> seen;
  CollectVarsFromShapeExprImpl(expr, seen, out);
  return out;
}

// Visitor to collect all MemRef objects from TileType variables. Also
// piggy-backs synthetic-parameter detection (prefetch.make_context and the
// SPMD identity ops) on the same body walk so callers do not need a separate
// IR traversal.
class MemRefCollectorVisitor : public ir::IRVisitor {
 public:
  MemRefCollectorVisitor() = default;

  [[nodiscard]] const std::vector<MemRefPtr>& GetMemRefs() const { return memrefs_; }
  [[nodiscard]] const std::map<const ir::Var*, std::shared_ptr<const TileType>>& GetMemRefTileTypes() const {
    return memref_tile_types_;
  }

  /// Returns true when the visited body invokes prefetch.make_context. Drives
  /// PTOCodegen's decision to append the hidden runtime-owned SDMA workspace
  /// pointer to the emitted func.func signature.
  [[nodiscard]] bool UsesSdmaWorkspace() const { return uses_sdma_workspace_; }

  /// Returns true when the visited body registers deferred task completion.
  /// Drives the hidden raw dispatch-args pointer shared with the kernel wrapper.
  [[nodiscard]] bool UsesDeferredCompletion() const { return uses_deferred_completion_; }

  /// Returns true when the visited body invokes tile.get_block_idx or
  /// tile.get_block_num. Drives PTOCodegen's decision to append two synthetic
  /// i32 params to the emitted func.func signature; the kernel wrapper
  /// resolves those values from intrinsic.h::get_block_idx(args) /
  /// get_block_num(args) at dispatch time.
  [[nodiscard]] bool UsesSpmdBlockOps() const { return uses_spmd_block_ops_; }

  /// Returns true when the visited body invokes tile.get_subblock_idx. Drives
  /// PTOCodegen's decision to append a synthetic i32 param to the func.func
  /// signature; the kernel wrapper resolves it from
  /// intrinsic.h::get_sub_block_id(args) at dispatch time, rather than reading
  /// the ccec get_subblockid() register.
  [[nodiscard]] bool UsesSubblockOp() const { return uses_subblock_op_; }

  [[nodiscard]] const std::set<const ir::Var*>& GetFFTSWorkspaceVars() const { return ffts_workspace_vars_; }

  void VisitExpr_(const VarPtr& op) override {
    if (iter_arg_ids_.count(op->UniqueId())) return;
    if (auto tile_type = ir::GetTileTypeWithMemRef(op->GetType())) {
      AddMemRefIfUnique(ir::GetDefinedMemRef(tile_type), tile_type);
    }
  }

  void VisitExpr_(const ir::IterArgPtr& op) override {
    iter_arg_ids_.insert(op->UniqueId());
    ir::IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ir::CallPtr& op) override {
    if (op->op_) {
      if (!uses_sdma_workspace_ && ir::IsOp(op, "prefetch.make_context")) {
        uses_sdma_workspace_ = true;
      }
      if (!uses_deferred_completion_ && ir::IsOp(op, "pld.system.defer_wait")) {
        uses_deferred_completion_ = true;
      }
      if (!uses_spmd_block_ops_ &&
          (ir::IsOp(op, "tile.get_block_idx") || ir::IsOp(op, "tile.get_block_num"))) {
        uses_spmd_block_ops_ = true;
      }
      if (!uses_subblock_op_ && ir::IsOp(op, "tile.get_subblock_idx")) {
        uses_subblock_op_ = true;
      }
      if (ir::IsOp(op, "system.set_ffts") && op->args_.size() == 1) {
        if (auto workspace = As<ir::Var>(op->args_[0])) {
          ffts_workspace_vars_.insert(workspace.get());
        }
      }
    }
    ir::IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<MemRefPtr> memrefs_;
  std::set<const ir::Var*> seen_bases_;
  std::map<const ir::Var*, std::shared_ptr<const TileType>> memref_tile_types_;
  std::set<uint64_t> iter_arg_ids_;
  bool uses_sdma_workspace_ = false;
  bool uses_deferred_completion_ = false;
  bool uses_spmd_block_ops_ = false;
  bool uses_subblock_op_ = false;
  std::set<const ir::Var*> ffts_workspace_vars_;

  void AddMemRefIfUnique(const MemRefPtr& memref, const std::shared_ptr<const TileType>& tile_type) {
    const ir::Var* base_ptr = memref->base_.get();
    if (seen_bases_.insert(base_ptr).second) {
      memrefs_.push_back(memref);
      memref_tile_types_[base_ptr] = tile_type;
    } else {
      // Merge TileView properties when multiple tiles share the same allocation:
      // - Keep valid_shape from the original tile (e.g., from load)
      // - Take pad from the new tile if it has a non-null pad (e.g., from fillpad)
      // This ensures fillpad's pad_value is used while preserving the original valid_shape
      auto existing = memref_tile_types_[base_ptr];
      if (const auto& tile_view = tile_type->tile_view_;
          tile_view.has_value() && tile_view->pad != ir::PadValue::null) {
        // Merge: keep valid_shape from existing, take pad from new tile
        ir::TileView merged_view;
        if (const auto& existing_view = existing->tile_view_) {
          merged_view = *existing_view;
        }
        merged_view.pad = tile_view->pad;
        auto merged_tile_type = std::make_shared<TileType>(
            existing->shape_, existing->dtype_, existing->memref_, merged_view, existing->memory_space_);
        memref_tile_types_[base_ptr] = merged_tile_type;
      }
    }
  }
};

// ========================================================================
// Constructors
// ========================================================================

PTOCodegen::PTOCodegen() : backend_(backend::GetBackend()) {
  CHECK(backend_ != nullptr && backend_->GetHandler() != nullptr)
      << "PTOCodegen requires a configured backend that exposes a BackendHandler";
}

PTOCodegen::PTOCodegen(const backend::Backend* backend) : backend_(backend) {
  CHECK(backend != nullptr) << "Backend cannot be null";
  CHECK(backend->GetHandler() != nullptr) << "PTOCodegen requires a backend that exposes a BackendHandler";
}

const backend::BackendHandler* PTOCodegen::GetBackendHandler() const { return backend_->GetHandler(); }

// ========================================================================
// Generate entry and GenerateFunction
// ========================================================================

std::string PTOCodegen::Generate(const ProgramPtr& program, bool emit_tile_addr, bool emit_source_loc) {
  emit_tile_addr_ = emit_tile_addr;
  emit_source_loc_ = emit_source_loc;
  current_span_ = nullptr;
  stream_.str("");
  stream_.clear();
  fs_.constants_section.str("");
  fs_.constants_section.clear();
  fs_.body_section.str("");
  fs_.body_section.clear();
  gm_slot_buffer_offsets_.clear();
  needs_deferred_completion_adapter_ = false;
  PrepareGMSlotBufferLayout(program);

  const std::string target_arch = backend_->GetHandler()->GetPtoTargetArch();
  stream_ << "module attributes {pto.target_arch = \"" << target_arch << "\"} {\n";

  for (const auto& [gvar, func] : program->functions_) {
    INTERNAL_CHECK_SPAN(ir::IsInCoreType(func->func_type_), func->span_)
        << "PTO backend only supports InCore-variant functions (InCore, AIC, AIV), but function '"
        << func->name_ << "' has type " << ir::FunctionTypeToString(func->func_type_);
    GenerateFunction(func);
  }

  if (needs_deferred_completion_adapter_) {
    for (const auto& [gvar, func] : program->functions_) {
      CHECK_SPAN(func->name_ != kDeferredCompletionAdapterName, func->span_)
          << "Function name '" << kDeferredCompletionAdapterName
          << "' is reserved for PyPTO's deferred-completion runtime adapter";
    }
  }

  EmitDeferredCompletionAdapterDeclaration();

  stream_ << "}\n";
  return stream_.str();
}

void PTOCodegen::PrepareGMSlotBufferLayout(const ProgramPtr& program) {
  std::map<std::pair<int, int>, int> slot_size_by_pipe;
  std::map<std::pair<int, int>, int> slot_count_by_pipe;

  std::function<void(const std::vector<StmtPtr>&)> scan_stmts;
  scan_stmts = [&](const std::vector<StmtPtr>& stmts) {
    for (const auto& stmt : stmts) {
      auto call = transform_utils::GetCallFromStmt(stmt);
      if (ir::op_predicates::IsInitializePipe(call)) {
        const int pipe_id = call->GetKwarg<int>("id", 0);
        const int dir_mask = call->GetKwarg<int>("dir_mask", 0);
        const int slot_size = call->GetKwarg<int>("slot_size", 0);
        const int slot_num = call->GetKwarg<int>("slot_num", 0);
        if (dir_mask > 0 && slot_size > 0) {
          const auto key = std::make_pair(pipe_id, dir_mask);
          const int slot_count = gm_pipe::EffectiveSlotCount(dir_mask, slot_num);
          auto [nit, ninserted] = slot_count_by_pipe.emplace(key, slot_count);
          CHECK(ninserted || nit->second == slot_count)
              << "initialize_pipe for frontend pipe id " << pipe_id << " and dir_mask " << dir_mask
              << " uses inconsistent slot counts: " << nit->second << " and " << slot_count;
          auto [it, inserted] = slot_size_by_pipe.emplace(key, slot_size);
          CHECK(inserted || it->second == slot_size)
              << "initialize_pipe for frontend pipe id " << pipe_id << " and dir_mask " << dir_mask
              << " uses inconsistent slot_size values: " << it->second << " and " << slot_size;
        }
      }
      if (auto for_stmt = As<ir::ForStmt>(stmt)) {
        scan_stmts(FlattenBody(for_stmt->body_));
      } else if (auto if_stmt = As<ir::IfStmt>(stmt)) {
        scan_stmts(FlattenBody(if_stmt->then_body_));
        if (if_stmt->else_body_.has_value()) {
          scan_stmts(FlattenBody(if_stmt->else_body_.value()));
        }
      } else if (auto while_stmt = As<ir::WhileStmt>(stmt)) {
        scan_stmts(FlattenBody(while_stmt->body_));
      }
    }
  };

  for (const auto& [gvar, func] : program->functions_) {
    (void)gvar;
    if (func->body_) {
      scan_stmts(FlattenBody(func->body_));
    }
  }

  // Each pipe's region must advance by its FULL footprint — both rings of a bidirectional pipe,
  // and an explicit slot_num where given — or the next pipe's base lands inside this one. This
  // has to stay in step with ComputeGMPipeWorkspaceElements, which sizes the whole workspace;
  // both derive it from gm_pipe_layout.h.
  int64_t byte_offset = 0;
  for (const auto& [key, slot_size] : slot_size_by_pipe) {
    gm_slot_buffer_offsets_[key] = byte_offset;
    const int dir_mask = key.second;
    CHECK(gm_pipe::SlotCountForDirMask(dir_mask) > 0)
        << "initialize_pipe has invalid dir_mask for GM slot buffer: " << dir_mask;
    auto num_it = slot_count_by_pipe.find(key);
    const int slot_count = num_it != slot_count_by_pipe.end() ? num_it->second : 0;
    const int64_t pipe_bytes = gm_pipe::FootprintBytes(dir_mask, slot_count, slot_size);
    CHECK(byte_offset <= std::numeric_limits<int64_t>::max() - pipe_bytes)
        << "GM slot buffer offset overflow while assigning frontend pipe id " << key.first;
    byte_offset += pipe_bytes;
  }
}

// ========================================================================
// Distributed N6: inline peer-offset (CommContext) arithmetic
// ========================================================================

std::string PTOCodegen::EmitCommRemoteOffsetInline(const std::string& ctx_ssa, const std::string& peer_ssa,
                                                   const DataType& dtype) {
  // Sub-byte dtypes (bool / 4-bit) have no whole-byte element stride, so the
  // byte→element division at the bottom is ill-defined. Fail here, at the op
  // call site, where the CHECK message still has caller context.
  const size_t elem_bits = dtype.GetBit();
  CHECK(elem_bits >= 8 && elem_bits % 8 == 0)
      << "Distributed remote ops only support byte-sized element types, got " << dtype.ToString() << " ("
      << elem_bits << " bits)";
  const int64_t elem_size_bytes = static_cast<int64_t>(elem_bits / 8);

  namespace cl = codegen::distributed::comm_layout;
  // CommContext field indices, expressed in u64 slots (one ``pto.load_scalar``
  // step = one slot). Pinned via static_assert in
  // include/pypto/codegen/distributed/comm_layout.h so a runtime ABI shift
  // fails PyPTO compilation rather than silently emitting wrong addresses.
  const int64_t k_rank_idx = static_cast<int64_t>(cl::kRankIdOffset / cl::kWindowSlotStride);
  const int64_t k_win_idx = static_cast<int64_t>(cl::kWindowsInOffset / cl::kWindowSlotStride);

  // Every value below gets a fresh SSA name (constants are deduplicated into
  // the function's constants section), so a function may hold any number of
  // remote ops without name collisions.
  const std::string c_r = GetOrEmitConstant(k_rank_idx, DataType::INDEX);
  const std::string c_w = GetOrEmitConstant(k_win_idx, DataType::INDEX);

  // Read rankId (the low 32 bits of the (rankId, rankNum) 8-byte slot at
  // u64 index k_rank_idx).
  const std::string rk_pair = NewTemp();
  Emit(rk_pair + " = pto.load_scalar " + ctx_ssa + "[" + c_r + "] : !pto.ptr<i64> -> i64");
  const std::string rk_i32 = NewTemp();
  Emit(rk_i32 + " = arith.trunci " + rk_pair + " : i64 to i32");
  const std::string rk_idx = NewTemp();
  Emit(rk_idx + " = arith.index_cast " + rk_i32 + " : i32 to index");

  // local_base = windowsIn[rankId]
  const std::string lb_off = NewTemp();
  Emit(lb_off + " = arith.addi " + c_w + ", " + rk_idx + " : index");
  const std::string lbase = NewTemp();
  Emit(lbase + " = pto.load_scalar " + ctx_ssa + "[" + lb_off + "] : !pto.ptr<i64> -> i64");

  // peer_base = windowsIn[peer]
  const std::string pb_off = NewTemp();
  Emit(pb_off + " = arith.addi " + c_w + ", " + peer_ssa + " : index");
  const std::string pbase = NewTemp();
  Emit(pbase + " = pto.load_scalar " + ctx_ssa + "[" + pb_off + "] : !pto.ptr<i64> -> i64");

  // delta_bytes = peer_base - local_base; converted to an element offset
  // because pto.addptr takes element counts, not bytes.
  const std::string dbytes = NewTemp();
  Emit(dbytes + " = arith.subi " + pbase + ", " + lbase + " : i64");
  const std::string esize = GetOrEmitConstant(elem_size_bytes, DataType::INT64);
  const std::string delems_i = NewTemp();
  Emit(delems_i + " = arith.divsi " + dbytes + ", " + esize + " : i64");
  const std::string delems = NewTemp();
  Emit(delems + " = arith.index_cast " + delems_i + " : i64 to index");
  return delems;
}

std::string PTOCodegen::RegisterDeferredCompletionAdapter() {
  needs_deferred_completion_adapter_ = true;
  return kDeferredCompletionAdapterName;
}

void PTOCodegen::EmitDeferredCompletionAdapterDeclaration() {
  if (!needs_deferred_completion_adapter_) return;
  stream_ << "  func.func private @" << kDeferredCompletionAdapterName
          << "(!pto.ptr<i64>, !pto.ptr<i32>, i64, i64)\n";
}

void PTOCodegen::GenerateFunction(const FunctionPtr& func) {
  fs_.Reset();
  fs_.current_function = func;

  // Collect dyn-dim Vars from tensor-parameter shapes once. The same list
  // drives both name reservation (Site A below) and the trailing %argN: index
  // params on the MLIR signature (Site B further down) -- a single source of
  // truth keeps the two in lockstep.
  const std::vector<VarPtr> dyn_vars = CollectTensorShapeDynVars(func);

  // Attribute every symbol that appears in a parameter's valid_shape but NOT in
  // any physical shape to the parameter that declares it. Such a symbol gets no
  // trailing %argN slot (CollectTensorShapeDynVars walks shape_ only) and is
  // bound at the call site, so the kernel cannot materialize it. Recording the
  // origin here lets GetVarName name the parameter if the symbol ever reaches
  // an emitted expression.
  {
    std::set<const ir::Var*> shape_bound;
    for (const auto& dyn_var : dyn_vars) shape_bound.insert(dyn_var.get());
    for (const auto& param : func->params_) {
      auto tensor_type = ir::AsTensorTypeLike(param->GetType());
      if (!tensor_type) continue;
      std::vector<VarPtr> valid_vars;
      std::set<const ir::Var*> seen;
      for (const auto& dim : ir::GetEffectiveTensorValidShape(*tensor_type)) {
        CollectVarsFromShapeExprImpl(dim, seen, valid_vars);
      }
      // Report the author's parameter name, not its SSA-renamed form: the
      // diagnostic points at DSL source the user can actually edit.
      std::string param_name = ir::auto_name::GetCompatibleBaseName(param->name_hint_);
      if (param_name.empty()) param_name = param->name_hint_;
      for (const auto& valid_var : valid_vars) {
        if (shape_bound.count(valid_var.get()) == 0) {
          fs_.valid_shape_symbol_origin.emplace(valid_var.get(), param_name);
        }
      }
    }
  }

  // Reserve %argN names upfront so NewNamedTemp never collides with them
  for (size_t i = 0; i < func->params_.size(); i++) {
    fs_.used_ssa_names.insert("arg" + std::to_string(i));
  }
  // Reserve extra %argN slots for generated trailing signature args
  // (``dyn_vars`` computed at the top of GenerateFunction). Explicit
  // CommCtxType params are already included in func->params_.
  for (size_t i = 0; i < dyn_vars.size(); i++) {
    fs_.used_ssa_names.insert("arg" + std::to_string(func->params_.size() + i));
  }

  BuildVarToMemRefMapping(func);

  // One body walk: collects MemRefs and detects hidden runtime parameters.
  // The SDMA workspace and SPMD identity params are injected at codegen time
  // (not at IR level) when the function body invokes the corresponding ops.
  MemRefCollectorVisitor collector;
  if (func->body_) {
    collector.VisitStmt(func->body_);
  }
  const bool uses_sdma_workspace = collector.UsesSdmaWorkspace();
  const bool uses_deferred_completion = collector.UsesDeferredCompletion();
  const bool uses_spmd_params = collector.UsesSpmdBlockOps();
  const bool uses_subblock_param = collector.UsesSubblockOp();
  fs_.ffts_workspace_vars = collector.GetFFTSWorkspaceVars();
  if (uses_sdma_workspace) {
    fs_.used_ssa_names.insert("arg" + std::to_string(func->params_.size() + dyn_vars.size()));
  }
  if (uses_deferred_completion) {
    fs_.used_ssa_names.insert("__pypto_deferred_raw_args");
  }
  if (uses_spmd_params) {
    fs_.used_ssa_names.insert("__pypto_spmd_block_idx");
    fs_.used_ssa_names.insert("__pypto_spmd_block_num");
  }
  if (uses_subblock_param) {
    fs_.used_ssa_names.insert("__pypto_spmd_subblock_idx");
  }

  // Still collect fs_.memref_to_tile_type for GetTileBufTypeString fallback paths
  fs_.memref_to_tile_type = collector.GetMemRefTileTypes();

  // Per-var SSA binding: each tile variable gets its own SSA name — except in
  // PTOAS memory-planner mode (no addr baked), where variables denoting the
  // *same* buffer (same MemRef base+offset+size, e.g. a loop-carried
  // accumulator coalesced by MemoryReuse) must share one tile_buf handle. In
  // level3 that aliasing was carried by an identical `addr`; without addr, ptoas
  // PlanMemory would otherwise allocate them separately, so we instead emit a
  // single alloc_tile and let the op write in place (`outs(%acc)`).
  for (const auto& [tile_var, tile_type] : fs_.tile_var_allocs) {
    auto memref = ir::GetDefinedMemRef(tile_type);

    std::string type_str = GetTileBufTypeStringFromTileType(tile_type);

    std::string ssa_name;
    if (!emit_tile_addr_) {
      const std::string ident = MemRefIdentityKey(memref);
      auto it = fs_.memref_identity_to_mlir.find(ident);
      if (it != fs_.memref_identity_to_mlir.end()) {
        ssa_name = it->second;  // reuse the shared handle (in-place aliasing)
      } else {
        ssa_name = NewNamedTemp(tile_var->name_hint_);
        fs_.memref_identity_to_mlir[ident] = ssa_name;
      }
      // Same bytes does not mean same tile_buf type: a [1, N] row-major op result
      // and its [N, 1] col-major reshape view share base+offset+size. They still
      // share one handle (differently-typed reads become `pto.treshape` views of
      // it), but an MLIR SSA value has exactly one type, so callers that want to
      // *re-type* the handle — the IfStmt phi head-declaration — must not touch a
      // mixed-type identity. Record which identities are uniform.
      auto [type_it, fresh] = fs_.memref_identity_type.emplace(ident, type_str);
      if (!fresh && type_it->second != type_str) {
        fs_.memref_identity_mixed_types.insert(ident);
      }
    } else {
      ssa_name = NewNamedTemp(tile_var->name_hint_);
    }
    BindVarToMlir(tile_var, ssa_name);

    // Pre-populate type so body visitors (e.g., tile.reshape no-op check)
    // can query it before per-variable alloc_tile emission runs.
    fs_.ssa_to_tile_buf_type[ssa_name] = type_str;

    // Also maintain fs_.memref_to_mlir for compatibility (first var per allocation)
    const ir::Var* base_ptr = memref->base_.get();
    if (fs_.memref_to_mlir.find(base_ptr) == fs_.memref_to_mlir.end()) {
      fs_.memref_to_mlir[base_ptr] = ssa_name;
    }
  }

  // ``dyn_vars`` was computed at the top of GenerateFunction; it carries the
  // trailing %argN: index parameters in first-seen order.

  // Collect ordered DistributedTensor params and their materialized CommCtx
  // params (both in IR-param order) so get_comm_ctx aliases can resolve to the
  // explicit ctx pointer argument.
  std::vector<VarPtr> dist_tensor_params;
  std::vector<VarPtr> comm_ctx_params;
  for (const auto& param : func->params_) {
    if (As<ir::DistributedTensorType>(param->GetType())) {
      dist_tensor_params.push_back(param);
    } else if (ir::IsA<ir::CommCtxType>(param->GetType())) {
      comm_ctx_params.push_back(param);
    }
  }

  stream_ << "  func.func @" << func->name_ << "(";

  // Separate params into tensors and scalars for tensors-first dispatch order.
  // PTOParam dispatches args as [tensors..., scalars...] regardless of function
  // signature order, so the MLIR function signature must match that layout.
  // DistributedTensorType inherits TensorType and uses the same `!pto.ptr<T>`
  // signature slot — fold it into the tensor partition via ir::AsTensorTypeLike.
  std::vector<size_t> tensor_param_indices;
  std::vector<size_t> scalar_param_indices;
  for (size_t i = 0; i < func->params_.size(); i++) {
    if (ir::AsTensorTypeLike(func->params_[i]->GetType())) {
      tensor_param_indices.push_back(i);
    } else {
      scalar_param_indices.push_back(i);
    }
  }

  // Assign %argN names: tensors get indices 0..N_tensors-1, scalars get N_tensors..
  size_t scalar_start_idx = tensor_param_indices.size();
  std::set<const ir::Var*> param_keys;
  for (size_t j = 0; j < tensor_param_indices.size(); j++) {
    const auto& param = func->params_[tensor_param_indices[j]];
    BindVarToMlir(param, "%arg" + std::to_string(j));
    param_keys.insert(GetVarKey(param));
  }
  for (size_t j = 0; j < scalar_param_indices.size(); j++) {
    const auto& param = func->params_[scalar_param_indices[j]];
    BindVarToMlir(param, "%arg" + std::to_string(scalar_start_idx + j));
    param_keys.insert(GetVarKey(param));
  }

  // Emit signature: tensors first, then scalars
  bool first_param = true;
  for (size_t j = 0; j < tensor_param_indices.size(); j++) {
    if (!first_param) stream_ << ", ";
    first_param = false;
    const auto& param = func->params_[tensor_param_indices[j]];
    auto tensor_type = ir::AsTensorTypeLike(param->GetType());
    if (fs_.ffts_workspace_vars.count(param.get()) > 0) {
      auto extent = As<ir::ConstInt>(tensor_type->shape_[0]);
      INTERNAL_CHECK_SPAN(extent && tensor_type->dtype_ == DataType::INT64, param->span_)
          << "FFTS workspace must be a statically sized INT64 tensor";
    }
    stream_ << "%arg" << j << ": !pto.ptr<" << GetTypeString(tensor_type->dtype_) << ">";
  }
  for (size_t j = 0; j < scalar_param_indices.size(); j++) {
    if (!first_param) stream_ << ", ";
    first_param = false;
    const auto& param = func->params_[scalar_param_indices[j]];
    stream_ << "%arg" << (scalar_start_idx + j) << ": ";
    if (auto scalar_type = As<ScalarType>(param->GetType())) {
      stream_ << GetTypeString(scalar_type->dtype_);
    } else if (ir::IsA<ir::CommCtxType>(param->GetType())) {
      stream_ << "!pto.ptr<i64>";
    } else {
      stream_ << "!pto.ptr<f32>";
    }
  }

  // Pair each DistributedTensor param with its explicit CommCtxType param (in
  // IR-param order). The runtime CommContext is passed as a GM ``uint64_t*``
  // (see ``runtime/src/common/platform_comm/comm_context.h``); codegen indexes
  // its fields via ``pto.load_scalar`` and the ``comm_layout::k*`` constants.
  INTERNAL_CHECK_SPAN(dist_tensor_params.size() == comm_ctx_params.size(), func->span_)
      << "PTOCodegen: function '" << func->name_ << "' has " << dist_tensor_params.size()
      << " DistributedTensor params but " << comm_ctx_params.size()
      << " CommCtxType params; run MaterializeDistTensorCtx before PTO codegen";
  for (size_t i = 0; i < dist_tensor_params.size(); ++i) {
    fs_.dist_tensor_to_ctx[GetVarKey(dist_tensor_params[i])] = GetVarName(comm_ctx_params[i]);
  }

  // Append trailing index parameters for each unique dynamic dimension variable
  size_t next_arg_idx = func->params_.size();
  for (const auto& dyn_var : dyn_vars) {
    std::string arg_name = "%arg" + std::to_string(next_arg_idx++);
    stream_ << ", " << arg_name << ": index";
    BindVarToMlir(dyn_var, arg_name);
  }

  // Deferred completion registration needs the scheduler-owned AsyncCtx,
  // which is reachable only from kernel_entry's raw dispatch args. Keep this
  // hidden ABI before other runtime-owned arguments; the Python wrapper mirrors
  // the order exactly.
  if (uses_deferred_completion) {
    if (!first_param) stream_ << ", ";
    first_param = false;
    fs_.deferred_completion_raw_args_ssa = "%__pypto_deferred_raw_args";
    stream_ << fs_.deferred_completion_raw_args_ssa << ": !pto.ptr<i64>";
  }

  // Append the hidden SDMA workspace pointer after user-derived arguments and
  // before SPMD identity params. The Python wrapper mirrors this exact order.
  if (uses_sdma_workspace) {
    if (!first_param) stream_ << ", ";
    fs_.sdma_workspace_arg_ssa = "%arg" + std::to_string(next_arg_idx++);
    stream_ << fs_.sdma_workspace_arg_ssa << ": !pto.ptr<i8>";
  }

  // Append SPMD identity params after the dynamic-dim and SDMA workspace args,
  // in canonical order (block_idx, block_num, subblock_idx). Each is appended
  // independently based on the ops the function actually uses; the Python
  // kernel wrapper mirrors this exact order when forwarding the call args.
  // Named SSAs make the synthetic origin obvious in the emitted MLIR and let
  // lowerings refer to them via PTOCodegen::GetSpmd{Block,Subblock}*ArgSSA().
  if (uses_spmd_params) {
    fs_.spmd_block_idx_arg = "%__pypto_spmd_block_idx";
    fs_.spmd_block_num_arg = "%__pypto_spmd_block_num";
    stream_ << ", " << fs_.spmd_block_idx_arg << ": i32, " << fs_.spmd_block_num_arg << ": i32";
  }
  if (uses_subblock_param) {
    fs_.spmd_subblock_idx_arg = "%__pypto_spmd_subblock_idx";
    stream_ << ", " << fs_.spmd_subblock_idx_arg << ": i32";
  }

  stream_ << ")";
  switch (func->func_type_) {
    case ir::FunctionType::AIC:
      stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<cube>}";
      break;
    case ir::FunctionType::AIV:
      stream_ << " attributes {pto.kernel_kind = #pto.kernel_kind<vector>}";
      break;
    default:
      // Other function types like InCore are not expected here and have no kernel_kind.
      break;
  }
  stream_ << " {\n";
  indent_level_++;
  fs_.constants_indent = GetIndent();

  // Pre-emit alloc_tile address constants now that indent_level_ is set.
  // For addr constants specifically, codegen preserves the IR ConstInt
  // dtype 1:1 (other operands like valid_row/valid_col adapt to the
  // consumer's type via cast_to_index — see ComputeAllocTileFields).
  if (emit_tile_addr_) {
    for (const auto& [tile_var, tile_type] : fs_.tile_var_allocs) {
      auto memref = ir::GetDefinedMemRef(tile_type);
      if (auto const_offset = memref ? As<ir::ConstInt>(memref->byte_offset_) : nullptr) {
        GetOrEmitConstant(const_offset->value_, const_offset->dtype());
      }
    }
  }

  // Decide which declared multi-slot allocations become ptoas multi-buffer
  // regions. Runs here, after the constants indent is set (the region's shared
  // valid extent is emitted as constants) and before the body walk, which reads
  // the plan when it lowers each slot.
  PlanMultiBufferRegions(func);

  // Parameters are already bound; non-param tile vars are bound above in per-var SSA binding

  for (const auto& var : func->params_) {
    if (auto tensor_type = ir::AsTensorTypeLike(var->GetType())) {
      // Skip tensor view for GM slot buffer workspace parameter (raw pointer, no view needed)
      if (var->name_hint_ == "__gm_pipe_buffer") {
        RecordGMSlotBufferSSA(GetVarName(var), tensor_type->dtype_);
        continue;
      }
      if (fs_.ffts_workspace_vars.count(var.get()) > 0) continue;
      std::string tensor_view = NewNamedTemp(var->name_hint_ + "_view");
      BindTensorView(var, tensor_view);
      // Remember the base pointer so mid-body pl.read/pl.write resolve to !pto.ptr
      // even after a slice-assign rebinds the var to its tensor_view.
      RegisterBasePtr(var, GetVarName(var));

      for (const auto& j : tensor_type->shape_) {
        if (As<ir::ConstInt>(j)) {
          GetOrEmitConstant(GetConstIntValue(j), DataType::INDEX);
        }
      }
      // Pre-emit stride constants: use explicit tensor_view_.stride if available,
      // otherwise fall back to shape-based stride computation.
      bool has_explicit_stride =
          tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty();
      if (has_explicit_stride) {
        for (const auto& s : tensor_type->tensor_view_->stride) {
          if (As<ir::ConstInt>(s)) {
            GetOrEmitConstant(GetConstIntValue(s), DataType::INDEX);
          }
        }
      } else if (tensor_type->shape_.size() == 2) {
        if (As<ir::ConstInt>(tensor_type->shape_[1])) {
          GetOrEmitConstant(GetConstIntValue(tensor_type->shape_[1]), DataType::INDEX);
        }
        GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      } else {
        // 1-D and N-D (N>2): pre-emit constant 1 (innermost stride). For N>2,
        // other strides are computed dynamically via arith.muli in
        // EmitMakeTensorViews to support dynamic dims.
        GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      }
    }
  }

  auto saved_stream = std::move(stream_);
  stream_ = std::move(fs_.body_section);

  if (func->body_) {
    VisitStmt(func->body_);
  }

  std::string body_content = stream_.str();

  // Render the prologue before flushing constants so constants unique to a
  // shape/stride expression (e.g. the 2 in M * 2) are declared before use.
  stream_.str("");
  stream_.clear();
  EmitMakeTensorViews(func);
  EmitExtraAllocTiles();
  std::string prologue_content = stream_.str();

  stream_ = std::move(saved_stream);

  stream_ << fs_.constants_section.str();
  stream_ << prologue_content;
  stream_ << body_content;
  stream_ << GetIndent() << "return\n";

  indent_level_--;
  stream_ << "  }\n";
}

void PTOCodegen::BuildVarToMemRefMapping(const FunctionPtr& func) {
  class VarMemRefMapper : public ir::IRVisitor {
   public:
    std::map<const ir::Var*, const ir::Var*>& var_to_memref;    ///< tile var → base_ Ptr
    std::map<const ir::Var*, std::string>& memref_to_var_name;  ///< base_ Ptr → var name
    std::vector<std::pair<VarPtr, std::shared_ptr<const TileType>>>& tile_var_allocs;

    VarMemRefMapper(std::map<const ir::Var*, const ir::Var*>& mapping,
                    std::map<const ir::Var*, std::string>& reverse_mapping,
                    std::vector<std::pair<VarPtr, std::shared_ptr<const TileType>>>& allocs)
        : var_to_memref(mapping), memref_to_var_name(reverse_mapping), tile_var_allocs(allocs) {}

    void VisitStmt_(const AssignStmtPtr& op) override {
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        const auto memref = ir::GetDefinedMemRef(tile_type);
        const ir::Var* base_ptr = memref->base_.get();
        var_to_memref[op->var_.get()] = base_ptr;
        if (memref_to_var_name.find(base_ptr) == memref_to_var_name.end()) {
          memref_to_var_name[base_ptr] = op->var_->name_hint_;
        }
        tile_var_allocs.emplace_back(op->var_, tile_type);
      }
      ir::IRVisitor::VisitStmt_(op);
    }
  };

  VarMemRefMapper mapper(fs_.var_to_memref, fs_.memref_to_var_name, fs_.tile_var_allocs);
  if (func->body_) {
    mapper.VisitStmt(func->body_);
  }
}

void PTOCodegen::EmitMakeTensorViews(const FunctionPtr& func) {
  // RFC #1300 P7 (canonical codegen).
  //
  // Emit ``pto.make_tensor_view`` directly from the IR's canonical
  // ``(shape, stride, layout)`` triple. There are no implicit swaps or
  // post-emit dn_swap path here — every layout-aware transform (RFC §3.3
  // canonical promotion, ``MaterializeTensorStrides``) has already run by the
  // time codegen executes, so the IR's TensorView fields can be transcribed
  // verbatim.
  //
  // The one exception is the ordinary ``[M, 1]`` column-vector special case:
  // PTOAS *infers* DN for shape ``[M, 1]`` with degenerate strides regardless
  // of an ND declaration, so codegen forces DN + ``[1, M]`` strides. MX
  // layouts are explicit hardware contracts and bypass this legacy override.
  ir::var_collectors::VarDefUseCollector body_vars;
  if (func->body_) body_vars.VisitStmt(func->body_);

  for (const auto& param : func->params_) {
    auto tensor_type = ir::AsTensorTypeLike(param->GetType());
    if (!tensor_type) continue;
    // Core-group outlining keeps the complete public signature on both the
    // AIC and AIV functions.  Do not materialize a view for a tensor that the
    // outlined body does not reference: PTOAS cannot infer a non-ND layout for
    // such an unused view (notably the MX scale tensors on the AIV cast side).
    if (body_vars.var_uses.count(param.get()) == 0) continue;
    if (param->name_hint_ == "__gm_pipe_buffer") continue;         // GM slot buffer is a raw pointer
    if (fs_.ffts_workspace_vars.count(param.get()) > 0) continue;  // FFTS workspace stays a raw pointer

    // ptoas rejects a malformed view (bad strides / layout) on this line, so
    // attribute it to the parameter that declared the tensor.
    SpanScope param_loc(this, &param->span_);
    std::string tensor_view = fs_.tensor_to_view.at(GetVarKey(param));
    const size_t rank = tensor_type->shape_.size();

    // ``[..., M, 1]`` column-vector legacy path: PTOAS infers DN for any
    // shape whose innermost dim is constant 1, so the codegen forces DN to
    // match what ``tile.load`` produces (memory.cpp DeduceTileLoadType emits
    // a ColMajor BLayout tile whenever the load shape ends with a constant 1
    // — see test_tensor_expand_clone[broadcast_dim=2] where input
    // ``[B, N, 1]`` is loaded into a ColMajor tile and PTOAS TLoad enforces
    // ``tile.BLayout == tensor.Layout``).
    bool is_column_vector = false;
    if (rank >= 2) {
      auto last_dim = As<ir::ConstInt>(tensor_type->shape_.back());
      if (last_dim && last_dim->value_ == 1) {
        is_column_vector = true;
      }
    }

    ir::TensorLayout layout = ir::TensorLayout::ND;
    if (tensor_type->tensor_view_.has_value()) {
      layout = tensor_type->tensor_view_->layout;
    }
    const bool force_column_vector_dn = is_column_vector && !IsMxTensorLayout(layout);
    if (force_column_vector_dn) layout = ir::TensorLayout::DN;

    // Materialize one shape dimension as an MLIR SSA value.
    auto get_shape_dim_mlir = [&](size_t dim_idx) -> std::string {
      const auto& dim_expr = tensor_type->shape_[dim_idx];
      if (auto const_int = As<ir::ConstInt>(dim_expr)) {
        return GetOrEmitConstant(const_int->value_, DataType::INDEX);
      }
      return EmitCastToIndex(dim_expr, GetExprAsCode(dim_expr));
    };
    // Materialize a stride ExprPtr as an MLIR SSA value.
    auto get_stride_mlir = [&](const ir::ExprPtr& stride_expr) -> std::string {
      if (auto const_int = As<ir::ConstInt>(stride_expr)) {
        return GetOrEmitConstant(const_int->value_, DataType::INDEX);
      }
      return EmitCastToIndex(stride_expr, GetExprAsCode(stride_expr));
    };
    // Precompute shape dim SSA names. Dynamic shape exprs may need cast SSA
    // ops (``EmitCastToIndex``) emitted before the ``pto.make_tensor_view``
    // line — materialize them all up-front so the main statement is a single
    // contiguous line.
    std::vector<std::string> shape_dim_names(rank);
    for (size_t j = 0; j < rank; ++j) {
      shape_dim_names[j] = get_shape_dim_mlir(j);
    }

    // Emit one stride multiply ``lhs * shape_dim_names[dim_idx]`` and return
    // the resulting SSA, used for fallback stride derivation when
    // ``tensor_view_->stride`` is empty.
    auto emit_stride_mul = [&](const std::string& lhs, size_t dim_idx, size_t stride_slot) -> std::string {
      std::string mul_name = NewNamedTemp(param->name_hint_ + "_s" + std::to_string(stride_slot));
      Emit(mul_name + " = arith.muli " + lhs + ", " + shape_dim_names[dim_idx] + " : index");
      return mul_name;
    };

    // Build the stride SSA names. Prefer explicit ``tensor_view_->stride``;
    // fall back to canonical derivation per ``layout`` when absent
    // (``MaterializeTensorStrides`` should normally have populated it by now,
    // but the codegen tolerates absent strides for any path that constructs
    // IR ad-hoc and skips the pipeline).
    std::vector<std::string> stride_names(rank);
    bool has_explicit_stride =
        tensor_type->tensor_view_.has_value() && !tensor_type->tensor_view_->stride.empty();
    if (has_explicit_stride) {
      const auto& strides = tensor_type->tensor_view_->stride;
      CHECK(strides.size() == rank) << "EmitMakeTensorViews: explicit stride rank " << strides.size()
                                    << " does not match tensor shape rank " << rank;
      for (size_t j = 0; j < rank; ++j) {
        stride_names[j] = get_stride_mlir(strides[j]);
      }
    } else if (force_column_vector_dn) {
      // Forced-DN ``[..., M, 1]`` legacy stride pattern (PTOAS column-vector
      // convention): trailing pair degenerates to ``stride[rank-2]=1`` and
      // ``stride[rank-1]=shape[rank-1]=1``; outer dims walk row-major over the
      // ``M`` extent (``stride[rank-3]=shape[rank-2]``, ``stride[k-1]=stride[k]*shape[k]``).
      // For rank 2 this collapses to the legacy ``[1, shape[0]]``.
      stride_names[rank - 2] = GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      if (rank == 2) {
        stride_names[rank - 1] = shape_dim_names[0];
      } else {
        // rank >= 3: stride[rank-1] = shape[rank-1] (= 1), stride[rank-3] = shape[rank-2].
        stride_names[rank - 1] = shape_dim_names[rank - 1];
        stride_names[rank - 3] = shape_dim_names[rank - 2];
        for (int j = static_cast<int>(rank) - 4; j >= 0; --j) {
          size_t dim = static_cast<size_t>(j);
          stride_names[dim] = emit_stride_mul(stride_names[dim + 1], dim + 1, dim);
        }
      }
    } else if (layout == ir::TensorLayout::DN) {
      CHECK(rank >= 2) << "EmitMakeTensorViews: DN layout requires rank >= 2, got " << rank;
      // RFC §2.3 canonical DN: stride[-2]=1, stride[-1]=shape[-2], outer
      // strides walk row-major over the DN-block volume. Use direct shape
      // references for the trailing pair so 2D DN avoids a spurious
      // ``arith.muli %c1, shape`` step.
      stride_names[rank - 2] = GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      stride_names[rank - 1] = shape_dim_names[rank - 2];
      if (rank >= 3) {
        // stride[n-3] = shape[n-2] * shape[n-1] (one full DN-block volume).
        stride_names[rank - 3] = emit_stride_mul(shape_dim_names[rank - 2], rank - 1, rank - 3);
        for (int j = static_cast<int>(rank) - 4; j >= 0; --j) {
          size_t dim = static_cast<size_t>(j);
          stride_names[dim] = emit_stride_mul(stride_names[dim + 1], dim + 1, dim);
        }
      }
    } else {
      // Canonical ND (row-major): stride[-1]=1, stride[k]=stride[k+1]*shape[k+1].
      // For rank 2 specifically, stride[0] = shape[1] directly (avoids a
      // spurious ``arith.muli %c1, shape[1]`` step).
      stride_names[rank - 1] = GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      if (rank >= 2) {
        stride_names[rank - 2] = shape_dim_names[rank - 1];
        for (int j = static_cast<int>(rank) - 3; j >= 0; --j) {
          size_t dim = static_cast<size_t>(j);
          stride_names[dim] = emit_stride_mul(stride_names[dim + 1], dim + 1, dim);
        }
      }
    }

    // Buffer the statement so Emit() writes it as one line and can suffix the
    // parameter's source location.
    std::ostringstream view_line;
    view_line << tensor_view << " = pto.make_tensor_view ";
    view_line << GetVarName(param);

    // Emit shape (verbatim from IR — canonical).
    view_line << ", shape = [";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) view_line << ", ";
      view_line << shape_dim_names[j];
    }
    view_line << "],";

    // Emit strides.
    view_line << " strides = [";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) view_line << ", ";
      view_line << stride_names[j];
    }
    view_line << "]";

    std::string layout_str = "nd";
    switch (layout) {
      case ir::TensorLayout::DN:
        layout_str = "dn";
        break;
      case ir::TensorLayout::NZ:
        layout_str = "nz";
        break;
      case ir::TensorLayout::MX_A_ZZ:
        layout_str = "mx_a_zz";
        break;
      case ir::TensorLayout::MX_B_NN:
        layout_str = "mx_b_nn";
        break;
      case ir::TensorLayout::ND:
        break;
    }
    view_line << " {layout = #pto.layout<" << layout_str << ">} : ";

    view_line << "!pto.tensor_view<";
    for (size_t j = 0; j < rank; ++j) {
      if (j > 0) view_line << "x";
      view_line << "?";
    }
    view_line << "x" << GetTypeString(tensor_type->dtype_) << ">";
    Emit(view_line.str());
  }
}

PTOCodegen::AllocTileFields PTOCodegen::ComputeAllocTileFields(
    const std::shared_ptr<const ir::TileType>& tile_type, bool use_physical_valid_shape) {
  AllocTileFields fields;

  // Type string always uses dynamic valid dims (v_row=?, v_col=?); the actual
  // extent is conveyed via valid_row / valid_col operands below.
  fields.type_str = GetTileBufTypeStringFromTileType(tile_type);

  // Cast a non-index integer SSA to `index` (PTOAS expects index typed
  // valid_row / valid_col operands). Floating-point operands are rejected.
  auto cast_to_index = [&](const std::string& ssa, const ir::ExprPtr& expr) -> std::string {
    auto scalar_type = As<ScalarType>(expr->GetType());
    if (!scalar_type || scalar_type->dtype_ == DataType::INDEX) return ssa;
    CHECK(scalar_type->dtype_.IsInt())
        << "alloc_tile valid_row/valid_col operand must be integer or index typed, got "
        << GetTypeString(scalar_type->dtype_);
    std::string idx = NewTemp();
    Emit(idx + " = arith.index_cast " + ssa + " : " + GetTypeString(scalar_type->dtype_) + " to index");
    return idx;
  };

  // Widen an address expression to the `i64` the alloc_tile addr operand takes.
  // Mirrors cast_to_index above, in the other direction.
  auto cast_to_i64 = [&](const std::string& ssa, const ir::ExprPtr& expr) -> std::string {
    auto scalar_type = As<ScalarType>(expr->GetType());
    CHECK(scalar_type && (scalar_type->dtype_.IsInt() || scalar_type->dtype_ == DataType::INDEX))
        << "alloc_tile addr operand must be integer or index typed, got "
        << (scalar_type ? GetTypeString(scalar_type->dtype_) : std::string("non-scalar"));
    if (scalar_type->dtype_ == DataType::INT64) return ssa;
    std::string wide = NewTemp();
    const std::string from =
        scalar_type->dtype_ == DataType::INDEX ? "index" : GetTypeString(scalar_type->dtype_);
    Emit(wide + " = arith.index_cast " + ssa + " : " + from + " to i64");
    return wide;
  };

  // FP4 Vec tile_bufs use PTOAS's physical x2-carrier coordinates along the
  // BLayout packed axis. PyPTO keeps logical nibble shapes internally; matrix
  // spaces are excluded because TMATMUL_MX has its own logical-dimension ABI.
  const auto memory_space = tile_type->GetMemorySpace();
  const auto tile_view = ir::tile_view_semantics::GetEffectiveTileView(*tile_type);
  const bool packed_fp4_vec =
      tile_type->dtype_ == DataType::FP4 && memory_space.has_value() && *memory_space == ir::MemorySpace::Vec;
  const size_t packed_dim = tile_view.blayout == ir::TileLayout::col_major ? 0 : 1;

  // Lower a single valid_shape dim expression to an `index` SSA value.
  auto lower_dim = [&](const ir::ExprPtr& expr, size_t dim) -> std::string {
    if (!expr) return "";
    if (auto ci = As<ir::ConstInt>(expr)) {
      int64_t value = ci->value_;
      if (packed_fp4_vec && dim == packed_dim) {
        CHECK(value > 0 && value % 2 == 0)
            << "FP4 Vec valid_shape packed dimension must be a positive even logical extent for PTOAS, got "
            << value;
        value /= 2;
      }
      return GetOrEmitConstant(value, DataType::INDEX);
    }
    CHECK(!(packed_fp4_vec && dim == packed_dim)) << "Dynamic FP4 Vec valid_shape on the packed dimension is "
                                                     "not supported; provide a static even extent";
    return cast_to_index(GetExprAsCode(expr), expr);
  };

  // Source of truth for valid_row / valid_col operand values:
  //   - tile_view.valid_shape when populated (preferred — captures user intent
  //     such as a smaller load region or a runtime ctx_len);
  //   - tile_type->shape_ otherwise (physical dims).
  const std::vector<ir::ExprPtr>* dims = nullptr;
  if (const auto& tile_view = tile_type->tile_view_;
      !use_physical_valid_shape && tile_view.has_value() && !tile_view->valid_shape.empty()) {
    dims = &tile_view->valid_shape;
  } else if (!tile_type->shape_.empty()) {
    dims = &tile_type->shape_;
  }

  if (dims != nullptr) {
    if (dims->size() == 1) {
      // Match ExtractTileTypeInfo: 1-D tile maps to rows=1, cols=shape[0].
      fields.valid_row_ssa = GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      fields.valid_col_ssa = lower_dim((*dims)[0], 1);
    } else {
      if (dims->size() >= 1) fields.valid_row_ssa = lower_dim((*dims)[0], 0);
      if (dims->size() >= 2) fields.valid_col_ssa = lower_dim((*dims)[1], 1);
    }
  }

  auto memref = ir::GetDefinedMemRef(tile_type);
  if (memref && emit_tile_addr_) {
    if (auto const_offset = As<ir::ConstInt>(memref->byte_offset_)) {
      fields.addr_ssa = GetOrEmitConstant(const_offset->value_, const_offset->dtype());
    } else if (memref->byte_offset_) {
      // A runtime address: a declared allocation's slot index scaled to a byte
      // offset (`l0c[i % 2]`) and added to the base by AllocateMemoryAddr. The
      // `alloc_tile` addr operand is i64, and PTOAS lowers it to a runtime
      // `TASSIGN(tile, addr)`, so an SSA value is as valid here as a constant.
      fields.addr_ssa = cast_to_i64(GetExprAsCode(memref->byte_offset_), memref->byte_offset_);
    }
  }
  return fields;
}

void PTOCodegen::PlanMultiBufferRegions(const FunctionPtr& func) {
  fs_.multi_buffer_regions.clear();
  fs_.multi_buffer_region_order.clear();

  // PyPTO planner (ptoas --pto-level=level3): ptoas fans an explicit base address
  // out into the per-slot addresses without folding them, so its multi-buffer slot
  // narrowing falls back to conservative aliasing — measurably worse there than the
  // baked-address alloc_tile path. See hw-native-sys/PTOAS#1106.
  if (emit_tile_addr_) return;

  TilePhiBaseCollector phi_collector;
  CoLiveSlotCollector colive_collector;
  if (func->body_) {
    phi_collector.VisitStmt(func->body_);
    colive_collector.VisitStmt(func->body_);
  }

  /// One allocation's slots, accumulated over every tile bound to it. `blocker`
  /// is empty while the allocation can still become a region, and otherwise says
  /// what stopped it — the author has to hear that, because under this planner a
  /// slotted declaration has no fallback that keeps its slots apart.
  struct Candidate {
    uint64_t count = 1;
    std::string slot_type_str;
    /// The valid extent every slot must share, taken from the reference tile.
    /// Held as plain values plus a flag rather than an optional: `blocker` is what
    /// decides whether they are usable, and an optional here reads as if a null
    /// extent were a state the emission below has to handle.
    bool has_extents = false;
    int64_t valid_row = 0;
    int64_t valid_col = 0;
    ir::VarPtr first_tile;      ///< Diagnostic anchor: the first tile seen
    ir::VarPtr reference_tile;  ///< The first slot-selecting tile: geometry
    std::string blocker;
  };
  std::map<const ir::Var*, Candidate> candidates;
  std::vector<const ir::Var*> discovery_order;

  // Pass 1: find the slotted allocations, and take each one's geometry from the
  // first tile that actually *selects* a slot. A tile that binds the allocation
  // whole is still recorded (pass 2 rejects the allocation for it) but must not
  // supply the baseline — it names no slot, so its type says nothing about what
  // the slots hold. Reading geometry from whichever tile came first would also
  // make the outcome depend on discovery order.
  for (const auto& [tile_var, tile_type] : fs_.tile_var_allocs) {
    auto memref = ir::GetDefinedMemRef(tile_type);
    if (memref->slot_count_ <= 1) continue;
    const ir::Var* base = memref->base_.get();
    auto [it, fresh] = candidates.try_emplace(base);
    Candidate& candidate = it->second;
    if (fresh) {
      discovery_order.push_back(base);
      candidate.first_tile = tile_var;
      // Count comes from any binding — they all read it off one declaration, and
      // InitMemRef has already rejected a disagreement — so the diagnostic below
      // states the declared count even when no tile selects a slot.
      candidate.count = memref->slot_count_;
    }
    if (candidate.reference_tile || !memref->slot_index_.has_value() || !*memref->slot_index_) continue;
    candidate.slot_type_str = GetTileBufTypeStringFromTileType(tile_type);
    const auto reference_extents = StaticValidExtents(tile_type);
    candidate.has_extents = reference_extents.has_value();
    candidate.valid_row = candidate.has_extents ? reference_extents->first : 0;
    candidate.valid_col = candidate.has_extents ? reference_extents->second : 0;
    candidate.reference_tile = tile_var;
  }
  if (candidates.empty()) return;

  // Pass 2: every tile on a slotted allocation has to select a slot of the same
  // type — ptoas requires `multi_tile_get`'s result to equal the region's slot
  // type, and a user that selects no slot (a view of the region, an unsubscripted
  // binding) wants an address the region hands out to nobody.
  for (const auto& [tile_var, tile_type] : fs_.tile_var_allocs) {
    auto memref = ir::GetDefinedMemRef(tile_type);
    auto it = candidates.find(memref->base_.get());
    if (it == candidates.end()) continue;
    Candidate& candidate = it->second;
    if (!candidate.blocker.empty()) continue;  // first reason wins
    const std::string type_str = GetTileBufTypeStringFromTileType(tile_type);
    // Compared separately from the type string, which renders `v_row=?, v_col=?`
    // by design: the region declares ONE valid extent for every slot, so two slots
    // that print alike but differ in valid_shape would silently give one of them
    // the other's extent.
    const auto extents = StaticValidExtents(tile_type);
    const bool static_extents = extents.has_value();
    const int64_t tile_valid_row = static_extents ? extents->first : 0;
    const int64_t tile_valid_col = static_extents ? extents->second : 0;

    if (!memref->slot_index_.has_value() || !*memref->slot_index_) {
      candidate.blocker = "tile '" + tile_var->name_hint_ + "' binds it without selecting a slot";
    } else if (memref->slot_count_ != candidate.count) {
      candidate.blocker = "its tiles disagree on how many slots it has";
    } else if (candidate.count > kMaxMultiTileBufSlots) {
      candidate.blocker = "ptoas supports " + std::to_string(kMinMultiTileBufSlots) + " to " +
                          std::to_string(kMaxMultiTileBufSlots) + " slots, and it declares " +
                          std::to_string(candidate.count);
    } else if (type_str != candidate.slot_type_str) {
      candidate.blocker = "its slots hold differently shaped tiles, and ptoas slots are uniform";
    } else if (!static_extents || !candidate.has_extents) {
      // Either this tile or the reference slot has a runtime extent; name the one
      // that does, since that is the annotation to change.
      const auto& offender = static_extents ? candidate.reference_tile : tile_var;
      candidate.blocker = "tile '" + offender->name_hint_ +
                          "' has a runtime valid shape, and a region declares one static extent for "
                          "all its slots";
    } else if (tile_valid_row != candidate.valid_row || tile_valid_col != candidate.valid_col) {
      candidate.blocker = "its slots declare different valid shapes (" + std::to_string(candidate.valid_row) +
                          "x" + std::to_string(candidate.valid_col) + " and " +
                          std::to_string(tile_valid_row) + "x" + std::to_string(tile_valid_col) +
                          "), and a region declares one valid extent for all of them";
    } else if (!IsMultiBufferMemorySpace(tile_type->memory_space_)) {
      candidate.blocker = "ptoas multi-buffer covers the Vec, Mat and Acc memory spaces only";
    } else if (phi_collector.bases.count(memref->base_.get()) != 0) {
      candidate.blocker = "one of its slots is carried out of an if or a loop as a phi";
    } else if (colive_collector.bases.count(memref->base_.get()) != 0) {
      candidate.blocker =
          "two of its slots are live at once inside a loop, and ptoas guards only the first slot "
          "selected in an iteration — the second would be read while the next iteration overwrites "
          "it (ptoas 0.54). Take one slot per iteration, which is the shape the region form "
          "accelerates";
    }
  }

  for (const ir::Var* base : discovery_order) {
    Candidate& candidate = candidates.at(base);
    // Degrading to one alloc_tile per slot would silently undo the separation the
    // author declared — ptoas would be free to plan the slots on top of each
    // other. Say what is unsupported instead.
    CHECK_SPAN(candidate.blocker.empty(), candidate.first_tile->span_)
        << "The declared allocation 'pl.MemRef(\"" << base->name_hint_ << "\", slots=" << candidate.count
        << ")' cannot be lowered to a ptoas multi-buffer region because " << candidate.blocker
        << ". Under memory_planner=PTOAS the slots have no other way to stay apart — adjust the "
           "declaration, or compile with the default PyPTO memory planner.";

    // The valid extent is stated once on the region: pass 2 established that every
    // slot agrees on it, and that it is static — the region is declared in the
    // function head, where a runtime extent's SSA value is not yet in scope.
    MultiBufferRegion region;
    region.valid_row_ssa = GetOrEmitConstant(candidate.valid_row, DataType::INDEX);
    region.valid_col_ssa = GetOrEmitConstant(candidate.valid_col, DataType::INDEX);
    region.count = candidate.count;
    region.slot_type_str = candidate.slot_type_str;
    region.mtb_type_str = FormatMultiTileBufTypeString(region.slot_type_str, region.count);
    region.region_ssa = NewNamedTemp(base->name_hint_ + "_mb");

    fs_.multi_buffer_regions.emplace(base, std::move(region));
    fs_.multi_buffer_region_order.push_back(base);
  }
}

const PTOCodegen::MultiBufferRegion* PTOCodegen::GetMultiBufferRegion(const ir::MemRefPtr& memref) const {
  if (!memref || fs_.multi_buffer_regions.empty()) return nullptr;
  auto it = fs_.multi_buffer_regions.find(memref->base_.get());
  return it != fs_.multi_buffer_regions.end() ? &it->second : nullptr;
}

bool PTOCodegen::TryEmitMultiTileGet(const ir::MemRefPtr& memref, const std::string& tile_buf,
                                     const ir::Span& span) {
  const MultiBufferRegion* region = GetMultiBufferRegion(memref);
  if (region == nullptr) return false;

  // Eligibility already established that every tile on this allocation selects a
  // slot, so a missing index here is a planning bug, not an unsupported program.
  INTERNAL_CHECK_SPAN(memref->slot_index_.has_value() && *memref->slot_index_, span)
      << "Internal error: MemRef on multi-buffer region '" << memref->base_->name_hint_
      << "' carries no slot index";
  const ExprPtr& slot_index = *memref->slot_index_;

  // ptoas reads the slot as an `index` SSA and matches its affine form (`iv % N`,
  // `(iv ± c) % N`, a constant) to decide whether two accesses can touch the same
  // slot. Passing the index itself — not the byte offset InitMemRef derived from
  // it — is what keeps that analysis, and with it the per-slot event ids.
  std::string slot_ssa;
  if (auto const_index = As<ir::ConstInt>(slot_index)) {
    slot_ssa = GetOrEmitConstant(const_index->value_, DataType::INDEX);
  } else {
    // Check the type before lowering: GetExprAsCode already writes the expression
    // out, so a rejection after it would leave dead code in the stream.
    auto scalar_type = As<ScalarType>(slot_index->GetType());
    CHECK_SPAN(scalar_type && (scalar_type->dtype_.IsInt() || scalar_type->dtype_ == DataType::INDEX), span)
        << "A slot index must be an integer or index expression, got "
        << (scalar_type ? GetTypeString(scalar_type->dtype_) : std::string("a non-scalar"));
    slot_ssa = GetExprAsCode(slot_index);
    if (scalar_type->dtype_ != DataType::INDEX) {
      std::string idx = NewTemp();
      Emit(idx + " = arith.index_cast " + slot_ssa + " : " + GetTypeString(scalar_type->dtype_) +
           " to index");
      slot_ssa = idx;
    }
  }

  Emit(tile_buf + " = pto.multi_tile_get " + region->region_ssa + "[" + slot_ssa +
       "] : " + region->mtb_type_str + " -> " + region->slot_type_str);
  fs_.ssa_to_tile_buf_type[tile_buf] = region->slot_type_str;
  return true;
}

void PTOCodegen::EmitMultiBufferRegionAllocs() {
  for (const ir::Var* base : fs_.multi_buffer_region_order) {
    const MultiBufferRegion& region = fs_.multi_buffer_regions.at(base);
    // No `addr`: ptoas PlanMemory owns the region's placement, which is the whole
    // point of describing the slots to it. Under the PyPTO planner no region is
    // planned at all (see PlanMultiBufferRegions). Both extents are always present —
    // planning rejects an allocation whose valid shape it cannot state statically.
    // Region declarations are synthesized from the whole allocation rather than
    // one statement, so the base variable's span is the closest true source.
    SpanScope base_loc(this, &base->span_);
    Emit(region.region_ssa + " = pto.alloc_multi_tile valid_row = " + region.valid_row_ssa +
         " valid_col = " + region.valid_col_ssa + " : " + region.mtb_type_str);
  }
}

void PTOCodegen::EmitAllocTileForVar(const ir::VarPtr& tile_var,
                                     const std::shared_ptr<const ir::TileType>& tile_type) {
  auto var_key = GetVarKey(tile_var);
  if (!fs_.emitted_tile_alloc_vars.insert(var_key).second) {
    return;
  }

  auto mlir_it = fs_.var_to_mlir.find(var_key);
  INTERNAL_CHECK_SPAN(mlir_it != fs_.var_to_mlir.end(), tile_var->span_)
      << "Tile var " << tile_var->name_hint_ << " not found in fs_.var_to_mlir";
  std::string tile_buf = mlir_it->second;

  // In PTOAS mode several vars may share one handle (in-place aliasing); emit
  // the alloc_tile only once per handle so the shared buffer has a single def.
  if (!fs_.emitted_tile_alloc_names.insert(tile_buf).second) {
    return;
  }

  // A slot of a declared multi-slot allocation is taken from its region rather
  // than allocated: one `pto.alloc_multi_tile` backs all N, and this use selects
  // one. Falls through to the ordinary alloc_tile when no region was planned for
  // the allocation — it declares no slots, or the PyPTO planner is in use. A
  // slotted allocation this planner cannot describe never gets this far; see
  // PlanMultiBufferRegions.
  if (TryEmitMultiTileGet(ir::GetDefinedMemRef(tile_type), tile_buf, tile_var->span_)) {
    return;
  }

  AllocTileFields fields = ComputeAllocTileFields(tile_type);

  std::ostringstream line;
  line << tile_buf << " = pto.alloc_tile";
  if (!fields.addr_ssa.empty()) line << " addr = " << fields.addr_ssa;
  if (!fields.valid_row_ssa.empty()) line << " valid_row = " << fields.valid_row_ssa;
  if (!fields.valid_col_ssa.empty()) line << " valid_col = " << fields.valid_col_ssa;
  line << " : " << fields.type_str;
  Emit(line.str());

  fs_.ssa_to_tile_buf_type[tile_buf] = fields.type_str;
}

// ========================================================================
// Private helpers
// ========================================================================

std::string PTOCodegen::GetIndent() const { return std::string(static_cast<size_t>(indent_level_) * 2, ' '); }

std::string PTOCodegen::GetOrEmitConstant(int64_t value, DataType dt) {
  auto key = std::make_pair(value, dt.Code());
  auto it = fs_.emitted_numeric_constants.find(key);
  if (it != fs_.emitted_numeric_constants.end()) return it->second;

  std::string mlir_type = GetTypeString(dt);
  // MLIR's arith.constant requires signless integer return types (upstream
  // ArithOps.cpp ConstantOp::verify). For unsigned dtypes, emit the constant
  // at the signless type and bridge to the unsigned type via
  // builtin.unrealized_conversion_cast; some consumer ops (e.g. pto.tci) in
  // turn require the operand type to match the destination dtype exactly.
  bool is_unsigned = dt.IsUnsignedInt() && !mlir_type.empty() && mlir_type[0] == 'u';
  std::string signless_type = is_unsigned ? mlir_type.substr(1) : mlir_type;
  std::string ssa_suffix = "_" + mlir_type;

  std::string ssa_id;
  if (value == 0) {
    ssa_id = "c0" + ssa_suffix;
  } else if (value < 0) {
    uint64_t mag = static_cast<uint64_t>(-(value + 1)) + 1;
    ssa_id = "cn" + std::to_string(mag) + ssa_suffix;
  } else {
    ssa_id = "c" + std::to_string(value) + ssa_suffix;
  }

  std::string name;
  if (!fs_.used_ssa_names.count(ssa_id)) {
    fs_.used_ssa_names.insert(ssa_id);
    name = "%" + ssa_id;
  } else {
    name = NewTemp();
  }

  if (is_unsigned) {
    std::string signless_name = NewTemp();
    fs_.constants_section << fs_.constants_indent << signless_name << " = arith.constant " << value << " : "
                          << signless_type << "\n";
    fs_.constants_section << fs_.constants_indent << name << " = builtin.unrealized_conversion_cast "
                          << signless_name << " : " << signless_type << " to " << mlir_type << "\n";
  } else {
    fs_.constants_section << fs_.constants_indent << name << " = arith.constant " << value << " : "
                          << mlir_type << "\n";
  }
  fs_.emitted_numeric_constants[key] = name;
  return name;
}

std::string PTOCodegen::GetOrEmitConstant(double value, DataType dt) {
  int64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  auto key = std::make_pair(bits, dt.Code());
  auto it = fs_.emitted_numeric_constants.find(key);
  if (it != fs_.emitted_numeric_constants.end()) return it->second;

  std::string mlir_type = GetTypeString(dt);
  std::string ssa_id = "cst";
  if (!fs_.emitted_numeric_constants.empty()) {
    ssa_id += "_" + std::to_string(fs_.emitted_numeric_constants.size());
  }
  std::string name;
  if (!fs_.used_ssa_names.count(ssa_id)) {
    fs_.used_ssa_names.insert(ssa_id);
    name = "%" + ssa_id;
  } else {
    name = NewTemp();
  }
  std::ostringstream val_str;
  val_str << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  fs_.constants_section << fs_.constants_indent << name << " = arith.constant " << val_str.str() << " : "
                        << mlir_type << "\n";
  fs_.emitted_numeric_constants[key] = name;
  return name;
}

std::string PTOCodegen::GetTileBufForMemRef(const MemRefPtr& memref) const {
  INTERNAL_CHECK(memref != nullptr) << "Internal error: null MemRef passed to GetTileBufForMemRef";
  auto it = fs_.memref_to_mlir.find(memref->base_.get());
  INTERNAL_CHECK_SPAN(it != fs_.memref_to_mlir.end(), memref->span_)
      << "Internal error: no MLIR mapping for MemRef base '" << memref->base_->name_hint_ << "'";
  return it->second;
}

std::string PTOCodegen::AllocNewTileBuf(const std::string& tile_buf_type_string, const std::string& name_hint,
                                        const std::string& addr_ssa, const std::string& valid_row_ssa,
                                        const std::string& valid_col_ssa) {
  std::string name = NewNamedTemp(name_hint);
  fs_.extra_alloc_tiles.push_back(
      FunctionState::ExtraAllocTile{name, tile_buf_type_string, addr_ssa, valid_row_ssa, valid_col_ssa});
  fs_.ssa_to_tile_buf_type[name] = tile_buf_type_string;
  return name;
}

std::string PTOCodegen::TryGetSharedTileBufHandle(const ir::MemRefPtr& memref) const {
  if (emit_tile_addr_ || !memref) {
    return "";
  }
  const std::string ident = MemRefIdentityKey(memref);
  // A mixed-type identity's handle already carries another var's type; re-typing
  // it would make one SSA value have two types and ptoas would reject the module.
  if (fs_.memref_identity_mixed_types.count(ident) != 0) {
    return "";
  }
  auto it = fs_.memref_identity_to_mlir.find(ident);
  return it != fs_.memref_identity_to_mlir.end() ? it->second : std::string{};
}

bool PTOCodegen::DeclareTileBufAtHead(const std::string& ssa_name, const AllocTileFields& fields) {
  if (!fs_.emitted_tile_alloc_names.insert(ssa_name).second) {
    return false;  // already declared — in the head, or inline earlier in the body
  }
  fs_.extra_alloc_tiles.push_back(FunctionState::ExtraAllocTile{ssa_name, fields.type_str, fields.addr_ssa,
                                                                fields.valid_row_ssa, fields.valid_col_ssa});
  fs_.ssa_to_tile_buf_type[ssa_name] = fields.type_str;
  return true;
}

void PTOCodegen::SetCurrentResultBuf(const std::string& buf) { fs_.current_result_buf = buf; }

void PTOCodegen::RegisterTileBufType(const std::string& ssa_name, const std::string& type_string) {
  fs_.ssa_to_tile_buf_type[ssa_name] = type_string;
}

std::string PTOCodegen::GetSSATileBufType(const std::string& ssa_name) const {
  auto it = fs_.ssa_to_tile_buf_type.find(ssa_name);
  return it != fs_.ssa_to_tile_buf_type.end() ? it->second : std::string{};
}

void PTOCodegen::RegisterTileViewName(const std::string& ssa_name) { fs_.tile_view_names.insert(ssa_name); }

bool PTOCodegen::IsTileViewName(const std::string& ssa_name) const {
  return fs_.tile_view_names.count(ssa_name) > 0;
}

void PTOCodegen::RegisterSubviewMaterialization(const std::string& subview_ssa,
                                                const SubviewMaterializationInfo& info) {
  fs_.subview_materializations[subview_ssa] = info;
}

PTOCodegen::SubviewMaterializationInfo* PTOCodegen::GetSubviewMaterialization(
    const std::string& subview_ssa) {
  auto it = fs_.subview_materializations.find(subview_ssa);
  return it != fs_.subview_materializations.end() ? &it->second : nullptr;
}

const PTOCodegen::SubviewMaterializationInfo* PTOCodegen::GetSubviewMaterialization(
    const std::string& subview_ssa) const {
  auto it = fs_.subview_materializations.find(subview_ssa);
  return it != fs_.subview_materializations.end() ? &it->second : nullptr;
}

void PTOCodegen::RecordGMSlotBufferSSA(const std::string& ssa, const DataType& dtype) {
  CHECK(dtype == DataType::FP32) << "__gm_pipe_buffer must use FP32 elements, got " << dtype.ToString();
  fs_.gm_slot_buffer_ssa = ssa;
  fs_.gm_slot_buffer_dtype = dtype;
}

std::string PTOCodegen::GetGMSlotBufferSSA() const { return fs_.gm_slot_buffer_ssa; }

std::string PTOCodegen::GetCommCtxSSAFor(const ir::Var* dist_var) const {
  if (dist_var == nullptr) return "";
  auto it = fs_.dist_tensor_to_ctx.find(dist_var);
  if (it != fs_.dist_tensor_to_ctx.end()) return it->second;
  if (auto iter_arg = dynamic_cast<const ir::IterArg*>(dist_var)) {
    if (auto init_var = AsVarLike(iter_arg->initValue_)) return GetCommCtxSSAFor(init_var.get());
  }
  return "";
}

void PTOCodegen::RegisterCommCtxFor(const ir::VarPtr& dist_var, const std::string& ctx_ssa) {
  if (!dist_var || ctx_ssa.empty()) return;
  fs_.dist_tensor_to_ctx[GetVarKey(dist_var)] = ctx_ssa;
}

std::string PTOCodegen::GetGMSlotBufferSSAForPipe(int pipe_id, int dir_mask) {
  if (fs_.gm_slot_buffer_ssa.empty()) {
    return "";
  }

  const auto key = std::make_pair(pipe_id, dir_mask);
  auto it = fs_.gm_slot_buffer_region_by_pipe.find(key);
  if (it != fs_.gm_slot_buffer_region_by_pipe.end()) {
    return it->second;
  }

  auto offset_it = gm_slot_buffer_offsets_.find(key);
  INTERNAL_CHECK(offset_it != gm_slot_buffer_offsets_.end())
      << "Internal error: missing GM slot buffer offset for frontend pipe id " << pipe_id << " and dir_mask "
      << dir_mask;
  const int64_t byte_offset = offset_it->second;

  std::string region_ssa = fs_.gm_slot_buffer_ssa;
  if (byte_offset != 0) {
    const auto element_bytes = static_cast<int64_t>((fs_.gm_slot_buffer_dtype.GetBit() + 7) / 8);
    CHECK(element_bytes > 0) << "Unsupported __gm_pipe_buffer dtype: " << fs_.gm_slot_buffer_dtype.ToString();
    CHECK(byte_offset % element_bytes == 0)
        << "GM slot buffer byte offset must be aligned to " << fs_.gm_slot_buffer_dtype.ToString()
        << " elements, got " << byte_offset;
    const int64_t elem_offset = byte_offset / element_bytes;
    std::string offset_ssa = GetOrEmitConstant(elem_offset, DataType::INDEX);
    region_ssa = NewTemp();
    const std::string elem_type = GetTypeString(fs_.gm_slot_buffer_dtype);
    Emit(region_ssa + " = pto.addptr " + fs_.gm_slot_buffer_ssa + ", " + offset_ssa + " : <" + elem_type +
         "> -> <" + elem_type + ">");
  }

  fs_.gm_slot_buffer_region_by_pipe[key] = region_ssa;
  return region_ssa;
}

bool PTOCodegen::IsAICFunction() const {
  return fs_.current_function && fs_.current_function->func_type_ == ir::FunctionType::AIC;
}

bool PTOCodegen::IsAIVFunction() const {
  return fs_.current_function && fs_.current_function->func_type_ == ir::FunctionType::AIV;
}

bool PTOCodegen::IsDualAivDispatchFunction() const {
  return fs_.current_function && fs_.current_function->HasAttr(ir::kAttrDualAivDispatch) &&
         fs_.current_function->GetAttr<bool>(ir::kAttrDualAivDispatch, false);
}

void PTOCodegen::EmitExtraAllocTiles() {
  // Regions first: every `pto.multi_tile_get` in the body reads one, so the
  // declaration has to dominate them all.
  EmitMultiBufferRegionAllocs();
  // These allocations are hoisted out of the body (e.g. reshape outputs), so no
  // single statement owns them; the function is the closest true source.
  SpanScope func_loc(this, fs_.current_function ? &fs_.current_function->span_ : nullptr);
  for (const auto& alloc : fs_.extra_alloc_tiles) {
    std::ostringstream line;
    line << alloc.name << " = pto.alloc_tile";
    if (emit_tile_addr_ && !alloc.addr_ssa.empty()) {
      line << " addr = " << alloc.addr_ssa;
    }
    if (!alloc.valid_row_ssa.empty()) {
      line << " valid_row = " << alloc.valid_row_ssa;
    }
    if (!alloc.valid_col_ssa.empty()) {
      line << " valid_col = " << alloc.valid_col_ssa;
    }
    line << " : " << alloc.type_string;
    Emit(line.str());
  }
}

// ========================================================================
// Statement visitors
// ========================================================================

void PTOCodegen::VisitStmt(const ir::StmtPtr& stmt) {
  // Defensive: the first-class SplitAivScopeStmt region is consumed and erased
  // by LowerAutoVectorSplit (pass 20), well before codegen. There is no
  // ScopeStmt handler here, so a survivor would be silently unwrapped by the
  // base visitor — losing the region semantics. Fail loudly instead.
  INTERNAL_CHECK_SPAN(!ir::As<ir::SplitAivScopeStmt>(stmt), stmt->span_)
      << "Internal error: SplitAivScopeStmt reached PTO codegen; it must be lowered and erased by "
         "LowerAutoVectorSplit (pass 20).";
  // Primary location source: every op lowered under this statement is attributed
  // to the statement's source line unless a nested Call refines it (see
  // VisitExpr_(CallPtr)). The statement span is what passes reliably preserve —
  // they frequently rebuild the Call underneath it with a coarser span.
  SpanScope stmt_loc(this, &stmt->span_);
  ir::IRVisitor::VisitStmt(stmt);
}

void PTOCodegen::VisitStmt_(const AssignStmtPtr& op) {
  auto call = As<ir::Call>(op->value_);
  const bool is_set_validshape = ir::IsOp(call, "tile.set_validshape");
  const bool alias_result_to_in_place_input = ShouldAliasResultToInPlaceInput(op);
  const bool alias_array_update_to_input = ShouldAliasArrayUpdateResultToInput(op);

  if (ir::IsOp(call, "pld.tile.remote_load")) {
    auto result_tile_type = As<ir::TileType>(op->var_->GetType());
    INTERNAL_CHECK_SPAN(result_tile_type, call->span_)
        << "Internal error: pld.tile.remote_load result must be a TileType";
    const auto result_tile_view = ir::tile_view_semantics::GetEffectiveTileView(*result_tile_type);
    CheckExprVarsBound(result_tile_view.valid_shape, call->span_,
                       "pld.tile.remote_load inferred valid_shape");
  }

  if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
    if (!is_set_validshape && !alias_result_to_in_place_input) {
      EmitAllocTileForVar(op->var_, tile_type);
    }
  }

  if (call) {
    if (backend_ != nullptr && backend_->GetOpInfo(call->op_->name_) != nullptr) {
      std::string result_buf =
          op->var_->name_hint_;  // Seed for readable MLIR names when no tile buffer exists.
      std::shared_ptr<const TileType> result_tile_type;
      if (auto tile_type = ir::GetTileTypeWithMemRef(op->var_->GetType())) {
        if (alias_result_to_in_place_input) {
          result_buf = GetExprAsCode(call->args_[0]);
          INTERNAL_CHECK(!result_buf.empty())
              << "Internal error: " << call->op_->name_ << " result must alias the input tile SSA";
          BindVarToMlir(op->var_, result_buf);
        } else {
          // Prefer per-var SSA name from fs_.var_to_mlir (set during per-var alloc binding)
          auto var_it = fs_.var_to_mlir.find(GetVarKey(op->var_));
          if (var_it != fs_.var_to_mlir.end()) {
            result_buf = var_it->second;
          } else {
            result_buf = GetTileBufForMemRef(ir::GetDefinedMemRef(tile_type));
          }
        }
        result_tile_type = tile_type;
      } else if (auto tile_type = As<TileType>(op->var_->GetType())) {
        // A MemRef-less tile result (e.g. a cross-core tpop result, whose data
        // lives in the reserved C2V/V2C slot) still needs a %-SSA name bound so
        // consumers resolve it; its tile_buf type comes from the TileType since
        // there is no MemRef to read. Register it before the op codegen runs so
        // GetCurrentResultTileBufTypeString() can emit the `-> type` annotation.
        result_tile_type = tile_type;
        result_buf = NewNamedTemp(op->var_->name_hint_);
        BindVarToMlir(op->var_, result_buf);
        std::string type_str = GetTileBufTypeStringFromTileType(tile_type);
        if (!type_str.empty()) {
          fs_.ssa_to_tile_buf_type[result_buf] = type_str;
        }
      } else if (alias_array_update_to_input) {
        // array.update_element: alias the result Var to the input array's SSA so
        // the emitted pto.local_array_set mutates the same declare_local_array
        // storage in place (mirrors the SSA-functional -> in-place lowering).
        result_buf = GetExprAsCode(call->args_[0]);
        INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_)
            << "Internal error: array.update_element result must alias the input array SSA";
        BindVarToMlir(op->var_, result_buf);
      } else {
        // Pre-allocate a %-prefixed SSA name for non-tile backend ops (e.g., scalar
        // results like tile.getval, or i32 results like reserve_buffer / import_peer_buffer).
        // Register it in fs_.var_to_mlir so subsequent expressions can resolve the variable.
        result_buf = NewNamedTemp(op->var_->name_hint_);
        BindVarToMlir(op->var_, result_buf);
      }
      fs_.current_result_var = op->var_;
      fs_.current_result_buf = result_buf;
      fs_.current_result_tile_type = result_tile_type;
      VisitExpr(op->value_);
      // If codegen changed the result buffer (e.g., reshape allocated a new tile),
      // update variable mapping so subsequent references use the new buffer
      if (!fs_.current_result_buf.empty() && (is_set_validshape || fs_.current_result_buf != result_buf)) {
        BindVarToMlir(op->var_, fs_.current_result_buf);
      }
      fs_.current_result_var.reset();
      fs_.current_result_buf.clear();
      fs_.current_result_tile_type = nullptr;
      return;
    }
  }

  // Plain tensor alias: `lhs_tensor = rhs_var` with no Call on the RHS. This
  // arises when Simplify folds an empty loop-result ForStmt into a plain
  // AssignStmt — e.g. a constant-trip `pl.pipeline`'s statically-empty main
  // loop becomes `t__rv_vN_main = t__iter_vM`. The deleted ForStmt would have
  // registered the loop-result tensor view / SSA name for its return var (see
  // VisitStmt_(ForStmtPtr) in pto_control_flow_codegen.cpp), so a later
  // tile.store into the alias can resolve its view instead of
  // GetOrCreateTensorView tripping its INTERNAL_CHECK on the synthetic var.
  // We additionally propagate the base-ptr mapping so element-wise alias
  // consumers (pl.read / pl.write / store_scalar) resolve to the backing
  // pointer rather than the view SSA — as the IfStmt in-place-return path
  // (VisitStmt_(IfStmtPtr)) does for merged tensors.
  // Non-fatal: if the RHS has no registered view, fall through to the generic
  // handling rather than throwing eagerly on a view that may never be consumed.
  if (auto rhs_var = AsVarLike(op->value_)) {
    if (ir::AsTensorTypeLike(op->var_->GetType())) {
      const std::string view = TryGetTensorView(rhs_var);
      if (!view.empty()) {
        BindTensorView(op->var_, view);
        BindVarToMlir(op->var_, view);  // view name == SSA name, as in ForStmt
        RegisterBasePtr(op->var_, GetTensorBasePtr(rhs_var));
        const std::string comm_ctx = GetCommCtxSSAFor(rhs_var.get());
        if (As<ir::DistributedTensorType>(op->var_->GetType())) {
          INTERNAL_CHECK_SPAN(!comm_ctx.empty(), op->span_)
              << "Internal error: DistributedTensor alias '" << op->var_->name_hint_ << "' from source '"
              << rhs_var->name_hint_ << "' has no CommContext binding";
        }
        RegisterCommCtxFor(op->var_, comm_ctx);
        return;
      }
    } else if (!emit_tile_addr_ && As<TileType>(op->var_->GetType())) {
      // Bare tile SSA alias (`lhs = rhs`) under memory_planner=PTOAS. `lhs` and
      // `rhs` denote the identical tile value, so `lhs` must resolve to `rhs`'s
      // CURRENT SSA binding. This matters when `rhs` is a view (tile.reshape /
      // tile.transpose_view) that re-pointed itself at a typed view SSA of a
      // shared buffer — e.g. the `[N, 1]` col-major reshape of a `[1, N]`
      // row-major op result, which shares the op result's MemRef. `lhs` was
      // pre-bound (GenerateFunction) to that shared handle, whose SSA is typed
      // `[1, N]`; keeping it makes a later yield of `lhs` (an `m = m_new`
      // online-softmax carry) emit a `[1, N] -> [N, 1]` write-back tmov that
      // ptoas rejects for shape mismatch. Following `rhs` binds `lhs` to the
      // `[N, 1]` view SSA so the write-back has matching src/dst shapes — the
      // same shape the `s = pl.mul(...)`-style yield (no bare alias) already
      // gets. Under PyPTO (emit_tile_addr_) the baked address already aliases
      // the two allocs, so this is a no-op there and is left untouched.
      const std::string rhs_ssa = LookupVarName(rhs_var);
      if (!rhs_ssa.empty()) {
        BindVarToMlir(op->var_, rhs_ssa);
        return;
      }
    }
  }

  fs_.current_expr_value = "";
  VisitExpr(op->value_);
  // Register scalar/index/CommCtx result so subsequent expressions can look up
  // this variable. N7: CommCtxType is a singleton marker; the bound SSA is the
  // matching explicit ``!pto.ptr<i64>`` ctx ptr from the func.func signature
  // (no MLIR is emitted for ``pld.system.get_comm_ctx`` — its lambda just sets
  // ``current_expr_value`` to the ctx SSA). Treating it like a scalar here lets
  // downstream ``pld.system.rank(ctx)`` / ``pld.system.nranks(ctx)`` codegen
  // resolve ``ctx`` via the standard ``GetExprAsCode(call->args_[0])`` path.
  const auto& var_type = op->var_->GetType();
  if ((As<ScalarType>(var_type) || As<CommCtxType>(var_type)) && !fs_.current_expr_value.empty()) {
    BindVarToMlir(op->var_, fs_.current_expr_value);
  }
}

// ========================================================================
// Expression visitors
// ========================================================================

void PTOCodegen::VisitExpr_(const CallPtr& op) {
  const std::string& op_name = op->op_->name_;

  CHECK(backend_ != nullptr) << "Backend must not be null; use PTOCodegen(backend) or default backend";
  const auto* op_info = backend_->GetOpInfo(op_name);
  if (op_info == nullptr) {
    ThrowNoCodegenForCall(op_name);
  }
  // Refine to the Call's own (column-accurate) span when it is genuinely nested
  // in the enclosing statement; otherwise keep the statement span, which is the
  // trustworthy one for any Call a pass rebuilt.
  SpanScope call_loc(this, SpanContains(current_span_, op->span_) ? &op->span_ : nullptr);
  std::string mlir_line = op_info->codegen_func(op, *this);
  if (!mlir_line.empty()) {
    Emit(mlir_line);
  }
}

// ========================================================================
// CodegenBase interface and PTO-specific helper methods
// ========================================================================

std::string PTOCodegen::GetCurrentResultTarget() const { return fs_.current_result_buf; }

ir::VarPtr PTOCodegen::GetCurrentResultVar() const { return fs_.current_result_var; }

std::vector<ir::VarPtr> PTOCodegen::ResolveTupleResultElements(const ir::VarPtr& tuple_var,
                                                               size_t arity) const {
  INTERNAL_CHECK(tuple_var) << "Internal error: ResolveTupleResultElements requires non-null tuple_var";
  INTERNAL_CHECK(fs_.current_function)
      << "Internal error: ResolveTupleResultElements requires current_function";
  TupleConsumerCollector collector(tuple_var.get(), arity);
  collector.VisitStmt(fs_.current_function->body_);
  return collector.elements();
}

void PTOCodegen::Emit(const std::string& line) { stream_ << GetIndent() << line << LocSuffix() << "\n"; }

void PTOCodegen::EmitStructural(const std::string& line) { stream_ << GetIndent() << line << "\n"; }

std::string PTOCodegen::LocSuffix() const {
  if (!emit_source_loc_ || current_span_ == nullptr) return "";
  const ir::Span& span = *current_span_;
  // Without a filename there is nothing to attribute, and MLIR's FileLineColLoc
  // needs non-negative coordinates. Emitting nothing leaves the op exactly as it
  // was before locations existed (ptoas then reports the .pto line) — strictly
  // better than a misleading `loc("":0:0)`.
  if (span.filename_.empty() || !span.is_valid()) return "";
  // is_valid() admits an unknown column (-1); MLIR does not.
  const int column = span.begin_column_ > 0 ? span.begin_column_ : 1;
  return " loc(\"" + EscapeMlirString(span.filename_) + "\":" + std::to_string(span.begin_line_) + ":" +
         std::to_string(column) + ")";
}

std::string PTOCodegen::GetExprAsCode(const ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    return GetVarName(var);
  }
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return GetOrEmitConstant(const_int->value_, const_int->dtype());
  }
  if (auto const_float = As<ir::ConstFloat>(expr)) {
    return GetOrEmitConstant(const_float->value_, const_float->dtype());
  }

  // Fall back to visitor pattern for complex expressions (arithmetic, comparisons)
  fs_.current_expr_value = "";
  VisitExpr(expr);
  std::string result = fs_.current_expr_value;
  fs_.current_expr_value = "";
  if (!result.empty()) {
    return result;
  }

  LOG_ERROR << "GetExprAsCode for unsupported expression type";
  return "";
}

std::string PTOCodegen::GetTypeString(const DataType& dtype) const {
  const auto* handler = GetBackendHandler();
  INTERNAL_CHECK(handler) << "PTOCodegen requires a backend handler";
  if (!handler->SupportsIncoreDataType(dtype)) {
    const std::string arch = handler->GetPtoTargetArch();
    if (arch == "a2a3" && dtype.GetBit() == 4) {
      CHECK(false) << "The 4-bit dtype " << dtype.ToString()
                   << " is not supported for end-to-end in-core codegen on backend 'a2a3'. "
                      "A2/A3 exposes only an isolated FP16<->INT4 conversion, while direct packed "
                      "4-bit load/store and carrier ABI are unavailable";
    }
    CHECK(false) << "The 4-bit dtype " << dtype.ToString()
                 << " is not supported for end-to-end in-core codegen on backend '" << arch
                 << "'; A5 currently supports only FP4 among 4-bit dtypes";
  }
  return DataTypeToMLIR(dtype);
}

const ir::Var* PTOCodegen::GetVarKey(const VarPtr& var) const {
  INTERNAL_CHECK(var != nullptr) << "Internal error: variable key requested for null Var";
  return var.get();
}

void PTOCodegen::CheckExprVarsBound(const std::vector<ir::ExprPtr>& exprs, const ir::Span& span,
                                    const std::string& context) const {
  ir::var_collectors::VarDefUseCollector collector;
  for (const auto& expr : exprs) {
    collector.VisitExpr(expr);
  }
  for (const auto* var : collector.var_uses) {
    CHECK_SPAN(fs_.var_to_mlir.count(var) != 0, span)
        << context << " depends on unbound symbol '" << var->name_hint_
        << "'; pass a scalar, loop, or physical tensor-shape value that is available in the kernel";
  }
}

void PTOCodegen::BindVarToMlir(const VarPtr& var, const std::string& mlir_name) {
  fs_.var_to_mlir[GetVarKey(var)] = mlir_name;
}

void PTOCodegen::BindTensorView(const VarPtr& var, const std::string& tensor_view_name) {
  fs_.tensor_to_view[GetVarKey(var)] = tensor_view_name;
}

void PTOCodegen::BindVarToMemRef(const VarPtr& var, const ir::Var* base_ptr) {
  fs_.var_to_memref[GetVarKey(var)] = base_ptr;
}

std::string PTOCodegen::LookupVarName(const VarPtr& var) const {
  auto key = GetVarKey(var);
  auto it = fs_.var_to_mlir.find(key);
  if (it != fs_.var_to_mlir.end()) {
    return it->second;
  }
  auto memref_it = fs_.var_to_memref.find(key);
  if (memref_it != fs_.var_to_memref.end()) {
    auto mlir_it = fs_.memref_to_mlir.find(memref_it->second);
    if (mlir_it != fs_.memref_to_mlir.end()) {
      return mlir_it->second;
    }
  }
  if (auto tile_type = ir::GetTileTypeWithMemRef(var->GetType())) {
    return GetTileBufForMemRef(ir::GetDefinedMemRef(tile_type));
  }
  for (const auto& [mapped_var, mlir_name] : fs_.var_to_mlir) {
    if (mapped_var && mapped_var->name_hint_ == var->name_hint_) {
      return mlir_name;
    }
  }
  return "";
}

std::string PTOCodegen::DescribeUnbindableSymbol(const VarPtr& var) const {
  auto origin = fs_.valid_shape_symbol_origin.find(GetVarKey(var));
  if (origin != fs_.valid_shape_symbol_origin.end()) {
    return "it appears only in the valid_shape of parameter '" + origin->second +
           "', and MaterializeValidShapeSymbols did not turn it into a parameter (that pass runs "
           "last in the Default strategy — a custom pass list that omits it leaves the symbol "
           "unbound)";
  }
  return "it is not a physical tensor dimension, a scalar parameter, or a loop variable of this "
         "kernel";
}

std::string PTOCodegen::GetVarName(const VarPtr& var) const {
  // An unresolvable symbol must fail here. Emitting an empty operand instead
  // produces MLIR that ptoas rejects with an opaque "expected SSA operand"
  // several stages downstream, far from the annotation that caused it.
  std::string name = LookupVarName(var);
  CHECK_SPAN(!name.empty(), var->span_)
      << "PTO codegen cannot materialize symbol '" << var->name_hint_ << "' in function '"
      << (fs_.current_function ? fs_.current_function->name_ : "<unknown>")
      << "': " << DescribeUnbindableSymbol(var)
      << ". Pass the extent as a pl.Scalar[pl.INDEX] parameter and use it in "
         "pl.load(..., valid_shape=[...]) instead of naming it in the parameter's "
         "pl.TensorView(valid_shape=...) annotation.";
  return name;
}

std::string PTOCodegen::NewTemp() {
  std::string name = std::to_string(fs_.temp_counter++);
  while (fs_.used_ssa_names.count(name)) {
    name = std::to_string(fs_.temp_counter++);
  }
  fs_.used_ssa_names.insert(name);
  return "%" + name;
}

std::string PTOCodegen::NewNamedTemp(const std::string& name) {
  // Sanitize name to be a valid MLIR SSA identifier: [a-zA-Z_][a-zA-Z0-9_$.]*
  std::string sanitized = name;
  if (!sanitized.empty()) {
    for (auto& c : sanitized) {
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '.' && c != '$') {
        c = '_';
      }
    }
    if (std::isdigit(static_cast<unsigned char>(sanitized[0]))) {
      sanitized.insert(0, 1, '_');
    }
  }

  if (!sanitized.empty() && fs_.used_ssa_names.find(sanitized) == fs_.used_ssa_names.end()) {
    fs_.used_ssa_names.insert(sanitized);
    return "%" + sanitized;
  }
  return NewTemp();
}

void PTOCodegen::RegisterVarToMlir(const VarPtr& var, const std::string& mlir_name) {
  BindVarToMlir(var, mlir_name);
}

void PTOCodegen::RegisterTensorView(const VarPtr& var, const std::string& tensor_view_name) {
  BindTensorView(var, tensor_view_name);
}

void PTOCodegen::RegisterBasePtr(const VarPtr& var, const std::string& ptr_name) {
  if (var && !ptr_name.empty()) fs_.tensor_to_base_ptr[GetVarKey(var)] = ptr_name;
}

std::string PTOCodegen::GetTensorBasePtr(const VarPtr& tensor) const {
  auto it = fs_.tensor_to_base_ptr.find(GetVarKey(tensor));
  if (it != fs_.tensor_to_base_ptr.end()) return it->second;
  // For IterArg, follow initValue_ to the original tensor parameter (mirrors GetOrCreateTensorView).
  if (auto iter_arg = As<ir::IterArg>(tensor)) {
    if (auto init_var = AsVarLike(iter_arg->initValue_)) return GetTensorBasePtr(init_var);
  }
  return GetVarName(tensor);
}

int64_t PTOCodegen::GetConstIntValue(const ExprPtr& expr) const {
  if (auto const_int = As<ir::ConstInt>(expr)) {
    return const_int->value_;
  }
  LOG_ERROR << "Expected ConstInt expression";
  return 0;
}

std::string PTOCodegen::TryGetTensorView(const VarPtr& tensor_var) const {
  auto it = fs_.tensor_to_view.find(GetVarKey(tensor_var));
  if (it != fs_.tensor_to_view.end()) return it->second;
  // For IterArg, follow initValue_ chain to the original tensor parameter.
  if (auto iter_arg = As<ir::IterArg>(tensor_var)) {
    if (auto init_var = As<ir::Var>(iter_arg->initValue_)) return TryGetTensorView(init_var);
    if (auto init_iter = As<ir::IterArg>(iter_arg->initValue_)) return TryGetTensorView(init_iter);
  }
  return "";
}

bool PTOCodegen::NoteCacheBypassWarned(const ir::Var* tensor) {
  INTERNAL_CHECK(tensor != nullptr) << "Internal error: null tensor passed to NoteCacheBypassWarned";
  return fs_.cache_bypass_warned.insert(tensor).second;
}

std::string PTOCodegen::GetOrCreateTensorView(const VarPtr& tensor_var) {
  std::string view = TryGetTensorView(tensor_var);
  INTERNAL_CHECK_SPAN(!view.empty(), tensor_var->span_)
      << "Tensor view not found for parameter: " << tensor_var->name_hint_;
  return view;
}

std::string PTOCodegen::GetTensorViewTypeString(const ir::TensorType* tensor_type) const {
  std::ostringstream oss;
  oss << "!pto.tensor_view<";
  for (size_t i = 0; i < tensor_type->shape_.size(); i++) {
    if (i > 0) oss << "x";
    oss << "?";
  }
  oss << "x" << GetTypeString(tensor_type->dtype_) << ">";
  return oss.str();
}

std::string PTOCodegen::GetTileBufTypeString(const ir::Var* base_ptr) const {
  INTERNAL_CHECK(base_ptr != nullptr) << "Internal error: null base_ptr passed to GetTileBufTypeString";
  auto tile_it = fs_.memref_to_tile_type.find(base_ptr);
  INTERNAL_CHECK_SPAN(tile_it != fs_.memref_to_tile_type.end(), base_ptr->span_)
      << "Internal error: missing tile type for base Ptr '" << base_ptr->name_hint_ << "'";
  auto memory_space = tile_it->second->GetMemorySpace();
  INTERNAL_CHECK_SPAN(memory_space.has_value(), base_ptr->span_)
      << "Internal error: tile type must have memory_space";

  std::string loc = MemorySpaceToMLIR(*memory_space);
  auto c = ExtractTileTypeInfo(*tile_it->second, GetTypeString(tile_it->second->dtype_));
  return FormatTileBufTypeString(loc, c.dtype_str, c.rows, c.cols, c.blayout, c.slayout, c.fractal, c.pad,
                                 c.compact, c.v_row, c.v_col, c.v_row_dynamic, c.v_col_dynamic);
}

std::string PTOCodegen::GetTileBufTypeStringFromTileType(
    const std::shared_ptr<const ir::TileType>& tile_type) const {
  INTERNAL_CHECK(tile_type) << "Internal error: tile_type must not be null";
  auto memory_space = tile_type->GetMemorySpace();
  INTERNAL_CHECK(memory_space.has_value()) << "Internal error: tile_type must have memory_space";

  std::string loc = MemorySpaceToMLIR(*memory_space);
  auto c = ExtractTileTypeInfo(*tile_type, GetTypeString(tile_type->dtype_));
  return FormatTileBufTypeString(loc, c.dtype_str, c.rows, c.cols, c.blayout, c.slayout, c.fractal, c.pad,
                                 c.compact, c.v_row, c.v_col, c.v_row_dynamic, c.v_col_dynamic);
}

std::string PTOCodegen::GetViewTileBufTypeStringFromTileType(
    const std::shared_ptr<const ir::TileType>& tile_type) const {
  INTERNAL_CHECK(tile_type) << "Internal error: tile_type must not be null";
  auto memory_space = tile_type->GetMemorySpace();
  INTERNAL_CHECK(memory_space.has_value()) << "Internal error: tile_type must have memory_space";

  auto c = ExtractTileTypeInfo(*tile_type, GetTypeString(tile_type->dtype_));

  // `pto.alloc_tile` conveys the valid extent through `valid_row` / `valid_col`
  // operands, so ExtractTileTypeInfo always renders `v_row=?, v_col=?`. A view op
  // that takes NO such operands — `pto.treshape` — cannot: ptoas default-
  // constructs its destination tile from the result type alone, so a dynamic
  // valid leaves the tile's valid extent at zero and every consumer silently
  // becomes a no-op. Render static valid dims whenever the view's effective
  // valid_shape is statically known.
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile_type);
  const auto& valid = view.valid_shape;
  if (valid.size() == 1) {
    // Match ComputeAllocTileFields / ExtractTileTypeInfo: a 1-D valid_shape
    // maps to rows=1, cols=shape[0]. Without this a 1-D reshape view keeps the
    // dynamic zero-valid extent and its consumers become silent no-ops.
    if (auto v_col = As<ir::ConstInt>(valid[0])) {
      c.v_row = 1;
      c.v_col = v_col->value_;
      c.v_row_dynamic = false;
      c.v_col_dynamic = false;
    }
  } else if (valid.size() >= 2) {
    auto v_row = As<ir::ConstInt>(valid[0]);
    auto v_col = As<ir::ConstInt>(valid[1]);
    if (v_row && v_col) {
      c.v_row = v_row->value_;
      c.v_col = v_col->value_;
      c.v_row_dynamic = false;
      c.v_col_dynamic = false;
    }
  }
  return FormatTileBufTypeString(MemorySpaceToMLIR(*memory_space), c.dtype_str, c.rows, c.cols, c.blayout,
                                 c.slayout, c.fractal, c.pad, c.compact, c.v_row, c.v_col, c.v_row_dynamic,
                                 c.v_col_dynamic);
}

std::string PTOCodegen::GetExprTypeAnnotation(const ir::ExprPtr& expr) {
  if (auto var = As<ir::Var>(expr)) {
    auto key = GetVarKey(var);
    // Primary lookup: SSA name → tile_buf type (covers root allocs AND view results)
    auto mlir_it = fs_.var_to_mlir.find(key);
    if (mlir_it != fs_.var_to_mlir.end()) {
      auto ssa_it = fs_.ssa_to_tile_buf_type.find(mlir_it->second);
      if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
        return ssa_it->second;
      }
    }
    // Per-variable TileType: derives the type from the variable's own
    // TileType, which is correct for view op results (slice, reshape,
    // fillpad) whose type differs from the root alloc's type.
    if (auto tile_type = As<TileType>(var->GetType())) {
      if (tile_type->memref_.has_value()) {
        return GetTileBufTypeStringFromTileType(tile_type);
      }
    }
    // Fallback: var → memref → root alloc type
    auto memref_it = fs_.var_to_memref.find(key);
    if (memref_it != fs_.var_to_memref.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    if (auto scalar_type = As<ScalarType>(var->GetType())) {
      return GetTypeString(scalar_type->dtype_);
    }
  }
  if (auto iter_arg = As<ir::IterArg>(expr)) {
    auto key = GetVarKey(std::dynamic_pointer_cast<const ir::Var>(iter_arg));
    auto mlir_it = fs_.var_to_mlir.find(key);
    if (mlir_it != fs_.var_to_mlir.end()) {
      auto ssa_it = fs_.ssa_to_tile_buf_type.find(mlir_it->second);
      if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
        return ssa_it->second;
      }
    }
    if (auto tile_type = ir::GetTileTypeWithMemRef(iter_arg->GetType())) {
      return GetTileBufTypeStringFromTileType(tile_type);
    }
    auto memref_it = fs_.var_to_memref.find(key);
    if (memref_it != fs_.var_to_memref.end()) {
      return GetTileBufTypeString(memref_it->second);
    }
    if (auto scalar_type = As<ScalarType>(iter_arg->GetType())) {
      return GetTypeString(scalar_type->dtype_);
    }
  }
  if (auto const_float = As<ir::ConstFloat>(expr)) {
    return GetTypeString(const_float->dtype());
  }
  if (auto const_int = As<ir::ConstInt>(expr)) {
    // The SSA value produced by GetOrEmitConstant is cast back to the dtype's
    // MLIR type (via unrealized_conversion_cast for unsigned), so the use-site
    // annotation matches the declared dtype directly.
    return GetTypeString(const_int->dtype());
  }
  // Fallback: derive annotation from any ScalarType expression (e.g. Cast results,
  // arith expression results). Their SSA value carries the declared dtype.
  if (auto scalar_type = As<ScalarType>(expr->GetType())) {
    return GetTypeString(scalar_type->dtype_);
  }
  return "";
}

std::string PTOCodegen::GetCurrentResultTileBufTypeString() const {
  // Prefer the type registered by alloc_tile (always dynamic
  // `v_row=?, v_col=?` per ComputeAllocTileFields).
  if (!fs_.current_result_buf.empty()) {
    auto ssa_it = fs_.ssa_to_tile_buf_type.find(fs_.current_result_buf);
    if (ssa_it != fs_.ssa_to_tile_buf_type.end()) {
      return ssa_it->second;
    }
  }
  if (fs_.current_result_tile_type) {
    if (const auto& memref = fs_.current_result_tile_type->memref_) {
      return GetTileBufTypeString((*memref)->base_.get());
    }
  }
  return "";
}

std::string PTOCodegen::GetCurrentResultTileBufTypeStringFromTileType() const {
  if (fs_.current_result_tile_type && fs_.current_result_tile_type->memref_.has_value()) {
    return GetTileBufTypeStringFromTileType(fs_.current_result_tile_type);
  }
  return "";
}

std::pair<std::string, std::string> PTOCodegen::GetCurrentResultTpopValidShapeOperands() {
  if (!fs_.current_result_tile_type) {
    return {"", ""};
  }

  const auto& tile_view = fs_.current_result_tile_type->tile_view_;
  if (!tile_view) {
    return {"", ""};
  }

  const auto& valid_shape = tile_view->valid_shape;
  ExprPtr valid_row_expr;
  ExprPtr valid_col_expr;
  bool has_dynamic_valid_shape = false;
  if (valid_shape.size() >= 1 && valid_shape[0]) {
    valid_row_expr = valid_shape[0];
    has_dynamic_valid_shape = !As<ir::ConstInt>(valid_row_expr);
  }
  if (valid_shape.size() >= 2 && valid_shape[1]) {
    valid_col_expr = valid_shape[1];
    has_dynamic_valid_shape = has_dynamic_valid_shape || !As<ir::ConstInt>(valid_col_expr);
  }
  bool valid_shape_matches_shape = !valid_row_expr && !valid_col_expr;
  if (valid_row_expr && valid_col_expr) {
    const auto& shape = fs_.current_result_tile_type->shape_;
    ExprPtr shape_row_expr;
    ExprPtr shape_col_expr;
    if (shape.size() >= 2) {
      shape_row_expr = shape[0];
      shape_col_expr = shape[1];
    } else if (shape.size() == 1) {
      shape_row_expr = std::make_shared<ir::ConstInt>(1, DataType::INDEX, ir::Span::unknown());
      shape_col_expr = shape[0];
    }
    valid_shape_matches_shape =
        IsSameDimExpr(valid_row_expr, shape_row_expr) && IsSameDimExpr(valid_col_expr, shape_col_expr);
  }
  if (!has_dynamic_valid_shape && valid_shape_matches_shape) {
    return {"", ""};
  }

  auto cast_scalar_to_index = [&](const std::string& ssa, const ScalarType* scalar_type) -> std::string {
    bool is_integer_or_index = scalar_type->dtype_.IsInt() || scalar_type->dtype_ == DataType::INDEX;
    CHECK(is_integer_or_index && scalar_type->dtype_.GetBit() != 1)
        << "tpop valid_shape operand must be integer or index type, got "
        << GetTypeString(scalar_type->dtype_);
    if (scalar_type->dtype_ == DataType::INDEX) {
      return ssa;
    }
    std::string idx = NewTemp();
    std::string src_type = GetTypeString(scalar_type->dtype_);
    Emit(idx + " = arith.index_cast " + ssa + " : " + src_type + " to index");
    return idx;
  };

  auto get_index_operand = [&](const ExprPtr& expr, size_t dim_idx) -> std::string {
    if (expr) {
      if (auto const_int = As<ir::ConstInt>(expr)) {
        return GetOrEmitConstant(const_int->value_, DataType::INDEX);
      }
      std::string ssa = GetExprAsCode(expr);
      if (auto scalar_type = As<ScalarType>(expr->GetType())) {
        return cast_scalar_to_index(ssa, scalar_type.get());
      }
      return ssa;
    }

    const auto& shape = fs_.current_result_tile_type->shape_;
    ExprPtr shape_dim;
    if (shape.size() >= 2 && dim_idx < shape.size()) {
      shape_dim = shape[dim_idx];
    } else if (shape.size() == 1) {
      if (dim_idx == 0) {
        return GetOrEmitConstant(static_cast<int64_t>(1), DataType::INDEX);
      }
      shape_dim = shape[0];
    }
    INTERNAL_CHECK(shape_dim) << "Internal error: tpop result tile type is missing shape dim " << dim_idx;
    if (auto const_int = As<ir::ConstInt>(shape_dim)) {
      return GetOrEmitConstant(const_int->value_, DataType::INDEX);
    }
    std::string ssa = GetExprAsCode(shape_dim);
    if (auto scalar_type = As<ScalarType>(shape_dim->GetType())) {
      return cast_scalar_to_index(ssa, scalar_type.get());
    }
    return ssa;
  };

  return {get_index_operand(valid_row_expr, 0), get_index_operand(valid_col_expr, 1)};
}

}  // namespace codegen
}  // namespace pypto
