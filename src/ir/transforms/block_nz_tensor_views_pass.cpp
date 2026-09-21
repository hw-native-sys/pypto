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

/**
 * @file block_nz_tensor_views_pass.cpp
 * @brief BlockNzTensorViews pass — turn a logical NZ tensor into its blocked form.
 *
 * ``pl.Tensor[[E, N, K], pl.INT8, pl.NZ]`` asserts that the bytes in GM are
 * already in PTO-native NZ fractal order while keeping the *logical* shape and
 * slicing at the DSL level. pto-isa describes such a buffer with a blocked
 * **rank-5** GlobalTensor (``pto/common/pto_tile.hpp``):
 *
 *     shape   = [E, K/c0, N/16, 16, c0]
 *     strides = [K*N,     N*c0, 16*c0, c0, 1]      (c0 = 256 / dtype bits)
 *
 * The rank is fixed, not ``logical rank + 2``: the leading batch slot exists
 * whether or not the logical tensor has a leading axis. A logical rank-2
 * ``[N, K]`` weight therefore blocks to ``[1, K/c0, N/16, 16, c0]``, with the
 * batch materialised as 1 — blocking it to rank 4 instead produces a view PTOAS
 * refuses ("user-specified layout=nz requires a rank-5 view"). Logical rank 4+
 * has no canonical form yet and is rejected in ``CheckNzLogicalRank``.
 *
 * This pass rewrites the IR into exactly that form:
 *
 *   Phase 1 — every TensorType tagged ``TensorLayout::NZ`` gets its shape
 *             replaced by ``BlockNzShape``. The stride slot is left empty for
 *             ``MaterializeTensorStrides`` (pass 33) to fill; because a blocked
 *             NZ shape's row-major strides *are* pto-isa's NZ strides, that
 *             pass needs no NZ-specific rule.
 *
 *   Phase 2 — every ``tile.load`` reading such a tensor gets its offsets /
 *             shapes / valid_shape rewritten into blocked coordinates, while
 *             its result ``TileType`` is preserved verbatim: the GM partition
 *             becomes rank-5 but the destination tile stays the logical
 *             2-D ``[N_TILE, K_TILE]``.
 *
 * After this pass no logical-shaped NZ TensorType survives, so nothing
 * downstream — including codegen, which derives every rank from
 * ``TensorType::shape_`` — needs to know NZ is special. That is the reason the
 * blocking lives here rather than in the backend: ``EmitMakeTensorViews``,
 * ``GetTensorViewTypeString`` and the ``tile.load`` ``partition_view`` emitter
 * each read the rank independently and must agree.
 *
 * Ordering constraints (see docs/en/dev/passes/15-block_nz_tensor_views.md):
 *   * after ConvertTensorToTileOps / LowerCompositeOps — the ``tile.load`` ops
 *     Phase 2 rewrites must already exist;
 *   * after FlattenTileNdTo2D — declared as a ``TileOps2D`` requirement. The
 *     destination tile must already be the logical 2-D operand: blocking a
 *     still-ND-rank tile leaves a ``tile.load`` whose type annotation and
 *     argument ranks cannot both be printed, which the printer round-trip
 *     rejects. FlattenTileNdTo2D skips its ND2NZ source-window collapse for an
 *     NZ source, so the logical window is still intact when this pass runs.
 *
 * Milestone 1 scope: read-only, matmul operands only (``target_memory=Mat``),
 * whole-byte dtypes, static shapes, fractal-aligned shapes and slice offsets. A
 * slice offset may be symbolic when its alignment is *provable* — see
 * ``NzOffsetFactStore`` below and ``DivideIndexExactly`` in
 * ``tensor_view_semantics.h``. Everything outside that is rejected with a
 * diagnostic naming the authoring fix — an NZ tensor must never be silently
 * mis-addressed.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

/// Function-level stamp marking that Phase 1 already ran on this function.
///
/// Blocking is *not* idempotent — running it twice would block an already
/// blocked shape — and the structural ``IsBlockedNzShape`` test cannot tell a
/// blocked shape from a logical one that merely ends in ``[16, c0]``. The stamp
/// makes re-entry a no-op without relying on that ambiguity.
constexpr const char* kNzBlockedAttr = "nz_tensor_views_blocked";

/// True when a tensor-like type carries an NZ TensorView.
bool IsNzTensorType(const TypePtr& type) {
  auto tensor_type = AsTensorTypeLike(type);
  if (!tensor_type) return false;
  // Bind the optional before dereferencing so the access is locally provable
  // (bugprone-unchecked-optional-access does not follow a `&&` short-circuit).
  const auto& view = tensor_type->tensor_view_;
  return view.has_value() && view->layout == TensorLayout::NZ;
}

/// Rewrite an NZ-tagged TensorType / DistributedTensorType to its blocked
/// shape, recursing into TupleType. Returns the input unchanged when there is
/// nothing to block (identity-comparable by the caller).
TypePtr BlockNzType(const TypePtr& type, const Span& span) {
  if (!type) return type;

  if (auto tuple_type = As<TupleType>(type)) {
    std::vector<TypePtr> new_elements;
    new_elements.reserve(tuple_type->types_.size());
    bool changed = false;
    for (const auto& element : tuple_type->types_) {
      auto new_element = BlockNzType(element, span);
      if (new_element.get() != element.get()) changed = true;
      new_elements.push_back(std::move(new_element));
    }
    if (!changed) return type;
    return std::make_shared<TupleType>(std::move(new_elements));
  }

  if (auto dist_type = As<DistributedTensorType>(type)) {
    if (!IsNzTensorType(type)) return type;
    // A distributed NZ tensor would additionally need remote_load blocking,
    // which milestone 1 does not implement; remote_load.cpp rejects it. Refuse
    // here too so the diagnostic names the annotation rather than surfacing
    // later as an opaque layout error.
    CHECK_SPAN(false, span) << "NZ layout is not supported on a distributed tensor yet. "
                            << "Annotate the tensor as pl.ND or pl.DN.";
  }

  if (auto tensor_type = As<TensorType>(type)) {
    if (!IsNzTensorType(type)) return type;
    // ``IsNzTensorType`` already established the view exists, but that is not
    // locally provable, so repeat the test where the optional is dereferenced.
    const auto& maybe_view = tensor_type->tensor_view_;
    if (!maybe_view.has_value()) return type;
    auto blocked_shape = tensor_view_semantics::BlockNzShape(tensor_type->shape_, tensor_type->dtype_, span);
    // valid_shape is a per-dim companion of the logical shape; a partial NZ
    // region has no blocked representation in milestone 1.
    const TensorView& view = *maybe_view;
    CHECK_SPAN(view.valid_shape.empty(), span)
        << "NZ layout does not support a partial valid_shape yet; the whole tensor must be valid.";
    CHECK_SPAN(view.stride.empty(), span)
        << "NZ layout does not support an explicit stride yet: the blocked NZ stride is derived from "
        << "the shape. Drop the stride annotation.";
    return std::make_shared<TensorType>(std::move(blocked_shape), tensor_type->dtype_, tensor_type->memref_,
                                        view);
  }

  return type;
}

/// Whether ``op`` is a ``tensor.reshape`` that flattens the whole NZ tensor into
/// a single rank-1 view of every element.
///
/// The blocked form is a reordering of the logical index space, not of memory:
/// both spellings cover the same contiguous GM range in the same order. A view
/// of *all* of it therefore needs no coordinate rewrite, which is what lets an
/// NZ weight still be handed to ``prefetch.async_prefetch`` (it wants a flat
/// logical-1D source). Any other reshape does reinterpret coordinates and is
/// refused by the caller.
bool IsWholeTensorFlatten(const CallPtr& op) {
  if (!IsOp(op, "tensor.reshape")) return false;
  auto source_type = AsTensorTypeLike(op->args_[0]->GetType());
  if (!source_type) return false;
  auto shape_tuple = As<MakeTuple>(op->args_[1]);
  if (!shape_tuple || shape_tuple->elements_.size() != 1) return false;
  auto flat_extent = As<ConstInt>(shape_tuple->elements_[0]);
  if (!flat_extent) return false;
  int64_t logical_elements = 1;
  for (const auto& dim : source_type->shape_) {
    auto extent = As<ConstInt>(dim);
    if (!extent) return false;
    logical_elements *= extent->value_;
  }
  return logical_elements == flat_extent->value_;
}

/// Index bindings a symbolic slice offset has to be proven against.
///
/// A slice offset arrives at the ``tile.load`` as the SSA name it was bound to,
/// so the arithmetic that makes it fractal-aligned lives elsewhere in the
/// function — in the ``AssignStmt`` that defined it (``n0 = nb * 256``) or in
/// the ``ForStmt`` that bounds it (``for k0 in pl.pipeline(512, 4096, 512)``).
/// This collects both in one read-only walk, so the rewrite itself stays a
/// constant-time lookup and the pass stays O(N).
///
/// Collected from the *pre-mutation* body: the mutator only substitutes NZ
/// tensor Vars, so an index Var is the same node before and after.
class NzOffsetFactStore {
 public:
  explicit NzOffsetFactStore(const StmtPtr& body) {
    if (!body) return;
    Collector collector(this);
    collector.VisitStmt(body);
  }

  /// A view of this store. The callbacks capture ``this``, so the store must
  /// outlive every use of the returned facts.
  [[nodiscard]] tensor_view_semantics::NzOffsetFacts Facts() const {
    tensor_view_semantics::NzOffsetFacts facts;
    facts.definition = [this](const VarPtr& var) -> ExprPtr {
      auto it = definitions_.find(var);
      return it == definitions_.end() ? nullptr : it->second;
    };
    facts.loop_range = [this](const VarPtr& var) -> std::pair<ExprPtr, ExprPtr> {
      auto it = loop_bindings_.find(var);
      if (it == loop_bindings_.end()) return {nullptr, nullptr};
      return it->second;
    };
    facts.is_non_negative = [this](const VarPtr& var) {
      // The SPMD block index is a lane number, so it is never negative.
      return non_negative_vars_.count(var) != 0;
    };
    return facts;
  }

 private:
  class Collector : public IRVisitor {
   public:
    explicit Collector(NzOffsetFactStore* store) : store_(store) {}

   protected:
    void VisitStmt_(const AssignStmtPtr& op) override {
      // Scalars only: a tensor / tile definition can never be part of an index
      // expression, and keeping them out bounds the map to the index IR.
      if (op->var_ && As<ScalarType>(op->var_->GetType())) {
        store_->definitions_.emplace(op->var_, op->value_);
        // A block index or block count is a lane number, so it is non-negative
        // by construction. That is an operator fact rather than a structural
        // one, which is why it is recorded here rather than derived by the
        // geometry helpers.
        auto call = As<Call>(op->value_);
        if (call && (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num"))) {
          store_->non_negative_vars_.insert(op->var_);
        }
      }
      IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const ForStmtPtr& op) override {
      // Symbolic bounds are recorded too: the proofs recurse into them, so a
      // strided loop that starts at the block index is as provable as one that
      // starts at a literal.
      if (op->loop_var_ && op->start_ && op->step_) {
        store_->loop_bindings_.emplace(op->loop_var_, std::make_pair(op->start_, op->step_));
      }
      IRVisitor::VisitStmt_(op);
    }

   private:
    NzOffsetFactStore* store_;
  };

  std::unordered_map<VarPtr, ExprPtr> definitions_;
  std::unordered_map<VarPtr, std::pair<ExprPtr, ExprPtr>> loop_bindings_;
  std::unordered_set<VarPtr> non_negative_vars_;
};

/// Positions in a blocked NZ shape ``[B, C/c0, R/16, 16, c0]``.
constexpr size_t kNzColumnBlockDim = 1;
constexpr size_t kNzRowFractalDim = 2;

/// Largest inter-burst source gap the strided GM->L1 copy can encode: its
/// ``srcStride`` operand is a 16-bit field counting 32-byte blocks.
constexpr int64_t kNzMaxGmGapBlocks = 65535;

/// TEMPORARY correctness guard for an upstream pto-isa defect — delete this
/// function and its call site once hw-native-sys/pto-isa#317 lands.
///
/// ``TLoadGm2L1Nz2nz`` computes the GM gap between consecutive column blocks as
/// a ``uint32_t`` and hands it to ``TLoadInstrGm2L1``'s ``uint16_t gmGap`` with
/// no range test, so a gap above ``kNzMaxGmGapBlocks`` wraps and the load reads
/// the wrong fractals. Nothing below this pass notices: PTOAS assembles it, the
/// CCE compiler accepts it, and the kernel returns wrong numbers with no error.
/// This pass is the only layer that still knows which ``pl.NZ`` annotation in
/// the user's own source is responsible, which is why the guard lives here.
///
/// For a blocked view the gap reduces to the row extent the load leaves behind:
///
/// ```text
///   gmGap = (gStride1 - gShape2*gShape3*gShape4) * sizeof(T) / 32
///         = (R/16 - TR/16) * 16 * c0 * sizeof(T) / 32
///         = R - TR                                      (32-byte blocks)
/// ```
///
/// because ``c0 * sizeof(T) == 32`` holds for every NZ view by construction —
/// so the bound is dtype-independent, and a *wider* row tile is what lowers the
/// gap, not a narrower one.
///
/// The gap is the stride *between* bursts, and ``nBurst`` is the load's own
/// column-block extent, so a single-column-block load never consumes it — it
/// is exempt no matter how large the gap computes to.
///
/// ``blocked_partition`` must be the tuple codegen turns into the
/// ``pto.partition_view`` — ``valid_shape`` when the load carries one, else
/// ``shapes`` (``src/backend/common/pto_ops_memory.cpp``). That view *is*
/// pto-isa's ``gShape``, so reading ``shapes`` unconditionally would measure a
/// window the hardware never sees: a narrowed ``valid_shape`` loads *fewer*
/// row fractals and therefore leaves a *larger* gap behind.
void CheckNzGmGapFitsBurstStride(const std::vector<ExprPtr>& blocked_shape, const ExprPtr& blocked_partition,
                                 const Span& span) {
  auto sizes = As<MakeTuple>(blocked_partition);
  INTERNAL_CHECK_SPAN(sizes && sizes->elements_.size() == tensor_view_semantics::kNzBlockedRank, span)
      << "Internal error: the blocked tile.load partition must be a rank-"
      << tensor_view_semantics::kNzBlockedRank << " MakeTuple";
  INTERNAL_CHECK_SPAN(blocked_shape.size() == tensor_view_semantics::kNzBlockedRank, span)
      << "Internal error: the NZ tensor shape must be blocked before the GM gap check";

  // ``BlockNzShape`` rejects a dynamic ``shape[-2]`` on the tensor, so its
  // row-fractal extent is a constant. The loaded extent is dynamic when a
  // valid_shape narrows a ragged last tile; the gap only grows as fewer rows
  // load, so the proof takes the worst case -- nothing loaded -- and a load that
  // fits then fits for every run-time width.
  auto whole = As<ConstInt>(blocked_shape[kNzRowFractalDim]);
  INTERNAL_CHECK_SPAN(whole, span)
      << "Internal error: the blocked NZ tensor row-fractal extent must be static";
  auto loaded = As<ConstInt>(sizes->elements_[kNzRowFractalDim]);

  // ``TLoadGm2L1Nz2nz`` passes the load's column-block extent as ``nBurst``,
  // and the DMA applies ``gmGap`` only when stepping from one burst to the
  // next. At one burst the field is never read, so the truncation cannot reach
  // any source address and the load is correct however large the gap is.
  // Confirmed on device: the 65536-block gap that corrupts a two-column-block
  // load returns bit-exact data at one (pto-isa tload_gm2mat ST, NZ int16
  // 1_1_8_16_16 / 1_1_4104_16_16).
  auto column_blocks = As<ConstInt>(sizes->elements_[kNzColumnBlockDim]);
  INTERNAL_CHECK_SPAN(column_blocks, span)
      << "Internal error: the blocked NZ column-block extent must be static";
  if (column_blocks->value_ <= 1) return;

  const int64_t whole_rows = whole->value_ * tensor_view_semantics::kNzFractalRow;
  const int64_t loaded_rows = loaded ? loaded->value_ * tensor_view_semantics::kNzFractalRow : 0;
  const int64_t gap = whole_rows - loaded_rows;
  CHECK_SPAN(gap <= kNzMaxGmGapBlocks, span)
      << "NZ layout: this tile.load is refused because it would silently return wrong data. Its GM row "
      << "gap is " << gap << " 32-byte blocks (" << whole_rows << " tensor rows - " << loaded_rows
      << " loaded rows), above the " << kNzMaxGmGapBlocks
      << " that the hardware's strided GM->L1 copy can encode. pto-isa's TLoadGm2L1Nz2nz truncates that "
      << "gap to 16 bits with no range check, so the load would read the wrong fractals and the kernel "
      << "would return wrong numbers with no error at run time.\n"
      << "This is NOT an expected pl.NZ limitation: it is a temporary correctness guard, and it is "
      << "removed once pto-isa fixes the truncation (hw-native-sys/pto-isa#317).\n"
      << "Workarounds: load a taller row tile -- the gap is (rows - loaded rows), so it shrinks as the "
      << "tile grows -- or annotate the tensor rank-3 with the stacked axis as the batch "
      << "([LAYERS, K, N]), whose extent then rides the batch stride instead of the burst gap.";
}

/// Rewrite the elements of a ``MakeTuple`` coordinate argument into blocked NZ
/// form. ``facts`` is read only on the offsets path — a shape is a static
/// extent, never a symbolic expression.
ExprPtr BlockTupleArg(const ExprPtr& arg, const std::vector<ExprPtr>& parent_shape, DataType dtype,
                      const Span& span, bool is_offsets, const tensor_view_semantics::NzOffsetFacts& facts) {
  auto tuple = As<MakeTuple>(arg);
  INTERNAL_CHECK_SPAN(tuple, span) << "Internal error: an NZ coordinate argument must be a MakeTuple";
  auto blocked =
      is_offsets ? tensor_view_semantics::BlockNzOffsets(tuple->elements_, parent_shape, dtype, span, facts)
                 : tensor_view_semantics::BlockNzShape(tuple->elements_, dtype, span);
  return std::make_shared<MakeTuple>(std::move(blocked), tuple->span_);
}

class BlockNzMutator : public IRMutator {
 public:
  explicit BlockNzMutator(tensor_view_semantics::NzOffsetFacts facts) : facts_(std::move(facts)) {}

  void AddSubstitution(const VarPtr& old_var, const VarPtr& new_var) { var_cache_[old_var] = new_var; }

 protected:
  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) return it->second;
    auto new_type = BlockNzType(op->GetType(), op->span_);
    if (new_type.get() == op->GetType().get()) {
      var_cache_[op] = op;
      return op;
    }
    auto new_var = std::make_shared<Var>(op->name_hint_, std::move(new_type), op->span_);
    var_cache_[op] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    auto it = var_cache_.find(op);
    if (it != var_cache_.end()) return it->second;
    auto new_init = IRMutator::VisitExpr(op->initValue_);
    auto new_type = BlockNzType(op->GetType(), op->span_);
    if (new_init.get() == op->initValue_.get() && new_type.get() == op->GetType().get()) {
      var_cache_[op] = op;
      return op;
    }
    auto new_iter_arg = std::make_shared<IterArg>(op->name_hint_, std::move(new_type), new_init, op->span_);
    var_cache_[op] = new_iter_arg;
    return new_iter_arg;
  }

  ExprPtr VisitExpr_(const CallPtr& op) override {
    std::vector<ExprPtr> new_args;
    new_args.reserve(op->args_.size());
    bool args_changed = false;
    for (const auto& arg : op->args_) {
      auto new_arg = IRMutator::VisitExpr(arg);
      if (new_arg.get() != arg.get()) args_changed = true;
      new_args.push_back(std::move(new_arg));
    }

    // Scan *every* operand, not just the first. An NZ tensor can appear in any
    // position — `tile.store`'s destination is argument 2 — and phase 1 has
    // already blocked its type by the time we get here. An operand this pass
    // does not recognise must be rejected rather than left with logical
    // coordinates pointing into a blocked tensor.
    std::vector<size_t> nz_args;
    for (size_t i = 0; i < new_args.size(); ++i) {
      auto tensor = AsVarLike(new_args[i]);
      if (tensor && IsNzTensorType(tensor->GetType())) nz_args.push_back(i);
    }

    // A call to another function just forwards the tensor; the callee's own
    // params are blocked when that function is transformed.
    const bool is_function_call = static_cast<bool>(As<GlobalVar>(op->op_));
    if (!nz_args.empty() && !is_function_call) {
      // Name the store case directly: annotating an Out/InOut tensor pl.NZ is
      // the likely authoring mistake, and "read-only" is the actionable fact.
      CHECK_SPAN(!IsOp(op, "tile.store"), op->span_)
          << "NZ layout is read-only: an NZ tensor cannot be a store destination. "
          << "Annotate the output tensor as pl.ND.";
      const bool is_load = IsOp(op, "tile.load");
      const bool is_slice = IsOp(op, "tensor.slice");
      const bool is_flatten = IsWholeTensorFlatten(op);
      CHECK_SPAN((is_load || is_slice || is_flatten) && nz_args.size() == 1 && nz_args[0] == 0, op->span_)
          << "NZ layout currently supports only 'tile.load' and 'tensor.slice' reading the tensor as "
          << "their source, plus a whole-tensor 'tensor.reshape' flatten, but it is used by '"
          << op->op_->name_ << "' at argument " << nz_args[0]
          << ". NZ tensors are read-only matmul operands in this release.";
      auto logical_type = AsTensorTypeLike(op->args_[0]->GetType());
      INTERNAL_CHECK_SPAN(logical_type, op->span_)
          << "Internal error: the NZ source of '" << op->op_->name_ << "' must be a tensor";
      const std::vector<ExprPtr>& logical_shape = logical_type->shape_;
      // The flatten keeps its arguments: the element count is layout-invariant
      // and the result is already ND.
      if (!is_flatten) {
        args_changed = true;
        new_args = is_load ? BlockTileLoadArgs(op, std::move(new_args), logical_shape)
                           : BlockTensorSliceArgs(op, std::move(new_args), logical_shape);
      }
    }

    auto new_return_type = BlockNzType(op->GetType(), op->span_);
    const bool type_changed = new_return_type.get() != op->GetType().get();
    if (!args_changed && !type_changed) return op;

    // Direct ctor, not OpRegistry::Create: re-deducing ``tile.load``'s type
    // from the now rank-5 shapes argument would turn the destination tile into
    // a rank-5 TileType. The GM partition is blocked; the tile is not.
    return std::make_shared<Call>(op->op_, std::move(new_args), op->kwargs_, op->attrs_,
                                  std::move(new_return_type), op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_var_expr = IRMutator::VisitExpr(op->var_);
    auto new_value = IRMutator::VisitExpr(op->value_);
    auto new_var = As<Var>(new_var_expr);
    INTERNAL_CHECK(new_var) << "Internal error: BlockNzTensorViews visited an AssignStmt LHS to a non-Var";
    if (new_var.get() == op->var_.get() && new_value.get() == op->value_.get()) return op;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

 private:
  /// Rewrite ``tensor.slice``'s (shapes, offsets) into blocked coordinates.
  ///
  /// Only a window that narrows the *leading* axes is representable. A blocked
  /// NZ view carries no stride of its own — ``MaterializeTensorStrides`` derives
  /// a row-major one from the blocked shape — so the selected bytes have to be
  /// contiguous. A leading-axis window is: those axes fold into the batch slot,
  /// one whole ``R x C`` matrix per step. A row window is not: in NZ order one
  /// layer's rows sit inside *every* column block of the parent, so
  /// ``[layer*R, 0]`` selects ``C/c0`` disjoint runs that no contiguous stride
  /// describes.
  ///
  /// A rank-reducing scalar index (``w[r]``) needs no ``drop_dims`` afterwards:
  /// the axes it drops are leading ones, and the fold has already collapsed
  /// every leading axis into the single batch slot.
  std::vector<ExprPtr> BlockTensorSliceArgs(const CallPtr& op, std::vector<ExprPtr> args,
                                            const std::vector<ExprPtr>& logical_shape) {
    INTERNAL_CHECK_SPAN(args.size() >= 3, op->span_)
        << "Internal error: tensor.slice expects at least (tensor, shapes, offsets), got " << args.size();

    auto tensor = AsVarLike(args[0]);
    INTERNAL_CHECK_SPAN(tensor, op->span_) << "Internal error: the NZ tensor.slice source must be a variable";
    auto tensor_type = AsTensorTypeLike(tensor->GetType());
    INTERNAL_CHECK_SPAN(tensor_type, op->span_)
        << "Internal error: the NZ tensor.slice source must be a tensor";
    const DataType dtype = tensor_type->dtype_;

    auto shapes = As<MakeTuple>(args[1]);
    auto offsets = As<MakeTuple>(args[2]);
    INTERNAL_CHECK_SPAN(shapes && offsets, op->span_)
        << "Internal error: tensor.slice coordinate arguments must be MakeTuples";
    const size_t rank = shapes->elements_.size();
    INTERNAL_CHECK_SPAN(rank == logical_shape.size() && offsets->elements_.size() == rank, op->span_)
        << "Internal error: a tensor.slice window must have the source's own rank";

    // Milestone 1 reads whole fractals only, so a *narrowed* valid_shape has no
    // blocked form — the same rule ``BlockNzType`` applies to the tensor. The
    // frontend writes a full one for an ordinary index, which says nothing and
    // blocks like any other extent tuple.
    // An omitted valid_shape reaches the op as an empty tuple, which says the
    // same thing as no valid_shape at all: the window is valid throughout.
    auto valid_shape = args.size() >= 4 ? As<MakeTuple>(args[3]) : nullptr;
    if (valid_shape && valid_shape->elements_.empty()) valid_shape = nullptr;
    if (valid_shape) {
      CHECK_SPAN(valid_shape->elements_.size() == rank, op->span_)
          << "NZ layout needs a full-rank valid_shape on a tensor.slice to block it, but this one is rank "
          << valid_shape->elements_.size() << " against a rank-" << rank
          << " window. Drop the valid_shape, or annotate the tensor as pl.ND.";
      for (size_t axis = 0; axis < rank; ++axis) {
        auto valid = As<ConstInt>(valid_shape->elements_[axis]);
        auto size = As<ConstInt>(shapes->elements_[axis]);
        CHECK_SPAN((valid && size && valid->value_ == size->value_) ||
                       valid_shape->elements_[axis].get() == shapes->elements_[axis].get(),
                   op->span_)
            << "NZ layout does not support a tensor.slice whose valid_shape narrows the window: axis " << axis
            << " is valid over less than it spans, and a partial fractal has no blocked form. "
            << "Drop the valid_shape, or annotate the tensor as pl.ND.";
      }
    }

    // A rank-reducing index drops leading axes only — the trailing pair is the
    // fractal plane and was already required whole above.
    if (args.size() >= 5) {
      auto drop_dims = As<MakeTuple>(args[4]);
      INTERNAL_CHECK_SPAN(drop_dims, op->span_)
          << "Internal error: tensor.slice drop_dims must be a MakeTuple";
      for (const auto& dim_expr : drop_dims->elements_) {
        auto dim = As<ConstInt>(dim_expr);
        INTERNAL_CHECK_SPAN(dim, op->span_)
            << "Internal error: tensor.slice drop_dims entries must be ConstInt";
        CHECK_SPAN(dim->value_ >= 0 && static_cast<size_t>(dim->value_) + 2 < rank, op->span_)
            << "NZ layout supports dropping a leading axis only, but this tensor.slice drops axis "
            << dim->value_ << " of the trailing fractal plane. Index a leading axis, or annotate the "
            << "tensor as pl.ND.";
      }
    }

    // The trailing matrix must be taken whole, which is what makes the selected
    // bytes contiguous and the derived row-major stride correct.
    auto whole_axis = [&](size_t axis, const char* name) {
      auto extent = As<ConstInt>(logical_shape[axis]);
      auto size = As<ConstInt>(shapes->elements_[axis]);
      auto offset = As<ConstInt>(offsets->elements_[axis]);
      CHECK_SPAN(extent && size && offset && size->value_ == extent->value_ && offset->value_ == 0, op->span_)
          << "NZ layout supports slicing the leading axes only: shape[" << axis << "] (" << name
          << ") must take the whole axis at offset 0. A window inside the trailing matrix selects one run "
          << "per fractal column block, which no contiguous stride describes. Slice a leading axis, or "
          << "annotate the tensor as pl.ND.";
    };
    whole_axis(rank - 2, "rows");
    whole_axis(rank - 1, "cols");

    // The leading axes fold row-major into the one batch slot -- extents
    // multiply, offsets flatten against the parent's extents -- and a product
    // of windows is a contiguous flat run only when every axis less significant
    // than a narrowed one is taken whole. Slicing a [2, 4, R, C] tensor to
    // [2, 2, R, C] at [0, 1, 0, 0] names batches {1, 2, 5, 6}, while the fold
    // yields extent 4 at offset 1 -- the run {1, 2, 3, 4}. Refuse it rather
    // than load the wrong weights.
    bool rest_must_be_whole = false;
    for (size_t axis = 0; axis + 2 < rank; ++axis) {
      auto extent = As<ConstInt>(logical_shape[axis]);
      auto size = As<ConstInt>(shapes->elements_[axis]);
      auto offset = As<ConstInt>(offsets->elements_[axis]);
      const bool whole = extent && size && offset && size->value_ == extent->value_ && offset->value_ == 0;
      CHECK_SPAN(!rest_must_be_whole || whole, op->span_)
          << "NZ layout supports a window on one leading axis only: shape[" << axis
          << "] must take the whole axis at offset 0, because an axis before it already spans more "
          << "than one element. The leading axes fold into one batch, and a narrowed axis under a "
          << "spanning one selects a set no contiguous run describes. Slice a single leading axis, "
          << "or annotate the tensor as pl.ND.";
      // A symbolic extent could be anything, so it counts as spanning.
      if (!size || size->value_ > 1) rest_must_be_whole = true;
    }

    // The blocked window carries no unit axis to drop: the fold already
    // collapsed every leading axis into the batch slot, so the rank-reducing
    // tail arguments go away with it.
    std::vector<ExprPtr> blocked_args = {args[0],
                                         BlockTupleArg(args[1], logical_shape, dtype, op->span_,
                                                       /*is_offsets=*/false, facts_),
                                         BlockTupleArg(args[2], logical_shape, dtype, op->span_,
                                                       /*is_offsets=*/true, facts_)};
    if (valid_shape) {
      blocked_args.push_back(
          BlockTupleArg(args[3], logical_shape, dtype, op->span_, /*is_offsets=*/false, facts_));
    }
    return blocked_args;
  }

  /// Rewrite ``tile.load``'s (offsets, shapes, [valid_shape]) into blocked
  /// coordinates and enforce the milestone-1 scope guards.
  std::vector<ExprPtr> BlockTileLoadArgs(const CallPtr& op, std::vector<ExprPtr> args,
                                         const std::vector<ExprPtr>& logical_shape) {
    INTERNAL_CHECK_SPAN(args.size() >= 3, op->span_)
        << "Internal error: tile.load expects at least (tensor, offsets, shapes), got " << args.size();
    auto tensor = AsVarLike(args[0]);
    auto tensor_type = AsTensorTypeLike(tensor->GetType());
    const DataType dtype = tensor_type->dtype_;

    // pto-isa only offers NZ->NZ into a Mat tile for the matmul operand path
    // (docs/isa/TLOAD.md); an NZ source loaded into a Vec tile is a different
    // (unimplemented) lowering. ``target_memory`` is optional on tile.load, so
    // an omitted target is a rejection too — the Vec default would be wrong.
    std::optional<MemorySpace> target;
    for (const auto& [key, value] : op->kwargs_) {
      if (key == "target_memory") {
        target = AnyCast<MemorySpace>(value, "target_memory");
        break;
      }
    }
    CHECK_SPAN(target.has_value() && *target == MemorySpace::Mat, op->span_)
        << "NZ layout currently supports only matmul operand loads (target_memory=pl.Mem.Mat), got "
        << (target.has_value() ? MemorySpaceToString(*target) : std::string("no target_memory"))
        << ". An NZ tensor is a cube weight: load it into Mat, or annotate the tensor as pl.ND.";

    args[1] = BlockTupleArg(args[1], logical_shape, dtype, op->span_, /*is_offsets=*/true, facts_);
    args[2] = BlockTupleArg(args[2], logical_shape, dtype, op->span_, /*is_offsets=*/false, facts_);
    if (args.size() >= 4) {
      // A valid_shape may narrow the rows at run time -- a ragged last tile --
      // so it blocks through the path that proves a dynamic row extent rather
      // than the static-shape one.
      auto valid = As<MakeTuple>(args[3]);
      INTERNAL_CHECK_SPAN(valid, op->span_) << "Internal error: tile.load valid_shape must be a MakeTuple";
      args[3] = std::make_shared<MakeTuple>(
          tensor_view_semantics::BlockNzValidShape(valid->elements_, dtype, op->span_, facts_), valid->span_);
    }
    // Codegen builds the ``pto.partition_view`` — and so pto-isa's ``gShape`` —
    // from valid_shape when the load carries one, falling back to shapes
    // otherwise (``src/backend/common/pto_ops_memory.cpp``). Measure the same
    // tuple: a narrowed valid_shape loads fewer row fractals and leaves a
    // larger gap than shapes alone would suggest.
    CheckNzGmGapFitsBurstStride(tensor_type->shape_, args.size() >= 4 ? args[3] : args[2], op->span_);
    return args;
  }

  tensor_view_semantics::NzOffsetFacts facts_;
  std::unordered_map<VarPtr, VarPtr> var_cache_;
};

/// Block one function: params, return types, body. Returns the input unchanged
/// when the function carries no NZ tensor.
FunctionPtr TransformFunction(const FunctionPtr& func) {
  if (func->HasAttr(kNzBlockedAttr)) return func;

  bool params_changed = false;
  std::vector<VarPtr> new_params;
  new_params.reserve(func->params_.size());
  std::unordered_map<VarPtr, VarPtr> param_substitutions;
  for (const auto& old_param : func->params_) {
    auto new_type = BlockNzType(old_param->GetType(), old_param->span_);
    if (new_type.get() == old_param->GetType().get()) {
      new_params.push_back(old_param);
      continue;
    }
    auto new_param = std::make_shared<Var>(old_param->name_hint_, std::move(new_type), old_param->span_);
    new_params.push_back(new_param);
    param_substitutions.emplace(old_param, new_param);
    params_changed = true;
  }

  bool returns_changed = false;
  std::vector<TypePtr> new_return_types;
  new_return_types.reserve(func->return_types_.size());
  for (const auto& rt : func->return_types_) {
    auto new_rt = BlockNzType(rt, func->span_);
    if (new_rt.get() != rt.get()) returns_changed = true;
    new_return_types.push_back(std::move(new_rt));
  }

  // The store owns the maps the facts read, so it must outlive the mutator.
  NzOffsetFactStore fact_store(func->body_);
  BlockNzMutator mutator(fact_store.Facts());
  for (const auto& [old_var, new_var] : param_substitutions) {
    mutator.AddSubstitution(old_var, new_var);
  }
  StmtPtr new_body = func->body_;
  if (func->body_) new_body = mutator.VisitStmt(func->body_);
  const bool body_changed = new_body.get() != func->body_.get();

  if (!params_changed && !returns_changed && !body_changed) return func;

  auto new_func = MutableCopy(func);
  if (params_changed) new_func->params_ = std::move(new_params);
  if (returns_changed) new_func->return_types_ = std::move(new_return_types);
  if (body_changed) new_func->body_ = std::move(new_body);
  new_func->attrs_.emplace_back(kNzBlockedAttr, std::any(true));
  return new_func;
}

}  // namespace

namespace pass {

Pass BlockNzTensorViews() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    bool modified = false;
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      auto new_func = TransformFunction(func);
      if (new_func.get() != func.get()) modified = true;
      new_functions[gvar] = std::move(new_func);
    }
    if (!modified) return program;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "BlockNzTensorViews", kBlockNzTensorViewsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
