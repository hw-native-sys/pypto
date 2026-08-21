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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// Check if operation is a view operation (zero-copy metadata transform).
// A view op is one registered with set_output_memory_inherit_input() — its
// output reuses the input's MemRef view. Delegates to the shared registry
// predicate so InferTileMemorySpace and InitMemRef agree on the set.
bool IsViewOperation(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return false;
  return registry.GetEntry(op_name).OutputMemoryInheritsInput();
}

// Whether an inherit-input op *permutes* data rather than just reinterpreting
// it (tile.transpose).  A pure metadata view (slice/reshape/extract/...) aliases
// its input's buffer, but a permuting op's output must never alias the input:
// pto.ttrans is not in-place safe (the unaligned scalar path writes dst directly
// from src), so InitMemRef gives the transpose output a fresh buffer.
bool IsDataPermutingInheritOp(const OpPtr& op) { return IsOp(op, "tile.transpose"); }

// A tile owns no general-pool buffer ("buffer-less by design") when its defining
// value is a cross-core tpop result, or a non-permuting zero-copy view / plain
// alias chained off a buffer-less source. `source_buffer_less` reports whether a
// source Var is itself buffer-less (the MemRef-creating mutator queries the type;
// the HasMemRefs verifier consults its tracked set). tile.transpose is excluded —
// pto.ttrans is not in-place safe and always needs a fresh buffer.
template <typename SourceBufferLess>
bool ProducesBufferLessTile(const ExprPtr& value, const SourceBufferLess& source_buffer_less) {
  if (auto call = std::dynamic_pointer_cast<const Call>(value)) {
    if (!call->op_) return false;
    if (IsOp(call, "tile.tpop_from_aic") || IsOp(call, "tile.tpop_from_aiv")) {
      return true;
    }
    if (op_predicates::IsBufferAliasingViewOp(call->op_->name_) && !call->args_.empty()) {
      auto in = AsVarLike(call->args_[0]);
      return in && source_buffer_less(in.get());
    }
    return false;
  }
  auto v = AsVarLike(value);
  return v && source_buffer_less(v.get());
}

// Check if an operation's output should reuse the MemRef of a specific input argument.
// Returns the input arg index whose MemRef to share, or nullopt.
std::optional<size_t> GetOutputReusesInputArg(const std::string& op_name) {
  auto& registry = OpRegistry::GetInstance();
  if (!registry.IsRegistered(op_name)) return std::nullopt;
  return registry.GetEntry(op_name).GetOutputReusesInputArg();
}

/// Byte envelope touched by a static packed slice, relative to its first
/// element. This is deliberately distinct from the physical allocation size:
/// L0C row padding belongs to the root allocation and must not be applied again
/// to a view beginning part-way through that allocation.
std::optional<uint64_t> StaticSliceViewSpanBytes(const CallPtr& call, const ShapedTypePtr& parent,
                                                 const ShapedTypePtr& view) {
  if (!call || (!IsOp(call, "tensor.slice") && !IsOp(call, "tile.slice"))) return std::nullopt;
  if (!parent || !view || call->args_.size() < 2) return std::nullopt;
  // Use the requested pre-drop shape rather than the result rank so
  // tile.slice(..., drop_dims=...) keeps the parent strides of dimensions that
  // disappear from the result type.
  auto requested_shape = As<MakeTuple>(call->args_[1]);
  if (!requested_shape || parent->shape_.size() != requested_shape->elements_.size()) {
    return std::nullopt;
  }

  uint64_t max_linear_offset = 0;
  uint64_t stride = 1;
  for (size_t rev = 0; rev < parent->shape_.size(); ++rev) {
    const size_t i = parent->shape_.size() - 1 - rev;
    auto parent_dim = As<ConstInt>(parent->shape_[i]);
    auto view_dim = As<ConstInt>(requested_shape->elements_[i]);
    if (!parent_dim || !view_dim || parent_dim->value_ <= 0 || view_dim->value_ <= 0 ||
        view_dim->value_ > parent_dim->value_) {
      return std::nullopt;
    }

    const uint64_t parent_extent = static_cast<uint64_t>(parent_dim->value_);
    const uint64_t view_extent = static_cast<uint64_t>(view_dim->value_);
    if (view_extent - 1 > std::numeric_limits<uint64_t>::max() / stride) return std::nullopt;
    const uint64_t contribution = (view_extent - 1) * stride;
    if (max_linear_offset > std::numeric_limits<uint64_t>::max() - contribution) return std::nullopt;
    max_linear_offset += contribution;
    if (rev + 1 < parent->shape_.size()) {
      if (stride > std::numeric_limits<uint64_t>::max() / parent_extent) return std::nullopt;
      stride *= parent_extent;
    }
  }

  if (max_linear_offset == std::numeric_limits<uint64_t>::max()) return std::nullopt;
  return storage_size::StaticStorageBytes(max_linear_offset + 1, view->dtype_);
}

// ============================================================================
// Author-declared allocations (`pl.Tile[..., pl.MemRef("name"), ...]`)
// ============================================================================

/// Slot geometry derived for one declared allocation.
struct DeclaredAlloc {
  uint64_t slot_size = 0;  ///< Bytes per slot — the largest tile bound to any slot
  /// How many slots the author declared; 0 until the first binding records it.
  /// The sentinel must be a value a real declaration can never carry — 1 would
  /// collide with an ordinary unsubscripted declaration, and a later, genuinely
  /// different count would then overwrite it instead of tripping the mismatch
  /// check. `Record` rejects anything below 1, so 0 is safe.
  uint64_t slot_count = 0;

  /// Total bytes to allocate: every slot is the same size and they sit contiguously,
  /// which is what makes `slot_index * slot_size` a valid offset.
  ///
  /// Both factors are author-controlled (`slots=N` and the bound tile's shape), so
  /// the product is checked rather than assumed: wrapping would turn an absurd
  /// request into a *small* allocation and hand out addresses inside it.
  [[nodiscard]] uint64_t TotalSize() const {
    CHECK(slot_count == 0 || slot_size <= std::numeric_limits<uint64_t>::max() / slot_count)
        << "Declared allocation is too large: " << slot_count << " slots of " << slot_size
        << " bytes overflows a 64-bit size";
    return slot_size * slot_count;
  }
};

/// Base Ptr of a declared allocation -> its slot geometry.
using DeclaredAllocMap = std::map<const Var*, DeclaredAlloc>;

/// The declared-allocation MemRef `type` carries, or null when it carries none.
///
/// `MemRef::is_pinned_` is what tells an author's declaration apart from a
/// compiler allocation: re-parsing a post-allocation dump also puts MemRefs on
/// TileTypes, and those are the compiler's. Keying on an explicit field rather
/// than on "we are standing before InitMemRef" keeps the classification a
/// property of the data, so a dump can be reparsed and re-run without its
/// allocations turning into declared ones.
///
/// Returning the MemRef (not just its base) spares every caller a second,
/// unchecked unwrap of the same optional.
MemRefPtr GetDeclaredAlloc(const TypePtr& type) {
  auto tile_type = As<TileType>(type);
  if (!tile_type || !tile_type->memref_.has_value()) return nullptr;
  const auto& memref = *tile_type->memref_;
  if (!memref->base_ || !memref->is_pinned_) return nullptr;
  return memref;
}

/// Collect every declared allocation in a function, deriving each one's size
/// (the largest bound tile) and checking the bound tiles agree on memory space.
class DeclaredAllocCollector : public IRVisitor {
 public:
  explicit DeclaredAllocCollector(const backend::BackendHandler* handler) : handler_(handler) {}

  DeclaredAllocMap buffers;

  // Every binding reaches this pass on a Var's type, so one VarLike override
  // covers assignment LHSs and iter_args alike. Parameters are NOT visited —
  // the traversal starts at the body — which is sound because the parser refuses
  // a one-argument `pl.MemRef(...)` in a parameter annotation, so no declaration
  // can arrive on a param.
  void VisitVarLike_(const VarPtr& op) override {
    if (op) Record(op);
    IRVisitor::VisitVarLike_(op);
  }

 private:
  void Record(const VarPtr& var) {
    auto binding = GetDeclaredAlloc(var->GetType());
    if (!binding) return;
    const VarPtr& base = binding->base_;
    auto tile_type = As<TileType>(var->GetType());

    const MemorySpace space = tile_type->GetMemorySpace().value_or(MemorySpace::DDR);
    auto size = utils::StaticPhysicalAllocationBytes(tile_type, space, handler_);
    CHECK_SPAN(size.has_value(), var->span_)
        << "Tile '" << var->name_hint_ << "' is bound to the declared allocation '" << base->name_hint_
        << "' but has a dynamic shape; a declared allocation must be sized at compile time";

    auto& alloc = buffers[base.get()];
    // One slot must hold the largest tile bound to ANY slot: the slots are
    // uniform, so a per-slot size would make the stride inconsistent.
    alloc.slot_size = std::max(alloc.slot_size, *size);

    CHECK_SPAN(binding->slot_count_ >= 1, var->span_)
        << "Declared allocation '" << base->name_hint_ << "' must have at least one slot, got "
        << binding->slot_count_;
    if (alloc.slot_count == 0) {
      alloc.slot_count = binding->slot_count_;
    }
    CHECK_SPAN(alloc.slot_count == binding->slot_count_, var->span_)
        << "References to the declared allocation '" << base->name_hint_
        << "' disagree on how many slots it has (" << alloc.slot_count << " vs " << binding->slot_count_
        << "); one declaration has one slot count";

    if (tile_type->memory_space_.has_value()) {
      auto [it, inserted] = spaces_.emplace(base.get(), *tile_type->memory_space_);
      CHECK_SPAN(inserted || it->second == *tile_type->memory_space_, var->span_)
          << "Tiles bound to the declared allocation '" << base->name_hint_
          << "' must all live in the same memory space, but '" << var->name_hint_ << "' is "
          << MemorySpaceToString(*tile_type->memory_space_) << " while the allocation is already "
          << MemorySpaceToString(it->second);
    }
  }

  const backend::BackendHandler* handler_ = nullptr;
  std::map<const Var*, MemorySpace> spaces_;
};

// Mutator to initialize MemRef for variables
class InitMemRefMutator : public IRMutator {
 public:
  InitMemRefMutator(const DeclaredAllocMap& declared_allocs, const backend::BackendHandler* handler,
                    FunctionType func_type)
      : declared_allocs_(declared_allocs), handler_(handler), func_type_(func_type) {}

  /// Whether `type` is bound to one of this function's declared allocations.
  [[nodiscard]] bool HasUserBinding(const TypePtr& type) const {
    if (declared_allocs_.empty()) return false;
    auto binding = GetDeclaredAlloc(type);
    return binding && declared_allocs_.count(binding->base_.get()) > 0;
  }

  /// The MemRef a user binding asks for, sized to the slot it selects.
  /// Returns nullopt when `type` carries no binding.
  std::optional<MemRefPtr> UserBoundMemRef(const TypePtr& type) const {
    if (declared_allocs_.empty()) return std::nullopt;
    auto binding = GetDeclaredAlloc(type);
    if (!binding) return std::nullopt;
    auto it = declared_allocs_.find(binding->base_.get());
    if (it == declared_allocs_.end()) return std::nullopt;
    // Every bound tile gets the SAME base Ptr — that shared identity is what
    // makes them share storage — and the slot index becomes the byte offset:
    // `index * slot_size`. A constant index folds here, so a single-slot or
    // constant-slot declaration keeps the ConstInt offset every downstream pass
    // already expects. A runtime index survives as an expression that
    // AllocateMemoryAddr adds the base address to and codegen lowers into the
    // tile's address assignment.
    //
    // Size is ONE SLOT, not the whole allocation: `size_` is the extent of the
    // region this MemRef denotes, and `[offset, offset + size_)` is the range
    // `MayAlias` intersects and the address verifier bounds-checks. Sizing a slot
    // to the whole set would make slot 1 of two span `[S, 3S)` — overrunning the
    // allocation for the verifier, and overlapping slot 0 for MayAlias, which
    // would report the ping-pong's two halves as aliasing. The allocation itself
    // is sized to the full set separately, where the alloc statement is built.
    //
    // The slot geometry rides along on the resolved MemRef. Resolving the index
    // into an offset answers *where* the slot lands; it does not stop the MemRef
    // from being slot k of an N-slot allocation, and that is what lets PTO codegen
    // emit `pto.alloc_multi_tile` / `pto.multi_tile_get` instead of N unrelated
    // allocs. `is_pinned_` still clears here: the declaration is resolved, and the
    // flag is what confines MemRef rebuilds to the pre-InitMemRef window.
    return std::make_shared<MemRef>(binding->base_, SlotByteOffset(*binding, it->second),
                                    it->second.slot_size, Span::unknown(), /*is_pinned=*/false,
                                    binding->slot_count_, binding->slot_index_);
  }

  /// The byte offset a declaration's slot index denotes, folded when constant.
  ///
  /// Returns `binding`'s own offset untouched for an unsubscripted declaration —
  /// there is no slot arithmetic to do, and slot 0 of a 1-slot allocation is the
  /// allocation itself.
  static ExprPtr SlotByteOffset(const MemRef& binding, const DeclaredAlloc& alloc) {
    if (!binding.slot_index_.has_value() || !*binding.slot_index_) return binding.byte_offset_;
    const ExprPtr& slot_index = *binding.slot_index_;
    const auto& span = binding.span_;
    if (auto const_index = As<ConstInt>(slot_index)) {
      return std::make_shared<ConstInt>(
          static_cast<int64_t>(static_cast<uint64_t>(const_index->value_) * alloc.slot_size), DataType::INT64,
          span);
    }
    // INDEX for the runtime product: the index comes from loop variables, which are
    // INDEX-typed, and the existing dynamic-offset expressions (tile.slice views)
    // are built the same way. Codegen widens to the i64 the PTOAS `alloc_tile` addr
    // operand wants when it lowers the address.
    auto stride = std::make_shared<ConstInt>(static_cast<int64_t>(alloc.slot_size), DataType::INDEX, span);
    return std::make_shared<Mul>(slot_index, stride, DataType::INDEX, span);
  }

  // Resolve memory space from TileType::memory_space_ field (set by InferTileMemorySpace),
  // falling back to DDR when default_to_ddr is true.
  [[nodiscard]] static std::optional<MemorySpace> ResolveTileMemorySpace(const TypePtr& type,
                                                                         bool default_to_ddr = false) {
    if (auto tile_type = std::dynamic_pointer_cast<const TileType>(type)) {
      if (tile_type->memory_space_.has_value()) {
        return tile_type->memory_space_;
      }
    }

    if (default_to_ddr) {
      return MemorySpace::DDR;
    }
    return std::nullopt;
  }

  // Calculate allocation size and create MemRef with the given memory space.
  std::optional<MemRefPtr> CreateMemRef(const ShapedTypePtr& type, const VarPtr& var,
                                        std::optional<MemorySpace> memory_space) {
    const std::string var_name = var ? var->name_hint_ : "<anonymous>";
    const Span& err_span = var ? var->span_ : Span::unknown();
    INTERNAL_CHECK_SPAN(memory_space.has_value(), err_span)
        << "Internal error: memory_space must be resolved before CreateMemRef";

    if (As<TileType>(type)) {
      for (size_t i = 0; i < type->shape_.size(); ++i) {
        auto const_dim = As<ConstInt>(type->shape_[i]);
        INTERNAL_CHECK_SPAN(const_dim, err_span)
            << "InitMemRef requires static shape for variable '" << var_name << "', but shape element " << i
            << " is dynamic. Fix the upstream op to keep TileType.shape static and put runtime "
               "extent in TileView.valid_shape instead.";
        INTERNAL_CHECK_SPAN(const_dim->value_ > 0, err_span)
            << "InitMemRef requires positive shape for variable '" << var_name << "', but shape element " << i
            << " is " << const_dim->value_;
      }
    }

    auto static_size = utils::StaticPhysicalAllocationBytes(type, *memory_space, handler_);
    INTERNAL_CHECK_SPAN(!As<TileType>(type) || static_size.has_value(), err_span)
        << "InitMemRef cannot represent the physical allocation size for static tile variable '" << var_name
        << "' without overflowing 64-bit bytes";
    const uint64_t size_bytes = static_size.value_or(0);

    auto base =
        std::make_shared<Var>(BuildBasePtrName(*memory_space, next_id_++), GetPtrType(), Span::unknown());
    return std::make_shared<MemRef>(base, static_cast<int64_t>(0), size_bytes);
  }

  std::optional<MemorySpace> ExtractMemorySpaceFromType(const TypePtr& type) {
    auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(type);
    if (!shaped_type) {
      return std::nullopt;
    }
    return shaped_type->GetMemorySpace();
  }

  // Process IterArg variable (inherits MemRef from initValue)
  VarPtr ProcessIterArg(const VarPtr& old_var) {
    auto iter_arg = std::static_pointer_cast<const IterArg>(old_var);

    // Visit initValue to get its updated MemRef
    auto new_init = VisitExpr(iter_arg->initValue_);

    // Extract MemRef from initValue and create new type
    auto memref = GetTypeMemRef(new_init->GetType());
    auto old_var_expr = std::static_pointer_cast<const Expr>(old_var);
    auto source_memory_space = ExtractMemorySpaceFromType(new_init->GetType());
    TypePtr new_type = CloneTypeWithMemRefAndRemapExprs(
        old_var_expr->GetType(), memref, [this](const ExprPtr& expr) { return VisitExpr(expr); },
        source_memory_space);

    return std::make_shared<IterArg>(iter_arg->name_hint_, new_type, new_init, iter_arg->span_);
  }

  // Process normal Var variable (creates new MemRef based on usage)
  VarPtr ProcessNormalVar(const VarPtr& var) {
    auto var_expr = std::static_pointer_cast<const Expr>(var);
    TypePtr new_type = var_expr->GetType();

    // ArrayType lives on the on-core scalar register file / C stack and never
    // needs a runtime MemRef — codegen lowers it to a stack array directly.
    // Leave the var untouched so the original ArrayType (no memref_) propagates.
    if (std::dynamic_pointer_cast<const ArrayType>(var_expr->GetType())) {
      return var;
    }

    if (auto shaped_type = std::dynamic_pointer_cast<const ShapedType>(var_expr->GetType())) {
      // A tile must already have its space: InitMemRef declares
      // IRProperty::TileMemoryInferred as required, so an unset one here means
      // InferTileMemorySpace did not cover this function. Defaulting it to DDR
      // instead produced a "tile" in global memory that later reads as a real
      // placement -- a vector op with a DDR operand, and a memory planner
      // packing every such tile at offset 0. Fail with the tile's name rather
      // than let that travel. Non-tile ShapedTypes (tensors) keep the DDR
      // default, which is their correct home.
      if (auto tile_type = std::dynamic_pointer_cast<const TileType>(var_expr->GetType())) {
        if (IsInCoreType(func_type_)) {
          // A device tile with no space after pass 17 is a compiler bug:
          // InferTileMemorySpace covers every IsInCoreType function and is
          // declared to produce TileMemoryInferred.
          INTERNAL_CHECK_SPAN(tile_type->memory_space_.has_value(), var->span_)
              << "Internal error: tile '" << var->name_hint_
              << "' reached InitMemRef with no memory space; InferTileMemorySpace must place every "
                 "tile in a device function before this pass runs";
        } else {
          // Outside a device function there is no on-chip buffer for a tile to
          // live in, so no space can be inferred and none should be defaulted
          // -- the old DDR fallback produced a "tile" in global memory that
          // later read as a real placement. This is an authoring error, not a
          // compiler bug: tile ops belong in a device function.
          CHECK_SPAN(tile_type->memory_space_.has_value(), var->span_)
              << "The tile '" << var->name_hint_ << "' lives in a " << FunctionTypeToString(func_type_)
              << " function, which has no on-chip memory to place it in. Tiles are on-chip "
                 "hardware state, so tile ops belong in a device function -- move this code into "
                 "an InCore function, or into a pl.scope() the outliner turns into one.";
        }
      }
      // Resolve memory space once, pass to both CreateMemRef and CloneType
      auto memory_space = ResolveTileMemorySpace(var_expr->GetType(), /*default_to_ddr=*/true);
      // A declared allocation wins over a fresh one: the whole point is
      // that this tile lands in the buffer the kernel author named.
      auto memref = UserBoundMemRef(var_expr->GetType());
      if (!memref.has_value()) memref = CreateMemRef(shaped_type, var, memory_space);
      new_type = CloneTypeWithMemRefAndRemapExprs(
          var_expr->GetType(), memref, [this](const ExprPtr& expr) { return VisitExpr(expr); }, memory_space);
    } else {
      // Non-shaped types (e.g. ScalarType for dynamic-shape dimensions like M, N)
      // don't need MemRef initialization — return the original Var to preserve
      // pointer identity across all type annotations that reference it.
      return var;
    }

    return std::make_shared<Var>(var->name_hint_, new_type, var->span_);
  }

  // Create a new Var with MemRef initialized
  VarPtr GetNewVar(const VarPtr& old_var) {
    // Check cache first to prevent infinite recursion
    auto it = var_map_.find(old_var);
    if (it != var_map_.end()) {
      return it->second;
    }

    // Dispatch based on variable type
    VarPtr new_var;
    if (std::dynamic_pointer_cast<const IterArg>(old_var)) {
      new_var = ProcessIterArg(old_var);
    } else {
      new_var = ProcessNormalVar(old_var);
    }

    var_map_[old_var] = new_var;
    return new_var;
  }

  ExprPtr VisitExpr_(const VarPtr& op) override {
    return std::static_pointer_cast<const Expr>(GetNewVar(op));
  }

  ExprPtr VisitExpr_(const IterArgPtr& op) override {
    // IterArg extends Var, so cast to VarPtr for processing
    auto var_ptr = std::static_pointer_cast<const Var>(op);
    return std::static_pointer_cast<const Expr>(GetNewVar(var_ptr));
  }

  /**
   * @brief Create a view MemRef from a source expression for the LHS variable of an assignment.
   *
   * Creates a NEW MemRef with the same base_ Ptr as the source's MemRef,
   * accumulated byte offset, and computed size from the output shape.
   * Returns nullptr if the source has no MemRef.
   */
  StmtPtr ShareMemRefFrom(const ExprPtr& source, const AssignStmtPtr& op, const ExprPtr& new_value) {
    auto parent_memref_opt = GetTypeMemRef(source->GetType());
    if (!parent_memref_opt.has_value()) return nullptr;
    const auto& parent_memref = *parent_memref_opt;

    // Compute additional byte offset from the view op (if applicable)
    ExprPtr additional_offset = MakeZeroByteOffset();
    if (auto call = As<Call>(new_value)) {
      additional_offset = ComputeViewByteOffset(call, source->GetType());
    }

    // Accumulate: total_offset = parent.byte_offset + additional_offset
    ExprPtr total_offset = AddByteOffsets(parent_memref->byte_offset_, additional_offset);

    auto source_ms = ExtractMemorySpaceFromType(source->GetType());

    // Keep ordinary aliases/reshapes at the parent's physical range. A slice
    // narrows that range to the packed parent-layout envelope it can touch.
    // Crucially, this is a VIEW span, not a fresh allocation footprint: an Acc
    // slice beginning at row 16 of a 32-row INT32 allocation must end at the
    // root boundary, not acquire another 32 rows of L0C padding.
    uint64_t view_size = parent_memref->size_;  // default: same size as parent
    if (auto call = As<Call>(new_value)) {
      auto parent_shaped = std::dynamic_pointer_cast<const ShapedType>(source->GetType());
      auto out_shaped = std::dynamic_pointer_cast<const ShapedType>(op->var_->GetType());
      if (auto slice_span = StaticSliceViewSpanBytes(call, parent_shaped, out_shaped)) {
        view_size = *slice_span;
      }
    }

    // For pure aliases (no offset change, same size), share the SAME MemRef shared_ptr
    // to preserve base_ Ptr identity. MemoryReuse builds sharing groups by base_ Ptr —
    // variables sharing the same base_ are merged into one lifetime group.
    bool is_pure_alias = (view_size == parent_memref->size_);
    if (is_pure_alias) {
      if (auto const_offset = As<ConstInt>(additional_offset)) {
        is_pure_alias = (const_offset->value_ == 0);
      } else {
        is_pure_alias = false;
      }
    }
    MemRefPtr view_memref = is_pure_alias
                                ? parent_memref
                                : std::make_shared<MemRef>(parent_memref->base_, total_offset, view_size);

    std::optional<MemRefPtr> view_opt = view_memref;
    TypePtr new_type = CloneTypeWithMemRefAndRemapExprs(
        op->var_->GetType(), view_opt, [this](const ExprPtr& e) { return VisitExpr(e); }, source_ms);
    VarPtr new_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    var_map_[op->var_] = new_var;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

  // Rebuild an AssignStmt whose LHS tile var keeps its TileType + memory_space but
  // carries no MemRef. Used for tiles that own no general-pool buffer: cross-core
  // tpop results and zero-copy views over them (codegen lowers those to
  // pto.treshape over the source rather than a fresh alloc_tile).
  StmtPtr MakeMemRefLessAssign(const AssignStmtPtr& op, const ExprPtr& new_value) {
    auto var_expr = std::static_pointer_cast<const Expr>(op->var_);
    if (auto tile_type = std::dynamic_pointer_cast<const TileType>(var_expr->GetType())) {
      for (size_t i = 0; i < tile_type->shape_.size(); ++i) {
        INTERNAL_CHECK_SPAN(As<ConstInt>(tile_type->shape_[i]), op->var_->span_)
            << "InitMemRef requires static shape for variable '" << op->var_->name_hint_
            << "', but shape element " << i
            << " is dynamic. Fix the upstream op to keep TileType.shape static and put runtime "
               "extent in TileView.valid_shape instead.";
      }
    }
    TypePtr new_type = CloneTypeWithMemRefAndRemapExprs(
        var_expr->GetType(), std::nullopt, [this](const ExprPtr& expr) { return VisitExpr(expr); },
        ResolveTileMemorySpace(var_expr->GetType()));
    auto new_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    var_map_[op->var_] = new_var;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // First visit the value (RHS)
    auto new_value = VisitExpr(op->value_);

    // A tile that owns no general-pool buffer — a cross-core tpop result (its
    // data lives in the reserved C2V/V2C slot, addressed via the pipe), or a
    // zero-copy view / plain alias chained off one — stays MemRef-less, so
    // AllocateMemoryAddr reserves no phantom buffer and no fresh, disconnected
    // buffer is created.
    if (As<TileType>(op->var_->GetType()) && ProducesBufferLessTile(new_value, [](const Var* v) {
          return !GetTypeMemRef(v->GetType()).has_value();
        })) {
      return MakeMemRefLessAssign(op, new_value);
    }

    // Check if the RHS is a Call expression
    if (auto call = std::dynamic_pointer_cast<const Call>(op->value_)) {
      LOG_DEBUG << "Processing AssignStmt for " << op->var_->name_hint_ << " with call to "
                << call->op_->name_;

      // A view / in-place result physically IS its source's buffer, so binding
      // it to a different one cannot be honored. Say so instead of silently
      // dropping the binding — the user asked for something impossible.
      CHECK_SPAN(!op_predicates::OutputInheritsSourceBuffer(call->op_->name_) ||
                     !HasUserBinding(op->var_->GetType()),
                 op->var_->span_)
          << "Tile '" << op->var_->name_hint_ << "' is produced by '" << call->op_->name_
          << "', which lands in its source tile's allocation, so it cannot be given one of its own. "
             "Bind the source tile instead.";

      // Handle view operations: output should share MemRef with input tile.
      // A pure metadata view (slice/reshape/...) inherits its input's buffer.  A
      // permuting inherit-input op — tile.transpose — must NOT: pto.ttrans is
      // not in-place safe (the unaligned scalar path writes dst directly from
      // src), so the transpose output always gets a fresh buffer.
      if (IsViewOperation(call->op_->name_) && call->args_.size() > 0) {
        LOG_DEBUG << "Detected view operation: " << call->op_->name_;
        // Get the input tile (first argument) after mutation
        auto new_call = std::dynamic_pointer_cast<const Call>(new_value);
        if (new_call && !new_call->args_.empty()) {
          const bool may_inherit = !IsDataPermutingInheritOp(call->op_);
          if (may_inherit) {
            auto result = ShareMemRefFrom(new_call->args_[0], op, new_value);
            if (result) {
              LOG_DEBUG << "Sharing MemRef from input tile to " << op->var_->name_hint_;
              return result;
            }
            LOG_DEBUG << "Input tile has no MemRef yet";
          }
        }
      }

      // Handle ops whose output reuses a specific input arg's MemRef (registry-based)
      auto reuse_arg_idx = GetOutputReusesInputArg(call->op_->name_);
      if (reuse_arg_idx.has_value()) {
        auto new_call = std::dynamic_pointer_cast<const Call>(new_value);
        if (new_call && *reuse_arg_idx < new_call->args_.size()) {
          auto result = ShareMemRefFrom(new_call->args_[*reuse_arg_idx], op, new_value);
          if (result) {
            LOG_DEBUG << "Reusing MemRef from input arg " << *reuse_arg_idx << " for "
                      << op->var_->name_hint_;
            return result;
          }
        }
      }
    }

    // Tile alias: a = b where b is a tile Var — share b's MemRef.
    // Without this, the alias gets a fresh MemRef that is never written to,
    // breaking IfStmt return_vars that inherit the alias's empty MemRef.
    if (auto value_var = As<Var>(new_value)) {
      if (As<TileType>(op->var_->GetType())) {
        auto result = ShareMemRefFrom(value_var, op, new_value);
        if (result) return result;
      }
    }

    // Default case: visit the variable normally
    auto new_var = GetNewVar(op->var_);
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Manual traversal of ForStmt fields to ensure correct MemRef assignment:
    // - iter_args inherit MemRef from initValue (via ProcessIterArg)
    // - return_vars share MemRef with corresponding yield values

    // Step 1: Visit loop bounds and loop_var
    auto new_loop_var_expr = VisitExpr(op->loop_var_);
    auto new_loop_var = As<Var>(new_loop_var_expr);
    INTERNAL_CHECK_SPAN(new_loop_var, op->span_)
        << "Internal error: ForStmt loop_var is not a Var after mutation";
    auto new_start = VisitExpr(op->start_);
    auto new_stop = VisitExpr(op->stop_);
    auto new_step = VisitExpr(op->step_);

    // Step 2: Process iter_args (inherits MemRef from initValue via ProcessIterArg)
    std::vector<IterArgPtr> new_iter_args;
    new_iter_args.reserve(op->iter_args_.size());
    for (const auto& ia : op->iter_args_) {
      auto new_ia_expr = VisitExpr(ia);
      auto new_ia =
          std::dynamic_pointer_cast<const IterArg>(std::static_pointer_cast<const IRNode>(new_ia_expr));
      INTERNAL_CHECK_SPAN(new_ia, op->span_)
          << "Internal error: ForStmt iter_arg is not an IterArg after mutation";
      new_iter_args.push_back(new_ia);
    }

    // Register old->new IterArg mappings so body references are substituted
    for (size_t i = 0; i < op->iter_args_.size(); ++i) {
      if (new_iter_args[i].get() != op->iter_args_[i].get()) {
        var_remap_[op->iter_args_[i].get()] = new_iter_args[i];
      }
    }

    // Step 3: Visit body
    auto new_body = VisitStmt(op->body_);

    // Clean up IterArg remappings
    for (const auto& old_iter_arg : op->iter_args_) {
      var_remap_.erase(old_iter_arg.get());
    }

    // Step 4: Visit return_vars
    std::vector<VarPtr> new_return_vars;
    new_return_vars.reserve(op->return_vars_.size());
    for (const auto& rv : op->return_vars_) {
      auto new_rv_expr = VisitExpr(rv);
      auto new_rv = As<Var>(new_rv_expr);
      INTERNAL_CHECK_SPAN(new_rv, op->span_)
          << "Internal error: ForStmt return_var is not a Var after mutation";
      new_return_vars.push_back(new_rv);
    }

    auto new_for = MutableCopy(op);
    new_for->loop_var_ = new_loop_var;
    new_for->start_ = new_start;
    new_for->stop_ = new_stop;
    new_for->step_ = new_step;
    new_for->iter_args_ = new_iter_args;
    new_for->body_ = new_body;
    new_for->return_vars_ = new_return_vars;

    // Patch return_vars so each shares its iter_arg's MemRef (inherited from initValue).
    // This establishes the invariant that initValue/iter_arg/return_var all share the
    // same MemRef buffer — the loop accumulator lives in one place for the whole loop.
    // Any yield-vs-buffer mismatch is the concern of downstream passes.
    if (new_for->iter_args_.empty() || new_for->return_vars_.empty()) {
      return new_for;
    }

    auto get_iter_arg_var = [&](size_t i) -> VarPtr {
      if (i >= new_for->iter_args_.size()) return nullptr;
      return std::static_pointer_cast<const Var>(new_for->iter_args_[i]);
    };
    auto [patched, changed] =
        PatchReturnVarsFromYield(new_for->return_vars_, op->return_vars_, get_iter_arg_var);

    if (!changed) return new_for;

    new_for->return_vars_ = std::move(patched);
    return new_for;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    auto new_if = As<IfStmt>(result);
    if (!new_if || new_if->return_vars_.empty()) return result;

    auto then_yield = FindYieldStmt(new_if->then_body_);
    auto else_yield = new_if->else_body_.has_value() ? FindYieldStmt(new_if->else_body_.value()) : nullptr;
    if (!then_yield && !else_yield) return result;

    auto get_yield_var = [&](size_t i) -> VarPtr {
      VarPtr var = nullptr;
      if (then_yield && i < then_yield->value_.size()) var = As<Var>(then_yield->value_[i]);
      if (!var && else_yield && i < else_yield->value_.size()) var = As<Var>(else_yield->value_[i]);
      return var;
    };
    auto [patched, changed] = PatchReturnVarsFromYield(new_if->return_vars_, op->return_vars_, get_yield_var);

    if (!changed) return result;

    auto patched_if = MutableCopy(new_if);
    patched_if->return_vars_ = std::move(patched);
    return patched_if;
  }

 private:
  // Shared logic for ForStmt/IfStmt: patch each return_var to share its yield value's MemRef.
  // get_yield_var(i) resolves the yield variable for the i-th return_var.
  using YieldVarResolver = std::function<VarPtr(size_t)>;
  std::pair<std::vector<VarPtr>, bool> PatchReturnVarsFromYield(const std::vector<VarPtr>& new_return_vars,
                                                                const std::vector<VarPtr>& old_return_vars,
                                                                const YieldVarResolver& get_yield_var) {
    bool changed = false;
    std::vector<VarPtr> patched;
    patched.reserve(new_return_vars.size());

    for (size_t i = 0; i < new_return_vars.size(); ++i) {
      auto yield_var = get_yield_var(i);
      auto yield_tile = yield_var ? GetTileTypeWithMemRef(yield_var->GetType()) : nullptr;
      if (As<TileType>(new_return_vars[i]->GetType()) && yield_tile) {
        auto new_type = CloneTypeWithMemRef(new_return_vars[i]->GetType(), yield_tile->memref_,
                                            yield_tile->GetMemorySpace());
        auto new_rv =
            std::make_shared<Var>(new_return_vars[i]->name_hint_, new_type, new_return_vars[i]->span_);
        var_map_[old_return_vars[i]] = new_rv;
        patched.push_back(new_rv);
        changed = true;
      } else {
        patched.push_back(new_return_vars[i]);
      }
    }
    return {std::move(patched), changed};
  }

  std::map<VarPtr, VarPtr> var_map_;
  const DeclaredAllocMap& declared_allocs_;
  const backend::BackendHandler* handler_ = nullptr;
  FunctionType func_type_ = FunctionType::InCore;
  uint64_t next_id_ = 0;
};

// Insert alloc statements at the beginning of a function body.
StmtPtr InsertAllocsIntoBody(const StmtPtr& body, const std::vector<StmtPtr>& alloc_stmts) {
  if (alloc_stmts.empty()) return body;

  std::vector<StmtPtr> new_seq_stmts;
  new_seq_stmts.insert(new_seq_stmts.end(), alloc_stmts.begin(), alloc_stmts.end());

  const Span& span = body ? body->span_ : alloc_stmts.front()->span_;
  if (body) {
    if (auto seq = As<SeqStmts>(body)) {
      new_seq_stmts.insert(new_seq_stmts.end(), seq->stmts_.begin(), seq->stmts_.end());
    } else {
      new_seq_stmts.push_back(body);
    }
  }

  return SeqStmts::Flatten(std::move(new_seq_stmts), span);
}

/**
 * @brief Initialize MemRef for all variables in a function
 *
 * This transformation:
 * 1. Normalizes statement structure (ensures SeqStmts)
 * 2. Initializes the MemRef field for all Var nodes
 * 3. Creates tile.alloc operations for non-DDR MemRefs (addr=-1, unallocated)
 *
 * Memory space is read from TileType::memory_space_ (set by InferTileMemorySpace).
 * Variables without memory_space default to DDR.
 */
FunctionPtr TransformInitMemRef(const FunctionPtr& func) {
  // Step 1: Normalize statement structure to ensure SeqStmts
  auto normalized_func = NormalizeStmtStructure(func);

  const auto* ctx = PassContext::Current();
  const backend::BackendHandler* handler = nullptr;
  if (backend::BackendConfig::IsConfigured()) {
    handler = ctx ? ctx->GetBackendHandler() : backend::GetBackend()->GetHandler();
  }

  // Step 2: Resolve author-declared allocations (`pl.Tile[..., pl.MemRef("name"),
  // ...]`), then mutate variables to initialize their MemRef. They must be
  // collected up front: a declared allocation's size is the max over ALL tiles
  // bound to it, which is only known after the whole function has been seen.
  DeclaredAllocCollector declared_alloc_collector(handler);
  declared_alloc_collector.VisitStmt(normalized_func->body_);
  const DeclaredAllocMap& declared_allocs = declared_alloc_collector.buffers;

  // The isolation guarantee is enforced by MemoryReuse under PYPTO and by the
  // allocation constraints under DSA-RP. PTOAS replaces PyPTO memory planning
  // wholesale; honoring the declaration's allocation but not its isolation
  // would hand back exactly the coalescing the author declared it to prevent,
  // so reject the unsupported combination rather than degrade quietly.
  //
  // A MULTI-SLOT declaration is the exception: it lowers to a ptoas
  // `pto.alloc_multi_tile` region, and ptoas plans the N slots into disjoint
  // physical segments it is explicitly forbidden to alias-merge. The separation
  // the author asked for — slot k is not slot j — is therefore carried into ptoas
  // rather than lost, which is the whole reason the multi-buffer form exists. A
  // single-slot declaration has no such counterpart and stays rejected.
  if (ctx != nullptr && ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS) {
    for (const auto& [base, alloc] : declared_allocs) {
      CHECK(alloc.slot_count > 1)
          << "A single-slot declared allocation (pl.MemRef(\"" << base->name_hint_
          << "\")) is not supported under memory_planner=PTOAS: ptoas owns memory planning and "
             "would be free to coalesce the allocations you separated. Declare it with "
             "pl.MemRef(slots=N) — N slots become one ptoas multi-buffer region whose slots ptoas "
             "keeps disjoint — or compile with the default PyPTO memory planner.";
    }
  }

  InitMemRefMutator mutator(declared_allocs, handler, normalized_func->func_type_);

  std::vector<VarPtr> new_params;
  new_params.reserve(normalized_func->params_.size());
  for (const auto& var : normalized_func->params_) {
    auto new_param = mutator.GetNewVar(var);
    INTERNAL_CHECK_SPAN(new_param, var->span_) << "Failed to get new param";
    new_params.push_back(new_param);
  }

  auto new_body = mutator.VisitStmt(normalized_func->body_);

  auto result_func = MutableCopy(normalized_func);
  result_func->params_ = new_params;
  result_func->body_ = new_body;

  // Step 3: Collect ALL MemRefs (DDR gets tensor.alloc, on-chip gets tile.alloc)
  memref_collectors::MemRefWithSpaceCollector collector(/*skip_ddr=*/false);
  for (const auto& param : new_params) {
    collector.VisitExpr(param);
  }
  collector.VisitStmt(new_body);

  const auto& memrefs = collector.memrefs;
  if (memrefs.empty()) return result_func;

  // Deduplicate by base_ Ptr — one alloc per unique base
  std::set<const Var*> seen_bases;
  std::vector<StmtPtr> alloc_stmts;
  alloc_stmts.reserve(memrefs.size());
  for (const auto& [memref, memory_space] : memrefs) {
    if (seen_bases.insert(memref->base_.get()).second) {
      // A declared allocation covers all its slots; the MemRef that happens to be
      // seen first names only one of them, so take the size from the collector.
      auto declared = declared_allocs.find(memref->base_.get());
      const bool pinned = declared != declared_allocs.end();
      auto alloc_size = pinned ? std::make_optional(declared->second.TotalSize()) : std::optional<uint64_t>{};
      alloc_stmts.push_back(CreateAllocStatement(memref, memory_space, pinned, alloc_size));
    }
  }

  // Step 4: Insert alloc statements at the beginning of the function body
  auto final_body = InsertAllocsIntoBody(new_body, alloc_stmts);

  result_func->body_ = final_body;
  return result_func;
}

}  // namespace

// Factory function
namespace pass {
Pass InitMemRef() { return CreateFunctionPass(TransformInitMemRef, "InitMemRef", kInitMemRefProperties); }
}  // namespace pass

// ============================================================================
// HasMemRefs property verifier
// ============================================================================

namespace {

/**
 * @brief Checks all TileType variables have MemRef initialized.
 */
class HasMemRefsVerifier : public IRVisitor {
 public:
  explicit HasMemRefsVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op || !op->var_ || !op->var_->GetType()) return;
    auto tile_type = std::dynamic_pointer_cast<const TileType>(op->var_->GetType());
    if (tile_type && !tile_type->memref_.has_value()) {
      if (IsBufferLessByDesign(op)) {
        buffer_less_.insert(op->var_.get());
      } else {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "HasMemRefs", 0,
            "TileType variable '" + op->var_->name_hint_ + "' has no MemRef initialized", op->var_->span_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  // A cross-core tpop result (its data lives in the reserved C2V/V2C slot, not a
  // tile MemRef) and any zero-copy view / plain alias chained off such a result
  // legitimately carry no MemRef. Shares the rule with the MemRef-creating mutator
  // via ProducesBufferLessTile; here a source is buffer-less iff it is tracked.
  [[nodiscard]] bool IsBufferLessByDesign(const AssignStmtPtr& op) const {
    return ProducesBufferLessTile(op->value_, [this](const Var* v) { return buffer_less_.count(v) > 0; });
  }

  std::set<const Var*> buffer_less_;
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class HasMemRefsPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "HasMemRefs"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      HasMemRefsVerifier verifier(diagnostics);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateHasMemRefsPropertyVerifier() {
  return std::make_shared<HasMemRefsPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
