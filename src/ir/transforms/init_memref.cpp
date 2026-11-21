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

// Complexity exception for VF initialization validation: allocation and alias
// lookup remain indexed, but VerifyRegMemoryInitialization visits each helper
// with its call site's actual views, scalar values and initialized intervals.
// Its bound is O((N + E) log(N + E)), where E is the sum of helper-body sizes
// over VF call sites. E can be quadratic in the compact Program size when a
// large helper is reused many times. In/InOut and alias-return declarations
// alone cannot prove partially masked initialization. This exception requires
// approval under .claude/rules/pass-complexity.md before merging.

#include <algorithm>
#include <any>
#include <array>
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
#include <variant>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/op_rules.h"
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
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/implicit_vec_scratch.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/memref_collectors.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/normalize_stmt_structure.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/per_kernel_memory_planner.h"
#include "pypto/ir/transforms/utils/reg_memory_analysis.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/simplify.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/transforms/utils/vf_helpers.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

// A receive owns its FIFO slot. Views, physical writebacks and VF results can
// borrow that slot under fresh SSA names. Both allocation and HasMemRefs need
// the identity of the receive to distinguish two MemRef-less branch inputs.
class BorrowedTileRoots {
 public:
  [[nodiscard]] const Var* Find(const ExprPtr& value) const {
    auto var = AsVarLike(value);
    auto it = roots_.find(var.get());
    return it == roots_.end() ? nullptr : it->second;
  }

  [[nodiscard]] const Var* Definition(const VarPtr& result, const ExprPtr& value) const {
    if (op_predicates::IsTPop(As<Call>(value))) return result.get();
    return Find(op_predicates::GetBufferAliasSource(value));
  }

  void Bind(const VarPtr& result, const Var* root) {
    if (root) roots_[result.get()] = root;
  }

  void BindAlias(const VarPtr& result, const ExprPtr& source) { Bind(result, Find(source)); }

 private:
  std::map<const Var*, const Var*> roots_;
};

// Track declared tensor aliases and VF tile returns across tuple copies and control flow.
// A null field has no common operand and uses ordinary projection lowering.
class TupleResultStorage {
 public:
  struct Storage {
    TypePtr type;
    const Var* root = nullptr;  // Canonical FIFO receive for a borrowed Tile.
  };

  void Record(const VarPtr& result, const std::vector<ExprPtr>& args, const BorrowedTileRoots& borrowed) {
    auto& fields = fields_[result.get()];
    fields.clear();
    fields.reserve(args.size());
    for (const auto& arg : args) {
      fields.push_back(arg ? Storage{arg->GetType(), borrowed.Find(arg)} : Storage{});
    }
  }

  [[nodiscard]] const std::vector<Storage>* Find(const ExprPtr& value) const {
    auto var = AsVarLike(value);
    auto it = fields_.find(var.get());
    return it == fields_.end() ? nullptr : &it->second;
  }

  void Copy(const VarPtr& result, const ExprPtr& source) {
    if (const auto* fields = Find(source)) fields_[result.get()] = *fields;
  }

  [[nodiscard]] const Storage* Project(const ExprPtr& value) const {
    auto item = As<TupleGetItemExpr>(value);
    if (!item) return nullptr;
    const auto* fields = Find(item->tuple_);
    if (!fields) return nullptr;
    INTERNAL_CHECK_SPAN(item->index_ >= 0 && static_cast<size_t>(item->index_) < fields->size(), item->span_)
        << "Internal error: tuple result projection out of bounds";
    const auto& field = (*fields)[item->index_];
    return field.type ? &field : nullptr;
  }

  void Join(const VarPtr& result, const ExprPtr& lhs, const ExprPtr& rhs) {
    const auto* left = Find(lhs);
    const auto* right = Find(rhs);
    if (!left || !right || left->size() != right->size()) return;
    auto fields = *left;
    auto tuple = As<TupleType>(result->GetType());
    INTERNAL_CHECK_SPAN(tuple && tuple->types_.size() == fields.size(), result->span_)
        << "Internal error: tuple result arity changed at control-flow join";
    for (size_t i = 0; i < fields.size(); ++i) {
      auto& field = fields[i];
      const auto& other = (*right)[i];
      if (field.type == other.type && field.root == other.root) continue;
      if (field.root && field.root == other.root) {
        // Different views can borrow one slot. Preserve the join's logical
        // view rather than imposing one branch's metadata on both paths.
        field.type =
            CloneTypeWithMemRef(tuple->types_[i], std::nullopt, As<TileType>(field.type)->memory_space_);
      } else {
        field = {};
      }
    }
    fields_[result.get()] = std::move(fields);
  }

  void JoinBranches(const IfStmtPtr& branch, const YieldStmtPtr& lhs, const YieldStmtPtr& rhs) {
    if (!lhs || !rhs) return;
    for (size_t i = 0; i < branch->return_vars_.size() && i < lhs->value_.size() && i < rhs->value_.size();
         ++i) {
      if (As<TupleType>(branch->return_vars_[i]->GetType())) {
        Join(branch->return_vars_[i], lhs->value_[i], rhs->value_[i]);
      }
    }
  }

  void JoinLoop(const std::vector<VarPtr>& results, const std::vector<IterArgPtr>& carries,
                const StmtPtr& body) {
    auto yield = transform_utils::GetLastYieldStmt(body);
    if (!yield) return;
    for (size_t i = 0; i < results.size() && i < carries.size() && i < yield->value_.size(); ++i) {
      if (As<TupleType>(results[i]->GetType())) Join(results[i], carries[i], yield->value_[i]);
    }
  }

 private:
  std::map<const Var*, std::vector<Storage>> fields_;
};

/// Byte envelope of a static slice of an NZ-boxed accumulator (L0C) parent.
///
/// L0C stores 16x16 boxes column of boxes first, so the envelope is measured in
/// whole boxes, not in row-major-dense elements: it runs from the window's first
/// box to the end of its last one. The result is origin-independent, so it can
/// be derived from the shapes alone exactly like the row-major form below.
///
/// Only reached for the windows ``GetSliceAccumulatorGeometry`` accepts — full
/// parent row extent, box-aligned column origin — and for those the envelope is
/// not an approximation at all: the window's box columns are consecutive, so the
/// envelope is exactly the ``cols * rows * elem_bytes`` it occupies, and two
/// disjoint column windows therefore read as disjoint to lifetime analysis. A row
/// window would be strided and its envelope a gross over-approximation; that is
/// one of the reasons the guard excludes it rather than the other way round.
std::optional<uint64_t> StaticAccSliceViewSpanBytes(
    const tile_view_semantics::AccumulatorTileGeometry& geometry, const ShapedTypePtr& parent,
    const MakeTuplePtr& requested_shape, const DataType& dtype) {
  if (requested_shape->elements_.size() != 2 || parent->shape_.size() != 2) return std::nullopt;

  const std::array<int64_t, 2> box_extent = {geometry.box.rows, geometry.box.cols};
  const std::array<int64_t, 2> strides = {geometry.row_stride, geometry.col_stride};

  uint64_t last_box_start = 0;
  for (size_t i = 0; i < requested_shape->elements_.size(); ++i) {
    auto parent_dim = As<ConstInt>(parent->shape_[i]);
    auto view_dim = As<ConstInt>(requested_shape->elements_[i]);
    if (!parent_dim || !view_dim || view_dim->value_ <= 0 || view_dim->value_ > parent_dim->value_) {
      return std::nullopt;
    }
    // Round up: a window narrower than one box still occupies a whole box.
    const int64_t boxes = (view_dim->value_ + box_extent[i] - 1) / box_extent[i];
    const uint64_t contribution = static_cast<uint64_t>(boxes - 1) * static_cast<uint64_t>(box_extent[i]) *
                                  static_cast<uint64_t>(strides[i]);
    if (last_box_start > std::numeric_limits<uint64_t>::max() - contribution) return std::nullopt;
    last_box_start += contribution;
  }

  const uint64_t box_elements = static_cast<uint64_t>(geometry.box_elements);
  if (last_box_start > std::numeric_limits<uint64_t>::max() - box_elements) return std::nullopt;
  return storage_size::StaticStorageBytes(last_box_start + box_elements, dtype);
}

// The positive-stride descriptor covers a byte envelope, including its gaps.
// Only the individual addresses selected by the descriptor are read/written.
std::optional<uint64_t> StaticTensorViewSpanBytes(const TensorTypePtr& tensor,
                                                  const std::vector<ExprPtr>& shape) {
  const auto& view = tensor->tensor_view_;
  INTERNAL_CHECK(view && view->stride.size() == shape.size()) << "Tensor view stride rank mismatch";
  uint64_t elements = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    const auto extent = As<ConstInt>(shape[i]);
    const auto stride = As<ConstInt>(view->stride[i]);
    if (!extent || !stride || extent->value_ <= 0 || stride->value_ <= 0) return std::nullopt;
    uint64_t term;
    if (__builtin_mul_overflow(static_cast<uint64_t>(extent->value_ - 1),
                               static_cast<uint64_t>(stride->value_), &term) ||
        __builtin_add_overflow(elements, term, &elements)) {
      return std::nullopt;
    }
  }
  return storage_size::StaticStorageBytes(elements, tensor->dtype_);
}

/// Byte envelope touched by a static packed slice, relative to its first
/// element. This is deliberately distinct from the physical allocation size:
/// L0C row padding belongs to the root allocation and must not be applied again
/// to a view beginning part-way through that allocation.
std::optional<uint64_t> StaticSliceViewSpanBytes(const CallPtr& call, const ShapedTypePtr& parent,
                                                 const ShapedTypePtr& view) {
  if (!call || (!IsOp(call, "tensor.slice") && !IsOp(call, "tile.slice"))) return std::nullopt;
  if (!parent || !view || call->args_.size() < 3) return std::nullopt;
  // Use the requested pre-drop shape rather than the result rank so
  // tile.slice(..., drop_dims=...) keeps the parent strides of dimensions that
  // disappear from the result type.
  auto requested_shape = As<MakeTuple>(call->args_[1]);
  if (!requested_shape || parent->shape_.size() != requested_shape->elements_.size()) {
    return std::nullopt;
  }

  auto tensor = AsTensorTypeLike(parent);
  auto indices = As<TupleType>(call->args_[2]->GetType());
  if (indices &&
      HasRectangularSliceStrides(tensor, requested_shape->elements_.size(), indices->types_.size())) {
    auto extents = requested_shape->elements_;
    const auto offsets = ExtractTupleElements(call->args_[2], indices->types_.size());
    // A padded or clamped rectangle can extend past the parent; its GM
    // envelope ends at the final physical element available on each axis.
    for (size_t i = 0; i < extents.size(); ++i) {
      auto bound = As<ConstInt>(parent->shape_[i]);
      auto start = As<ConstInt>(offsets[i]);
      auto extent = As<ConstInt>(extents[i]);
      if (bound && start && extent && start->value_ >= 0 && bound->value_ >= 0) {
        const int64_t available = bound->value_ - start->value_;
        if (available < extent->value_) extents[i] = MakeIndexConst(std::max<int64_t>(0, available));
      }
    }
    return StaticTensorViewSpanBytes(tensor, extents).value_or(0);
  }

  // Keep the span in the same layout model as the byte offset that
  // ComputeViewByteOffset produced for this very slice — same predicate, same
  // inputs. Mixing the two would leave the MemRef internally inconsistent.
  if (auto geometry = GetSliceAccumulatorGeometry(call, parent)) {
    return StaticAccSliceViewSpanBytes(*geometry, parent, requested_shape, view->dtype_);
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
// Compiler-owned PTO level3 scratch
// ============================================================================

ExprPtr MakeStaticShape(const std::vector<int64_t>& dims, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(dims.size());
  for (int64_t dim : dims) {
    elements.push_back(std::make_shared<ConstInt>(dim, DataType::INDEX, span));
  }
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

// A private tile.create used only as scratch can keep its allocation. Other
// uses (including view construction and loop initializers) may require a
// differently typed native window parent over the same storage.
class PrivateScratchAnalysis : public IRVisitor {
 public:
  explicit PrivateScratchAnalysis(const StmtPtr& body) { VisitStmt(body); }

  [[nodiscard]] bool IsPrivate(const ExprPtr& expression) const {
    const auto var = AsVarLike(expression);
    const auto tile = var ? As<TileType>(var->GetType()) : nullptr;
    return var && created_.count(var.get()) && !other_uses_.count(var.get()) && tile &&
           (!tile->memref_ || !(*tile->memref_)->is_pinned_);
  }

 protected:
  void VisitStmt_(const AssignStmtPtr& statement) override {
    if (IsOp(As<Call>(statement->value_), "tile.create")) created_.insert(statement->var_.get());
    VisitExpr(statement->value_);
  }
  void VisitExpr_(const CallPtr& call) override {
    for (size_t i = 0; i < call->args_.size(); ++i) {
      if (i != 2 || !IsOp(call, "tile.gather_compare")) VisitExpr(call->args_[i]);
    }
  }
  void VisitVarLike_(const VarPtr& var) override { other_uses_.insert(var.get()); }
  void VisitExpr_(const IterArgPtr& argument) override { other_uses_.insert(argument.get()); }
  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }

 private:
  std::set<const Var*> created_;
  std::set<const Var*> other_uses_;
};

struct PtoScratchSpec {
  ExprPtr shape;
  DataType dtype;
  std::string name_component;
  std::optional<size_t> replace_arg = std::nullopt;
};

/// Materialize scratch before MemRef collection.
///
/// In addressless A2/A3 code, aliased gather scratch gets a private allocation.
/// Workspace operands carry no caller-readable result (the registry contract),
/// so no copy is needed. This keeps native window parents valid in both Tile
/// and Buffer IR without moving allocations into either emitter.
///
/// Optional compiler-owned scratch (`tile.ci`, narrowing `tile.cast`, required
/// `tile.sort32`) is inserted only when absent. Caller-owned tmp operands on
/// `tile.sel` / `tile.sels` / `tile.prelu` are preserved when present; those
/// ops require an explicit tmp in the IR, so this pass never synthesizes one
/// for them. `tile.col_sum` / `tile.row_expand_add` likewise keep caller tmp.
class MaterializePtoScratchMutator : public IRMutator {
 public:
  explicit MaterializePtoScratchMutator(const backend::Backend* target,
                                        const StmtPtr& addressless_body = nullptr)
      : backend_(target) {
    if (addressless_body) private_scratch_.emplace(addressless_body);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (!call) return IRMutator::VisitStmt_(op);

    auto spec = GetScratchSpec(call, op->span_);
    if (!spec.has_value()) return IRMutator::VisitStmt_(op);

    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"dtype", spec->dtype},
        {"target_memory", MemorySpace::Vec},
    };
    auto scratch_create = OpRegistry::GetInstance().Create("tile.create", {spec->shape}, kwargs, call->span_);
    const std::string scratch_name =
        auto_name::BuildName(auto_name::GetBaseName(op->var_->name_hint_), spec->name_component, "tmp",
                             static_cast<int>(scratch_counter_++));
    auto scratch_var = std::make_shared<Var>(scratch_name, scratch_create->GetType(), op->span_);
    auto scratch_assign = std::make_shared<AssignStmt>(scratch_var, scratch_create, op->span_);

    std::vector<ExprPtr> args = call->args_;
    if (spec->replace_arg) {
      args[*spec->replace_arg] = scratch_var;
    } else {
      args.push_back(scratch_var);
    }
    auto new_call = std::make_shared<Call>(call->op_, std::move(args), call->kwargs_, call->attrs_,
                                           call->GetType(), call->span_);
    auto new_assign = MutableCopy(op);
    new_assign->value_ = std::move(new_call);
    return SeqStmts::Flatten({std::move(scratch_assign), std::move(new_assign)}, op->span_);
  }

 private:
  std::optional<PtoScratchSpec> GetScratchSpec(const CallPtr& call, const Span& span) const {
    if (private_scratch_) {
      if (!IsOp(call, "tile.gather_compare") || call->args_.size() != 3 ||
          private_scratch_->IsPrivate(call->args_[2])) {
        return std::nullopt;
      }
      const auto workspace = backend::StaticLogicalBufferDescriptor(call->args_[2]->GetType());
      // Preserve caller validation: do not silently repair invalid shapes,
      // layouts, validity, or capacity while isolating an otherwise dense alias.
      if (!backend::IsAlignedDenseVecBuffer(workspace) || workspace->dtype_ != DataType::UINT8 ||
          workspace->shape_[0] != 1 || workspace->valid_shape_ != workspace->shape_) {
        return std::nullopt;
      }
      return PtoScratchSpec{MakeStaticShape(workspace->shape_, span), DataType::UINT8, "gather", 2};
    }
    const auto query = QueryImplicitVecScratch(call, *backend_);
    CHECK_SPAN(!std::holds_alternative<std::string>(query), span)
        << "InitMemRef: " << std::get<std::string>(query);
    if (const auto* spec = std::get_if<ImplicitVecScratch>(&query)) {
      return PtoScratchSpec{std::make_shared<MakeTuple>(spec->shape, span), spec->dtype, spec->name};
    }
    return std::nullopt;
  }

  const backend::Backend* backend_;
  std::optional<PrivateScratchAnalysis> private_scratch_;
  std::size_t scratch_counter_ = 0;
};

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
                    FunctionType func_type, ProgramPtr program)
      : declared_allocs_(declared_allocs),
        handler_(handler),
        func_type_(func_type),
        program_(std::move(program)) {}

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
    if (const auto tensor = AsTensorTypeLike(type); tensor && tensor->tensor_view_) {
      const auto& view = *tensor->tensor_view_;
      if (!view.stride.empty() && (view.layout == TensorLayout::ND || view.layout == TensorLayout::DN)) {
        static_size = StaticTensorViewSpanBytes(tensor, tensor->shape_);
      }
    }
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

    auto result = std::make_shared<IterArg>(iter_arg->name_hint_, new_type, new_init, iter_arg->span_);
    borrowed_.BindAlias(result, new_init);
    tuple_results_.Copy(result, new_init);
    return result;
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
          // A device tile with no space after pass 20 is a compiler bug:
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

  /// Validate a declaration against the storage an alias actually inherits.
  /// Type deduction and SSA can carry the source's pinned declaration onto an
  /// unannotated result (notably gather_row). Its presence alone is therefore
  /// not a request for another buffer. Resolve it to a byte address before
  /// comparing: base identity alone would also accept a different slot, while
  /// MemRef pointer equality would reject independently rebuilt equal indices.
  void CheckAliasBinding(const AssignStmtPtr& op, const MemRefPtr& inherited) const {
    auto requested = UserBoundMemRef(op->var_->GetType());
    if (!requested) return;
    CHECK_SPAN(inherited && CompareBaseAddress(*requested, inherited) == AddressRelation::kSame,
               op->var_->span_)
        << "Tile '" << op->var_->name_hint_
        << "' lands in its source tile's allocation, but its binding selects a different or "
           "unproven address. Bind the source tile instead.";
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
    if (!parent_memref_opt.has_value()) {
      CheckAliasBinding(op, nullptr);
      return nullptr;
    }
    const auto& parent_memref = *parent_memref_opt;

    // A nested slice can narrow a strided window's byte envelope back to its
    // dense size without removing the parent's row gaps. Preserve that fact
    // through every storage-sharing view instead of inferring it at codegen
    // from only the final MemRef size.
    bool non_dense_source = non_dense_view_memrefs_.count(parent_memref) != 0;
    if (auto parent_tile = As<TileType>(source->GetType())) {
      const auto dense_bytes = utils::StaticPhysicalAllocationBytes(parent_tile, MemorySpace::Vec, handler_);
      non_dense_source = non_dense_source || !dense_bytes || *dense_bytes != parent_memref->size_;
    }
    if (auto call = As<Call>(new_value); call && IsOp(call, "tile.reinterpret_view")) {
      auto source_tile = As<TileType>(source->GetType());
      auto result_tile = As<TileType>(op->var_->GetType());
      const bool fp4_alias = (source_tile && source_tile->dtype_.IsPackedFp4()) ||
                             (result_tile && result_tile->dtype_.IsPackedFp4());
      // A reverse packed-to-byte alias drops the same pitch/offset metadata as
      // byte-to-packed, so neither direction can hide a strided ancestor.
      CHECK_SPAN(!fp4_alias || (!non_dense_source && As<ConstInt>(parent_memref->byte_offset_)), call->span_)
          << "FP4E2M1X2 UB alias requires a contiguous byte-carrier tile with a static UB offset; "
             "unsupported slice provenance survives nested slices, aliases and reshapes. Slice the "
             "GM tensor and load the window instead";
    }

    // Compute additional byte offset from the view op (if applicable)
    ExprPtr additional_offset = MakeZeroByteOffset();
    if (auto call = As<Call>(new_value)) {
      additional_offset = ComputeViewByteOffset(call, source->GetType());
    }

    auto source_ms = ExtractMemorySpaceFromType(source->GetType());

    // Keep ordinary aliases/reshapes at the parent's physical range. A slice
    // narrows that range to the packed parent-layout envelope it can touch.
    // Crucially, this is a VIEW span, not a fresh allocation footprint: an Acc
    // slice beginning at row 16 of a 32-row INT32 allocation must end at the
    // root boundary, not acquire another 32 rows of L0C padding. The envelope
    // is measured in the same layout model as the byte offset just computed —
    // physical strides for rectangular GM slices, NZ boxes for eligible
    // accumulator windows, and row-major elements for packed slices.
    uint64_t view_size = parent_memref->size_;  // default: same size as parent
    if (auto call = As<Call>(new_value)) {
      auto parent_shaped = std::dynamic_pointer_cast<const ShapedType>(source->GetType());
      auto out_shaped = std::dynamic_pointer_cast<const ShapedType>(op->var_->GetType());
      if (IsWindowTensorView(call)) {
        auto tensor = AsTensorTypeLike(op->var_->GetType());
        view_size = StaticTensorViewSpanBytes(tensor, tensor->shape_).value_or(0);
      } else if (auto slice_span = StaticSliceViewSpanBytes(call, parent_shaped, out_shaped)) {
        view_size = *slice_span;
      }
    }

    // Full aliases share the exact MemRef; subviews retain the source's slots
    // while narrowing the window. Later reanchoring uses the same construction.
    MemRefPtr view_memref = MakeAliasingMemRef(parent_memref, additional_offset, view_size, op->span_);
    CheckAliasBinding(op, view_memref);
    if (non_dense_source) non_dense_view_memrefs_.insert(view_memref);

    std::optional<MemRefPtr> view_opt = view_memref;
    TypePtr new_type = CloneTypeWithMemRefAndRemapExprs(
        op->var_->GetType(), view_opt, [this](const ExprPtr& e) { return VisitExpr(e); }, source_ms);
    VarPtr new_var = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
    var_map_[op->var_] = new_var;
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

  // Rebuild an AssignStmt whose LHS tile var keeps its TileType + memory_space but
  // carries no MemRef. Used for tiles that own no general-pool buffer: cross-core
  // tpop results and their physical writebacks / zero-copy views (codegen lowers views to
  // pto.treshape over the source rather than a fresh alloc_tile).
  StmtPtr MakeMemRefLessAssign(const AssignStmtPtr& op, const ExprPtr& new_value, const Var* root) {
    // A FIFO slot has no statically provable allocation address.
    CheckAliasBinding(op, nullptr);
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
    borrowed_.Bind(new_var, root);
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

  // A projected alias keeps the operand's complete view, including borrowed FIFO storage.
  StmtPtr BindAliasedResult(const TupleResultStorage::Storage& storage, const AssignStmtPtr& op,
                            const ExprPtr& value) {
    CheckAliasBinding(op, GetTypeMemRef(storage.type).value_or(nullptr));
    auto var = std::make_shared<Var>(op->var_->name_hint_, storage.type, op->var_->span_);
    borrowed_.Bind(var, storage.root);
    var_map_[op->var_] = var;
    return std::make_shared<AssignStmt>(var, value, op->span_, op->leading_comments_);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // First visit the value (RHS)
    auto new_value = VisitExpr(op->value_);

    if (auto call = As<Call>(new_value)) {
      auto helper = ResolveVFCall(call, program_);
      auto aliases = helper ? VFReturnedArguments(helper, call) : TupleResultAliasOperands(call);
      if (helper || !aliases.empty()) {
        // Replace aliased result types with the actual operand views. Fresh fields
        // such as a transfer event retain the result type deduced by the operator.
        auto tuple = As<TupleType>(call->GetType());
        std::vector<TypePtr> types;
        types.reserve(aliases.size());
        for (size_t i = 0; i < aliases.size(); ++i) {
          types.push_back(aliases[i] ? aliases[i]->GetType() : tuple->types_[i]);
        }
        TypePtr type = types.size() == 1 ? types[0] : std::make_shared<TupleType>(types);
        auto value =
            std::make_shared<Call>(call->op_, call->args_, call->kwargs_, call->attrs_, type, call->span_);
        if (aliases.size() == 1) return BindAliasedResult({type, borrowed_.Find(aliases[0])}, op, value);
        auto var = std::make_shared<Var>(op->var_->name_hint_, type, op->var_->span_);
        var_map_[op->var_] = var;
        tuple_results_.Record(var, aliases, borrowed_);
        return std::make_shared<AssignStmt>(var, value, op->span_, op->leading_comments_);
      }
    }
    if (tuple_results_.Find(new_value)) {
      auto var = std::make_shared<Var>(op->var_->name_hint_, new_value->GetType(), op->var_->span_);
      var_map_[op->var_] = var;
      tuple_results_.Copy(var, new_value);
      return std::make_shared<AssignStmt>(var, new_value, op->span_, op->leading_comments_);
    }
    if (auto storage = tuple_results_.Project(new_value)) return BindAliasedResult(*storage, op, new_value);

    // FIFO storage is provided by the pipe. Follow physical aliases before
    // allocating a Tile, including writebacks with a fresh SSA result.
    if (As<TileType>(op->var_->GetType())) {
      if (auto root = borrowed_.Definition(op->var_, new_value)) {
        return MakeMemRefLessAssign(op, new_value, root);
      }
    }

    if (auto call = As<Call>(op->value_)) {
      LOG_DEBUG << "Processing AssignStmt for " << op->var_->name_hint_ << " with call to "
                << call->op_->name_;

      // Handle view operations: output should share MemRef with input tile.
      // A pure metadata view (slice/reshape/...) inherits its input's buffer.  A
      // permuting inherit-input op — tile.transpose — must NOT: pto.ttrans is
      // not in-place safe (the unaligned scalar path writes dst directly from
      // src), so the transpose output always gets a fresh buffer.  Both halves
      // of that statement are `IsBufferAliasingViewOp`, which is
      // `OutputMemoryInheritsInput() && IsInplaceSafe()` — it excludes transpose
      // without naming it, and will exclude any future inherit-input op
      // registered `not_inplace_safe()`.
      if (op_predicates::IsBufferAliasingViewOp(call->op_->name_) && !call->args_.empty()) {
        LOG_DEBUG << "Detected view operation: " << call->op_->name_;
        // Get the input tile (first argument) after mutation
        auto new_call = std::dynamic_pointer_cast<const Call>(new_value);
        if (new_call && !new_call->args_.empty()) {
          auto result = ShareMemRefFrom(new_call->args_[0], op, new_value);
          if (result) {
            LOG_DEBUG << "Sharing MemRef from input tile to " << op->var_->name_hint_;
            return result;
          }
          LOG_DEBUG << "Input tile has no MemRef yet";
        }
      }

      // An InCore scalar write returns the destination descriptor at the same
      // GM origin. Elsewhere, use explicit storage reuse: host collectives
      // acquire their physical storage later in LowerHostTensorCollectives.
      const auto reuse_arg_idx = IsInCoreType(func_type_) && IsOp(call, "tensor.write")
                                     ? ResultAliasedArgIndex(call)
                                     : op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
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

    // Plain tile and InCore GM tensor aliases share the source MemRef.
    // Without this, the alias gets a fresh MemRef that is never written to,
    // breaking IfStmt return_vars that inherit the alias's empty MemRef.
    if (auto value_var = AsVarLike(new_value)) {
      if (As<TileType>(op->var_->GetType()) ||
          (IsInCoreType(func_type_) && AsTensorTypeLike(op->var_->GetType()))) {
        auto result = ShareMemRefFrom(value_var, op, new_value);
        if (result) return result;
      }
    }

    // Default case: visit the variable normally
    auto new_var = GetNewVar(op->var_);
    return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
  }

  StmtPtr VisitStmt_(const SimtLaunchStmtPtr& op) override {
    auto launch = MutableCopy(op);
    for (auto& dim : launch->block_dims_) dim = VisitExpr(dim);
    for (auto& arg : launch->args_) arg = VisitExpr(arg);
    for (size_t i = 0; i < launch->results_.size(); ++i) {
      // MaterializeSimtOutputs has already selected (and initialized) the
      // writable Tile storage. GM results are reference updates. Neither kind
      // allocates another buffer at the result definition.
      const auto& argument = launch->args_[launch->result_param_indices_[i]];
      auto result =
          std::make_shared<Var>(op->results_[i]->name_hint_, argument->GetType(), op->results_[i]->span_);
      borrowed_.BindAlias(result, argument);
      var_map_[op->results_[i]] = result;
      launch->results_[i] = std::move(result);
    }
    return launch;
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
    tuple_results_.JoinLoop(new_return_vars, new_iter_args, new_body);

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

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    auto loop = As<WhileStmt>(result);
    if (!loop || loop->return_vars_.empty()) return result;
    tuple_results_.JoinLoop(loop->return_vars_, loop->iter_args_, loop->body_);
    // Match the ForStmt invariant: the result is the carried storage even if
    // the loop executes zero times. A fresh return MemRef would lose both
    // alias identity and initialization established before the loop.
    auto get_iter_arg_var = [&](size_t i) -> VarPtr {
      return i < loop->iter_args_.size() ? loop->iter_args_[i] : nullptr;
    };
    auto [patched, changed] =
        PatchReturnVarsFromYield(loop->return_vars_, op->return_vars_, get_iter_arg_var);
    if (!changed) return result;
    auto patched_loop = MutableCopy(loop);
    patched_loop->return_vars_ = std::move(patched);
    return patched_loop;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto result = IRMutator::VisitStmt_(op);
    auto new_if = As<IfStmt>(result);
    if (!new_if || new_if->return_vars_.empty()) return result;

    auto then_yield = FindYieldStmt(new_if->then_body_);
    auto else_yield = new_if->else_body_.has_value() ? FindYieldStmt(new_if->else_body_.value()) : nullptr;
    tuple_results_.JoinBranches(new_if, then_yield, else_yield);
    if (!then_yield && !else_yield) return result;

    auto get_yield_var = [&](size_t i) -> VarPtr {
      auto then_var =
          then_yield && i < then_yield->value_.size() ? AsVarLike(then_yield->value_[i]) : nullptr;
      auto else_var =
          else_yield && i < else_yield->value_.size() ? AsVarLike(else_yield->value_[i]) : nullptr;
      // A GM phi aliases its source descriptor only when both paths yield
      // that storage. PTO checks the corresponding descriptor SSA at emission.
      if (AsTensorTypeLike(new_if->return_vars_[i]->GetType()) &&
          (!then_var || !else_var ||
           GetTypeMemRef(then_var->GetType()) != GetTypeMemRef(else_var->GetType()))) {
        return nullptr;
      }
      auto var = then_var ? then_var : else_var;
      if (auto tile = var ? As<TileType>(var->GetType()) : nullptr; tile && !tile->memref_) {
        // Only a single borrowed slot can remain an alias after the join.
        // Distinct slots need the ordinary allocated phi and branch copies:
        // lifetime analysis cannot tie the selected handle to either slot.
        auto root = borrowed_.Find(then_var);
        if (!root || root != borrowed_.Find(else_var)) return nullptr;
      }
      return var;
    };
    auto [patched, changed] = PatchReturnVarsFromYield(new_if->return_vars_, op->return_vars_, get_yield_var);

    if (!changed) return result;

    auto patched_if = MutableCopy(new_if);
    patched_if->return_vars_ = std::move(patched);
    return patched_if;
  }

 private:
  // Share return storage with loop carries (For/While) or branch yield values (If).
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
      auto source = yield_var ? std::dynamic_pointer_cast<const ShapedType>(yield_var->GetType()) : nullptr;
      const auto& result_type = new_return_vars[i]->GetType();
      const bool share =
          (As<TileType>(result_type) && yield_var && As<TileType>(yield_var->GetType())) ||
          (IsInCoreType(func_type_) && AsTensorTypeLike(result_type) && source && source->memref_);
      if (share) {
        auto new_type = CloneTypeWithMemRef(result_type, source->memref_, source->GetMemorySpace());
        auto new_rv =
            std::make_shared<Var>(new_return_vars[i]->name_hint_, new_type, new_return_vars[i]->span_);
        borrowed_.BindAlias(new_rv, yield_var);
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
  BorrowedTileRoots borrowed_;
  std::set<MemRefPtr> non_dense_view_memrefs_;
  const DeclaredAllocMap& declared_allocs_;
  const backend::BackendHandler* handler_ = nullptr;
  FunctionType func_type_ = FunctionType::InCore;
  ProgramPtr program_;
  TupleResultStorage tuple_results_;
  uint64_t next_id_ = 0;
};

// Insert alloc statements at the beginning of a function body.
/**
 * @brief Initialize MemRef for all variables in a function
 *
 * This transformation:
 * 1. Simplifies scalar constants and normalizes statement structure
 * 2. Materializes compiler-owned PTO scratch required by level3
 * 3. Initializes the MemRef field for all Var nodes
 * 4. Creates tile.alloc operations for non-DDR MemRefs (addr=-1, unallocated)
 *
 * Memory space is read from TileType::memory_space_ (set by InferTileMemorySpace).
 * Variables without memory_space default to DDR.
 */
FunctionPtr TransformInitMemRef(const FunctionPtr& func, const ProgramPtr& program) {
  // Helper storage is borrowed from each launch caller; allocating here would
  // give the helper an unrelated UB/GM buffer instead of its supplied operand.
  if (func->func_type_ == FunctionType::SIMT || func->func_type_ == FunctionType::VF) return func;
  // Lowering may introduce constant SSA aliases after the early Simplify pass.
  // Normalize them with the shared transformation before building addresses,
  // including when InitMemRef is invoked outside the default pipeline.
  auto normalized_func =
      NormalizeStmtStructure(transform_utils::SimplifyFunction(func, /*constants_only=*/true));

  const auto* ctx = PassContext::Current();
  const backend::Backend* target = nullptr;
  if (backend::BackendConfig::IsConfigured()) {
    target = ctx ? ctx->GetBackend() : backend::GetBackend();
  }
  const auto* handler = target ? target->GetHandler() : nullptr;

  // PTOAS level2 owns implicit-tmp materialization as part of PlanMemory. PyPTO
  // and DSA-RP instead emit fixed addresses and invoke level3. CI/TCVT scratch
  // requirements come from the backend, while TSORT32 scratch is level-driven.
  const auto planner = GetFunctionMemoryPlanner(func);
  if (handler != nullptr && (planner == MemoryPlanner::PyPTO || planner == MemoryPlanner::DsaRP)) {
    MaterializePtoScratchMutator materializer(target);
    normalized_func = materializer.VisitFunction(normalized_func);
  } else if (handler != nullptr && planner == MemoryPlanner::PtoAS && handler->RequiresLevel3TmpScratch()) {
    MaterializePtoScratchMutator materializer(target, normalized_func->body_);
    normalized_func = materializer.VisitFunction(normalized_func);
  }

  // Step 3: Resolve author-declared allocations (`pl.Tile[..., pl.MemRef("name"),
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
  if (planner == MemoryPlanner::PtoAS) {
    for (const auto& [base, alloc] : declared_allocs) {
      CHECK(alloc.slot_count > 1)
          << "A single-slot declared allocation (pl.MemRef(\"" << base->name_hint_
          << "\")) is not supported under memory_planner=PTOAS: ptoas owns memory planning and "
             "would be free to coalesce the allocations you separated. Declare it with "
             "pl.MemRef(slots=N) — N slots become one ptoas multi-buffer region whose slots ptoas "
             "keeps disjoint — or compile with the default PyPTO memory planner.";
    }
  }

  InitMemRefMutator mutator(declared_allocs, handler, normalized_func->func_type_, program);

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
  // The custom function rebuild bypasses IRMutator::VisitFunction. Keep
  // parameter-valued attrs (including external return contracts) attached to
  // the fresh Vars whose tensor types now carry memory references.
  for (auto& [key, value] : result_func->attrs_) {
    if (auto remapped =
            MapAttrExprs(value, [&mutator](const ExprPtr& expr) { return mutator.VisitExpr(expr); })) {
      value = std::move(*remapped);
    }
  }
  VerifyRegMemoryInitialization(result_func, program);

  // Step 4: Collect ALL MemRefs (DDR gets tensor.alloc, on-chip gets tile.alloc)
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

  // Step 5: Insert alloc statements at the beginning of the function body
  auto final_body = InsertAllocsIntoBody(new_body, alloc_stmts);

  result_func->body_ = final_body;
  return result_func;
}

}  // namespace

// Factory function
namespace pass {
Pass InitMemRef() {
  return CreateProgramPass(
      [](const ProgramPtr& program) {
        return_lineage::ScopedTraceCache lineage_cache;
        auto result = MutableCopy(program);
        for (auto& [_, function] : result->functions_) function = TransformInitMemRef(function, program);
        return ProgramPtr(result);
      },
      "InitMemRef", kInitMemRefProperties);
}
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
  HasMemRefsVerifier(std::vector<Diagnostic>& diagnostics, ProgramPtr program)
      : diagnostics_(diagnostics), program_(std::move(program)) {}

  void VisitExpr_(const IterArgPtr& op) override {
    IRVisitor::VisitExpr_(op);
    TrackBufferLessAlias(op, op->initValue_);
    tuple_results_.Copy(op, op->initValue_);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    TrackLoopResults(op->return_vars_, op->iter_args_);
    tuple_results_.JoinLoop(op->return_vars_, op->iter_args_, op->body_);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    TrackLoopResults(op->return_vars_, op->iter_args_);
    tuple_results_.JoinLoop(op->return_vars_, op->iter_args_, op->body_);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    IRVisitor::VisitStmt_(op);
    auto then_yield = FindYieldStmt(op->then_body_);
    auto else_yield = op->else_body_ ? FindYieldStmt(*op->else_body_) : nullptr;
    if (!then_yield || !else_yield) return;
    tuple_results_.JoinBranches(op, then_yield, else_yield);
    for (size_t i = 0;
         i < op->return_vars_.size() && i < then_yield->value_.size() && i < else_yield->value_.size(); ++i) {
      if (borrowed_.Find(then_yield->value_[i]) == borrowed_.Find(else_yield->value_[i])) {
        TrackBufferLessAlias(op->return_vars_[i], then_yield->value_[i]);
      }
    }
  }

  void VisitStmt_(const SimtLaunchStmtPtr& op) override {
    for (const auto& result : op->results_) {
      if (As<TileType>(result->GetType()) && !GetTypeMemRef(result->GetType())) {
        diagnostics_.emplace_back(DiagnosticSeverity::Error, "HasMemRefs", 0,
                                  "SIMT Tile result has no initialized output storage", result->span_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op || !op->var_ || !op->var_->GetType()) return;
    if (auto call = As<Call>(op->value_)) {
      if (auto helper = ResolveVFCall(call, program_)) {
        auto aliases = VFReturnedArguments(helper, call);
        if (aliases.size() == 1) {
          TrackBufferLessAlias(op->var_, aliases[0]);
        } else {
          tuple_results_.Record(op->var_, aliases, borrowed_);
        }
      }
    }
    tuple_results_.Copy(op->var_, op->value_);
    if (auto storage = tuple_results_.Project(op->value_)) borrowed_.Bind(op->var_, storage->root);
    auto tile_type = std::dynamic_pointer_cast<const TileType>(op->var_->GetType());
    if (tile_type && !tile_type->memref_.has_value()) {
      borrowed_.Bind(op->var_, borrowed_.Definition(op->var_, op->value_));
      if (!borrowed_.Find(op->var_)) {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "HasMemRefs", 0,
            "TileType variable '" + op->var_->name_hint_ + "' has no MemRef initialized", op->var_->span_);
      }
    }
    IRVisitor::VisitStmt_(op);
  }

 private:
  void TrackBufferLessAlias(const VarPtr& var, const ExprPtr& source) {
    auto tile = As<TileType>(var->GetType());
    if (tile && !tile->memref_) borrowed_.BindAlias(var, source);
  }

  void TrackLoopResults(const std::vector<VarPtr>& results, const std::vector<IterArgPtr>& iter_args) {
    for (size_t i = 0; i < results.size() && i < iter_args.size(); ++i) {
      TrackBufferLessAlias(results[i], iter_args[i]);
    }
  }

  BorrowedTileRoots borrowed_;
  std::vector<Diagnostic>& diagnostics_;
  ProgramPtr program_;
  TupleResultStorage tuple_results_;
};

}  // namespace

class HasMemRefsPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "HasMemRefs"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    return_lineage::ScopedTraceCache lineage_cache;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_ || func->func_type_ == FunctionType::VF) continue;
      HasMemRefsVerifier verifier(diagnostics, program);
      verifier.VisitStmt(func->body_);
    }
  }
};

PropertyVerifierPtr CreateHasMemRefsPropertyVerifier() {
  return std::make_shared<HasMemRefsPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
