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
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/core/logging.h"
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
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/property_verifier_registry.h"

namespace pypto {
namespace ir {
namespace {

std::vector<int64_t> StaticExtents(const std::vector<ExprPtr>& extents, const Span& span) {
  std::vector<int64_t> result;
  result.reserve(extents.size());
  for (const auto& extent : extents) {
    auto value = As<ConstInt>(extent);
    CHECK_SPAN(value, span) << "LowerTileToBuffer: this recipe requires static physical and valid extents";
    result.push_back(value->value_);
  }
  return result;
}

BufferTypePtr DenseDescriptor(const TileTypePtr& tile, const Span& span) {
  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  CHECK_SPAN(tile->shape_.size() == 2 && backend::IsDenseBufferTransferDtype(tile->dtype_) &&
                 tile->GetMemorySpace() == MemorySpace::Vec && view.blayout == TileLayout::row_major &&
                 view.slayout == TileLayout::none_box && view.fractal == 512 && view.pad == PadValue::null &&
                 view.compact == CompactMode::null,
             span)
      << "LowerTileToBuffer: this recipe requires dense rank-2 Vec FP16/BF16/FP32/INT32 tiles";
  const auto view_offset = As<ConstInt>(view.start_offset);
  CHECK_SPAN(!view.start_offset || (view_offset && view_offset->value_ == 0), span)
      << "LowerTileToBuffer: nonzero tile view offsets require an explicit Buffer view recipe";
  const auto shape = StaticExtents(tile->shape_, span);
  const auto valid = StaticExtents(view.valid_shape.empty() ? tile->shape_ : view.valid_shape, span);
  if (!view.stride.empty()) {
    const auto strides = StaticExtents(view.stride, span);
    CHECK_SPAN(strides == std::vector<int64_t>({shape[1], 1}), span)
        << "LowerTileToBuffer: strided tiles require an explicit Buffer view recipe";
  }
  return std::make_shared<BufferType>(shape, tile->dtype_, MemorySpace::Vec, valid);
}

// A cube-space tile keeps its resolved fractal layout. Its physical extents
// are already whole boxes, so the descriptor is exact without a byte view.
BufferTypePtr MatrixDescriptor(const TileTypePtr& tile, MemorySpace space, const Span& span) {
  const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
  CHECK_SPAN(tile->shape_.size() == 2 && backend::IsMatrixBufferDtype(space, tile->dtype_) &&
                 view.pad == PadValue::null && view.stride.empty(),
             span)
      << "LowerTileToBuffer: " << MemorySpaceToString(space)
      << " recipes require unpadded rank-2 FP16/BF16/FP32/INT8 cube operands or FP32/INT32 accumulators, got "
      << tile->dtype_.ToString() << " rank " << tile->shape_.size();
  const auto view_offset = As<ConstInt>(view.start_offset);
  CHECK_SPAN(!view.start_offset || (view_offset && view_offset->value_ == 0), span)
      << "LowerTileToBuffer: nonzero tile view offsets require an explicit Buffer view recipe";
  const auto shape = StaticExtents(tile->shape_, span);
  const auto valid = StaticExtents(view.valid_shape.empty() ? tile->shape_ : view.valid_shape, span);
  return std::make_shared<BufferType>(shape, tile->dtype_, space, valid, view.blayout, view.slayout,
                                      view.fractal, view.pad, view.compact);
}

BufferTypePtr StorageDescriptor(const TileTypePtr& tile, const Span& span) {
  const auto space = tile->GetMemorySpace();
  if (space && backend::IsMatrixBufferSpace(*space)) return MatrixDescriptor(tile, *space, span);
  return DenseDescriptor(tile, span);
}

/// Planned members retain their descriptors; storage is represented once.
struct StorageMember {
  TileTypePtr tile;  // Null for a write view that no Tile variable names.
  MemRefPtr memory;
  BufferTypePtr descriptor;
  Span span;
  const Call* write_view = nullptr;  // The call writing through that view.
};

struct BufferStorage {
  VarPtr handle;
  std::vector<StorageMember> members;
  std::vector<StmtPtr> definitions;
};

class StorageIndex : public IRVisitor {
 public:
  explicit StorageIndex(bool addressed) : addressed_(addressed) {}

  void VisitExpr(const ExprPtr& expr) override {
    if (!expr) return;
    // Allocation passes attach planned storage to SSA variables. A producer
    // Call retains its logical deduced type and is not an allocation identity.
    auto variable = AsVarLike(expr);
    if (auto tile = variable ? As<TileType>(variable->GetType()) : nullptr;
        tile && types_.insert(tile.get()).second) {
      auto memory = GetDefinedMemRef(tile);
      CHECK_SPAN(memory && memory->base_ && !memory->is_pinned_ && memory->slot_count_ == 1 &&
                     !memory->slot_index_.has_value(),
                 expr->span_)
          << "LowerTileToBuffer: tile storage must be planned; multi-slot storage needs its own recipe";
      auto offset = As<ConstInt>(memory->byte_offset_);
      CHECK_SPAN(offset && offset->value_ >= 0, expr->span_)
          << "LowerTileToBuffer: expected a static nonnegative storage window address";
      roots[memory->base_.get()].members.push_back(
          {tile, memory, StorageDescriptor(tile, expr->span_), expr->span_});
    }
    IRVisitor::VisitExpr(expr);
  }

  // Each member participates in a fixed number of indexed scans, O(N log N).
  // The resulting aliases are ordinary SSA definitions, not an IR side table.
  void Finalize() {
    for (auto& [base, storage] : roots) FinalizeRoot(base, storage);
  }

  std::unordered_map<const Var*, BufferStorage> roots;
  std::unordered_map<const TileType*, VarPtr> handles;
  std::unordered_map<const Call*, VarPtr> write_views;

 protected:
  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (auto call = As<Call>(assign->value_); call && IsOp(call, "tile.alloc")) {
      INTERNAL_CHECK_SPAN(declarations_.emplace(assign->var_.get(), call).second, assign->span_)
          << "Internal error: duplicate planned allocation definition";
    }
    if (auto call = As<Call>(assign->value_); call && IsOp(call, "tile.matmul_acc")) {
      IndexProductView(call, assign->var_);
    }
    IRVisitor::VisitStmt_(assign);
  }

  // tile.matmul_acc may accumulate into a wider valid rectangle than its
  // product. Both native forms write only the product rectangle, and PTOAS
  // requires a static pto.tmatmul[.acc] destination to equal it, so such an
  // accumulator gets a same-storage view with the product's valid extents,
  // matching the legacy emitter's native write.
  void IndexProductView(const CallPtr& call, const VarPtr& result) {
    const auto accumulator = As<TileType>(result->GetType());
    const auto lhs = As<TileType>(call->args_[1]->GetType());
    const auto rhs = As<TileType>(call->args_[2]->GetType());
    INTERNAL_CHECK_SPAN(accumulator && lhs && rhs, call->span_)
        << "Internal error: tile.matmul_acc requires Tile accumulator and operands";
    const auto descriptor = StorageDescriptor(accumulator, call->span_);
    const std::vector<int64_t> product{StorageDescriptor(lhs, call->span_)->valid_shape_[0],
                                       StorageDescriptor(rhs, call->span_)->valid_shape_[1]};
    if (product == descriptor->valid_shape_) return;
    auto view = std::make_shared<BufferType>(
        descriptor->shape_, descriptor->dtype_, descriptor->memory_space_, product, descriptor->blayout_,
        descriptor->slayout_, descriptor->fractal_, descriptor->pad_, descriptor->compact_);
    auto memory = GetDefinedMemRef(accumulator);
    INTERNAL_CHECK_SPAN(memory && memory->base_, call->span_)
        << "Internal error: tile.matmul_acc accumulator storage must be planned";
    roots[memory->base_.get()].members.push_back({nullptr, memory, view, call->span_, call.get()});
  }

  // Initializers are visited once at their lexical binding. Body references
  // must not expand the initializer chains of enclosing loops.
  void VisitExpr_(const IterArgPtr& argument) override { VisitVarLike_(argument); }
  void VisitStmt_(const ForStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }
  void VisitStmt_(const WhileStmtPtr& loop) override {
    for (const auto& argument : loop->iter_args_) VisitExpr(argument->initValue_);
    IRVisitor::VisitStmt_(loop);
  }

 private:
  using ViewKey = std::tuple<uint64_t, std::string, std::vector<int64_t>, std::vector<int64_t>>;

  static VarPtr Define(BufferStorage& storage, const std::string& suffix, const std::string& op,
                       const std::vector<ExprPtr>& args, const BufferTypePtr& descriptor, const Span& span) {
    auto handle = std::make_shared<Var>(storage.handle->name_hint_ + suffix, descriptor, span);
    auto call = OpRegistry::GetInstance().CreateInternal(op, args, {}, descriptor, span);
    storage.definitions.push_back(std::make_shared<AssignStmt>(handle, call, span));
    return handle;
  }

  void FinalizeRoot(const Var* base, BufferStorage& storage) {
    auto declaration = declarations_.find(base);
    CHECK_SPAN(declaration != declarations_.end(), base->span_)
        << "LowerTileToBuffer: storage root requires a planned tile.alloc capacity";
    const auto& call = declaration->second;
    const auto& span =
        call->span_.is_valid() && !call->span_.filename_.empty() ? call->span_ : storage.members.front().span;
    INTERNAL_CHECK_SPAN(call->args_.size() == 2, span) << "Internal error: malformed planned tile.alloc";
    auto size = As<ConstInt>(call->args_[1]);
    auto space = As<ConstInt>(call->args_[0]);
    std::optional<MemorySpace> memory_space;
    for (const auto candidate :
         {MemorySpace::Vec, MemorySpace::Mat, MemorySpace::Left, MemorySpace::Right, MemorySpace::Acc}) {
      if (space && space->value_ == static_cast<int64_t>(candidate)) memory_space = candidate;
    }
    CHECK_SPAN(size && size->value_ > 0 && memory_space, span)
        << "LowerTileToBuffer: storage root requires a static positive Vec, Mat, Left, Right or Acc "
           "allocation capacity";
    for (const auto& member : storage.members) {
      INTERNAL_CHECK_SPAN(member.descriptor->memory_space_ == *memory_space, member.span)
          << "Internal error: a planned tile's memory space differs from its allocation";
    }
    const auto capacity = static_cast<uint64_t>(size->value_);
    // Address placement rebases each member. Only a full-capacity member can
    // establish the origin; guessing the minimum interior address loses bytes.
    std::optional<int64_t> origin = addressed_ ? std::nullopt : std::optional<int64_t>(0);
    for (const auto& member : storage.members) {
      if (member.memory->size_ != capacity) continue;
      const auto address = As<ConstInt>(member.memory->byte_offset_)->value_;
      CHECK_SPAN(!origin || *origin == address, span)
          << "LowerTileToBuffer: full-capacity storage members disagree on the allocation origin";
      origin = address;
    }
    if (backend::IsMatrixBufferSpace(*memory_space)) {
      FinalizeMatrixRoot(base, storage, capacity, origin, span);
      return;
    }
    CHECK_SPAN(origin, span) << "LowerTileToBuffer: addressed storage needs a full-capacity MemRef anchor; "
                                "interior windows alone do not determine the planned allocation origin";

    std::map<ViewKey, BufferTypePtr> descriptors;
    std::vector<ViewKey> member_keys;
    for (const auto& member : storage.members) {
      const auto address = As<ConstInt>(member.memory->byte_offset_)->value_;
      const auto bytes = backend::DenseBufferBytes(member.descriptor);
      CHECK_SPAN(address >= *origin && bytes, span)
          << "LowerTileToBuffer: dense storage window must begin within its allocation";
      const auto offset = static_cast<uint64_t>(address - *origin);
      CHECK_SPAN(offset <= capacity && *bytes <= capacity - offset, span)
          << "LowerTileToBuffer: descriptor window exceeds the planned allocation capacity";
      ViewKey key{offset, member.descriptor->dtype_.ToString(), member.descriptor->shape_,
                  member.descriptor->valid_shape_};
      descriptors.emplace(key, member.descriptor);
      member_keys.push_back(std::move(key));
    }
    const auto& first = *descriptors.begin();
    const bool direct = descriptors.size() == 1 && std::get<0>(first.first) == 0 &&
                        backend::DenseBufferBytes(first.second) == capacity;
    CHECK_SPAN(direct || capacity % 32 == 0, span)
        << "LowerTileToBuffer: byte storage views require allocation capacity aligned to 32 bytes";
    auto root_type =
        direct ? first.second
               : std::make_shared<BufferType>(std::vector<int64_t>{static_cast<int64_t>(capacity / 32), 32},
                                              DataType::UINT8, MemorySpace::Vec);
    storage.handle = std::make_shared<Var>(base->name_hint_ + "_buffer", root_type, span);
    std::vector<ExprPtr> args{std::make_shared<MakeTuple>(std::vector<ExprPtr>{}, span)};
    if (addressed_) args.push_back(std::make_shared<ConstInt>(*origin, DataType::INDEX, span));
    auto allocation = OpRegistry::GetInstance().CreateInternal("buffer.alloc", args, {}, root_type, span);
    storage.definitions.push_back(std::make_shared<AssignStmt>(storage.handle, allocation, span));

    std::map<ViewKey, VarPtr> views;
    std::map<std::pair<uint64_t, uint64_t>, VarPtr> byte_windows;
    for (const auto& [key, descriptor] : descriptors) {
      if (direct) {
        views.emplace(key, storage.handle);
        continue;
      }
      const auto offset = std::get<0>(key);
      const auto dense_bytes = backend::DenseBufferBytes(descriptor);
      INTERNAL_CHECK_SPAN(dense_bytes.has_value(), span)
          << "Internal error: storage view descriptor must have a dense byte extent";
      const auto bytes = *dense_bytes;
      CHECK_SPAN(offset % 32 == 0 && bytes % 32 == 0, span)
          << "LowerTileToBuffer: static byte views require offsets and windows aligned to 32 bytes";
      auto window = storage.handle;
      if (offset != 0 || bytes != capacity) {
        auto [entry, inserted] = byte_windows.emplace(std::make_pair(offset, bytes), nullptr);
        if (inserted) {
          auto type = std::make_shared<BufferType>(std::vector<int64_t>{static_cast<int64_t>(bytes / 32), 32},
                                                   DataType::UINT8, MemorySpace::Vec);
          std::vector<ExprPtr> indices{
              std::make_shared<ConstInt>(static_cast<int64_t>(offset / 32), DataType::INDEX, span),
              std::make_shared<ConstInt>(0, DataType::INDEX, span)};
          entry->second = Define(storage, "_bytes", "buffer.subview",
                                 {window, std::make_shared<MakeTuple>(indices, span)}, type, span);
        }
        window = entry->second;
      }
      views.emplace(key, Define(storage, "_view", "buffer.reshape", {window}, descriptor, span));
    }
    for (size_t i = 0; i < storage.members.size(); ++i) {
      INTERNAL_CHECK_SPAN(storage.members[i].tile, storage.members[i].span)
          << "Internal error: only matrix storage carries untyped write views";
      handles.emplace(storage.members[i].tile.get(), views.at(member_keys[i]));
    }
  }

  // Every descriptor field except the root's shared memory space.
  using MatrixKey = std::tuple<uint64_t, std::string, std::vector<int64_t>, std::vector<int64_t>, int, int,
                               uint64_t, int, int>;

  static MatrixKey MakeMatrixKey(uint64_t address, const BufferTypePtr& type) {
    return {address,
            type->dtype_.ToString(),
            type->shape_,
            type->valid_shape_,
            static_cast<int>(type->blayout_),
            static_cast<int>(type->slayout_),
            type->fractal_,
            static_cast<int>(type->pad_),
            static_cast<int>(type->compact_)};
  }

  // Fractal descriptors have no byte-view form. Addressed planners already
  // placed every window, so each distinct window is its own allocation at its
  // final effective address; windows sharing an address alias exactly as the
  // planner decided. Without addresses, one allocation can carry only one
  // window, relabelled in place by same-size reshapes (NZ <-> ZN).
  void FinalizeMatrixRoot(const Var* base, BufferStorage& storage, uint64_t capacity,
                          std::optional<int64_t> origin, const Span& span) {
    std::map<MatrixKey, VarPtr> handles_by_window;
    std::vector<std::pair<MatrixKey, BufferTypePtr>> member_keys;
    for (const auto& member : storage.members) {
      const auto address = As<ConstInt>(member.memory->byte_offset_)->value_;
      const auto bytes = backend::PhysicalBufferBytes(member.descriptor);
      INTERNAL_CHECK_SPAN(bytes.has_value(), member.span)
          << "Internal error: matrix storage descriptor must have a physical byte extent";
      CHECK_SPAN(*bytes <= capacity, member.span)
          << "LowerTileToBuffer: descriptor window exceeds the planned allocation capacity";
      if (origin) {
        CHECK_SPAN(address >= *origin && static_cast<uint64_t>(address - *origin) <= capacity - *bytes,
                   member.span)
            << "LowerTileToBuffer: descriptor window exceeds the planned allocation capacity";
      }
      member_keys.emplace_back(MakeMatrixKey(static_cast<uint64_t>(address), member.descriptor),
                               member.descriptor);
    }
    VarPtr root;
    for (size_t i = 0; i < storage.members.size(); ++i) {
      const auto& [key, descriptor] = member_keys[i];
      auto [entry, inserted] = handles_by_window.emplace(key, nullptr);
      if (inserted) {
        if (addressed_ || !root) {
          std::vector<ExprPtr> args{std::make_shared<MakeTuple>(std::vector<ExprPtr>{}, span)};
          if (addressed_) {
            args.push_back(
                std::make_shared<ConstInt>(static_cast<int64_t>(std::get<0>(key)), DataType::INDEX, span));
          }
          auto handle = std::make_shared<Var>(base->name_hint_ + "_buffer", descriptor, span);
          auto allocation =
              OpRegistry::GetInstance().CreateInternal("buffer.alloc", args, {}, descriptor, span);
          storage.definitions.push_back(std::make_shared<AssignStmt>(handle, allocation, span));
          entry->second = handle;
          if (!root) {
            root = handle;
            storage.handle = root;
          }
        } else {
          CHECK_SPAN(
              std::get<0>(key) == 0 && backend::PhysicalBufferBytes(descriptor) ==
                                           backend::PhysicalBufferBytes(As<BufferType>(root->GetType())),
              storage.members[i].span)
              << "LowerTileToBuffer: an addressless " << MemorySpaceToString(descriptor->memory_space_)
              << " allocation can hold only one window; distinct windows require separate allocations";
          entry->second = Define(storage, "_view", "buffer.reshape", {root}, descriptor, span);
        }
      }
      if (storage.members[i].tile) {
        handles.emplace(storage.members[i].tile.get(), entry->second);
      } else {
        write_views.emplace(storage.members[i].write_view, entry->second);
      }
    }
  }

  bool addressed_;
  std::unordered_set<const TileType*> types_;
  std::unordered_map<const Var*, CallPtr> declarations_;
};

class TileToBufferMutator : public IRMutator {
 public:
  explicit TileToBufferMutator(const StorageIndex& storage) : storage_(storage) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& var) override {
    if (As<TileType>(var->GetType())) return Handle(var);
    auto alias = tensor_aliases_.find(var.get());
    return alias == tensor_aliases_.end() ? IRMutator::VisitExpr_(var) : alias->second;
  }

  ExprPtr VisitExpr_(const IterArgPtr& var) override {
    if (As<TileType>(var->GetType())) return Handle(var);
    if (auto alias = tensor_aliases_.find(var.get()); alias != tensor_aliases_.end()) {
      return alias->second;
    }
    auto found = var_remap_.find(var.get());
    INTERNAL_CHECK_SPAN(found != var_remap_.end(), var->span_)
        << "Internal error: Buffer conversion encountered an unbound scalar carry";
    return found->second;
  }

  ExprPtr VisitExpr_(const CallPtr& call) override {
    INTERNAL_CHECK_SPAN(call->op_, call->span_) << "Internal error: device call has no operator";
    CHECK_SPAN(false, call->span_) << "LowerTileToBuffer: no scalar or nested-call recipe for '"
                                   << call->op_->name_ << "'";
    return call;
  }

  ExprPtr VisitExpr_(const SubmitPtr& submit) override {
    CHECK_SPAN(false, submit->span_) << "LowerTileToBuffer: a device function cannot submit tasks";
    return submit;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& assign) override {
    if (auto call = As<Call>(assign->value_)) return LowerCall(call, assign->var_);
    if (As<TileType>(assign->var_->GetType())) {
      auto source = AsVarLike(assign->value_);
      INTERNAL_CHECK_SPAN(source && Handle(source) == Handle(assign->var_), assign->span_)
          << "Internal error: Tile storage legalization left an implicit alias transfer";
      return Empty(assign->span_);
    }
    if (As<TensorType>(assign->var_->GetType())) {
      auto source = AsVarLike(VisitExpr(assign->value_));
      CHECK_SPAN(source, assign->span_)
          << "LowerTileToBuffer: GM assignments require a normalized tensor parameter alias";
      tensor_aliases_[assign->var_.get()] = source;
      return Empty(assign->span_);
    }
    return IRMutator::VisitStmt_(assign);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& eval) override {
    if (auto call = As<Call>(eval->expr_)) return LowerCall(call, nullptr);
    return IRMutator::VisitStmt_(eval);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& branch) override {
    // Distributed GM windows need a separate region-result and device ABI recipe.
    for (const auto& result : branch->return_vars_) {
      CHECK_SPAN(!As<DistributedTensorType>(result->GetType()), result->span_)
          << "LowerTileToBuffer: distributed tensor branch results require a separate conversion recipe";
    }
    auto condition = VisitExpr(branch->condition_);
    YieldContext then_context(branch->return_vars_);
    auto then_body = LowerRegion(branch->then_body_, then_context);
    YieldContext else_context(branch->return_vars_);
    std::optional<StmtPtr> else_body;
    if (branch->else_body_) else_body = LowerRegion(*branch->else_body_, else_context);
    std::vector<VarPtr> results;
    for (size_t i = 0; i < branch->return_vars_.size(); ++i) {
      const auto& result = branch->return_vars_[i];
      if (As<TileType>(result->GetType())) continue;
      if (As<TensorType>(result->GetType())) {
        CHECK_SPAN(
            then_context.tensor_values[i] && then_context.tensor_values[i] == else_context.tensor_values[i],
            branch->span_)
            << "LowerTileToBuffer: GM branch results must alias the same parameter in both arms";
        tensor_aliases_[result.get()] = then_context.tensor_values[i];
      } else {
        results.push_back(result);
      }
    }
    return std::make_shared<IfStmt>(condition, then_body, else_body, results, branch->span_,
                                    branch->leading_comments_);
  }

  StmtPtr VisitStmt_(const YieldStmtPtr& yield) override {
    if (!yield_context_) return IRMutator::VisitStmt_(yield);
    INTERNAL_CHECK_SPAN(yield->value_.size() == yield_context_->results.size(), yield->span_)
        << "Internal error: device region yield/result arity mismatch";
    std::vector<ExprPtr> values;
    for (size_t i = 0; i < yield->value_.size(); ++i) {
      const auto& result = yield_context_->results[i];
      const auto& value = yield->value_[i];
      if (As<TileType>(result->GetType())) {
        INTERNAL_CHECK_SPAN(Handle(value) == Handle(result), yield->span_)
            << "Internal error: Tile storage legalization left an implicit region transfer";
      } else if (As<TensorType>(result->GetType())) {
        yield_context_->tensor_values[i] = VisitExpr(value);
      } else {
        values.push_back(VisitExpr(value));
      }
    }
    return std::make_shared<YieldStmt>(values, yield->span_, yield->leading_comments_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& loop) override { return LowerLoop(loop); }
  StmtPtr VisitStmt_(const WhileStmtPtr& loop) override { return LowerLoop(loop); }

 private:
  struct YieldContext {
    explicit YieldContext(const std::vector<VarPtr>& results)
        : results(results), tensor_values(results.size()) {}
    const std::vector<VarPtr>& results;
    std::vector<ExprPtr> tensor_values;
  };

  StmtPtr LowerRegion(const StmtPtr& body, YieldContext& context) {
    auto* outer = yield_context_;
    yield_context_ = &context;
    auto lowered = VisitStmt(body);
    yield_context_ = outer;
    return lowered;
  }

  template <typename LoopPtr>
  StmtPtr LowerLoop(const LoopPtr& loop) {
    INTERNAL_CHECK_SPAN(loop->iter_args_.size() == loop->return_vars_.size(), loop->span_)
        << "Internal error: device loop carry/result arity mismatch";
    auto lowered = std::make_shared<std::remove_const_t<typename LoopPtr::element_type>>(*loop);
    if constexpr (std::is_same_v<LoopPtr, ForStmtPtr>) {
      lowered->start_ = VisitExpr(loop->start_);
      lowered->stop_ = VisitExpr(loop->stop_);
      lowered->step_ = VisitExpr(loop->step_);
    }
    lowered->iter_args_.clear();
    lowered->return_vars_.clear();
    std::vector<ExprPtr> initial_values(loop->iter_args_.size());
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      const auto& result = loop->return_vars_[i];
      // Distributed GM windows need a separate region-result and device ABI recipe.
      CHECK_SPAN(
          !As<DistributedTensorType>(argument->GetType()) && !As<DistributedTensorType>(result->GetType()),
          loop->span_)
          << "LowerTileToBuffer: distributed tensor loop carries require a separate conversion recipe";
      auto initial = VisitExpr(argument->initValue_);
      initial_values[i] = initial;
      if (As<TileType>(argument->GetType())) {
        INTERNAL_CHECK_SPAN(initial == Handle(argument) && initial == Handle(result), loop->span_)
            << "Internal error: Tile storage legalization left an implicit loop entry transfer";
      } else if (As<TensorType>(argument->GetType())) {
        tensor_aliases_[argument.get()] = initial;
      } else {
        CHECK_SPAN(As<ScalarType>(argument->GetType()), argument->span_)
            << "LowerTileToBuffer: loop carries require scalar, Tile, or normalized GM values";
        auto scalar =
            std::make_shared<IterArg>(argument->name_hint_, argument->GetType(), initial, argument->span_);
        var_remap_[argument.get()] = scalar;
        lowered->iter_args_.push_back(std::move(scalar));
        lowered->return_vars_.push_back(result);
      }
    }
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) {
      lowered->condition_ = VisitExpr(loop->condition_);
    } else {
      lowered->attrs_ = MutateScopeAttrs(loop->attrs_).first;
    }
    YieldContext context(loop->return_vars_);
    lowered->body_ = LowerRegion(loop->body_, context);
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      if (As<TensorType>(argument->GetType())) {
        CHECK_SPAN(context.tensor_values[i] == initial_values[i], loop->span_)
            << "LowerTileToBuffer: GM loop results must retain their initial parameter alias";
        tensor_aliases_[loop->return_vars_[i].get()] = initial_values[i];
      }
      var_remap_.erase(argument.get());
    }
    return lowered;
  }

  static StmtPtr Empty(const Span& span) { return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span); }

  VarPtr Handle(const ExprPtr& value) const {
    auto tile = As<TileType>(value->GetType());
    INTERNAL_CHECK_SPAN(tile, value->span_) << "Internal error: Buffer conversion expected a Tile operand";
    auto found = storage_.handles.find(tile.get());
    INTERNAL_CHECK_SPAN(found != storage_.handles.end(), value->span_)
        << "Internal error: missing indexed storage view for a Tile operand";
    return found->second;
  }

  MakeTuplePtr Valid(const ExprPtr& tile) const {
    const auto type = As<BufferType>(Handle(tile)->GetType());
    std::vector<ExprPtr> extents;
    for (const auto extent : type->valid_shape_) {
      extents.push_back(std::make_shared<ConstInt>(extent, DataType::INDEX, tile->span_));
    }
    return std::make_shared<MakeTuple>(std::move(extents), tile->span_);
  }

  StmtPtr Operation(const std::string& name, const std::vector<ExprPtr>& args, const Span& span) const {
    return std::make_shared<EvalStmt>(OpRegistry::GetInstance().CreateInternal(name, args, span), span);
  }

  StmtPtr LowerCall(const CallPtr& call, const VarPtr& result) {
    INTERNAL_CHECK_SPAN(call->op_, call->span_) << "Internal error: device call has no operator";
    // Pipeline membership only constrains storage planning, whose result is
    // already fixed in the MemRefs (AllocateMemoryAddr strips it; PTOAS keeps
    // it). Every other attribute still needs an explicit conversion contract.
    CHECK_SPAN(StripAttr(call->attrs_, kPipelineMembershipAttr).empty(), call->span_)
        << "LowerTileToBuffer: device-call attributes require an explicit conversion contract";
    for (const auto& [logical, physical] : {std::pair{"tile.get_block_idx", "buffer.get_block_idx"},
                                            std::pair{"tile.get_block_num", "buffer.get_block_num"},
                                            std::pair{"tile.get_subblock_idx", "buffer.get_subblock_idx"}}) {
      if (!IsOp(call, logical)) continue;
      CHECK_SPAN(result && call->args_.empty() && call->kwargs_.empty(), call->span_)
          << "LowerTileToBuffer: SPMD queries require an SSA result and no operands or kwargs";
      auto query = OpRegistry::GetInstance().CreateInternal(physical, {}, {}, call->span_);
      return std::make_shared<AssignStmt>(result, query, call->span_);
    }
    if (IsOp(call, "tile.alloc")) {
      INTERNAL_CHECK_SPAN(result, call->span_) << "Internal error: allocation has no pointer definition";
      auto found = storage_.roots.find(result.get());
      if (found == storage_.roots.end()) return Empty(call->span_);
      return std::make_shared<SeqStmts>(found->second.definitions, call->span_);
    }
    if (IsOp(call, "tile.create")) {
      INTERNAL_CHECK_SPAN(result, call->span_) << "Internal error: tile.create has no result";
      (void)Handle(result);
      return Empty(call->span_);
    }
    if (IsOp(call, "tile.reshape")) {
      INTERNAL_CHECK_SPAN(result, call->span_) << "Internal error: tile.reshape has no result";
      const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
      ValidateKwargs(call->kwargs_, entry.GetOp()->GetAttrs(), call->op_->name_);
      const auto expected = As<TileType>(entry.GetDeduceType()(call->args_, call->kwargs_));
      const auto source = As<TileType>(call->args_[0]->GetType());
      const auto destination = As<TileType>(result->GetType());
      const auto source_memory = GetDefinedMemRef(source);
      const auto destination_memory = GetDefinedMemRef(destination);
      const auto descriptor = As<BufferType>(Handle(result)->GetType());
      INTERNAL_CHECK_SPAN(expected && expected->dtype_ == descriptor->dtype_ &&
                              StaticExtents(expected->shape_, call->span_) == descriptor->shape_ &&
                              StaticExtents(tile_view_semantics::GetEffectiveTileView(*expected).valid_shape,
                                            call->span_) == descriptor->valid_shape_ &&
                              source_memory->base_ == destination_memory->base_ &&
                              structural_equal(source_memory->byte_offset_, destination_memory->byte_offset_),
                          call->span_)
          << "Internal error: planned tile.reshape must preserve its validated storage window";
      // StorageIndex already materialized this typed alias at the allocation.
      // Logical reshape has no data transfer and needs no second alias handle.
      return Empty(call->span_);
    }
    if (IsOp(call, "tile.load")) {
      INTERNAL_CHECK_SPAN(result && call->args_.size() >= 3, call->span_)
          << "Internal error: malformed tile.load";
      CHECK_SPAN(GetIntKwarg(call->kwargs_, "cache", 0) == 0, call->span_)
          << "LowerTileToBuffer: cache-policy loads require a Buffer transfer recipe";
      return Operation("buffer.load",
                       {VisitExpr(call->args_[0]), VisitExpr(call->args_[1]), Valid(result), Handle(result)},
                       call->span_);
    }
    if (IsOp(call, "tile.store")) {
      INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
          << "Internal error: dense rank-2 tile.store requires three operands";
      CHECK_SPAN(
          GetIntKwarg(call->kwargs_, "atomic", 0) == 0 && GetIntKwarg(call->kwargs_, "st_phase", 0) == 0,
          call->span_)
          << "LowerTileToBuffer: atomic and phased stores require a Buffer transfer recipe";
      CHECK_SPAN(!GetOptionalDoubleKwarg(call->kwargs_, "pre_quant") &&
                     !GetKwargOr<bool>(call->kwargs_, "pre_relu", false),
                 call->span_)
          << "LowerTileToBuffer: fix-pipe pre_quant/pre_relu stores require a Buffer transfer recipe";
      auto output = VisitExpr(call->args_[2]);
      if (result) tensor_aliases_[result.get()] = output;
      return Operation("buffer.store",
                       {Handle(call->args_[0]), VisitExpr(call->args_[1]), Valid(call->args_[0]), output},
                       call->span_);
    }
    if (const auto* recipe = As<GlobalVar>(call->op_)
                                 ? nullptr
                                 : backend::FindLogicalBufferElementwiseRecipe(call->op_->name_)) {
      INTERNAL_CHECK_SPAN(result, call->span_) << "Internal error: Tile elementwise recipe has no result";
      const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
      INTERNAL_CHECK_SPAN(call->args_.size() == entry.GetArgumentCount(), call->span_)
          << "Internal error: malformed Tile elementwise recipe operand count";
      ValidateKwargs(call->kwargs_, entry.GetOp()->GetAttrs(), call->op_->name_);
      const auto destination = Handle(result);
      const auto dtype = As<BufferType>(destination->GetType())->dtype_;
      std::vector<ExprPtr> args;
      for (const auto& input : recipe->inputs) {
        INTERNAL_CHECK_SPAN(input.logical_index < call->args_.size(), call->span_)
            << "Internal error: malformed Tile elementwise recipe operands";
        const auto& source = call->args_[input.logical_index];
        if (input.kind == backend::BufferElementwiseOperandKind::Buffer) {
          args.push_back(Handle(source));
        } else {
          // Scalar representation belongs to the lowering contract. The emitter
          // consumes the resulting type and never repairs element operands.
          auto value = VisitExpr(source);
          const auto source_dtype = GetScalarDtype(value);
          CHECK_SPAN(source_dtype.IsSignedInt() || source_dtype == DataType::INDEX ||
                         source_dtype == DataType::FP16 || source_dtype == DataType::BF16 ||
                         source_dtype == DataType::FP32,
                     source->span_)
              << "LowerTileToBuffer: element scalar conversion requires a signed integer, INDEX, "
                 "FP16, BF16 or FP32 value";
          if (source_dtype == DataType::INDEX) value = MakeCast(value, DataType::INT64, source->span_);
          args.push_back(GetScalarDtype(value) == dtype ? value : MakeCast(value, dtype, source->span_));
        }
      }
      args.push_back(destination);
      // Shape and dtype of tile.full have already selected the physical
      // destination; they are not instruction attributes. Other registered
      // kwargs retain their typed schema and are checked during construction.
      const auto kwargs = IsOp(call, "tile.full") ? decltype(call->kwargs_){} : call->kwargs_;
      auto lowered = OpRegistry::GetInstance().CreateInternal(recipe->buffer_op, args, kwargs, call->span_);
      return std::make_shared<EvalStmt>(lowered, call->span_);
    }
    if (IsOp(call, "tile.transpose_view")) {
      // StorageIndex already declared the relabelled window at the source's
      // storage: an addressed alias allocation, or an addressless reshape.
      INTERNAL_CHECK_SPAN(result && call->args_.size() == 1, call->span_)
          << "Internal error: tile.transpose_view requires a source and a result";
      const auto source = GetDefinedMemRef(As<TileType>(call->args_[0]->GetType()));
      const auto destination = GetDefinedMemRef(As<TileType>(result->GetType()));
      INTERNAL_CHECK_SPAN(
          source->base_ == destination->base_ &&
              structural_equal(source->byte_offset_, destination->byte_offset_) &&
              backend::PhysicalBufferBytes(As<BufferType>(Handle(call->args_[0])->GetType())) ==
                  backend::PhysicalBufferBytes(As<BufferType>(Handle(result)->GetType())),
          call->span_)
          << "Internal error: planned tile.transpose_view must relabel its source storage window";
      return Empty(call->span_);
    }
    if (IsOp(call, "tile.extract")) {
      INTERNAL_CHECK_SPAN(result && call->args_.size() == 4, call->span_)
          << "Internal error: tile.extract requires source, offsets, shape and a result";
      const auto& entry = OpRegistry::GetInstance().GetEntry(call->op_->name_);
      ValidateKwargs(call->kwargs_, entry.GetOp()->GetAttrs(), call->op_->name_);
      // The static window shape already selected the destination descriptor.
      return Operation(
          "buffer.extract",
          {Handle(call->args_[0]), VisitExpr(call->args_[1]), VisitExpr(call->args_[2]), Handle(result)},
          call->span_);
    }
    if (IsOp(call, "tile.matmul")) {
      INTERNAL_CHECK_SPAN(result && call->args_.size() == 2, call->span_)
          << "Internal error: tile.matmul requires lhs, rhs and a result";
      return Operation("buffer.matmul", {Handle(call->args_[0]), Handle(call->args_[1]), Handle(result)},
                       call->span_);
    }
    if (IsOp(call, "tile.matmul_acc")) {
      INTERNAL_CHECK_SPAN(result && (call->args_.size() == 3 || call->args_.size() == 4), call->span_)
          << "Internal error: tile.matmul_acc requires acc, lhs, rhs, an optional init_cond and a result";
      const auto destination = Handle(result);
      // MaterializeSemanticAliases binds the reused accumulator to its result.
      INTERNAL_CHECK_SPAN(Handle(call->args_[0]) == destination, call->span_)
          << "Internal error: tile.matmul_acc must accumulate in place; its accumulator and result need the "
             "same planned storage";
      // A wider accumulator is written through its product-shaped view.
      const auto view = storage_.write_views.find(call.get());
      const std::vector<ExprPtr> operands{Handle(call->args_[1]), Handle(call->args_[2]),
                                          view == storage_.write_views.end() ? destination : view->second};
      if (call->args_.size() == 3) return Operation("buffer.matmul_acc", operands, call->span_);
      // init_cond overwrites instead of accumulating. A literal selects one
      // form; a runtime predicate selects between both explicit writes.
      const auto& init = call->args_[3];
      std::optional<bool> literal;
      if (auto value = As<ConstInt>(init)) literal = value->value_ != 0;
      if (auto value = As<ConstBool>(init)) literal = value->value_;
      if (literal) return Operation(*literal ? "buffer.matmul" : "buffer.matmul_acc", operands, call->span_);
      return std::make_shared<IfStmt>(
          VisitExpr(init), Operation("buffer.matmul", operands, call->span_),
          std::optional<StmtPtr>(Operation("buffer.matmul_acc", operands, call->span_)),
          std::vector<VarPtr>{}, call->span_);
    }
    if (IsOp(call, "tile.move")) {
      INTERNAL_CHECK_SPAN(result && !call->args_.empty(), call->span_)
          << "Internal error: tile.move requires a source and a result";
      auto source = Handle(call->args_[0]);
      auto target = Handle(result);
      return source == target ? Empty(call->span_) : Operation("buffer.copy", {source, target}, call->span_);
    }
    CHECK_SPAN(false, call->span_) << "LowerTileToBuffer: no conversion recipe for '" << call->op_->name_
                                   << "'";
    return Empty(call->span_);
  }

  const StorageIndex& storage_;
  YieldContext* yield_context_ = nullptr;
  // Input SSA uses distinct identities for branch-local definitions. Retaining
  // their mappings is linear and avoids copying the outer map at each region.
  std::unordered_map<const Var*, ExprPtr> tensor_aliases_;
};

ProgramPtr TransformProgram(const ProgramPtr& program) {
  const auto* context = PassContext::Current();
  const bool addressed = !context || context->GetMemoryPlanner() != MemoryPlanner::PtoAS;
  auto& verifiers = PropertyVerifierRegistry::GetInstance();
  verifiers.VerifyOrThrow({IRProperty::TileStorageLegalized}, program);
  if (addressed) verifiers.VerifyOrThrow({IRProperty::TileStorageAllocated}, program);
  std::vector<FunctionPtr> functions;
  functions.reserve(program->functions_.size());
  for (const auto& [global, function] : program->functions_) {
    if (!IsInCoreType(function->func_type_) || function->ir_stage_ == FunctionIRStage::Buffer) {
      functions.push_back(function);
      continue;
    }
    for (const auto& param : function->params_) {
      CHECK_SPAN(!As<TileType>(param->GetType()), param->span_)
          << "LowerTileToBuffer: device Tile parameters require the Buffer helper ABI recipe";
    }
    StorageIndex storage(addressed);
    storage.VisitStmt(function->body_);
    storage.Finalize();
    TileToBufferMutator mutator(storage);
    auto lowered = std::make_shared<Function>(*function);
    lowered->body_ = mutator.VisitStmt(function->body_);
    lowered->ir_stage_ = FunctionIRStage::Buffer;
    functions.push_back(std::move(lowered));
  }
  auto result = std::make_shared<Program>(functions, program->name_, program->span_);
  verifiers.VerifyOrThrow({IRProperty::BufferIR}, result);
  return result;
}

}  // namespace

namespace pass {

Pass LowerTileToBuffer() {
  return CreateProgramPass(TransformProgram, "LowerTileToBuffer", kLowerTileToBufferProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
