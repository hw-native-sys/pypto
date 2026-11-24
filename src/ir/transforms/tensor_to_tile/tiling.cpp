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
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/op_rules.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/ir/cast_saturation.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/storage_size.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/op_conversion_registry.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/acc_init_builder.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/l0c_footprint.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/tensor_to_tile/internal.h"
#include "src/ir/transforms/utils/tile_cast_legalization.h"

namespace pypto {
namespace ir {
namespace tensor_to_tile {
namespace {

struct Shape {
  int64_t rows;
  int64_t cols;
};

struct Window {
  Shape size;
  ExprPtr row;
  ExprPtr col;
  std::vector<ExprPtr> batch = {};
};

struct TensorRegion {
  std::vector<AssignStmtPtr> calculations;
  VarPtr result;
  VarPtr destination;
  VarPtr writeback;
};

using UseCounts = std::unordered_map<const Var*, size_t>;
using TileEmitter = std::function<StmtPtr(const Window&)>;
using FitsTile = std::function<bool(Shape)>;

class UseCounter : public IRVisitor {
 public:
  UseCounts counts;

 protected:
  void VisitVarLike_(const VarPtr& var) override { ++counts[var.get()]; }
  void VisitStmt_(const AssignStmtPtr& assign) override { VisitExpr(assign->value_); }
};

size_t Uses(const UseCounts& counts, const VarPtr& var) {
  auto it = counts.find(var.get());
  return it == counts.end() ? 0 : it->second;
}

bool ZeroOffset(const ExprPtr& expr, size_t rank) {
  const auto tuple = As<MakeTuple>(expr);
  if (!tuple || tuple->elements_.size() != rank) return false;
  return std::all_of(tuple->elements_.begin(), tuple->elements_.end(), [](const ExprPtr& value) {
    const auto offset = As<ConstInt>(value);
    return offset && offset->value_ == 0;
  });
}

VarPtr IndexVar(const std::string& name, const Span& span) {
  return std::make_shared<Var>(name, std::make_shared<ScalarType>(DataType::INDEX), span);
}

int64_t LargestFitting(int64_t extent, const std::function<bool(int64_t)>& fits) {
  int64_t low = 0;
  int64_t high = extent;
  while (low < high) {
    const int64_t candidate = low + (high - low) / 2 + 1;
    if (fits(candidate)) {
      low = candidate;
    } else {
      high = candidate - 1;
    }
  }
  return low;
}

using EmitBand = std::function<StmtPtr(const ExprPtr&, int64_t)>;

StmtPtr EmitAxis(int64_t extent, int64_t block, const std::string& name, const Span& span,
                 const EmitBand& emit) {
  if (extent == block) return emit(MakeIndexConst(0, span), block);
  const int64_t interior = (extent / block) * block;
  auto iv = IndexVar(name, span);
  auto loop = std::make_shared<ForStmt>(iv, MakeIndexConst(0, span), MakeIndexConst(interior, span),
                                        MakeIndexConst(block, span), std::vector<IterArgPtr>{},
                                        emit(iv, block), std::vector<VarPtr>{}, span);
  if (interior == extent) return loop;
  return SeqStmts::Flatten({loop, emit(MakeIndexConst(interior, span), extent - interior)}, span);
}

UseCounts CountUses(const StmtPtr& body) {
  UseCounter counter;
  counter.VisitStmt(body);
  return std::move(counter.counts);
}

std::optional<std::vector<int64_t>> StaticExtents(const TensorTypePtr& type) {
  if (!type || type->shape_.empty()) return std::nullopt;
  if (const auto& view = type->tensor_view_) {
    if (view->layout != TensorLayout::ND || !view->valid_shape.empty()) return std::nullopt;
    // The window planner emits native DMA. Sampled inner axes need a
    // different access path even when the Tensor has static ND extents.
    if (!view->stride.empty()) {
      if (view->stride.size() != type->shape_.size()) return std::nullopt;
      const auto& inner = view->stride.back();
      if (ProveValidExtentEqual(inner, MakeIndexConst(1, inner->span_)) != ProofResult::kTrue) {
        return std::nullopt;
      }
    }
  }
  std::vector<int64_t> result;
  for (const auto& dim : type->shape_) {
    auto extent = As<ConstInt>(dim);
    if (!extent || extent->value_ <= 0 || extent->value_ > std::numeric_limits<int64_t>::max() - 32) {
      return std::nullopt;
    }
    result.push_back(extent->value_);
  }
  return result;
}

Shape PlaneShape(const std::vector<int64_t>& shape) {
  return {shape.size() >= 2 ? shape[shape.size() - 2] : 1, shape.back()};
}

// Match native DMA's logical ND column heuristic before padding. A scalar
// and rank-ND windows stay row-major; a tail of a rank-2 column stays DN.
bool IsColumnPlane(const std::vector<int64_t>& extents) {
  const auto shape = PlaneShape(extents);
  return extents.size() == 2 && shape.rows > 1 && shape.cols == 1;
}

int64_t AlignUp(int64_t extent, int64_t alignment) {
  return extent + (alignment - extent % alignment) % alignment;
}

Shape VectorBox(Shape valid, bool column_vector, bool align_rows, int64_t alignment) {
  return {column_vector || align_rows ? AlignUp(valid.rows, alignment) : valid.rows,
          column_vector ? 1 : AlignUp(valid.cols, alignment)};
}

Window ProjectBatch(const Window& window, const std::vector<int64_t>& shape, const Span& span) {
  Window projected{window.size, window.row, window.col};
  const size_t axes = shape.size() > 2 ? shape.size() - 2 : 0;
  const size_t first = window.batch.size() - axes;
  for (size_t axis = 0; axis < axes; ++axis) {
    projected.batch.push_back(shape[axis] == 1 ? MakeIndexConst(0, span) : window.batch[first + axis]);
  }
  return projected;
}

std::optional<TensorRegion> MatchRegion(const InCoreScopeStmtPtr& scope, const UseCounts& uses) {
  if (!scope || scope->split_ != SplitMode::None || !scope->attrs_.empty()) return std::nullopt;
  const auto seq = As<SeqStmts>(scope->body_);
  const auto statements = seq ? seq->stmts_ : std::vector<StmtPtr>{scope->body_};
  if (statements.empty()) return std::nullopt;

  TensorRegion region;
  for (const auto& stmt : statements) {
    auto assign = As<AssignStmt>(stmt);
    if (!assign || !As<TensorType>(assign->var_->GetType()) || !As<Call>(assign->value_)) {
      return std::nullopt;
    }
    region.calculations.push_back(assign);
  }

  const auto& last = region.calculations.back();
  const auto call = As<Call>(last->value_);
  if (IsOp(call, "tensor.assemble")) {
    if (region.calculations.size() == 1 || call->args_.size() != 3 || !call->attrs_.empty() ||
        call->GetKwarg<int>("atomic", 0) != 0) {
      return std::nullopt;
    }
    auto destination = AsVarLike(call->args_[0]);
    auto dest_type = destination ? As<TensorType>(destination->GetType()) : nullptr;
    auto source = AsVarLike(call->args_[1]);
    auto src_type = source ? As<TensorType>(source->GetType()) : nullptr;
    auto extents = StaticExtents(dest_type);
    if (!extents || !src_type || extents != StaticExtents(src_type) ||
        !ZeroOffset(call->args_[2], extents->size()) ||
        source != region.calculations[region.calculations.size() - 2]->var_) {
      return std::nullopt;
    }
    region.destination = destination;
    region.writeback = last->var_;
    region.calculations.pop_back();
  }
  region.result = region.calculations.back()->var_;
  if (!region.destination && Uses(uses, region.result) == 0) return std::nullopt;

  // A replacement can export the product and its writeback. Any other SSA
  // definition required by a caller must remain available in its original form.
  auto internal = CountUses(scope->body_);
  for (const auto& calc : region.calculations) {
    // Deferred writeback runs outside this scope and can only name a captured destination.
    if (calc->var_ == region.destination) return std::nullopt;
    if (calc->var_ != region.result && Uses(uses, calc->var_) != Uses(internal, calc->var_)) {
      return std::nullopt;
    }
  }
  return region;
}

ExprPtr MakeShape(Shape shape, size_t rank, const Span& span) {
  std::vector<ExprPtr> dims(rank > 2 ? rank - 2 : 0, MakeIndexConst(1, span));
  if (rank >= 2) dims.push_back(MakeIndexConst(shape.rows, span));
  dims.push_back(MakeIndexConst(shape.cols, span));
  return std::make_shared<MakeTuple>(std::move(dims), span);
}

ExprPtr MakeOffsets(const Window& window, size_t rank, const Span& span) {
  auto offsets = window.batch;
  if (rank >= 2) offsets.push_back(window.row);
  offsets.push_back(window.col);
  return std::make_shared<MakeTuple>(std::move(offsets), span);
}

VarPtr BindCall(std::vector<StmtPtr>& body, const std::string& name, const CallPtr& call) {
  auto value = std::make_shared<Var>(name, call->GetType(), call->span_);
  body.push_back(std::make_shared<AssignStmt>(value, call, call->span_));
  return value;
}

VarPtr Load(std::vector<StmtPtr>& body, const VarPtr& src, const Window& window, MemorySpace space,
            const std::string& name, const Span& span, std::optional<Shape> physical = std::nullopt) {
  const auto rank = As<TensorType>(src->GetType())->shape_.size();
  auto& registry = OpRegistry::GetInstance();
  auto shape = MakeShape(physical.value_or(window.size), rank, span);
  auto valid = MakeShape(window.size, rank, span);
  auto call = registry.Create("tile.load", {src, MakeOffsets(window, rank, span), shape, valid},
                              {{"target_memory", space}}, span);
  if (rank == 2) return BindCall(body, name, call);
  // Native tile conversions and scratch planning use 2D windows, including
  // a one-row window over a rank-one Tensor.
  auto loaded = BindCall(body, name + "_panel", call);
  return BindCall(body, name,
                  registry.Create("tile.reshape",
                                  {loaded, MakeShape(physical.value_or(window.size), 2, span)}, {}, span));
}

StmtPtr Store(const VarPtr& value, const VarPtr& destination, const Window& window, const Span& span) {
  const auto rank = As<TensorType>(destination->GetType())->shape_.size();
  auto call = OpRegistry::GetInstance().Create(
      "tile.store", {value, MakeOffsets(window, rank, span), destination}, {}, span);
  return std::make_shared<EvalStmt>(call, span);
}

// Prefer consecutive columns when choosing independently executable windows for ND transfers.
std::optional<Shape> ChooseWindow(Shape whole, bool split_rows, bool split_cols, const FitsTile& fits) {
  const int64_t min_rows = split_rows ? 1 : whole.rows;
  int64_t cols = split_cols
                     ? LargestFitting(whole.cols, [&](int64_t width) { return fits({min_rows, width}); })
                     : whole.cols;
  if (cols == 0 || !fits({min_rows, cols})) return std::nullopt;

  // Vector transfers benefit from a 32-byte aligned column origin even with
  // eight-bit operands. A static remainder retains its exact logical width.
  constexpr int64_t kTransferGranule = 32;
  if (cols < whole.cols && cols >= kTransferGranule) {
    const int64_t aligned = cols - cols % kTransferGranule;
    if (fits({min_rows, aligned})) cols = aligned;
  }

  const int64_t rows = split_rows
                           ? LargestFitting(whole.rows, [&](int64_t height) { return fits({height, cols}); })
                           : whole.rows;
  if (rows == 0) return std::nullopt;
  return Shape{rows, cols};
}

// Emit a loop for the interior and a static remainder on each axis.
StmtPtr EmitGrid(Shape whole, Shape tile, const std::string& name, const Span& span,
                 const TileEmitter& emit) {
  return EmitAxis(whole.rows, tile.rows, name + "_m", span, [&](const ExprPtr& row, int64_t rows) {
    return EmitAxis(whole.cols, tile.cols, name + "_n", span,
                    [&](const ExprPtr& col, int64_t cols) { return emit(Window{{rows, cols}, row, col}); });
  });
}

StmtPtr EmitBatchedGrid(const std::vector<int64_t>& whole, Shape tile, const std::string& name,
                        const Span& span, const TileEmitter& emit) {
  auto plane = PlaneShape(whole);
  std::vector<ExprPtr> coordinates;
  std::function<StmtPtr(size_t)> batch = [&](size_t axis) -> StmtPtr {
    if (axis + 2 >= whole.size()) {
      return EmitGrid(plane, tile, name, span, [&](const Window& window) {
        return emit(Window{window.size, window.row, window.col, coordinates});
      });
    }
    return EmitAxis(whole[axis], 1, name + "_batch_" + std::to_string(axis), span,
                    [&](const ExprPtr& offset, int64_t) {
                      coordinates.push_back(offset);
                      auto body = batch(axis + 1);
                      coordinates.pop_back();
                      return body;
                    });
  };
  return batch(0);
}

StmtPtr CopyTensor(const VarPtr& source, const VarPtr& destination, const std::vector<int64_t>& whole,
                   Shape tile, int64_t alignment, const Span& span) {
  return EmitBatchedGrid(whole, tile, destination->name_hint_ + "_copy", span, [&](const Window& window) {
    std::vector<StmtPtr> body;
    auto box = VectorBox(window.size, IsColumnPlane(whole), false, alignment);
    auto value = Load(body, source, window, MemorySpace::Vec, "copy_tile", span, box);
    body.push_back(Store(value, destination, window, span));
    return SeqStmts::Flatten(std::move(body), span);
  });
}

// Defer writeback to a captured destination until all potentially overlapping inputs have been read.
StmtPtr Materialize(const InCoreScopeStmtPtr& scope, const TensorRegion& region,
                    const backend::BackendHandler& handler, std::vector<Diagnostic>& diagnostics,
                    const std::function<StmtPtr(const VarPtr&)>& compute, const std::string& description,
                    uint64_t divisor) {
  const auto result_type = As<TensorType>(region.result->GetType());
  const auto extents = StaticExtents(result_type);
  if (!extents) return nullptr;
  const auto out = PlaneShape(*extents);
  const auto backend_limit = handler.GetCapacityBytes(MemorySpace::Vec) / divisor;
  const auto result_bytes = result_type->dtype_.GetByte();
  if (result_bytes == 0) return nullptr;
  const auto alignment = static_cast<int64_t>(32 / result_bytes);
  std::optional<Shape> copy_tile;
  if (region.destination) {
    auto destination_type = As<TensorType>(region.destination->GetType());
    if (destination_type->dtype_ != result_type->dtype_) return nullptr;
    copy_tile = ChooseWindow(out, true, true, [&](Shape window) {
      uint64_t remaining = backend_limit;
      for (auto rows : {window.rows, out.rows % window.rows}) {
        for (auto cols : {window.cols, out.cols % window.cols}) {
          if (rows == 0 || cols == 0) continue;
          const auto box = VectorBox({rows, cols}, IsColumnPlane(*extents), false, alignment);
          if (static_cast<uint64_t>(box.rows) > remaining / static_cast<uint64_t>(box.cols)) return false;
          auto bytes = storage_size::StaticStorageBytes(
              static_cast<uint64_t>(box.rows) * static_cast<uint64_t>(box.cols), result_type->dtype_);
          if (!bytes || *bytes > remaining) return false;
          remaining -= *bytes;
        }
      }
      return true;
    });
    if (!copy_tile) return nullptr;
  }

  const auto& span = scope->span_;
  const auto shape = std::make_shared<MakeTuple>(result_type->shape_, span);
  auto create = OpRegistry::GetInstance().Create(
      "tensor.create", {shape}, {{"dtype", result_type->dtype_}, {"layout", TensorLayout::ND}}, span);
  std::vector<StmtPtr> body;
  auto storage = BindCall(body, region.result->name_hint_ + "_storage", create);
  body.push_back(std::make_shared<InCoreScopeStmt>(SplitMode::None, scope->name_hint_, compute(storage), span,
                                                   scope->leading_comments_));
  // Let outlining resolve the post-call storage name for the original result.
  // The kernel sees its destination exclusively through store operations.
  body.push_back(std::make_shared<AssignStmt>(region.result, storage, span));
  if (region.destination) {
    auto copy = CopyTensor(region.result, region.destination, *extents, *copy_tile, alignment, span);
    body.push_back(
        std::make_shared<InCoreScopeStmt>(SplitMode::None, scope->name_hint_ + "_writeback", copy, span));
    body.push_back(std::make_shared<AssignStmt>(region.writeback, region.destination, span));
  }
  {
    std::ostringstream message;
    message << "Scope '" << scope->name_hint_ << "'. " << description;
    if (region.destination) {
      uint64_t elements = 1;
      bool static_bytes = true;
      for (const auto extent : *extents) {
        if (elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(extent)) {
          static_bytes = false;
          break;
        }
        elements *= static_cast<uint64_t>(extent);
      }
      const auto bytes =
          static_bytes ? storage_size::StaticStorageBytes(elements, result_type->dtype_) : std::nullopt;
      message << " A compiler-owned GM output";
      if (bytes) message << " of " << *bytes << " bytes";
      message << " and a separate writeback task preserve reads of overlapping input/output views.";
    }
    diagnostics.emplace_back(DiagnosticSeverity::PerfHint, "TensorTiling", 0, "PH-TENSOR-001", message.str(),
                             span);
  }
  return SeqStmts::Flatten(std::move(body), span);
}

// Vector windows and partial reductions.
using ValueMap = std::unordered_map<const Var*, VarPtr>;

struct VectorProgram {
  std::vector<const ConversionEntry*> conversions;
  std::unordered_map<const Var*, std::vector<int64_t>> shapes;
  std::vector<int64_t> extents{1, 1};
  Shape domain{1, 1};
  int64_t alignment = 1;
  bool align_rows = false;
  bool split_rows = true;
  bool split_cols = true;
};

std::optional<VectorProgram> Analyze(const TensorRegion& region) {
  VectorProgram program;
  auto record = [&](const VarPtr& var) {
    if (program.shapes.count(var.get()) != 0) return true;
    auto type = As<TensorType>(var->GetType());
    auto shape = StaticExtents(type);
    if (!shape || type->dtype_.GetBit() < 8) return false;
    program.alignment = std::max(program.alignment, static_cast<int64_t>(32 / type->dtype_.GetByte()));
    if (shape->size() > program.extents.size()) {
      program.extents.insert(program.extents.begin(), shape->size() - program.extents.size(), 1);
    }
    auto extent = program.extents.end() - static_cast<std::ptrdiff_t>(shape->size());
    for (auto size : *shape) {
      *extent = std::max(*extent, size);
      ++extent;
    }
    program.shapes.emplace(var.get(), std::move(*shape));
    return true;
  };

  const auto& registry = OpConversionRegistry::GetInstance();
  for (const auto& assignment : region.calculations) {
    const auto call = As<Call>(assignment->value_);
    const auto* conversion = registry.Lookup(call->op_->name_);
    if (As<GlobalVar>(call->op_) || !call->attrs_.empty() || !conversion ||
        conversion->partition == TensorPartition::None || !conversion->input_reqs.empty() ||
        !record(assignment->var_)) {
      return std::nullopt;
    }
    if (IsOp(call, "tensor.div")) {
      const auto dtype = As<TensorType>(assignment->var_->GetType())->dtype_;
      if (call->GetKwarg<bool>("high_precision", false) ||
          (dtype != DataType::FP32 && dtype != DataType::FP16)) {
        return std::nullopt;
      }
    }
    for (const auto& arg : call->args_) {
      if (As<ScalarType>(arg->GetType())) continue;
      auto var = AsVarLike(arg);
      if (!var || !record(var)) return std::nullopt;
    }
    if (conversion->partition == TensorPartition::ReduceRows) program.split_rows = false;
    if (conversion->partition == TensorPartition::ReduceColumns) program.split_cols = false;
    program.conversions.push_back(conversion);
  }

  // Shape-preserving calls and broadcasts share a rectangular iteration
  // domain. A singleton axis has a constant origin; a reduction is legal only
  // when the complete reduced dimension remains inside its window.
  program.domain = PlaneShape(program.extents);
  // Column vectors have aligned physical heights. The expanded tiles must
  // use that same height so that row expansion and later pointwise ops agree.
  program.align_rows = !program.split_cols;
  for (const auto& [value, shape] : program.shapes) {
    const auto plane = PlaneShape(shape);
    if (plane.cols == 1 && plane.rows > 1 && plane.rows == program.domain.rows) {
      program.align_rows = true;
    }
    auto extent = program.extents.end() - static_cast<std::ptrdiff_t>(shape.size());
    for (auto size : shape) {
      if (size != 1 && size != *extent) return std::nullopt;
      ++extent;
    }
  }
  const auto& result_shape = program.shapes.at(region.result.get());
  const auto result = PlaneShape(result_shape);
  // A single result must cover the complete batch grid. A leading broadcast
  // on an input is allowed, but does not change which result is written.
  if (result_shape.size() != program.extents.size()) {
    if (result_shape.size() != 1 || program.domain.rows != 1 || program.extents.size() != 2) {
      return std::nullopt;
    }
  }
  for (size_t i = 0; i + 2 < result_shape.size(); ++i) {
    if (result_shape[i] != program.extents[i]) return std::nullopt;
  }
  if ((result.rows == 1 && program.domain.rows > 1 && program.split_rows) ||
      (result.cols == 1 && program.domain.cols > 1 && program.split_cols)) {
    return std::nullopt;
  }
  return program;
}

Window ForValue(const std::vector<int64_t>& extents, const Window& window, const Span& span) {
  auto shape = PlaneShape(extents);
  auto local = ProjectBatch(window, extents, span);
  if (shape.rows == 1) {
    local.size.rows = 1;
    local.row = MakeIndexConst(0, span);
  }
  if (shape.cols == 1) {
    local.size.cols = 1;
    local.col = MakeIndexConst(0, span);
  }
  return local;
}

Shape BoxForValue(const VectorProgram& program, const std::vector<int64_t>& extents, Shape valid) {
  const auto shape = PlaneShape(extents);
  // Native reductions consume row-major input even when the logical domain
  // has one column. Load true DN columns first, then expand into padded ND
  // storage. Plain pointwise columns keep their contiguous DN layout.
  const bool single_column_reduction =
      (!program.split_rows || !program.split_cols) && program.domain.cols == 1;
  const bool column = IsColumnPlane(extents) && !single_column_reduction;
  const bool align_rows = program.align_rows && shape.rows == program.domain.rows;
  return VectorBox(valid, column, align_rows, program.alignment);
}

VarPtr ExpandBroadcast(std::vector<StmtPtr>& stmts, const VarPtr& value, Shape source, Shape result,
                       Shape valid, Shape box, const Span& span) {
  if (source.rows == result.rows && source.cols == result.cols) return value;

  // Equalize the shapes before ordinary binary lowering. A native pointwise
  // instruction reads a full rectangle from both operands, even when tensor
  // shape inference has allowed one operand to broadcast.
  auto& registry = OpRegistry::GetInstance();
  const auto type = As<TileType>(value->GetType());
  const bool scalar_input = source.rows == 1 && source.cols == 1;
  const bool scalar_to_column = scalar_input && box.cols == 1 && box.rows > 1;
  // TEXPANDS writes row-major storage. A row of scalar copies can be viewed
  // as a column vector because a column vector has the same contiguous layout.
  const auto expand_box = scalar_to_column ? Shape{1, box.rows} : box;
  const auto expand_valid = scalar_to_column ? Shape{1, valid.rows} : valid;
  auto target =
      BindCall(stmts, "broadcast_box",
               registry.Create("tile.create", {MakeShape(expand_box, 2, span)},
                               {{"dtype", type->dtype_}, {"target_memory", MemorySpace::Vec}}, span));
  if (expand_box.rows != expand_valid.rows || expand_box.cols != expand_valid.cols) {
    target = BindCall(stmts, "broadcast_valid",
                      registry.Create("tile.set_validshape",
                                      {target, MakeIndexConst(expand_valid.rows, span),
                                       MakeIndexConst(expand_valid.cols, span)},
                                      {}, span));
  }

  if (scalar_input) {
    const auto rank = type->shape_.size();
    const auto zero = MakeIndexConst(0, span);
    auto indices = std::make_shared<MakeTuple>(std::vector<ExprPtr>(rank, zero), span);
    auto scalar =
        BindCall(stmts, "broadcast_scalar", registry.Create("tile.read", {value, indices}, {}, span));
    auto filled =
        BindCall(stmts, "broadcast_tile", registry.Create("tile.expands", {target, scalar}, {}, span));
    if (!scalar_to_column) return filled;
    return BindCall(stmts, "broadcast_column",
                    registry.Create("tile.reshape", {filled, MakeShape(box, 2, span)}, {}, span));
  }
  auto vector = value;
  if (type->shape_.size() == 1) {
    auto row =
        std::make_shared<MakeTuple>(std::vector<ExprPtr>{MakeIndexConst(1, span), type->shape_.back()}, span);
    vector = BindCall(stmts, "broadcast_row", registry.Create("tile.reshape", {value, row}, {}, span));
  }
  const std::string op = source.cols == result.cols ? "tile.col_expand" : "tile.row_expand";
  return BindCall(stmts, "broadcast_tile", registry.Create(op, {target, vector}, {}, span));
}

struct VectorTile {
  std::vector<StmtPtr> stmts;
  VarPtr result;
};

VectorTile LowerVectorTile(const TensorRegion& region, const VectorProgram& program, const Window& window) {
  VectorTile tile;
  ValueMap values;
  for (size_t i = 0; i < region.calculations.size(); ++i) {
    const auto& assignment = region.calculations[i];
    const auto call = As<Call>(assignment->value_);
    const auto& span = call->span_;
    const auto& result_extents = program.shapes.at(assignment->var_.get());
    const auto result_shape = PlaneShape(result_extents);
    const auto result_window = ForValue(result_extents, window, span);
    std::vector<ExprPtr> args;
    for (const auto& arg : call->args_) {
      if (As<ScalarType>(arg->GetType())) {
        args.push_back(arg);
        continue;
      }
      const auto var = AsVarLike(arg);
      const auto& extents = program.shapes.at(var.get());
      auto known = values.find(var.get());
      if (known == values.end()) {
        const auto view = ForValue(extents, window, span);
        const auto box = BoxForValue(program, extents, view.size);
        // DMA preserves a captured column vector's DN layout. If a column
        // reduction needs padded row-major storage, copy into that storage
        // after the load; padding the load itself asks DMA to transpose it.
        const auto shape = PlaneShape(extents);
        const bool padded_column = IsColumnPlane(extents) && box.cols > 1;
        const auto load_box = padded_column ? VectorBox(view.size, true, false, program.alignment) : box;
        auto loaded =
            Load(tile.stmts, var, view, MemorySpace::Vec, var->name_hint_ + "_tile", span, load_box);
        if (padded_column) {
          loaded = ExpandBroadcast(tile.stmts, loaded, shape, {shape.rows, box.cols}, view.size, box, span);
        }
        known = values.emplace(var.get(), loaded).first;
      }
      auto operand = known->second;
      const auto partition = program.conversions[i]->partition;
      const auto operand_shape = PlaneShape(extents);
      if (partition == TensorPartition::Pointwise) {
        const auto box = BoxForValue(program, result_extents, result_window.size);
        operand =
            ExpandBroadcast(tile.stmts, operand, operand_shape, result_shape, result_window.size, box, span);
      }
      args.push_back(operand);
    }

    // The normal conversion produces the tile instructions and any workspace
    // declarations. The planner inventories those same declarations.
    auto converted = program.conversions[i]->func(args, call->kwargs_, span, ConversionContext{});
    tile.stmts.insert(tile.stmts.end(), converted.prologue.begin(), converted.prologue.end());
    auto value =
        std::make_shared<Var>(assignment->var_->name_hint_ + "_tile", converted.result->GetType(), span);
    tile.stmts.push_back(std::make_shared<AssignStmt>(value, converted.result, span));
    values.emplace(assignment->var_.get(), value);
  }
  tile.result = values.at(region.result.get());
  return tile;
}

enum class ReductionAxis { Rows, Columns };

struct SplitReduction {
  ReductionAxis axis;
  const char* combine_op;
  bool preserve_row_order;
};

std::optional<SplitReduction> ChooseSplitReduction(const TensorRegion& region, const VectorProgram& program) {
  // A terminal FP32 reduction can stream its reduced axis when a native
  // pointwise op combines the partials. Producers must be window-local.
  const auto last = As<Call>(region.calculations.back()->value_);
  const bool row_sum = IsOp(last, "tensor.row_sum");
  const bool row_max = IsOp(last, "tensor.row_max");
  if (!row_sum && !row_max && !IsOp(last, "tensor.col_sum")) return std::nullopt;
  if (As<TensorType>(region.result->GetType())->dtype_ != DataType::FP32 ||
      As<TensorType>(last->args_.front()->GetType())->dtype_ != DataType::FP32) {
    return std::nullopt;
  }
  if (std::any_of(program.conversions.begin(), program.conversions.end() - 1,
                  [](const auto* conv) { return conv->partition != TensorPartition::Pointwise; })) {
    return std::nullopt;
  }
  const auto input = AsVarLike(last->args_.front());
  const auto source = PlaneShape(program.shapes.at(input.get()));
  const bool reduce_columns = row_sum || row_max;
  // Another calculation may set a larger domain through singleton broadcasting.
  // Combining partials across that domain would count this input repeatedly.
  if (reduce_columns ? source.cols != program.domain.cols : source.rows != program.domain.rows) {
    return std::nullopt;
  }
  const auto axis = reduce_columns ? ReductionAxis::Columns : ReductionAxis::Rows;
  const bool sequential = !reduce_columns && !last->GetKwarg<bool>("is_binary", false);
  return SplitReduction{axis, row_max ? "tile.maximum" : "tile.add", sequential};
}

// Continue a sequential column sum by making the previous sum the first row
// of the next input. Reducing a chunk from zero and then adding its partial
// would regroup FP32 additions and violate col_sum's default row order.
VectorTile ContinueSequentialColumnSum(VectorTile tile, const VarPtr& carry) {
  const auto assignment = As<AssignStmt>(tile.stmts.back());
  const auto reduction = As<Call>(assignment->value_);
  const auto input = reduction->args_.front();
  const auto type = As<TileType>(input->GetType());
  const auto& span = reduction->span_;
  const auto rows = As<ConstInt>(type->shape_[0])->value_;
  const auto cols = As<ConstInt>(type->shape_[1])->value_;
  const auto valid = type->tile_view_ && !type->tile_view_->valid_shape.empty()
                         ? type->tile_view_->valid_shape
                         : type->shape_;
  const auto valid_rows = As<ConstInt>(valid[0])->value_;
  const auto box =
      VectorBox({rows + 1, cols}, cols == 1, false, static_cast<int64_t>(32 / type->dtype_.GetByte()));
  tile.stmts.pop_back();
  auto& registry = OpRegistry::GetInstance();
  auto storage =
      BindCall(tile.stmts, "sequential_input",
               registry.Create("tile.create", {MakeShape(box, 2, span)},
                               {{"dtype", type->dtype_}, {"target_memory", MemorySpace::Vec}}, span));
  const auto zero = MakeIndexConst(0, span);
  const auto one = MakeIndexConst(1, span);
  auto first =
      BindCall(tile.stmts, "sequential_seed",
               registry.Create("tile.assemble",
                               {storage, carry, MakeOffsets({{1, cols}, zero, zero}, 2, span)}, {}, span));
  auto rest =
      BindCall(tile.stmts, "sequential_rows",
               registry.Create("tile.assemble",
                               {first, input, MakeOffsets({{rows, cols}, one, zero}, 2, span)}, {}, span));
  auto narrowed = BindCall(tile.stmts, "sequential_valid",
                           registry.Create("tile.set_validshape",
                                           {rest, MakeIndexConst(valid_rows + 1, span), valid[1]}, {}, span));
  tile.result = BindCall(tile.stmts, "sequential_sum", registry.Create("tile.col_sum", {narrowed}, {}, span));
  return tile;
}

VectorTile LowerReduction(const TensorRegion& region, const VectorProgram& program, const Window& output,
                          SplitReduction reduction, int64_t step) {
  const auto& span = region.result->span_;
  auto index = std::make_shared<Var>("reduce_axis", std::make_shared<ScalarType>(DataType::INDEX), span);
  const auto zero = MakeIndexConst(0, span);
  const bool rows = reduction.axis == ReductionAxis::Rows;
  const auto extent = rows ? program.domain.rows : program.domain.cols;
  const auto interior = (extent / step) * step;
  auto compute = [&](const ExprPtr& offset, int64_t count, const VarPtr& carry) {
    auto window = output;
    if (rows) {
      window.row = offset;
      window.size.rows = count;
    } else {
      window.col = offset;
      window.size.cols = count;
    }
    auto tile = LowerVectorTile(region, program, window);
    if (carry && reduction.preserve_row_order) return ContinueSequentialColumnSum(std::move(tile), carry);
    return tile;
  };

  auto total = compute(zero, step, nullptr);
  auto combine = [&](VectorTile& tile, const VarPtr& previous) {
    if (reduction.preserve_row_order) return tile.result;
    return BindCall(
        tile.stmts, "reduced",
        OpRegistry::GetInstance().Create(reduction.combine_op, {previous, tile.result}, {}, span));
  };
  if (interior > step) {
    auto carry = std::make_shared<IterArg>("reduce_carry", total.result->GetType(), total.result, span);
    auto result = std::make_shared<Var>("reduce_result", total.result->GetType(), span);
    auto partial = compute(index, step, carry);
    auto value = combine(partial, carry);
    partial.stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{value}, span));
    total.stmts.push_back(std::make_shared<ForStmt>(
        index, MakeIndexConst(step, span), MakeIndexConst(interior, span), MakeIndexConst(step, span),
        std::vector<IterArgPtr>{carry}, SeqStmts::Flatten(std::move(partial.stmts), span),
        std::vector<VarPtr>{result}, span));
    total.result = result;
  }
  if (interior < extent) {
    auto tail = compute(MakeIndexConst(interior, span), extent - interior, total.result);
    total.result = combine(tail, total.result);
    total.stmts.insert(total.stmts.end(), tail.stmts.begin(), tail.stmts.end());
  }
  return total;
}

bool ReserveTile(const TileTypePtr& type, const backend::Backend& target, uint64_t* remaining) {
  constexpr uint64_t kAllocationAlignment = 32;
  const auto bytes = utils::StaticPhysicalAllocationBytes(type, MemorySpace::Vec, target.GetHandler());
  if (!bytes || *bytes > *remaining) return false;
  const uint64_t padding = (kAllocationAlignment - *bytes % kAllocationAlignment) % kAllocationAlignment;
  if (padding > *remaining - *bytes) return false;
  *remaining -= *bytes + padding;
  return true;
}

bool ReserveCastStep(const CallPtr& call, const backend::Backend& target, uint64_t* remaining) {
  if (!ReserveTile(As<TileType>(call->GetType()), target, remaining)) return false;
  if (call->args_.size() != 1) return true;
  const auto source = As<TileType>(call->args_.front()->GetType());
  const auto dtype = call->GetKwarg<DataType>("target_type");
  const bool saturating = GetEmittedSaturationMode(call) == static_cast<int>(SaturationMode::kOn);
  // The conversion has no scratch operand yet. Budget the same backend plan
  // InitMemRef will use to materialize it.
  const auto plan =
      target.PlanOpScratch("tile.cast", {source->dtype_, std::nullopt, dtype, saturating, source});
  if (!plan) return false;
  if (plan->status == backend::OpScratchStatus::NotRequired) return true;
  if (plan->status != backend::OpScratchStatus::Required || !plan->spec) return false;

  const auto& spec = *plan->spec;
  auto scratch = std::make_shared<TileType>(std::vector<int64_t>{spec.rows, spec.cols}, spec.dtype,
                                            std::nullopt, std::nullopt, MemorySpace::Vec);
  return ReserveTile(scratch, target, remaining);
}

bool ReserveCastChain(const CallPtr& call, const backend::Backend& target, uint64_t* remaining) {
  const auto source = As<TileType>(call->args_.front()->GetType());
  const auto dtype = call->GetKwarg<DataType>("target_type");
  // Tensor windowing excludes packed dtypes. The same native path will be
  // selected during LegalizeTileCast, with its intermediate results and scratch.
  const auto chain = utils::PlanTileCastChain(target.GetHandler()->GetTcvtAdjacency(), source->dtype_, dtype,
                                              /*allow_direct_fp4_to_fp8=*/false);
  if (chain.empty()) return false;
  if (chain.size() == 1) return ReserveCastStep(call, target, remaining);

  auto args = call->args_;
  const auto mode = call->GetKwarg<int>("mode");
  for (size_t i = 0; i + 1 < chain.size(); ++i) {
    auto intermediate = utils::MakeIntermediateTileCast(args.front(), chain[i], mode, call->span_);
    if (!ReserveCastStep(intermediate, target, remaining)) return false;
    args.front() = std::move(intermediate);
  }
  auto final = OpRegistry::GetInstance().Create("tile.cast", args, call->kwargs_, call->span_);
  return ReserveCastStep(final, target, remaining);
}

bool CountTileBytes(const std::vector<StmtPtr>& stmts, const backend::Backend& target, uint64_t* remaining) {
  for (const auto& stmt : stmts) {
    if (const auto loop = As<ForStmt>(stmt)) {
      const auto seq = As<SeqStmts>(loop->body_);
      if (!seq || !CountTileBytes(seq->stmts_, target, remaining)) return false;
      continue;
    }
    if (As<YieldStmt>(stmt)) continue;
    const auto assignment = As<AssignStmt>(stmt);
    if (!assignment) return false;
    const auto assignment_type = assignment->var_->GetType();
    if (As<ScalarType>(assignment_type)) continue;
    const auto call = As<Call>(assignment->value_);
    if (IsOp(call, "tile.cast")) {
      if (!ReserveCastChain(call, target, remaining)) return false;
    } else if (!ReserveTile(As<TileType>(assignment_type), target, remaining)) {
      return false;
    }
  }
  return true;
}

using LowerTile = std::function<VectorTile(const Window&)>;

bool FitsGrid(const TensorRegion& region, const VectorProgram& program, Shape domain, Shape tile,
              const backend::Backend& target, uint64_t remaining, const LowerTile& lower) {
  const auto& span = region.result->span_;
  const auto zero = MakeIndexConst(0, span);

  // Each interior and remainder template can introduce an independent buffer
  // before memory reuse. Budget all of them and let the allocator coalesce
  // compatible lifetimes afterwards.
  const int64_t rows[] = {tile.rows, domain.rows % tile.rows};
  const int64_t cols[] = {tile.cols, domain.cols % tile.cols};
  for (int64_t height : rows) {
    if (height == 0) continue;
    for (int64_t width : cols) {
      if (width == 0) continue;
      std::vector<ExprPtr> batch(program.extents.size() - 2, zero);
      auto lowered = lower({{height, width}, zero, zero, std::move(batch)});
      if (!CountTileBytes(lowered.stmts, target, &remaining)) return false;
    }
  }
  return true;
}

StmtPtr SplitVectorPipeline(const InCoreScopeStmtPtr& scope, const TensorRegion& region,
                            const backend::Backend& target, std::vector<Diagnostic>& diagnostics,
                            uint64_t divisor);

StmtPtr TryVector(const InCoreScopeStmtPtr& scope, const TensorRegion& region, const backend::Backend& target,
                  std::vector<Diagnostic>& diagnostics, uint64_t divisor) {
  const auto& handler = *target.GetHandler();
  const auto program = Analyze(region);
  if (!program || handler.GetCapacityBytes(MemorySpace::Vec) == 0) {
    return nullptr;
  }
  const auto capacity = handler.GetCapacityBytes(MemorySpace::Vec) / divisor;
  if (capacity == 0) return nullptr;
  auto domain = program->domain;
  auto extents = program->extents;
  LowerTile lower = [&](const Window& window) { return LowerVectorTile(region, *program, window); };
  const auto fits = [&](Shape shape) {
    return FitsGrid(region, *program, domain, shape, target, capacity, lower);
  };
  auto tile = ChooseWindow(domain, program->split_rows, program->split_cols, fits);
  std::string reduction_note;
  if (!tile) {
    auto reduction = ChooseSplitReduction(region, *program);
    if (!reduction) return SplitVectorPipeline(scope, region, target, diagnostics, divisor);
    if (reduction->preserve_row_order) {
      reduction_note = " Column sums retain sequential row order across windows.";
    } else if (std::string(reduction->combine_op) == "tile.add") {
      reduction_note = " Partial FP32 sums regroup additions across windows.";
    }
    const bool rows = reduction->axis == ReductionAxis::Rows;
    // Find a reduction chunk while holding one independent row or column.
    // Then grow the independent axis with the complete loop and tail in the
    // budget. Neither search changes the output's logical shape.
    domain = rows ? Shape{1, domain.cols} : Shape{domain.rows, 1};
    extents[extents.size() - 2] = domain.rows;
    extents.back() = domain.cols;
    const auto unit = Shape{1, 1};
    const auto reduced = rows ? Shape{program->domain.rows, 1} : Shape{1, program->domain.cols};
    auto chunk = ChooseWindow(reduced, rows, !rows, [&](Shape shape) {
      auto test = [&](const Window& window) {
        return LowerReduction(region, *program, window, *reduction, rows ? shape.rows : shape.cols);
      };
      return FitsGrid(region, *program, unit, unit, target, capacity, test);
    });
    if (!chunk) return SplitVectorPipeline(scope, region, target, diagnostics, divisor);
    const auto step = rows ? chunk->rows : chunk->cols;
    const auto extent = rows ? reduced.rows : reduced.cols;
    const auto count = extent / step + (extent % step != 0);
    reduction_note += " Each output window streams " + std::to_string(extent) +
                      (rows ? " input rows in " : " input columns in ") + std::to_string(count) +
                      " chunks of up to " + std::to_string(step) + ".";
    lower = [&, reduction = *reduction, step](const Window& window) {
      return LowerReduction(region, *program, window, reduction, step);
    };
    tile = ChooseWindow(domain, !rows, rows, fits);
    if (!tile) return SplitVectorPipeline(scope, region, target, diagnostics, divisor);
  }

  return Materialize(
      scope, region, handler, diagnostics,
      [&](const VarPtr& storage) {
        const auto& span = scope->span_;
        return EmitBatchedGrid(extents, *tile, region.result->name_hint_, span, [&](const Window& window) {
          auto lowered = lower(window);
          auto output = ForValue(program->shapes.at(region.result.get()), window, span);
          lowered.stmts.push_back(Store(lowered.result, storage, output, span));
          return SeqStmts::Flatten(std::move(lowered.stmts), span);
        });
      },
      "Vector output windows " + std::to_string(tile->rows) + " x " + std::to_string(tile->cols) +
          ". The bounded window plan uses a Vec budget of " + std::to_string(capacity) + " bytes." +
          reduction_note,
      divisor);
}

StmtPtr SplitVectorPipeline(const InCoreScopeStmtPtr& scope, const TensorRegion& stage,
                            const backend::Backend& target, std::vector<Diagnostic>& diagnostics,
                            uint64_t divisor) {
  if (stage.calculations.size() < 2) return nullptr;

  // A reduction can require a complete column or row, and the rest of the
  // chain may need more buffers than fit beside it. Individual stages can
  // use different window sizes and share their results through GM.
  std::vector<StmtPtr> kernels;
  for (size_t i = 0; i < stage.calculations.size(); ++i) {
    const auto& stmt = stage.calculations[i];
    const bool last = i + 1 == stage.calculations.size();
    TensorRegion part{
        {stmt}, stmt->var_, last ? stage.destination : nullptr, last ? stage.writeback : nullptr};
    auto isolated = std::make_shared<InCoreScopeStmt>(
        SplitMode::None, scope->name_hint_ + "_vec_" + std::to_string(i), scope->body_, scope->span_,
        i == 0 ? scope->leading_comments_ : std::vector<std::string>{});
    auto tiled = TryVector(isolated, part, target, diagnostics, divisor);
    if (!tiled) return nullptr;
    kernels.push_back(tiled);
  }
  return SeqStmts::Flatten(std::move(kernels), scope->span_);
}

// Matrix product windows and K tails.
struct Operand {
  VarPtr source;
  std::vector<int64_t> shape;
  bool transpose;
};

struct Matmul {
  Operand lhs;
  Operand rhs;
  std::vector<int64_t> output;
  Shape plane;
  int64_t k;
  DataType dtype;
  DataType accumulator;
  std::vector<std::pair<std::string, std::any>> kwargs;
};

std::optional<Operand> GetOperand(const ExprPtr& arg, bool transpose) {
  auto var = AsVarLike(arg);
  auto shape = StaticExtents(As<TensorType>(arg->GetType()));
  if (!var || !shape || shape->size() < 2) return std::nullopt;
  return Operand{var, std::move(*shape), transpose};
}

bool BroadcastsTo(const Operand& operand, const std::vector<int64_t>& output) {
  if (operand.shape.size() > output.size()) return false;
  const size_t first = output.size() - operand.shape.size();
  for (size_t i = 0; i + 2 < operand.shape.size(); ++i) {
    if (operand.shape[i] != 1 && operand.shape[i] != output[first + i]) return false;
  }
  return true;
}

std::optional<Matmul> AnalyzeMatmul(const AssignStmtPtr& product) {
  const auto call = As<Call>(product->value_);
  if (!IsOp(call, "tensor.matmul") || call->args_.size() != 2 || !call->attrs_.empty() ||
      call->GetKwarg<bool>("c_matrix_nz", false)) {
    return std::nullopt;
  }
  auto a = GetOperand(call->args_[0], call->GetKwarg<bool>("a_trans", false));
  auto b = GetOperand(call->args_[1], call->GetKwarg<bool>("b_trans", false));
  const auto result_type = As<TensorType>(product->var_->GetType());
  auto output = StaticExtents(result_type);
  if (!a || !b || !output || output->size() < 2 || !BroadcastsTo(*a, *output) || !BroadcastsTo(*b, *output)) {
    return std::nullopt;
  }

  const auto dtype = As<TensorType>(a->source->GetType())->dtype_;
  if (dtype != As<TensorType>(b->source->GetType())->dtype_ ||
      (dtype != DataType::BF16 && dtype != DataType::FP16 && dtype != DataType::FP32 &&
       dtype != DataType::INT8)) {
    return std::nullopt;
  }

  auto lhs = PlaneShape(a->shape);
  auto rhs = PlaneShape(b->shape);
  if (a->transpose) std::swap(lhs.rows, lhs.cols);
  if (b->transpose) std::swap(rhs.rows, rhs.cols);
  auto plane = PlaneShape(*output);
  if (lhs.rows != plane.rows || rhs.cols != plane.cols || lhs.cols != rhs.rows) return std::nullopt;
  if (std::max({plane.rows, plane.cols, lhs.cols}) > std::numeric_limits<int>::max() / 2) {
    return std::nullopt;
  }
  auto accumulator = MatmulAccumulatorDataType(dtype, dtype);
  if (!CubeWritebackSupportsDataType(accumulator, result_type->dtype_)) return std::nullopt;

  std::vector<std::pair<std::string, std::any>> kwargs;
  for (const auto& kwarg : call->kwargs_) {
    if (kwarg.first == "tf32_mode") kwargs.push_back(kwarg);
  }
  return Matmul{std::move(*a), std::move(*b), std::move(*output), plane,
                lhs.cols,      dtype,         accumulator,        std::move(kwargs)};
}

struct Blocking {
  Shape tile;
  int64_t k_step;
  int64_t k_main;
  int64_t alignment;
};

std::optional<Blocking> ChooseBlocking(const Matmul& matmul, const backend::BackendHandler& handler,
                                       uint64_t divisor) {
  const auto operand_bytes = matmul.dtype.GetByte();
  const auto accumulator_bytes = matmul.accumulator.GetByte();
  const auto mat_capacity = handler.GetCapacityBytes(MemorySpace::Mat) / divisor;
  const auto acc_capacity = handler.GetCapacityBytes(MemorySpace::Acc) / divisor;
  if (mat_capacity == 0 || acc_capacity == 0 || operand_bytes == 0 || accumulator_bytes == 0) {
    return std::nullopt;
  }
  const auto alignment =
      std::max<int64_t>(handler.GetL0FractalAlignment(), static_cast<int64_t>(32 / operand_bytes));

  // Each non-aligned output axis may add a separate remainder template. Keep
  // room for those declarations even if the memory planner cannot reuse them.
  const int sections = (matmul.plane.rows % alignment ? 2 : 1) * (matmul.plane.cols % alignment ? 2 : 1);
  const int64_t k_main = matmul.k - matmul.k % alignment;
  const int operand_sections = sections * (k_main > 0 && k_main < matmul.k ? 2 : 1);
  utils::L0TileConfig config;
  config.M = static_cast<int>(AlignUp(matmul.plane.rows, alignment));
  config.N = static_cast<int>(AlignUp(matmul.plane.cols, alignment));
  config.K = static_cast<int>(std::max(k_main, alignment));
  config.bytes_a = config.bytes_b = static_cast<int>(operand_bytes);
  config.bytes_c = static_cast<int>(accumulator_bytes);
  config.l0a_bytes =
      std::min(handler.GetCapacityBytes(MemorySpace::Left) / divisor, mat_capacity / 2) / operand_sections;
  config.l0b_bytes =
      std::min(handler.GetCapacityBytes(MemorySpace::Right) / divisor, mat_capacity / 2) / operand_sections;
  config.l0c_bytes = acc_capacity / sections;
  config.l0c_align_m = handler.GetL0cMAlignment(matmul.accumulator);
  config.align_m = config.align_n = config.align_k = static_cast<int>(alignment);
  config.min_m = config.min_n = config.min_k =
      std::max(static_cast<int>(alignment), handler.GetMinL0TileDim());
  config.independent_output_tiles = true;
  const auto selected = utils::ChooseL0Tile(config);
  if (selected.m <= 0 || selected.n <= 0 || selected.k <= 0 || config.K % selected.k != 0) {
    return std::nullopt;
  }
  const auto step = [&](int64_t extent, int64_t chosen) {
    // On aligned axes use divisors to avoid otherwise unnecessary tile types.
    return extent % alignment == 0 ? std::gcd(extent, chosen) : std::min(extent, chosen);
  };
  return Blocking{{step(matmul.plane.rows, selected.m), step(matmul.plane.cols, selected.n)},
                  selected.k,
                  k_main,
                  alignment};
}

VarPtr LoadOperand(std::vector<StmtPtr>& body, const Operand& operand, Window window, Shape physical,
                   MemorySpace space, const std::string& name, const Span& span) {
  auto& registry = OpRegistry::GetInstance();
  window = ProjectBatch(window, operand.shape, span);
  if (operand.transpose) {
    std::swap(window.size.rows, window.size.cols);
    std::swap(window.row, window.col);
    std::swap(physical.rows, physical.cols);
  }
  auto tile = Load(body, operand.source, window, MemorySpace::Mat, name + "_mat", span, physical);
  if (operand.transpose) {
    tile = BindCall(body, name + "_view", registry.Create("tile.transpose_view", {tile}, {}, span));
  }
  return BindCall(body, name, registry.Create("tile.move", {tile}, {{"target_memory", space}}, span));
}

// Stage only the final, incomplete K panel. Both operands must have
// actual zeros in the padded cells because the Cube reads whole K fractals.
std::optional<Operand> StageKTail(std::vector<StmtPtr>& body, const InCoreScopeStmtPtr& scope,
                                  const Operand& operand, bool rows, const Blocking& blocking, DataType dtype,
                                  uint64_t capacity, const std::string& name) {
  auto& registry = OpRegistry::GetInstance();
  const auto& span = scope->span_;
  const size_t axis = operand.shape.size() - (rows ? 2 : 1);
  const int64_t tail = operand.shape[axis] - blocking.k_main;
  auto extents = operand.shape;
  extents[axis] = blocking.alignment;
  auto plane = PlaneShape(extents);
  const auto elem_bytes = dtype.GetByte();
  if (elem_bytes == 0) return std::nullopt;
  const auto granule = static_cast<int64_t>(32 / elem_bytes);

  // The A2/A3 sub-block fill for multi-byte lanes passes its remaining row
  // count to an 8-bit vector_dup repeat. Keep horizontal padding within one
  // repeat field so every row of a narrow K panel is explicitly zero-filled.
  constexpr int64_t kFillPadRepeatMax = 255;
  const int64_t max_rows = !rows && elem_bytes > 1 ? kFillPadRepeatMax : plane.rows;
  const auto tile = ChooseWindow(plane, !rows, rows, [&](Shape shape) {
    const auto box = VectorBox(shape, false, false, granule);
    return box.rows <= max_rows &&
           static_cast<uint64_t>(box.rows) <= capacity / (4 * elem_bytes) / static_cast<uint64_t>(box.cols);
  });
  if (!tile) return std::nullopt;

  std::vector<ExprPtr> dims;
  dims.reserve(extents.size());
  for (const auto extent : extents) dims.push_back(MakeIndexConst(extent, span));
  auto tensor =
      BindCall(body, name,
               registry.Create("tensor.create", {std::make_shared<MakeTuple>(std::move(dims), span)},
                               {{"dtype", dtype}, {"layout", TensorLayout::ND}}, span));
  auto grid = EmitBatchedGrid(extents, *tile, name, span, [&](const Window& window) {
    std::vector<StmtPtr> stmts;
    auto source = window;
    if (rows) {
      source.row = MakeIndexConst(blocking.k_main, span);
      source.size.rows = tail;
    } else {
      source.col = MakeIndexConst(blocking.k_main, span);
      source.size.cols = tail;
    }
    auto physical = VectorBox(window.size, false, false, granule);
    auto load = Load(stmts, operand.source, source, MemorySpace::Vec, name + "_tile", span, physical);
    auto pad = BindCall(stmts, name + "_pad",
                        registry.Create("tile.fillpad", {load}, {{"pad_value", PadValue::zero}}, span));
    auto valid_rows = MakeIndexConst(window.size.rows, span);
    auto valid_cols = MakeIndexConst(window.size.cols, span);
    auto valid = BindCall(stmts, name + "_valid",
                          registry.Create("tile.set_validshape", {pad, valid_rows, valid_cols}, {}, span));
    stmts.push_back(Store(valid, tensor, window, span));
    return SeqStmts::Flatten(std::move(stmts), span);
  });
  body.push_back(
      std::make_shared<InCoreScopeStmt>(SplitMode::None, scope->name_hint_ + "_" + name, grid, span));
  return Operand{tensor, std::move(extents), operand.transpose};
}

using TailOperands = std::pair<Operand, Operand>;

VarPtr EmitMatmul(std::vector<StmtPtr>& body, const Matmul& matmul, const Blocking& blocking,
                  const std::optional<TailOperands>& tail, const Window& window, const Span& span) {
  auto& registry = OpRegistry::GetInstance();
  auto box =
      Shape{AlignUp(window.size.rows, blocking.alignment), AlignUp(window.size.cols, blocking.alignment)};
  const auto shape = As<MakeTuple>(MakeShape(box, 2, span))->elements_;
  const auto valid = As<MakeTuple>(MakeShape(window.size, 2, span))->elements_;
  auto dot = [&](std::vector<StmtPtr>& stmts, const Operand& a, const Operand& b, int64_t k,
                 const ExprPtr& offset, const VarPtr& carry, const ExprPtr& first = nullptr) {
    auto left = LoadOperand(stmts, a, {{window.size.rows, k}, window.row, offset, window.batch},
                            {box.rows, k}, MemorySpace::Left, "left", span);
    auto right = LoadOperand(stmts, b, {{k, window.size.cols}, offset, window.col, window.batch},
                             {k, box.cols}, MemorySpace::Right, "right", span);
    if (!carry) {
      return BindCall(stmts, "product", registry.Create("tile.matmul", {left, right}, matmul.kwargs, span));
    }
    std::vector<ExprPtr> args{carry, left, right};
    if (first) args.push_back(first);
    return BindCall(stmts, "partial", registry.Create("tile.matmul_acc", args, matmul.kwargs, span));
  };

  const auto zero = MakeIndexConst(0, span);
  VarPtr result;
  if (blocking.k_step == blocking.k_main) {
    result = dot(body, matmul.lhs, matmul.rhs, blocking.k_step, zero, nullptr);
  } else if (blocking.k_main > 0) {
    auto init = acc_init::BuildNarrowedAccInit(shape, valid, matmul.accumulator, "product_init", span);
    body.insert(body.end(), init.stmts.begin(), init.stmts.end());
    auto index = std::make_shared<Var>("reduce_k", std::make_shared<ScalarType>(DataType::INDEX), span);
    auto carry = std::make_shared<IterArg>("product_carry", init.value->GetType(), init.value, span);
    result = std::make_shared<Var>("product", init.value->GetType(), span);
    std::vector<StmtPtr> reduction;
    auto partial =
        dot(reduction, matmul.lhs, matmul.rhs, blocking.k_step, index, carry, MakeEq(index, zero, span));
    reduction.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{partial}, span));
    body.push_back(std::make_shared<ForStmt>(
        index, zero, MakeIndexConst(blocking.k_main, span), MakeIndexConst(blocking.k_step, span),
        std::vector<IterArgPtr>{carry}, SeqStmts::Flatten(std::move(reduction), span),
        std::vector<VarPtr>{result}, span));
  }
  if (tail) {
    result = dot(body, tail->first, tail->second, blocking.alignment, zero, result);
  }
  return result;
}

StmtPtr TryMatmul(const InCoreScopeStmtPtr& scope, const TensorRegion& region, const backend::Backend& target,
                  std::vector<Diagnostic>& diagnostics, uint64_t divisor) {
  const auto& handler = *target.GetHandler();
  const auto& product = region.calculations.front();
  const auto matmul = AnalyzeMatmul(product);
  if (!matmul) return nullptr;
  const bool has_epilogue = region.calculations.size() > 1;
  const auto blocking = ChooseBlocking(*matmul, handler, divisor);
  if (!blocking) return nullptr;
  const auto& span = scope->span_;
  StmtPtr epilogue;
  if (has_epilogue) {
    auto tail = region;
    tail.calculations.erase(tail.calculations.begin());
    auto vector_scope = std::make_shared<InCoreScopeStmt>(SplitMode::None, scope->name_hint_ + "_epilogue",
                                                          scope->body_, span);
    epilogue = TryVector(vector_scope, tail, target, diagnostics, divisor);
    if (!epilogue) return nullptr;
  }

  std::vector<StmtPtr> kernels;
  std::optional<TailOperands> tail;
  if (blocking->k_main < matmul->k) {
    const auto capacity = handler.GetCapacityBytes(MemorySpace::Vec) / divisor;
    auto a = StageKTail(kernels, scope, matmul->lhs, matmul->lhs.transpose, *blocking, matmul->dtype,
                        capacity, "lhs_k_tail");
    auto b = StageKTail(kernels, scope, matmul->rhs, !matmul->rhs.transpose, *blocking, matmul->dtype,
                        capacity, "rhs_k_tail");
    if (!a || !b) return nullptr;
    tail = TailOperands{std::move(*a), std::move(*b)};
  }

  const auto compute = [&](const VarPtr& storage) {
    return EmitBatchedGrid(matmul->output, blocking->tile, "product", span, [&](const Window& window) {
      std::vector<StmtPtr> body;
      auto value = EmitMatmul(body, *matmul, *blocking, tail, window, span);
      body.push_back(Store(value, storage, window, span));
      return SeqStmts::Flatten(std::move(body), span);
    });
  };
  TensorRegion gemm{{product}, product->var_, nullptr, nullptr};
  auto cube = Materialize(
      scope, epilogue ? gemm : region, handler, diagnostics, compute,
      "Matrix windows " + std::to_string(blocking->tile.rows) + " x " + std::to_string(blocking->tile.cols) +
          " with K panels of " + std::to_string(blocking->k_step) +
          ". The plan bounds Mat, Left, Right and Acc storage by their backend capacities.",
      divisor);
  if (!cube) return nullptr;
  kernels.push_back(cube);
  if (epilogue) kernels.push_back(epilogue);
  return SeqStmts::Flatten(std::move(kernels), span);
}

// Stage mixed matrix and vector computations through GM.
using Calculations = std::vector<AssignStmtPtr>;

bool IsProduct(const AssignStmtPtr& stmt) { return IsOp(As<Call>(stmt->value_), "tensor.matmul"); }

// A GM-backed accumulator can enter a stage through a vector addition.
// Keep the predicated Cube form in the ordinary pipeline.
std::optional<Calculations> ExpandAccumulations(const Calculations& calculations) {
  auto& registry = OpRegistry::GetInstance();
  Calculations result;
  result.reserve(calculations.size());
  for (const auto& stmt : calculations) {
    auto call = As<Call>(stmt->value_);
    if (!IsOp(call, "tensor.matmul_acc")) {
      result.push_back(stmt);
      continue;
    }
    if (call->args_.size() != 3 || !call->attrs_.empty()) return std::nullopt;
    auto type = As<TensorType>(stmt->var_->GetType());
    if (!StaticExtents(type)) return std::nullopt;

    auto kwargs = call->kwargs_;
    kwargs.emplace_back("out_dtype", type->dtype_);
    const auto& span = call->span_;
    auto product = registry.Create("tensor.matmul", {call->args_[1], call->args_[2]}, kwargs, span);
    auto partial = std::make_shared<Var>(stmt->var_->name_hint_ + "_product", product->GetType(), span);
    result.push_back(std::make_shared<AssignStmt>(partial, product, span));
    auto sum = registry.Create("tensor.add", {call->args_[0], partial}, {}, span);
    result.push_back(std::make_shared<AssignStmt>(stmt->var_, sum, span));
  }
  return result;
}

UseCounts CountRangeUses(Calculations::const_iterator begin, Calculations::const_iterator end,
                         const Span& span) {
  return CountUses(SeqStmts::Flatten(std::vector<StmtPtr>(begin, end), span));
}

std::vector<TensorRegion> PartitionStages(const Calculations& calculations, const TensorRegion& region,
                                          const Span& span) {
  std::vector<TensorRegion> stages;
  const auto uses = CountRangeUses(calculations.begin(), calculations.end(), span);
  auto append = [&](auto begin, auto end) {
    stages.push_back(TensorRegion{Calculations(begin, end), (*(end - 1))->var_, nullptr, nullptr});
  };

  for (auto begin = calculations.begin(); begin != calculations.end();) {
    if (IsProduct(*begin)) {
      append(begin, begin + 1);
      ++begin;
      continue;
    }
    const auto end = std::find_if(begin + 1, calculations.end(), IsProduct);
    const auto local_uses = CountRangeUses(begin, end, span);
    // Each windowed region exports one tensor. Materialize individual results
    // when a vector run has multiple values used in subsequent stages.
    const bool escapes = std::any_of(begin, end - 1, [&](const AssignStmtPtr& stmt) {
      return Uses(uses, stmt->var_) != Uses(local_uses, stmt->var_);
    });
    if (escapes) {
      for (auto it = begin; it != end; ++it) append(it, it + 1);
    } else {
      append(begin, end);
    }
    begin = end;
  }
  stages.back().destination = region.destination;
  stages.back().writeback = region.writeback;
  return stages;
}

StmtPtr TryMatrixPipeline(const InCoreScopeStmtPtr& scope, const TensorRegion& region,
                          const backend::Backend& target, std::vector<Diagnostic>& diagnostics,
                          uint64_t divisor) {
  const auto calculations = ExpandAccumulations(region.calculations);
  if (!calculations || calculations->size() < 2 ||
      std::none_of(calculations->begin(), calculations->end(), IsProduct)) {
    return nullptr;
  }
  const auto stages = PartitionStages(*calculations, region, scope->span_);
  auto emit_stage = [&](size_t i) {
    const auto& stage = stages[i];
    auto comments = i == 0 ? scope->leading_comments_ : std::vector<std::string>{};
    auto isolated =
        std::make_shared<InCoreScopeStmt>(SplitMode::None, scope->name_hint_ + "_stage_" + std::to_string(i),
                                          scope->body_, scope->span_, std::move(comments));
    if (IsProduct(stage.calculations.front())) {
      return TryMatmul(isolated, stage, target, diagnostics, divisor);
    }
    return TryVector(isolated, stage, target, diagnostics, divisor);
  };

  std::vector<StmtPtr> emitted(stages.size());
  for (size_t i = 0; i < stages.size(); ++i) {
    emitted[i] = emit_stage(i);
    if (!emitted[i]) return nullptr;
  }
  return SeqStmts::Flatten(std::move(emitted), scope->span_);
}

// Source spans survive kernel splitting. Include both endpoints so the index
// never guesses from generated names or merges scopes with a common start.
std::string SpanKey(const Span& span) {
  return span.filename_ + ":" + std::to_string(span.begin_line_) + ":" + std::to_string(span.begin_column_) +
         ":" + std::to_string(span.end_line_) + ":" + std::to_string(span.end_column_);
}

class FailedScopeFinder : public IRVisitor {
 public:
  std::vector<InCoreScopeStmtPtr> scopes;

 protected:
  void VisitStmt_(const InCoreScopeStmtPtr& scope) override { scopes.push_back(scope); }
  // These scopes already own their dispatch and storage contract.
  void VisitStmt_(const SpmdScopeStmtPtr&) override {}
  void VisitStmt_(const ClusterScopeStmtPtr&) override {}
  void VisitStmt_(const RuntimeScopeStmtPtr&) override {}
  void VisitStmt_(const RegScopeStmtPtr&) override {}
};

// Trace failed kernels through the actual call graph. Deduplication preserves
// caller spans while replacing callees, so every original scope using a shared
// kernel remains identifiable even after wrappers and split kernels appear.
struct CapacityCaller {
  std::string function;
  Span span;
};
using CapacityCallers = std::unordered_map<std::string, std::vector<CapacityCaller>>;

class CapacityCallerCollector : public IRVisitor {
 public:
  CapacityCallerCollector(std::string function, CapacityCallers& callers)
      : function_(std::move(function)), callers_(callers) {}

  void VisitExpr(const ExprPtr& expr) override {
    if (expr && expressions_.insert(expr.get()).second) IRVisitor::VisitExpr(expr);
  }
  void VisitStmt(const StmtPtr& stmt) override {
    if (stmt && statements_.insert(stmt.get()).second) IRVisitor::VisitStmt(stmt);
  }

 protected:
  void VisitExpr_(const CallPtr& call) override {
    Add(call->op_, call->span_);
    IRVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const SubmitPtr& submit) override {
    Add(submit->op_, submit->span_);
    IRVisitor::VisitExpr_(submit);
  }

 private:
  void Add(const OpPtr& callee, const Span& span) {
    if (As<GlobalVar>(callee)) callers_[callee->name_].push_back({function_, span});
  }
  std::string function_;
  CapacityCallers& callers_;
  std::unordered_set<const Expr*> expressions_;
  std::unordered_set<const Stmt*> statements_;
};

class ReplaceRetriedScopes : public IRMutator {
 public:
  explicit ReplaceRetriedScopes(const std::unordered_map<const InCoreScopeStmt*, StmtPtr>& replacements)
      : replacements_(replacements) {}

 protected:
  StmtPtr VisitStmt_(const InCoreScopeStmtPtr& scope) override {
    auto it = replacements_.find(scope.get());
    return it == replacements_.end() ? scope : it->second;
  }

 private:
  const std::unordered_map<const InCoreScopeStmt*, StmtPtr>& replacements_;
};

}  // namespace

void TensorAllocationRetry::CheckMatLoadCapacity(const ProgramPtr& program) const {
  const auto* context = PassContext::Current();
  if ((context && context->GetMemoryPlanner() != MemoryPlanner::PyPTO) ||
      !backend::BackendConfig::IsConfigured()) {
    return;
  }
  const auto* target = context ? context->GetBackend() : backend::GetBackend();
  if (!target || !target->GetHandler()) return;

  // Only compiler-managed Tensor scopes are eligible for recovery. A hand-
  // authored Tile kernel retains the original pass and allocation behavior.
  std::unordered_set<std::string> eligible;
  for (const auto& [gvar, func] : original_->functions_) {
    if (func->func_type_ != FunctionType::Opaque && !IsOrchestrationLike(func->func_type_)) continue;
    FailedScopeFinder finder;
    finder.VisitStmt(func->body_);
    const auto uses = CountUses(func->body_);
    for (const auto& scope : finder.scopes) {
      const auto region = MatchRegion(scope, uses);
      if (!region) continue;
      const auto calculations = ExpandAccumulations(region->calculations);
      if (calculations &&
          std::any_of(calculations->begin(), calculations->end(),
                      [](const AssignStmtPtr& stmt) { return AnalyzeMatmul(stmt).has_value(); })) {
        eligible.insert(SpanKey(scope->span_));
      }
    }
  }

  class MatLoadCapacityChecker : public IRVisitor {
   public:
    MatLoadCapacityChecker(FunctionPtr func, const backend::Backend& target)
        : func_(std::move(func)), target_(target) {}

   protected:
    void VisitStmt_(const AssignStmtPtr& assign) override {
      const auto call = As<Call>(assign->value_);
      const auto type = As<TileType>(assign->var_->GetType());
      // ND loads may be streamed into separate planes by FlattenTileNdTo2D.
      // This check sees its already-lowered rank-2 Mat allocation requests.
      if (IsOp(call, "tile.load") && type && type->shape_.size() == 2 &&
          type->GetMemorySpace() == MemorySpace::Mat) {
        const auto bytes = utils::StaticPhysicalAllocationBytes(type, MemorySpace::Mat, target_.GetHandler());
        const auto limit = target_.GetMemSize(MemorySpace::Mat);
        if (bytes && limit != 0 && *bytes > limit) {
          throw TensorAllocationFailure(func_, MemorySpace::Mat, *bytes, limit,
                                        ". A single Mat load cannot fit before L0 tiling");
        }
      }
      if (call && !As<GlobalVar>(call->op_)) {
        const auto* entry = OpConversionRegistry::GetInstance().Lookup(call->op_->name_);
        if (entry) {
          for (const auto& [index, req] : entry->input_reqs) {
            if (req.demanded_space.Get() != MemorySpace::Mat || index >= call->args_.size()) continue;
            const auto source = As<TensorType>(call->args_[index]->GetType());
            const auto shape = StaticExtents(source);
            if (!shape || shape->size() != 2) continue;
            // The conversion contract loads this complete rank-2 operand into
            // Mat. Its logical bytes are a lower bound on the physical request:
            // padding can only increase it, and reuse cannot shrink one buffer.
            // Check before conversion so a later consumer's shape deduction
            // cannot hide this already-impossible allocation in a mixed region.
            const auto rows = static_cast<uint64_t>((*shape)[0]);
            const auto cols = static_cast<uint64_t>((*shape)[1]);
            if (rows > std::numeric_limits<uint64_t>::max() / cols) continue;
            const auto bytes = storage_size::StaticStorageBytes(rows * cols, source->dtype_);
            const auto limit = target_.GetMemSize(MemorySpace::Mat);
            if (bytes && limit != 0 && *bytes > limit) {
              throw TensorAllocationFailure(func_, MemorySpace::Mat, *bytes, limit,
                                            ". A whole-operand Mat load cannot fit before Tensor conversion");
            }
          }
        }
      }
      IRVisitor::VisitStmt_(assign);
    }

   private:
    FunctionPtr func_;
    const backend::Backend& target_;
  };

  std::vector<TensorAllocationFailure::Request> failures;
  for (const auto& [gvar, func] : program->functions_) {
    if (eligible.count(SpanKey(func->span_)) == 0) continue;
    MatLoadCapacityChecker checker(func, *target);
    try {
      checker.VisitStmt(func->body_);
    } catch (const TensorAllocationFailure& failure) {
      failures.insert(failures.end(), failure.requests.begin(), failure.requests.end());
    }
  }
  if (!failures.empty()) throw TensorAllocationFailure(std::move(failures));
}

ProgramPtr TensorAllocationRetry::Retry(const TensorAllocationFailure& failure, const ProgramPtr& current) {
  if (!backend::BackendConfig::IsConfigured()) return nullptr;
  const auto* context = PassContext::Current();
  if (context && context->GetMemoryPlanner() != MemoryPlanner::PyPTO) return nullptr;
  const auto* target = context ? context->GetBackend() : backend::GetBackend();
  if (!target || !target->GetHandler()) return nullptr;

  // Index source scopes once. Owner identity prevents equal source positions
  // in different orchestration functions from being conflated.
  using ScopeInfo = std::pair<InCoreScopeStmtPtr, size_t>;
  std::vector<UseCounts> owner_uses;
  owner_uses.reserve(original_->functions_.size());
  std::map<std::pair<std::string, std::string>, ScopeInfo> source_scopes;
  for (const auto& [gvar, func] : original_->functions_) {
    if (func->func_type_ != FunctionType::Opaque && !IsOrchestrationLike(func->func_type_)) continue;
    FailedScopeFinder finder;
    finder.VisitStmt(func->body_);
    owner_uses.push_back(CountUses(func->body_));
    for (const auto& scope : finder.scopes) {
      auto [it, inserted] = source_scopes.emplace(std::make_pair(gvar->name_, SpanKey(scope->span_)),
                                                  std::make_pair(scope, owner_uses.size() - 1));
      if (!inserted) it->second.first = nullptr;  // Ambiguous source locations remain unsupported.
    }
  }

  CapacityCallers callers;
  std::unordered_map<const Function*, std::string> symbols;
  for (const auto& [gvar, func] : current->functions_) {
    symbols.emplace(func.get(), gvar->name_);
    CapacityCallerCollector collector(gvar->name_, callers);
    collector.VisitStmt(func->body_);
  }
  std::vector<std::string> pending;
  for (const auto& request : failure.requests) {
    const auto symbol = symbols.find(request.function.get());
    if (symbol == symbols.end()) return nullptr;
    pending.push_back(symbol->second);
  }

  // One reverse traversal handles all failures, without copying an origin set
  // through every wrapper. Stop at original scope calls, retaining only the
  // callers of failed kernels rather than all users of a representative span.
  std::unordered_set<std::string> visited;
  std::unordered_set<const InCoreScopeStmt*> retried;
  std::vector<ScopeInfo> affected;
  for (size_t index = 0; index < pending.size(); ++index) {
    const auto name = pending[index];
    if (!visited.insert(name).second) continue;
    const auto found = callers.find(name);
    if (found == callers.end()) continue;
    for (const auto& caller : found->second) {
      const auto source = source_scopes.find(std::make_pair(caller.function, SpanKey(caller.span)));
      if (source == source_scopes.end()) {
        pending.push_back(caller.function);
        continue;
      }
      const auto& info = source->second;
      if (!info.first) return nullptr;
      if (retried.insert(info.first.get()).second) affected.push_back(info);
    }
  }
  if (affected.empty()) return nullptr;

  for (const auto& [failed_scope, use_index] : affected) {
    const auto region = MatchRegion(failed_scope, owner_uses[use_index]);
    if (!region) return nullptr;

    const auto previous = plans_.find(failed_scope.get());
    uint64_t divisor = previous == plans_.end() ? 1 : previous->second.divisor;
    std::vector<Diagnostic> diagnostics;
    StmtPtr replacement;
    // A uint64_t capacity admits at most 64 strict halvings. Skip identical
    // candidate IR locally instead of recompiling a plan that already failed.
    for (size_t attempt = 0; attempt < 64; ++attempt) {
      if (previous != plans_.end() || attempt != 0) {
        if (divisor > std::numeric_limits<uint64_t>::max() / 2) return nullptr;
        divisor *= 2;
      }
      diagnostics.clear();
      replacement = TryMatmul(failed_scope, *region, *target, diagnostics, divisor);
      if (!replacement) {
        diagnostics.clear();
        replacement = TryVector(failed_scope, *region, *target, diagnostics, divisor);
      }
      if (!replacement) {
        diagnostics.clear();
        replacement = TryMatrixPipeline(failed_scope, *region, *target, diagnostics, divisor);
      }
      if (!replacement) return nullptr;
      if (previous == plans_.end() ||
          !structural_equal(replacement, previous->second.replacement, /*enable_auto_mapping=*/true)) {
        break;
      }
      replacement = nullptr;
    }
    if (!replacement) return nullptr;
    plans_.insert_or_assign(failed_scope.get(), Plan{divisor, replacement, std::move(diagnostics)});
  }

  std::unordered_map<const InCoreScopeStmt*, StmtPtr> replacements;
  for (const auto& [scope, plan] : plans_) replacements.emplace(scope, plan.replacement);
  ReplaceRetriedScopes mutator(replacements);
  std::vector<FunctionPtr> functions;
  for (const auto& [gvar, func] : original_->functions_) {
    auto body = mutator.VisitStmt(func->body_);
    if (body == func->body_) {
      functions.push_back(func);
    } else {
      auto changed = MutableCopy(func);
      changed->body_ = body;
      functions.push_back(changed);
    }
  }
  return std::make_shared<Program>(functions, original_->name_, original_->span_);
}

void TensorAllocationRetry::EmitSelectedDiagnostics() const {
  const auto* context = PassContext::Current();
  if (context && context->GetDiagnosticPhase() == DiagnosticPhase::None) return;
  for (const auto& [scope, plan] : plans_) {
    EmitDiagnostics(plan.diagnostics, "TensorTiling");
  }
}

}  // namespace tensor_to_tile
}  // namespace ir
}  // namespace pypto
