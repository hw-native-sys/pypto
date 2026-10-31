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

#include "pypto/ir/transforms/utils/cross_core_pipe.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/core_side_ops.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace cross_core_pipe {

namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

struct ConsumerFifo {
  std::vector<const Var*> pops;
  size_t next_free = 0;
  bool valid = true;
};

// Inspect the finalized FIFO schedule, not merely the number of live values:
// the emitted tfree has no tile operand and releases the oldest slot. Nested
// scopes must preserve their incoming FIFO ownership; otherwise reject the
// unsupported schedule rather than silently releasing an enclosing tile.
// Checkpoint/rewind avoids copying live queues at each nesting level: O(N).
int RequiredConsumerDepth(const std::vector<StmtPtr>& stmts, core_affinity::CoreSide side, ConsumerFifo& fifo,
                          bool diagnose = true) {
  int peak = static_cast<int>(fifo.pops.size() - fifo.next_free);
  auto visit_body = [&](const StmtPtr& body) {
    const size_t entry_tail = fifo.pops.size();
    const size_t entry_head = fifo.next_free;
    const int body_peak = RequiredConsumerDepth(FlattenBody(body), side, fifo, diagnose);
    const bool unchanged = fifo.next_free == entry_head && fifo.pops.size() == entry_tail;
    const bool empty_on_both_sides = entry_head == entry_tail && fifo.next_free == fifo.pops.size();
    if (!diagnose && (!fifo.valid || !(unchanged || empty_on_both_sides))) {
      fifo.valid = false;
      return;
    }
    CHECK_SPAN(unchanged || empty_on_both_sides, body->span_)
        << "Automatic cross-core pipe does not support FIFO ownership escaping a control-flow body; "
           "place/consume and release each gathered tile within its scope.";
    fifo.pops.resize(entry_tail);
    fifo.next_free = entry_head;
    peak = std::max(peak, body_peak);
  };
  for (const auto& stmt : stmts) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
    auto call = assign ? std::dynamic_pointer_cast<const Call>(assign->value_)
                       : (eval ? std::dynamic_pointer_cast<const Call>(eval->expr_) : nullptr);
    if (call && call->op_) {
      if (call->op_->name_ == core_side_ops::TPopOp(side)) {
        INTERNAL_CHECK_SPAN(assign, stmt->span_) << "Internal error: tpop must bind a tile";
        fifo.pops.push_back(assign->var_.get());
        peak = std::max(peak, static_cast<int>(fifo.pops.size() - fifo.next_free));
      } else if (call->op_->name_ == core_side_ops::TFreeOp(side)) {
        auto tile = call->args_.empty() ? nullptr : AsVarLike(call->args_[0]);
        if (!diagnose &&
            !(tile && fifo.next_free < fifo.pops.size() && fifo.pops[fifo.next_free] == tile.get())) {
          fifo.valid = false;
          return peak;
        }
        CHECK_SPAN(tile && fifo.next_free < fifo.pops.size() && fifo.pops[fifo.next_free] == tile.get(),
                   stmt->span_)
            << "Automatic cross-core pipe requires FIFO release order; a retained earlier tile "
               "must be consumed before a later gathered tile is released.";
        ++fifo.next_free;
      }
    }
    if (auto loop = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      visit_body(loop->body_);
    } else if (auto loop = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      visit_body(loop->body_);
    } else if (auto branch = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      visit_body(branch->then_body_);
      if (branch->else_body_) {
        visit_body(*branch->else_body_);
      }
    }
  }
  return peak;
}

}  // namespace

bool FitsAutomaticPipeCapacity(const std::vector<StmtPtr>& stmts, core_affinity::CoreSide side,
                               int capacity) {
  ConsumerFifo fifo;
  const int required = RequiredConsumerDepth(stmts, side, fifo, /*diagnose=*/false);
  return fifo.valid && required <= capacity;
}

std::optional<int64_t> TryGetNonNegativeConstInt(const ExprPtr& expr) {
  auto value = transform_utils::EvalConstInt(expr);
  if (!value.has_value() || *value < 0) return std::nullopt;
  return value;
}

std::optional<int64_t> TryGetTileSlotSizeBytes(const TypePtr& type) {
  auto tile_type = std::dynamic_pointer_cast<const TileType>(type);
  if (!tile_type) return std::nullopt;

  int64_t element_count = 1;
  for (const auto& dim : tile_type->shape_) {
    auto dim_value = TryGetNonNegativeConstInt(dim);
    if (!dim_value.has_value()) return std::nullopt;
    INTERNAL_CHECK(*dim_value == 0 || element_count <= std::numeric_limits<int64_t>::max() / *dim_value)
        << "Tile element count overflow while inferring cross-core slot size";
    element_count *= *dim_value;
  }

  const int64_t bit_width = static_cast<int64_t>(tile_type->dtype_.GetBit());
  INTERNAL_CHECK(bit_width > 0) << "Unsupported dtype for cross-core slot size inference: "
                                << tile_type->dtype_.ToString();
  INTERNAL_CHECK(element_count <= (std::numeric_limits<int64_t>::max() - 7) / bit_width)
      << "Tile byte size overflow while inferring cross-core slot size";
  return (element_count * bit_width + 7) / 8;
}

void RecordObservedSlotSize(PipeDirectionMetadata& metadata, int64_t slot_size) {
  metadata.has_ops = true;
  if (std::find(metadata.observed_slot_sizes.begin(), metadata.observed_slot_sizes.end(), slot_size) ==
      metadata.observed_slot_sizes.end()) {
    metadata.observed_slot_sizes.push_back(slot_size);
  }
  if (!metadata.slot_size_bytes.has_value()) {
    metadata.slot_size_bytes = slot_size;
    return;
  }
  if (metadata.slot_size_bytes.value() != slot_size) {
    metadata.has_inconsistent_slot_size = true;
    metadata.slot_size_bytes = std::max(metadata.slot_size_bytes.value(), slot_size);
  }
}

void RecordTileSlotSize(PipeDirectionMetadata& metadata, const TypePtr& type) {
  metadata.has_ops = true;
  auto slot_size = TryGetTileSlotSizeBytes(type);
  if (slot_size.has_value()) {
    RecordObservedSlotSize(metadata, slot_size.value());
  }
}

void MergeDirectionMetadata(PipeDirectionMetadata& dst, const PipeDirectionMetadata& src) {
  dst.has_ops = dst.has_ops || src.has_ops;
  dst.fixpipe_bf16 = dst.fixpipe_bf16 || src.fixpipe_bf16;
  dst.fixpipe_relu = dst.fixpipe_relu || src.fixpipe_relu;
  dst.has_inconsistent_slot_size = dst.has_inconsistent_slot_size || src.has_inconsistent_slot_size;
  for (int64_t slot_size : src.observed_slot_sizes) {
    RecordObservedSlotSize(dst, slot_size);
  }
}

CrossCorePipeMetadata MergeCrossCorePipeMetadata(const CrossCorePipeMetadata& lhs,
                                                 const CrossCorePipeMetadata& rhs) {
  CrossCorePipeMetadata merged;
  MergeDirectionMetadata(merged.c2v, lhs.c2v);
  MergeDirectionMetadata(merged.c2v, rhs.c2v);
  MergeDirectionMetadata(merged.v2c, lhs.v2c);
  MergeDirectionMetadata(merged.v2c, rhs.v2c);
  merged.has_reserve_buffer = lhs.has_reserve_buffer || rhs.has_reserve_buffer;
  merged.has_import_peer_buffer = lhs.has_import_peer_buffer || rhs.has_import_peer_buffer;
  merged.has_aic_initialize_pipe = lhs.has_aic_initialize_pipe || rhs.has_aic_initialize_pipe;
  merged.has_aiv_initialize_pipe = lhs.has_aiv_initialize_pipe || rhs.has_aiv_initialize_pipe;
  return merged;
}

int BuildDirMask(const CrossCorePipeMetadata& metadata) {
  int dir_mask = 0;
  if (metadata.c2v.has_ops) dir_mask |= core_affinity::kDirMaskC2V;
  if (metadata.v2c.has_ops) dir_mask |= core_affinity::kDirMaskV2C;
  return dir_mask;
}

int GetPtoasImplicitSlotNum(int dir_mask) {
  return dir_mask == (core_affinity::kDirMaskC2V | core_affinity::kDirMaskV2C) ? 4 : 8;
}

std::optional<int64_t> GetCommonSlotSizeBytes(const CrossCorePipeMetadata& metadata) {
  std::optional<int64_t> common_slot_size;
  for (const auto* direction : {&metadata.c2v, &metadata.v2c}) {
    if (!direction->has_ops) continue;
    if (!direction->slot_size_bytes.has_value()) {
      return std::nullopt;
    }
    if (!common_slot_size.has_value()) {
      common_slot_size = direction->slot_size_bytes;
      continue;
    }
    common_slot_size = std::max(common_slot_size.value(), direction->slot_size_bytes.value());
  }
  return common_slot_size;
}

std::string BuildPipeBufferName(const std::string& func_name, core_affinity::PipeDirection direction) {
  return func_name +
         ((direction == core_affinity::PipeDirection::C2V) ? "_c2v_slot_buffer" : "_v2c_slot_buffer");
}

CallPtr CreateSystemOpCall(const std::string& op_name,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span) {
  return CreateSystemOpCall(op_name, {}, kwargs, span);
}

CallPtr CreateSystemOpCall(const std::string& op_name, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span) {
  return OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
}

CallPtr CreateReserveBuffer(const std::string& buffer_name, int64_t size_bytes, const Span& span) {
  INTERNAL_CHECK_SPAN(size_bytes >= 0 && size_bytes <= std::numeric_limits<int>::max(), span)
      << "Cross-core reserve_buffer size out of range: " << size_bytes;
  return CreateSystemOpCall("system.reserve_buffer",
                            {{"name", std::any(buffer_name)},
                             {"size", std::any(static_cast<int>(size_bytes))},
                             {"base", std::any(kAutoBufferBase)}},
                            span);
}

CallPtr CreateImportPeerBuffer(const std::string& buffer_name, const std::string& peer_func,
                               const Span& span) {
  return CreateSystemOpCall("system.import_peer_buffer",
                            {{"name", std::any(buffer_name)}, {"peer_func", std::any(peer_func)}}, span);
}

CallPtr CreateInitializePipe(core_affinity::CoreSide side, int dir_mask, int slot_size_bytes,
                             const ExprPtr& c2v_consumer_buf, const ExprPtr& v2c_consumer_buf, int slot_num,
                             std::optional<int> local_slot_num, const Span& span) {
  INTERNAL_CHECK_SPAN(slot_size_bytes >= 0 && slot_size_bytes <= std::numeric_limits<int>::max(), span)
      << "Cross-core slot_size out of range: " << slot_size_bytes;
  INTERNAL_CHECK_SPAN(slot_num > 0, span) << "Cross-core slot_num must be positive: " << slot_num;
  // Always emit the transport depth so GM sizing and peer ring indexing agree.
  // The optional local depth separately describes the on-chip reserve.
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dir_mask", std::any(dir_mask)},
                                                          {"slot_size", std::any(slot_size_bytes)},
                                                          {"slot_num", std::any(slot_num)}};
  if (local_slot_num.has_value()) {
    kwargs.emplace_back("local_slot_num", local_slot_num.value());
  }
  const std::string op_name = core_side_ops::InitializePipeOp(side);
  return CreateSystemOpCall(op_name, {c2v_consumer_buf, v2c_consumer_buf}, kwargs, span);
}

void CollectCrossCorePipeMetadata(const std::vector<StmtPtr>& stmts, CrossCorePipeMetadata& metadata) {
  for (const auto& stmt : stmts) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
    CallPtr call;
    if (assign) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (eval) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    if (op) {
      if (IsOp(op, "system.reserve_buffer")) {
        metadata.has_reserve_buffer = true;
      } else if (IsOp(op, "system.import_peer_buffer")) {
        metadata.has_import_peer_buffer = true;
      } else if (IsOp(op, "system.aic_initialize_pipe")) {
        metadata.has_aic_initialize_pipe = true;
      } else if (IsOp(op, "system.aiv_initialize_pipe")) {
        metadata.has_aiv_initialize_pipe = true;
      } else if (IsOp(op, "tile.tpush_to_aiv") && call->args_.size() == 1) {
        auto type = As<TileType>(call->args_[0]->GetType());
        if (type && call->GetKwarg<bool>("fixpipe_bf16", false)) {
          metadata.c2v.fixpipe_bf16 = true;
          metadata.c2v.fixpipe_relu = call->GetKwarg<bool>("fixpipe_relu", false);
          RecordTileSlotSize(metadata.c2v,
                             std::make_shared<TileType>(type->shape_, DataType::BF16, std::nullopt,
                                                        type->tile_view_, type->memory_space_));
        } else {
          RecordTileSlotSize(metadata.c2v, call->args_[0]->GetType());
        }
      } else if (IsOp(op, "tile.tpush_to_aic") && call->args_.size() == 1) {
        RecordTileSlotSize(metadata.v2c, call->args_[0]->GetType());
      } else if (IsOp(op, "tile.tpop_from_aiv") && assign) {
        RecordTileSlotSize(metadata.v2c, assign->var_->GetType());
      } else if (IsOp(op, "tile.tpop_from_aic") && assign) {
        RecordTileSlotSize(metadata.c2v, assign->var_->GetType());
      }
    }

    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(for_stmt->body_), metadata);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(if_stmt->then_body_), metadata);
      const auto& else_body = if_stmt->else_body_;
      if (else_body) {
        CollectCrossCorePipeMetadata(FlattenBody(*else_body), metadata);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(while_stmt->body_), metadata);
    }
  }
}

CrossCorePipeMetadata CollectDominatingPipeSetupMetadata(const std::vector<StmtPtr>& stmts) {
  CrossCorePipeMetadata metadata;
  for (const auto& stmt : stmts) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
    CallPtr call;
    if (assign) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (eval) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    CrossCorePipeMetadata stmt_metadata;
    CollectCrossCorePipeMetadata({stmt}, stmt_metadata);
    if (stmt_metadata.HasCrossCoreOps()) {
      break;
    }
    if (op) {
      if (IsOp(op, "system.reserve_buffer")) {
        metadata.has_reserve_buffer = true;
      } else if (IsOp(op, "system.import_peer_buffer")) {
        metadata.has_import_peer_buffer = true;
      } else if (IsOp(op, "system.aic_initialize_pipe")) {
        metadata.has_aic_initialize_pipe = true;
      } else if (IsOp(op, "system.aiv_initialize_pipe")) {
        metadata.has_aiv_initialize_pipe = true;
      }
    }
  }
  return metadata;
}

namespace {
AutomaticPipeSetup BuildPipeSetup(const std::string& func_name, const std::string& aic_name,
                                  const std::string& aiv_name, const std::vector<StmtPtr>& aic_stmts,
                                  const std::vector<StmtPtr>& aiv_stmts, std::optional<int> slot_num_override,
                                  std::optional<int> local_slot_num_override, const Span& span,
                                  const CrossCorePipeMetadata& combined, int pipe_id) {
  const int dir_mask = BuildDirMask(combined);
  auto common_slot_size = GetCommonSlotSizeBytes(combined);
  if (dir_mask == 0 || !common_slot_size.has_value()) {
    return {};
  }

  // Always emit the transport depth, avoiding PTOAS's different implicit default.
  // Local storage follows it unless the supporting backend receives an explicit
  // local depth. GM allocation continues to consume only the transport signature.
  if (slot_num_override.has_value()) {
    INTERNAL_CHECK_SPAN(slot_num_override.value() > 0, span)
        << "Cross-core slot_num override must be positive: " << slot_num_override.value();
  }
  const int effective_slot_num = slot_num_override.value_or(kDefaultAutoPipeSlotNum);
  if (local_slot_num_override.has_value()) {
    // Keep allocation and both peer initializers in agreement. Merely shrinking
    // reserve_buffer would leave PTOAS indexing at the full transport depth.
    CHECK_SPAN(PassContext::Current()->GetBackendHandler()->SupportsLocalPipeDepth(), span)
        << "pl.cross_core_slot(local_slot_num=...) is only supported on A2/A3; "
           "omit local_slot_num on A5";
    CHECK_SPAN(*local_slot_num_override > 0 && *local_slot_num_override <= effective_slot_num, span)
        << "Cross-core local_slot_num (" << *local_slot_num_override << ") must be positive and <= slot_num ("
        << effective_slot_num << ")";
  }
  const int local_depth = local_slot_num_override.value_or(effective_slot_num);
  ConsumerFifo aic_fifo;
  ConsumerFifo aiv_fifo;
  const int required_slot_num =
      std::max(RequiredConsumerDepth(aic_stmts, core_affinity::CoreSide::AIC, aic_fifo),
               RequiredConsumerDepth(aiv_stmts, core_affinity::CoreSide::AIV, aiv_fifo));
  CHECK_SPAN(effective_slot_num >= required_slot_num, span)
      << "Automatic cross-core pipe slot_num=" << effective_slot_num << " is insufficient for "
      << required_slot_num
      << " simultaneously retained popped tiles; set cross_core_slot(slot_num=" << required_slot_num
      << ") or place/consume each gathered tile before gathering the next.";
  CHECK_SPAN(local_depth >= required_slot_num, span)
      << "Automatic cross-core pipe local_slot_num=" << local_depth << " is insufficient for "
      << required_slot_num
      << " simultaneously retained popped tiles; increase local_slot_num or place/consume "
         "each gathered tile before gathering the next.";
  // Bound-check the slot size before multiplying so an oversized inferred size
  // can't overflow the int64 buffer_size computation.
  const int64_t slot_size_i64 = common_slot_size.value();
  INTERNAL_CHECK_SPAN(slot_size_i64 >= 0 && slot_size_i64 <= std::numeric_limits<int>::max(), span)
      << "Cross-core slot_size out of range: " << slot_size_i64;
  const int slot_size_bytes = static_cast<int>(slot_size_i64);
  const int64_t buffer_size = slot_size_i64 * local_depth;
  AutomaticPipeSetup setup;

  std::shared_ptr<Var> aic_v2c_reserve_var;
  std::shared_ptr<Var> aic_c2v_import_var;
  std::shared_ptr<Var> aiv_c2v_reserve_var;
  std::shared_ptr<Var> aiv_v2c_import_var;

  auto zero_i32 = [&]() { return std::make_shared<ConstInt>(0, DataType::INT32, span); };
  auto var_as_expr = [](const std::shared_ptr<Var>& v) -> ExprPtr {
    return std::static_pointer_cast<const Expr>(v);
  };

  if (dir_mask & core_affinity::kDirMaskV2C) {
    const auto v2c_name = BuildPipeBufferName(func_name, core_affinity::PipeDirection::V2C);
    auto v2c_reserve = CreateReserveBuffer(v2c_name, buffer_size, span);
    aic_v2c_reserve_var = std::make_shared<Var>(v2c_name, v2c_reserve->GetType(), span);
    setup.aic_stmts.push_back(std::make_shared<AssignStmt>(aic_v2c_reserve_var, v2c_reserve, span));
    auto v2c_import = CreateImportPeerBuffer(v2c_name, aic_name, span);
    aiv_v2c_import_var = std::make_shared<Var>(v2c_name + "_import", v2c_import->GetType(), span);
    setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(aiv_v2c_import_var, v2c_import, span));
  }

  if (dir_mask & core_affinity::kDirMaskC2V) {
    const auto c2v_name = BuildPipeBufferName(func_name, core_affinity::PipeDirection::C2V);
    auto c2v_reserve = CreateReserveBuffer(c2v_name, buffer_size, span);
    aiv_c2v_reserve_var = std::make_shared<Var>(c2v_name, c2v_reserve->GetType(), span);
    setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(aiv_c2v_reserve_var, c2v_reserve, span));
    auto c2v_import = CreateImportPeerBuffer(c2v_name, aiv_name, span);
    aic_c2v_import_var = std::make_shared<Var>(c2v_name + "_import", c2v_import->GetType(), span);
    setup.aic_stmts.push_back(std::make_shared<AssignStmt>(aic_c2v_import_var, c2v_import, span));
  }

  // AIC: c2v operand = import on Cube; v2c operand = reserve on Cube (matches PTO codegen order).
  const ExprPtr aic_c2v_arg = aic_c2v_import_var ? var_as_expr(aic_c2v_import_var) : ExprPtr(zero_i32());
  const ExprPtr aic_v2c_arg = aic_v2c_reserve_var ? var_as_expr(aic_v2c_reserve_var) : ExprPtr(zero_i32());
  // AIV: c2v operand = reserve on Vector; v2c operand = import on Vector.
  const ExprPtr aiv_c2v_arg = aiv_c2v_reserve_var ? var_as_expr(aiv_c2v_reserve_var) : ExprPtr(zero_i32());
  const ExprPtr aiv_v2c_arg = aiv_v2c_import_var ? var_as_expr(aiv_v2c_import_var) : ExprPtr(zero_i32());

  auto initialize = [&](core_affinity::CoreSide side, const ExprPtr& c2v, const ExprPtr& v2c) {
    auto call = CreateInitializePipe(side, dir_mask, slot_size_bytes, c2v, v2c, effective_slot_num,
                                     local_slot_num_override, span);
    auto kwargs = call->kwargs_;
    if (pipe_id != 0) kwargs.emplace_back("id", pipe_id);
    if (combined.c2v.fixpipe_bf16) {
      kwargs.emplace_back("fixpipe_bf16", true);
      kwargs.emplace_back("fixpipe_relu", combined.c2v.fixpipe_relu);
    }
    return std::make_shared<EvalStmt>(
        CreateSystemOpCall(core_side_ops::InitializePipeOp(side), call->args_, kwargs, span), span);
  };
  setup.aic_stmts.push_back(initialize(core_affinity::CoreSide::AIC, aic_c2v_arg, aic_v2c_arg));
  setup.aiv_stmts.push_back(initialize(core_affinity::CoreSide::AIV, aiv_c2v_arg, aiv_v2c_arg));

  return setup;
}
}  // namespace

AutomaticPipeSetup BuildAutomaticPipeSetup(const std::string& func_name, const std::string& aic_name,
                                           const std::string& aiv_name, const std::vector<StmtPtr>& aic_stmts,
                                           const std::vector<StmtPtr>& aiv_stmts,
                                           std::optional<int> slot_num_override,
                                           std::optional<int> local_slot_num_override, const Span& span) {
  CrossCorePipeMetadata aic_metadata;
  CollectCrossCorePipeMetadata(aic_stmts, aic_metadata);
  CrossCorePipeMetadata aiv_metadata;
  CollectCrossCorePipeMetadata(aiv_stmts, aiv_metadata);
  CrossCorePipeMetadata combined = MergeCrossCorePipeMetadata(aic_metadata, aiv_metadata);

  if (!combined.HasCrossCoreOps() || aic_metadata.HasAnySetup() || aiv_metadata.HasAnySetup()) {
    return {};
  }

  if (combined.c2v.fixpipe_bf16 && combined.v2c.has_ops) {
    // PTOAS Fixpipe pipes are C2V-only. Give V2C its own identity and size,
    // preserving both directions' original transport depth and lifetime checks.
    auto c2v = combined;
    c2v.v2c = {};
    auto v2c = combined;
    v2c.c2v = {};
    auto setup = BuildPipeSetup(func_name, aic_name, aiv_name, aic_stmts, aiv_stmts, slot_num_override,
                                local_slot_num_override, span, c2v, 0);
    auto other = BuildPipeSetup(func_name, aic_name, aiv_name, aic_stmts, aiv_stmts, slot_num_override,
                                local_slot_num_override, span, v2c, 1);
    setup.aic_stmts.insert(setup.aic_stmts.end(), other.aic_stmts.begin(), other.aic_stmts.end());
    setup.aiv_stmts.insert(setup.aiv_stmts.end(), other.aiv_stmts.begin(), other.aiv_stmts.end());
    return setup;
  }
  return BuildPipeSetup(func_name, aic_name, aiv_name, aic_stmts, aiv_stmts, slot_num_override,
                        local_slot_num_override, span, combined, 0);
}

std::vector<StmtPtr> PrependPipeSetup(const std::vector<StmtPtr>& prologue,
                                      const std::vector<StmtPtr>& body) {
  if (prologue.empty()) return body;
  std::vector<StmtPtr> result;
  result.reserve(prologue.size() + body.size());
  result.insert(result.end(), prologue.begin(), prologue.end());
  result.insert(result.end(), body.begin(), body.end());
  return result;
}

std::string FormatObservedSlotSizes(const std::vector<int64_t>& slot_sizes) {
  std::string result;
  for (size_t i = 0; i < slot_sizes.size(); ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(slot_sizes[i]);
  }
  return result;
}

}  // namespace cross_core_pipe
}  // namespace ir
}  // namespace pypto
