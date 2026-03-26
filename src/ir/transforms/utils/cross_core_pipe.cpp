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

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace cross_core_pipe {

namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

}  // namespace

std::optional<int64_t> TryGetConstIntValue(const ExprPtr& expr) {
  auto const_int = std::dynamic_pointer_cast<const ConstInt>(expr);
  if (!const_int || const_int->value_ < 0) return std::nullopt;
  return const_int->value_;
}

std::optional<int64_t> TryGetTileSlotSizeBytes(const TypePtr& type) {
  auto tile_type = std::dynamic_pointer_cast<const TileType>(type);
  if (!tile_type) return std::nullopt;

  int64_t element_count = 1;
  for (const auto& dim : tile_type->shape_) {
    auto dim_value = TryGetConstIntValue(dim);
    if (!dim_value.has_value()) return std::nullopt;
    CHECK(*dim_value == 0 || element_count <= std::numeric_limits<int64_t>::max() / *dim_value)
        << "Tile element count overflow while inferring cross-core slot size";
    element_count *= *dim_value;
  }

  const int64_t bit_width = static_cast<int64_t>(tile_type->dtype_.GetBit());
  CHECK(bit_width > 0) << "Unsupported dtype for cross-core slot size inference: "
                       << tile_type->dtype_.ToString();
  CHECK(element_count <= (std::numeric_limits<int64_t>::max() - 7) / bit_width)
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

int GetSlotNumForDirMask(int dir_mask) {
  return dir_mask == (core_affinity::kDirMaskC2V | core_affinity::kDirMaskV2C) ? 4 : 8;
}

std::optional<int64_t> GetCommonSlotSizeBytes(const CrossCorePipeMetadata& metadata) {
  std::optional<int64_t> common_slot_size;
  for (const auto* direction : {&metadata.c2v, &metadata.v2c}) {
    if (!direction->has_ops) continue;
    if (direction->has_inconsistent_slot_size || !direction->slot_size_bytes.has_value()) {
      return std::nullopt;
    }
    if (!common_slot_size.has_value()) {
      common_slot_size = direction->slot_size_bytes;
      continue;
    }
    if (common_slot_size.value() != direction->slot_size_bytes.value()) {
      return std::nullopt;
    }
  }
  return common_slot_size;
}

std::string BuildPipeBufferName(const std::string& func_name, core_affinity::PipeDirection direction) {
  return func_name +
         ((direction == core_affinity::PipeDirection::C2V) ? "_c2v_slot_buffer" : "_v2c_slot_buffer");
}

CallPtr CreateSystemOpCall(const std::string& op_name,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span) {
  return OpRegistry::GetInstance().Create(op_name, {}, kwargs, span);
}

CallPtr CreateReserveBuffer(const std::string& buffer_name, int64_t size_bytes, const Span& span) {
  CHECK(size_bytes >= 0 && size_bytes <= std::numeric_limits<int>::max())
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
                             const Span& span) {
  CHECK(slot_size_bytes >= 0 && slot_size_bytes <= std::numeric_limits<int>::max())
      << "Cross-core slot_size out of range: " << slot_size_bytes;
  const std::string op_name =
      (side == core_affinity::CoreSide::AIC) ? "system.aic_initialize_pipe" : "system.aiv_initialize_pipe";
  return CreateSystemOpCall(
      op_name, {{"dir_mask", std::any(dir_mask)}, {"slot_size", std::any(slot_size_bytes)}}, span);
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
      const std::string& op_name = op->name_;
      if (op_name == "system.reserve_buffer") {
        metadata.has_reserve_buffer = true;
      } else if (op_name == "system.import_peer_buffer") {
        metadata.has_import_peer_buffer = true;
      } else if (op_name == "system.aic_initialize_pipe") {
        metadata.has_aic_initialize_pipe = true;
      } else if (op_name == "system.aiv_initialize_pipe") {
        metadata.has_aiv_initialize_pipe = true;
      } else if (op_name == "tile.tpush_to_aiv" && call->args_.size() == 1) {
        RecordTileSlotSize(metadata.c2v, call->args_[0]->GetType());
      } else if (op_name == "tile.tpush_to_aic" && call->args_.size() == 1) {
        RecordTileSlotSize(metadata.v2c, call->args_[0]->GetType());
      } else if (op_name == "tile.tpop_from_aiv" && assign) {
        RecordTileSlotSize(metadata.v2c, assign->var_->GetType());
      } else if (op_name == "tile.tpop_from_aic" && assign) {
        RecordTileSlotSize(metadata.c2v, assign->var_->GetType());
      }
    }

    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(for_stmt->body_), metadata);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(if_stmt->then_body_), metadata);
      if (if_stmt->else_body_.has_value()) {
        CollectCrossCorePipeMetadata(FlattenBody(if_stmt->else_body_.value()), metadata);
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
    if (op) {
      const std::string& op_name = op->name_;
      if (op_name == "system.reserve_buffer") {
        metadata.has_reserve_buffer = true;
      } else if (op_name == "system.import_peer_buffer") {
        metadata.has_import_peer_buffer = true;
      } else if (op_name == "system.aic_initialize_pipe") {
        metadata.has_aic_initialize_pipe = true;
      } else if (op_name == "system.aiv_initialize_pipe") {
        metadata.has_aiv_initialize_pipe = true;
      }
    }

    CrossCorePipeMetadata stmt_metadata;
    CollectCrossCorePipeMetadata({stmt}, stmt_metadata);
    if (stmt_metadata.HasCrossCoreOps()) {
      break;
    }
  }
  return metadata;
}

AutomaticPipeSetup BuildAutomaticPipeSetup(const std::string& func_name, const std::string& aic_name,
                                           const std::string& aiv_name, const std::vector<StmtPtr>& aic_stmts,
                                           const std::vector<StmtPtr>& aiv_stmts, const Span& span) {
  CrossCorePipeMetadata aic_metadata;
  CollectCrossCorePipeMetadata(aic_stmts, aic_metadata);
  CrossCorePipeMetadata aiv_metadata;
  CollectCrossCorePipeMetadata(aiv_stmts, aiv_metadata);
  CrossCorePipeMetadata combined = MergeCrossCorePipeMetadata(aic_metadata, aiv_metadata);

  if (!combined.HasCrossCoreOps() || aic_metadata.HasAnySetup() || aiv_metadata.HasAnySetup()) {
    return {};
  }

  const int dir_mask = BuildDirMask(combined);
  auto common_slot_size = GetCommonSlotSizeBytes(combined);
  if (dir_mask == 0 || !common_slot_size.has_value()) {
    return {};
  }

  const int64_t buffer_size = common_slot_size.value() * GetSlotNumForDirMask(dir_mask);
  CHECK(common_slot_size.value() <= std::numeric_limits<int>::max())
      << "Cross-core slot_size out of range: " << common_slot_size.value();
  const int slot_size_bytes = static_cast<int>(common_slot_size.value());
  AutomaticPipeSetup setup;

  if (dir_mask & core_affinity::kDirMaskV2C) {
    setup.aic_stmts.push_back(std::make_shared<AssignStmt>(
        std::make_shared<Var>(BuildPipeBufferName(func_name, core_affinity::PipeDirection::V2C),
                              GetUnknownType(), span),
        CreateReserveBuffer(BuildPipeBufferName(func_name, core_affinity::PipeDirection::V2C), buffer_size,
                            span),
        span));
    setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(
        std::make_shared<Var>(BuildPipeBufferName(func_name, core_affinity::PipeDirection::V2C) + "_import",
                              GetUnknownType(), span),
        CreateImportPeerBuffer(BuildPipeBufferName(func_name, core_affinity::PipeDirection::V2C), aic_name,
                               span),
        span));
  }

  if (dir_mask & core_affinity::kDirMaskC2V) {
    setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(
        std::make_shared<Var>(BuildPipeBufferName(func_name, core_affinity::PipeDirection::C2V),
                              GetUnknownType(), span),
        CreateReserveBuffer(BuildPipeBufferName(func_name, core_affinity::PipeDirection::C2V), buffer_size,
                            span),
        span));
    setup.aic_stmts.push_back(std::make_shared<AssignStmt>(
        std::make_shared<Var>(BuildPipeBufferName(func_name, core_affinity::PipeDirection::C2V) + "_import",
                              GetUnknownType(), span),
        CreateImportPeerBuffer(BuildPipeBufferName(func_name, core_affinity::PipeDirection::C2V), aiv_name,
                               span),
        span));
  }

  setup.aic_stmts.push_back(std::make_shared<EvalStmt>(
      CreateInitializePipe(core_affinity::CoreSide::AIC, dir_mask, slot_size_bytes, span), span));
  setup.aiv_stmts.push_back(std::make_shared<EvalStmt>(
      CreateInitializePipe(core_affinity::CoreSide::AIV, dir_mask, slot_size_bytes, span), span));

  return setup;
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
