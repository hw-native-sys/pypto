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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_CROSS_CORE_PIPE_H_
#define PYPTO_IR_TRANSFORMS_UTILS_CROSS_CORE_PIPE_H_

#include <any>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace cross_core_pipe {

struct PipeDirectionMetadata {
  bool has_ops = false;
  bool has_inconsistent_slot_size = false;
  std::optional<int64_t> slot_size_bytes;
  std::vector<int64_t> observed_slot_sizes;
};

struct CrossCorePipeMetadata {
  PipeDirectionMetadata c2v;
  PipeDirectionMetadata v2c;
  bool has_reserve_buffer = false;
  bool has_import_peer_buffer = false;
  bool has_aic_initialize_pipe = false;
  bool has_aiv_initialize_pipe = false;

  [[nodiscard]] bool HasCrossCoreOps() const { return c2v.has_ops || v2c.has_ops; }
  [[nodiscard]] bool HasAnySetup() const {
    return has_reserve_buffer || has_import_peer_buffer || has_aic_initialize_pipe || has_aiv_initialize_pipe;
  }
};

struct AutomaticPipeSetup {
  std::vector<StmtPtr> aic_stmts;
  std::vector<StmtPtr> aiv_stmts;
};

constexpr int kAutoBufferBase = -1;

/// Ring depth an automatically built cross-core pipe gets when the enclosing scope carries no
/// `pl.cross_core_slot(slot_num=N)`. Deep enough to double-buffer the cube<->vector handoff while
/// keeping the consumer-side ring (L1 for V2C, UB for C2V) affordable — the full-tile slots make a
/// deeper default overflow on-chip memory for most real tile shapes. `BuildAutomaticPipeSetup`
/// always emits this value explicitly on `initialize_pipe`, so it is a pypto policy number and is
/// deliberately independent of PTOAS's own fallback below.
constexpr int kDefaultAutoPipeSlotNum = 2;

/// Compile-time integer value of @p expr, or ``nullopt`` when it is not one
/// **or is negative**. The non-negative half of the contract is what the name
/// carries: callers here read tile shape dimensions and slot sizes, where a
/// negative is nonsense rather than a value to propagate. For the plain
/// signed reading, use ``transform_utils::EvalConstInt``, which this wraps.
std::optional<int64_t> TryGetNonNegativeConstInt(const ExprPtr& expr);
std::optional<int64_t> TryGetTileSlotSizeBytes(const TypePtr& type);
void RecordObservedSlotSize(PipeDirectionMetadata& metadata, int64_t slot_size);
void RecordTileSlotSize(PipeDirectionMetadata& metadata, const TypePtr& type);
void MergeDirectionMetadata(PipeDirectionMetadata& dst, const PipeDirectionMetadata& src);
CrossCorePipeMetadata MergeCrossCorePipeMetadata(const CrossCorePipeMetadata& lhs,
                                                 const CrossCorePipeMetadata& rhs);
int BuildDirMask(const CrossCorePipeMetadata& metadata);
/// Ring depth PTOAS derives itself when `initialize_pipe` carries no `slot_num` clause
/// (`slotNum = dirMask == 3 ? 4 : 8`). This mirrors PTOAS and must not be retuned as a policy knob:
/// only hand-written `pl.system.{aic,aiv}_initialize_pipe` still reaches that path, since automatic
/// pipes always emit an explicit `slot_num`.
int GetPtoasImplicitSlotNum(int dir_mask);
std::optional<int64_t> GetCommonSlotSizeBytes(const CrossCorePipeMetadata& metadata);
std::string BuildPipeBufferName(const std::string& func_name, core_affinity::PipeDirection direction);

CallPtr CreateSystemOpCall(const std::string& op_name,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span);
CallPtr CreateSystemOpCall(const std::string& op_name, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span);
CallPtr CreateReserveBuffer(const std::string& buffer_name, int64_t size_bytes, const Span& span);
CallPtr CreateImportPeerBuffer(const std::string& buffer_name, const std::string& peer_func,
                               const Span& span);
// `slot_num` is the ring depth emitted on the initialize_pipe op. It is always emitted, so PTOAS
// never falls back to its own `dir_mask` derivation for a pipe pypto built.
CallPtr CreateInitializePipe(core_affinity::CoreSide side, int dir_mask, int slot_size_bytes,
                             const ExprPtr& c2v_consumer_buf, const ExprPtr& v2c_consumer_buf, int slot_num,
                             const Span& span);

void CollectCrossCorePipeMetadata(const std::vector<StmtPtr>& stmts, CrossCorePipeMetadata& metadata);
CrossCorePipeMetadata CollectDominatingPipeSetupMetadata(const std::vector<StmtPtr>& stmts);

// `slot_num_override` (from pl.cross_core_slot(slot_num=N)) overrides the ring depth used to size
// the reserved buffer and the emitted initialize_pipe `slot_num` attribute. nullopt selects
// `kDefaultAutoPipeSlotNum`; either way the attribute is emitted.
AutomaticPipeSetup BuildAutomaticPipeSetup(const std::string& func_name, const std::string& aic_name,
                                           const std::string& aiv_name, const std::vector<StmtPtr>& aic_stmts,
                                           const std::vector<StmtPtr>& aiv_stmts,
                                           std::optional<int> slot_num_override, const Span& span);

std::vector<StmtPtr> PrependPipeSetup(const std::vector<StmtPtr>& prologue, const std::vector<StmtPtr>& body);

std::string FormatObservedSlotSizes(const std::vector<int64_t>& slot_sizes);

}  // namespace cross_core_pipe
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_CROSS_CORE_PIPE_H_
