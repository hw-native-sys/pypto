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

#ifndef PYPTO_CODEGEN_GM_PIPE_LAYOUT_H_
#define PYPTO_CODEGEN_GM_PIPE_LAYOUT_H_

#include <cstdint>

#include "pypto/ir/transforms/utils/core_affinity.h"

namespace pypto {
namespace codegen {
namespace gm_pipe {

/// Layout rules for the GM ring buffers backing a cross-core (cube<->vector) pipe.
///
/// Two places must agree on these numbers: the orchestration analysis that sizes the injected
/// `__gm_pipe_buffer` workspace, and the PTO codegen that assigns each frontend pipe its byte
/// offset *within* that workspace. When they disagree, either the allocation is too small or two
/// independent pipes overlap — both silent, both corrupting. Keep the rules here, once.

/// Slots per ring, used when `initialize_pipe` carries no explicit `slot_num`. Mirrors PTOAS's own
/// derivation (`cross_core_pipe::GetPtoasImplicitSlotNum`), so it applies only to hand-written
/// `pl.system.{aic,aiv}_initialize_pipe` — automatic pipes always carry an explicit `slot_num`
/// (`cross_core_pipe::kDefaultAutoPipeSlotNum` by default), which takes precedence below.
/// Returns 0 for a dir_mask that does not describe a GM-backed pipe.
inline int SlotCountForDirMask(int dir_mask) {
  const int bidirectional = ir::core_affinity::kDirMaskC2V | ir::core_affinity::kDirMaskV2C;
  if (dir_mask == bidirectional) {
    return 4;
  }
  if (dir_mask == ir::core_affinity::kDirMaskC2V || dir_mask == ir::core_affinity::kDirMaskV2C) {
    return 8;
  }
  return 0;
}

/// Independent rings in the GM buffer. A bidirectional pipe is TWO rings laid out back to back:
/// on a2a3 the C2V ring starts at the pipe's base and the V2C ring at base + slot_num * slot_size
/// (pto-isa, `TPipe`). Returns 0 for a dir_mask that does not describe a GM-backed pipe.
inline int RingCountForDirMask(int dir_mask) {
  const int bidirectional = ir::core_affinity::kDirMaskC2V | ir::core_affinity::kDirMaskV2C;
  if (dir_mask == bidirectional) {
    return 2;
  }
  if (dir_mask == ir::core_affinity::kDirMaskC2V || dir_mask == ir::core_affinity::kDirMaskV2C) {
    return 1;
  }
  return 0;
}

/// Slots per ring for a pipe, honouring an explicit `slot_num` (<= 0 means "not specified").
/// Returns 0 when `dir_mask` is not a GM-backed pipe, whatever `slot_num` says — an explicit
/// slot count must not paper over a direction mask we cannot lay out.
inline int EffectiveSlotCount(int dir_mask, int slot_num) {
  const int dir_slots = SlotCountForDirMask(dir_mask);
  if (dir_slots <= 0) {
    return 0;
  }
  return slot_num > 0 ? slot_num : dir_slots;
}

/// Total bytes one pipe occupies in the GM workspace: rings * slots * slot_size.
/// Returns 0 for a dir_mask that does not describe a GM-backed pipe.
inline int64_t FootprintBytes(int dir_mask, int slot_num, int slot_size) {
  const int64_t slots = EffectiveSlotCount(dir_mask, slot_num);
  const int64_t rings = RingCountForDirMask(dir_mask);
  return slots * rings * static_cast<int64_t>(slot_size);
}

}  // namespace gm_pipe
}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_GM_PIPE_LAYOUT_H_
