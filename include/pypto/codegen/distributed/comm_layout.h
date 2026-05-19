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

#ifndef PYPTO_CODEGEN_DISTRIBUTED_COMM_LAYOUT_H_
#define PYPTO_CODEGEN_DISTRIBUTED_COMM_LAYOUT_H_

#include <cstddef>
#include <cstdint>

#include "platform_comm/comm_context.h"  // runtime submodule (runtime/src/common)

namespace pypto {
namespace codegen {
namespace distributed {
namespace comm_layout {

// Field offsets and size of the runtime CommContext struct. The values come
// from `offsetof(::CommContext, ...)` on the runtime header, then the
// static_asserts below pin them to the literal numbers that distributed
// codegen embeds into the emitted CommRemotePtr() inline function.
//
// If the runtime CommContext layout ever shifts (field reorder, insert, type
// change), pypto's `cmake --build` fails here — codegen would otherwise
// silently emit kernels that index the wrong byte and garble cross-rank DMA.
// Treat any failure as a deliberate ABI bump: re-verify the runtime/codegen
// pair end-to-end before updating the literals.

inline constexpr std::size_t kRankIdOffset = offsetof(::CommContext, rankId);
inline constexpr std::size_t kRankNumOffset = offsetof(::CommContext, rankNum);
inline constexpr std::size_t kWindowsInOffset = offsetof(::CommContext, windowsIn);
inline constexpr std::size_t kWindowsOutOffset = offsetof(::CommContext, windowsOut);
inline constexpr std::size_t kWindowSlotStride = sizeof(std::uint64_t);
inline constexpr std::size_t kCommCtxSize = sizeof(::CommContext);

static_assert(kRankIdOffset == 16, "CommContext.rankId offset drift");
static_assert(kRankNumOffset == 20, "CommContext.rankNum offset drift");
static_assert(kWindowsInOffset == 32, "CommContext.windowsIn offset drift");
static_assert(kWindowsOutOffset == 544, "CommContext.windowsOut offset drift");
static_assert(kWindowSlotStride == 8, "CommContext window slot stride drift");
static_assert(kCommCtxSize == 1056, "CommContext size drift");

}  // namespace comm_layout
}  // namespace distributed
}  // namespace codegen
}  // namespace pypto

#endif  // PYPTO_CODEGEN_DISTRIBUTED_COMM_LAYOUT_H_
