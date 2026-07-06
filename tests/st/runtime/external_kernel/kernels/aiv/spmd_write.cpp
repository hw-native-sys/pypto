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
 * SPMD Multi-Block Write Kernel (AIV) — hand-written test fixture.
 *
 * Used by test_external_kernel.py to exercise integrating an existing C++
 * kernel into a PyPTO program via @pl.function(external_source=...). Each SPMD
 * block writes float(block_idx) at a cacheline-aligned offset:
 *
 *   out[(base_cl + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)
 *
 * Args:
 *   args[0] = output Tensor* (INOUT)
 *   args[1] = scalar: base_cl (starting cache line index for this task)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#include "intrinsic.h"

static constexpr int32_t FLOATS_PER_CACHE_LINE = 16;

#ifdef PTO_CPUSTUB_HPP
#define dcci(...) \
    do {          \
    } while (0)
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    int32_t base_cl = static_cast<int32_t>(args[1]);
    int32_t block_idx = get_block_idx(args);
    int32_t offset = (base_cl + block_idx) * FLOATS_PER_CACHE_LINE;

    out[offset] = static_cast<float>(block_idx);

    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
