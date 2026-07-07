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

// Shared cacheline constant for the SPMD write kernel. It lives in a sibling
// directory so the entry .cpp includes it via a relative "../common/..." path
// — exercising PyPTO's multi-file external-kernel support (the entry file is
// referenced at its original location, so relative includes resolve). A plain
// constant (not a function) keeps it usable from AICore code on both the CCEC
// device toolchain and the host simulator.

#ifndef TESTS_ST_RUNTIME_EXTERNAL_KERNEL_KERNELS_COMMON_CACHELINE_OFFSET_H_
#define TESTS_ST_RUNTIME_EXTERNAL_KERNEL_KERNELS_COMMON_CACHELINE_OFFSET_H_

#include <cstdint>

static constexpr int32_t FLOATS_PER_CACHE_LINE = 16;

#endif  // TESTS_ST_RUNTIME_EXTERNAL_KERNEL_KERNELS_COMMON_CACHELINE_OFFSET_H_
