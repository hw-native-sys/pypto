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

// Standalone launch envelope for ACL hardware or the simulator; not the PyPTO scheduler ABI.
#include <stdint.h>

#include "acl/acl.h"

extern "C" __global__[aicore] void pypto_vf_launch(__gm__ int64_t*);

extern "C" void launch_pypto_vf(int64_t* args, void* stream) {
  pypto_vf_launch<<<1, nullptr, stream>>>(args);
}
