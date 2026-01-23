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

#ifndef PYPTO_IR_TRANSFORM_INSERT_SYNC_PASS_H_
#define PYPTO_IR_TRANSFORM_INSERT_SYNC_PASS_H_

#include <memory>

#include "pypto/ir/function.h"
#include "pypto/ir/transform/base/pass.h"

namespace pypto {
namespace ir {

/**
 * @brief Pass for inserting synchronization operations (sync_src, sync_dst, bars)
 *
 * This pass analyzes data dependencies between operations based on MemRef.
 * It inserts:
 * - sync_src/sync_dst pairs for cross-pipe dependencies
 * - bar_v/bar_m for intra-pipe dependencies in Vector/Cube units
 */
class InsertSyncPass : public Pass {
 public:
  InsertSyncPass() = default;
  ~InsertSyncPass() override = default;

  /**
   * @brief Execute the insert sync pass
   *
   * @param func Input function
   * @return Function with synchronization operations inserted
   */
  FunctionPtr Run(const FunctionPtr& func) override;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_INSERT_SYNC_PASS_H_
