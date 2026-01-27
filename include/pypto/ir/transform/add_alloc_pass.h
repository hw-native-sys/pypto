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

#ifndef PYPTO_IR_TRANSFORM_ADD_ALLOC_PASS_H_
#define PYPTO_IR_TRANSFORM_ADD_ALLOC_PASS_H_

#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/memref.h"
#include "pypto/ir/transform/base/pass.h"

namespace pypto {
namespace ir {

/**
 * @brief Pass to add alloc operations for all MemRef objects in TileType variables
 *
 * This pass traverses all TileType variables in a Function and creates alloc operations
 * for each unique MemRef. The alloc operations are added at the beginning of the function.
 *
 * The pass:
 * 1. Identifies all TileType variables in the function
 * 2. Collects all unique MemRef objects from these TileType variables
 * 3. Creates an alloc operation for each unique MemRef
 * 4. Prepends these alloc operations to the function body
 *
 * Each alloc operation has no input/output arguments but is bound to a MemRef pointer
 * to track memory allocation for that specific buffer.
 */
class AddAllocPass : public Pass {
 public:
  AddAllocPass() = default;

  [[nodiscard]] std::string Name() const { return "AddAllocPass"; }

  [[nodiscard]] FunctionPtr Run(const FunctionPtr& func) override;

 private:
  /**
   * @brief Collect all unique MemRef objects from TileType variables in a statement
   *
   * @param stmt Statement to traverse
   * @param memrefs Vector to accumulate unique MemRef objects
   */
  void CollectMemRefsFromStatement(const StmtPtr& stmt, std::vector<MemRefPtr>& memrefs);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_ADD_ALLOC_PASS_H_
