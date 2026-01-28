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
#include <utility>
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
  explicit AddAllocPass(bool addOp = true) : addOp_(addOp) {}

  [[nodiscard]] std::string Name() const { return "AddAllocPass"; }

  [[nodiscard]] FunctionPtr Run(const FunctionPtr& func) override;

 private:
  bool addOp_;  // Whether to add alloc operations
  /**
   * @brief Collect all unique MemRef objects from TileType variables in a statement
   *
   * @param stmt Statement to traverse
   * @param memrefs Vector to accumulate unique MemRef objects
   */
  void CollectMemRefsFromStatement(const StmtPtr& stmt, std::vector<MemRefPtr>& memrefs);

  /**
   * @brief Allocate memory addresses for non-DDR memory spaces
   *
   * Groups MemRefs by memory space and allocates non-overlapping 32-byte aligned addresses
   * for each non-DDR space (UB, L1, L0A, L0B, L0C). DDR MemRefs keep their original addresses.
   * Creates new MemRef objects with updated addresses, sorted by address.
   *
   * @param memrefs Vector of all MemRef objects to allocate
   * @return Vector of (old MemRef, new MemRef) pairs sorted by allocated address
   */
  std::vector<std::pair<const MemRef*, MemRefPtr>> AllocateMemoryAddresses(
      const std::vector<MemRefPtr>& memrefs);

  /**
   * @brief Create block.alloc statements for MemRefs
   *
   * Creates an alloc operation for each MemRef in the provided ordered list.
   *
   * @param memref_pairs Vector of (old MemRef, new MemRef) pairs, already sorted by address
   * @return Vector of alloc statements
   */
  std::vector<StmtPtr> CreateAllocStatements(
      const std::vector<std::pair<const MemRef*, MemRefPtr>>& memref_pairs);

  /**
   * @brief Prepend alloc statements to function body
   *
   * Adds alloc statements at the beginning of the function body, handling both SeqStmts
   * and single statement cases.
   *
   * @param body Original function body
   * @param alloc_stmts Vector of alloc statements to prepend
   * @return New function body with alloc statements prepended
   */
  StmtPtr PrependAllocStatements(const StmtPtr& body, const std::vector<StmtPtr>& alloc_stmts);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORM_ADD_ALLOC_PASS_H_
