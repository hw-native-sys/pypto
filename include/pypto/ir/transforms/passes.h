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

#ifndef PYPTO_IR_TRANSFORMS_PASSES_H_
#define PYPTO_IR_TRANSFORMS_PASSES_H_

#include <functional>
#include <memory>
#include <string>

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

/**
 * @brief Internal base class for pass implementations
 *
 * This is an internal class used for implementing passes via pimpl pattern.
 * Concrete passes should inherit from this class in their source files.
 */
class PassImpl {
 public:
  virtual ~PassImpl() = default;

  /**
   * @brief Execute the pass on a program
   *
   * @param program Input program to transform
   * @return Transformed program
   */
  virtual ProgramPtr operator()(const ProgramPtr& program) = 0;

  /**
   * @brief Get the name of the pass (for debugging)
   */
  [[nodiscard]] virtual std::string GetName() const { return "UnnamedPass"; }
};

/**
 * @brief Base class for IR transformation passes
 *
 * Pass is a standalone class (not inheriting from IRMutator) that provides transformations
 * on Program level. Each pass operates on entire Programs, returning transformed IR.
 * Passes maintain immutability - they return new IR instances rather than modifying in place.
 *
 * The Pass class uses a pimpl pattern to hide implementation details.
 * Users should create passes using factory functions (CreateIdentity, CreateInitMemRef, etc.)
 * rather than instantiating Pass directly.
 */
class Pass {
 public:
  Pass();
  explicit Pass(std::shared_ptr<PassImpl> impl);
  ~Pass();

  // Copy and move constructors/assignment
  Pass(const Pass& other);
  Pass& operator=(const Pass& other);
  Pass(Pass&& other) noexcept;
  Pass& operator=(Pass&& other) noexcept;

  /**
   * @brief Execute the pass on a program (primary API)
   *
   * This is the main entry point for pass execution using function call operator.
   *
   * @param program Input program to transform
   * @return Transformed program (may be the same pointer if no changes were made)
   */
  ProgramPtr operator()(const ProgramPtr& program) const;

  /**
   * @brief Execute the pass on a program (backward compatible API)
   *
   * This method provides backward compatibility with existing code.
   * It delegates to operator().
   *
   * @param program Input program to transform
   * @return Transformed program
   */
  [[nodiscard]] ProgramPtr run(const ProgramPtr& program) const;

 private:
  std::shared_ptr<PassImpl> impl_;
};

// Factory functions for built-in passes
namespace pass {

// Utility functions for creating custom passes

/**
 * @brief Create a pass from a program-level transform function
 *
 * @param transform Function that transforms a Program
 * @param name Optional name for the pass (for debugging)
 * @return Pass that applies the transform
 */
Pass CreateProgramPass(std::function<ProgramPtr(const ProgramPtr&)> transform, const std::string& name = "");

/**
 * @brief Create a pass from a function-level transform function
 *
 * The returned pass applies the function transform to each function in the program.
 *
 * @param transform Function that transforms a Function
 * @param name Optional name for the pass (for debugging)
 * @return Pass that applies the transform to each function
 */
Pass CreateFunctionPass(std::function<FunctionPtr(const FunctionPtr&)> transform,
                        const std::string& name = "");

/**
 * @brief Create an identity pass for testing
 *
 * Appends "_identity" to function names to verify pass execution.
 */
Pass Identity();

/**
 * @brief Create an init memref pass
 *
 * Initializes MemRef for all variables in functions.
 * Sets memory space to UB by default, or DDR for block.load/block.store operands.
 */
Pass InitMemRef();

/**
 * @brief Create a basic memory reuse pass
 *
 * Uses dependency analysis to identify memory reuse opportunities.
 * Variables with non-overlapping lifetimes in the same memory space can share MemRef objects.
 */
Pass BasicMemoryReuse();

/**
 * @brief Create an insert sync pass
 *
 * Analyzes data dependencies and inserts synchronization operations
 * (sync_src, sync_dst, bar_v, bar_m) for correct execution across hardware pipes.
 */
Pass InsertSync();

/**
 * @brief Create an add alloc pass
 *
 * This pass traverses all TileType variables in each Function and creates alloc operations
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
 *
 * @return Pass that adds alloc operations
 */
Pass AddAlloc();

}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASSES_H_
