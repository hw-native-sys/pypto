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
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"

namespace pypto {
namespace ir {

/**
 * @brief Internal base class for pass implementations
 *
 * Most passes should use CreateFunctionPass() or CreateProgramPass() helpers.
 * Only inherit from PassImpl for complex passes with custom state.
 */
class PassImpl {
 public:
  virtual ~PassImpl() = default;

  /**
   * @brief Execute the pass on a program
   */
  virtual ProgramPtr operator()(const ProgramPtr& program) = 0;

  /**
   * @brief Get the name of the pass (for debugging)
   */
  [[nodiscard]] virtual std::string GetName() const { return "UnnamedPass"; }

  /**
   * @brief Get properties required before this pass can run
   */
  [[nodiscard]] virtual IRPropertySet GetRequiredProperties() const { return {}; }

  /**
   * @brief Get properties produced (guaranteed) after this pass runs
   */
  [[nodiscard]] virtual IRPropertySet GetProducedProperties() const { return {}; }

  /**
   * @brief Get properties invalidated (broken) by this pass
   */
  [[nodiscard]] virtual IRPropertySet GetInvalidatedProperties() const { return {}; }
};

/**
 * @brief Base class for IR transformation passes
 *
 * Pass uses a pimpl pattern to hide implementation details.
 * Users should create passes using factory functions.
 */
class Pass {
 public:
  Pass();
  explicit Pass(std::shared_ptr<PassImpl> impl);
  ~Pass();

  // Copy and move
  Pass(const Pass& other);
  Pass& operator=(const Pass& other);
  Pass(Pass&& other) noexcept;
  Pass& operator=(Pass&& other) noexcept;

  /**
   * @brief Execute the pass on a program (primary API)
   */
  ProgramPtr operator()(const ProgramPtr& program) const;

  /**
   * @brief Execute the pass on a program (backward compatible API)
   */
  [[nodiscard]] ProgramPtr run(const ProgramPtr& program) const;

  /**
   * @brief Get the name of the pass
   */
  [[nodiscard]] std::string GetName() const;

  /**
   * @brief Get properties required before this pass can run
   */
  [[nodiscard]] IRPropertySet GetRequiredProperties() const;

  /**
   * @brief Get properties produced (guaranteed) after this pass runs
   */
  [[nodiscard]] IRPropertySet GetProducedProperties() const;

  /**
   * @brief Get properties invalidated (broken) by this pass
   */
  [[nodiscard]] IRPropertySet GetInvalidatedProperties() const;

 private:
  std::shared_ptr<PassImpl> impl_;
};

// Factory functions for built-in passes
namespace pass {

/**
 * @brief Create a pass from a function-level transform function (RECOMMENDED)
 *
 * @param transform Function that transforms a Function
 * @param name Optional name for the pass (for debugging)
 * @param properties Optional property declarations
 * @return Pass that applies the transform to each function
 */
Pass CreateFunctionPass(std::function<FunctionPtr(const FunctionPtr&)> transform,
                        const std::string& name = "", const PassProperties& properties = {});

/**
 * @brief Create a pass from a program-level transform function
 *
 * @param transform Function that transforms a Program
 * @param name Optional name for the pass (for debugging)
 * @param properties Optional property declarations
 * @return Pass that applies the transform
 */
Pass CreateProgramPass(std::function<ProgramPtr(const ProgramPtr&)> transform, const std::string& name = "",
                       const PassProperties& properties = {});

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
 * Uses the globally configured backend to obtain pipe info.
 */
Pass InsertSync();

/**
 * @brief Create an add alloc pass
 *
 * Traverses all TileType variables and creates alloc operations for each unique MemRef.
 * The alloc operations are added at the beginning of the function.
 */
Pass AddAlloc();

/**
 * @brief Create an SSA verification pass
 */
Pass VerifySSA();

/**
 * @brief Create a type checking pass
 */
Pass TypeCheck();

/**
 * @brief Create an SSA conversion pass
 */
Pass ConvertToSSA();

/**
 * @brief Outline InCore scopes into separate functions
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions
 */
Pass OutlineIncoreScopes();

/**
 * @brief Create a verifier pass with configurable rules
 *
 * @param disabled_rules Vector of rule names to disable
 * @return Pass that runs IR verification
 */
Pass RunVerifier(const std::vector<std::string>& disabled_rules = {});

/**
 * @brief Create a pass that flattens nested call expressions
 */
Pass FlattenCallExpr();

/**
 * @brief Create a pass that normalizes statement structure
 */
Pass NormalizeStmtStructure();

/**
 * @brief Create a pass that recursively flattens single-statement blocks
 */
Pass FlattenSingleStmt();

}  // namespace pass

/**
 * @brief Controls when property verification runs in a PassPipeline
 */
enum class VerificationMode {
  None,           ///< No automatic verification
  Before,         ///< Verify required properties before each pass
  After,          ///< Verify produced properties after each pass
  BeforeAndAfter  ///< Verify both before and after each pass
};

/**
 * @brief A pipeline of passes with property tracking and verification
 *
 * PassPipeline maintains a sequence of passes and tracks IR properties
 * as passes are executed. It validates that pass requirements are met
 * and optionally runs property verifiers before/after each pass.
 *
 * Usage:
 * @code
 *   PassPipeline pipeline;
 *   pipeline.AddPass(pass::ConvertToSSA());
 *   pipeline.AddPass(pass::FlattenCallExpr());
 *   pipeline.AddPass(pass::RunVerifier());
 *
 *   // Static validation (no execution)
 *   auto errors = pipeline.Validate();
 *
 *   // Execute with property tracking
 *   auto result = pipeline.Run(program);
 * @endcode
 */
class PassPipeline {
 public:
  PassPipeline();

  /**
   * @brief Add a pass to the pipeline
   */
  void AddPass(Pass pass);

  /**
   * @brief Set verification mode
   */
  void SetVerificationMode(VerificationMode mode);

  /**
   * @brief Set initial properties (properties known to hold before the pipeline runs)
   */
  void SetInitialProperties(const IRPropertySet& properties);

  /**
   * @brief Execute all passes with property tracking
   * @param program Input program
   * @return Transformed program
   * @throws ValueError if a pass's required properties are not satisfied
   */
  [[nodiscard]] ProgramPtr Run(const ProgramPtr& program) const;

  /**
   * @brief Static validation: check property flow without executing passes
   * @return List of error messages (empty if valid)
   */
  [[nodiscard]] std::vector<std::string> Validate() const;

  /**
   * @brief Get the names of all passes in the pipeline
   */
  [[nodiscard]] std::vector<std::string> GetPassNames() const;

 private:
  std::vector<Pass> passes_;
  VerificationMode verification_mode_;
  IRPropertySet initial_properties_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASSES_H_
