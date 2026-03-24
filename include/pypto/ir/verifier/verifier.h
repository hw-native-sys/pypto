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

#ifndef PYPTO_IR_VERIFIER_VERIFIER_H_
#define PYPTO_IR_VERIFIER_VERIFIER_H_

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/program.h"

namespace pypto {
namespace ir {

/**
 * @brief Base class for IR property verifiers
 *
 * Each verifier implements a specific check on IR programs.
 * Verifiers can detect errors or warnings and add them to a diagnostics vector.
 * Each verifier receives a ProgramPtr and internally decides whether to iterate
 * over functions or check program-level properties.
 *
 * To create a new property verifier:
 * 1. Inherit from PropertyVerifier
 * 2. Implement GetName() to return a unique name
 * 3. Implement Verify() to perform the verification logic
 *
 * Example:
 * @code
 *   class MyVerifier : public PropertyVerifier {
 *    public:
 *     std::string GetName() const override { return "MyVerifier"; }
 *     void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
 *       for (const auto& [gv, func] : program->functions_) {
 *         // Verification logic per function
 *       }
 *     }
 *   };
 * @endcode
 */
class PropertyVerifier {
 public:
  virtual ~PropertyVerifier() = default;

  /**
   * @brief Get the name of this verifier
   * @return Unique name (e.g., "SSAVerify", "TypeCheck")
   */
  [[nodiscard]] virtual std::string GetName() const = 0;

  /**
   * @brief Verify a program and collect diagnostics
   * @param program Program to verify
   * @param diagnostics Vector to append diagnostics to
   *
   * This method should examine the program and add any detected issues
   * to the diagnostics vector. It should not throw exceptions - all issues
   * should be reported through diagnostics.
   */
  virtual void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) = 0;
};

/// Shared pointer to a property verifier
using PropertyVerifierPtr = std::shared_ptr<PropertyVerifier>;

/**
 * @brief Factory function for creating SSA property verifier
 * @return Shared pointer to SSA PropertyVerifier
 */
PropertyVerifierPtr CreateSSAPropertyVerifier();

/**
 * @brief Factory function for creating type check property verifier
 * @return Shared pointer to TypeCheck PropertyVerifier
 */
PropertyVerifierPtr CreateTypeCheckPropertyVerifier();

/**
 * @brief Factory function for creating no nested call property verifier
 * @return Shared pointer to NoNestedCall PropertyVerifier
 */
PropertyVerifierPtr CreateNoNestedCallPropertyVerifier();

/**
 * @brief Factory function for creating NormalizedStmtStructure property verifier
 * @return Shared pointer to NormalizedStmtStructure PropertyVerifier
 */
PropertyVerifierPtr CreateNormalizedStmtPropertyVerifier();

/**
 * @brief Factory function for creating NoRedundantBlocks property verifier
 *
 * Verifies that no SeqStmts has exactly one child (should be unwrapped),
 * and no SeqStmts contains a nested SeqStmts (should be flattened).
 * @return Shared pointer to NoRedundantBlocks PropertyVerifier
 */
PropertyVerifierPtr CreateNoRedundantBlocksPropertyVerifier();

/**
 * @brief Factory function for creating SplitIncoreOrch property verifier
 * @return Shared pointer to SplitIncoreOrch PropertyVerifier
 */
PropertyVerifierPtr CreateSplitIncoreOrchPropertyVerifier();

/**
 * @brief Factory function for creating ClusterOutlined property verifier
 * @return Shared pointer to ClusterOutlined PropertyVerifier
 */
PropertyVerifierPtr CreateClusterOutlinedPropertyVerifier();

/**
 * @brief Factory function for creating HierarchyOutlined property verifier
 * @return Shared pointer to HierarchyOutlined PropertyVerifier
 */
PropertyVerifierPtr CreateHierarchyOutlinedPropertyVerifier();

/**
 * @brief Factory function for creating HasMemRefs property verifier
 * @return Shared pointer to HasMemRefs PropertyVerifier
 */
PropertyVerifierPtr CreateHasMemRefsPropertyVerifier();

/**
 * @brief Factory function for creating IncoreTileOps property verifier
 * @return Shared pointer to IncoreTileOps PropertyVerifier
 */
PropertyVerifierPtr CreateIncoreTileOpsPropertyVerifier();

/**
 * @brief Factory function for creating MixedKernelExpanded property verifier
 *
 * Verifies that no InCore function contains both Cube and Vector tile ops, and
 * that split AIC/AIV functions keep cross-core tpop results in their required
 * bridge memory spaces.
 * @return Shared pointer to MixedKernelExpanded PropertyVerifier
 */
PropertyVerifierPtr CreateMixedKernelExpandedPropertyVerifier();

/**
 * @brief Factory function for creating AllocatedMemoryAddr property verifier
 *
 * Verifies that all non-DDR MemRefs have valid allocated addresses and
 * that total memory usage per space does not exceed platform buffer limits.
 * @return Shared pointer to AllocatedMemoryAddr PropertyVerifier
 */
PropertyVerifierPtr CreateAllocatedMemoryAddrPropertyVerifier();

/**
 * @brief Factory function for creating TileOps2D property verifier
 *
 * Verifies that all tile op calls (excluding tile.load, tile.store,
 * tile.reshape) in InCore functions operate on ≤2D tiles.
 * @return Shared pointer to TileOps2D PropertyVerifier
 */
PropertyVerifierPtr CreateTileOps2DPropertyVerifier();

/**
 * @brief Factory function for creating BreakContinueCheck property verifier
 *
 * Verifies that break/continue statements only appear inside sequential
 * (ForKind::Sequential) or while loops. Reports errors for break/continue
 * in parallel or unrolled loops, or outside any loop.
 * @return Shared pointer to BreakContinueCheck PropertyVerifier
 */
PropertyVerifierPtr CreateBreakContinuePropertyVerifier();

/**
 * @brief Factory function for creating TileMemoryInferred property verifier
 *
 * Verifies that all TileType variables in InCore functions have
 * memory_space_ set (not nullopt).
 * @return Shared pointer to TileMemoryInferred PropertyVerifier
 */
PropertyVerifierPtr CreateTileMemoryInferredPropertyVerifier();

/**
 * @brief Factory function for creating UseAfterDef property verifier
 *
 * Verifies that every Var reference in an expression is dominated by a
 * definition (function parameter, ForStmt/WhileStmt loop variable,
 * iter_arg, return_var, or AssignStmt).
 * @return Shared pointer to UseAfterDef PropertyVerifier
 */
PropertyVerifierPtr CreateUseAfterDefPropertyVerifier();

/**
 * @brief Factory function for creating StructuredCtrlFlow property verifier
 *
 * Verifies that no BreakStmt or ContinueStmt remains in InCore-type function
 * bodies (InCore, AIC, AIV). Host and Orchestration functions are skipped
 * because they support break/continue natively.
 * @return Shared pointer to StructuredCtrlFlow PropertyVerifier
 */
PropertyVerifierPtr CreateStructuredCtrlFlowPropertyVerifier();

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_VERIFIER_VERIFIER_H_
