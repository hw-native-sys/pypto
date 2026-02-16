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

namespace pypto {
namespace ir {

/**
 * @brief Internal base class for pass implementations
 *
 * This is an internal class used for implementing passes via pimpl pattern.
 *
 * Most passes should use CreateFunctionPass() or CreateProgramPass() helpers instead
 * of directly inheriting from this class. Only inherit from PassImpl for complex
 * passes that need:
 * - Custom state management
 * - Complex helper methods
 * - Program-level analysis (not just per-function transformations)
 *
 * For simple function-level transformations, use CreateFunctionPass() which
 * automatically handles the Program → Program transformation.
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
 * Users should create passes using factory functions (CreateInitMemRef, etc.)
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
//
// These helpers simplify pass creation by eliminating boilerplate code.
// Most passes should use these instead of inheriting from PassImpl.

/**
 * @brief Create a pass from a function-level transform function (RECOMMENDED)
 *
 * This is the recommended way to create passes that apply transformations to each
 * function independently. The helper automatically handles the Program → Program
 * transformation by applying your function to each function in the program.
 *
 * Example:
 *   Pass MyPass() {
 *     return CreateFunctionPass([](const FunctionPtr& func) {
 *       // Transform the function
 *       return transformed_func;
 *     }, "MyPass");
 *   }
 *
 * @param transform Function that transforms a Function
 * @param name Optional name for the pass (for debugging)
 * @return Pass that applies the transform to each function
 */
Pass CreateFunctionPass(std::function<FunctionPtr(const FunctionPtr&)> transform,
                        const std::string& name = "");

/**
 * @brief Create a pass from a program-level transform function
 *
 * Use this for passes that need to transform the entire program at once,
 * such as inter-procedural optimizations or whole-program analysis.
 * For most cases, prefer CreateFunctionPass() instead.
 *
 * @param transform Function that transforms a Program
 * @param name Optional name for the pass (for debugging)
 * @return Pass that applies the transform
 */
Pass CreateProgramPass(std::function<ProgramPtr(const ProgramPtr&)> transform, const std::string& name = "");

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

/**
 * @brief Create an SSA verification pass
 *
 * This pass verifies SSA form of IR by checking:
 * 1. Each variable is assigned only once (MULTIPLE_ASSIGNMENT)
 * 2. No variable name shadowing across scopes (NAME_SHADOWING)
 * 3. ForStmt with iter_args must have YieldStmt as last statement (MISSING_YIELD)
 * 4. IfStmt with return_vars must have YieldStmt in both then and else branches (MISSING_YIELD)
 *
 * The pass collects all errors and generates a verification report instead of
 * throwing exceptions, allowing detection of all issues in a single run.
 *
 * @return Pass that performs SSA verification
 */
Pass VerifySSA();

/**
 * @brief Create a type checking pass
 *
 * This pass checks type consistency in control flow constructs:
 * 1. ForStmt: iter_args initValue, yield values, and return_vars must have matching types
 * 2. IfStmt: then and else yield values must have matching types
 * 3. Shape consistency for TensorType and TileType
 *
 * The pass collects all errors and generates a type checking report instead of
 * throwing exceptions, allowing detection of all issues in a single run.
 *
 * @return Pass that performs type checking
 */
Pass TypeCheck();

/**
 * @brief Create an SSA conversion pass
 *
 * This pass converts non-SSA IR to SSA form by:
 * 1. Renaming variables with version suffixes (x -> x_0, x_1, x_2)
 * 2. Adding phi nodes (return_vars + YieldStmt) for IfStmt control flow divergence
 * 3. Converting loop-modified variables to iter_args + return_vars pattern
 *
 * The pass handles:
 * - Straight-line code: multiple assignments to the same variable
 * - If statements: variables modified in one or both branches
 * - For loops: variables modified inside the loop body
 * - Mixed SSA/non-SSA: preserves existing SSA structure while converting non-SSA parts
 *
 * @return Pass that converts to SSA form
 */
Pass ConvertToSSA();

/**
 * @brief Outline InCore scopes into separate functions
 *
 * This pass transforms ScopeStmt(InCore) nodes into separate Function(InCore) definitions
 * and replaces the scope with a Call to the outlined function.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Only processes Opaque functions (InCore functions are left unchanged)
 *
 * Transformation:
 * 1. For each ScopeStmt(InCore) in an Opaque function:
 *    - Analyze body to determine external variable references (inputs)
 *    - Analyze body to determine internal definitions used after scope (outputs)
 *    - Extract body into new Function(InCore) with appropriate params/returns
 *    - Replace scope with Call to the outlined function + output assignments
 * 2. Add outlined functions to the program
 *
 * @return Pass that outlines InCore scopes
 */
Pass OutlineIncoreScopes();

/**
 * @brief Create a verifier pass with configurable rules
 *
 * This pass creates an IRVerifier with default rules (SSAVerify, TypeCheck)
 * and allows disabling specific rules. The verifier collects all diagnostics
 * and logs them without throwing exceptions (unless there are errors).
 *
 * @param disabled_rules Vector of rule names to disable (e.g., {"TypeCheck"})
 * @return Pass that runs IR verification
 */
Pass RunVerifier(const std::vector<std::string>& disabled_rules = {});

/**
 * @brief Create a pass that flattens nested call expressions into three-address code
 *
 * This pass ensures that call expressions do not appear in nested contexts:
 * 1. Call arguments cannot be calls
 * 2. If conditions cannot be calls
 * 3. For loop ranges (start/stop/step) cannot be calls
 * 4. Binary/unary expression operands cannot be calls
 *
 * Nested calls are extracted into temporary variables (named _t0, _t1, etc.)
 * and inserted as AssignStmt before the statement containing the nested call.
 * For if/for statements, extracted statements are inserted into the last OpStmts
 * before the if/for, or a new OpStmts is created if needed.
 *
 * Example transformation:
 *   c = foo(bar(a))  =>  _t0 = bar(a); c = foo(_t0)
 *
 * @return Pass that flattens nested call expressions
 */
Pass FlattenCallExpr();

/**
 * @brief Create a pass that normalizes statement structure
 *
 * This pass ensures IR is in a normalized form:
 * 1. Function/IfStmt/ForStmt body must be SeqStmts
 * 2. Consecutive AssignStmt/EvalStmt in SeqStmts are wrapped in OpStmts
 *
 * Example transformations:
 *   Function body = AssignStmt(x, 1)
 *   => Function body = SeqStmts([OpStmts([AssignStmt(x, 1)])])
 *
 *   SeqStmts([AssignStmt(a, 1), AssignStmt(b, 2), IfStmt(...)])
 *   => SeqStmts([OpStmts([AssignStmt(a, 1), AssignStmt(b, 2)]), IfStmt(...)])
 *
 * @return Pass that normalizes statement structure
 */
Pass NormalizeStmtStructure();

/**
 * @brief Create a pass that recursively flattens single-statement blocks
 *
 * This pass simplifies IR by removing unnecessary nesting:
 * - SeqStmts with only one statement is replaced by that statement
 * - OpStmts with only one statement is replaced by that statement
 * - Process is applied recursively
 *
 * Example transformations:
 *   SeqStmts([OpStmts([AssignStmt(x, 1)])])
 *   => AssignStmt(x, 1)
 *
 *   SeqStmts([OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])])
 *   => OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])
 *
 * Note: This pass does NOT enforce that Function/IfStmt/ForStmt body must be SeqStmts.
 * It will flatten them if they contain only a single statement.
 *
 * @return Pass that flattens single-statement blocks
 */
Pass FlattenSingleStmt();

}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASSES_H_
