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

#ifndef PYPTO_IR_TRANSFORMS_PASS_CONTEXT_H_
#define PYPTO_IR_TRANSFORMS_PASS_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/transforms/ir_property.h"

namespace pypto {
namespace ir {

// Forward declare Pass to avoid circular include (pass_context.h <-> passes.h)
class Pass;

/**
 * @brief Controls when property verification runs
 */
enum class VerificationMode {
  None,           ///< No automatic verification
  Before,         ///< Verify required properties before each pass
  After,          ///< Verify produced properties after each pass
  BeforeAndAfter  ///< Verify both before and after each pass
};

/**
 * @brief Abstract base class for pass instrumentation
 *
 * PassInstruments are callbacks that run before/after each pass execution.
 * Subclass this to implement custom instrumentation (verification, logging, profiling, etc.).
 */
class PassInstrument {
 public:
  virtual ~PassInstrument() = default;

  /**
   * @brief Called before a pass is executed
   * @param pass The pass about to run
   * @param program The program before transformation
   */
  virtual void RunBeforePass(const Pass& pass, const ProgramPtr& program) = 0;

  /**
   * @brief Called after a pass is executed
   * @param pass The pass that just ran
   * @param program The program after transformation
   */
  virtual void RunAfterPass(const Pass& pass, const ProgramPtr& program) = 0;

  /**
   * @brief Get the name of this instrument
   */
  [[nodiscard]] virtual std::string GetName() const = 0;
};

using PassInstrumentPtr = std::shared_ptr<PassInstrument>;

/**
 * @brief Instrument that verifies IR properties before/after passes
 *
 * Uses PropertyVerifierRegistry to check that passes' required properties hold
 * before execution and produced properties hold after execution.
 */
class VerificationInstrument : public PassInstrument {
 public:
  explicit VerificationInstrument(VerificationMode mode);

  void RunBeforePass(const Pass& pass, const ProgramPtr& program) override;
  void RunAfterPass(const Pass& pass, const ProgramPtr& program) override;
  [[nodiscard]] std::string GetName() const override;

 private:
  VerificationMode mode_;
};

/**
 * @brief Context that holds instruments and manages a thread-local stack
 *
 * PassContext provides a `with`-style nesting mechanism. When active, Pass::operator()
 * will run the context's instruments before/after each pass execution.
 *
 * Usage (Python):
 * @code
 *   with PassContext([VerificationInstrument(VerificationMode.AFTER)]):
 *       result = some_pass(program)  # instruments fire automatically
 * @endcode
 */
class PassContext {
 public:
  /**
   * @brief Create a context with instruments and optional verification level
   * @param instruments List of pass instruments
   * @param verification_level Verification level (default: Basic)
   */
  explicit PassContext(std::vector<PassInstrumentPtr> instruments,
                       VerificationLevel verification_level = VerificationLevel::Basic);

  /**
   * @brief Push this context onto the thread-local stack
   */
  void EnterContext();

  /**
   * @brief Pop this context from the thread-local stack
   */
  void ExitContext();

  /**
   * @brief Run all instruments' RunBeforePass
   */
  void RunBeforePass(const Pass& pass, const ProgramPtr& program);

  /**
   * @brief Run all instruments' RunAfterPass
   */
  void RunAfterPass(const Pass& pass, const ProgramPtr& program);

  /**
   * @brief Get the verification level for this context
   */
  [[nodiscard]] VerificationLevel GetVerificationLevel() const;

  /**
   * @brief Get the currently active context (top of thread-local stack)
   * @return Pointer to current context, or nullptr if none
   */
  static PassContext* Current();

 private:
  std::vector<PassInstrumentPtr> instruments_;
  VerificationLevel verification_level_;
  PassContext* previous_;

  static thread_local PassContext* current_;
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_PASS_CONTEXT_H_
