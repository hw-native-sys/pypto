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

#ifndef PYPTO_RUNTIME_AICPU_H_
#define PYPTO_RUNTIME_AICPU_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "pypto/runtime/instruction.h"
#include "pypto/runtime/memory.h"
#include "pypto/runtime/program.h"
#include "pypto/runtime/task.h"

namespace pypto {
namespace runtime {

/**
 * @brief Task registry for AICORE callables
 *
 * Maps task names to callable implementations.
 */
using TaskRegistry = std::map<std::string, TaskCallable>;

/**
 * @brief AICPU interpreter
 *
 * Executes a RuntimeProgram synchronously using a fetch-decode-execute loop.
 * The interpreter has its own register file and program counter.
 * Execution happens on the calling thread (the AICPU host).
 */
class AICPUInterpreter {
 public:
  AICPUInterpreter(int id, std::shared_ptr<TaskQueue> task_queue, std::shared_ptr<SharedMemory> memory,
                   std::shared_ptr<TaskRegistry> task_registry);
  ~AICPUInterpreter();

  /**
   * @brief Load a program onto this AICPU
   * @param program Program to load
   */
  void LoadProgram(std::shared_ptr<RuntimeProgram> program);

  /**
   * @brief Execute the loaded program synchronously
   *
   * Runs the fetch-decode-execute loop on the calling thread until
   * a HALT instruction is encountered or an error occurs.
   */
  void Execute();

  /**
   * @brief Get interpreter ID
   */
  int GetId() const { return id_; }

 private:

  /**
   * @brief Execute a single instruction
   * @return false if HALT encountered
   */
  bool ExecuteInstruction(const InstructionPtr& instr);

  /**
   * @brief Get value from register
   */
  Value GetRegisterValue(const Register& reg);

  /**
   * @brief Set value in register
   */
  void SetRegisterValue(const Register& reg, const Value& value);

  /**
   * @brief Execute CONST instruction
   */
  void ExecuteConst(const ConstInstruction* instr);

  /**
   * @brief Execute arithmetic instruction
   */
  void ExecuteArithmetic(const ArithmeticInstruction* instr);

  /**
   * @brief Execute JUMP instruction
   */
  void ExecuteJump(const JumpInstruction* instr);

  /**
   * @brief Execute conditional jump instruction
   */
  void ExecuteConditionalJump(const ConditionalJumpInstruction* instr);

  /**
   * @brief Execute DISPATCH instruction
   */
  void ExecuteDispatch(const DispatchInstruction* instr);

  /**
   * @brief Execute WAIT instruction
   */
  void ExecuteWait(const WaitInstruction* instr);

  /**
   * @brief Execute WAIT_ALL instruction
   */
  void ExecuteWaitAll(const WaitAllInstruction* instr);

  /**
   * @brief Execute memory instruction
   */
  void ExecuteMemory(const MemoryInstruction* instr);

  /**
   * @brief Convert Value to int64_t (for arithmetic and comparison)
   */
  int64_t ValueToInt(const Value& val);

  /**
   * @brief Check if value is zero (for conditional jumps)
   */
  bool IsZero(const Value& val);

  int id_;
  std::shared_ptr<RuntimeProgram> program_;
  std::shared_ptr<TaskQueue> task_queue_;
  std::shared_ptr<SharedMemory> memory_;
  std::shared_ptr<TaskRegistry> task_registry_;

  // Interpreter state
  size_t pc_;  ///< Program counter
  std::unordered_map<Register, Value, RegisterIdHash> registers_;  ///< Unlimited virtual registers
  std::unordered_map<Register, TaskHandlePtr, RegisterIdHash> task_handles_;  ///< Task handle storage
};

}  // namespace runtime
}  // namespace pypto

#endif  // PYPTO_RUNTIME_AICPU_H_
