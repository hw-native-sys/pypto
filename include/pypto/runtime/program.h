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

#ifndef PYPTO_RUNTIME_PROGRAM_H_
#define PYPTO_RUNTIME_PROGRAM_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "pypto/runtime/instruction.h"

namespace pypto {
namespace runtime {

/**
 * @brief Runtime program that executes on AICPU
 *
 * A RuntimeProgram is a sequence of instructions with labels.
 * Labels are resolved to PC offsets when the program is finalized.
 */
class RuntimeProgram {
 public:
  RuntimeProgram() = default;

  /**
   * @brief Add an instruction to the program
   * @param instr Instruction to add
   */
  void AddInstruction(InstructionPtr instr);

  /**
   * @brief Add a label at the current position
   * @param label Label name
   *
   * The label will point to the next instruction added.
   */
  void AddLabel(const std::string& label);

  /**
   * @brief Get instruction at given PC
   * @param pc Program counter
   * @return Instruction pointer, or nullptr if PC is out of bounds
   */
  [[nodiscard]] InstructionPtr GetInstruction(size_t pc) const;

  /**
   * @brief Get PC for a label
   * @param label Label name
   * @return PC offset, or -1 if label not found
   */
  int GetLabelPC(const std::string& label) const;

  /**
   * @brief Get total number of instructions
   */
  size_t GetInstructionCount() const { return instructions_.size(); }

  /**
   * @brief Convert program to Python-like syntax
   * @return Multi-line string representation of the program
   */
  std::string ToPythonSyntax() const;

  /**
   * @brief Helper to add CONST instruction
   */
  void Const(const Register& dst, Value value);

  /**
   * @brief Helper to add ADD instruction
   */
  void Add(const Register& dst, const Register& src1, const Register& src2);

  /**
   * @brief Helper to add SUB instruction
   */
  void Sub(const Register& dst, const Register& src1, const Register& src2);

  /**
   * @brief Helper to add MUL instruction
   */
  void Mul(const Register& dst, const Register& src1, const Register& src2);

  /**
   * @brief Helper to add CMP_EQ instruction
   */
  void CmpEq(const Register& dst, const Register& src1, const Register& src2);

  /**
   * @brief Helper to add JUMP instruction
   */
  void Jump(const std::string& label);

  /**
   * @brief Helper to add JUMP_IF_ZERO instruction
   */
  void JumpIfZero(const Register& cond, const std::string& label);

  /**
   * @brief Helper to add JUMP_IF_NOT_ZERO instruction
   */
  void JumpIfNotZero(const Register& cond, const std::string& label);

  /**
   * @brief Helper to add HALT instruction
   */
  void Halt();

  /**
   * @brief Helper to add DISPATCH instruction
   */
  void Dispatch(const Register& handle, const std::string& task_name, const std::vector<Register>& args);

  /**
   * @brief Helper to add WAIT instruction
   */
  void Wait(const Register& handle);

  /**
   * @brief Helper to add WAIT_ALL instruction
   */
  void WaitAll(const std::vector<Register>& handles);

  /**
   * @brief Helper to add STORE_MEM instruction
   */
  void StoreMem(int64_t addr, const Register& reg);

  /**
   * @brief Helper to add LOAD_MEM instruction
   */
  void LoadMem(const Register& reg, int64_t addr);

  /**
   * @brief Helper to add NOP instruction
   */
  void Nop();

  /**
   * @brief Generate a register ID from an integer
   * @param n Register number
   * @return Register for the given number
   */
  static Register Reg(int n);

 private:
  std::vector<InstructionPtr> instructions_;
  std::map<std::string, size_t> labels_;  // label -> PC mapping
};

}  // namespace runtime
}  // namespace pypto

#endif  // PYPTO_RUNTIME_PROGRAM_H_
