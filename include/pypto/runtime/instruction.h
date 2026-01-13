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

#ifndef PYPTO_RUNTIME_INSTRUCTION_H_
#define PYPTO_RUNTIME_INSTRUCTION_H_

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace pypto {
namespace runtime {

/**
 * @brief Opcode enumeration for runtime instructions
 */
enum class Opcode {
  // Value creation
  CONST,  ///< Create a constant value

  // Arithmetic
  ADD,      ///< Addition
  SUB,      ///< Subtraction
  MUL,      ///< Multiplication
  CMP_EQ,   ///< Compare equal

  // Control flow
  JUMP,              ///< Unconditional jump
  JUMP_IF_ZERO,      ///< Conditional jump if zero
  JUMP_IF_NOT_ZERO,  ///< Conditional jump if not zero
  HALT,              ///< Stop execution

  // Task dispatch
  DISPATCH,   ///< Dispatch task to AICORE
  WAIT,       ///< Wait for task completion
  WAIT_ALL,   ///< Wait for all tasks

  // Memory operations
  STORE_MEM,  ///< Store to shared memory
  LOAD_MEM,   ///< Load from shared memory

  // Utility
  NOP  ///< No operation
};

/**
 * @brief Convert opcode to string
 */
const char* OpcodeToString(Opcode op);

/**
 * @brief Register identifier (symbolic value reference)
 *
 * Registers are unlimited and dynamically created.
 * Examples: r0, r1, r2, ..., r9999, ...
 *
 * Internally uses an integer ID for efficiency instead of string.
 */
class Register {
 public:
  /**
   * @brief Construct from integer ID (e.g., 0 for r0, 1 for r1)
   */
  explicit Register(int id) : id_(id) {}

  /**
   * @brief Construct from string representation (e.g., "r0", "r1")
   * @throws std::invalid_argument if string format is invalid
   */
  explicit Register(const std::string& str);

  /**
   * @brief Get the integer ID
   */
  [[nodiscard]] int GetId() const { return id_; }

  /**
   * @brief Convert to string representation (e.g., "r0", "r1")
   */
  [[nodiscard]] std::string ToString() const;

  /**
   * @brief Equality comparison
   */
  bool operator==(const Register& other) const { return id_ == other.id_; }
  bool operator!=(const Register& other) const { return id_ != other.id_; }

  /**
   * @brief Less-than comparison (for ordered containers)
   */
  bool operator<(const Register& other) const { return id_ < other.id_; }

 private:
  int id_;
};

/**
 * @brief Hash function for Register (for unordered containers)
 */
struct RegisterIdHash {
  size_t operator()(const Register& reg) const {
    return std::hash<int>()(reg.GetId());
  }
};

/**
 * @brief Value type that can be held in a register
 *
 * Registers can reference any of these types:
 * - int64_t: Integer constants
 * - double: Float constants
 * - std::string: String constants or tensor names
 * - void*: Opaque handles (task handles, tensor references, etc.)
 */
using Value = std::variant<std::monostate, int64_t, double, std::string, void*>;

/**
 * @brief Base class for all instructions
 */
class Instruction {
 public:
  explicit Instruction(Opcode op) : opcode_(op) {}
  virtual ~Instruction() = default;

  [[nodiscard]] Opcode GetOpcode() const { return opcode_; }

  /**
   * @brief Convert instruction to Python-like syntax string
   */
  [[nodiscard]] virtual std::string ToPythonSyntax() const = 0;

 protected:
  Opcode opcode_;
};

using InstructionPtr = std::shared_ptr<Instruction>;

/**
 * @brief CONST instruction: Create a constant value
 * Syntax: CONST(r_dst, value)
 */
class ConstInstruction : public Instruction {
 public:
  ConstInstruction(Register dst, Value value) : Instruction(Opcode::CONST), dst_(dst), value_(std::move(value)) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;

  [[nodiscard]] const Register& GetDst() const { return dst_; }
  [[nodiscard]] const Value& GetValue() const { return value_; }

 private:
  Register dst_;
  Value value_;
};

/**
 * @brief Arithmetic instruction (ADD, SUB, MUL, CMP_EQ)
 * Syntax: OP(r_dst, r_src1, r_src2)
 */
class ArithmeticInstruction : public Instruction {
 public:
  ArithmeticInstruction(Opcode op, Register dst, Register src1, Register src2)
      : Instruction(op), dst_(dst), src1_(src1), src2_(src2) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;

  [[nodiscard]] const Register& GetDst() const { return dst_; }
  [[nodiscard]] const Register& GetSrc1() const { return src1_; }
  [[nodiscard]] const Register& GetSrc2() const { return src2_; }

 private:
  Register dst_;
  Register src1_;
  Register src2_;
};

/**
 * @brief JUMP instruction: Unconditional jump
 * Syntax: JUMP(label)
 */
class JumpInstruction : public Instruction {
 public:
  explicit JumpInstruction(std::string label) : Instruction(Opcode::JUMP), label_(std::move(label)) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;

  [[nodiscard]] const std::string& GetLabel() const { return label_; }

 private:
  std::string label_;
};

/**
 * @brief Conditional jump instruction (JUMP_IF_ZERO, JUMP_IF_NOT_ZERO)
 * Syntax: JUMP_IF_ZERO(r_cond, label)
 */
class ConditionalJumpInstruction : public Instruction {
 public:
  ConditionalJumpInstruction(Opcode op, Register cond, std::string label)
      : Instruction(op), cond_(cond), label_(std::move(label)) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;

  [[nodiscard]] const Register& GetCond() const { return cond_; }
  [[nodiscard]] const std::string& GetLabel() const { return label_; }

 private:
  Register cond_;
  std::string label_;
};

/**
 * @brief HALT instruction: Stop execution
 * Syntax: HALT()
 */
class HaltInstruction : public Instruction {
 public:
  HaltInstruction() : Instruction(Opcode::HALT) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;
};

/**
 * @brief DISPATCH instruction: Dispatch task to AICORE
 * Syntax: DISPATCH(r_handle, task_name, r_arg1, r_arg2, ...)
 */
class DispatchInstruction : public Instruction {
 public:
  DispatchInstruction(Register handle, std::string task_name, std::vector<Register> args)
      : Instruction(Opcode::DISPATCH), handle_(handle), task_name_(std::move(task_name)), args_(std::move(args)) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;

  [[nodiscard]] const Register& GetHandle() const { return handle_; }
  [[nodiscard]] const std::string& GetTaskName() const { return task_name_; }
  [[nodiscard]] const std::vector<Register>& GetArgs() const { return args_; }

 private:
  Register handle_;
  std::string task_name_;
  std::vector<Register> args_;
};

/**
 * @brief WAIT instruction: Wait for task completion
 * Syntax: WAIT(r_handle)
 */
class WaitInstruction : public Instruction {
 public:
  explicit WaitInstruction(Register handle) : Instruction(Opcode::WAIT), handle_(handle) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;

  [[nodiscard]] const Register& GetHandle() const { return handle_; }

 private:
  Register handle_;
};

/**
 * @brief WAIT_ALL instruction: Wait for all tasks
 * Syntax: WAIT_ALL(r_handle1, r_handle2, ...)
 */
class WaitAllInstruction : public Instruction {
 public:
  explicit WaitAllInstruction(std::vector<Register> handles) : Instruction(Opcode::WAIT_ALL), handles_(std::move(handles)) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;

  [[nodiscard]] const std::vector<Register>& GetHandles() const { return handles_; }

 private:
  std::vector<Register> handles_;
};

/**
 * @brief Memory instruction (STORE_MEM, LOAD_MEM)
 * Syntax: STORE_MEM(addr, r_src) or LOAD_MEM(r_dst, addr)
 */
class MemoryInstruction : public Instruction {
 public:
  MemoryInstruction(Opcode op, int64_t addr, Register reg)
      : Instruction(op), addr_(addr), reg_(reg) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;

  [[nodiscard]] int64_t GetAddr() const { return addr_; }
  [[nodiscard]] const Register& GetReg() const { return reg_; }

 private:
  int64_t addr_;
  Register reg_;
};

/**
 * @brief NOP instruction: No operation
 * Syntax: NOP()
 */
class NopInstruction : public Instruction {
 public:
  NopInstruction() : Instruction(Opcode::NOP) {}

  [[nodiscard]] std::string ToPythonSyntax() const override;
};

}  // namespace runtime
}  // namespace pypto

#endif  // PYPTO_RUNTIME_INSTRUCTION_H_
