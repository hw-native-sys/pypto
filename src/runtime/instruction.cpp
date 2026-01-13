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

#include "pypto/runtime/instruction.h"

#include <sstream>

namespace pypto {
namespace runtime {

const char* OpcodeToString(Opcode op) {
  switch (op) {
    case Opcode::CONST:
      return "CONST";
    case Opcode::ADD:
      return "ADD";
    case Opcode::SUB:
      return "SUB";
    case Opcode::MUL:
      return "MUL";
    case Opcode::CMP_EQ:
      return "CMP_EQ";
    case Opcode::JUMP:
      return "JUMP";
    case Opcode::JUMP_IF_ZERO:
      return "JUMP_IF_ZERO";
    case Opcode::JUMP_IF_NOT_ZERO:
      return "JUMP_IF_NOT_ZERO";
    case Opcode::HALT:
      return "HALT";
    case Opcode::DISPATCH:
      return "DISPATCH";
    case Opcode::WAIT:
      return "WAIT";
    case Opcode::WAIT_ALL:
      return "WAIT_ALL";
    case Opcode::STORE_MEM:
      return "STORE_MEM";
    case Opcode::LOAD_MEM:
      return "LOAD_MEM";
    case Opcode::NOP:
      return "NOP";
    default:
      return "UNKNOWN";
  }
}

namespace {
// Helper to convert Value to string
std::string ValueToString(const Value& val) {
  if (std::holds_alternative<std::monostate>(val)) {
    return "None";
  } else if (std::holds_alternative<int64_t>(val)) {
    return std::to_string(std::get<int64_t>(val));
  } else if (std::holds_alternative<double>(val)) {
    std::ostringstream oss;
    oss << std::get<double>(val);
    return oss.str();
  } else if (std::holds_alternative<std::string>(val)) {
    return "\"" + std::get<std::string>(val) + "\"";
  } else if (std::holds_alternative<void*>(val)) {
    std::ostringstream oss;
    oss << std::get<void*>(val);
    return oss.str();
  }
  return "unknown";
}
}  // namespace

std::string ConstInstruction::ToPythonSyntax() const {
  return "CONST(" + dst_ + ", " + ValueToString(value_) + ")";
}

std::string ArithmeticInstruction::ToPythonSyntax() const {
  return std::string(OpcodeToString(opcode_)) + "(" + dst_ + ", " + src1_ + ", " + src2_ + ")";
}

std::string JumpInstruction::ToPythonSyntax() const {
  return "JUMP(\"" + label_ + "\")";
}

std::string ConditionalJumpInstruction::ToPythonSyntax() const {
  return std::string(OpcodeToString(opcode_)) + "(" + cond_ + ", \"" + label_ + "\")";
}

std::string HaltInstruction::ToPythonSyntax() const {
  return "HALT()";
}

std::string DispatchInstruction::ToPythonSyntax() const {
  std::ostringstream oss;
  oss << "DISPATCH(" << handle_ << ", \"" << task_name_ << "\"";
  for (const auto& arg : args_) {
    oss << ", " << arg;
  }
  oss << ")";
  return oss.str();
}

std::string WaitInstruction::ToPythonSyntax() const {
  return "WAIT(" + handle_ + ")";
}

std::string WaitAllInstruction::ToPythonSyntax() const {
  std::ostringstream oss;
  oss << "WAIT_ALL(";
  for (size_t i = 0; i < handles_.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << handles_[i];
  }
  oss << ")";
  return oss.str();
}

std::string MemoryInstruction::ToPythonSyntax() const {
  std::ostringstream oss;
  if (opcode_ == Opcode::STORE_MEM) {
    oss << "STORE_MEM(0x" << std::hex << addr_ << ", " << reg_ << ")";
  } else {  // LOAD_MEM
    oss << "LOAD_MEM(" << reg_ << ", 0x" << std::hex << addr_ << ")";
  }
  return oss.str();
}

std::string NopInstruction::ToPythonSyntax() const {
  return "NOP()";
}

}  // namespace runtime
}  // namespace pypto
