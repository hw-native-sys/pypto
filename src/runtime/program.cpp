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

#include "pypto/runtime/program.h"

#include <sstream>

namespace pypto {
namespace runtime {

void RuntimeProgram::AddInstruction(InstructionPtr instr) {
  instructions_.push_back(std::move(instr));
}

void RuntimeProgram::AddLabel(const std::string& label) {
  labels_[label] = instructions_.size();
}

InstructionPtr RuntimeProgram::GetInstruction(size_t pc) const {
  if (pc >= instructions_.size()) {
    return nullptr;
  }
  return instructions_[pc];
}

int RuntimeProgram::GetLabelPC(const std::string& label) const {
  auto it = labels_.find(label);
  if (it == labels_.end()) {
    return -1;
  }
  return static_cast<int>(it->second);
}

std::string RuntimeProgram::ToPythonSyntax() const {
  std::ostringstream oss;

  // Build reverse map: PC -> label(s)
  std::map<size_t, std::vector<std::string>> pc_to_labels;
  for (const auto& [label, pc] : labels_) {
    pc_to_labels[pc].push_back(label);
  }

  for (size_t pc = 0; pc < instructions_.size(); ++pc) {
    // Print labels for this PC
    if (pc_to_labels.count(pc)) {
      for (const auto& label : pc_to_labels[pc]) {
        oss << label << ":\n";
      }
    }

    // Print instruction
    oss << "    " << instructions_[pc]->ToPythonSyntax() << "\n";
  }

  return oss.str();
}

// Helper methods

void RuntimeProgram::Const(const Register& dst, Value value) {
  AddInstruction(std::make_shared<ConstInstruction>(dst, std::move(value)));
}

void RuntimeProgram::Add(const Register& dst, const Register& src1, const Register& src2) {
  AddInstruction(std::make_shared<ArithmeticInstruction>(Opcode::ADD, dst, src1, src2));
}

void RuntimeProgram::Sub(const Register& dst, const Register& src1, const Register& src2) {
  AddInstruction(std::make_shared<ArithmeticInstruction>(Opcode::SUB, dst, src1, src2));
}

void RuntimeProgram::Mul(const Register& dst, const Register& src1, const Register& src2) {
  AddInstruction(std::make_shared<ArithmeticInstruction>(Opcode::MUL, dst, src1, src2));
}

void RuntimeProgram::CmpEq(const Register& dst, const Register& src1, const Register& src2) {
  AddInstruction(std::make_shared<ArithmeticInstruction>(Opcode::CMP_EQ, dst, src1, src2));
}

void RuntimeProgram::Jump(const std::string& label) {
  AddInstruction(std::make_shared<JumpInstruction>(label));
}

void RuntimeProgram::JumpIfZero(const Register& cond, const std::string& label) {
  AddInstruction(std::make_shared<ConditionalJumpInstruction>(Opcode::JUMP_IF_ZERO, cond, label));
}

void RuntimeProgram::JumpIfNotZero(const Register& cond, const std::string& label) {
  AddInstruction(std::make_shared<ConditionalJumpInstruction>(Opcode::JUMP_IF_NOT_ZERO, cond, label));
}

void RuntimeProgram::Halt() {
  AddInstruction(std::make_shared<HaltInstruction>());
}

void RuntimeProgram::Dispatch(const Register& handle, const std::string& task_name, const std::vector<Register>& args) {
  AddInstruction(std::make_shared<DispatchInstruction>(handle, task_name, args));
}

void RuntimeProgram::Wait(const Register& handle) {
  AddInstruction(std::make_shared<WaitInstruction>(handle));
}

void RuntimeProgram::WaitAll(const std::vector<Register>& handles) {
  AddInstruction(std::make_shared<WaitAllInstruction>(handles));
}

void RuntimeProgram::StoreMem(int64_t addr, const Register& reg) {
  AddInstruction(std::make_shared<MemoryInstruction>(Opcode::STORE_MEM, addr, reg));
}

void RuntimeProgram::LoadMem(const Register& reg, int64_t addr) {
  AddInstruction(std::make_shared<MemoryInstruction>(Opcode::LOAD_MEM, addr, reg));
}

void RuntimeProgram::Nop() {
  AddInstruction(std::make_shared<NopInstruction>());
}

Register RuntimeProgram::Reg(int n) {
  return Register(n);
}

}  // namespace runtime
}  // namespace pypto
