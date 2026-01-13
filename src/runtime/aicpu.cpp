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

#include "pypto/runtime/aicpu.h"

#include <iostream>
#include <stdexcept>

namespace pypto {
namespace runtime {

AICPUInterpreter::AICPUInterpreter(int id, std::shared_ptr<TaskQueue> task_queue,
                                   std::shared_ptr<SharedMemory> memory,
                                   std::shared_ptr<TaskRegistry> task_registry)
    : id_(id),
      task_queue_(std::move(task_queue)),
      memory_(std::move(memory)),
      task_registry_(std::move(task_registry)),
      pc_(0) {}

AICPUInterpreter::~AICPUInterpreter() = default;

void AICPUInterpreter::LoadProgram(std::shared_ptr<RuntimeProgram> program) {
  program_ = std::move(program);
  pc_ = 0;
  registers_.clear();
  task_handles_.clear();
}

void AICPUInterpreter::Execute() {
  if (!program_) {
    throw std::runtime_error("No program loaded on AICPU");
  }

  try {
    while (pc_ < program_->GetInstructionCount()) {
      InstructionPtr instr = program_->GetInstruction(pc_);
      if (!instr) {
        std::cerr << "AICPU[" << id_ << "] Invalid instruction at PC=" << pc_ << '\n';
        break;
      }

      // Increment PC before execution (allows jumps to override)
      pc_++;

      // Execute instruction
      bool continue_execution = ExecuteInstruction(instr);
      if (!continue_execution) {
        break;  // HALT encountered
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "AICPU[" << id_ << "] Execution failed: " << e.what() << '\n';
    throw;  // Re-throw to let caller handle
  }
}

bool AICPUInterpreter::ExecuteInstruction(const InstructionPtr& instr) {
  switch (instr->GetOpcode()) {
    case Opcode::CONST:
      ExecuteConst(dynamic_cast<const ConstInstruction*>(instr.get()));
      break;

    case Opcode::ADD:
    case Opcode::SUB:
    case Opcode::MUL:
    case Opcode::CMP_EQ:
      ExecuteArithmetic(dynamic_cast<const ArithmeticInstruction*>(instr.get()));
      break;

    case Opcode::JUMP:
      ExecuteJump(dynamic_cast<const JumpInstruction*>(instr.get()));
      break;

    case Opcode::JUMP_IF_ZERO:
    case Opcode::JUMP_IF_NOT_ZERO:
      ExecuteConditionalJump(dynamic_cast<const ConditionalJumpInstruction*>(instr.get()));
      break;

    case Opcode::HALT:
      return false;  // Stop execution

    case Opcode::DISPATCH:
      ExecuteDispatch(dynamic_cast<const DispatchInstruction*>(instr.get()));
      break;

    case Opcode::WAIT:
      ExecuteWait(dynamic_cast<const WaitInstruction*>(instr.get()));
      break;

    case Opcode::WAIT_ALL:
      ExecuteWaitAll(dynamic_cast<const WaitAllInstruction*>(instr.get()));
      break;

    case Opcode::STORE_MEM:
    case Opcode::LOAD_MEM:
      ExecuteMemory(dynamic_cast<const MemoryInstruction*>(instr.get()));
      break;

    case Opcode::NOP:
      // Do nothing
      break;

    default:
      throw std::runtime_error("Unknown opcode: " + std::to_string(static_cast<int>(instr->GetOpcode())));
  }

  return true;  // Continue execution
}

Value AICPUInterpreter::GetRegisterValue(const Register& reg) {
  auto it = registers_.find(reg);
  if (it == registers_.end()) {
    return std::monostate{};  // Uninitialized register
  }
  return it->second;
}

void AICPUInterpreter::SetRegisterValue(const Register& reg, const Value& value) {
  registers_[reg] = value;
}

void AICPUInterpreter::ExecuteConst(const ConstInstruction* instr) {
  SetRegisterValue(instr->GetDst(), instr->GetValue());
}

void AICPUInterpreter::ExecuteArithmetic(const ArithmeticInstruction* instr) {
  Value src1_val = GetRegisterValue(instr->GetSrc1());
  Value src2_val = GetRegisterValue(instr->GetSrc2());

  int64_t src1 = ValueToInt(src1_val);
  int64_t src2 = ValueToInt(src2_val);

  int64_t result;
  switch (instr->GetOpcode()) {
    case Opcode::ADD:
      result = src1 + src2;
      break;
    case Opcode::SUB:
      result = src1 - src2;
      break;
    case Opcode::MUL:
      result = src1 * src2;
      break;
    case Opcode::CMP_EQ:
      result = (src1 == src2) ? 1 : 0;
      break;
    default:
      throw std::runtime_error("Unknown arithmetic opcode");
  }

  SetRegisterValue(instr->GetDst(), result);
}

void AICPUInterpreter::ExecuteJump(const JumpInstruction* instr) {
  int target_pc = program_->GetLabelPC(instr->GetLabel());
  if (target_pc < 0) {
    throw std::runtime_error("Label not found: " + instr->GetLabel());
  }
  pc_ = static_cast<size_t>(target_pc);
}

void AICPUInterpreter::ExecuteConditionalJump(const ConditionalJumpInstruction* instr) {
  Value cond_val = GetRegisterValue(instr->GetCond());
  bool is_zero = IsZero(cond_val);

  bool should_jump = (instr->GetOpcode() == Opcode::JUMP_IF_ZERO) ? is_zero : !is_zero;

  if (should_jump) {
    int target_pc = program_->GetLabelPC(instr->GetLabel());
    if (target_pc < 0) {
      throw std::runtime_error("Label not found: " + instr->GetLabel());
    }
    pc_ = static_cast<size_t>(target_pc);
  }
}

void AICPUInterpreter::ExecuteDispatch(const DispatchInstruction* instr) {
  // Get task callable from registry
  auto it = task_registry_->find(instr->GetTaskName());
  if (it == task_registry_->end()) {
    throw std::runtime_error("Task not found in registry: " + instr->GetTaskName());
  }

  TaskCallable callable = it->second;

  // Resolve argument registers to values
  std::vector<Value> args;
  for (const auto& arg_reg : instr->GetArgs()) {
    args.push_back(GetRegisterValue(arg_reg));
  }

  // Create task handle
  auto handle = std::make_shared<TaskHandle>();

  // Create and enqueue task
  Task task(instr->GetTaskName(), std::move(args), handle, callable);
  task_queue_->Enqueue(std::move(task));

  // Store handle in register
  task_handles_[instr->GetHandle()] = handle;

  // Store handle pointer as a value (for future WAIT instructions)
  SetRegisterValue(instr->GetHandle(), static_cast<void*>(handle.get()));
}

void AICPUInterpreter::ExecuteWait(const WaitInstruction* instr) {
  auto it = task_handles_.find(instr->GetHandle());
  if (it == task_handles_.end()) {
    throw std::runtime_error("Task handle not found: " + instr->GetHandle().ToString());
  }

  TaskHandlePtr handle = it->second;
  handle->Wait();

  // Optionally, we could store the result in a register
  Value result = handle->GetResult();
  // Create a result register by using a higher ID (handle_id + 10000 for results)
  Register result_reg(instr->GetHandle().GetId() + 10000);
  SetRegisterValue(result_reg, result);
}

void AICPUInterpreter::ExecuteWaitAll(const WaitAllInstruction* instr) {
  for (const auto& handle_reg : instr->GetHandles()) {
    auto it = task_handles_.find(handle_reg);
    if (it == task_handles_.end()) {
      throw std::runtime_error("Task handle not found: " + handle_reg.ToString());
    }
    it->second->Wait();
  }
}

void AICPUInterpreter::ExecuteMemory(const MemoryInstruction* instr) {
  if (instr->GetOpcode() == Opcode::STORE_MEM) {
    Value val = GetRegisterValue(instr->GetReg());
    memory_->Write(instr->GetAddr(), val);
  } else {  // LOAD_MEM
    Value val = memory_->Read(instr->GetAddr());
    SetRegisterValue(instr->GetReg(), val);
  }
}

int64_t AICPUInterpreter::ValueToInt(const Value& val) {
  if (std::holds_alternative<int64_t>(val)) {
    return std::get<int64_t>(val);
  } else if (std::holds_alternative<double>(val)) {
    return static_cast<int64_t>(std::get<double>(val));
  } else if (std::holds_alternative<std::monostate>(val)) {
    return 0;
  } else {
    throw std::runtime_error("Cannot convert value to int");
  }
}

bool AICPUInterpreter::IsZero(const Value& val) { return ValueToInt(val) == 0; }

}  // namespace runtime
}  // namespace pypto
