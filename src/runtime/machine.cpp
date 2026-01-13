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

#include "pypto/runtime/machine.h"

#include <stdexcept>

namespace pypto {
namespace runtime {

RuntimeMachine::RuntimeMachine(int num_aicore) : num_aicore_(num_aicore) {
  if (num_aicore <= 0) {
    throw std::invalid_argument("num_aicore must be positive");
  }

  // Create shared resources
  task_queue_ = std::make_shared<TaskQueue>();
  memory_ = std::make_shared<SharedMemory>();
  task_registry_ = std::make_shared<TaskRegistry>();

  // Create single AICPU host (id=0)
  aicpu_ = std::make_shared<AICPUInterpreter>(0, task_queue_, memory_, task_registry_);

  // Create AICORE workers
  for (int i = 0; i < num_aicore; ++i) {
    auto aicore = std::make_shared<AICoreWorker>(i, task_queue_, memory_);
    aicores_.push_back(aicore);
  }
}

RuntimeMachine::~RuntimeMachine() {
  Stop();
}

void RuntimeMachine::Start() {
  // Start AICORE workers if not already started
  for (auto& aicore : aicores_) {
    if (!aicore->IsRunning()) {
      aicore->Start();
    }
  }
}

void RuntimeMachine::Stop() {
  // Stop task queue
  task_queue_->Stop();

  // Stop all AICORE workers
  for (auto& aicore : aicores_) {
    aicore->Stop();
  }
}

void RuntimeMachine::RegisterTask(const std::string& name, TaskCallable callable) {
  (*task_registry_)[name] = std::move(callable);
}

void RuntimeMachine::LoadAndRunProgram(std::shared_ptr<RuntimeProgram> program) {
  // Start AICORE workers if needed
  Start();

  // Load program onto AICPU host
  aicpu_->LoadProgram(std::move(program));

  // Execute synchronously on calling thread (AICPU host)
  aicpu_->Execute();
}

}  // namespace runtime
}  // namespace pypto
