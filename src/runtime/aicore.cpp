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

#include "pypto/runtime/aicore.h"

#include <iostream>

namespace pypto {
namespace runtime {

AICoreWorker::AICoreWorker(int id, std::shared_ptr<TaskQueue> task_queue,
                           std::shared_ptr<SharedMemory> memory)
    : id_(id), task_queue_(std::move(task_queue)), memory_(std::move(memory)), running_(false) {}

AICoreWorker::~AICoreWorker() { Stop(); }

void AICoreWorker::Start() {
  if (running_.load()) {
    return;
  }

  running_.store(true);

  // Start the worker thread
  worker_thread_ = std::thread(&AICoreWorker::WorkerLoop, this);
}

void AICoreWorker::Stop() {
  if (!running_.load()) {
    return;
  }

  running_.store(false);

  // Wait for the worker thread to finish
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

bool AICoreWorker::IsRunning() const { return running_.load(); }

void AICoreWorker::WorkerLoop() {
  while (running_.load()) {
    auto task_opt = task_queue_->Dequeue();

    if (!task_opt.has_value()) {
      // Queue stopped
      break;
    }

    Task& task = task_opt.value();

    try {
      // Execute the task callable
      Value result = task.callable(task.args);

      // Mark task as completed
      task.handle->SetCompleted(result);
    } catch (const std::exception& e) {
      std::cerr << "AICORE[" << id_ << "] Task \"" << task.task_name << "\" failed: " << e.what() << '\n';
      task.handle->SetCompleted(std::monostate{});
    }
  }
}

}  // namespace runtime
}  // namespace pypto
