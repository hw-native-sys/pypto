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

#ifndef PYPTO_RUNTIME_AICORE_H_
#define PYPTO_RUNTIME_AICORE_H_

#include <atomic>
#include <memory>
#include <thread>

#include "pypto/runtime/memory.h"
#include "pypto/runtime/task.h"

namespace pypto {
namespace runtime {

/**
 * @brief AICORE worker thread
 *
 * Continuously pulls tasks from the queue and executes them.
 * Simulates a single AICORE unit using a CPU thread.
 */
class AICoreWorker {
 public:
  AICoreWorker(int id, std::shared_ptr<TaskQueue> task_queue, std::shared_ptr<SharedMemory> memory);
  ~AICoreWorker();

  /**
   * @brief Start the worker thread
   */
  void Start();

  /**
   * @brief Stop the worker thread
   */
  void Stop();

  /**
   * @brief Check if worker is running
   */
  [[nodiscard]] bool IsRunning() const;

  /**
   * @brief Get worker ID
   */
  [[nodiscard]] int GetId() const { return id_; }

 private:
  /**
   * @brief Main worker loop (runs in separate thread)
   */
  void WorkerLoop();

  int id_;
  std::shared_ptr<TaskQueue> task_queue_;
  std::shared_ptr<SharedMemory> memory_;
  std::atomic<bool> running_;
  std::thread worker_thread_;  ///< Worker thread that executes tasks
};

}  // namespace runtime
}  // namespace pypto

#endif  // PYPTO_RUNTIME_AICORE_H_
