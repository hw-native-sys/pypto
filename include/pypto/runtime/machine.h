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

#ifndef PYPTO_RUNTIME_MACHINE_H_
#define PYPTO_RUNTIME_MACHINE_H_

#include <memory>
#include <vector>

#include "pypto/runtime/aicpu.h"
#include "pypto/runtime/aicore.h"
#include "pypto/runtime/memory.h"
#include "pypto/runtime/program.h"
#include "pypto/runtime/task.h"

namespace pypto {
namespace runtime {

/**
 * @brief Runtime Machine - simulated NPU device (AICPU host)
 *
 * The RuntimeMachine IS the AICPU host that executes programs synchronously.
 * It manages:
 * - Single AICPU interpreter (runs on calling thread)
 * - Multiple AICORE workers (compute threads)
 * - Shared memory
 * - Task queue for AICORE dispatch
 */
class RuntimeMachine {
 public:
  /**
   * @brief Create a runtime machine
   * @param num_aicore Number of AICORE units
   */
  explicit RuntimeMachine(int num_aicore);

  ~RuntimeMachine();

  /**
   * @brief Register a task implementation
   * @param name Task name
   * @param callable Task callable
   */
  void RegisterTask(const std::string& name, TaskCallable callable);

  /**
   * @brief Load and execute a program synchronously on AICPU host
   * 
   * This method blocks until the program completes (HALT instruction).
   * The program runs on the calling thread (which is the AICPU host).
   * 
   * @param program Program to execute
   */
  void LoadAndRunProgram(std::shared_ptr<RuntimeProgram> program);

  /**
   * @brief Get shared memory
   */
  std::shared_ptr<SharedMemory> GetMemory() const { return memory_; }

  /**
   * @brief Get number of AICOREs
   */
  int GetNumAICORE() const { return num_aicore_; }

 private:
  /**
   * @brief Start AICORE workers
   */
  void Start();

  /**
   * @brief Stop AICORE workers
   */
  void Stop();

  int num_aicore_;

  // Shared resources
  std::shared_ptr<TaskQueue> task_queue_;
  std::shared_ptr<SharedMemory> memory_;
  std::shared_ptr<TaskRegistry> task_registry_;

  // Components
  std::shared_ptr<AICPUInterpreter> aicpu_;  // Single AICPU host
  std::vector<std::shared_ptr<AICoreWorker>> aicores_;
};

}  // namespace runtime
}  // namespace pypto

#endif  // PYPTO_RUNTIME_MACHINE_H_
