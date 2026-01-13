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

#ifndef PYPTO_RUNTIME_TASK_H_
#define PYPTO_RUNTIME_TASK_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "pypto/runtime/instruction.h"

namespace pypto {
namespace runtime {

/**
 * @brief Task handle for async task tracking
 *
 * Used to wait for task completion and retrieve results.
 */
class TaskHandle {
 public:
  TaskHandle() : completed_(false), result_(std::monostate{}) {}

  /**
   * @brief Wait for task to complete
   */
  void Wait();

  /**
   * @brief Wait for task to complete with pre/post callbacks
   * Callbacks can be used to release/acquire locks (e.g., Python GIL)
   */
  void WaitWithCallbacks(const std::function<void()>& pre_wait, const std::function<void()>& post_wait);

  /**
   * @brief Check if task is completed (non-blocking)
   */
  [[nodiscard]] bool IsCompleted() const;

  /**
   * @brief Set task as completed with result
   */
  void SetCompleted(const Value& result);

  /**
   * @brief Get task result (blocks until completed)
   */
  Value GetResult();

 private:
  std::atomic<bool> completed_;
  Value result_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

using TaskHandlePtr = std::shared_ptr<TaskHandle>;

/**
 * @brief Callable task that executes on AICORE
 *
 * Tasks are Python-style callables for simulation.
 * For real hardware, these would be kernel launches.
 */
using TaskCallable = std::function<Value(const std::vector<Value>&)>;

/**
 * @brief Task to be executed on AICORE
 */
struct Task {
  std::string task_name;    ///< Name of the task
  std::vector<Value> args;  ///< Arguments to the task
  TaskHandlePtr handle;     ///< Handle for completion tracking
  TaskCallable callable;    ///< Callable to execute

  Task(std::string name, std::vector<Value> args_vec, TaskHandlePtr hdl, TaskCallable fn)
      : task_name(std::move(name)),
        args(std::move(args_vec)),
        handle(std::move(hdl)),
        callable(std::move(fn)) {}
};

/**
 * @brief Thread-safe task queue for AICORE dispatch
 */
class TaskQueue {
 public:
  TaskQueue() : stopped_(false) {}

  /**
   * @brief Enqueue a task
   */
  void Enqueue(Task task);

  /**
   * @brief Dequeue a task (blocks if empty)
   * @return Task, or nullopt if queue is stopped
   */
  std::optional<Task> Dequeue();

  /**
   * @brief Stop the queue (unblocks all waiting threads)
   */
  void Stop();

  /**
   * @brief Check if queue is stopped
   */
  bool IsStopped() const;

  /**
   * @brief Get queue size
   */
  size_t Size() const;

 private:
  std::queue<Task> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stopped_;
};

}  // namespace runtime
}  // namespace pypto

#endif  // PYPTO_RUNTIME_TASK_H_
