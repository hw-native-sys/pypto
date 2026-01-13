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

#include "pypto/runtime/task.h"

namespace pypto {
namespace runtime {

// TaskHandle implementation

void TaskHandle::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return completed_.load(); });
}

void TaskHandle::WaitWithCallbacks(const std::function<void()>& pre_wait, const std::function<void()>& post_wait) {
  if (pre_wait) {
    pre_wait();
  }

  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return completed_.load(); });

  if (post_wait) {
    post_wait();
  }
}

bool TaskHandle::IsCompleted() const {
  return completed_.load();
}

void TaskHandle::SetCompleted(const Value& result) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    result_ = result;
    completed_.store(true);
  }
  cv_.notify_all();
}

Value TaskHandle::GetResult() {
  Wait();
  std::lock_guard<std::mutex> lock(mutex_);
  return result_;
}

// TaskQueue implementation

void TaskQueue::Enqueue(Task task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
      return;
    }
    queue_.push(std::move(task));
  }
  cv_.notify_one();
}

std::optional<Task> TaskQueue::Dequeue() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });

  if (stopped_ && queue_.empty()) {
    return std::nullopt;
  }

  Task task = std::move(queue_.front());
  queue_.pop();
  return task;
}

void TaskQueue::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_.store(true);
  }
  cv_.notify_all();
}

bool TaskQueue::IsStopped() const {
  return stopped_.load();
}

size_t TaskQueue::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

}  // namespace runtime
}  // namespace pypto
