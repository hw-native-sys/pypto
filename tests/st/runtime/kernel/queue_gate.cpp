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

/**
 * @file testing.cpp
 * @brief Implementation of Python bindings for testing utilities
 *
 * This module provides internal testing utilities that should not be used
 * in production code. It is exposed as pypto.testing in Python.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

#include "pypto/core/error.h"

namespace nb = nanobind;

namespace {
class QueueGate {
 public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    condition_.notify_all();
    if (!condition_.wait_for(lock, std::chrono::seconds(30), [&] { return released_; })) {
      throw pypto::RuntimeError("Test queue gate timed out");
    }
  }
  void WaitEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!condition_.wait_for(lock, std::chrono::seconds(15), [&] { return entered_; })) {
      throw pypto::RuntimeError("Test callback did not enter the queue gate");
    }
  }
  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool entered_ = false;
  bool released_ = false;
};
}  // namespace

NB_MODULE(_torch_npu_test, m) {
  nb::class_<QueueGate>(m, "QueueGate")
      .def("wait_entered", &QueueGate::WaitEntered, nb::call_guard<nb::gil_scoped_release>())
      .def("release", &QueueGate::Release, nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "block_queue",
      [] {
        auto gate = std::make_shared<QueueGate>();
        at_npu::native::OpCommand::RunOpApiV2("PyPTOTestQueueGate", [gate] {
          gate->Wait();
          return 0;
        });
        return gate;
      },
      nb::call_guard<nb::gil_scoped_release>());
}
