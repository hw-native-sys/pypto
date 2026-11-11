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

// Test-only native gate: every queue operation and reference release is GIL-free.
#include <Python.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "worker/native_submission.h"

namespace {
struct Gate {
  std::atomic<int> refs{1};
  std::atomic<bool> started{false};
  std::atomic<bool> finished{false};
  std::mutex mutex;
  std::condition_variable ready;
  bool open{false};
  SimplerNativeSubmission api;

  Gate() : api{this, Retain, Release, Submit} {}
  static void Retain(void* context) noexcept { ++static_cast<Gate*>(context)->refs; }
  static void Release(void* context) noexcept {
    auto* gate = static_cast<Gate*>(context);
    if (--gate->refs == 0) delete gate;
  }
  static int Submit(void* context, char*, size_t) noexcept {
    auto* gate = static_cast<Gate*>(context);
    gate->started = true;
    std::unique_lock<std::mutex> lock(gate->mutex);
    if (!gate->ready.wait_for(lock, std::chrono::seconds(10), [gate] { return gate->open; })) {
      return -1;
    }
    gate->finished = true;
    return 0;
  }
};

Gate* Get(PyObject* capsule) {
  auto* api =
      static_cast<SimplerNativeSubmission*>(PyCapsule_GetPointer(capsule, SIMPLER_NATIVE_SUBMISSION_CAPSULE));
  return api ? static_cast<Gate*>(api->context) : nullptr;
}
}  // namespace

extern "C" PyObject* make_gate() {
  auto* gate = new Gate();
  return PyCapsule_New(&gate->api, SIMPLER_NATIVE_SUBMISSION_CAPSULE, [](PyObject* capsule) {
    if (auto* gate = Get(capsule)) Gate::Release(gate);
  });
}

extern "C" int gate_started(PyObject* capsule) { return Get(capsule)->started; }
extern "C" int gate_finished(PyObject* capsule) { return Get(capsule)->finished; }

extern "C" void open_gate_later(PyObject* capsule) {
  auto* gate = Get(capsule);
  Gate::Retain(gate);
  std::thread([gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    {
      std::lock_guard<std::mutex> lock(gate->mutex);
      gate->open = true;
    }
    gate->ready.notify_all();
    Gate::Release(gate);
  }).detach();
}

// Called through ctypes.PyDLL: deliberately retain the GIL while waiting.
extern "C" int wait_finished_with_gil(PyObject* capsule) {
  auto* gate = Get(capsule);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!gate->finished && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return gate->finished;
}
