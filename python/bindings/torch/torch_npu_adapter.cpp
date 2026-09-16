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
 * @file torch_npu_adapter.cpp
 * @brief Native queue submission and lifetime management for prepared kernels.
 */

#include <acl/acl.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "worker/chip_worker.h"

namespace nb = nanobind;

namespace {
std::mutex launch_mutex;

void Require(bool condition, const char* message) {
  if (!condition) throw pypto::ValueError(message);
}

// Everything captured by the taskQueue callback is native. Python owns the ticket
// and Worker on the admission side; releasing a callback never acquires the GIL.
struct LaunchState {
  ChipWorker* worker;
  int32_t callable_id;
  c10_npu::NPUStream stream;
  ChipStorageTaskArgs args{};
  aclrtEvent completion = nullptr;
  std::vector<at::Tensor> tensors;
  std::vector<c10::Storage> storages;
  std::atomic<bool> callback_finished{false};
  std::atomic<bool> enqueued{false};
  std::mutex mutex;
  std::exception_ptr error;

  LaunchState(ChipWorker* worker, int32_t callable_id, c10_npu::NPUStream stream)
      : worker(worker), callable_id(callable_id), stream(stream) {}

  ~LaunchState() {
    if (completion != nullptr) aclrtDestroyEvent(completion);
  }

  void ReleaseCompletedOwners() {
    if (completion != nullptr) {
      Require(aclrtDestroyEvent(completion) == ACL_SUCCESS, "Cannot release kernel completion event");
      completion = nullptr;
    }
    tensors.clear();
    storages.clear();
  }

  void CheckError() {
    std::lock_guard<std::mutex> lock(mutex);
    if (error) std::rethrow_exception(error);
  }
};

class LaunchTicket {
 public:
  explicit LaunchTicket(std::shared_ptr<LaunchState> state) : state_(std::move(state)) {}

  void Enqueue() {
    Require(!state_->enqueued.exchange(true), "Kernel ticket can only be enqueued once");
    Require(c10_npu::getCurrentNPUStream(state_->stream.device_index()) == state_->stream,
            "Kernel call stream changed before enqueue");
    auto state = state_;
    // Record before any launch: even a partially enqueued invocation must never
    // lose allocator protection. Deduplicate by StorageImpl, including aliases.
    std::unordered_set<const c10::StorageImpl*> recorded;
    for (const auto& storage : state->storages) {
      if (recorded.insert(storage.unsafeGetStorageImpl()).second) {
        c10_npu::NPUCachingAllocator::recordStream(storage.data_ptr(), state->stream);
      }
    }
    at_npu::native::OpCommand::RunOpApiV2("PyPTOKernel", [state]() -> int {
      try {
        std::lock_guard<std::mutex> launch_lock(launch_mutex);
        state->worker->kernel_launch(state->callable_id, &state->args, state->stream.stream(false));
        Require(aclrtRecordEvent(state->completion, state->stream.stream(false)) == ACL_SUCCESS,
                "Failed to record kernel completion; retain all owners");
      } catch (...) {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->error = std::current_exception();
        state->callback_finished.store(true, std::memory_order_release);
        throw;  // Preserve the framework's synchronous/asynchronous error channel.
      }
      state->callback_finished.store(true, std::memory_order_release);
      return 0;
    });
  }

  bool Done() {
    c10_npu::NPUGuard device_guard(state_->stream.device_index());
    state_->CheckError();
    if (state_->completion == nullptr) return true;
    if (!state_->callback_finished.load(std::memory_order_acquire)) return false;
    aclrtEventRecordedStatus status{};
    Require(aclrtQueryEventStatus(state_->completion, &status) == ACL_SUCCESS,
            "Failed to query kernel completion; retain all owners");
    const bool done = status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
    state_->CheckError();
    if (done) state_->ReleaseCompletedOwners();
    return done;
  }

  void Quiesce() {
    c10_npu::NPUGuard device_guard(state_->stream.device_index());
    // Error-only cleanup: a caller-stream fence can be missing after partial
    // enqueue. Drain the host queue first, then prove *all* device streams idle.
    try {
      state_->stream.synchronize();
    } catch (...) {
      // A sticky framework queue error is not itself proof of live device work.
    }
    Require(state_->callback_finished.load(std::memory_order_acquire),
            "Cannot release a kernel ticket whose host callback has not finished");
    Require(aclrtSynchronizeDevice() == ACL_SUCCESS,
            "Cannot establish device quiescence after failed kernel submission; owners retained");
    state_->ReleaseCompletedOwners();
  }

  void Wait() {
    c10_npu::NPUGuard device_guard(state_->stream.device_index());
    // Framework synchronization drains its host queue as well as device work.
    // If either fails, the manager retains this ticket and all its owners.
    state_->stream.synchronize();
    state_->CheckError();
    Require(state_->callback_finished.load(std::memory_order_acquire),
            "Kernel callback did not finish; submission was rejected");
    state_->ReleaseCompletedOwners();
  }

 private:
  std::shared_ptr<LaunchState> state_;
};

bool FrameworkAlive() { return c10_npu::NpuSysCtrl::GetInstance().GetInitFlag(); }

uintptr_t BorrowContext(int32_t device_id) {
  Require(FrameworkAlive(), "Cannot borrow a context after torch_npu teardown");
  int32_t current = -1;
  aclrtContext context = nullptr;
  Require(aclrtGetDevice(&current) == ACL_SUCCESS && current == device_id,
          "Kernel lifecycle must borrow the current torch NPU device");
  Require(aclrtGetCurrentContext(&context) == ACL_SUCCESS && context != nullptr,
          "Cannot borrow the current torch NPU context");
  return reinterpret_cast<uintptr_t>(context);
}

void BindContext(uintptr_t context) {
  Require(FrameworkAlive() && context != 0, "Kernel lifecycle cannot use a torn-down framework context");
  Require(aclrtSetCurrentContext(reinterpret_cast<aclrtContext>(context)) == ACL_SUCCESS,
          "Cannot bind the borrowed framework context to the kernel lifecycle thread");
}

c10_npu::NPUStream EagerStream(int64_t stream_id, int32_t device_id) {
  auto stream = c10_npu::getCurrentNPUStream(device_id);
  Require(stream.id() == stream_id, "Kernel frame is not on the current NPU stream");
  aclmdlRICaptureStatus capture_status{};
  aclmdlRI model = nullptr;
  Require(aclmdlRICaptureGetInfo(stream.stream(false), &capture_status, &model) == ACL_SUCCESS,
          "Cannot query capture state for PyPTO eager submission");
  Require(capture_status == ACL_MODEL_RI_CAPTURE_STATUS_NONE,
          "PyPTO eager kernel submission does not support graph capture");
  return stream;
}

std::shared_ptr<LaunchTicket> Prepare(ChipWorker* worker, int32_t callable_id, nb::list objects,
                                      const std::vector<uint32_t>& dtypes,
                                      const std::vector<uint64_t>& scalar_bits, int64_t stream_id,
                                      int32_t device_id) {
  Require(worker && worker->initialized() && worker->device_id() == device_id,
          "Kernel Worker must be live on the call device");
  Require(callable_id >= 0, "Expected a prepared kernel callable id");
  Require(objects.size() == dtypes.size() && objects.size() <= CHIP_MAX_TENSOR_ARGS &&
              scalar_bits.size() <= CHIP_MAX_SCALAR_ARGS,
          "Kernel argument pools exceed the pinned native ABI");
  auto stream = EagerStream(stream_id, device_id);
  auto state = std::make_shared<LaunchState>(worker, callable_id, stream);
  for (size_t i = 0; i < objects.size(); ++i) {
    Require(THPVariable_Check(objects[i].ptr()), "Kernel arguments must be torch tensors");
    at::Tensor tensor = THPVariable_Unpack(objects[i].ptr());
    Require(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device().index() == device_id &&
                tensor.is_contiguous(),
            "Kernel tensors must be contiguous on the call NPU");
    Require(tensor.dim() > 0 && tensor.dim() <= MAX_TENSOR_DIMS && dtypes[i] <= 14,
            "Kernel tensor rank or dtype is outside the pinned ABI");
    ChipTensor view{};
    view.buffer = {reinterpret_cast<uint64_t>(tensor.data_ptr()), static_cast<uint64_t>(tensor.nbytes())};
    view.ndims = static_cast<uint32_t>(tensor.dim());
    view.dtype = static_cast<DataType>(dtypes[i]);
    view.address_space = AddressSpace::DEVICE;
    view.start_offset = 0;  // data_ptr already includes the logical storage offset.
    for (int64_t d = 0; d < tensor.dim(); ++d) {
      Require(tensor.size(d) > 0 && tensor.size(d) <= UINT32_MAX && tensor.stride(d) > 0 &&
                  tensor.stride(d) <= UINT32_MAX,
              "Kernel launch requires positive u32 tensor extents and strides");
      view.shapes[d] = static_cast<uint32_t>(tensor.size(d));
      view.strides[d] = static_cast<uint32_t>(tensor.stride(d));
    }
    state->args.add_tensor(view);
    state->storages.push_back(tensor.storage());
    state->tensors.push_back(std::move(tensor));
  }
  for (auto value : scalar_bits) state->args.add_scalar(value);
  Require(aclrtCreateEvent(&state->completion) == ACL_SUCCESS, "Cannot create kernel completion event");
  return std::make_shared<LaunchTicket>(std::move(state));
}
}  // namespace

NB_MODULE(_torch_npu, m) {
  m.attr("simpler_revision") = PYPTO_SIMPLER_REVISION;
  nb::class_<LaunchTicket>(m, "LaunchTicket")
      .def("enqueue", &LaunchTicket::Enqueue, nb::call_guard<nb::gil_scoped_release>())
      .def("done", &LaunchTicket::Done, nb::call_guard<nb::gil_scoped_release>())
      .def("wait", &LaunchTicket::Wait, nb::call_guard<nb::gil_scoped_release>())
      .def("quiesce", &LaunchTicket::Quiesce, nb::call_guard<nb::gil_scoped_release>());
  m.def("framework_alive", &FrameworkAlive);
  m.def("borrow_context", &BorrowContext);
  m.def("bind_context", &BindContext);
  // Deliberately never decref: unsafe shutdown must not run native destructors
  // during later Python module clearing, after the framework has destroyed ACL.
  m.def("retain_until_exit", [](nb::object owner) { Py_INCREF(owner.ptr()); });
  m.def("check_eager", [](int64_t stream_id, int32_t device_id) { EagerStream(stream_id, device_id); });
  m.def("prepare", &Prepare, nb::keep_alive<0, 1>());
}
