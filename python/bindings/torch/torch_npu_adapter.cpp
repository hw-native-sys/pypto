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

// Optional Torch queue bridge. Simpler execution is reached only through Python Worker.submit.
#include <acl/acl.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"

namespace nb = nanobind;
namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw pypto::ValueError(message);
}

void CheckAcl(aclError result, const char* operation) {
  if (result != ACL_SUCCESS) {
    throw pypto::RuntimeError(std::string(operation) + " failed with ACL error " + std::to_string(result));
  }
}

bool FrameworkAlive() { return c10_npu::NpuSysCtrl::GetInstance().GetInitFlag(); }

uintptr_t BorrowContext(int32_t device_id) {
  Require(FrameworkAlive(), "Cannot borrow a context after torch_npu teardown");
  int32_t current = -1;
  aclrtContext context = nullptr;
  CheckAcl(aclrtGetDevice(&current), "aclrtGetDevice");
  Require(current == device_id, "Kernel lifecycle must borrow the current torch NPU device");
  CheckAcl(aclrtGetCurrentContext(&context), "aclrtGetCurrentContext");
  Require(context != nullptr, "Cannot borrow the current torch NPU context");
  return reinterpret_cast<uintptr_t>(context);
}

void BindContext(uintptr_t context) {
  Require(FrameworkAlive() && context != 0, "Kernel lifecycle cannot use a torn-down framework context");
  CheckAcl(aclrtSetCurrentContext(reinterpret_cast<aclrtContext>(context)), "aclrtSetCurrentContext");
}

c10_npu::NPUStream CallStream(int64_t stream_id, int32_t device_id) {
  auto stream = c10_npu::getCurrentNPUStream(device_id);
  Require(stream.id() == stream_id, "Kernel frame is not on the current NPU stream");
  return stream;
}

// The queue owns only C++ state. The pending Python ticket owns callback and
// all of its Python captures before enqueue; queue-thread destruction must never
// DECREF a Python object without the GIL. Tensor/Storage release is explicit and
// occurs after RunHandle retirement, with failed submissions retained by Python.
struct QueueState {
  c10_npu::NPUStream stream;
  uintptr_t context;
  std::vector<at::Tensor> tensors;
  std::vector<c10::Storage> storages;
  std::atomic<bool> enqueued{false};

  QueueState(c10_npu::NPUStream stream, uintptr_t context) : stream(stream), context(context) {}
};

class QueueTicket {
 public:
  explicit QueueTicket(std::shared_ptr<QueueState> state) : state_(std::move(state)) {}

  void Enqueue(nb::handle callback) {
    Require(!state_->enqueued.exchange(true), "Kernel ticket can only be enqueued once");
    Require(c10_npu::getCurrentNPUStream(state_->stream.device_index()) == state_->stream,
            "Kernel call stream changed before enqueue");
    Require(BorrowContext(state_->stream.device_index()) == state_->context,
            "Kernel borrowed context changed before enqueue");
    auto state = state_;
    PyObject* borrowed_callback = callback.ptr();
    // Protect allocator reuse even if enqueue only partially succeeds.
    std::unordered_set<const c10::StorageImpl*> recorded;
    for (const auto& storage : state->storages) {
      if (recorded.insert(storage.unsafeGetStorageImpl()).second) {
        c10_npu::NPUCachingAllocator::recordStream(storage.data_ptr(), state->stream);
      }
    }
    // RunOpApiV2 may execute inline or enqueue. Release the caller GIL in both
    // cases so a queue callback can acquire it even when the queue is full.
    nb::gil_scoped_release release;
    at_npu::native::OpCommand::RunOpApiV2("PyPTOKernel", [state, borrowed_callback]() -> int {
      std::string error;
      {
        nb::gil_scoped_acquire gil;
        try {
          // The Python callback validates the session's captured context, calls
          // public Worker.submit, and publishes its RunHandle or exception.
          nb::borrow<nb::object>(borrowed_callback)(reinterpret_cast<uintptr_t>(state->stream.stream(false)));
        } catch (nb::python_error& exc) {
          // Never export a Python-owning exception into the framework queue.
          // The original exception is already retained on the pending ticket.
          error = exc.what();
        }
      }
      if (!error.empty()) throw pypto::RuntimeError(error);
      return 0;
    });
  }

  void Synchronize() { state_->stream.synchronize(); }

  void Release() {
    // Python only calls this after RunHandle.wait successfully retires the run.
    state_->tensors.clear();
    state_->storages.clear();
  }

 private:
  std::shared_ptr<QueueState> state_;
};

std::shared_ptr<QueueTicket> Prepare(nb::list objects, int64_t stream_id, int32_t device_id) {
  auto stream = CallStream(stream_id, device_id);
  auto state = std::make_shared<QueueState>(stream, BorrowContext(device_id));
  for (auto object : objects) {
    Require(THPVariable_Check(object.ptr()), "Kernel arguments must be torch tensors");
    at::Tensor tensor = THPVariable_Unpack(object.ptr());
    Require(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device().index() == device_id &&
                tensor.is_contiguous(),
            "Kernel tensors must be contiguous on the current NPU device");
    state->storages.push_back(tensor.storage());
    state->tensors.push_back(std::move(tensor));
  }
  return std::make_shared<QueueTicket>(std::move(state));
}
}  // namespace

NB_MODULE(_torch_npu, m) {
  nb::class_<QueueTicket>(m, "QueueTicket")
      .def("enqueue", &QueueTicket::Enqueue)
      .def("synchronize", &QueueTicket::Synchronize, nb::call_guard<nb::gil_scoped_release>())
      .def("release", &QueueTicket::Release);
  m.def("prepare", &Prepare);
  m.def("framework_alive", &FrameworkAlive);
  m.def("borrow_context", &BorrowContext);
  m.def("bind_context", &BindContext);
  // Intentionally survive interpreter finalization if safe cleanup failed.
  m.def("retain_until_exit", [](nb::object owner) { Py_INCREF(owner.ptr()); });
  m.def("check_call", [](int64_t stream_id, int32_t device_id) {
    auto stream = CallStream(stream_id, device_id);
    aclmdlRICaptureStatus status{};
    aclmdlRI model = nullptr;
    CheckAcl(aclmdlRICaptureGetInfo(stream.stream(false), &status, &model), "aclmdlRICaptureGetInfo");
    Require(status == ACL_MODEL_RI_CAPTURE_STATUS_NONE || status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
            "Kernel initialization encountered an invalidated graph capture");
    return status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
  });
  m.def("storage_shape", [](nb::handle object) {
    Require(THPVariable_Check(object.ptr()), "Storage shape requires a torch tensor");
    const at::Tensor& tensor = THPVariable_Unpack(object.ptr());
    Require(tensor.device().type() == c10::DeviceType::PrivateUse1, "Storage shape requires an NPU tensor");
    return at_npu::native::get_npu_storage_sizes(tensor);
  });
}
