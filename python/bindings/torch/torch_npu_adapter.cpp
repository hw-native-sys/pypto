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
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <cstdint>
#include <string>
#include <unordered_set>

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

// PendingTicket owns callback and all Tensor/Storage references before this
// function is called. The queue captures only the native stream and a borrowed
// callback pointer; no Python reference is destroyed on an arbitrary queue thread.
void Enqueue(nb::list objects, int64_t stream_id, int32_t device_id, uintptr_t context, nb::handle callback) {
  auto stream = CallStream(stream_id, device_id);
  Require(BorrowContext(device_id) == context, "Kernel borrowed context changed before enqueue");
  std::unordered_set<const c10::StorageImpl*> recorded;
  for (auto object : objects) {
    Require(THPVariable_Check(object.ptr()), "Kernel arguments must be torch tensors");
    const at::Tensor& tensor = THPVariable_Unpack(object.ptr());
    Require(tensor.device().type() == c10::DeviceType::PrivateUse1 && tensor.device().index() == device_id &&
                tensor.is_contiguous(),
            "Kernel tensors must be contiguous on the current NPU device");
    const auto& storage = tensor.storage();
    if (recorded.insert(storage.unsafeGetStorageImpl()).second) {
      c10_npu::NPUCachingAllocator::recordStream(storage.data_ptr(), stream);
    }
  }
  PyObject* borrowed_callback = callback.ptr();
  // Queue admission may block. Release the caller GIL so the queue can enter
  // Python, both when callbacks execute inline and when taskQueue is enabled.
  nb::gil_scoped_release release;
  at_npu::native::OpCommand::RunOpApiV2("PyPTOKernel", [stream, borrowed_callback]() -> int {
    std::string error;
    {
      nb::gil_scoped_acquire gil;
      try {
        nb::borrow<nb::object>(borrowed_callback)(reinterpret_cast<uintptr_t>(stream.stream(false)));
      } catch (nb::python_error& exc) {
        // The Python callback retains the original exception on PendingTicket.
        // Destroy the Python-owning exception while the GIL is still held.
        error = exc.what();
      }
    }
    if (!error.empty()) throw pypto::RuntimeError(error);
    return 0;
  });
}
}  // namespace

NB_MODULE(_torch_npu, m) {
  m.def("enqueue", &Enqueue);
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
