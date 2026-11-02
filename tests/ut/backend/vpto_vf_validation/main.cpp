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

// Check actual device computations and untouched tail lanes across VF calls.
#include <array>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>

#include "acl/acl.h"
#include "tensormap_and_ringbuffer/tensor.h"

using simpler::tmr::Tensor;

extern "C" void launch_pypto_vf(int64_t*, void*);

static void check(aclError result, const char* operation) {
  if (result != ACL_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(result));
    std::exit(1);
  }
}

static int parse_nonnegative(const char* text) {
  char* end = nullptr;
  errno = 0;
  const auto value = std::strtol(text, &end, 10);
  if (errno || end == text || *end || value < 0 || value > INT_MAX) {
    std::fprintf(stderr, "Invalid nonnegative integer: %s\n", text);
    std::exit(2);
  }
  return static_cast<int>(value);
}

int main(int argc, char** argv) {
  // The simulator defaults to device zero; hardware pytest always passes both values.
  if (argc != 1 && argc != 3) {
    std::fprintf(stderr, "Usage: %s [device_id repetitions]\n", argv[0]);
    return 2;
  }
  const int device_id = argc == 3 ? parse_nonnegative(argv[1]) : 0;
  const int repetitions = argc == 3 ? parse_nonnegative(argv[2]) : 1;
  if (repetitions == 0) {
    return 2;
  }
  constexpr int kCount = 1024;
  constexpr int kOffset = 32;
  constexpr int kStorageCount = kCount + 2 * kOffset;
  constexpr size_t kBytes = kStorageCount * sizeof(float);
  constexpr float kSentinel = -777.0f;
  std::array<float, kStorageCount> lhs{}, rhs{}, output{};
  float *device_lhs = nullptr, *device_rhs = nullptr, *device_out = nullptr;
  aclrtStream stream = nullptr;
  check(aclInit(nullptr), "aclInit");
  check(aclrtSetDevice(device_id), "aclrtSetDevice");
  check(aclrtCreateStream(&stream), "aclrtCreateStream");
  check(aclrtMalloc(reinterpret_cast<void**>(&device_lhs), kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc lhs");
  check(aclrtMalloc(reinterpret_cast<void**>(&device_rhs), kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc rhs");
  check(aclrtMalloc(reinterpret_cast<void**>(&device_out), kBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc out");
  std::array<Tensor, 3> tensors{};
  constexpr uint32_t shapes[] = {1, kCount};
  const std::array<float*, 3> buffers = {device_lhs, device_rhs, device_out};
  Tensor* device_tensors = nullptr;
  int64_t* device_args = nullptr;
  check(aclrtMalloc(reinterpret_cast<void**>(&device_tensors), sizeof(tensors), ACL_MEM_MALLOC_HUGE_FIRST),
        "malloc tensors");
  check(aclrtMalloc(reinterpret_cast<void**>(&device_args), 4 * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST),
        "malloc args");
  for (size_t i = 0; i < tensors.size(); ++i) {
    tensors[i].init_external(buffers[i], kBytes, shapes, 2, DataType::FLOAT32, 0);
    tensors[i].start_offset = kOffset;
  }
  check(aclrtMemcpy(device_tensors, sizeof(tensors), tensors.data(), sizeof(tensors),
                    ACL_MEMCPY_HOST_TO_DEVICE),
        "copy tensors");
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    for (int i = 0; i < kStorageCount; ++i) {
      lhs[i] = static_cast<float>(i % 31) * 0.25f - 4.0f + repeat;
      rhs[i] = static_cast<float>(i % 17) * 0.5f + 2.0f - 2 * repeat;
    }
    check(aclrtMemcpy(device_lhs, kBytes, lhs.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy lhs");
    check(aclrtMemcpy(device_rhs, kBytes, rhs.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy rhs");
    for (int n : {0, 1, 63, 64, 65, 127, 128, 1023, 1024}) {
      output.fill(kSentinel);
      check(aclrtMemcpy(device_out, kBytes, output.data(), kBytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "reset output");
      const std::array<int64_t, 4> args = {reinterpret_cast<int64_t>(device_tensors),
                                           reinterpret_cast<int64_t>(device_tensors + 1),
                                           reinterpret_cast<int64_t>(device_tensors + 2), n};
      check(aclrtMemcpy(device_args, sizeof(args), args.data(), sizeof(args), ACL_MEMCPY_HOST_TO_DEVICE),
            "copy args");
      launch_pypto_vf(device_args, stream);
      check(aclrtSynchronizeStream(stream), "synchronize");
      check(aclrtMemcpy(output.data(), kBytes, device_out, kBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
      for (int i = 0; i < kStorageCount; ++i) {
        const float expected = i >= kOffset && i < kOffset + n ? lhs[i] + rhs[i] : kSentinel;
        if (output[i] != expected) {
          std::fprintf(stderr, "FAIL repeat=%d n=%d index=%d actual=%g expected=%g\n", repeat, n, i,
                       output[i], expected);
          return 2;
        }
      }
      std::printf("PASS n=%d: active lanes and unchanged tail\n", n);
      std::fflush(stdout);
    }
  }
  check(aclrtFree(device_args), "free args");
  check(aclrtFree(device_tensors), "free tensors");
  check(aclrtFree(device_lhs), "free lhs");
  check(aclrtFree(device_rhs), "free rhs");
  check(aclrtFree(device_out), "free out");
  check(aclrtDestroyStream(stream), "destroy stream");
  check(aclrtResetDevice(device_id), "reset device");
  check(aclFinalize(), "aclFinalize");
}
