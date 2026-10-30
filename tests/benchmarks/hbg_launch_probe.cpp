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

// Standalone Linux LD_PRELOAD diagnostic, never linked into the product. Build
// against the installed CANN headers, e.g. with CANN=$ASCEND_HOME_PATH:
// g++ -std=c++17 -O2 -fPIC -shared -I "$CANN/pkg_inc" -I "$CANN/pkg_inc/profiling"
//   -I "$CANN/pkg_inc/runtime" -I "$CANN/pkg_inc/runtime/runtime"
//   tests/benchmarks/hbg_launch_probe.cpp -ldl -o /tmp/hbg_launch_probe.so
// PYPTO_LAUNCH_PROBE_OUTPUT must name a new CSV file. Records stay in bounded
// memory until normal process exit; a missing footer means an incomplete run.
// Run only in a separate diagnostic pass: timestamps, TLS and atomics perturb
// the enclosing submission even though logging is outside the measured APIs.
#include <dlfcn.h>
#include <fcntl.h>
#include <runtime/kernel.h>
#include <runtime/rts/rts_kernel.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr uint64_t kCapacity = 65536;

struct Record {
  std::atomic<bool> ready{false};
  const char* api = nullptr;
  uint64_t start_ns = 0;
  uint64_t duration_ns = 0;
  uintptr_t stream = 0;
  int64_t tid = 0;
  uint32_t blocks = 0;
  rtError_t result = RT_ERROR_NONE;
};

Record records[kCapacity];
std::atomic<uint64_t> record_count{0};
int output_fd = -1;
pid_t owner_pid = 0;

[[noreturn]] void Fail(const char* message) {
  std::fprintf(stderr, "hbg_launch_probe: %s\n", message);
  std::_Exit(127);
}

void Write(const char* data, size_t size) {
  while (size != 0) {
    const ssize_t written = write(output_fd, data, size);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) Fail("cannot write diagnostic output");
    data += written;
    size -= static_cast<size_t>(written);
  }
}

uint64_t Now() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) Fail("CLOCK_MONOTONIC failed");
  return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + static_cast<uint64_t>(value.tv_nsec);
}

template <typename Function>
Function Resolve(const char* symbol) {
  dlerror();
  void* address = dlsym(RTLD_NEXT, symbol);
  const char* error = dlerror();
  if (error != nullptr || address == nullptr) {
    // A Python extension may have loaded the runtime in a local ELF scope.
    // Look only for an already-loaded runtime: never initialize a second one.
    void* runtime = dlopen("libruntime.so", RTLD_NOW | RTLD_NOLOAD);
    if (runtime == nullptr) Fail(dlerror());
    dlerror();
    address = dlsym(runtime, symbol);
    error = dlerror();
    if (error != nullptr) Fail(error);
    // Retain this loader reference until exit so the cached function stays live.
  }
  if (address == nullptr) Fail(symbol);
  return reinterpret_cast<Function>(address);
}

template <typename Function>
rtError_t Observe(const char* api, rtStream_t stream, uint32_t blocks, Function call) {
  if (output_fd < 0 || getpid() != owner_pid) return call();
  const int incoming_errno = errno;
  const uint64_t index = record_count.fetch_add(1, std::memory_order_relaxed);
  if (index >= kCapacity) {
    errno = incoming_errno;
    return call();
  }
  // Resolve the thread identity before opening the interval. The API return
  // code and errno are forwarded unchanged; records carry no tensor data.
  const int64_t tid = syscall(SYS_gettid);
  errno = incoming_errno;
  const uint64_t start = Now();
  const rtError_t result = call();
  const int outgoing_errno = errno;
  const uint64_t end = Now();
  Record& record = records[index];
  record.api = api;
  record.start_ns = start;
  record.duration_ns = end - start;
  record.stream = reinterpret_cast<uintptr_t>(stream);
  record.tid = tid;
  record.blocks = blocks;
  record.result = result;
  record.ready.store(true, std::memory_order_release);
  errno = outgoing_errno;
  return result;
}

__attribute__((constructor)) void Open() {
  const char* path = std::getenv("PYPTO_LAUNCH_PROBE_OUTPUT");
  if (path == nullptr || path[0] == '\0') return;
  // Refuse overwrites so a new process cannot silently replace earlier evidence.
  output_fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (output_fd < 0) Fail("cannot create PYPTO_LAUNCH_PROBE_OUTPUT");
  owner_pid = getpid();
  // Touch the record pages before any launch, keeping first-touch faults out of
  // the surrounding submission measurements as far as practical.
  for (auto& record : records) record.ready.store(false, std::memory_order_relaxed);
}

__attribute__((destructor)) void Close() {
  // Fork children inherit the descriptor and buffers but never own the trace.
  // The harness also removes the environment before spawning exec children.
  if (output_fd < 0 || getpid() != owner_pid) return;
  const char header[] = "api,pid,tid,start_ns,duration_ns,stream,blocks,result\n";
  Write(header, sizeof(header) - 1);
  const uint64_t total = record_count.load(std::memory_order_acquire);
  const uint64_t retained = total < kCapacity ? total : kCapacity;
  uint64_t incomplete = 0;
  char line[256];
  for (uint64_t i = 0; i < retained; ++i) {
    const Record& record = records[i];
    if (!record.ready.load(std::memory_order_acquire)) {
      ++incomplete;
      continue;
    }
    const int size = std::snprintf(
        line, sizeof(line), "%s,%" PRId64 ",%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%" PRIuPTR ",%u,%d\n",
        record.api, static_cast<int64_t>(owner_pid), record.tid, record.start_ns, record.duration_ns,
        record.stream, record.blocks, static_cast<int>(record.result));
    if (size < 0 || static_cast<size_t>(size) >= sizeof(line)) Fail("record formatting failed");
    Write(line, static_cast<size_t>(size));
  }
  const int size = std::snprintf(
      line, sizeof(line), "# total=%" PRIu64 " dropped=%" PRIu64 " incomplete=%" PRIu64 " complete=1\n",
      total, total - retained, incomplete);
  if (size < 0 || static_cast<size_t>(size) >= sizeof(line)) Fail("footer formatting failed");
  Write(line, static_cast<size_t>(size));
  if (close(output_fd) != 0) Fail("cannot close diagnostic output");
  output_fd = -1;
}
}  // namespace

extern "C" rtError_t rtKernelLaunchWithHandleV2(void* handle, const uint64_t tiling_key, uint32_t blocks,
                                                rtArgsEx_t* args, rtSmDesc_t* sm_desc, rtStream_t stream,
                                                const rtTaskCfgInfo_t* config) {
  const int incoming_errno = errno;
  static const auto launch = Resolve<decltype(&rtKernelLaunchWithHandleV2)>("rtKernelLaunchWithHandleV2");
  errno = incoming_errno;
  return Observe("rtKernelLaunchWithHandleV2", stream, blocks,
                 [&] { return launch(handle, tiling_key, blocks, args, sm_desc, stream, config); });
}

extern "C" rtError_t rtsLaunchCpuKernel(const rtFuncHandle handle, uint32_t blocks, rtStream_t stream,
                                        const rtKernelLaunchCfg_t* config, rtCpuKernelArgs_t* args) {
  const int incoming_errno = errno;
  static const auto launch = Resolve<decltype(&rtsLaunchCpuKernel)>("rtsLaunchCpuKernel");
  errno = incoming_errno;
  return Observe("rtsLaunchCpuKernel", stream, blocks,
                 [&] { return launch(handle, blocks, stream, config, args); });
}
