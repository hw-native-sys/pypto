# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile the URMA async transfer adapter against the A5 runtime headers."""

import os
import shutil
import subprocess

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.backend.pto_backend import (
    _URMA_TRANSFER_ADAPTER_TEMPLATE,
    _URMA_TRANSFER_INCLUDE,
    _generate_urma_transfer_adapter,
)


def test_urma_adapter_compiles_with_a5_runtime(tmp_path):
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("A5 device compilation requires CANN (ASCEND_HOME_PATH)")
    pytest.importorskip("simpler_setup")
    pto_isa_root = os.environ.get("PTO_ISA_ROOT")
    if not pto_isa_root or not os.path.isdir(pto_isa_root):
        pytest.skip("PTO_ISA_ROOT is not available")
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            dst: pld.DistributedTensor[[16, 64], pl.FP32],
            src: pld.DistributedTensor[[16, 64], pl.FP32],
            peer: pl.Scalar[pl.INT32],
        ):
            _out = pld.tensor.put_async(dst, peer=peer, src=src, engine=pld.CommEngine.URMA)
            _back = pld.tensor.get_async(dst, peer=peer, src=src, engine=pld.CommEngine.URMA)

    func = P.get_function("kernel")
    assert func is not None
    adapter = _generate_urma_transfer_adapter(func)
    assert "pypto_urma_put_async_f32" in adapter
    assert "pypto_urma_get_async_f32" in adapter
    assert "pypto_urma_transfer_impl<float, true>" in adapter
    assert "pypto_urma_transfer_impl<float, false>" in adapter

    source = tmp_path / "urma_adapter.cpp"
    source.write_text(
        "#include <pto/pto-inst.hpp>\n"
        + _URMA_TRANSFER_INCLUDE
        + adapter
        + 'extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {\n'
        + "  pypto_urma_put_async_f32(args, args, nullptr, nullptr, 0, 0, 0, 0, 0, 0);\n"
        + "  pypto_urma_get_async_f32(args, args, nullptr, nullptr, 0, 0, 0, 0, 0, 0);\n"
        + "}\n"
    )
    binary = KernelCompiler("a5").compile_incore(
        str(source),
        runtime_name="tensormap_and_ringbuffer",
        pto_isa_root=pto_isa_root,
    )
    assert binary


def test_urma_adapter_accepts_local_memory_outside_remote_domain(tmp_path):
    """Execute the generated guards with CPU stubs; keep endpoint/window bounds."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required")
    preamble = r"""
#include <cassert>
#include <cstdint>
#define __aicore__
#define __gm__
#define __CPU_SIM
constexpr int SIMPLER_ERROR_ASYNC_COMPLETION_INVALID = 1;
constexpr int PIPE_ALL = 0;
int errors = 0, submissions = 0;
uint64_t submitted_dst = 0, submitted_src = 0;
struct AsyncCtx {};
AsyncCtx get_async_ctx(int64_t*) { return {}; }
void pipe_barrier(int) {}
struct Workspace { uint64_t addr = 1; };
struct CommContext {
  uint32_t rankNum = 2, rankId = 0;
  uint64_t windowsIn[2] = {0x10000, 0x20000};
  uint64_t winSize = 256;
  Workspace workspace;
};
enum class CommWorkspaceKind { URMA };
template <CommWorkspaceKind> Workspace& comm_workspace(CommContext& c) { return c.workspace; }
namespace pto2 {
namespace detail {
void set_urma_error(AsyncCtx&, int32_t) { ++errors; }
void defer_flush(AsyncCtx&) {}
}
namespace urma_backend {
template <typename T> T* peer_mr_ptr(uint8_t*, uint32_t peer, uint64_t off) {
  assert(peer == 1);
  return reinterpret_cast<T*>(0x20000 + off);
}
}
}
namespace pto {
constexpr int DYNAMIC = -1;
template <int...> struct Shape { template <typename... A> Shape(A...) {} };
template <int...> struct Stride { template <typename... A> Stride(A...) {} };
template <typename T, typename S, typename R> struct GlobalTensor {
  T* ptr;
  GlobalTensor(T* p, S, R) : ptr(p) {}
};
}
struct Request { uint64_t dst, src; };
template <typename T> Request UrmaTput(T dst, T src, uint8_t*, uint32_t) {
  return {reinterpret_cast<uint64_t>(dst.ptr), reinterpret_cast<uint64_t>(src.ptr)};
}
template <typename T> Request UrmaTget(T dst, T src, uint8_t* w, uint32_t p) {
  return UrmaTput(dst, src, w, p);
}
bool send_request_entry(AsyncCtx&, Request request) {
  ++submissions;
  submitted_dst = request.dst;
  submitted_src = request.src;
  return true;
}
"""
    cases = r"""
int main() {
  CommContext comm;
  int64_t raw_args = 0;
  auto* ctx = reinterpret_cast<int64_t*>(&comm);
  auto* remote = reinterpret_cast<float*>(0x10040);
  // Cover local pointers both below and above the remote domain's window.
  for (uint64_t address : {uint64_t{0x8000}, uint64_t{0x30000}}) {
    auto* local = reinterpret_cast<float*>(address);
    errors = submissions = 0;
    pypto_urma_transfer_impl<float, true>(&raw_args, ctx, remote, local, 4, 8, 32, 1, 64, 64);
    assert(errors == 0 && submissions == 1);
    assert(submitted_dst == 0x20044 && submitted_src == address + 8);
    pypto_urma_transfer_impl<float, false>(&raw_args, ctx, local, remote, 8, 4, 32, 1, 64, 64);
    assert(errors == 0 && submissions == 2);
    assert(submitted_dst == address + 8 && submitted_src == 0x20044);
    // Local view bounds remain enforced in each direction.
    pypto_urma_transfer_impl<float, true>(&raw_args, ctx, remote, local, 0, 40, 32, 1, 64, 64);
    pypto_urma_transfer_impl<float, false>(&raw_args, ctx, local, remote, 40, 0, 32, 1, 64, 64);
    assert(errors == 2 && submissions == 2);
    // Remote view bounds and the selected communication window remain enforced.
    pypto_urma_transfer_impl<float, true>(&raw_args, ctx, remote, local, 40, 0, 32, 1, 64, 64);
    auto* outside = reinterpret_cast<float*>(0x10100);
    pypto_urma_transfer_impl<float, false>(&raw_args, ctx, local, outside, 0, 0, 32, 1, 64, 64);
    assert(errors == 4 && submissions == 2);
  }
}
"""
    source = tmp_path / "adapter_guards.cpp"
    source.write_text(
        "#include <initializer_list>\n"
        + preamble
        + _URMA_TRANSFER_ADAPTER_TEMPLATE.replace("__PYPTO_URMA_WRAPPERS__", "")
        + cases
    )
    binary = tmp_path / "adapter_guards"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
