# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile the deferred error path with the runtime's opaque task handles."""

import re
import shutil
import subprocess
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import codegen, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend._ptoas_preprocess import preprocess_ptoas_output
from pypto.backend.pto_backend import _DEFERRED_COMPLETION_ADAPTER, _compile_pto_module


def _generated_deferred_cpp(output_dir):
    """Use production codegen and PTOAS to obtain the adapter's declaration."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")

    @pl.program
    class DeferredProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def deferred_kernel(
            self,
            signal: pld.DistributedTensor[[4, 8], pl.INT32],
            expected: pl.Scalar[pl.INT32],
        ):
            # Production outlining stamps this marker on a validated waiter.
            pl.func_attr({"deferred_completion_waiter": True})
            pld.system.defer_wait(signal, offsets=[1, 2], expected=expected, cmp=pld.WaitCmp.Ge)

    program = passes.materialize_dist_tensor_ctx()(passes.convert_to_ssa()(DeferredProgram))
    mlir = codegen.PTOCodegen().generate(program)
    return preprocess_ptoas_output(_compile_pto_module(mlir, "deferred_kernel", str(output_dir)))


@pytest.mark.parametrize("runtime_name", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize(
    "with_ptoas_declaration", [False, pytest.param(True, marks=pytest.mark.ptoas_compile)]
)
def test_invalid_task_handle_reports_error_without_fabricating_identity(
    tmp_path, runtime_name, with_ptoas_declaration, ascend_backend
):
    runtime = Path(__file__).resolve().parents[3] / "runtime" / "src" / "common"
    if not (runtime / runtime_name / "task_id.h").is_file():
        pytest.skip("runtime submodule is unavailable")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    # PTOAS emits this prototype for the private IR declaration. The wrapper
    # inserts its adapter definition separately, before the processed output.
    declaration = ""
    if with_ptoas_declaration:
        # This isolated task-handle test has no PTO-ISA types or intrinsics.
        # Keep the emitted declaration verbatim; the SDK test below compiles
        # the full processed output, including PTOAS's hardware helpers.
        generated = _generated_deferred_cpp(tmp_path)
        match = re.search(
            r'extern "C"\s+__aicore__\s+void\s+pypto_register_counter_completion\s*\([^;{}]*\)\s*;',
            generated,
        )
        assert match is not None, generated
        declaration = match.group(0)
    source = tmp_path / "adapter.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include "task_id.h"
#define __aicore__
#define __gm__
#define __CPU_SIM
#define __CCE_KT_TEST__
constexpr int MAX_COMPLETIONS_PER_TASK = 64;
constexpr int SIMPLER_ERROR_ASYNC_COMPLETION_INVALID = -9;
constexpr int PIPE_ALL = 0;
struct AsyncCtx {
    uint32_t* completion_count;
    int32_t* completion_error_code;
    void* completion_entries;
    uint32_t completion_capacity;
    TaskId task_token;
};
static uint32_t count = 7;
static int32_t error = 0;
static AsyncCtx original{&count, &error, nullptr, 0, TaskId::invalid()};
static int flushes = 0;
static int fences = 0;
AsyncCtx get_async_ctx(int64_t*) { return original; }
bool async_ctx_is_deferred(const AsyncCtx& ctx) { return ctx.task_token.is_valid(); }
void dsb(int) { ++fences; }
void pipe_barrier(int) { ++fences; }
namespace pto2::detail {
void defer_flush_range(volatile void* addr, uint32_t size) {
    assert(size == 4);
    assert(addr == &count || addr == &error);
    ++flushes;
}
void defer_flush(AsyncCtx& ctx) { assert(ctx.task_token.is_valid()); }
void defer_error(AsyncCtx& ctx, int32_t code) { *ctx.completion_error_code = code; }
}
void save_expected_notification_counter(AsyncCtx&, volatile void*, uint32_t) {}
"""
        + _DEFERRED_COMPLETION_ADAPTER
        + declaration
        + r"""
int main() {
    pypto_register_counter_completion(nullptr, nullptr, 0, 0);
    assert(count == 0);
    assert(error == SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    assert(!original.task_token.is_valid());
    assert(flushes == 2 && fences == 2);
}
"""
    )
    binary = tmp_path / "adapter"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-I{runtime / runtime_name}",
            f"-I{runtime / 'task_interface'}",
            str(source),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
@pytest.mark.parametrize("runtime_name", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.ptoas_compile
def test_adapter_compiles_with_runtime_async_api(tmp_path, platform, runtime_name, ascend_backend):
    pytest.importorskip("simpler_setup")
    if shutil.which("g++-15") is None:
        pytest.skip("GCC 15 is required for the real CPU simulator compiler")
    from pypto.runtime.kernel_compiler import KernelCompiler  # noqa: PLC0415

    # The opaque-handle test above isolates the error path. Compile against the
    # installed SDK too, so AsyncCtx, slab, and helper API drift cannot hide
    # behind its minimal stand-ins when the runtime gitlink is advanced.
    declaration = _generated_deferred_cpp(tmp_path)
    source = tmp_path / "adapter.cpp"
    source.write_text(
        '#include <pto/pto-inst.hpp>\n#include "async_kernel_api.h"\n'
        + _DEFERRED_COMPLETION_ADAPTER
        + declaration
        + 'extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {\n'
        + "    pypto_register_counter_completion(args, reinterpret_cast<int32_t*>(args[0]), 0, 1);\n"
        + "}\n"
    )
    binary = KernelCompiler(platform).compile_incore(str(source), runtime_name=runtime_name)
    assert binary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
