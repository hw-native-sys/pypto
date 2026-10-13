# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile the deferred error path with the runtime's opaque task handles."""

import shutil
import subprocess
from pathlib import Path

import pytest
from pypto.backend.pto_backend import _DEFERRED_COMPLETION_ADAPTER


@pytest.mark.parametrize("runtime_name", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_invalid_task_handle_reports_error_without_fabricating_identity(tmp_path, runtime_name):
    runtime = Path(__file__).resolve().parents[3] / "runtime" / "src" / "common"
    if not (runtime / runtime_name / "task_id.h").is_file():
        pytest.skip("runtime submodule is unavailable")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
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
