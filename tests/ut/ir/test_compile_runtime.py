# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Selecting the target Simpler runtime ABI for a compilation.

The runtime name travels through ``PassContext`` rather than being a
codegen-only argument, so passes that must legalize IR for a specific runtime
can read it. These tests pin the whole path: the context round-trip, the
precedence ``ir.compile()`` applies, and the value that lands in the generated
``kernel_config.py``.

Note that ``tests/ut/conftest.py`` wraps every unit test in a ``PassContext``,
so ``ir.compile(runtime=...)`` is always a conflict here — selecting the runtime
by wrapping the call in a context is the path these tests exercise, and the one
users take in practice.
"""

import pypto.language as pl
import pytest
from pypto.ir.compile import compile as ir_compile
from pypto.pypto_core import passes

DEFAULT_RUNTIME = passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER
GRAPH_EXECUTION_RUNTIME = passes.RuntimeKind.HOST_BUILD_GRAPH


@pl.program
class VectorAdd:
    """Minimal orchestration entry, so compilation emits a ``kernel_config.py``."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
            tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
            tile_c: pl.Tile[[128, 128], pl.FP32] = pl.add(tile_a, tile_b)
            pl.store(tile_c, [0, 0], c)
        return c


# ---------------------------------------------------------------------------
# PassContext round-trip
# ---------------------------------------------------------------------------


def test_pass_context_default_runtime_is_tensormap_and_ringbuffer():
    ctx = passes.PassContext([])
    assert ctx.get_runtime() == DEFAULT_RUNTIME


@pytest.mark.parametrize("runtime", [DEFAULT_RUNTIME, GRAPH_EXECUTION_RUNTIME])
def test_pass_context_runtime_round_trip(runtime):
    ctx = passes.PassContext([], runtime=runtime)
    assert ctx.get_runtime() == runtime


def test_pass_context_runtime_is_independent_of_memory_planner():
    ctx = passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, runtime=GRAPH_EXECUTION_RUNTIME)
    assert ctx.get_memory_planner() == passes.MemoryPlanner.PTOAS
    assert ctx.get_runtime() == GRAPH_EXECUTION_RUNTIME


# ---------------------------------------------------------------------------
# Validation and precedence
# ---------------------------------------------------------------------------


def test_compile_rejects_runtime_when_a_context_is_active(tmp_path):
    # Same rule as memory_planner: an explicit argument alongside an active
    # context is ambiguous, so it is refused rather than silently ranked.
    with passes.PassContext([], runtime=GRAPH_EXECUTION_RUNTIME):
        with pytest.raises(RuntimeError, match="runtime while a PassContext is already active"):
            ir_compile(
                VectorAdd,
                runtime=DEFAULT_RUNTIME,
                skip_ptoas=True,
                platform="a2a3",
                output_dir=str(tmp_path / "conflict"),
            )


# ---------------------------------------------------------------------------
# The value that lands in the artifact
# ---------------------------------------------------------------------------

# ``kernel_config.py`` is only written when ptoas is not skipped, so these tests
# need ``skip_ptoas=False`` — but they are about the runtime name, not about
# kernel compilation. Stub the ptoas invocation, as the neighbouring codegen
# tests do, so the assertions do not depend on a working ptoas toolchain.
_STUB_PTOAS_OUTPUT = """\
#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void stub_kernel(__gm__ float* v1) {}
"""


@pytest.fixture
def stub_ptoas(monkeypatch):
    monkeypatch.setattr(
        "pypto.backend.pto_backend._compile_pto_module",
        lambda _pto_code, _module_name, _output_dir, _memory_planner=None: _STUB_PTOAS_OUTPUT,
    )


def _compile_and_read_config(out_dir) -> str:
    ir_compile(
        VectorAdd,
        skip_ptoas=False,
        platform="a2a3",
        output_dir=str(out_dir),
        dump_passes=False,
    )
    return (out_dir / "kernel_config.py").read_text()


@pytest.mark.parametrize("runtime", [DEFAULT_RUNTIME, GRAPH_EXECUTION_RUNTIME])
def test_context_runtime_reaches_kernel_config(tmp_path, runtime, stub_ptoas):
    with passes.PassContext([], runtime=runtime):
        config = _compile_and_read_config(tmp_path / passes.runtime_kind_to_name(runtime))
    name = passes.runtime_kind_to_name(runtime)
    assert f'"runtime": "{name}"' in config
    assert f"# Runtime configuration for {name}." in config


def test_compile_without_explicit_runtime_defaults_to_tensormap(tmp_path, stub_ptoas):
    config = _compile_and_read_config(tmp_path / "default")
    assert f'"runtime": "{passes.runtime_kind_to_name(DEFAULT_RUNTIME)}"' in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
