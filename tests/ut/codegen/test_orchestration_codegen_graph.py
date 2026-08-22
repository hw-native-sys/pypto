# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Orchestration codegen for ``FunctionType.Graph``.

A Graph function is emitted once as a **named file-scope function**, and each
call site launches it with ``rt_submit_graph``. Named rather than an inlined
lambda because the runtime identifies a graph by a bare function pointer: a
lambda would mint one pointer per syntactic occurrence and burn through the
16-entry Definition cache.

The assertions that matter most are the ones covering silent failures. A
boundary scalar bound by value instead of by reference severs the pointer
identity the runtime uses to track it, and the value is frozen at its first-call
number on every later replay — with no warning anywhere.
"""

import tempfile
from pathlib import Path
from unittest import mock

import pypto.language as pl
import pytest
from pypto.ir.compile import compile as ir_compile
from pypto.pypto_core import passes

# ``kernel_config.py`` is only written when ptoas is not skipped, but these tests
# are about the emitted orchestration and manifest, not kernel compilation. Stub
# the ptoas invocation as the neighbouring codegen tests do.
_STUB_PTOAS_OUTPUT = """\
#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void stub_kernel(__gm__ float* v1) {}
"""


@pl.program
class Decoder:
    """One recordable layer, launched four times from the entry."""

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        layer_idx: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        base = layer_idx * 128
        with pl.at(level=pl.Level.CORE_GROUP):
            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [base, 0], [128, 128])
            pl.store(t, [0, 0], c)
        return c

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[512, 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        for i in pl.range(4):
            c = self.layer(a, c, i)
        return c


@pytest.fixture(scope="module")
def artifacts() -> dict[str, str]:
    """Compile Decoder for host_build_graph and return the emitted files."""
    out_dir = tempfile.mkdtemp()
    with (
        mock.patch(
            "pypto.backend.pto_backend._compile_pto_module",
            lambda _code, _name, _dir, _planner=None: _STUB_PTOAS_OUTPUT,
        ),
        passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH),
    ):
        ir_compile(
            Decoder,
            skip_ptoas=False,
            platform="a2a3",
            output_dir=out_dir,
            dump_passes=False,
        )
    root = Path(out_dir)
    return {
        str(path.relative_to(root)): path.read_text()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".py"}
    }


@pytest.fixture(scope="module")
def orch(artifacts) -> str:
    return artifacts["orchestration/main.cpp"]


def _graph_body(orch: str) -> str:
    """The emitted graph function, up to the entry that follows it."""
    start = orch.index("static void pypto_graph_layer")
    return orch[start : orch.index("aicpu_orchestration_entry")]


def _entry_body(orch: str) -> str:
    return orch[orch.index("aicpu_orchestration_entry") :]


# ---------------------------------------------------------------------------
# The emitted graph function
# ---------------------------------------------------------------------------


def test_graph_is_a_named_file_scope_function(orch):
    assert "static void pypto_graph_layer(const CoreTaskArgs& args) {" in orch


def test_boundary_tensors_are_bound_from_the_task_args(orch):
    body = _graph_body(orch)
    assert "const ChipTensor& a = args.tensor(0).ref();" in body
    assert "const ChipTensor& c = args.tensor(1).ref();" in body


def test_boundary_scalars_are_bound_by_reference(orch):
    """The single most consequential line in the whole feature.

    The runtime tracks a boundary scalar by the *address* of its argument slot.
    Copying it into a local (``uint64_t base = args.scalar(1);``) severs that
    link, so the value is frozen at the first call's number and silently reused
    on every replay.
    """
    body = _graph_body(orch)
    assert "const uint64_t& layer_idx = args.scalar(0);" in body
    assert "const uint64_t& base = args.scalar(1);" in body


def test_graph_body_allocates_nothing(orch):
    # A bare alloc_tensors inside the region poisons the recording outright.
    assert "alloc_tensors(" not in _graph_body(orch)


def test_graph_body_task_vars_do_not_collide_with_the_entry(orch):
    # The graph body is emitted by a second codegen instance whose counters
    # restart at 0; without a prefix both would declare `params_t0`.
    assert "g0_params_t0" in _graph_body(orch)
    assert "params_t0" in _entry_body(orch)


# ---------------------------------------------------------------------------
# The call site
# ---------------------------------------------------------------------------


def test_call_site_submits_the_graph_by_key_and_pointer(orch):
    assert "rt_submit_graph(&pypto_graph_layer," in _entry_body(orch)


def test_derived_scalar_is_computed_at_the_call_site(orch):
    """LegalizeGraphBoundary moved ``base = layer_idx * 128`` out here.

    Inside the region it would have had no argument slot; out here it is an
    ordinary pass-through scalar the runtime can patch on replay.
    """
    entry = _entry_body(orch)
    assert "(i * 128)" in entry
    assert entry.count("add_scalar") == 2  # layer_idx and the hoisted base


def test_graph_result_is_not_chained(orch):
    # rt_submit_graph yields a valid task id only on a cache hit, so the call
    # site must not bind or depend on its result.
    entry = _entry_body(orch)
    assert "= rt_submit_graph" not in entry


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_graph_is_not_a_kernel(artifacts):
    """A Graph has no .cpp under kernels/, so it must not reach KERNELS.

    Registering it would put an entry in the manifest pointing at a file that
    does not exist, which fails at runtime rather than at build time.
    """
    config = artifacts["kernel_config.py"]
    assert "pypto_graph_layer" not in config
    assert '"name": "layer"' not in config
    # The real kernel outlined out of the graph body is still listed.
    assert "layer_incore_0" in config


def test_artifact_targets_the_graph_runtime(artifacts):
    assert '"runtime": "host_build_graph"' in artifacts["kernel_config.py"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
