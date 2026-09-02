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

import os
import re
import shutil
import subprocess
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
def artifact_root() -> Path:
    """Compile Decoder for host_build_graph and return the output directory."""
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
    return Path(out_dir)


@pytest.fixture(scope="module")
def artifacts(artifact_root) -> dict[str, str]:
    """The emitted files, keyed by path relative to the output directory."""
    return {
        str(path.relative_to(artifact_root)): path.read_text()
        for path in artifact_root.rglob("*")
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
    assert "static void pypto_graph_layer(const GraphTaskArgs& args) {" in orch


def test_boundary_tensors_are_bound_from_the_task_args(orch):
    body = _graph_body(orch)
    assert "const Tensor& a = args.tensor(0).ref();" in body
    assert "const Tensor& c = args.tensor(1).ref();" in body


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
    # Anchored: `"params_t0" in entry` is also true of `g0_params_t0`, so a
    # substring check passes even if the entry gained a prefix too — which is
    # the regression this guards against.
    entry = _entry_body(orch)
    assert re.search(r"(?<![0-9A-Za-z_])params_t0\b", entry), entry


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


# ---------------------------------------------------------------------------
# The generated file against the real runtime headers
# ---------------------------------------------------------------------------


def _runtime_include_args(repo_root: Path) -> list[str]:
    """``-I`` flags for the a2a3 host_build_graph orchestration headers.

    The same directories, in the same order, that simpler's own kernel compiler
    puts on an orchestration TU: ``get_orchestration_include_dirs`` followed by
    the runtime's ``build_config`` ``orchestration.include_dirs``. Handing the
    compiler every directory that holds a header instead lets a runtime that is
    not under test shadow the one that is -- ``common/hierarchical/types.h``
    wins over ``host_build_graph/types.h`` and drags in
    ``task_interface/buffer.h``, whose global ``Tensor`` then collides with the
    runtime's own ``Tensor`` alias. The tree root comes last, for the includes
    spelled with a directory prefix (``common/host_phase_kind.h``).
    """
    src = repo_root / "runtime" / "src"
    runtime_dir = src / "a2a3" / "runtime" / "host_build_graph"
    return [
        f"-I{path}"
        for path in (
            runtime_dir / "runtime",
            runtime_dir / "orchestration",
            runtime_dir / "common",
            src / "a2a3" / "runtime",
            src / "common" / "host_build_graph",
            src / "common" / "task_interface",
            src / "common",
            src / "a2a3" / "platform" / "include",
            src / "common" / "platform" / "include",
            src,
        )
    ]


def test_generated_orchestration_compiles_against_the_pinned_runtime(artifact_root):
    """Type-check the emitted file — the only check that covers the graph ABI.

    Every other test here matches generated *text*, so it agrees with whatever
    codegen currently emits. That cannot catch a mismatch with the runtime's
    actual types: `rt_submit_graph` takes `void (*)(const GraphTaskArgs&)` and
    `GraphTaskArgs` is a different `Arg` instantiation from `CoreTaskArgs`, while
    `args.tensor(i).ref()` yields `const simpler::hbg::Tensor&` (aliased
    `Tensor`). Emitting `CoreTaskArgs` or a boundary tensor type there is a
    hard compile error in the generated file that a string assertion happily
    confirms instead of catching.
    """
    repo_root = Path(__file__).resolve().parents[3]
    main_cpp = artifact_root / "orchestration" / "main.cpp"
    assert main_cpp.is_file(), f"no orchestration emitted at {main_cpp}"

    if not (repo_root / "runtime" / "src" / "a2a3" / "runtime" / "host_build_graph").is_dir():
        pytest.skip("runtime submodule not checked out")
    candidates = (os.environ.get("CXX"), "g++-15", "g++", "c++")
    compiler = next(
        (found for name in candidates if name and (found := shutil.which(name))),
        None,
    )
    if compiler is None:
        pytest.skip("no C++ compiler available")

    result = subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", *_runtime_include_args(repo_root), str(main_cpp)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return

    # A missing header means this environment's include layout differs from the
    # one the flags above assume — not a codegen defect, so do not fail on it.
    # Anything the compiler blames on the generated file is.
    if "fatal error:" in result.stderr and str(main_cpp) not in result.stderr.split("fatal error:")[0][-200:]:
        pytest.skip(f"runtime headers not resolvable here: {result.stderr.strip().splitlines()[0]}")
    blamed = [line for line in result.stderr.splitlines() if "error:" in line]
    raise AssertionError(
        "generated orchestration does not compile against the pinned runtime:\n" + "\n".join(blamed[:15])
    )


@pl.program
class _DynDim:
    """A Graph whose body reads a dynamic extent of a boundary tensor.

    `LegalizeGraphBoundary` only forbids a *launch* under a non-constant bound,
    so a dynamic loop that launches nothing is legal here.
    """

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        a: pl.Tensor[[pl.dynamic("N"), 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        n = pl.tensor.dim(a, 0)
        for _ in pl.range(n):
            pass
        with pl.at(level=pl.Level.CORE_GROUP):
            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
            pl.store(t, [0, 0], c)
        return c

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[pl.dynamic("N"), 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c = self.layer(a, c)
        return c


@pl.program
class _SubmitDeps:
    """A Graph whose body submits two tasks with an explicit TaskId edge."""

    @pl.function(type=pl.FunctionType.AIV)
    def k1(
        self, x: pl.Tensor[[128, 128], pl.FP32], o: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        t: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
        o = pl.store(t, [0, 0], o)
        return o

    @pl.function(type=pl.FunctionType.AIV)
    def k2(
        self, x: pl.Tensor[[128, 128], pl.FP32], o: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        t: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
        o = pl.store(t, [0, 0], o)
        return o

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ):
        with pl.manual_scope():
            c, t1 = pl.spmd_submit(self.k1, a, c, core_num=1)
            d, _ = pl.spmd_submit(self.k2, c, d, core_num=1, deps=[t1])
        return c, d

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        d: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ):
        c, d = self.layer(a, c, d)
        return c, d


@pl.program
class _BatchedAllocs:
    """A Graph whose body allocates 20 tensors, each consumed by a launch.

    The interleaving is the point: a launch between two creates does not close
    the batch, so codegen packs all 20 into ``ceil(20 / 16) = 2``
    ``alloc_tensors`` calls -- two recorded nodes, not twenty.

    Under a constant-trip loop so the allocations stay in the region: Step C of
    ``LegalizeGraphBoundary`` hoists a *top-level* region allocation out to the
    call site, which is what ``_RegionAllocs`` below covers. The loop body is
    still one statement list, which is what the batching rule is about.
    """

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self, x: pl.Tensor[[128, 128], pl.FP32], o: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        t: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
        o = pl.store(t, [0, 0], o)
        return o

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        # Chained so every buffer stays live: an unused create would be folded
        # away and the batch would never reach the packing boundary.
        for _ in pl.range(2):
            s0: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s0 = self.kernel(a, s0)
            s1: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s1 = self.kernel(s0, s1)
            s2: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s2 = self.kernel(s1, s2)
            s3: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s3 = self.kernel(s2, s3)
            s4: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s4 = self.kernel(s3, s4)
            s5: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s5 = self.kernel(s4, s5)
            s6: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s6 = self.kernel(s5, s6)
            s7: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s7 = self.kernel(s6, s7)
            s8: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s8 = self.kernel(s7, s8)
            s9: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s9 = self.kernel(s8, s9)
            s10: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s10 = self.kernel(s9, s10)
            s11: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s11 = self.kernel(s10, s11)
            s12: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s12 = self.kernel(s11, s12)
            s13: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s13 = self.kernel(s12, s13)
            s14: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s14 = self.kernel(s13, s14)
            s15: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s15 = self.kernel(s14, s15)
            s16: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s16 = self.kernel(s15, s16)
            s17: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s17 = self.kernel(s16, s17)
            s18: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s18 = self.kernel(s17, s18)
            s19: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            s19 = self.kernel(s18, s19)
            c = self.kernel(s19, c)
        return c

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c = self.layer(a, c)
        return c


@pl.program
class _RegionAllocs:
    """A Graph whose body allocates two scratch tensors at the top level.

    Step C hoists both to the call site, so the recorded region carries no
    allocation node at all and the entry allocates the buffers instead.
    """

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self, x: pl.Tensor[[128, 128], pl.FP32], o: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        t: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
        o = pl.store(t, [0, 0], o)
        return o

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        s0: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
        s0 = self.kernel(a, s0)
        s1: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
        s1 = self.kernel(s0, s1)
        c = self.kernel(s1, c)
        return c

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c = self.layer(a, c)
        return c


def _compile_orch(program) -> str:
    """Compile one program for host_build_graph and return orchestration/main.cpp."""
    out_dir = tempfile.mkdtemp()
    with (
        mock.patch(
            "pypto.backend.pto_backend._compile_pto_module",
            lambda _code, _name, _dir, _planner=None: _STUB_PTOAS_OUTPUT,
        ),
        passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH),
    ):
        ir_compile(program, skip_ptoas=False, platform="a2a3", output_dir=out_dir, dump_passes=False)
    return (Path(out_dir) / "orchestration" / "main.cpp").read_text()


def test_graph_helper_defines_the_dynamic_dims_its_body_reads():
    """The helper binds its own tensor parameters, so it needs its own symbols.

    The entry emits these from its parameter shapes; a Graph body reading one
    without the matching definition names an undeclared variable.
    """
    body = _graph_body(_compile_orch(_DynDim))
    assert "int64_t N = (int64_t)a.shapes[0];" in body, body
    # The loop that reads it is what makes the definition load-bearing.
    assert "i < N;" in body, body


def test_graph_helper_binds_submit_task_ids():
    """`GenerateSubmitReturnAliases` needs the tuple mappings from
    `OrchestrationInfoCollector`, which the entry sets and the graph instance
    must set too — otherwise the TaskId is never bound and the dependency edge
    inside the region is lost."""
    body = _graph_body(_compile_orch(_SubmitDeps))
    assert "TaskId t1 = task_0_outs.task_id();" in body, body
    assert "] = t1;" in body, body


def _alloc_batch_sizes(body: str) -> list[int]:
    """Operand count of each ``alloc_tensors`` call in @p body, in order."""
    return [len(args.split(",")) for args in re.findall(r"alloc_tensors\(([^)]*)\)", body)]


def test_graph_allocations_are_packed_sixteen_to_a_node():
    """Pins codegen to the node count `LegalizeGraphBoundary` charges it.

    The pass and the verifier both estimate a Graph's recorded node count from
    `alloc_batching::BatchedAllocationNodes`, and reject the region when the
    total leaves `1..GRAPH_MAX_NODES`. That estimate is only correct because
    codegen packs `kAllocTensorsArgs` creates per `alloc_tensors` and lets a
    launch sit between two creates without closing the batch -- which nothing
    on the codegen side asserted, so a change here would desynchronise the two
    silently: an over-count rejects a legal Graph, an under-count admits one the
    runtime then declines to cache and replays as ordinary tasks.

    The counterpart is
    `test_legalize_graph_boundary.py::test_interleaved_allocations_batch_across_the_statement_list`,
    which pins the same 20 creates to 2 nodes on the pass side.
    """
    body = _graph_body(_compile_orch(_BatchedAllocs))
    # 16 + 4, not 20 single-create calls and not a batch closed by each launch.
    assert _alloc_batch_sizes(body) == [16, 4], body
    # The other half of the node total the pass computes: 21 launches + 2
    # allocation nodes. A launch that stopped being emitted would make the
    # batching assertion above pass while the totals still diverged.
    assert body.count("rt_submit_") == 21, body


@pl.program
class _SwappedReturns:
    """A Graph returning its two InOut params in the reverse of their declared order.

    `InOutUseDiscipline` requires the caller to read a written param through the
    call's return value, so a Graph that writes several InOut params returns them
    all -- and nothing forces that return order to match the parameter order.
    """

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        x: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        y: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> tuple[pl.Tensor[[128, 128], pl.FP32], pl.Tensor[[128, 128], pl.FP32]]:
        with pl.at(level=pl.Level.CORE_GROUP):
            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
            pl.store(t, [0, 0], x)
            u: pl.Tile[[128, 128], pl.FP32] = pl.add(t, t)
            pl.store(u, [0, 0], y)
        return y, x

    @pl.function(type=pl.FunctionType.InCore)
    def consume(
        self,
        u: pl.Tensor[[128, 128], pl.FP32],
        v: pl.Tensor[[128, 128], pl.FP32],
        o: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tu: pl.Tile[[128, 128], pl.FP32] = pl.load(u, [0, 0], [128, 128])
        tv: pl.Tile[[128, 128], pl.FP32] = pl.load(v, [0, 0], [128, 128])
        s: pl.Tile[[128, 128], pl.FP32] = pl.add(tu, tv)
        return pl.store(s, [0, 0], o)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        # Freshly created region tensors, not entry params: with entry params the
        # compiler resolves the aliasing without emitting a binding at all, and
        # the mis-binding stays invisible.
        for _i in pl.range(2):
            p: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            q: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], pl.FP32)
            qq, pp = self.layer(a, p, q)
            out = self.consume(qq, pp, out)
        return out


def _task_args(entry: str, params_var: str) -> list[str]:
    """The `add_input` / `add_inout` operands of the @p params_var argument block."""
    block = re.search(rf"{params_var};(.*?)rt_submit_", entry, re.S)
    assert block is not None, entry
    return re.findall(r"add_(?:input|inout)\((\w+)\)", block.group(1))


def test_multi_param_graph_returns_bind_to_the_right_call_site_tensors():
    """Return position j binds the arg of the param j *returns*, not the j-th Out param.

    `layer` returns `(y, x)`, so `qq, pp = layer(a, p, q)` means `qq` aliases `q`
    and `pp` aliases `p`. While Graph was excluded from `NormalizeReturnOrder`'s
    param-return canonicalization, the return->param map came back all-nullopt and
    codegen guessed positionally -- return 0 to the first Out/InOut param (`x`) --
    binding every result to the wrong tensor with no diagnostic (#2601). Both
    sides are same-shaped tensors, so nothing downstream notices; the numbers just
    come out wrong.
    """
    entry = _entry_body(_compile_orch(_SwappedReturns))

    # The launch itself is unremarkable: args in parameter order.
    assert _task_args(entry, "GraphTaskArgs params_t0")[1:] == ["p", "q"], entry

    # The consumer is where the mapping shows. `consume(qq, pp, out)` must read
    # qq -> q and pp -> p; `["p", "q"]` here is exactly the swap this fixes.
    assert _task_args(entry, "CoreTaskArgs params_t1")[:2] == ["q", "p"], entry


def test_a_region_allocation_is_hoisted_to_the_call_site():
    """Step C: the recorded region allocates nothing; the entry does it instead.

    The graph heap is never reclaimed mid-run, so an allocation the region makes
    for itself is held for the whole run and the live set grows with the number
    of submissions. Moving it to the call site puts it back on the ordinary
    reclaimable heap.

    Asserted on the emitted C++ rather than on the pass's parameter list because
    the two halves have to agree: the buffer has to leave the region *and*
    arrive as a boundary tensor the launch declares as an output. Declared
    ``add_input``, the launch would not register as a writer of it.
    """
    orch = _compile_orch(_RegionAllocs)
    body = _graph_body(orch)
    assert "alloc_tensors(" not in body, body
    # Both scratch buffers arrive as boundary tensors, after the two the user
    # wrote. They are appended, so the original indices do not move.
    assert "const Tensor& a = args.tensor(0).ref();" in body, body
    assert "const Tensor& c = args.tensor(1).ref();" in body, body
    assert "const Tensor& s0 = args.tensor(2).ref();" in body, body
    assert "const Tensor& s1 = args.tensor(3).ref();" in body, body

    entry = _entry_body(orch)
    assert "alloc_tensors(" in entry, entry
    # `InOut`, not `Input`: the region writes them, and a caller that hoisted the
    # allocation out of its own loop needs successive launches ordered.
    assert entry.count(".add_inout(s0") == 1, entry
    assert entry.count(".add_inout(s1") == 1, entry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
