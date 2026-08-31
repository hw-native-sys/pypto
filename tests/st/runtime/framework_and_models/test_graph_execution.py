# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end coverage for ``@pl.jit.graph`` under the host_build_graph runtime.

The unit tests around ``LegalizeGraphBoundary`` and orchestration codegen cover
the compile side, including one that compiles the emitted
``orchestration/main.cpp`` against the pinned runtime headers. None of them can
show that a *recorded* graph replays correctly, which is the half where this
feature fails quietly:

* **A frozen boundary scalar is a wrong answer.** The runtime tracks a boundary
  scalar by the address of its argument slot. A value the region derives has no
  slot, so it is classified as static data and every replay reuses the first
  call's number, with no diagnostic. ``test_per_layer_accumulate`` is shaped to
  catch it — each call reads its own band of a weight tensor whose bands hold
  distinct values, so a frozen offset re-adds band 0 and misses by a wide margin.
* **A declined recording is silent too.** ``rt_submit_graph`` falls back to
  running the body as ordinary tasks when ``rt_graph_args_cacheable`` refuses the
  boundary — correct numbers, no speedup, no log line. The runtime exposes no
  Python-visible counter for that, so the guard is on the compile side:
  ``TestGraphIsNotSilentlyDroppedAtCompile`` asserts the lowered IR really
  carries a ``FunctionType.Graph`` function.

The runtime ABI comes from the enclosing ``PassContext``, which is what
``@pl.jit`` reads when it lowers; ``graph_runtime`` supplies one.

**Run this file with ``--forked``.** A ``ChipWorker`` binds one
``(platform, device_id, runtime)``, and this file deliberately mixes the two: the
Graph cases need ``host_build_graph`` while the A/B twin runs on the default
runtime. Sharing one process poisons the lane — the first Graph test fails 507018
and every later test in that process goes with it, including unrelated ones. CI
gives the file its own ``--forked`` step for that reason.
"""

import pypto.language as pl
import pytest
import torch
from pypto.ir.printer import python_print
from pypto.pypto_core import passes

ROWS = 128
COLS = 128
LAYERS = 4


@pytest.fixture
def graph_runtime():
    """Compile under host_build_graph — a Graph is rejected under any other."""
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        yield


def _banded_weights() -> torch.Tensor:
    """Weights whose layer *i* band holds ``i + 1``.

    Distinct per band on purpose: a frozen offset re-reads one band, so the sum
    lands far outside tolerance instead of coincidentally matching.
    """
    w = torch.empty(LAYERS * ROWS, COLS, dtype=torch.float32)
    for i in range(LAYERS):
        w[i * ROWS : (i + 1) * ROWS, :] = float(i + 1)
    return w


def _band_total() -> float:
    return float(sum(i + 1 for i in range(LAYERS)))


# --- Kernels ---


@pl.jit.graph
def accumulate_band(w: pl.Tensor, acc: pl.InOut[pl.Tensor], layer_idx: pl.Scalar[pl.INDEX]):
    """``base`` is derived from the boundary scalar, so Step A hoists it out.

    Left in the region it would have no argument slot and every replay would
    re-add layer 0's band.
    """
    base = layer_idx * ROWS
    with pl.at(level=pl.Level.CORE_GROUP):
        band = pl.load(w, [base, 0], [ROWS, COLS])
        cur = pl.load(acc, [0, 0], [ROWS, COLS])
        pl.store(pl.add(cur, band), [0, 0], acc)
    return acc


@pl.jit
def per_layer_accumulate(w: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    for i in pl.range(LAYERS):
        acc = accumulate_band(w, acc, i)
    return acc


@pl.jit
def no_graph_per_layer_accumulate(w: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    """Same arithmetic, no Graph — the body is inlined rather than called.

    An Orchestration entry launches tasks; it does not call another
    Orchestration. Being *callable* is what ``Graph`` adds, so the twin can only
    mirror the maths, not the structure.
    """
    for i in pl.range(LAYERS):
        base = i * ROWS
        with pl.at(level=pl.Level.CORE_GROUP):
            band = pl.load(w, [base, 0], [ROWS, COLS])
            cur = pl.load(acc, [0, 0], [ROWS, COLS])
            pl.store(pl.add(cur, band), [0, 0], acc)
    return acc


@pl.jit.graph
def accumulate_view(w: pl.Tensor, acc: pl.InOut[pl.Tensor], layer_idx: pl.Scalar[pl.INDEX]):
    """Step B: a boundary tensor sliced by a per-layer offset.

    The slice is hoisted to the call site and arrives as its own boundary
    tensor, so each replay addresses its own window.
    """
    wl = pl.tensor.slice(w, [ROWS, COLS], [layer_idx * ROWS, 0])
    with pl.at(level=pl.Level.CORE_GROUP):
        band = pl.load(wl, [0, 0], [ROWS, COLS])
        cur = pl.load(acc, [0, 0], [ROWS, COLS])
        pl.store(pl.add(cur, band), [0, 0], acc)
    return acc


@pl.jit
def boundary_view_accumulate(w: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    for i in pl.range(LAYERS):
        acc = accumulate_view(w, acc, i)
    return acc


@pl.jit.graph
def double_via_scratch(a: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    """Step C: a constant-shaped allocation inside the region.

    Recorded as a kernel-less node, so the region's topology is more than one
    task per call.
    """
    tmp = pl.create_tensor([ROWS, COLS], pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(pl.add(t, t), [0, 0], tmp)
    with pl.at(level=pl.Level.CORE_GROUP):
        u = pl.load(tmp, [0, 0], [ROWS, COLS])
        v = pl.load(acc, [0, 0], [ROWS, COLS])
        pl.store(pl.add(u, v), [0, 0], acc)
    return acc


@pl.jit
def region_alloc_accumulate(a: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    for _ in pl.range(LAYERS):
        acc = double_via_scratch(a, acc)
    return acc


@pl.jit.graph
def add_layer(a: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(a, [0, 0], [ROWS, COLS])
        c = pl.load(acc, [0, 0], [ROWS, COLS])
        pl.store(pl.add(c, t), [0, 0], acc)
    return acc


@pl.jit.graph
def double_layer(acc: pl.InOut[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        c = pl.load(acc, [0, 0], [ROWS, COLS])
        pl.store(pl.add(c, c), [0, 0], acc)
    return acc


@pl.jit
def two_distinct_graphs(a: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    """Two Graph functions interleaved.

    The runtime keys a recording on the emitted function's address, so these
    must stay two recordings — sharing one would apply ``add_layer``'s topology
    to ``double_layer`` and change the result.
    """
    for _ in pl.range(LAYERS):
        acc = add_layer(a, acc)
        acc = double_layer(acc)
    return acc


@pl.jit.graph
def scale_once(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(pl.add(t, t), [0, 0], out)
    return out


@pl.jit
def single_launch(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    out = scale_once(a, out)
    return out


# --- Device execution ---


class TestGraphExecution:
    """Numerical results after record-and-replay."""

    def test_single_launch(self, test_config, graph_runtime):
        single_launch._cache.clear()
        a = torch.full((ROWS, COLS), 3.0, dtype=torch.float32)
        out = torch.zeros((ROWS, COLS), dtype=torch.float32)

        single_launch(a, out, config=test_config)

        expected = a * 2
        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
            f"max diff = {(out - expected).abs().max().item()}"
        )

    def test_per_layer_accumulate(self, test_config, graph_runtime):
        """The one that catches a frozen boundary scalar.

        Every band must be added exactly once. A frozen offset adds band 0 four
        times — 4.0 instead of 10.0 — nowhere near tolerance.
        """
        per_layer_accumulate._cache.clear()
        w = _banded_weights()
        acc = torch.zeros((ROWS, COLS), dtype=torch.float32)

        per_layer_accumulate(w, acc, config=test_config)

        expected = torch.full((ROWS, COLS), _band_total())
        assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-5), (
            f"max diff = {(acc - expected).abs().max().item()}; a frozen per-layer "
            f"offset would give {float(LAYERS)} everywhere"
        )

    def test_no_graph_per_layer_accumulate(self, test_config):
        """Same golden without a Graph, so a failure here is arithmetic.

        Deliberately outside ``graph_runtime``: this one runs on the default
        runtime, which is what makes the pair an A/B on the feature.
        """
        no_graph_per_layer_accumulate._cache.clear()
        w = _banded_weights()
        acc = torch.zeros((ROWS, COLS), dtype=torch.float32)

        no_graph_per_layer_accumulate(w, acc, config=test_config)

        expected = torch.full((ROWS, COLS), _band_total())
        assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-5), (
            f"max diff = {(acc - expected).abs().max().item()}"
        )

    def test_boundary_view(self, test_config, graph_runtime):
        boundary_view_accumulate._cache.clear()
        w = _banded_weights()
        acc = torch.zeros((ROWS, COLS), dtype=torch.float32)

        boundary_view_accumulate(w, acc, config=test_config)

        expected = torch.full((ROWS, COLS), _band_total())
        assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-5), (
            f"max diff = {(acc - expected).abs().max().item()}"
        )

    def test_region_alloc(self, test_config, graph_runtime):
        region_alloc_accumulate._cache.clear()
        a = torch.full((ROWS, COLS), 1.5, dtype=torch.float32)
        acc = torch.zeros((ROWS, COLS), dtype=torch.float32)

        region_alloc_accumulate(a, acc, config=test_config)

        expected = a * 2 * LAYERS
        assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-5), (
            f"max diff = {(acc - expected).abs().max().item()}"
        )

    def test_two_distinct_graphs(self, test_config, graph_runtime):
        two_distinct_graphs._cache.clear()
        a = torch.full((ROWS, COLS), 1.0, dtype=torch.float32)
        acc = torch.zeros((ROWS, COLS), dtype=torch.float32)

        two_distinct_graphs(a, acc, config=test_config)

        expected = torch.zeros((ROWS, COLS), dtype=torch.float32)
        for _ in range(LAYERS):
            expected = (expected + a) * 2
        assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-5), (
            f"max diff = {(acc - expected).abs().max().item()}"
        )

    def test_replay_serves_a_second_call(self, test_config, graph_runtime):
        """The second invocation replays rather than rebuilds.

        Its *correctness* is the point: replay patches boundary addresses and
        scalars, so fresh inputs must give a fresh answer rather than the
        recorded one.
        """
        per_layer_accumulate._cache.clear()
        w = _banded_weights()

        first = torch.zeros((ROWS, COLS), dtype=torch.float32)
        per_layer_accumulate(w, first, config=test_config)

        second = torch.full((ROWS, COLS), 100.0, dtype=torch.float32)
        per_layer_accumulate(w, second, config=test_config)

        assert torch.allclose(first, torch.full((ROWS, COLS), _band_total()), rtol=1e-5, atol=1e-5)
        assert torch.allclose(
            second, torch.full((ROWS, COLS), 100.0 + _band_total()), rtol=1e-5, atol=1e-5
        ), "the second call reused the first call's result instead of replaying against its own boundary"


# --- Compile side ---


class TestGraphIsNotSilentlyDroppedAtCompile:
    """A declined recording is numerically correct, so goldens cannot see it.

    ``rt_submit_graph`` runs the body as ordinary tasks when the boundary is not
    cacheable, and the runtime exposes no Python-visible counter for it. What is
    checkable is that the compiler produced a Graph at all: a region that
    silently lowered to ordinary tasks carries no ``FunctionType.Graph``.
    """

    @staticmethod
    def _lower(fn, *args, config):
        with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
            return python_print(fn.lower(*args, config=config))

    def test_graph_survives_lowering(self, test_config):
        src = self._lower(
            per_layer_accumulate,
            torch.zeros(LAYERS * ROWS, COLS),
            torch.zeros(ROWS, COLS),
            config=test_config,
        )
        assert "type=pl.FunctionType.Graph" in src, src
        assert "def accumulate_band" in src, src

    def test_the_derived_offset_moves_to_the_caller(self, test_config):
        """Step A hoisted ``base``, so the entry scales it, not the region.

        Inside the region it would have no argument slot and replay would reuse
        the first call's value — which is what ``test_per_layer_accumulate``
        would then fail on.
        """
        src = self._lower(
            per_layer_accumulate,
            torch.zeros(LAYERS * ROWS, COLS),
            torch.zeros(ROWS, COLS),
            config=test_config,
        )
        entry = src[src.index("def per_layer_accumulate") :]
        assert f"* {ROWS}" in entry, entry

    def test_two_graphs_stay_two_functions(self, test_config):
        src = self._lower(
            two_distinct_graphs,
            torch.zeros(ROWS, COLS),
            torch.zeros(ROWS, COLS),
            config=test_config,
        )
        assert src.count("type=pl.FunctionType.Graph") == 2, src

    def test_a_graph_under_the_default_runtime_is_rejected(self, test_config):
        """The default runtime has neither `rt_submit_graph` nor `GraphTaskArgs`.

        Compiling a Graph against it used to emit orchestration C++ naming
        undeclared symbols; it is a compile-time error now.
        """
        with pytest.raises(ValueError, match="requires the host_build_graph runtime"):
            per_layer_accumulate.lower(
                torch.zeros(LAYERS * ROWS, COLS),
                torch.zeros(ROWS, COLS),
                config=test_config,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
