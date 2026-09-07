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

from examples.advanced import moe_graph_predicate as moe

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


def _graph_boundary_arity(src: str) -> tuple[int, int]:
    """(tensor params, scalar params) of the one Graph function in ``src``.

    Parameter *order* differs between the two surfaces by construction, so the
    comparable quantity is how many of each kind cross the boundary.
    """
    body = src[src.index("type=pl.FunctionType.Graph") :]
    # Anchor on `def` first: the decorator `@pl.function(type=..., level=...)`
    # closes its own paren before the signature opens one, so slicing from the
    # first "(" to the first ")" yields an empty string and a vacuous (0, 0).
    signature = body[body.index("def ") :]
    signature = signature[signature.index("(") : signature.index(") ->")]
    arity = signature.count("pl.Tensor"), signature.count("pl.Scalar")
    # Guard the guard: a slice that stops matching the printer would make every
    # comparison against this helper trivially true.
    assert arity != (0, 0), f"failed to parse a Graph signature out of:\n{signature}"
    return arity


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


@pl.jit
def scope_form_accumulate(w: pl.Tensor, acc: pl.InOut[pl.Tensor]):
    """``per_layer_accumulate`` written with the scope form instead.

    Same arithmetic and the same Graph, expressed as `with pl.graph(...)` in
    place rather than as a separate `@pl.jit.graph` function. OutlineGraphScopes
    lifts the region back out, so this must reach the runtime as the same
    program — which is what ``test_scope_form_matches_the_decorator_form``
    checks, and what makes this case a real A/B rather than a second copy.
    """
    for i in pl.range(LAYERS):
        with pl.graph("accumulate_band_scope"):
            base = i * ROWS
            with pl.at(level=pl.Level.CORE_GROUP):
                band = pl.load(w, [base, 0], [ROWS, COLS])
                cur = pl.load(acc, [0, 0], [ROWS, COLS])
                pl.store(pl.add(cur, band), [0, 0], acc)
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

    def test_scope_form_accumulate(self, test_config, graph_runtime):
        """The scope form must hit the same golden as the decorator form.

        Shares ``test_per_layer_accumulate``'s shape on purpose: a frozen
        per-layer offset gives 4.0 instead of 10.0, so this catches a Step A
        regression reached through the scope path specifically.
        """
        scope_form_accumulate._cache.clear()
        w = _banded_weights()
        acc = torch.zeros((ROWS, COLS), dtype=torch.float32)

        scope_form_accumulate(w, acc, config=test_config)

        expected = torch.full((ROWS, COLS), _band_total())
        assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-5), (
            f"max diff = {(acc - expected).abs().max().item()}; a frozen per-layer "
            f"offset would give {float(LAYERS)} everywhere"
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

    def test_scope_form_lowers_to_a_graph_function(self, test_config):
        """`with pl.graph(...)` must reach codegen as a Graph, not inlined tasks.

        A region that silently lowered to ordinary tasks is numerically correct,
        so the device test above cannot tell — only the lowered IR can.
        """
        src = self._lower(
            scope_form_accumulate,
            torch.zeros(LAYERS * ROWS, COLS),
            torch.zeros(ROWS, COLS),
            config=test_config,
        )
        assert "type=pl.FunctionType.Graph" in src, src
        assert "def accumulate_band_scope" in src, src

    def test_scope_form_matches_the_decorator_form(self, test_config):
        """Both surfaces lower to the same *shape*, which is the pass's contract.

        Not the same text: the outliner orders a region's boundary parameters by
        capture order while the decorator form uses the signature the user wrote,
        so the two argument lists are permutations of each other. What must agree
        is what the runtime sees — one Graph function, the same boundary arity,
        and the derived per-layer offset computed at the caller (Step A) rather
        than inside the region, where replay would freeze it.
        """
        args = (torch.zeros(LAYERS * ROWS, COLS), torch.zeros(ROWS, COLS))
        scope_src = self._lower(scope_form_accumulate, *args, config=test_config)
        decorator_src = self._lower(per_layer_accumulate, *args, config=test_config)

        for src in (scope_src, decorator_src):
            assert src.count("type=pl.FunctionType.Graph") == 1, src
            # The hoisted multiply lands in the entry, outside the Graph body.
            entry = src[src.rindex("type=pl.FunctionType.Orchestration") :]
            assert f"* {ROWS}" in entry, entry

        assert _graph_boundary_arity(scope_src) == _graph_boundary_arity(decorator_src), (
            "the scope form and @pl.jit.graph no longer present the same boundary\n"
            f"=== scope ===\n{scope_src}\n=== decorator ===\n{decorator_src}"
        )

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


# --- MoE routing: a Graph plus per-expert dispatch predicates ---
#
# The program under test is the example itself, imported rather than restated —
# a copy here would let the two drift, and the example is the artifact users
# actually run.


class TestMoeGraphPredicate:
    """Run-time routing inside a recorded graph.

    A recorded topology cannot branch, and under HBG the orchestration cannot
    read a count a task has yet to produce — the graph is built before the device
    runs anything. So every expert is enumerated at build time and each one's
    dispatch is gated by a predicate the scheduler evaluates on device.

    Asserted per expert *band*: an unrouted expert's band keeps its initial
    value, so a failure names the expert that ran when it should not have, or the
    reverse.
    """

    @staticmethod
    def _inputs():
        torch.manual_seed(0)
        return (
            torch.randn(moe.D, moe.D, dtype=torch.float32) * 0.1,
            torch.randn(moe.EXPERTS * moe.D, moe.D, dtype=torch.float32) * 0.1,
            torch.randn(moe.EXPERTS * moe.D, moe.D, dtype=torch.float32) * 0.1,
            torch.randn(moe.EXPERTS * moe.D, moe.D, dtype=torch.float32) * 0.1,
        )

    def _check(self, counts, test_config):
        x, w_gate, w_up, w_down = self._inputs()
        out = torch.zeros((moe.EXPERTS * moe.D, moe.D), dtype=torch.float32)

        moe.moe_decode(x, w_gate, w_up, w_down, counts, out, config=test_config)

        want = moe.expected_bands(x, w_gate, w_up, w_down, counts)
        for e in range(moe.EXPERTS):
            band = slice(e * moe.D, (e + 1) * moe.D)
            routed = int(counts[e, 0]) > 0
            assert torch.allclose(out[band], want[band], rtol=3e-2, atol=3e-2), (
                f"expert {e} (count={int(counts[e, 0])}) "
                f"{'should have run' if routed else 'should have been retired inline'}; "
                f"max diff = {(out[band] - want[band]).abs().max().item()}"
            )

    def test_only_the_routed_experts_contribute(self, test_config, graph_runtime):
        """Experts 1 and 3 receive nothing, so their bands must stay zero."""
        moe.moe_decode._cache.clear()
        self._check(torch.tensor([[1], [0], [1], [0]], dtype=torch.int32), test_config)

    def test_replay_re_reads_the_predicate(self, test_config, graph_runtime):
        """Invert the routing on a *reused* compiled program.

        The second call hits the JIT cache, so it replays the graph recorded by
        the first. A replay that froze the predicate operand — its address or its
        value — would still route experts 0 and 2 and fail here. This is the case
        that distinguishes "the predicate is patched on replay" from "the first
        call's routing happened to be right".
        """
        moe.moe_decode._cache.clear()
        self._check(torch.tensor([[1], [0], [1], [0]], dtype=torch.int32), test_config)
        assert len(moe.moe_decode._cache) == 1

        self._check(torch.tensor([[0], [1], [0], [1]], dtype=torch.int32), test_config)
        assert len(moe.moe_decode._cache) == 1, "second call recompiled instead of replaying"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
