# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the GraphBoundaryLegalized property verifier.

``LegalizeGraphBoundary`` rejects illegal graphs as it rewrites them; this
verifier re-states the resulting invariants program-wide so a later pass that
reintroduces a violation is caught.

That safety net matters more here than for a typical property. Almost every
host_build_graph constraint degrades to a *silent* non-graph fallback in a
release build: the program stays numerically correct and merely loses the
speedup, which no correctness test can detect. This verifier is the automated
detector, so every case below asserts on the rule name and the message.
"""

import pypto.language as pl
import pytest
from pypto.pypto_core import passes

GRAPH_BODY = (
    "        with pl.at(level=pl.Level.CORE_GROUP):\n"
    "            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])\n"
    "            pl.store(t, [0, 0], c)\n"
    "        return c\n"
)


def _verify(prog):
    # Outline first. The property describes IR at the point LegalizeGraphBoundary
    # runs — well after OutlineIncoreScopes — where a `pl.at` scope has become a
    # task launch. Verifying the pre-outlining shape would ask the body checks
    # about a topology that does not exist yet: every Graph would read as
    # launching zero tasks.
    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        prog = passes.outline_incore_scopes()(passes.convert_to_ssa()(prog))
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.GraphBoundaryLegalized)
    return passes.PropertyVerifierRegistry.verify(props, prog)


def _graph_diags(prog):
    return [d for d in _verify(prog) if d.rule_name == "GraphBoundaryLegalized"]


# ---------------------------------------------------------------------------
# The legal shape
# ---------------------------------------------------------------------------


def test_well_formed_graph_passes():
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        f"{GRAPH_BODY}"
        "    @pl.function\n"
        "    def main(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(a, c)\n"
        "        return c\n"
    )
    assert _graph_diags(pl.parse_program(src)) == []


# ---------------------------------------------------------------------------
# Signature contract
# ---------------------------------------------------------------------------


def test_graph_without_tensor_parameters_is_rejected():
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:\n"
        "        return n\n"
        "    @pl.function\n"
        "    def main(self) -> pl.Scalar[pl.INDEX]:\n"
        "        return self.layer(0)\n"
    )
    # Two independent violations, both genuine: no tensor to patch on replay,
    # and no task to record.
    messages = [d.message for d in _graph_diags(pl.parse_program(src))]
    assert any("empty boundary" in m for m in messages), messages
    assert any("launches no tasks" in m for m in messages), messages


def test_runtime_allocated_output_is_rejected():
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.Out[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        f"{GRAPH_BODY}"
        "    @pl.function\n"
        "    def main(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(a, c)\n"
        "        return c\n"
    )
    diags = _graph_diags(pl.parse_program(src))
    assert any("the runtime allocates it" in d.message for d in diags)


# ---------------------------------------------------------------------------
# Who may call a Graph
# ---------------------------------------------------------------------------


def test_graph_called_from_another_graph_is_rejected():
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def inner(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        f"{GRAPH_BODY}"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def outer(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.inner(a, c)\n"
        "        return c\n"
        "    @pl.function\n"
        "    def main(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.outer(a, c)\n"
        "        return c\n"
    )
    diags = _graph_diags(pl.parse_program(src))
    assert any("already recording" in d.message for d in diags)


def test_submit_may_omit_only_an_out_tail():
    """The Submit prefix rule covers runtime-allocated `Out` params, nothing else.

    Requiring `args == params` would reject legal IR; accepting any short arg
    list lets a missing `In` through to codegen, where it surfaces as an
    INTERNAL_CHECK in the direction handling instead of a diagnostic here.
    """
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.InCore)\n"
        "    def kernel(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "out: pl.InOut[pl.Tensor[[128, 128], pl.FP32]], "
        "scale: pl.Scalar[pl.INDEX]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128], "
        "target_memory=pl.MemorySpace.Vec)\n"
        "        return pl.store(t, [0, 0], out)\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        with pl.manual_scope():\n"
        "            c, tid = pl.submit(self.kernel, a, c)\n"
        "        return c\n"
        "    @pl.function\n"
        "    def main(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(a, c)\n"
        "        return c\n"
    )
    messages = [d.message for d in _graph_diags(pl.parse_program(src))]
    # `scale` is an omitted In, not a runtime-allocated Out tail.
    assert any("without supplying 'scale" in m for m in messages), messages


def test_tensor_full_in_the_region_is_reported():
    """Re-derived here, not trusted from the pass that rejected it."""
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        _s: pl.Tensor[[16], pl.FP32] = pl.full([16], dtype=pl.FP32, value=0.0)\n"
        f"{GRAPH_BODY}"
        "    @pl.function\n"
        "    def main(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(a, c)\n"
        "        return c\n"
    )
    messages = [d.message for d in _graph_diags(pl.parse_program(src))]
    assert any("tensor.full" in m for m in messages), messages


def test_runtime_shaped_allocation_is_reported():
    """A frozen shape is a wrong address layout, so the property must catch it."""
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]], n: pl.Scalar[pl.INDEX]) "
        "-> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        _s: pl.Tensor[[n], pl.FP32] = pl.create_tensor([n], pl.FP32)\n"
        f"{GRAPH_BODY}"
        "    @pl.function\n"
        "    def main(self, a: pl.Tensor[[128, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]], n: pl.Scalar[pl.INDEX]) "
        "-> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(a, c, n)\n"
        "        return c\n"
    )
    messages = [d.message for d in _graph_diags(pl.parse_program(src))]
    assert any("not a compile-time constant" in m for m in messages), messages


# ---------------------------------------------------------------------------
# Replay-invariant values — the audit must agree with the pass about these
# ---------------------------------------------------------------------------


def test_a_surviving_scalar_alias_of_a_boundary_parameter_is_reported():
    """A rename does not inherit the parameter's argument slot.

    Orchestration codegen emits a surviving `n = batch` as a value copy
    (`int64_t n = batch;`) and then `add_scalar(n)`. Recording matches a boundary
    scalar by the *address* its value came from, comparing against
    `&boundary_args->scalar(i)`, so the copy has no match, is recorded as
    `STATIC_VALUE`, and every later replay reuses the first call's number.

    `LegalizeGraphBoundary` substitutes these away, so none should reach codegen.
    The property is that none *survives*: this input is the pre-pass shape, and
    the verifier must still call it a leak.
    """
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, a: pl.Tensor[[512, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]], batch: pl.Scalar[pl.INDEX]) "
        "-> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        n = batch\n"
        "        with pl.at(level=pl.Level.CORE_GROUP):\n"
        "            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [n, 0], [128, 128])\n"
        "            pl.store(t, [0, 0], c)\n"
        "        return c\n"
        "    @pl.function\n"
        "    def main(self, a: pl.Tensor[[512, 128], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(a, c, 0)\n"
        "        return c\n"
    )
    messages = [d.message for d in _graph_diags(pl.parse_program(src))]
    assert any("no argument slot" in m for m in messages), messages


def test_a_constant_trip_loop_offset_is_not_reported():
    """Frozen at a value every replay recomputes identically.

    The offset has no argument slot and does end up baked into the recording —
    but constant loop bounds mean iteration `i` supplies the same literal on
    every call, so the baked copy is the correct one.
    """
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, w: pl.Tensor[[128, 256], pl.FP32], "
        "acc: pl.InOut[pl.Tensor[[128, 256], pl.FP32]]) -> pl.Tensor[[128, 256], pl.FP32]:\n"
        "        for i in pl.range(2):\n"
        "            off = i * 128\n"
        "            with pl.at(level=pl.Level.CORE_GROUP):\n"
        "                t: pl.Tile[[128, 128], pl.FP32] = pl.load(w, [0, off], [128, 128])\n"
        "                pl.store(t, [0, off], acc)\n"
        "        return acc\n"
        "    @pl.function\n"
        "    def main(self, w: pl.Tensor[[128, 256], pl.FP32], "
        "acc: pl.InOut[pl.Tensor[[128, 256], pl.FP32]]) -> pl.Tensor[[128, 256], pl.FP32]:\n"
        "        self.layer(w, acc)\n"
        "        return acc\n"
    )
    diags = _graph_diags(pl.parse_program(src))
    assert not diags, [d.message for d in diags]


def test_a_scalar_read_from_a_tensor_is_reported():
    """The leak the widening must still catch.

    Nothing pins a boundary buffer's *contents*, so a value read out of one
    genuinely differs between calls — and it has no argument slot to be patched
    through, which is exactly the silent freeze this property exists to detect.
    """
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, a: pl.Tensor[[512, 128], pl.FP32], idx: pl.Tensor[[4], pl.INT32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        row: pl.Scalar[pl.INDEX] = pl.tensor.read(idx, [0])\n"
        "        with pl.at(level=pl.Level.CORE_GROUP):\n"
        "            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [row, 0], [128, 128])\n"
        "            pl.store(t, [0, 0], c)\n"
        "        return c\n"
        "    @pl.function\n"
        "    def main(self, a: pl.Tensor[[512, 128], pl.FP32], idx: pl.Tensor[[4], pl.INT32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(a, idx, c)\n"
        "        return c\n"
    )
    messages = [d.message for d in _graph_diags(pl.parse_program(src))]
    assert any("can differ between calls" in m for m in messages), messages


def test_a_boundary_view_with_an_invariant_window_is_not_reported():
    """A view the pass deliberately leaves in place is not a leak.

    Replay restores it as `boundary.start_offset + packed_offset` with the delta
    recorded on the first call; an invariant offset makes that delta right every
    time, and there is no call site name to hoist the view to.
    """
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, w: pl.Tensor[[512, 256], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        for i in pl.range(2):\n"
        "            off = i * 128\n"
        "            band: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [off, 0])\n"
        "            with pl.at(level=pl.Level.CORE_GROUP):\n"
        "                t: pl.Tile[[128, 128], pl.FP32] = pl.load(band, [0, 0], [128, 128])\n"
        "                pl.store(t, [0, 0], c)\n"
        "        return c\n"
        "    @pl.function\n"
        "    def main(self, w: pl.Tensor[[512, 256], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]]) -> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(w, c)\n"
        "        return c\n"
    )
    diags = _graph_diags(pl.parse_program(src))
    assert not diags, [d.message for d in diags]


def test_a_boundary_view_offset_by_a_boundary_scalar_is_reported():
    """The window that really can move between calls.

    A boundary scalar is patched per call, but the view's `packed_offset` is
    not — so the replayed window is call one's while the address is call two's.
    """
    src = (
        "@pl.program\n"
        "class P:\n"
        "    @pl.function(type=pl.FunctionType.Graph)\n"
        "    def layer(self, w: pl.Tensor[[512, 256], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]], row: pl.Scalar[pl.INDEX]) "
        "-> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        band: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.slice(w, [128, 128], [row, 0])\n"
        "        with pl.at(level=pl.Level.CORE_GROUP):\n"
        "            t: pl.Tile[[128, 128], pl.FP32] = pl.load(band, [0, 0], [128, 128])\n"
        "            pl.store(t, [0, 0], c)\n"
        "        return c\n"
        "    @pl.function\n"
        "    def main(self, w: pl.Tensor[[512, 256], pl.FP32], "
        "c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]], row: pl.Scalar[pl.INDEX]) "
        "-> pl.Tensor[[128, 128], pl.FP32]:\n"
        "        self.layer(w, c, row)\n"
        "        return c\n"
    )
    messages = [d.message for d in _graph_diags(pl.parse_program(src))]
    assert any("window can differ between calls" in m for m in messages), messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
