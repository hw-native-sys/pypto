# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``FunctionType.Graph``.

A Graph function is a callable orchestration fragment: its body is orchestration
code, but each call site is one task launch that the ``host_build_graph`` runtime
records once and replays thereafter. The runtime identifies the recording by the
address of the emitted C++ function, so the type carries no cache key.

Graph is authored like every other function type — ``type=pl.FunctionType.Graph``
— which is also how it prints, so print->parse works through the same path as the
rest of the enum.
"""

import tempfile

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.printer import python_print
from pypto.language.parser.text_parser import parse_program


def _layer(program: ir.Program) -> ir.Function:
    """Fetch the Graph function, narrowing away the Optional for type checking."""
    func = program.get_function("layer")
    assert func is not None
    return func


@pl.program
class GraphInProgram:
    """The primary authoring path: a Graph method inside ``@pl.program``."""

    @pl.function(type=pl.FunctionType.Graph)
    def layer(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
            pl.store(t, [0, 0], c)
        return c


# ---------------------------------------------------------------------------
# The enum itself
# ---------------------------------------------------------------------------


def test_graph_is_a_distinct_function_type():
    assert ir.FunctionType.Graph not in {
        ir.FunctionType.Opaque,
        ir.FunctionType.Orchestration,
        ir.FunctionType.InCore,
        ir.FunctionType.AIC,
        ir.FunctionType.AIV,
        ir.FunctionType.Group,
        ir.FunctionType.Spmd,
        ir.FunctionType.Inline,
    }


def test_graph_derives_chip_level_and_orchestrator_role():
    """A Graph body orchestrates tasks, so it is a chip-level Orchestrator.

    This is what makes ``{CHIP, Orchestrator}`` ambiguous: code that must single
    out the compilation *entry* can no longer key on level+role, because a Graph
    now matches too. ``IsChipOrch`` in materialize_comm_domain_scopes_pass.cpp
    excludes Graph explicitly for exactly this reason.
    """
    func = _layer(GraphInProgram)
    assert func.func_type == ir.FunctionType.Graph
    assert func.level == ir.Level.CHIP
    assert func.role == ir.Role.Orchestrator


# Serialization round-trip is covered for every non-Opaque type, Graph included,
# by ``test_function_type_serialization`` in test_function_type.py.


# ---------------------------------------------------------------------------
# Both decorator paths
# ---------------------------------------------------------------------------


def test_graph_type_on_the_standalone_path():
    # The standalone path builds the IR inside the decorator, rather than
    # deferring to the @pl.program walker, so it is a genuinely separate route
    # to the same type.
    @pl.function(type=pl.FunctionType.Graph)
    def standalone(
        a: pl.Tensor[[128, 128], pl.FP32],
        c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
            pl.store(t, [0, 0], c)
        return c

    assert standalone.func_type == ir.FunctionType.Graph


def test_plain_function_is_unaffected():
    @pl.function
    def plain(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        return x

    assert plain.func_type == ir.FunctionType.Opaque


# ---------------------------------------------------------------------------
# The gap this change opens, reported rather than crashed on
# ---------------------------------------------------------------------------


def test_compiling_a_graph_call_reports_that_codegen_is_missing():
    """Until the graph-launch emission path lands, say so instead of crashing.

    A Graph callee reaches `InferFunctionCoreType`, which recognises only
    AIC/AIV and aborts with an internal error naming nothing actionable. This
    change makes the type authorable, so it also has to make the gap legible.

    The Graph body opens its own device scope, which is how one is actually
    written — and is only reachable because the scope outliners now admit Graph
    bodies. Without that the compile would stop earlier on a leftover scope, and
    this diagnostic would be unreachable for every realistic program.

    The message is matched in full: it names the authoring form, so a spelling
    that no longer exists would otherwise survive here unnoticed.
    """
    from pypto.backend.pto_backend import PartialCodegenError  # noqa: PLC0415
    from pypto.ir.compile import compile as ir_compile  # noqa: PLC0415

    @pl.program
    class UsesGraph:
        @pl.function(type=pl.FunctionType.Graph)
        def layer(
            self,
            a: pl.Tensor[[128, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                pl.store(t, [0, 0], c)
            return c

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[128, 128], pl.FP32],
            c: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        ) -> pl.Tensor[[128, 128], pl.FP32]:
            c = self.layer(a, c)
            return c

    expected = (
        "Graph function 'layer' cannot be compiled yet: type=pl.FunctionType.Graph is authorable, "
        "but the orchestration codegen that emits a graph launch is not in place yet."
    )
    with tempfile.TemporaryDirectory() as out_dir:
        with pytest.raises(PartialCodegenError) as excinfo:
            ir_compile(UsesGraph, skip_ptoas=True, platform="a2a3", output_dir=out_dir, dump_passes=False)

    # PartialCodegenError word-wraps each message into a table cell, so compare
    # against the text with the gutter and the wrapping collapsed away — a
    # substring match on one line would not notice a stale spelling in another.
    reported = " ".join(str(excinfo.value).replace("|", " ").split())
    assert expected in reported


# ---------------------------------------------------------------------------
# print -> parse
# ---------------------------------------------------------------------------


def test_printed_form_is_the_authored_form():
    text = python_print(GraphInProgram)
    assert "type=pl.FunctionType.Graph" in text


def test_graph_function_roundtrips():
    reparsed = parse_program(python_print(GraphInProgram))
    ir.assert_structural_equal(GraphInProgram, reparsed)
    assert _layer(reparsed).func_type == ir.FunctionType.Graph


# ---------------------------------------------------------------------------
# Parser treatment
# ---------------------------------------------------------------------------


def test_graph_body_folds_tensor_dim_to_the_signature_symbol():
    """A Graph body gets the same dynamic-extent folding as an Orchestration one.

    `_fold_tensor_dim` rewrites ``pl.tensor.dim(x, 0)`` into the symbol the
    signature already binds for that extent. Gated strictly on Orchestration, a
    Graph body instead mints a *second* runtime scalar for the same extent, and
    a shape built from it disagrees structurally with a callee or tensor type
    that uses the declared symbol.

    The two programs below are the same body under the two function types; the
    decorator's `type=` must be a literal, so they cannot share a builder.
    """
    N = pl.dynamic("N")

    @pl.program
    class AsGraph:
        @pl.function(type=pl.FunctionType.Graph)
        def layer(
            self,
            x: pl.Tensor[[N, 64], pl.FP32],
            out: pl.InOut[pl.Tensor[[N, 64], pl.FP32]],
        ) -> pl.Tensor[[N, 64], pl.FP32]:
            n = pl.tensor.dim(x, 0)
            for _ in pl.range(n):
                pass
            return out

    @pl.program
    class AsOrchestration:
        @pl.function(type=pl.FunctionType.Orchestration)
        def layer(
            self,
            x: pl.Tensor[[N, 64], pl.FP32],
            out: pl.InOut[pl.Tensor[[N, 64], pl.FP32]],
        ) -> pl.Tensor[[N, 64], pl.FP32]:
            n = pl.tensor.dim(x, 0)
            for _ in pl.range(n):
                pass
            return out

    graph_text = python_print(_layer(AsGraph))
    assert "pl.tensor.dim" not in graph_text, "the extent symbol was not reused"

    # Same body, same parser output — only the decorator line differs.
    orch_text = python_print(_layer(AsOrchestration))
    assert graph_text.split("\n", 1)[1] == orch_text.split("\n", 1)[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
