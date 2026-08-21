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
    diags = _graph_diags(pl.parse_program(src))
    assert len(diags) == 1
    assert "empty boundary" in diags[0].message


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
