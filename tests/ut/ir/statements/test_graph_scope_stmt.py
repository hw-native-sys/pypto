# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the GraphScopeStmt IR node."""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir.printer import python_print
from pypto.language.parser.diagnostics import ParserSyntaxError


def _vars(span: ir.Span) -> tuple[ir.Var, ir.Var]:
    var_x = ir.Var("x", ir.TensorType([64], DataType.FP32), span)
    var_y = ir.Var("y", ir.TensorType([64], DataType.FP32), span)
    return var_x, var_y


def _make_body(span: ir.Span, var_x: ir.Var | None = None, var_y: ir.Var | None = None) -> ir.Stmt:
    if var_x is None or var_y is None:
        var_x, var_y = _vars(span)
    return ir.AssignStmt(var_y, var_x, span)


def test_construct():
    """GraphScopeStmt carries the region name and reports ScopeKind.Graph."""
    span = ir.Span("test.py", 1, 1, 1, 10)
    scope = ir.GraphScopeStmt("layer", body=_make_body(span), span=span)

    assert isinstance(scope, ir.ScopeStmt)
    assert scope.scope_kind == ir.ScopeKind.Graph
    assert scope.name_hint == "layer"
    assert isinstance(scope.body, ir.AssignStmt)


def test_structural_equal_same():
    """Two nodes with the same name and body compare structurally equal."""
    span = ir.Span("test.py", 1, 1, 1, 10)
    var_x, var_y = _vars(span)
    scope1 = ir.GraphScopeStmt("layer", body=_make_body(span, var_x, var_y), span=span)
    scope2 = ir.GraphScopeStmt("layer", body=_make_body(span, var_x, var_y), span=span)
    assert ir.structural_equal(scope1, scope2)


def test_structural_unequal_name():
    """The region name is part of the node's identity.

    It becomes the outlined function's name and hence the runtime's graph key,
    so two regions differing only in name are genuinely different programs.
    """
    span = ir.Span("test.py", 1, 1, 1, 10)
    var_x, var_y = _vars(span)
    scope1 = ir.GraphScopeStmt("layer_a", body=_make_body(span, var_x, var_y), span=span)
    scope2 = ir.GraphScopeStmt("layer_b", body=_make_body(span, var_x, var_y), span=span)
    assert not ir.structural_equal(scope1, scope2)


def test_not_equal_to_other_scope_kinds():
    """A Graph region is not a Cluster region with the same body."""
    span = ir.Span("test.py", 1, 1, 1, 10)
    var_x, var_y = _vars(span)
    graph = ir.GraphScopeStmt("layer", body=_make_body(span, var_x, var_y), span=span)
    cluster = ir.ClusterScopeStmt("layer", body=_make_body(span, var_x, var_y), span=span)
    assert not ir.structural_equal(graph, cluster)


def test_serialize_roundtrip():
    """A .pto serialize -> deserialize round-trip is a byte-level fixpoint.

    (Free Vars in the body get fresh identities on deserialize, so
    structural_equal is not a reliable cross-roundtrip check for any scope
    node — re-serialize and compare bytes instead.)
    """
    span = ir.Span("test.py", 1, 1, 1, 10)
    scope = ir.GraphScopeStmt("layer", body=_make_body(span), span=span)
    data = ir.serialize(scope)
    restored = ir.deserialize(data)

    assert isinstance(restored, ir.GraphScopeStmt)
    assert restored.name_hint == "layer"
    assert ir.serialize(restored) == data


def test_empty_name_is_rejected_by_the_parser():
    """The name is the graph key, so no path auto-generates one.

    The parser rejects it first, which is what a user hits; ``IRBuilder``'s own
    ``CHECK`` behind it is the backstop for IR built by other means.
    """
    with pytest.raises(ParserSyntaxError, match=r"must not be empty"):

        @pl.program
        class Bad:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.graph(""):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y


def test_print_reparse_round_trip():
    """``with pl.graph("name"):`` prints and reparses to the same source."""
    src = (
        "@pl.function(type=pl.FunctionType.Orchestration)\n"
        "def main(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:\n"
        '    with pl.graph("layer"):\n'
        "        with pl.at(level=pl.Level.CORE_GROUP):\n"
        "            y: pl.Tensor[[64], pl.FP32] = pl.tensor.add(x, x)\n"
        "    return y"
    )
    func = pl.parse(src)
    out1 = python_print(func, format=False)
    assert 'pl.graph("layer")' in out1
    out2 = python_print(pl.parse(out1), format=False)
    assert out1 == out2, f"Round-trip diverged:\n=== first ===\n{out1}\n=== second ===\n{out2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
