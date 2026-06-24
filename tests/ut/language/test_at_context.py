# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests that ``pl.at(...)`` forwards its keyword arguments into ``AtContext``.

Regression: ``pl.at(..., allow_early_resolve=True)`` was accepted by ``at()`` but
silently dropped because the flag was not forwarded into the returned
``AtContext`` (PR #1819 review). Every accepted keyword must reach the context.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError


def test_at_forwards_allow_early_resolve():
    """pl.at(..., allow_early_resolve=True) carries the flag onto AtContext."""
    ctx = pl.at(ir.Level.CORE_GROUP, allow_early_resolve=True)
    assert ctx.allow_early_resolve is True


def test_at_allow_early_resolve_defaults_false():
    """allow_early_resolve defaults to False on AtContext when omitted."""
    ctx = pl.at(ir.Level.CORE_GROUP)
    assert ctx.allow_early_resolve is False


def test_at_forwards_name_hint():
    """pl.at(..., name_hint=...) reaches AtContext (sibling-kwarg forwarding guard)."""
    ctx = pl.at(ir.Level.CORE_GROUP, name_hint="fused_scope")
    assert ctx.name_hint == "fused_scope"


def test_at_forwards_windowize():
    """pl.at(..., windowize=True) carries the flag onto AtContext."""
    ctx = pl.at(ir.Level.CORE_GROUP, windowize=True)
    assert ctx.windowize is True


def test_at_windowize_defaults_false():
    """windowize defaults to False on AtContext when omitted."""
    ctx = pl.at(ir.Level.CORE_GROUP)
    assert ctx.windowize is False


def test_windowize_reaches_outlined_incore_function():
    """The parser and scope outliners preserve the explicit opt-in."""
    program = pl.parse_program(
        """
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[128, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, windowize=True):
            tile = pl.load(x, [0, 0], [128, 128])
            result = pl.store(tile, [0, 0], out)
        return result
"""
    )

    for outline_pass in (
        passes.outline_hierarchy_scopes(),
        passes.outline_incore_scopes(),
        passes.outline_cluster_scopes(),
    ):
        program = outline_pass(program)

    outlined = program.get_function("main_incore_0")
    assert outlined is not None
    assert outlined.attrs["windowize"] is True


def test_windowize_requires_boolean_literal():
    with pytest.raises(ParserSyntaxError, match="windowize must be a boolean literal"):
        pl.parse_program(
            """
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self):
        with pl.at(level=pl.Level.CORE_GROUP, windowize=1):
            pass
"""
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
