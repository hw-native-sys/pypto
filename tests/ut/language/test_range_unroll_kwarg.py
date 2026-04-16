# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the ``unroll=`` kwarg on ``pl.range``.

Validates the DSL surface and that the parser propagates ``unroll_factor``
into ``ForStmt.attrs_``. The actual unroll lowering is covered by
``test_partial_unroll_tile_loops.py``.
"""

from typing import cast

import pypto.language as pl
import pytest
from pypto import ir


def _outer_for(program: ir.Program) -> ir.ForStmt:
    func = list(program.functions.values())[0]
    body = cast(ir.SeqStmts, func.body)
    return cast(ir.ForStmt, body.stmts[0])


class TestRangeUnrollKwargParser:
    """Verify the parser stashes ``unroll`` into ``ForStmt.attrs_["unroll_factor"]``."""

    def test_unroll_factor_lands_in_for_attrs(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(8, unroll=4):
                    x = pl.add(x, 1.0)
                return x

        for_stmt = _outer_for(P)
        assert dict(for_stmt.attrs).get("unroll_factor") == 4

    def test_unroll_one_is_accepted(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(8, unroll=1):
                    x = pl.add(x, 1.0)
                return x

        assert dict(_outer_for(P).attrs).get("unroll_factor") == 1

    def test_unroll_default_omits_attr(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(8):
                    x = pl.add(x, 1.0)
                return x

        assert "unroll_factor" not in dict(_outer_for(P).attrs)

    def test_unroll_with_init_values_parses(self):
        """``unroll=`` composes with ``init_values=`` — loop-carried state is supported."""

        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(8, init_values=(x,), unroll=2):
                    acc = pl.add(acc, 1.0)
                    acc_rv = pl.yield_(acc)
                return acc_rv

        for_stmt = _outer_for(P)
        assert dict(for_stmt.attrs).get("unroll_factor") == 2
        assert len(for_stmt.iter_args) == 1


class TestRangeUnrollKwargRejection:
    """The parser must reject invalid combinations with clear errors."""

    def test_unroll_zero_rejected(self):
        with pytest.raises(Exception, match=r"unroll factor must be >= 1"):

            @pl.program
            class _P:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.range(8, unroll=0):
                        x = pl.add(x, 1.0)
                    return x

    def test_unroll_negative_rejected(self):
        with pytest.raises(Exception, match=r"unroll factor must be >= 1"):

            @pl.program
            class _P:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.range(8, unroll=-2):
                        x = pl.add(x, 1.0)
                    return x

    def test_unroll_with_chunk_rejected(self):
        with pytest.raises(Exception, match=r"unroll= and chunk= are mutually exclusive"):

            @pl.program
            class _P:
                @pl.function
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                        for i in pl.range(8, chunk=4, unroll=2):
                            x = pl.add(x, 1.0)
                    return x

    def test_unroll_runtime_value_rejected(self):
        """Non-constant ``unroll`` is rejected by the parser (must be a literal int)."""
        with pytest.raises(Exception, match=r"unroll must be a compile-time constant"):

            @pl.program
            class _P:
                @pl.function
                def main(
                    self, x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT64]
                ) -> pl.Tensor[[64], pl.FP32]:
                    for i in pl.range(8, unroll=n):  # pyright: ignore[reportArgumentType]
                        x = pl.add(x, 1.0)
                    return x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
