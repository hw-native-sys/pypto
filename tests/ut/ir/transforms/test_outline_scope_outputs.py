# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scope outputs follow external uses, including when that set is empty."""

import pypto.language as pl
import pytest
from pypto import ir, passes


def test_nested_scope_propagates_external_use():
    """A wrapper must pass the outer return's requirement to its direct child."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.cluster():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kernel"):
                    y = pl.add(x, x)
            return y

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y = pl.add(x, x)
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.cluster():
                y = self.kernel(x)
            return y

    after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
    ir.assert_structural_equal(after, passes.convert_to_ssa()(Expected))


@pytest.mark.parametrize("nested", [False, True], ids=["function-body", "spmd-body"])
def test_terminal_scope_does_not_export_local_tile_alias(nested):
    """A local in-place result is not an output just because it aliases a write.

    Both a terminal function-body scope and a direct child of an SPMD wrapper
    have known-empty external value uses. The caller-owned store target still
    escapes through the independent store-output path.
    """

    @pl.program
    class Before:
        @pl.function
        def main(
            self,
            x: pl.Tensor[[16, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kernel"):
                tile = pl.load(x, [0, 0], [16, 32])
                padded = pl.tile.fillpad_inplace(tile)
                pl.store(padded, [0, 0], out)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            tile = pl.load(x, [0, 0], [16, 32])
            padded = pl.tile.fillpad_inplace(tile)
            _stored = pl.store(padded, [0, 0], out)
            return out

    class WrapScope(ir.IRMutator):
        def visit_in_core_scope_stmt(self, op):
            return ir.SpmdScopeStmt(1, body=op, span=op.span)

    before = passes.convert_to_ssa()(Before)
    if nested:
        before = WrapScope().visit_program(before)
    after = passes.outline_incore_scopes()(before)
    ir.assert_structural_equal(
        after.get_function("kernel"), passes.convert_to_ssa()(Expected).get_function("kernel")
    )
    # A Tile declared as an InCore return would fail here, even if unused.
    passes.convert_tensor_to_tile_ops()(passes.outline_cluster_scopes()(after))


def test_terminal_scope_does_not_export_loop_local_write_alias():
    """A post-write binding defined inside a loop stays inside that loop."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[16, 32], pl.FP32]):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kernel"):
                work = pl.create_tensor([16, 32], dtype=pl.FP32)
                for _i in pl.range(2):
                    _updated = pl.assemble(work, x, [0, 0])

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, x: pl.Tensor[[16, 32], pl.FP32]):
            work = pl.create_tensor([16, 32], dtype=pl.FP32)
            for _i in pl.range(2):
                _updated = pl.assemble(work, x, [0, 0])

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16, 32], pl.FP32]):
            self.kernel(x)

    after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
    ir.assert_structural_equal(after, passes.convert_to_ssa()(Expected))


def test_live_write_alias_remains_an_output():
    """A real outer use selects an SSA writer's result without any fallback."""

    @pl.program
    class Before:
        @pl.function
        def main(
            self, x: pl.Tensor[[16, 32], pl.FP32], dst: pl.Tensor[[16, 32], pl.FP32]
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kernel"):
                updated = pl.assemble(dst, x, [0, 0])
            return updated

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, dst: pl.Out[pl.Tensor[[16, 32], pl.FP32]], x: pl.Tensor[[16, 32], pl.FP32]
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            _updated = pl.assemble(dst, x, [0, 0])
            return dst

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[16, 32], pl.FP32], dst: pl.Tensor[[16, 32], pl.FP32]
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            updated = self.kernel(dst, x)
            return updated

    after = passes.outline_incore_scopes()(passes.convert_to_ssa()(Before))
    ir.assert_structural_equal(after, passes.convert_to_ssa()(Expected))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
