# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OutlineIncoreScopes pass.

OutlineIncoreScopes outlines `HierarchyScopeStmt(level=CORE_GROUP)` into
`Function(InCore)` and promotes the parent function from `Opaque` to
`Orchestration`. It runs after OutlineHierarchyScopes, which handles all
non-CORE_GROUP Hierarchy scopes.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


class TestOutlineIncoreScopes:
    """Test OutlineIncoreScopes pass."""

    def test_outline_simple_core_group_scope(self):
        """A single CORE_GROUP scope becomes an InCore function; main is promoted to Orchestration."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)

        func_types = {gv.name: func.func_type for gv, func in After.functions.items()}
        # Parent promoted
        assert func_types["main"] == ir.FunctionType.Orchestration
        # Exactly one outlined InCore function with "core_group" in its name
        incore_funcs = [(n, t) for n, t in func_types.items() if t == ir.FunctionType.InCore]
        assert len(incore_funcs) == 1
        assert "core_group" in incore_funcs[0][0]

    def test_outline_preserves_non_core_group_scopes(self):
        """Non-CORE_GROUP Hierarchy scopes are left intact for OutlineHierarchyScopes.

        Run with verification disabled because OutlineIncoreScopes claims to
        produce HierarchyOutlined; a leftover HOST scope (which would normally
        have been removed by OutlineHierarchyScopes earlier in the pipeline)
        intentionally fails that property — we only care that the pass itself
        is a no-op for non-CORE_GROUP scopes.
        """

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        with passes.PassContext([], passes.VerificationLevel.NONE):
            Before = passes.convert_to_ssa()(Before)
            After = passes.outline_incore_scopes()(Before)
        # Pass is a no-op — no CORE_GROUP scope present.
        ir.assert_structural_equal(After, Before)

    def test_outline_split_propagates_to_incore_function(self):
        """`pl.split(...)` on a CORE_GROUP scope is forwarded to the outlined InCore fn."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
                ):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)
        incore_funcs = [f for _, f in After.functions.items() if f.func_type == ir.FunctionType.InCore]
        assert len(incore_funcs) == 1
        # `split` attr round-trips as the SplitMode's underlying int value.
        attrs = dict(incore_funcs[0].attrs)
        assert attrs.get("split") == ir.SplitMode.UP_DOWN.value

    def test_pipeline_order_outlines_nested_core_group(self):
        """Hierarchy then Incore outlining cleanly handles a CORE_GROUP nested inside HOST."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        program = passes.convert_to_ssa()(Before)
        program = passes.outline_hierarchy_scopes()(program)
        program = passes.outline_incore_scopes()(program)

        func_types = {gv.name: func.func_type for gv, func in program.functions.items()}
        # The inner HOST function (which originally wrapped the CORE_GROUP scope)
        # must have been promoted to Orchestration when its CORE_GROUP child got
        # outlined. Distinguish it from the further-outlined CORE_GROUP function
        # (whose name extends `main_host_worker_…` with `_core_group_…`) by
        # filtering out names that *also* contain `core_group`.
        host_only_funcs = [n for n in func_types if "host_worker" in n and "core_group" not in n]
        assert len(host_only_funcs) == 1
        assert func_types[host_only_funcs[0]] == ir.FunctionType.Orchestration
        # An InCore function exists.
        assert any(t == ir.FunctionType.InCore for t in func_types.values())
        # main itself (which only contained the HOST scope, not a CORE_GROUP
        # directly) stays Opaque.
        assert func_types["main"] == ir.FunctionType.Opaque

    def test_no_core_group_passthrough(self):
        """Functions without CORE_GROUP scopes pass through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)
        ir.assert_structural_equal(After, Before)

    def test_outline_skips_non_opaque_functions(self):
        """Already-typed (InCore/Orchestration/...) functions are not touched."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)

        compute = After.get_function("compute")
        assert compute is not None
        assert compute.func_type == ir.FunctionType.InCore

        main = After.get_function("main")
        assert main is not None
        # main got promoted because its CORE_GROUP scope was outlined.
        assert main.func_type == ir.FunctionType.Orchestration

    def test_multiple_core_group_scopes_in_one_function(self):
        """Two sibling CORE_GROUP scopes both get outlined; parent promoted once."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                with pl.at(level=pl.Level.CORE_GROUP):
                    z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
                return z

        Before = passes.convert_to_ssa()(Before)
        After = passes.outline_incore_scopes()(Before)

        func_types = {gv.name: func.func_type for gv, func in After.functions.items()}
        assert func_types["main"] == ir.FunctionType.Orchestration
        incore_count = sum(1 for t in func_types.values() if t == ir.FunctionType.InCore)
        assert incore_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
