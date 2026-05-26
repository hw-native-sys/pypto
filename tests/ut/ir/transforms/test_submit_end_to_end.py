# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end integration test for the Submit IR migration.

The original motivation: dumps captured after InferTileMemorySpace
(pass 18) print kernel submissions as ``self.stage1(...)`` with an
implicit tuple-augmented return type — visually indistinguishable from
a plain function call. With the Submit IR kind plus the parser flip,
those mid-pipeline dumps now use the source-level
``pl.submit(self.stage1, ..., deps=[...])`` form, matching what users
write in the DSL. DeriveCallDirections (pass 34) lowers Submit → Call
so late passes and codegen are unaffected.
"""

import pypto.language as pl
import pytest
from pypto import passes


def _ssa_then_print(prog) -> str:
    """Run only the early passes (through ConvertToSSA / Simplify) and return
    the printed IR. This mirrors what a mid-pipeline dump captures."""
    prog = passes.inline_functions()(prog)
    prog = passes.unroll_loops()(prog)
    prog = passes.ctrl_flow_transform()(prog)
    prog = passes.convert_to_ssa()(prog)
    prog = passes.simplify()(prog)
    return prog.as_python()


def test_submit_visible_in_mid_pipeline_dump():
    """The pl.submit form survives the early passes and is visible in the
    post-Simplify dump — the canonical mid-pipeline reading point."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def stage1(
            self,
            x: pl.Tensor[[16, 256], pl.FP32],
            scratch: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
        ) -> pl.Tensor[[16, 256], pl.FP32]:
            return scratch

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[16, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
        ) -> pl.Tensor[[16, 256], pl.FP32]:
            with pl.manual_scope():
                _scratch, _tid = pl.submit(self.stage1, x, out)
            return out

    text = _ssa_then_print(Prog)
    # The user's wish: pl.submit visibility in mid-pipeline dumps.
    assert "pl.submit(self.stage1" in text, text
    # And the legacy bare-Call form must NOT be how kernel submission is
    # rendered — otherwise the user's complaint persists.
    assert "= self.stage1(" not in text, text


def test_submit_with_deps_visible_in_mid_pipeline_dump():
    """When ``deps=[tid]`` is attached, the mid-pipeline dump surfaces the
    typed Submit.deps_ field as a ``deps=[t1]`` kwarg."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.manual_scope():
                a, a_tid = pl.submit(self.producer, x)
                b, _ = pl.submit(self.consumer, a, deps=[a_tid])
            return b

    text = _ssa_then_print(Prog)
    assert "pl.submit(self.producer" in text
    assert "pl.submit(self.consumer" in text
    assert "deps=[" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
