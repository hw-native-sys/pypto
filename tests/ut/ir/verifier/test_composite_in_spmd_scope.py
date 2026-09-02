# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CompositeInSpmdScope: warn when a composite collective sits inside ``pl.spmd``.

The lowering never reads the block index, so every block of a ``pl.spmd(N)``
scope runs the whole peer loop: the transfer is duplicated N times rather than
divided N ways, and the barrier — expecting a compile-time ``1`` while N blocks
each notify ``+1`` — releases after a peer's *first* block.  None of that fails
today (blocks write byte-identical content, and the epilogue still zeroes the
signal), so without this check the cost is entirely silent.

Behaviour is pinned by ``tests/ut/ir/transforms/test_composite_in_spmd_partitioning.py``.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import passes

SIZE = 16
NRANKS = 2


def _verify(prog):
    """Diagnostics from the CompositeInSpmdScope check alone.

    PrePipeline: the composite Call must still exist — LowerCompositeOps
    replaces it during the pipeline.
    """
    checks = passes.DiagnosticCheckSet()
    checks.insert(passes.DiagnosticCheck.CompositeInSpmdScope)
    return passes.DiagnosticCheckRegistry.run_checks(checks, passes.DiagnosticPhase.PRE_PIPELINE, prog)


def _messages(prog):
    return [d.message for d in _verify(prog)]


def _allgather_in_spmd(width):
    @pl.program
    class InSpmd:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]:
            for _ in pl.spmd(width, name_hint="ag_spmd"):
                data = pld.tensor.allgather(inp, data, signal)
            return data

    return InSpmd


def _allgather_bare():
    @pl.program
    class Bare:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]:
            return pld.tensor.allgather(inp, data, signal)

    return Bare


class TestCompositeInSpmdIsWarned:
    def test_composite_in_spmd_warns(self):
        """A composite inside pl.spmd produces exactly one warning."""
        messages = _messages(_allgather_in_spmd(8))
        assert len(messages) == 1, f"expected one warning, got {messages}"
        assert "does NOT parallelise" in messages[0]
        assert "pld.tensor.allgather" in messages[0]

    def test_warning_names_the_static_width(self):
        """A compile-time width is reported as the actual multiplier, not 'N'."""
        assert "8 blocks" in _messages(_allgather_in_spmd(8))[0]
        assert "4 blocks" in _messages(_allgather_in_spmd(4))[0]

    def test_warning_points_at_the_host_rail(self):
        """The message must offer the path that does exist."""
        message = _messages(_allgather_in_spmd(2))[0]
        assert "core_num" in message and "HOST" in message


class TestNoFalsePositives:
    def test_bare_composite_is_silent(self):
        """A composite outside any spmd scope is the supported form."""
        assert _messages(_allgather_bare()) == []

    def test_spmd_without_a_composite_is_silent(self):
        """The check keys on the collective, not on pl.spmd itself."""

        @pl.program
        class SpmdOnly:
            @pl.function(type=pl.FunctionType.InCore)
            def add_step(
                self,
                x: pl.Tensor[[NRANKS, SIZE], pl.FP32],
                out: pl.Out[pl.Tensor[[NRANKS, SIZE], pl.FP32]],
            ) -> pl.Tensor[[NRANKS, SIZE], pl.FP32]:
                for _ in pl.spmd(4, name_hint="plain"):
                    tile = pl.load(x, [0, 0], [NRANKS, SIZE])
                    out = pl.store(pl.add(tile, tile), [0, 0], out)
                return out

        assert _messages(SpmdOnly) == []

    def test_spmd_width_one_is_silent(self):
        """pl.spmd(1) runs the collective exactly once — no duplication, no warning."""
        assert _messages(_allgather_in_spmd(1)) == []


class TestRegistryWiring:
    def test_check_is_selectable(self):
        checks = passes.DiagnosticCheckSet()
        checks.insert(passes.DiagnosticCheck.CompositeInSpmdScope)
        assert checks.contains(passes.DiagnosticCheck.CompositeInSpmdScope)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
