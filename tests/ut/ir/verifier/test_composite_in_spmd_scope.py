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


def _allreduce_in_spmd(width):
    @pl.program
    class ARInSpmd:
        @pl.function(type=pl.FunctionType.InCore)
        def ar_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            for _ in pl.spmd(width, name_hint="ar_spmd"):
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    return ARInSpmd


_AAV_SIZE = 16
_AAV_NRANKS = 2
_AAV_MAX_RECV = 2
_AAV_TOTAL = _AAV_NRANKS * _AAV_MAX_RECV


def _all_to_all_v_in_spmd(width):
    @pl.program
    class A2AVInSpmd:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[_AAV_TOTAL, _AAV_SIZE], pl.FP32],
            counts: pl.Tensor[[_AAV_NRANKS, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[_AAV_TOTAL, _AAV_SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[_AAV_TOTAL, _AAV_SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[_AAV_NRANKS, 1], pl.INT32]],
            recv_counts: pl.InOut[pld.DistributedTensor[[_AAV_NRANKS, 1], pl.INT32]],
        ) -> pl.Tensor[[_AAV_TOTAL, _AAV_SIZE], pl.FP32]:
            for _ in pl.spmd(width, name_hint="a2av_spmd"):
                data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv_counts)
            row = pl.load(data, [0, 0], [1, _AAV_SIZE])
            return pl.store(row, [0, 0], out)

    return A2AVInSpmd


def _allgather_via_inline_helper_in_spmd(width):
    """The composite sits in an ``Inline`` helper called from inside ``pl.spmd``.

    ``InlineFunctions`` (pass 01) splices ``helper``'s body into ``gather_step``
    at every call site, but this check runs PrePipeline — before that splice —
    so ``helper``'s body is still a separate function when the check runs.
    """

    @pl.program
    class InlineInSpmd:
        @pl.function(type=pl.FunctionType.Inline)
        def helper(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]:
            return pld.tensor.allgather(inp, data, signal)

        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]:
            for _ in pl.spmd(width, name_hint="ag_spmd"):
                data = self.helper(inp, data, signal)
            return data

    return InlineInSpmd


def _allgather_in_nested_spmd(outer_width, inner_width):
    """The composite sits under two nested ``pl.spmd`` scopes."""

    @pl.program
    class NestedSpmd:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]:
            for _ in pl.spmd(outer_width, name_hint="outer_spmd"):
                for _ in pl.spmd(inner_width, name_hint="inner_spmd"):
                    data = pld.tensor.allgather(inp, data, signal)
            return data

    return NestedSpmd


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


class TestMultiCoreAlternativeIsAccurate:
    """The remediation text must name only an alternative that actually exists.

    Only ``pld.tensor.allreduce`` has a working multi-core knob today
    (HOST-rail ``core_num``). ``pld.tensor.all_to_all_v`` accepts ``core_num``
    too, but only on the managed CHIP/L2 rail, with only ``core_num=1``
    implemented. Every other composite has no multi-core alternative on any
    rail — naming one would send a caller looking for a kwarg that is not
    there.
    """

    def test_allreduce_points_at_the_host_rail(self):
        message = _messages(_allreduce_in_spmd(2))[0]
        assert "core_num" in message and "HOST" in message

    def test_all_to_all_v_points_at_chip_l2_not_host(self):
        message = _messages(_all_to_all_v_in_spmd(2))[0]
        assert "core_num" in message
        assert "CHIP/L2" in message
        assert "HOST" not in message

    def test_allgather_names_no_false_alternative(self):
        """No knob exists for allgather — the message must not claim HOST core_num."""
        message = _messages(_allgather_in_spmd(2))[0]
        assert "HOST" not in message
        assert "core_num" not in message
        assert "No multi-core alternative" in message


class TestInlineCalleeDescent:
    """A composite reached only through an ``Inline`` helper call must still warn.

    This check runs PrePipeline — before ``InlineFunctions`` (pass 01) splices
    the helper's body into its call sites — so the composite sits in the
    helper's own function body when this check runs, with no enclosing
    ``pl.spmd`` visible unless the check follows the call into it.
    """

    def test_composite_behind_inline_helper_in_spmd_warns(self):
        messages = _messages(_allgather_via_inline_helper_in_spmd(8))
        assert len(messages) == 1, f"expected one warning, got {messages}"
        assert "does NOT parallelise" in messages[0]
        assert "8 blocks" in messages[0]

    def test_composite_behind_inline_helper_in_spmd_one_is_silent(self):
        """pl.spmd(1) around the call site is still the safe, single-block form."""
        assert _messages(_allgather_via_inline_helper_in_spmd(1)) == []


class TestNestedSpmdWidthsCompound:
    """Nested ``pl.spmd`` widths multiply; only the innermost being 1 is not enough.

    ``pl.spmd(8): pl.spmd(1): allgather(...)`` still runs the collective 8
    times — once per outer block, each running the inner scope's single
    block — so it must warn, and the multiplier it reports must be the
    product of every enclosing width (8), not the innermost width alone (1).
    """

    def test_inner_width_one_does_not_suppress_an_outer_width(self):
        messages = _messages(_allgather_in_nested_spmd(outer_width=8, inner_width=1))
        assert len(messages) == 1, f"expected one warning, got {messages}"
        assert "8 blocks" in messages[0]

    def test_every_level_known_one_is_silent(self):
        """Only when every enclosing width is provably 1 is there no duplication."""
        assert _messages(_allgather_in_nested_spmd(outer_width=1, inner_width=1)) == []

    def test_compile_time_widths_multiply(self):
        """Two non-trivial nested widths report their product, not either alone."""
        messages = _messages(_allgather_in_nested_spmd(outer_width=4, inner_width=2))
        assert "8 blocks" in messages[0]


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
