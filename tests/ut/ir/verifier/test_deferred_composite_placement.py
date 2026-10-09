# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for DeferredCompositePlacementValid (deferred-placement check).

``pld.tensor.*(defer=True)`` must sit in a task-level ``pl.at(CORE_GROUP)`` body.
Bare Orchestration placement is rejected at pipeline input so LowerCompositeOps
cannot silently skip the flag on non-InCore rails.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.pypto_core import passes


def _verify(prog):
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.DeferredCompositePlacementValid)
    return passes.PropertyVerifierRegistry.verify(props, prog)


def _legal_deferred_allgather():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[2, 64], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ):
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer") as _ag_tid:
                    pld.tensor.allgather(inp, data, signal, defer=True)

    return Prog


def _bare_orch_deferred_allgather():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[2, 64], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ):
            # Illegal: defer=True outside a task-level pl.at.
            pld.tensor.allgather(inp, data, signal, defer=True)

    return Prog


def _nested_spmd_deferred_allgather():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[2, 64], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ):
            with pl.manual_scope():
                with pl.spmd(4, name_hint="outer"):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer") as _ag_tid:
                        pld.tensor.allgather(inp, data, signal, defer=True)

    return Prog


def _early_resolve_deferred_allgather():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[2, 64], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ):
            with pl.manual_scope():
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="ag_defer",
                    allow_early_resolve=True,
                ) as _ag_tid:
                    pld.tensor.allgather(inp, data, signal, defer=True)

    return Prog


def _nested_if_deferred_allgather():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[2, 64], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
            flag: pl.Scalar[pl.BOOL],
        ):
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer") as _ag_tid:
                    if flag:
                        pld.tensor.allgather(inp, data, signal, defer=True)

    return Prog


def _orch_if_wrapping_deferred_pl_at():
    """Orchestration ``if`` around a whole ``pl.at`` task — legal CFG edge."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, 64], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[2, 64], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
            flag: pl.Scalar[pl.BOOL],
        ):
            with pl.manual_scope():
                if flag:
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer") as _ag_tid:
                        pld.tensor.allgather(inp, data, signal, defer=True)

    return Prog


class TestDeferredCompositePlacementValid:
    def test_legal_task_level_placement_passes(self):
        diags = _verify(_legal_deferred_allgather())
        assert diags == []

    def test_bare_orchestration_rejected(self):
        diags = _verify(_bare_orch_deferred_allgather())
        assert len(diags) >= 1
        assert any(d.rule_name == "DeferredCompositePlacementValid" for d in diags)
        assert any("defer=True" in d.message for d in diags)

    def test_nested_under_spmd_rejected(self):
        diags = _verify(_nested_spmd_deferred_allgather())
        assert len(diags) >= 1
        assert any("task-level" in d.message or "pl.spmd" in d.message for d in diags)

    def test_allow_early_resolve_rejected(self):
        diags = _verify(_early_resolve_deferred_allgather())
        assert len(diags) >= 1
        assert any("allow_early_resolve" in d.message for d in diags)

    def test_nested_under_if_rejected(self):
        diags = _verify(_nested_if_deferred_allgather())
        assert len(diags) >= 1
        assert any("if/for/while" in d.message for d in diags)

    def test_orch_if_wrapping_pl_at_passes(self):
        """``if`` outside ``pl.at`` (orchestration CFG) must not false-reject."""
        diags = _verify(_orch_if_wrapping_deferred_pl_at())
        assert diags == [], [d.message for d in diags]

    def test_post_outline_stamped_body_still_passes(self):
        """VerificationInstrument re-checks after OutlineIncoreScopes."""
        prog = _legal_deferred_allgather()
        ssa = passes.convert_to_ssa()(prog)
        outlined = passes.outline_incore_scopes()(ssa)
        diags = _verify(outlined)
        assert diags == [], [d.message for d in diags]
