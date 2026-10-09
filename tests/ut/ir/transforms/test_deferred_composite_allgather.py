# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""UT: ``defer=True`` allgather outlines, lowers, and splits."""

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir, passes
from pypto.language.parser.diagnostics import InvalidOperationError

SIZE = 64
NR = 2

_ALLGATHER = ir.get_op("pld.tensor.allgather").name
_DEFER_WAIT = ir.get_op("pld.system.defer_wait").name
_SYSTEM_WAIT = ir.get_op("pld.system.wait").name


def _build_deferred_allgather_program():
    # Chip-orch only: host window buffers stamp opaque ``name=`` kwargs that the
    # IR printer omits, which breaks RoundtripInstrument after ConvertToSSA.
    @pl.program
    class DeferredAllGather:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, NR * SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ) -> pl.Tensor[[1, NR * SIZE], pl.FP32]:
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer") as ag_tid:
                    pld.tensor.allgather(inp, data, signal, defer=True)

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="ag_consume",
                    deps=[ag_tid],
                ):
                    for r in pl.range(NR):
                        chunk = pl.load(data, [r, 0], [1, SIZE])
                        pl.store(chunk, [0, r * SIZE], out)
            return out

    return DeferredAllGather


def _build_deferred_allgather_no_tid_binding():
    """defer=True without ``as tid`` — outline must still emit Submit for split."""

    @pl.program
    class DeferredAllGatherNoTid:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
        ):
            with pl.manual_scope():
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer"):
                    pld.tensor.allgather(inp, data, signal, defer=True)

    return DeferredAllGatherNoTid


class TestDeferredCompositeAllGather:
    def test_dsl_accepts_defer_kwarg(self):
        prog = _build_deferred_allgather_program()
        chip = prog.get_function("chip_orch")
        assert chip is not None

        found = False

        class Finder(ir.IRVisitor):
            def visit_call(self, op):  # noqa: N802
                nonlocal found
                if hasattr(op.op, "name") and op.op.name == _ALLGATHER:
                    assert op.kwargs.get("defer") is True
                    found = True
                super().visit_call(op)

        Finder().visit_stmt(chip.body)
        assert found

    def test_outline_stamps_deferred_waiter_and_lower_emits_defer_wait(self):
        prog = _build_deferred_allgather_program()
        ssa = passes.convert_to_ssa()(prog)
        outlined = passes.outline_incore_scopes()(ssa)

        ag = outlined.get_function("ag_defer")
        assert ag is not None
        assert ag.attrs.get("deferred_completion_waiter") is True

        # Post-outline structural verify must not false-fail on stamped InCore bodies.
        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.DeferredCompositePlacementValid)
        diags = passes.PropertyVerifierRegistry.verify(props, outlined)
        assert diags == [], [d.message for d in diags]

        tiled = passes.convert_tensor_to_tile_ops()(outlined)
        lowered = passes.lower_composite_ops()(tiled)
        ag_lowered = lowered.get_function("ag_defer")
        assert ag_lowered is not None

        found_defer_wait = False
        found_blocking_wait = False

        class Finder(ir.IRVisitor):
            def visit_call(self, op):  # noqa: N802
                nonlocal found_defer_wait, found_blocking_wait
                name = getattr(op.op, "name", "")
                if name == _DEFER_WAIT:
                    found_defer_wait = True
                if name == _SYSTEM_WAIT:
                    found_blocking_wait = True
                super().visit_call(op)

        Finder().visit_stmt(ag_lowered.body)
        assert found_defer_wait, "defer=True allgather must emit pld.system.defer_wait"
        assert not found_blocking_wait, "defer=True allgather must not emit blocking pld.system.wait"

        # Split before ExpandMixedKernel: push / wait / epi siblings.
        # Satisfy SplitDeferredCompositeKernels' required properties without the
        # full Default tile pipeline.
        prepared = passes.flatten_tile_nd_to_2d()(lowered)
        prepared = passes.infer_tile_memory_space()(prepared)
        split = passes.split_deferred_composite_kernels()(prepared)
        assert split.get_function("ag_defer") is None
        assert split.get_function("ag_defer_push") is not None
        assert split.get_function("ag_defer_wait") is not None
        assert split.get_function("ag_defer_epi") is not None
        wait_fn = split.get_function("ag_defer_wait")
        assert wait_fn is not None
        assert wait_fn.attrs.get("deferred_completion_waiter") is True

    def test_outline_without_as_tid_still_emits_submit_and_splits(self):
        """Bare ``pl.at`` (no ``as tid``) must still outline as Submit so split can chain deps."""
        prog = _build_deferred_allgather_no_tid_binding()
        ssa = passes.convert_to_ssa()(prog)
        outlined = passes.outline_incore_scopes()(ssa)
        assert outlined.get_function("ag_defer") is not None

        chip = outlined.get_function("chip_orch")
        assert chip is not None
        found_submit = False
        found_plain_call = False

        class Finder(ir.IRVisitor):
            def visit_submit(self, op):  # noqa: N802
                nonlocal found_submit
                name = getattr(op.op, "name", "")
                if name == "ag_defer":
                    found_submit = True
                super().visit_submit(op)

            def visit_call(self, op):  # noqa: N802
                nonlocal found_plain_call
                name = getattr(op.op, "name", "")
                if name == "ag_defer":
                    found_plain_call = True
                super().visit_call(op)

        Finder().visit_stmt(chip.body)
        assert found_submit, "deferred waiter without as tid must outline as Submit"
        assert not found_plain_call, "deferred waiter must not outline as plain Call"

        tiled = passes.convert_tensor_to_tile_ops()(outlined)
        lowered = passes.lower_composite_ops()(tiled)
        prepared = passes.flatten_tile_nd_to_2d()(lowered)
        prepared = passes.infer_tile_memory_space()(prepared)
        split = passes.split_deferred_composite_kernels()(prepared)
        assert split.get_function("ag_defer") is None
        assert split.get_function("ag_defer_push") is not None
        assert split.get_function("ag_defer_wait") is not None
        assert split.get_function("ag_defer_epi") is not None

        chip_split = split.get_function("chip_orch")
        assert chip_split is not None
        submit_names: list[str] = []

        class SplitFinder(ir.IRVisitor):
            def visit_submit(self, op):  # noqa: N802
                name = getattr(op.op, "name", "")
                if name.startswith("ag_defer"):
                    submit_names.append(name)
                    if name == "ag_defer_wait":
                        assert len(op.deps) == 1
                    if name == "ag_defer_epi":
                        assert len(op.deps) == 1
                super().visit_submit(op)

        SplitFinder().visit_stmt(chip_split.body)
        assert "ag_defer_push" in submit_names
        assert "ag_defer_wait" in submit_names
        assert "ag_defer_epi" in submit_names

    def test_split_avoids_phase_name_collision(self):
        """Pre-existing ``*_push`` outline must not collide with split phase names."""

        @pl.program
        class CollisionProg:
            @pl.function(type=pl.FunctionType.Orchestration)
            def chip_orch(
                self,
                inp: pl.Tensor[[1, SIZE], pl.FP32],
                out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
                data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
            ) -> pl.Tensor[[1, SIZE], pl.FP32]:
                with pl.manual_scope():
                    # Occupies the preferred push name before the deferred scope splits.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer_push"):
                        chunk = pl.load(inp, [0, 0], [1, SIZE])
                        pl.store(chunk, [0, 0], out)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_defer") as ag_tid:
                        pld.tensor.allgather(inp, data, signal, defer=True)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ag_consume", deps=[ag_tid]):
                        chunk = pl.load(data, [0, 0], [1, SIZE])
                        pl.store(chunk, [0, 0], out)
                return out

        prog = CollisionProg
        ssa = passes.convert_to_ssa()(prog)
        outlined = passes.outline_incore_scopes()(ssa)
        assert outlined.get_function("ag_defer_push") is not None
        assert outlined.get_function("ag_defer") is not None
        tiled = passes.convert_tensor_to_tile_ops()(outlined)
        lowered = passes.lower_composite_ops()(tiled)
        prepared = passes.flatten_tile_nd_to_2d()(lowered)
        prepared = passes.infer_tile_memory_space()(prepared)
        split = passes.split_deferred_composite_kernels()(prepared)
        assert split.get_function("ag_defer") is None
        # Original unrelated outline keeps the bare ``_push`` name.
        assert split.get_function("ag_defer_push") is not None
        # Split claims a fresh suffix instead of duplicating.
        assert split.get_function("ag_defer_push_1") is not None
        assert split.get_function("ag_defer_wait") is not None
        assert split.get_function("ag_defer_epi") is not None

    def test_defer_rejected_on_allreduce(self):
        with pytest.raises(InvalidOperationError, match="defer"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.Orchestration)
                def chip_orch(
                    self,
                    data: pl.InOut[pld.DistributedTensor[[NR, SIZE], pl.FP32]],
                    signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
                ):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ar"):
                        data = pld.tensor.allreduce(data, signal, defer=True)  # type: ignore[call-arg]
