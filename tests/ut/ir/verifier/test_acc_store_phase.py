# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the final accumulator/store phase-pairing verifier.

On A2/A3, ``acc_phase=pl.AccPhase.Final`` is producer-side check-and-set and
``st_phase=pl.STPhase.Final`` is consumer-side check-and-clear. Either missing side can
stall the device rather than producing a normal runtime error, so
``AccStorePhaseValid`` rejects the mismatch immediately after Inline helpers
are expanded, while the IR still carries the user's source spans.
"""

import pypto
import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes


def _verify(program):
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AccStorePhaseValid)
    return passes.PropertyVerifierRegistry.verify(props, program)


def _gemv_program(
    *,
    producer_phase=pl.AccPhase.Final,
    store_phase=pl.STPhase.Final,
):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv(lhs, rhs, acc_phase=producer_phase)
            result_alias = result
            out = pl.store(result_alias, [0, 0], out, st_phase=store_phase)

    return Prog


def _double_store_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            out2: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)
            out2 = pl.store(result, [0, 0], out2, st_phase=pl.STPhase.Final)

    return Prog


def _unconsumed_final_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
            _ = result

    return Prog


def _gemv_acc_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 256], pl.FP32],
            b: pl.Tensor[[256, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs0 = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs0 = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.gemv(lhs0, rhs0, acc_phase=pl.AccPhase.Partial)
            lhs1 = pl.load(a, [0, 128], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs1 = pl.load(b, [128, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv_acc(acc, lhs1, rhs1, acc_phase=pl.AccPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def _gemv_bias_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            bias: pl.Tensor[[1, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            bias_tile = pl.load(bias, [0, 0], [1, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv_bias(lhs, rhs, bias_tile, acc_phase=pl.AccPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def _cross_if_boundary_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            cond: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
            if cond < 1:
                out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def _pair_inside_if_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            cond: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            if cond < 1:
                lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
                rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                result = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
                out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def _inline_final_producer_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Inline)
        def final_gemv(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
        ) -> pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[1, 64])]:
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            return pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)

        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            result = self.final_gemv(a, b)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def test_matching_final_gemv_and_store_are_accepted():
    assert _verify(_gemv_program()) == []


def test_all_phase_aware_gemv_variants_are_accepted():
    assert _verify(_gemv_acc_program()) == []
    assert _verify(_gemv_bias_program()) == []


def test_final_producer_with_plain_store_is_rejected():
    diagnostics = _verify(_gemv_program(store_phase=pl.STPhase.Unspecified))
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "st_phase=pl.STPhase.Unspecified" in diagnostics[0].message
    assert "st_phase=pl.STPhase.Final" in diagnostics[0].message


def test_invalid_store_phase_reports_a_diagnostic():
    """Malformed IR is diagnosed without formatting the invalid enum value."""

    class _StampInvalidStorePhase(ir.IRMutator):
        def visit_call(self, op):
            rewritten = super().visit_call(op)
            assert isinstance(rewritten, ir.Call)
            if rewritten.op.name != ir.get_op("tile.store").name:
                return rewritten
            kwargs = dict(rewritten.kwargs)
            kwargs["st_phase"] = 99
            return ir.Call(
                rewritten.op,
                list(rewritten.args),
                kwargs,
                dict(rewritten.attrs),
                rewritten.type,
                rewritten.span,
            )

    malformed = _StampInvalidStorePhase().visit_program(_gemv_program())
    diagnostics = _verify(malformed)

    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "must encode pl.STPhase.Unspecified (0) or pl.STPhase.Final (3)" in diagnostics[0].message
    assert "got 99" in diagnostics[0].message


def test_final_store_without_final_producer_is_rejected():
    diagnostics = _verify(_gemv_program(producer_phase=pl.AccPhase.Unspecified))
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "unit flag that was never set" in diagnostics[0].message


def test_unconsumed_final_producer_is_rejected():
    diagnostics = _verify(_unconsumed_final_program())
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "must be consumed exactly once" in diagnostics[0].message


def test_second_final_store_of_same_value_is_rejected():
    diagnostics = _verify(_double_store_program())
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "does not have a live matching" in diagnostics[0].message


def test_partial_producer_and_plain_store_remain_legal():
    assert (
        _verify(_gemv_program(producer_phase=pl.AccPhase.Partial, store_phase=pl.STPhase.Unspecified)) == []
    )


def test_pair_may_not_cross_a_control_flow_boundary():
    diagnostics = _verify(_cross_if_boundary_program())
    assert diagnostics
    assert all(diag.rule_name == "AccStorePhaseValid" for diag in diagnostics)
    assert any("before entering an if statement" in diag.message for diag in diagnostics)


def test_balanced_pair_inside_a_branch_is_accepted():
    assert _verify(_pair_inside_if_program()) == []


def test_phase_property_is_produced_after_inline_not_structural():
    assert not passes.get_structural_properties().contains(passes.IRProperty.AccStorePhaseValid)
    assert passes.inline_functions().get_produced_properties().contains(passes.IRProperty.AccStorePhaseValid)


def test_default_pipeline_accepts_pair_joined_by_inline_expansion():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    try:
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(_inline_final_producer_program())
    finally:
        backend.reset_for_testing()


def test_default_pipeline_rejects_mismatch_after_inline_functions():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    try:
        with pytest.raises(pypto.Error, match="AccStorePhaseValid"):
            PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
                _gemv_program(store_phase=pl.STPhase.Unspecified)
            )
    finally:
        backend.reset_for_testing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
