# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Whole-function vector AutoTile planning and emission tests.

The pass is intentionally all-or-nothing: an explicitly marked static tensor
DAG becomes one exact SPMD/InCore vector schedule, or compilation fails.  These
tests run the real pre-outline pipeline and pin both that admission contract and
the planned algorithm (loads, phases, loops, and live-out stores).
"""

from __future__ import annotations

from collections import Counter

import pypto.language as pl
import pytest
from pypto import LogLevel, backend, ir, passes, set_log_level
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


@pytest.fixture(autouse=True)
def _ascend_910b_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_auto_tile(program: ir.Program) -> ir.Program:
    """Run the exact Default-pipeline prefix that precedes AutoTile."""
    program = passes.convert_to_ssa()(program)
    program = passes.simplify()(program)
    program = passes.normalize_stmt_structure()(program)
    program = passes.flatten_call_expr()(program)
    return passes.auto_tile()(program)


def _run_default(program: ir.Program) -> ir.Program:
    """Run AutoTile through the complete production pass pipeline."""
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


class _Structure(ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.ops: Counter[str] = Counter()
        self.spmd = 0
        self.spmd_core_nums: list[int] = []
        self.pipeline_loops = 0
        self.atomic_stores = 0

    def visit_call(self, op: ir.Call) -> None:
        self.ops[op.op.name] += 1
        if op.op.name == "tensor.assemble" and op.kwargs.get("atomic", 0) != 0:
            self.atomic_stores += 1
        super().visit_call(op)

    def visit_spmd_scope_stmt(self, op: ir.SpmdScopeStmt) -> None:
        self.spmd += 1
        assert isinstance(op.core_num, ir.ConstInt)
        self.spmd_core_nums.append(int(op.core_num.value))
        super().visit_spmd_scope_stmt(op)

    def visit_for_stmt(self, op: ir.ForStmt) -> None:
        if op.kind == ir.ForKind.Pipeline:
            self.pipeline_loops += 1
        super().visit_for_stmt(op)


def _structure(program: ir.Program) -> _Structure:
    result = _Structure()
    result.visit_program(program)
    return result


def _function(program: ir.Program, name: str) -> ir.Function:
    function = program.get_function(name)
    assert function is not None
    return function


def _logged_plan(program: ir.Program, capfd: pytest.CaptureFixture[str]) -> tuple[ir.Program, str]:
    """Run AutoTile and return its single machine-readable selected-plan log."""
    capfd.readouterr()
    set_log_level(LogLevel.INFO)
    try:
        after = _run_auto_tile(program)
    finally:
        set_log_level(LogLevel.INFO)
    lines = [line for line in capfd.readouterr().err.splitlines() if "AutoTile[" in line]
    assert len(lines) == 1
    return after, lines[0]


@pl.program
class PointwiseProgram:
    @pl.function(attrs={"auto_tile": True})
    def pointwise(self, x: pl.Tensor[[64, 256], pl.FP32]) -> pl.Tensor[[64, 256], pl.FP32]:
        a: pl.Tensor[[64, 256], pl.FP32] = pl.exp(x)
        b: pl.Tensor[[64, 256], pl.FP32] = pl.add(a, x)
        out: pl.Tensor[[64, 256], pl.FP32] = pl.mul(b, a)
        return out


@pl.program
class RepeatedInputProgram:
    @pl.function(attrs={"auto_tile": True})
    def square(self, x: pl.Tensor[[64, 256], pl.FP32]) -> pl.Tensor[[64, 256], pl.FP32]:
        out: pl.Tensor[[64, 256], pl.FP32] = pl.mul(x, x)
        return out


@pl.program
class ExplicitOutProgram:
    @pl.function(attrs={"auto_tile": True})
    def pointwise(
        self,
        x: pl.Tensor[[64, 256], pl.FP32],
        y: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> pl.Tensor[[64, 256], pl.FP32]:
        y = pl.exp(x)
        return y


@pl.program
class ExplicitMultiOutProgram:
    @pl.function(attrs={"auto_tile": True})
    def pointwise(
        self,
        x: pl.Tensor[[64, 256], pl.FP32],
        first: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
        second: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> tuple[pl.Tensor[[64, 256], pl.FP32], pl.Tensor[[64, 256], pl.FP32]]:
        first = pl.exp(x)
        second = pl.add(first, 1.0)
        return first, second


@pl.program
class ExplicitOutDirectCallProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[64, 256], pl.FP32],
        y: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> pl.Tensor[[64, 256], pl.FP32]:
        y = pl.exp(x)
        return y

    @pl.function
    def main(
        self,
        x: pl.Tensor[[64, 256], pl.FP32],
        y: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> pl.Tensor[[64, 256], pl.FP32]:
        out: pl.Tensor[[64, 256], pl.FP32] = self.kernel(x, y)
        return out


@pl.program
class ExplicitOutSubmitProgram:
    @pl.function(attrs={"auto_tile": True})
    def kernel(
        self,
        x: pl.Tensor[[64, 256], pl.FP32],
        y: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> pl.Tensor[[64, 256], pl.FP32]:
        y = pl.exp(x)
        return y

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[64, 256], pl.FP32],
        y: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> pl.Tensor[[64, 256], pl.FP32]:
        with pl.manual_scope():
            out, _tid = pl.submit(self.kernel, x, y)
        return out


@pl.program
class MultiOutputProgram:
    @pl.function(attrs={"auto_tile": True})
    def live_out(
        self, x: pl.Tensor[[64, 256], pl.FP32]
    ) -> tuple[pl.Tensor[[64, 256], pl.FP32], pl.Tensor[[64, 256], pl.FP32]]:
        a: pl.Tensor[[64, 256], pl.FP32] = pl.exp(x)
        b: pl.Tensor[[64, 256], pl.FP32] = pl.add(a, 1.0)
        return a, b


@pl.program
class WideMultiOutputProgram:
    @pl.function(attrs={"auto_tile": True})
    def live_out(
        self, x: pl.Tensor[[128, 8192], pl.FP32]
    ) -> tuple[pl.Tensor[[128, 8192], pl.FP32], pl.Tensor[[128, 8192], pl.FP32]]:
        a: pl.Tensor[[128, 8192], pl.FP32] = pl.exp(x)
        b: pl.Tensor[[128, 8192], pl.FP32] = pl.add(a, 1.0)
        return a, b


@pl.program
class WidePointwiseProgram:
    @pl.function(attrs={"auto_tile": True})
    def wide(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.FP32]:
        a: pl.Tensor[[128, 8192], pl.FP32] = pl.add(x, 1.0)
        out: pl.Tensor[[128, 8192], pl.FP32] = pl.mul(a, 2.0)
        return out


@pl.program
class RowReductionProgram:
    @pl.function(attrs={"auto_tile": True})
    def reduce(self, x: pl.Tensor[[64, 4096], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
        out: pl.Tensor[[64, 1], pl.FP32] = pl.row_sum(x)
        return out


@pl.program
class NarrowRowReductionProgram:
    @pl.function(attrs={"auto_tile": True})
    def reduce(self, x: pl.Tensor[[16384, 16], pl.FP32]) -> pl.Tensor[[16384, 1], pl.FP32]:
        out: pl.Tensor[[16384, 1], pl.FP32] = pl.row_sum(x)
        return out


@pl.program
class Bf16RowReductionProgram:
    @pl.function(attrs={"auto_tile": True})
    def reduce(self, x: pl.Tensor[[64, 4096], pl.BF16]) -> pl.Tensor[[64, 1], pl.BF16]:
        out: pl.Tensor[[64, 1], pl.BF16] = pl.row_sum(x)
        return out


@pl.program
class ReductionApplyProgram:
    @pl.function(attrs={"auto_tile": True})
    def rms(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.FP32]:
        sq: pl.Tensor[[128, 8192], pl.FP32] = pl.mul(x, x)
        total: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(sq)
        mean: pl.Tensor[[128, 1], pl.FP32] = pl.mul(total, 1.0 / 8192.0)
        norm: pl.Tensor[[128, 1], pl.FP32] = pl.rsqrt(pl.add(mean, 1.0e-6))
        out: pl.Tensor[[128, 8192], pl.FP32] = pl.mul(x, norm)
        return out


@pl.program
class ReductionMultiOutputProgram:
    @pl.function(attrs={"auto_tile": True})
    def reduce_live_out(
        self, x: pl.Tensor[[64, 1024], pl.FP32]
    ) -> tuple[pl.Tensor[[64, 1], pl.FP32], pl.Tensor[[64, 1024], pl.FP32]]:
        total: pl.Tensor[[64, 1], pl.FP32] = pl.row_sum(x)
        out: pl.Tensor[[64, 1024], pl.FP32] = pl.mul(x, total)
        return total, out


@pl.program
class SoftmaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def softmax(self, x: pl.Tensor[[32, 8192], pl.FP32]) -> pl.Tensor[[32, 8192], pl.FP32]:
        maximum: pl.Tensor[[32, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[32, 8192], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[32, 8192], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[32, 1], pl.FP32] = pl.row_sum(exponent)
        out: pl.Tensor[[32, 8192], pl.FP32] = pl.div(exponent, total)
        return out


@pl.program
class SmallSoftmaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def softmax(self, x: pl.Tensor[[8, 128], pl.FP32]) -> pl.Tensor[[8, 128], pl.FP32]:
        maximum: pl.Tensor[[8, 1], pl.FP32] = pl.row_max(x)
        shifted: pl.Tensor[[8, 128], pl.FP32] = pl.sub(x, maximum)
        exponent: pl.Tensor[[8, 128], pl.FP32] = pl.exp(shifted)
        total: pl.Tensor[[8, 1], pl.FP32] = pl.row_sum(exponent)
        out: pl.Tensor[[8, 128], pl.FP32] = pl.div(exponent, total)
        return out


@pl.program
class Int8OutputProgram:
    @pl.function(attrs={"auto_tile": True})
    def cast_output(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.INT8]:
        wide: pl.Tensor[[128, 8192], pl.FP32] = pl.add(x, 1.0)
        out: pl.Tensor[[128, 8192], pl.INT8] = pl.cast(wide, pl.INT8)
        return out


@pl.program
class ReductionInt8OutputProgram:
    @pl.function(attrs={"auto_tile": True})
    def cast_output(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.INT8]:
        total: pl.Tensor[[128, 1], pl.FP32] = pl.row_sum(x)
        centered: pl.Tensor[[128, 8192], pl.FP32] = pl.sub(x, total)
        out: pl.Tensor[[128, 8192], pl.INT8] = pl.cast(centered, pl.INT8)
        return out


@pl.program
class NativeCastProgram:
    @pl.function(attrs={"auto_tile": True})
    def cast_output(self, x: pl.Tensor[[128, 512], pl.FP32]) -> pl.Tensor[[128, 512], pl.FP16]:
        out: pl.Tensor[[128, 512], pl.FP16] = pl.cast(x, pl.FP16)
        return out


@pl.program
class Bf16ToFp16CastProgram:
    @pl.function(attrs={"auto_tile": True})
    def cast_output(self, x: pl.Tensor[[48, 47000], pl.BF16]) -> pl.Tensor[[48, 47000], pl.FP16]:
        out: pl.Tensor[[48, 47000], pl.FP16] = pl.cast(x, pl.FP16)
        return out


@pl.program
class Fp16ToBf16CastProgram:
    @pl.function(attrs={"auto_tile": True})
    def cast_output(self, x: pl.Tensor[[128, 512], pl.FP16]) -> pl.Tensor[[128, 512], pl.BF16]:
        out: pl.Tensor[[128, 512], pl.BF16] = pl.cast(x, pl.BF16)
        return out


@pl.program
class RaggedPointwiseProgram:
    @pl.function(attrs={"auto_tile": True})
    def ragged(self, x: pl.Tensor[[130, 66], pl.FP32]) -> pl.Tensor[[130, 66], pl.FP32]:
        a: pl.Tensor[[130, 66], pl.FP32] = pl.abs(x)
        out: pl.Tensor[[130, 66], pl.FP32] = pl.add(a, x)
        return out


@pl.program
class Fp16PointwiseProgram:
    @pl.function(attrs={"auto_tile": True})
    def fp16(
        self,
        x: pl.Tensor[[128, 512], pl.FP16],
        y: pl.Tensor[[128, 512], pl.FP16],
    ) -> pl.Tensor[[128, 512], pl.FP16]:
        out: pl.Tensor[[128, 512], pl.FP16] = pl.mul(x, y)
        return out


@pl.program
class Bf16PointwiseProgram:
    @pl.function(attrs={"auto_tile": True})
    def bf16(
        self,
        x: pl.Tensor[[128, 512], pl.BF16],
        y: pl.Tensor[[128, 512], pl.BF16],
    ) -> pl.Tensor[[128, 512], pl.BF16]:
        out: pl.Tensor[[128, 512], pl.BF16] = pl.add(x, y)
        return out


@pl.program
class PointwiseVocabularyProgram:
    @pl.function(attrs={"auto_tile": True})
    def vocabulary(
        self,
        x: pl.Tensor[[64, 256], pl.FP32],
        y: pl.Tensor[[64, 256], pl.FP32],
    ) -> pl.Tensor[[64, 256], pl.FP32]:
        remainder: pl.Tensor[[64, 256], pl.FP32] = pl.fmod(x, y)
        lower: pl.Tensor[[64, 256], pl.FP32] = pl.maximum(remainder, -1.0)
        out: pl.Tensor[[64, 256], pl.FP32] = pl.minimum(lower, 1.0)
        return out


@pl.program
class HighPrecisionRsqrtProgram:
    @pl.function(attrs={"auto_tile": True})
    def rsqrt(self, x: pl.Tensor[[128, 8192], pl.FP32]) -> pl.Tensor[[128, 8192], pl.FP32]:
        out: pl.Tensor[[128, 8192], pl.FP32] = pl.rsqrt(x, high_precision=True)
        return out


@pl.program
class BroadcastProgram:
    @pl.function(attrs={"auto_tile": True})
    def broadcast(
        self,
        x: pl.Tensor[[64, 256], pl.FP32],
        row: pl.Tensor[[64, 1], pl.FP32],
        col: pl.Tensor[[1, 256], pl.FP32],
    ) -> pl.Tensor[[64, 256], pl.FP32]:
        with_row: pl.Tensor[[64, 256], pl.FP32] = pl.add(x, row)
        out: pl.Tensor[[64, 256], pl.FP32] = pl.add(col, with_row)
        return out


@pl.program
class RowExpandRepeatAlignedProgram:
    @pl.function(attrs={"auto_tile": True})
    def row_expand(
        self,
        x: pl.Tensor[[1, 64], pl.FP32],
        row: pl.Tensor[[1, 1], pl.FP32],
    ) -> pl.Tensor[[1, 64], pl.FP32]:
        out: pl.Tensor[[1, 64], pl.FP32] = pl.row_expand_add(x, row)
        return out


@pl.program
class RowExpandCountModeProgram:
    @pl.function(attrs={"auto_tile": True})
    def row_expand(
        self,
        x: pl.Tensor[[1, 256], pl.FP32],
        row: pl.Tensor[[1, 1], pl.FP32],
    ) -> pl.Tensor[[1, 256], pl.FP32]:
        out: pl.Tensor[[1, 256], pl.FP32] = pl.row_expand_add(x, row)
        return out


@pl.program
class RowMaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def reduce(self, x: pl.Tensor[[64, 4096], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
        out: pl.Tensor[[64, 1], pl.FP32] = pl.row_max(x)
        return out


@pl.program
class ColSumProgram:
    @pl.function(attrs={"auto_tile": True})
    def reduce(self, x: pl.Tensor[[2048, 8], pl.FP32]) -> pl.Tensor[[1, 8], pl.FP32]:
        out: pl.Tensor[[1, 8], pl.FP32] = pl.col_sum(x)
        return out


@pl.program
class ColMaxProgram:
    @pl.function(attrs={"auto_tile": True})
    def reduce(self, x: pl.Tensor[[2048, 64], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
        out: pl.Tensor[[1, 64], pl.FP32] = pl.col_max(x)
        return out


@pl.program
class CalledAutoTileProgram:
    @pl.function(attrs={"auto_tile": True})
    def worker(self, x: pl.Tensor[[64, 256], pl.FP32]) -> pl.Tensor[[64, 256], pl.FP32]:
        out: pl.Tensor[[64, 256], pl.FP32] = pl.exp(x)
        return out

    @pl.function
    def main(self, x: pl.Tensor[[64, 256], pl.FP32]) -> pl.Tensor[[64, 256], pl.FP32]:
        out: pl.Tensor[[64, 256], pl.FP32] = self.worker(x)
        return out


def test_unmarked_and_false_markers_are_noops():
    @pl.program
    class Program:
        @pl.function
        def unmarked(self, x: pl.Tensor[[16, 64], pl.FP32]) -> pl.Tensor[[16, 64], pl.FP32]:
            out: pl.Tensor[[16, 64], pl.FP32] = pl.exp(x)
            return out

        @pl.function(attrs={"auto_tile": False})
        def disabled(self, x: pl.Tensor[[16, 64], pl.FP32]) -> pl.Tensor[[16, 64], pl.FP32]:
            out: pl.Tensor[[16, 64], pl.FP32] = pl.exp(x)
            return out

    prepared = passes.flatten_call_expr()(
        passes.normalize_stmt_structure()(passes.simplify()(passes.convert_to_ssa()(Program)))
    )
    after = passes.auto_tile()(prepared)
    ir.assert_structural_equal(after, prepared)


def test_pointwise_is_one_kernel_and_marker_is_consumed():
    after = _run_auto_tile(PointwiseProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.ops["tensor.slice"] >= 1
    assert structure.ops["tensor.exp"] == 1
    assert structure.ops["tensor.add"] == 1
    assert structure.ops["tensor.mul"] == 1
    assert structure.ops["tensor.assemble"] == 1
    assert "auto_tile" not in _function(after, "pointwise").attrs


def test_repeated_operand_is_one_load_but_two_uses():
    after = _run_auto_tile(RepeatedInputProgram)
    structure = _structure(after)
    assert structure.ops["tensor.slice"] == 1
    assert structure.ops["tensor.mul"] == 1


@pytest.mark.parametrize(
    ("program", "expected_params", "expected_assembles"),
    [(ExplicitOutProgram, 2, 1), (ExplicitMultiOutProgram, 3, 2)],
)
def test_explicit_out_parameters_are_reused_without_duplicate_lifted_outputs(
    program, expected_params: int, expected_assembles: int
):
    after = _run_auto_tile(program)
    function = _function(after, "pointwise")
    structure = _structure(after)
    assert len(function.params) == expected_params
    assert list(function.param_directions).count(ir.ParamDirection.Out) == expected_assembles
    assert structure.ops["tensor.create"] == 0
    assert structure.ops["tensor.assemble"] == expected_assembles


@pytest.mark.parametrize("program", [ExplicitOutDirectCallProgram, ExplicitOutSubmitProgram])
def test_called_explicit_out_kernel_preserves_its_declared_signature(program):
    function = _function(_run_auto_tile(program), "kernel")
    assert len(function.params) == 2
    assert list(function.param_directions) == [ir.ParamDirection.In, ir.ParamDirection.Out]


@pytest.mark.parametrize("kind", ["count", "type"])
def test_explicit_out_mapping_is_all_or_none_and_type_exact(kind: str):
    if kind == "count":

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def kernel(
                self,
                x: pl.Tensor[[64, 256], pl.FP32],
                only: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 256], pl.FP32], pl.Tensor[[64, 256], pl.FP32]]:
                first: pl.Tensor[[64, 256], pl.FP32] = pl.exp(x)
                second: pl.Tensor[[64, 256], pl.FP32] = pl.add(first, 1.0)
                return first, second

    else:

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def kernel(
                self,
                x: pl.Tensor[[64, 256], pl.FP32],
                wrong: pl.Out[pl.Tensor[[64, 256], pl.FP16]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                out: pl.Tensor[[64, 256], pl.FP32] = pl.exp(x)
                return out

    with pytest.raises(ValueError, match="AutoTile"):
        _run_auto_tile(Program)


def test_inout_parameter_is_not_an_implicit_output_sink():
    @pl.program
    class Program:
        @pl.function(attrs={"auto_tile": True})
        def kernel(
            self,
            x: pl.Tensor[[64, 256], pl.FP32],
            state: pl.InOut[pl.Tensor[[64, 256], pl.FP32]],
        ) -> pl.Tensor[[64, 256], pl.FP32]:
            out: pl.Tensor[[64, 256], pl.FP32] = pl.add(x, state)
            return out

    function = _function(_run_auto_tile(Program), "kernel")
    assert len(function.params) == 3
    assert list(function.param_directions) == [
        ir.ParamDirection.In,
        ir.ParamDirection.InOut,
        ir.ParamDirection.Out,
    ]


def test_returned_intermediate_stays_live_for_two_distinct_stores():
    after = _run_auto_tile(MultiOutputProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.ops["tensor.exp"] == 1
    assert structure.ops["tensor.assemble"] == 2
    function = _function(after, "live_out")
    assert len(function.params) == 3  # one input plus two lifted Out buffers


def test_wide_multi_output_streams_with_both_live_outs_carried():
    after = _run_auto_tile(WideMultiOutputProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.pipeline_loops == 1
    assert structure.ops["tensor.exp"] == 1
    assert structure.ops["tensor.assemble"] == 2
    assert len(_function(after, "live_out").params) == 3


def test_oversized_pointwise_uses_a_two_stage_strip_pipeline():
    after = _run_auto_tile(WidePointwiseProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.pipeline_loops == 1
    assert structure.ops["tensor.slice"] == 1
    assert structure.ops["tensor.assemble"] == 1


def test_folded_and_spanning_reductions_emit_planned_phases():
    folded = _structure(_run_auto_tile(RowReductionProgram))
    spanning = _structure(_run_auto_tile(ReductionApplyProgram))
    assert folded.spmd == 1
    assert folded.ops["tensor.row_sum"] >= 1
    assert folded.ops["tensor.assemble"] == 1
    assert spanning.spmd == 1
    assert spanning.ops["tensor.row_sum"] >= 1
    assert spanning.ops["tensor.rsqrt"] >= 1
    assert spanning.ops["tensor.assemble"] >= 1
    assert spanning.pipeline_loops >= 1


def test_narrow_row_reduction_prices_the_lowerings_128_element_scratch():
    after = _run_auto_tile(NarrowRowReductionProgram)
    structure = _structure(after)
    # A 48-way grid would fit if the scratch were incorrectly priced like the
    # 16-column input.  Tensor-to-tile lowering pads it to 128 columns, so the
    # first capacity-safe balanced grid is two 48-core waves.
    assert structure.spmd_core_nums == [96]


def test_grounded_pointwise_chain_and_row_expand_count_mode_costs(capfd):
    _, pointwise = _logged_plan(PointwiseProgram, capfd)
    # One [32,64] strip executes exp -> add -> mul twice.  Only exp starts the
    # vector stream: (2*32+31 + 2*32 + 2*32) * 2 = 446 cycles.
    assert "strip=32x64" in pointwise
    assert "compute_cycles=446 " in pointwise

    _, aligned = _logged_plan(RowExpandRepeatAlignedProgram, capfd)
    _, count_mode = _logged_plan(RowExpandCountModeProgram, capfd)
    # FP32 has 64 elements/repeat. [1,64] pays 2*1+24. [1,256]
    # additionally satisfies cols/epr > rows and therefore pays the grounded
    # 16-cycle count-mask dispatch floor: 2*4+24+16 = 48.
    assert "compute_cycles=26 " in aligned
    assert "compute_cycles=48 " in count_mode


def test_grounded_reduction_tables_and_fallback_are_observable(capfd):
    _, row = _logged_plan(NarrowRowReductionProgram, capfd)
    _, col = _logged_plan(ColSumProgram, capfd)
    _, bf16 = _logged_plan(Bf16RowReductionProgram, capfd)
    # These are exact interpolation-table goldens, including the selected
    # grid's wave division. They are intentionally independent of Fusebox.
    assert "compute_cycles=2510 " in row
    assert "compute_cycles=38941 " in col
    assert "reduction_model=grounded" in row
    assert "reduction_model=grounded" in col
    assert "reduction_model=legacy_fallback" in bf16


def test_reduction_chunk_is_chosen_by_cost_not_first_capacity_fit(capfd):
    after, plan = _logged_plan(RowReductionProgram, capfd)
    structure = _structure(after)
    assert structure.spmd_core_nums == [16]
    # 1456 is the largest capacity-safe emitted chunk, but the exact
    # reduction/traffic roofline selects 512. This pins enumeration by modeled
    # cost instead of the former descending first-fit policy.
    assert "chunks=8x512+0" in plan
    assert "largest_feasible_chunk=1456" in plan
    assert "feasible_chunks=91" in plan


def test_generated_softmax_work_and_phase_traffic_are_exact(capfd):
    _, plan = _logged_plan(SoftmaxProgram, capfd)
    assert "chunks=17x480+32" in plan
    # Statistics are the exact generated init/update primitive tallies; apply
    # is the source DAG replay with persistent max/sum substituted.
    assert "compute_cycles=13612 " in plan
    assert "phase_compute=[0,10316,3296,0]" in plan
    # The input is read once by each pass and the one output is stored only by
    # apply. Padding remains UB-only and therefore does not inflate GM traffic.
    assert "phase_input_bytes=[0,1048576,1048576,0]" in plan
    assert "phase_output_bytes=[0,0,1048576,0]" in plan


def test_reduction_multi_live_out_uses_a_capacity_safe_materialized_region():
    after = _run_auto_tile(ReductionMultiOutputProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.pipeline_loops == 0
    assert structure.ops["tensor.row_sum"] == 1
    assert structure.ops["tensor.assemble"] == 2


def test_exact_wide_softmax_uses_online_stats_and_apply_passes():
    after = _run_auto_tile(SoftmaxProgram)
    structure = _structure(after)
    assert structure.spmd == 1
    assert structure.ops["tensor.row_max"] >= 1
    assert structure.ops["tensor.row_sum"] >= 1
    # One logical output is assembled once in the full-chunk loop and once for
    # the peeled tail in the static IR.
    assert structure.ops["tensor.assemble"] == 2
    assert structure.pipeline_loops >= 1


def test_terminal_fp32_to_int8_uses_dtype_exact_tiling():
    for program in (Int8OutputProgram, ReductionInt8OutputProgram):
        after = _run_auto_tile(program)
        structure = _structure(after)
        assert structure.spmd == 1
        # Reduction schedules replay the same two-hop path in both full-chunk
        # and tail regions, so static IR can contain more than two calls.
        assert structure.ops["tensor.cast"] >= 2
        assert structure.ops["tensor.cast"] % 2 == 0
        assert structure.ops["tensor.assemble"] >= 1


@pytest.mark.parametrize(
    ("program", "expected_targets"),
    [
        (NativeCastProgram, [pl.FP16]),
        (Bf16ToFp16CastProgram, [pl.FP32, pl.FP16]),
        (Fp16ToBf16CastProgram, [pl.FP32, pl.BF16]),
        (Int8OutputProgram, [pl.FP16, pl.INT8]),
    ],
)
def test_cast_plans_contain_the_complete_native_910b_conversion_path(program, expected_targets):
    class CastTargets(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.targets: list[ir.DataType] = []

        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == "tensor.cast":
                self.targets.append(op.kwargs["target_type"])
            super().visit_call(op)

    after = _run_auto_tile(program)
    casts = CastTargets()
    casts.visit_program(after)
    assert casts.targets == expected_targets
    assert _structure(after).spmd == 1
    tiled = passes.outline_hierarchy_scopes()(after)
    tiled = passes.outline_incore_scopes()(tiled)
    tiled = passes.outline_cluster_scopes()(tiled)
    tiled = passes.convert_tensor_to_tile_ops()(tiled)
    ir.assert_structural_equal(passes.legalize_tile_cast()(tiled), tiled)


@pytest.mark.parametrize(
    ("program", "op"),
    [
        (RaggedPointwiseProgram, "tensor.add"),
        (Fp16PointwiseProgram, "tensor.mul"),
        (Bf16PointwiseProgram, "tensor.add"),
        (PointwiseVocabularyProgram, "tensor.fmod"),
        (HighPrecisionRsqrtProgram, "tensor.rsqrt"),
    ],
)
def test_ragged_and_half_width_pointwise_are_admitted(program, op):
    structure = _structure(_run_auto_tile(program))
    assert structure.spmd == 1
    assert structure.ops[op] >= 1
    assert structure.ops["tensor.assemble"] >= 1


def test_unified_broadcasts_normalize_to_explicit_row_and_column_ops():
    structure = _structure(_run_auto_tile(BroadcastProgram))
    assert structure.ops["tensor.row_expand_add"] == 1
    assert structure.ops["tensor.col_expand_add"] == 1
    assert structure.ops["tensor.add"] == 0


@pytest.mark.parametrize(
    ("program", "op"),
    [
        (RowMaxProgram, "tensor.row_max"),
        (ColSumProgram, "tensor.col_sum"),
        (ColMaxProgram, "tensor.col_max"),
    ],
)
def test_all_supported_reduction_directions_and_kinds_are_admitted(program, op):
    structure = _structure(_run_auto_tile(program))
    assert structure.spmd == 1
    assert structure.ops[op] >= 1
    assert structure.ops["tensor.assemble"] >= 1


def test_aligned_terminal_col_sum_remains_one_non_atomic_kernel():
    structure = _structure(_run_auto_tile(ColSumProgram))
    assert structure.spmd == 1
    assert structure.ops["tensor.full"] == 0
    assert structure.atomic_stores == 0


def test_called_marked_helper_keeps_its_output_internal_and_lowers_fully():
    standalone = _run_auto_tile(CalledAutoTileProgram)
    worker = _function(standalone, "worker")
    assert len(worker.params) == 1
    assert _structure(standalone).ops["tensor.create"] == 1

    lowered = _run_default(CalledAutoTileProgram)
    function_types = {function.func_type for function in lowered.functions.values()}
    assert ir.FunctionType.AIV in function_types
    assert ir.FunctionType.Spmd in function_types


@pytest.mark.parametrize(
    "program",
    [
        PointwiseProgram,
        ExplicitOutProgram,
        ExplicitMultiOutProgram,
        WidePointwiseProgram,
        MultiOutputProgram,
        WideMultiOutputProgram,
        RaggedPointwiseProgram,
        Fp16PointwiseProgram,
        Bf16PointwiseProgram,
        PointwiseVocabularyProgram,
        HighPrecisionRsqrtProgram,
        BroadcastProgram,
        RowReductionProgram,
        NarrowRowReductionProgram,
        RowMaxProgram,
        ReductionApplyProgram,
        ReductionMultiOutputProgram,
        SoftmaxProgram,
        Int8OutputProgram,
        ReductionInt8OutputProgram,
        NativeCastProgram,
        Bf16ToFp16CastProgram,
        Fp16ToBf16CastProgram,
        ColSumProgram,
        ColMaxProgram,
    ],
)
def test_supported_programs_survive_the_complete_default_pipeline(program):
    lowered = _run_default(program)
    function_types = {function.func_type for function in lowered.functions.values()}
    assert ir.FunctionType.Orchestration in function_types
    assert ir.FunctionType.Spmd in function_types
    assert ir.FunctionType.AIV in function_types
    assert all("auto_tile" not in function.attrs for function in lowered.functions.values())


def test_marked_graph_requires_an_explicit_supported_backend():
    backend.reset_for_testing()
    with pytest.raises(ValueError, match="explicitly configured backend"):
        _run_auto_tile(PointwiseProgram)

    backend.set_backend_type(BackendType.Ascend950)
    with pytest.raises(ValueError, match="Ascend910B only"):
        _run_auto_tile(PointwiseProgram)


@pytest.mark.parametrize("kind", ["matmul", "row_min", "row_prod", "full", "rank3"])
def test_explicit_marker_fails_when_whole_graph_is_unsupported(kind: str):
    if kind == "matmul":

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out: pl.Tensor[[64, 64], pl.FP32] = pl.matmul(a, b)
                return out
    elif kind == "row_min":

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
                out: pl.Tensor[[64, 1], pl.FP32] = pl.row_min(x)
                return out
    elif kind == "row_prod":

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 1], pl.FP32]:
                out: pl.Tensor[[64, 1], pl.FP32] = pl.row_prod(x)
                return out
    elif kind == "full":

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(self) -> pl.Tensor[[64, 64], pl.FP32]:
                out: pl.Tensor[[64, 64], pl.FP32] = pl.full([64, 64], dtype=pl.FP32, value=0.0)
                return out
    else:

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(self, x: pl.Tensor[[2, 16, 64], pl.FP32]) -> pl.Tensor[[2, 16, 64], pl.FP32]:
                out: pl.Tensor[[2, 16, 64], pl.FP32] = pl.exp(x)
                return out

    with pytest.raises(ValueError, match="AutoTile"):
        _run_auto_tile(Program)


@pytest.mark.parametrize(
    "kind",
    ["non_commutative_lhs", "ambiguous", "high_precision", "bf16_div"],
)
def test_unsupported_broadcast_contracts_fail_during_auto_tile_admission(kind: str):
    if kind == "non_commutative_lhs":

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(
                self,
                x: pl.Tensor[[64, 256], pl.FP32],
                row: pl.Tensor[[64, 1], pl.FP32],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                out: pl.Tensor[[64, 256], pl.FP32] = pl.sub(row, x)
                return out
    elif kind == "ambiguous":

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(
                self,
                x: pl.Tensor[[64, 256], pl.FP32],
                scalar_tensor: pl.Tensor[[1, 1], pl.FP32],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                out: pl.Tensor[[64, 256], pl.FP32] = pl.add(x, scalar_tensor)
                return out
    elif kind == "high_precision":

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(
                self,
                x: pl.Tensor[[64, 256], pl.FP32],
                row: pl.Tensor[[64, 1], pl.FP32],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                out: pl.Tensor[[64, 256], pl.FP32] = pl.div(x, row, high_precision=True)
                return out
    else:

        @pl.program
        class Program:
            @pl.function(attrs={"auto_tile": True})
            def unsupported(
                self,
                x: pl.Tensor[[64, 256], pl.BF16],
                row: pl.Tensor[[64, 1], pl.BF16],
            ) -> pl.Tensor[[64, 256], pl.BF16]:
                out: pl.Tensor[[64, 256], pl.BF16] = pl.div(x, row)
                return out

    with pytest.raises(ValueError, match="AutoTile"):
        _run_auto_tile(Program)


def test_auto_tile_is_idempotent_after_consuming_the_marker():
    once = _run_auto_tile(PointwiseProgram)
    twice = passes.auto_tile()(once)
    ir.assert_structural_equal(twice, once)


@pytest.mark.parametrize(
    ("program", "entry", "inputs", "expected", "block_zero"),
    [
        (
            PointwiseProgram,
            "pointwise",
            lambda torch: (torch.randn(64, 256),),
            lambda torch, x: torch.mul(torch.add(torch.exp(x), x), torch.exp(x)),
            (slice(0, 64), slice(0, 64)),
        ),
        (
            RepeatedInputProgram,
            "square",
            lambda torch: (torch.randn(64, 256),),
            lambda torch, x: x * x,
            (slice(0, 64), slice(0, 64)),
        ),
        (
            SmallSoftmaxProgram,
            "softmax",
            lambda torch: (torch.randn(8, 128),),
            lambda torch, x: torch.softmax(x, dim=1),
            (slice(0, 1), slice(0, 128)),
        ),
        (
            BroadcastProgram,
            "broadcast",
            lambda torch: (
                torch.randn(64, 256),
                torch.randn(64, 1),
                torch.randn(1, 256),
            ),
            lambda torch, x, row, col: x + row + col,
            (slice(0, 64), slice(0, 64)),
        ),
    ],
)
def test_emitted_algorithm_matches_torch(program, entry, inputs, expected, block_zero):
    torch = pytest.importorskip("torch")
    from pypto.debug import torch_codegen  # noqa: PLC0415

    torch.manual_seed(0)
    values = inputs(torch)
    after = _run_auto_tile(program)
    namespace: dict[str, object] = {}
    # The upstream Torch interpreter intentionally maps get_block_idx() to
    # block zero.  Compare precisely that scheduled region; silicon STs cover
    # concurrent execution of the complete SPMD grid.
    exec(torch_codegen(after), namespace)  # noqa: S102
    reference = expected(torch, *values)
    actual = namespace[entry](*values, torch.empty_like(reference))
    assert torch.allclose(actual[block_zero], reference[block_zero], rtol=1e-4, atol=1e-4)
