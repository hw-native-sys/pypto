# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ExpandMxPackedQuant."""

from collections import Counter

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _run_default_through(program: ir.Program, last_pass: str) -> ir.Program:
    manager = PassManager(OptimizationStrategy.Default)
    stop = manager.pass_names.index(last_pass)
    pipeline = passes.PassPipeline()
    for pass_obj in manager.passes[: stop + 1]:
        pipeline.add_pass(pass_obj)
    return pipeline.run(program)


def _chunk_lifetime_stats(program: ir.Program) -> tuple[list[set[int]], int]:
    groups: list[set[int]] = []
    current_bases: set[int] = set()
    barrier_count = 0

    class _Collector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):
            name = stmt.var.name_hint
            tile_type = stmt.var.type
            if "__chunk_keep_tmp_" in name and isinstance(tile_type, ir.TileType):
                if tile_type.memref is not None:
                    current_bases.add(tile_type.memref.base_.unique_id)
            super().visit_assign_stmt(stmt)

        def visit_call(self, call):
            nonlocal barrier_count, current_bases
            if call.op.name == "system.bar_all":
                barrier_count += 1
                groups.append(current_bases)
                current_bases = set()
            super().visit_call(call)

    _Collector().visit_program(program)
    return groups, barrier_count


def _static_shape(expr: ir.Expr) -> tuple[int, ...]:
    tile_type = expr.type
    assert isinstance(tile_type, ir.TileType)
    shape: list[int] = []
    for dim in tile_type.shape:
        assert isinstance(dim, ir.ConstInt)
        shape.append(dim.value)
    return tuple(shape)


def _static_tuple(expr: ir.Expr) -> tuple[int, ...]:
    assert isinstance(expr, ir.MakeTuple)
    values: list[int] = []
    for element in expr.elements:
        assert isinstance(element, ir.ConstInt)
        values.append(element.value)
    return tuple(values)


def _expanded_packing_ops(
    program: ir.Program,
) -> tuple[
    list[tuple[int, ...]],
    list[tuple[int, ...]],
    list[tuple[int, ...]],
    Counter[tuple[int, ...]],
    Counter[tuple[int, ...]],
    list[tuple[int, ...]],
]:
    source_load_shapes: list[tuple[int, ...]] = []
    result_load_shapes: list[tuple[int, ...]] = []
    quant_input_shapes: list[tuple[int, ...]] = []
    data_store_offsets: Counter[tuple[int, ...]] = Counter()
    scale_store_offsets: Counter[tuple[int, ...]] = Counter()
    transpose_output_shapes: list[tuple[int, ...]] = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, call):
            if call.op.name == "tile.load" and call.type.dtype == ir.DataType.FP32:
                source_load_shapes.append(_static_shape(call))
            elif call.op.name == "tile.load" and call.type.dtype in {
                ir.DataType.FP8E4M3FN,
                ir.DataType.FP8E8M0,
            }:
                result_load_shapes.append(_static_shape(call))
            elif call.op.name == "tile.tquant_mx":
                quant_input_shapes.append(_static_shape(call.args[0]))
            elif call.op.name == "tile.store":
                value_type = call.args[0].type
                if isinstance(value_type, ir.TileType):
                    offsets = _static_tuple(call.args[1])
                    if value_type.dtype == ir.DataType.FP8E4M3FN and _static_shape(call.args[0]) == (16, 64):
                        data_store_offsets[offsets] += 1
                    elif value_type.dtype == ir.DataType.FP8E8M0 and _static_shape(call.args[0]) == (1, 32):
                        scale_store_offsets[offsets] += 1
            elif call.op.name == "tile.transpose":
                transpose_output_shapes.append(_static_shape(call))
            super().visit_call(call)

    _Collector().visit_program(program)
    return (
        source_load_shapes,
        result_load_shapes,
        quant_input_shapes,
        data_store_offsets,
        scale_store_offsets,
        transpose_output_shapes,
    )


class TestExpandMxPackedQuant:
    """Packed MX expansion bounds async lifetimes with explicit pipe drains."""

    def test_store_fusion_does_not_cross_destination_read(self):
        """A fused GM write must not become visible before an earlier read."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                q_out: pl.InOut[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
                s_out: pl.Out[pl.Tensor[[1, 32], pl.FP8E8M0]],
                snapshot_out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
            ) -> tuple[
                pl.Tensor[[16, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 32], pl.FP8E8M0],
                pl.Tensor[[16, 64], pl.FP8E4M3FN],
            ]:
                quant, scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), layout=pl.MX_A_ZZ)
                old_q = pl.load(q_out, [0, 0], [16, 64])
                snapshot_out = pl.store(old_q, [0, 0], snapshot_out)
                q_out = pl.store(quant, [0, 0], q_out)
                s_out = pl.store(scale, [0, 0], s_out)
                return q_out, s_out, snapshot_out

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        kernel = expanded.get_function("kernel")
        assert kernel is not None
        q_out = next(param for param in kernel.params if param.name_hint.startswith("q_out"))
        events: list[str] = []

        class _AccessCollector(ir.IRVisitor):
            def visit_call(self, call):
                if (
                    call.op.name == "tile.load"
                    and isinstance(call.args[0], ir.Var)
                    and call.args[0].unique_id == q_out.unique_id
                ):
                    events.append("read")
                elif (
                    call.op.name == "tile.store"
                    and isinstance(call.args[2], ir.Var)
                    and call.args[2].unique_id == q_out.unique_id
                ):
                    events.append("write")
                super().visit_call(call)

        _AccessCollector().visit_program(expanded)

        assert events == ["read", "write"]
        passes.run_verifier()(expanded)

    def test_dynamic_source_offset_uses_assemble_fallback(self):
        """A dynamic aggregate load stays live for slice-based expansion."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[32, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                q_out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
                s_out: pl.Out[pl.Tensor[[1, 32], pl.FP8E8M0]],
            ) -> tuple[
                pl.Tensor[[16, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 32], pl.FP8E8M0],
            ]:
                loaded = pl.load(src, [row_offset, 0], [16, 64])
                quant, scale = pl.quant_mx(loaded, layout=pl.MX_A_ZZ)
                q_out = pl.store(quant, [0, 0], q_out)
                s_out = pl.store(scale, [0, 0], s_out)
                return q_out, s_out

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        op_counts: Counter[str] = Counter()

        class _OpCollector(ir.IRVisitor):
            def visit_call(self, call):
                op_counts[call.op.name] += 1
                super().visit_call(call)

        _OpCollector().visit_program(expanded)

        assert op_counts["tile.slice"] == 1
        assert op_counts["tile.store"] == 2
        passes.run_verifier()(expanded)
        assert "__FREE_VAR" not in ir.python_print(expanded)

    def test_store_fusion_does_not_cross_a_future_destination_update(self):
        """A destination SSA value defined after quantization cannot be hoisted."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                prior: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                q_out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
                s_out: pl.Out[pl.Tensor[[1, 32], pl.FP8E8M0]],
            ) -> tuple[
                pl.Tensor[[16, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 32], pl.FP8E8M0],
            ]:
                quant, scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), layout=pl.MX_A_ZZ)
                prior_tile = pl.load(prior, [0, 0], [16, 64])
                q_out = pl.store(prior_tile, [0, 0], q_out)
                q_out = pl.store(quant, [0, 0], q_out)
                s_out = pl.store(scale, [0, 0], s_out)
                return q_out, s_out

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        passes.run_verifier()(expanded)
        assert "__FREE_VAR" not in ir.python_print(expanded)

    def test_store_fusion_does_not_escape_control_flow(self):
        """Stores guarded by an if must remain conditional after expansion."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                cond: pl.Scalar[pl.BOOL],
                q_out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
                s_out: pl.Out[pl.Tensor[[1, 32], pl.FP8E8M0]],
            ) -> tuple[
                pl.Tensor[[16, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 32], pl.FP8E8M0],
            ]:
                quant, scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), layout=pl.MX_A_ZZ)
                if cond:
                    q_out = pl.store(quant, [0, 0], q_out)
                    s_out = pl.store(scale, [0, 0], s_out)
                return q_out, s_out

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        store_depths: list[int] = []
        if_depth = 0

        class _ControlFlowCollector(ir.IRVisitor):
            def visit_if_stmt(self, stmt):
                nonlocal if_depth
                if_depth += 1
                super().visit_if_stmt(stmt)
                if_depth -= 1

            def visit_call(self, call):
                if call.op.name == "tile.store":
                    store_depths.append(if_depth)
                super().visit_call(call)

        _ControlFlowCollector().visit_program(expanded)

        assert store_depths == [1, 1]
        passes.run_verifier()(expanded)

    def test_store_only_aliases_survive_assemble_fallback(self):
        """Fallback stores must retain the tuple-projection definitions."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
                s_out: pl.Out[pl.Tensor[[1, 32], pl.FP8E8M0]],
            ) -> tuple[
                pl.Tensor[[16, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 32], pl.FP8E8M0],
            ]:
                transformed = pl.abs(pl.load(src, [0, 0], [16, 64]))
                quant, scale = pl.quant_mx(transformed, layout=pl.MX_A_ZZ)
                q_out = pl.store(quant, [0, 0], q_out)
                s_out = pl.store(scale, [0, 0], s_out)
                return q_out, s_out

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        assemble_fractals: list[tuple[int, int]] = []

        class _AssembleViewCollector(ir.IRVisitor):
            def visit_call(self, call):
                if call.op.name == "tile.assemble":
                    target_type = call.args[0].type
                    source_type = call.args[1].type
                    assert isinstance(target_type, ir.TileType)
                    assert isinstance(source_type, ir.TileType)
                    target_fractal = (
                        target_type.tile_view.fractal
                        if target_type.tile_view is not None
                        else ir.TileView().fractal
                    )
                    source_fractal = (
                        source_type.tile_view.fractal
                        if source_type.tile_view is not None
                        else ir.TileView().fractal
                    )
                    assemble_fractals.append((target_fractal, source_fractal))
                super().visit_call(call)

        _AssembleViewCollector().visit_program(expanded)

        passes.run_verifier()(expanded)
        assert "__FREE_VAR" not in ir.python_print(expanded)
        assert assemble_fractals == [(512, 512), (32, 32)]

    def test_b_packing_does_not_reuse_user_tensor_as_scratch(self):
        """A shape-compatible InOut tensor is user data, not compiler scratch."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                unrelated: pl.InOut[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
                q_out: pl.Out[pl.Tensor[[64, 16], pl.FP8E4M3FN]],
                s_out: pl.Out[pl.Tensor[[1, 32], pl.FP8E8M0]],
            ) -> tuple[
                pl.Tensor[[64, 16], pl.FP8E4M3FN],
                pl.Tensor[[1, 32], pl.FP8E8M0],
                pl.Tensor[[16, 64], pl.FP8E4M3FN],
            ]:
                quant, scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), layout=pl.MX_B_NN)
                q_out = pl.store(quant, [0, 0], q_out)
                s_out = pl.store(scale, [0, 0], s_out)
                return q_out, s_out, unrelated

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        kernel = expanded.get_function("kernel")
        assert kernel is not None
        unrelated = next(param for param in kernel.params if param.name_hint.startswith("unrelated"))
        store_op_name = ir.get_op("tile.store").name
        store_target_ids: set[int] = set()

        class _StoreTargetCollector(ir.IRVisitor):
            def visit_call(self, call):
                if call.op.name == store_op_name and isinstance(call.args[2], ir.Var):
                    store_target_ids.add(call.args[2].unique_id)
                super().visit_call(call)

        _StoreTargetCollector().visit_program(expanded)

        assert unrelated.unique_id not in store_target_ids
        passes.run_verifier()(expanded)

    def test_isolated_large_k_uses_full_pack_box_order(self):
        """Isolated large-K packed quant keeps host full-pack order (mb outer, kb inner).

        For M=32,K=128 boxes are (mb,kb)=(0,0),(0,1),(1,0),(1,1) — not chunk-cat
        (0,0),(1,0),(0,1),(1,1). Scale assemble offsets are box_id*32.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                a_q_out: pl.Out[pl.Tensor[[32, 128], pl.FP8E4M3FN]],
                a_s_out: pl.Out[pl.Tensor[[1, 128], pl.FP8E8M0]],
            ) -> tuple[pl.Tensor[[32, 128], pl.FP8E4M3FN], pl.Tensor[[1, 128], pl.FP8E8M0]]:
                a_q, a_s = pl.quant_mx(pl.load(a, [0, 0], [32, 128]), layout=pl.MX_A_ZZ)
                a_q_out = pl.store(a_q, [0, 0], a_q_out)
                a_s_out = pl.store(a_s, [0, 0], a_s_out)
                return a_q_out, a_s_out

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        q_offsets: list[tuple[int, int]] = []
        s_offsets: list[tuple[int, int]] = []

        class _Collect(ir.IRVisitor):
            def visit_call(self, call):
                if call.op.name == "tile.assemble" and len(call.args) >= 3:
                    src = call.args[1]
                    if isinstance(src.type, ir.TileType) and src.type.dtype == ir.DataType.UINT8:
                        shape = _static_shape(src)
                        off = _static_tuple(call.args[2])
                        if shape == (16, 64):
                            q_offsets.append(off)
                        elif shape == (1, 32):
                            s_offsets.append(off)
                super().visit_call(call)

        _Collect().visit_program(expanded)
        # Full ZZ: quant at (mb*16, kb*64); scale at (0, (mb*kb_count+kb)*32).
        assert q_offsets == [(0, 0), (0, 64), (16, 0), (16, 64)]
        assert s_offsets == [(0, 0), (0, 32), (0, 64), (0, 96)]
        passes.run_verifier()(expanded)

    def test_cosplits_packed_quant_with_matmul_mx_k128(self):
        """Large-K packed quant + matmul co-splits into K=64 quant/matmul_acc."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                b: pl.Tensor[[128, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[4, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                aq, a_s = pl.quant_mx(pl.load(a, [0, 0], [16, 128]), layout=pl.MX_A_ZZ)
                tb = pl.load(b, [0, 0], [128, 32], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [4, 32], target_memory=pl.Mem.Mat)
                c = pl.matmul_mx(aq, a_s, tb, tbs)
                return pl.store(c, [0, 0], out)

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                b: pl.Tensor[[128, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[4, 32], pl.FP8E8M0, pl.MX_B_NN],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                out = pl.create_tensor([16, 32], dtype=pl.FP32)
                return self.kernel(a, b, b_s, out)

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")

        mx = []
        mx_acc = []
        packed_layout_quants = []

        class _Collect(ir.IRVisitor):
            def visit_call(self, call):
                if call.op.name == ir.get_op("tile.matmul_mx").name:
                    mx.append(call)
                elif call.op.name == ir.get_op("tile.matmul_mx_acc").name:
                    mx_acc.append(call)
                elif call.op.name == "tile.tquant_mx":
                    kwargs = dict(call.kwargs) if call.kwargs is not None else {}
                    if "layout" in kwargs:
                        packed_layout_quants.append(call)
                super().visit_call(call)

        _Collect().visit_program(expanded)
        assert len(mx) == 1
        assert len(mx_acc) == 1
        lhs = mx[0].args[0]
        assert isinstance(lhs.type.shape[1], ir.ConstInt)
        assert lhs.type.shape[1].value == 64
        lhs_acc = mx_acc[0].args[1]
        assert isinstance(lhs_acc.type.shape[1], ir.ConstInt)
        assert lhs_acc.type.shape[1].value == 64
        # Co-split + Expand: no packed-layout tquant_mx remains.
        assert packed_layout_quants == []
        passes.run_verifier()(expanded)

    def test_inserts_reshape_for_flat_scale_matmul_k64(self):
        """K=64 packed quant + matmul with flat [1,G] scales: pass inserts reshape."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                b_nk: pl.Tensor[[32, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                aq, a_s = pl.quant_mx(pl.load(a, [0, 0], [16, 64]), layout=pl.MX_A_ZZ)
                bq, b_s = pl.quant_mx(pl.load(b_nk, [0, 0], [32, 64]), layout=pl.MX_B_NN)
                c = pl.matmul_mx(aq, a_s, bq, b_s)
                return pl.store(c, [0, 0], out)

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP32],
                b_nk: pl.Tensor[[32, 64], pl.FP32],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                out = pl.create_tensor([16, 32], dtype=pl.FP32)
                return self.kernel(a, b_nk, out)

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")

        logical_scale_reshapes: list[tuple[int, int]] = []
        mx_calls: list = []

        class _Collect(ir.IRVisitor):
            def visit_call(self, call):
                if call.op.name == ir.get_op("tile.matmul_mx").name:
                    mx_calls.append(call)
                elif call.op.name == ir.get_op("tile.reshape").name and len(call.args) >= 2:
                    shape = _static_tuple(call.args[1])
                    if shape in ((16, 2), (2, 32)):
                        logical_scale_reshapes.append(shape)
                super().visit_call(call)

        _Collect().visit_program(expanded)
        assert len(mx_calls) == 1
        assert (16, 2) in logical_scale_reshapes
        assert (2, 32) in logical_scale_reshapes
        lhs_s = mx_calls[0].args[1]
        rhs_s = mx_calls[0].args[3]
        assert _static_shape(lhs_s) == (16, 2)
        assert _static_shape(rhs_s) == (2, 32)
        passes.run_verifier()(expanded)

    def test_assemble_packing_and_pipe_all_drains(self):
        """Assemble path: per-box load/quant + Vec assemble; drains bound reuse.

        Shapes use K=64 (kb==1). Expand only emits assemble (no store-fusion).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[32, 64], pl.FP32],
                b_nk: pl.Tensor[[64, 64], pl.FP32],
                a_q_out: pl.Out[pl.Tensor[[32, 64], pl.FP8E4M3FN]],
                a_s_out: pl.Out[pl.Tensor[[1, 64], pl.FP8E8M0]],
                b_q_out: pl.Out[pl.Tensor[[64, 64], pl.FP8E4M3FN]],
                b_s_out: pl.Out[pl.Tensor[[1, 128], pl.FP8E8M0]],
            ) -> tuple[
                pl.Tensor[[32, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 64], pl.FP8E8M0],
                pl.Tensor[[64, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 128], pl.FP8E8M0],
            ]:
                a_q, a_s = pl.quant_mx(pl.load(a, [0, 0], [32, 64]), layout=pl.MX_A_ZZ)
                b_q, b_s = pl.quant_mx(pl.load(b_nk, [0, 0], [64, 64]), layout=pl.MX_B_NN)
                a_q_out = pl.store(a_q, [0, 0], a_q_out)
                a_s_out = pl.store(a_s, [0, 0], a_s_out)
                b_q_out = pl.store(b_q, [0, 0], b_q_out)
                b_s_out = pl.store(b_s, [0, 0], b_s_out)
                return a_q_out, a_s_out, b_q_out, b_s_out

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        (
            source_load_shapes,
            result_load_shapes,
            quant_shapes,
            data_offsets,
            scale_offsets,
            transpose_shapes,
        ) = _expanded_packing_ops(expanded)

        op_counts: Counter[str] = Counter()

        class _OpCollector(ir.IRVisitor):
            def visit_call(self, call):
                op_counts[call.op.name] += 1
                super().visit_call(call)

        _OpCollector().visit_program(expanded)

        # Aggregate FP32 loads remain; per-box reloads are [16,64].
        box_loads = [s for s in source_load_shapes if s == (16, 64)]
        assert box_loads == [(16, 64)] * 6  # A:2 + B:4
        assert result_load_shapes == []
        assert quant_shapes == [(32, 32)] * 6
        # No per-box GM stores; user stores whole results after assemble.
        assert data_offsets == Counter()
        assert scale_offsets == Counter()
        assert transpose_shapes == [(64, 64)]
        # A: 2q+2s assemble; B: 4q+4s assemble.
        assert op_counts["tile.assemble"] == 12

        _, expanded_barriers = _chunk_lifetime_stats(expanded)
        # Drain after A's boxes, after B's boxes, and after B transpose.
        assert expanded_barriers == 3

        after = _run_default_through(Before, "MemoryReuse")
        chunk_groups, barrier_count = _chunk_lifetime_stats(after)

        # Per-box chunk keeps load/quant/scale (3) alive across each drain.
        assert [len(group) for group in chunk_groups[:2]] == [2 * 3, 4 * 3]
        assert chunk_groups[0] & chunk_groups[1]
        assert barrier_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
