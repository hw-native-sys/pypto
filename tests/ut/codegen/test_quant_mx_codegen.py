# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen tests for MX quantization operators."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.runtime import RunConfig

_TQUANT_MX_RAW = ir.get_op("tile.tquant_mx_raw").name
_TMOV_X2ZZ = ir.get_op("tile.tmov_x2zz").name


def _optimize(program):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _generate_mlir(program):
    optimized = _optimize(program)
    target = next(
        function for function in optimized.functions.values() if ir.is_incore_type(function.func_type)
    )
    return codegen.PTOCodegen().generate(ir.Program([target], target.name, optimized.span))


def _allocation_range(mlir, name_fragment):
    line = next(line for line in mlir.splitlines() if name_fragment in line and "pto.alloc_tile" in line)
    match = re.search(r"addr = %c(\d+)_i64.*dtype=([^,]+), rows=(\d+), cols=(\d+)", line)
    assert match is not None, line
    address, dtype, rows, cols = match.groups()
    element_bytes = {"i8": 1, "ui8": 1, "f16": 2, "bf16": 2, "f32": 4}[dtype]
    return int(address), int(address) + int(rows) * int(cols) * element_bytes


class TestQuantMxCodegen:
    @pytest.mark.parametrize(("case", "slot_num"), [("direct", 4), ("none", 4), ("split", 4), ("none", 2)])
    def test_public_mx_gather_compiles(self, case, slot_num, tmp_path):
        """Both public MX scale orientations survive an explicit AIV/AIC crossing."""

        @pl.jit
        def direct(
            query: pl.Tensor[[64, 128], pl.BF16],
            rhs: pl.Tensor[[64, 128], pl.BF16],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            with pl.spmd(1, optimizations=[pl.cross_core_slot(slot_num=slot_num)]):
                _block = pl.tile.get_block_idx()
                q_mx, q_scale = pl.quant_mx(pl.load(query, [0, 0], [64, 128]), group_axis=1)
                b_mx, b_scale = pl.quant_mx(pl.load(rhs, [0, 0], [64, 128]), group_axis=0)
                output = pl.store(pl.matmul_mx(q_mx, q_scale, b_mx, b_scale), [0, 0], output)
            return output

        @pl.jit
        def none_gather(
            query: pl.Tensor[[64, 128], pl.BF16],
            rhs: pl.Tensor[[64, 128], pl.BF16],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            with pl.spmd(1, optimizations=[pl.cross_core_slot(slot_num=slot_num)]):
                _block = pl.tile.get_block_idx()
                for _lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    q_mx_lane, q_scale_lane = pl.quant_mx(pl.load(query, [0, 0], [64, 128]), group_axis=1)
                    b_mx_lane, b_scale_lane = pl.quant_mx(pl.load(rhs, [0, 0], [64, 128]), group_axis=0)
                    q_mx = pl.aic_gather(q_mx_lane)
                    q_scale = pl.aic_gather(q_scale_lane)
                    b_mx = pl.aic_gather(b_mx_lane)
                    b_scale = pl.aic_gather(b_scale_lane)
                output = pl.store(pl.matmul_mx(q_mx, q_scale, b_mx, b_scale), [0, 0], output)
            return output

        @pl.jit
        def split_gather(
            query: pl.Tensor[[64, 128], pl.BF16],
            rhs: pl.Tensor[[128, 64], pl.FP8E4M3FN],
            rhs_scale: pl.Tensor[[4, 64], pl.FP8E8M0, pl.MX_B_NN],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            with pl.spmd(1, optimizations=[pl.cross_core_slot(slot_num=slot_num)]):
                _block = pl.tile.get_block_idx()
                b_mx = pl.load(rhs, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
                b_scale = pl.load(rhs_scale, [0, 0], [4, 64], target_memory=pl.Mem.Mat)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    q_mx_lane, q_scale_lane = pl.quant_mx(
                        pl.load(query, [lane * 32, 0], [32, 128]), group_axis=1
                    )
                    q_mx = pl.aic_gather(q_mx_lane)
                    q_scale = pl.aic_gather(q_scale_lane)
                output = pl.store(pl.matmul_mx(q_mx, q_scale, b_mx, b_scale), [0, 0], output)
            return output

        kernel = {
            "direct": direct,
            "none": none_gather,
            "split": split_gather,
        }[case]
        config = RunConfig(platform="a5", save_kernels_dir=str(tmp_path))
        if slot_num == 2:
            with pytest.raises(ValueError, match="slot_num=2 is insufficient for 4"):
                kernel.compile(config=config)
            return
        compiled = kernel.compile(config=config)
        mlir = "\n".join(path.read_text() for path in Path(compiled.output_dir).rglob("*.pto"))
        assert "pto.tmatmul.mx" in mlir
        scale_pops = [line for line in mlir.splitlines() if "pto.tpop" in line and "f8E8M0" in line]
        assert len(scale_pops) == (1 if case == "split" else 2)
        assert all("fractal=32" in line for line in scale_pops)
        assert any("blayout=row_major, slayout=row_major" in line for line in scale_pops)
        if case != "split":
            assert any("blayout=col_major, slayout=col_major" in line for line in scale_pops)

    @pytest.mark.parametrize(
        ("group_axis", "src_shape", "out_shape"),
        [
            (1, (16, 64), (16, 64)),
            (0, (32, 64), (64, 32)),
        ],
    )
    def test_packed_quant_codegen(self, group_axis, src_shape, out_shape):
        src_rows, src_cols = src_shape
        out_rows, out_cols = out_shape

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[src_rows, src_cols], pl.BF16],
                out: pl.Tensor[[out_rows, out_cols], pl.FP8E4M3FN],
            ):
                quant, _scale = pl.quant_mx(
                    pl.load(src, [0, 0], [src_rows, src_cols]),
                    group_axis=group_axis,
                )
                pl.store(quant, [0, 0], out)

        mlir = _generate_mlir(Program)
        tquant = next(line for line in mlir.splitlines() if "pto.tquant.mx" in line)
        x2zz = next(line for line in mlir.splitlines() if "pto.tmov" in line and "x2zz_tmp_static" in line)
        assert tquant.split("outs(", 1)[1].split(" : ", 1)[0].count(",") == 3
        assert "quant_type MXFP8" in tquant
        assert f"grpAxis = #pto<mx_group_axis axis{group_axis}>" in tquant
        assert f"grpAxis = #pto<mx_group_axis axis{group_axis}>" in x2zz
        assert ("tq_src_kn" in mlir) == (group_axis == 0)
        assert "tquant_dst_static" in mlir

        zz_alloc = next(
            line for line in mlir.splitlines() if "tq_exp_zz" in line and "pto.alloc_tile" in line
        )
        if group_axis == 1:
            # Axis1 ZZ keeps [M, G] (with align16 row padding on the physical rows).
            assert f"cols={src_cols // 32}" in zz_alloc
            assert "tq_scale_nn" not in mlir
        else:
            # Axis0 DN [M̂,N] -> ZZ [N,M̂]; public MX_B is the zero-copy transpose_view.
            # Packed-B lowers after [N,K]->[K,N], so M̂=K/32=src_cols/32 and N=src_rows.
            zz_rows = src_rows
            zz_cols = src_cols // 32
            assert f"rows={zz_rows}" in zz_alloc and f"cols={zz_cols}" in zz_alloc
            assert "blayout=row_major, slayout=row_major" in zz_alloc
            assert "tq_scale_nn" in mlir
            assert "transpose_view" in mlir or "treshape" in mlir
            # Must not emit a Vec layout-conversion tmov on the scale path.
            scale_tmovs = [
                line
                for line in mlir.splitlines()
                if "pto.tmov" in line and "x2zz" not in line and ("tq_scale" in line or "scale_nn" in line)
            ]
            assert scale_tmovs == []

        ranges = [_allocation_range(mlir, name) for name in ("tq_max", "tq_scaling", "tq_dst", "tq_exp")]
        for index, lhs in enumerate(ranges):
            assert all(lhs[1] <= rhs[0] or rhs[1] <= lhs[0] for rhs in ranges[index + 1 :])

    def test_raw_is_value_returning_after_optimization(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[16, 64], pl.FP16]):
                _quant, _scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), group_axis=1)

        class Collector(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.eval_count = 0
                self.assign_count = 0
                self.x2zz_assign = 0

            def visit_eval_stmt(self, op):
                self.eval_count += isinstance(op.expr, ir.Call) and op.expr.op.name in (
                    _TQUANT_MX_RAW,
                    _TMOV_X2ZZ,
                )
                super().visit_eval_stmt(op)

            def visit_assign_stmt(self, op):
                if isinstance(op.value, ir.Call):
                    if op.value.op.name == _TQUANT_MX_RAW:
                        self.assign_count += 1
                    elif op.value.op.name == _TMOV_X2ZZ:
                        self.x2zz_assign += 1
                super().visit_assign_stmt(op)

        collector = Collector()
        collector.visit_program(_optimize(Program))
        assert (collector.eval_count, collector.assign_count, collector.x2zz_assign) == (0, 1, 1)

    def test_tuple_alias_chain_reaches_codegen(self):
        """Ordinary aliases of the composite tuple must preserve projection lowering."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
            ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
                pair = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), group_axis=1)
                alias = pair
                second_alias = alias
                quant = second_alias[0]
                return pl.store(quant, [0, 0], out)

        mlir = _generate_mlir(Program)
        assert "pto.tquant.mx" in mlir

    def test_not_registered_on_ascend910b(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, src: pl.Tensor[[16, 64], pl.FP16]):
                _quant, _scale = pl.quant_mx(pl.load(src, [0, 0], [16, 64]), group_axis=1)

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        target = next(
            function for function in optimized.functions.values() if ir.is_incore_type(function.func_type)
        )
        with pytest.raises(ValueError, match=r"tile\.tquant_mx_raw"):
            codegen.PTOCodegen().generate(ir.Program([target], target.name, optimized.span))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
