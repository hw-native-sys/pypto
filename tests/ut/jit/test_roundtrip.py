# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Round-trip tests: @pl.jit equivalents of example programs.

For each program pattern, a hand-written @pl.program serves as ground truth.
Both the JIT output and the ground truth are run through PassManager before
comparison, so the test verifies that JIT produces the same compiled IR as the
equivalent @pl.program written by hand.

Coverage
--------
01_elementwise.py  -- TileAdd 128x128, TileMul 128x128,
                       TileAdd 64x64, TileMul 64x64
02_fused_ops.py    -- FusedAddScale, FusedAddRelu,
                       FusedMatmulBias, FusedLinearRelu
03_matmul.py       -- Matmul, MatmulAcc
05_activation.py   -- SiLU, GELU, SwiGLU, GeGLU
06_softmax.py      -- TileSoftmax
07_normalization.py -- RMSNorm, LayerNorm
08_assemble.py     -- TileAssembleAccMat, TileAssembleVec,
                       TileAssembleRowByRow, TileAssembleDoubleLoop,
                       TileAssembleLoopColBroadcast, TileAssembleDoubleLoopBroadcast

Intentionally excluded (require features outside @pl.jit scope)
---------------------------------------------------------------
04_concat.py       -- Orchestration has no Out param; output is created with
                       pl.create_tensor inside the orchestrator. @pl.jit cannot
                       infer the return type in this pattern.
09_dyn_valid_shape.py -- Uses module-level @pl.function (not @pl.jit.incore)
                       and pl.tensor.read for scalar config tensors.
examples/models/   -- Use module-level @pl.function called directly (not via
                       @pl.jit.incore), which @pl.jit dep discovery does not cover.
"""

import pypto.language as pl
import pytest
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.decorator import jit
from pypto.pypto_core import ir

# ---------------------------------------------------------------------------
# 01_elementwise.py
# ---------------------------------------------------------------------------


class TestElementwise:
    def test_tile_add_128x128(self):
        """Multi-function JIT round-trip for 128x128 FP32 add."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileAddRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [128, 128])
                tile_b = pl.load(b, [0, 0], [128, 128])
                tile_c = pl.add(tile_a, tile_b)
                out_c = pl.store(tile_c, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_add(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAddRef)

        @jit.incore
        def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [128, 128])
            tile_b = pl.load(b, [0, 0], [128, 128])
            tile_c = pl.add(tile_a, tile_b)
            out_c = pl.store(tile_c, [0, 0], c)
            return out_c

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = tile_add(a, b, out_c)
            return out_c

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)
        got = orchestrator.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)

    def test_tile_mul_128x128(self):
        """Multi-function JIT round-trip for 128x128 FP32 mul."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileMulRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_mul(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [128, 128])
                tile_b = pl.load(b, [0, 0], [128, 128])
                tile_c = pl.mul(tile_a, tile_b)
                out_c = pl.store(tile_c, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.tile_mul(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileMulRef)

        @jit.incore
        def tile_mul(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [128, 128])
            tile_b = pl.load(b, [0, 0], [128, 128])
            tile_c = pl.mul(tile_a, tile_b)
            out_c = pl.store(tile_c, [0, 0], c)
            return out_c

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = tile_mul(a, b, out_c)
            return out_c

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)
        got = orchestrator.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)

    def test_tile_add_64x64(self):
        """Multi-function JIT round-trip for 64x64 FP32 add."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileAdd64Ref:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_add(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 64])
                tile_b = pl.load(b, [0, 0], [64, 64])
                tile_c = pl.add(tile_a, tile_b)
                out_c = pl.store(tile_c, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out_c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_add(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAdd64Ref)

        @jit.incore
        def tile_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [64, 64])
            tile_b = pl.load(b, [0, 0], [64, 64])
            tile_c = pl.add(tile_a, tile_b)
            out_c = pl.store(tile_c, [0, 0], c)
            return out_c

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = tile_add(a, b, out_c)
            return out_c

        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        c = torch.empty(64, 64)
        got = orchestrator.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)

    def test_tile_mul_64x64(self):
        """Multi-function JIT round-trip for 64x64 FP32 mul."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileMul64Ref:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_mul(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 64])
                tile_b = pl.load(b, [0, 0], [64, 64])
                tile_c = pl.mul(tile_a, tile_b)
                out_c = pl.store(tile_c, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out_c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.tile_mul(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileMul64Ref)

        @jit.incore
        def tile_mul(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [64, 64])
            tile_b = pl.load(b, [0, 0], [64, 64])
            tile_c = pl.mul(tile_a, tile_b)
            out_c = pl.store(tile_c, [0, 0], c)
            return out_c

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = tile_mul(a, b, out_c)
            return out_c

        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        c = torch.empty(64, 64)
        got = orchestrator.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)


# ---------------------------------------------------------------------------
# 02_fused_ops.py
# ---------------------------------------------------------------------------


class TestFusedOps:
    def test_fused_add_scale(self):
        """(a + b) * 2.0."""
        torch = pytest.importorskip("torch")

        @pl.program
        class FusedAddScaleRef:
            @pl.function(type=pl.FunctionType.InCore)
            def fused_add_scale(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [128, 128])
                tile_b = pl.load(b, [0, 0], [128, 128])
                tile_sum = pl.add(tile_a, tile_b)
                tile_c = pl.mul(tile_sum, 2.0)
                out_c = pl.store(tile_c, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.fused_add_scale(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(FusedAddScaleRef)

        @jit.incore
        def fused_add_scale(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [128, 128])
            tile_b = pl.load(b, [0, 0], [128, 128])
            tile_sum = pl.add(tile_a, tile_b)
            tile_c = pl.mul(tile_sum, 2.0)
            out_c = pl.store(tile_c, [0, 0], c)
            return out_c

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = fused_add_scale(a, b, out_c)
            return out_c

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)
        got = orchestrator.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)

    def test_fused_add_relu(self):
        """relu(a + b)."""
        torch = pytest.importorskip("torch")

        @pl.program
        class FusedAddReluRef:
            @pl.function(type=pl.FunctionType.InCore)
            def fused_add_relu(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [128, 128])
                tile_b = pl.load(b, [0, 0], [128, 128])
                tile_sum = pl.add(tile_a, tile_b)
                tile_c = pl.relu(tile_sum)
                out_c = pl.store(tile_c, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out_c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.fused_add_relu(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(FusedAddReluRef)

        @jit.incore
        def fused_add_relu(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [128, 128])
            tile_b = pl.load(b, [0, 0], [128, 128])
            tile_sum = pl.add(tile_a, tile_b)
            tile_c = pl.relu(tile_sum)
            out_c = pl.store(tile_c, [0, 0], c)
            return out_c

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = fused_add_relu(a, b, out_c)
            return out_c

        a = torch.randn(128, 128)
        b = torch.randn(128, 128)
        c = torch.empty(128, 128)
        got = orchestrator.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)

    def test_fused_matmul_bias(self):
        """c = matmul(a, b) + bias."""
        torch = pytest.importorskip("torch")

        @pl.program
        class FusedMatmulBiasRef:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out = pl.store(tile_c_l0c, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def add_bias_kernel(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [64, 64])
                tile_bias = pl.load(bias, [0, 0], [64, 64])
                tile_c = pl.add(tile_x, tile_bias)
                out = pl.store(tile_c, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mm_out = pl.create_tensor([64, 64], dtype=pl.FP32)
                mm_out = self.matmul_kernel(a, b, mm_out)
                c = self.add_bias_kernel(mm_out, bias, c)
                return c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(FusedMatmulBiasRef)

        @jit.incore
        def matmul_kernel(a: pl.Tensor, b: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
            out = pl.store(tile_c_l0c, [0, 0], output)
            return out

        @jit.incore
        def add_bias_kernel(x: pl.Tensor, bias: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [64, 64])
            tile_bias = pl.load(bias, [0, 0], [64, 64])
            tile_c = pl.add(tile_x, tile_bias)
            out = pl.store(tile_c, [0, 0], output)
            return out

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, bias: pl.Tensor, c: pl.Out[pl.Tensor]):
            mm_out = pl.create_tensor([64, 64], dtype=pl.FP32)
            mm_out = matmul_kernel(a, b, mm_out)
            c = add_bias_kernel(mm_out, bias, c)
            return c

        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        bias = torch.randn(64, 64)
        c = torch.empty(64, 64)
        got = orchestrator.compile_for_test(a, b, bias, c)
        ir.assert_structural_equal(got, expected)

    def test_fused_linear_relu(self):
        """y = relu(matmul(x, w) + bias)."""
        torch = pytest.importorskip("torch")

        @pl.program
        class FusedLinearReluRef:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_kernel(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                w: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_x_l1 = pl.load(x, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_w_l1 = pl.load(w, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_x_l0a = pl.move(tile_x_l1, target_memory=pl.MemorySpace.Left)
                tile_w_l0b = pl.move(tile_w_l1, target_memory=pl.MemorySpace.Right)
                tile_out_l0c = pl.matmul(tile_x_l0a, tile_w_l0b)
                out = pl.store(tile_out_l0c, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def add_bias_relu_kernel(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [64, 64])
                tile_bias = pl.load(bias, [0, 0], [64, 64])
                tile_biased = pl.add(tile_x, tile_bias)
                tile_y = pl.relu(tile_biased)
                out = pl.store(tile_y, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                w: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                y: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mm_out = pl.create_tensor([64, 64], dtype=pl.FP32)
                mm_out = self.matmul_kernel(x, w, mm_out)
                y = self.add_bias_relu_kernel(mm_out, bias, y)
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(FusedLinearReluRef)

        @jit.incore
        def matmul_kernel(x: pl.Tensor, w: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_x_l1 = pl.load(x, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tile_w_l1 = pl.load(w, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tile_x_l0a = pl.move(tile_x_l1, target_memory=pl.MemorySpace.Left)
            tile_w_l0b = pl.move(tile_w_l1, target_memory=pl.MemorySpace.Right)
            tile_out_l0c = pl.matmul(tile_x_l0a, tile_w_l0b)
            out = pl.store(tile_out_l0c, [0, 0], output)
            return out

        @jit.incore
        def add_bias_relu_kernel(x: pl.Tensor, bias: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [64, 64])
            tile_bias = pl.load(bias, [0, 0], [64, 64])
            tile_biased = pl.add(tile_x, tile_bias)
            tile_y = pl.relu(tile_biased)
            out = pl.store(tile_y, [0, 0], output)
            return out

        @jit
        def orchestrator(x: pl.Tensor, w: pl.Tensor, bias: pl.Tensor, y: pl.Out[pl.Tensor]):
            mm_out = pl.create_tensor([64, 64], dtype=pl.FP32)
            mm_out = matmul_kernel(x, w, mm_out)
            y = add_bias_relu_kernel(mm_out, bias, y)
            return y

        x = torch.randn(64, 64)
        w = torch.randn(64, 64)
        bias = torch.randn(64, 64)
        y = torch.empty(64, 64)
        got = orchestrator.compile_for_test(x, w, bias, y)
        ir.assert_structural_equal(got, expected)


# ---------------------------------------------------------------------------
# 03_matmul.py
# ---------------------------------------------------------------------------


class TestMatmul:
    def test_matmul_program(self):
        """64x64 matmul."""
        torch = pytest.importorskip("torch")

        @pl.program
        class MatmulRef:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out_c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.matmul(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(MatmulRef)

        @jit.incore
        def matmul(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
            out_c = pl.store(tile_c_l0c, [0, 0], c)
            return out_c

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = matmul(a, b, out_c)
            return out_c

        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        c = torch.empty(64, 64)
        got = orchestrator.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)

    def test_matmulacc_program(self):
        """K=64 split into two K=32 chunks."""
        torch = pytest.importorskip("torch")

        @pl.program
        class MatmulaccRef:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_acc(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a0_l1 = pl.load(a, [0, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
                tile_b0_l1 = pl.load(b, [0, 0], [32, 64], target_memory=pl.MemorySpace.Mat)
                tile_a0_l0a = pl.move(tile_a0_l1, target_memory=pl.MemorySpace.Left)
                tile_b0_l0b = pl.move(tile_b0_l1, target_memory=pl.MemorySpace.Right)
                acc: pl.Tile[[64, 64], pl.FP32] = pl.matmul(tile_a0_l0a, tile_b0_l0b)
                tile_a1_l1 = pl.load(a, [0, 32], [64, 32], target_memory=pl.MemorySpace.Mat)
                tile_b1_l1 = pl.load(b, [32, 0], [32, 64], target_memory=pl.MemorySpace.Mat)
                tile_a1_l0a = pl.move(tile_a1_l1, target_memory=pl.MemorySpace.Left)
                tile_b1_l0b = pl.move(tile_b1_l1, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul_acc(acc, tile_a1_l0a, tile_b1_l0b)
                out_c = pl.store(acc, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out_c: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.matmul_acc(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(MatmulaccRef)

        @jit.incore
        def matmul_acc(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            tile_a0_l1 = pl.load(a, [0, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
            tile_b0_l1 = pl.load(b, [0, 0], [32, 64], target_memory=pl.MemorySpace.Mat)
            tile_a0_l0a = pl.move(tile_a0_l1, target_memory=pl.MemorySpace.Left)
            tile_b0_l0b = pl.move(tile_b0_l1, target_memory=pl.MemorySpace.Right)
            acc: pl.Tile[[64, 64], pl.FP32] = pl.matmul(tile_a0_l0a, tile_b0_l0b)
            tile_a1_l1 = pl.load(a, [0, 32], [64, 32], target_memory=pl.MemorySpace.Mat)
            tile_b1_l1 = pl.load(b, [32, 0], [32, 64], target_memory=pl.MemorySpace.Mat)
            tile_a1_l0a = pl.move(tile_a1_l1, target_memory=pl.MemorySpace.Left)
            tile_b1_l0b = pl.move(tile_b1_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul_acc(acc, tile_a1_l0a, tile_b1_l0b)
            out_c = pl.store(acc, [0, 0], c)
            return out_c

        @jit
        def orchestrator(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = matmul_acc(a, b, out_c)
            return out_c

        a = torch.randn(64, 64)
        b = torch.randn(64, 64)
        c = torch.empty(64, 64)
        got = orchestrator.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)


# ---------------------------------------------------------------------------
# 05_activation.py
# ---------------------------------------------------------------------------


class TestActivation:
    def test_silu(self):
        """SiLU activation."""
        torch = pytest.importorskip("torch")

        @pl.program
        class SiluRef:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_silu(
                self,
                x: pl.Tensor[[32, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 128])
                x_neg = pl.mul(tile_x, -1.0)
                exp_neg = pl.exp(x_neg)
                denom = pl.add(exp_neg, 1.0)
                sigmoid = pl.recip(denom)
                result = pl.mul(tile_x, sigmoid)
                out = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def silu_orch(
                self,
                x: pl.Tensor[[32, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                output = self.kernel_silu(x, output)
                return output

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(SiluRef)

        @jit.incore
        def kernel_silu(x: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 128])
            x_neg = pl.mul(tile_x, -1.0)
            exp_neg = pl.exp(x_neg)
            denom = pl.add(exp_neg, 1.0)
            sigmoid = pl.recip(denom)
            result = pl.mul(tile_x, sigmoid)
            out = pl.store(result, [0, 0], output)
            return out

        @jit
        def silu_orch(x: pl.Tensor, output: pl.Out[pl.Tensor]):
            output = kernel_silu(x, output)
            return output

        x = torch.randn(32, 128)
        out = torch.empty(32, 128)
        got = silu_orch.compile_for_test(x, out)
        ir.assert_structural_equal(got, expected)

    def test_gelu(self):
        """GELU activation."""
        torch = pytest.importorskip("torch")

        @pl.program
        class GeluRef:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_gelu(
                self,
                x: pl.Tensor[[32, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 128])
                x_scaled = pl.mul(tile_x, 1.702)
                x_neg = pl.mul(x_scaled, -1.0)
                exp_neg = pl.exp(x_neg)
                denom = pl.add(exp_neg, 1.0)
                sigmoid = pl.recip(denom)
                result = pl.mul(tile_x, sigmoid)
                out = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def gelu_orch(
                self,
                x: pl.Tensor[[32, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                output = self.kernel_gelu(x, output)
                return output

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(GeluRef)

        @jit.incore
        def kernel_gelu(x: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 128])
            x_scaled = pl.mul(tile_x, 1.702)
            x_neg = pl.mul(x_scaled, -1.0)
            exp_neg = pl.exp(x_neg)
            denom = pl.add(exp_neg, 1.0)
            sigmoid = pl.recip(denom)
            result = pl.mul(tile_x, sigmoid)
            out = pl.store(result, [0, 0], output)
            return out

        @jit
        def gelu_orch(x: pl.Tensor, output: pl.Out[pl.Tensor]):
            output = kernel_gelu(x, output)
            return output

        x = torch.randn(32, 128)
        out = torch.empty(32, 128)
        got = gelu_orch.compile_for_test(x, out)
        ir.assert_structural_equal(got, expected)

    def test_swiglu(self):
        """SwiGLU activation."""
        torch = pytest.importorskip("torch")

        @pl.program
        class SwigluRef:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_swiglu(
                self,
                gate: pl.Tensor[[32, 128], pl.FP32],
                up: pl.Tensor[[32, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_gate = pl.load(gate, [0, 0], [32, 128])
                tile_up = pl.load(up, [0, 0], [32, 128])
                gate_neg = pl.mul(tile_gate, -1.0)
                exp_neg = pl.exp(gate_neg)
                denom = pl.add(exp_neg, 1.0)
                sigmoid = pl.recip(denom)
                swish = pl.mul(tile_gate, sigmoid)
                result = pl.mul(swish, tile_up)
                out = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def swiglu_orch(
                self,
                gate: pl.Tensor[[32, 128], pl.FP32],
                up: pl.Tensor[[32, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                output = self.kernel_swiglu(gate, up, output)
                return output

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(SwigluRef)

        @jit.incore
        def kernel_swiglu(gate: pl.Tensor, up: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_gate = pl.load(gate, [0, 0], [32, 128])
            tile_up = pl.load(up, [0, 0], [32, 128])
            gate_neg = pl.mul(tile_gate, -1.0)
            exp_neg = pl.exp(gate_neg)
            denom = pl.add(exp_neg, 1.0)
            sigmoid = pl.recip(denom)
            swish = pl.mul(tile_gate, sigmoid)
            result = pl.mul(swish, tile_up)
            out = pl.store(result, [0, 0], output)
            return out

        @jit
        def swiglu_orch(gate: pl.Tensor, up: pl.Tensor, output: pl.Out[pl.Tensor]):
            output = kernel_swiglu(gate, up, output)
            return output

        gate = torch.randn(32, 128)
        up = torch.randn(32, 128)
        out = torch.empty(32, 128)
        got = swiglu_orch.compile_for_test(gate, up, out)
        ir.assert_structural_equal(got, expected)

    def test_geglu(self):
        """GeGLU activation."""
        torch = pytest.importorskip("torch")

        @pl.program
        class GegluRef:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_geglu(
                self,
                gate: pl.Tensor[[32, 128], pl.FP32],
                up: pl.Tensor[[32, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                tile_gate = pl.load(gate, [0, 0], [32, 128])
                tile_up = pl.load(up, [0, 0], [32, 128])
                gate_scaled = pl.mul(tile_gate, 1.702)
                gate_neg = pl.mul(gate_scaled, -1.0)
                exp_neg = pl.exp(gate_neg)
                denom = pl.add(exp_neg, 1.0)
                sigmoid = pl.recip(denom)
                gelu_gate = pl.mul(tile_gate, sigmoid)
                result = pl.mul(gelu_gate, tile_up)
                out = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def geglu_orch(
                self,
                gate: pl.Tensor[[32, 128], pl.FP32],
                up: pl.Tensor[[32, 128], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 128], pl.FP32]],
            ) -> pl.Tensor[[32, 128], pl.FP32]:
                output = self.kernel_geglu(gate, up, output)
                return output

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(GegluRef)

        @jit.incore
        def kernel_geglu(gate: pl.Tensor, up: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_gate = pl.load(gate, [0, 0], [32, 128])
            tile_up = pl.load(up, [0, 0], [32, 128])
            gate_scaled = pl.mul(tile_gate, 1.702)
            gate_neg = pl.mul(gate_scaled, -1.0)
            exp_neg = pl.exp(gate_neg)
            denom = pl.add(exp_neg, 1.0)
            sigmoid = pl.recip(denom)
            gelu_gate = pl.mul(tile_gate, sigmoid)
            result = pl.mul(gelu_gate, tile_up)
            out = pl.store(result, [0, 0], output)
            return out

        @jit
        def geglu_orch(gate: pl.Tensor, up: pl.Tensor, output: pl.Out[pl.Tensor]):
            output = kernel_geglu(gate, up, output)
            return output

        gate = torch.randn(32, 128)
        up = torch.randn(32, 128)
        out = torch.empty(32, 128)
        got = geglu_orch.compile_for_test(gate, up, out)
        ir.assert_structural_equal(got, expected)


# ---------------------------------------------------------------------------
# 06_softmax.py
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_tile_softmax(self):
        """Row-wise softmax."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileSoftmaxRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_softmax(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 64])
                max_tmp = pl.create_tile([64, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                row_max: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)
                shifted = pl.row_expand_sub(tile_a, row_max)
                exp_shifted = pl.exp(shifted)
                sum_tmp = pl.create_tile([64, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                row_sum: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(exp_shifted, sum_tmp)
                result = pl.row_expand_div(exp_shifted, row_sum)
                out = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                output = self.tile_softmax(a, output)
                return output

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileSoftmaxRef)

        @jit.incore
        def tile_softmax(a: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [64, 64])
            max_tmp = pl.create_tile([64, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_max: pl.Tile[[64, 1], pl.FP32] = pl.row_max(tile_a, max_tmp)
            shifted = pl.row_expand_sub(tile_a, row_max)
            exp_shifted = pl.exp(shifted)
            sum_tmp = pl.create_tile([64, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_sum: pl.Tile[[64, 1], pl.FP32] = pl.row_sum(exp_shifted, sum_tmp)
            result = pl.row_expand_div(exp_shifted, row_sum)
            out = pl.store(result, [0, 0], output)
            return out

        @jit
        def orchestrator(a: pl.Tensor, output: pl.Out[pl.Tensor]):
            output = tile_softmax(a, output)
            return output

        a = torch.randn(64, 64)
        out = torch.empty(64, 64)
        got = orchestrator.compile_for_test(a, out)
        ir.assert_structural_equal(got, expected)


# ---------------------------------------------------------------------------
# 07_normalization.py
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_rms_norm(self):
        """RMS normalization."""
        torch = pytest.importorskip("torch")

        @pl.program
        class RMSNormRef:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_rms_norm(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 64])
                tile_gamma = pl.load(gamma, [0, 0], [1, 64])
                squared = pl.mul(tile_x, tile_x)
                tmp = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                mean_sq: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(squared, tmp)
                mean_sq_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean_sq, [1, 32])
                mean_sq_T = pl.mul(mean_sq_T, 0.015625)
                mean_sq = pl.reshape(mean_sq_T, [32, 1])
                mean_sq_T = pl.reshape(mean_sq, [1, 32])
                rms_T = pl.add(mean_sq_T, 1e-5)
                rms_T = pl.sqrt(rms_T)
                rms = pl.reshape(rms_T, [32, 1])
                normalized = pl.row_expand_div(tile_x, rms)
                result = pl.col_expand_mul(normalized, tile_gamma)
                out = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def rms_norm_orch(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                output = self.kernel_rms_norm(x, gamma, output)
                return output

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(RMSNormRef)

        @jit.incore
        def kernel_rms_norm(x: pl.Tensor, gamma: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 64])
            tile_gamma = pl.load(gamma, [0, 0], [1, 64])
            squared = pl.mul(tile_x, tile_x)
            tmp = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            mean_sq: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(squared, tmp)
            mean_sq_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean_sq, [1, 32])
            mean_sq_T = pl.mul(mean_sq_T, 0.015625)
            mean_sq = pl.reshape(mean_sq_T, [32, 1])
            mean_sq_T = pl.reshape(mean_sq, [1, 32])
            rms_T = pl.add(mean_sq_T, 1e-5)
            rms_T = pl.sqrt(rms_T)
            rms = pl.reshape(rms_T, [32, 1])
            normalized = pl.row_expand_div(tile_x, rms)
            result = pl.col_expand_mul(normalized, tile_gamma)
            out = pl.store(result, [0, 0], output)
            return out

        @jit
        def rms_norm_orch(x: pl.Tensor, gamma: pl.Tensor, output: pl.Out[pl.Tensor]):
            output = kernel_rms_norm(x, gamma, output)
            return output

        x = torch.randn(32, 64)
        gamma = torch.randn(1, 64)
        out = torch.empty(32, 64)
        got = rms_norm_orch.compile_for_test(x, gamma, out)
        ir.assert_structural_equal(got, expected)

    def test_layer_norm(self):
        """Layer normalization."""
        torch = pytest.importorskip("torch")

        @pl.program
        class LayerNormRef:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_layer_norm(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                beta: pl.Tensor[[1, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 64])
                tile_gamma = pl.load(gamma, [0, 0], [1, 64])
                tile_beta = pl.load(beta, [0, 0], [1, 64])
                tmp = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                mean: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_x, tmp)
                mean_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean, [1, 32])
                mean_T = pl.mul(mean_T, 0.015625)
                mean = pl.reshape(mean_T, [32, 1])
                centered = pl.row_expand_sub(tile_x, mean)
                squared = pl.mul(centered, centered)
                tmp2 = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                var: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(squared, tmp2)
                var_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(var, [1, 32])
                var_T = pl.mul(var_T, 0.015625)
                var = pl.reshape(var_T, [32, 1])
                var_T = pl.reshape(var, [1, 32])
                var_eps_T = pl.add(var_T, 1e-5)
                std_T = pl.sqrt(var_eps_T)
                std = pl.reshape(std_T, [32, 1])
                normalized = pl.row_expand_div(centered, std)
                scaled = pl.col_expand_mul(normalized, tile_gamma)
                beta_full = pl.col_expand(scaled, tile_beta)
                result = pl.add(scaled, beta_full)
                out = pl.store(result, [0, 0], output)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def layer_norm_orch(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                beta: pl.Tensor[[1, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                output = self.kernel_layer_norm(x, gamma, beta, output)
                return output

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(LayerNormRef)

        @jit.incore
        def kernel_layer_norm(x: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, output: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 64])
            tile_gamma = pl.load(gamma, [0, 0], [1, 64])
            tile_beta = pl.load(beta, [0, 0], [1, 64])
            tmp = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            mean: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(tile_x, tmp)
            mean_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(mean, [1, 32])
            mean_T = pl.mul(mean_T, 0.015625)
            mean = pl.reshape(mean_T, [32, 1])
            centered = pl.row_expand_sub(tile_x, mean)
            squared = pl.mul(centered, centered)
            tmp2 = pl.create_tile([32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            var: pl.Tile[[32, 1], pl.FP32] = pl.row_sum(squared, tmp2)
            var_T: pl.Tile[[1, 32], pl.FP32] = pl.reshape(var, [1, 32])
            var_T = pl.mul(var_T, 0.015625)
            var = pl.reshape(var_T, [32, 1])
            var_T = pl.reshape(var, [1, 32])
            var_eps_T = pl.add(var_T, 1e-5)
            std_T = pl.sqrt(var_eps_T)
            std = pl.reshape(std_T, [32, 1])
            normalized = pl.row_expand_div(centered, std)
            scaled = pl.col_expand_mul(normalized, tile_gamma)
            beta_full = pl.col_expand(scaled, tile_beta)
            result = pl.add(scaled, beta_full)
            out = pl.store(result, [0, 0], output)
            return out

        @jit
        def layer_norm_orch(x: pl.Tensor, gamma: pl.Tensor, beta: pl.Tensor, output: pl.Out[pl.Tensor]):
            output = kernel_layer_norm(x, gamma, beta, output)
            return output

        x = torch.randn(32, 64)
        gamma = torch.randn(1, 64)
        beta = torch.randn(1, 64)
        out = torch.empty(32, 64)
        got = layer_norm_orch.compile_for_test(x, gamma, beta, out)
        ir.assert_structural_equal(got, expected)


# ---------------------------------------------------------------------------
# 08_assemble.py
# ---------------------------------------------------------------------------


class TestAssemble:
    def test_assemble_acc_mat(self):
        """Acc->Mat assemble."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileAssembleAccMatRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_assemble(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                a: pl.Tensor[[32, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
                tile_a_l1 = pl.load(a, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
                tile_a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_src = pl.matmul(tile_a, tile_b)
                result = pl.tile.assemble(tile_x, tile_src, [0, 16])
                result_vec = pl.move(result, target_memory=pl.MemorySpace.Vec)
                out_y = pl.store(result_vec, [0, 0], y)
                return out_y

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                a: pl.Tensor[[32, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                y = self.tile_assemble(x, a, b, y)
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAssembleAccMatRef)

        @jit.incore
        def tile_assemble(x: pl.Tensor, a: pl.Tensor, b: pl.Tensor, y: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
            tile_a_l1 = pl.load(a, [0, 0], [32, 16], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            tile_a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_src = pl.matmul(tile_a, tile_b)
            result = pl.tile.assemble(tile_x, tile_src, [0, 16])
            result_vec = pl.move(result, target_memory=pl.MemorySpace.Vec)
            out_y = pl.store(result_vec, [0, 0], y)
            return out_y

        @jit
        def orchestrator(x: pl.Tensor, a: pl.Tensor, b: pl.Tensor, y: pl.Out[pl.Tensor]):
            y = tile_assemble(x, a, b, y)
            return y

        x = torch.randn(32, 32)
        a = torch.randn(32, 16)
        b = torch.randn(16, 16)
        y = torch.empty(32, 32)
        got = orchestrator.compile_for_test(x, a, b, y)
        ir.assert_structural_equal(got, expected, enable_auto_mapping=True)

    def test_assemble_vec(self):
        """Vec->Vec single-shot assemble."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileAssembleVecRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_assemble(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[32, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
                tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
                result = pl.tile.assemble(tile_x, tile_src, [0, 0])
                out_y = pl.store(result, [0, 0], y)
                return out_y

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[32, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                y = self.tile_assemble(x, src, y)
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAssembleVecRef)

        @jit.incore
        def tile_assemble(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
            tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
            result = pl.tile.assemble(tile_x, tile_src, [0, 0])
            out_y = pl.store(result, [0, 0], y)
            return out_y

        @jit
        def orchestrator(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            y = tile_assemble(x, src, y)
            return y

        x = torch.randn(32, 32)
        src = torch.randn(32, 16)
        y = torch.empty(32, 32)
        got = orchestrator.compile_for_test(x, src, y)
        ir.assert_structural_equal(got, expected)

    def test_assemble_row_by_row(self):
        """Loop + pl.slice + assemble."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileAssembleRowByRowRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_assemble(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[32, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
                tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
                for i in pl.range(32):
                    row = pl.slice(tile_src, [1, 16], [i, 0])
                    tile_x = pl.tile.assemble(tile_x, row, [i, 0])
                out_y = pl.store(tile_x, [0, 0], y)
                return out_y

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[32, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                y = self.tile_assemble(x, src, y)
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAssembleRowByRowRef)

        @jit.incore
        def tile_assemble(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
            tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
            for i in pl.range(32):
                row = pl.slice(tile_src, [1, 16], [i, 0])
                tile_x = pl.tile.assemble(tile_x, row, [i, 0])
            out_y = pl.store(tile_x, [0, 0], y)
            return out_y

        @jit
        def orchestrator(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            y = tile_assemble(x, src, y)
            return y

        x = torch.randn(32, 32)
        src = torch.randn(32, 16)
        y = torch.empty(32, 32)
        got = orchestrator.compile_for_test(x, src, y)
        ir.assert_structural_equal(got, expected)

    def test_assemble_double_loop(self):
        """Nested loops + pl.slice."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileAssembleDoubleLoopRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_assemble(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[32, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
                tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
                for b in pl.range(4):
                    for i in pl.range(8):
                        row = b * 8 + i
                        tile_row = pl.slice(tile_src, [1, 16], [row, 0])
                        tile_x = pl.tile.assemble(tile_x, tile_row, [row, 0])
                out_y = pl.store(tile_x, [0, 0], y)
                return out_y

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[32, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                y = self.tile_assemble(x, src, y)
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAssembleDoubleLoopRef)

        @jit.incore
        def tile_assemble(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
            tile_src = pl.load(src, [0, 0], [32, 16], target_memory=pl.MemorySpace.Vec)
            for b in pl.range(4):
                for i in pl.range(8):
                    row = b * 8 + i
                    tile_row = pl.slice(tile_src, [1, 16], [row, 0])
                    tile_x = pl.tile.assemble(tile_x, tile_row, [row, 0])
            out_y = pl.store(tile_x, [0, 0], y)
            return out_y

        @jit
        def orchestrator(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            y = tile_assemble(x, src, y)
            return y

        x = torch.randn(32, 32)
        src = torch.randn(32, 16)
        y = torch.empty(32, 32)
        got = orchestrator.compile_for_test(x, src, y)
        ir.assert_structural_equal(got, expected)

    def test_assemble_loop_col_broadcast(self):
        """Loop with column broadcast."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileAssembleLoopColBroadcastRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_assemble(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[32, 8], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
                tile_src = pl.load(src, [0, 0], [32, 8], target_memory=pl.MemorySpace.Vec)
                for c in pl.range(4):
                    tile_x = pl.tile.assemble(tile_x, tile_src, [0, c * 8])
                out_y = pl.store(tile_x, [0, 0], y)
                return out_y

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[32, 8], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                y = self.tile_assemble(x, src, y)
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAssembleLoopColBroadcastRef)

        @jit.incore
        def tile_assemble(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
            tile_src = pl.load(src, [0, 0], [32, 8], target_memory=pl.MemorySpace.Vec)
            for c in pl.range(4):
                tile_x = pl.tile.assemble(tile_x, tile_src, [0, c * 8])
            out_y = pl.store(tile_x, [0, 0], y)
            return out_y

        @jit
        def orchestrator(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            y = tile_assemble(x, src, y)
            return y

        x = torch.randn(32, 32)
        src = torch.randn(32, 8)
        y = torch.empty(32, 32)
        got = orchestrator.compile_for_test(x, src, y)
        ir.assert_structural_equal(got, expected)

    def test_assemble_double_loop_broadcast(self):
        """Nested loops, quadrant broadcast."""
        torch = pytest.importorskip("torch")

        @pl.program
        class TileAssembleDoubleLoopBroadcastRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_assemble(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[16, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
                tile_src = pl.load(src, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                for b in pl.range(2):
                    for c in pl.range(2):
                        tile_x = pl.tile.assemble(tile_x, tile_src, [b * 16, c * 16])
                out_y = pl.store(tile_x, [0, 0], y)
                return out_y

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                src: pl.Tensor[[16, 16], pl.FP32],
                y: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                y = self.tile_assemble(x, src, y)
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAssembleDoubleLoopBroadcastRef)

        @jit.incore
        def tile_assemble(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            tile_x = pl.load(x, [0, 0], [32, 32], target_memory=pl.MemorySpace.Vec)
            tile_src = pl.load(src, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
            for b in pl.range(2):
                for c in pl.range(2):
                    tile_x = pl.tile.assemble(tile_x, tile_src, [b * 16, c * 16])
            out_y = pl.store(tile_x, [0, 0], y)
            return out_y

        @jit
        def orchestrator(x: pl.Tensor, src: pl.Tensor, y: pl.Out[pl.Tensor]):
            y = tile_assemble(x, src, y)
            return y

        x = torch.randn(32, 32)
        src = torch.randn(16, 16)
        y = torch.empty(32, 32)
        got = orchestrator.compile_for_test(x, src, y)
        ir.assert_structural_equal(got, expected)


# ---------------------------------------------------------------------------
# bind_dynamic: dynamic dimension round-trip
# ---------------------------------------------------------------------------


class TestDynamic:
    def test_multi_func_bind_dynamic_dim0(self):
        """Multi-function JIT: @jit.incore dep with pl.dynamic + bind_dynamic on dim 0.

        The dep marks dim 0 of all tensor params as dynamic ("M").  The JIT
        should produce the same compiled IR as an equivalent @pl.program whose
        tensor annotations use the same DynVar.
        """
        torch = pytest.importorskip("torch")

        # Hand-written ground truth using pl.dynamic
        M = pl.dynamic("M")

        @pl.program
        class TileAddDynRef:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_add_dyn(
                self,
                a: pl.Tensor[[M, 128], pl.FP32],
                b: pl.Tensor[[M, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[M, 128], pl.FP32]],
            ) -> pl.Tensor[[M, 128], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 128])
                tile_b = pl.load(b, [0, 0], [64, 128])
                tile_c = pl.add(tile_a, tile_b)
                out_c = pl.store(tile_c, [0, 0], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator_dyn(
                self,
                a: pl.Tensor[[M, 128], pl.FP32],
                b: pl.Tensor[[M, 128], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, 128], pl.FP32]],
            ) -> pl.Tensor[[M, 128], pl.FP32]:
                out_c = self.tile_add_dyn(a, b, out_c)
                return out_c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        expected = pm.run_passes(TileAddDynRef)

        @jit.incore
        def tile_add_dyn(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
            M = pl.dynamic("M")
            a.bind_dynamic(0, M)
            b.bind_dynamic(0, M)
            c.bind_dynamic(0, M)
            tile_a = pl.load(a, [0, 0], [64, 128])
            tile_b = pl.load(b, [0, 0], [64, 128])
            tile_c = pl.add(tile_a, tile_b)
            out_c = pl.store(tile_c, [0, 0], c)
            return out_c

        @jit
        def orchestrator_dyn(a: pl.Tensor, b: pl.Tensor, out_c: pl.Out[pl.Tensor]):
            out_c = tile_add_dyn(a, b, out_c)
            return out_c

        a = torch.randn(64, 128)
        b = torch.randn(64, 128)
        c = torch.empty(64, 128)
        got = orchestrator_dyn.compile_for_test(a, b, c)
        ir.assert_structural_equal(got, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
