# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Round-trip tests: @pl.jit equivalents of examples/ programs.

For each @pl.program in examples/, we write an equivalent @pl.jit version and
verify structural equality of the parsed IR using ir.assert_structural_equal.

Coverage
--------
examples/kernels/01_elementwise.py  -- TileAddProgram, TileMulProgram,
                                       TileAdd64Program, TileMul64Program
examples/kernels/02_fused_ops.py    -- FusedAddScaleProgram, FusedAddReluProgram,
                                       FusedMatmulBiasProgram, FusedLinearReluProgram
examples/kernels/03_matmul.py       -- MatmulProgram, MatmulaccProgram
examples/kernels/05_activation.py   -- SiluProgram, GeluProgram,
                                       SwigluProgram, GegluProgram
examples/kernels/06_softmax.py      -- TileSoftmaxProgram
examples/kernels/07_normalization.py -- RMSNormProgram, LayerNormProgram
examples/kernels/08_assemble.py     -- TileAssembleAccMatProgram,
                                       TileAssembleVecProgram,
                                       TileAssembleRowByRowProgram,
                                       TileAssembleDoubleLoopProgram,
                                       TileAssembleLoopColBroadcastProgram,
                                       TileAssembleDoubleLoopBroadcastProgram

Intentionally excluded (require features outside @pl.jit scope)
---------------------------------------------------------------
examples/kernels/04_concat.py       -- Orchestration has no Out param; output
                                       is created with pl.create_tensor inside
                                       the orchestrator.  @pl.jit cannot infer
                                       the return type in this pattern.
examples/kernels/09_dyn_valid_shape.py -- Uses module-level @pl.function (not
                                       @pl.jit.incore) and pl.tensor.read for
                                       scalar config tensors.
examples/models/                    -- Use module-level @pl.function called
                                       directly (not via @pl.jit.incore), which
                                       @pl.jit dep discovery does not cover.
"""

import pypto.language as pl
import pytest
from pypto.jit.decorator import jit
from pypto.pypto_core import ir

# ---------------------------------------------------------------------------
# 01_elementwise.py
# ---------------------------------------------------------------------------


class TestElementwise:
    def test_tile_add_128x128(self):
        """TileAddProgram: Style B round-trip for 128x128 FP32 add."""
        torch = pytest.importorskip("torch")
        from examples.kernels.elementwise import TileAddProgram  # noqa: PLC0415

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
        got = orchestrator(a, b, c)
        ir.assert_structural_equal(got, TileAddProgram)

    def test_tile_mul_128x128(self):
        """TileMulProgram: Style B round-trip for 128x128 FP32 mul."""
        torch = pytest.importorskip("torch")
        from examples.kernels.elementwise import TileMulProgram  # noqa: PLC0415

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
        got = orchestrator(a, b, c)
        ir.assert_structural_equal(got, TileMulProgram)

    def test_tile_add_64x64(self):
        """TileAdd64Program: Style B round-trip for 64x64 FP32 add."""
        torch = pytest.importorskip("torch")
        from examples.kernels.elementwise import TileAdd64Program  # noqa: PLC0415

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
        got = orchestrator(a, b, c)
        ir.assert_structural_equal(got, TileAdd64Program)

    def test_tile_mul_64x64(self):
        """TileMul64Program: Style B round-trip for 64x64 FP32 mul."""
        torch = pytest.importorskip("torch")
        from examples.kernels.elementwise import TileMul64Program  # noqa: PLC0415

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
        got = orchestrator(a, b, c)
        ir.assert_structural_equal(got, TileMul64Program)


# ---------------------------------------------------------------------------
# 02_fused_ops.py
# ---------------------------------------------------------------------------


class TestFusedOps:
    def test_fused_add_scale(self):
        """FusedAddScaleProgram: (a + b) * 2.0."""
        torch = pytest.importorskip("torch")
        from examples.kernels.fused_ops import FusedAddScaleProgram  # noqa: PLC0415

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
        got = orchestrator(a, b, c)
        ir.assert_structural_equal(got, FusedAddScaleProgram)

    def test_fused_add_relu(self):
        """FusedAddReluProgram: relu(a + b)."""
        torch = pytest.importorskip("torch")
        from examples.kernels.fused_ops import FusedAddReluProgram  # noqa: PLC0415

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
        got = orchestrator(a, b, c)
        ir.assert_structural_equal(got, FusedAddReluProgram)

    def test_fused_matmul_bias(self):
        """FusedMatmulBiasProgram: c = matmul(a, b) + bias."""
        torch = pytest.importorskip("torch")
        from examples.kernels.fused_ops import FusedMatmulBiasProgram  # noqa: PLC0415

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
        got = orchestrator(a, b, bias, c)
        ir.assert_structural_equal(got, FusedMatmulBiasProgram)

    def test_fused_linear_relu(self):
        """FusedLinearReluProgram: y = relu(matmul(x, w) + bias)."""
        torch = pytest.importorskip("torch")
        from examples.kernels.fused_ops import FusedLinearReluProgram  # noqa: PLC0415

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
        got = orchestrator(x, w, bias, y)
        ir.assert_structural_equal(got, FusedLinearReluProgram)


# ---------------------------------------------------------------------------
# 03_matmul.py
# ---------------------------------------------------------------------------


class TestMatmul:
    def test_matmul_program(self):
        """MatmulProgram: 64x64 matmul."""
        torch = pytest.importorskip("torch")
        from examples.kernels.matmul import MatmulProgram  # noqa: PLC0415

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
        got = orchestrator(a, b, c)
        ir.assert_structural_equal(got, MatmulProgram)

    def test_matmulacc_program(self):
        """MatmulaccProgram: K=64 split into two K=32 chunks."""
        torch = pytest.importorskip("torch")
        from examples.kernels.matmul import MatmulaccProgram  # noqa: PLC0415

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
        got = orchestrator(a, b, c)
        ir.assert_structural_equal(got, MatmulaccProgram)


# ---------------------------------------------------------------------------
# 05_activation.py
# ---------------------------------------------------------------------------


class TestActivation:
    def test_silu(self):
        """SiluProgram: SiLU activation."""
        torch = pytest.importorskip("torch")
        from examples.kernels.activation import SiluProgram  # noqa: PLC0415

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
        got = silu_orch(x, out)
        ir.assert_structural_equal(got, SiluProgram)

    def test_gelu(self):
        """GeluProgram: GELU activation."""
        torch = pytest.importorskip("torch")
        from examples.kernels.activation import GeluProgram  # noqa: PLC0415

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
        got = gelu_orch(x, out)
        ir.assert_structural_equal(got, GeluProgram)

    def test_swiglu(self):
        """SwigluProgram: SwiGLU activation."""
        torch = pytest.importorskip("torch")
        from examples.kernels.activation import SwigluProgram  # noqa: PLC0415

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
        got = swiglu_orch(gate, up, out)
        ir.assert_structural_equal(got, SwigluProgram)

    def test_geglu(self):
        """GegluProgram: GeGLU activation."""
        torch = pytest.importorskip("torch")
        from examples.kernels.activation import GegluProgram  # noqa: PLC0415

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
        got = geglu_orch(gate, up, out)
        ir.assert_structural_equal(got, GegluProgram)


# ---------------------------------------------------------------------------
# 06_softmax.py
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_tile_softmax(self):
        """TileSoftmaxProgram: row-wise softmax."""
        torch = pytest.importorskip("torch")
        from examples.kernels.softmax import TileSoftmaxProgram  # noqa: PLC0415

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
        got = orchestrator(a, out)
        ir.assert_structural_equal(got, TileSoftmaxProgram)


# ---------------------------------------------------------------------------
# 07_normalization.py
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_rms_norm(self):
        """RMSNormProgram: RMS normalization."""
        torch = pytest.importorskip("torch")
        from examples.kernels.normalization import RMSNormProgram  # noqa: PLC0415

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
        got = rms_norm_orch(x, gamma, out)
        ir.assert_structural_equal(got, RMSNormProgram)

    def test_layer_norm(self):
        """LayerNormProgram: Layer normalization."""
        torch = pytest.importorskip("torch")
        from examples.kernels.normalization import LayerNormProgram  # noqa: PLC0415

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
        got = layer_norm_orch(x, gamma, beta, out)
        ir.assert_structural_equal(got, LayerNormProgram)


# ---------------------------------------------------------------------------
# 08_assemble.py
# ---------------------------------------------------------------------------


class TestAssemble:
    def test_assemble_acc_mat(self):
        """TileAssembleAccMatProgram: Acc->Mat assemble."""
        torch = pytest.importorskip("torch")
        from examples.kernels.assemble import TileAssembleAccMatProgram  # noqa: PLC0415

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
        got = orchestrator(x, a, b, y)
        ir.assert_structural_equal(got, TileAssembleAccMatProgram)

    def test_assemble_vec(self):
        """TileAssembleVecProgram: Vec->Vec single-shot assemble."""
        torch = pytest.importorskip("torch")
        from examples.kernels.assemble import TileAssembleVecProgram  # noqa: PLC0415

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
        got = orchestrator(x, src, y)
        ir.assert_structural_equal(got, TileAssembleVecProgram)

    def test_assemble_row_by_row(self):
        """TileAssembleRowByRowProgram: loop + pl.slice + assemble."""
        torch = pytest.importorskip("torch")
        from examples.kernels.assemble import TileAssembleRowByRowProgram  # noqa: PLC0415

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
        got = orchestrator(x, src, y)
        ir.assert_structural_equal(got, TileAssembleRowByRowProgram)

    def test_assemble_double_loop(self):
        """TileAssembleDoubleLoopProgram: nested loops + pl.slice."""
        torch = pytest.importorskip("torch")
        from examples.kernels.assemble import TileAssembleDoubleLoopProgram  # noqa: PLC0415

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
        got = orchestrator(x, src, y)
        ir.assert_structural_equal(got, TileAssembleDoubleLoopProgram)

    def test_assemble_loop_col_broadcast(self):
        """TileAssembleLoopColBroadcastProgram: loop with column broadcast."""
        torch = pytest.importorskip("torch")
        from examples.kernels.assemble import TileAssembleLoopColBroadcastProgram  # noqa: PLC0415

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
        got = orchestrator(x, src, y)
        ir.assert_structural_equal(got, TileAssembleLoopColBroadcastProgram)

    def test_assemble_double_loop_broadcast(self):
        """TileAssembleDoubleLoopBroadcastProgram: nested loops, quadrant broadcast."""
        torch = pytest.importorskip("torch")
        from examples.kernels.assemble import TileAssembleDoubleLoopBroadcastProgram  # noqa: PLC0415

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
        got = orchestrator(x, src, y)
        ir.assert_structural_equal(got, TileAssembleDoubleLoopBroadcastProgram)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
