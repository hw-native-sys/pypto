# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
FFN activation functions using PyPTO language DSL.

Programs:
  GeluProgram    — GELU:   output = x * sigmoid(1.702 * x)        (128x128)
  SwigluProgram  — SwiGLU: output = gate * sigmoid(gate) * up     (64x64)
  SiluProgram    — SiLU:   output = x * sigmoid(x)                (64x64)
  GegluProgram   — GeGLU:  output = gate * sigmoid(1.702 * gate) * up  (64x64)
"""

import pypto.language as pl


@pl.program
class GeluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_gelu(
        self,
        x: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_x: pl.Tile[[128, 128], pl.FP32] = pl.load(x, [0, 0], [128, 128])
        x_scaled: pl.Tile[[128, 128], pl.FP32] = pl.mul(tile_x, 1.702)  # type: ignore[reportArgumentType]
        x_neg: pl.Tile[[128, 128], pl.FP32] = pl.mul(x_scaled, -1.0)  # type: ignore[reportArgumentType]
        exp_neg: pl.Tile[[128, 128], pl.FP32] = pl.exp(x_neg)
        denom: pl.Tile[[128, 128], pl.FP32] = pl.add(exp_neg, 1.0)  # type: ignore[reportArgumentType]
        sigmoid: pl.Tile[[128, 128], pl.FP32] = pl.recip(denom)
        result: pl.Tile[[128, 128], pl.FP32] = pl.mul(tile_x, sigmoid)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(result, [0, 0], [128, 128], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def gelu_orch(
        self,
        x: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        output: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        output = self.kernel_gelu(x, output)
        return output


@pl.program
class SwigluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_swiglu(
        self,
        gate: pl.Tensor[[64, 64], pl.FP32],
        up: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # Swish(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
        # Note: -1.0 and 1.0 are inlined as literals; outer variables
        # are not accessible inside the DSL kernel closure.
        tile_gate: pl.Tile[[64, 64], pl.FP32] = pl.load(gate, [0, 0], [64, 64])
        tile_up: pl.Tile[[64, 64], pl.FP32] = pl.load(up, [0, 0], [64, 64])
        gate_neg: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_gate, -1.0)  # type: ignore[reportArgumentType]
        exp_neg: pl.Tile[[64, 64], pl.FP32] = pl.exp(gate_neg)
        denom: pl.Tile[[64, 64], pl.FP32] = pl.add(exp_neg, 1.0)  # type: ignore[reportArgumentType]
        sigmoid: pl.Tile[[64, 64], pl.FP32] = pl.recip(denom)
        swish: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_gate, sigmoid)
        result: pl.Tile[[64, 64], pl.FP32] = pl.mul(swish, tile_up)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def swiglu_orch(
        self,
        gate: pl.Tensor[[64, 64], pl.FP32],
        up: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        output = self.kernel_swiglu(gate, up, output)
        return output


@pl.program
class SiluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_silu(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        tile_x: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        x_neg: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_x, -1.0)  # type: ignore[reportArgumentType]
        exp_neg: pl.Tile[[64, 64], pl.FP32] = pl.exp(x_neg)
        denom: pl.Tile[[64, 64], pl.FP32] = pl.add(exp_neg, 1.0)  # type: ignore[reportArgumentType]
        sigmoid: pl.Tile[[64, 64], pl.FP32] = pl.recip(denom)
        result: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_x, sigmoid)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def silu_orch(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        output = self.kernel_silu(x, output)
        return output


@pl.program
class GegluProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_geglu(
        self,
        gate: pl.Tensor[[64, 64], pl.FP32],
        up: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        # GeGLU(gate, up) = GELU(gate) * up
        # GELU approximation: gate * sigmoid(1.702 * gate)
        tile_gate: pl.Tile[[64, 64], pl.FP32] = pl.load(gate, [0, 0], [64, 64])
        tile_up: pl.Tile[[64, 64], pl.FP32] = pl.load(up, [0, 0], [64, 64])
        gate_scaled: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_gate, 1.702)  # type: ignore[reportArgumentType]
        gate_scaled_neg: pl.Tile[[64, 64], pl.FP32] = pl.mul(gate_scaled, -1.0)  # type: ignore[reportArgumentType]
        exp_neg: pl.Tile[[64, 64], pl.FP32] = pl.exp(gate_scaled_neg)
        denom: pl.Tile[[64, 64], pl.FP32] = pl.add(exp_neg, 1.0)  # type: ignore[reportArgumentType]
        sigmoid: pl.Tile[[64, 64], pl.FP32] = pl.recip(denom)
        gelu_gate: pl.Tile[[64, 64], pl.FP32] = pl.mul(tile_gate, sigmoid)
        result: pl.Tile[[64, 64], pl.FP32] = pl.mul(gelu_gate, tile_up)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], [64, 64], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def geglu_orch(
        self,
        gate: pl.Tensor[[64, 64], pl.FP32],
        up: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        output: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
        output = self.kernel_geglu(gate, up, output)
        return output
