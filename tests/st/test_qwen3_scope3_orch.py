# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Qwen3 Scope 3 orchestration decomposition tests.

Tests each incore function call from the Qwen3 Scope 3 orchestration
independently. Derived from the pass dump at:
  build_output/Qwen3Scope3_*/passes_dump/19_after_AllocateMemoryAddr.py

The orchestration calls 6 incore functions per batch tile:
  0. Output projection + residual  (scope3_incore_0)
  1. RMSNorm squared sum → inv_rms (scope3_incore_1)
  2. Post-attention normalize      (scope3_incore_2)
  3. MLP gate/up + SiLU            (scope3_incore_3)
  4. Down projection accumulation  (scope3_incore_4)
  5. Final residual → BF16 output  (scope3_incore_5)

Each test exercises one incore function in isolation with scaled-down
dimensions (BATCH=4, HIDDEN=512, INTERMEDIATE=128) while preserving
the tiling structure of the full model.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime.runner import RunConfig

# Test dimensions (scaled down from full Qwen3: B=16, H=5120, I=25600)
BATCH = 16
HIDDEN = 512
INTERMEDIATE = 128

# Tiling constants (same as full model)
K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64

# Derived
HIDDEN_BLOCKS = HIDDEN // K_CHUNK  # 4
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK  # 8
MLP_OUT_BLOCKS = INTERMEDIATE // MLP_OUT_CHUNK  # 2

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN


class _Scope3TestBase(PTOTestCase):
    """Base class with Qwen3 Scope 3 defaults (Ascend950 / Default strategy)."""

    __test__ = False

    def __init__(self, config=None):
        if config is None:
            config = RunConfig(rtol=2e-2, atol=2e-2)
        super().__init__(config)

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend950


# ---------------------------------------------------------------------------
# Test 0: Output Projection + Residual (scope3_incore_0)
#   resid1 = matmul(cast(attn_out, BF16), wo) + cast(hidden_states, FP32)
#   Tiled across Q_OUT_CHUNK=64, K_CHUNK=128.
# ---------------------------------------------------------------------------


class _OutProjResidCase(_Scope3TestBase):
    __test__ = False

    def get_name(self) -> str:
        return "qwen3_s3_incore0_outproj_resid"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("attn_out", [BATCH, HIDDEN], DataType.FP32, init_value=torch.randn),
            TensorSpec("hidden_states", [BATCH, HIDDEN], DataType.BF16, init_value=torch.randn),
            TensorSpec("wo", [HIDDEN, HIDDEN], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [BATCH, HIDDEN], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def compute(
                self,
                attn_out: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
                hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
                wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
                out: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
            ) -> pl.Tensor[[BATCH, HIDDEN], pl.FP32]:
                with pl.auto_incore():
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                        o0 = ob * Q_OUT_CHUNK
                        o_acc = pl.create_tensor([BATCH, Q_OUT_CHUNK], dtype=pl.FP32)
                        o_acc = pl.mul(o_acc, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.cast(
                                pl.slice(attn_out, [BATCH, K_CHUNK], [0, k0]),
                                target_type=pl.BF16,
                            )
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH, Q_OUT_CHUNK], [0, o0]),
                            target_type=pl.FP32,
                        )
                        out = pl.assemble(out, pl.add(o_acc, resid), [0, o0])
                return out

        return Prog

    def compute_expected(self, tensors, params=None):
        attn_out = tensors["attn_out"]
        hidden_states = tensors["hidden_states"]
        wo = tensors["wo"]
        o_proj = torch.matmul(attn_out.bfloat16().float(), wo.float())
        tensors["out"][:] = o_proj + hidden_states.float()


# ---------------------------------------------------------------------------
# Test 1: RMSNorm inv_rms (scope3_incore_1)
#   inv_rms = rsqrt(sum(x²) / HIDDEN + eps)
# ---------------------------------------------------------------------------


class _RMSNormInvRMSCase(_Scope3TestBase):
    __test__ = False

    def get_name(self) -> str:
        return "qwen3_s3_incore1_rmsnorm"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("resid1", [BATCH, HIDDEN], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [BATCH, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def compute(
                self,
                resid1: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
                out: pl.Tensor[[BATCH, 1], pl.FP32],
            ) -> pl.Tensor[[BATCH, 1], pl.FP32]:
                with pl.auto_incore():
                    sq_sum = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
                    sq_sum = pl.mul(sq_sum, 0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1, [BATCH, K_CHUNK], [0, k0])
                        sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                    inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))
                    out = pl.assemble(out, inv_rms, [0, 0])
                return out

        return Prog

    def compute_expected(self, tensors, params=None):
        resid1 = tensors["resid1"]
        sq_sum = resid1.pow(2).sum(dim=-1, keepdim=True)
        tensors["out"][:] = torch.rsqrt(sq_sum * HIDDEN_INV + EPS)


# ---------------------------------------------------------------------------
# Test 2: Post-attention Normalize (scope3_incore_2)
#   post_norm = cast(x * inv_rms * gamma, BF16)
# ---------------------------------------------------------------------------


class _PostNormCase(_Scope3TestBase):
    __test__ = False

    def get_name(self) -> str:
        return "qwen3_s3_incore2_post_norm"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("resid1", [BATCH, HIDDEN], DataType.FP32, init_value=torch.randn),
            TensorSpec("inv_rms", [BATCH, 1], DataType.FP32, init_value=torch.randn),
            TensorSpec("post_rms_weight", [1, HIDDEN], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [BATCH, HIDDEN], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def compute(
                self,
                resid1: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
                inv_rms: pl.Tensor[[BATCH, 1], pl.FP32],
                post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
                out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
                with pl.auto_incore():
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1, [BATCH, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                        out = pl.assemble(out, pl.cast(normed, target_type=pl.BF16), [0, k0])
                return out

        return Prog

    def compute_expected(self, tensors, params=None):
        resid1 = tensors["resid1"]
        inv_rms = tensors["inv_rms"]
        weight = tensors["post_rms_weight"]
        tensors["out"][:] = (resid1 * inv_rms * weight).bfloat16()


# ---------------------------------------------------------------------------
# Test 3: MLP Gate/Up + SiLU (scope3_incore_3)
#   gate = post_norm @ w_gate,  up = post_norm @ w_up
#   out  = cast(gate * sigmoid(gate) * up, BF16)
#
#   Tests a single MLP_OUT_CHUNK column (one iteration of the ob loop).
# ---------------------------------------------------------------------------


class _MLPGateUpSiLUCase(_Scope3TestBase):
    __test__ = False

    def get_name(self) -> str:
        return "qwen3_s3_incore3_gate_up_silu"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("post_norm", [BATCH, HIDDEN], DataType.BF16, init_value=torch.randn),
            TensorSpec("w_gate", [HIDDEN, MLP_OUT_CHUNK], DataType.BF16, init_value=torch.randn),
            TensorSpec("w_up", [HIDDEN, MLP_OUT_CHUNK], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [BATCH, MLP_OUT_CHUNK], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def compute(
                self,
                post_norm: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
                w_gate: pl.Tensor[[HIDDEN, MLP_OUT_CHUNK], pl.BF16],
                w_up: pl.Tensor[[HIDDEN, MLP_OUT_CHUNK], pl.BF16],
                out: pl.Tensor[[BATCH, MLP_OUT_CHUNK], pl.BF16],
            ) -> pl.Tensor[[BATCH, MLP_OUT_CHUNK], pl.BF16]:
                with pl.auto_incore():
                    gate_acc = pl.create_tensor([BATCH, MLP_OUT_CHUNK], dtype=pl.FP32)
                    up_acc = pl.create_tensor([BATCH, MLP_OUT_CHUNK], dtype=pl.FP32)
                    gate_acc = pl.mul(gate_acc, 0.0)
                    up_acc = pl.mul(up_acc, 0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        post_chunk = pl.slice(post_norm, [BATCH, K_CHUNK], [0, k0])
                        wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, 0])
                        wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, 0])
                        gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                        up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                    mlp_out = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                    out = pl.assemble(out, pl.cast(mlp_out, target_type=pl.BF16), [0, 0])
                return out

        return Prog

    def compute_expected(self, tensors, params=None):
        post_norm = tensors["post_norm"]
        w_gate = tensors["w_gate"]
        w_up = tensors["w_up"]
        gate = torch.matmul(post_norm.float(), w_gate.float())
        up = torch.matmul(post_norm.float(), w_up.float())
        tensors["out"][:] = (gate * torch.sigmoid(gate) * up).bfloat16()


# ---------------------------------------------------------------------------
# Test 4: Down Projection (scope3_incore_4)
#   out = down_proj + matmul(mlp_chunk_bf16, w_down)
#   Tiled across K_CHUNK=128 with parallel(chunk=4).
# ---------------------------------------------------------------------------


class _DownProjectionCase(_Scope3TestBase):
    __test__ = False

    def get_name(self) -> str:
        return "qwen3_s3_incore4_down_proj"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("down_proj", [BATCH, HIDDEN], DataType.FP32, init_value=torch.randn),
            TensorSpec("mlp_chunk", [BATCH, MLP_OUT_CHUNK], DataType.BF16, init_value=torch.randn),
            TensorSpec("w_down", [MLP_OUT_CHUNK, HIDDEN], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [BATCH, HIDDEN], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def compute(
                self,
                down_proj: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
                mlp_chunk: pl.Tensor[[BATCH, MLP_OUT_CHUNK], pl.BF16],
                w_down: pl.Tensor[[MLP_OUT_CHUNK, HIDDEN], pl.BF16],
                out: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
            ) -> pl.Tensor[[BATCH, HIDDEN], pl.FP32]:
                with pl.auto_incore():
                    for dob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                        d0 = dob * K_CHUNK
                        down_prev = pl.slice(down_proj, [BATCH, K_CHUNK], [0, d0])
                        w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                        down_next = pl.add(down_prev, pl.matmul(mlp_chunk, w_down_chunk))
                        out = pl.assemble(out, down_next, [0, d0])
                return out

        return Prog

    def compute_expected(self, tensors, params=None):
        down_proj = tensors["down_proj"]
        mlp_chunk = tensors["mlp_chunk"]
        w_down = tensors["w_down"]
        tensors["out"][:] = down_proj + torch.matmul(mlp_chunk.float(), w_down.float())


# ---------------------------------------------------------------------------
# Test 5: Final Residual (scope3_incore_5)
#   out = cast(down_proj + resid1, BF16)
#   Tiled across K_CHUNK=128 with parallel(chunk=4).
# ---------------------------------------------------------------------------


class _FinalResidualCase(_Scope3TestBase):
    __test__ = False

    def get_name(self) -> str:
        return "qwen3_s3_incore5_final_resid"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("down_proj", [BATCH, HIDDEN], DataType.FP32, init_value=torch.randn),
            TensorSpec("resid1", [BATCH, HIDDEN], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [BATCH, HIDDEN], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def compute(
                self,
                down_proj: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
                resid1: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
                out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
                with pl.auto_incore():
                    for ob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                        o0 = ob * K_CHUNK
                        down_acc = pl.add(
                            pl.slice(down_proj, [BATCH, K_CHUNK], [0, o0]),
                            pl.slice(resid1, [BATCH, K_CHUNK], [0, o0]),
                        )
                        out = pl.assemble(out, pl.cast(down_acc, target_type=pl.BF16), [0, o0])
                return out

        return Prog

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = (tensors["down_proj"] + tensors["resid1"]).bfloat16()


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


@pytest.mark.a5
class TestQwen3Scope3Orch:
    """Test each incore function call from the Qwen3 Scope 3 orchestration."""

    def test_incore0_output_proj_residual(self, test_runner):
        """Output projection: cast(attn_out, BF16) x wo + hidden_states."""
        result = test_runner.run(_OutProjResidCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_incore1_rmsnorm_inv_rms(self, test_runner):
        """RMSNorm: inv_rms = rsqrt(mean(x^2) + eps)."""
        result = test_runner.run(_RMSNormInvRMSCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_incore2_post_norm(self, test_runner):
        """Post-attention normalize: cast(x * inv_rms * gamma, BF16)."""
        result = test_runner.run(_PostNormCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_incore3_mlp_gate_up_silu(self, test_runner):
        """MLP gate/up + SiLU: cast(silu(gate) * up, BF16)."""
        result = test_runner.run(_MLPGateUpSiLUCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_incore4_down_projection(self, test_runner):
        """Down projection: down_proj + matmul(mlp_chunk, w_down)."""
        result = test_runner.run(_DownProjectionCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_incore5_final_residual(self, test_runner):
        """Final residual: cast(down_proj + resid1, BF16)."""
        result = test_runner.run(_FinalResidualCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
