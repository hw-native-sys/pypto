# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3 decode scope-3 mixed-kernel runtime system test.

This scope covers:
  1. Output projection: attn_out x wo, accumulated in Q_OUT_CHUNK tiles
  2. Residual addition with hidden_states
  3. Post-attention RMSNorm
  4. MLP: gate/up projections, SiLU activation, down projection
  5. Final residual addition

The original file was a standalone script executed via:
    python tests/st/runtime/test_qwen3_decode_scope3_mixed.py -d <device>

It is now structured as a standard pytest ST case so it can be collected and
run together with the rest of tests/st/, while preserving the same program and
reference compute logic. The __main__ block keeps a thin compatibility layer
for translating the old -d/-p flags into pytest's --device/--platform options.
"""

import sys
from pathlib import Path
from typing import Any

_ST_DIR = Path(__file__).resolve().parents[1]
if str(_ST_DIR) not in sys.path:
    sys.path.insert(0, str(_ST_DIR))

_PROJECT_ROOT = _ST_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pypto.language as pl  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from harness.core.harness import DataType, PTOTestCase, TensorSpec  # noqa: E402
from pypto.backend import BackendType  # noqa: E402
from pypto.runtime.runner import RunConfig  # noqa: E402

BATCH = 16
HIDDEN = 5120
INTERMEDIATE = 25600

EPS = 1e-6

K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
DOWN_OUT_CHUNK = 64
BATCH_TILE = 16


def build_qwen3_scope3_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_CFG = batch
    HIDDEN_CFG = hidden_size
    INTER_CFG = intermediate_size

    HIDDEN_BLOCKS = HIDDEN_CFG // K_CHUNK
    Q_OUT_BLOCKS = HIDDEN_CFG // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = INTER_CFG // MLP_OUT_CHUNK
    hidden_inv = 1.0 / HIDDEN_CFG

    @pl.program
    class Qwen3Scope3:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3(
            self,
            attn_out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    for ob in pl.parallel(0, Q_OUT_BLOCKS):
                        o0 = ob * Q_OUT_CHUNK
                        zero_resid1 = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        resid1_tile = pl.assemble(resid1_tile, zero_resid1, [0, o0])

                    # Output projection: attn_out × wo, tiled by Q_OUT_CHUNK.
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8, chunk_policy="leading_full"):
                        o0 = ob * Q_OUT_CHUNK
                        o_acc = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]),
                            target_type=pl.FP32,
                        )
                        resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(
                            sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE])
                        )
                    inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, hidden_inv), EPS))

                    post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                    down_proj_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    for zi in pl.range(HIDDEN_BLOCKS):
                        z0 = zi * K_CHUNK
                        down_zero_chunk = pl.full([BATCH_TILE, K_CHUNK], dtype=pl.FP32, value=0.0)
                        down_proj_tile = pl.assemble(down_proj_tile, down_zero_chunk, [0, z0])

                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [BATCH_TILE, 1])), gamma
                        )
                        post_norm_tile = pl.assemble(
                            post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                        )

                    for ob in pl.range(MLP_OUT_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK
                        gate_acc = pl.full([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        up_acc = pl.full([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)

                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                            up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                        for dob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
                            d0 = dob * K_CHUNK
                            for doff in pl.range(0, K_CHUNK, DOWN_OUT_CHUNK):
                                d1 = d0 + doff
                                down_prev = pl.slice(down_proj_tile, [BATCH_TILE, DOWN_OUT_CHUNK], [0, d1])
                                w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, DOWN_OUT_CHUNK], [o0, d1])
                                down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                                down_proj_tile = pl.assemble(down_proj_tile, down_next, [0, d1])

                    # Final residual: down_proj + resid1, write to output.
                    for ob in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
                        o0 = ob * K_CHUNK
                        down_acc = pl.add(
                            pl.slice(down_proj_tile, [BATCH_TILE, K_CHUNK], [0, o0]),
                            pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, o0]),
                        )
                        out = pl.assemble(out, pl.cast(down_acc, target_type=pl.BF16), [b0, o0])

            return out

    return Qwen3Scope3


def golden(tensors: dict, params: dict | None = None) -> None:
    """Reference computation for Scope 3.

    Steps:
      1. Output projection: attn_out (cast BF16) × wo, FP32 accumulation + residual
      2. Post-attention RMSNorm
      3. SwiGLU MLP: gate/up projections → silu(gate) * up → down projection
      4. Final residual addition → BF16 output
    """
    attn_out = tensors["attn_out"]
    hidden_states = tensors["hidden_states"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    eps = 1e-6

    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    tensors["out"][:] = (down + resid1).bfloat16()


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
) -> list[TensorSpec]:
    def init_attn_out():
        fan_in = hidden_size
        return (torch.randn([batch, hidden_size], dtype=torch.float32) / (fan_in**0.5)).to(torch.bfloat16)

    def init_hidden_states():
        fan_in = hidden_size
        return (torch.randn([batch, hidden_size], dtype=torch.float32) / (fan_in**0.5)).to(torch.bfloat16)

    def init_wo():
        fan_in = hidden_size
        return (torch.randn([hidden_size, hidden_size], dtype=torch.float32) / (fan_in**0.5)).to(
            torch.bfloat16
        )

    def init_post_rms_weight():
        fan_in = hidden_size
        return torch.randn([1, hidden_size], dtype=torch.float32) / (fan_in**0.5)

    def init_w_gate():
        fan_in = intermediate_size
        return (torch.randn([hidden_size, intermediate_size], dtype=torch.float32) / (fan_in**0.5)).to(
            torch.bfloat16
        )

    def init_w_up():
        fan_in = intermediate_size
        return (torch.randn([hidden_size, intermediate_size], dtype=torch.float32) / (fan_in**0.5)).to(
            torch.bfloat16
        )

    def init_w_down():
        fan_in = hidden_size
        return (torch.randn([intermediate_size, hidden_size], dtype=torch.float32) / (fan_in**0.5)).to(
            torch.bfloat16
        )

    return [
        TensorSpec("attn_out", [batch, hidden_size], DataType.BF16, init_value=init_attn_out),
        TensorSpec("hidden_states", [batch, hidden_size], DataType.BF16, init_value=init_hidden_states),
        TensorSpec("wo", [hidden_size, hidden_size], DataType.BF16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], DataType.FP32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], DataType.BF16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], DataType.BF16, init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], DataType.BF16, init_value=init_w_down),
        TensorSpec("out", [batch, hidden_size], DataType.BF16, is_output=True),
    ]


class Qwen3DecodeScope3MixedTestCase(PTOTestCase):
    """Shared ST test case for Qwen3 decode scope-3 mixed kernel."""

    __test__ = False
    compute_expected = staticmethod(golden)

    def __init__(
        self,
        batch: int = BATCH,
        hidden_size: int = HIDDEN,
        intermediate_size: int = INTERMEDIATE,
        *,
        backend_type: BackendType | None = None,
        config: RunConfig | None = None,
    ):
        super().__init__(config or RunConfig(rtol=1e-3, atol=1e-3), backend_type=backend_type)
        self._batch = batch
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size

    def get_name(self) -> str:
        return f"qwen3_decode_scope3_mixed_b{self._batch}_h{self._hidden_size}_i{self._intermediate_size}"

    def define_tensors(self) -> list[TensorSpec]:
        return build_tensor_specs(
            batch=self._batch,
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
        )

    def get_program(self) -> Any:
        return build_qwen3_scope3_program(
            batch=self._batch,
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
        )


class TestQwen3DecodeScope3Mixed:
    """Pytest entry points for the Qwen3 decode scope-3 ST coverage."""

    @pytest.mark.hardware
    def test_qwen3_decode_scope3_mixed(self, test_runner):
        """Run the original a2a3 hardware case under the shared ST harness."""
        result = test_runner.run(Qwen3DecodeScope3MixedTestCase(backend_type=BackendType.Ascend910B))
        assert result.passed, f"Qwen3 decode scope-3 mixed test failed: {result.error}"

    @pytest.mark.a5
    def test_qwen3_decode_scope3_mixed_a5(self, test_runner):
        """Run the same scope-3 test on the Ascend 950 backend."""
        if test_runner.config.platform.endswith("sim"):
            pytest.skip("a5sim CPU stub does not support BF16 TMATMUL for this mixed-kernel case yet")
        result = test_runner.run(Qwen3DecodeScope3MixedTestCase(backend_type=BackendType.Ascend950))
        assert result.passed, f"Qwen3 decode scope-3 mixed A5 test failed: {result.error}"


def _build_pytest_args(argv: list[str]) -> list[str]:
    """Translate legacy standalone flags into pytest-compatible arguments."""
    pytest_args = [__file__, "-v"]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"-d", "--device"}:
            if i + 1 >= len(argv):
                raise SystemExit("Missing value for -d/--device")
            pytest_args.extend(["--device", argv[i + 1]])
            i += 2
            continue
        if arg in {"-p", "--platform"}:
            if i + 1 >= len(argv):
                raise SystemExit("Missing value for -p/--platform")
            pytest_args.extend(["--platform", argv[i + 1]])
            i += 2
            continue
        pytest_args.append(arg)
        i += 1
    return pytest_args


if __name__ == "__main__":
    raise SystemExit(pytest.main(_build_pytest_args(sys.argv[1:])))
