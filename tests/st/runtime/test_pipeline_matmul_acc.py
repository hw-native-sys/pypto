# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime regression test for #1352: acc→acc pto.tmov in pipeline matmul_acc.

Reproduces the gate_up_silu pattern from the Qwen3-32B decode kernel that
triggered the bug: two independent accumulators (gate_acc, up_acc) are built
inside a ``pl.at(CORE_GROUP, split=UP_DOWN)`` scope, each following the
prolog-then-pipeline pattern:

    gate_acc = pl.matmul(x_chunk_0, wg_0, out_dtype=pl.FP32)
    gate_acc = pl.matmul_acc(gate_acc, x_chunk_1, wg_1)
    for kb in pl.pipeline(2, NUM_CHUNKS, stage=2):
        gate_acc = pl.matmul_acc(gate_acc, x_chunk_k, wg_k)

Three conditions combine to expose the bug:

1. **Mat-resident inputs** — ``pl.slice`` passes Mat-space tiles to
   ``pl.matmul`` / ``pl.matmul_acc``, so AutoTileMatmulL0 sees them and
   inserts an inner K-loop (K_CHUNK=128, N=256 → effective L0B = 32 KB <
   128×256×2 B = 64 KB, so ChooseL0Tile picks K_L0=64).

2. **pl.pipeline(stage=2)** — LowerPipelineLoops replicates the loop body,
   compounding the IterArg nesting around the ``_l0_c`` accumulator variable
   introduced by AutoTileMatmulL0.

3. **CORE_GROUP + split=UP_DOWN** — the AIC/AIV split causes MemoryReuse to
   run only on the AIC function.  With two concurrent same-shape ``[16, 256]``
   FP32 accumulators live in the function, AllocateMemoryAddr assigns them
   addr=0 and addr=16384 respectively.  MemoryReuse fails to unify the
   ``_l0_c`` IterArg base with the outer acc base, so YieldFixupMutator
   inserts ``tile.move acc@0 → acc@16384 → acc@0``.  Without the fix in
   ``src/backend/common/pto_ops_common.cpp``, ptoas rejects these with
   ``'pto.tmov' op expects a supported tmov address-space pair for this
   target``.

Usage note on sequential consumption
-------------------------------------
Because Path 1 aliases *both* gate_acc and up_acc to the same physical
buffer (acc@0) in the generated code, the final add must read each
accumulator **at the point where acc@0 holds its correct value**:

- After the gate pipeline completes, acc@0 = gate result.
  → cast gate_acc to BF16 (Vec space) **before** the up pipeline starts.
- After the up pipeline completes, acc@0 = up result.
  → cast up_acc to BF16 **after** the up pipeline.

This is exactly the access order used in the Qwen3-32B gate_up_silu kernel:
gate is consumed (cast + silu) between the two pipeline loops, so by the time
up pipeline overwrites acc@0 the gate value has already been saved.

The output is ``gate_result (FP32) * up_result (FP32)``, cast and stored as BF16.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

# ---------------------------------------------------------------------------
# Kernel dimensions (match gate_up_silu in qwen3_32b_decode_scope3.py)
# ---------------------------------------------------------------------------
_BATCH_TILE = 16
_K_CHUNK = 128
_MLP_OUT_CHUNK = 256
_NUM_CHUNKS = 4  # total K chunks; prolog uses 2, pipeline(2,4,stage=2) does 2


class TestPipelineMatmulAccGateUp(PTOTestCase):
    """Dual-accumulator gate/up matmul via matmul_acc + pl.pipeline(stage=2).

    Computes:
        gate_result[B, N] = x @ wg       (BF16 inputs, FP32 accum)
        up_result[B, N]   = x @ wu       (BF16 inputs, FP32 accum)
        out[B, N]         = (gate_result * up_result), cast to BF16

    The program uses the exact prolog-then-pipeline pattern from the Qwen3-32B
    gate_up_silu kernel that triggered the acc→acc pto.tmov bug (#1352).

    Note: only ``split=UP_DOWN`` is used (no ``auto_chunk``).  ``split=UP_DOWN``
    alone is what forces the AIC/AIV split that triggers MemoryReuse on the AIC
    function — the root condition for #1352.  ``auto_chunk`` is deliberately
    omitted to avoid it incorrectly distributing K-pipeline iterations across
    cores when no ``pl.parallel`` outer loop is present.
    """

    __test__ = False

    def __init__(
        self,
        batch: int = _BATCH_TILE,
        k_chunk: int = _K_CHUNK,
        n: int = _MLP_OUT_CHUNK,
        num_chunks: int = _NUM_CHUNKS,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self.BATCH = batch
        self.K_CHUNK = k_chunk
        self.N = n
        self.NUM_CHUNKS = num_chunks
        self.K = k_chunk * num_chunks

    def get_name(self) -> str:
        return f"pipeline_matmul_acc_gate_up_{self.BATCH}x{self.K}x{self.N}_chunks{self.NUM_CHUNKS}"

    def define_tensors(self) -> list[TensorSpec]:
        K = self.K
        return [
            TensorSpec(
                "x", [self.BATCH, K], DataType.BF16, init_value=lambda: (torch.rand(self.BATCH, K) - 0.5) * 2
            ),
            TensorSpec(
                "wg", [K, self.N], DataType.BF16, init_value=lambda: (torch.rand(K, self.N) - 0.5) / K**0.5
            ),
            TensorSpec(
                "wu", [K, self.N], DataType.BF16, init_value=lambda: (torch.rand(K, self.N) - 0.5) / K**0.5
            ),
            TensorSpec("out", [self.BATCH, self.N], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        BATCH = self.BATCH
        K_CHUNK = self.K_CHUNK
        N = self.N
        NUM_CHUNKS = self.NUM_CHUNKS

        @pl.program
        class PipelineGateUpProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[BATCH, K_CHUNK * NUM_CHUNKS], pl.BF16],
                wg: pl.Tensor[[K_CHUNK * NUM_CHUNKS, N], pl.BF16],
                wu: pl.Tensor[[K_CHUNK * NUM_CHUNKS, N], pl.BF16],
                out: pl.Out[pl.Tensor[[BATCH, N], pl.BF16]],
            ) -> pl.Tensor[[BATCH, N], pl.BF16]:
                # Mirror the exact gate_up_silu structure from qwen3_32b:
                # split=UP_DOWN triggers the AIC/AIV split that causes
                # MemoryReuse to run on the AIC function, where the two
                # concurrent same-shape accumulators expose the MemRef-base
                # mismatch that triggers the acc→acc pto.tmov bug (#1352).
                #
                # out_tile is a separate DDR BF16 staging tensor used as an
                # intermediate write target before the final assemble into the
                # caller-provided ``out``.  Assembling a BF16 Vec tile directly
                # into the caller's DDR ND tensor (via TStore NZ→ND) fails with
                # a layout mismatch on A2/A3; routing through a fresh DDR
                # staging tensor (allocated here via pl.create_tensor) and then
                # assembling that into ``out`` avoids the restriction.  The
                # name "out_tile" is historical; this is a Tensor, not a Tile.
                out_tile = pl.create_tensor([BATCH, N], dtype=pl.BF16)

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
                    name_hint="gate_up_pipeline_acc",
                ):
                    # Prolog: unrolled first two K_CHUNK slices
                    x0 = pl.slice(x, [BATCH, K_CHUNK], [0, 0])
                    x1 = pl.slice(x, [BATCH, K_CHUNK], [0, K_CHUNK])
                    wg0 = pl.slice(wg, [K_CHUNK, N], [0, 0])
                    wg1 = pl.slice(wg, [K_CHUNK, N], [K_CHUNK, 0])
                    wu0 = pl.slice(wu, [K_CHUNK, N], [0, 0])
                    wu1 = pl.slice(wu, [K_CHUNK, N], [K_CHUNK, 0])

                    # Gate projection — mirrors the exact structure in qwen3_32b
                    # gate_up_silu: each of gate and up has its OWN independent
                    # pl.pipeline loop.  Two separate pipeline loops over the
                    # same K range produce two independent IterArg chains for
                    # gate_acc and up_acc.  LowerPipelineLoops replicates each
                    # loop body separately, making MemoryReuse more likely to
                    # leave the _l0_c bases of gate and up ununified with the
                    # outer acc bases — the condition that exposes #1352.
                    gate_acc = pl.matmul(x0, wg0, out_dtype=pl.FP32)
                    gate_acc = pl.matmul_acc(gate_acc, x1, wg1)
                    for kb in pl.pipeline(2, NUM_CHUNKS, stage=2):
                        k0 = kb * K_CHUNK
                        xk = pl.slice(x, [BATCH, K_CHUNK], [0, k0])
                        wgk = pl.slice(wg, [K_CHUNK, N], [k0, 0])
                        gate_acc = pl.matmul_acc(gate_acc, xk, wgk)

                    # --- match the exact Qwen3-32B consumption ordering ---
                    # In the model, gate_acc is first consumed by:
                    #   sigmoid = recip(add(exp(neg(gate_acc)), 1.0))
                    # which is a Vec operation that forces the compiler to
                    # emit a tpush for gate_acc BEFORE the up pipeline starts.
                    # Without this intermediate use, both tpushes would be
                    # scheduled after both pipelines, so gate_acc's tpop would
                    # read acc@0 = up_final (wrong).
                    #
                    # Here we approximate sigmoid with a no-op-equivalent
                    # pl.add(gate_acc, 0.0) that forces the tpush between the
                    # two pipelines.  gate_fp32 is a Vec FP32 tile that
                    # correctly captures gate_final at this point.
                    gate_fp32 = pl.add(gate_acc, 0.0)

                    # Up projection — independent pipeline loop, same shape.
                    # AIC overwrites acc@0 with up values here; gate_fp32 is
                    # already in Vec space so it is unaffected.
                    up_acc = pl.matmul(x0, wu0, out_dtype=pl.FP32)
                    up_acc = pl.matmul_acc(up_acc, x1, wu1)
                    for kb in pl.pipeline(2, NUM_CHUNKS, stage=2):
                        k0 = kb * K_CHUNK
                        xk = pl.slice(x, [BATCH, K_CHUNK], [0, k0])
                        wuk = pl.slice(wu, [K_CHUNK, N], [k0, 0])
                        up_acc = pl.matmul_acc(up_acc, xk, wuk)

                    # gate_fp32 × up_acc (FP32 tmul, supported on A2/A3).
                    # up_acc's tpush happens after the up pipeline, reading
                    # acc@0 = up_final ✓.
                    combined = pl.mul(gate_fp32, up_acc)
                    combined_bf16 = pl.cast(combined, pl.BF16)
                    out_tile = pl.assemble(out_tile, combined_bf16, [0, 0])

                out = pl.assemble(out, out_tile, [0, 0])
                return out

        return PipelineGateUpProgram

    def compute_expected(self, tensors, params=None):
        x = tensors["x"].to(torch.float32)
        wg = tensors["wg"].to(torch.float32)
        wu = tensors["wu"].to(torch.float32)
        gate = torch.matmul(x, wg)
        up = torch.matmul(x, wu)
        tensors["out"][:] = (gate * up).to(torch.bfloat16)


# =============================================================================
# pytest test functions
# =============================================================================


class TestPipelineMatmulAccOperations:
    """Regression suite for acc→acc pto.tmov elision in pipeline loops (#1352)."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_pipeline_matmul_acc_gate_up(self, test_runner, platform):
        """gate+up dual-accumulator matmul_acc with pl.pipeline(stage=2).

        This reproduces the exact gate_up_silu kernel shape from Qwen3-32B
        that triggered 'pto.tmov expects a supported tmov address-space pair'
        on Ascend 910B.  Without the fix in pto_ops_common.cpp the kernel
        fails to compile; with the fix it compiles and produces numerically
        correct results.
        """
        # BF16 output: product of two scaled matmul results (~K=512 accumulations).
        # Inputs are scaled (weights by 1/sqrt(K)) to keep output magnitude small,
        # matching the qwen3_32b_decode_scope3 initialization pattern.
        cfg = RunConfig(rtol=1e-3, atol=1e-3)
        result = test_runner.run(TestPipelineMatmulAccGateUp(platform=platform, config=cfg))
        assert result.passed, f"Test failed: {result.error}"
