# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for matrix multiplication operation using PyPTO frontend.

This test validates the matmul operation implementation through the
pto-testing-framework, ensuring correct code generation and execution.
Each test case accepts an optional ``platform`` parameter so a single
class can run on multiple platforms via ``@pytest.mark.parametrize``.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from examples.kernels.matmul import matmul_acc_64
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig


class TestMatmul(PTOTestCase):
    """Matmul: C = A @ B."""

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul(a, b, out_c)
                return out_c

        return MatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


class TestMatmulBTranspose(PTOTestCase):
    """Matmul with B transposed: C = A @ B^T.

    B is stored as [N, K] in memory and transposed during the load to L1.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_btranspose_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulBTransposeProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_btranspose(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b_nat = pl.load(b, offsets=[0, 0], shapes=[N, K], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.tile.transpose_view(tile_b_nat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul_btranspose(a, b, out_c)
                return out_c

        return MatmulBTransposeProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].to(torch.float32), tensors["b"].to(torch.float32).T)


class TestMatmulATranspose(PTOTestCase):
    """Matmul with A transposed: C = A^T @ B.

    A is stored as [K, M] in memory and transposed during the load to L1.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_atranspose_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.K, self.M], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulATransposeProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_atranspose(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_nat = pl.load(a, offsets=[0, 0], shapes=[K, M], target_memory=pl.MemorySpace.Mat)
                tile_a_l1 = pl.tile.transpose_view(tile_a_nat)
                tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul_atranspose(a, b, out_c)
                return out_c

        return MatmulATransposeProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].to(torch.float32).T, tensors["b"].to(torch.float32))


class TestMatmulABTranspose(PTOTestCase):
    """Matmul with both A and B transposed: C = A^T @ B^T.

    A is stored as [K, M] and B as [N, K] in memory, both transposed during load.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 64, n: int = 64, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_abtranspose_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.K, self.M], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulABTransposeProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_abtranspose(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a_nat = pl.load(a, offsets=[0, 0], shapes=[K, M], target_memory=pl.MemorySpace.Mat)
                tile_a_l1 = pl.tile.transpose_view(tile_a_nat)
                tile_b_nat = pl.load(b, offsets=[0, 0], shapes=[N, K], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.tile.transpose_view(tile_b_nat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
                out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul_abtranspose(a, b, out_c)
                return out_c

        return MatmulABTransposeProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].to(torch.float32).T, tensors["b"].to(torch.float32).T)


class TestMatmulAutoL0(PTOTestCase):
    """Matmul on Mat-resident tiles — AutoTileMatmulL0 inserts L0 splits.

    Unlike ``TestMatmul`` (which moves to Left/Right explicitly and gives the
    pass nothing to do), this case calls ``pl.matmul`` on L1 tiles, mirroring
    the pattern used in models such as qwen3_decode.  K is sized so the
    chooser must split: with FP32 + double-buffered L0a/L0b on 910B
    (effective 32 KB each), K=128 forces k=64 and a 2-iter K-loop.
    """

    __test__ = False

    def __init__(self, m: int = 64, k: int = 128, n: int = 128, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_autol0_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulAutoL0Program:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_c = pl.matmul(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul(a, b, out_c)
                return out_c

        return MatmulAutoL0Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


class TestMatmulAutoL0BF16(PTOTestCase):
    """BF16 matmul on Mat-resident tiles with FP32 accumulator.

    Mirrors the per-matmul shape in qwen3_decode kv_proj/q_proj
    (M=BATCH=16, K=K_CHUNK=128, N=OUT_CHUNK=256), where AutoTileMatmulL0 is
    expected to K-split (k=64, 2 iterations) because K*N exceeds L0b/2.
    """

    __test__ = False

    def __init__(self, m: int = 16, k: int = 128, n: int = 256, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"matmul_autol0_bf16_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.BF16, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.BF16, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MatmulAutoL0BF16Program:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                b: pl.Tensor[[K, N], pl.BF16],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, offsets=[0, 0], shapes=[K, N], target_memory=pl.MemorySpace.Mat)
                tile_c = pl.matmul(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                b: pl.Tensor[[K, N], pl.BF16],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.matmul(a, b, out_c)
                return out_c

        return MatmulAutoL0BF16Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].to(torch.float32), tensors["b"].to(torch.float32))


class TestMatmulOuterPipelinedBF16(PTOTestCase):
    """BF16 matmul mirroring qwen3 kv_proj's outer-pl.pipeline + if/else pattern.

    A single matmul call gets wrapped in a hand-coded ``pl.pipeline(stage=2)``
    over K_total/K_chunk chunks: ``kb == 0`` does ``pl.matmul`` (init), all
    later iterations do ``pl.matmul_acc``.  AutoTileMatmulL0 then K-tiles the
    inner per-chunk matmul into a 2-iter loop, producing the same nested
    pipeline shape as kv_proj (outer stage=2 around inner stage=2 with
    if/else in between).
    """

    __test__ = False

    def __init__(
        self,
        m: int = 16,
        k_chunk: int = 128,
        n: int = 256,
        num_chunks: int = 8,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self.M = m
        self.K_CHUNK = k_chunk
        self.N = n
        self.NUM_CHUNKS = num_chunks
        self.K = k_chunk * num_chunks

    def get_name(self) -> str:
        return f"matmul_outer_pipe_bf16_{self.M}x{self.K}x{self.N}_chunks{self.NUM_CHUNKS}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.BF16, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.BF16, init_value=torch.randn),
            TensorSpec("c", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N, K_CHUNK, NUM_CHUNKS = self.M, self.K, self.N, self.K_CHUNK, self.NUM_CHUNKS

        @pl.program
        class OuterPipeBF16Program:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                b: pl.Tensor[[K, N], pl.BF16],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="outer_pipe_matmul"):
                    acc = pl.create_tensor([M, N], dtype=pl.FP32)
                    for kb in pl.pipeline(NUM_CHUNKS, stage=2):
                        k0 = kb * K_CHUNK
                        tile_a = a[:, k0 : k0 + K_CHUNK]
                        tile_b = b[k0 : k0 + K_CHUNK, 0:N]
                        if kb == 0:
                            acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
                        else:
                            acc = pl.matmul_acc(acc, tile_a, tile_b)
                    out_c = pl.assemble(out_c, acc, [0, 0])
                return out_c

        return OuterPipeBF16Program

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"].to(torch.float32), tensors["b"].to(torch.float32))


# ---------------------------------------------------------------------------
# Dual-accumulator gate/up pipeline matmul_acc (#1352 regression)
# Kernel dimensions match gate_up_silu in qwen3_32b_decode_scope3.py.
# ---------------------------------------------------------------------------
_GATE_UP_BATCH_TILE = 16
_GATE_UP_K_CHUNK = 128
_GATE_UP_MLP_OUT_CHUNK = 256
_GATE_UP_NUM_CHUNKS = 4  # total K chunks; prolog uses 2, pipeline(2,4,stage=2) does 2


class TestPipelineMatmulAccGateUp(PTOTestCase):
    """Dual-accumulator gate/up matmul via matmul_acc + pl.pipeline(stage=2).

    Runtime regression for #1352: acc→acc pto.tmov in pipeline matmul_acc.

    Reproduces the gate_up_silu pattern from the Qwen3-32B decode kernel that
    triggered the bug: two independent accumulators (gate_acc, up_acc) are built
    inside a ``pl.at(CORE_GROUP, optimizations=[pl.split(UP_DOWN)])`` scope, each following the
    prolog-then-pipeline pattern::

        gate_acc = pl.matmul(x_chunk_0, wg_0, out_dtype=pl.FP32)
        gate_acc = pl.matmul_acc(gate_acc, x_chunk_1, wg_1)
        for kb in pl.pipeline(2, NUM_CHUNKS, stage=2):
            gate_acc = pl.matmul_acc(gate_acc, x_chunk_k, wg_k)

    Computes::

        gate_result[B, N] = x @ wg       (BF16 inputs, FP32 accum)
        up_result[B, N]   = x @ wu       (BF16 inputs, FP32 accum)
        out[B, N]         = (gate_result * up_result), cast to BF16

    Three conditions combine to expose the bug:

    1. **Mat-resident inputs** — ``pl.slice`` passes Mat-space tiles to
       ``pl.matmul`` / ``pl.matmul_acc``, so AutoTileMatmulL0 sees them and
       inserts an inner K-loop (K_CHUNK=128, N=256 → effective L0B = 32 KB <
       128×256×2 B = 64 KB, so ChooseL0Tile picks K_L0=64).

    2. **pl.pipeline(stage=2)** — LowerPipelineLoops replicates the loop body,
       compounding the IterArg nesting around the ``_l0_c`` accumulator variable
       introduced by AutoTileMatmulL0.

    3. **CORE_GROUP + split=UP_DOWN** — the AIC/AIV split causes MemoryReuse to
       run only on the AIC function.  gate is consumed (to Vec) before the up
       pipeline, so up *reuses* gate's freed Acc buffer.  Before the fix,
       MemoryReuse's greedy pass retyped up's producer/init vars onto gate's
       buffer but left up's loop-carried iter_arg/return_var on its own buffer,
       splitting the chain; YieldFixupMutator then bridged the two Acc buffers
       with ``tile.move acc→acc`` ops that ptoas rejects on Ascend 910B with
       ``'pto.tmov' op expects a supported tmov address-space pair for this
       target``.  The fix (``AlignLoopCarriesToInitMutator`` in
       ``src/ir/transforms/memory_reuse_pass.cpp``) re-aligns loop-carried
       MemRefs to their reused init top-down, so the whole up chain lands on the
       single reused Acc buffer and no acc→acc move is ever inserted.

    Note: only ``split=UP_DOWN`` is used (no ``auto_chunk``).  ``split=UP_DOWN``
    alone is what forces the AIC/AIV split that triggers MemoryReuse on the AIC
    function — the root condition for #1352.  ``auto_chunk`` is deliberately
    omitted to avoid it incorrectly distributing K-pipeline iterations across
    cores when no ``pl.parallel`` outer loop is present.

    Usage note on sequential consumption
    -------------------------------------
    Because MemoryReuse legitimately reuses *one* physical Acc buffer for both
    gate_acc and up_acc (gate is dead by the time up runs), the final add must
    read each accumulator **at the point where that buffer holds its correct
    value**:

    - After the gate pipeline completes, the buffer = gate result.
      → cast gate_acc to BF16 (Vec space) **before** the up pipeline starts.
    - After the up pipeline completes, the buffer = up result.
      → read up_acc **after** the up pipeline.

    This is exactly the access order used in the Qwen3-32B gate_up_silu kernel:
    gate is consumed (cast + silu) between the two pipeline loops, so by the time
    the up pipeline overwrites the shared buffer the gate value has already been
    saved.
    """

    __test__ = False

    def __init__(
        self,
        batch: int = _GATE_UP_BATCH_TILE,
        k_chunk: int = _GATE_UP_K_CHUNK,
        n: int = _GATE_UP_MLP_OUT_CHUNK,
        num_chunks: int = _GATE_UP_NUM_CHUNKS,
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
                    # read the shared Acc buffer = up_final (wrong).
                    #
                    # Here we approximate sigmoid with a no-op-equivalent
                    # pl.add(gate_acc, 0.0) that forces the tpush between the
                    # two pipelines.  gate_fp32 is a Vec FP32 tile that
                    # correctly captures gate_final at this point.
                    gate_fp32 = pl.add(gate_acc, 0.0)

                    # Up projection — independent pipeline loop, same shape.
                    # AIC overwrites the shared Acc buffer with up values here;
                    # gate_fp32 is already in Vec space so it is unaffected.
                    up_acc = pl.matmul(x0, wu0, out_dtype=pl.FP32)
                    up_acc = pl.matmul_acc(up_acc, x1, wu1)
                    for kb in pl.pipeline(2, NUM_CHUNKS, stage=2):
                        k0 = kb * K_CHUNK
                        xk = pl.slice(x, [BATCH, K_CHUNK], [0, k0])
                        wuk = pl.slice(wu, [K_CHUNK, N], [k0, 0])
                        up_acc = pl.matmul_acc(up_acc, xk, wuk)

                    # gate_fp32 × up_acc (FP32 tmul, supported on A2/A3).
                    # up_acc's tpush happens after the up pipeline, reading
                    # the shared Acc buffer = up_final ✓.
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

_MATMUL_SHAPES = [(64, 64, 64), (128, 64, 128), (64, 128, 64)]
_TRANSPOSE_SHAPES = [(64, 64, 64), (128, 64, 128), (64, 128, 64), (32, 64, 32)]
# Shapes chosen so AutoTileMatmulL0 must K-split (FP32, double-buffered L0a/b
# = 32 KB effective): K=128 with N=128 exceeds L0b at k=128, forcing k=64 and
# splitting the K-loop in two.  K-split accumulates in a different order than
# the torch.matmul reference, so per-element FP32 rounding can drift by up to
# ~K * eps_fp32 (≈1.5e-5 at K=128).  Golden tolerance is loosened to 1e-4 for
# these shapes — see _AUTOL0_RTOL / _AUTOL0_ATOL below.
_AUTOL0_SHAPES = [
    (64, 128, 128),
    (128, 128, 128),
    (128, 128, 64),
    (64, 128, 256),
]
# Tolerance for AutoL0 K-split: HW reduces K=64 chunks in a different order
# than torch's BLAS reference, so the strict 1e-5 default is too tight.
_AUTOL0_RTOL = 1e-4
_AUTOL0_ATOL = 1e-5
# BF16 matmul mirroring qwen3_decode kv_proj/q_proj per-matmul shape
# (BATCH=16, K_CHUNK=128, OUT_CHUNK=256). Same 2-iter K-loop, BF16 inputs +
# FP32 accumulator.
_AUTOL0_BF16_SHAPES = [(16, 128, 256)]


class TestSharedKVMatmul(PTOTestCase):
    """``qk = q @ kv^T`` and ``pv = p @ kv`` where a NON-SQUARE ``kv`` [N, K] is
    one sliced, shared operand feeding both b_trans=True and b_trans=False.

    A single sliced ``kv`` is consumed by both a b_trans=True and a b_trans=False
    matmul (issue #1776). The compiler must emit ONE GM->L1 load and reinterpret
    it for the transposed use via a zero-copy ``tile.transpose_view`` view (NZ<->ZN)
    aliasing the same L1 buffer. Because ``kv`` is non-square, ``qk`` and ``pv``
    have different shapes and cannot be summed, so each is a separate output.
    """

    __test__ = False

    def __init__(self, m: int = 16, k: int = 64, n: int = 128, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n  # non-square kv [N, K]

    def get_name(self) -> str:
        return f"shared_kv_matmul_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        M, K, N = self.M, self.K, self.N
        return [
            TensorSpec("q", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("p", [M, N], DataType.FP32, init_value=torch.randn),
            # Sliced down to [N, K] so the load is a real (partial) tensor.slice.
            TensorSpec("kv_src", [2 * N, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c_qk", [M, N], DataType.FP32, is_output=True),
            TensorSpec("c_pv", [M, K], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N
        NN = 2 * N

        @pl.program
        class SharedKVProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def shared_kv(
                self,
                q: pl.Tensor[[M, K], pl.FP32],
                p: pl.Tensor[[M, N], pl.FP32],
                kv_src: pl.Tensor[[NN, K], pl.FP32],
                c_qk: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                c_pv: pl.Out[pl.Tensor[[M, K], pl.FP32]],
            ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, K], pl.FP32]]:
                # ONE sliced NON-SQUARE KV consumed by both matmuls -> ONE GM->L1 load.
                kv = kv_src[0:N, 0:K]
                # b_trans=True reads kv transposed via a zero-copy tile.transpose_view view.
                qk = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)  # [M, N]
                # b_trans=False reads the same buffer in its natural orientation.
                pv = pl.matmul(p, kv, out_dtype=pl.FP32)  # [M, K]
                c_qk = pl.assemble(c_qk, qk, offset=[0, 0])
                c_pv = pl.assemble(c_pv, pv, offset=[0, 0])
                return c_qk, c_pv

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                q: pl.Tensor[[M, K], pl.FP32],
                p: pl.Tensor[[M, N], pl.FP32],
                kv_src: pl.Tensor[[NN, K], pl.FP32],
                out_qk: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                out_pv: pl.Out[pl.Tensor[[M, K], pl.FP32]],
            ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, K], pl.FP32]]:
                out_qk, out_pv = self.shared_kv(q, p, kv_src, out_qk, out_pv)
                return out_qk, out_pv

        return SharedKVProgram

    def compute_expected(self, tensors, params=None):
        q = tensors["q"].to(torch.float32)
        p = tensors["p"].to(torch.float32)
        kv = tensors["kv_src"][: self.N, : self.K].to(torch.float32)
        tensors["c_qk"][:] = torch.matmul(q, kv.T)
        tensors["c_pv"][:] = torch.matmul(p, kv)


class TestATransMatmul(PTOTestCase):
    """``c = a^T @ b`` via a 2D ``a_trans=True`` matmul (issue #1776).

    The LHS/Left cube operand loads in its NATURAL (NZ) orientation and is
    reinterpreted by a zero-copy ``tile.transpose_view`` view (NZ<->ZN), exactly
    mirroring the b_trans path but on the A operand. Only the b_trans view path
    had numerical coverage; this validates that the A-operand NZ<->ZN duality is
    numerically equivalent on real hardware.
    """

    __test__ = False

    def __init__(self, m: int = 16, k: int = 64, n: int = 128, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"a_trans_matmul_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        M, K, N = self.M, self.K, self.N
        return [
            TensorSpec("a", [K, M], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class ATransProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def a_trans_mm(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                # a_trans=True: a loads natural (NZ) and is reinterpreted as its
                # transpose via a zero-copy tile.transpose_view view (NZ<->ZN).
                cm = pl.matmul(a, b, a_trans=True, out_dtype=pl.FP32)  # a^T @ b -> [M, N]
                out_c = pl.assemble(c, cm, offset=[0, 0])
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[K, M], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.a_trans_mm(a, b, c)
                return out_c

        return ATransProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["c"][:] = torch.matmul(a.T, b)


class TestMixedAddBTrans(PTOTestCase):
    """Mixed-kernel probe: a Vec compute result feeds a ``b_trans=True`` 2D matmul.

    ``bt = b0 + b1`` is a vector (Vec/UB) op result; ``c = a @ bt^T``. The
    transposed operand originates from a compute op, NOT a load, so the
    transpose cannot ride a transposed load. This exercises whether the 2D
    ``tile.transpose_view`` + Vec->Mat ``tile.move`` path transposes a
    compute-sourced operand correctly. Probe for the ND batch_matmul migration.
    """

    __test__ = False

    def __init__(self, m: int = 16, k: int = 64, n: int = 128, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"mixed_add_btrans_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        M, K, N = self.M, self.K, self.N
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b0", [N, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b1", [N, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class MixedAddBTransProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def mixed_add_btrans(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b0: pl.Tensor[[N, K], pl.FP32],
                b1: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                bt = pl.add(b0, b1)  # Vec compute -> [N, K]
                cm = pl.matmul(a, bt, b_trans=True, out_dtype=pl.FP32)  # a @ bt^T -> [M, N]
                out_c = pl.assemble(c, cm, offset=[0, 0])
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b0: pl.Tensor[[N, K], pl.FP32],
                b1: pl.Tensor[[N, K], pl.FP32],
                out_c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out_c = self.mixed_add_btrans(a, b0, b1, out_c)
                return out_c

        return MixedAddBTransProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        bt = tensors["b0"].to(torch.float32) + tensors["b1"].to(torch.float32)
        tensors["c"][:] = torch.matmul(a, bt.T)


class TestMatmulOperations:
    """Test suite for matrix multiplication (matmul) operations."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _MATMUL_SHAPES)
    def test_matmul(self, test_runner, platform, m, k, n):
        """Test matmul with configurable matrix dimensions."""
        result = test_runner.run(TestMatmul(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _TRANSPOSE_SHAPES)
    def test_matmul_btranspose(self, test_runner, platform, m, k, n):
        """Test matmul with B transposed (C = A @ B^T)."""
        result = test_runner.run(TestMatmulBTranspose(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _TRANSPOSE_SHAPES)
    def test_matmul_atranspose(self, test_runner, platform, m, k, n):
        """Test matmul with A transposed (C = A^T @ B)."""
        result = test_runner.run(TestMatmulATranspose(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _TRANSPOSE_SHAPES)
    def test_matmul_abtranspose(self, test_runner, platform, m, k, n):
        """Test matmul with both A and B transposed (C = A^T @ B^T)."""
        result = test_runner.run(TestMatmulABTranspose(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", [(16, 64, 128)])
    def test_matmul_mixed_add_btranspose(self, test_runner, platform, m, k, n):
        """Mixed-kernel: a Vec compute (add) result feeds a b_trans=True 2D matmul."""
        result = test_runner.run(TestMixedAddBTrans(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    def test_matmulacc(self, test_config):
        """Test matmul_acc_64 (@pl.jit): K=64 split into two K=32 chunks."""
        matmul_acc_64._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(64, 64, dtype=torch.float32)
        b = torch.randn(64, 64, dtype=torch.float32)
        c = torch.zeros((64, 64), dtype=torch.float32)
        matmul_acc_64(a, b, c, config=test_config)
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3), (
            f"matmul_acc_64 failed: max diff = {(c - expected).abs().max().item()}"
        )

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _AUTOL0_SHAPES)
    def test_matmul_autol0(self, test_runner, platform, m, k, n):
        """Matmul on Mat-resident operands — exercises AutoTileMatmulL0 K-split."""
        cfg = RunConfig(platform=platform, rtol=_AUTOL0_RTOL, atol=_AUTOL0_ATOL)
        result = test_runner.run(TestMatmulAutoL0(m=m, k=k, n=n, platform=platform, config=cfg))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _AUTOL0_BF16_SHAPES)
    def test_matmul_autol0_bf16(self, test_runner, platform, m, k, n):
        """BF16 matmul on Mat-resident operands — qwen3 kv_proj per-matmul shape."""
        cfg = RunConfig(platform=platform, rtol=_AUTOL0_RTOL, atol=_AUTOL0_ATOL)
        result = test_runner.run(TestMatmulAutoL0BF16(m=m, k=k, n=n, platform=platform, config=cfg))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(
        reason="Reproducer for the qwen3_decode runtime hang: outer "
        "pl.pipeline(stage=2) + if/else matmul/matmul_acc, with "
        "AutoTileMatmulL0 K-tiling inside, hangs at runtime on a2a3. "
        "PTO output is structurally correct; suspect ptoas (simpler) "
        "synchronization codegen for nested branched pipelines. See "
        "KNOWN_ISSUES.md."
    )
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_matmul_outer_pipelined_bf16(self, test_runner, platform):
        """qwen3 kv_proj-shaped pattern: outer pl.pipeline(stage=2) wrapping
        if/else matmul/matmul_acc with AutoTileMatmulL0 K-tiling inside."""
        result = test_runner.run(TestMatmulOuterPipelinedBF16(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_pipeline_matmul_acc_gate_up(self, test_runner, platform):
        """gate+up dual-accumulator matmul_acc with pl.pipeline(stage=2) (#1352).

        This reproduces the exact gate_up_silu kernel shape from Qwen3-32B
        that triggered 'pto.tmov expects a supported tmov address-space pair'
        on Ascend 910B.  Without the MemoryReuse fix (AlignLoopCarriesToInit)
        the kernel fails to compile; with the fix the up accumulator's chain
        lands on a single reused Acc buffer, so no acc→acc move is emitted and
        the kernel compiles and produces numerically correct results.
        """
        # BF16 output: product of two scaled matmul results (~K=512 accumulations).
        # Inputs are scaled (weights by 1/sqrt(K)) to keep output magnitude small,
        # matching the qwen3_32b_decode_scope3 initialization pattern.
        cfg = RunConfig(rtol=1e-3, atol=1e-3)
        result = test_runner.run(TestPipelineMatmulAccGateUp(platform=platform, config=cfg))
        assert result.passed, f"Test failed: {result.error}"

    # --- Shared-KV / NZ<->ZN b_trans view test (#1776) -------------------------

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", [(16, 64, 128), (16, 128, 64)])
    def test_shared_kv_matmul(self, test_runner, platform, m, k, n):
        """One GM->L1 load + zero-copy NZ<->ZN view feeding QK and PV over a
        NON-SQUARE kv [N, K]; both outputs must match.

        FP32 cube matmul vs torch fp32 golden differs by accumulation-order rounding,
        so use a realistic tolerance rather than bit-exact.
        """
        cfg = RunConfig(platform=platform, rtol=1e-3, atol=1e-3)
        result = test_runner.run(TestSharedKVMatmul(m=m, k=k, n=n, platform=platform, config=cfg))
        assert result.passed, f"Test failed: {result.error}"

    # --- a_trans NZ<->ZN view test (#1776) -------------------------------------

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", [(16, 64, 128), (16, 128, 64)])
    def test_a_trans_matmul(self, test_runner, platform, m, k, n):
        """2D a_trans=True matmul: the LHS loads natural and is reinterpreted by a
        zero-copy NZ<->ZN view on the Left cube operand, mirroring the b_trans
        view but on the A side (#1776). Validates a_trans view numerics."""
        cfg = RunConfig(platform=platform, rtol=1e-3, atol=1e-3)
        result = test_runner.run(TestATransMatmul(m=m, k=k, n=n, platform=platform, config=cfg))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
