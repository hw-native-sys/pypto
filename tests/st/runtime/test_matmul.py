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
from examples.kernels.matmul import MatmulaccProgram
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec


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
                tile_b_l1 = pl.load(
                    b, offsets=[0, 0], shapes=[N, K], target_memory=pl.MemorySpace.Mat, transpose=True
                )
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
                tile_a_l1 = pl.load(
                    a, offsets=[0, 0], shapes=[K, M], target_memory=pl.MemorySpace.Mat, transpose=True
                )
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
                tile_a_l1 = pl.load(
                    a, offsets=[0, 0], shapes=[K, M], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_b_l1 = pl.load(
                    b, offsets=[0, 0], shapes=[N, K], target_memory=pl.MemorySpace.Mat, transpose=True
                )
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


class TestMatmulAcc(PTOTestCase):
    """Test matmul with accumulation (K-split into two chunks).

    Uses MatmulaccProgram which splits K=64 into two K=32 chunks:
    first chunk via pl.matmul, second via pl.matmul_acc.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "matmulacc_64x64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MatmulaccProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = torch.matmul(tensors["a"], tensors["b"])


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


# =============================================================================
# pytest test functions
# =============================================================================

_MATMUL_SHAPES = [(64, 64, 64), (128, 64, 128), (64, 128, 64)]
_TRANSPOSE_SHAPES = [(64, 64, 64), (128, 64, 128), (64, 128, 64), (32, 64, 32)]
# Shapes chosen so AutoTileMatmulL0 must K-split (FP32, double-buffered L0a/b
# = 32 KB effective): K=128 with N=128 exceeds L0b at k=128, forcing k=64 and
# splitting the K-loop in two.  K-iter count is kept at 2 to stay within the
# 1e-5 golden tolerance — deeper K (more iters) flakes due to FP32
# accumulation order vs. reference (torch.matmul).
_AUTOL0_SHAPES = [
    (64, 128, 128),
    (128, 128, 128),
    (128, 128, 64),
    (64, 128, 256),
]
# BF16 matmul mirroring qwen3_decode kv_proj/q_proj per-matmul shape
# (BATCH=16, K_CHUNK=128, OUT_CHUNK=256). Same 2-iter K-loop, BF16 inputs +
# FP32 accumulator.
_AUTOL0_BF16_SHAPES = [(16, 128, 256)]


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
    def test_matmulacc(self, test_runner, platform):
        """Test matmul with accumulation (K split into two chunks)."""
        result = test_runner.run(TestMatmulAcc(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _AUTOL0_SHAPES)
    def test_matmul_autol0(self, test_runner, platform, m, k, n):
        """Matmul on Mat-resident operands — exercises AutoTileMatmulL0 K-split."""
        result = test_runner.run(TestMatmulAutoL0(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _AUTOL0_BF16_SHAPES)
    def test_matmul_autol0_bf16(self, test_runner, platform, m, k, n):
        """BF16 matmul on Mat-resident operands — qwen3 kv_proj per-matmul shape."""
        result = test_runner.run(TestMatmulAutoL0BF16(m=m, k=k, n=n, platform=platform))
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
