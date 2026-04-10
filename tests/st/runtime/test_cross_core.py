# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Cross-Core Communication (TPUSH/TPOP) System Tests.

Tests:
  V2CTest : Vector→Cube, updown split. output = (a + b) @ (a - b)
  C2VTest : Cube→Vector, left-right split. c += a @ b (parallel over N in blocks)
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M = 32
K = 64
N = 512
N_BLOCK = 64
N_BLOCKS = N // N_BLOCK


@pl.program
class V2CProgram:
    """V2C updown-split cross-core program.

    Vector producer: loads tiles a and b, computes add and sub, pushes both to Cube.
    Cube consumer: pops tiles, performs matmul, stores result.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[32, 32], pl.FP32],
        b: pl.Tensor[[32, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)
        ):
            a_plus_b = pl.add(a, b)
            sub = pl.sub(a, b)
            out = pl.matmul(a_plus_b, sub)
            output = pl.assemble(output, out, [0, 0])
        return output


class V2CTest(PTOTestCase):
    """Cross-core V2C: output = (a + b) @ (a - b)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_tpush_tpop_v2c_updown_32x32"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [32, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return V2CProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].float()
        b = tensors["b"].float()
        tensors["output"][:] = torch.matmul(a + b, a - b)


@pl.program
class C2VProgram:
    """C2V left-right-split cross-core program.

    Cube producer: computes matmul in blocks over N, pushes results to Vector.
    Vector consumer: accumulates result into output tensor.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.auto_incore(split=pl.SplitMode.LEFT_RIGHT):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


class C2VTest(PTOTestCase):
    """Cross-core C2V: c += a @ b (parallel over N in blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_tpop_c2v_leftright"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return C2VProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        c_prev = tensors["c"].clone()
        tensors["c"][:] = c_prev + torch.matmul(a, b)


@pl.program
class BiDirectProgram:
    """Bidirectional (V→C→V) updown-split cross-core program.

    Vector sends data to Cube for matmul, Cube sends results back to Vector.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                a_add = pl.add(a, 1.0)
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a_add, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


class BiDirectTest(PTOTestCase):
    """Cross-core V->C->V: c += (a+1) @ b (parallel over N in blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_tpop_bidirect_updown"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return BiDirectProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        c_prev = tensors["c"].clone()
        tensors["c"][:] = c_prev + torch.matmul(a + 1, b)


class TestCrossCore:
    """Cross-core communication system tests."""

    def test_tpush_tpop_v2c_updown(self, test_runner):
        """V2C updown pipe: compile through full pipeline and verify kernel artifacts."""
        test_case = V2CTest()
        result = test_runner.run(test_case)
        assert result.passed, f"Cross-core V2C compilation failed: {result.error}"

    def test_tpop_c2v_leftright(self, test_runner):
        """C2V left-right pipe: compile through full pipeline and verify correctness."""
        test_case = C2VTest()
        result = test_runner.run(test_case)
        assert result.passed, f"Cross-core C2V compilation failed: {result.error}"

    def test_tpop_bidirect_updown(self, test_runner):
        """bidirect updown pipe: compile through full pipeline and verify correctness."""
        test_case = BiDirectTest()
        result = test_runner.run(test_case)
        assert result.passed, f"Cross-core bidirect compilation failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
