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
  V2CUDTest      : Vector→Cube, updown split.      output = (a + b) @ (a - b)
  V2CLRTest      : Vector→Cube, left-right split.  output = (a + b) @ (a - b)
  V2CNoSplitTest : Vector→Cube, no split.          output = (a + b) @ (a - b)
  C2VLRTest      : Cube→Vector, left-right split.  c += a @ b (parallel over N in blocks)
  C2VUDTest      : Cube→Vector, updown split.      c += a @ b (parallel over N in blocks)
  C2VNoSplitTest : Cube→Vector, no split.          c += a @ b (parallel over N in blocks)
  BiDirectUDTest : V↔C, updown split.              c += (a+1) @ b (parallel over N in blocks)
  BiDirectLRTest : V↔C, left-right split.          c += (a+1) @ b (parallel over N in blocks)
  BiDirectNoSplitTest : V↔C, no split.             c += (a+1) @ b (parallel over N in blocks)
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
class V2CUDProgram:
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


class V2CUDTest(PTOTestCase):
    """Cross-core V2C updown: output = (a + b) @ (a - b)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_v2c_updown"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [32, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return V2CUDProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].float()
        b = tensors["b"].float()
        tensors["output"][:] = torch.matmul(a + b, a - b)


@pl.program
class V2CLRProgram:
    """V2C left-right-split cross-core program.

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
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT)
        ):
            a_plus_b = pl.add(a, b)
            sub = pl.sub(a, b)
            out = pl.matmul(a_plus_b, sub)
            output = pl.assemble(output, out, [0, 0])
        return output


class V2CLRTest(PTOTestCase):
    """Cross-core V2C left-right: output = (a + b) @ (a - b)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_v2c_leftright"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [32, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return V2CLRProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].float()
        b = tensors["b"].float()
        tensors["output"][:] = torch.matmul(a + b, a - b)


@pl.program
class V2CNoSplitProgram:
    """V2C no-split cross-core program.

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
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.NONE),
        ):
            a_plus_b = pl.add(a, b)
            sub = pl.sub(a, b)
            out = pl.matmul(a_plus_b, sub)
            output = pl.assemble(output, out, [0, 0])
        return output


class V2CNoSplitTest(PTOTestCase):
    """Cross-core V2C no-split: output = (a + b) @ (a - b)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_v2c_nosplit"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [32, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [32, 32], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [32, 32], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return V2CNoSplitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].float()
        b = tensors["b"].float()
        tensors["output"][:] = torch.matmul(a + b, a - b)


@pl.program
class C2VLRProgram:
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
        with pl.at(
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT)
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


class C2VLRTest(PTOTestCase):
    """Cross-core C2V left-right: c += a @ b (parallel over N in blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_c2v_leftright"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return C2VLRProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        c_prev = tensors["c"].clone()
        tensors["c"][:] = c_prev + torch.matmul(a, b)


@pl.program
class C2VUDProgram:
    """C2V updown-split cross-core program.

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
        with pl.at(
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


class C2VUDTest(PTOTestCase):
    """Cross-core C2V updown: c += a @ b (parallel over N in blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_c2v_updown"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return C2VUDProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        c_prev = tensors["c"].clone()
        tensors["c"][:] = c_prev + torch.matmul(a, b)


@pl.program
class C2VNoSplitProgram:
    """C2V no-split cross-core program.

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
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.NONE),
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


class C2VNoSplitTest(PTOTestCase):
    """Cross-core C2V no-split: c += a @ b (parallel over N in blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_c2v_nosplit"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return C2VNoSplitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        c_prev = tensors["c"].clone()
        tensors["c"][:] = c_prev + torch.matmul(a, b)


@pl.program
class BiDirectUDProgram:
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
        with pl.at(
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                a_add = pl.add(a, 1.0)
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a_add, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


class BiDirectUDTest(PTOTestCase):
    """Cross-core V->C->V updown: c += (a+1) @ b (parallel over N in blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_bidirect_updown"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return BiDirectUDProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        c_prev = tensors["c"].clone()
        tensors["c"][:] = c_prev + torch.matmul(a + 1, b)


@pl.program
class BiDirectLRProgram:
    """Bidirectional (V→C→V) left-right-split cross-core program.

    Vector sends data to Cube for matmul, Cube sends results back to Vector.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT)
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                a_add = pl.add(a, 1.0)
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a_add, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


class BiDirectLRTest(PTOTestCase):
    """Cross-core V->C->V left-right: c += (a+1) @ b (parallel over N in blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_bidirect_leftright"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return BiDirectLRProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        c_prev = tensors["c"].clone()
        tensors["c"][:] = c_prev + torch.matmul(a + 1, b)


@pl.program
class BiDirectNoSplitProgram:
    """Bidirectional (V→C→V) no-split cross-core program.

    Vector sends data to Cube for matmul, Cube sends results back to Vector.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.NONE),
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                a_add = pl.add(a, 1.0)
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a_add, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


class BiDirectNoSplitTest(PTOTestCase):
    """Cross-core V->C->V no-split: c += (a+1) @ b (parallel over N in blocks)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_bidirect_nosplit"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return BiDirectNoSplitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        c_prev = tensors["c"].clone()
        tensors["c"][:] = c_prev + torch.matmul(a + 1, b)


class TestCrossCore:
    """Cross-core communication system tests.

    The 9 cases below intentionally do not parametrize over ``platform``.
    They exercise the cross-core compile/runtime pipeline using the default
    backend (Ascend910B) and rely on the session-wide ``--platform`` value
    chosen by CI to pick the execution target (a2a3, a2a3sim or a5sim).
    """

    def test_tpush_tpop_v2c_updown(self, test_runner):
        """V2C updown pipe: compile through full pipeline and verify kernel artifacts."""
        result = test_runner.run(V2CUDTest())
        assert result.passed, f"Cross-core V2C updown compilation failed: {result.error}"

    def test_tpush_tpop_v2c_leftright(self, test_runner):
        """V2C left-right pipe: compile through full pipeline and verify kernel artifacts."""
        result = test_runner.run(V2CLRTest())
        assert result.passed, f"Cross-core V2C left-right compilation failed: {result.error}"

    def test_tpush_tpop_v2c_nosplit(self, test_runner):
        """V2C no-split pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(V2CNoSplitTest())
        assert result.passed, f"Cross-core V2C no-split compilation failed: {result.error}"

    def test_tpop_c2v_leftright(self, test_runner):
        """C2V left-right pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(C2VLRTest())
        assert result.passed, f"Cross-core C2V left-right compilation failed: {result.error}"

    def test_tpop_c2v_updown(self, test_runner):
        """C2V updown pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(C2VUDTest())
        assert result.passed, f"Cross-core C2V updown compilation failed: {result.error}"

    def test_tpop_c2v_nosplit(self, test_runner):
        """C2V no-split pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(C2VNoSplitTest())
        assert result.passed, f"Cross-core C2V no-split compilation failed: {result.error}"

    def test_tpop_bidirect_updown(self, test_runner):
        """Bidirect updown pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(BiDirectUDTest())
        assert result.passed, f"Cross-core bidirect updown compilation failed: {result.error}"

    def test_tpop_bidirect_leftright(self, test_runner):
        """Bidirect left-right pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(BiDirectLRTest())
        assert result.passed, f"Cross-core bidirect left-right compilation failed: {result.error}"

    def test_tpop_bidirect_nosplit(self, test_runner):
        """Bidirect no-split pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(BiDirectNoSplitTest())
        assert result.passed, f"Cross-core bidirect no-split compilation failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
