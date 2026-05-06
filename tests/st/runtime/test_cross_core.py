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
  MultiPipeNoSplitTest : two explicit V→C pipes with ids 0 and 1.
"""

import sys
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType

M = 32
K = 64
N = 512
N_BLOCK = 64
N_BLOCKS = N // N_BLOCK
MULTI_PIPE_DIM = 16
MULTI_PIPE_SLOT_SIZE = MULTI_PIPE_DIM * MULTI_PIPE_DIM * 4
MULTI_PIPE_BUFFER_SIZE = MULTI_PIPE_SLOT_SIZE * 8

_PLATFORM_TO_BACKEND: dict[str, BackendType] = {
    "a2a3": BackendType.Ascend910B,
    "a2a3sim": BackendType.Ascend910B,
    "a5": BackendType.Ascend950,
    "a5sim": BackendType.Ascend950,
}
_DEFAULT_PLATFORM = "a2a3"


def _resolve_platform(config: pytest.Config) -> str:
    """Resolve the effective platform from the session-wide allowlist."""
    raw_platform = str(config.getoption("--platform") or "")
    tokens = [tok.strip() for tok in raw_platform.split(",") if tok.strip()]
    valid_platforms = tuple(dict.fromkeys(tok for tok in tokens if tok in _PLATFORM_TO_BACKEND))
    if tokens and not valid_platforms:
        raise pytest.UsageError(
            "tests/st/runtime/test_cross_core.py supports --platform values (a2a3, a2a3sim, a5, or a5sim)"
        )
    return valid_platforms[0] if valid_platforms else _DEFAULT_PLATFORM


def _resolve_backend_type(config: pytest.Config) -> BackendType:
    """Resolve backend from the selected platform, defaulting to a2a3."""
    platform = _resolve_platform(config)
    try:
        return _PLATFORM_TO_BACKEND[platform]
    except KeyError as exc:
        raise pytest.UsageError(
            f"Unsupported --platform {platform!r} for tests/st/runtime/test_cross_core.py"
        ) from exc


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Drive backend selection from the session-wide --platform filter."""
    if "backend_type" not in metafunc.fixturenames and "platform" not in metafunc.fixturenames:
        return

    platform = _resolve_platform(metafunc.config)
    backend_type = _resolve_backend_type(metafunc.config)
    if "backend_type" in metafunc.fixturenames:
        metafunc.parametrize("backend_type", [backend_type], ids=[platform])
    if "platform" in metafunc.fixturenames:
        metafunc.parametrize("platform", [platform])


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


@pl.program
class MultiPipeNoSplitProgram:
    """Explicit two-pipe no-split V2C program."""

    @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
    def vector_multi_pipe(
        self,
        a: pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32],
        b: pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32],
        output: pl.Out[pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32]],
    ):
        v2c_peer_0 = pl.import_peer_buffer(name="v2c_slot_buffer_0", peer_func="cube_multi_pipe")
        v2c_peer_1 = pl.import_peer_buffer(name="v2c_slot_buffer_1", peer_func="cube_multi_pipe")
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_peer_0,
            dir_mask=2,
            slot_size=MULTI_PIPE_SLOT_SIZE,
            id=0,
        )
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_peer_1,
            dir_mask=2,
            slot_size=MULTI_PIPE_SLOT_SIZE,
            id=1,
        )

        a_tile: pl.Tile[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32, pl.Mem.Vec] = pl.load(
            a,
            [0, 0],
            [MULTI_PIPE_DIM, MULTI_PIPE_DIM],
        )
        b_tile: pl.Tile[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32, pl.Mem.Vec] = pl.load(
            b,
            [0, 0],
            [MULTI_PIPE_DIM, MULTI_PIPE_DIM],
        )
        sum_tile: pl.Tile[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32, pl.Mem.Vec] = pl.add(a_tile, b_tile)
        diff_tile: pl.Tile[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32, pl.Mem.Vec] = pl.sub(a_tile, b_tile)
        # Push in reverse id order so the test fails if ids collapse onto one FIFO.
        pl.tpush_to_aic(diff_tile, split=0, id=1)
        pl.tpush_to_aic(sum_tile, split=0, id=0)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_multi_pipe(
        self,
        a: pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32],
        b: pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32],
        output: pl.Out[pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32]],
    ) -> pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32]:
        v2c_buf_0 = pl.reserve_buffer(
            name="v2c_slot_buffer_0",
            size=MULTI_PIPE_BUFFER_SIZE,
            base=pl.AUTO,
        )
        v2c_buf_1 = pl.reserve_buffer(
            name="v2c_slot_buffer_1",
            size=MULTI_PIPE_BUFFER_SIZE,
            base=pl.AUTO,
        )
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_buf_0,
            dir_mask=2,
            slot_size=MULTI_PIPE_SLOT_SIZE,
            id=0,
        )
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_buf_1,
            dir_mask=2,
            slot_size=MULTI_PIPE_SLOT_SIZE,
            id=1,
        )
        sum_tile: pl.Tile[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32, pl.Mem.Mat] = pl.tpop_from_aiv(
            split=0,
            id=0,
        )
        diff_tile: pl.Tile[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32, pl.Mem.Mat] = pl.tpop_from_aiv(
            split=0,
            id=1,
        )
        sum_left = pl.move(sum_tile, target_memory=pl.Mem.Left)
        diff_right = pl.move(diff_tile, target_memory=pl.Mem.Right)
        result: pl.Tile[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32] = pl.matmul(sum_left, diff_right)
        pl.tfree_to_aiv(sum_tile, id=0)
        pl.tfree_to_aiv(diff_tile, id=1)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32],
        b: pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32],
        output: pl.Out[pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32]],
    ) -> pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32]:
        result = self.cube_multi_pipe(a, b, output)
        self.vector_multi_pipe(a, b, output)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32],
        b: pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32],
        output: pl.Out[pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32]],
    ) -> pl.Tensor[[MULTI_PIPE_DIM, MULTI_PIPE_DIM], pl.FP32]:
        result = self.group_func(a, b, output)
        return result


class MultiPipeNoSplitTest(PTOTestCase):
    """Explicit two-pipe V->C setup: output = (a + b) @ (a - b)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_multiple_pipes_nosplit"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [MULTI_PIPE_DIM, MULTI_PIPE_DIM], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [MULTI_PIPE_DIM, MULTI_PIPE_DIM], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [MULTI_PIPE_DIM, MULTI_PIPE_DIM], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return MultiPipeNoSplitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = torch.matmul(tensors["a"] + tensors["b"], tensors["a"] - tensors["b"])


class TestCrossCore:
    """Cross-core communication system tests."""

    def test_tpush_tpop_v2c_updown(self, test_runner, backend_type):
        """V2C updown pipe: compile through full pipeline and verify kernel artifacts."""
        result = test_runner.run(V2CUDTest(backend_type=backend_type))
        assert result.passed, f"Cross-core V2C updown compilation failed: {result.error}"

    def test_tpush_tpop_v2c_leftright(self, test_runner, backend_type):
        """V2C left-right pipe: compile through full pipeline and verify kernel artifacts."""
        result = test_runner.run(V2CLRTest(backend_type=backend_type))
        assert result.passed, f"Cross-core V2C left-right compilation failed: {result.error}"

    def test_tpush_tpop_v2c_nosplit(self, test_runner, backend_type):
        """V2C no-split pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(V2CNoSplitTest(backend_type=backend_type))
        assert result.passed, f"Cross-core V2C no-split compilation failed: {result.error}"

    def test_tpop_c2v_leftright(self, test_runner, backend_type):
        """C2V left-right pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(C2VLRTest(backend_type=backend_type))
        assert result.passed, f"Cross-core C2V left-right compilation failed: {result.error}"

    def test_tpop_c2v_updown(self, test_runner, backend_type):
        """C2V updown pipe: compile through full pipeline and verify correctness."""
        result = test_runner.run(C2VUDTest(backend_type=backend_type))
        assert result.passed, f"Cross-core C2V updown compilation failed: {result.error}"

    def test_tpop_c2v_nosplit(self, test_runner, backend_type, platform):
        """C2V no-split pipe: compile through full pipeline and verify correctness."""
        if platform == "a5sim":
            pytest.xfail("950 backend cross-core C2V no-split not yet stable on sim")
        result = test_runner.run(C2VNoSplitTest(backend_type=backend_type))
        assert result.passed, f"Cross-core C2V no-split compilation failed: {result.error}"

    def test_tpop_bidirect_updown(self, test_runner, backend_type, platform):
        """Bidirect updown pipe: compile through full pipeline and verify correctness."""
        if platform == "a5sim":
            pytest.xfail("950 backend cross-core bidirect updown not yet stable on sim")
        result = test_runner.run(BiDirectUDTest(backend_type=backend_type))
        assert result.passed, f"Cross-core bidirect updown compilation failed: {result.error}"

    def test_tpop_bidirect_leftright(self, test_runner, backend_type, platform):
        """Bidirect left-right pipe: compile through full pipeline and verify correctness."""
        if platform == "a5sim":
            pytest.xfail("950 backend cross-core bidirect left-right not yet stable on sim")
        result = test_runner.run(BiDirectLRTest(backend_type=backend_type))
        assert result.passed, f"Cross-core bidirect left-right compilation failed: {result.error}"

    def test_tpop_bidirect_nosplit(self, test_runner, backend_type, platform):
        """Bidirect no-split pipe: compile through full pipeline and verify correctness."""
        if platform == "a5sim":
            pytest.xfail("950 backend cross-core bidirect no-split not yet stable on sim")
        result = test_runner.run(BiDirectNoSplitTest(backend_type=backend_type))
        assert result.passed, f"Cross-core bidirect no-split compilation failed: {result.error}"

    def test_multiple_pipes_nosplit(self, test_runner, backend_type, platform):
        """Explicit multiple pipe ids: compile through full pipeline and verify correctness."""
        if platform == "a5sim":
            pytest.xfail("950 backend explicit multi-pipe no-split not yet validated on sim")
        result = test_runner.run(MultiPipeNoSplitTest(backend_type=backend_type))
        assert result.passed, f"Cross-core explicit multi-pipe no-split failed: {result.error}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
