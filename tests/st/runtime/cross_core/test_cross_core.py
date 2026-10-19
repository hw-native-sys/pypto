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
from harness import st
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.pypto_core.passes import MemoryPlanner

M = 32
K = 64
N = 512
N_BLOCK = 64
N_BLOCKS = N // N_BLOCK
# Bidirectional cube<->vec fused into one no-split pl.spmd scope. Square per-tile
# ([ROW_TILE, SPMD_N]) so the result can be transposed in place.
SPMD_N = 16
SPMD_ROW_TILE = 16
MULTI_PIPE_DIM = 16
MULTI_PIPE_SLOT_SIZE = MULTI_PIPE_DIM * MULTI_PIPE_DIM * 4
MULTI_PIPE_BUFFER_SIZE = MULTI_PIPE_SLOT_SIZE * 8

# Explicit slot_num / local_slot_num pipe sizing.
# slot_num pins the GM ring depth (here 4, below the default 8); local_slot_num
# pins the local slot count. Reserve-buffer size is arch-dependent:
#   a3 -> slot_size * local_slot_num ; a5 -> slot_size * slot_num.
# Keeping local_slot_num == slot_num makes a single buffer size correct on both.
# NOTE: a5 does not accept the operand at all -- ptoas rejects `local_slot_num`
# on pto.{aic,aiv}_initialize_pipe regardless of its value, so this case is
# a2/a3-only in practice; see the xfail on test_explicit_slot_num.
SLOTNUM_DIM = 16
SLOTNUM_SLOT_SIZE = SLOTNUM_DIM * SLOTNUM_DIM * 4
SLOTNUM_SLOT_NUM = 4
SLOTNUM_LOCAL_SLOT_NUM = 4
SLOTNUM_BUFFER_SIZE = SLOTNUM_SLOT_SIZE * SLOTNUM_LOCAL_SLOT_NUM


_LOAD_POLICIES = list(pl.CachePolicy)
_STORE_POLICIES = [
    policy
    for policy in pl.CachePolicy
    if policy not in (pl.CachePolicy.NOT_ALLOC_KEEP, pl.CachePolicy.NOT_ALLOC_DROP)
]


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
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
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
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
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
        with pl.at(level=pl.Level.CORE_GROUP):
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
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
            for cb in pl.range(0, N_BLOCKS // 4):
                for ci in pl.range(0, 4):
                    nb = cb * 4 + ci
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
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            for cb in pl.range(0, N_BLOCKS // 4):
                for ci in pl.range(0, 4):
                    nb = cb * 4 + ci
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
        with pl.at(level=pl.Level.CORE_GROUP):
            for cb in pl.range(0, N_BLOCKS // 4):
                for ci in pl.range(0, 4):
                    nb = cb * 4 + ci
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
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            for cb in pl.range(0, N_BLOCKS // 4):
                for ci in pl.range(0, 4):
                    nb = cb * 4 + ci
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
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
            for cb in pl.range(0, N_BLOCKS // 4):
                for ci in pl.range(0, 4):
                    nb = cb * 4 + ci
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
        with pl.at(level=pl.Level.CORE_GROUP):
            for cb in pl.range(0, N_BLOCKS // 4):
                for ci in pl.range(0, 4):
                    nb = cb * 4 + ci
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
class BidirectSpmdNoSplitProgram:
    """Bidirectional cube<->vec + ``tile.transpose`` fused in one no-split scope.

    ``out = ((a + 1) @ b + 1).T`` per row-tile. A vector op produces the matmul
    operand (V->C) and a vector op consumes the result (C->V), so the fused scope
    runs under dual-AIV dispatch. The ``pl.transpose`` of the consumed vector tile
    then exercises the path this test guards: on the secondary subblock that
    transpose is replayed as a zero-valid tile, which lowers to a ``pto.ttrans``
    that 507018-hangs the AICore unless SplitVectorKernel rewrites it to an empty
    ``tile.create`` (#1761).
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, SPMD_N], pl.FP32],
        out: pl.Tensor[[M, SPMD_N], pl.FP32],
    ) -> pl.Tensor[[M, SPMD_N], pl.FP32]:
        for ob in pl.spmd(M // SPMD_ROW_TILE, name_hint="bidirect_spmd"):
            m0 = ob * SPMD_ROW_TILE
            a_slice = pl.slice(a, [SPMD_ROW_TILE, K], [m0, 0])
            a_add = pl.add(a_slice, 1.0)  # vector produces the matmul operand (V->C)
            c_tile = pl.matmul(a_add, b)  # cube
            c_vec = pl.add(c_tile, 1.0)  # vector consumes the matmul result (C->V)
            c_t = pl.transpose(c_vec, axis1=0, axis2=1)  # replayed zero-valid on subblock 1
            out = pl.assemble(out, c_t, [m0, 0])
        return out


class BidirectSpmdNoSplitTest(PTOTestCase):
    """Cross-core V<->C bidirectional + transpose, no split, in one pl.spmd scope."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_bidirect_spmd_nosplit"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, SPMD_N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [M, SPMD_N], DataType.FP32, is_output=True, init_value=torch.randn),
        ]

    def get_program(self) -> Any:
        return BidirectSpmdNoSplitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"]
        b = tensors["b"]
        for m0 in range(0, M, SPMD_ROW_TILE):
            c = torch.matmul(a[m0 : m0 + SPMD_ROW_TILE] + 1.0, b) + 1.0
            tensors["out"][m0 : m0 + SPMD_ROW_TILE] = c.t()


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


@pl.program
class ExplicitSlotNumProgram:
    """Two explicit V2C pipes with caller-pinned slot_num / local_slot_num."""

    @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
    def vector_slotnum(
        self,
        a: pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32],
        b: pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32],
        output: pl.Out[pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32]],
    ):
        v2c_peer_0 = pl.import_peer_buffer(name="v2c_slotnum_buffer_0", peer_func="cube_slotnum")
        v2c_peer_1 = pl.import_peer_buffer(name="v2c_slotnum_buffer_1", peer_func="cube_slotnum")
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_peer_0,
            dir_mask=2,
            slot_size=SLOTNUM_SLOT_SIZE,
            slot_num=SLOTNUM_SLOT_NUM,
            local_slot_num=SLOTNUM_LOCAL_SLOT_NUM,
            id=0,
        )
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_peer_1,
            dir_mask=2,
            slot_size=SLOTNUM_SLOT_SIZE,
            slot_num=SLOTNUM_SLOT_NUM,
            local_slot_num=SLOTNUM_LOCAL_SLOT_NUM,
            id=1,
        )
        a_tile: pl.Tile[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32, pl.Mem.Vec] = pl.load(
            a, [0, 0], [SLOTNUM_DIM, SLOTNUM_DIM]
        )
        b_tile: pl.Tile[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32, pl.Mem.Vec] = pl.load(
            b, [0, 0], [SLOTNUM_DIM, SLOTNUM_DIM]
        )
        sum_tile: pl.Tile[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32, pl.Mem.Vec] = pl.add(a_tile, b_tile)
        diff_tile: pl.Tile[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32, pl.Mem.Vec] = pl.sub(a_tile, b_tile)
        pl.tpush_to_aic(diff_tile, split=0, id=1)
        pl.tpush_to_aic(sum_tile, split=0, id=0)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_slotnum(
        self,
        a: pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32],
        b: pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32],
        output: pl.Out[pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32]],
    ) -> pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32]:
        v2c_buf_0 = pl.reserve_buffer(name="v2c_slotnum_buffer_0", size=SLOTNUM_BUFFER_SIZE, base=pl.AUTO)
        v2c_buf_1 = pl.reserve_buffer(name="v2c_slotnum_buffer_1", size=SLOTNUM_BUFFER_SIZE, base=pl.AUTO)
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_buf_0,
            dir_mask=2,
            slot_size=SLOTNUM_SLOT_SIZE,
            slot_num=SLOTNUM_SLOT_NUM,
            local_slot_num=SLOTNUM_LOCAL_SLOT_NUM,
            id=0,
        )
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            v2c_buf_1,
            dir_mask=2,
            slot_size=SLOTNUM_SLOT_SIZE,
            slot_num=SLOTNUM_SLOT_NUM,
            local_slot_num=SLOTNUM_LOCAL_SLOT_NUM,
            id=1,
        )
        sum_tile: pl.Tile[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32, pl.Mem.Mat] = pl.tpop_from_aiv(split=0, id=0)
        diff_tile: pl.Tile[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32, pl.Mem.Mat] = pl.tpop_from_aiv(split=0, id=1)
        sum_left = pl.move(sum_tile, target_memory=pl.Mem.Left)
        diff_right = pl.move(diff_tile, target_memory=pl.Mem.Right)
        result: pl.Tile[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32] = pl.matmul(sum_left, diff_right)
        pl.tfree_to_aiv(sum_tile, id=0)
        pl.tfree_to_aiv(diff_tile, id=1)
        return pl.store(result, [0, 0], output)

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32],
        b: pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32],
        output: pl.Out[pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32]],
    ) -> pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32]:
        result = self.cube_slotnum(a, b, output)
        self.vector_slotnum(a, b, output)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32],
        b: pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32],
        output: pl.Out[pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32]],
    ) -> pl.Tensor[[SLOTNUM_DIM, SLOTNUM_DIM], pl.FP32]:
        result = self.group_func(a, b, output)
        return result


class ExplicitSlotNumTest(PTOTestCase):
    """Explicit slot_num / local_slot_num V->C setup: output = (a + b) @ (a - b)."""

    __test__ = False

    def get_name(self) -> str:
        return "cross_core_explicit_slot_num"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [SLOTNUM_DIM, SLOTNUM_DIM], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [SLOTNUM_DIM, SLOTNUM_DIM], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [SLOTNUM_DIM, SLOTNUM_DIM], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ExplicitSlotNumProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = torch.matmul(tensors["a"] + tensors["b"], tensors["a"] - tensors["b"])


class TestCrossCore:
    """Cross-core communication system tests."""

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

    def test_tpop_bidirect_spmd_nosplit(self, test_runner):
        """Bidirect cube<->vec + transpose fused in one no-split pl.spmd scope.

        On-board guard for #1761: the secondary-subblock replay of the scope's
        ``tile.transpose`` must not 507018-hang the AICore.
        """
        result = test_runner.run(BidirectSpmdNoSplitTest())
        assert result.passed, f"Cross-core bidirect spmd no-split failed: {result.error}"

    def test_multiple_pipes_nosplit(self, test_runner):
        """Explicit multiple pipe ids: compile through full pipeline and verify correctness."""
        result = test_runner.run(MultiPipeNoSplitTest())
        assert result.passed, f"Cross-core explicit multi-pipe no-split failed: {result.error}"

    # ptoas rejects the operand outright for the 950 frontend pipe lowering,
    # whatever its value --
    #   error: 'pto.aic_initialize_pipe' op 'local_slot_num' is only supported
    #          for a2/a3 frontend pipe lowering
    # Deselected rather than platform_xfail-ed: xfail says "this ought to work
    # and does not", which would keep a permanent platform limitation on the
    # report as if it were a defect awaiting a fix.
    @pytest.mark.platforms(
        "a2a3",
        "a2a3sim",
        reason=(
            "local_slot_num on pto.{aic,aiv}_initialize_pipe is an a2/a3-only operand, so the "
            "manual pl.{aic,aiv}_initialize_pipe route this case exercises does not exist on 950"
        ),
    )
    def test_explicit_slot_num(self, test_runner):
        """Explicit slot_num / local_slot_num: compile through full pipeline and verify correctness."""
        result = test_runner.run(ExplicitSlotNumTest())
        assert result.passed, f"Cross-core explicit slot_num failed: {result.error}"


def _l2_hint_case(load_hint, store_hint, planner):
    """Exercise explicit hints beside ordinary transfers inside a tile loop."""

    @pl.jit.incore
    def hinted_kernel(
        x: pl.Tensor[[64, 32], pl.FP32],
        y: pl.Tensor[[64, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        for i, (out_iter,) in pl.range(2, init_values=(out,)):
            a = pl.load(x, [i * 32, 0], [32, 32], cache=load_hint)
            b = pl.load(y, [i * 32, 0], [32, 32])
            updated = pl.store(pl.add(a, b), [i * 32, 0], out_iter, cache=store_hint)
            out = pl.yield_(updated)
        return out

    @pl.jit
    def hinted_add(
        x: pl.Tensor[[64, 32], pl.FP32],
        y: pl.Tensor[[64, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        return hinted_kernel(x, y, out)

    x = (torch.arange(64 * 32).reshape(64, 32) % 127 - 63).float()
    y = (torch.arange(64 * 32).reshape(64, 32) % 31 - 15).float()
    return st.case(
        hinted_add,
        x,
        y,
        torch.full_like(x, float("nan")),
        name=f"l2_{load_hint.name}_{store_hint.name}_{planner.name}",
        golden=lambda tensors: tensors["x"] + tensors["y"],
        memory_planner=planner,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.platforms("a5", "a5sim")
@st.cases(
    *[
        _l2_hint_case(load_hint, store_hint, planner)
        for load_hint in _LOAD_POLICIES
        for store_hint in _STORE_POLICIES
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
    ]
)
def test_l2_hints_preserve_results(case_run):
    """Every A5 load/store hint pair preserves exact results."""
    case_run.assert_passed()


def _scope_rebind_case(policy, planner, explicit_default):
    override = pl.CachePolicy.DEFAULT if explicit_default else None

    @pl.jit
    def scoped_rebind(x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_cache_policy(out, policy)
            value = pl.add(x, x)
            out = pl.assemble(out, value, [0, 0])
            first = pl.load(out, [0, 0], [32, 32])
            out = pl.store(pl.add(first, first), [0, 0], out)
            second = pl.load(out, [0, 0], [32, 32], cache=override)
            out = pl.store(second, [0, 0], out)
        return out

    name = f"l2_scope_rebind_{policy.name}_{explicit_default}_{planner.name}"
    x = (torch.arange(32 * 32).reshape(32, 32) % 127 - 63).float()
    case = st.case(
        scoped_rebind,
        x,
        torch.full_like(x, float("nan")),
        name=name,
        golden=lambda tensors: tensors["x"] * 4,
        memory_planner=planner,
        rtol=0.0,
        atol=0.0,
    )
    return pytest.param(policy, explicit_default, case, id=name)


@pytest.fixture
def _scope_rebind_run(_st_case, test_runner, tmp_path, monkeypatch):
    # CI runs inline under --forked; keep its sources as well as batch artifacts.
    monkeypatch.setattr(test_runner.config, "save_kernels", True)
    monkeypatch.setattr(test_runner.config, "save_kernels_dir", str(tmp_path))
    run = st.CaseRun(_st_case, test_runner.run(_st_case))
    return run, run.work_dir or (tmp_path / _st_case.name)


@pytest.mark.platforms("a5", "a5sim")
@pytest.mark.parametrize(
    "policy, explicit_default, _st_case",
    [
        _scope_rebind_case(policy, planner, explicit_default)
        for policy in (
            pl.CachePolicy.NORMAL_FIRST_VICTIM,
            pl.CachePolicy.NORMAL_LAST_VICTIM,
            pl.CachePolicy.NORMAL_PERSISTENT,
        )
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
        for explicit_default in (False, True)
    ],
)
def test_scope_cache_policy_survives_gm_rebinding(_scope_rebind_run, policy, explicit_default):
    """Check inherited ISA modes as well as values after assemble/store rebindings."""
    _assert_scope_rebind_hints(_scope_rebind_run, policy, explicit_default)


def _assert_scope_rebind_hints(scope_rebind_run, policy, explicit_default):
    case_run, work_dir = scope_rebind_run
    case_run.assert_passed()
    sources = list((work_dir / "kernels").rglob("*.cpp"))
    assert sources, "the compiled case must retain its kernel sources"
    cpp = "\n".join(path.read_text() for path in sources)
    hint = {
        pl.CachePolicy.NORMAL_FIRST_VICTIM: "NormalFirstVictim",
        pl.CachePolicy.NORMAL_LAST_VICTIM: "NormalLastVictim",
        pl.CachePolicy.NORMAL_PERSISTENT: "NormalPersistent",
    }[policy]
    assert cpp.count(f"TLOAD<pto::TLoadL2Hint::{hint}>") == (1 if explicit_default else 2)
    assert "TSTORE<pto::TStoreL2Hint" not in cpp, "scope policies govern reads only"


def _scope_conditional_rebind_case(policy, planner, take_then):
    @pl.jit
    def scoped_conditional_rebind(
        x: pl.Tensor[[32, 32], pl.FP32],
        flags: pl.Tensor[[1], pl.INT32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_cache_policy(out, policy)
            if pl.read(flags, [0]) > 0:
                out = pl.assemble(out, pl.add(x, x), [0, 0])
            else:
                value = pl.load(x, [0, 0], [32, 32])
                out = pl.store(pl.add(value, pl.add(value, value)), [0, 0], out)
            first = pl.load(out, [0, 0], [32, 32])
            out = pl.store(pl.add(first, first), [0, 0], out)
            second = pl.load(out, [0, 0], [32, 32])
            out = pl.store(second, [0, 0], out)
        return out

    name = f"l2_scope_if_{take_then}_{policy.name}_{planner.name}"
    x = (torch.arange(32 * 32).reshape(32, 32) % 127 - 63).float()
    factor = 4 if take_then else 6
    case = st.case(
        scoped_conditional_rebind,
        x,
        torch.tensor([int(take_then)], dtype=torch.int32),
        torch.full_like(x, float("nan")),
        name=name,
        golden=lambda tensors: tensors["x"] * factor,
        memory_planner=planner,
        rtol=0.0,
        atol=0.0,
    )
    return pytest.param(policy, case, id=name)


@pytest.mark.platforms("a5", "a5sim")
@pytest.mark.parametrize(
    "policy, _st_case",
    [
        _scope_conditional_rebind_case(policy, planner, take_then)
        for policy in (
            pl.CachePolicy.NORMAL_FIRST_VICTIM,
            pl.CachePolicy.NORMAL_LAST_VICTIM,
            pl.CachePolicy.NORMAL_PERSISTENT,
        )
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
        for take_then in (False, True)
    ],
)
def test_scope_cache_policy_survives_conditional_gm_rebinding(_scope_rebind_run, policy):
    """Both runtime branches retain the selected read mode after their GM merge."""
    _assert_scope_rebind_hints(_scope_rebind_run, policy, False)


def _scope_loop_rebind_case(policy, planner, explicit_default, use_while, steps):
    override = pl.CachePolicy.DEFAULT if explicit_default else None

    @pl.jit.inline
    def for_rebind(
        x: pl.Tensor[[32, 32], pl.FP32],
        out: pl.Tensor[[32, 32], pl.FP32],
        limit: pl.Scalar[pl.INT32],
    ):
        for i, (carried,) in pl.range(limit, init_values=(out,)):
            updated = pl.assemble(carried, pl.add(carried, x), [0, 0])
            out = pl.yield_(updated)
        return out

    @pl.jit.inline
    def while_rebind(
        x: pl.Tensor[[32, 32], pl.FP32],
        out: pl.Tensor[[32, 32], pl.FP32],
        limit: pl.Scalar[pl.INT32],
    ):
        for i, carried in pl.while_(init_values=(0, out)):
            pl.cond(i < limit)
            updated = pl.assemble(carried, pl.add(carried, x), [0, 0])
            _i, out = pl.yield_(i + 1, updated)
        return out

    rebind = while_rebind if use_while else for_rebind

    @pl.jit
    def scoped_loop_rebind(
        x: pl.Tensor[[32, 32], pl.FP32],
        counts: pl.Tensor[[1], pl.INT32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_cache_policy(out, policy)
            out = pl.assemble(out, pl.add(x, x), [0, 0])
            limit = pl.read(counts, [0])
            out = rebind(x, out, limit)
            second = pl.load(out, [0, 0], [32, 32], cache=override)
            out = pl.store(pl.add(second, second), [0, 0], out)
        return out

    kind = "while" if use_while else "for"
    name = f"l2_scope_loop_{kind}_{steps}_{policy.name}_{explicit_default}_{planner.name}"
    x = (torch.arange(32 * 32).reshape(32, 32) % 127 - 63).float()
    case = st.case(
        scoped_loop_rebind,
        x,
        torch.tensor([steps], dtype=torch.int32),
        torch.full_like(x, float("nan")),
        name=name,
        golden=lambda tensors: tensors["x"] * (4 + 2 * steps),
        memory_planner=planner,
        rtol=0.0,
        atol=0.0,
    )
    return pytest.param(policy, explicit_default, case, id=name)


@pytest.mark.platforms("a5", "a5sim")
@pytest.mark.parametrize(
    "policy, explicit_default, _st_case",
    [
        _scope_loop_rebind_case(policy, planner, explicit_default, use_while, steps)
        for policy in (
            pl.CachePolicy.NORMAL_FIRST_VICTIM,
            pl.CachePolicy.NORMAL_LAST_VICTIM,
            pl.CachePolicy.NORMAL_PERSISTENT,
        )
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
        for explicit_default in (False, True)
        for use_while in (False, True)
        for steps in (0, 2)
    ],
)
def test_scope_cache_policy_survives_loop_results(_scope_rebind_run, policy, explicit_default):
    """Check zero/multiple iterations and ISA hints after for/while GM results."""
    _assert_scope_rebind_hints(_scope_rebind_run, policy, explicit_default)


def _acc_store_cache_case(policy, planner, quantized, with_relu):
    scale = 0.5 if quantized else None

    @pl.jit.incore
    def acc_store(
        x: pl.Tensor[[16, 64], pl.FP16],
        y: pl.Tensor[[32, 64], pl.FP16],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
    ):
        lhs = pl.load(x, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
        rhs = pl.load(y, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
        acc = pl.matmul(
            pl.tile.move(lhs, target_memory=pl.Mem.Left),
            pl.tile.move(pl.tile.transpose_view(rhs), target_memory=pl.Mem.Right),
        )
        return pl.store(acc, [0, 0], out, cache=policy, pre_quant=scale, pre_relu=with_relu)

    @pl.jit
    def acc_writeback(
        x: pl.Tensor[[16, 64], pl.FP16],
        y: pl.Tensor[[32, 64], pl.FP16],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
    ):
        return acc_store(x, y, out)

    def golden(tensors):
        value = tensors["x"].float() @ tensors["y"].float().T
        if with_relu:
            value = torch.relu(value)
        return (value * (0.5 if quantized else 1.0)).half()

    name = f"l2_acc_{policy.name}_quant{quantized}_relu{with_relu}_{planner.name}"
    case = st.case(
        acc_writeback,
        (torch.arange(16 * 64).reshape(16, 64) % 7 - 3).half(),
        (torch.arange(32 * 64).reshape(32, 64) % 5 - 2).half(),
        torch.full((16, 32), float("nan"), dtype=torch.float16),
        name=name,
        golden=golden,
        memory_planner=planner,
        rtol=0.0,
        atol=0.0,
    )
    return pytest.param(policy, case, id=name)


@pytest.mark.platforms("a5", "a5sim")
@pytest.mark.parametrize(
    "policy, _st_case",
    [
        _acc_store_cache_case(policy, planner, quantized, with_relu)
        for policy in _STORE_POLICIES
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
        for quantized in (False, True)
        for with_relu in (False, True)
    ],
)
def test_acc_store_cache_policy_is_honored_or_rejected(case_run, policy):
    """Do not mistake a numerically correct ACC store for an honored cache hint."""
    if policy in (pl.CachePolicy.DEFAULT, pl.CachePolicy.NORMAL_FIRST_VICTIM):
        case_run.assert_passed()
    else:
        assert not case_run.passed, "unsupported ACC cache modes must fail before dispatch"
        error = case_run.error or ""
        assert "A5 ACC-to-GM store cache policy" in error, error
        assert "PTO-ISA" in error and "l2Control" in error, error


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
