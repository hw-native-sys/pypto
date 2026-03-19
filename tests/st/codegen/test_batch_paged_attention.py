# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for Batch Paged Attention kernels using PyPTO frontend.

Each kernel from batch_paged_attention_example.py is tested in isolation
with a dedicated PTOTestCase subclass and a standalone @pl.program that
wraps just the kernel under test plus a thin orchestration.

KernelQkMatmul (CUBE):
  Per-batch: sij[b] = query[qi_row] @ key_cache[phys_block * bs].T
  where qi_row = b * num_heads + q_offset, phys_block from block_table

KernelSoftmaxPrepare (VECTOR):
  Per-batch: mask invalid columns with -inf, scale, row_max, exp, row_sum

KernelPvMatmul (CUBE):
  Per-batch: oi_new[b] = pij[b] @ value_cache[phys_block * bs]

KernelOnlineUpdate (VECTOR):
  Per-batch: online softmax update with is_first/is_last branching
"""

import struct
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

from examples.ir_parser.batch_paged_attention_example import BuildBatchPagedAttentionProgram

DEFAULT_SCALE = 1.0


# ---------------------------------------------------------------------------
# QK Matmul (batched)
# ---------------------------------------------------------------------------
class BatchQKMatmulTestCase(PTOTestCase):
    """Test case for batched QK matmul kernel.

    Computes per-batch: sij[b] = qi[b] @ kj[b].T
    where qi/kj addresses are derived from block_table + scalar offsets.
    """

    def __init__(
        self,
        batch: int = 2,
        num_heads: int = 16,
        head_dim: int = 16,
        block_size: int = 16,
        q_tile: int = 16,
        block_num: int = 4,
        block_idx: int = 0,
        q_offset: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.q_tile = q_tile
        self.block_num = block_num
        self.block_idx = block_idx
        self.q_offset = q_offset

    def get_name(self) -> str:
        return f"batch_qk_matmul_{self.batch}b_{self.num_heads}h_{self.head_dim}d"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        query_rows = self.batch * self.num_heads
        key_cache_rows = self.batch * self.block_num * self.block_size
        batch_q_tile = self.batch * self.q_tile
        bt_size = self.batch * self.block_num

        block_table = torch.arange(bt_size, dtype=torch.int32)
        config = torch.tensor(
            [self.batch, self.block_idx, self.q_offset, self.block_num, self.num_heads],
            dtype=torch.int64,
        )

        return [
            TensorSpec("query", [query_rows, self.head_dim], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [key_cache_rows, self.head_dim], DataType.BF16, init_value=torch.randn),
            TensorSpec("block_table", [bt_size], DataType.INT32, init_value=block_table),
            TensorSpec("config", [5], DataType.INT64, init_value=config),
            TensorSpec("sij_batch", [batch_q_tile, self.block_size], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        batch = self.batch
        num_heads = self.num_heads
        head_dim = self.head_dim
        block_size = self.block_size
        q_tile = self.q_tile
        block_num = self.block_num

        query_rows = batch * num_heads
        key_cache_rows = batch * block_num * block_size
        batch_q_tile = batch * q_tile
        block_table_flat_size = batch * block_num

        @pl.program
        class BatchQKMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def KernelQkMatmul(
                self,
                query: pl.Tensor[[query_rows, head_dim], pl.BF16],
                key_cache: pl.Tensor[[head_dim, key_cache_rows], pl.BF16, pl.DN],
                sij_batch: pl.Out[pl.Tensor[[batch_q_tile, block_size], pl.FP32]],
                block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
                batch_count: pl.Scalar[pl.INDEX],
                block_idx: pl.Scalar[pl.INDEX],
                q_offset: pl.Scalar[pl.INDEX],
                block_num_p: pl.Scalar[pl.INDEX],
                num_heads_p: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[batch_q_tile, block_size], pl.FP32]:
                for b in pl.range(batch_count):
                    qi_row = b * num_heads_p + q_offset
                    phys_block = pl.read(block_table, b * block_num_p + block_idx)
                    kj_row = phys_block * block_size

                    qi_l1 = pl.load(
                        query,
                        [qi_row, 0],
                        [q_tile, head_dim],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    kj_l1 = pl.load(
                        key_cache,
                        [0, kj_row],
                        [head_dim, block_size],
                        target_memory=pl.MemorySpace.Mat,
                        transpose=True,
                    )
                    qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)
                    kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)
                    sij_l0c = pl.matmul(qi_l0a, kj_l0b)
                    sij_batch_new = pl.store(sij_l0c, [b * q_tile, 0], sij_batch)
                return sij_batch_new

            @pl.function(type=pl.FunctionType.Orchestration)
            def Orchestrator(
                self,
                query: pl.Tensor[[query_rows, head_dim], pl.BF16],
                key_cache: pl.Tensor[[head_dim, key_cache_rows], pl.BF16, pl.DN],
                block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
                config: pl.Tensor[[5], pl.INT64],
                sij_batch: pl.Out[pl.Tensor[[batch_q_tile, block_size], pl.FP32]],
            ) -> pl.Tensor[[batch_q_tile, block_size], pl.FP32]:
                batch_count: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                block_idx: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                q_offset: pl.Scalar[pl.INT64] = pl.tensor.read(config, [2])
                block_num_p: pl.Scalar[pl.INT64] = pl.tensor.read(config, [3])
                num_heads_p: pl.Scalar[pl.INT64] = pl.tensor.read(config, [4])

                sij_batch = self.KernelQkMatmul(
                    query,
                    key_cache,
                    sij_batch,
                    block_table,
                    batch_count,
                    block_idx,
                    q_offset,
                    block_num_p,
                    num_heads_p,
                )
                return sij_batch

        return BatchQKMatmulProgram

    def compute_expected(self, tensors, params=None):
        query = tensors["query"].float()
        key_cache = tensors["key_cache"].float()
        block_table = tensors["block_table"]
        config = tensors["config"]

        batch_count = int(config[0].item())
        block_idx = int(config[1].item())
        q_offset = int(config[2].item())
        block_num = int(config[3].item())
        num_heads = int(config[4].item())

        q_tile = 16
        block_size = 16

        for b in range(batch_count):
            qi_row = b * num_heads + q_offset
            phys_block = int(block_table[b * block_num + block_idx].item())
            kj_row = phys_block * block_size

            qi = query[qi_row : qi_row + q_tile, :]
            kj = key_cache[kj_row : kj_row + block_size, :]

            tensors["sij_batch"][b * q_tile : (b + 1) * q_tile, :] = qi @ kj.T


# ---------------------------------------------------------------------------
# Softmax Prepare (batched)
# ---------------------------------------------------------------------------
class BatchSoftmaxPrepareTestCase(PTOTestCase):
    """Test case for batched softmax prepare kernel.

    Per-batch: mask invalid columns, scale, row_max, exp, BF16 cast, row_sum.
    context_lens + block_idx determine valid_len for each batch element.
    """

    def __init__(
        self,
        batch: int = 2,
        num_heads: int = 16,
        block_size: int = 16,
        q_tile: int = 16,
        block_idx: int = 1,
        context_lens: list[int] | None = None,
        scale: float = DEFAULT_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch = batch
        self.num_heads = num_heads
        self.block_size = block_size
        self.q_tile = q_tile
        self.block_idx = block_idx
        self.context_lens = context_lens or [20, 24]
        self.scale = scale

    def get_name(self) -> str:
        return f"batch_softmax_prepare_{self.batch}b_{self.block_size}bs_bi{self.block_idx}"

    def define_tensors(self) -> list[TensorSpec]:
        batch_q_tile = self.batch * self.q_tile

        context_lens_t = torch.tensor(self.context_lens, dtype=torch.int32)
        config = torch.tensor([self.batch, self.block_idx], dtype=torch.int64)

        return [
            TensorSpec("sij_batch", [batch_q_tile, self.block_size], DataType.FP32, init_value=torch.randn),
            TensorSpec("context_lens", [self.batch], DataType.INT32, init_value=context_lens_t),
            TensorSpec("scale_config", [1], DataType.FP32, init_value=self.scale),
            TensorSpec("config", [2], DataType.INT64, init_value=config),
            TensorSpec("pij_batch", [batch_q_tile, self.block_size], DataType.BF16, is_output=True),
            TensorSpec("mij_batch", [batch_q_tile, 1], DataType.FP32, is_output=True),
            TensorSpec("lij_batch", [batch_q_tile, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        batch = self.batch
        block_size = self.block_size
        q_tile = self.q_tile

        batch_q_tile = batch * q_tile

        @pl.program
        class BatchSoftmaxPrepareProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def KernelSoftmaxPrepare(
                self,
                sij_batch: pl.Tensor[[batch_q_tile, block_size], pl.FP32],
                pij_batch: pl.Out[pl.Tensor[[batch_q_tile, block_size], pl.BF16]],
                mij_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
                lij_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
                scale_value: pl.Scalar[pl.FP32],
                context_lens: pl.Tensor[[batch], pl.INT32],
                batch_count: pl.Scalar[pl.INT64],
                block_idx: pl.Scalar[pl.INT64],
            ) -> tuple[
                pl.Tensor[[batch_q_tile, block_size], pl.BF16],
                pl.Tensor[[batch_q_tile, 1], pl.FP32],
                pl.Tensor[[batch_q_tile, 1], pl.FP32],
            ]:
                for b in pl.range(batch_count):
                    cur_seq = pl.read(context_lens, b)
                    start = block_idx * block_size
                    remaining = cur_seq - start
                    valid_len = pl.max(pl.min(remaining, block_size), 0)

                    s_tile = pl.load(
                        sij_batch,
                        [b * q_tile, 0],
                        [q_tile, block_size],
                        valid_shapes=[q_tile, valid_len],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    s_tile = pl.tile.fillpad(s_tile)

                    scaled = pl.mul(s_tile, scale_value)
                    tmp_tile = pl.create_tile(
                        [q_tile, block_size],
                        dtype=pl.FP32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    mi_tile = pl.row_max(scaled, tmp_tile)
                    sij_centered = pl.row_expand_sub(scaled, mi_tile)
                    exp_tile = pl.exp(sij_centered)
                    pij_tile_f16 = pl.cast(exp_tile, target_type=pl.BF16)
                    pij_tile = pl.cast(pij_tile_f16, target_type=pl.FP32)
                    li_tile = pl.row_sum(pij_tile, tmp_tile)

                    pij_batch = pl.store(pij_tile_f16, [b * q_tile, 0], [q_tile, block_size], pij_batch)
                    mij_batch = pl.store(mi_tile, [b * q_tile, 0], [q_tile, 1], mij_batch)
                    lij_batch = pl.store(li_tile, [b * q_tile, 0], [q_tile, 1], lij_batch)
                return pij_batch, mij_batch, lij_batch

            @pl.function(type=pl.FunctionType.Orchestration)
            def Orchestrator(
                self,
                sij_batch: pl.Tensor[[batch_q_tile, block_size], pl.FP32],
                context_lens: pl.Tensor[[batch], pl.INT32],
                scale_config: pl.Tensor[[1], pl.FP32],
                config: pl.Tensor[[2], pl.INT64],
                pij_batch: pl.Out[pl.Tensor[[batch_q_tile, block_size], pl.BF16]],
                mij_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
                lij_batch: pl.Out[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[batch_q_tile, block_size], pl.BF16],
                pl.Tensor[[batch_q_tile, 1], pl.FP32],
                pl.Tensor[[batch_q_tile, 1], pl.FP32],
            ]:
                batch_count: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                block_idx: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                scale_value: pl.Scalar[pl.FP32] = pl.tensor.read(scale_config, [0])

                pij_batch, mij_batch, lij_batch = self.KernelSoftmaxPrepare(
                    sij_batch,
                    pij_batch,
                    mij_batch,
                    lij_batch,
                    scale_value,
                    context_lens,
                    batch_count,
                    block_idx,
                )
                return pij_batch, mij_batch, lij_batch

        return BatchSoftmaxPrepareProgram

    def compute_expected(self, tensors, params=None):
        sij_batch = tensors["sij_batch"]
        context_lens = tensors["context_lens"]
        scale_value = tensors["scale_config"][0].item()
        config = tensors["config"]

        batch_count = int(config[0].item())
        block_idx = int(config[1].item())

        q_tile = 16
        block_size = 16

        for b in range(batch_count):
            cur_seq = int(context_lens[b].item())
            start = block_idx * block_size
            remaining = cur_seq - start
            valid_len = max(min(remaining, block_size), 0)

            s = sij_batch[b * q_tile : (b + 1) * q_tile, :].float().clone()

            if valid_len < block_size:
                s[:, valid_len:] = float("-inf")

            s_scaled = s * scale_value
            mi = s_scaled.max(dim=1, keepdim=True).values
            pij = torch.exp(s_scaled - mi)
            pij_f16 = pij.to(torch.float16)
            pij_f32 = pij_f16.float()
            li = pij_f32.sum(dim=1, keepdim=True)

            tensors["pij_batch"][b * q_tile : (b + 1) * q_tile, :] = pij_f16
            tensors["mij_batch"][b * q_tile : (b + 1) * q_tile, :] = mi
            tensors["lij_batch"][b * q_tile : (b + 1) * q_tile, :] = li


# ---------------------------------------------------------------------------
# PV Matmul (batched)
# ---------------------------------------------------------------------------
class BatchPVMatmulTestCase(PTOTestCase):
    """Test case for batched PV matmul kernel.

    Computes per-batch: oi_new[b] = pij[b] @ vj[b]
    where vj address is derived from block_table lookup.
    """

    def __init__(
        self,
        batch: int = 2,
        num_heads: int = 16,
        head_dim: int = 16,
        block_size: int = 16,
        q_tile: int = 16,
        block_num: int = 4,
        block_idx: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.q_tile = q_tile
        self.block_num = block_num
        self.block_idx = block_idx

    def get_name(self) -> str:
        return f"batch_pv_matmul_{self.batch}b_{self.num_heads}h_{self.head_dim}d"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        key_cache_rows = self.batch * self.block_num * self.block_size
        batch_q_tile = self.batch * self.q_tile
        bt_size = self.batch * self.block_num

        block_table = torch.arange(bt_size, dtype=torch.int32)
        config = torch.tensor(
            [self.batch, self.block_idx, self.block_num],
            dtype=torch.int64,
        )

        return [
            TensorSpec("pij_batch", [batch_q_tile, self.block_size], DataType.BF16, init_value=torch.randn),
            TensorSpec(
                "value_cache",
                [key_cache_rows, self.head_dim],
                DataType.BF16,
                init_value=torch.randn,
            ),
            TensorSpec("block_table", [bt_size], DataType.INT32, init_value=block_table),
            TensorSpec("config", [3], DataType.INT64, init_value=config),
            TensorSpec("oi_new_batch", [batch_q_tile, self.head_dim], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        batch = self.batch
        head_dim = self.head_dim
        block_size = self.block_size
        q_tile = self.q_tile
        block_num = self.block_num

        key_cache_rows = batch * block_num * block_size
        batch_q_tile = batch * q_tile
        block_table_flat_size = batch * block_num

        @pl.program
        class BatchPVMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def KernelPvMatmul(
                self,
                pij_batch: pl.Tensor[[batch_q_tile, block_size], pl.BF16],
                value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
                oi_new_batch: pl.Out[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
                block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
                batch_count: pl.Scalar[pl.INDEX],
                block_idx: pl.Scalar[pl.INDEX],
                block_num_p: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[batch_q_tile, head_dim], pl.FP32]:
                for b in pl.range(batch_count):
                    phys_block = pl.read(block_table, b * block_num_p + block_idx)
                    vj_row = phys_block * block_size

                    pij_l1 = pl.load(
                        pij_batch,
                        [b * q_tile, 0],
                        [q_tile, block_size],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    vj_l1 = pl.load(
                        value_cache,
                        [vj_row, 0],
                        [block_size, head_dim],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)
                    vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)
                    oi_l0c = pl.matmul(pij_l0a, vj_l0b)
                    oi_new_batch_new = pl.store(oi_l0c, [b * q_tile, 0], oi_new_batch)
                return oi_new_batch_new

            @pl.function(type=pl.FunctionType.Orchestration)
            def Orchestrator(
                self,
                pij_batch: pl.Tensor[[batch_q_tile, block_size], pl.BF16],
                value_cache: pl.Tensor[[key_cache_rows, head_dim], pl.BF16],
                block_table: pl.Tensor[[block_table_flat_size], pl.INT32],
                config: pl.Tensor[[3], pl.INT64],
                oi_new_batch: pl.Out[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
            ) -> pl.Tensor[[batch_q_tile, head_dim], pl.FP32]:
                batch_count: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                block_idx: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                block_num_p: pl.Scalar[pl.INT64] = pl.tensor.read(config, [2])

                oi_new_batch = self.KernelPvMatmul(
                    pij_batch,
                    value_cache,
                    oi_new_batch,
                    block_table,
                    batch_count,
                    block_idx,
                    block_num_p,
                )
                return oi_new_batch

        return BatchPVMatmulProgram

    def compute_expected(self, tensors, params=None):
        pij_batch = tensors["pij_batch"].float()
        value_cache = tensors["value_cache"].float()
        block_table = tensors["block_table"]
        config = tensors["config"]

        batch_count = int(config[0].item())
        block_idx = int(config[1].item())
        block_num = int(config[2].item())

        q_tile = 16
        block_size = 16

        for b in range(batch_count):
            phys_block = int(block_table[b * block_num + block_idx].item())
            vj_row = phys_block * block_size

            pij = pij_batch[b * q_tile : (b + 1) * q_tile, :]
            vj = value_cache[vj_row : vj_row + block_size, :]

            tensors["oi_new_batch"][b * q_tile : (b + 1) * q_tile, :] = pij @ vj


# ---------------------------------------------------------------------------
# Online Update (batched)
# ---------------------------------------------------------------------------
class BatchOnlineUpdateTestCase(PTOTestCase):
    """Test case for batched online update kernel.

    Handles all four (is_first, is_last) combinations:
      - (1, 1): copy mij->mi, lij->li, oi_new->oi; out = oi_new / lij
      - (1, 0): copy mij->mi, lij->li, oi_new->oi; out unchanged
      - (0, 1): full online update; out = oi_updated / li_updated
      - (0, 0): full online update; out unchanged
    """

    def __init__(
        self,
        batch: int = 2,
        num_heads: int = 16,
        head_dim: int = 16,
        q_tile: int = 16,
        is_first: int = 0,
        is_last: int = 1,
        q_offset: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_tile = q_tile
        self.is_first = is_first
        self.is_last = is_last
        self.q_offset = q_offset

    def get_name(self) -> str:
        return (
            f"batch_online_update_{self.batch}b_{self.num_heads}h_{self.head_dim}d"
            f"_f{self.is_first}_l{self.is_last}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        batch_q_tile = self.batch * self.q_tile
        out_rows = self.batch * self.num_heads

        config = torch.tensor(
            [self.is_first, self.is_last, self.batch, self.q_offset, self.num_heads],
            dtype=torch.int64,
        )

        return [
            TensorSpec("mij_batch", [batch_q_tile, 1], DataType.FP32, init_value=0.5),
            TensorSpec("lij_batch", [batch_q_tile, 1], DataType.FP32, init_value=1.5),
            TensorSpec("oi_new_batch", [batch_q_tile, self.head_dim], DataType.FP32, init_value=0.3),
            TensorSpec("config", [5], DataType.INT64, init_value=config),
            TensorSpec("mi_batch", [batch_q_tile, 1], DataType.FP32, init_value=0.4, is_output=True),
            TensorSpec("li_batch", [batch_q_tile, 1], DataType.FP32, init_value=2.0, is_output=True),
            TensorSpec(
                "oi_batch",
                [batch_q_tile, self.head_dim],
                DataType.FP32,
                init_value=0.2,
                is_output=True,
            ),
            TensorSpec("out_tensor", [out_rows, self.head_dim], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        batch = self.batch
        num_heads = self.num_heads
        head_dim = self.head_dim
        q_tile = self.q_tile

        batch_q_tile = batch * q_tile
        out_rows = batch * num_heads

        @pl.program
        class BatchOnlineUpdateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def KernelOnlineUpdate(  # noqa: PLR0913
                self,
                mij_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
                lij_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
                oi_new_batch: pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
                mi_batch: pl.InOut[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
                li_batch: pl.InOut[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
                oi_batch: pl.InOut[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
                out_tensor: pl.Out[pl.Tensor[[out_rows, head_dim], pl.FP32]],
                is_first: pl.Scalar[pl.INT64],
                is_last: pl.Scalar[pl.INT64],
                batch_count: pl.Scalar[pl.INT64],
                q_offset: pl.Scalar[pl.INT64],
                num_heads_p: pl.Scalar[pl.INT64],
            ) -> tuple[
                pl.Tensor[[batch_q_tile, 1], pl.FP32],
                pl.Tensor[[batch_q_tile, 1], pl.FP32],
                pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
                pl.Tensor[[out_rows, head_dim], pl.FP32],
            ]:
                for b in pl.range(batch_count):
                    dst_row = b * num_heads_p + q_offset

                    if is_first:
                        mij_tile = pl.load(
                            mij_batch,
                            [b * q_tile, 0],
                            [q_tile, 1],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        lij_tile = pl.load(
                            lij_batch,
                            [b * q_tile, 0],
                            [q_tile, 1],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        oi_new_tile = pl.load(
                            oi_new_batch,
                            [b * q_tile, 0],
                            [q_tile, head_dim],
                            target_memory=pl.MemorySpace.Vec,
                        )

                        mi_batch = pl.store(mij_tile, [b * q_tile, 0], [q_tile, 1], mi_batch)
                        li_batch = pl.store(lij_tile, [b * q_tile, 0], [q_tile, 1], li_batch)
                        oi_batch = pl.store(oi_new_tile, [b * q_tile, 0], [q_tile, head_dim], oi_batch)

                        if is_last:
                            dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
                            out_tensor = pl.store(dst_tile, [dst_row, 0], [q_tile, head_dim], out_tensor)
                    else:
                        mij_tile = pl.load(
                            mij_batch,
                            [b * q_tile, 0],
                            [q_tile, 1],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        lij_tile = pl.load(
                            lij_batch,
                            [b * q_tile, 0],
                            [q_tile, 1],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        oi_new_tile = pl.load(
                            oi_new_batch,
                            [b * q_tile, 0],
                            [q_tile, head_dim],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        mi_tile = pl.load(
                            mi_batch,
                            [b * q_tile, 0],
                            [q_tile, 1],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        li_tile = pl.load(
                            li_batch,
                            [b * q_tile, 0],
                            [q_tile, 1],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        oi_tile = pl.load(
                            oi_batch,
                            [b * q_tile, 0],
                            [q_tile, head_dim],
                            target_memory=pl.MemorySpace.Vec,
                        )

                        mi_tile_nd = pl.reshape(mi_tile, [1, q_tile])
                        mij_tile_nd = pl.reshape(mij_tile, [1, q_tile])
                        li_tile_nd = pl.reshape(li_tile, [1, q_tile])
                        lij_tile_nd = pl.reshape(lij_tile, [1, q_tile])

                        mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
                        alpha = pl.exp(pl.sub(mi_tile_nd, mi_new))
                        beta = pl.exp(pl.sub(mij_tile_nd, mi_new))
                        li_updated = pl.add(pl.mul(alpha, li_tile_nd), pl.mul(beta, lij_tile_nd))

                        mi_new_dn = pl.reshape(mi_new, [q_tile, 1])
                        li_updated_dn = pl.reshape(li_updated, [q_tile, 1])

                        mi_batch = pl.store(mi_new_dn, [b * q_tile, 0], [q_tile, 1], mi_batch)
                        li_batch = pl.store(li_updated_dn, [b * q_tile, 0], [q_tile, 1], li_batch)

                        alpha_dn = pl.reshape(alpha, [q_tile, 1])
                        beta_dn = pl.reshape(beta, [q_tile, 1])
                        oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
                        oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
                        oi_updated = pl.add(oi_scaled, oi_new_scaled)

                        if is_last:
                            dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                            out_tensor = pl.store(dst_tile, [dst_row, 0], [q_tile, head_dim], out_tensor)
                        else:
                            oi_batch = pl.store(
                                oi_updated,
                                [b * q_tile, 0],
                                [q_tile, head_dim],
                                oi_batch,
                            )

                return mi_batch, li_batch, oi_batch, out_tensor

            @pl.function(type=pl.FunctionType.Orchestration)
            def Orchestrator(
                self,
                mij_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
                lij_batch: pl.Tensor[[batch_q_tile, 1], pl.FP32],
                oi_new_batch: pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
                config: pl.Tensor[[5], pl.INT64],
                mi_batch: pl.InOut[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
                li_batch: pl.InOut[pl.Tensor[[batch_q_tile, 1], pl.FP32]],
                oi_batch: pl.InOut[pl.Tensor[[batch_q_tile, head_dim], pl.FP32]],
                out_tensor: pl.Out[pl.Tensor[[out_rows, head_dim], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[batch_q_tile, 1], pl.FP32],
                pl.Tensor[[batch_q_tile, 1], pl.FP32],
                pl.Tensor[[batch_q_tile, head_dim], pl.FP32],
                pl.Tensor[[out_rows, head_dim], pl.FP32],
            ]:
                is_first: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                is_last: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                batch_count: pl.Scalar[pl.INT64] = pl.tensor.read(config, [2])
                q_offset: pl.Scalar[pl.INT64] = pl.tensor.read(config, [3])
                num_heads_p: pl.Scalar[pl.INT64] = pl.tensor.read(config, [4])

                mi_batch, li_batch, oi_batch, out_tensor = self.KernelOnlineUpdate(
                    mij_batch,
                    lij_batch,
                    oi_new_batch,
                    mi_batch,
                    li_batch,
                    oi_batch,
                    out_tensor,
                    is_first,
                    is_last,
                    batch_count,
                    q_offset,
                    num_heads_p,
                )
                return mi_batch, li_batch, oi_batch, out_tensor

        return BatchOnlineUpdateProgram

    def compute_expected(self, tensors, params=None):
        config = tensors["config"]
        is_first = bool(int(config[0].item()))
        is_last = bool(int(config[1].item()))
        batch_count = int(config[2].item())
        q_offset = int(config[3].item())
        num_heads = int(config[4].item())

        q_tile = 16

        mij_in = tensors["mij_batch"]
        lij_in = tensors["lij_batch"]
        oi_new_in = tensors["oi_new_batch"]

        for b in range(batch_count):
            dst_row = b * num_heads + q_offset
            bs = b * q_tile
            be = (b + 1) * q_tile

            mij = mij_in[bs:be, :]
            lij = lij_in[bs:be, :]
            oi_new = oi_new_in[bs:be, :]

            if is_first:
                tensors["mi_batch"][bs:be, :] = mij.clone()
                tensors["li_batch"][bs:be, :] = lij.clone()
                tensors["oi_batch"][bs:be, :] = oi_new.clone()

                if is_last:
                    tensors["out_tensor"][dst_row : dst_row + q_tile, :] = oi_new / lij
            else:
                mi = tensors["mi_batch"][bs:be, :].clone()
                li = tensors["li_batch"][bs:be, :].clone()
                oi = tensors["oi_batch"][bs:be, :].clone()

                mi_new = torch.maximum(mi, mij)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(mij - mi_new)
                li_updated = alpha * li + beta * lij
                oi_updated = alpha * oi + beta * oi_new

                tensors["mi_batch"][bs:be, :] = mi_new
                tensors["li_batch"][bs:be, :] = li_updated

                if is_last:
                    tensors["out_tensor"][dst_row : dst_row + q_tile, :] = oi_updated / li_updated
                else:
                    tensors["oi_batch"][bs:be, :] = oi_updated


# ---------------------------------------------------------------------------
# Full Batch Paged Attention (integration test)
# ---------------------------------------------------------------------------
class BatchPagedAttentionTestCase(PTOTestCase):
    """Integration test for the full batch paged attention pipeline.

    Delegates program construction to BuildBatchPagedAttentionProgram so the ST
    always exercises the same program definition as the example.

    Tensor layout (all 2D, flattened):
      query:       [batch * num_heads, head_dim]                    BF16
      key_cache:   [total_pool_blocks * block_size, head_dim]       BF16
      value_cache: [total_pool_blocks * block_size, head_dim]       BF16
      out:         [batch * num_heads, head_dim]                    FP32
    """

    def __init__(
        self,
        batch: int = 2,
        num_heads: int = 16,
        head_dim: int = 16,
        block_size: int = 16,
        context_len: int = 64,
        max_model_len: int = 256,
        scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config.atol = 1e-3
        self.config.rtol = 1e-3
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.context_len = context_len
        self.max_model_len = max_model_len
        self.scale = scale
        self.max_num_blocks_per_req = max_model_len // block_size

    def get_name(self) -> str:
        return f"batch_paged_attention_{self.batch}bat_{self.num_heads}h_{self.head_dim}d_{self.block_size}bs"

    def define_tensors(self) -> list[TensorSpec]:
        b = self.batch
        h = self.num_heads
        d = self.head_dim
        bs = self.block_size
        max_blocks = self.max_num_blocks_per_req
        total_pool_rows = b * max_blocks * bs

        scale_bits = struct.unpack("I", struct.pack("f", self.scale))[0]
        config = torch.tensor(
            [b, h, 1, d, bs, max_blocks, scale_bits],
            dtype=torch.int64,
        )
        block_table = torch.randint(
            0, max(b * max_blocks, 1), size=(b, max_blocks), dtype=torch.int32
        ).flatten()
        context_lens = torch.full((b,), self.context_len, dtype=torch.int32)

        return [
            TensorSpec("query", [b * h, d], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [total_pool_rows, d], DataType.BF16, init_value=torch.randn),
            TensorSpec("value_cache", [total_pool_rows, d], DataType.BF16, init_value=torch.randn),
            TensorSpec("block_table", [b * max_blocks], DataType.INT32, init_value=block_table),
            TensorSpec("context_lens", [b], DataType.INT32, init_value=context_lens),
            TensorSpec("out", [b * h, d], DataType.FP32, is_output=True),
            TensorSpec("config", [7], DataType.INT64, init_value=config),
            TensorSpec(
                "size_query",
                [1],
                DataType.INT64,
                init_value=torch.tensor([b * h * d * 2], dtype=torch.int64),
            ),
            TensorSpec(
                "size_key_cache",
                [1],
                DataType.INT64,
                init_value=torch.tensor([total_pool_rows * d * 2], dtype=torch.int64),
            ),
            TensorSpec(
                "size_value_cache",
                [1],
                DataType.INT64,
                init_value=torch.tensor([total_pool_rows * d * 2], dtype=torch.int64),
            ),
        ]

    def get_program(self) -> Any:
        return BuildBatchPagedAttentionProgram(
            batch=self.batch,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
        )

    def compute_expected(self, tensors, params=None):
        config = tensors["config"]
        batch = int(config[0].item())
        num_heads = int(config[1].item())
        head_dim = int(config[3].item())
        block_size = int(config[4].item())
        max_num_blocks_per_req = int(config[5].item())
        scale_bits = int(config[6].item())
        scale_value = struct.unpack("f", struct.pack("I", scale_bits & 0xFFFFFFFF))[0]

        query = tensors["query"].float().reshape(batch, num_heads, head_dim)
        total_pool_blocks = batch * max_num_blocks_per_req
        key_cache = tensors["key_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
        value_cache = tensors["value_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
        block_table = tensors["block_table"].reshape(batch, max_num_blocks_per_req)
        context_lens = tensors["context_lens"]

        out = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
        q_tile = 16
        max_bn = int((context_lens.max().item() + block_size - 1) // block_size)

        for q_offset in range(0, num_heads, q_tile):
            q_tile_size = min(q_tile, num_heads - q_offset)
            qi = query[:, q_offset : q_offset + q_tile_size, :]
            oi, li, mi = None, None, None

            for bn in range(max_bn):
                valid_lens = torch.clamp(context_lens - bn * block_size, min=0, max=block_size)
                if not (valid_lens > 0).any():
                    break
                block_indices = block_table[:, bn]
                kj_all = key_cache[block_indices].float()
                vj_all = value_cache[block_indices].float()
                sij = torch.bmm(qi, kj_all.transpose(1, 2)) * scale_value
                pos = torch.arange(block_size).unsqueeze(0)
                valid_mask = (pos < valid_lens.unsqueeze(1)).unsqueeze(1)
                sij = sij.masked_fill(~valid_mask, float("-inf"))
                mij = sij.max(dim=-1, keepdim=True)[0].clamp(min=-1e30)
                pij = torch.exp(sij - mij).masked_fill(~valid_mask, 0.0)
                pij = pij.to(torch.float16).to(torch.float32)
                lij = pij.sum(dim=-1, keepdim=True)
                oi_new = torch.bmm(pij, vj_all)
                if bn == 0:
                    oi, li, mi = oi_new, lij, mij
                else:
                    mi_new = torch.maximum(mi, mij)
                    alpha = torch.exp(mi - mi_new)
                    beta = torch.exp(mij - mi_new)
                    li = alpha * li + beta * lij
                    oi = alpha * oi + beta * oi_new
                    mi = mi_new

            out[:, q_offset : q_offset + q_tile_size, :] = oi / li

        tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


# ---------------------------------------------------------------------------
# Pytest test class
# ---------------------------------------------------------------------------
class TestBatchPagedAttentionKernels:
    """Tests for the four batched Paged Attention kernels.

    Each test instantiates the corresponding PTOTestCase and runs it through
    the test_runner fixture, which handles kernel compilation and result
    validation against compute_expected.
    """

    @pytest.mark.parametrize("batch,num_heads,head_dim,block_size", [(2, 16, 16, 16)])
    def test_batch_qk_matmul(self, test_runner, batch, num_heads, head_dim, block_size):
        test_case = BatchQKMatmulTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"Batch QK matmul test failed: {result.error}"

    @pytest.mark.skip(reason="Under debugging and fixing")
    @pytest.mark.parametrize(
        "batch,block_size,block_idx,context_lens",
        [
            (2, 16, 1, [20, 24]),
        ],
    )
    def test_batch_softmax_prepare(self, test_runner, batch, block_size, block_idx, context_lens):
        test_case = BatchSoftmaxPrepareTestCase(
            batch=batch,
            block_size=block_size,
            block_idx=block_idx,
            context_lens=context_lens,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"Batch softmax prepare test failed: {result.error}"

    @pytest.mark.parametrize("batch,num_heads,head_dim,block_size", [(2, 16, 16, 16)])
    def test_batch_pv_matmul(self, test_runner, batch, num_heads, head_dim, block_size):
        test_case = BatchPVMatmulTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"Batch PV matmul test failed: {result.error}"

    @pytest.mark.skip(reason="Under debugging and fixing")
    @pytest.mark.parametrize(
        "batch,num_heads,head_dim,is_first,is_last",
        [
            (2, 16, 16, 1, 1),
            (2, 16, 16, 1, 0),
            (2, 16, 16, 0, 1),
            (2, 16, 16, 0, 0),
        ],
    )
    def test_batch_online_update(self, test_runner, batch, num_heads, head_dim, is_first, is_last):
        test_case = BatchOnlineUpdateTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            is_first=is_first,
            is_last=is_last,
        )
        result = test_runner.run(test_case)
        assert result.passed, (
            f"Batch online update test failed (is_first={is_first}, is_last={is_last}): {result.error}"
        )

    @pytest.mark.skip(reason="Under debugging and fixing")
    @pytest.mark.parametrize(
        "batch,num_heads,head_dim,block_size,context_len,max_model_len",
        [
            (2, 16, 16, 16, 64, 256),
        ],
    )
    def test_batch_paged_attention(
        self, test_runner, batch, num_heads, head_dim, block_size, context_len, max_model_len
    ):
        test_case = BatchPagedAttentionTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            context_len=context_len,
            max_model_len=max_model_len,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"Batch paged attention test failed: {result.error}"
