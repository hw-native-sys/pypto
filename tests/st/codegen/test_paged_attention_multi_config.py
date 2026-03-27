# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for Paged Attention Multi-Config implementation using PyPTO frontend.

Multi-config interface: N_UNROLL=8, BLOCK_SIZE=64, HEAD_DIM=128.
Orchestration pre-extracts block_indices via pl.slice() from block_table.
Kernels receive block_indices tensor view instead of block_table + bt_offset.

Module-level InCore kernels:
  kernel_aiv_hub:         Zero-initialise oi, li, mi accumulators
  kernel_softmax_prepare: Two-pass softmax across n_blocks (global row_max, then exp+sum)
  kernel_online_update:   Online softmax update with inplace mi/li/oi

Factory-generated InCore kernels:
  make_kernel_qk_matmul(key_cache_rows):  Multi-block QK matmul
  make_kernel_pv_matmul(key_cache_rows):   SplitK PV matmul
"""

import struct
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

from examples.ir_parser.paged_attention_multi_config_example import (
    BLOCK_SIZE,
    HEAD_DIM,
    N_UNROLL,
    N_UNROLL_Q,
    Q_TILE,
    build_paged_attention_multi_config_program,
    kernel_aiv_hub,
    kernel_online_update,
    kernel_softmax_prepare,
    make_kernel_pv_matmul,
    make_kernel_qk_matmul,
)


class AivHubTestCase(PTOTestCase):
    """Test case for kernel_aiv_hub: zero-initialise oi, li, mi accumulators.

    Verifies that all three output tensors are set to zero.
    """

    def get_name(self) -> str:
        return "aiv_hub_zero_init"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("oi", [Q_TILE, HEAD_DIM], DataType.FP32, is_output=True),
            TensorSpec("li", [Q_TILE, 1], DataType.FP32, is_output=True),
            TensorSpec("mi", [Q_TILE, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class AivHubProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                oi: pl.Out[pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]],
                li: pl.Out[pl.Tensor[[Q_TILE, 1], pl.FP32]],
                mi: pl.Out[pl.Tensor[[Q_TILE, 1], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
            ]:
                oi, li, mi = kernel_aiv_hub(oi, li, mi)
                return oi, li, mi

        return AivHubProgram

    def compute_expected(self, tensors, params=None):
        tensors["oi"][:] = torch.zeros(16, 128, dtype=torch.float32)
        tensors["li"][:] = torch.zeros(16, 1, dtype=torch.float32)
        tensors["mi"][:] = torch.zeros(16, 1, dtype=torch.float32)


class SoftmaxPrepareTestCase(PTOTestCase):
    """Test case for kernel_softmax_prepare with two-pass softmax.

    Uses N_UNROLL=8 block grouping with global row_max across all blocks,
    then exp with that max and row_sum accumulation.
    """

    def __init__(self, n_blocks: int = 2, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.scale = scale
        # BF16 exp+cast in softmax produces ~1e-3 precision; default 1e-5 is too tight
        self.config.atol = 1e-3
        self.config.rtol = 1e-3

    def get_name(self) -> str:
        return f"softmax_prepare_nb{self.n_blocks}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("sij_buf", [N_UNROLL_Q, BLOCK_SIZE], DataType.FP32, init_value=torch.randn),
            TensorSpec("config", [2], DataType.FP32, init_value=torch.tensor([self.scale, 0.0])),
            TensorSpec("pij_buf", [N_UNROLL_Q, BLOCK_SIZE], DataType.BF16, is_output=True),
            TensorSpec("mi_out", [Q_TILE, 1], DataType.FP32, is_output=True),
            TensorSpec("li_out", [Q_TILE, 1], DataType.FP32, is_output=True),
            TensorSpec(
                "n_blocks_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([self.n_blocks], dtype=torch.int64),
            ),
        ]

    def get_program(self) -> Any:
        @pl.program
        class SoftmaxPrepareProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                sij_buf: pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32],
                config: pl.Tensor[[2], pl.FP32],
                pij_buf: pl.Out[pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16]],
                mi_out: pl.InOut[pl.Tensor[[Q_TILE, 1], pl.FP32]],
                li_out: pl.InOut[pl.Tensor[[Q_TILE, 1], pl.FP32]],
                n_blocks_cfg: pl.Tensor[[1], pl.INT64],
            ) -> tuple[
                pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
            ]:
                scale: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                n_blocks: pl.Scalar[pl.INT64] = pl.tensor.read(n_blocks_cfg, [0])
                pij_buf, mi_out, li_out = kernel_softmax_prepare(
                    sij_buf,
                    scale,
                    pij_buf,
                    mi_out,
                    li_out,
                    n_blocks,
                )
                return pij_buf, mi_out, li_out

        return SoftmaxPrepareProgram

    def compute_expected(self, tensors, params=None):
        q_tile = 16
        block_size = 64
        n_unroll_q = 1024
        sij_buf = tensors["sij_buf"].float()
        scale = float(tensors["config"][0].item())
        n_blocks = int(tensors["n_blocks_cfg"][0].item())

        # Pass 1: find global row_max across all blocks
        global_max = None
        for i in range(n_blocks):
            s_tile = sij_buf[i * q_tile : (i + 1) * q_tile, :]
            scaled = s_tile * scale
            local_max = scaled.max(dim=-1, keepdim=True)[0]
            if global_max is None:
                global_max = local_max
            else:
                global_max = torch.maximum(global_max, local_max)

        # Pass 2: exp(scaled - global_max), cast to bf16, row_sum
        li_acc = None
        pij_out = torch.zeros(n_unroll_q, block_size, dtype=torch.bfloat16)
        for i in range(n_blocks):
            s_tile = sij_buf[i * q_tile : (i + 1) * q_tile, :]
            scaled = s_tile * scale
            centered = scaled - global_max
            exp_tile = torch.exp(centered)
            pij_bf16 = exp_tile.to(torch.bfloat16)
            pij_f32 = pij_bf16.to(torch.float32)
            pij_out[i * q_tile : (i + 1) * q_tile, :] = pij_bf16
            li_local = pij_f32.sum(dim=-1, keepdim=True)
            if li_acc is None:
                li_acc = li_local
            else:
                li_acc = li_acc + li_local

        tensors["pij_buf"][:] = pij_out
        tensors["mi_out"][:] = global_max
        tensors["li_out"][:] = li_acc


class QKMatmulTestCase(PTOTestCase):
    """Test case for multi-block QK matmul with pre-extracted block indices.

    Computes: sij[i] = qi @ key_cache[block_indices[i]].T for i in range(n_blocks)
    Results are vertically stacked in sij_buf.
    """

    def __init__(self, n_blocks: int = 2, num_cache_blocks: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.num_cache_blocks = num_cache_blocks
        self.key_cache_rows = num_cache_blocks * BLOCK_SIZE

    def get_name(self) -> str:
        return f"qk_matmul_nb{self.n_blocks}"

    def define_tensors(self) -> list[TensorSpec]:
        block_indices = torch.randint(0, self.num_cache_blocks, (N_UNROLL,), dtype=torch.int32)
        return [
            TensorSpec("qi", [Q_TILE, HEAD_DIM], DataType.BF16, init_value=torch.randn),
            TensorSpec(
                "key_cache",
                [self.key_cache_rows, HEAD_DIM],
                DataType.BF16,
                init_value=torch.randn,
            ),
            TensorSpec("sij_buf", [N_UNROLL_Q, BLOCK_SIZE], DataType.FP32, is_output=True),
            TensorSpec("block_indices", [N_UNROLL], DataType.INT32, init_value=block_indices),
            TensorSpec(
                "n_blocks_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([self.n_blocks], dtype=torch.int64),
            ),
        ]

    def get_program(self) -> Any:
        key_cache_rows = self.key_cache_rows
        kernel_qk = make_kernel_qk_matmul(key_cache_rows)

        @pl.program
        class QKMatmulProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                qi: pl.Tensor[[Q_TILE, HEAD_DIM], pl.BF16],
                key_cache: pl.Tensor[[key_cache_rows, HEAD_DIM], pl.BF16],
                sij_buf: pl.Out[pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32]],
                block_indices: pl.Tensor[[N_UNROLL], pl.INT32],
                n_blocks_cfg: pl.Tensor[[1], pl.INT64],
            ) -> pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32]:
                n_blocks: pl.Scalar[pl.INT64] = pl.tensor.read(n_blocks_cfg, [0])
                sij_buf = kernel_qk(qi, key_cache, sij_buf, block_indices, n_blocks)
                return sij_buf

        return QKMatmulProgram

    def compute_expected(self, tensors, params=None):
        q_tile = 16
        block_size = 64
        head_dim = 128
        n_unroll_q = 1024
        qi = tensors["qi"].float()
        key_cache = tensors["key_cache"].float().reshape(-1, block_size, head_dim)
        block_indices = tensors["block_indices"]
        n_blocks = int(tensors["n_blocks_cfg"][0].item())

        sij_buf = torch.zeros(n_unroll_q, block_size, dtype=torch.float32)
        for i in range(n_blocks):
            bidx = int(block_indices[i].item())
            kj = key_cache[bidx]  # [block_size, head_dim]
            sij = torch.mm(qi, kj.T)  # [q_tile, block_size]
            sij_buf[i * q_tile : (i + 1) * q_tile, :] = sij
        tensors["sij_buf"][:] = sij_buf


class PVMatmulTestCase(PTOTestCase):
    """Test case for SplitK PV matmul with pre-extracted block indices.

    First block via matmul, remaining blocks via matmul_acc.
    Accumulates pij[i] @ vj[block_indices[i]] across n_blocks.
    """

    def __init__(self, n_blocks: int = 2, num_cache_blocks: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.num_cache_blocks = num_cache_blocks
        self.key_cache_rows = num_cache_blocks * BLOCK_SIZE

    def get_name(self) -> str:
        return f"pv_matmul_nb{self.n_blocks}"

    def define_tensors(self) -> list[TensorSpec]:
        block_indices = torch.randint(0, self.num_cache_blocks, (N_UNROLL,), dtype=torch.int32)
        return [
            TensorSpec("pij_buf", [N_UNROLL_Q, BLOCK_SIZE], DataType.BF16, init_value=torch.randn),
            TensorSpec(
                "value_cache",
                [self.key_cache_rows, HEAD_DIM],
                DataType.BF16,
                init_value=torch.randn,
            ),
            TensorSpec("oi_new", [Q_TILE, HEAD_DIM], DataType.FP32, is_output=True),
            TensorSpec("block_indices", [N_UNROLL], DataType.INT32, init_value=block_indices),
            TensorSpec(
                "n_blocks_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([self.n_blocks], dtype=torch.int64),
            ),
        ]

    def get_program(self) -> Any:
        key_cache_rows = self.key_cache_rows
        kernel_pv = make_kernel_pv_matmul(key_cache_rows)

        @pl.program
        class PVMatmulProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                pij_buf: pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16],
                value_cache: pl.Tensor[[key_cache_rows, HEAD_DIM], pl.BF16],
                oi_new: pl.Out[pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]],
                block_indices: pl.Tensor[[N_UNROLL], pl.INT32],
                n_blocks_cfg: pl.Tensor[[1], pl.INT64],
            ) -> pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]:
                n_blocks: pl.Scalar[pl.INT64] = pl.tensor.read(n_blocks_cfg, [0])
                oi_new = kernel_pv(pij_buf, value_cache, oi_new, block_indices, n_blocks)
                return oi_new

        return PVMatmulProgram

    def compute_expected(self, tensors, params=None):
        head_dim = 128
        block_size = 64
        q_tile = 16
        pij_buf = tensors["pij_buf"].float()
        value_cache = tensors["value_cache"].float().reshape(-1, block_size, head_dim)
        block_indices = tensors["block_indices"]
        n_blocks = int(tensors["n_blocks_cfg"][0].item())

        oi = torch.zeros(q_tile, head_dim, dtype=torch.float32)
        for i in range(n_blocks):
            bidx = int(block_indices[i].item())
            pij_tile = pij_buf[i * q_tile : (i + 1) * q_tile, :]  # [q_tile, block_size]
            vj = value_cache[bidx]  # [block_size, head_dim]
            oi += torch.mm(pij_tile, vj)
        tensors["oi_new"][:] = oi


class OnlineUpdateTestCase(PTOTestCase):
    """Test case for kernel_online_update with INDEX-typed is_first/is_last flags.

    Four flag combinations:
      - is_first=1, is_last=1: copy + normalize dst = oi_new / lij
      - is_first=1, is_last=0: copy; dst = zeros
      - is_first=0, is_last=1: full online update + normalize
      - is_first=0, is_last=0: full online update; dst = zeros
    """

    def __init__(self, is_first: int = 0, is_last: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.is_first = is_first
        self.is_last = is_last

    def get_name(self) -> str:
        return f"online_update_f{self.is_first}_l{self.is_last}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("mij", [Q_TILE, 1], DataType.FP32, init_value=0.5),
            TensorSpec("lij", [Q_TILE, 1], DataType.FP32, init_value=1.5),
            TensorSpec("oi_new", [Q_TILE, HEAD_DIM], DataType.FP32, init_value=0.3),
            TensorSpec(
                "config",
                [2],
                DataType.INT64,
                init_value=torch.tensor([self.is_first, self.is_last], dtype=torch.int64),
            ),
            TensorSpec("mi", [Q_TILE, 1], DataType.FP32, init_value=0.4, is_output=True),
            TensorSpec("li", [Q_TILE, 1], DataType.FP32, init_value=2.0, is_output=True),
            TensorSpec("oi", [Q_TILE, HEAD_DIM], DataType.FP32, init_value=0.2, is_output=True),
            TensorSpec("dst", [Q_TILE, HEAD_DIM], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class OnlineUpdateProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mij: pl.Tensor[[Q_TILE, 1], pl.FP32],
                lij: pl.Tensor[[Q_TILE, 1], pl.FP32],
                oi_new: pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
                config: pl.Tensor[[2], pl.INT64],
                mi: pl.InOut[pl.Tensor[[Q_TILE, 1], pl.FP32]],
                li: pl.InOut[pl.Tensor[[Q_TILE, 1], pl.FP32]],
                oi: pl.InOut[pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]],
                dst: pl.Out[pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[Q_TILE, 1], pl.FP32],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
                pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
                pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
            ]:
                is_first: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                is_last: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                mi, li, oi, dst = kernel_online_update(
                    mij,
                    lij,
                    oi_new,
                    mi,
                    li,
                    oi,
                    dst,
                    is_first,
                    is_last,
                )
                return mi, li, oi, dst

        return OnlineUpdateProgram

    def compute_expected(self, tensors, params=None):
        is_first = bool(int(tensors["config"][0]))
        is_last = bool(int(tensors["config"][1]))

        mij = tensors["mij"]
        lij = tensors["lij"]
        oi_new = tensors["oi_new"]
        mi = tensors["mi"]
        li = tensors["li"]
        oi = tensors["oi"]

        if is_first:
            tensors["mi"][:] = mij
            tensors["li"][:] = lij
            tensors["oi"][:] = oi_new
            if is_last:
                tensors["dst"][:] = oi_new / lij
            else:
                tensors["dst"][:] = torch.zeros_like(tensors["dst"])
        else:
            mi_new = torch.maximum(mi, mij)
            alpha = torch.exp(mi - mi_new)
            beta = torch.exp(mij - mi_new)
            li_updated = alpha * li + beta * lij
            oi_updated = alpha * oi + beta * oi_new

            tensors["mi"][:] = mi_new
            tensors["li"][:] = li_updated
            tensors["oi"][:] = oi_updated

            if is_last:
                tensors["dst"][:] = oi_updated / li_updated
            else:
                tensors["dst"][:] = torch.zeros_like(oi_new)


class PagedAttentionMultiConfigTestCase(PTOTestCase):
    """Test case for full paged attention multi-config program.

    Delegates program construction to paged_attention_multi_config_example.py so
    that the ST always exercises the same program definition as the example.
    """

    def __init__(
        self,
        batch: int = 4,
        num_heads: int = 16,
        head_dim: int = HEAD_DIM,
        block_size: int = BLOCK_SIZE,
        context_len: int = 1024,
        max_model_len: int = 2048,
        scale: float = 1.0,
        q_tile: int = Q_TILE,
        n_unroll: int = N_UNROLL,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config.atol = 2e-2
        self.config.rtol = 2e-2
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.context_len = context_len
        self.max_model_len = max_model_len
        self.scale = scale
        self.q_tile = q_tile
        self.n_unroll = n_unroll
        self.max_num_blocks_per_req = max_model_len // block_size

    def get_name(self) -> str:
        return (
            f"paged_attention_multi_config"
            f"_{self.batch}bat_{self.num_heads}h_{self.head_dim}d_{self.block_size}bs"
        )

    def define_tensors(self) -> list[TensorSpec]:
        B = self.batch
        H = self.num_heads
        D = self.head_dim
        BS = self.block_size
        max_blocks = self.max_num_blocks_per_req
        total_pool_rows = B * max_blocks * BS

        scale_bits = struct.unpack("I", struct.pack("f", self.scale))[0]
        config = torch.tensor(
            [B, H, 1, D, BS, max_blocks, scale_bits, self.q_tile, self.n_unroll],
            dtype=torch.int64,
        )
        block_table = torch.randint(
            0,
            max(B * max_blocks, 1),
            size=(B, max_blocks),
            dtype=torch.int32,
        ).flatten()
        context_lens = torch.full((B,), self.context_len, dtype=torch.int32)

        query_rows = B * H
        key_cache_rows = total_pool_rows
        block_table_flat_size = B * max_blocks

        size_query = torch.tensor([query_rows * D * 2], dtype=torch.int64)
        size_key_cache = torch.tensor([key_cache_rows * D * 2], dtype=torch.int64)
        size_value_cache = torch.tensor([key_cache_rows * D * 2], dtype=torch.int64)

        return [
            TensorSpec("query", [query_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [key_cache_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("value_cache", [key_cache_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("block_table", [block_table_flat_size], DataType.INT32, init_value=block_table),
            TensorSpec("context_lens", [B], DataType.INT32, init_value=context_lens),
            TensorSpec("out", [query_rows, D], DataType.FP32, is_output=True),
            TensorSpec("config", [9], DataType.INT64, init_value=config),
            TensorSpec("size_query", [1], DataType.INT64, init_value=size_query),
            TensorSpec("size_key_cache", [1], DataType.INT64, init_value=size_key_cache),
            TensorSpec("size_value_cache", [1], DataType.INT64, init_value=size_value_cache),
        ]

    def get_program(self) -> Any:
        return build_paged_attention_multi_config_program(
            batch=self.batch,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            context_len=self.context_len,
            q_tile=self.q_tile,
            n_unroll=self.n_unroll,
        )

    def compute_expected(self, tensors, params=None):
        config = tensors["config"]
        batch = int(config[0].item())
        num_heads = int(config[1].item())
        head_dim = int(config[3].item())
        block_size = int(config[4].item())
        max_num_blocks_per_req = int(config[5].item())
        scale_bits = int(config[6].item())
        scale = struct.unpack("f", struct.pack("I", scale_bits & 0xFFFFFFFF))[0]

        query = tensors["query"].float().reshape(batch, num_heads, head_dim)
        total_pool_blocks = batch * max_num_blocks_per_req
        key_cache = tensors["key_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
        value_cache = tensors["value_cache"].float().reshape(total_pool_blocks, block_size, head_dim)
        block_table = tensors["block_table"].reshape(batch, max_num_blocks_per_req)
        context_lens = tensors["context_lens"]

        out = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
        q_tile = int(config[7].item())
        n_unroll = int(config[8].item())

        def _update(oi_a, li_a, mi_a, oi_new, li_new, mi_new):
            if oi_a is None or li_a is None or mi_a is None:
                return oi_new, li_new, mi_new
            mi_u = torch.maximum(mi_a, mi_new)
            a = torch.exp(mi_a - mi_u)
            b_ = torch.exp(mi_new - mi_u)
            return a * oi_a + b_ * oi_new, a * li_a + b_ * li_new, mi_u

        for b in range(batch):
            cur_seq = int(context_lens[b].item())
            max_bn_b = (cur_seq + block_size - 1) // block_size

            for q_idx in range(num_heads // q_tile):
                q_off = q_idx * q_tile
                qi = query[b, q_off : q_off + q_tile, :]

                oi_acc, li_acc, mi_acc = None, None, None

                for bn in range(0, max_bn_b, n_unroll):
                    n_blocks = min(n_unroll, max_bn_b - bn)

                    all_sij = []
                    for i in range(n_blocks):
                        bidx = int(block_table[b, bn + i].item())
                        kj = key_cache[bidx]
                        sij = torch.mm(qi, kj.T) * scale
                        all_sij.append(sij)

                    global_max = all_sij[0].max(dim=-1, keepdim=True)[0]
                    for sij in all_sij[1:]:
                        local_max = sij.max(dim=-1, keepdim=True)[0]
                        global_max = torch.maximum(global_max, local_max)
                    global_max = global_max.clamp(min=-1e30)

                    li_group = torch.zeros(q_tile, 1)
                    oi_group = torch.zeros(q_tile, head_dim, dtype=torch.float32)
                    for i, sij in enumerate(all_sij):
                        pij = torch.exp(sij - global_max).to(torch.bfloat16).to(torch.float32)
                        li_group += pij.sum(dim=-1, keepdim=True)
                        bidx = int(block_table[b, bn + i].item())
                        vj = value_cache[bidx]
                        oi_group += torch.mm(pij, vj)

                    oi_acc, li_acc, mi_acc = _update(
                        oi_acc,
                        li_acc,
                        mi_acc,
                        oi_group,
                        li_group,
                        global_max,
                    )

                assert oi_acc is not None and li_acc is not None
                out[b, q_off : q_off + q_tile, :] = oi_acc / li_acc

        tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


# ── PTOAS mixin and variants ────────────────────────────────────────────────


class PTOASTestCaseMixin:
    """Mixin for test cases using PTO backend and Default optimization strategy."""

    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO


class AivHubPTOASTestCase(PTOASTestCaseMixin, AivHubTestCase):
    """Test aiv_hub with PTO backend and Default optimization strategy."""

    def get_name(self) -> str:
        return "aiv_hub_ptoas_zero_init"


class SoftmaxPreparePTOASTestCase(PTOASTestCaseMixin, SoftmaxPrepareTestCase):
    """Test softmax prepare with PTO backend and Default optimization strategy."""

    def get_name(self) -> str:
        return f"softmax_prepare_ptoas_nb{self.n_blocks}"


class QKMatmulPTOASTestCase(PTOASTestCaseMixin, QKMatmulTestCase):
    """Test QK matmul with PTO backend and Default optimization strategy."""

    def get_name(self) -> str:
        return f"qk_matmul_ptoas_nb{self.n_blocks}"


class PVMatmulPTOASTestCase(PTOASTestCaseMixin, PVMatmulTestCase):
    """Test PV matmul with PTO backend and Default optimization strategy."""

    def get_name(self) -> str:
        return f"pv_matmul_ptoas_nb{self.n_blocks}"


class OnlineUpdatePTOASTestCase(PTOASTestCaseMixin, OnlineUpdateTestCase):
    """Test online update with PTO backend and Default optimization strategy."""

    def get_name(self) -> str:
        return f"online_update_ptoas_f{self.is_first}_l{self.is_last}"


class PagedAttentionMultiConfigPTOASTestCase(PTOASTestCaseMixin, PagedAttentionMultiConfigTestCase):
    """Test paged attention multi-config with PTO backend and Default optimization strategy."""

    def get_name(self) -> str:
        return (
            f"paged_attention_multi_config_ptoas_{self.batch}bat_{self.num_heads}h_"
            f"{self.head_dim}d_{self.block_size}bs"
        )


# ── Test class ───────────────────────────────────────────────────────────────


class TestPagedAttentionMultiConfigKernels:
    """Integration tests for the Paged Attention Multi-Config kernels.

    Each test instantiates the corresponding PTOTestCase and runs it through
    the test_runner fixture.
    """

    def test_aiv_hub_ptoas(self, test_runner):
        """Test aiv_hub with PTO backend and Default optimization."""
        test_case = AivHubPTOASTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Aiv hub PTOAS test failed: {result.error}"

    @pytest.mark.parametrize("n_blocks", [1, 2, 4])
    def test_softmax_prepare_ptoas(self, test_runner, n_blocks):
        """Test softmax prepare with PTO backend and Default optimization."""
        test_case = SoftmaxPreparePTOASTestCase(n_blocks=n_blocks)
        result = test_runner.run(test_case)
        assert result.passed, f"Softmax prepare PTOAS test failed (n_blocks={n_blocks}): {result.error}"

    @pytest.mark.parametrize("n_blocks", [1, 2, 4])
    def test_qk_matmul_ptoas(self, test_runner, n_blocks):
        """Test QK matmul with PTO backend and Default optimization."""
        test_case = QKMatmulPTOASTestCase(n_blocks=n_blocks)
        result = test_runner.run(test_case)
        assert result.passed, f"QK matmul PTOAS test failed (n_blocks={n_blocks}): {result.error}"

    @pytest.mark.parametrize("n_blocks", [1, 2, 4])
    def test_pv_matmul_ptoas(self, test_runner, n_blocks):
        """Test PV matmul with PTO backend and Default optimization."""
        test_case = PVMatmulPTOASTestCase(n_blocks=n_blocks)
        result = test_runner.run(test_case)
        assert result.passed, f"PV matmul PTOAS test failed (n_blocks={n_blocks}): {result.error}"

    @pytest.mark.parametrize(
        "is_first,is_last",
        [
            (1, 1),  # single group: first + last
            (1, 0),  # first group, more to come
            (0, 1),  # last group
            (0, 0),  # middle group
        ],
    )
    def test_online_update_ptoas(self, test_runner, is_first, is_last):
        """Test online update with PTO backend and Default optimization."""
        test_case = OnlineUpdatePTOASTestCase(is_first=is_first, is_last=is_last)
        result = test_runner.run(test_case)
        assert result.passed, (
            f"Online update PTOAS test failed (is_first={is_first}, is_last={is_last}): {result.error}"
        )

    @pytest.mark.parametrize(
        "batch,num_heads,head_dim,block_size,context_len,max_model_len,q_tile,n_unroll",
        [
            # Original small-scale case
            (4, 16, 128, 64, 1024, 2048, 16, 64),
            # Case1-aligned: block_size=128, q_tile=16
            (4, 16, 128, 128, 1024, 2048, 16, 8),
            # Case2-aligned: num_heads=64, q_tile=64
            (4, 64, 128, 64, 1024, 2048, 64, 8),
            # Case3-aligned: head_dim=256, q_tile=32 (q_tile=64 exceeds Vec limit)
            (4, 64, 256, 64, 1024, 2048, 32, 8),
        ],
    )
    def test_paged_attention_multi_config_ptoas(
        self,
        test_runner,
        batch,
        num_heads,
        head_dim,
        block_size,
        context_len,
        max_model_len,
        q_tile,
        n_unroll,
    ):
        """Test paged attention multi-config with PTO backend and Default optimization."""
        test_case = PagedAttentionMultiConfigPTOASTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            context_len=context_len,
            max_model_len=max_model_len,
            q_tile=q_tile,
            n_unroll=n_unroll,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"Paged attention multi-config PTOAS test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
