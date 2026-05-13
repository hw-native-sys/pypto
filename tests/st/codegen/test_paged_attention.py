# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for Paged Attention implementation using PyPTO frontend.

QK Matmul Kernel:
  Computes: sij = qi @ kj_t                           -> (num_heads, num_heads)

Softmax Prepare Kernel (aiv_softmax_prepare.cpp):
  Computes: sij_scaled = sij * scale
            mij = row_max(sij_scaled)                 -> (num_heads, 1)
            pij = exp(sij_scaled - mij)               -> (num_heads, block_size)
            lij = row_sum(pij)                        -> (num_heads, 1)

PV Matmul Kernel:
  Computes: oi_new = pij @ vj                         -> (num_heads, head_dim)

Online Update Kernel (aiv_online_update.cpp):
  - is_first=1, is_last=0: Copy mij->mi, lij->li, oi_new->oi (first block, more to come)
  - is_first=1, is_last=1: Copy + normalize dst = oi_new / lij (single block case)
  - is_first=0, is_last=0: Full online update, store oi (middle blocks)
  - is_first=0, is_last=1: Full online update + normalize dst = oi_updated / li_updated (last block)
"""

import struct
from typing import Any

import pypto.language as pl
import pytest
import torch
from examples.models.paged_attention import (
    build_paged_attention_program,
    build_paged_attention_unaligned_program,
    kernel_online_update,
    kernel_pv_matmul,
    kernel_qk_matmul,
    kernel_softmax_prepare,
    kernel_softmax_prepare_unaligned,
)
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

DEFAULT_SCALE = 0.0884


class QKMatmulTestCase(PTOTestCase):
    """Test case for QK matmul kernel.

    Computes: sij = qi @ kj_t  -> (num_heads, num_heads)
    Memory flow: GM -> Mat (target_memory=pl.MemorySpace.Mat)
                 -> Left/Right (target_memory=pl.MemorySpace.Left/Right) -> Acc -> GM
    """

    def __init__(self, num_heads: int = 16, head_dim: int = 16, block_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size

    def get_name(self) -> str:
        return f"qk_matmul_{self.num_heads}h_{self.head_dim}d_b{self.block_size}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "qi", [self.num_heads, self.head_dim], DataType.BF16, init_value=torch.randn
            ),  # query: [num_heads, head_dim]
            TensorSpec(
                "kj", [self.block_size, self.head_dim], DataType.BF16, init_value=torch.randn
            ),  # key: [block_size, head_dim]
            TensorSpec(
                "sij", [self.num_heads, self.block_size], DataType.FP32, is_output=True
            ),  # attention score output: [num_heads, block_size]
        ]

    def get_program(self) -> Any:
        @pl.program
        class QKMatmulProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                qi: pl.Tensor[[16, 128], pl.BF16],
                kj_t: pl.Tensor[[128, 128], pl.BF16],
                out_sij: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_sij = kernel_qk_matmul(qi, kj_t, out_sij)
                return out_sij

        return QKMatmulProgram

    def compute_expected(self, tensors, params=None):
        # sij = qi @ kj_t
        qi = tensors["qi"].to(torch.float32)
        kj = tensors["kj"].to(torch.float32)
        tensors["sij"][:] = torch.matmul(qi, kj.T)


class SoftmaxPrepareTestCase(PTOTestCase):
    """Test case for softmax_prepare kernel.

    Computes:
      sij_scaled = sij * scale
      mij = row_max(sij_scaled)        -> (num_heads, 1)
      pij = exp(sij_scaled - mij)      -> (num_heads, block_size)
      lij = row_sum(pij)               -> (num_heads, 1)
    """

    def __init__(self, num_heads: int = 16, block_size: int = 16, scale: float = DEFAULT_SCALE, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.block_size = block_size
        self.scale = scale

    def get_name(self) -> str:
        return f"softmax_prepare_{self.num_heads}h_{self.block_size}b"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "sij", [self.num_heads, self.block_size], DataType.FP32, init_value=1.0
            ),  # attention scores input: [num_heads, block_size]
            TensorSpec(
                "config", [1], DataType.FP32, init_value=self.scale
            ),  # single-element FP32 tensor storing the scale factor
            TensorSpec(
                "pij", [self.num_heads, self.block_size], DataType.BF16, is_output=True
            ),  # exp(sij_scaled - mij) output: [num_heads, block_size]
            TensorSpec(
                "mij", [self.num_heads, 1], DataType.FP32, is_output=True
            ),  # row-max output: [num_heads, 1]
            TensorSpec(
                "lij", [self.num_heads, 1], DataType.FP32, is_output=True
            ),  # row-sum of pij output: [num_heads, 1]
        ]

    def get_program(self) -> Any:
        @pl.program
        class SoftmaxPrepareProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                sij: pl.Tensor[[16, 128], pl.FP32],
                config: pl.Tensor[[1], pl.FP32],
                pij_out: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
                mij_out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
                lij_out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 128], pl.BF16],
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 1], pl.FP32],
            ]:
                # Read scale value from config tensor
                scale: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                pij_out, mij_out, lij_out = kernel_softmax_prepare(sij, scale, pij_out, mij_out, lij_out)
                return pij_out, mij_out, lij_out

        return SoftmaxPrepareProgram

    def compute_expected(self, tensors, params=None):
        # Read scale directly from the FP32 config tensor
        scale = tensors["config"][0]

        sij = tensors["sij"]
        sij_scaled = sij * scale
        mij = torch.max(sij_scaled, axis=1, keepdims=True).values
        pij = torch.exp(sij_scaled - mij)
        pij_bf16 = pij.to(torch.bfloat16)
        pij = pij_bf16.to(torch.float32)
        lij = torch.sum(pij, axis=1, keepdims=True)

        tensors["pij"][:] = pij_bf16
        tensors["mij"][:] = mij
        tensors["lij"][:] = lij


class SoftmaxPrepareUnalignedTestCase(PTOTestCase):
    """Test case for softmax_prepare with unaligned valid_len.

    Uses pl.load(..., valid_shapes=[16, valid_len]) + fillpad(pad_value=min)
    to handle non-block-aligned KV cache lengths. Columns beyond valid_len
    are padded with -inf before softmax.
    """

    def __init__(
        self,
        num_heads: int = 16,
        block_size: int = 128,
        valid_len: int = 100,
        scale: float = DEFAULT_SCALE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.block_size = block_size
        self.valid_len = valid_len
        self.scale = scale

    def get_name(self) -> str:
        return f"softmax_prepare_unaligned_{self.num_heads}h_{self.block_size}b_{self.valid_len}v"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "sij", [self.num_heads, self.block_size], DataType.FP32, init_value=torch.randn
            ),  # attention scores input
            TensorSpec(
                "config", [1], DataType.FP32, init_value=self.scale
            ),  # single-element FP32 tensor storing the scale factor
            TensorSpec(
                "valid_len_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([self.valid_len], dtype=torch.int64),
            ),  # valid_len as INT64 config
            TensorSpec("pij", [self.num_heads, self.block_size], DataType.BF16, is_output=True),  # exp output
            TensorSpec("mij", [self.num_heads, 1], DataType.FP32, is_output=True),  # row-max output
            TensorSpec("lij", [self.num_heads, 1], DataType.FP32, is_output=True),  # row-sum of pij output
        ]

    def get_program(self) -> Any:
        @pl.program
        class SoftmaxPrepareUnalignedProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                sij: pl.Tensor[[16, 128], pl.FP32],
                config: pl.Tensor[[1], pl.FP32],
                valid_len_cfg: pl.Tensor[[1], pl.INT64],
                pij_out: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
                mij_out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
                lij_out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 128], pl.BF16],
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 1], pl.FP32],
            ]:
                scale: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                valid_len: pl.Scalar[pl.INT64] = pl.tensor.read(valid_len_cfg, [0])
                pij_out, mij_out, lij_out = kernel_softmax_prepare_unaligned(
                    sij, scale, valid_len, pij_out, mij_out, lij_out
                )
                return pij_out, mij_out, lij_out

        return SoftmaxPrepareUnalignedProgram

    def compute_expected(self, tensors, params=None):
        scale = tensors["config"][0]
        valid_len = int(tensors["valid_len_cfg"][0].item())
        block_size = tensors["sij"].shape[1]  # derive from tensor shape (golden_writer can't access self.*)

        sij = tensors["sij"].clone()
        # Mask columns beyond valid_len to -inf (fillpad with PadValue.min)
        sij[:, valid_len:block_size] = float("-inf")
        sij_scaled = sij * scale
        mij = torch.max(sij_scaled, axis=1, keepdims=True).values
        pij = torch.exp(sij_scaled - mij)
        pij_bf16 = pij.to(torch.bfloat16)
        pij = pij_bf16.to(torch.float32)
        lij = torch.sum(pij, axis=1, keepdims=True)

        tensors["pij"][:] = pij_bf16
        tensors["mij"][:] = mij
        tensors["lij"][:] = lij


class PVMatmulTestCase(PTOTestCase):
    """Test case for PV matmul kernel.

    Computes: oi_new = pij @ vj  -> (num_heads, head_dim)
    Memory flow: GM -> Mat (target_memory=pl.MemorySpace.Mat)
                 -> Left/Right (target_memory=pl.MemorySpace.Left/Right) -> Acc -> GM
    """

    def __init__(self, num_heads: int = 16, block_size: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"pv_matmul_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "pij", [self.num_heads, self.block_size], DataType.BF16, init_value=torch.randn
            ),  # attention probability: [num_heads, block_size]
            TensorSpec(
                "vj", [self.block_size, self.head_dim], DataType.BF16, init_value=torch.randn
            ),  # value tensor: [block_size, head_dim]
            TensorSpec(
                "oi_new", [self.num_heads, self.head_dim], DataType.FP32, is_output=True
            ),  # new attention output: [num_heads, head_dim]
        ]

    def get_program(self) -> Any:
        @pl.program
        class PVMatmulProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                pij: pl.Tensor[[16, 128], pl.BF16],
                vj: pl.Tensor[[128, 128], pl.BF16],
                out_oi: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                out_oi = kernel_pv_matmul(pij, vj, out_oi)
                return out_oi

        return PVMatmulProgram

    def compute_expected(self, tensors, params=None):
        # oi_new = pij @ vj
        pij = tensors["pij"].to(torch.float32)
        vj = tensors["vj"].to(torch.float32)
        tensors["oi_new"][:] = torch.matmul(pij, vj)


class OnlineUpdateTestCase(PTOTestCase):
    """Unified test case for online_update kernel.

    is_first and is_last are typed pl.Scalar[pl.BOOL] in the InCore function
    signature, but read from the config tensor as pl.Scalar[pl.INT64] in the
    Orchestration function.  The kernel handles all four flag combinations:

      - is_first=1, is_last=1: copy mij->mi, lij->li, oi_new->oi; dst=oi_new/lij
      - is_first=1, is_last=0: copy mij->mi, lij->li, oi_new->oi; dst unchanged
      - is_first=0, is_last=1: full online update; dst=oi_updated/li_updated
      - is_first=0, is_last=0: full online update; dst=zeros

    is_first and is_last are accepted as constructor arguments and written into
    the config TensorSpec so the test harness can exercise all four paths.
    """

    def __init__(
        self, num_heads: int = 16, head_dim: int = 16, is_first: int = 0, is_last: int = 1, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.is_first = is_first
        self.is_last = is_last

    def get_name(self) -> str:
        return f"online_update_{self.num_heads}h_{self.head_dim}d_f{self.is_first}_l{self.is_last}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "mij", [self.num_heads, 1], DataType.FP32, init_value=0.5
            ),  # current block row-max: [num_heads, 1]
            TensorSpec(
                "lij", [self.num_heads, 1], DataType.FP32, init_value=1.5
            ),  # current block row-sum: [num_heads, 1]
            TensorSpec(
                "oi_new", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.3
            ),  # current block attention output: [num_heads, head_dim]
            TensorSpec(
                "config",
                [2],
                DataType.INT64,
                init_value=torch.tensor([self.is_first, self.is_last], dtype=torch.int64),
            ),  # [is_first, is_last]
            TensorSpec(
                "mi", [self.num_heads, 1], DataType.FP32, init_value=0.4, is_output=True
            ),  # accumulated row-max (in/out): [num_heads, 1]
            TensorSpec(
                "li", [self.num_heads, 1], DataType.FP32, init_value=2.0, is_output=True
            ),  # accumulated row-sum (in/out): [num_heads, 1]
            TensorSpec(
                "oi", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.2, is_output=True
            ),  # accumulated attention output (in/out): [num_heads, head_dim]
            TensorSpec(
                "dst", [self.num_heads, self.head_dim], DataType.FP32, is_output=True
            ),  # final normalized output: [num_heads, head_dim]
        ]

    def get_program(self) -> Any:
        @pl.program
        class OnlineUpdateProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 128], pl.FP32],
                config: pl.Tensor[[2], pl.INT64],
                mi: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
                li: pl.InOut[pl.Tensor[[16, 1], pl.FP32]],
                oi: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],
                dst: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 128], pl.FP32],
                pl.Tensor[[16, 128], pl.FP32],
            ]:
                # Read is_first and is_last from config tensor
                is_first: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                is_last: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                mi, li, oi, dst = kernel_online_update(mij, lij, oi_new, mi, li, oi, dst, is_first, is_last)
                return mi, li, oi, dst

        return OnlineUpdateProgram

    def compute_expected(self, tensors, params=None):
        """Compute expected outputs for all four (is_first, is_last) combinations.

        Mirrors the branching logic of OnlineUpdateProgram.online_update using
        the same intermediate names (mi_new, alpha, beta, li_updated, oi_updated)
        so the expected values align with the hardware kernel's behaviour.
        """
        is_first = bool(int(tensors["config"][0]))
        is_last = bool(int(tensors["config"][1]))

        mij = tensors["mij"]
        lij = tensors["lij"]
        oi_new = tensors["oi_new"]
        mi = tensors["mi"]
        li = tensors["li"]
        oi = tensors["oi"]

        if is_first:
            # First block: copy mij->mi, lij->li, oi_new->oi
            tensors["mi"][:] = mij
            tensors["li"][:] = lij
            tensors["oi"][:] = oi_new
            if is_last:
                # Single block: normalize dst = oi_new / lij
                tensors["dst"][:] = oi_new / lij
            else:
                # First but not last: kernel does not write dst; zero it for comparison
                tensors["dst"][:] = torch.zeros_like(tensors["dst"])
        else:
            # Not first: full online update
            mi_new = torch.maximum(mi, mij)
            alpha = torch.exp(mi - mi_new)
            beta = torch.exp(mij - mi_new)
            li_updated = alpha * li + beta * lij
            oi_updated = alpha * oi + beta * oi_new

            tensors["mi"][:] = mi_new
            tensors["li"][:] = li_updated
            tensors["oi"][:] = oi_updated

            if is_last:
                # Last block: normalize dst = oi_updated / li_updated
                tensors["dst"][:] = oi_updated / li_updated
            else:
                # Middle block: kernel stores zeros to dst
                tensors["dst"][:] = torch.zeros_like(oi_new)


class PagedAttentionTestCase(PTOTestCase):
    """Test case for paged attention using build_paged_attention_program.

    Delegates program construction to paged_attention_example.py so that the ST
    always exercises the same program definition as the example.

    Tensor layout (all 2D, flattened):
      query:       [batch * num_heads, head_dim]                    BF16
      key_cache:   [total_pool_blocks * block_size, head_dim]       BF16
      value_cache: [total_pool_blocks * block_size, head_dim]       BF16
      out:         [batch * num_heads, head_dim]                    FP32
    """

    def __init__(
        self,
        batch: int = 64,
        num_heads: int = 16,
        head_dim: int = 128,
        block_size: int = 128,
        context_len: int = 8192,
        max_model_len: int = 32768,
        scale: float = 1.0,
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
        self.max_num_blocks_per_req = max_model_len // block_size

    def get_name(self) -> str:
        return f"paged_attention_{self.batch}bat_{self.num_heads}h_{self.head_dim}d_{self.block_size}bs"

    def define_tensors(self) -> list[TensorSpec]:
        B = self.batch
        H = self.num_heads
        D = self.head_dim
        BS = self.block_size
        max_blocks = self.max_num_blocks_per_req
        total_pool_rows = B * max_blocks * BS

        scale_bits = struct.unpack("I", struct.pack("f", self.scale))[0]
        config = torch.tensor(
            [B, H, 1, D, BS, max_blocks, scale_bits],
            dtype=torch.int64,
        )

        def make_block_table():
            return torch.randint(0, max(B * max_blocks, 1), size=(B, max_blocks), dtype=torch.int32).flatten()

        context_lens = torch.full((B,), self.context_len, dtype=torch.int32)

        return [
            TensorSpec("query", [B * H, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [total_pool_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("value_cache", [total_pool_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("block_table", [B * max_blocks], DataType.INT32, init_value=make_block_table),
            TensorSpec("context_lens", [B], DataType.INT32, init_value=context_lens),
            TensorSpec("out", [B * H, D], DataType.FP32, is_output=True),
            TensorSpec("config", [7], DataType.INT64, init_value=config),
        ]

    def get_program(self) -> Any:
        return build_paged_attention_program(
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
                pij = pij.to(torch.bfloat16).to(torch.float32)
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


class UnalignedPagedAttentionTestCase(PTOTestCase):
    """Full paged attention with unaligned context_len (not a multiple of block_size).

    Uses build_paged_attention_unaligned_program which calls
    kernel_softmax_prepare_unaligned (set_validshape + fillpad) instead of
    slicing sij to valid_len.
    """

    def __init__(
        self,
        batch: int = 64,
        num_heads: int = 16,
        head_dim: int = 128,
        block_size: int = 128,
        context_len: int = 8100,
        max_model_len: int = 32768,
        scale: float = 1.0,
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
        self.max_num_blocks_per_req = max_model_len // block_size

    def get_name(self) -> str:
        return (
            f"paged_attention_unaligned_{self.batch}bat_{self.num_heads}h_"
            f"{self.head_dim}d_{self.block_size}bs_{self.context_len}ctx"
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
            [B, H, 1, D, BS, max_blocks, scale_bits],
            dtype=torch.int64,
        )

        def make_block_table():
            return torch.randint(0, max(B * max_blocks, 1), size=(B, max_blocks), dtype=torch.int32).flatten()

        context_lens = torch.full((B,), self.context_len, dtype=torch.int32)

        return [
            TensorSpec("query", [B * H, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [total_pool_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("value_cache", [total_pool_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("block_table", [B * max_blocks], DataType.INT32, init_value=make_block_table),
            TensorSpec("context_lens", [B], DataType.INT32, init_value=context_lens),
            TensorSpec("out", [B * H, D], DataType.FP32, is_output=True),
            TensorSpec("config", [7], DataType.INT64, init_value=config),
        ]

    def get_program(self) -> Any:
        return build_paged_attention_unaligned_program(
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
                pij = pij.to(torch.bfloat16).to(torch.float32)
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


class PTOASTestCaseMixin:
    """Mixin for test cases using PTO backend and Default optimization strategy."""

    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class QKMatmulPTOASTestCase(PTOASTestCaseMixin, QKMatmulTestCase):
    """Test QK matmul with PTO backend and PTOAS optimization strategy."""

    def get_name(self) -> str:
        return f"qk_matmul_ptoas_{self.num_heads}h_{self.head_dim}d_b{self.block_size}"


class SoftmaxPreparePTOASTestCase(PTOASTestCaseMixin, SoftmaxPrepareTestCase):
    """Test softmax prepare with PTO backend and PTOAS optimization strategy."""

    def get_name(self) -> str:
        return f"softmax_prepare_ptoas_{self.num_heads}h_{self.block_size}b"


class PVMatmulPTOASTestCase(PTOASTestCaseMixin, PVMatmulTestCase):
    """Test PV matmul with PTO backend and PTOAS optimization strategy."""

    def get_name(self) -> str:
        return f"pv_matmul_ptoas_{self.num_heads}h_{self.head_dim}d"


class OnlineUpdatePTOASTestCase(PTOASTestCaseMixin, OnlineUpdateTestCase):
    """Test online update with PTO backend and PTOAS optimization strategy."""

    def get_name(self) -> str:
        return f"online_update_ptoas_{self.num_heads}h_{self.head_dim}d_f{self.is_first}_l{self.is_last}"


class PagedAttentionPTOASTestCase(PTOASTestCaseMixin, PagedAttentionTestCase):
    """Test paged attention with PTO backend and PTOAS optimization strategy."""

    def get_name(self) -> str:
        return f"paged_attention_ptoas_{self.batch}bat_{self.num_heads}h_{self.head_dim}d_{self.block_size}bs"


class SoftmaxPrepareUnalignedPTOASTestCase(PTOASTestCaseMixin, SoftmaxPrepareUnalignedTestCase):
    """Test unaligned softmax prepare with PTO backend and PTOAS optimization strategy."""

    def get_name(self) -> str:
        return f"softmax_prepare_unaligned_ptoas_{self.num_heads}h_{self.block_size}b_{self.valid_len}v"


class UnalignedPagedAttentionPTOASTestCase(PTOASTestCaseMixin, UnalignedPagedAttentionTestCase):
    """Test full paged attention with unaligned context_len using PTOAS."""

    def get_name(self) -> str:
        return (
            f"paged_attention_unaligned_ptoas_{self.batch}bat_{self.num_heads}h_"
            f"{self.head_dim}d_{self.block_size}bs_{self.context_len}ctx"
        )


class TestPagedAttentionKernels:
    """Integration tests for Paged Attention kernels (PTO backend + PTOAS).

    Each test instantiates the corresponding PTOTestCase and runs it through
    the test_runner fixture, which handles kernel compilation and result
    validation against compute_expected.
    """

    @pytest.mark.parametrize("num_heads,head_dim,block_size", [(16, 128, 128)])
    def test_qk_matmul_ptoas(self, test_runner, num_heads, head_dim, block_size):
        """Test QK matmul with PTO backend and PTOAS optimization."""
        test_case = QKMatmulPTOASTestCase(num_heads=num_heads, head_dim=head_dim, block_size=block_size)
        result = test_runner.run(test_case)
        assert result.passed, f"QK matmul PTOAS test failed: {result.error}"

    @pytest.mark.parametrize("num_heads,block_size", [(16, 128)])
    def test_softmax_prepare_ptoas(self, test_runner, num_heads, block_size):
        """Test softmax prepare with PTO backend and PTOAS optimization."""
        test_case = SoftmaxPreparePTOASTestCase(num_heads=num_heads, block_size=block_size)
        result = test_runner.run(test_case)
        assert result.passed, f"Softmax prepare PTOAS test failed: {result.error}"

    @pytest.mark.parametrize("num_heads,block_size,head_dim", [(16, 128, 128)])
    def test_pv_matmul_ptoas(self, test_runner, num_heads, block_size, head_dim):
        """Test PV matmul with PTO backend and PTOAS optimization."""
        test_case = PVMatmulPTOASTestCase(num_heads=num_heads, block_size=block_size, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"PV matmul PTOAS test failed: {result.error}"

    @pytest.mark.parametrize(
        "num_heads,head_dim,is_first,is_last",
        [
            (16, 128, 1, 1),  # single block: first + last
            (16, 128, 1, 0),  # first block, more to come
            (16, 128, 0, 1),  # last block
            (16, 128, 0, 0),  # middle block
        ],
    )
    def test_online_update_ptoas(self, test_runner, num_heads, head_dim, is_first, is_last):
        """Test online update with PTO backend and PTOAS optimization."""
        test_case = OnlineUpdatePTOASTestCase(
            num_heads=num_heads, head_dim=head_dim, is_first=is_first, is_last=is_last
        )
        result = test_runner.run(test_case)
        assert result.passed, (
            f"Online update PTOAS test failed (is_first={is_first}, is_last={is_last}): {result.error}"
        )

    @pytest.mark.parametrize(
        "batch,num_heads,head_dim,block_size,context_len,max_model_len",
        [
            (64, 16, 128, 128, 8192, 32768),
        ],
    )
    def test_paged_attention_ptoas(
        self, test_runner, batch, num_heads, head_dim, block_size, context_len, max_model_len
    ):
        """Test paged attention with PTO backend and PTOAS optimization."""
        test_case = PagedAttentionPTOASTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            context_len=context_len,
            max_model_len=max_model_len,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"Paged attention PTOAS test failed: {result.error}"

    # ── Unaligned variants ────────────────────────────────────────────

    @pytest.mark.parametrize(
        "num_heads,block_size,valid_len",
        [
            (16, 128, 100),  # unaligned
            (16, 128, 128),  # full block (aligned)
        ],
    )
    def test_softmax_prepare_unaligned_ptoas(self, test_runner, num_heads, block_size, valid_len):
        """Test unaligned softmax prepare with PTO backend and PTOAS optimization."""
        test_case = SoftmaxPrepareUnalignedPTOASTestCase(
            num_heads=num_heads, block_size=block_size, valid_len=valid_len
        )
        result = test_runner.run(test_case)
        assert result.passed, (
            f"Softmax prepare unaligned PTOAS test failed (valid_len={valid_len}): {result.error}"
        )

    @pytest.mark.parametrize(
        "batch,num_heads,head_dim,block_size,context_len,max_model_len",
        [
            (64, 16, 128, 128, 8100, 32768),  # context_len=8100 not a multiple of block_size=128
        ],
    )
    def test_paged_attention_unaligned_ptoas(
        self, test_runner, batch, num_heads, head_dim, block_size, context_len, max_model_len
    ):
        """Test full paged attention with unaligned context_len using set_validshape."""
        test_case = UnalignedPagedAttentionPTOASTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            context_len=context_len,
            max_model_len=max_model_len,
        )
        result = test_runner.run(test_case)
        assert result.passed, (
            f"Paged attention unaligned PTOAS test failed (context_len={context_len}): {result.error}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
