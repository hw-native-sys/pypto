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

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

DEFAULT_SCALE = 0.0884


class QKMatmulTestCase(PTOTestCase):
    """Test case for QK matmul kernel.

    Computes: sij = qi @ kj_t  -> (num_heads, num_heads)
    Memory flow: GM -> Mat (target_memory=pl.MemorySpace.Mat)
                 -> Left/Right (target_memory=pl.MemorySpace.Left/Right) -> Acc -> GM
    """

    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"qk_matmul_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "qi", [self.num_heads, self.head_dim], DataType.FP32, init_value=2.0
            ),  # query: [num_heads, head_dim]
            TensorSpec(
                "kj_t", [self.head_dim, self.num_heads], DataType.FP32, init_value=3.0
            ),  # transposed key: [head_dim, num_heads]
            TensorSpec(
                "sij", [self.num_heads, self.num_heads], DataType.FP32, is_output=True
            ),  # attention score output: [num_heads, num_heads]
        ]

    def get_program(self) -> Any:
        @pl.program
        class QKMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_matmul(
                self,
                qi: pl.Tensor[[16, 16], pl.FP32],
                kj_t: pl.Tensor[[16, 16], pl.FP32],
                sij: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                qi_l1 = pl.load(qi, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)  # Load qi to L1
                kj_l1 = pl.load(kj_t, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)  # Load kj_t to L1
                qi_l0a = pl.move(qi_l1, target_memory=pl.MemorySpace.Left)  # Move qi L1 -> Left
                kj_l0b = pl.move(kj_l1, target_memory=pl.MemorySpace.Right)  # Move kj_t L1 -> Right
                sij_l0c = pl.matmul(qi_l0a, kj_l0b)  # Compute qi @ kj_t in Acc
                out_sij = pl.l0c_store(sij_l0c, [0, 0], [16, 16], sij)  # Store Acc -> GM
                return out_sij

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, qi: pl.Tensor[[16, 16], pl.FP32], kj_t: pl.Tensor[[16, 16], pl.FP32]
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_sij: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                out_sij = self.qk_matmul(qi, kj_t, out_sij)
                return out_sij

        return QKMatmulProgram

    def compute_expected(self, tensors, params=None):
        # sij = qi @ kj_t
        tensors["sij"][:] = torch.matmul(tensors["qi"], tensors["kj_t"])


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
            @pl.function(type=pl.FunctionType.InCore)
            def softmax_prepare(
                self,
                sij: pl.Tensor[[16, 128], pl.FP32],
                scale: pl.Scalar[pl.FP32],
                pij: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
                mij: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
                lij: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[16, 128], pl.BF16], pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 1], pl.FP32]
            ]:
                # Load sij to UB (target_memory=pl.MemorySpace.Vec)
                sij_tile = pl.load(sij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)

                # Scale: sij * scale_factor
                sij_scaled = pl.mul(sij_tile, scale)

                # Create temp tile for row reduction
                tmp_tile = pl.create_tile([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)

                # Row max: mij = max(sij_scaled, axis=1) -> [16, 1] DN format
                mij_tile = pl.row_max(sij_scaled, tmp_tile)

                # Row broadcast subtraction: sij_scaled - mij
                sij_centered = pl.row_expand_sub(sij_scaled, mij_tile)

                # Exp: exp(sij_centered)
                pij_tile = pl.exp(sij_centered)

                # Cast to BF16 for output, then back to FP32 for row sum computation
                pij_tile_bf16 = pl.cast(pij_tile, target_type=pl.BF16)
                pij_tile = pl.cast(pij_tile_bf16, target_type=pl.FP32)

                # Row sum: lij = sum(pij, axis=1) -> [16, 1] DN format
                lij_tile = pl.row_sum(pij_tile, tmp_tile)

                # Store results
                pij_out = pl.store(pij_tile_bf16, [0, 0], [16, 128], pij)
                mij_out = pl.store(mij_tile, [0, 0], [16, 1], mij)
                lij_out = pl.store(lij_tile, [0, 0], [16, 1], lij)

                return pij_out, mij_out, lij_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                sij: pl.Tensor[[16, 128], pl.FP32],
                config: pl.Tensor[[1], pl.FP32],
            ) -> tuple[
                pl.Tensor[[16, 128], pl.BF16], pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 1], pl.FP32]
            ]:
                # Read scale value from config tensor
                scale: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                pij_out: pl.Tensor[[16, 128], pl.BF16] = pl.create_tensor([16, 128], dtype=pl.BF16)
                mij_out: pl.Tensor[[16, 1], pl.FP32] = pl.create_tensor([16, 1], dtype=pl.FP32)
                lij_out: pl.Tensor[[16, 1], pl.FP32] = pl.create_tensor([16, 1], dtype=pl.FP32)
                pij_out, mij_out, lij_out = self.softmax_prepare(sij, scale, pij_out, mij_out, lij_out)
                return pij_out, mij_out, lij_out

        return SoftmaxPrepareProgram

    def compute_expected(self, tensors, params=None):
        # Read scale directly from the FP32 config tensor
        scale = tensors["config"][0]

        sij = tensors["sij"]
        sij_scaled = sij * scale
        mij = torch.max(sij_scaled, axis=1, keepdims=True).values
        pij = torch.exp(sij_scaled - mij)

        # Cast to BF16 and back to FP32 for consistent row sum
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

    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"pv_matmul_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "pij", [self.num_heads, self.num_heads], DataType.FP32, init_value=0.1
            ),  # attention probability: [num_heads, num_heads]
            TensorSpec(
                "vj", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.5
            ),  # value tensor: [num_heads, head_dim]
            TensorSpec(
                "oi_new", [self.num_heads, self.head_dim], DataType.FP32, is_output=True
            ),  # new attention output: [num_heads, head_dim]
        ]

    def get_program(self) -> Any:
        @pl.program
        class PVMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def pv_matmul(
                self,
                pij: pl.Tensor[[16, 16], pl.FP32],
                vj: pl.Tensor[[16, 16], pl.FP32],
                oi_new: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pij_l1 = pl.load(pij, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)  # Load pij to L1
                vj_l1 = pl.load(vj, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)  # Load vj to L1
                pij_l0a = pl.move(pij_l1, target_memory=pl.MemorySpace.Left)  # Move pij L1 -> Left
                vj_l0b = pl.move(vj_l1, target_memory=pl.MemorySpace.Right)  # Move vj L1 -> Right
                oi_l0c = pl.matmul(pij_l0a, vj_l0b)  # Compute pij @ vj in Acc
                out_oi = pl.l0c_store(oi_l0c, [0, 0], [16, 16], oi_new)  # Store Acc -> GM
                return out_oi

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, pij: pl.Tensor[[16, 16], pl.FP32], vj: pl.Tensor[[16, 16], pl.FP32]
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_oi: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                out_oi = self.pv_matmul(pij, vj, out_oi)
                return out_oi

        return PVMatmulProgram

    def compute_expected(self, tensors, params=None):
        # oi_new = pij @ vj
        tensors["oi_new"][:] = torch.matmul(tensors["pij"], tensors["vj"])


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
            @pl.function(type=pl.FunctionType.InCore)
            def online_update(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 128], pl.FP32],
                mi: pl.Tensor[[16, 1], pl.FP32],
                li: pl.Tensor[[16, 1], pl.FP32],
                oi: pl.Tensor[[16, 128], pl.FP32],
                dst: pl.Tensor[[16, 128], pl.FP32],
                is_first: pl.Scalar[pl.BOOL],
                is_last: pl.Scalar[pl.BOOL],
            ) -> tuple[
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 128], pl.FP32],
                pl.Tensor[[16, 128], pl.FP32],
            ]:
                # Load all inputs
                mij_tile = pl.load(mij, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec)
                lij_tile = pl.load(lij, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec)
                oi_new_tile = pl.load(oi_new, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
                mi_tile = pl.load(mi, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec)
                li_tile = pl.load(li, [0, 0], [16, 1], target_memory=pl.MemorySpace.Vec)
                oi_tile = pl.load(oi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec)

                if is_first:
                    # First block: copy mij->mi, lij->li, oi_new->oi
                    mi_out = pl.store(mij_tile, [0, 0], [16, 1], mi)
                    li_out = pl.store(lij_tile, [0, 0], [16, 1], li)
                    oi_out = pl.store(oi_new_tile, [0, 0], [16, 128], oi)
                    if is_last:
                        # Single block: normalize dst = oi_new / lij
                        dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
                        dst_out = pl.store(dst_tile, [0, 0], [16, 128], dst)
                else:
                    # Not first: full online update
                    # Reshape DN [16,1] -> ND [1,16] for element-wise ops
                    mi_tile_nd = pl.reshape(mi_tile, [1, 16])
                    mij_tile_nd = pl.reshape(mij_tile, [1, 16])
                    li_tile_nd = pl.reshape(li_tile, [1, 16])
                    lij_tile_nd = pl.reshape(lij_tile, [1, 16])

                    # mi_new = max(mi, mij): new running row maximum
                    mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
                    # alpha = exp(mi - mi_new): rescale factor for accumulated oi
                    mi_diff = pl.sub(mi_tile_nd, mi_new)
                    alpha = pl.exp(mi_diff)
                    # beta = exp(mij - mi_new): rescale factor for current oi_new
                    mij_diff = pl.sub(mij_tile_nd, mi_new)
                    beta = pl.exp(mij_diff)

                    # li_updated = alpha * li + beta * lij: updated normalizer
                    li_scaled = pl.mul(alpha, li_tile_nd)
                    lij_scaled = pl.mul(beta, lij_tile_nd)
                    li_updated = pl.add(li_scaled, lij_scaled)

                    # oi_updated = alpha * oi + beta * oi_new: updated attention output
                    alpha_dn = pl.reshape(alpha, [16, 1])  # Reshape [1,16] -> [16,1] DN for row_expand_mul
                    oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
                    beta_dn = pl.reshape(beta, [16, 1])  # Reshape [1,16] -> [16,1] DN for row_expand_mul
                    oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
                    oi_updated = pl.add(oi_scaled, oi_new_scaled)

                    mi_new_dn = pl.reshape(mi_new, [16, 1])  # Reshape back to DN [16,1] for store
                    li_updated_dn = pl.reshape(li_updated, [16, 1])  # Reshape back to DN [16,1] for store

                    mi_out = pl.store(mi_new_dn, [0, 0], [16, 1], mi)
                    li_out = pl.store(li_updated_dn, [0, 0], [16, 1], li)

                    if is_last:
                        # Last block: normalize dst = oi_updated / li_updated
                        dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                        dst_out = pl.store(dst_tile, [0, 0], [16, 128], dst)
                        oi_out = pl.store(oi_updated, [0, 0], [16, 128], oi)
                    else:
                        # Middle block: no normalize
                        zero_tile = pl.block.full([16, 128], dtype=pl.FP32, value=0.0)
                        dst_out = pl.store(zero_tile, [0, 0], [16, 128], dst)
                        oi_out = pl.store(oi_updated, [0, 0], [16, 128], oi)

                return mi_out, li_out, oi_out, dst_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 128], pl.FP32],
                config: pl.Tensor[[2], pl.INT64],
                mi: pl.Tensor[[16, 1], pl.FP32],
                li: pl.Tensor[[16, 1], pl.FP32],
                oi: pl.Tensor[[16, 128], pl.FP32],
                dst: pl.Tensor[[16, 128], pl.FP32],
            ) -> tuple[
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 1], pl.FP32],
                pl.Tensor[[16, 128], pl.FP32],
                pl.Tensor[[16, 128], pl.FP32],
            ]:
                # Read is_first and is_last from config tensor
                is_first: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                is_last: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                mi, li, oi, dst = self.online_update(mij, lij, oi_new, mi, li, oi, dst, is_first, is_last)
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


class TestPagedAttentionKernels:
    """Integration tests for the four Paged Attention kernels.

    Each test instantiates the corresponding PTOTestCase and runs it through
    the test_runner fixture, which handles kernel compilation and result
    validation against compute_expected.
    """

    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_qk_matmul(self, test_runner, num_heads, head_dim):
        test_case = QKMatmulTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"QK matmul test failed: {result.error}"

    @pytest.mark.parametrize("num_heads,block_size", [(16, 128)])
    def test_softmax_prepare(self, test_runner, num_heads, block_size):
        test_case = SoftmaxPrepareTestCase(num_heads=num_heads, block_size=block_size)
        result = test_runner.run(test_case)
        assert result.passed, f"Softmax prepare test failed: {result.error}"

    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_pv_matmul(self, test_runner, num_heads, head_dim):
        test_case = PVMatmulTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"PV matmul test failed: {result.error}"

    @pytest.mark.parametrize(
        "num_heads,head_dim,is_first,is_last",
        [
            (16, 128, 1, 1),  # single block: first + last
            (16, 128, 1, 0),  # first block, more to come
            (16, 128, 0, 1),  # last block
            (16, 128, 0, 0),  # middle block
        ],
    )
    def test_online_update(self, test_runner, num_heads, head_dim, is_first, is_last):
        test_case = OnlineUpdateTestCase(
            num_heads=num_heads, head_dim=head_dim, is_first=is_first, is_last=is_last
        )
        result = test_runner.run(test_case)
        assert result.passed, (
            f"Online update test failed (is_first={is_first}, is_last={is_last}): {result.error}"
        )
