"""
Tests for Paged Attention implementation using PyPTO frontend.

Online Update Kernel Behavior (aiv_online_update.cpp):
  - is_first=1, is_last=0: Copy mij->mi, lij->li, oi_new->oi (first block, more to come)
  - is_first=1, is_last=1: Copy + normalize dst = oi_new / lij (single block case)
  - is_first=0, is_last=0: Full online update, store oi (middle blocks)
  - is_first=0, is_last=1: Full online update + normalize dst = oi / li (last block)

Softmax Prepare Kernel (aiv_softmax_prepare.cpp):
  Computes: sij_scale = sij * scale
            mij = row_max(sij_scale)
            pij = exp(sij_scale - mij)
            lij = row_sum(pij)
"""

from typing import Any, List
import struct

import numpy as np
import pytest
import torch

from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec


DEFAULT_SCALE = 0.0884


class QKMatmulTestCase(PTOTestCase):
    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"qk_matmul_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("qi", [self.num_heads, self.head_dim], DataType.FP32, init_value=2.0),
            TensorSpec("kj_t", [self.head_dim, self.num_heads], DataType.FP32, init_value=3.0),
            TensorSpec("sij", [self.num_heads, self.num_heads], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class QKMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_matmul(
                self,
                qi: pl.Tensor[[16, 16], pl.FP32],
                kj_t: pl.Tensor[[16, 16], pl.FP32],
                sij: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                qi_l1 = pl.load(qi, [0, 0], [16, 16], target_memory=2)
                kj_l1 = pl.load(kj_t, [0, 0], [16, 16], target_memory=2)
                qi_l0a = pl.move(qi_l1, target_memory=3)
                kj_l0b = pl.move(kj_l1, target_memory=4)
                sij_l0c = pl.matmul(qi_l0a, kj_l0b)
                out_sij = pl.l0c_store(sij_l0c, [0, 0], [16, 16], sij)
                return out_sij

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, qi: pl.Tensor[[16, 16], pl.FP32], kj_t: pl.Tensor[[16, 16], pl.FP32]
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_sij = self.qk_matmul(qi, kj_t)
                return out_sij

        return QKMatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["sij"][:] = np.matmul(tensors["qi"], tensors["kj_t"])


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

    def define_tensors(self) -> List[TensorSpec]:
        import struct
        # Pack scale as float32 bits into int32
        scale_bits = struct.unpack('i', struct.pack('f', self.scale))[0]
        return [
            TensorSpec("sij", [self.num_heads, self.block_size], DataType.FP32, init_value=1.0),
            TensorSpec("config", [1], DataType.INT32, init_value=scale_bits),
            TensorSpec("pij", [self.num_heads, self.block_size], DataType.FP32, is_output=True),
            TensorSpec("mij", [self.num_heads, 1], DataType.FP32, is_output=True),
            TensorSpec("lij", [self.num_heads, 1], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl
        
        scale_value = self.scale

        @pl.program
        class SoftmaxPrepareProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def softmax_prepare(
                self,
                sij: pl.Tensor[[16, 16], pl.FP32],
                scale: pl.Scalar[pl.FP32],
                pij: pl.Tensor[[16, 16], pl.FP32],
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 1], pl.FP32]]:
                # Load sij to UB (target_memory=1)
                sij_tile = pl.load(sij, [0, 0], [16, 16], target_memory=1)
                
                # Scale: sij * scale_factor
                sij_scaled = pl.muls(sij_tile, scale)
                
                # Create temp tile for row reduction
                tmp_tile = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=1)
                
                # Row max: mij = max(sij_scaled, axis=1) -> [16, 1] DN format
                mij_tile = pl.row_max(sij_scaled, tmp_tile)
                
                # Row broadcast subtraction: sij_scaled - mij
                sij_centered = pl.row_expand_sub(sij_scaled, mij_tile)
                
                # Exp: exp(sij_centered)
                pij_tile = pl.exp(sij_centered)
                
                # Row sum: lij = sum(pij, axis=1) -> [16, 1] DN format
                lij_tile = pl.row_sum(pij_tile, tmp_tile)
                
                # Store results
                pij_out = pl.store(pij_tile, [0, 0], [16, 16], pij)
                mij_out = pl.store(mij_tile, [0, 0], [16, 1], mij)
                lij_out = pl.store(lij_tile, [0, 0], [16, 1], lij)
                
                return pij_out, mij_out, lij_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                sij: pl.Tensor[[16, 16], pl.FP32],
                config: pl.Tensor[[1], pl.INT32],
            ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 1], pl.FP32]]:
                # Read scale value from config tensor
                scale: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                pij_out, mij_out, lij_out = self.softmax_prepare(sij, scale)
                return pij_out, mij_out, lij_out

        return SoftmaxPrepareProgram

    def compute_expected(self, tensors, params=None):
        # Unpack scale from config tensor (INT32 bits representing float32)
        scale_bits = int(tensors["config"][0])
        scale = struct.unpack('f', struct.pack('i', scale_bits))[0]

        sij = tensors["sij"].numpy()
        sij_scaled = sij * scale
        mij = np.max(sij_scaled, axis=1, keepdims=True)
        pij = np.exp(sij_scaled - mij)
        lij = np.sum(pij, axis=1, keepdims=True)

        tensors["pij"][:] = torch.from_numpy(pij)
        tensors["mij"][:] = torch.from_numpy(mij)
        tensors["lij"][:] = torch.from_numpy(lij)


class PVMatmulTestCase(PTOTestCase):
    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"pv_matmul_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("pij", [self.num_heads, self.num_heads], DataType.FP32, init_value=0.1),
            TensorSpec("vj", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.5),
            TensorSpec("oi_new", [self.num_heads, self.head_dim], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class PVMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def pv_matmul(
                self,
                pij: pl.Tensor[[16, 16], pl.FP32],
                vj: pl.Tensor[[16, 16], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                pij_l1 = pl.load(pij, [0, 0], [16, 16], target_memory=2)
                vj_l1 = pl.load(vj, [0, 0], [16, 16], target_memory=2)
                pij_l0a = pl.move(pij_l1, target_memory=3)
                vj_l0b = pl.move(vj_l1, target_memory=4)
                oi_l0c = pl.matmul(pij_l0a, vj_l0b)
                out_oi = pl.l0c_store(oi_l0c, [0, 0], [16, 16], oi_new)
                return out_oi

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, pij: pl.Tensor[[16, 16], pl.FP32], vj: pl.Tensor[[16, 16], pl.FP32]
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                out_oi = self.pv_matmul(pij, vj)
                return out_oi

        return PVMatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["oi_new"][:] = np.matmul(tensors["pij"], tensors["vj"])


class OnlineUpdateTestCase(PTOTestCase):
    """Unified test case for online_update kernel.
    
    is_first and is_last are passed as pl.Scalar[pl.BOOL] function parameters,
    not as __init__ parameters. The kernel handles all four cases based on
    these runtime parameters.
    """

    def __init__(self, num_heads: int = 16, head_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def get_name(self) -> str:
        return f"online_update_{self.num_heads}h_{self.head_dim}d"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("mij", [self.num_heads, 1], DataType.FP32, init_value=0.5),
            TensorSpec("lij", [self.num_heads, 1], DataType.FP32, init_value=1.5),
            TensorSpec("oi_new", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.3),
            TensorSpec("config", [2], DataType.INT64, init_value=[0, 1]),  # [is_first=0, is_last=1]
            TensorSpec("mi", [self.num_heads, 1], DataType.FP32, init_value=0.4, is_output=True),
            TensorSpec("li", [self.num_heads, 1], DataType.FP32, init_value=2.0, is_output=True),
            TensorSpec("oi", [self.num_heads, self.head_dim], DataType.FP32, init_value=0.2, is_output=True),
            TensorSpec("dst", [self.num_heads, self.head_dim], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class OnlineUpdateProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def online_update(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                mi: pl.Tensor[[16, 1], pl.FP32],
                li: pl.Tensor[[16, 1], pl.FP32],
                oi: pl.Tensor[[16, 16], pl.FP32],
                is_first: pl.Scalar[pl.BOOL],
                is_last: pl.Scalar[pl.BOOL],
                dst: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 1], pl.FP32],
                       pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                # Load all inputs
                mij_tile = pl.load(mij, [0, 0], [16, 1], target_memory=1)
                lij_tile = pl.load(lij, [0, 0], [16, 1], target_memory=1)
                oi_new_tile = pl.load(oi_new, [0, 0], [16, 16], target_memory=1)
                mi_tile = pl.load(mi, [0, 0], [16, 1], target_memory=1)
                li_tile = pl.load(li, [0, 0], [16, 1], target_memory=1)
                oi_tile = pl.load(oi, [0, 0], [16, 16], target_memory=1)

                if is_first:
                    # First block: copy mij->mi, lij->li, oi_new->oi
                    mi_out = pl.store(mij_tile, [0, 0], [16, 1], mi)
                    li_out = pl.store(lij_tile, [0, 0], [16, 1], li)
                    oi_out = pl.store(oi_new_tile, [0, 0], [16, 16], oi)
                    if is_last:
                        # Single block: normalize dst = oi_new / lij
                        dst_tile = pl.row_expand_div(oi_new_tile, lij_tile)
                        dst_out = pl.store(dst_tile, [0, 0], [16, 16], dst)
                    else:
                        # First but not last: no dst output
                        zero_tile = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=1)
                        dst_out = pl.store(zero_tile, [0, 0], [16, 16], dst)
                else:
                    # Not first: full online update
                    mi_tile_nd = pl.reshape(mi_tile, [1, 16])
                    mij_tile_nd = pl.reshape(mij_tile, [1, 16])
                    li_tile_nd = pl.reshape(li_tile, [1, 16])
                    lij_tile_nd = pl.reshape(lij_tile, [1, 16])

                    mi_new = pl.maximum(mi_tile_nd, mij_tile_nd)
                    mi_diff = pl.sub(mi_tile_nd, mi_new)
                    alpha = pl.exp(mi_diff)
                    mij_diff = pl.sub(mij_tile_nd, mi_new)
                    beta = pl.exp(mij_diff)

                    li_scaled = pl.mul(alpha, li_tile_nd)
                    lij_scaled = pl.mul(beta, lij_tile_nd)
                    li_updated = pl.add(li_scaled, lij_scaled)

                    alpha_dn = pl.reshape(alpha, [16, 1])
                    oi_scaled = pl.row_expand_mul(oi_tile, alpha_dn)
                    beta_dn = pl.reshape(beta, [16, 1])
                    oi_new_scaled = pl.row_expand_mul(oi_new_tile, beta_dn)
                    oi_updated = pl.add(oi_scaled, oi_new_scaled)

                    mi_new_dn = pl.reshape(mi_new, [16, 1])
                    li_updated_dn = pl.reshape(li_updated, [16, 1])

                    mi_out = pl.store(mi_new_dn, [0, 0], [16, 1], mi)
                    li_out = pl.store(li_updated_dn, [0, 0], [16, 1], li)

                    if is_last:
                        # Last block: normalize dst = oi / li
                        dst_tile = pl.row_expand_div(oi_updated, li_updated_dn)
                        dst_out = pl.store(dst_tile, [0, 0], [16, 16], dst)
                        oi_out = pl.store(oi_updated, [0, 0], [16, 16], oi)
                    else:
                        # Middle block: no normalize
                        oi_out = pl.store(oi_updated, [0, 0], [16, 16], oi)
                        zero_tile = pl.create_tile([16, 16], dtype=pl.FP32, target_memory=1)
                        dst_out = pl.store(zero_tile, [0, 0], [16, 16], dst)

                return mi_out, li_out, oi_out, dst_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                mij: pl.Tensor[[16, 1], pl.FP32],
                lij: pl.Tensor[[16, 1], pl.FP32],
                oi_new: pl.Tensor[[16, 16], pl.FP32],
                config: pl.Tensor[[2], pl.INT64],
                mi: pl.Tensor[[16, 1], pl.FP32],
                li: pl.Tensor[[16, 1], pl.FP32],
                oi: pl.Tensor[[16, 16], pl.FP32],
            ) -> tuple[pl.Tensor[[16, 1], pl.FP32], pl.Tensor[[16, 1], pl.FP32],
                       pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
                # Read is_first and is_last from config tensor
                is_first: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                is_last: pl.Scalar[pl.INT64] = pl.tensor.read(config, [1])
                mi, li, oi, dst = self.online_update(mij, lij, oi_new, mi, li, oi, is_first, is_last)
                return mi, li, oi, dst

        return OnlineUpdateProgram

    def compute_expected(self, tensors, params=None):
        """Compute expected output based on config values."""
        # Read is_first and is_last from config tensor
        is_first = int(tensors["config"][0])
        is_last = int(tensors["config"][1])

        mij = tensors["mij"].numpy()
        lij = tensors["lij"].numpy()
        oi_new = tensors["oi_new"].numpy()
        mi = tensors["mi"].numpy().copy()
        li = tensors["li"].numpy().copy()
        oi = tensors["oi"].numpy().copy()

        # Default test case: is_first=0, is_last=1 (last block case)
        mi_new = np.maximum(mi, mij)
        alpha = np.exp(mi - mi_new)
        beta = np.exp(mij - mi_new)
        li_updated = alpha * li + beta * lij
        oi_updated = alpha * oi + beta * oi_new

        tensors["mi"][:] = torch.from_numpy(mi_new)
        tensors["li"][:] = torch.from_numpy(li_updated)
        tensors["oi"][:] = torch.from_numpy(oi_updated)
        tensors["dst"][:] = torch.from_numpy(oi_updated / li_updated)


class TestPagedAttentionKernels:
    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_qk_matmul(self, test_runner, num_heads, head_dim):
        test_case = QKMatmulTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"QK matmul test failed: {result.error}"

    @pytest.mark.parametrize("num_heads,block_size", [(16, 16)])
    def test_softmax_prepare(self, test_runner, num_heads, block_size):
        test_case = SoftmaxPrepareTestCase(num_heads=num_heads, block_size=block_size)
        result = test_runner.run(test_case)
        assert result.passed, f"Softmax prepare test failed: {result.error}"

    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_pv_matmul(self, test_runner, num_heads, head_dim):
        test_case = PVMatmulTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"PV matmul test failed: {result.error}"

    @pytest.mark.parametrize("num_heads,head_dim", [(16, 16)])
    def test_online_update(self, test_runner, num_heads, head_dim):
        test_case = OnlineUpdateTestCase(num_heads=num_heads, head_dim=head_dim)
        result = test_runner.run(test_case)
        assert result.passed, f"Online update test failed: {result.error}"
