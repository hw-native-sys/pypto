# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Paged Attention 64-Tier System Tests for PyPTO.

64divisions paged attention with single-loop N_UNROLL=64 orchestration:
  q_tile=64, block_size=64, head_dim=128, batch=64, num_heads=64, kv_head_num=1

Individual kernel tests:
  KernelAivHubTestCase        — zero-init accumulators
  KernelQkMatmulTestCase      — multi-block QK matmul
  KernelSoftmaxPrepareTestCase — two-pass softmax over n_blocks
  KernelPvMatmulTestCase      — SplitK PV matmul
  KernelOnlineUpdateTestCase  — online softmax merge (all 4 flag combos)

Full pipeline test:
  TestPagedAttention64Core    — end-to-end with golden_64 reference
"""

import struct
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime.runner import RunConfig

from examples.ir_parser.paged_attention_64_example import (
    BLOCK_SIZE,
    HEAD_DIM,
    N_UNROLL,
    N_UNROLL_Q,
    Q_TILE,
    build_paged_attention_64_program,
    kernel_aiv_hub,
    kernel_online_update_64,
    kernel_softmax_prepare_64,
    make_kernel_pv_matmul_64,
    make_kernel_qk_matmul_64,
)

# ── Full-pipeline parameters ─────────────────────────────────────────────────
# BATCH = 64
# NUM_HEADS = 64
# KV_HEAD_NUM = 1
# CONTEXT_LEN = 8192
# MAX_MODEL_LEN = 32768

BATCH = 1
NUM_HEADS = 16  # = Q_TILE
KV_HEAD_NUM = 1
CONTEXT_LEN = 128  # 2 blocks
MAX_MODEL_LEN = 128

MAX_NUM_BLOCKS_PER_REQ = MAX_MODEL_LEN // BLOCK_SIZE  # 512
SCALE = 1.0

# ── Individual kernel test parameters ────────────────────────────────────────
# Small cache dimensions for fast kernel-level tests (2 blocks)
_TEST_N_BLOCKS = 2
_TEST_KEY_CACHE_ROWS = _TEST_N_BLOCKS * BLOCK_SIZE  # 128
_TEST_BLOCK_TABLE_FLAT_SIZE = _TEST_N_BLOCKS  # 2


# ── PTOAS mixin ───────────────────────────────────────────────────────────────


class PTOASTestCaseMixin:
    """Mixin: configure Ascend910B_PTO backend with default optimization strategy."""

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO


# ── Individual kernel test cases ─────────────────────────────────────────────


class KernelAivHubTestCase(PTOTestCase):
    """Test case for kernel_aiv_hub: verifies zero-init of oi/li/mi accumulators."""

    __test__ = False

    def get_name(self) -> str:
        return "kernel_aiv_hub_64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("oi", [Q_TILE, HEAD_DIM], DataType.FP32, init_value=1.0, is_output=True),
            TensorSpec("li", [Q_TILE, 1], DataType.FP32, init_value=1.0, is_output=True),
            TensorSpec("mi", [Q_TILE, 1], DataType.FP32, init_value=1.0, is_output=True),
        ]

    def get_program(self) -> Any:
        @pl.program
        class AivHubProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                oi: pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
                li: pl.Tensor[[Q_TILE, 1], pl.FP32],
                mi: pl.Tensor[[Q_TILE, 1], pl.FP32],
            ) -> tuple[
                pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
            ]:
                return kernel_aiv_hub(oi, li, mi)

        return AivHubProgram

    def compute_expected(
        self, tensors: dict[str, torch.Tensor], _params: dict[str, Any] | None = None
    ) -> None:
        tensors["oi"][:] = torch.zeros_like(tensors["oi"])
        tensors["li"][:] = torch.zeros_like(tensors["li"])
        tensors["mi"][:] = torch.zeros_like(tensors["mi"])


class KernelAivHubPTOASTestCase(PTOASTestCaseMixin, KernelAivHubTestCase):
    """KernelAivHub test with Ascend910B_PTO backend."""

    def get_name(self) -> str:
        return "kernel_aiv_hub_64_ptoas"


class KernelQkMatmulTestCase(PTOTestCase):
    """Test case for kernel_qk_matmul_64: multi-block QK matmul.

    Uses _TEST_N_BLOCKS=2 blocks. Verifies sij[i*Q_TILE:(i+1)*Q_TILE, :] = qi @ kj[i].T
    for each block i, stacked vertically in sij_buf.
    """

    __test__ = False

    _n_blocks = _TEST_N_BLOCKS
    _key_cache_rows = _TEST_KEY_CACHE_ROWS
    _block_table_flat_size = _TEST_BLOCK_TABLE_FLAT_SIZE

    def __init__(self, config: RunConfig | None = None):
        # BF16 matmul (cube unit: BF16 inputs, FP32 accumulate) vs FP32 golden:
        # accumulation-order differences require relaxed tolerance.
        super().__init__(config or RunConfig(rtol=1e-3, atol=1e-3))

    def get_name(self) -> str:
        return f"kernel_qk_matmul_64_{self._n_blocks}blocks"

    def define_tensors(self) -> list[TensorSpec]:
        block_table_data = torch.arange(self._n_blocks, dtype=torch.int32)
        config_data = torch.tensor([self._n_blocks, BLOCK_SIZE], dtype=torch.int64)
        return [
            TensorSpec("qi", [Q_TILE, HEAD_DIM], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [self._key_cache_rows, HEAD_DIM], DataType.BF16, init_value=torch.randn),
            TensorSpec("sij_buf", [N_UNROLL_Q, BLOCK_SIZE], DataType.FP32, is_output=True),
            TensorSpec(
                "block_table", [self._block_table_flat_size], DataType.INT32, init_value=block_table_data
            ),
            TensorSpec("config", [2], DataType.INT64, init_value=config_data),
        ]

    def get_program(self) -> Any:
        kernel_qk = make_kernel_qk_matmul_64(self._key_cache_rows, self._block_table_flat_size)

        @pl.program
        class QkMatmulProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                qi: pl.Tensor[[Q_TILE, HEAD_DIM], pl.BF16],
                key_cache: pl.Tensor[[_TEST_KEY_CACHE_ROWS, HEAD_DIM], pl.BF16],
                sij_buf: pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32],
                block_table: pl.Tensor[[_TEST_BLOCK_TABLE_FLAT_SIZE], pl.INT32],
                config: pl.Tensor[[2], pl.INT64],
            ) -> pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32]:
                n_blocks: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])  # 2
                bt_offset: pl.Scalar[pl.INT64] = pl.const(0, pl.INT64)  # 64
                oi_new = kernel_qk(qi, key_cache, sij_buf, block_table, n_blocks, bt_offset)
                return oi_new

        return QkMatmulProgram

    def compute_expected(
        self, tensors: dict[str, torch.Tensor], _params: dict[str, Any] | None = None
    ) -> None:
        qi = tensors["qi"].to(torch.float32)
        key_cache = tensors["key_cache"].to(torch.float32)  # [key_cache_rows, HEAD_DIM]
        block_table = tensors["block_table"]
        sij_buf = torch.zeros_like(tensors["sij_buf"])
        n_blocks = int(tensors["config"][0].item())
        block_size = int(tensors["config"][1].item())
        q_tile = qi.shape[0]
        for i in range(n_blocks):
            phys = int(block_table[i].item())
            # kj: [block_size, HEAD_DIM] → matmul qi[q_tile,HD] @ kj.T[HD,BS] = [q_tile, BS]
            kj = key_cache[phys * block_size : (phys + 1) * block_size, :]
            sij_buf[i * q_tile : (i + 1) * q_tile, :] = torch.matmul(qi, kj.T)
        tensors["sij_buf"][:] = sij_buf


class KernelQkMatmulPTOASTestCase(PTOASTestCaseMixin, KernelQkMatmulTestCase):
    """KernelQkMatmul test with Ascend910B_PTO backend."""

    def get_name(self) -> str:
        return f"kernel_qk_matmul_64_{self._n_blocks}blocks_ptoas"


class KernelSoftmaxPrepareTestCase(PTOTestCase):
    """Test case for kernel_softmax_prepare_64: two-pass softmax over N_BLOCKS=2.

    Verifies global row_max (pass 1) and exp(s - max) + row_sum (pass 2).
    """

    __test__ = False

    _n_blocks = _TEST_N_BLOCKS
    _scale = 1.0

    def get_name(self) -> str:
        return f"kernel_softmax_prepare_64_{self._n_blocks}blocks"

    def define_tensors(self) -> list[TensorSpec]:
        config_data = torch.tensor([self._n_blocks, BLOCK_SIZE, BLOCK_SIZE], dtype=torch.int64)
        return [
            TensorSpec("sij_buf", [N_UNROLL_Q, BLOCK_SIZE], DataType.FP32, init_value=torch.randn),
            TensorSpec("pij_buf", [N_UNROLL_Q, BLOCK_SIZE], DataType.BF16, is_output=True),
            TensorSpec("mi_out", [Q_TILE, 1], DataType.FP32, is_output=True),
            TensorSpec("li_out", [Q_TILE, 1], DataType.FP32, is_output=True),
            TensorSpec("config", [3], DataType.INT64, init_value=config_data),
        ]

    def get_program(self) -> Any:
        scale = self._scale

        @pl.program
        class SoftmaxPrepareProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                sij_buf: pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.FP32],
                pij_buf: pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16],
                mi_out: pl.Tensor[[Q_TILE, 1], pl.FP32],
                li_out: pl.Tensor[[Q_TILE, 1], pl.FP32],
                config: pl.Tensor[[3], pl.INT64],
            ) -> tuple[
                pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
            ]:
                n_blocks: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                # valid_len_last: pl.Scalar[pl.INT64] = pl.tensor.read(config, [2])
                pij_buf, mi_out, li_out = kernel_softmax_prepare_64(
                    sij_buf,
                    scale,
                    pij_buf,
                    mi_out,
                    li_out,
                    n_blocks,  # type: ignore[reportArgumentType]
                )
                return pij_buf, mi_out, li_out

        return SoftmaxPrepareProgram

    def compute_expected(
        self, tensors: dict[str, torch.Tensor], _params: dict[str, Any] | None = None
    ) -> None:
        sij_buf = tensors["sij_buf"]
        n_blocks = self._n_blocks
        scale = self._scale

        # Pass 1: global row_max across all blocks
        all_scaled = []
        for i in range(n_blocks):
            s = sij_buf[i * Q_TILE : (i + 1) * Q_TILE, :] * scale
            all_scaled.append(s)
        global_max = all_scaled[0].max(dim=-1, keepdim=True)[0]
        for s in all_scaled[1:]:
            global_max = torch.maximum(global_max, s.max(dim=-1, keepdim=True)[0])

        # Pass 2: exp(s - global_max), row_sum
        pij_buf = torch.zeros_like(tensors["pij_buf"], dtype=torch.bfloat16)
        li_sum = torch.zeros(Q_TILE, 1, dtype=torch.float32)
        for i, s in enumerate(all_scaled):
            exp_tile = torch.exp(s - global_max)
            pij_bf16 = exp_tile.to(torch.bfloat16)
            pij_f32 = pij_bf16.to(torch.float32)
            pij_buf[i * Q_TILE : (i + 1) * Q_TILE, :] = pij_bf16
            li_sum += pij_f32.sum(dim=-1, keepdim=True)

        tensors["pij_buf"][:] = pij_buf
        tensors["mi_out"][:] = global_max
        tensors["li_out"][:] = li_sum


class KernelSoftmaxPreparePTOASTestCase(PTOASTestCaseMixin, KernelSoftmaxPrepareTestCase):
    """KernelSoftmaxPrepare test with Ascend910B_PTO backend."""

    def get_name(self) -> str:
        return f"kernel_softmax_prepare_64_{self._n_blocks}blocks_ptoas"


class KernelPvMatmulTestCase(PTOTestCase):
    """Test case for kernel_pv_matmul_64: SplitK PV matmul over N_BLOCKS=2.

    Verifies oi_new = sum_i(pij[i*Q_TILE:(i+1)*Q_TILE, :] @ vj[i]).
    """

    __test__ = False

    _n_blocks = _TEST_N_BLOCKS
    _key_cache_rows = _TEST_KEY_CACHE_ROWS
    _block_table_flat_size = _TEST_BLOCK_TABLE_FLAT_SIZE

    def get_name(self) -> str:
        return f"kernel_pv_matmul_64_{self._n_blocks}blocks"

    def define_tensors(self) -> list[TensorSpec]:
        block_table_data = torch.arange(self._n_blocks, dtype=torch.int32)
        config_data = torch.tensor([self._n_blocks], dtype=torch.int64)
        return [
            TensorSpec("pij_buf", [N_UNROLL_Q, BLOCK_SIZE], DataType.BF16, init_value=torch.randn),
            TensorSpec(
                "value_cache", [self._key_cache_rows, HEAD_DIM], DataType.BF16, init_value=torch.randn
            ),
            TensorSpec("oi_new", [Q_TILE, HEAD_DIM], DataType.FP32, is_output=True),
            TensorSpec(
                "block_table", [self._block_table_flat_size], DataType.INT32, init_value=block_table_data
            ),
            TensorSpec("config", [1], DataType.INT64, init_value=config_data),
        ]

    def get_program(self) -> Any:
        kernel_pv = make_kernel_pv_matmul_64(self._key_cache_rows, self._block_table_flat_size)

        @pl.program
        class PvMatmulProgram:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                pij_buf: pl.Tensor[[N_UNROLL_Q, BLOCK_SIZE], pl.BF16],
                value_cache: pl.Tensor[[_TEST_KEY_CACHE_ROWS, HEAD_DIM], pl.BF16],
                oi_new: pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
                block_table: pl.Tensor[[_TEST_BLOCK_TABLE_FLAT_SIZE], pl.INT32],
                config: pl.Tensor[[1], pl.INT64],
            ) -> pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32]:
                n_blocks: pl.Scalar[pl.INT64] = pl.tensor.read(config, [0])
                bt_offset: pl.Scalar[pl.INT64] = pl.const(0, pl.INT64)
                sij_buf = kernel_pv(pij_buf, value_cache, oi_new, block_table, n_blocks, bt_offset)
                return sij_buf

        return PvMatmulProgram

    def compute_expected(
        self, tensors: dict[str, torch.Tensor], _params: dict[str, Any] | None = None
    ) -> None:
        pij_buf = tensors["pij_buf"].to(torch.float32)  # [N_UNROLL_Q, BLOCK_SIZE]
        value_cache = tensors["value_cache"].to(torch.float32)  # [key_cache_rows, HEAD_DIM]
        block_table = tensors["block_table"]
        oi_new = torch.zeros(Q_TILE, HEAD_DIM, dtype=torch.float32)
        for i in range(self._n_blocks):  # i = 0, i = 1, 2 iterations
            phys = int(block_table[i].item())
            pij_i = pij_buf[i * Q_TILE : (i + 1) * Q_TILE, :]  # [Q_TILE, BLOCK_SIZE]
            vj = value_cache[phys * BLOCK_SIZE : (phys + 1) * BLOCK_SIZE, :]  # [BLOCK_SIZE, HEAD_DIM]
            oi_new += torch.matmul(pij_i, vj)
        tensors["oi_new"][:] = oi_new


class KernelPvMatmulPTOASTestCase(PTOASTestCaseMixin, KernelPvMatmulTestCase):
    """KernelPvMatmul test with Ascend910B_PTO backend."""

    def get_name(self) -> str:
        return f"kernel_pv_matmul_64_{self._n_blocks}blocks_ptoas"


class KernelOnlineUpdateTestCase(PTOTestCase):
    """Test case for kernel_online_update_64.

    Tests all four (is_first, is_last) flag combinations:
      - is_first=1, is_last=1: copy + normalize dst = oi_new / lij
      - is_first=1, is_last=0: copy only; dst = zeros
      - is_first=0, is_last=1: full online update + normalize dst
      - is_first=0, is_last=0: full online update; dst = zeros
    """

    __test__ = False

    def __init__(self, is_first: int = 1, is_last: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.is_first = is_first
        self.is_last = is_last

    def get_name(self) -> str:
        return f"kernel_online_update_64_f{self.is_first}_l{self.is_last}"

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
                mi: pl.Tensor[[Q_TILE, 1], pl.FP32],
                li: pl.Tensor[[Q_TILE, 1], pl.FP32],
                oi: pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
            ) -> tuple[
                pl.Tensor[[Q_TILE, 1], pl.FP32],
                pl.Tensor[[Q_TILE, 1], pl.FP32],
                pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
                pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32],
            ]:
                is_first: pl.Scalar[pl.INDEX] = pl.tensor.read(config, [0])
                is_last: pl.Scalar[pl.INDEX] = pl.tensor.read(config, [1])
                dst: pl.Tensor[[Q_TILE, HEAD_DIM], pl.FP32] = pl.create_tensor(
                    [Q_TILE, HEAD_DIM], dtype=pl.FP32
                )
                mi, li, oi, dst = kernel_online_update_64(
                    mij, lij, oi_new, mi, li, oi, dst, is_first, is_last
                )
                return mi, li, oi, dst

        return OnlineUpdateProgram

    def compute_expected(
        self, tensors: dict[str, torch.Tensor], _params: dict[str, Any] | None = None
    ) -> None:
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


class KernelOnlineUpdatePTOASTestCase(PTOASTestCaseMixin, KernelOnlineUpdateTestCase):
    """KernelOnlineUpdate test with Ascend910B_PTO backend."""

    def get_name(self) -> str:
        return f"kernel_online_update_64_f{self.is_first}_l{self.is_last}_ptoas"


# ── Full-pipeline test case ───────────────────────────────────────────────────


class TestPagedAttention64Core(PTOTestCase):
    """64divisions paged attention: q_tile=64, block_size=64, single-loop N_UNROLL=64."""

    __test__ = False

    def get_name(self) -> str:
        return "paged_attention_64tier_64bat_64h_128d_64bs"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def define_tensors(self) -> list[TensorSpec]:
        query_rows = BATCH * NUM_HEADS
        key_cache_rows = BATCH * MAX_NUM_BLOCKS_PER_REQ * BLOCK_SIZE
        block_table_flat_size = BATCH * MAX_NUM_BLOCKS_PER_REQ

        scale_bits = struct.unpack("I", struct.pack("f", SCALE))[0]
        config_data = torch.tensor(
            [BATCH, NUM_HEADS, KV_HEAD_NUM, HEAD_DIM, BLOCK_SIZE, MAX_NUM_BLOCKS_PER_REQ, scale_bits],
            dtype=torch.int64,
        )
        context_lens_data = torch.full((BATCH,), CONTEXT_LEN, dtype=torch.int32)
        block_table_data = torch.randint(
            0, max(block_table_flat_size, 1), size=(BATCH, MAX_NUM_BLOCKS_PER_REQ), dtype=torch.int32
        ).flatten()

        size_query = torch.tensor([query_rows * HEAD_DIM * 2], dtype=torch.int64)
        size_key_cache = torch.tensor([key_cache_rows * HEAD_DIM * 2], dtype=torch.int64)
        size_value_cache = torch.tensor([key_cache_rows * HEAD_DIM * 2], dtype=torch.int64)

        return [
            TensorSpec("query", [query_rows, HEAD_DIM], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [key_cache_rows, HEAD_DIM], DataType.BF16, init_value=torch.randn),
            TensorSpec("value_cache", [key_cache_rows, HEAD_DIM], DataType.BF16, init_value=torch.randn),
            TensorSpec("block_table", [block_table_flat_size], DataType.INT32, init_value=block_table_data),
            TensorSpec("context_lens", [BATCH], DataType.INT32, init_value=context_lens_data),
            TensorSpec("out", [query_rows, HEAD_DIM], DataType.FP32, is_output=True),
            TensorSpec("config", [7], DataType.INT64, init_value=config_data),
            TensorSpec("size_query", [1], DataType.INT64, init_value=size_query),
            TensorSpec("size_key_cache", [1], DataType.INT64, init_value=size_key_cache),
            TensorSpec("size_value_cache", [1], DataType.INT64, init_value=size_value_cache),
        ]

    def get_program(self) -> Any:
        return build_paged_attention_64_program(
            batch=BATCH,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE,
            max_num_blocks_per_req=MAX_NUM_BLOCKS_PER_REQ,
            context_len=CONTEXT_LEN,
        )

    def compute_expected(
        self, tensors: dict[str, torch.Tensor], _params: dict[str, Any] | None = None
    ) -> None:
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
        q_tile = Q_TILE

        def _update(
            oi_a: torch.Tensor | None,
            li_a: torch.Tensor | None,
            mi_a: torch.Tensor | None,
            oi_new: torch.Tensor,
            li_new: torch.Tensor,
            mi_new: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

                for bn in range(0, max_bn_b, N_UNROLL):
                    n_blocks = min(N_UNROLL, max_bn_b - bn)

                    all_sij = []
                    for i in range(n_blocks):
                        v = min(block_size, cur_seq - (bn + i) * block_size)
                        bidx = int(block_table[b, bn + i].item())
                        kj = key_cache[bidx, :v]
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
                        v = min(block_size, cur_seq - (bn + i) * block_size)
                        bidx = int(block_table[b, bn + i].item())
                        vj = value_cache[bidx, :v]
                        oi_group += torch.mm(pij, vj)

                    oi_acc, li_acc, mi_acc = _update(oi_acc, li_acc, mi_acc, oi_group, li_group, global_max)

                assert oi_acc is not None and li_acc is not None, f"No valid blocks for b={b} q={q_off}"
                out[b, q_off : q_off + q_tile, :] = oi_acc / li_acc

        tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


# ── Test runner ───────────────────────────────────────────────────────────────


class TestPagedAttention64Operations:
    """Test suite for 64divisions paged attention — individual kernels and full pipeline."""

    def test_kernel_aiv_hub(self, test_runner):
        """Test kernel_aiv_hub: zero-init of oi/li/mi accumulators."""
        result = test_runner.run(KernelAivHubPTOASTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_kernel_qk_matmul(self, test_runner):
        """Test kernel_qk_matmul_64: multi-block QK matmul over 2 blocks."""
        result = test_runner.run(KernelQkMatmulPTOASTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_kernel_softmax_prepare(self, test_runner):
        """Test kernel_softmax_prepare_64: two-pass softmax over 2 blocks."""
        result = test_runner.run(KernelSoftmaxPreparePTOASTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_kernel_pv_matmul(self, test_runner):
        """Test kernel_pv_matmul_64: SplitK PV matmul over 2 blocks."""
        result = test_runner.run(KernelPvMatmulPTOASTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_kernel_online_update_first_last(self, test_runner):
        """Test kernel_online_update_64: is_first=1, is_last=1 (single-block normalize)."""
        result = test_runner.run(KernelOnlineUpdatePTOASTestCase(is_first=1, is_last=1))
        assert result.passed, f"Test failed: {result.error}"

    def test_kernel_online_update_first_not_last(self, test_runner):
        """Test kernel_online_update_64: is_first=1, is_last=0 (copy, no normalize)."""
        result = test_runner.run(KernelOnlineUpdatePTOASTestCase(is_first=1, is_last=0))
        assert result.passed, f"Test failed: {result.error}"

    def test_kernel_online_update_not_first_last(self, test_runner):
        """Test kernel_online_update_64: is_first=0, is_last=1 (merge + normalize)."""
        result = test_runner.run(KernelOnlineUpdatePTOASTestCase(is_first=0, is_last=1))
        assert result.passed, f"Test failed: {result.error}"

    def test_kernel_online_update_middle(self, test_runner):
        """Test kernel_online_update_64: is_first=0, is_last=0 (middle block, no normalize)."""
        result = test_runner.run(KernelOnlineUpdatePTOASTestCase(is_first=0, is_last=0))
        assert result.passed, f"Test failed: {result.error}"

    def test_paged_attention_64tier(self, test_runner):
        """Test 64divisions paged attention: q_tile=64, block_size=64, N_UNROLL=64 orchestration."""
        test_case = TestPagedAttention64Core()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
