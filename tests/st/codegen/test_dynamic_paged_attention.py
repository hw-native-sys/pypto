# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Dynamic shape tests for Paged Attention kernels.

Dynamic shapes — InCore kernel type annotations use pl.dynamic() variables
(Q_HEADS, HEAD_DIM_DYN, BLOCK_SIZE_DYN) instead of literal numbers, while
load operations use concrete closure variables (_Q_TILE, _HEAD_DIM, _BLOCK_SIZE)
captured from build_dynamic_paged_attention_program() for tile sizes.
Matches the DynShapeAddTestCase pattern from test_dynamic_shape.py.

Test cases:
  DynamicPagedAttentionTestCase     — full paged attention with dynamic kernel shapes
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

from examples.ir_parser.dynamic_paged_attention_example import (
    build_dynamic_paged_attention_program,
)

# ---------------------------------------------------------------------------
# Test Case — DynamicPagedAttentionTestCase
# Full paged attention with dynamic InCore kernel type annotations.
# ---------------------------------------------------------------------------


class DynamicPagedAttentionTestCase(PTOTestCase):
    """Full paged attention with dynamic-shape InCore kernel type annotations.

    InCore kernels (init_inplace, qk_matmul, softmax_prepare, pv_matmul,
    online_update) annotate their tensor shapes with pl.dynamic() variables
    (Q_HEADS, HEAD_DIM_DYN, BLOCK_SIZE_DYN) instead of literal numbers.
    Load operations inside the kernels use concrete closure variables (_Q_TILE,
    _HEAD_DIM, _BLOCK_SIZE) captured from build_dynamic_paged_attention_program(),
    matching the DynShapeAddTestCase pattern from test_dynamic_shape.py.

    Unlike PagedAttentionTestCase there is no config tensor — all shape
    parameters are derived from tensor dimensions at runtime (mirroring the
    orchestration function's pl.tensor.dim() calls).  Scale is hardcoded to 1.0.
    """

    def __init__(
        self,
        batch: int = 64,
        num_heads: int = 16,
        head_dim: int = 128,
        block_size: int = 128,
        context_len: int | list[int] = 8192,
        max_model_len: int = 32768,
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
        self.max_num_blocks_per_req = max_model_len // block_size

    def get_name(self) -> str:
        return (
            f"dynamic_paged_attention_{self.batch}bat_{self.num_heads}h_{self.head_dim}d_{self.block_size}bs"
        )

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def get_program(self) -> Any:
        return build_dynamic_paged_attention_program(
            q_tile=16,
            head_dim=self.head_dim,
            block_size=self.block_size,
        )

    def define_tensors(self) -> list[TensorSpec]:
        B = self.batch
        H = self.num_heads
        D = self.head_dim
        BS = self.block_size
        max_blocks = self.max_num_blocks_per_req  # max KV blocks per request
        # Total rows in the flat KV-cache pool: all requests × max blocks × tokens per block
        total_pool_rows = B * max_blocks * BS

        # Build a random page table: each request gets max_blocks physical block indices
        # sampled from [0, B*max_blocks).  Flattened to 1-D to match orchestration input.
        block_table = torch.randint(
            0, max(B * max_blocks, 1), size=(B, max_blocks), dtype=torch.int32
        ).flatten()

        # context_lens can be a scalar (all requests have equal length) or a per-request list
        if isinstance(self.context_len, list):
            if len(self.context_len) != B:
                raise ValueError(
                    f"context_len list length {len(self.context_len)} does not match batch size B={B}"
                )
            context_lens = torch.tensor(self.context_len, dtype=torch.int32)
        else:
            context_lens = torch.full((B,), self.context_len, dtype=torch.int32)

        return [
            TensorSpec("query", [B * H, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("key_cache", [total_pool_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("value_cache", [total_pool_rows, D], DataType.BF16, init_value=torch.randn),
            TensorSpec("block_table", [B * max_blocks], DataType.INT32, init_value=block_table),
            TensorSpec("context_lens", [B], DataType.INT32, init_value=context_lens),
            TensorSpec("out", [B * H, D], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors, params=None):
        context_lens = tensors["context_lens"]
        query = tensors["query"]
        key_cache = tensors["key_cache"]
        value_cache = tensors["value_cache"]
        block_table_flat = tensors["block_table"]

        # Derive shape parameters from tensor dimensions — no config tensor, mirrors
        # the orchestration function's pl.tensor.dim() derivations.
        batch = context_lens.shape[0]
        num_heads = query.shape[0] // batch
        head_dim = query.shape[1]
        # block_size = value_cache rows / block_table rows (rows per physical block)
        block_size = value_cache.shape[0] // block_table_flat.shape[0]
        max_num_blocks_per_req = block_table_flat.shape[0] // batch
        scale = 1.0  # hardcoded to match orchestration function

        # Reshape flat tensors into 3-D views for batch-matmul operations
        query_3d = query.float().reshape(batch, num_heads, head_dim)
        total_pool_blocks = batch * max_num_blocks_per_req
        key_cache_3d = key_cache.float().reshape(total_pool_blocks, block_size, head_dim)
        value_cache_3d = value_cache.float().reshape(total_pool_blocks, block_size, head_dim)
        block_table = block_table_flat.reshape(batch, max_num_blocks_per_req)

        out = torch.zeros((batch, num_heads, head_dim), dtype=torch.float32)
        q_tile = 16
        # Upper bound on KV blocks to iterate over (driven by longest request in batch)
        max_bn = int((context_lens.max().item() + block_size - 1) // block_size)

        for q_offset in range(0, num_heads, q_tile):
            q_tile_size = min(q_tile, num_heads - q_offset)
            qi = query_3d[:, q_offset : q_offset + q_tile_size, :]
            # Online softmax state across KV blocks
            oi, li, mi = None, None, None

            for bn in range(max_bn):
                # valid_lens[b]: how many tokens of block bn are valid for request b
                valid_lens = torch.clamp(context_lens - bn * block_size, min=0, max=block_size)
                if not (valid_lens > 0).any():
                    break
                block_indices = block_table[:, bn]
                kj_all = key_cache_3d[block_indices].float()
                vj_all = value_cache_3d[block_indices].float()

                # Stage 1: QK matmul + mask padding columns with -inf
                sij = torch.bmm(qi, kj_all.transpose(1, 2)) * scale
                pos = torch.arange(block_size).unsqueeze(0)
                valid_mask = (pos < valid_lens.unsqueeze(1)).unsqueeze(1)
                sij = sij.masked_fill(~valid_mask, float("-inf"))

                # Stage 2: softmax prepare — row_max, exp(sij - max), BF16 cast, row_sum
                mij = sij.max(dim=-1, keepdim=True)[0].clamp(min=-1e30)
                pij = torch.exp(sij - mij).masked_fill(~valid_mask, 0.0)
                # Simulate BF16 cast to match InCore kernel precision
                pij = pij.to(torch.bfloat16).to(torch.float32)
                lij = pij.sum(dim=-1, keepdim=True)

                # Stage 3: PV matmul
                oi_new = torch.bmm(pij, vj_all)

                # Stage 4: online softmax accumulator update
                if bn == 0:
                    # First block: initialise accumulators directly
                    oi, li, mi = oi_new, lij, mij
                else:
                    # Subsequent block: rescale old and new partial results then merge
                    mi_new = torch.maximum(mi, mij)
                    alpha = torch.exp(mi - mi_new)  # rescale factor for old accumulator
                    beta = torch.exp(mij - mi_new)  # rescale factor for new block
                    li = alpha * li + beta * lij
                    oi = alpha * oi + beta * oi_new
                    mi = mi_new

            # Final normalisation: oi / li, write into the corresponding output slice
            out[:, q_offset : q_offset + q_tile_size, :] = oi / li

        tensors["out"][:] = out.reshape(batch * num_heads, head_dim)


# ---------------------------------------------------------------------------
# pytest test suite
# ---------------------------------------------------------------------------


class TestDynamicPagedAttentionKernels:
    """Integration tests for the dynamic shapes pattern.

    test_dynamic_paged_attention:
        Exercises the full 5-kernel paged attention pipeline (init_inplace,
        qk_matmul, softmax_prepare, pv_matmul, online_update) where InCore
        kernel type annotations use pl.dynamic() variables for the tile shape dims.
    """

    @pytest.mark.parametrize(
        "batch,num_heads,head_dim,block_size,context_len,max_model_len",
        [
            (256, 16, 128, 128, 8192, 32768),
            (256, 64, 128, 64, 8192, 32768),
            (64, 64, 256, 64, 8192, 32768),
            # Variable context lengths: each of 4 requests has a different length
            (4, 16, 128, 128, [512, 4096, 8192, 768], 32768),
        ],
    )
    def test_dynamic_paged_attention(
        self, test_runner, batch, num_heads, head_dim, block_size, context_len, max_model_len
    ):
        """Test full paged attention with dynamic InCore kernel type annotations."""
        test_case = DynamicPagedAttentionTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            context_len=context_len,
            max_model_len=max_model_len,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"Dynamic paged attention test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
