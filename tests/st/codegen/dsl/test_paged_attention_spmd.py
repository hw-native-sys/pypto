# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
SPMD tests for Paged Attention kernels.

Test cases:
  PagedAttentionSpmdTestCase     — full SPMD paged attention pipeline
"""

import importlib
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

_spmd_module = importlib.import_module("examples.models.09_paged_attention_spmd")
build_paged_attention_spmd_program = _spmd_module.build_paged_attention_spmd_program
build_tensor_specs = _spmd_module.build_tensor_specs
golden = _spmd_module.golden


class PagedAttentionSpmdTestCase(PTOTestCase):
    """Full SPMD paged attention pipeline backed by the example golden.

    Unlike the dynamic-shape variant, this test reuses the example's config
    tensor and tensor-spec builder directly. Runtime metadata such as
    ``active_num_blocks`` is derived from ``context_len`` so the system test
    matches the example's orchestration entry as closely as possible.
    """

    def __init__(
        self,
        batch: int = 4,
        num_heads: int = 16,
        head_dim: int = 128,
        block_size: int = 128,
        context_len: int | Sequence[int] | torch.Tensor = 8192,
        max_model_len: int = 32768,
        scale: float = 1.0,
        q_tile: int = 16,
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
        self.max_num_blocks_per_req = max_model_len // block_size
        self.active_num_blocks = self._resolve_active_num_blocks()

    def _context_lens_tensor(self) -> torch.Tensor:
        if isinstance(self.context_len, torch.Tensor):
            context_lens = self.context_len.to(dtype=torch.int32)
        elif isinstance(self.context_len, Sequence) and not isinstance(self.context_len, (str, bytes)):
            context_lens = torch.tensor(list(self.context_len), dtype=torch.int32)
        else:
            context_lens = torch.full((self.batch,), int(self.context_len), dtype=torch.int32)

        if context_lens.numel() != self.batch:
            raise ValueError(
                f"context_len must provide exactly {self.batch} elements, got {context_lens.numel()}"
            )
        return context_lens

    def _resolve_active_num_blocks(self) -> int:
        context_lens = self._context_lens_tensor()
        max_context_len = int(context_lens.max().item()) if self.batch > 0 else 0
        max_blocks_from_context = (max_context_len + self.block_size - 1) // self.block_size
        return min(self.max_num_blocks_per_req, max_blocks_from_context)

    def get_name(self) -> str:
        if isinstance(self.context_len, Sequence) and not isinstance(
            self.context_len, (str, bytes, torch.Tensor)
        ):
            context_tag = "varctx"
        elif isinstance(self.context_len, torch.Tensor) and self.context_len.numel() > 1:
            context_tag = "varctx"
        else:
            context_tag = f"{int(self._context_lens_tensor()[0].item())}ctx"
        return (
            f"paged_attention_spmd_{self.batch}bat_{self.num_heads}h_"
            f"{self.head_dim}d_{self.block_size}bs_{context_tag}"
        )

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def get_program(self) -> Any:
        return build_paged_attention_spmd_program(
            batch=self.batch,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            q_tile=self.q_tile,
        )

    def define_tensors(self) -> list[TensorSpec]:
        runtime_specs = build_tensor_specs(
            batch=self.batch,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            active_num_blocks=self.active_num_blocks,
            context_len=self._context_lens_tensor(),
            scale=self.scale,
        )
        dtype_map = {
            torch.bfloat16: DataType.BF16,
            torch.float32: DataType.FP32,
            torch.int32: DataType.INT32,
            torch.int64: DataType.INT64,
        }
        return [
            TensorSpec(
                spec.name,
                list(spec.shape),
                dtype_map[spec.dtype],
                init_value=spec.init_value,
                is_output=spec.is_output,
            )
            for spec in runtime_specs
        ]

    def compute_expected(self, tensors, params=None):
        golden(tensors, params)


class PTOASTestCaseMixin:
    """Mixin for test cases using PTO backend and Default optimization strategy."""

    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


class PagedAttentionSpmdPTOASTestCase(PTOASTestCaseMixin, PagedAttentionSpmdTestCase):
    """SPMD paged attention using the PTO backend + PTOAS toolchain."""

    def get_name(self) -> str:
        base_name = super().get_name()
        return f"{base_name}_ptoas"


class TestPagedAttentionSpmdKernels:
    """Integration tests for the SPMD paged attention pipeline.

    test_paged_attention_spmd_ptoas:
        Exercises the full SPMD paged attention pipeline implemented in
        ``examples.models.09_paged_attention_spmd``.
    """

    @pytest.mark.parametrize(
        "batch,num_heads,head_dim,block_size,context_len,max_model_len",
        [
            (256, 16, 128, 128, 8192, 32768),
            (256, 64, 128, 64, 8192, 32768),
            (64, 64, 256, 64, 8192, 32768),
            # (4, 16, 128, 128, [512, 4096, 8192, 768], 32768),
        ],
    )
    def test_paged_attention_spmd_ptoas(
        self, test_runner, batch, num_heads, head_dim, block_size, context_len, max_model_len
    ):
        """Test full SPMD paged attention with the PTO backend + PTOAS toolchain."""
        test_case = PagedAttentionSpmdPTOASTestCase(
            batch=batch,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            context_len=context_len,
            max_model_len=max_model_len,
        )
        result = test_runner.run(test_case)
        assert result.passed, f"SPMD paged attention test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
