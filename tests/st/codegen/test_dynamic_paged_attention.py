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
load operations use concrete closure variables for tile sizes.
Matches the DynShapeAddTestCase pattern from test_dynamic_shape.py.

Test cases:
  DynamicPagedAttentionTestCase     — full paged attention with dynamic kernel shapes
"""

from typing import Any

import pytest
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

from examples.ir_parser.dynamic_paged_attention_example import (
    build_dynamic_paged_attention_program,
)
from tests.st.codegen.test_paged_attention import PagedAttentionTestCase

# ---------------------------------------------------------------------------
# Test Case — DynamicPagedAttentionTestCase
# Full paged attention with dynamic InCore kernel type annotations.
# ---------------------------------------------------------------------------


class DynamicPagedAttentionTestCase(PagedAttentionTestCase):
    """Full paged attention with dynamic-shape InCore kernel type annotations.

    InCore kernels (init_inplace, qk_matmul, softmax_prepare, pv_matmul,
    online_update) annotate their tensor shapes with pl.dynamic() variables
    (Q_HEADS, HEAD_DIM_DYN, BLOCK_SIZE_DYN) instead of literal numbers.
    Load operations inside the kernels use concrete closure variables, matching
    the DynShapeAddTestCase pattern from test_dynamic_shape.py.

    Inherits define_tensors and compute_expected from PagedAttentionTestCase —
    the pipeline logic and golden are identical to the static version.
    """

    def get_name(self) -> str:
        return (
            f"dynamic_paged_attention_{self.batch}bat_{self.num_heads}h_{self.head_dim}d_{self.block_size}bs"
        )

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B_PTO

    def get_program(self) -> Any:
        return build_dynamic_paged_attention_program(
            batch=self.batch,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
        )


# ---------------------------------------------------------------------------
# pytest test suite
# ---------------------------------------------------------------------------


class TestDynamicPagedAttentionKernels:
    """Integration tests for the dynamic shapes pattern.

    test_dynamic_paged_attention:
        Exercises the full 4-kernel paged attention pipeline where InCore kernel
        type annotations use pl.dynamic() variables for the tile shape dims.
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
