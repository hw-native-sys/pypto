# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for DAG (Directed Acyclic Graph) operations using PyPTO frontend.

This test validates complex multi-kernel orchestration with mixed operations,
ensuring correct code generation and execution for DAG-structured computations.

The JIT entry is imported from examples/models/vector_dag.py to keep a single
source of truth and ensure examples are guarded by tests.
"""

import pytest
import torch
from examples.models.vector_dag import golden, vector_dag


class TestDAGOperations:
    """Test suite for DAG operations."""

    def test_vector_dag(self, test_config):
        """Test vector DAG computation with 128x128 shape.

        Implements: f = (a + b + 1)(a + b + 2) + (a + b)
        """
        vector_dag._cache.clear()
        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        f = torch.zeros((128, 128), dtype=torch.float32)

        vector_dag(a, b, f, config=test_config)

        # Reference via the example's golden() function (single source of truth).
        ref_tensors = {"a": a, "b": b, "f": torch.zeros_like(f)}
        golden(ref_tensors)
        expected = ref_tensors["f"]
        assert torch.allclose(f, expected, rtol=1e-5, atol=1e-5), (
            f"vector_dag failed: max diff = {(f - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
