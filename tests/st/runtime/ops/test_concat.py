# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile.concat (column-wise concatenation) using @pl.jit."""

import pytest
import torch
from examples.beginner.concat import tile_concat_32x32


class TestConcatOperations:
    """Test suite for tile.concat operations."""

    def test_tile_concat_32x32(self, test_config):
        """Test tile concatenation: 32x16 + 32x16 -> 32x32.

        Uses random data on purpose: with a constant-filled ``a``, a concat that
        overwrites rows of its own source before reading them still produces the
        expected output (every row holds the same value), so the corruption is
        invisible. Distinct per-row values are what make such a defect fail here.
        """
        tile_concat_32x32._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(32, 16, dtype=torch.float32)
        b = torch.randn(32, 16, dtype=torch.float32)
        c = torch.zeros((32, 32), dtype=torch.float32)
        tile_concat_32x32(a, b, c, config=test_config)
        expected = torch.cat([a, b], dim=1)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"tile_concat_32x32 failed: max diff = {(c - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
