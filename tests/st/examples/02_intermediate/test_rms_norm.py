# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
RMSNorm System Tests for PyPTO.

One RMS normalization pattern is demonstrated:
  1. RMSNorm  — x / sqrt(mean(x^2) + eps) * gamma
"""

import pytest
import torch
from examples.kernels.normalization import rms_norm


class TestRMSNormCore:
    """RMSNorm with 32x64 input: normalize by RMS across hidden dim, then scale by gamma."""

    def test_rms_norm_core(self, test_config):
        rms_norm._cache.clear()
        torch.manual_seed(0)
        x = torch.randn(32, 64, dtype=torch.float32)
        gamma = torch.randn(1, 64, dtype=torch.float32)
        output = torch.zeros_like(x)
        rms_norm(x, gamma, output, config=test_config)

        hidden_size = 64
        eps = 1e-5
        mean_sq = (x**2).sum(dim=-1, keepdim=True) / hidden_size
        rms = torch.sqrt(mean_sq + eps)
        expected = (x / rms) * gamma

        assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), (
            f"rms_norm failed: max diff = {(output - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
