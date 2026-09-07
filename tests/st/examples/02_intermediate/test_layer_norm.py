# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
LayerNorm System Tests for PyPTO.

One layer normalization pattern is demonstrated:
  1. LayerNorm  — (x - mean) / sqrt(var + eps) * gamma + beta
"""

import pytest
import torch
from examples.intermediate.normalization import layer_norm
from harness import st

HIDDEN_SIZE = 64
EPS = 1e-5


def _layer_norm_case():
    """LayerNorm with 32x64 input: normalize across hidden dim, then scale and shift."""
    torch.manual_seed(0)
    x = torch.randn(32, HIDDEN_SIZE, dtype=torch.float32)
    gamma = torch.randn(1, HIDDEN_SIZE, dtype=torch.float32)
    beta = torch.randn(1, HIDDEN_SIZE, dtype=torch.float32)
    output = torch.zeros_like(x)

    def golden(_):
        mean = x.sum(dim=-1, keepdim=True) / HIDDEN_SIZE
        centered = x - mean
        var = (centered**2).sum(dim=-1, keepdim=True) / HIDDEN_SIZE
        std = torch.sqrt(var + EPS)
        return (centered / std) * gamma + beta

    return st.case(layer_norm, x, gamma, beta, output, name="layer_norm_core", golden=golden)


@st.cases(_layer_norm_case())
def test_layer_norm_core(case_run):
    """LayerNorm matches the torch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
