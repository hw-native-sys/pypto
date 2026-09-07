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
from examples.intermediate.normalization import rms_norm
from harness import st

HIDDEN_SIZE = 64
EPS = 1e-5


def _rms_norm_case():
    """RMSNorm with 32x64 input: normalize by RMS across hidden dim, then scale by gamma."""
    torch.manual_seed(0)
    x = torch.randn(32, HIDDEN_SIZE, dtype=torch.float32)
    gamma = torch.randn(1, HIDDEN_SIZE, dtype=torch.float32)
    output = torch.zeros_like(x)

    def golden(_):
        mean_sq = (x**2).sum(dim=-1, keepdim=True) / HIDDEN_SIZE
        rms = torch.sqrt(mean_sq + EPS)
        return (x / rms) * gamma

    return st.case(rms_norm, x, gamma, output, name="rms_norm_core", golden=golden)


@st.cases(_rms_norm_case())
def test_rms_norm_core(case_run):
    """RMSNorm matches the torch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
