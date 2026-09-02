# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Basic Fused Operations System Tests for PyPTO.

Corresponds to examples.beginner.scalar_ops / examples.beginner.activation /
examples.intermediate.fused_linear (split from the former 02_fused_ops.py),
implemented using @pl.jit.

Four fused operation patterns are demonstrated:
  1. fused_add_scale     — vector: c = (a + b) * 2.0
  2. fused_add_relu      — vector: c = relu(a + b)
  3. fused_matmul_bias   — cube + vector: c = matmul(a, b) + bias
  4. fused_linear_relu   — cube + vector: y = relu(matmul(x, w) + bias)

Verifies that fused kernels produce results matching PyTorch reference
implementations across three fusion patterns:
  - Vector-only fusion (add+scale, add+relu)
  - Cube+vector fusion (matmul+bias)
  - Full linear layer (matmul+bias+relu)
"""

import pytest
import torch
from examples.beginner.activation import fused_add_relu
from examples.beginner.scalar_ops import fused_add_scale
from examples.intermediate.fused_linear import fused_linear_relu, fused_matmul_bias
from harness import st


def _add_scale_case():
    """Fused add and scale: c = (a + b) * 2.0."""
    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    c = torch.zeros((128, 128), dtype=torch.float32)
    return st.case(fused_add_scale, a, b, c, name="fused_add_scale", golden=lambda _: (a + b) * 2.0)


def _add_relu_case():
    """Fused add and relu: c = relu(a + b)."""
    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    c = torch.zeros((128, 128), dtype=torch.float32)
    return st.case(fused_add_relu, a, b, c, name="fused_add_relu", golden=lambda _: torch.relu(a + b))


def _matmul_bias_case():
    """Fused matmul and bias add: c = matmul(a, b) + bias."""
    torch.manual_seed(0)
    a = torch.full((64, 64), 2.0, dtype=torch.float32)
    b = torch.full((64, 64), 3.0, dtype=torch.float32)
    bias = torch.randn(64, 64, dtype=torch.float32)
    c = torch.zeros((64, 64), dtype=torch.float32)
    return st.case(
        fused_matmul_bias,
        a,
        b,
        bias,
        c,
        name="fused_matmul_bias",
        golden=lambda _: torch.matmul(a, b) + bias,
        rtol=1e-3,
        atol=1e-3,
    )


def _linear_relu_case():
    """Fused linear layer with relu: y = relu(matmul(x, w) + bias)."""
    torch.manual_seed(0)
    x = torch.full((64, 64), 2.0, dtype=torch.float32)
    w = torch.full((64, 64), 3.0, dtype=torch.float32)
    bias = torch.randn(64, 64, dtype=torch.float32)
    y = torch.zeros((64, 64), dtype=torch.float32)
    return st.case(
        fused_linear_relu,
        x,
        w,
        bias,
        y,
        name="fused_linear_relu",
        golden=lambda _: torch.relu(torch.matmul(x, w) + bias),
        rtol=1e-3,
        atol=1e-3,
    )


@st.cases(_add_scale_case(), _add_relu_case(), _matmul_bias_case(), _linear_relu_case())
def test_basic_fused_ops(case_run):
    """Each fused kernel matches its PyTorch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
