# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
FFN Module System Tests for PyPTO.

Three FFN patterns are demonstrated (all on 64x64 tiles):
  1. FFN + GELU   — GELU(hidden @ gate_proj) @ down_proj
  2. FFN + SwiGLU — SwiGLU(hidden @ gate_proj, hidden @ up_proj) @ down_proj
  3. FFN + ReLU   — ReLU(hidden @ gate_proj) @ down_proj
"""

import pytest
import torch
from examples.models.ffn import ffn_gelu, ffn_relu, ffn_swiglu
from harness import st

SHAPE = (64, 64)
# Two chained matmuls in FP32 on 64x64 — looser than the 1e-5 an elementwise
# kernel gets, and unchanged from before the migration.
TOL = {"rtol": 3e-3, "atol": 3e-3}


def _ungated_case(kernel, name, activation):
    """FFN whose gate projection feeds one activation: activation(hidden @ gate) @ down."""
    torch.manual_seed(0)
    hidden = torch.randn(*SHAPE, dtype=torch.float32)
    gate = torch.randn(*SHAPE, dtype=torch.float32)
    down = torch.randn(*SHAPE, dtype=torch.float32)
    output = torch.zeros(*SHAPE, dtype=torch.float32)
    return st.case(
        kernel,
        hidden,
        gate,
        down,
        output,
        name=name,
        golden=lambda _: activation(hidden @ gate) @ down,
        **TOL,
    )


def _swiglu_case():
    """FFN + SwiGLU: (gate_out * sigmoid(gate_out) * up_out) @ down."""
    torch.manual_seed(0)
    hidden = torch.randn(*SHAPE, dtype=torch.float32)
    gate = torch.randn(*SHAPE, dtype=torch.float32)
    up = torch.randn(*SHAPE, dtype=torch.float32)
    down = torch.randn(*SHAPE, dtype=torch.float32)
    output = torch.zeros(*SHAPE, dtype=torch.float32)

    def golden(_):
        gate_out = hidden @ gate
        up_out = hidden @ up
        return (gate_out * torch.sigmoid(gate_out) * up_out) @ down

    return st.case(ffn_swiglu, hidden, gate, up, down, output, name="ffn_swiglu_64x64", golden=golden, **TOL)


@st.cases(
    _ungated_case(ffn_gelu, "ffn_gelu_64x64", lambda h: h * torch.sigmoid(1.702 * h)),
    _swiglu_case(),
    _ungated_case(ffn_relu, "ffn_relu_64x64", torch.relu),
)
def test_ffn_activation(case_run):
    """Each FFN variant matches its torch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
