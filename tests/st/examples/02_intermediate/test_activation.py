# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Activation Function System Tests for PyPTO.

Four activation patterns are demonstrated, all on 32x128 input:
  1. SiLU   — x * sigmoid(x)
  2. GELU   — x * sigmoid(1.702 * x)
  3. SwiGLU — gate * sigmoid(gate) * up
  4. GeGLU  — gate * sigmoid(1.702 * gate) * up
"""

import pytest
import torch
from examples.beginner.activation import geglu, gelu, silu, swiglu
from harness import st

SHAPE = (32, 128)


def _unary_case(kernel, name, golden_of_x):
    """A one-input activation: output = f(x)."""
    torch.manual_seed(0)
    x = torch.randn(*SHAPE, dtype=torch.float32)
    output = torch.zeros_like(x)
    return st.case(kernel, x, output, name=name, golden=lambda _: golden_of_x(x))


def _gated_case(kernel, name, golden_of_gate_up):
    """A gated activation: output = f(gate, up)."""
    torch.manual_seed(0)
    gate = torch.randn(*SHAPE, dtype=torch.float32)
    up = torch.randn(*SHAPE, dtype=torch.float32)
    output = torch.zeros_like(gate)
    return st.case(kernel, gate, up, output, name=name, golden=lambda _: golden_of_gate_up(gate, up))


@st.cases(
    _unary_case(silu, "silu", lambda x: x * torch.sigmoid(x)),
    _unary_case(gelu, "gelu", lambda x: x * torch.sigmoid(1.702 * x)),
    _gated_case(swiglu, "swiglu", lambda gate, up: gate * torch.sigmoid(gate) * up),
    _gated_case(geglu, "geglu", lambda gate, up: gate * torch.sigmoid(1.702 * gate) * up),
)
def test_activation(case_run):
    """Each activation kernel matches its torch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
