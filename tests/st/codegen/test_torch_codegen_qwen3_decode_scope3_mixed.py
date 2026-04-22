# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System test for torch codegen on Qwen3 decode scope3 mixed kernel.

Generates executable PyTorch code from the runtime scope3-mixed IR via
torch_codegen, runs it with test tensors, and compares outputs to the
golden reference.
"""

import pytest
import torch
from harness.core.harness import PLATFORMS, platform_to_backend
from pypto.backend import reset_for_testing, set_backend_type
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

from tests.st.runtime.test_qwen3_decode_scope3_mixed import (
    build_qwen3_scope3_program,
    golden,
)


def _build_tensors(batch: int, hidden_size: int, intermediate_size: int) -> dict[str, torch.Tensor]:
    attn_out = (torch.randn([batch, hidden_size], dtype=torch.float32) / (hidden_size**0.5)).to(
        torch.bfloat16
    )
    hidden_states = (torch.randn([batch, hidden_size], dtype=torch.float32) / (hidden_size**0.5)).to(
        torch.bfloat16
    )
    wo = (torch.randn([hidden_size, hidden_size], dtype=torch.float32) / (hidden_size**0.5)).to(
        torch.bfloat16
    )
    post_rms_weight = torch.randn([1, hidden_size], dtype=torch.float32) / (hidden_size**0.5)
    w_gate = (
        torch.randn([hidden_size, intermediate_size], dtype=torch.float32) / (intermediate_size**0.5)
    ).to(torch.bfloat16)
    w_up = (torch.randn([hidden_size, intermediate_size], dtype=torch.float32) / (intermediate_size**0.5)).to(
        torch.bfloat16
    )
    w_down = (torch.randn([intermediate_size, hidden_size], dtype=torch.float32) / (hidden_size**0.5)).to(
        torch.bfloat16
    )
    out = torch.zeros([batch, hidden_size], dtype=torch.bfloat16)

    return {
        "attn_out": attn_out,
        "hidden_states": hidden_states,
        "wo": wo,
        "post_rms_weight": post_rms_weight,
        "w_gate": w_gate,
        "w_up": w_up,
        "w_down": w_down,
        "out": out,
    }


def _run_scope3_generated_code_and_check(
    code: str,
    tensors: dict[str, torch.Tensor],
    expected: torch.Tensor,
) -> None:
    ns: dict = {}
    exec(code, ns)  # noqa: S102

    codegen_out = tensors["out"].clone()
    ns["scope3"](
        tensors["attn_out"],
        tensors["hidden_states"],
        tensors["wo"],
        tensors["post_rms_weight"],
        tensors["w_gate"],
        tensors["w_up"],
        tensors["w_down"],
        codegen_out,
    )

    assert torch.allclose(codegen_out, expected, rtol=5e-2, atol=5e-2), (
        f"out max abs diff = {(expected - codegen_out).abs().max().item():.6e}"
    )


def _run_codegen_after_default_pass_and_check(
    program,
    tensors: dict[str, torch.Tensor],
    expected: torch.Tensor,
    platform: str,
) -> None:
    backend_type = platform_to_backend(platform)

    reset_for_testing()
    set_backend_type(backend_type)
    try:
        transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
        code = torch_codegen(transformed, check_shapes=True)
    finally:
        reset_for_testing()

    assert "_cross_core_rt.push_to_" in code
    assert "_cross_core_rt.pop_from_" in code

    _run_scope3_generated_code_and_check(code, tensors, expected)


def test_qwen3_decode_scope3_mixed_codegen_vs_golden():
    """Torch codegen of qwen3 decode scope3 mixed should match golden."""
    # Keep dimensions moderate for system-test runtime while preserving
    # K_CHUNK/Q_OUT_CHUNK/MLP_OUT_CHUNK divisibility.
    batch = 16
    hidden_size = 512
    intermediate_size = 1024

    program = build_qwen3_scope3_program(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    code = torch_codegen(program, check_shapes=True)

    torch.manual_seed(42)
    tensors = _build_tensors(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    golden_tensors = {k: v.clone() for k, v in tensors.items()}
    golden(golden_tensors)
    golden_out = golden_tensors["out"]
    _run_scope3_generated_code_and_check(code, tensors, golden_out)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_qwen3_decode_scope3_mixed_codegen_after_default_pass_vs_golden(platform: str):
    """Pass-expanded qwen3 decode scope3 mixed should match golden."""
    batch = 16
    hidden_size = 512
    intermediate_size = 1024

    program = build_qwen3_scope3_program(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    torch.manual_seed(42)
    tensors = _build_tensors(
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    golden_tensors = {k: v.clone() for k, v in tensors.items()}
    golden(golden_tensors)
    golden_out = golden_tensors["out"]

    _run_codegen_after_default_pass_and_check(
        program=program,
        tensors=tensors,
        expected=golden_out,
        platform=platform,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
