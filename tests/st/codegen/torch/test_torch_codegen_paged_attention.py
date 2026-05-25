# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for torch codegen on paged attention variants.

Generates executable PyTorch code from paged attention IR via torch_codegen,
runs it with test tensors, and compares the output to the golden reference.
"""

import pytest
import torch
from pypto.debug import torch_codegen

# ---------------------------------------------------------------------------
# Paged attention (static shapes, config tensor)
# ---------------------------------------------------------------------------


def test_paged_attention_codegen_vs_golden():
    """Torch codegen of paged attention should match the golden reference."""
    from examples.models.paged_attention import (  # noqa: PLC0415
        build_paged_attention_program,
        build_tensors,
        golden,
    )

    batch, num_heads, head_dim, block_size = 1, 16, 128, 128
    max_num_blocks_per_req, context_len, scale = 2, 200, 1.0

    program = build_paged_attention_program(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
    )
    code = torch_codegen(program, check_shapes=True)

    torch.manual_seed(42)
    query, key_cache, value_cache, block_table, context_lens, out, config = build_tensors(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
        context_len=context_len,
        scale=scale,
    )
    tensors = {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "context_lens": context_lens,
        "out": out,
        "config": config,
    }

    golden_tensors = {k: v.clone() for k, v in tensors.items()}
    golden(golden_tensors)
    golden_out = golden_tensors["out"]

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    codegen_out = tensors["out"].clone()
    ns["paged_attention"](
        tensors["query"],
        tensors["key_cache"],
        tensors["value_cache"],
        tensors["block_table"],
        tensors["context_lens"],
        codegen_out,
        tensors["config"],
    )

    assert torch.allclose(codegen_out, golden_out, rtol=5e-2, atol=5e-2), (
        f"max abs diff = {(golden_out - codegen_out).abs().max().item():.6e}"
    )


# ---------------------------------------------------------------------------
# Dynamic paged attention (shape derived from tensors at runtime)
# ---------------------------------------------------------------------------


def test_dynamic_paged_attention_codegen_vs_golden():
    """Torch codegen of dynamic paged attention should match the golden reference."""
    from examples.models.paged_attention_dynamic import (  # noqa: PLC0415
        build_dynamic_paged_attention_program,
        build_tensors,
        golden,
    )

    batch, num_heads, head_dim, block_size = 1, 16, 128, 128
    max_num_blocks_per_req, context_len = 2, 200

    program = build_dynamic_paged_attention_program(
        q_tile=16,
        head_dim=head_dim,
        block_size=block_size,
    )
    code = torch_codegen(program, check_shapes=True)

    torch.manual_seed(42)
    query, key_cache, value_cache, block_table, context_lens, out = build_tensors(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks_per_req=max_num_blocks_per_req,
        context_len=context_len,
    )
    tensors = {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "context_lens": context_lens,
        "out": out,
    }

    golden_tensors = {k: v.clone() for k, v in tensors.items()}
    golden(golden_tensors)
    golden_out = golden_tensors["out"]

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    codegen_out = tensors["out"].clone()
    ns["paged_attention"](
        tensors["query"],
        tensors["key_cache"],
        tensors["value_cache"],
        tensors["block_table"],
        tensors["context_lens"],
        codegen_out,
    )

    assert torch.allclose(codegen_out, golden_out, rtol=5e-2, atol=5e-2), (
        f"max abs diff = {(golden_out - codegen_out).abs().max().item():.6e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
