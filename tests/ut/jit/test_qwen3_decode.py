# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration test for the Qwen3-32B JIT decode example.

Verifies that the cross-file ``@pl.jit.inline`` composition in
``examples/models/qwen3_jit/`` compiles end-to-end through the full
pass pipeline, producing the expected post-pass IR shape (one
Orchestration entry + several InCore-class kernels)."""

import pytest

# Module-level skip — tests need torch to build random input tensors.
torch = pytest.importorskip("torch")

from pypto.pypto_core import ir  # noqa: E402

from examples.models.qwen3_jit.config import (  # noqa: E402
    BATCH,
    CACHE_ROWS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HIDDEN,
    MAX_SEQ,
)
from examples.models.qwen3_jit.qwen3_decode import qwen3_decode  # noqa: E402


def _make_args():
    def randn(shape, dtype):
        return torch.empty(shape, dtype=dtype).normal_()

    return [
        randn([BATCH, HIDDEN], torch.bfloat16),
        randn([1, HIDDEN], torch.float32),
        randn([HIDDEN, HIDDEN], torch.bfloat16),
        randn([HIDDEN, KV_HIDDEN], torch.bfloat16),
        randn([HIDDEN, KV_HIDDEN], torch.bfloat16),
        torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32),
        randn([MAX_SEQ, HEAD_DIM], torch.float32),
        randn([MAX_SEQ, HEAD_DIM], torch.float32),
        randn([CACHE_ROWS, HEAD_DIM], torch.bfloat16),
        randn([CACHE_ROWS, HEAD_DIM], torch.bfloat16),
        randn([HIDDEN, HIDDEN], torch.bfloat16),
        randn([1, HIDDEN], torch.float32),
        randn([HIDDEN, INTERMEDIATE], torch.bfloat16),
        randn([HIDDEN, INTERMEDIATE], torch.bfloat16),
        randn([INTERMEDIATE, HIDDEN], torch.bfloat16),
        torch.empty([BATCH, HIDDEN], dtype=torch.bfloat16),
    ]


class TestQwen3JITCompile:
    """End-to-end compile of the Qwen3 JIT example."""

    def test_qwen3_decode_compile_for_test(self):
        """compile_for_test runs the full pipeline; the post-pass IR drops all
        Inline functions and outlines pl.at scopes into InCore-class kernels."""
        post_pass = qwen3_decode.compile_for_test(*_make_args())
        names = sorted(f.name for f in post_pass.functions.values())

        # No FunctionType.Inline survives the InlineFunctions pass. Note that
        # we can't compare by *name* because OutlineIncoreScopes names each
        # outlined kernel after its ``pl.at(name_hint=...)`` — and a kernel's
        # name_hint may incidentally match the inline utility's Python name
        # (e.g. ``post_rmsnorm`` appears as both, but the post-pass instance
        # is the outlined InCore function, not the inline source).
        for fn in post_pass.functions.values():
            assert fn.func_type != ir.FunctionType.Inline, (
                f"FunctionType.Inline function '{fn.name}' should have been "
                f"spliced and removed by InlineFunctions"
            )

        # The entry survives.
        assert "qwen3_decode" in names

        # OutlineIncoreScopes extracts each ``pl.at`` block into a separate
        # InCore-class function. Expect one per name_hint in the kernel files.
        expected_outlined_hints = {
            "rmsnorm",  # input_rmsnorm scope
            "post_rmsnorm",  # post_rmsnorm scope
            "q_proj",  # q_projection scope
            "k_proj",  # k_projection scope
            "v_proj",  # v_projection scope
            "out_proj_residual",  # out_projection_residual scope (with split)
            "down_proj_residual",  # down_projection_residual scope (with split)
            "gate_proj",  # mlp_block first scope
            "up_proj",  # mlp_block second scope
            "silu",  # mlp_block third scope
            "rope_kv_cache",  # rope_kv_cache_update scope
        }
        for hint in expected_outlined_hints:
            assert any(n.startswith(hint) or n == hint for n in names), (
                f"Expected an outlined function for ``pl.at(name_hint='{hint}')``; got functions: {names}"
            )

    def test_qwen3_decode_post_pass_has_orchestration_entry(self):
        """The entry function lands as Orchestration after outlining."""
        post_pass = qwen3_decode.compile_for_test(*_make_args())
        entry = post_pass.get_function("qwen3_decode")
        assert entry is not None
        assert entry.func_type == ir.FunctionType.Orchestration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
