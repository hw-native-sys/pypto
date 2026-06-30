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
    def empty(shape, dtype):
        # ``compile_for_test`` consumes only tensor *shape* and *dtype* metadata
        # (via ``_bind_args``); it never reads the tensor values. Filling the
        # Qwen3-32B weights + 512K-row KV cache with ``torch.*.normal_()`` is
        # ~37s of pure overhead per call (916M elements), so allocate
        # uninitialized memory instead.
        return torch.empty(shape, dtype=dtype)

    return [
        empty([BATCH, HIDDEN], torch.bfloat16),
        empty([1, HIDDEN], torch.float32),
        empty([HIDDEN, HIDDEN], torch.bfloat16),
        empty([HIDDEN, KV_HIDDEN], torch.bfloat16),
        empty([HIDDEN, KV_HIDDEN], torch.bfloat16),
        torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32),
        empty([MAX_SEQ, HEAD_DIM], torch.float32),
        empty([MAX_SEQ, HEAD_DIM], torch.float32),
        empty([CACHE_ROWS, HEAD_DIM], torch.bfloat16),
        empty([CACHE_ROWS, HEAD_DIM], torch.bfloat16),
        empty([HIDDEN, HIDDEN], torch.bfloat16),
        empty([1, HIDDEN], torch.float32),
        empty([HIDDEN, INTERMEDIATE], torch.bfloat16),
        empty([HIDDEN, INTERMEDIATE], torch.bfloat16),
        empty([INTERMEDIATE, HIDDEN], torch.bfloat16),
        torch.empty([BATCH, HIDDEN], dtype=torch.bfloat16),
    ]


# Module-level cache so the (non-trivial) end-to-end compile runs once and is
# shared by both tests below — they assert on the *same* post-pass program, so
# compiling per test only doubles the pipeline cost. A *function*-scoped fixture
# (memoized here) is used rather than ``scope="class"`` on purpose: a
# class-scoped fixture sets up *before* the function-scoped autouse fixtures in
# ``tests/ut/conftest.py``, so the compile would escape the per-test
# ``PYPTO_PROG_BUILD_DIR`` redirect and leave stale build_output dirs in the
# repo. A function-scoped fixture runs after those autouse fixtures, so the
# shared compile's artifacts land in pytest's tmp dir.
_POST_PASS: list = []


@pytest.fixture
def post_pass():
    """Compile the Qwen3 decode example once; shared across this module's tests."""
    if not _POST_PASS:
        _POST_PASS.append(qwen3_decode.compile_for_test(*_make_args()))
    return _POST_PASS[0]


class TestQwen3JITCompile:
    """End-to-end compile of the Qwen3 JIT example."""

    def test_qwen3_decode_compile_for_test(self, post_pass):
        """compile_for_test runs the full pipeline; the post-pass IR drops all
        Inline functions and outlines pl.at scopes into InCore-class kernels."""
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

    def test_qwen3_decode_post_pass_has_orchestration_entry(self, post_pass):
        """The entry function lands as Orchestration after outlining."""
        entry = post_pass.get_function("qwen3_decode")
        assert entry is not None
        assert entry.func_type == ir.FunctionType.Orchestration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
