# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end ST test for issue #1580 — the orchestration phi-node naming swap.

Problem scenario: a ``pl.Scalar`` (``global_c_idx``) is defined inside a
``pl.spmd`` body, and the *same name* is reused in a later ``pl.parallel`` +
``pl.at`` + ``pl.range`` scope that writes two output tensors. ``ConvertToSSA``
promotes the scalar into the ``pl.parallel`` ``init_values`` tuple, mixed in
with the tensor carries, and a passed-through ``tile`` carry trails the real
outputs. The carry tuple becomes ``(global_c_idx, out_b, out_c, tile)`` — a
scalar leads, the outputs sit in the middle, a non-output tensor trails.

Before the fix, the orchestration codegen mapped each return-tuple element onto
the callee's Out params by positional tail-alignment, which assumed all
non-output elements form a leading prefix. The interleaved layout rotated the
output names by one, so the generated orchestration C++ referenced an
undeclared ``out_b__rv_v*`` and aliased ``out_c`` / ``tile`` to the wrong
values — the kernel failed to compile.

This is *not* a dynamic-shape issue; it reproduces with fully static dims.

The fix traces each return-tuple element back to the callee arg it aliases, so
the kernel now compiles and runs. All three outputs are independent row-wise
copies of ``x`` cast to FP32, so each must equal ``x.float()``.
"""

import pypto.language as pl
import pytest
import torch

B = 64
DIM = 512
TILE = 8


@pl.jit
def mixed_carry_phi_swap_kernel(
    x: pl.Tensor[[B, DIM], pl.BF16],
    out_a: pl.Out[pl.Tensor[[B, DIM], pl.FP32]],
    out_b: pl.Out[pl.Tensor[[B, DIM], pl.FP32]],
    out_c: pl.Out[pl.Tensor[[B, DIM], pl.FP32]],
):
    # Scope A: a pl.spmd body defines "global_c_idx" and writes out_a.
    for idx in pl.spmd(B, name_hint="scope_a"):
        global_c_idx = idx
        tile = pl.cast(x[global_c_idx : global_c_idx + 1, :], target_type=pl.FP32)
        out_a = pl.assemble(out_a, tile, [global_c_idx, 0])

    # Scope B: reuses "global_c_idx" + "tile" with two tensor carries, so the
    # parallel-loop carry becomes (scalar, out_b, out_c, tile) — the mixed
    # scalar/tensor carry that triggered the phi-naming swap.
    for batch_base_idx in pl.parallel(0, B // TILE):
        batch_base = batch_base_idx * TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scope_b"):
            for inner in pl.range(TILE):
                global_c_idx = batch_base + inner
                tile = pl.cast(x[global_c_idx : global_c_idx + 1, :], target_type=pl.FP32)
                out_b[global_c_idx : global_c_idx + 1, :] = tile
                out_c[global_c_idx : global_c_idx + 1, :] = tile
    return out_a, out_b, out_c


class TestMixedCarryPhiSwap:
    """Regression test for issue #1580 — mixed scalar/tensor parallel-loop carry."""

    def test_mixed_scalar_tensor_carry_runs(self, test_config):
        """The kernel compiles and all three outputs equal ``x`` cast to FP32."""
        mixed_carry_phi_swap_kernel._cache.clear()
        torch.manual_seed(0)

        x = torch.randn(B, DIM, dtype=torch.bfloat16)
        out_a = torch.zeros((B, DIM), dtype=torch.float32)
        out_b = torch.zeros((B, DIM), dtype=torch.float32)
        out_c = torch.zeros((B, DIM), dtype=torch.float32)

        mixed_carry_phi_swap_kernel(x, out_a, out_b, out_c, config=test_config)

        expected = x.to(torch.float32)
        for name, out in (("out_a", out_a), ("out_b", out_b), ("out_c", out_c)):
            assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3), (
                f"{name} mismatch: max diff = {(out - expected).abs().max().item()}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
