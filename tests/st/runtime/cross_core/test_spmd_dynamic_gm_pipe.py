# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression test (issue #1768): single-scope cube→vec fusion with a dynamic dim.

A cube matmul feeding a vector epilogue, fused into ONE ``pl.spmd`` scope with a
dynamic block count, makes ``InjectGMPipeBuffer`` (Ascend910B) inject a GM pipe
buffer whose size is ``slot_size * (m // ROW_TILE)`` — a function of the dynamic
token dim ``m``. Orchestration codegen used to hoist that buffer's
``alloc_tensors`` to the top of ``PTO2_SCOPE()``, *above* the body-local
``int64_t m = orch_args.tensor(0).shapes[0];`` that sizes it, so the generated
host C++ failed to compile with ``'m...' was not declared in this scope``.

Only this exact combination triggers it:
  * single-scope cube→vec fusion  ⇒ a GM pipe buffer is injected, and
  * a dynamic block count         ⇒ the pipe size references a body-local.
A static dim (constant ``core_num``) or a multi-scope layout (matmul in its own
spmd scope ⇒ no GM pipe buffer) both compile fine and are not regressions here.

``core_num = m // ROW_TILE`` is kept well under the physical core count so each
block writes its own in-bounds row-tile and the kernel runs end-to-end against a
torch golden.
"""

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402

M_DYN = pl.dynamic("M_DYN")
K = 64
N = 16
ROW_TILE = 16
M = 64  # concrete token dim: M // ROW_TILE = 4 fused cube→vec blocks


@pl.jit
def fused_matmul_epilogue_dyn(
    a: pl.Tensor[[M_DYN, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP32],
    out: pl.Out[pl.Tensor[[M_DYN, N], pl.FP32]],
):
    """``out[i] = (a[i] + 1) @ b + 1`` per row-tile, fused in one spmd scope.

    A vector op produces the matmul operand (V->C) and a vector op consumes the
    matmul result (C->V), so the single no-split scope runs under dual-AIV
    dispatch and gets an injected GM pipe buffer sized by the dynamic block
    count ``m // ROW_TILE``.
    """
    a.bind_dynamic(0, M_DYN)
    out.bind_dynamic(0, M_DYN)
    m = pl.tensor.dim(a, 0)
    for ob in pl.spmd(m // ROW_TILE, name_hint="hc"):
        m0 = ob * ROW_TILE
        a_slice = pl.slice(a, [ROW_TILE, K], [m0, 0])
        a_add = pl.add(a_slice, 1.0)  # vector produces matmul operand (V->C)
        c_tile = pl.matmul(a_add, b)  # cube
        c_vec = pl.add(c_tile, 1.0)  # vector consumes matmul result (C->V)
        out = pl.assemble(out, c_vec, [m0, 0])
    return out


class TestSpmdDynamicGMPipe:
    """Single-scope cube→vec fusion dispatched over a dynamic dim (issue #1768)."""

    def test_dynamic_token_dim_compiles_and_runs(self, test_config):
        """``pl.spmd(m // ROW_TILE)`` over a fused matmul+epilogue compiles and runs."""
        fused_matmul_epilogue_dyn._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        out = torch.zeros(M, N, dtype=torch.float32)

        fused_matmul_epilogue_dyn(a, b, out, config=test_config)

        expected = torch.empty(M, N, dtype=torch.float32)
        for m0 in range(0, M, ROW_TILE):
            expected[m0 : m0 + ROW_TILE] = torch.matmul(a[m0 : m0 + ROW_TILE] + 1.0, b) + 1.0
        assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3), (
            f"dynamic-dim cube→vec fusion: max diff = {(out - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
