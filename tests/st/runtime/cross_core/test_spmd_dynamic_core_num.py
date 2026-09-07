# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression test (issue #1579): ``pl.spmd`` with a composite dynamic ``core_num``.

The issue: dispatching ``pl.spmd`` with a block count that is a composite
expression of a dynamic dim — a ``dyn * static`` multiply (e.g. ``b_dim * 3 // 4``)
or ``dyn // static`` (e.g. ``b_dim // 2``) — broke orchestration C++ codegen with
``'CORES_inline1' was not declared in this scope``.

Cause: scope outlining moves the ``core_num`` expression onto the dispatched
Spmd function as a function attr whose Vars are defined in the *caller*
Orchestration function. The final per-function ``Simplify`` scalar-DCE deleted
the defining scalar — it sees no in-function use, because the only consumer is
a sibling function's attr — so codegen referenced an undeclared variable. The
fix makes ``Simplify`` program-aware so those scalars are preserved.

``core_num`` is kept ``<= b_dim`` (and well under the physical core count) so
each spmd block ``idx`` writes its own in-bounds row ``out[idx]`` and the
kernels run end-to-end against a torch golden.
"""

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402
from harness import st  # noqa: E402

B_DYN = pl.dynamic("B_DYN")
B = 8  # concrete batch size used by the tests


@pl.jit
def spmd_core_num_mul(
    x: pl.Tensor[[B_DYN, 64, 64], pl.BF16],
    out: pl.Out[pl.Tensor[[B_DYN, 64, 64], pl.FP32]],
):
    """core_num = b_dim * 3 // 4 — composite containing a dyn*static multiply."""
    x.bind_dynamic(0, B_DYN)
    out.bind_dynamic(0, B_DYN)
    b_dim = pl.tensor.dim(x, 0)
    cores = b_dim * 3 // 4  # composite: (dynamic * static) // static
    for idx in pl.spmd(cores, name_hint="mul"):
        out[idx : idx + 1, :, :] = pl.cast(x[idx : idx + 1, :, :], target_type=pl.FP32)
    return out


@pl.jit
def spmd_core_num_floordiv(
    x: pl.Tensor[[B_DYN, 64, 64], pl.BF16],
    out: pl.Out[pl.Tensor[[B_DYN, 64, 64], pl.FP32]],
):
    """core_num = b_dim // 2 — composite dyn // static. Each block handles two rows."""
    x.bind_dynamic(0, B_DYN)
    out.bind_dynamic(0, B_DYN)
    b_dim = pl.tensor.dim(x, 0)
    cores = b_dim // 2  # composite: dynamic // static
    for g in pl.spmd(cores, name_hint="div"):
        r = g * 2
        out[r : r + 1, :, :] = pl.cast(x[r : r + 1, :, :], target_type=pl.FP32)
        out[r + 1 : r + 2, :, :] = pl.cast(x[r + 1 : r + 2, :, :], target_type=pl.FP32)
    return out


def _dyn_mul_case():
    """``pl.spmd(b_dim * 3 // 4)`` compiles and casts the dispatched rows.

    Only rows ``[0, cores)`` are dispatched and written; the tail is undefined by
    contract, so the golden marks it NaN — ``validate_golden`` treats a NaN
    golden element as don't-care, which is the same region the original
    ``out[:cores]`` slice compared.
    """
    torch.manual_seed(0)
    x = torch.randn(B, 64, 64, dtype=torch.bfloat16)
    out = torch.zeros(B, 64, 64, dtype=torch.float32)
    cores = B * 3 // 4

    def golden(_):
        expected = torch.full((B, 64, 64), float("nan"), dtype=torch.float32)
        expected[:cores] = x.float()[:cores]
        return expected

    return st.case(
        spmd_core_num_mul, x, out, name="spmd_core_num_dyn_mul", golden=golden, rtol=1e-3, atol=1e-3
    )


def _dyn_floordiv_case():
    """``pl.spmd(b_dim // 2)`` compiles and casts every element."""
    torch.manual_seed(0)
    x = torch.randn(B, 64, 64, dtype=torch.bfloat16)
    out = torch.zeros(B, 64, 64, dtype=torch.float32)
    return st.case(
        spmd_core_num_floordiv,
        x,
        out,
        name="spmd_core_num_dyn_floordiv",
        golden=lambda _: x.float(),
        rtol=1e-3,
        atol=1e-3,
    )


@st.cases(_dyn_mul_case(), _dyn_floordiv_case())
def test_spmd_dynamic_core_num(case_run):
    """``pl.spmd`` dispatched with a composite dynamic ``core_num`` (issue #1579)."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
