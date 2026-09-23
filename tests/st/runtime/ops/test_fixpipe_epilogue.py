# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime st for the FIXPIPE epilogue (``pre_quant`` / ``pre_relu``).

The cube's fix-pipe can multiply an accumulator by an FP32 scale and apply ReLU
while it drains L0C. Every other layer of coverage for this checks a *description*
of that: the unit tests assert emitted ``.pto`` text, ptoas assembles it, and the
dtype tables are transcribed from pto-isa. None of them executes the instruction.
These cases do, and they pin the three claims that cannot be checked any other
way:

1. **The scale reaches the hardware with the value the author wrote.**
   ``EncodeFixpipePreQuant`` packs it as the FP32 bit pattern in the low 32 bits
   of a configuration register. A wrong layout still assembles and still runs --
   it just multiplies by something else.

2. **The ReLU runs on the accumulator, ahead of the scale** -- the pair is
   ``maximum(tile, 0) * scale``, not ``maximum(tile * scale, 0)``. pto-isa calls
   it ``ReluPreMode`` for that reason. The two are the same function for every
   ``scale > 0``, so only ``acc_to_gm_negative_scale`` separates them: with
   ``s < 0`` they disagree at **every** element (one keeps the originally
   *positive* entries and negates them, the other keeps the negative ones), and
   that case is what measured the order in the first place -- the first version
   of this file asserted the opposite and the device said no.

3. **The quantizing writeback clamps to the destination range** rather than
   overflowing to infinity, which is where it parts company with an ordinary
   ``pl.cast`` to FP16. The clamp is last, after the multiply.

Coverage includes the scaled Acc->Mat (``pto.tinsert``) score-reduction chain
enabled by PTOAS 0.65, where an INT32 accumulator becomes an FP16 L1 tile and
feeds a second Cube matmul. This is the form previously blocked by PTOAS#1570.

Golden: torch. The Acc-to-GM cases remain **A2/A3 only** because the wider A5
table has no device evidence. The new Acc-to-Mat case is deliberately narrower:
it covers only the PTOAS 0.65-validated ``INT32 -> FP16`` pair and runs on both
A2/A3 and A5.
"""

import pypto.language as pl
import pytest
import torch
from harness import st

M = 128
N = 128
K = 64
H = 32

# 2**-10 -- exactly representable, and the constant CANN writes with
# `SetFixpipePreQuantFlag` for this kernel family (FP32 word 0x3A800000).
SCALE = 1.0 / 1024
# Deliberately not a power of two, so a truncated or re-rounded scale shows up.
SCALE_INEXACT = 0.003
# The order discriminator; see the module docstring.
SCALE_NEGATIVE = -1.0 / 512

FP16_MAX = 65504.0


# ---------------------------------------------------------------------------
# Kernels -- Acc -> GM (pto.tstore)
# ---------------------------------------------------------------------------


@pl.jit
def score_dequant_relu_to_gm(k: pl.Tensor, q: pl.Tensor, out: pl.Out[pl.Tensor]):
    """The same epilogue on the direct-to-GM writeback."""
    with pl.at(level=pl.Level.CORE_GROUP):
        k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(k_mat, target_memory=pl.Mem.Left),
            pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
        )
        pl.tile.store(acc, [0, 0], out, pre_quant=SCALE, pre_relu=True)
    return out


@pl.jit
def score_negative_scale_to_gm(k: pl.Tensor, q: pl.Tensor, out: pl.Out[pl.Tensor]):
    """A **negative** scale: the case that pins ReLU-before-scale.

    ``relu(x) * s`` keeps the entries that were positive before scaling and
    negates them; ``relu(x * s)`` keeps the negative ones and leaves them
    positive. The two disagree at every element, so this cannot pass under a
    swapped order -- and it is what established the real one.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(k_mat, target_memory=pl.Mem.Left),
            pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
        )
        pl.tile.store(acc, [0, 0], out, pre_quant=SCALE_NEGATIVE, pre_relu=True)
    return out


@pl.jit
def score_inexact_scale_no_relu_to_gm(k: pl.Tensor, q: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``pre_quant`` with no activation, and a scale that is not a power of two.

    Isolates the scale from the ReLU, and makes the packed FP32 word carry a
    full mantissa -- a truncated or re-rounded encoding shows up here, where
    ``SCALE``'s ``0x3A800000`` would survive several wrong layouts unharmed.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(k_mat, target_memory=pl.Mem.Left),
            pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
        )
        pl.tile.store(acc, [0, 0], out, pre_quant=SCALE_INEXACT)
    return out


@pl.jit
def score_dequant_saturates_to_gm(k: pl.Tensor, q: pl.Tensor, out: pl.Out[pl.Tensor]):
    """A scale large enough that the dequantized value leaves the FP16 range.

    ``DEQF16`` clamps to +/-FP16_MAX; an ordinary ``pl.cast`` to FP16 would
    produce an infinity here instead.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(k_mat, target_memory=pl.Mem.Left),
            pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
        )
        pl.tile.store(acc, [0, 0], out, pre_quant=4096.0, pre_relu=True)
    return out


# ---------------------------------------------------------------------------
# Kernels -- Acc -> Mat
# ---------------------------------------------------------------------------


@pl.jit
def score_dequant_relu_to_mat(k: pl.Tensor, q: pl.Tensor, w: pl.Tensor, out: pl.Out[pl.Tensor]):
    """Scaled/ReLU INT32 L0C -> FP16 L1, followed by an FP16 matmul."""
    with pl.at(level=pl.Level.CORE_GROUP):
        k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(k_mat, target_memory=pl.Mem.Left),
            pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
        )
        score_mat = pl.tile.create([M, N], pl.FP16, target_memory=pl.Mem.Mat)
        score_mat = pl.tile.assemble(score_mat, acc, [0, 0], pre_quant=SCALE, pre_relu=True)
        w_mat = pl.tile.load(w, [0, 0], [N, H], target_memory=pl.Mem.Mat)
        y = pl.tile.matmul(
            pl.tile.move(score_mat, target_memory=pl.Mem.Left),
            pl.tile.move(w_mat, target_memory=pl.Mem.Right),
        )
        pl.tile.store(y, [0, 0], out)
    return out


@pl.jit
def matmul_relu_to_mat(a: pl.Tensor, b: pl.Tensor, e: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``relu(a @ b) @ e`` with the ReLU on the cube and no scale at all.

    ``pre_relu`` without ``pre_quant`` attaches to the ordinary FP32 -> BF16
    writeback, a different instruction form from the quantizing one above.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        a_mat = pl.tile.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        b_mat = pl.tile.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(
            pl.tile.move(a_mat, target_memory=pl.Mem.Left),
            pl.tile.move(b_mat, target_memory=pl.Mem.Right),
        )
        s_mat = pl.tile.create([M, N], pl.BF16, target_memory=pl.Mem.Mat)
        s_mat = pl.tile.assemble(s_mat, acc, [0, 0], pre_relu=True)
        e_mat = pl.tile.load(e, [0, 0], [N, H], target_memory=pl.Mem.Mat)
        y = pl.tile.matmul(
            pl.tile.move(s_mat, target_memory=pl.Mem.Left),
            pl.tile.move(e_mat, target_memory=pl.Mem.Right),
        )
        pl.store(y, [0, 0], out)
    return out


# ---------------------------------------------------------------------------
# Goldens
# ---------------------------------------------------------------------------


def _fixpipe_dequant(acc_i32: torch.Tensor, scale: float, *, relu: bool) -> torch.Tensor:
    """The fix-pipe's own order: activate the accumulator, scale, then clamp.

    The ReLU is a *pre*-quant stage -- pto-isa spells it ``ReluPreMode`` -- so it
    sees the raw INT32 accumulator and the multiply happens afterwards. Written
    in this order on purpose: ``relu(x * scale)`` is the same function for every
    ``scale > 0``, so only ``acc_to_gm_negative_scale`` tells them apart, and it
    is what measured this.
    """
    activated = torch.clamp(acc_i32.float(), min=0.0) if relu else acc_i32.float()
    return torch.clamp(activated * scale, -FP16_MAX, FP16_MAX).to(torch.float16)


def _int_score(k: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``k @ q.T`` in INT32 -- exact, so it contributes no tolerance."""
    return k.int() @ q.int().t()


def _gm_case(kernel, name, scale, *, relu, low=-4, high=5, seed=0, **kwargs):
    """Acc->GM: the dequantized score lands straight in an FP16 tensor."""
    torch.manual_seed(seed)
    k = torch.randint(low, high, (M, K), dtype=torch.int8)
    q = torch.randint(low, high, (N, K), dtype=torch.int8)
    out = torch.zeros((M, N), dtype=torch.float16)
    return st.case(
        kernel,
        k,
        q,
        out,
        name=name,
        golden=lambda _: _fixpipe_dequant(_int_score(k, q), scale, relu=relu),
        **kwargs,
    )


def _relu_only_case(kernel, name, **kwargs):
    """``relu(a @ b) @ e`` with a BF16 on-chip intermediate."""
    torch.manual_seed(0)
    # BF16 throughout: the scratch the ReLU writes is BF16 (the unscaled
    # writeback's only narrowing target here), and tile.matmul requires both
    # operands to share a dtype, so `e` must match it.
    a = torch.randn(M, K, dtype=torch.bfloat16)
    b = torch.randn(K, N, dtype=torch.bfloat16)
    e = torch.randn(N, H, dtype=torch.bfloat16)
    out = torch.zeros((M, H), dtype=torch.float32)

    def golden(_):
        activated = torch.clamp(a.float() @ b.float(), min=0.0)
        return activated.to(torch.bfloat16).float() @ e.float()

    return st.case(kernel, a, b, e, out, name=name, golden=golden, **kwargs)


def _scaled_mat_case(kernel, name, **kwargs):
    """``dequant_relu(k @ q.T) @ w`` with an FP16 on-chip intermediate."""
    torch.manual_seed(0)
    k = torch.randint(-4, 5, (M, K), dtype=torch.int8)
    q = torch.randint(-4, 5, (N, K), dtype=torch.int8)
    w = torch.randn(N, H, dtype=torch.float16)
    out = torch.zeros((M, H), dtype=torch.float32)

    def golden(_):
        score = _fixpipe_dequant(_int_score(k, q), SCALE, relu=True)
        return score.float() @ w.float()

    return st.case(kernel, k, q, w, out, name=name, golden=golden, **kwargs)


# A direct FP16 store rounds once and stops, so it is tight.
_STORE_TOL = {"rtol": 1e-3, "atol": 1e-3}


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    _gm_case(score_dequant_relu_to_gm, "acc_to_gm_dequant_relu", SCALE, relu=True, **_STORE_TOL),
    # A scale with a full mantissa, and no ReLU, so the multiply is on its own.
    _gm_case(
        score_inexact_scale_no_relu_to_gm,
        "acc_to_gm_inexact_scale",
        SCALE_INEXACT,
        relu=False,
        **_STORE_TOL,
    ),
    # The order discriminator. `relu(x)*s` and `relu(x*s)` differ at every
    # element for s < 0, so passing this is direct evidence for the documented
    # order rather than agreement by construction.
    _gm_case(
        score_negative_scale_to_gm,
        "acc_to_gm_negative_scale",
        SCALE_NEGATIVE,
        relu=True,
        **_STORE_TOL,
    ),
    # Wide inputs * 4096 leave the FP16 range, so this asserts the DEQF16 clamp
    # rather than the infinity a plain cast would give.
    _gm_case(
        score_dequant_saturates_to_gm,
        "acc_to_gm_saturates",
        4096.0,
        relu=True,
        low=-100,
        high=101,
        **_STORE_TOL,
    ),
)
def test_acc_to_gm_epilogue(case_run):
    """``pto.tstore`` with a scale and ReLU, straight into an FP16 tensor."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(_relu_only_case(matmul_relu_to_mat, "acc_to_mat_relu_only", rtol=2e-2, atol=2e-2))
def test_acc_to_mat_relu_without_scale(case_run):
    """``pre_relu`` alone rides the unscaled FP32 -> BF16 narrowing."""
    case_run.assert_passed()


@pytest.mark.platforms(
    "a2a3",
    "a5",
    reason="Scaled INT32 Acc-to-FP16 Mat TINSERT is enabled on PTOAS 0.65 for both architectures.",
)
@st.cases(
    _scaled_mat_case(
        score_dequant_relu_to_mat,
        "acc_to_mat_dequant_relu_then_matmul",
        rtol=2e-2,
        atol=2e-2,
    )
)
def test_acc_to_mat_scaled_relu(case_run):
    """PTOAS 0.65 preserves the scale on the A2/A3 and A5 TINSERT paths."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
