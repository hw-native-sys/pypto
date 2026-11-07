# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 runtime tests for the Tensor surface of MX matmul.

``pl.matmul_mx`` applied straight to GM tensors, with no ``pl.load``: the Tensor
dispatch has only the base form -- no Tensor-level ``_acc`` / ``_bias`` exist --
and all four A5 FP8 pairs run here.

Each case runs **both** surfaces in one kernel over byte-identical operands: the
Tensor dispatch and a ``@pl.jit.incore`` Tile reference, so a disagreement can
only mean the surfaces differ, not that the inputs did. Both outputs are
compared to the same exact FP64-derived golden on raw FP32 words rather than
with ``torch.isclose``, which holds for ``(-0.0, +0.0)`` even at
``rtol = atol = 0`` and so cannot see a sign-of-zero error. Comparing both
against one golden also compares them against each other, so no separate
cross-check is needed.

The in-kernel Tile reference is kept deliberately even though the golden alone
would catch a wrong Tensor result: on a red run it says *which* layer broke.
Tensor wrong while Tile is right points at the Tensor dispatch; both wrong
points at the shared MX stack underneath. Deleting it would save ~30 lines and
cost that distinction on exactly the runs where it is most expensive to
re-derive.

Operands come from ``_matmul_mx_data._make_inputs`` -- the same generator the
Tile-surface cases in ``test_matmul_mx.py`` use -- windowed to the exact extents
the Tensor op takes. The exactness budget and the per-K-group extremes scheme
are documented there.

This file is a separate module from ``test_matmul_mx.py`` because CI runs it in
its own ``onboard-tests-a5`` step; see that step's comment for why the sessions
are split.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec
from runtime.ops._matmul_mx_data import (
    _MX_FP8_PAIR_VALUES,
    _PL_DTYPES,
    A_OFF,
    A_SCALE_OFF,
    B_OFF,
    B_SCALE_OFF,
    K_GROUPS,
    REQUIRED_TORCH_DTYPES,
    K,
    M,
    N,
    _make_inputs,
    _matmul_mx_golden,
    _pack_a_scale,
    _pack_b_scale,
    _unpack_a_scale,
    _unpack_b_scale,
)

if not all(hasattr(torch, name) for name in REQUIRED_TORCH_DTYPES):
    pytest.skip("torch MXFP4/MXFP8/E8M0 dtypes required", allow_module_level=True)


# ---------------------------------------------------------------------------
# Tensor surface: pl.matmul_mx straight on GM tensors
# ---------------------------------------------------------------------------
# The Tensor dispatch has only the base form -- no Tensor-level `_acc` / `_bias`
# exist -- and it takes whole tensors rather than windows, so its operands are the
# exact [M, K] / [K, N] window the Tile case reads through its offsets. Pulling
# them out of `_make_inputs` makes "same inputs" literal: the two surfaces see
# byte-identical data, scales and dtypes, so comparing both against the one exact
# golden also compares them against each other.


def _window_operands(
    lhs_dtype: DataType, rhs_dtype: DataType
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The exact-size operands the Tile case reads, re-packed for the Tensor case.

    Scales are unpacked to logical codes, windowed, then re-packed for the smaller
    tensor: the MX physical layout depends on the tensor's own extents, so slicing
    the packed bytes would not give the window's layout.
    """
    a, a_scale, b, b_scale, _ = _make_inputs(lhs_dtype, rhs_dtype)
    a_window = a[A_OFF[0] : A_OFF[0] + M, A_OFF[1] : A_OFF[1] + K].contiguous()
    b_window = b[B_OFF[0] : B_OFF[0] + K, B_OFF[1] : B_OFF[1] + N].contiguous()
    a_codes = _unpack_a_scale(a_scale.view(torch.uint8))[
        A_SCALE_OFF[0] : A_SCALE_OFF[0] + M, A_SCALE_OFF[1] : A_SCALE_OFF[1] + K_GROUPS
    ].contiguous()
    b_codes = _unpack_b_scale(b_scale.view(torch.uint8))[
        B_SCALE_OFF[0] : B_SCALE_OFF[0] + K_GROUPS, B_SCALE_OFF[1] : B_SCALE_OFF[1] + N
    ].contiguous()
    return (
        a_window,
        _pack_a_scale(a_codes).view(torch.float8_e8m0fnu),
        b_window,
        _pack_b_scale(b_codes).view(torch.float8_e8m0fnu),
    )


def _tensor_vs_tile_entry(lhs_dtype: DataType, rhs_dtype: DataType):
    """One entry running both surfaces over the same GM operands, in one kernel.

    The Tile reference lives in the same program on purpose: both outputs come
    from one device run on one set of bytes, so ``out_tensor != out_tile`` can
    only mean the surfaces disagree.
    """
    lhs_pl = _PL_DTYPES[lhs_dtype]
    rhs_pl = _PL_DTYPES[rhs_dtype]

    @pl.jit.incore
    def mx_tile_reference(
        a: pl.Tensor[[M, K], lhs_pl],
        a_scale: pl.Tensor[[M, K_GROUPS], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[K, N], rhs_pl],
        b_scale: pl.Tensor[[K_GROUPS, N], pl.FP8E8M0, pl.MX_B_NN],
        out_tile: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        lhs = pl.load(a, [0, 0], [M, K])
        lhs_scale = pl.load(a_scale, [0, 0], [M, K_GROUPS])
        rhs = pl.load(b, [0, 0], [K, N])
        rhs_scale = pl.load(b_scale, [0, 0], [K_GROUPS, N])
        return pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out_tile)

    @pl.jit
    def mx_tensor_matmul(
        a: pl.Tensor[[M, K], lhs_pl],
        a_scale: pl.Tensor[[M, K_GROUPS], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[K, N], rhs_pl],
        b_scale: pl.Tensor[[K_GROUPS, N], pl.FP8E8M0, pl.MX_B_NN],
        out_tensor: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_tile: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            out_tensor = pl.assemble(out_tensor, pl.matmul_mx(a, a_scale, b, b_scale), [0, 0])
        out_tile = mx_tile_reference(a, a_scale, b, b_scale, out_tile)
        return out_tensor, out_tile

    return mx_tensor_matmul


def _tensor_case_tensors(lhs_dtype: DataType, rhs_dtype: DataType) -> list[TensorSpec]:
    a, a_scale, b, b_scale = _window_operands(lhs_dtype, rhs_dtype)
    # Sentinel-filled outputs: an op that writes nothing cannot pass by accident.
    sentinel = torch.full((M, N), float("nan"), dtype=torch.float32)
    return [
        TensorSpec("a", [M, K], lhs_dtype, init_value=a),
        TensorSpec("a_scale", [M, K_GROUPS], DataType.FP8E8M0, init_value=a_scale),
        TensorSpec("b", [K, N], rhs_dtype, init_value=b),
        TensorSpec("b_scale", [K_GROUPS, N], DataType.FP8E8M0, init_value=b_scale),
        TensorSpec("out_tensor", [M, N], DataType.FP32, is_output=True, init_value=sentinel),
        TensorSpec("out_tile", [M, N], DataType.FP32, is_output=True, init_value=sentinel.clone()),
    ]


def _tensor_case_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    base = _matmul_mx_golden(
        tensors["a"],
        _unpack_a_scale(tensors["a_scale"].view(torch.uint8)),
        tensors["b"],
        _unpack_b_scale(tensors["b_scale"].view(torch.uint8)),
        k_valid=K,
    )
    return {"out_tensor": base, "out_tile": base.clone()}


def _assert_bitwise_equal(actual, expected) -> None:
    """Compare both surfaces' FP32 bytes against the one shared exact golden.

    The default elementwise check is ``torch.isclose``, and ``isclose(-0.0, +0.0)``
    holds even at ``rtol = atol = 0`` -- so a sign-of-zero difference would pass a
    tolerance-based claim of exactness. Comparing the raw bytes does not.

    Both outputs are checked against the *same* golden values (``base`` and
    ``base.clone()``), so "the two surfaces agree with each other" is implied by
    this loop rather than needing its own assert: anything that makes them
    disagree makes at least one of them disagree with the golden first. An
    explicit cross-check used to follow, and was removed because it could not
    fail -- an exhaustive single-word perturbation of both outputs fired this
    loop every time and the cross-check never.
    """
    assert actual.keys() == expected.keys(), f"{sorted(actual)} != {sorted(expected)}"
    for name in sorted(actual):
        got = actual[name].contiguous().view(torch.int32)
        ref = expected[name].contiguous().view(torch.int32)
        differing = int((got != ref).sum())
        assert differing == 0, f"{name} differs from the golden in {differing} of {got.numel()} FP32 words"


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _tensor_vs_tile_entry(lhs_dtype, rhs_dtype),
            name=f"tensor_matmul_mx_{lhs_dtype.value}_x_{rhs_dtype.value}_vs_tile",
            tensors=_tensor_case_tensors(lhs_dtype, rhs_dtype),
            golden=_tensor_case_golden,
            compare=_assert_bitwise_equal,
            platform="a5",
        )
        for lhs_dtype, rhs_dtype, _pair_id in _MX_FP8_PAIR_VALUES
    ]
)
def test_tensor_matmul_mx_matches_the_tile_path(case_run):
    """Tensor ``pl.matmul_mx`` over all four FP8 pairs, bit-exact and tile-identical."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
