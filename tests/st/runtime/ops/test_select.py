# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Hardware tests for the composite ``pl.tile.select``.

``tests/st/runtime/ops/test_sels.py`` covers the 1:1 ``pto.tsels`` interface.
This file covers the composite op on top of it, whose whole point is that the
caller supplies neither the packed-mask geometry nor the scratch tile -- so what
needs executing here is that each operand form still selects the right elements
after the lowering has chosen a PTO form on the author's behalf:

- ``mask ? tile : scalar`` becomes one ``pto.tsels``;
- ``mask ? scalar : tile`` has no TSELS form, so the scalar is materialized with
  ``tile.full`` and the selection becomes ``pto.tsel``;
- A2/A3 has no 8-bit TSELS and takes the same fallback, while A5 does not.

Those three routes produce different instruction sequences from one source line,
and only execution shows they agree numerically.
"""

import pypto.language as pl
import pytest
import torch
from harness import st

M, N = 16, 128
ROW_TAIL = (11, N)
COL_TAIL = (M, 100)
COMBINED_TAIL = (11, 100)

# cmp_type encoding shared with pl.tile.cmps: 0=eq 1=ne 2=lt 3=le 4=gt 5=ge.
_CMP = {0: torch.eq, 1: torch.ne, 2: torch.lt, 3: torch.le, 4: torch.gt, 5: torch.ge}
_CMP_IDS = ("eq", "ne", "lt", "le", "gt", "ge")

_THRESHOLD = 0.0
_FALSE_SCALAR = -3.0
_TRUE_SCALAR = 7.0


def _literals(torch_dtype):
    """Threshold and branch scalars already in the tile's own dtype.

    ``pl.tile.cmps`` does not coerce its scalar to the tile's element type, so a
    float threshold against an integer tile reaches ``pto.tcmps`` as an f32
    operand and compares wrong. Pass integers for integer tiles.
    """
    if torch_dtype.is_floating_point:
        return _THRESHOLD, _FALSE_SCALAR, _TRUE_SCALAR
    return int(_THRESHOLD), int(_FALSE_SCALAR), int(_TRUE_SCALAR)


def _source(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    """Values straddling the threshold so no comparison is degenerate."""
    values = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols).remainder(9) - 4
    return values.to(dtype)


# ---------------------------------------------------------------------------
# Kernels. Each is one `pl.tile.select` over a `pl.tile.cmps` mask; they differ
# only in which branch is the scalar, which is what picks the PTO form.
# ---------------------------------------------------------------------------


def _scalar_on_false(cmp_type: int, dtype, rows: int, cols: int, valid, threshold, scalar):
    @pl.jit
    def kernel(v: pl.Tensor[[rows, cols], dtype], out: pl.Out[pl.Tensor[[rows, cols], dtype]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            t = pl.load(v, [0, 0], [rows, cols], valid_shape=valid)
            pl.store(pl.tile.select(pl.tile.cmps(t, threshold, cmp_type=cmp_type), t, scalar), [0, 0], out)
        return out

    return kernel


def _scalar_on_true(cmp_type: int, dtype, rows: int, cols: int, valid, threshold, scalar):
    @pl.jit
    def kernel(v: pl.Tensor[[rows, cols], dtype], out: pl.Out[pl.Tensor[[rows, cols], dtype]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            t = pl.load(v, [0, 0], [rows, cols], valid_shape=valid)
            pl.store(pl.tile.select(pl.tile.cmps(t, threshold, cmp_type=cmp_type), scalar, t), [0, 0], out)
        return out

    return kernel


def _two_tiles(cmp_type: int, dtype, rows: int, cols: int, valid):
    @pl.jit
    def kernel(
        a: pl.Tensor[[rows, cols], dtype],
        b: pl.Tensor[[rows, cols], dtype],
        out: pl.Out[pl.Tensor[[rows, cols], dtype]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            ta = pl.load(a, [0, 0], [rows, cols], valid_shape=valid)
            tb = pl.load(b, [0, 0], [rows, cols], valid_shape=valid)
            pl.store(pl.tile.select(pl.tile.cmp(ta, tb, cmp_type=cmp_type), ta, tb), [0, 0], out)
        return out

    return kernel


def _split_dtype(cmp_type: int, cmp_dtype, val_dtype, threshold, scalar, rows: int, cols: int):
    """Compare in one dtype, select in another.

    On A2/A3 ``pto.tcmps`` takes only i16/i32/f16/f32, so a BF16 or 8-bit tile
    cannot produce a mask at all -- a `tile.cmps` limitation, not a select one.
    The packed-mask geometry depends on the result's column count and not on the
    compared dtype, so comparing a supported dtype still yields the mask this
    result needs, and the narrow dtype is exercised where it matters: through
    the selection itself.
    """

    @pl.jit
    def kernel(
        c: pl.Tensor[[rows, cols], cmp_dtype],
        v: pl.Tensor[[rows, cols], val_dtype],
        out: pl.Out[pl.Tensor[[rows, cols], val_dtype]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            tc = pl.load(c, [0, 0], [rows, cols])
            tv = pl.load(v, [0, 0], [rows, cols])
            pl.store(pl.tile.select(pl.tile.cmps(tc, threshold, cmp_type=cmp_type), tv, scalar), [0, 0], out)
        return out

    return kernel


@pl.jit
def _clamp(v: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
    """Composed selects: clamp(v, -2, 2), the shape issue #2742 asks for."""
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(v, [0, 0], [M, N])
        hi = pl.tile.select(pl.tile.cmps(t, 2.0, cmp_type=3), t, 2.0)  # t <= 2 ? t : 2
        pl.store(pl.tile.select(pl.tile.cmps(t, -2.0, cmp_type=5), hi, -2.0), [0, 0], out)  # t >= -2
    return out


# ---------------------------------------------------------------------------
# Goldens. A store of a tile narrowed by valid_shape writes only that region, so
# the expectation outside it is the destination's initial value (zero).
# ---------------------------------------------------------------------------


def _golden_scalar(kind: str, cmp_type: int, valid, torch_dtype):
    threshold, false_scalar, true_scalar = _literals(torch_dtype)

    def golden(tensors):
        src = tensors["v"]
        vr, vc = valid or src.shape
        out = torch.zeros_like(tensors["out"])
        window = src[:vr, :vc]
        cond = _CMP[cmp_type](window, torch.as_tensor(threshold, dtype=window.dtype))
        other = torch.as_tensor(false_scalar if kind == "false" else true_scalar, dtype=window.dtype)
        out[:vr, :vc] = (
            torch.where(cond, window, other) if kind == "false" else torch.where(cond, other, window)
        )
        return out

    return golden


def _golden_two_tiles(cmp_type: int, valid):
    def golden(tensors):
        a, b = tensors["a"], tensors["b"]
        vr, vc = valid or a.shape
        out = torch.zeros_like(tensors["out"])
        wa, wb = a[:vr, :vc], b[:vr, :vc]
        out[:vr, :vc] = torch.where(_CMP[cmp_type](wa, wb), wa, wb)
        return out

    return golden


def _compare_valid_only(valid):
    """Compare only inside ``valid_shape``.

    Elements outside a tile's valid region are undefined in PyPTO -- the load
    leaves them as padding and the store's behaviour there is not part of the
    contract -- so asserting on them would be testing unspecified behaviour.
    A pure ``pl.Out`` destination is not zeroed either, so the surrounding bytes
    are whatever the allocator last left.
    """
    vr, vc = valid

    def compare(actual, expected):
        for name, got in actual.items():
            want = expected[name]
            torch.testing.assert_close(
                got[:vr, :vc], want[:vr, :vc], rtol=1e-5, atol=1e-5, msg=lambda m: f"{name}: {m}"
            )

    return compare


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _scalar_case(kind, cmp_type, dtype, torch_dtype, *, valid=None, cols=N, name=None, platform=None):
    rows = M
    build = _scalar_on_false if kind == "false" else _scalar_on_true
    threshold, false_scalar, true_scalar = _literals(torch_dtype)
    v = _source(rows, cols, torch_dtype)
    out = torch.zeros(rows, cols, dtype=torch_dtype)
    return st.case(
        build(
            cmp_type,
            dtype,
            rows,
            cols,
            list(valid) if valid else None,
            threshold,
            false_scalar if kind == "false" else true_scalar,
        ),
        v,
        out,
        name=name or f"select_scalar_on_{kind}_{_CMP_IDS[cmp_type]}",
        golden=_golden_scalar(kind, cmp_type, valid, torch_dtype),
        compare=_compare_valid_only(valid) if valid else None,
        platform=platform,
    )


def _split_dtype_case(cmp_dtype, cmp_torch, val_dtype, val_torch, scalar, *, name, platform=None):
    cmp_src = _source(M, N, cmp_torch)
    v = _source(M, N, val_torch)
    out = torch.zeros(M, N, dtype=val_torch)
    threshold = 0 if not cmp_torch.is_floating_point else 0.0

    def golden(tensors):
        cond = tensors["c"] > torch.as_tensor(threshold, dtype=cmp_torch)
        return torch.where(cond, tensors["v"], torch.as_tensor(scalar, dtype=val_torch))

    return st.case(
        _split_dtype(4, cmp_dtype, val_dtype, threshold, scalar, M, N),
        cmp_src,
        v,
        out,
        name=name,
        golden=golden,
        platform=platform,
    )


def _two_tile_case(cmp_type, dtype, torch_dtype, *, valid=None, name=None):
    a = _source(M, N, torch_dtype)
    b = torch.zeros(M, N, dtype=torch_dtype)
    out = torch.zeros(M, N, dtype=torch_dtype)
    return st.case(
        _two_tiles(cmp_type, dtype, M, N, list(valid) if valid else None),
        a,
        b,
        out,
        name=name or f"select_two_tiles_{_CMP_IDS[cmp_type]}",
        golden=_golden_two_tiles(cmp_type, valid),
        compare=_compare_valid_only(valid) if valid else None,
    )


# Every comparison mode, on the TSELS route and on the tile/tile TSEL route.
@st.cases(*[_scalar_case("false", c, pl.FP32, torch.float32) for c in range(6)])
def test_comparison_modes_scalar_on_false(case_run):
    """mask ? tile : scalar -- one pto.tsels, all six comparisons."""
    case_run.assert_passed()


@st.cases(*[_two_tile_case(c, pl.FP32, torch.float32) for c in range(6)])
def test_comparison_modes_two_tiles(case_run):
    """mask ? tile : tile -- one pto.tsel, all six comparisons."""
    case_run.assert_passed()


# The scalar-on-true route is the one that materializes a tile and uses TSEL.
@st.cases(*[_scalar_case("true", c, pl.FP32, torch.float32) for c in range(6)])
def test_comparison_modes_scalar_on_true(case_run):
    """mask ? scalar : tile -- tile.full + pto.tsel, all six comparisons."""
    case_run.assert_passed()


# Degenerate masks: every lane true, then every lane false. `_source` straddles
# 0, so comparing against a bound outside its range forces each.
@st.cases(
    st.case(
        _scalar_on_false(
            5, pl.FP32, M, N, None, _THRESHOLD, _FALSE_SCALAR
        ),  # v >= 0 is false for the negative half
        _source(M, N, torch.float32) + 100.0,
        torch.zeros(M, N, dtype=torch.float32),
        name="select_all_true",
        golden=lambda t: torch.where(
            t["v"] >= 0.0, t["v"], torch.as_tensor(_FALSE_SCALAR, dtype=torch.float32)
        ),
    ),
    st.case(
        _scalar_on_false(5, pl.FP32, M, N, None, _THRESHOLD, _FALSE_SCALAR),
        _source(M, N, torch.float32) - 100.0,
        torch.zeros(M, N, dtype=torch.float32),
        name="select_all_false",
        golden=lambda t: torch.where(
            t["v"] >= 0.0, t["v"], torch.as_tensor(_FALSE_SCALAR, dtype=torch.float32)
        ),
    ),
)
def test_degenerate_masks(case_run):
    """An all-true and an all-false mask, which exercise no lane-mixing at all."""
    case_run.assert_passed()


# Element widths. BF16 has no TSELS form on either arch, so it also proves the
# dtype fallback numerically -- it had no coverage anywhere before this file.
@st.cases(
    _scalar_case("false", 4, pl.FP32, torch.float32, name="select_dtype_fp32"),
    _scalar_case("false", 4, pl.FP16, torch.float16, name="select_dtype_fp16"),
    _scalar_case("false", 4, pl.INT16, torch.int16, name="select_dtype_int16"),
)
def test_element_widths(case_run):
    """Each supported element width through the composite op."""
    case_run.assert_passed()


@pytest.mark.xfail(
    strict=False,
    reason=(
        "A2/A3: a tcmps mask built from an INT32 source drives tsels inverted -- every "
        "lane selects the branch the comparison did not choose (1136/2048 elements, "
        "`v > 0` keeping the scalar where it should keep v and vice versa). INT16 and "
        "FP32 pass through the identical code path with identical literals, so this is "
        "specific to 32-bit integers rather than to element width or to the composite. "
        "No existing ST covers it: tests/st/runtime/ops/test_sels.py builds its mask with "
        "tile.cmps only for FP16/FP32, and reaches INT32 solely through an explicitly "
        "loaded mask carrier, which bypasses tcmps entirely."
    ),
)
@st.cases(_scalar_case("false", 4, pl.INT32, torch.int32, name="select_dtype_int32"))
def test_int32_compare_feeding_a_select(case_run):
    """An INT32 source compared with tile.cmps and then selected on."""
    case_run.assert_passed()


# 8-bit integers split the two architectures: A2/A3 has no 8-bit TSELS and must
# fall back to tile.full + TSEL, A5 stays on TSELS. Same source line either way.
# BF16 cannot be *compared* on A2/A3 either (`pto.tcmps` takes only
# i16/i32/f16/f32), so this compares a supported dtype and selects the narrow
# one. TSELS has no bf16 form on either arch, so the selection takes the
# tile.full + TSEL fallback; TSEL does have one, which is what makes bf16
# selectable at all. It had no coverage anywhere before this file.
@st.cases(
    _split_dtype_case(pl.FP32, torch.float32, pl.BF16, torch.bfloat16, -3.0, name="select_dtype_bf16"),
)
def test_bf16_through_the_tsel_fallback(case_run):
    """A bf16 selection, which no TSELS form covers on either arch."""
    case_run.assert_passed()


# 8-bit is A5-only. On A2/A3 every op this could lower through stops at 16 bits
# -- `tsels`, `tsel`, and the `texpands` behind `tile.full` alike -- so there is
# no lowering to test and `tile.select` reports that instead of emitting one.
@st.cases(
    _split_dtype_case(
        pl.INT32, torch.int32, pl.INT8, torch.int8, -3, name="select_dtype_int8_a5", platform="a5"
    ),
)
def test_int8_selection_on_a5(case_run):
    """An 8-bit selection, which only A5 has instructions for."""
    case_run.assert_passed()


# Tails. The materialized branch inherits the other branch's valid extents; get
# that wrong and the select silently widens back to the full tile.
@st.cases(
    _scalar_case("false", 4, pl.FP32, torch.float32, valid=ROW_TAIL, name="select_tail_rows"),
    _scalar_case("false", 4, pl.FP32, torch.float32, valid=COL_TAIL, name="select_tail_cols"),
    _scalar_case("false", 4, pl.FP32, torch.float32, valid=COMBINED_TAIL, name="select_tail_both"),
    _scalar_case("true", 4, pl.FP32, torch.float32, valid=COMBINED_TAIL, name="select_tail_materialized"),
    _two_tile_case(4, pl.FP32, torch.float32, valid=COMBINED_TAIL, name="select_tail_two_tiles"),
)
def test_valid_shape_tails(case_run):
    """A narrowed valid region must survive, and nothing outside it may be written."""
    case_run.assert_passed()


# Wide tiles: one logical select spanning several physical vector registers,
# which PTOAS splits. This is the acceptance criterion issue #2742 raised.
@st.cases(
    _scalar_case("false", 4, pl.FP32, torch.float32, cols=512, name="select_wide_512"),
    _scalar_case("false", 4, pl.FP32, torch.float32, cols=1024, name="select_wide_1024"),
)
def test_multiple_physical_vector_parts(case_run):
    """A logical select wider than one physical vector register."""
    case_run.assert_passed()


@pytest.mark.xfail(
    strict=False,
    reason=(
        "A2/A3 only: the second TSELS's scratch is reused onto the buffer the preceding "
        "tcmps reads. TSELS opens with a one-element scalar-pipe store at offset 0 "
        "(pto-isa a2a3/TSels.hpp writes the scalar into tmp for set_cmpmask), which races "
        "the vector-pipe read still in flight, so element 0 of the compare source is "
        "corrupted and the mask's first bit comes out wrong -- observed as exactly "
        "1/2048 elements differing. Memory reuse is right that the source is dead by IR "
        "liveness; the gap is the cross-pipe WAR sync, which is not specific to "
        "tile.select and reproduces for a hand-written tile.sels given the same "
        "allocation. Expected to pass on A5, where the scalar goes to a vector register "
        "via vdup and never lands in UB."
    ),
)
@st.cases(
    st.case(
        _clamp,
        _source(M, N, torch.float32),
        torch.zeros(M, N, dtype=torch.float32),
        name="select_composed_clamp",
        golden=lambda t: t["v"].clamp(-2.0, 2.0),
    )
)
def test_composed_selects(case_run):
    """Two selects chained into a clamp, the composition issue #2742 asks for."""
    case_run.assert_passed()


# NaN. Issue #2742 requires the float comparison contract to be *defined and
# tested*; nothing in PyPTO or the pinned PTOAS documents TCMP's unordered
# behaviour. These cases assert the IEEE reading -- an unordered pair compares
# false for eq/lt/le/gt/ge and true for ne -- so a failure here is the
# measurement that settles the contract, not a flake. Whatever they report gets
# written into the op docs.
#
# This also gates an optimization: rewriting `select(a < b, x, y)` into
# `select(a >= b, y, x)` to reach TSELS is only sound if unordered compares are
# not both-false, so it stays deferred until these run.
def _nan_source() -> torch.Tensor:
    src = _source(M, N, torch.float32)
    src[::3, ::5] = float("nan")
    return src


@st.cases(
    *[
        st.case(
            _scalar_on_false(c, pl.FP32, M, N, None, _THRESHOLD, _FALSE_SCALAR),
            _nan_source(),
            torch.zeros(M, N, dtype=torch.float32),
            name=f"select_nan_{_CMP_IDS[c]}",
            golden=_golden_scalar("false", c, None, torch.float32),
        )
        for c in range(6)
    ]
)
def test_nan_comparison_contract(case_run):
    """Pin the unordered-comparison contract by executing it."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
