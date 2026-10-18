# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device tests for cast saturation and A2/A3 scratch planning.

FP16 -> INT8 is the pair the A2/A3 assembler lowers two different ways: without
saturation it emulates the target's overflow behavior with a chunked vector
helper, and with saturation it issues the native conversion. The two are only
required to agree on values the destination can already represent, so the cases
below split accordingly:

  ``in-range``  -- every requested mode, plus the default, on finite inputs the
                   INT8 range holds. All four must produce the same exact bytes:
                   that equivalence is what makes saturation safe to have as the
                   default for a quantizer whose final input is already bounded.
  ``overflow``  -- saturating only (explicit and defaulted), on finite inputs
                   outside the range, up to the FP16 extrema. Saturation is
                   defined to clamp, so the expectation is exact. OFF's behavior
                   there is architecture-defined and is deliberately not pinned
                   by a test.

Non-finite inputs (NaN / Inf) are not covered: neither mode defines them.
The planner matrix additionally covers all four scratch-backed conversions,
checking exact finite results and preserved output outside partial tiles.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto.pypto_core.passes import MemoryPlanner

N = 128

# Finite FP16 values the INT8 range holds after truncation, including the
# boundaries themselves and half-way magnitudes that a rounding-mode mistake
# would move. trunc(127.5) == 127 and trunc(-128.5) == -128 are in range.
_IN_RANGE = [
    0.0,
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    126.9375,
    -126.9375,
    127.0,
    -127.0,
    127.5,
    -128.0,
    -128.5,
]

# Finite FP16 values outside the INT8 range, up to the FP16 extrema.
_OVERFLOW = [128.0, -129.0, 200.0, -200.0, 1000.0, -1000.0, 65504.0, -65504.0]


def _tile(values: list[float]) -> torch.Tensor:
    """Repeat ``values`` across a [1, N] FP16 row so every lane is exercised."""
    repeats = (N + len(values) - 1) // len(values)
    flat = (values * repeats)[:N]
    return torch.tensor(flat, dtype=torch.float16).reshape(1, N)


def _make_program(saturation_mode: str | None):
    """One FP16 -> INT8 kernel per requested mode; ``None`` omits the kwarg entirely."""

    @pl.jit.incore
    def cast_kernel(
        src: pl.Tensor[[1, N], pl.FP16],
        out: pl.InOut[pl.Tensor[[1, N], pl.INT8]],
    ) -> pl.Tensor[[1, N], pl.INT8]:
        tile = pl.load(src, [0, 0], [1, N])
        quantized = pl.cast(tile, pl.INT8, mode="trunc", saturation_mode=saturation_mode)
        return pl.store(quantized, [0, 0], out)

    @pl.jit
    def cast_program(
        src: pl.Tensor[[1, N], pl.FP16],
        out: pl.InOut[pl.Tensor[[1, N], pl.INT8]],
    ) -> pl.Tensor[[1, N], pl.INT8]:
        return cast_kernel(src, out)

    return cast_program


# PTOAS's non-saturating FP16 -> INT8 helper sizes its scratch from the tile
# width; [1, 128] resolves to 256 bytes (see TcvtScratchCapacityBytes). Passing
# it explicitly alongside ON exercises the level-3 two-operand tcvt form, which
# the compiler never generates for a saturating cast on its own.
@pl.jit.incore
def _cast_on_explicit_tmp_kernel(
    src: pl.Tensor[[1, N], pl.FP16],
    out: pl.InOut[pl.Tensor[[1, N], pl.INT8]],
) -> pl.Tensor[[1, N], pl.INT8]:
    tile = pl.load(src, [0, 0], [1, N])
    tmp: pl.Tile[[1, 256], pl.INT8, pl.Mem.Vec] = pl.tile.create(
        [1, 256], dtype=pl.INT8, target_memory=pl.Mem.Vec
    )
    quantized = pl.tile.cast(tile, pl.INT8, mode="trunc", tmp=tmp, saturation_mode="on")
    return pl.store(quantized, [0, 0], out)


# FP32 -> INT8 is not a native A2/A3 tcvt; LegalizeTileCast expands it to
# FP32 -> FP16 -> INT8 and puts the requested saturation on the final hop.
@pl.jit.incore
def _cast_multihop_on_kernel(
    src: pl.Tensor[[1, N], pl.FP32],
    out: pl.InOut[pl.Tensor[[1, N], pl.INT8]],
) -> pl.Tensor[[1, N], pl.INT8]:
    tile = pl.load(src, [0, 0], [1, N])
    quantized = pl.cast(tile, pl.INT8, mode="trunc", saturation_mode="on")
    return pl.store(quantized, [0, 0], out)


@pl.jit
def _cast_on_explicit_tmp_program(
    src: pl.Tensor[[1, N], pl.FP16],
    out: pl.InOut[pl.Tensor[[1, N], pl.INT8]],
) -> pl.Tensor[[1, N], pl.INT8]:
    return _cast_on_explicit_tmp_kernel(src, out)


@pl.jit
def _cast_multihop_on_program(
    src: pl.Tensor[[1, N], pl.FP32],
    out: pl.InOut[pl.Tensor[[1, N], pl.INT8]],
) -> pl.Tensor[[1, N], pl.INT8]:
    return _cast_multihop_on_kernel(src, out)


def _saturation_cases():
    for mode in ("on", "on_explicit_tmp", "off", None):
        kernel = _cast_on_explicit_tmp_program if mode == "on_explicit_tmp" else _make_program(mode)
        yield st.case(
            kernel,
            _tile(_IN_RANGE),
            torch.zeros(1, N, dtype=torch.int8),
            name=f"cast_saturation_{mode or 'default'}_in_range",
            golden=_saturating_golden,
            rtol=0,
            atol=0,
        )
    for mode in ("on", None):
        yield st.case(
            _make_program(mode),
            _tile(_OVERFLOW),
            torch.zeros(1, N, dtype=torch.int8),
            name=f"cast_saturation_{mode or 'default'}_overflow",
            golden=_saturating_golden,
            rtol=0,
            atol=0,
        )
    for values, tag in ((_IN_RANGE, "in_range"), (_OVERFLOW, "overflow")):
        # Construct FP32 inputs directly: casting the FP16 fixture would hide
        # rounding on the first hop of the FP32 -> FP16 -> INT8 legalization.
        src = torch.tensor(
            (values * ((N + len(values) - 1) // len(values)))[:N], dtype=torch.float32
        ).reshape(1, N)
        yield st.case(
            _cast_multihop_on_program,
            src,
            torch.zeros(1, N, dtype=torch.int8),
            name=f"cast_saturation_multihop_on_{tag}",
            golden=_saturating_golden,
            rtol=0,
            atol=0,
        )


def _saturating_golden(tensors):
    return torch.clamp(torch.trunc(tensors["src"].float()), -128, 127).to(torch.int8)


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(*_saturation_cases())
def test_cast_saturation(case_run):
    """Preserve all eight original saturation cases through the shared JIT harness."""
    case_run.assert_passed()


def _make_planner_cast(dst_dtype, rows, valid_rows, valid_cols, cols=N, saturation_mode="off"):
    @pl.jit
    def cast_program(src: pl.Tensor, out: pl.InOut[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(src, [0, 0], [rows, cols], valid_shape=[valid_rows, valid_cols])
            converted = pl.cast(tile, dst_dtype, mode="trunc", saturation_mode=saturation_mode)
            pl.store(converted, [0, 0], out)
        return out

    return cast_program


def _planner_cases():
    for src_dtype, dst_dtype, torch_src, torch_dst in (
        (pl.FP32, pl.INT16, torch.float32, torch.int16),
        (pl.FP16, pl.INT16, torch.float16, torch.int16),
        (pl.FP16, pl.INT8, torch.float16, torch.int8),
        (pl.FP16, pl.UINT8, torch.float16, torch.uint8),
    ):
        for rows, valid_rows, valid_cols in ((1, 1, N), (4, 3, 96)):
            # OFF is only specified here for finite, in-range values. Fractional
            # values verify truncation, while the 77 sentinel detects writes
            # outside the partial tile's valid rectangle.
            if torch_dst == torch.uint8:
                values = [0.0, 0.5, 1.5, 127.5, 128.5, 254.5, 255.5]
            elif torch_dst == torch.int16:
                # 32752 is the largest FP16 value below INT16's upper bound.
                values = _IN_RANGE + [-32768.0, -32752.0, -1024.5, 1024.5, 32752.0]
            else:
                values = _IN_RANGE
            count = rows * N
            src = torch.tensor(
                (values * ((count + len(values) - 1) // len(values)))[:count], dtype=torch_src
            ).reshape(rows, N)
            expected = torch.full((rows, N), 77, dtype=torch_dst)
            expected[:valid_rows, :valid_cols] = torch.trunc(src[:valid_rows, :valid_cols].float()).to(
                torch_dst
            )
            for planner in MemoryPlanner:
                yield st.case(
                    _make_planner_cast(dst_dtype, rows, valid_rows, valid_cols),
                    src,
                    torch.full((rows, N), 77, dtype=torch_dst),
                    name=f"cast_scratch_{src_dtype}_{dst_dtype}_{valid_rows}x{valid_cols}_{planner.name}",
                    memory_planner=planner,
                    # Exercise tile.cast's static-view bridge even if Buffer IR
                    # gains support for these conversions in the future.
                    enable_buffer_ir=False,
                    golden=lambda _tensors, expected=expected: expected,
                    rtol=0,
                    atol=0,
                )


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(*_planner_cases())
def test_cast_scratch_across_planners(case_run):
    """Implicit PTOAS scratch must produce the same exact values as explicit scratch."""
    case_run.assert_passed()


def _narrowing_tail_cases():
    # A2/A3 processes every row's aligned head before any masked tail. Cover
    # native conversions too, and use 192 valid columns to exercise FP16's
    # 128-element head plus a tail. The scratch matrix already covers FP32 OFF.
    for src_dtype, dst_dtype, torch_src, torch_dst, cols, valid_cols, saturation in (
        (pl.FP32, pl.FP16, torch.float32, torch.float16, 128, 96, "off"),
        (pl.FP32, pl.INT16, torch.float32, torch.int16, 128, 96, "on"),
        (pl.FP16, pl.INT8, torch.float16, torch.int8, 256, 192, "off"),
        (pl.FP16, pl.UINT8, torch.float16, torch.uint8, 256, 192, "on"),
    ):
        values = [0.5, 1.5, 2.5, 63.5, 126.5]
        count = 4 * cols
        src = torch.tensor(
            (values * ((count + len(values) - 1) // len(values)))[:count], dtype=torch_src
        ).reshape(4, cols)
        expected = torch.full((4, cols), 77, dtype=torch_dst)
        valid = src[:3, :valid_cols]
        expected[:3, :valid_cols] = (valid if torch_dst.is_floating_point else torch.trunc(valid)).to(
            torch_dst
        )
        for planner in MemoryPlanner:
            yield st.case(
                _make_planner_cast(dst_dtype, 4, 3, valid_cols, cols, saturation),
                src,
                torch.full((4, cols), 77, dtype=torch_dst),
                name=f"cast_tail_{src_dtype}_{dst_dtype}_{saturation}_{planner.name}",
                memory_planner=planner,
                enable_buffer_ir=False,
                golden=lambda _tensors, expected=expected: expected,
                rtol=0,
                atol=0,
            )


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(*_narrowing_tail_cases())
def test_narrowing_cast_head_tail(case_run):
    """A later output row must not overwrite an earlier row's unread input tail."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
