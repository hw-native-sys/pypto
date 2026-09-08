# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device tests for ``saturation_mode`` on the FP16 -> INT8 cast.

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
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

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


class CastMultiHopSaturationCase(PTOTestCase):
    """FP32 -> INT8 with ON, which reaches the destination through an FP16 hop."""

    __test__ = False

    def __init__(self, *, values: list[float], tag: str):
        super().__init__(platform="a2a3")
        self.values = values
        self.tag = tag

    def get_name(self) -> str:
        return f"cast_saturation_multihop_on_{self.tag}"

    def define_tensors(self) -> list[TensorSpec]:
        values = self.values

        def _src() -> torch.Tensor:
            repeats = (N + len(values) - 1) // len(values)
            return torch.tensor((values * repeats)[:N], dtype=torch.float32).reshape(1, N)

        return [
            TensorSpec("src", [1, N], DataType.FP32, init_value=_src),
            TensorSpec("out", [1, N], DataType.INT8, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        return _cast_multihop_on_program.specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        # Every value here is exactly representable in FP16, so the widening hop
        # is lossless and only the final INT8 hop rounds and clamps.
        truncated = torch.trunc(tensors["src"].to(torch.float32))
        tensors["out"][:] = torch.clamp(truncated, -128.0, 127.0).to(torch.int8)


class CastSaturationCase(PTOTestCase):
    """One FP16 -> INT8 cast at one saturation mode over one input set."""

    __test__ = False

    def __init__(self, *, saturation_mode: str | None, values: list[float], tag: str):
        super().__init__(platform="a2a3")
        self.saturation_mode = saturation_mode
        self.values = values
        self.tag = tag

    def get_name(self) -> str:
        return f"cast_saturation_{self.saturation_mode or 'default'}_{self.tag}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [1, N], DataType.FP16, init_value=lambda: _tile(self.values)),
            TensorSpec("out", [1, N], DataType.INT8, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        if self.saturation_mode == "on_explicit_tmp":
            return _cast_on_explicit_tmp_program.specialize()
        return _make_program(self.saturation_mode).specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        truncated = torch.trunc(tensors["src"].to(torch.float32))
        # Every value in the in-range set already lies inside [-128, 127], so the
        # clamp is a no-op there and this one expression serves both case shapes.
        tensors["out"][:] = torch.clamp(truncated, -128.0, 127.0).to(torch.int8)


class TestCastSaturation:
    """FP16 -> INT8 saturation behavior on Ascend 910B."""

    @pytest.mark.parametrize(
        "saturation_mode",
        ["on", "on_explicit_tmp", "off", None],
        ids=["on", "on-explicit-tmp", "off", "default"],
    )
    def test_in_range_values_agree_across_modes(self, test_runner, saturation_mode):
        """Every mode -- including ON with a caller-supplied tmp -- agrees on representable inputs."""
        result = test_runner.run(
            CastSaturationCase(saturation_mode=saturation_mode, values=_IN_RANGE, tag="in_range")
        )
        assert result.passed, f"in-range cast failed for saturation_mode={saturation_mode}: {result.error}"

    @pytest.mark.parametrize("saturation_mode", ["on", None], ids=["explicit-on", "default"])
    def test_saturation_clamps_out_of_range_values(self, test_runner, saturation_mode):
        """Clamping to the INT8 endpoints holds up to the FP16 extrema -- and is the default.

        Running the defaulted kernel through the same expectation is what pins
        ``"on"`` as the default on hardware, rather than only in the IR.
        """
        result = test_runner.run(
            CastSaturationCase(saturation_mode=saturation_mode, values=_OVERFLOW, tag="overflow")
        )
        assert result.passed, f"saturating cast failed to clamp: {result.error}"

    @pytest.mark.parametrize(
        "values, tag", [(_IN_RANGE, "in_range"), (_OVERFLOW, "overflow")], ids=["in-range", "overflow"]
    )
    def test_multi_hop_cast_saturates_on_its_final_hop(self, test_runner, values, tag):
        """FP32 -> INT8 goes through FP16, and the requested clamp still applies at the end."""
        result = test_runner.run(CastMultiHopSaturationCase(values=values, tag=tag))
        assert result.passed, f"multi-hop saturating cast failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
