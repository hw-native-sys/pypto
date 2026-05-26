# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for issue #1509: writing a source whose ``valid_shape`` is
narrower than its ``static_shape`` (typical when ISA alignment forces a Vec
tile to pad rows to 32 bytes) into a target slot sized to ``valid_shape``.

Two write paths must produce identical, valid_shape-respecting output:

  1. Direct API:        ``pl.assemble(output, src, [r, 0])``
  2. Subscript-write:   ``output[r:r+H, 0:W] = src``

Subscript-write is parser sugar over ``pl.assemble``, so post-parse both paths
produce the same IR; this test exercises end-to-end runtime equivalence and
verifies that the source's static-but-invalid padding (filled with a sentinel)
does NOT leak into the output.
"""

from collections.abc import Callable
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import ONBOARD_PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 64
N = 32
SRC_ROWS = 16
SRC_COLS_STATIC = 8  # ISA-padded static extent (32-byte row alignment for FP32 Vec)
SRC_COLS_VALID = 4  # logical extent the user wants written
T_OFFSET_ROW = 8  # non-zero offset to exercise offset semantics
PAD_SENTINEL = -1000.0  # marks "static-but-invalid" cells of the source


@pl.program
class AssembleDirectNarrowValidProgram:
    """Write via direct ``pl.assemble`` — source has narrow valid_shape."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[SRC_ROWS, SRC_COLS_STATIC], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            narrowed = pl.set_validshape(src, SRC_ROWS, SRC_COLS_VALID)
            output = pl.assemble(output, narrowed, [T_OFFSET_ROW, 0])
        return output


@pl.program
class AssembleSubscriptNarrowValidProgram:
    """Write via subscript-write sugar — semantically equivalent to the direct path."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[SRC_ROWS, SRC_COLS_STATIC], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            narrowed = pl.set_validshape(src, SRC_ROWS, SRC_COLS_VALID)
            output[T_OFFSET_ROW : T_OFFSET_ROW + SRC_ROWS, 0:SRC_COLS_VALID] = narrowed
        return output


@pl.program
class AssembleDirectDstEqValidProgram:
    """``dst.shape == src.valid_shape`` — the whole target is the valid region.

    Issue #1509's canonical "narrow write" shape: caller has an ISA-padded
    source ``[16, 8]`` whose logical content is only ``[16, 4]``, and a target
    sized exactly to the logical content ``[16, 4]``. ``offset = [0, 0]``;
    ``offset + src.static = [16, 8]`` would OOB the ``[16, 4]`` target —
    only a valid_shape-respecting write is safe.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[SRC_ROWS, SRC_COLS_STATIC], pl.FP32],
        output: pl.Out[pl.Tensor[[SRC_ROWS, SRC_COLS_VALID], pl.FP32]],
    ) -> pl.Tensor[[SRC_ROWS, SRC_COLS_VALID], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            narrowed = pl.set_validshape(src, SRC_ROWS, SRC_COLS_VALID)
            output = pl.assemble(output, narrowed, [0, 0])
        return output


@pl.program
class AssembleSubscriptDstEqValidProgram:
    """Subscript-write variant of ``AssembleDirectDstEqValidProgram``."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[SRC_ROWS, SRC_COLS_STATIC], pl.FP32],
        output: pl.Out[pl.Tensor[[SRC_ROWS, SRC_COLS_VALID], pl.FP32]],
    ) -> pl.Tensor[[SRC_ROWS, SRC_COLS_VALID], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            narrowed = pl.set_validshape(src, SRC_ROWS, SRC_COLS_VALID)
            output[:, :] = narrowed
        return output


def _make_src() -> torch.Tensor:
    """Build the source tensor.

    Layout: valid cols 0..SRC_COLS_VALID hold sequential payload values; the
    static-but-invalid cols SRC_COLS_VALID..SRC_COLS_STATIC hold PAD_SENTINEL.
    If the runtime mistakenly writes ``static_shape`` cells, the sentinel would
    appear in the output and fail the assertion.
    """
    src = torch.full((SRC_ROWS, SRC_COLS_STATIC), PAD_SENTINEL, dtype=torch.float32)
    payload = torch.arange(SRC_ROWS * SRC_COLS_VALID, dtype=torch.float32).reshape(SRC_ROWS, SRC_COLS_VALID)
    src[:, :SRC_COLS_VALID] = payload
    return src


def _expected(src: torch.Tensor) -> torch.Tensor:
    """Output starts zero-initialised. Only the [T_OFFSET_ROW:, 0:SRC_COLS_VALID]
    slot receives src's valid region; the rest stays zero — including the
    [..., SRC_COLS_VALID:SRC_COLS_STATIC] sub-window in the target row band,
    which must NOT carry the PAD_SENTINEL.
    """
    expected = torch.zeros((M, N), dtype=torch.float32)
    expected[T_OFFSET_ROW : T_OFFSET_ROW + SRC_ROWS, :SRC_COLS_VALID] = src[:, :SRC_COLS_VALID]
    return expected


def _expected_dst_eq_valid(src: torch.Tensor) -> torch.Tensor:
    """Output is sized to the valid region; the whole target receives src's valid
    region. If the runtime wrote ``static_shape`` cells, it would OOB the target.
    """
    return src[:, :SRC_COLS_VALID].clone()


class _AssembleNarrowValidTestCase(PTOTestCase):
    """Shared scaffolding parametrised by program, output shape, and expected fn."""

    __test__ = False

    def __init__(
        self,
        name: str,
        program_cls: Any,
        output_shape: list[int],
        expected_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        self._name = name
        self._program_cls = program_cls
        self._output_shape = output_shape
        self._expected_fn = expected_fn
        self._src = _make_src()

    def get_name(self) -> str:
        return self._name

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [SRC_ROWS, SRC_COLS_STATIC], DataType.FP32, init_value=self._src),
            TensorSpec("output", self._output_shape, DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return self._program_cls

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["output"][:] = self._expected_fn(self._src)


class TestAssembleNarrowValidShape:
    """Issue #1509: assemble / subscript-write must honour source's valid_shape,
    writing only the valid region (mirroring tile.store's tile-side semantics).

    Covers two shape families:

    * Large target with a narrow slot at non-zero offset (issue's typical
      "loop-carry writeback into GM" pattern).
    * Target sized exactly to the valid region, ``offset = [0, 0]`` —
      ``offset + src.static`` would OOB the target, so this case can only
      succeed if the runtime honours ``valid_shape``.
    """

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_assemble_direct(self, test_runner, platform):
        result = test_runner.run(
            _AssembleNarrowValidTestCase(
                "assemble_narrow_valid_direct",
                AssembleDirectNarrowValidProgram,
                [M, N],
                _expected,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_assemble_subscript_write(self, test_runner, platform):
        result = test_runner.run(
            _AssembleNarrowValidTestCase(
                "assemble_narrow_valid_subscript",
                AssembleSubscriptNarrowValidProgram,
                [M, N],
                _expected,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_assemble_direct_dst_eq_valid(self, test_runner, platform):
        result = test_runner.run(
            _AssembleNarrowValidTestCase(
                "assemble_narrow_valid_dst_eq_valid_direct",
                AssembleDirectDstEqValidProgram,
                [SRC_ROWS, SRC_COLS_VALID],
                _expected_dst_eq_valid,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", ONBOARD_PLATFORMS)
    def test_assemble_subscript_dst_eq_valid(self, test_runner, platform):
        result = test_runner.run(
            _AssembleNarrowValidTestCase(
                "assemble_narrow_valid_dst_eq_valid_subscript",
                AssembleSubscriptDstEqValidProgram,
                [SRC_ROWS, SRC_COLS_VALID],
                _expected_dst_eq_valid,
                platform=platform,
            )
        )
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
