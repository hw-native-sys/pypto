# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device sentinel test for ``pl.gather_row(..., valid_shape=[<runtime>, ...])``.

``shapes`` sizes the ``pto.subview`` carved out of the L1 accumulator and must be
compile-time constant (ptoas types ``sizes`` as a static ``I64ArrayAttr``). The
optional ``valid_shape`` carries the *runtime* transfer extent instead, feeding
the subview's ``valid_row``/``valid_col`` operands and the GM
``pto.partition_view`` sizes. These tests prove on real hardware that a runtime
row count genuinely limits how many rows move.

**Why a sentinel, and why it is discriminating.** A partial gather is only
observable if the untouched part of the destination holds something recognisable,
so each case fills the accumulator twice:

1. A **full-window** gather (plain 5-arg form) from pool rows ``[SENT_BASE, SENT_BASE+ROWS)``
   paints all ``ROWS`` slots with sentinel values.
2. A **dynamic** gather from pool rows ``[0, r)`` with ``valid_shape=[r, HEAD_DIM]``
   overwrites only the first ``r`` slots.

Because ``src[p, :] == p`` (row-id encoding, FP16-exact for ``p <= 255``), the
expected result is ``out[i, :] == i`` for ``i < r`` and ``out[i, :] == SENT_BASE + i``
for ``i >= r``. If the runtime extent were ignored and the full window moved,
every row would read ``i`` and the tail assertion fails immediately — which is
exactly the silent-wrong-data failure this guards against. The two source regions
are disjoint and their values differ by ``SENT_BASE`` in every element, so an
off-by-one or a partially-applied extent is caught too.

The gathered tile carries the matmul-operand NZ (boxed) layout and so cannot be
pushed straight to GM; as in ``test_paged_gather.py`` it is read back via
``eye @ gathered``, which reproduces the gathered rows exactly.

The row count arrives as a **GM scalar read** rather than a kernel parameter, so
it cannot be constant-folded into the static ``shapes`` — the value is genuinely
unknown at compile time.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

HEAD_DIM = 128
ROWS = 128  # L1 accumulator rows == the static gather window
SENT_BASE = 128  # sentinel source rows: [128, 256)
POOL_ROWS = SENT_BASE + ROWS  # 256 — max row id 255 stays FP16-exact

# Runtime extents under test. r >= 1 is required by the device library:
# TLoadGm2L1Nd2nz asserts `gShape3 > 0`, and gShape3 is the transfer row count.
# r == ROWS is the control case (dynamic operand that happens to equal the window).
VALID_ROWS = [1, 63, 127, ROWS]


def _make_src() -> torch.Tensor:
    """src[p, c] = p — row-id encoding, so a gathered slot reveals its source row."""
    rows = torch.arange(POOL_ROWS, dtype=torch.float32).reshape(POOL_ROWS, 1)
    return rows.expand(POOL_ROWS, HEAD_DIM).to(torch.float16).contiguous()


def _gather_row_dynamic_golden(src: torch.Tensor, r: int) -> torch.Tensor:
    """Sentinel fill of the whole window, then the first r rows overwritten."""
    out = src[SENT_BASE : SENT_BASE + ROWS, :HEAD_DIM].clone()
    out[:r, :] = src[:r, :HEAD_DIM]
    return out


@pl.program
class GatherRowDynamicValidShapeProgram:
    """Sentinel-fill an L1 accumulator, then overwrite a runtime number of rows.

    ``n`` is read from GM so the extent is opaque to the compiler. The first
    ``pl.gather_row`` omits ``valid_shape`` (the pre-existing 5-arg form) and
    paints the sentinel; the second passes the runtime ``rows`` and must touch
    only that many slots.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[POOL_ROWS, HEAD_DIM], pl.FP16],
        n: pl.Tensor[[1], pl.INT32],
        eye: pl.Tensor[[ROWS, ROWS], pl.FP16],
        output: pl.Out[pl.Tensor[[ROWS, HEAD_DIM], pl.FP32]],
    ) -> pl.Tensor[[ROWS, HEAD_DIM], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            rows = pl.cast(pl.read(n, [0]), pl.INDEX)
            kv = pl.create_l1([ROWS, HEAD_DIM], pl.FP16)
            # Sentinel: every slot gets pool row SENT_BASE + i.
            kv = pl.gather_row(kv, src, [0, 0], [SENT_BASE, 0], [ROWS, HEAD_DIM])
            # Dynamic: only the first `rows` slots get pool row i.
            kv = pl.gather_row(kv, src, [0, 0], [0, 0], [ROWS, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
            result = pl.matmul(eye, kv, out_dtype=pl.FP32)
            output = pl.assemble(output, result, [0, 0])
        return output


class GatherRowDynamicValidShapeTestCase(PTOTestCase):
    """One case per runtime extent in VALID_ROWS."""

    __test__ = False

    def __init__(self, valid_rows: int, *, platform: str | None = None):
        super().__init__(None, platform=platform)
        self._valid_rows = valid_rows

    def get_name(self) -> str:
        return f"gather_row_dynamic_valid_shape_r{self._valid_rows}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_program(self) -> Any:
        return GatherRowDynamicValidShapeProgram

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [POOL_ROWS, HEAD_DIM], DataType.FP16, init_value=_make_src),
            TensorSpec(
                "n",
                [1],
                DataType.INT32,
                init_value=torch.tensor([self._valid_rows], dtype=torch.int32),
            ),
            TensorSpec(
                "eye",
                [ROWS, ROWS],
                DataType.FP16,
                init_value=lambda: torch.eye(ROWS, dtype=torch.float16),
            ),
            TensorSpec("output", [ROWS, HEAD_DIM], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors, params=None):
        # eye @ gathered == gathered, so the golden is the accumulator contents.
        tensors["output"][:] = _gather_row_dynamic_golden(tensors["src"], self._valid_rows).to(torch.float32)


@pl.program
class GatherRowTwoRunProgram:
    """Two contiguous runs into one accumulator, split at a runtime row ``r1``.

    The motivating shape: a page-aligned KV window that straddles a block boundary
    is 1-2 contiguous runs whose split point is only known at runtime. Run 2 writes
    at the runtime offset ``r1``, so its static ``shapes`` window spans rows
    ``[r1, r1 + ROWS)`` — past the end of the tile. Only ``ROWS - r1`` rows are
    actually transferred, so the *write* stays in bounds, but whether the declared
    window is tolerated is a device question, which is what this exercises.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[POOL_ROWS, HEAD_DIM], pl.FP16],
        n: pl.Tensor[[1], pl.INT32],
        eye: pl.Tensor[[ROWS, ROWS], pl.FP16],
        output: pl.Out[pl.Tensor[[ROWS, HEAD_DIM], pl.FP32]],
    ) -> pl.Tensor[[ROWS, HEAD_DIM], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            r1 = pl.cast(pl.read(n, [0]), pl.INDEX)
            kv = pl.create_l1([ROWS, HEAD_DIM], pl.FP16)
            # Run 1: pool rows [0, r1) -> slots [0, r1).
            kv = pl.gather_row(kv, src, [0, 0], [0, 0], [ROWS, HEAD_DIM], valid_shape=[r1, HEAD_DIM])
            # Run 2: pool rows [SENT_BASE, SENT_BASE + ROWS - r1) -> slots [r1, ROWS).
            kv = pl.gather_row(
                kv, src, [r1, 0], [SENT_BASE, 0], [ROWS, HEAD_DIM], valid_shape=[ROWS - r1, HEAD_DIM]
            )
            result = pl.matmul(eye, kv, out_dtype=pl.FP32)
            output = pl.assemble(output, result, [0, 0])
        return output


class GatherRowTwoRunTestCase(GatherRowDynamicValidShapeTestCase):
    """Two-run split at a runtime boundary."""

    __test__ = False

    def get_name(self) -> str:
        return f"gather_row_two_run_r{self._valid_rows}"

    def get_program(self) -> Any:
        return GatherRowTwoRunProgram

    def compute_expected(self, tensors, params=None):
        r = self._valid_rows
        src = tensors["src"]
        out = torch.empty(ROWS, HEAD_DIM, dtype=torch.float16)
        out[:r, :] = src[:r, :HEAD_DIM]
        out[r:, :] = src[SENT_BASE : SENT_BASE + (ROWS - r), :HEAD_DIM]
        tensors["output"][:] = out.to(torch.float32)


class TestGatherRowDynamicValidShape:
    """A runtime valid_shape limits a GM -> L1 gather on real hardware."""

    @pytest.mark.parametrize("valid_rows", VALID_ROWS)
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_row_dynamic_valid_shape(self, test_runner, platform, valid_rows):
        result = test_runner.run(GatherRowDynamicValidShapeTestCase(valid_rows, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("valid_rows", [1, 63, 127])
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_gather_row_two_run_split(self, test_runner, platform, valid_rows):
        """Two runs split at a runtime row — the page-boundary shape this exists for."""
        result = test_runner.run(GatherRowTwoRunTestCase(valid_rows, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
