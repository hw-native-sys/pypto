# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime regression: a row-narrowed matmul Acc and its two readers (issues #2510, #2470).

``mad`` is issued with ``M = aMatrix.GetValidRow()``, so a matmul whose left
operand is row-narrowed lays its L0C result out with an N-fractal stride of
``ceil(validRow/16)*16`` instead of the physical row count, and the tile carries
``CompactMode::normal`` to say so. Every L0C reader recomputes that pitch only
for a compact tile, so a chain that loses the mode — or a transport that rewrites
``validRow`` before the read — walks L0C at a stride ``mad`` never wrote at. With
a 64-row box valid to 16 rows the reader's N-fractal ``j`` picks up the matmul's
fractal ``4j``, and only the first 16 columns of each ``N_TILE`` survive.

The two cases here are the same arithmetic through the two readers:

* ``mixed`` keeps the accumulator in the scope, so a vector epilogue reads it
  through the Cube→Vector FIFO — the shape #2510 reported, where the transport
  normalization used to widen the pushed tile's rows to the physical box.
* ``staged`` sends the accumulator to GM (the ``TSTORE`` reader) over a K wide
  enough that the compiler synthesizes its own K-accumulation loop, whose seed is
  where the chain used to lose the mode (#2470).
* ``carried`` writes that K loop by hand -- a ``pl.create_tensor`` seed seeded before a
  ``pl.pipeline`` and rebound under ``if k0 == 0`` -- which is how the model kernel in
  #2470 spells it. The carry is typed from the seed alone, so the narrowing every matmul
  in the body produced used to die at the loop boundary and the store read the full box.

The first two returned 14336 of 65536 elements wrong before their fix; the third returned
75264 of 131072 wrong in the issue's own reproducer, its 48-row tail garbage rather than
the untouched zeros the others left.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

M_TILE = 64  # physical accumulator rows
VALID_ROWS = 16  # rows that actually hold data
N_TILE = 128
K_ONE_BLOCK = 256  # one L0 K block: a single tile.matmul
K_MULTI_BLOCK = 2048  # several L0 K blocks: a synthesized K-accumulation loop
K_BLOCK = 512  # per-iteration K slice of the hand-written carry loop
SCALE = 1.0 / 4096.0


def _int8(shape: list[int]) -> torch.Tensor:
    return torch.randint(-127, 128, shape, dtype=torch.int32).to(torch.int8)


def _expected(tensors: dict[str, torch.Tensor]) -> None:
    """``out[:VALID_ROWS] = (x @ w.T) * scale``; the rest is pl.fillpad's zero."""
    x = tensors["x"].to(torch.float64)
    w = tensors["w"].to(torch.float64)
    scale = tensors["scale"].to(torch.float64)
    out = torch.zeros(M_TILE, N_TILE, dtype=torch.float64)
    out[:VALID_ROWS] = (x[:VALID_ROWS] @ w.T) * scale[:VALID_ROWS]
    tensors["out"][:] = out.to(torch.float32)


def _io_tensors(k: int) -> list[TensorSpec]:
    return [
        TensorSpec("x", [M_TILE, k], DataType.INT8, init_value=lambda: _int8([M_TILE, k])),
        TensorSpec("w", [N_TILE, k], DataType.INT8, init_value=lambda: _int8([N_TILE, k])),
        TensorSpec(
            "scale",
            [M_TILE, 1],
            DataType.FP32,
            init_value=lambda: torch.full((M_TILE, 1), SCALE, dtype=torch.float32),
        ),
        TensorSpec("out", [M_TILE, N_TILE], DataType.FP32, init_value=torch.zeros, is_output=True),
    ]


class _MixedEpilogueCase(PTOTestCase):
    """Cube matmul + vector dequant in ONE scope: the accumulator crosses the C2V FIFO."""

    __test__ = False

    def __init__(self, *, platform=None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"c2v_narrowed_acc_mixed_k{K_ONE_BLOCK}"

    def define_tensors(self) -> list[TensorSpec]:
        return _io_tensors(K_ONE_BLOCK)

    def get_program(self) -> Any:
        k = K_ONE_BLOCK

        @pl.program
        class MixedEpilogueProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[M_TILE, k], pl.INT8],
                w: pl.Tensor[[N_TILE, k], pl.INT8],
                scale: pl.Tensor[[M_TILE, 1], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
            ) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
                xk = pl.slice(x, [M_TILE, k], [0, 0], valid_shape=[VALID_ROWS, k])
                acc = pl.matmul(xk, w, b_trans=True, out_dtype=pl.INT32)  # cube
                deq = pl.row_expand_mul(pl.cast(acc, target_type=pl.FP32, mode="none"), scale)
                out[:] = pl.fillpad(pl.set_validshape(deq, VALID_ROWS, N_TILE), pad_value=pl.PadValue.zero)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[M_TILE, k], pl.INT8],
                w: pl.Tensor[[N_TILE, k], pl.INT8],
                scale: pl.Tensor[[M_TILE, 1], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
            ) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
                out = self.kernel(x, w, scale, out)
                return out

        return MixedEpilogueProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _expected(tensors)


class _StagedKSplitCase(PTOTestCase):
    """Cube matmul over a split K stores INT32 to GM; a second kernel dequants it."""

    __test__ = False

    def __init__(self, *, platform=None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"c2v_narrowed_acc_staged_k{K_MULTI_BLOCK}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            *_io_tensors(K_MULTI_BLOCK),
            TensorSpec("acc_gm", [M_TILE, N_TILE], DataType.INT32, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        k = K_MULTI_BLOCK

        @pl.program
        class StagedKSplitProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def cube(
                self,
                x: pl.Tensor[[M_TILE, k], pl.INT8],
                w: pl.Tensor[[N_TILE, k], pl.INT8],
                acc_gm: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.INT32]],
            ) -> pl.Tensor[[M_TILE, N_TILE], pl.INT32]:
                xk = pl.slice(x, [M_TILE, k], [0, 0], valid_shape=[VALID_ROWS, k])
                acc_gm[:] = pl.matmul(xk, w, b_trans=True, out_dtype=pl.INT32)
                return acc_gm

            @pl.function(type=pl.FunctionType.InCore)
            def dequant(
                self,
                acc_gm: pl.Tensor[[M_TILE, N_TILE], pl.INT32],
                scale: pl.Tensor[[M_TILE, 1], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
            ) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
                deq = pl.row_expand_mul(pl.cast(acc_gm[:], target_type=pl.FP32, mode="none"), scale)
                out[:] = pl.fillpad(pl.set_validshape(deq, VALID_ROWS, N_TILE), pad_value=pl.PadValue.zero)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[M_TILE, k], pl.INT8],
                w: pl.Tensor[[N_TILE, k], pl.INT8],
                scale: pl.Tensor[[M_TILE, 1], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
                acc_gm: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.INT32]],
            ) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
                acc_gm = self.cube(x, w, acc_gm)
                out = self.dequant(acc_gm, scale, out)
                return out

        return StagedKSplitProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _expected(tensors)


class _CarriedKLoopCase(PTOTestCase):
    """A hand-written K loop whose accumulator is seeded by ``pl.create_tensor``."""

    __test__ = False

    def __init__(self, *, platform=None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"c2v_narrowed_acc_carried_k{K_MULTI_BLOCK}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            *_io_tensors(K_MULTI_BLOCK),
            TensorSpec("acc_gm", [M_TILE, N_TILE], DataType.INT32, init_value=torch.zeros),
        ]

    def get_program(self) -> Any:
        k = K_MULTI_BLOCK

        @pl.program
        class CarriedKLoopProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def cube(
                self,
                x: pl.Tensor[[M_TILE, k], pl.INT8],
                w: pl.Tensor[[N_TILE, k], pl.INT8],
                acc_gm: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.INT32]],
            ) -> pl.Tensor[[M_TILE, N_TILE], pl.INT32]:
                acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, k, K_BLOCK, stage=2):
                    xk = pl.slice(x, [M_TILE, K_BLOCK], [0, k0], valid_shape=[VALID_ROWS, K_BLOCK])
                    wk = pl.slice(w, [N_TILE, K_BLOCK], [0, k0])
                    if k0 == 0:
                        acc = pl.matmul(xk, wk, b_trans=True, out_dtype=pl.INT32)
                    else:
                        acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
                acc_gm[:] = acc
                return acc_gm

            @pl.function(type=pl.FunctionType.InCore)
            def dequant(
                self,
                acc_gm: pl.Tensor[[M_TILE, N_TILE], pl.INT32],
                scale: pl.Tensor[[M_TILE, 1], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
            ) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
                deq = pl.row_expand_mul(pl.cast(acc_gm[:], target_type=pl.FP32, mode="none"), scale)
                out[:] = pl.fillpad(pl.set_validshape(deq, VALID_ROWS, N_TILE), pad_value=pl.PadValue.zero)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                x: pl.Tensor[[M_TILE, k], pl.INT8],
                w: pl.Tensor[[N_TILE, k], pl.INT8],
                scale: pl.Tensor[[M_TILE, 1], pl.FP32],
                out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
                acc_gm: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.INT32]],
            ) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
                acc_gm = self.cube(x, w, acc_gm)
                out = self.dequant(acc_gm, scale, out)
                return out

        return CarriedKLoopProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _expected(tensors)


class TestNarrowedAccEpilogue:
    """A narrowed accumulator must survive both of its readers."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
    def test_vector_epilogue_in_the_same_scope(self, test_runner, platform):
        """The Cube→Vector push path (#2510)."""
        result = test_runner.run(_MixedEpilogueCase(platform=platform))
        assert result.passed, f"mixed cube+vector epilogue failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
    def test_gm_staged_accumulator_over_a_split_k(self, test_runner, platform):
        """The TSTORE path over a compiler-synthesized K loop (#2470)."""
        result = test_runner.run(_StagedKSplitCase(platform=platform))
        assert result.passed, f"GM-staged accumulator failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("platform", [pytest.param("a2a3", id="a2a3")])
    def test_gm_stored_accumulator_carried_by_a_hand_written_loop(self, test_runner, platform):
        """The TSTORE path over a user-written carry seeded by pl.create_tensor (#2470)."""
        result = test_runner.run(_CarriedKLoopCase(platform=platform))
        assert result.passed, f"carried accumulator failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
