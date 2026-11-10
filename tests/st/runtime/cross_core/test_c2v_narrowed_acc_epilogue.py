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

The ``..._across_a_split_boundary`` tests cross the same accumulator into TWO vector
lanes. A split push used to widen its rows to the box for the lanes' sake, which skews the
compact L0C read exactly like #2510 (1792 of 2048 valid elements wrong at 16 of 64 rows),
so the shape was refused outright. On A2/A3 each lane locates its own band and no lane
reads a row past the valid extent, so the push now keeps the producer's rows; a runtime
extent additionally pops the full per-lane box and carries the extent on its consumers.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime import RunConfig

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


def _explicit_split_epilogue(mode: pl.SplitMode, valid_rows: int):
    """An explicit split region over an accumulator narrowed through its left operand."""

    row_step = M_TILE // 2 if mode == pl.SplitMode.UP_DOWN else 0
    col_step = N_TILE // 2 if mode == pl.SplitMode.LEFT_RIGHT else 0

    @pl.jit
    def narrowed_acc_explicit_split(
        x: pl.Tensor[[M_TILE, K_ONE_BLOCK], pl.INT8],
        w: pl.Tensor[[N_TILE, K_ONE_BLOCK], pl.INT8],
        out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
    ) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="narrowed_split"):
            xk = pl.slice(x, [M_TILE, K_ONE_BLOCK], [0, 0], valid_shape=[valid_rows, K_ONE_BLOCK])
            acc = pl.matmul(xk, w, b_trans=True, out_dtype=pl.INT32)
            for lane in pl.split_aiv(2, mode=mode):
                shard = pl.aiv_shard(acc)
                row = lane * row_step
                col = lane * col_step
                out[row : row + (M_TILE - row_step), col : col + (N_TILE - col_step)] = pl.cast(
                    shard, target_type=pl.FP32, mode="none"
                )
        return out

    return narrowed_acc_explicit_split


@pl.jit
def narrowed_acc_explicit_split(
    x: pl.Tensor[[M_TILE, K_ONE_BLOCK], pl.INT8],
    w: pl.Tensor[[N_TILE, K_ONE_BLOCK], pl.INT8],
    out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
    """The same crossing through a ``pl.split_aiv`` region, rows reaching lane 0 only."""
    half = M_TILE // 2
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="narrowed_region"):
        xk = pl.slice(x, [M_TILE, K_ONE_BLOCK], [0, 0], valid_shape=[VALID_ROWS, K_ONE_BLOCK])
        acc = pl.matmul(xk, w, b_trans=True, out_dtype=pl.INT32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            out[aiv_id * half : aiv_id * half + half, :] = pl.cast(shard, target_type=pl.FP32, mode="none")
    return out


@pl.jit
def narrowed_acc_explicit_runtime(
    x: pl.Tensor[[M_TILE, K_ONE_BLOCK], pl.INT8],
    w: pl.Tensor[[N_TILE, K_ONE_BLOCK], pl.INT8],
    valid_rows: pl.Tensor[[1], pl.INT32],
    out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
    """An explicit split crossing whose row extent is only known at runtime."""
    n = pl.tensor.read(valid_rows, [0])
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="narrowed_split_runtime",
    ):
        xk = pl.slice(x, [M_TILE, K_ONE_BLOCK], [0, 0], valid_shape=[n, K_ONE_BLOCK])
        acc = pl.matmul(xk, w, b_trans=True, out_dtype=pl.INT32)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            row = lane * (M_TILE // 2)
            out[row : row + M_TILE // 2, :] = pl.cast(shard, target_type=pl.FP32, mode="none")
    return out


@pl.jit
def narrowed_acc_region_runtime(
    x: pl.Tensor[[M_TILE, K_ONE_BLOCK], pl.INT8],
    w: pl.Tensor[[N_TILE, K_ONE_BLOCK], pl.INT8],
    valid_rows: pl.Tensor[[1], pl.INT32],
    out: pl.InOut[pl.Tensor[[M_TILE, N_TILE], pl.FP32]],
) -> pl.Tensor[[M_TILE, N_TILE], pl.FP32]:
    """The runtime-extent crossing through a ``pl.split_aiv`` region."""
    half = M_TILE // 2
    n = pl.tensor.read(valid_rows, [0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="narrowed_region_runtime"):
        xk = pl.slice(x, [M_TILE, K_ONE_BLOCK], [0, 0], valid_shape=[n, K_ONE_BLOCK])
        acc = pl.matmul(xk, w, b_trans=True, out_dtype=pl.INT32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            out[aiv_id * half : aiv_id * half + half, :] = pl.cast(shard, target_type=pl.FP32, mode="none")
    return out


@pl.jit
def narrowed_acc_explicit_runtime_row_sum(
    x: pl.Tensor[[M_TILE, K_ONE_BLOCK], pl.INT8],
    w: pl.Tensor[[N_TILE, K_ONE_BLOCK], pl.INT8],
    valid_rows: pl.Tensor[[1], pl.INT32],
    out: pl.InOut[pl.Tensor[[M_TILE, 1], pl.FP32]],
) -> pl.Tensor[[M_TILE, 1], pl.FP32]:
    """A row reduction after the runtime-extent crossing: its result shape differs from the shard's."""
    n = pl.tensor.read(valid_rows, [0])
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="narrowed_split_row_sum",
    ):
        xk = pl.slice(x, [M_TILE, K_ONE_BLOCK], [0, 0], valid_shape=[n, K_ONE_BLOCK])
        acc = pl.matmul(xk, w, b_trans=True, out_dtype=pl.INT32)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            row = lane * (M_TILE // 2)
            out[row : row + M_TILE // 2, :] = pl.row_sum(pl.cast(shard, target_type=pl.FP32, mode="none"))
    return out


@pl.jit
def narrowed_acc_explicit_runtime_two_stores(
    x: pl.Tensor[[M_TILE, K_ONE_BLOCK], pl.INT8],
    w: pl.Tensor[[N_TILE, K_ONE_BLOCK], pl.INT8],
    valid_rows: pl.Tensor[[1], pl.INT32],
    out: pl.InOut[pl.Tensor[[M_TILE, 2 * N_TILE], pl.FP32]],
) -> pl.Tensor[[M_TILE, 2 * N_TILE], pl.FP32]:
    """Two slice writes into one tensor: the first store's result is the second's destination."""
    n = pl.tensor.read(valid_rows, [0])
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="narrowed_split_two_stores",
    ):
        xk = pl.slice(x, [M_TILE, K_ONE_BLOCK], [0, 0], valid_shape=[n, K_ONE_BLOCK])
        acc = pl.matmul(xk, w, b_trans=True, out_dtype=pl.INT32)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            row = lane * (M_TILE // 2)
            f = pl.cast(shard, target_type=pl.FP32, mode="none")
            out[row : row + M_TILE // 2, 0:N_TILE] = f
            out[row : row + M_TILE // 2, N_TILE : 2 * N_TILE] = pl.add(f, f)
    return out


def _split_operands(bound: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded INT8 ``x`` and ``w`` over one K block, with values in ``(-bound, bound)``."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(1 - bound, bound, (M_TILE, K_ONE_BLOCK), dtype=torch.int32, generator=generator)
    w = torch.randint(1 - bound, bound, (N_TILE, K_ONE_BLOCK), dtype=torch.int32, generator=generator)
    return x.to(torch.int8), w.to(torch.int8)


def _split_case(kernel, valid_rows: int, name: str, *, runtime_extent: bool = False):
    """``out[:valid_rows] = x @ w.T``; the rows past it keep their zero init.

    ``out`` is ``InOut`` so that init reaches the device: the kernel writes only
    the valid rows, and the rows past them must come back untouched. With
    ``runtime_extent`` the kernel reads ``valid_rows`` from a tensor instead of
    closing over it.

    INT8 x INT8 over one 256-wide K block is bounded by 256 * 127 * 127 < 2**24,
    so the FP32 result is exact and compared without tolerance.
    """
    x, w = _split_operands(128)
    expected = torch.zeros(M_TILE, N_TILE, dtype=torch.float32)
    expected[:valid_rows] = (x[:valid_rows].int() @ w.int().T).float()
    extent = [torch.tensor([valid_rows], dtype=torch.int32)] if runtime_extent else []
    return st.case(
        kernel,
        x,
        w,
        *extent,
        torch.zeros(M_TILE, N_TILE, dtype=torch.float32),
        name=name,
        golden=lambda _: expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.platforms("a2a3")
@st.cases(
    # Both lanes are fully valid; static partial rows use the explicit box partition.
    _split_case(
        _explicit_split_epilogue(pl.SplitMode.UP_DOWN, M_TILE), M_TILE, "narrowed_acc_explicit_split_ud_full"
    ),
    # The split is on the columns; the narrowed rows ride along on both lanes.
    _split_case(
        _explicit_split_epilogue(pl.SplitMode.LEFT_RIGHT, 40), 40, "narrowed_acc_explicit_split_lr_v40"
    ),
    _split_case(narrowed_acc_explicit_split, VALID_ROWS, "narrowed_acc_explicit_split_ud_v16"),
)
def test_vector_epilogue_across_a_split_boundary(case_run):
    """A compact accumulator crosses a split C2V boundary at its own row extent."""
    case_run.assert_passed()


@pytest.mark.platforms("a2a3")
@st.cases(
    # Past the box half: lane 1 holds rows 32..39, which it used to read from rows 8..15.
    _split_case(narrowed_acc_explicit_runtime, 40, "narrowed_acc_explicit_runtime_v40", runtime_extent=True),
    # Within the box half: lane 1 is empty, so its store must be skipped.
    _split_case(narrowed_acc_explicit_runtime, 16, "narrowed_acc_explicit_runtime_v16", runtime_extent=True),
    _split_case(narrowed_acc_region_runtime, 40, "narrowed_acc_region_runtime_v40", runtime_extent=True),
)
def test_vector_epilogue_across_a_split_boundary_at_a_runtime_extent(case_run):
    """A runtime row extent pops the full per-lane box and carries the extent on its consumers."""
    case_run.assert_passed()


@pytest.mark.parametrize(
    "kernel, message",
    [
        (narrowed_acc_explicit_runtime_row_sum, "reshapes the logical valid region"),
        (narrowed_acc_explicit_runtime_two_stores, "store whose result feeds another store"),
    ],
)
def test_explicit_runtime_split_reports_unsupported_consumers(kernel, message):
    """Retired AUTO forms must respect the existing explicit-boundary admission rules."""
    with pytest.raises(ValueError, match=message):
        kernel.lower(config=RunConfig(platform="a2a3"))


@pytest.mark.parametrize("valid_rows", [33, 40])
def test_explicit_static_split_does_not_rebalance_ragged_rows(valid_rows):
    """The authored box partition is preserved instead of rebalancing its valid rows."""
    kernel = _explicit_split_epilogue(pl.SplitMode.UP_DOWN, valid_rows)
    with pytest.raises(ValueError, match="leaves the two AIV lanes"):
        kernel.lower(config=RunConfig(platform="a2a3"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
