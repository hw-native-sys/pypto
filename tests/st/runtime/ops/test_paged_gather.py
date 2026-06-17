# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end (on-device) tests for ``pl.paged_gather``.

``pl.paged_gather`` gathers scattered rows of a paged KV pool directly into an
on-chip buffer (L1 / ``Mem.Mat`` by default, or UB / ``Mem.Vec``) using a
fully-scalar per-row ``GM -> on-chip`` DMA loop on the Cube core. The physical
row for each logical index is resolved through the paged ``block_table``::

    phys = block_table[idx // block_size] * block_size + idx % block_size
    out[i, :] = src[phys, col_off : col_off + size]

These tests write the gathered tile back to GM and compare against a torch
paged-gather golden, validating the scalar index translation + GM->L1 loads
numerically on real hardware.

**Why these inputs are discriminating.** ``src[r, c] = r`` encodes each row's
physical index into its values (FP16-exact for r <= 255), so ``out[i, :]`` must
equal the constant ``phys_i``. The ``block_table`` is a non-identity reverse
permutation and the indices spread across blocks/offsets, so an implementation
that skipped the page-table lookup (``phys == idx``) or mis-computed the offset
would mismatch immediately.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# Paged KV pool geometry.
_BLOCK_SIZE = 16  # tokens per page block
_PHYS_BLOCKS = 16  # physical blocks in the pool
_POOL_ROWS = _BLOCK_SIZE * _PHYS_BLOCKS  # 256
_HIDDEN = 128
_NUM_IDX = 16  # gathered rows (== max_indices: fully-defined output)


def _make_src(torch_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """src[r, c] = r — row-id encoding so out[i, :] must equal phys_i."""
    rows = torch.arange(_POOL_ROWS, dtype=torch.float32).reshape(_POOL_ROWS, 1)
    return rows.expand(_POOL_ROWS, _HIDDEN).to(torch_dtype).contiguous()


def _make_block_table() -> torch.Tensor:
    """Non-identity reverse permutation: logical block b -> physical (PHYS_BLOCKS-1-b)."""
    return torch.tensor([_PHYS_BLOCKS - 1 - b for b in range(_PHYS_BLOCKS)], dtype=torch.int32).contiguous()


def _make_indices() -> torch.Tensor:
    """Logical token indices spread across blocks and intra-block offsets."""
    return torch.tensor([(i * 7 + 3) % _POOL_ROWS for i in range(_NUM_IDX)], dtype=torch.int32).contiguous()


def _paged_gather_golden(
    src: torch.Tensor, indices: torch.Tensor, block_table: torch.Tensor, block_size: int, size: int
) -> torch.Tensor:
    n = indices.shape[0]
    out = torch.empty(n, size, dtype=src.dtype)
    for i in range(n):
        li = int(indices[i].item())
        phys = int(block_table[li // block_size].item()) * block_size + (li % block_size)
        out[i, :] = src[phys, :size]
    return out


@pl.program
class PagedGatherMatProgram:
    """Gather paged KV rows into L1 (Mem.Mat), read them back via an identity
    matmul, and store the matmul result to GM.

    The gathered L1 tile carries the matmul-operand NZ (boxed) layout, so it
    cannot be pushed straight to GM — an NZ L1 tile has no cross-core producer
    pipe to the vector unit (ptoas: ``'pto.tpush' op tile type must map to a
    supported producer pipe``). Instead we verify the gather the way an L1
    operand is meant to be consumed: ``eye @ gathered``. ``eye`` is the
    ``[N, N]`` identity (A / left operand) and ``gathered`` is the ``[N, D]``
    right operand, so the product equals the gathered rows exactly (each
    identity row selects one gathered row; row ids <= 255 are FP16-exact). The
    matmul result is an Acc tile, which reaches GM through the supported
    Acc -> Vec -> GM path.
    """

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[_POOL_ROWS, _HIDDEN], pl.FP16],
        idx: pl.Tensor[[_NUM_IDX], pl.INT32],
        bt: pl.Tensor[[_PHYS_BLOCKS], pl.INT32],
        eye: pl.Tensor[[_NUM_IDX, _NUM_IDX], pl.FP16],
        output: pl.Out[pl.Tensor[[_NUM_IDX, _HIDDEN], pl.FP32]],
    ) -> pl.Tensor[[_NUM_IDX, _HIDDEN], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            gathered = pl.paged_gather(
                src,
                idx,
                bt,
                block_size=_BLOCK_SIZE,
                size=_HIDDEN,
                max_indices=_NUM_IDX,
                space=pl.MemorySpace.Mat,
                is_b_matrix=True,
            )
            result = pl.matmul(eye, gathered, out_dtype=pl.FP32)
            output = pl.assemble(output, result, [0, 0])
        return output


@pl.program
class PagedGatherVecProgram:
    """Gather paged KV rows into UB (Mem.Vec), then store to GM."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        src: pl.Tensor[[_POOL_ROWS, _HIDDEN], pl.FP16],
        idx: pl.Tensor[[_NUM_IDX], pl.INT32],
        bt: pl.Tensor[[_PHYS_BLOCKS], pl.INT32],
        output: pl.Out[pl.Tensor[[_NUM_IDX, _HIDDEN], pl.FP16]],
    ) -> pl.Tensor[[_NUM_IDX, _HIDDEN], pl.FP16]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.paged_gather(
                src,
                idx,
                bt,
                block_size=_BLOCK_SIZE,
                size=_HIDDEN,
                max_indices=_NUM_IDX,
                space=pl.MemorySpace.Vec,
            )
            output = pl.assemble(output, out, [0, 0])
        return output


class _PagedGatherBaseTestCase(PTOTestCase):
    __test__ = False

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [_POOL_ROWS, _HIDDEN], DataType.FP16, init_value=_make_src),
            TensorSpec("idx", [_NUM_IDX], DataType.INT32, init_value=_make_indices),
            TensorSpec("bt", [_PHYS_BLOCKS], DataType.INT32, init_value=_make_block_table),
            TensorSpec("output", [_NUM_IDX, _HIDDEN], DataType.FP16, is_output=True),
        ]

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = _paged_gather_golden(
            tensors["src"], tensors["idx"], tensors["bt"], _BLOCK_SIZE, _HIDDEN
        )


class PagedGatherMatTestCase(_PagedGatherBaseTestCase):
    def get_name(self) -> str:
        return "paged_gather_mat"

    def get_program(self) -> Any:
        return PagedGatherMatProgram

    def define_tensors(self) -> list[TensorSpec]:
        # The Mat path reads the gathered L1 tile back via ``eye @ gathered``
        # (see PagedGatherMatProgram), so it adds an identity operand and the
        # matmul result is FP32.
        return [
            TensorSpec("src", [_POOL_ROWS, _HIDDEN], DataType.FP16, init_value=_make_src),
            TensorSpec("idx", [_NUM_IDX], DataType.INT32, init_value=_make_indices),
            TensorSpec("bt", [_PHYS_BLOCKS], DataType.INT32, init_value=_make_block_table),
            TensorSpec(
                "eye",
                [_NUM_IDX, _NUM_IDX],
                DataType.FP16,
                init_value=lambda: torch.eye(_NUM_IDX, dtype=torch.float16),
            ),
            TensorSpec("output", [_NUM_IDX, _HIDDEN], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors, params=None):
        # eye @ gathered == gathered, so the golden is the gathered rows (FP32).
        tensors["output"][:] = _paged_gather_golden(
            tensors["src"], tensors["idx"], tensors["bt"], _BLOCK_SIZE, _HIDDEN
        ).to(torch.float32)


class PagedGatherVecTestCase(_PagedGatherBaseTestCase):
    def get_name(self) -> str:
        return "paged_gather_vec"

    def get_program(self) -> Any:
        return PagedGatherVecProgram


class TestPagedGather:
    """On-device paged gather into L1 / UB, validated against a torch golden."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_paged_gather_mat(self, test_runner, platform):
        result = test_runner.run(PagedGatherMatTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_paged_gather_vec(self, test_runner, platform):
        result = test_runner.run(PagedGatherVecTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
