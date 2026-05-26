# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for LowerTransposeLoadParamLayout pass (RFC #1300 P6).

The pass leaves InCore parameter signatures untouched and instead prepends a
``b_dn = tensor.as_layout(b, layout=DN)`` AssignStmt at the top of the InCore
body for each param ``b`` loaded via ``tile.load(transpose=True)``. Body uses
of ``b`` are substituted with ``b_dn`` (which has the canonical
``[..., b_dim, a_dim] DN`` view per RFC §3.3 + §4.2), and the matching
``tile.load`` calls have their ``offsets`` / ``shapes`` / ``valid_shapes``
trailing pair swapped while ``transpose=True`` is flipped to
``transpose=False``. Non-InCore (orch) call sites are not touched — they pass
their original ND args straight through to the kernel.

``tensor.as_layout`` has a thin ``pl.tensor.as_layout`` wrapper (internal API);
the printer emits it as ``pl.tensor.as_layout(...)`` and the parser accepts that
form, so the post-pass IR round-trips through print/parse. Tests therefore use
the standard Before/Expected ``@pl.program`` pattern: drive the pass with a
``Before`` program and compare the result against an ``Expected`` program that
spells out the post-pass IR (the body-prepended ``as_layout`` binding, the
rewritten ``tile.load`` window, and ``transpose=False``).
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


class TestBTransposePromotesParam:
    """``C = A @ B^T`` with B loaded via ``transpose=True`` — param promoted to DN."""

    def test_btranspose_basic(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                c: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def matmul_incore(
                a: pl.Tensor[[64, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                # Param ``b`` is untouched; the body prepends a DN view binding.
                b_dn_view: pl.Tensor[
                    [128, 32], pl.FP32, pl.TensorView(stride=[1, 128], layout=pl.TensorLayout.DN)
                ] = pl.tensor.as_layout(b, layout=pl.TensorLayout.DN)
                tile_a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    a, [0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                # tile.load now reads from b_dn_view: window swapped, transpose=False.
                tile_b: pl.Tile[
                    [128, 32],
                    pl.FP32,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.tile.load(
                    b_dn_view, [0, 0], [128, 32], [128, 32], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_a_l0a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Left] = pl.tile.move(
                    tile_a, target_memory=pl.Mem.Left
                )
                tile_b_l0b: pl.Tile[[128, 32], pl.FP32, pl.Mem.Right] = pl.tile.move(
                    tile_b, target_memory=pl.Mem.Right
                )
                tile_c: pl.Tile[[64, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a_l0a, tile_b_l0b)
                c_store: pl.Tensor[[64, 32], pl.FP32] = pl.tile.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def orchestrator(
                self, a: pl.Tensor[[64, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                # Orch is untouched — its call site passes ``b`` straight through.
                c: pl.Tensor[[64, 32], pl.FP32] = pl.tensor.create(
                    [64, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                c_result: pl.Tensor[[64, 32], pl.FP32] = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_btranspose_non_square(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[32, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 32], pl.FP32]],
            ) -> pl.Tensor[[128, 32], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [32, 64], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 64], pl.FP32], b: pl.Tensor[[32, 64], pl.FP32]
            ) -> pl.Tensor[[128, 32], pl.FP32]:
                c: pl.Tensor[[128, 32], pl.FP32] = pl.create_tensor([128, 32], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def matmul_incore(
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[32, 64], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 32], pl.FP32]],
            ) -> pl.Tensor[[128, 32], pl.FP32]:
                # Body prepends b_dn_view = as_layout(b, DN); LHS carries [64, 32] DN.
                b_dn_view: pl.Tensor[
                    [64, 32], pl.FP32, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)
                ] = pl.tensor.as_layout(b, layout=pl.TensorLayout.DN)
                tile_a: pl.Tile[[128, 64], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    a, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_b: pl.Tile[
                    [64, 32],
                    pl.FP32,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.tile.load(
                    b_dn_view, [0, 0], [64, 32], [64, 32], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_a_l0a: pl.Tile[[128, 64], pl.FP32, pl.Mem.Left] = pl.tile.move(
                    tile_a, target_memory=pl.Mem.Left
                )
                tile_b_l0b: pl.Tile[[64, 32], pl.FP32, pl.Mem.Right] = pl.tile.move(
                    tile_b, target_memory=pl.Mem.Right
                )
                tile_c: pl.Tile[[128, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a_l0a, tile_b_l0b)
                c_store: pl.Tensor[[128, 32], pl.FP32] = pl.tile.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def orchestrator(
                self, a: pl.Tensor[[128, 64], pl.FP32], b: pl.Tensor[[32, 64], pl.FP32]
            ) -> pl.Tensor[[128, 32], pl.FP32]:
                c: pl.Tensor[[128, 32], pl.FP32] = pl.tensor.create(
                    [128, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                c_result: pl.Tensor[[128, 32], pl.FP32] = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestATransposePromotesParam:
    """``C = A^T @ B`` — A param promoted to canonical DN."""

    def test_atranspose_basic(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[128, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [128, 32], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 64], pl.FP32], b: pl.Tensor[[128, 32], pl.FP32]
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                c: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def matmul_incore(
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[128, 32], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                # Only ``a`` is promoted; ``b`` (no transpose load) is untouched.
                a_dn_view: pl.Tensor[
                    [64, 128], pl.FP32, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)
                ] = pl.tensor.as_layout(a, layout=pl.TensorLayout.DN)
                tile_a: pl.Tile[
                    [64, 128],
                    pl.FP32,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.tile.load(
                    a_dn_view, [0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_b: pl.Tile[[128, 32], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    b, [0, 0], [128, 32], [128, 32], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_a_l0a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Left] = pl.tile.move(
                    tile_a, target_memory=pl.Mem.Left
                )
                tile_b_l0b: pl.Tile[[128, 32], pl.FP32, pl.Mem.Right] = pl.tile.move(
                    tile_b, target_memory=pl.Mem.Right
                )
                tile_c: pl.Tile[[64, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a_l0a, tile_b_l0b)
                c_store: pl.Tensor[[64, 32], pl.FP32] = pl.tile.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def orchestrator(
                self, a: pl.Tensor[[128, 64], pl.FP32], b: pl.Tensor[[128, 32], pl.FP32]
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                c: pl.Tensor[[64, 32], pl.FP32] = pl.tensor.create(
                    [64, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                c_result: pl.Tensor[[64, 32], pl.FP32] = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestABTransposePromotesBothParams:
    """``C = A^T @ B^T`` — both params promoted."""

    def test_abtranspose_basic(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(b, [0, 0], [32, 128], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[128, 64], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                c: pl.Tensor[[64, 32], pl.FP32] = pl.create_tensor([64, 32], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def matmul_incore(
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                # Body prepends one as_layout binding per promoted param.
                a_dn_view: pl.Tensor[
                    [64, 128], pl.FP32, pl.TensorView(stride=[1, 64], layout=pl.TensorLayout.DN)
                ] = pl.tensor.as_layout(a, layout=pl.TensorLayout.DN)
                b_dn_view: pl.Tensor[
                    [128, 32], pl.FP32, pl.TensorView(stride=[1, 128], layout=pl.TensorLayout.DN)
                ] = pl.tensor.as_layout(b, layout=pl.TensorLayout.DN)
                tile_a: pl.Tile[
                    [64, 128],
                    pl.FP32,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.tile.load(
                    a_dn_view, [0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_b: pl.Tile[
                    [128, 32],
                    pl.FP32,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.tile.load(
                    b_dn_view, [0, 0], [128, 32], [128, 32], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_a_l0a: pl.Tile[[64, 128], pl.FP32, pl.Mem.Left] = pl.tile.move(
                    tile_a, target_memory=pl.Mem.Left
                )
                tile_b_l0b: pl.Tile[[128, 32], pl.FP32, pl.Mem.Right] = pl.tile.move(
                    tile_b, target_memory=pl.Mem.Right
                )
                tile_c: pl.Tile[[64, 32], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a_l0a, tile_b_l0b)
                c_store: pl.Tensor[[64, 32], pl.FP32] = pl.tile.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def orchestrator(
                self, a: pl.Tensor[[128, 64], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                c: pl.Tensor[[64, 32], pl.FP32] = pl.tensor.create(
                    [64, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                c_result: pl.Tensor[[64, 32], pl.FP32] = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestNoOpCases:
    """Pass is a no-op when no parameter needs promotion."""

    def test_no_transpose_unchanged(self):
        M, K, N = 64, 128, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[M, K], pl.FP32], b: pl.Tensor[[K, N], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Before)


class TestStridedParamFlipsCorrectly:
    """Regression for #1212 / #1213: when an InCore param's TensorView carries
    parent-derived strides (a sliced sub-view of a larger tensor — set up here
    via an explicit ``pl.TensorView(stride=...)`` annotation, the same shape
    ``SliceInputStridesOptimizer`` produces in the default pipeline), the body-
    prepended ``tensor.as_layout`` flip must propagate those strides through
    the §4.2 canonical-pair swap. The output DN view must carry the swapped
    parent stride, not the slice-shape-derived packed stride — otherwise PTOAS
    walks rows at the wrong stride and silently miscompiles."""

    def test_strided_nd_param_flips_to_strided_dn(self):
        """Parent buffer is ``[T, K_parent] ND`` with row stride ``K_parent``;
        a slice ``[T, K_slice]`` annotated with that parent stride must flip
        to ``[K_slice, T] DN`` with stride ``[1, K_parent]`` — preserving the
        parent's row stride at the trailing slot.
        """
        T, K_slice, K_parent = 16, 512, 16384

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[T, K_slice], pl.FP32],
                # Slice of a `[T, K_parent]` parent: strided-ND with the
                # parent's row stride preserved at the outer slot.
                b_slice: pl.Tensor[  # noqa: E501
                    [T, K_slice],
                    pl.FP32,
                    pl.TensorView(stride=[K_parent, 1], layout=pl.TensorLayout.ND),
                ],
                c: pl.Out[pl.Tensor[[T, T], pl.FP32]],
            ) -> pl.Tensor[[T, T], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [T, K_slice], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(
                    b_slice, [0, 0], [T, K_slice], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_a_l0 = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0 = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0, tile_b_l0)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[T, K_slice], pl.FP32],
                b_slice: pl.Tensor[  # noqa: E501
                    [T, K_slice],
                    pl.FP32,
                    pl.TensorView(stride=[K_parent, 1], layout=pl.TensorLayout.ND),
                ],
            ) -> pl.Tensor[[T, T], pl.FP32]:
                c: pl.Tensor[[T, T], pl.FP32] = pl.create_tensor([T, T], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b_slice, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def matmul_incore(
                a: pl.Tensor[[16, 512], pl.FP32],
                # Param is untouched: still [T, K_slice] ND with parent's stride.
                b_slice: pl.Tensor[  # noqa: E501
                    [16, 512],
                    pl.FP32,
                    pl.TensorView(stride=[16384, 1], layout=pl.TensorLayout.ND),
                ],
                c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                # Critical: the DN view inherits the parent's stride,
                # trailing-pair-swapped to [1, K_parent] — NOT slice-shape-derived
                # packed strides. See #1212 / #1213.
                b_slice_dn_view: pl.Tensor[  # noqa: E501
                    [512, 16],
                    pl.FP32,
                    pl.TensorView(stride=[1, 16384], layout=pl.TensorLayout.DN),
                ] = pl.tensor.as_layout(b_slice, layout=pl.TensorLayout.DN)
                tile_a: pl.Tile[[16, 512], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    a, [0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_b: pl.Tile[
                    [512, 16],
                    pl.FP32,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.tile.load(
                    b_slice_dn_view, [0, 0], [512, 16], [512, 16], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_a_l0: pl.Tile[[16, 512], pl.FP32, pl.Mem.Left] = pl.tile.move(
                    tile_a, target_memory=pl.Mem.Left
                )
                tile_b_l0: pl.Tile[[512, 16], pl.FP32, pl.Mem.Right] = pl.tile.move(
                    tile_b, target_memory=pl.Mem.Right
                )
                tile_c: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a_l0, tile_b_l0)
                c_store: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def orchestrator(
                self,
                a: pl.Tensor[[16, 512], pl.FP32],
                b_slice: pl.Tensor[  # noqa: E501
                    [16, 512],
                    pl.FP32,
                    pl.TensorView(stride=[16384, 1], layout=pl.TensorLayout.ND),
                ],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                c: pl.Tensor[[16, 16], pl.FP32] = pl.tensor.create(
                    [16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                c_result: pl.Tensor[[16, 16], pl.FP32] = self.matmul_incore(a, b_slice, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestMixedUseAccepted:
    """A param loaded with both transpose=True and transpose=False coexists as two
    cube loads: the transposed load is redirected to a body-local DN view, while
    the non-transposed load keeps reading the original ND param. This unblocks
    FlashAttention-style fused QK+PV on a shared KV tensor (issue #1532)."""

    def test_mixed_transpose_modes_split_into_two_loads(self):
        M, K, N = 64, 128, 32

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_incore(
                self,
                a: pl.Tensor[[N, K], pl.FP32],
                b: pl.Tensor[[N, K], pl.FP32],
                c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [N, K], target_memory=pl.MemorySpace.Mat, transpose=True)
                tile_b = pl.load(a, [0, 0], [N, K], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0a, tile_b_l0b)
                c_store = pl.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[N, K], pl.FP32], b: pl.Tensor[[N, K], pl.FP32]
            ) -> pl.Tensor[[M, N], pl.FP32]:
                c: pl.Tensor[[M, N], pl.FP32] = pl.create_tensor([M, N], dtype=pl.FP32)
                c_result = self.matmul_incore(a, b, c)
                return c_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def matmul_incore(
                a: pl.Tensor[[32, 128], pl.FP32],
                b: pl.Tensor[[32, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                # Body prepends a DN view binding for ``a`` (only ``a`` is promoted —
                # ``b`` has no transposed load). ``tile_a`` redirects to ``a_dn_view``
                # with the load window swapped to canonical DN coords; ``tile_b``
                # still reads ``a`` directly with ``transpose=False``.
                a_dn_view: pl.Tensor[
                    [128, 32], pl.FP32, pl.TensorView(stride=[1, 128], layout=pl.TensorLayout.DN)
                ] = pl.tensor.as_layout(a, layout=pl.TensorLayout.DN)
                tile_a: pl.Tile[
                    [128, 32],
                    pl.FP32,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.tile.load(
                    a_dn_view, [0, 0], [128, 32], [128, 32], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_b: pl.Tile[[32, 128], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    a, [0, 0], [32, 128], [32, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_a_l0a: pl.Tile[[128, 32], pl.FP32, pl.Mem.Left] = pl.tile.move(
                    tile_a, target_memory=pl.Mem.Left
                )
                tile_b_l0b: pl.Tile[[32, 128], pl.FP32, pl.Mem.Right] = pl.tile.move(
                    tile_b, target_memory=pl.Mem.Right
                )
                tile_c: pl.Tile[[128, 128], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a_l0a, tile_b_l0b)
                c_store: pl.Tensor[[64, 32], pl.FP32] = pl.tile.store(tile_c, [0, 0], c)
                return c_store

            @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def orchestrator(
                self, a: pl.Tensor[[32, 128], pl.FP32], b: pl.Tensor[[32, 128], pl.FP32]
            ) -> pl.Tensor[[64, 32], pl.FP32]:
                c: pl.Tensor[[64, 32], pl.FP32] = pl.tensor.create(
                    [64, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                c_result: pl.Tensor[[64, 32], pl.FP32] = self.matmul_incore(a, b, c)
                return c_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Expected)


class TestPartialLoadPromotion:
    """A param with a partial-window transpose load: param shape swap is based on the
    full TensorType shape, not the load window."""

    def test_partial_load_square_tensor(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
                tile_k = pl.load(
                    key_cache, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat, transpose=True
                )
                tile_a_l0 = pl.move(tile_a, target_memory=pl.MemorySpace.Left)
                tile_k_l0 = pl.move(tile_k, target_memory=pl.MemorySpace.Right)
                tile_c = pl.matmul(tile_a_l0, tile_k_l0)
                out_store = pl.store(tile_c, [0, 0], out)
                return out_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
                out_result = self.kernel(a, key_cache, out)
                return out_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                # Shape stays [128, 128] (square) but layout flips to DN — the
                # trailing-pair swap on the full TensorType shape is identity here.
                key_cache_dn_view: pl.Tensor[
                    [128, 128], pl.BF16, pl.TensorView(stride=[1, 128], layout=pl.TensorLayout.DN)
                ] = pl.tensor.as_layout(key_cache, layout=pl.TensorLayout.DN)
                tile_a: pl.Tile[[64, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    a, [0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat, transpose=False
                )
                # tile.load reads from the binding's LHS, load window swapped
                # ([64, 128] -> [128, 64]) and transpose=False.
                tile_k: pl.Tile[
                    [128, 64],
                    pl.BF16,
                    pl.Mem.Mat,
                    pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major),
                ] = pl.tile.load(
                    key_cache_dn_view, [0, 0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False
                )
                tile_a_l0: pl.Tile[[64, 128], pl.BF16, pl.Mem.Left] = pl.tile.move(
                    tile_a, target_memory=pl.Mem.Left
                )
                tile_k_l0: pl.Tile[[128, 64], pl.BF16, pl.Mem.Right] = pl.tile.move(
                    tile_k, target_memory=pl.Mem.Right
                )
                tile_c: pl.Tile[[64, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a_l0, tile_k_l0)
                out_store: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(tile_c, [0, 0], out)
                return out_store

            @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 128], pl.BF16],
                key_cache: pl.Tensor[[128, 128], pl.BF16],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.create(
                    [64, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                out_result: pl.Tensor[[64, 64], pl.FP32] = self.kernel(a, key_cache, out)
                return out_result

        After = passes.lower_transpose_load_param_layout()(Before)
        ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
