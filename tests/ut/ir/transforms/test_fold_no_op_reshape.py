# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for FoldNoOpReshape pass.

The pass rewrites ``lhs = tile.reshape(rhs, shape)`` AssignStmts into plain
``lhs = rhs`` whenever both sides share the same MemRef root and produce
identical TileBufSignatures, or when neither side owns a MemRef (a cross-core
tpop result or a view off one) and the two tile types are structurally equal.
PTO codegen previously dropped emission of such reshapes via a peephole;
folding into the IR makes codegen 1:1.

FoldNoOpReshape runs late in the pipeline (#38), after AllocateMemoryAddr
(#37). The prerequisite passes below mirror that ordering so the IR carries
the memrefs / allocated addresses the pass operates on. Tests follow the
Before/Expected pattern: ``_run_prereqs_and_fold`` runs the prereqs plus the
fold pass on ``Before``; ``_run_prereqs_only`` runs the prereqs alone on the
hand-written ``Expected`` (or on ``Before`` itself for no-op scenarios).
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend910B backend; the MemoryReuse hazard guard needs one."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


_PREREQS = (
    passes.convert_to_ssa,
    passes.outline_incore_scopes,
    passes.flatten_tile_nd_to_2d,
    passes.infer_tile_memory_space,
    passes.init_mem_ref,
    passes.memory_reuse,
    passes.allocate_memory_addr,
)


def _run_prereqs_only(program: ir.Program) -> ir.Program:
    """Run the pre-required passes without FoldNoOpReshape."""
    pipeline = passes.PassPipeline()
    for make_pass in _PREREQS:
        pipeline.add_pass(make_pass())
    return pipeline.run(program)


def _run_prereqs_and_fold(program: ir.Program) -> ir.Program:
    """Run the pre-required passes, then FoldNoOpReshape."""
    pipeline = passes.PassPipeline()
    for make_pass in _PREREQS:
        pipeline.add_pass(make_pass())
    pipeline.add_pass(passes.fold_no_op_reshape())
    return pipeline.run(program)


class TestFoldNoOpReshape:
    def test_noop_reshape_is_folded(self):
        """A reshape that preserves shape and shares MemRef must be folded out."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                # Identical shape + same MemRef after MemoryReuse → no-op reshape.
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(tile_a, [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                # FoldNoOpReshape rewrites the no-op reshape into a Var-to-Var alias.
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = tile_a
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_b, [0, 0], output)
                return result

        After = _run_prereqs_and_fold(Before)
        ExpectedIR = _run_prereqs_only(Expected)
        ir.assert_structural_equal(After, ExpectedIR)

    def test_genuine_reshape_kept(self):
        """A reshape that changes physical shape must NOT be folded."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                reshaped: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(tile_a, [4096, 1])
                flat: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(reshaped, [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(flat, [0, 0], output)
                return result

        # Physical-shape changing reshapes ([64,64] <-> [4096,1]) must remain:
        # FoldNoOpReshape is a no-op here, so After equals the prereq-only IR.
        After = _run_prereqs_and_fold(Before)
        ExpectedIR = _run_prereqs_only(Before)
        ir.assert_structural_equal(After, ExpectedIR)

    def test_same_allocation_different_window_is_kept(self):
        """Equal signatures do not make sibling subranges the same value.

        Capacity-overflow reuse can give two tiles one allocation base while
        keeping them at distinct byte offsets. Folding this reshape would turn
        a real PTO operation into an alias of the wrong half of the buffer.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                input_a: pl.Tensor[[16, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                tile_a: pl.Tile[[16, 64], pl.FP32, pl.MemRef("shared", 0, 4096), pl.MemorySpace.Vec] = (
                    pl.tile.load(input_a, [0, 0], [16, 64])
                )
                tile_b: pl.Tile[[16, 64], pl.FP32, pl.MemRef("shared", 4096, 4096), pl.MemorySpace.Vec] = (
                    pl.tile.reshape(tile_a, [16, 64])
                )
                result: pl.Tensor[[16, 64], pl.FP32] = pl.tile.store(tile_b, [0, 0], output)
                return result

        after = passes.fold_no_op_reshape()(Before)
        ir.assert_structural_equal(after, Before)

    def test_buffer_less_identity_reshape_is_folded(self):
        """An identity reshape of a MemRef-less tpop result folds to an alias.

        The popped tile owns no buffer, so there is no MemRef to compare; equal
        tile types prove the reshape is a no-op instead. Folding is what keeps the
        runtime valid extent: a surviving reshape lowers to a ``pto.treshape``
        view, which has no operands to carry ``vr``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main(
                self,
                vr: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                t: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = (
                    pl.tile.tpop_from_aic(split=0)
                )
                r: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = (
                    pl.tile.reshape(t, [16, 128])
                )
                pl.system.tfree_to_aic(t, split=0)
                result: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(r, [0, 0], output)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def main(
                self,
                vr: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                t: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = (
                    pl.tile.tpop_from_aic(split=0)
                )
                r: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = t
                pl.system.tfree_to_aic(t, split=0)
                result: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(r, [0, 0], output)
                return result

        After = _run_prereqs_and_fold(Before)
        ExpectedIR = _run_prereqs_only(Expected)
        ir.assert_structural_equal(After, ExpectedIR)

    def test_buffer_less_reshape_with_a_different_valid_extent_is_kept(self):
        """Equal signatures do not prove a MemRef-less LHS has the source's extent.

        ``TileBufSignature`` records any symbolic valid dim only as "dynamic", so
        ``[vr, 128]`` and ``[vc, 128]`` compare equal there. Aliasing ``r`` to
        ``t`` would hand every consumer ``vr`` rows where the IR says ``vc``.

        The pass is applied directly: ConvertToSSA re-deduces ``r``'s type from
        the reshape call, which would erase the hand-written ``vc``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main(
                self,
                vr: pl.Scalar[pl.INDEX],
                vc: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                t: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vr, 128])] = (
                    pl.tile.tpop_from_aic(split=0)
                )
                r: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[vc, 128])] = (
                    pl.tile.reshape(t, [16, 128])
                )
                pl.system.tfree_to_aic(t, split=0)
                result: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(r, [0, 0], output)
                return result

        after = passes.fold_no_op_reshape()(Before)
        ir.assert_structural_equal(after, Before)

    def test_buffer_less_shape_change_is_kept(self):
        """A reshape of a MemRef-less tile that changes its shape is a real view."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main(
                self,
                output: pl.Out[pl.Tensor[[128, 16], pl.FP32]],
            ) -> pl.Tensor[[128, 16], pl.FP32]:
                t: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
                r: pl.Tile[[128, 16], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t, [128, 16])
                pl.system.tfree_to_aic(t, split=0)
                result: pl.Tensor[[128, 16], pl.FP32] = pl.tile.store(r, [0, 0], output)
                return result

        After = _run_prereqs_and_fold(Before)
        ExpectedIR = _run_prereqs_only(Before)
        ir.assert_structural_equal(After, ExpectedIR)

    def test_pass_runs_without_error_on_simple_kernel(self):
        """Smoke test: pass should not crash on a kernel without trivial reshapes."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0, 0], [64, 64])
                y: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(t, t)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(y, [0, 0], output)
                return result

        # No tile.reshape in the input — FoldNoOpReshape is a no-op, so After
        # equals the prereq-only IR.
        After = _run_prereqs_and_fold(Before)
        ExpectedIR = _run_prereqs_only(Before)
        ir.assert_structural_equal(After, ExpectedIR)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
