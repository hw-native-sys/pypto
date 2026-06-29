# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SplitVectorKernel pass."""

import re

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir import op as ir_op
from pypto.ir.printer import python_print


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend950 backend before each test and reset afterward."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def _run_split_vector_kernel(program):
    """Run convert_to_ssa then split_vector_kernel.

    Runs under the conftest's default verification context (BEFORE_AND_AFTER
    property verification + print->parse roundtrip). SplitVectorKernel requires
    the ``MixedKernelExpanded`` property, so every fixture supplies the
    cross-core pipe scaffolding (``aic_initialize_pipe`` / ``aiv_initialize_pipe``
    + ``reserve_buffer`` / ``import_peer_buffer`` and a matching ``tfree_to_aic`` /
    ``tfree_to_aiv``) that ``ExpandMixedKernel`` would add in the real pipeline.
    The scaffolding is inert for SplitVectorKernel — it is passed through
    untouched — so the before/after contract still asserts exactly the split
    rewrite (mode inference, shape halving, offset adjustment). Round-trip
    correctness — including def-use closure on Var refs embedded inside type
    annotations — is exercised by the same roundtrip instrument.
    """
    ssa = passes.convert_to_ssa()(program)
    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.split_vector_kernel())
    return pipeline.run(ssa)


def _assert_split_matches_expected(before_program, expected_program):
    actual = _run_split_vector_kernel(before_program)
    ir.assert_structural_equal(actual, passes.convert_to_ssa()(expected_program))


class TestSplitVectorKernelNoSplitPassthrough:
    """SplitVectorKernel leaves non-split AIV functions untouched.

    After the convergence refactor, SplitVectorKernel no longer halves bodies:
    LowerAutoVectorSplit converts AUTO ``pl.split`` mixed InCore functions into
    the explicit ``split_aiv`` form upstream, and SplitVectorKernel only stamps
    attrs for those (see TestSplitVectorKernelExplicitSplitAivBypass). A function
    with no split mode (and no no-split dual-AIV marker) is passed through
    structurally unchanged. The per-op halving tests these classes used to hold
    moved to ``test_lower_auto_vector_split.py`` (they assert the same facts via
    the shared ``split_axis`` machinery, now reached through LowerAutoVectorSplit).
    """

    def test_no_split_when_none(self):
        """Functions with no split should not be modified."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main_aiv(
                self,
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                return out_0_store

        result = _run_split_vector_kernel(Before)
        ir.assert_structural_equal(result, passes.convert_to_ssa()(Before))

    def test_reshape_of_rank1_load_unchanged_when_no_split(self):
        """A rank-1 load + reshape is left untouched when the function has no split mode."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main_aiv(
                self,
                scale: pl.Tensor[[128], pl.FP32],
                data: pl.Tensor[[16, 128], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                scale_row: pl.Tile[[128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    scale, [0], [128], target_memory=pl.MemorySpace.Vec
                )
                scale_2d: pl.Tile[[1, 128], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(scale_row, [1, 128])
                prev: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 128], target_memory=pl.MemorySpace.Vec
                )
                result: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.col_expand_mul(prev, scale_2d)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(result, [0, 0], out_0)
                return out_0_store

        result = _run_split_vector_kernel(Before)
        ir.assert_structural_equal(result, passes.convert_to_ssa()(Before))


class TestSplitVectorKernelNoSplitA2A3:
    """Tests for Ascend910B no-split mixed-kernel dual-dispatch lowering."""

    def test_no_split_dual_dispatch_producer_replays_compute_and_tpush_on_lane1(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                a_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec
                )
                b_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec
                )
                summed: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(a_tile, b_tile)
                pl.tpush_to_aic(summed, split=0)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    a_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec
                    )
                    b_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec
                    )
                    summed: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(a_tile, b_tile)
                    pl.tpush_to_aic(summed, split=0)
                    return out
                else:
                    a_tile_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    b_tile_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    summed_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.add(a_tile_lane1, b_tile_lane1)
                    pl.tpush_to_aic(summed_lane1, split=0)
                    return out

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_dual_dispatch_rewrites_lane1_tile_load_to_create(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        span = ir.Span.unknown()
        zero = ir.ConstInt(0, pl.INDEX, span)
        dim = ir.ConstInt(16, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([dim, dim], span)
        valid_shapes = ir.MakeTuple([dim, dim], span)

        data = ir.Var("data", ir.TensorType([16, 16], pl.FP32), span)
        out = ir.Var("out", ir.TensorType([16, 16], pl.FP32), span)

        load_view = ir.TileView(valid_shape=[dim, dim])
        load_type = ir.TileType([16, 16], pl.FP32, None, load_view, ir.MemorySpace.Vec)
        loaded = ir.Var("loaded", load_type, span)
        # Canonical 4-operand tile.load ([tensor, offsets, shapes, valid_shapes])
        # with the full kwarg set the DSL/IR builder emits (target_memory +
        # transpose). Matching the canonical operand/kwarg arity is what lets the
        # hand-built body survive the print->parse roundtrip verifier.
        load_call = ir.Call(
            ir.Op("tile.load"),
            [data, offsets, shapes, valid_shapes],
            {"target_memory": ir.MemorySpace.Vec},
            load_type,
            span,
        )

        tpush_call = ir.Call(ir.Op("tile.tpush_to_aic"), [loaded], {"split": 0}, ir.UnknownType(), span)

        # Cross-core pipe scaffolding the MixedKernelExpanded property requires
        # for an AIV function that uses a V2C op (tpush_to_aic): a dominating
        # import_peer_buffer + aiv_initialize_pipe. Built via the IR op builders
        # so the Calls are canonical and survive the roundtrip verifier. This is
        # the same scaffolding ExpandMixedKernel injects in the real pipeline.
        peer_buf_call = ir_op.system.import_peer_buffer(
            name="v2c_slot_buffer", peer_func="main_aic", span=span
        )
        peer_buf = ir.Var("peer_buf", peer_buf_call.type, span)
        init_pipe_call = ir_op.system.aiv_initialize_pipe(
            v2c_consumer_buf=peer_buf, dir_mask=2, slot_size=512, span=span
        )

        body = ir.SeqStmts(
            [
                ir.AssignStmt(peer_buf, peer_buf_call, span),
                ir.EvalStmt(init_pipe_call, span),
                ir.AssignStmt(loaded, load_call, span),
                ir.EvalStmt(tpush_call, span),
                ir.ReturnStmt([out], span),
            ],
            span,
        )
        func = ir.Function(
            "main_aiv",
            [(data, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
            [out.type],
            body,
            span,
            ir.FunctionType.AIV,
            attrs={"dual_aiv_dispatch": True},
        )

        actual = _run_split_vector_kernel(ir.Program([func], "tile_load_program", span))
        printed = python_print(actual)

        assert "if subblock_idx == 0:" in printed
        assert printed.count("pl.tile.load(") == 1
        assert printed.count("pl.tile.create(") == 1
        assert printed.count("pl.tile.tpush_to_aic(") == 2
        assert re.search(
            r"loaded__ssa_v0_\d+: pl.Tile\[\[16, 16\], pl.FP32, pl.Mem.Vec, "
            r"pl.TileView\(valid_shape=\[0, 0\]\)\] = pl.tile.create",
            printed,
        )

    def test_no_split_dual_dispatch_rewrites_lane1_tile_slice_to_create(self):
        """Lane1 replay rewrites a producer ``tile.slice`` into ``tile.create``.

        A ``tile.slice`` is a pure view with no cross-core sync, so the replay
        lane only needs an empty tile of the slice's result shape. Forcing the
        slice's explicit ``valid_shape`` to a static 0 would emit a
        ``v_row=0, v_col=0`` subview that pto-isa cannot compile (no
        ``GetValidRow`` overload for a static mask of 0); the rewrite to
        ``tile.create`` yields a dynamic-valid empty tile instead (gh#1649).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        span = ir.Span.unknown()
        zero = ir.ConstInt(0, pl.INDEX, span)
        dim = ir.ConstInt(16, pl.INDEX, span)
        sub = ir.ConstInt(8, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([dim, dim], span)
        valid_shapes = ir.MakeTuple([dim, dim], span)
        slice_shape = ir.MakeTuple([dim, sub], span)

        data = ir.Var("data", ir.TensorType([16, 16], pl.FP32), span)
        out = ir.Var("out", ir.TensorType([16, 8], pl.FP32), span)

        load_type = ir.TileType(
            [16, 16], pl.FP32, None, ir.TileView(valid_shape=[dim, dim]), ir.MemorySpace.Vec
        )
        loaded = ir.Var("loaded", load_type, span)
        # Canonical 4-operand tile.load + full kwarg set (see the load->create
        # test above) so the hand-built body round-trips under verification. The
        # producer ``tile.slice`` below is already canonical: a 3-operand
        # ``[tile, shape, offset]`` slice (no explicit valid_shape) is exactly
        # what the DSL/IR builder emits, so it needs no padding.
        load_call = ir.Call(
            ir.Op("tile.load"),
            [data, offsets, shapes, valid_shapes],
            {"target_memory": ir.MemorySpace.Vec},
            load_type,
            span,
        )

        slice_type = ir.TileType(
            [16, 8], pl.FP32, None, ir.TileView(valid_shape=[dim, sub]), ir.MemorySpace.Vec
        )
        sliced = ir.Var("sliced", slice_type, span)
        slice_call = ir.Call(ir.Op("tile.slice"), [loaded, slice_shape, offsets], {}, slice_type, span)

        tpush_call = ir.Call(ir.Op("tile.tpush_to_aic"), [sliced], {"split": 0}, ir.UnknownType(), span)

        # Cross-core pipe scaffolding required by MixedKernelExpanded for an AIV
        # function using a V2C op (see the load->create test above).
        peer_buf_call = ir_op.system.import_peer_buffer(
            name="v2c_slot_buffer", peer_func="main_aic", span=span
        )
        peer_buf = ir.Var("peer_buf", peer_buf_call.type, span)
        init_pipe_call = ir_op.system.aiv_initialize_pipe(
            v2c_consumer_buf=peer_buf, dir_mask=2, slot_size=512, span=span
        )

        body = ir.SeqStmts(
            [
                ir.AssignStmt(peer_buf, peer_buf_call, span),
                ir.EvalStmt(init_pipe_call, span),
                ir.AssignStmt(loaded, load_call, span),
                ir.AssignStmt(sliced, slice_call, span),
                ir.EvalStmt(tpush_call, span),
                ir.ReturnStmt([out], span),
            ],
            span,
        )
        func = ir.Function(
            "main_aiv",
            [(data, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
            [out.type],
            body,
            span,
            ir.FunctionType.AIV,
            attrs={"dual_aiv_dispatch": True},
        )

        actual = _run_split_vector_kernel(ir.Program([func], "tile_slice_program", span))
        printed = python_print(actual)

        assert "if subblock_idx == 0:" in printed
        # Lane0 keeps the real slice; lane1 replaces it (and the load) with create.
        assert printed.count("pl.tile.slice(") == 1
        assert printed.count("pl.tile.create(") == 2
        # The lane1 slice result is a dynamic-valid empty [16, 8] tile, never a
        # static v_row=0/v_col=0 subview.
        assert re.search(
            r"sliced__ssa_v0_\d+: pl.Tile\[\[16, 8\], pl.FP32, pl.Mem.Vec, "
            r"pl.TileView\(valid_shape=\[0, 0\]\)\] = pl.tile.create",
            printed,
        )

    def test_no_split_dual_dispatch_rewrites_lane1_transpose_to_create(self):
        """Lane1 replay rewrites a ``tile.transpose`` into ``tile.create``.

        ``tile.transpose`` lowers to a pto-isa op that hangs the AICore (507018)
        when every operand is a zero-valid replay tile -- the same static/zero
        hazard gh#1649 hit for subview slices. The replay result is discarded, so
        lane1 emits an empty tile of the transposed shape instead (gh#1761).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        span = ir.Span.unknown()
        zero = ir.ConstInt(0, pl.INDEX, span)
        one = ir.ConstInt(1, pl.INDEX, span)
        dim = ir.ConstInt(16, pl.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple([dim, dim], span)
        valid_shapes = ir.MakeTuple([dim, dim], span)

        data = ir.Var("data", ir.TensorType([16, 16], pl.FP32), span)
        out = ir.Var("out", ir.TensorType([16, 16], pl.FP32), span)

        load_type = ir.TileType(
            [16, 16], pl.FP32, None, ir.TileView(valid_shape=[dim, dim]), ir.MemorySpace.Vec
        )
        loaded = ir.Var("loaded", load_type, span)
        load_call = ir.Call(
            ir.Op("tile.load"),
            [data, offsets, shapes, valid_shapes],
            {"target_memory": ir.MemorySpace.Vec},
            load_type,
            span,
        )

        transpose_type = ir.TileType(
            [16, 16], pl.FP32, None, ir.TileView(valid_shape=[dim, dim]), ir.MemorySpace.Vec
        )
        transposed = ir.Var("transposed", transpose_type, span)
        transpose_call = ir.Call(ir.Op("tile.transpose"), [loaded, zero, one], {}, transpose_type, span)

        tpush_call = ir.Call(ir.Op("tile.tpush_to_aic"), [transposed], {"split": 0}, ir.UnknownType(), span)

        peer_buf_call = ir_op.system.import_peer_buffer(
            name="v2c_slot_buffer", peer_func="main_aic", span=span
        )
        peer_buf = ir.Var("peer_buf", peer_buf_call.type, span)
        init_pipe_call = ir_op.system.aiv_initialize_pipe(
            v2c_consumer_buf=peer_buf, dir_mask=2, slot_size=512, span=span
        )

        body = ir.SeqStmts(
            [
                ir.AssignStmt(peer_buf, peer_buf_call, span),
                ir.EvalStmt(init_pipe_call, span),
                ir.AssignStmt(loaded, load_call, span),
                ir.AssignStmt(transposed, transpose_call, span),
                ir.EvalStmt(tpush_call, span),
                ir.ReturnStmt([out], span),
            ],
            span,
        )
        func = ir.Function(
            "main_aiv",
            [(data, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
            [out.type],
            body,
            span,
            ir.FunctionType.AIV,
            attrs={"dual_aiv_dispatch": True},
        )

        actual = _run_split_vector_kernel(ir.Program([func], "tile_transpose_program", span))
        printed = python_print(actual)

        assert "if subblock_idx == 0:" in printed
        # Lane0 keeps the real transpose; lane1 replaces it (and the load) with create.
        assert printed.count("pl.tile.transpose(") == 1
        assert printed.count("pl.tile.create(") == 2
        then_branch, lane1 = printed.split("else:", 1)
        # Lane 0 keeps the real transpose; lane 1 replaces it with an empty create.
        assert "pl.tile.transpose(" in then_branch
        assert "pl.tile.transpose(" not in lane1
        assert re.search(
            r"transposed__ssa_v0_\d+: pl.Tile\[\[16, 16\], pl.FP32, pl.Mem.Vec, "
            r"pl.TileView\(valid_shape=\[0, 0\]\)\] = pl.tile.create",
            lane1,
        )

    def test_no_split_dual_dispatch_hoists_import_peer_buffer_and_pipe_init(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                peer_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=peer_buf)
                zero_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                    [16, 16], dtype=pl.FP32, value=0.0
                )
                pl.tpush_to_aic(zero_tile, split=0)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                peer_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=peer_buf)
                if subblock_idx == 0:
                    zero_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                        [16, 16], dtype=pl.FP32, value=0.0
                    )
                    pl.tpush_to_aic(zero_tile, split=0)
                    return out
                else:
                    zero_tile_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.full([16, 16], dtype=pl.FP32, value=0.0)
                    pl.tpush_to_aic(zero_tile_lane1, split=0)
                    return out

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_dual_dispatch_consumer_keeps_only_tpop_tfree_on_lane1(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                pl.tfree_to_aic(z_vec)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(z_vec, [0, 0], out)
                return updated

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
                    pl.tfree_to_aic(z_vec)
                    updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(z_vec, [0, 0], out)
                    return updated
                else:
                    z_vec_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tpop_from_aic(split=0)
                    pl.tfree_to_aic(z_vec_lane1)
                    updated_lane1: pl.Tensor[[16, 16], pl.FP32] = out
                    return out

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_dual_dispatch_lane1_replays_empty_tiles_after_tpop(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=0
                )
                incremented: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(z_vec, 1.0)
                pl.tfree_to_aic(z_vec)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(incremented, [0, 0], out)
                return updated

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
                    incremented: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(z_vec, 1.0)
                    pl.tfree_to_aic(z_vec)
                    updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(incremented, [0, 0], out)
                    return updated
                else:
                    z_vec_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tpop_from_aic(split=0)
                    incremented_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.add(z_vec_lane1, 1.0)
                    pl.tfree_to_aic(z_vec_lane1)
                    updated_lane1: pl.Tensor[[16, 16], pl.FP32] = out
                    return out

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_dual_dispatch_lane1_loop_init_uses_empty_accumulator(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                    [16, 16], dtype=pl.FP32, value=0.0
                )
                for kb, (acc_iter,) in pl.range(4, init_values=(acc,)):
                    z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    next_acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_iter, z_vec)
                    pl.tfree_to_aic(z_vec)
                    acc_final: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.yield_(next_acc)
                incremented: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_final, 1.0)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(incremented, [0, 0], out)
                return updated

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                        [16, 16], dtype=pl.FP32, value=0.0
                    )
                    for kb, (acc_iter,) in pl.range(4, init_values=(acc,)):
                        z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
                        next_acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_iter, z_vec)
                        pl.tfree_to_aic(z_vec)
                        acc_final: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.yield_(next_acc)
                    incremented: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_final, 1.0)
                    updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(incremented, [0, 0], out)
                    return updated
                else:
                    acc_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.full([16, 16], dtype=pl.FP32, value=0.0)
                    for kb_lane1, (acc_iter_lane1,) in pl.range(4, init_values=(acc_lane1,)):
                        z_vec_lane1: pl.Tile[
                            [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                        ] = pl.tpop_from_aic(split=0)
                        next_acc_lane1: pl.Tile[
                            [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                        ] = pl.add(acc_iter_lane1, z_vec_lane1)
                        pl.tfree_to_aic(z_vec_lane1)
                        acc_final_lane1: pl.Tile[
                            [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                        ] = pl.yield_(next_acc_lane1)
                    incremented_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.add(acc_final_lane1, 1.0)
                    updated_lane1: pl.Tensor[[16, 16], pl.FP32] = out
                    return out

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_dual_dispatch_lane1_while_init_uses_empty_accumulator(self):
        """Cover lane1 replay for while-loop tile/scalar carried state."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.full(
                    [16, 16], dtype=pl.FP32, value=0.0
                )
                count: pl.Scalar[pl.INDEX] = 0
                for acc_iter, count_iter in pl.while_(init_values=(acc, count)):
                    pl.cond(count_iter < 4)
                    z_vec: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    next_acc: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_iter, z_vec)
                    next_count: pl.Scalar[pl.INDEX] = count_iter + 1
                    pl.tfree_to_aic(z_vec)
                    acc_final, count_final = pl.yield_(next_acc, next_count)
                incremented: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc_final, 1.0)
                updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(incremented, [0, 0], out)
                return updated

        actual = _run_split_vector_kernel(Before)
        printed = python_print(actual)
        lane1 = printed.split("else:", 1)[1]

        assert re.search(
            r"acc__ssa_v0_\d+: pl.Tile\[\[16, 16\], pl.FP32, pl.Mem.Vec, "
            r"pl.TileView\(valid_shape=\[0, 0\]\)\] = pl.tile.full",
            lane1,
        )
        assert re.search(
            r"for \(?acc_iter_\d+, count_iter_\d+\)? in pl.while_"
            r"\(init_values=\(acc__ssa_v0_\d+, count__ssa_v0_\d+\)\)",
            lane1,
        )
        assert "pl.while_(init_values=(acc__ssa_v0, count__ssa_v0))" not in lane1
        assert re.search(r"pl.cond\(count_iter_\d+ < 4\)", lane1)
        assert re.search(
            r"incremented__ssa_v0_\d+: pl.Tile\[\[16, 16\], pl.FP32, pl.Mem.Vec, "
            r"pl.TileView\(valid_shape=\[0, 0\]\)\]",
            lane1,
        )


def _count_op_calls(stmt, op_name: str) -> int:
    """Recursively count Call expressions to ``op_name`` in a statement tree."""
    count = 0

    def visit_expr(expr):
        nonlocal count
        if expr is None:
            return
        if isinstance(expr, ir.Call) and expr.op is not None and expr.op.name == op_name:
            count += 1

    def visit_stmt(s):
        nonlocal count
        if s is None:
            return
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                visit_stmt(child)
        elif isinstance(s, ir.AssignStmt):
            visit_expr(s.value)
        elif isinstance(s, ir.EvalStmt):
            visit_expr(s.expr)
        elif isinstance(s, ir.ForStmt):
            visit_stmt(s.body)
        elif isinstance(s, ir.WhileStmt):
            visit_stmt(s.body)
        elif isinstance(s, ir.IfStmt):
            visit_stmt(s.then_body)
            if s.else_body is not None:
                visit_stmt(s.else_body)

    visit_stmt(stmt)
    return count


def _find_assign_tile_shape(stmt, var_name_prefix: str) -> list[int] | None:
    """Return the static tile shape of the first AssignStmt whose Var name starts with prefix."""
    result: list[int] | None = None

    def visit_stmt(s):
        nonlocal result
        if result is not None or s is None:
            return
        if isinstance(s, ir.SeqStmts):
            for child in s.stmts:
                visit_stmt(child)
        elif isinstance(s, ir.AssignStmt):
            if s.var.name_hint.startswith(var_name_prefix) and isinstance(s.var.type, ir.TileType):
                dims: list[int] = []
                for d in s.var.type.shape:
                    assert isinstance(d, ir.ConstInt)
                    dims.append(int(d.value))
                result = dims
        elif isinstance(s, ir.ForStmt):
            visit_stmt(s.body)
        elif isinstance(s, ir.WhileStmt):
            visit_stmt(s.body)
        elif isinstance(s, ir.IfStmt):
            visit_stmt(s.then_body)
            if s.else_body is not None:
                visit_stmt(s.else_body)

    visit_stmt(stmt)
    return result


class TestSplitVectorKernelExplicitSplitAivBypass:
    """SplitVectorKernel stamps attrs for split_aiv kernels and passes them through.

    This is the SOLE split path through SplitVectorKernel after the convergence
    refactor (the per-op halving driver was deleted). A ``split_aiv`` kernel —
    hand-written, or produced upstream by LowerAutoVectorSplit — has already
    lowered its explicit ``tile.aiv_shard`` / ``tile.aic_gather`` into
    split-stamped tpush/tpop and carries already-halved compute tiles plus its
    own ``get_subblock_idx``. The ``split_aiv`` marker must make SplitVectorKernel
    leave the body untouched: it only stamps ``split`` + ``dual_aiv_dispatch``.
    """

    def test_split_aiv_bypass_keeps_half_tiles_and_single_subblock_idx(self):
        @pl.program
        class Before:
            @pl.function(
                type=pl.FunctionType.AIC,
                attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
            )
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                peer_buf = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="main_aiv")
                pl.aic_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=peer_buf)
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            # AIV body is already in the post-split form: half [8, 128] tiles, one
            # hand-written get_subblock_idx, and a store offset that already uses it.
            @pl.function(
                type=pl.FunctionType.AIV,
                attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True},
            )
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0 + subblock_idx * 8, 0], out_0)
                return out_0_store

        actual = _run_split_vector_kernel(Before)
        aiv = actual.get_function("main_aiv")
        assert aiv is not None

        # (i) split + dual_aiv_dispatch stamped; split_aiv marker preserved.
        assert aiv.attrs["split"] == pl.SplitMode.UP_DOWN.value
        assert aiv.attrs["dual_aiv_dispatch"] is True
        assert aiv.attrs["split_aiv"] is True

        # (ii) compute tile UNCHANGED — still [8, 128], not re-halved to [4, 128].
        assert _find_assign_tile_shape(aiv.body, "z_vec") == [8, 128]

        # (iii) exactly one get_subblock_idx — no second one injected.
        assert _count_op_calls(aiv.body, "tile.get_subblock_idx") == 1


class TestSplitVectorKernelStandaloneSetValidshape:
    """set_validshape split-axis operand localization on the standalone split path.

    The input is already-split AIC/AIV functions carrying a function-level
    ``split`` attr (not the ``split_aiv`` marker), so SplitVectorKernel's
    standalone arm (ProcessStandaloneSplitFunction -> split_axis::ProcessStmts)
    halves them -- the same localization the auto ``pl.split`` path gets via
    LowerAutoVectorSplit.
    """

    def test_set_validshape_split_dim_operand_localized(self):
        """set_validshape: the split-dim valid operand is localized to the halved box.

        Halving only the result type left the explicit row operand at its full
        pre-split extent (e.g. 16 on an 8-row physical tile), which PTOAS rejects
        with 'set_validshape op expects row operand <= shape dim'. The split-dim
        operand must be localized exactly like the result type's valid_shape; the
        non-split col operand is left untouched.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                peer_buf = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="main_aiv")
                pl.aic_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=peer_buf)
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=1
                )
                pl.tfree_to_aic(z_vec)
                narrowed: pl.Tile[
                    [16, 128],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(valid_shape=[16, 64]),
                ] = pl.tile.set_validshape(z_vec, 16, 64)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(narrowed, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                peer_buf = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="main_aiv")
                pl.aic_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=peer_buf)
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(
                type=pl.FunctionType.AIV,
                attrs={"split": pl.SplitMode.UP_DOWN, "dual_aiv_dispatch": True},
            )
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                pl.tfree_to_aic(z_vec)
                narrowed: pl.Tile[
                    [8, 128],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(valid_shape=[8, 64]),
                ] = pl.tile.set_validshape(z_vec, 8, 64)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(
                    narrowed, [0 + subblock_idx * 8, 0], out_0
                )
                return out_0_store

        _assert_split_matches_expected(Before, Expected)

    def test_set_validshape_replicated_operand_not_localized(self):
        """set_validshape: a replicated valid operand (< half the physical box) is preserved.

        When the valid extent is smaller than the halved physical dim it is a
        replicated extent both AIV lanes share (e.g. a fused-attention head count,
        valid_row=5 on a [16]->[8] split), not a row partition. Localizing it would
        subtract half on lane 1 and collapse it to 0, silently corrupting that
        lane. The operand must stay verbatim on both lanes; only the result type's
        valid_shape is localized (a harmless annotation on a non-subview tile).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                peer_buf = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="main_aiv")
                pl.aic_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=peer_buf)
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                    split=1
                )
                pl.tfree_to_aic(z_vec)
                narrowed: pl.Tile[
                    [16, 128],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(valid_shape=[5, 64]),
                ] = pl.tile.set_validshape(z_vec, 5, 64)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(narrowed, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
                peer_buf = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="main_aiv")
                pl.aic_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=peer_buf)
                x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
                y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
                z_tile = pl.matmul(x_left, y_right)
                pl.tpush_to_aiv(z_tile, split=1)

            @pl.function(
                type=pl.FunctionType.AIV,
                attrs={"split": pl.SplitMode.UP_DOWN, "dual_aiv_dispatch": True},
            )
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                pl.tfree_to_aic(z_vec)
                narrowed: pl.Tile[
                    [8, 128],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(valid_shape=[pl.max(pl.min(5 - subblock_idx * 8, 8), 0), 64]),
                ] = pl.tile.set_validshape(z_vec, 5, 64)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(
                    narrowed, [0 + subblock_idx * 8, 0], out_0
                )
                return out_0_store

        _assert_split_matches_expected(Before, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
