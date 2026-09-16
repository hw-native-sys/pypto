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

    def test_external_dual_dispatch_keeps_signature_only_body(self, tmp_path):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        source = tmp_path / "external.cpp"
        source.write_text('extern "C" void kernel_entry(long long* args) { (void)args; }\n')

        @pl.program
        class Before:
            @pl.function(
                type=pl.FunctionType.AIV,
                external_source=source,
                attrs={"dual_aiv_dispatch": True},
            )
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]: ...

        result = _run_split_vector_kernel(Before)
        ir.assert_structural_equal(result, passes.convert_to_ssa()(Before))
        printed = python_print(result)
        assert "external_source" in printed
        assert "get_subblock_idx" not in printed

    def test_external_split_stamps_dual_dispatch_without_body_rewrite(self, tmp_path):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        source = tmp_path / "external.cpp"
        source.write_text('extern "C" void kernel_entry(long long* args) { (void)args; }\n')

        @pl.program
        class Before:
            @pl.function(
                type=pl.FunctionType.AIV,
                external_source=source,
                attrs={"split": pl.SplitMode.UP_DOWN},
            )
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]: ...

        result = _run_split_vector_kernel(Before)
        aiv = result.get_function("main_aiv")
        assert aiv is not None
        assert aiv.attrs.get("dual_aiv_dispatch") is True
        assert "get_subblock_idx" not in python_print(result)

    def test_external_split_aiv_overrides_false_dual_dispatch_attr(self, tmp_path):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)
        source = tmp_path / "external.cpp"
        source.write_text('extern "C" void kernel_entry(long long* args) { (void)args; }\n')

        @pl.program
        class Before:
            @pl.function(
                type=pl.FunctionType.AIV,
                external_source=source,
                attrs={"split_aiv": True, "dual_aiv_dispatch": False},
            )
            def main_aiv(
                self,
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]: ...

        result = _run_split_vector_kernel(Before)
        aiv = result.get_function("main_aiv")
        assert aiv is not None
        assert aiv.attrs.get("dual_aiv_dispatch") is True
        assert "get_subblock_idx" not in python_print(result)

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
        """Lane1 replay rewrites a producer ``tile.load`` into ``tile.create``.

        The replay lane never consumes the loaded data — only the ``tpush`` has
        to happen on both lanes — so lane1 gets an empty tile of the load's
        result shape instead of a second GM read.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                loaded: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec
                )
                pl.tpush_to_aic(loaded, split=0)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    loaded: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        data, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec
                    )
                    pl.tpush_to_aic(loaded, split=0)
                    return out
                else:
                    loaded_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    pl.tpush_to_aic(loaded_lane1, split=0)
                    return out

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_dual_dispatch_rewrites_lane1_tile_slice_to_create(self):
        """Lane1 replay rewrites a producer ``tile.slice`` into ``tile.create``.

        A ``tile.slice`` is a pure view with no cross-core sync, so the replay
        lane only needs an empty tile of the slice's result shape. Forcing the
        slice's explicit ``valid_shape`` to a static 0 would emit a
        ``v_row=0, v_col=0`` subview that pto-isa cannot compile (no
        ``GetValidRow`` overload for a static mask of 0); ``Expected`` pins the
        rewrite to a dynamic-valid empty ``tile.create`` instead (gh#1649).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
            ) -> pl.Tensor[[16, 8], pl.FP32]:
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                loaded: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec
                )
                sliced: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.tile.slice(loaded, [16, 8], [0, 0])
                pl.tpush_to_aic(sliced, split=0)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
            ) -> pl.Tensor[[16, 8], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    loaded: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        data, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec
                    )
                    sliced: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.tile.slice(
                        loaded, [16, 8], [0, 0]
                    )
                    pl.tpush_to_aic(sliced, split=0)
                    return out
                else:
                    # Both the load and the slice collapse to empty creates; the
                    # slice result is a dynamic-valid empty [16, 8] tile, never a
                    # static v_row=0 / v_col=0 subview.
                    loaded_lane1: pl.Tile[
                        [16, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    sliced_lane1: pl.Tile[
                        [16, 8], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([16, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    pl.tpush_to_aic(sliced_lane1, split=0)
                    return out

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_dual_dispatch_rewrites_lane1_transpose_to_create(self):
        """Lane1 replay rewrites a ``tile.transpose`` into ``tile.create``.

        ``tile.transpose`` lowers to a pto-isa op that hangs the AICore (507018)
        when every operand is a zero-valid replay tile -- the same static/zero
        hazard gh#1649 hit for subview slices. The replay result is discarded, so
        lane1 emits an empty tile of the transposed shape instead (gh#1761).

        The tile is deliberately non-square (``[16, 8] -> [8, 16]``): with a
        square one the replay ``tile.create`` would have the same shape whether
        the rewrite took the transpose's result shape or its source shape, so
        the comparison would not pin which.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
            ) -> pl.Tensor[[8, 16], pl.FP32]:
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                loaded: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    data, [0, 0], [16, 8], target_memory=pl.MemorySpace.Vec
                )
                transposed: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.transpose(loaded, 0, 1)
                pl.tpush_to_aic(transposed, split=0)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[8, 16], pl.FP32]],
            ) -> pl.Tensor[[8, 16], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    loaded: pl.Tile[[16, 8], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        data, [0, 0], [16, 8], target_memory=pl.MemorySpace.Vec
                    )
                    transposed: pl.Tile[[8, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.transpose(
                        loaded, 0, 1
                    )
                    pl.tpush_to_aic(transposed, split=0)
                    return out
                else:
                    # Lane 1 keeps no transpose at all -- the hazard the rewrite
                    # avoids -- and its empty tile carries the TRANSPOSED shape.
                    loaded_lane1: pl.Tile[
                        [16, 8], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([16, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    transposed_lane1: pl.Tile[
                        [8, 16], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([8, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    pl.tpush_to_aic(transposed_lane1, split=0)
                    return out

        _assert_split_matches_expected(Before, Expected)

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
                    pl.TileView(valid_shape=[pl.min(pl.max(5, subblock_idx * 8) - subblock_idx * 8, 8), 64]),
                ] = pl.tile.set_validshape(z_vec, 5, 64)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(
                    narrowed, [0 + subblock_idx * 8, 0], out_0
                )
                return out_0_store

        _assert_split_matches_expected(Before, Expected)


class TestSplitVectorKernelManualTpopDeferredValid:
    """Where a hand-written ``tile.tpop_from_aic``'s split-axis extent may land.

    pto-isa reads lane 1's band offset off the POPPED tile's own runtime valid
    extents, while PTO codegen always transports the producer's full physical
    box. The halving may therefore materialize the lane's extent onto the pop
    only for the lane pairs the transport has a code for -- equal, one apart, or
    an empty lane 1 (``split_axis::BoundaryCarriesLaneExtent``). Any other pair
    DEFERS: the pop declares its box, which is where the producer wrote lane 1,
    and the lane's extent lands on the pop's consumers instead.

    This is the same pairing ``LocalizeExplicitBoundaryValid`` gives a
    ``pl.split_aiv`` region's ``tile.aiv_shard``; these tests pin it for the
    hand-written transport form, which reaches it through
    ``RebuildTpopWithHalvedShape``.
    """

    def test_runtime_split_axis_extent_defers_onto_the_consumer(self):
        """A RUNTIME extent has no compile-time lane pair, so the pop takes the box.

        Materializing ``clamp(vr - lane * 8, 0, 8)`` on the pop would send lane 1
        to row ``vr - 8``; the producer wrote it at row 8. The extent moves to the
        ``tile.add`` that consumes the pop, where no band offset depends on it.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self,
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                vr: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[vr, 128])] = (
                    pl.tpop_from_aic(split=1)
                )
                inc: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[vr, 128])] = (
                    pl.tile.add(z_vec, 1.0)
                )
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(inc, [0, 0], out_0)
                return out_0_store

        @pl.program
        class Expected:
            @pl.function(
                type=pl.FunctionType.AIV,
                attrs={"split": pl.SplitMode.UP_DOWN, "dual_aiv_dispatch": True},
            )
            def main_aiv(
                self,
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                vr: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[8, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1)
                inc: pl.Tile[
                    [8, 128],
                    pl.FP32,
                    pl.MemorySpace.Vec,
                    pl.TileView(
                        valid_shape=[pl.min(pl.max(vr, subblock_idx * 8) - subblock_idx * 8, 8), 128]
                    ),
                ] = pl.tile.add(z_vec, 1.0)
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(inc, [0 + subblock_idx * 8, 0], out_0)
                return out_0_store

        _assert_split_matches_expected(Before, Expected)

    def test_ragged_static_extent_defers_instead_of_being_refused(self):
        """12 of a 16-row box leaves the lanes 8 and 4 -- no code places that pair.

        The hand-written form has no partition left to choose: the author fixed
        the lanes at the box half by writing the transport, and there is no
        ``ResolveLaneStride`` to re-cut them. So the pop defers rather than
        reporting a decision nobody can revisit (``SplitOrigin::kManualTransport``).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[12, 128])] = (
                    pl.tpop_from_aic(split=1)
                )
                inc: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[12, 128])] = (
                    pl.tile.add(z_vec, 1.0)
                )
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(inc, [0, 0], out_0)
                return out_0_store

        result = _run_split_vector_kernel(Before)
        printed = ir.python_print(result)
        # The pop carries the transport's box (no valid_shape survives on it) and
        # keeps the EVEN code -- lane 1's band is at the box half.
        assert "pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=1)" in printed, printed
        # The lane's own extent is still materialized, on the consumer.
        assert "pl.tile.add" in printed or "pl.tile.adds" in printed, printed
        assert "subblock_idx" in printed, printed

    def test_placeable_odd_extent_keeps_the_lane_extent_on_the_pop(self):
        """15 of 16 leaves the lanes 8 and 7 -- one apart, which the ODD code places.

        A placeable pair must NOT defer: the odd codes exist precisely so the
        per-lane extent can ride on the transport, and pto-isa reads it off the
        popped tile to find lane 1 at ``(e1 + 1) * pitch``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[15, 128])] = (
                    pl.tpop_from_aic(split=1)
                )
                inc: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[15, 128])] = (
                    pl.tile.add(z_vec, 1.0)
                )
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(inc, [0, 0], out_0)
                return out_0_store

        printed = ir.python_print(_run_split_vector_kernel(Before))
        assert "pl.tile.tpop_from_aic(split=3)" in printed, printed
        # The pop itself carries the per-lane clamp, NOT the box.
        pop_line = next(line for line in printed.splitlines() if "tpop_from_aic" in line)
        pop_decl = printed[: printed.index(pop_line) + len(pop_line)]
        assert "valid_shape" in pop_decl, printed

    def test_narrowed_column_is_still_refused_on_the_manual_path(self):
        """The FIFO's valid-shape contract is a LOWERING fact, so it is not deferred.

        No choice of partition lets the transport carry a narrowed column extent:
        pto-isa strides the GM slot view with the popped tile's own ``validCol``
        against the producer's full-box pitch. Unlike the lane-pair rule, this one
        holds for a hand-written transport exactly as for ``tile.aiv_shard``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
            def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[16, 64])] = (
                    pl.tpop_from_aic(split=2)
                )
                inc: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[16, 64])] = (
                    pl.tile.add(z_vec, 1.0)
                )
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(inc, [0, 0], out_0)
                return out_0_store

        with pytest.raises(ValueError, match="LEFT_RIGHT splits the column axis"):
            _run_split_vector_kernel(Before)

    def test_deferred_pop_stored_directly_is_refused(self):
        """A store off a deferred pop would write the transport's padding as data.

        The extent has to land on a consumer, and a store carries nothing onward.
        Same refusal ``RejectDeferredValidStore`` makes inside a region.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self,
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                vr: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[vr, 128])] = (
                    pl.tpop_from_aic(split=1)
                )
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                pl.tfree_to_aic(z_vec)
                return out_0_store

        with pytest.raises(ValueError, match="reads a hand-written tile.tpop_from_aic directly"):
            _run_split_vector_kernel(Before)

    def test_deferred_pop_pad_fill_is_refused(self):
        """A pad fill off a deferred pop has no per-lane boundary left to fill up to."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main_aiv(
                self,
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
                vr: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[vr, 128])] = (
                    pl.tpop_from_aic(split=1)
                )
                filled: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec] = pl.tile.fillpad(z_vec, 0.0)
                pl.tfree_to_aic(z_vec)
                out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(filled, [0, 0], out_0)
                return out_0_store

        with pytest.raises(ValueError, match="fills the padding of a hand-written"):
            _run_split_vector_kernel(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
