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
    Native C2V setup may be shared by both lanes; other scaffolding passes
    through unchanged. The before/after contract asserts the split
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
    with no split mode is passed through unchanged unless its native C2V
    constructor requires both AIV lanes to participate. The per-op halving
    tests these classes used to hold moved to ``test_lower_auto_vector_split.py``
    (they assert the same facts via the shared ``split_axis`` machinery, now
    reached through LowerAutoVectorSplit).
    """

    def test_native_c2v_preserves_primary_body_and_shares_initialization(self):
        """No-split shapes stay intact while both constructor participants run."""

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
                slot_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
                pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    z_vec: pl.Tile[[16, 128], pl.FP32, pl.MemorySpace.Vec, pl.TileView()] = pl.tpop_from_aic(
                        split=0
                    )
                    pl.tfree_to_aic(z_vec)
                    out_0_store: pl.Tensor[[16, 128], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
                    return out_0_store
                else:
                    return out_0

        _assert_split_matches_expected(Before, Expected)
        result = _run_split_vector_kernel(Before)
        ir.assert_structural_equal(result, passes.split_vector_kernel()(result))
        original = Before.get_function("main_aiv")
        assert original is not None
        assert "dual_aiv_dispatch" not in original.attrs

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
    """Tests for Ascend910 no-split mixed-kernel dual-dispatch lowering."""

    def test_external_dual_dispatch_keeps_signature_only_body(self, tmp_path):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
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
        backend.set_backend_type(BackendType.Ascend910)
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
        backend.set_backend_type(BackendType.Ascend910)
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
        backend.set_backend_type(BackendType.Ascend910)

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
        backend.set_backend_type(BackendType.Ascend910)

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
        backend.set_backend_type(BackendType.Ascend910)

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
        backend.set_backend_type(BackendType.Ascend910)

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

    def test_no_split_dual_dispatch_replaces_empty_replay_reductions(self):
        """Keep real reductions on lane 0 and only their empty result on lane 1.

        Both CPU-SIM and A2/A3 row reductions require nonempty inputs. The
        replay lane still needs its push to balance the cube handshake.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                loaded = pl.tile.load(data, [0, 0], [16, 8], target_memory=pl.MemorySpace.Vec)
                scratch = pl.tile.create([16, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                reduced = pl.tile.row_max(loaded, scratch)
                pl.tpush_to_aic(reduced, split=0)
                minimum = pl.tile.row_min(loaded, scratch)
                pl.tpush_to_aic(minimum, split=0)
                total = pl.tile.row_sum(loaded, scratch)
                pl.tpush_to_aic(total, split=0)
                product = pl.tile.row_prod(loaded, scratch)
                pl.tpush_to_aic(product, split=0)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
            def main_aiv(
                self,
                data: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                subblock_idx = pl.tile.get_subblock_idx()
                slot_buf = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
                if subblock_idx == 0:
                    loaded = pl.tile.load(data, [0, 0], [16, 8], target_memory=pl.MemorySpace.Vec)
                    scratch = pl.tile.create([16, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    reduced = pl.tile.row_max(loaded, scratch)
                    pl.tpush_to_aic(reduced, split=0)
                    minimum = pl.tile.row_min(loaded, scratch)
                    pl.tpush_to_aic(minimum, split=0)
                    total = pl.tile.row_sum(loaded, scratch)
                    pl.tpush_to_aic(total, split=0)
                    product = pl.tile.row_prod(loaded, scratch)
                    pl.tpush_to_aic(product, split=0)
                    return out
                else:
                    loaded_lane1: pl.Tile[
                        [16, 8], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([16, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    scratch_lane1: pl.Tile[
                        [16, 8], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
                    ] = pl.tile.create([16, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    reduced_lane1: pl.Tile[
                        [16, 1],
                        pl.FP32,
                        pl.MemorySpace.Vec,
                        pl.TileView(blayout=pl.TileLayout.col_major, valid_shape=[0, 0]),
                    ] = pl.tile.create([16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    pl.tpush_to_aic(reduced_lane1, split=0)
                    minimum_lane1: pl.Tile[
                        [16, 1],
                        pl.FP32,
                        pl.MemorySpace.Vec,
                        pl.TileView(blayout=pl.TileLayout.col_major, valid_shape=[0, 0]),
                    ] = pl.tile.create([16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    pl.tpush_to_aic(minimum_lane1, split=0)
                    total_lane1: pl.Tile[
                        [16, 1],
                        pl.FP32,
                        pl.MemorySpace.Vec,
                        pl.TileView(blayout=pl.TileLayout.col_major, valid_shape=[0, 0]),
                    ] = pl.tile.create([16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    pl.tpush_to_aic(total_lane1, split=0)
                    product_lane1: pl.Tile[
                        [16, 1],
                        pl.FP32,
                        pl.MemorySpace.Vec,
                        pl.TileView(blayout=pl.TileLayout.col_major, valid_shape=[0, 0]),
                    ] = pl.tile.create([16, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    pl.tpush_to_aic(product_lane1, split=0)
                    return out

        _assert_split_matches_expected(Before, Expected)

    def test_no_split_dual_dispatch_hoists_import_peer_buffer_and_pipe_init(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)

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
        backend.set_backend_type(BackendType.Ascend910)

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
        backend.set_backend_type(BackendType.Ascend910)

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
        backend.set_backend_type(BackendType.Ascend910)

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
        backend.set_backend_type(BackendType.Ascend910)

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
        if isinstance(expr, ir.Call) and expr.op.name == op_name:
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


@pytest.mark.parametrize(
    "expression, result_shape, result_dtype",
    [
        ("pl.reshape(popped, [8, 32])", [8, 32], "FP32"),
        ("pl.tile.reinterpret_view(popped, pl.INT32)", [16, 16], "INT32"),
        ("pl.tile.transpose_view(popped)", [16, 16], "FP32"),
    ],
)
def test_replay_tpop_views_preserve_fifo_aliases(expression, result_shape, result_dtype):
    """Empty lane-1 views keep their FIFO source and allocate no payload storage."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    program = pl.parse(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def main_aiv(self):
        pl.func_attr({{"dual_aiv_dispatch": True}})
        local = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=8192)
        peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="main_aic")
        pl.aiv_initialize_pipe(dir_mask=3, slot_size=1024, c2v_consumer_buf=local, v2c_consumer_buf=peer)
        popped: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        view: pl.Tile[{result_shape}, pl.{result_dtype}, pl.MemorySpace.Vec] = {expression}
        pl.tpush_to_aic(view, split=0)
        pl.tfree_to_aic(popped)
""")
    transformed = _run_split_vector_kernel(program)
    function = transformed.get_function("main_aiv")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    branch = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.IfStmt))
    assert isinstance(branch.else_body, ir.SeqStmts)
    replay = branch.else_body
    calls = [
        stmt.value
        for stmt in replay.stmts
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    ]
    assert all(call.op.name != ir.get_op("tile.create").name for call in calls), python_print(transformed)
    view_names = {
        ir.get_op(name).name for name in ("tile.reshape", "tile.reinterpret_view", "tile.transpose_view")
    }
    views = [call for call in calls if call.op.name in view_names]
    assert len(views) == 1, python_print(transformed)
    popped = next(
        stmt.var
        for stmt in replay.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tile.tpop_from_aic").name
    )
    ir.assert_structural_equal(views[0].args[0], popped)
    tile_type = views[0].type
    assert isinstance(tile_type, ir.TileType)
    assert len(tile_type.shape) == len(result_shape)
    for dim, expected in zip(tile_type.shape, result_shape):
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == expected
    assert tile_type.dtype == getattr(pl, result_dtype)
    assert tile_type.tile_view is not None
    assert len(tile_type.tile_view.valid_shape) == 2
    for dim in tile_type.tile_view.valid_shape:
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == 0
    text = python_print(transformed)
    for operation in ("tile.tpop_from_aic", "tile.tpush_to_aic", "system.tfree_to_aic"):
        assert text.count(f"pl.{operation}(") == 2, text


def test_native_c2v_data_return_has_actionable_diagnostic():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def main(self) -> pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec]:
            buf = pl.reserve_buffer(name="c2v", size=2048, base=0)
            pl.aiv_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
            received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(
                shape=[16, 16], dtype=pl.FP32, split=0
            )
            pl.tfree_to_aic(received)
            return received

    with pytest.raises(ValueError, match="returns to alias parameters.*explicit split_aiv"):
        _run_split_vector_kernel(Before)


def test_native_c2v_preserves_authored_dual_dispatch():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
        def main(self, out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
            buf = pl.reserve_buffer(name="c2v", size=2048, base=0)
            pl.aiv_initialize_pipe(buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
            received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(
                shape=[16, 16], dtype=pl.FP32, split=0
            )
            updated = pl.store(received, [0, 0], out)
            pl.tfree_to_aic(received)
            return updated

    ir.assert_structural_equal(_run_split_vector_kernel(Before), passes.convert_to_ssa()(Before))


def _assert_extent_after_substitution(expr, values, expected):
    """Use the compiler's scalar simplifier to check the selected lane's extent."""
    span = ir.Span.unknown()

    class Substitute(ir.IRMutator):
        def visit_var(self, op):
            if op.name_hint in values:
                assert isinstance(op.type, ir.ScalarType)
                return ir.ConstInt(values[op.name_hint], op.type.dtype, op.span)
            return super().visit_var(op)

        def visit_iter_arg(self, op):
            if op.name_hint in values:
                return self.visit_var(op)
            return super().visit_iter_arg(op)

    value = Substitute().visit_expr(expr)
    before = ir.Program(
        [ir.Function("extent", [], [ir.ScalarType(pl.INDEX)], ir.ReturnStmt([value], span), span)],
        "Extent",
        span,
    )
    wanted = ir.Program(
        [
            ir.Function(
                "extent",
                [],
                [ir.ScalarType(pl.INDEX)],
                ir.ReturnStmt([ir.ConstInt(expected, pl.INDEX, span)], span),
                span,
            )
        ],
        "Extent",
        span,
    )
    ir.assert_structural_equal(passes.simplify()(before), wanted)


@pytest.mark.parametrize(
    ("mode", "shape", "view", "axis", "narrow_after_pop"),
    [
        ("UP_DOWN", [31, 16], "", 0, False),
        ("LEFT_RIGHT", [16, 31], "", 1, False),
        ("UP_DOWN", [32, 16], ", pl.TileView(valid_shape=[31, 16])", 0, False),
        pytest.param("UP_DOWN", [32, 16], "", 0, True, id="even_then_narrow_rows"),
        pytest.param("LEFT_RIGHT", [16, 32], "", 1, True, id="even_then_narrow_cols"),
    ],
)
@pytest.mark.parametrize("lane_kind", ["omit", "zero", "one", "swapped"])
@pytest.mark.parametrize("carry", [False, True])
def test_explicit_pop_lane_localizes_odd_validity(
    mode, shape, view, axis, narrow_after_pop, lane_kind, carry
):
    """The 31-cell band is 16/15 by LOGICAL lane, including its consumers.

    Output placement still follows the executing physical AIV. A swapped pop
    on physical lane zero must read 15 cells, rather than pairing logical lane
    one's band offset with physical lane zero's 16-cell valid extent.
    """
    operand = {"omit": "", "zero": "0", "one": "1", "swapped": "lane"}[lane_kind]
    lane_kw = f", aiv_subblockid={operand}" if operand else ""
    split = (1 if axis == 0 else 2) if narrow_after_pop else (3 if axis == 0 else 4)
    loop = "for step in pl.range(2):\n            value = pl.tile.add(value, 1.0)" if carry else ""
    valid = [31, 16] if axis == 0 else [16, 31]
    narrow = f"narrowed = pl.tile.set_validshape(value, {valid[0]}, {valid[1]})" if narrow_after_pop else ""
    consumer_input = "narrowed" if narrow_after_pop else "value"
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, lane: pl.Scalar[pl.INT64], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]):
        pl.func_attr({{"split": pl.SplitMode.{mode}}})
        buf = pl.system.reserve_buffer(name="c2v", size=65536)
        pl.system.aiv_initialize_pipe(dir_mask=1, slot_size=4096, slot_num=16, c2v_consumer_buf=buf)
        received: pl.Tile[{shape}, pl.FP32, pl.Mem.Vec{view}] = pl.system.tpop_from_aic(
            split={split}{lane_kw}
        )
        value = pl.tile.add(received, 1.0)
        {loop}
        {narrow}
        incremented = pl.tile.add({consumer_input}, 1.0)
        pl.system.tfree_to_aic(received)
        saved = pl.tile.store(incremented, [0, 0], out)
        return saved
""")
    after = _run_split_vector_kernel(before)
    tiles = {}
    physical_vars = []
    stores = []
    physical_name = ir.get_op("tile.get_subblock_idx").name
    pop_name = ir.get_op("tile.tpop_from_aic").name
    add_name = ir.get_op("tile.adds").name
    store_name = ir.get_op("tile.store").name

    class Collect(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call):
                name = op.value.op.name
                if name == physical_name:
                    physical_vars.append(op.var)
                elif name in (pop_name, add_name):
                    tiles[name] = op
                elif name == store_name:
                    stores.append(op.value)
            super().visit_assign_stmt(op)

    Collect().visit_program(after)
    assert len(tiles) == 2 and physical_vars and len(stores) == 1
    pop = tiles[pop_name].value
    assert len(pop.args) == (0 if lane_kind == "omit" else 1)
    for physical in (0, 1):
        logical = {"omit": physical, "zero": 0, "one": 1, "swapped": 1 - physical}[lane_kind]
        values = {var.name_hint: physical for var in physical_vars}
        if lane_kind == "swapped":
            values[pop.args[0].name_hint] = logical
        for name, assignment in tiles.items():
            tile_type = assignment.var.type
            assert isinstance(tile_type, ir.TileType)
            extent = tile_type.get_effective_tile_view().valid_shape[axis]
            expected = 16 if logical == 0 else 15
            if narrow_after_pop and name == pop_name:
                expected = 16
            _assert_extent_after_substitution(extent, values, expected)
        # Selecting another input band must not relocate the physical output.
        _assert_extent_after_substitution(stores[0].args[1].elements[axis], values, physical * 16)


@pytest.mark.parametrize("pop_on_backedge", [True, False], ids=["pop_on_backedge", "pop_in_init"])
def test_full_pop_carry_tracks_the_lane_that_produced_the_value(pop_on_backedge):
    """The init and backedge supply different origins without changing the tile type."""
    physical_value = "value = pl.tile.full([32, 16], dtype=pl.FP32, value=0.0)"
    explicit_value = (
        "received: pl.Tile[[32, 16], pl.FP32, pl.Mem.Vec] = "
        "pl.system.tpop_from_aic(split=1, aiv_subblockid=0)\n"
        "value = pl.tile.add(received, 1.0)\n"
        "pl.system.tfree_to_aic(received)"
    )
    init, backedge = (physical_value, explicit_value) if pop_on_backedge else (explicit_value, physical_value)
    init = init.replace("\n", "\n        ")
    backedge = backedge.replace("\n", "\n            ")
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def vector(self, steps: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[32, 16], pl.FP32]]):
        pl.func_attr({{"split": pl.SplitMode.UP_DOWN}})
        buf = pl.system.reserve_buffer(name="c2v", size=65536)
        pl.system.aiv_initialize_pipe(dir_mask=1, slot_size=4096, slot_num=16, c2v_consumer_buf=buf)
        {init}
        for step in pl.range(steps):
            inner = pl.tile.set_validshape(value, 31, 16)
            out = pl.tile.store(inner, [0, 0], out)
            {backedge}
        outer = pl.tile.set_validshape(value, 31, 16)
        return pl.tile.store(outer, [0, 0], out)
""")
    program = _run_split_vector_kernel(before)
    loops = []
    narrowed = []
    yields = []
    physical = []

    class Collect(ir.IRVisitor):
        def visit_for_stmt(self, op):
            loops.append(op)
            super().visit_for_stmt(op)

        def visit_yield_stmt(self, op):
            yields.append(op)
            super().visit_yield_stmt(op)

        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call):
                if op.value.op.name == ir.get_op("tile.set_validshape").name:
                    narrowed.append(op.var)
                elif op.value.op.name == ir.get_op("tile.get_subblock_idx").name:
                    physical.append(op.var)
            super().visit_assign_stmt(op)

    Collect().visit_program(program)
    assert len(loops) == len(yields) == len(physical) == 1
    assert len(narrowed) == 2
    lane = loops[0].iter_args[-1]
    exit_lane = loops[0].return_vars[-1]
    assert isinstance(lane.type, ir.ScalarType) and lane.type.dtype == pl.INDEX
    backedge_lane = yields[0].value[-1]
    if pop_on_backedge:
        ir.assert_structural_equal(lane.initValue, physical[0])
        _assert_extent_after_substitution(backedge_lane, {}, 0)
    else:
        _assert_extent_after_substitution(lane.initValue, {}, 0)
        ir.assert_structural_equal(backedge_lane, physical[0])
    for tile, selector in zip(narrowed, (lane, exit_lane), strict=True):
        assert isinstance(tile.type, ir.TileType)
        extent = tile.type.get_effective_tile_view().valid_shape[0]
        _assert_extent_after_substitution(extent, {selector.name_hint: 0}, 16)
        _assert_extent_after_substitution(extent, {selector.name_hint: 1}, 15)


def test_explicit_pop_leaves_unrelated_loop_carry_without_a_lane_companion():
    """A pop stored directly to the output has no effect on a separate tile carry."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def vector(self, steps: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[32, 16], pl.FP32]]):
            pl.func_attr({"split": pl.SplitMode.UP_DOWN})
            buf = pl.system.reserve_buffer(name="c2v", size=65536)
            pl.system.aiv_initialize_pipe(dir_mask=1, slot_size=4096, slot_num=16, c2v_consumer_buf=buf)
            value = pl.tile.full([32, 16], dtype=pl.FP32, value=0.0)
            for step in pl.range(steps):
                received: pl.Tile[[32, 16], pl.FP32, pl.Mem.Vec] = pl.system.tpop_from_aic(
                    split=1, aiv_subblockid=0
                )
                out = pl.tile.store(received, [0, 0], out)
                pl.system.tfree_to_aic(received)
                if step == 0:
                    value = pl.tile.add(value, 1.0)
                else:
                    value = pl.tile.full([32, 16], dtype=pl.FP32, value=2.0)
            return pl.tile.store(value, [0, 0], out)

    program = _run_split_vector_kernel(Before)
    vector = program.get_function("vector")
    assert vector is not None
    assert isinstance(vector.body, ir.SeqStmts)
    loop = next(stmt for stmt in vector.body.stmts if isinstance(stmt, ir.ForStmt))
    assert len(loop.iter_args) == 2
    assert all(not isinstance(arg.type, ir.ScalarType) for arg in loop.iter_args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
