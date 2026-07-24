# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for system operation DSL parsing and round-trip."""

import re

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir.op import system_ops
from pypto.pypto_core import DataType


class TestSystemOpsParsing:
    """Tests for parsing pl.system.* operations in the DSL."""

    def test_sync_src_round_trip(self):
        """Test round-trip for pl.system.sync_src."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                return x

        printed = Before.as_python()
        assert "pl.system.sync_src(" in printed
        assert "set_pipe=pl.PipeType.MTE2" in printed
        assert "wait_pipe=pl.PipeType.V" in printed
        assert "event_id=0" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_sync_dst_round_trip(self):
        """Test round-trip for pl.system.sync_dst."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                return x

        printed = Before.as_python()
        assert "pl.system.sync_dst(" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_cross_core_sync_static_round_trip(self):
        """Static cross-core event ids and pipe enums survive Python printing."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_set(event_id=3, pipe=pl.PipeType.MTE3, ffts_mode=1)
                pl.system.sync_wait(event_id=3, pipe=pl.PipeType.MTE2)
                return x

        printed = Before.as_python()
        assert "pl.system.sync_set(pipe=pl.PipeType.MTE3, event_id=3, ffts_mode=1)" in printed
        assert "pl.system.sync_wait(pipe=pl.PipeType.MTE2, event_id=3)" in printed

        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_set_ffts_round_trip(self):
        """The A3 FFTS workspace setup survives Python printing and reparsing."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main(
                self,
                ffts_workspace: pl.Tensor[[256], pl.INT64],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                pl.system.set_ffts(ffts_workspace)
                pl.system.sync_wait(event_id=3, pipe=pl.PipeType.MTE2)
                return x

        printed = Before.as_python()
        assert "pl.system.set_ffts(ffts_workspace)" in printed

        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_cross_core_sync_dynamic_round_trip(self):
        """An index SSA event id is printed as the dynamic sync operand."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def main(
                self, x: pl.Tensor[[64], pl.FP32], event_id: pl.Scalar[pl.INDEX]
            ) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_set(event_id, pipe=pl.PipeType.MTE3)
                pl.system.sync_wait(event_id, pipe=pl.PipeType.MTE2)
                return x

        printed = Before.as_python()
        assert "pl.system.sync_set(event_id, pipe=pl.PipeType.MTE3)" in printed
        assert "pl.system.sync_wait(event_id, pipe=pl.PipeType.MTE2)" in printed

        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_bar_v_round_trip(self):
        """Test round-trip for pl.system.bar_v."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.bar_v()
                return x

        printed = Before.as_python()
        assert "pl.system.bar_v()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_bar_m_round_trip(self):
        """Test round-trip for pl.system.bar_m."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.bar_m()
                return x

        printed = Before.as_python()
        assert "pl.system.bar_m()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_bar_all_round_trip(self):
        """Test round-trip for pl.system.bar_all."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.bar_all()
                return x

        printed = Before.as_python()
        assert "pl.system.bar_all()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_fence_round_trip(self):
        """Test round-trip for pl.system.fence."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.fence()
                return x

        printed = Before.as_python()
        assert "pl.system.fence()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_cacheinvalid_round_trip_scalar(self):
        """Round-trip for the scalar-write (all-ones shapes) form."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.cacheinvalid(x, [1], [4])
                return x

        printed = Before.as_python()
        assert "pl.system.cacheinvalid(x, [1], [4])" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_cacheinvalid_round_trip_region(self):
        """Round-trip for the partition-view (region) form."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
                pl.system.cacheinvalid(x, [16, 16], [0, 0])
                return x

        printed = Before.as_python()
        assert "pl.system.cacheinvalid(x, [16, 16], [0, 0])" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_cacheinvalid_rejects_float_offset(self):
        """A non-integer offset is rejected at the IR wrapper, not deep in codegen."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(64, DataType.INT32, span)
        tensor = ir.Var("x", ir.TensorType([dim], DataType.FP32), span)
        with pytest.raises(TypeError, match="offsets must be integers"):
            system_ops.cacheinvalid(tensor, [1], [3.5])  # type: ignore[list-item]  # intentionally wrong type

    def test_cacheinvalid_rejects_rank_mismatch(self):
        """offsets/shapes length must match the tensor rank."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(16, DataType.INT32, span)
        tensor = ir.Var("x", ir.TensorType([dim, dim], DataType.FP32), span)
        with pytest.raises(ValueError, match="offsets must match tensor rank 2"):
            system_ops.cacheinvalid(tensor, [1, 1], [0])

    def test_syncall_round_trip(self):
        """Test round-trip for pl.system.syncall with an explicit core_type."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.syncall(core_type="aiv_only")
                return x

        printed = Before.as_python()
        assert re.search(r"""pl\.system\.syncall\(core_type=(["'])aiv_only\1\)""", printed)

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_syncall_default_core_type_round_trip(self):
        """Test round-trip for pl.system.syncall with the default core_type."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.syncall()
                return x

        printed = Before.as_python()
        assert re.search(r"""pl\.system\.syncall\(core_type=(["'])mix\1\)""", printed)

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_syncall_invalid_core_type_raises(self):
        """syncall rejects an unknown core_type at construction time."""
        with pytest.raises(ValueError, match="core_type"):
            pl.system.syncall(core_type="bogus")

    def test_syncall_soft_round_trip(self):
        """Round-trip for the soft (GM-polling) form of pl.system.syncall."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                x: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Tensor[[512, 128], pl.FP32],
                ws: pl.Tensor[[32], pl.INT32],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                off = pl.tile.get_block_idx() * 128
                t = pl.load(x, [off, 0], [128, 128])
                pl.system.syncall(mode="soft", core_type="aiv_only", gm_workspace=ws, used_cores=4)
                return pl.store(t, [off, 0], out)

        printed = Before.as_python()
        # The high-level surface (mode/core_type/gm_workspace/used_cores) round-trips;
        # the synthesized scratch is threaded back via the internal scratch= kwarg.
        assert 'pl.system.syncall(mode="soft", core_type="aiv_only", gm_workspace=ws, used_cores=4' in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_syncall_soft_validation(self):
        """Soft syncall validates mode, core_type, gm_workspace, and used_cores."""
        with pytest.raises(ValueError, match="mode"):
            pl.system.syncall(mode="bogus")
        # An unknown core_type is rejected.
        with pytest.raises(ValueError, match="core_type"):
            pl.system.syncall(mode="soft", core_type="bogus_type", used_cores=4)
        # aiv_only, aic_only, and mix are all supported; each still requires a
        # shared gm_workspace.
        for ct in ("aiv_only", "aic_only", "mix"):
            with pytest.raises(ValueError, match="gm_workspace"):
                pl.system.syncall(mode="soft", core_type=ct, used_cores=4)

    def test_multiple_system_ops_round_trip(self):
        """Test round-trip with multiple system ops in a single function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.bar_v()
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.bar_all()
                return x

        printed = Before.as_python()
        assert "pl.system.sync_src(" in printed
        assert "pl.system.bar_v()" in printed
        assert "pl.system.sync_dst(" in printed
        assert "pl.system.bar_all()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_sync_with_different_pipe_types(self):
        """Test sync ops with various PipeType enum values."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
                return x

        printed = Before.as_python()
        assert "pl.PipeType.MTE1" in printed
        assert "pl.PipeType.M" in printed
        assert "pl.PipeType.MTE3" in printed
        assert "pl.PipeType.S" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)


class TestCrossCoreParsing:
    """Tests for parsing pl.system.* cross-core operations via print-parse round-trip."""

    def _build_program_with_system_stmt(self, stmt_code: str) -> ir.Program:
        """Build a program from printed IR text containing a system op statement."""
        program_text = f"""\
import pypto.language as pl

@pl.program
class test_program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        {stmt_code}
        return x
"""
        return pl.parse_program(program_text)

    def test_aic_initialize_pipe_round_trip(self):
        """Test round-trip for pl.system.aic_initialize_pipe."""
        prog = self._build_program_with_system_stmt(
            "pl.system.aic_initialize_pipe(dir_mask=1, slot_size=256)"
        )
        printed = prog.as_python()
        assert "pl.system.aic_initialize_pipe(" in printed
        assert "dir_mask=1" in printed
        assert "slot_size=256" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_aiv_initialize_pipe_round_trip(self):
        """Test round-trip for pl.system.aiv_initialize_pipe."""
        prog = self._build_program_with_system_stmt(
            "pl.system.aiv_initialize_pipe(dir_mask=2, slot_size=512)"
        )
        printed = prog.as_python()
        assert "pl.system.aiv_initialize_pipe(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_initialize_pipe_explicit_slot_num_round_trip(self):
        """Explicit slot_num / local_slot_num attributes should round-trip through print/parse."""
        prog = self._build_program_with_system_stmt(
            "pl.system.aic_initialize_pipe(dir_mask=1, slot_size=256, slot_num=16, local_slot_num=4)"
        )
        printed = prog.as_python()
        assert "slot_num=16" in printed
        assert "local_slot_num=4" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_reserve_buffer_round_trip(self):
        """Test round-trip for pl.system.reserve_buffer."""
        prog = self._build_program_with_system_stmt('pl.system.reserve_buffer(name="shared_buf", size=1024)')
        printed = prog.as_python()
        assert "pl.system.reserve_buffer(" in printed
        assert "shared_buf" in printed
        assert "size=1024" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_import_peer_buffer_round_trip(self):
        """Test round-trip for pl.system.import_peer_buffer."""
        prog = self._build_program_with_system_stmt(
            'pl.system.import_peer_buffer(name="shared_buf", peer_func="aiv_kernel")'
        )
        printed = prog.as_python()
        assert "pl.system.import_peer_buffer(" in printed
        assert "shared_buf" in printed
        assert "aiv_kernel" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_tpush_to_aiv_round_trip(self):
        """Test round-trip for pl.tile.tpush_to_aiv with tile param."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC)
            def kernel(self, t: pl.Tile[[64], pl.FP32]) -> pl.Tile[[64], pl.FP32]:
                pl.tile.tpush_to_aiv(t, split=0)
                return t

        printed = Before.as_python()
        assert "pl.tile.tpush_to_aiv(" in printed
        assert "split=0" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_tpush_to_aic_round_trip(self):
        """Test round-trip for pl.tile.tpush_to_aic with tile param."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(self, t: pl.Tile[[64], pl.FP32]) -> pl.Tile[[64], pl.FP32]:
                pl.tile.tpush_to_aic(t, split=0)
                return t

        printed = Before.as_python()
        assert "pl.tile.tpush_to_aic(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_tpop_from_aic_round_trip(self):
        """Test round-trip for pl.tile.tpop_from_aic (zero-arg op)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(self) -> pl.Tile[[64], pl.FP32]:
                received: pl.Tile[[64], pl.FP32] = pl.tile.tpop_from_aic()
                return received

        printed = Before.as_python()
        assert "pl.tile.tpop_from_aic(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_tpop_from_aiv_round_trip(self):
        """Test round-trip for pl.tile.tpop_from_aiv (zero-arg op)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC)
            def kernel(self) -> pl.Tile[[64], pl.FP32]:
                received: pl.Tile[[64], pl.FP32] = pl.tile.tpop_from_aiv()
                return received

        printed = Before.as_python()
        assert "pl.tile.tpop_from_aiv(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_short_alias_tpush_to_aic(self):
        """Test pl.tpush_to_aic short alias for pl.tile.tpush_to_aic."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(self, t: pl.Tile[[64], pl.FP32]) -> pl.Tile[[64], pl.FP32]:
                pl.tpush_to_aic(t, split=0)
                return t

        printed = Before.as_python()
        assert "pl.tile.tpush_to_aic(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_short_alias_tpop_from_aic(self):
        """Test pl.tpop_from_aic short alias for pl.tile.tpop_from_aic."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(self) -> pl.Tile[[64], pl.FP32]:
                received: pl.Tile[[64], pl.FP32] = pl.tpop_from_aic()
                return received

        printed = Before.as_python()
        assert "pl.tile.tpop_from_aic(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_short_alias_aic_initialize_pipe(self):
        """Test pl.aic_initialize_pipe short alias."""
        prog = self._build_program_with_system_stmt("pl.aic_initialize_pipe(dir_mask=1, slot_size=256)")
        printed = prog.as_python()
        assert "pl.system.aic_initialize_pipe(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_short_alias_reserve_buffer(self):
        """Test pl.reserve_buffer short alias."""
        prog = self._build_program_with_system_stmt('pl.reserve_buffer(name="buf", size=512)')
        printed = prog.as_python()
        assert "pl.system.reserve_buffer(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_short_alias_tpush_to_aiv(self):
        """Test pl.tpush_to_aiv short alias."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC)
            def kernel(self, t: pl.Tile[[64], pl.FP32]) -> pl.Tile[[64], pl.FP32]:
                pl.tpush_to_aiv(t, split=0)
                return t

        printed = Before.as_python()
        assert "pl.tile.tpush_to_aiv(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_short_alias_tpop_from_aiv(self):
        """Test pl.tpop_from_aiv short alias."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC)
            def kernel(self) -> pl.Tile[[64], pl.FP32]:
                received: pl.Tile[[64], pl.FP32] = pl.tpop_from_aiv()
                return received

        printed = Before.as_python()
        assert "pl.tile.tpop_from_aiv(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_short_alias_aiv_initialize_pipe(self):
        """Test pl.aiv_initialize_pipe short alias."""
        prog = self._build_program_with_system_stmt("pl.aiv_initialize_pipe(dir_mask=2, slot_size=512)")
        printed = prog.as_python()
        assert "pl.system.aiv_initialize_pipe(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_short_alias_import_peer_buffer(self):
        """Test pl.import_peer_buffer short alias."""
        prog = self._build_program_with_system_stmt(
            'pl.import_peer_buffer(name="buf", peer_func="aic_kernel")'
        )
        printed = prog.as_python()
        assert "pl.system.import_peer_buffer(" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(prog, reparsed)

    def test_function_type_aic_round_trip(self):
        """Test round-trip for @pl.function(type=pl.FunctionType.AIC) decorator."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIC)
            def aic_kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        printed = Before.as_python()
        assert "@pl.function(type=pl.FunctionType.AIC" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_function_type_aiv_round_trip(self):
        """Test round-trip for @pl.function(type=pl.FunctionType.AIV) decorator."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def aiv_kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        printed = Before.as_python()
        assert "@pl.function(type=pl.FunctionType.AIV" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)

    def test_function_type_group_round_trip(self):
        """Test round-trip for @pl.function(type=pl.FunctionType.Group) decorator."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Group)
            def group_func(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

        printed = Before.as_python()
        assert "@pl.function(type=pl.FunctionType.Group" in printed
        reparsed = pl.parse_program(printed)
        ir.assert_structural_equal(Before, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
