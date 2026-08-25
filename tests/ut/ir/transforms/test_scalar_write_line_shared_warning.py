# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Warning for multi-block ``pl.write`` that may share a 64-byte cache line.

A scalar write reaches DDR one whole 64-byte line at a time, carrying the
neighbouring elements as the writing block last saw them. Two ``pl.spmd`` blocks
writing *different* elements of one line therefore lose each other's stores,
silently and non-deterministically -- disjoint indices are not enough.

The check proves the negation: each block must own whole, private 64-byte lines.
It reports two distinguishable outcomes -- an index it can analyse and show is
interleaved, and an index it cannot analyse at all.
"""

import pypto.language as pl
import pytest
from pypto import passes

_RULE = "ScalarWriteLineShared"

N = 64  # INT32 -> 16 elements per 64-byte line
LINE = 16
BLOCKS = 24


def _warnings(program) -> list:
    """Run the pre-pipeline warning checks and keep only this rule's output."""
    checks = passes.DiagnosticCheckRegistry.get_warning_checks()
    diagnostics = passes.DiagnosticCheckRegistry.run_checks(
        checks, passes.DiagnosticPhase.PRE_PIPELINE, program
    )
    return [d for d in diagnostics if d.rule_name == _RULE]


class TestInterleavedIsReported:
    """An index the model can analyse, and show lands inside a shared line."""

    def test_grid_strided_fill(self):
        """The repro shape: blocks 0..15 all write inside out[0:16]."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="fill"):
                    blk = pl.tile.get_block_idx()
                    for i in pl.range(pl.cast(blk, pl.INDEX), N, BLOCKS):
                        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        message = found[0].message
        # The analysable case names the measured stride and the sharing factor,
        # rather than asking the author to go and check.
        assert "4 bytes apart" in message
        assert "16 of them share each 64-byte cache line" in message
        assert "'out'" in message and "fill" in message
        assert "16 x INT32" in message
        assert "24 concurrent blocks (" in message

    def test_block_owns_half_a_line(self):
        """A clean power-of-two stride that is still under 64 bytes."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(8, name_hint="half"):
                    blk = pl.tile.get_block_idx()
                    base = pl.cast(blk, pl.INDEX) * 8
                    for i in pl.range(base, base + 8):
                        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        assert "32 bytes apart" in found[0].message


class TestIndeterminateIsReported:
    """An index the model cannot analyse gets the "confirm it yourself" form."""

    def test_index_read_from_another_tensor(self):
        """The shape real plan-building code uses: a runtime row index."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, rows: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="gather"):
                    blk = pl.tile.get_block_idx()
                    for g in pl.range(pl.cast(blk, pl.INDEX), N, BLOCKS):
                        dst = pl.cast(pl.read(rows, [g]), pl.INDEX)
                        pl.write(out, [dst], pl.cast(g, pl.INT32))
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        message = found[0].message
        assert "the index is computed at runtime" in message
        assert "16 x INT32" in message


class TestProvenDisjointIsSilent:
    """Layouts the model can prove safe must not warn."""

    def test_block_owns_exactly_one_line(self):
        """``--aligned`` in the repro: block b owns out[16b : 16b+16]."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[LINE * BLOCKS], pl.INT32],
                out: pl.Out[pl.Tensor[[LINE * BLOCKS], pl.INT32]],
            ) -> pl.Tensor[[LINE * BLOCKS], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="aligned"):
                    blk = pl.tile.get_block_idx()
                    base = pl.cast(blk, pl.INDEX) * LINE
                    for i in pl.range(base, base + LINE):
                        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        assert _warnings(Prog) == []

    def test_block_owns_several_whole_lines(self):
        """A stride of 4 lines, spanning all of them, is still disjoint."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[LINE * 4 * 8], pl.INT32],
                out: pl.Out[pl.Tensor[[LINE * 4 * 8], pl.INT32]],
            ) -> pl.Tensor[[LINE * 4 * 8], pl.INT32]:
                with pl.spmd(8, name_hint="wide"):
                    blk = pl.tile.get_block_idx()
                    base = pl.cast(blk, pl.INDEX) * (LINE * 4)
                    for i in pl.range(base, base + LINE * 4):
                        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        assert _warnings(Prog) == []

    def test_span_overflowing_the_stride_is_reported(self):
        """Stride is a line multiple, but each block reaches into the next."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[LINE * 8 + LINE], pl.INT32],
                out: pl.Out[pl.Tensor[[LINE * 8 + LINE], pl.INT32]],
            ) -> pl.Tensor[[LINE * 8 + LINE], pl.INT32]:
                with pl.spmd(8, name_hint="overlap"):
                    blk = pl.tile.get_block_idx()
                    base = pl.cast(blk, pl.INDEX) * LINE
                    # 4 elements past this block's own line.
                    for i in pl.range(base, base + LINE + 4):
                        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        assert len(_warnings(Prog)) == 1


class TestNegativeCoefficient:
    """A reversed instance->address mapping gives the block index a negative
    coefficient, which the interval arithmetic must orient correctly."""

    def test_reversed_line_ownership_is_silent(self):
        """Block b owns the line at the far end; still one whole line each."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[LINE * BLOCKS], pl.INT32],
                out: pl.Out[pl.Tensor[[LINE * BLOCKS], pl.INT32]],
            ) -> pl.Tensor[[LINE * BLOCKS], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="reversed"):
                    blk = pl.tile.get_block_idx()
                    base = (BLOCKS - 1 - pl.cast(blk, pl.INDEX)) * LINE
                    for i in pl.range(base, base + LINE):
                        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        assert _warnings(Prog) == []

    def test_reversed_interleaved_is_reported(self):
        """Same reversal, but one element per block -- still shares lines."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[N], pl.INT32],
                out: pl.Out[pl.Tensor[[N], pl.INT32]],
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="revfill"):
                    blk = pl.tile.get_block_idx()
                    idx = BLOCKS - 1 - pl.cast(blk, pl.INDEX)
                    pl.write(out, [idx], pl.cast(pl.read(src, [idx]) + 1, pl.INT32))
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        assert "4 bytes apart" in found[0].message


class TestNotApplicable:
    """Shapes the check must leave alone."""

    def test_single_block_is_silent(self):
        """One writer owns every line it touches, at any layout."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(1, name_hint="fill"):
                    blk = pl.tile.get_block_idx()
                    for i in pl.range(pl.cast(blk, pl.INDEX), N, 1):
                        pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        assert _warnings(Prog) == []

    def test_tile_write_is_silent(self):
        """A Tile is on-chip and never written back through the data cache."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, out: pl.Out[pl.Tensor[[BLOCKS, LINE], pl.INT32]]
            ) -> pl.Tensor[[BLOCKS, LINE], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="tilewrite"):
                    blk = pl.tile.get_block_idx()
                    t = pl.tile.full([1, LINE], dtype=pl.INT32, value=0)
                    for c in pl.range(LINE):
                        pl.write(t, [0, c], pl.cast(c, pl.INT32))
                    out = pl.store(t, [pl.cast(blk, pl.INDEX), 0], out)
                return out

        assert _warnings(Prog) == []

    def test_scope_local_tensor_is_silent(self):
        """A buffer created inside the body is not visible to another block."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="scratch"):
                    blk = pl.tile.get_block_idx()
                    buf = pl.create_tensor([N], dtype=pl.INT32)
                    for i in pl.range(pl.cast(blk, pl.INDEX), N, BLOCKS):
                        pl.write(buf, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        assert _warnings(Prog) == []

    def test_hoisted_tensor_is_reported(self):
        """The ``--internal`` variant: a buffer declared outside the scope."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                buf = pl.create_tensor([N], dtype=pl.INT32)
                with pl.spmd(BLOCKS, name_hint="fill"):
                    blk = pl.tile.get_block_idx()
                    for i in pl.range(pl.cast(blk, pl.INDEX), N, BLOCKS):
                        pl.write(buf, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                with pl.spmd(1, name_hint="copy"):
                    # An spmd body must read the block index or dispatch a kernel.
                    _ = pl.tile.get_block_idx()
                    for j in pl.range(N):
                        pl.write(out, [j], pl.read(buf, [j]))
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        assert "'buf'" in found[0].message


class TestPerWriteReporting:
    """Each unprovable write is reported on its own."""

    def test_two_writes_report_twice(self):
        @pl.program
        class Prog:
            @pl.function
            def main(
                self, rows: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="gather"):
                    blk = pl.tile.get_block_idx()
                    for g in pl.range(pl.cast(blk, pl.INDEX), N, BLOCKS):
                        dst = pl.cast(pl.read(rows, [g]), pl.INDEX)
                        alt = pl.cast(pl.read(rows, [g]) + 1, pl.INDEX)
                        pl.write(out, [dst], pl.cast(g, pl.INT32))
                        pl.write(out, [alt], pl.cast(g, pl.INT32))
                return out

        assert len(_warnings(Prog)) == 2


class TestDtypeWidth:
    """A line holds a different number of elements per dtype."""

    def test_fp16_reports_32_per_line(self):
        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[128], pl.FP16], out: pl.Out[pl.Tensor[[128], pl.FP16]]
            ) -> pl.Tensor[[128], pl.FP16]:
                with pl.spmd(4, name_hint="fill"):
                    blk = pl.tile.get_block_idx()
                    for i in pl.range(pl.cast(blk, pl.INDEX), 128, 4):
                        pl.write(out, [i], pl.read(src, [i]))
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        assert "32 x FP16" in found[0].message
        assert "2 bytes apart" in found[0].message

    def test_fp16_block_owns_32_elements(self):
        """32 FP16 is exactly one line -- must be silent."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[128], pl.FP16], out: pl.Out[pl.Tensor[[128], pl.FP16]]
            ) -> pl.Tensor[[128], pl.FP16]:
                with pl.spmd(4, name_hint="aligned"):
                    blk = pl.tile.get_block_idx()
                    base = pl.cast(blk, pl.INDEX) * 32
                    for i in pl.range(base, base + 32):
                        pl.write(out, [i], pl.read(src, [i]))
                return out

        assert _warnings(Prog) == []


class TestMultiDimensionalIndex:
    """The flat offset must fold row-major strides."""

    def test_row_per_block_is_silent(self):
        """Each block owns one 16-INT32 row = exactly one line."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[BLOCKS, LINE], pl.INT32],
                out: pl.Out[pl.Tensor[[BLOCKS, LINE], pl.INT32]],
            ) -> pl.Tensor[[BLOCKS, LINE], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="rows"):
                    blk = pl.tile.get_block_idx()
                    r = pl.cast(blk, pl.INDEX)
                    for c in pl.range(LINE):
                        pl.write(out, [r, c], pl.read(src, [r, c]))
                return out

        assert _warnings(Prog) == []

    def test_column_per_block_is_reported(self):
        """Down a column of a row-major tensor, consecutive blocks are one
        element apart -- the textbook false-sharing layout."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                src: pl.Tensor[[LINE, BLOCKS], pl.INT32],
                out: pl.Out[pl.Tensor[[LINE, BLOCKS], pl.INT32]],
            ) -> pl.Tensor[[LINE, BLOCKS], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="cols"):
                    blk = pl.tile.get_block_idx()
                    c = pl.cast(blk, pl.INDEX)
                    for r in pl.range(LINE):
                        pl.write(out, [r, c], pl.read(src, [r, c]))
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        assert "4 bytes apart" in found[0].message


class TestInstanceSources:
    """Multiplicity comes from instance count, not from any one construct."""

    def test_parallel_loop_is_an_instance_dimension(self):
        """``pl.parallel`` iterations are concurrent task instances."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                for g in pl.parallel(8):
                    with pl.spmd(1, name_hint="w"):
                        _ = pl.tile.get_block_idx()
                        pl.write(out, [g], pl.cast(pl.read(src, [g]) + 1, pl.INT32))
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        message = found[0].message
        assert "8 concurrent task instances (pl.parallel)" in message
        assert "4 bytes apart" in message

    def test_parallel_loop_owning_whole_lines_is_silent(self):
        """Instance b owns out[16b : 16b+16] -- one whole line each."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[LINE * 8], pl.INT32], out: pl.Out[pl.Tensor[[LINE * 8], pl.INT32]]
            ) -> pl.Tensor[[LINE * 8], pl.INT32]:
                for g in pl.parallel(8):
                    with pl.spmd(1, name_hint="w"):
                        _ = pl.tile.get_block_idx()
                        base = g * LINE
                        for i in pl.range(base, base + LINE):
                            pl.write(out, [i], pl.cast(pl.read(src, [i]) + 1, pl.INT32))
                return out

        assert _warnings(Prog) == []

    def test_dispatched_kernel_inherits_multiplicity(self):
        """The writes live in the callee; the instance count is at the call site."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, rows: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                blk = pl.tile.get_block_idx()
                for g in pl.range(pl.cast(blk, pl.INDEX), N, BLOCKS):
                    dst = pl.cast(pl.read(rows, [g]), pl.INDEX)
                    pl.write(out, [dst], pl.cast(g, pl.INT32))
                return out

            @pl.function
            def main(
                self, rows: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(BLOCKS, name_hint="disp"):
                    out = self.kernel(rows, out)
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        message = found[0].message
        assert "dispatched from a multi-instance scope" in message
        assert "in function 'kernel'" in message


class TestSingleInstanceIsExempt:
    """A single instance runs its body sequentially on one core -- no check."""

    def test_incore_scope_with_runtime_index_is_silent(self):
        """The unprovable index that would report under pl.spmd(24)."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, rows: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    for g in pl.range(N):
                        dst = pl.cast(pl.read(rows, [g]), pl.INDEX)
                        pl.write(out, [dst], pl.cast(g, pl.INT32))
                return out

        assert _warnings(Prog) == []

    def test_sequential_loop_is_not_an_instance_dimension(self):
        """``pl.range`` iterates in order on one core."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self, src: pl.Tensor[[N], pl.INT32], out: pl.Out[pl.Tensor[[N], pl.INT32]]
            ) -> pl.Tensor[[N], pl.INT32]:
                with pl.spmd(1, name_hint="w"):
                    _ = pl.tile.get_block_idx()
                    for g in pl.range(N):
                        pl.write(out, [g], pl.cast(pl.read(src, [g]) + 1, pl.INT32))
                return out

        assert _warnings(Prog) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
