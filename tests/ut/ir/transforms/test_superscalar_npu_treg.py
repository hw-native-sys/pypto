# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end tests for the SuperscalarNPU TREG pass strategy.

The strategy lowers tensor ops to tile ops, assigns the TREG register file as
the on-chip memory space, and allocates TREG block indices via register
renaming (MemoryReuse coalescing + the SuperscalarNPU allocator policy).
Codegen is not implemented for this backend, so these tests stop after the
passes and inspect the printed IR.
"""

import pypto.language as pl
import pytest
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.ir.printer import python_print

from pypto import backend, passes

# 128x128 FP32 tile = 64 KiB = 16 TREG blocks (4 KiB each).
_TILE_BYTES = 128 * 128 * 4
_BLOCKS_PER_TILE = _TILE_BYTES // 4096  # 16


@pytest.fixture(autouse=True)
def _superscalar_backend():
    """Run every test in this file under the SuperscalarNPU backend."""
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.SuperscalarNPU)
    try:
        yield
    finally:
        backend.reset_for_testing()


def _run(program):
    pm = PassManager.get_strategy(OptimizationStrategy.SuperscalarNPU)
    # Codegen is unimplemented and the truncated pipeline does not establish
    # every property a full Ascend run would, so verification is disabled.
    with passes.PassContext([], passes.VerificationLevel.NONE):
        return pm.run_passes(program)


class TestElementwiseLowering:
    """A simple add kernel lowers to TREG tiles with register-renamed blocks."""

    def test_treg_assignment_and_register_renaming(self):
        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    ta = pl.load(a, [0, 0], [128, 128])
                    tb = pl.load(b, [0, 0], [128, 128])
                    tc = pl.add(ta, tb)
                    pl.store(tc, [0, 0], out)
                return out

        text = python_print(_run(Prog), prefix="pl")

        # On-chip tiles live in TREG; SuperscalarNPU has no Vec/Mat at all.
        assert "pl.Mem.TREG" in text
        assert "pl.Mem.Vec" not in text
        assert "pl.Mem.Mat" not in text

        # Register renaming: ta, tb, tc need only two live registers because the
        # MemoryReuse pass coalesces tc back onto ta after ta's last use, so only
        # two TREG allocations survive (not three).
        assert text.count("pl.tile.alloc(pl.Mem.TREG") == 2

    def test_addresses_are_block_indices(self):
        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    ta = pl.load(a, [0, 0], [128, 128])
                    tb = pl.load(b, [0, 0], [128, 128])
                    tc = pl.add(ta, tb)
                    pl.store(tc, [0, 0], out)
                return out

        text = python_print(_run(Prog), prefix="pl")

        # TREG addresses are *block indices*, not byte addresses. The first tile
        # lands on block 0; the second starts at block 16 (one tile = 16 blocks),
        # NOT at byte offset 65536.
        assert f"pl.const({_BLOCKS_PER_TILE}, pl.INT64), {_TILE_BYTES})" in text
        # A byte-addressed second tile would read pl.const(65536, ...); ensure no
        # TREG MemRef uses the byte offset as its address.
        assert f"pl.const({_TILE_BYTES}, pl.INT64)" not in text


class TestRegisterPressure:
    """Exceeding 256 live TREG blocks raises a user-facing error."""

    def test_pressure_exceeded_raises(self):
        # 512x512 FP32 = 1 MiB = 256 blocks per tile; the two loaded operands are
        # live simultaneously at the add, needing 512 blocks > 256 available.
        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 512], pl.FP32],
                b: pl.Tensor[[512, 512], pl.FP32],
            ) -> pl.Tensor[[512, 512], pl.FP32]:
                out: pl.Tensor[[512, 512], pl.FP32] = pl.create_tensor([512, 512], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP):
                    ta = pl.load(a, [0, 0], [512, 512])
                    tb = pl.load(b, [0, 0], [512, 512])
                    tc = pl.add(ta, tb)
                    pl.store(tc, [0, 0], out)
                return out

        with pytest.raises(ValueError, match="Register pressure exceeded in TREG"):
            _run(Prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
