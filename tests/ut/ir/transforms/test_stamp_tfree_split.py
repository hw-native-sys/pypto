# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the StampTfreeSplit pass.

The pass copies each cross-core tpop's ``split`` (and pipe ``id``) onto its
matching ``tfree`` op so codegen reads them directly. It also performs the
tpop/tfree direction and pipe-id consistency checks (moved out of codegen).
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _stamp(program):
    """Run convert_to_ssa then stamp_tfree_split, returning the transformed program.

    Runs under the ambient ``conftest`` context (BEFORE_AND_AFTER property
    verification + the print->parse roundtrip instrument), so the pass output is
    checked for free on top of the structural comparison each test performs.
    """
    ssa = passes.convert_to_ssa()(program)
    return passes.stamp_tfree_split()(ssa)


def _run_prereqs_only(program):
    """Normalize a hand-written ``Expected`` with the prerequisite pass only.

    ``StampTfreeSplit`` runs after ``ConvertToSSA``, so ``Expected`` is written
    in pre-SSA DSL form and lifted the same way -- without ever running the pass
    under test, which would make the golden self-referential.
    """
    return passes.convert_to_ssa()(program)


def test_tfree_gets_split_from_tpop():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=2)
            pl.tfree_to_aic(t)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=2)
            # split copied off the originating tpop by the pass.
            pl.tfree_to_aic(t, split=2)

    After = _stamp(Before)
    ir.assert_structural_equal(After, _run_prereqs_only(Expected))


def test_tfree_gets_split_and_id_from_tpop():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf, id=3)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1, id=3)
            pl.tfree_to_aic(t)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf, id=3)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1, id=3)
            # both split and the pipe id are copied off the originating tpop.
            pl.tfree_to_aic(t, split=1, id=3)

    After = _stamp(Before)
    ir.assert_structural_equal(After, _run_prereqs_only(Expected))


def test_tfree_id_mismatch_raises():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf, id=3)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1, id=3)
            pl.tfree_to_aic(t, id=0)

    with pytest.raises(ValueError, match="does not match originating"):
        _stamp(Before)


def test_tfree_direction_mismatch_raises():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            pl.tfree_to_aiv(t)

    with pytest.raises(ValueError, match="requires its tile argument to come from"):
        _stamp(Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
