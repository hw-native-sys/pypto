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
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.ir.printer import python_print


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _stamp(program) -> str:
    """Run convert_to_ssa then stamp_tfree_split (no verification) and print."""
    ssa = passes.convert_to_ssa()(program)
    with passes.PassContext([]):
        after = passes.stamp_tfree_split()(ssa)
    return python_print(after)


def _tfree_line(text: str) -> str:
    return next(line.strip() for line in text.splitlines() if "tfree_to_ai" in line)


def test_tfree_gets_split_from_tpop():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=2)
            pl.tfree_to_aic(t)

    line = _tfree_line(_stamp(Prog))
    assert "tfree_to_aic" in line
    assert "split=2" in line, line


def test_tfree_gets_split_and_id_from_tpop():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf, id=3)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1, id=3)
            pl.tfree_to_aic(t)

    line = _tfree_line(_stamp(Prog))
    assert "split=1" in line and "id=3" in line, line


def test_tfree_id_mismatch_raises():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf, id=3)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=1, id=3)
            pl.tfree_to_aic(t, id=0)

    with pytest.raises(Exception, match="does not match originating"):
        _stamp(Prog)


def test_tfree_direction_mismatch_raises():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def consumer(self):
            buf = pl.reserve_buffer(name="c2v", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=buf)
            t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            pl.tfree_to_aiv(t)

    with pytest.raises(Exception, match="requires its tile argument to come from"):
        _stamp(Prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
