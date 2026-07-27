# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verifier tests for explicit tile buffer-slot leases."""

import pypto.language as pl
import pytest
from pypto import passes


def test_lifetime_verifier_rejects_use_after_release():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(self):
            buffers = pl.create_tile_buffers(2, [16, 128], pl.FP32, pl.Mem.Acc)
            slot = buffers[0]
            lhs = pl.create_tile([16, 32], pl.BF16, pl.Mem.Left)
            rhs = pl.create_tile([32, 128], pl.BF16, pl.Mem.Right)
            pl.tile.release(slot)
            _bad = pl.tile.matmul(lhs, rhs, out=slot)

    with pytest.raises(Exception, match="released.*slot|slot.*released"):
        passes.verify_tile_buffer_lifetime(Program)


def test_lifetime_verifier_rejects_release_of_ordinary_tile():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(self):
            ordinary = pl.create_tile([16, 128], pl.FP32, pl.Mem.Acc)
            pl.tile.release(ordinary)

    with pytest.raises(Exception, match="release.*selected slot"):
        passes.verify_tile_buffer_lifetime(Program)


def test_lifetime_verifier_tracks_destination_result_alias():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(self):
            buffers = pl.create_tile_buffers(2, [16, 128], pl.FP32, pl.Mem.Acc)
            slot = buffers[0]
            lhs = pl.create_tile([16, 32], pl.BF16, pl.Mem.Left)
            rhs = pl.create_tile([32, 128], pl.BF16, pl.Mem.Right)
            result = pl.tile.matmul(lhs, rhs, out=slot)
            pl.tile.release(result)
            _bad = pl.tile.matmul_acc(result, lhs, rhs, out=slot)

    with pytest.raises(Exception, match="released.*result|result.*released"):
        passes.verify_tile_buffer_lifetime(Program)


def test_lifetime_verifier_is_a_structural_property():
    structural = passes.get_structural_properties()
    assert structural.contains(passes.IRProperty.TileBufferLifetimeValid)
