# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen smoke tests for dynamic valid_shape (single-block @pl.jit kernel).

The pre-JIT version of this test exercised a per-block loop with an in-DSL
``if/else`` that selected ``vlen`` per iteration.  In the @pl.jit world the
specializer's alpha-renamer rewrites the rebinding of ``vlen`` in the
else-branch to a distinct alias, which then fails ``ConvertToSSA`` ("used
outside its defining scope").  The current recommended workaround --
documented in ``examples/kernels/09_dyn_valid_shape.py`` -- is to push the
per-call/per-iteration choice of ``vlen`` to the caller and pass a single
scalar parameter.

These tests verify that the JIT pipeline (specialize + full pass pipeline)
succeeds for both vlen values that previously appeared inside the if/else:

  * full-block vlen (= BLOCK_COL): ``valid_shape`` matches the physical
    tile shape; ``fillpad`` is a no-op.
  * partial-block vlen (< BLOCK_COL): ``valid_shape`` < physical;
    ``fillpad`` writes the padding region.
"""

import pytest
import torch
from examples.kernels.dyn_valid_shape import BLOCK_COL, Q_TILE, dyn_valid_shape

# Original tests carried this constant for the multi-block tensor row count
# (2 blocks of Q_TILE=64).  The single-block @pl.jit kernel is per-block, so
# the constant only survives as a documentation marker.
N_ROW = Q_TILE


class TestLoopDynValidShape:
    """Codegen smoke for dynamic valid_shape across both block lengths.

    The two cases mirror the two branches of the original in-DSL ``if/else``:
    the partial-last-block path (``vlen < BLOCK_COL``) and the full-block
    path (``vlen == BLOCK_COL``).
    """

    def test_partial_block(self):
        """Partial vlen (48) -- mirrors the ``is_last`` branch of the old loop."""
        dyn_valid_shape._cache.clear()
        data = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        out = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        program = dyn_valid_shape.compile_for_test(data, 2.0, 48, out)
        # Post-pass program must be non-empty and well-formed.
        assert program is not None
        assert len(program.functions) >= 1, (
            f"expected >= 1 function in post-pass IR, got {len(program.functions)}"
        )

    def test_full_block(self):
        """Full vlen (= BLOCK_COL) -- mirrors the non-last branch of the old loop."""
        dyn_valid_shape._cache.clear()
        data = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        out = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        program = dyn_valid_shape.compile_for_test(data, 2.0, BLOCK_COL, out)
        assert program is not None
        assert len(program.functions) >= 1, (
            f"expected >= 1 function in post-pass IR, got {len(program.functions)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
