# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen smoke tests for dynamic valid_shape branch selection.

The pre-JIT version of this test exercised a single-call kernel that
selected ``vlen`` via an in-DSL ``if/else`` based on an ``is_last`` flag.
In the @pl.jit world the specializer's alpha-renamer rewrites the
rebinding of ``vlen`` in the else-branch to a distinct alias, which then
fails ``ConvertToSSA`` ("used outside its defining scope").  The current
recommended workaround -- documented in
``examples/kernels/09_dyn_valid_shape.py`` -- is to push the
``vlen`` selection to the caller.

These tests verify that the JIT pipeline succeeds for both branches of
the original ``if/else``:

  * is_last=True  -> ``vlen = last_valid_len`` (partial)
  * is_last=False -> ``vlen = full_len`` (full)
"""

import pytest
import torch
from examples.kernels.dyn_valid_shape import BLOCK_COL, Q_TILE, dyn_valid_shape


class TestDynValidShapeIfElse:
    """Codegen smoke for the two branches of the (now caller-side) if/else.

    The original kernel computed ``vlen`` from an ``is_last`` flag inside
    the kernel.  Each test below picks the same ``vlen`` value the kernel
    would have used if the corresponding branch had been taken.
    """

    def test_last_block(self):
        """is_last=True path: partial valid_len (48) -- vlen < physical."""
        dyn_valid_shape._cache.clear()
        data = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        out = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        program = dyn_valid_shape.compile_for_test(data, 2.0, 48, out)
        assert program is not None
        assert len(program.functions) >= 1, (
            f"expected >= 1 function in post-pass IR, got {len(program.functions)}"
        )

    def test_full_block(self):
        """is_last=False path: full valid_len (= BLOCK_COL) -- fillpad no-op."""
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
