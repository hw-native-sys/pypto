# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lowering smoke tests for dynamic valid_shape selected by an in-DSL if/else.

``dyn_valid_shape_if_else`` reads ``[is_last, last_valid_len, full_len]`` from
a config tensor and picks the tile's valid length with an in-DSL ``if``/``else``.
Because the three values are read at runtime rather than passed as scalar
parameters, the branch is not folded at specialization time -- it lowers to an
``scf.if`` whose result feeds the tile's ``valid_col``.

One specialization covers both branches (only ``cfg`` contents differ), so
these tests vary ``is_last`` to confirm the shared kernel lowers for either
config:

  * is_last=1 -> ``vlen = last_valid_len`` (partial, ``vlen < BLOCK_COL``)
  * is_last=0 -> ``vlen = full_len``       (full, fillpad is a no-op)

The generated PTO is asserted in
``tests/ut/codegen/test_dynamic_valid_shape_if_else.py``.
"""

import pytest
import torch
from examples.intermediate.dyn_valid_shape import BLOCK_COL, Q_TILE, dyn_valid_shape_if_else


class TestDynValidShapeIfElse:
    """Lowering smoke for both branches of the in-DSL if/else."""

    def test_last_block(self):
        """is_last=1 path: partial valid_len (48) -- vlen < physical."""
        data = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        out = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        cfg = torch.tensor([1, 48, BLOCK_COL], dtype=torch.int64)
        program = dyn_valid_shape_if_else.lower(data, cfg, out)
        assert program is not None
        assert len(program.functions) >= 1, (
            f"expected >= 1 function in post-pass IR, got {len(program.functions)}"
        )

    def test_full_block(self):
        """is_last=0 path: full valid_len (= BLOCK_COL) -- fillpad no-op."""
        data = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        out = torch.zeros((Q_TILE, BLOCK_COL), dtype=torch.float32)
        cfg = torch.tensor([0, 48, BLOCK_COL], dtype=torch.int64)
        program = dyn_valid_shape_if_else.lower(data, cfg, out)
        assert program is not None
        assert len(program.functions) >= 1, (
            f"expected >= 1 function in post-pass IR, got {len(program.functions)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
