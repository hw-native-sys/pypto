# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lowering smoke tests for the per-block loop + if/else valid_shape idiom.

``dyn_valid_shape_loop`` reads ``[n_blocks, last_valid_len, block_size]`` from
a config tensor and loops over blocks, selecting the partial valid length on
the last iteration.  All three values are runtime reads, so the loop lowers to
an ``scf.for`` with a runtime trip count and the per-iteration ``if``/``else``
to an ``scf.if`` nested inside it -- the ragged-tail idiom used by the
paged-attention kernels.

The cases below vary the config the single specialization runs with:

  * ragged tail: last block partial (``last_valid_len < block_size``)
  * uniform: every block full (``last_valid_len == block_size``), so the
    per-iteration fillpad is a no-op
"""

import pytest
import torch
from examples.intermediate.dyn_valid_shape import BLOCK_COL, N_ROW, dyn_valid_shape_loop

# sij_buf holds N_ROW rows = 2 blocks of Q_TILE.
N_BLOCKS = 2


class TestLoopDynValidShape:
    """Lowering smoke for the loop + if/else valid_shape selection."""

    def test_ragged_tail(self):
        """Last block partial (48) -- the if/else takes its ``is_last`` branch."""
        sij_buf = torch.zeros((N_ROW, BLOCK_COL), dtype=torch.float32)
        out = torch.zeros((N_ROW, BLOCK_COL), dtype=torch.float32)
        cfg = torch.tensor([N_BLOCKS, 48, BLOCK_COL], dtype=torch.int64)
        program = dyn_valid_shape_loop.lower(sij_buf, cfg, out)
        # Post-pass program must be non-empty and well-formed.
        assert program is not None
        assert len(program.functions) >= 1, (
            f"expected >= 1 function in post-pass IR, got {len(program.functions)}"
        )

    def test_uniform_blocks(self):
        """Every block full (= BLOCK_COL) -- fillpad is a no-op on all iterations."""
        sij_buf = torch.zeros((N_ROW, BLOCK_COL), dtype=torch.float32)
        out = torch.zeros((N_ROW, BLOCK_COL), dtype=torch.float32)
        cfg = torch.tensor([N_BLOCKS, BLOCK_COL, BLOCK_COL], dtype=torch.int64)
        program = dyn_valid_shape_loop.lower(sij_buf, cfg, out)
        assert program is not None
        assert len(program.functions) >= 1, (
            f"expected >= 1 function in post-pass IR, got {len(program.functions)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
