# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end (compile-pipeline) test for the issue #1578 scenario.

A 3D+ tensor with a *dynamic* dimension that flows into a tile shape inside an
``pl.at`` (InCore) scope yields a >2D tile with a dynamic extent, which cannot be
flattened directly. The user handles it by **writing the chunk loop themselves**:
they iterate the dynamic dimension with ``pl.range`` in a static ``CHUNK`` step
and load each chunk as a static physical ``[1, CHUNK, 512]`` tile whose
``valid_shapes`` carries the runtime tail ``min(CHUNK, s - c)``. The chunk size
is the user's choice (it strongly affects performance).

``FlattenTileNdTo2D`` then only needs to lower the per-chunk ``[1, CHUNK, 512]``
tile to ``[CHUNK, 512]`` while **preserving the dynamic ``valid_shape``**
(``ComputeMergedValidShape``) so the runtime tail survives. This test pins that
the full ``@pl.jit`` pipeline compiles such a kernel.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from annotations that reference module-level DynVar names and from DSL-only
# subscript-assignment syntax.
# pyright: reportUndefinedVariable=false, reportIndexIssue=false, reportUnusedVariable=false

import pypto.language as pl
import pytest
import torch

B_DYN = pl.dynamic("B_DYN")
S_DYN = pl.dynamic("S_DYN")

# User-chosen physical chunk for the dynamic S dimension (performance knob).
# Must fit Vec memory: CHUNK * 512 * (bf16 + f32 bytes) <= UB capacity.
CHUNK = 16


@pl.jit
def cast_3d_dynamic(
    x: pl.Tensor[[B_DYN, S_DYN, 512], pl.BF16],
    out: pl.Out[pl.Tensor[[B_DYN, S_DYN, 512], pl.FP32]],
):
    """Issue #1578: cast a 3D tensor with a dynamic middle dim, user-chunked.

    The user iterates the dynamic S dim in CHUNK steps and loads each chunk as a
    static ``[1, CHUNK, 512]`` tile, clamping the tail with
    ``valid_shapes=[1, min(CHUNK, s - c), 512]`` so the last (partial) chunk does
    not read out of bounds.
    """
    b_dim = pl.tensor.dim(x, 0)
    s_dim = pl.tensor.dim(x, 1)
    for b in pl.parallel(0, b_dim):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="inner"):
            # User-written chunk loop over the dynamic S dim, carrying the output
            # tensor through ``init_values`` so each chunk's store threads forward.
            # A loop with init_values requires the assignment form of pl.yield_
            # (the LHS supplies the post-loop binding); bare pl.yield_ is rejected.
            for c, (o,) in pl.range(0, s_dim, CHUNK, init_values=(out,)):
                valid = pl.min(CHUNK, s_dim - c)
                t = pl.load(x, [b, c, 0], [1, CHUNK, 512], valid_shapes=[1, valid, 512])
                t = pl.cast(t, target_type=pl.FP32)
                o = pl.store(t, [b, c, 0], o)
                chunk_out = pl.yield_(o)  # noqa: F841 — parser requires the yield-LHS binding
    return out


class TestFlattenDynamicTile3D:
    """Compile-pipeline guard for issue #1578."""

    def test_3d_dynamic_tile_now_compiles(self):
        """A user-chunked dynamic >2D tile compiles end-to-end: flatten lowers the
        static per-chunk tile to 2D while preserving the dynamic valid tail."""
        cast_3d_dynamic._cache.clear()
        # Concrete arg shapes only seed specialization; B_DYN/S_DYN stay symbolic,
        # so the [1, CHUNK, 512] tile keeps a dynamic valid extent at pass time.
        # S=40 with CHUNK=16 exercises multiple chunks plus a partial tail (16 + 16 + 8).
        x = torch.zeros((16, 40, 512), dtype=torch.bfloat16)
        out = torch.zeros((16, 40, 512), dtype=torch.float32)

        # Runs the full pass pipeline (no raise). Before the fix, FlattenTileNdTo2D
        # dropped the dynamic valid_shape when flattening the >2D per-chunk tile.
        program = cast_3d_dynamic.compile_for_test(x, out)
        assert program is not None, "compile_for_test returned None"
        # The kernel's InCore function survives the pipeline.
        names = [fn.name for fn in program.functions.values()]
        assert any("cast_3d_dynamic" in n or "inner" in n for n in names), (
            f"expected the kernel function in post-pass IR, got {names}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
