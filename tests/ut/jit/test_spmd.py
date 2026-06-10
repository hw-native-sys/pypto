# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests: @pl.jit / @pl.jit.inline support pl.spmd().

Guards issue #1432. There is no separate "JIT parser" — the JIT specializer
rewrites @pl.jit functions into ordinary @pl.program source, which the shared
parser consumes. Both pl.spmd() forms therefore flow through unchanged:

  - the ``for i in pl.spmd(N):`` loop form, and
  - the ``with pl.spmd(N): kernel(...)`` scope form.

The sibling-loop test additionally guards PR #1414's specializer alpha-renamer
fix in the SPMD context: two sibling loops inside a pl.spmd block may reuse the
same loop-local names without the renamer emitting an out-of-scope bridge.
"""

import pypto.language as pl
import pytest
from pypto.jit.decorator import jit
from pypto.pypto_core import ir

# Module-level constants — the JIT specializer inlines module-level ints at
# their use sites, but does NOT capture function-local closure variables.
_BATCH = 16
_HIDDEN = 512
_K_CHUNK = 128
_HALF = _HIDDEN // 2  # columns of `hidden` handled by each SPMD block
_CHUNKS = _HALF // _K_CHUNK  # pl.pipeline iterations per block


def test_jit_spmd_for_loop():
    """``for i in pl.spmd(N)`` inside a plain @pl.jit entry compiles end-to-end."""
    torch = pytest.importorskip("torch")

    @jit
    def spmd_add(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        for i in pl.spmd(2):
            offset = i * 64
            tile_a = pl.load(a, [offset, 0], [64, 128])
            tile_b = pl.load(b, [offset, 0], [64, 128])
            c = pl.store(pl.add(tile_a, tile_b), [offset, 0], c)
        return c

    post = spmd_add.compile_for_test(torch.randn(128, 128), torch.randn(128, 128), torch.empty(128, 128))
    func_types = {f.func_type for f in post.functions.values()}
    assert ir.FunctionType.Spmd in func_types, (
        f"expected an Spmd function from `for i in pl.spmd()`, got {func_types}"
    )
    assert any(
        f.name == "spmd_add" and f.func_type == ir.FunctionType.Orchestration for f in post.functions.values()
    )


def test_jit_spmd_with_form():
    """``with pl.spmd(N): kernel(...)`` dispatching a @pl.jit.incore dep."""
    torch = pytest.importorskip("torch")

    @jit.incore
    def add_kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        # @jit.incore already establishes the InCore context — no pl.at needed.
        tile_a = pl.load(a, [0, 0], [64, 128])
        tile_b = pl.load(b, [0, 0], [64, 128])
        c = pl.store(pl.add(tile_a, tile_b), [0, 0], c)
        return c

    @jit
    def entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.spmd(2):
            c = add_kernel(a, b, c)
        return c

    post = entry.compile_for_test(torch.randn(128, 128), torch.randn(128, 128), torch.empty(128, 128))
    func_types = {f.func_type for f in post.functions.values()}
    assert ir.FunctionType.Spmd in func_types, (
        f"expected an Spmd function from `with pl.spmd()`, got {func_types}"
    )


def test_jit_inline_helper_spmd_for_loop():
    """A @pl.jit.inline helper using ``for i in pl.spmd(N)`` is spliced + dispatched."""
    torch = pytest.importorskip("torch")

    @jit.inline
    def spmd_helper(a: pl.Tensor, b: pl.Tensor, c: pl.Tensor):
        for i in pl.spmd(2):
            offset = i * 64
            tile_a = pl.load(a, [offset, 0], [64, 128])
            tile_b = pl.load(b, [offset, 0], [64, 128])
            c = pl.store(pl.add(tile_a, tile_b), [offset, 0], c)
        return c

    @jit
    def entry(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        c = spmd_helper(a, b, c)
        return c

    post = entry.compile_for_test(torch.randn(128, 128), torch.randn(128, 128), torch.empty(128, 128))
    func_types = {f.func_type for f in post.functions.values()}
    # InlineFunctions splices away every Inline helper.
    assert ir.FunctionType.Inline not in func_types, (
        "FunctionType.Inline helper should have been spliced by InlineFunctions"
    )
    assert ir.FunctionType.Spmd in func_types, (
        f"expected an Spmd function from the inline helper, got {func_types}"
    )


def test_jit_spmd_sibling_pipeline_loops_reuse_names():
    """Two sibling pl.pipeline loops inside a pl.spmd block may reuse loop-local names.

    Regression for issue #1432 / PR #1414: before #1414 the JIT specializer's
    alpha-renamer treated the second loop's ``k0`` / ``chunk`` as loop-carried
    rebinds of the first loop's locals, emitting a bridge that reads them
    outside their defining scope — which ConvertToSSA rejects. ``compile_for_test``
    raises if the specializer regresses.
    """
    torch = pytest.importorskip("torch")

    @jit.inline
    def rmsnorm_like(a: pl.Tensor, out: pl.Tensor):
        # Each SPMD block owns a column-half of `hidden`; `k0` and `chunk` are
        # reused across the two sibling pl.pipeline loops on purpose.
        for blk in pl.spmd(2):
            base = blk * _HALF
            acc = pl.full([1, _BATCH], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(_CHUNKS, stage=2):
                k0 = base + kb * _K_CHUNK
                chunk = pl.cast(a[:, k0 : k0 + _K_CHUNK], target_type=pl.FP32)
                acc = pl.add(acc, pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, _BATCH]))
            for kb in pl.pipeline(_CHUNKS, stage=2):  # sibling: reuses kb/k0/chunk
                k0 = base + kb * _K_CHUNK
                chunk = pl.cast(a[:, k0 : k0 + _K_CHUNK], target_type=pl.FP32)
                out = pl.assemble(out, pl.cast(chunk, target_type=pl.BF16), [0, k0])
        return out

    @jit
    def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
        out = rmsnorm_like(a, out)
        return out

    post = entry.compile_for_test(
        # Input is BF16 so the `pl.cast(..., FP32)` in `rmsnorm_like` is a
        # real cross-dtype promotion. Same-dtype casts are rejected at IR
        # construction time, so this fixture must not feed FP32 in.
        torch.empty([_BATCH, _HIDDEN], dtype=torch.bfloat16).normal_(),
        torch.empty([_BATCH, _HIDDEN], dtype=torch.bfloat16),
    )
    func_types = {f.func_type for f in post.functions.values()}
    assert ir.FunctionType.Spmd in func_types, (
        f"expected an Spmd function from sibling pl.pipeline loops, got {func_types}"
    )


def test_jit_spmd_with_form_as_tid_captures_and_wires_deps():
    """``with pl.spmd(N) as tid:`` (capture form) compiles end-to-end through @pl.jit.

    Exercises the full pipeline on the SPMD producer-TaskId capture: an inline
    ``as tid`` body (auto-outlined like the for-form) plus a downstream dispatch
    wired via ``deps=[tid0]``. Two separate Out buffers keep the dependency on the
    explicit ``deps=`` edge (not on a chained tuple-return). Asserts not just that
    two Spmd wrappers exist, but that the ``deps=[tid0]`` edge actually survives JIT
    specialization + lowering — otherwise the specializer could drop/rebind it and
    this test would still pass.
    """
    torch = pytest.importorskip("torch")

    @jit
    def entry(a: pl.Tensor, out1: pl.Out[pl.Tensor], out2: pl.Out[pl.Tensor]):
        with pl.spmd(2, name_hint="stage1") as tid0:
            i = pl.tile.get_block_idx()
            t = pl.load(a, [i * 64, 0], [64, 128])
            out1 = pl.store(pl.add(t, t), [i * 64, 0], out1)
        with pl.spmd(2, name_hint="stage2", deps=[tid0]) as tid1:  # noqa: F841
            j = pl.tile.get_block_idx()
            u = pl.load(a, [j * 64, 0], [64, 128])
            out2 = pl.store(pl.add(u, u), [j * 64, 0], out2)
        return out2

    post = entry.compile_for_test(torch.randn(128, 128), torch.empty(128, 128), torch.empty(128, 128))
    spmd_fns = [f for f in post.functions.values() if f.func_type == ir.FunctionType.Spmd]
    assert len(spmd_fns) == 2, (
        f"expected two Spmd functions from the two captured pl.spmd() dispatches, got {len(spmd_fns)}"
    )
    spmd_fn_names = {f.name for f in spmd_fns}

    # Inspect the lowered orchestration IR: the full pipeline now KEEPS the
    # captured dispatches as Submits of the outlined Spmd functions
    # (DeriveCallDirections preserves Submit-ness), and the surviving deps=[tid0]
    # edge rides as the consumer Submit's typed ``deps_`` field. Exactly one of
    # the two dispatches must carry a single dep edge — proving the specializer
    # neither dropped nor duplicated the JIT-specific dependency wiring.
    orch = next(f for f in post.functions.values() if f.func_type == ir.FunctionType.Orchestration)
    assert orch.name == "entry"
    spmd_dispatches = []

    def _walk(node):
        if isinstance(node, ir.SeqStmts):
            for s in node.stmts:
                _walk(s)
        elif isinstance(node, ir.AssignStmt):
            if isinstance(node.value, (ir.Call, ir.Submit)) and node.value.op.name in spmd_fn_names:
                spmd_dispatches.append(node.value)
        elif hasattr(node, "body") and node.body is not None:
            _walk(node.body)

    _walk(orch.body)
    assert len(spmd_dispatches) == 2, (
        f"expected two spmd dispatches, got {[c.op.name for c in spmd_dispatches]}"
    )

    def _dep_count(disp):
        # Deps live only on Submit::deps_ (ManualDepsOnSubmitOnly invariant);
        # a plain Call dispatch never carries dep edges.
        assert isinstance(disp, ir.Submit), f"spmd dispatch must be a Submit, got {type(disp).__name__}"
        return len(disp.deps)

    dep_counts = sorted(_dep_count(c) for c in spmd_dispatches)
    assert dep_counts == [0, 1], (
        f"expected the deps=[tid0] edge to survive as exactly one dep on the consumer "
        f"dispatch, got per-dispatch dep counts {dep_counts}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
