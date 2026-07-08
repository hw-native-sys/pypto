# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen checks for the ptoas multi-buffer switch (use_ptoas_multi_buffer).

A `pl.pipeline(stage=N)` loop over an i-dependent vec load normally lowers to
pypto's own body-replication ping-pong (several `pto.alloc_tile` at disjoint
addresses). With `use_ptoas_multi_buffer=True`, `ConvertToPtoasMultiBuffer`
instead keeps a single loop body and emits ptoas multi-buffer ops: one hoisted
`pto.alloc_multi_tile ... count=N` and a `pto.multi_tile_get %mb[i%N]` per
iteration (same slot for load + consume), delegating slot rotation + the
cross-iteration double-buffer overlap to ptoas.

The switch **auto-forces `memory_planner=PTOAS`** (--pto-level=level2): the
overlap only materializes when ptoas PlanMemory assigns the N slots concrete
disjoint addresses, so the emitted region is address-less (level2).

These are codegen-level checks on the emitted `.pto` MLIR text (the JIT compile
may raise post-codegen when the simpler runtime is absent — the assertion is on
the `.pto`, which materializes first).
"""

import shutil

import pypto.language as pl
import pypto.pypto_core.ir as _ir
import pytest
import torch
from pypto import backend as _backend
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes
from pypto.runtime import RunConfig

T, N = 256, 64


# Two distinct kernels: `@pl.jit` caches by function + arg signature (not by the
# active PassContext), so a single kernel compiled once would reuse its artifact
# regardless of the switch. Separate functions keep the on/off caches disjoint.
# The load offset is i-DEPENDENT (`[_i*64, 0]`) so consecutive iterations touch
# different data — the only case where double-buffering is observable.
@pl.jit
def mbuf_pipe_add_off(
    a: pl.Tensor[[T, N], pl.FP32],
    c: pl.Out[pl.Tensor[[T, N], pl.FP32]],
):
    """Accumulate a rolling i-dependent [64, N] vec tile across a 2-stage pipeline."""
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.load(a, [0, 0], [64, N])
        for _i in pl.pipeline(1, 4, stage=2):  # noqa: B007 - pipeline index
            t = pl.load(a, [_i * 64, 0], [64, N])
            acc = pl.add(acc, t)
        c = pl.store(acc, [0, 0], c)
    return c


@pl.jit
def mbuf_pipe_add_on(
    a: pl.Tensor[[T, N], pl.FP32],
    c: pl.Out[pl.Tensor[[T, N], pl.FP32]],
):
    """Same kernel as ``mbuf_pipe_add_off``; compiled with the switch on."""
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.load(a, [0, 0], [64, N])
        for _i in pl.pipeline(1, 4, stage=2):  # noqa: B007 - pipeline index
            t = pl.load(a, [_i * 64, 0], [64, N])
            acc = pl.add(acc, t)
        c = pl.store(acc, [0, 0], c)
    return c


def _emit_pto(kernel, dump_dir, use_multi_buffer: bool) -> str:
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    cfg = RunConfig(
        platform="a2a3",
        codegen_only=True,
        save_kernels=True,
        save_kernels_dir=str(dump_dir),
        use_ptoas_multi_buffer=use_multi_buffer,
    )
    try:
        kernel(torch.randn(T, N), torch.empty(T, N), config=cfg)
    except Exception:  # noqa: BLE001, S110 - post-codegen (simpler) failures are irrelevant here
        pass
    ptos = sorted(dump_dir.rglob("*.pto"))
    assert ptos, f"codegen emitted no .pto under {dump_dir}"
    return ptos[0].read_text()


def test_switch_off_uses_ordinary_alloc_tile(tmp_path):
    """Default (switch off): no multi-buffer ops; ordinary alloc_tile ping-pong."""
    text = _emit_pto(mbuf_pipe_add_off, tmp_path / "off", use_multi_buffer=False)
    assert "pto.alloc_multi_tile" not in text
    assert "pto.multi_tile_get" not in text
    assert "pto.alloc_tile" in text


def test_switch_on_emits_same_slot_multi_buffer(tmp_path):
    """Switch on: rotating vec tile lowers to a same-slot multi-buffer region.

    The switch auto-forces memory_planner=PTOAS (level2), so the region is
    address-less and ptoas PlanMemory assigns the N slots. The single loop body
    keeps one `multi_tile_get` per iteration (slot i%N, no prefetch split).
    """
    text = _emit_pto(mbuf_pipe_add_on, tmp_path / "on", use_multi_buffer=True)
    assert "pto.alloc_multi_tile" in text
    assert "count=2" in text
    assert "pto.multi_tile_get" in text
    # level2 (ptoas PlanMemory owns addresses): the region carries no baked addr.
    assert "pto.alloc_multi_tile addr" not in text
    # Same-slot form: no prefetch guard (scf.if) was emitted around the load.
    assert "scf.if" not in text


@pl.jit
def mbuf_pipe_add_dump(
    a: pl.Tensor[[T, N], pl.FP32],
    c: pl.Out[pl.Tensor[[T, N], pl.FP32]],
):
    """Same kernel; compiled via the explicit RunConfig field with dump_passes on."""
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.load(a, [0, 0], [64, N])
        for _i in pl.pipeline(1, 4, stage=2):  # noqa: B007 - pipeline index
            t = pl.load(a, [_i * 64, 0], [64, N])
            acc = pl.add(acc, t)
        c = pl.store(acc, [0, 0], c)
    return c


def test_runconfig_field_survives_dump_passes(tmp_path):
    """Regression: RunConfig.use_ptoas_multi_buffer must reach the pass even with
    dump_passes=True. Dump mode reconstructs the PassContext, and previously
    dropped use_ptoas_multi_buffer there, silently making the pass a no-op."""
    dump_dir = tmp_path / "dump_on"
    cfg = RunConfig(
        platform="a2a3",
        codegen_only=True,
        save_kernels=True,
        save_kernels_dir=str(dump_dir),
        dump_passes=True,
        use_ptoas_multi_buffer=True,
    )
    try:
        mbuf_pipe_add_dump(torch.randn(T, N), torch.empty(T, N), config=cfg)
    except Exception:  # noqa: BLE001, S110 - post-codegen (simpler) failures are irrelevant here
        pass
    ptos = sorted(dump_dir.rglob("*.pto"))
    assert ptos, f"codegen emitted no .pto under {dump_dir}"
    text = ptos[0].read_text()
    assert "pto.alloc_multi_tile" in text
    assert "pto.multi_tile_get" in text


@pl.jit
def mbuf_compile_for_test(
    a: pl.Tensor[[T, N], pl.FP32],
    c: pl.Out[pl.Tensor[[T, N], pl.FP32]],
):
    """Same kernel; driven through ``JITFunction.compile_for_test``."""
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.load(a, [0, 0], [64, N])
        for _i in pl.pipeline(1, 4, stage=2):  # noqa: B007 - pipeline index
            t = pl.load(a, [_i * 64, 0], [64, N])
            acc = pl.add(acc, t)
        c = pl.store(acc, [0, 0], c)
    return c


def test_compile_for_test_honors_pass_context_and_cache_key():
    """``compile_for_test`` takes no RunConfig, so it must honor an ambient
    PassContext — and fold that switch into its cache key.

    Without the switch in the key, a switch-on ``compile_for_test`` would reuse
    the switch-off ``_cache`` entry for the same kernel + args (the returned
    Program is always recomputed, so only the cached CompiledProgram collides).
    """
    mbuf_compile_for_test._cache.clear()
    a, c = torch.randn(T, N), torch.empty(T, N)

    off = mbuf_compile_for_test.compile_for_test(a, c)
    assert "multi_buffer_alloc" not in off.as_python()
    off_entries = len(mbuf_compile_for_test._cache)
    assert off_entries == 1, f"compile_for_test should populate its cache, got {off_entries} entries"

    with passes.PassContext([], use_ptoas_multi_buffer=True):
        on = mbuf_compile_for_test.compile_for_test(a, c)
    # The ambient PassContext must reach the pass pipeline.
    assert "multi_buffer_alloc" in on.as_python()
    assert "multi_buffer_load_slot" in on.as_python()

    # Distinct cache key: switch-on must not reuse the switch-off entry.
    assert len(mbuf_compile_for_test._cache) == off_entries + 1, (
        "compile_for_test cache key must fold in use_ptoas_multi_buffer; "
        f"switch-on reused the switch-off entry ({len(mbuf_compile_for_test._cache)} entries)"
    )


def test_multi_buffer_ops_round_trip():
    """The pass-synthesized multi_buffer ops must survive print -> parse.

    `tile.multi_buffer_alloc` / `tile.multi_buffer_load_slot` have no user DSL
    surface, but IR dumps reparse, so the printer must serialize them and the
    parser recover them (structural equality after a round-trip)."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910B)
    try:

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[T, N], pl.FP32],
                c: pl.Out[pl.Tensor[[T, N], pl.FP32]],
            ):
                with pl.at(level=pl.Level.CORE_GROUP):
                    acc = pl.load(a, [0, 0], [64, N])
                    for _i in pl.pipeline(1, 4, stage=2):  # noqa: B007
                        t = pl.load(a, [_i * 64, 0], [64, N])
                        acc = pl.add(acc, t)
                    c = pl.store(acc, [0, 0], c)
                return c

        with passes.PassContext([], use_ptoas_multi_buffer=True):
            out = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)

        printed = out.as_python()
        assert "multi_buffer_alloc" in printed
        assert "multi_buffer_load_slot" in printed
        reparsed = pl.parse_program(printed)
        _ir.assert_structural_equal(out, reparsed)
    finally:
        _backend.reset_for_testing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
