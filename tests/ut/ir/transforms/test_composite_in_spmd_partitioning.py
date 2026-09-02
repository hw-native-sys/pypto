# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Does an InCore composite collective partition its work across ``pl.spmd`` blocks?

``pld.tensor.allreduce`` requires InCore callers to keep ``core_num=1``, and
``LowerTensorAllReduceRule`` enforces that with a ``CHECK_SPAN``.  Until
2026-08-28 both the docstring and that message pointed users at an enclosing
``pl.spmd(...)`` as the multi-core InCore path.  It is not one, and this file
tests what the lowering actually does under such a scope.

The claim under test: ``lower_composite_ops_pass.cpp`` never reads the block
index, so every block of an enclosing ``pl.spmd(N)`` runs the *whole* peer loop
— the transfer is duplicated ``N`` times rather than partitioned ``N`` ways.

**What a unit test can and cannot show here.** The N-fold duplication is a
*runtime* property: ``pl.spmd(N)`` emits one body that N AIV blocks each
execute.  Counting ops in the IR therefore cannot show an N-fold op count.
What it *can* show, and what decides the question, is whether the emitted body
is a function of ``N`` at all:

* if the lowered composite body is **identical** for ``N`` = 1, 2, 8, and
* it contains **no block-index read**,

then every block issues the same puts to the same peers with the same offsets,
which is duplication rather than partitioning.  Conversely a partitioned
lowering must either stride by the block index or vary with ``N``.

Tests here pin **current** behaviour.  If plan 92 Phase 1 lands a partitioned
lowering, ``test_composite_body_does_not_read_block_index`` and
``test_lowered_body_is_independent_of_spmd_width`` are expected to flip and
should be rewritten, not deleted.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir, passes

SIZE = 16
NRANKS = 2

# Ops whose count tracks how much communication the lowering emits.
_OP_PUT = ir.get_op("pld.tile.put").name
_OP_NOTIFY = ir.get_op("pld.system.notify").name
_OP_WAIT = ir.get_op("pld.system.wait").name
_OP_REMOTE_LOAD = ir.get_op("pld.tile.remote_load").name

# A partitioned lowering has to read the block index under one of these names.
# Resolved through the registry rather than compared as bare literals, per
# .claude/rules/operator-identity-checks.md.
_BLOCK_INDEX_OPS = frozenset(
    {
        ir.get_op("tile.get_block_idx").name,
        ir.get_op("tensor.get_block_idx").name,
    }
)


class _OpNameCollector(ir.IRVisitor):
    """Record ``op.name`` for every Call in the program."""

    def __init__(self) -> None:
        super().__init__()
        self.op_names: list[str] = []

    def visit_call(self, op: ir.Call) -> None:
        self.op_names.append(op.op.name)
        super().visit_call(op)


def _collect_op_names(prog) -> list[str]:
    collector = _OpNameCollector()
    collector.visit_program(prog)
    return collector.op_names


def _counts(prog) -> dict[str, int]:
    names = _collect_op_names(prog)
    return {
        "put": names.count(_OP_PUT),
        "notify": names.count(_OP_NOTIFY),
        "wait": names.count(_OP_WAIT),
        "remote_load": names.count(_OP_REMOTE_LOAD),
        "block_idx": sum(names.count(b) for b in _BLOCK_INDEX_OPS),
    }


# Width-independence is asserted on every communication op the schedules can
# emit — mesh allreduce also transfers peer data via ``pld.tile.remote_load``
# (see the allreduce rule in lower_composite_ops_pass.cpp), so a width-dependent
# lowering that switches put/notify/wait totals for remote_loads would otherwise
# slip through the count comparison.
_COMMUNICATION_KEYS = ("put", "notify", "wait", "remote_load")


def _build_allgather_in_spmd(width: int | None):
    """Allgather in an InCore function, optionally wrapped in ``pl.spmd(width)``.

    ``width=None`` builds the unwrapped baseline the composite STs use.
    """
    if width is None:

        @pl.program
        class Bare:
            @pl.function(type=pl.FunctionType.InCore)
            def gather_step(
                self,
                inp: pl.Tensor[[1, SIZE], pl.FP32],
                data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
            ) -> pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]:
                return pld.tensor.allgather(inp, data, signal)

        return Bare

    @pl.program
    class InSpmd:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[NRANKS, SIZE], pl.FP32]:
            for _ in pl.spmd(width, name_hint="ag_spmd"):
                data = pld.tensor.allgather(inp, data, signal)
            return data

    return InSpmd


# ============================================================================
# Q1 — is a composite inside pl.spmd even accepted?
# ============================================================================


@pytest.mark.parametrize("width", [1, 2, 8])
def test_composite_inside_spmd_lowers(width):
    """An InCore composite nested in ``pl.spmd(N)`` must survive LowerCompositeOps.

    A rejection here would be a *harder* blocker than duplication — it would
    mean the pypto-lib rails cannot host a composite at all, and plan 93's
    split-phase design would have to solve placement too.
    """
    program = _build_allgather_in_spmd(width)
    after = passes.lower_composite_ops()(program)
    op_names = set(_collect_op_names(after))

    assert "pld.tensor.allgather" not in op_names, (
        "the composite Call must not survive the pass inside pl.spmd"
    )
    assert _OP_PUT in op_names, "lowered body inside pl.spmd emits no pld.tile.put"


# ============================================================================
# Q2 — does the composite consume the block index?
# ============================================================================


@pytest.mark.parametrize("width", [2, 8])
def test_composite_ignores_the_enclosing_block_index(width):
    """The only block-index read is ``pl.spmd``'s own loop variable.

    ``pl.spmd(N)`` materialises its induction variable as a block-index read, so
    a program-wide count is 1 even though the composite never consumes it.  What
    identifies the composite as block-unaware is that adding that scaffolding
    leaves its emitted communication untouched: same puts, same notifies, same
    waits.  The IR bears this out directly — the push is

        for peer in pl.range(nranks_idx):
            pld.tile.put(data, peer, inp, stage, [my_rank, 0], [0, 0], [1, SIZE])

    whose bounds are ``nranks`` and whose offsets are ``my_rank``.  Neither is a
    function of the block index, so all N blocks push identical bytes to
    identical destinations.
    """
    bare = _counts(passes.lower_composite_ops()(_build_allgather_in_spmd(None)))
    wrapped = _counts(passes.lower_composite_ops()(_build_allgather_in_spmd(width)))

    assert bare["block_idx"] == 0, "unwrapped composite should read no block index"
    assert wrapped["block_idx"] == 1, (
        f"expected exactly 1 block-index read (pl.spmd's loop variable), got "
        f"{wrapped['block_idx']} — if the composite started reading it, plan 92's premise is void"
    )
    for key in _COMMUNICATION_KEYS:
        assert bare[key] == wrapped[key], (
            f"{key} count changed when wrapped in pl.spmd({width}) "
            f"({bare[key]} -> {wrapped[key]}); the composite is spmd-aware after all"
        )


# ============================================================================
# Q3 — does the emitted work depend on the spmd width at all?
# ============================================================================


def test_lowered_body_is_independent_of_spmd_width():
    """The decisive check: identical emitted communication for N = 1, 2, 8.

    A partitioned lowering must vary with N (different bounds) or stride by the
    block index (ruled out by Q2).  Identical counts across widths mean every
    block issues the same puts to the same peers — N-fold duplicate traffic,
    not an N-way split.
    """
    counts = {w: _counts(passes.lower_composite_ops()(_build_allgather_in_spmd(w))) for w in (1, 2, 8)}

    baseline = {k: v for k, v in counts[1].items() if k != "block_idx"}
    for width in (2, 8):
        got = {k: v for k, v in counts[width].items() if k != "block_idx"}
        assert got == baseline, (
            f"emitted communication differs between pl.spmd(1) and pl.spmd({width}): "
            f"{baseline} vs {got} — the lowering DOES vary with the spmd width, "
            "so it may already partition; plan 92 needs revisiting"
        )


# ============================================================================
# Q4 — what the duplication does to the barrier
# ============================================================================


def test_barrier_expected_value_does_not_scale_with_spmd_width():
    """The wait's expected credit is a compile-time 1 regardless of spmd width.

    This is the sharper half of the finding.  Under ``pl.spmd(N)`` every block
    runs the notify loop, so a peer's signal cell receives ``+N`` per source
    rank — but the wait still expects ``Ge(1)``.  The barrier therefore releases
    once a peer's *first* block has notified, while its remaining N-1 blocks may
    still be pushing.

    It is not a data race *today* only because every block writes byte-identical
    content, so an early reader still observes correct values; and the signal
    still returns to zero because the epilogue subtracts ``-1`` per block too.
    Both properties are incidental.  Any future partitioned lowering — where
    blocks push *different* slices — turns this into a real race, which is why
    ``EmitBarrier`` taking a compile-time generation (rather than a runtime
    credit count) is a prerequisite for plan 92 Phase 1 and plan 93.
    """
    for width in (1, 2, 8):
        after = passes.lower_composite_ops()(_build_allgather_in_spmd(width))
        text = str(after)
        assert "pld.system.wait(signal, [" in text, "expected a lowered wait on the signal"
        assert "pl.const(1, pl.INT32), cmp=" in text, (
            f"under pl.spmd({width}) the wait's expected value is no longer the "
            "compile-time constant 1 — the barrier became width-aware; update plan 92"
        )


# ============================================================================
# Q5 — mode coverage: does the finding hold for allreduce mesh AND ring?
# ============================================================================
#
# ``allgather`` above has a single schedule.  ``allreduce`` has two, and they
# barrier very differently: mesh issues a ready barrier through ``EmitBarrier``,
# while ring hand-rolls loop-resident notify/wait pairs against a
# ``[2*(NR-1), NR]`` signal because ``EmitBarrier`` is top-level-only.  A finding
# from mesh does not automatically carry to ring, so both are exercised here.

_RING_SIGNAL_ROWS = 2 * (NRANKS - 1)


def _build_allreduce_in_spmd(width: int | None, mode: str):
    """Allreduce (mesh or ring) in an InCore function, optionally in ``pl.spmd``."""
    sig_rows = NRANKS if mode == "mesh" else _RING_SIGNAL_ROWS
    sig_cols = 1 if mode == "mesh" else NRANKS

    if width is None:

        @pl.program
        class BareAR:
            @pl.function(type=pl.FunctionType.InCore)
            def ar_step(
                self,
                data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
            ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
                return pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode=mode)

        return BareAR

    @pl.program
    class ARInSpmd:
        @pl.function(type=pl.FunctionType.InCore)
        def ar_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[sig_rows, sig_cols], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            for _ in pl.spmd(width, name_hint="ar_spmd"):
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode=mode)
            return data

    return ARInSpmd


@pytest.mark.parametrize("mode", ["mesh", "ring"])
def test_allreduce_emitted_work_is_independent_of_spmd_width(mode):
    """Both allreduce schedules emit spmd-width-independent communication.

    Confirms the allgather finding generalises across the two barrier styles,
    so plan 92's Phase 0 conclusion is not mesh-only.
    """
    bare = _counts(passes.lower_composite_ops()(_build_allreduce_in_spmd(None, mode)))
    for width in (2, 8):
        wrapped = _counts(passes.lower_composite_ops()(_build_allreduce_in_spmd(width, mode)))
        for key in _COMMUNICATION_KEYS:
            assert bare[key] == wrapped[key], (
                f'mode="{mode}": {key} count changed under pl.spmd({width}) '
                f"({bare[key]} -> {wrapped[key]}); this schedule may already partition"
            )


# ============================================================================
# Q6 — the guidance itself
# ============================================================================


def test_incore_core_num_rejection_does_not_recommend_spmd():
    """The ``core_num > 1`` rejection must not point InCore users at ``pl.spmd``.

    Until 2026-08-28 both this message and the ``core_num`` docstring offered an
    enclosing ``pl.spmd(...)`` as the multi-core InCore path.  Every other test
    in this file shows it is not one: the work is duplicated rather than
    divided, and the barrier releases early.  Following that advice silently
    multiplied traffic by the spmd width, so the guidance is pinned here
    against reintroduction.
    """

    @pl.program
    class MultiCoreInCore:
        @pl.function(type=pl.FunctionType.InCore)
        def ar_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[NRANKS, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            return pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, core_num=2)

    with pytest.raises(ValueError, match=r"NOT a multi-core InCore path") as excinfo:
        passes.lower_composite_ops()(MultiCoreInCore)

    message = str(excinfo.value)
    assert "HOST orchestrator" in message, (
        f"rejection should name the HOST rail as the multi-core path, got: {message}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
