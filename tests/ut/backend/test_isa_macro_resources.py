# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise sparse ISA signal declarations, physical ownership, and detached query results."""

import pytest
from pypto.pypto_core import backend, ir, sync


@pytest.fixture(params=[backend.BackendType.Ascend910, backend.BackendType.Ascend950])
def target(request) -> backend.Backend:
    """Exercise each backend without touching the global target."""
    return backend.get_backend_instance(request.param)


def _get(target: backend.Backend, name: str) -> sync.IsaMacroResources:
    """Require a real declaration; missing rules cannot be interpreted as zero usage."""
    result = target.find_isa_macro_resources(name)
    assert result is not None, name
    return result


def test_registered_macros_claim_signals_and_results_are_owned(target: backend.Backend) -> None:
    """A published row always reserves a signal; no empty or unavailable placeholders are needed."""
    names = target.get_isa_macro_names()
    assert names
    assert names == sorted(set(names))
    assert target.find_isa_macro_resources("pto::NOT_A_MACRO") is None
    for name in names:
        row = _get(target, name)
        assert row.name == name
        assert row.target in {"a2a3", "a5"}
        assert row.claims
        assert all(
            claim.pool.kind in {sync.ResourceKind.PipelineEvent, sync.ResourceKind.IntraBlockSignal}
            for claim in row.claims
        )
    expected_names = list(names)
    names.clear()
    assert target.get_isa_macro_names() == expected_names


@pytest.mark.parametrize(
    "name",
    [
        "pto::TADD",
        "pto::TCVT",
        "pto::TPRINT",
        "pto::TMATMUL",
        "pto::TMOV",
        "pto::TSTORE",
        "pto::TRANDOM",
        "pto::comm::TNOTIFY",
        "pto::comm::TWAIT",
        "pto::comm::TTEST",
    ],
)
def test_macros_without_signal_occupancy_have_no_entry(target: backend.Backend, name: str) -> None:
    """Barriers, unit flags, GM notifications and missing implementations need no signal entry.

    None means no registered signal claim, not proof that an arbitrary implementation is safe.
    """
    assert target.find_isa_macro_resources(name) is None


@pytest.mark.parametrize("name", ["pto::Event", "pto::TPipe", "pto::GridPipe", "pto::TSync_Custom"])
def test_helper_apis_are_not_a_separate_public_catalog(target: backend.Backend, name: str) -> None:
    """A public macro carries its helper's signal usage without publishing helper-only entries."""
    assert target.find_isa_macro_resources(name) is None


def test_comm_and_tile_gather_have_distinct_names(target: backend.Backend) -> None:
    """Equal macro spellings from different namespaces must never overwrite one another."""
    comm = _get(target, "pto::comm::TGATHER")
    tile = target.find_isa_macro_resources("pto::TGATHER")
    if comm.target == "a2a3":
        assert tile is not None
        assert tile.name == "pto::TGATHER"
        assert any(c.pool.src == ir.PipeType.V and c.pool.dst == ir.PipeType.S for c in tile.claims)
    else:
        # A5's tile gather has no hidden signal, while the communication collective still does.
        assert tile is None
    assert comm.name == "pto::comm::TGATHER"
    assert any(c.pool.src == ir.PipeType.MTE2 and c.pool.dst == ir.PipeType.MTE3 for c in comm.claims)


@pytest.mark.parametrize("macro", ["TPUT", "TGET"])
def test_transfer_reserves_both_ping_pong_directions(target: backend.Backend, macro: str) -> None:
    """One explicit staging tile can split into two lanes, occupying ID 0/1 in both banks."""
    claims = _get(target, f"pto::comm::{macro}").claims
    assert {(c.pool.src, c.pool.dst, tuple(c.ids)) for c in claims} == {
        (ir.PipeType.MTE2, ir.PipeType.MTE3, (0, 1)),
        (ir.PipeType.MTE3, ir.PipeType.MTE2, (0, 1)),
    }
    assert all(c.source == sync.IsaIdSource.Fixed for c in claims)
    assert all(
        c.pool.kind == sync.ResourceKind.PipelineEvent and c.pool.core == sync.CoreScope.AIV for c in claims
    )


def test_async_ids_are_parameters_with_shared_session_identity(target: backend.Backend) -> None:
    """Submission pairs its hardware set/wait locally while the asynchronous session outlives it."""
    row = _get(target, "pto::comm::TPUT_ASYNC")
    claims = [c for c in row.claims if c.source == sync.IsaIdSource.Parameter]
    assert claims
    assert all(c.parameter == "session.syncId" and c.owner == "session" for c in claims)
    assert all(c.offsets == [0] and c.ids == [] for c in claims)
    assert all(c.lifetime == sync.IsaResourceLifetime.Invocation for c in claims)


@pytest.mark.parametrize("macro", ["TPUSH", "TPOP", "TFREE", "TALLOC"])
def test_pipe_credits_share_one_binding_and_logical_offsets(target: backend.Backend, macro: str) -> None:
    """Per-call release would let unrelated macros steal the pipe's ready/free flags."""
    row = _get(target, f"pto::{macro}")
    signals = [c for c in row.claims if c.pool.kind == sync.ResourceKind.IntraBlockSignal]
    assert signals
    assert all(c.source == sync.IsaIdSource.Parameter for c in signals)
    assert all(c.lifetime == sync.IsaResourceLifetime.Owner and c.owner == "pipe" for c in signals)
    assert all(c.parameter == "pipe.FlagID" and c.ids == [] for c in signals)
    assert {offset for c in signals for offset in c.offsets} == {0, 1, 2, 3}
    assert all(c.pool.core == sync.CoreScope.CoreGroup for c in signals)
    assert not any(c.pool.kind == sync.ResourceKind.BufferCredit for c in row.claims)


def test_simt_gather_reserves_its_cross_core_signal(target: backend.Backend) -> None:
    """A5 SIMT gather hands scratch from AIV to AIC using signal 2 in the shared bank."""
    row = _get(target, "pto::MGATHER")
    signals = [c for c in row.claims if c.pool.kind == sync.ResourceKind.IntraBlockSignal]
    if row.target == "a5":
        assert signals
        assert {id_ for c in signals for id_ in c.ids} == {2}
        assert all(
            c.pool.core == sync.CoreScope.CoreGroup and c.source == sync.IsaIdSource.Fixed for c in signals
        )
        assert all(c.lifetime == sync.IsaResourceLifetime.Invocation for c in signals)
    else:
        assert not signals


def test_reduction_ping_pong_events_use_the_vector_input_bank(target: backend.Backend) -> None:
    """Root accumulator uses ID 0 and remote inputs use ID 1/2 on MTE2→V, not MTE3→MTE2."""
    claims = _get(target, "pto::comm::TREDUCE").claims
    vector_input = [c for c in claims if c.pool.src == ir.PipeType.MTE2 and c.pool.dst == ir.PipeType.V]
    assert vector_input
    assert {id_ for c in vector_input for id_ in c.ids} == {0, 1, 2}
    assert all(c.source == sync.IsaIdSource.Fixed and c.pool.core == sync.CoreScope.AIV for c in vector_input)
    store_to_load = [
        c
        for c in claims
        if c.pool.src == ir.PipeType.MTE3
        and c.pool.dst == ir.PipeType.MTE2
        and c.source == sync.IsaIdSource.Fixed
    ]
    assert {id_ for c in store_to_load for id_ in c.ids} == {0}


def test_helper_allocation_modes_are_not_reported_as_fixed_zero(target: backend.Backend) -> None:
    """PTO_FLAG_TEST allocates helper tokens, while __PTO_AUTO__ omits the helper entirely."""
    row = _get(target, "pto::TMRGSORT")
    claims = row.claims
    allocated = [c for c in claims if c.source == sync.IsaIdSource.CompilerAllocated]
    assert allocated
    catalog = sync.find_resource_catalog(row.target)
    assert catalog is not None
    for claim in allocated:
        pool = catalog.find_pool(claim.pool)
        assert pool is not None
        assert claim.modes == [sync.IsaSyncMode.FlagTest]
        assert claim.ids == (pool.legal_ids or list(range(pool.capacity)))
        # This describes downstream ISA allocation, whose choices PyPTO does not yet control.
        # Compiler reservation policy must not hide a hardware ID that the macro may occupy.
        assert set(pool.reserved_ids) <= set(claim.ids)
        assert claim.pool.src == ir.PipeType.V and claim.pool.dst == ir.PipeType.S
        assert pool.reserved_ids
    assert all(sync.IsaSyncMode.Auto not in c.modes for c in claims)
    fixed = [c for c in claims if c.source == sync.IsaIdSource.Fixed]
    assert fixed and all(c.modes == [sync.IsaSyncMode.Manual] and c.ids == [0] for c in fixed)


def test_nested_results_are_detached(target: backend.Backend) -> None:
    """Mutating returned vectors or mutable PoolKey objects cannot corrupt future queries."""
    row = _get(target, "pto::comm::TPUT")
    claim = row.claims[0]
    original = claim.pool.replica
    claim.pool.replica = 81
    claim.ids.clear()
    row.claims.clear()
    assert claim.pool.replica == original and claim.ids
    assert row.claims
    assert _get(target, row.name).claims[0].pool.replica == original
    with pytest.raises(AttributeError):
        setattr(row, "name", "pto::INVALID")


def test_collective_helpers_keep_flag_test_allocation(target: backend.Backend) -> None:
    """Collective single-chunk helpers and explicit ping-pong flags coexist in one implementation."""
    for name in ("pto::comm::TBROADCAST", "pto::comm::TREDUCE"):
        claims = _get(target, name).claims
        assert any(c.source == sync.IsaIdSource.Fixed for c in claims)
        assert any(c.source == sync.IsaIdSource.CompilerAllocated for c in claims)


def test_syncall_occupies_the_same_signal_pool_as_pipes(target: backend.Backend) -> None:
    """Global hard synchronization aliases the same logical signal pool used by TPipe."""
    row = _get(target, "pto::SYNCALL")
    assert {id_ for c in row.claims for id_ in c.ids} == {11, 12, 13, 14}
    assert all(c.pool.kind == sync.ResourceKind.IntraBlockSignal for c in row.claims)
    assert all(c.source == sync.IsaIdSource.Fixed for c in row.claims)
    assert all(c.lifetime == sync.IsaResourceLifetime.Invocation for c in row.claims)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
