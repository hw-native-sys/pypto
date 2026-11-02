# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify backend-owned synchronization catalogs, target dispatch, and query snapshots."""

import gc
from typing import Any, cast

import pytest
from pypto.pypto_core import backend, ir, sync


@pytest.fixture(params=["a2a3", "a5"])
def catalog(request: pytest.FixtureRequest) -> sync.ResourceCatalog:
    """Borrow both target catalogs without choosing a process-global backend."""
    # Pytest 7 adds param only to parametrized requests, outside FixtureRequest's type contract.
    selected = sync.find_resource_catalog(cast(str, getattr(request, "param")))
    assert selected is not None
    return selected


def _pipeline_key(*, replica: int = 0) -> sync.PoolKey:
    """Use one modeled AIV direction while allowing independent-instance queries."""
    return sync.PoolKey(src=ir.PipeType.MTE2, dst=ir.PipeType.MTE3, replica=replica)


@pytest.mark.parametrize(
    "configured_backend", [None, backend.BackendType.Ascend910, backend.BackendType.Ascend950]
)
@pytest.mark.parametrize(
    ("queried_backend", "target"),
    [(backend.BackendType.Ascend910, "a2a3"), (backend.BackendType.Ascend950, "a5")],
)
def test_target_lookup_uses_owning_backend(
    configured_backend: backend.BackendType | None,
    queried_backend: backend.BackendType,
    target: str,
) -> None:
    """Both entry points borrow the owning handler's catalog regardless of global selection."""
    # The shared UT fixture resets backend state around each case. Exercise both
    # unconfigured use and a target different from the active compilation target.
    assert not backend.is_backend_configured()
    if configured_backend is not None:
        backend.set_backend_type(configured_backend)
    handler = backend.get_backend_instance(queried_backend).get_handler()
    owned = handler.get_sync_resource_catalog()
    assert owned.target == handler.get_pto_target_arch() == target
    assert handler.get_sync_resource_catalog() is owned
    assert sync.find_resource_catalog(target) is owned

    # The buffer-credit difference catches accidental dispatch through the
    # global handler even if both targets expose identical event capacities.
    credits = owned.find_pool(sync.PoolKey(kind=sync.ResourceKind.BufferCredit))
    assert credits is not None and credits.capacity == (32 if target == "a5" else 0)
    assert backend.is_backend_configured() == (configured_backend is not None)
    if configured_backend is not None:
        assert backend.get_backend_type() == configured_backend


def test_backend_catalogs_are_distinct() -> None:
    """Each backend owns a stable catalog instead of sharing mutable target facts."""
    h910 = backend.get_backend_instance(backend.BackendType.Ascend910).get_handler()
    h950 = backend.get_backend_instance(backend.BackendType.Ascend950).get_handler()
    c910 = h910.get_sync_resource_catalog()
    c950 = h950.get_sync_resource_catalog()
    assert c910 is not c950
    assert c910 is sync.find_resource_catalog("a2a3")
    assert c950 is sync.find_resource_catalog("a5")
    assert c910.target == "a2a3" and c950.target == "a5"
    assert not backend.is_backend_configured()


@pytest.mark.parametrize(
    "configured_backend", [None, backend.BackendType.Ascend910, backend.BackendType.Ascend950]
)
def test_unknown_target_preserves_backend_state(configured_backend: backend.BackendType | None) -> None:
    """An unknown target must neither choose a backend nor fall back to the active target."""
    assert not backend.is_backend_configured()
    if configured_backend is not None:
        backend.set_backend_type(configured_backend)
    assert sync.find_resource_catalog("unknown-target") is None
    assert backend.is_backend_configured() == (configured_backend is not None)
    if configured_backend is not None:
        assert backend.get_backend_type() == configured_backend


def test_target_capacities(catalog: sync.ResourceCatalog) -> None:
    """Keep hardware families distinct, including a modeled zero-capacity family on A2/A3."""
    pipeline = catalog.find_pool(_pipeline_key())
    signals = catalog.find_pool(sync.PoolKey(kind=sync.ResourceKind.IntraBlockSignal))
    credits = catalog.find_pool(sync.PoolKey(kind=sync.ResourceKind.BufferCredit))
    assert pipeline is not None and pipeline.capacity == 8
    assert pipeline.legal_ids == [] and pipeline.reserved_ids == []
    # Signal capacities count logical bases; the A5 +16 lowering companion is not another bank.
    assert signals is not None and signals.capacity == 16
    assert signals.key.core == sync.CoreScope.CoreGroup
    assert signals.legal_ids == [] and signals.reserved_ids == []
    assert credits is not None
    assert credits.capacity == (32 if catalog.target == "a5" else 0)
    assert credits.legal_ids == [] and credits.reserved_ids == []


@pytest.mark.parametrize("core", [sync.CoreScope.AIC, sync.CoreScope.AIV])
@pytest.mark.parametrize(("src", "dst"), [(ir.PipeType.V, ir.PipeType.S), (ir.PipeType.S, ir.PipeType.V)])
def test_scalar_vector_reservation_policy(
    catalog: sync.ResourceCatalog, core: sync.CoreScope, src: ir.PipeType, dst: ir.PipeType
) -> None:
    """Both V/S directions reserve ID 7 on each executor without changing hardware legality."""
    key = sync.PoolKey(core=core, src=src, dst=dst, replica=3)
    reserved = catalog.find_pool(key)
    assert reserved is not None and reserved.key == key
    assert reserved.capacity == 8 and reserved.legal_ids == []
    assert reserved.reserved_ids == [7]
    assert reserved.requires_contiguous_ids is False


@pytest.mark.parametrize("core", [sync.CoreScope.AIC, sync.CoreScope.AIV])
def test_reservation_policy_does_not_spill_to_other_directions(
    catalog: sync.ResourceCatalog, core: sync.CoreScope
) -> None:
    """The V/S exclusion must not consume IDs in neighboring independent event banks."""
    ordinary = catalog.find_pool(sync.PoolKey(core=core, src=ir.PipeType.S, dst=ir.PipeType.MTE2))
    assert ordinary is not None and ordinary.capacity == 8 and ordinary.reserved_ids == []
    assert catalog.find_pool(sync.PoolKey(core=core, src=ir.PipeType.V, dst=ir.PipeType.V)) is None


def test_pipeline_directions_and_replicas_remain_distinct(catalog: sync.ResourceCatalog) -> None:
    """Equal capacities do not collapse directed pools or independent physical instances."""
    forward_key = _pipeline_key()
    reverse_key = sync.PoolKey(src=forward_key.dst, dst=forward_key.src)
    forward = catalog.find_pool(forward_key)
    reverse = catalog.find_pool(reverse_key)
    replica = catalog.find_pool(_pipeline_key(replica=3))
    aic = catalog.find_pool(sync.PoolKey(core=sync.CoreScope.AIC, src=forward_key.src, dst=forward_key.dst))
    assert forward is not None and reverse is not None and replica is not None and aic is not None
    assert forward.key == forward_key
    assert reverse.key == reverse_key
    assert forward.key != reverse.key
    assert aic.key.core == sync.CoreScope.AIC and aic.key != forward.key
    assert replica.key.replica == 3 and replica.key != forward.key
    assert replica.capacity == forward.capacity
    assert replica.legal_ids == forward.legal_ids and replica.reserved_ids == forward.reserved_ids


def test_signal_aliases_query_one_canonical_domain(catalog: sync.ResourceCatalog) -> None:
    """Signal APIs name an executor pipe, while ownership belongs to one shared core-group bank."""
    aic = sync.PoolKey(
        kind=sync.ResourceKind.IntraBlockSignal,
        core=sync.CoreScope.AIC,
        src=ir.PipeType.MTE2,
        dst=ir.PipeType.V,
        replica=5,
    )
    aiv = sync.PoolKey(
        kind=sync.ResourceKind.IntraBlockSignal,
        core=sync.CoreScope.AIV,
        src=ir.PipeType.MTE3,
        dst=ir.PipeType.S,
        replica=5,
    )
    assert aic != aiv  # Raw lookup values are not canonicalized by construction or equality.
    first = catalog.find_pool(aic)
    second = catalog.find_pool(aiv)
    assert first is not None and second is not None
    canonical = sync.PoolKey(
        kind=sync.ResourceKind.IntraBlockSignal, core=sync.CoreScope.CoreGroup, replica=5
    )
    assert first.key == second.key == canonical
    assert first.capacity == second.capacity == 16
    assert first.legal_ids == second.legal_ids == []
    assert first.reserved_ids == second.reserved_ids == []
    assert aic.core == sync.CoreScope.AIC and aic.src == ir.PipeType.MTE2


def test_buffer_credit_canonicalization_preserves_executor(catalog: sync.ResourceCatalog) -> None:
    """Credits ignore API pipe spellings without inheriting the signals' shared-core alias rule."""
    query = sync.PoolKey(
        kind=sync.ResourceKind.BufferCredit,
        src=ir.PipeType.MTE2,
        dst=ir.PipeType.MTE3,
        replica=2,
    )
    result = catalog.find_pool(query)
    assert result is not None
    assert result.key == sync.PoolKey(kind=sync.ResourceKind.BufferCredit, replica=2)
    query.core = sync.CoreScope.AIC
    assert catalog.find_pool(query) is None


def test_queries_return_owned_snapshots(catalog: sync.ResourceCatalog) -> None:
    """Returned mutable keys/lists must not become writable aliases into read-only target facts."""
    query = sync.PoolKey(src=ir.PipeType.V, dst=ir.PipeType.S)
    snapshot = catalog.find_pool(query)
    assert snapshot is not None
    original_legal = snapshot.legal_ids
    original_reserved = snapshot.reserved_ids
    copied_key = snapshot.key
    copied_key.replica = 9
    legal = snapshot.legal_ids
    reserved = snapshot.reserved_ids
    legal.append(snapshot.capacity)
    reserved.clear()
    assert snapshot.key == query
    assert snapshot.legal_ids == original_legal and snapshot.reserved_ids == original_reserved
    # Query inputs and earlier results may be mutated/destroyed independently of the catalog.
    query.src = ir.PipeType.MTE2
    again = catalog.find_pool(sync.PoolKey(src=ir.PipeType.V, dst=ir.PipeType.S))
    assert again is not None
    assert again.key.src == ir.PipeType.V and again.reserved_ids == original_reserved
    with pytest.raises(AttributeError):
        setattr(snapshot, "capacity", 0)


@pytest.mark.parametrize("target", ["a2a3", "a5"])
def test_builtin_catalog_lifetime_and_immutability(target: str) -> None:
    """Dropping a Python wrapper must not delete the backend's process-lifetime native catalog."""
    borrowed = sync.find_resource_catalog(target)
    assert borrowed is not None and borrowed.target == target
    assert sync.find_resource_catalog(target) is borrowed
    query = sync.PoolKey(src=ir.PipeType.V, dst=ir.PipeType.S, replica=2)
    snapshot = borrowed.find_pool(query)
    assert snapshot is not None and snapshot.key == query
    assert snapshot.capacity == 8 and snapshot.reserved_ids == [7]
    with pytest.raises(AttributeError):
        setattr(borrowed, "target", "changed")

    # A detached result survives its wrapper, and querying again preserves the native pool policy.
    del borrowed
    gc.collect()
    recovered = sync.find_resource_catalog(target)
    assert recovered is not None and recovered.target == target
    assert snapshot.key == query and snapshot.capacity == 8 and snapshot.reserved_ids == [7]
    again = recovered.find_pool(query)
    assert again is not None and again.key == snapshot.key
    assert again.capacity == snapshot.capacity and again.legal_ids == snapshot.legal_ids
    assert again.reserved_ids == snapshot.reserved_ids
    assert again.requires_contiguous_ids == snapshot.requires_contiguous_ids
    with pytest.raises(TypeError):
        sync.ResourceCatalog()


def test_invalid_and_absent_domains_return_none(catalog: sync.ResourceCatalog) -> None:
    """Constructible enum combinations can still be invalid domains; queries reject them without guessing."""
    assert catalog.find_pool(sync.PoolKey()) is None  # ALL is a barrier concept, not an event endpoint.
    assert catalog.find_pool(sync.PoolKey(src=ir.PipeType.V, dst=ir.PipeType.V)) is None
    assert (
        catalog.find_pool(
            sync.PoolKey(core=sync.CoreScope.CoreGroup, src=ir.PipeType.MTE2, dst=ir.PipeType.MTE3)
        )
        is None
    )


@pytest.mark.parametrize("replica", [-1, 2**32])
def test_replica_conversion_rejects_out_of_range_values(replica: int) -> None:
    """The unsigned replica field must not wrap negative or overflowing Python integers."""
    with pytest.raises(TypeError):
        sync.PoolKey(replica=replica)


def test_keys_require_typed_enums() -> None:
    """Untyped callers cannot introduce string enum spellings that bypass native domain validation."""
    with pytest.raises(TypeError):
        sync.PoolKey(kind=cast(Any, "pipeline_event"))
    with pytest.raises(TypeError):
        sync.PoolKey(core=cast(Any, "aiv"))
    with pytest.raises(TypeError):
        sync.PoolKey(src=cast(Any, "MTE2"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
