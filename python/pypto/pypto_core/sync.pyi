# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read-only hardware synchronization resource queries, independent of global backend state.

Pool keys are mutable lookup values. Catalogs and query results are immutable;
collection and nested-key properties return snapshots rather than writable aliases.
Pools model resource identity and ID capacity; backend instruction-legality checks
remain separate from these queries.
"""

from enum import Enum

from .ir import PipeType

class ResourceKind(Enum):
    """Disjoint resource families; equal numeric IDs do not imply shared hardware ownership."""

    PipelineEvent = 0
    IntraBlockSignal = 1
    BufferCredit = 2

class CoreScope(Enum):
    """Concrete AIC/AIV executors and the canonical CoreGroup scope for shared signal banks."""

    AIC = 0
    AIV = 1
    CoreGroup = 2

class PoolKey:
    """Physical conflict domain; pipeline events retain their directed source/destination pair."""

    def __init__(
        self,
        kind: ResourceKind = ResourceKind.PipelineEvent,
        core: CoreScope = CoreScope.AIV,
        src: PipeType = PipeType.ALL,
        dst: PipeType = PipeType.ALL,
        replica: int = 0,
    ) -> None:
        """Store typed lookup evidence; find_pool rejects invalid or unmodeled combinations."""
        ...
    kind: ResourceKind
    core: CoreScope  # Pipeline events require a concrete AIC/AIV executor.
    src: PipeType  # Canonicalization ignores pipe spellings for non-pipeline families.
    dst: PipeType
    replica: int  # Unsigned independent-instance identity; it is not an event-ID lane offset.
    def __eq__(self, other: object) -> bool:
        """Compare raw fields; find_pool results explicitly canonicalize physical aliases."""
        ...

class PoolSpec:
    """Immutable ID-space facts and allocation exclusions, without instruction-legality certification."""

    @property
    def key(self) -> PoolKey:
        """Return a detached canonical domain key, preserving the requested replica."""
        ...
    @property
    def capacity(self) -> int:
        """Hardware ID capacity; dense legal IDs occupy the half-open range [0, capacity)."""
        ...
    @property
    def legal_ids(self) -> list[int]:
        """Snapshot of a restricted hardware ID set; empty means the dense capacity range."""
        ...
    @property
    def reserved_ids(self) -> list[int]:
        """Snapshot of policy exclusions; these do not make authored hardware IDs illegal."""
        ...
    @property
    def requires_contiguous_ids(self) -> bool:
        """Whether the catalog requires a group to occupy consecutive IDs in this pool."""
        ...

class ResourceCatalog:
    """Borrowed backend-owned facts; Python cannot construct or register custom catalogs."""

    @property
    def target(self) -> str:
        """Canonical hardware target represented by this catalog."""
        ...
    def find_pool(self, key: PoolKey) -> PoolSpec | None:
        """Return an owned canonical pool snapshot, or None for an invalid/unmodeled domain."""
        ...

def find_resource_catalog(target: str) -> ResourceCatalog | None:
    """Borrow a backend catalog without ISA discovery or changing the active target; unknown is None."""
