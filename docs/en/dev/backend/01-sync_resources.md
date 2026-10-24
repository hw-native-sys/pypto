# Synchronization hardware resources

Each backend owns a synchronization resource catalog recording its resource
domains, ID ranges, and compiler reservation policy. Callers with an explicit
backend use `BackendHandler::GetSyncResourceCatalog()` in C++ or
`handler.get_sync_resource_catalog()` in Python. Callers with a target string
can use `pypto::sync::FindResourceCatalog` or `sync.find_resource_catalog`;
these select the same backend-owned catalog for `a2a3` or `a5`. Queries do not
allocate IDs, discover an ISA installation, or change the active backend.

## Query through the backend

An explicit handler selects its own catalog even when the process-global backend
is unconfigured or configured for another target:

```python
from pypto.pypto_core import backend, sync

target = backend.get_backend_instance(backend.BackendType.Ascend950)
handler = target.get_handler()
catalog = handler.get_sync_resource_catalog()
assert catalog.target == handler.get_pto_target_arch() == "a5"
assert sync.find_resource_catalog("a5") is catalog
```

The corresponding native query borrows a process-lifetime catalog:

```cpp
const auto* target = pypto::backend::GetBackendInstance(pypto::backend::BackendType::Ascend950);
const auto& catalog = target->GetHandler()->GetSyncResourceCatalog();
```

## Resource domains

`PoolKey` names a resource family, core scope, directed source/destination pipe
pair, and replica. Pipeline event banks are independent for each AIC/AIV executor
and direction. Intra-block signals canonicalize executor aliases to `CoreGroup`
and ignore pipe spelling. Non-pipeline resources use canonical `ALL/ALL` pipes.
Replica is a caller-supplied independent-instance identity; queries preserve it
while consulting the same replica-zero capability prototype.

An accepted key identifies a domain in the resource model. It does not certify
that each named pipe or instruction can execute on the selected core. The backend
remains responsible for instruction and core legality.

`PoolSpec` separates hardware legality from allocation policy:

| Field | Meaning |
| ----- | ------- |
| `capacity` | Exclusive numeric upper bound for hardware IDs |
| `legal_ids` | Restricted legal set; an empty list means `[0, capacity)` |
| `reserved_ids` | Hardware-legal IDs excluded by automatic allocation policy |
| `requires_contiguous_ids` | Contiguous IDs required for a future allocation |

The built-in rules follow the PTOAS resource model:

| Resource | A2/A3 capacity | A5 capacity |
| -------- | -------------- | ----------- |
| Pipeline event, per core and directed pipe pair | 8 | 8 |
| Intra-block signal, logical base IDs | 16 | 16 |
| Buffer credit | 0 | 32 |

On A5, the physical `base + 16` companion does not create an independently
allocatable signal base. A zero capacity records an unavailable resource family.
For `V -> S` and `S -> V`, ID 7 is compiler-reserved but remains hardware-legal
for an explicitly authored flag.

```python
from pypto.pypto_core import ir, sync

catalog = sync.find_resource_catalog("a5")
assert catalog is not None
pool = catalog.find_pool(
    sync.PoolKey(core=sync.CoreScope.AIV, src=ir.PipeType.V, dst=ir.PipeType.S)
)
assert pool is not None
assert pool.capacity == 8
assert pool.legal_ids == []
assert pool.reserved_ids == [7]
```

Unknown targets and invalid or unmodeled pool queries return `None` in Python
and null pointer / `std::nullopt` respectively in C++. Built-in catalogs live for
the process lifetime. Query results are independent snapshots; modifying a
returned list or key cannot change the catalog. Native construction validates
pool domains, ID sets, and duplicate canonical keys with `pypto::ValueError`.

## Native rules and ownership

The responsibilities are separated into three layers:

| Layer | Responsibility |
| ----- | -------------- |
| Backend | Own target facts and expose the catalog through its handler |
| Shared resource model (`pypto::sync`) | Define typed domains, validate and canonicalize keys, and return independent pool snapshots |
| Compilation consumer | Maintain occupancy and allocation state for one compilation; this catalog does not implement an allocator |

The Ascend910 and Ascend950 handlers each own a static catalog. Shared Ascend
constants and initialization rules live in `src/backend/common/sync_resources.cpp`;
the handlers supply their target-specific facts, including buffer-credit
capacity. Target-string lookup uses explicit backend instances and matches each
handler's `GetPtoTargetArch()`, so it does not maintain a second target registry
or consult the global backend selection.

`src/sync/resource_catalog.cpp` implements the shared resource model without
registering hardware targets. Queries validate and canonicalize a `PoolKey`,
look up the corresponding prototype, and return a `PoolSpec` snapshot with the
requested replica. Python exposes backend-owned catalogs and cannot construct
or register custom catalogs.

The capacity and reservation rules correspond to PTOAS's
`kTotalEventIdNum` and `kBlockSyncSetWaitEventIdNum` in
`include/PTO/Transforms/InsertSync/SyncEventIdAllocation.h`, the V/S reservation
in `lib/PTO/Transforms/InsertSync/SyncEventIdAllocation.cpp`, and
`ResourceModel::bufidCapacity` in
`include/PTO/Transforms/InsertSync/UnifiedSyncModel.h`.
Changes to these facts belong in the backend rules and their catalog tests.

This component supplies facts for subsequent synchronization work. It does not
include a macro registry or resource allocator. Hidden ISA macro footprints,
event-ID allocation, and resource-pressure repair remain outside this interface.
The default automatic synchronization path continues to use PTOAS.
