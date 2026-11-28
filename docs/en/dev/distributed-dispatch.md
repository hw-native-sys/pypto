# Bounded distributed dispatch

The budgeted dispatch path uses Simpler's completion events, identity capacity,
`RunHandle.finalized`, and hierarchical completion callbacks. With the current
pinned runtime, `DistributedWorker` still uses the existing two Host metadata
frames and finalizes the oldest handle on submission pressure.

## Ownership and completion

A dispatch frame owns Host scratch, argument wrappers, retained communication
domains, and cleanup callbacks. Public model parameters and caller-owned mutable
state are referenced, not copied. A single background observer waits for any
native parent to finish; it does not poll or allocate a thread per request.
It calls `result()` to finish native cleanup and PyPTO result publication before
returning the frame. A blocked submitter waits on the frame condition variable,
not on the oldest handle. Explicit `result()` remains idempotent.

Reclamation does not consume failures on the caller's behalf. A `done` query
only reports terminal state; an unobserved failure remains available to
`result()` and is reported by `close()` if the caller drops the handle.
An interrupted or refused wait does not return a frame: retirement requires
the native handle's final result and cleanup outcome to have been published.
A cached terminal `TimeoutError` is a failure, distinct from an expired wait.

Persistent submissions carry an explicit predecessor-success dependency. A
successor may prepare, but it cannot activate until its predecessor's native
cleanup, PyPTO cleanup, and result postprocessing have all succeeded. Failure
poisons the persistent worker and prevents reuse; it does not authorize the
successor. This dependency is local to the persistent pipeline, not a change to
ordinary independent Simpler submissions.

Unknown communication declarations and requests that reset windows retain a
conservative predecessor wait before issuing allocation/copy controls. Only
prewarmed, validated domain-pool submissions avoid that fallback.

Caller-owned shared windows use the existing `CommDomain` and `DomainTensor`
lifecycle. Submissions that bind the same shared domain wait for its prior uses
to finish. Each frame holds the domain and its buffer arguments through
completion or whole-worker teardown. Caller-owned windows are excluded from
automatic pool allocation.

## Capacity policy

Capacity is established during initialization, before dispatch admission:

- Extra Host frames have a budget of `min(64 MiB, free Host pages / 64)`.
  `free Host pages` is `SC_AVPHYS_PAGES * SC_PAGE_SIZE`, not Linux
  `MemAvailable`. Shared tensor storage is charged in whole OS pages; Python
  object accounting is an estimate. The native identity-directory capacity is
  an additional upper bound.
- Each distinct nonempty shared Tensor storage also charges one descriptor per
  extra frame. At most one quarter of currently free `RLIMIT_NOFILE` capacity
  is used for these optional copies; unknown descriptor occupancy permits only
  the baseline. Aliases are counted once. Before fork, declared domain sizes
  also cap Host replication by the device cache budget; actual device mappings
  may reduce the capacity further after initialization.
- The first domain set is provisioned on all ranks. For each rank, extra sets
  have a budget of `min(128 MiB, free device bytes / 64)`. Cost includes actual
  window mapping sizes, a conservative 2 MiB allowance per opaque domain
  context, and any larger observed free-memory reduction during provisioning.
  The smallest capacity across ranks is applied to every rank.
- One baseline frame/domain set is permitted independently of the extra-copy
  budget. Simulator backends without device-memory telemetry retain that one
  domain set. Unneeded Host candidates are discarded before admission.

These are internal conservative cache budgets, not a hardware capacity claim
or a performance optimum. The memory measurements can change due to other
processes, and Python object accounting is not a strict process-memory bound.
Allocator failures remain errors; partially constructed workers close through
the normal cleanup path. No live-worker expansion is implemented. Increasing
capacity does not clone model weights or caller-owned state.

## Complete communication-resource contract

An audited artifact producer explicitly writes
`orchestration/communication_resources.json`. Fresh IR compilation alone does
not prove the behavior of external/opaque leaf functions and never grants this
contract automatically. The versioned producer promise is that opaque
communication only accesses the declared domains and all other memory accesses
are represented by actual `TaskArgs`. Host syntax validation rejects dynamic
domain specifications, extra workspace/initializer controls, and scalar
arguments other than rank or declared-domain contexts.

The writer requires an explicit completeness promise, dependency closure, and rank count:

```python
from pathlib import Path
from pypto.runtime.communication_resources import write_resource_contract

write_resource_contract(
    Path(artifact_root), complete_resources=True,
    external_dependencies=(Path(custom_library),), world_size=8,
)
```

The dependency tuple must describe the complete external application/helper
code dependency closure. Passing an empty tuple explicitly promises there is
no unbound application code; unknown dependencies must not be omitted. The
automatically hashed scope is `orchestration/`, `next_levels/`, and
`distributed_meta.json`, including final binaries, objects, local headers and
configuration. Custom includes/helpers in other directories must appear in the
external dependency tuple. The platform runtime keeps its existing separate
ABI/provenance checks. An artifact without a final
native shared object is not eligible. Hashes bind the producer's promise; they
do not prove kernel behavior or authenticate a producer. Missing, changed, or
unsupported contracts retain conservative submission. No model-specific hash
grants domain-pool access.
