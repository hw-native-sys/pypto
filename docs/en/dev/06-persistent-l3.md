# Persistent L3 execution

Prepared distributed programs normally reuse one Simpler worker but enter a
new `Worker.run()` for every dispatch. Programs that allocate communication
windows therefore allocate and release their CommDomains on every call.

Use `persistent=True` to retain generated CommDomains for the lifetime of the
prepared worker:

```python
with decode.prepare(persistent=True) as worker:
    for _ in range(100):
        worker(x, weights, output)
```

The persistent path is opt-in. The default `prepare()` behavior is unchanged.

## Lifecycle

Each PyPTO submission builds its persistent orchestration synchronously in the
calling thread and calls Simpler `Worker.submit()` directly. The returned
`DistributedRunHandle` owns that request until its native completion fence and
cleanup finish. The first use of a generated CommDomain allocates its physical
window; later calls receive a retained lease for the same handle. Closing the
prepared worker stops admission, drains every published handle, and then
releases all retained domains. Request and domain-release errors are propagated
to the caller.

Generated HOST orchestration entries accept an internal `_domain_provider`
keyword. Normal dispatch leaves it unset and continues to call
`orch.allocate_domain`. Persistent dispatch supplies a provider keyed by the
compiled program and generated domain name. Existing generated artifacts must
be regenerated before they can use persistent execution.

## Window contents

Persistent execution restores each retained CommDomain window to zero before
reuse by default. This gives every repeated dispatch fresh-window semantics,
including programs whose communication buffers store synchronization state.
The first dispatch uses the runtime's freshly initialized allocation and does
not perform an additional reset.

Disable the reset only when the program manages the reused communication-buffer
contents itself:

```python
with decode.prepare(
    persistent=True,
    reset_persistent_windows=False,
) as worker:
    worker(x, weights, output)
```

When reset is disabled, later dispatches observe the previous contents
unchanged. The caller must manually zero the affected communication buffers
before reuse or use a protocol such as epochs that safely manages all retained
signal and data state. Reusing stale state without either mechanism can produce
incorrect results or deadlock.

With the default reset enabled, PyPTO synchronously zeros every named local
buffer before reuse. For each reset request, it creates one zero-filled
POSIX-shared-memory host `Buffer` per distinct named-buffer size and reuses that
staging `Buffer` across all matching domains and workers. The staging buffers
remain live until the `Worker.run()` fence returns, so peak staging memory is
the sum of those distinct sizes. The reset copies are part of each repeated
request's host overhead.

This whole-buffer staging is required by the current Simpler Buffer API: `copy_to`
derives the transfer length from the source `Buffer` and exposes neither a
destination offset nor a public Buffer subview. Generated PyPTO domains cover
their windows exactly with named buffers. Reset rejects artifacts with unnamed
window slack because the Buffer API cannot restore that slack to fresh-window
state.

## Multiple compiled programs

Persistent execution supports the existing multi-program prepared worker:

```python
with prefill.prepare(extra_compiled=[decode], persistent=True) as worker:
    worker.run(prefill, prefill_x, weights, kv_cache)
    worker.run(decode, decode_x, weights, kv_cache)
    worker.run(decode, decode_x, weights, kv_cache)
```

Domains are isolated by `(compiled program, generated domain name)`. Prefill's
`comm_d0` and decode's `comm_d0` therefore remain distinct even when their
generated names match. All prepared programs still must satisfy the normal
platform, runtime, and device-ID compatibility checks.

Graph construction remains serialized by `DistributedWorker.submit()` and
Simpler. Once accepted, persistent runs follow the same bounded asynchronous
dispatch contract as ordinary runs: up to two PyPTO metadata frames may be
published, while the backend's negotiated depth determines how many device
runs can actually overlap.

## Runtime dependency

This implementation does not modify Simpler. Each request uses the public
`Worker.submit()` boundary, and its native handle remains attached to PyPTO's
bounded dispatch handle until completion. PyPTO detaches retained CommDomains
from Simpler's per-run release set and releases them when the prepared worker
closes. That retention currently depends on Simpler's private Worker-level
live-domain registry and active run-resource journal (`_live_domains` and
`_building_run_resources.live_domains`); a future public retention API should
encapsulate this lifecycle.

## CLUSTER programs: explicit shared global windows

CLUSTER_0 programs support synchronous multi-program sessions. Prepare all
programs before startup with `extra_compiled`, and select each program with
`run(compiled, ...)`. Each call has its own completion fence. Global domains
remain isolated by program unless explicitly bound to the same session key:

```python
from pypto.ir import SharedDomainBinding

with compute.prepare(
    extra_compiled=[communication],
    persistent=True,
    reset_persistent_windows=False,
    shared_domains=[
        SharedDomainBinding(compute, domain=0, key="activations", slots={0: "values"}),
        SharedDomainBinding(communication, domain=0, key="activations", slots={0: "values"}),
    ],
) as worker:
    worker.run(compute, lhs, rhs, echo)
    worker.run(communication, peer_values)
```

The first program stores results into each rank's global window. Its return
fence ensures all writes have drained before the second program performs
TLOAD. The communication program receives no host copy of the saved data.
One compiled program is one run; the session does not split programs into
phases or add frontend window-copy operations.

Inspect the immutable `compiled.comm_domains` descriptors for domain and slot
ordinals. Generated names are internal and may differ between programs.
`slots` maps each program-local slot ordinal to a canonical physical slot name
and must cover every slot exactly once. Canonical names can reorder slots.
The session validates ordered members, topology, runtime/profile, window and
aligned slot sizes, dtype, shape, strides and byte offsets before reuse.
Equal sizes do not imply sharing. Distinct keys and omitted bindings remain
separate. Shared layouts require static slot sizes. Preparation snapshots the
configuration; changing the original mapping later has no effect, and sharing
does not enter JIT compilation identity.

The session installs one bounded aggregate source package with a canonical
runtime helper and one prefork token pool per node. All programs use the same
actual retained R1 domain and its `(domain_id, generation)` ordering state.
Slot translation selects the real runtime Buffer; it does not wrap or rename
the imported global-domain view. Helper/bootstrap contract mismatches fail
before startup. Cold schema-4 artifacts preserve the domain descriptors;
older artifacts without them must be regenerated for explicit sharing.

The default reset clears every reused global mapping, including signal state
and padding, before the run's first task submission. Runtime control copies
use chunks of at most 8 MiB. Fresh allocations need no additional reset.
With `reset_persistent_windows=False`, data, signals and user epochs all retain
their previous contents. Kernel protocols using credits or signals must manage
their own reset/epoch progression. The compute-then-TLOAD example needs only
the run fences. Ordering-token storage remains stable across both programs.

Normal close explicitly releases retained domains after the last fence.
Failure close drains the worker tree before dropping domain, I/O or prefork
references. An uncertain close retains its journals, refuses new runs, and
permits a close retry. CLUSTER sessions expose synchronous `run` only.
Their retention uses R1's public `retain_after_run` contract and does not
detach domains from private runtime journals.

For validation, SIM global-domain ranks must share one host/POSIX shared-memory
namespace. The SIM backend imports peer windows through `shm_open`; it does
not provide cross-host SIM global windows. True multi-host validation uses the
A3 fabric profile and physical devices. Ordinary remote tensor I/O is a
separate capability.
