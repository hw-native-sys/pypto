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

PyPTO resets each named buffer with `copy_to(dst, src)`, whose default transfer
length is the source `Buffer` size. Generated domains cover the requested
window exactly with named buffers. Reset rejects artifacts with unnamed slack
inside that requested extent because it has no named buffer to restore there.
The runtime's physical mapping may be larger due to allocation alignment, as
on A5. Reset accepts this extra capacity and copies only the named buffers;
it does not expose or reset the mapping's trailing padding. A physical mapping
smaller than the requested window is rejected before copying that domain's buffers.

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
bounded dispatch handle until completion. Each request acquires its domains
through `Orchestrator.get_or_allocate_domain(..., retain_after_run=True)`.
Simpler validates the configuration on reuse, registers the domain with the
current run's completion fence, and retains its allocation across successful
runs. PyPTO releases the domains after draining all dispatch handles when the
prepared worker closes; Simpler owns cleanup on failure. No private domain
registry or run-resource journal is modified by PyPTO.

## Worker-owned external domains

A prepared worker can create a retained domain with `create_comm_domain` and pass
its named windows to independently compiled HOST entries. Each entry declares a
static, contiguous ND `pld.DistributedTensor` for each window. Each invocation
selects the owner and slot through its actual `DomainTensor` arguments.
Internal persistent-window reset applies only to program-owned allocations.

The schema 4 sidecar marks external parameters with `is_distributed` in declaration order.
Reload uses these parameters to reconstruct the scalar suffix. Each window
occupies one scalar slot holding its buffer index. The paired hidden context at
the end occupies three uint64 slots for the eight-byte little-endian owner ID,
domain ID and generation. The runner passes these scalars through the native HOST callback’s `TaskArgs`. Ordinary tensors and typed Python scalars remain available by name for HOST slicing and control flow. At each
invocation the HOST entry resolves each distinct ref through the active
orchestrator, materializes each member's named buffer and device context, and
attaches one ordering buffer per distinct domain to a CHIP dispatch.
HOST orchestration passes distributed windows from the start of their named buffer.
Submissions sharing a domain wait for the previous use to complete. Release
and worker close also drain accepted uses. Cold reload requires callers to
create the runtime resources and supply the windows again.

An interrupted wait keeps the dispatch arguments and its domain alive. If native cancellation cannot confirm completion, the worker retains them until native teardown succeeds.
