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

The PyPTO worker starts one background dispatcher and sends requests to it
through a Python queue. Every request executes inside its own Simpler
`Worker.run()` completion fence. The first use of a generated CommDomain
allocates its physical window; later calls receive a retained lease for the
same handle. Closing the prepared worker stops the dispatcher and releases all
retained domains. Request or domain-release errors are propagated to the
caller instead of being silently discarded by the background thread.

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

With the default reset enabled, PyPTO synchronously zeros every local window
before reuse. A 1 MiB read-only host chunk is allocated before the chip workers
fork and is copied repeatedly for larger windows. The reset copy is part of
each repeated request's host overhead.

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

Requests execute serially through one queue. Persistent mode does not make one
worker execute multiple L3 DAGs concurrently.

## Runtime dependency

This implementation does not modify Simpler. Each queued request uses the
public `Worker.run()` completion boundary. PyPTO detaches retained CommDomains
from Simpler's per-run release set and releases them when the prepared worker
closes. That retention currently depends on Simpler's private live-domain and
deferred-release hooks; a future public retention API should encapsulate this
lifecycle.
