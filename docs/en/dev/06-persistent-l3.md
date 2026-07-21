# Persistent L3 execution

Prepared distributed programs normally reuse one Simpler worker but enter a
new `Worker.run()` for every dispatch. Programs that allocate communication
windows therefore allocate and release their CommDomains on every call.

Use `persistent=True` to keep one L3 orchestration active for the lifetime of
the prepared worker:

```python
with decode.prepare(persistent=True) as worker:
    for _ in range(100):
        worker(x, weights, output)
```

The persistent path is opt-in. The default `prepare()` behavior is unchanged.

## Lifecycle

The PyPTO worker starts one background orchestration and sends requests to it
through a Python queue. The first use of a generated CommDomain allocates its
physical window. Later calls receive a retained lease for the same handle.
Closing the prepared worker stops the orchestration and releases all retained
domains through the normal Simpler `Worker.run()` cleanup path. An error from
that final drain or domain release is propagated by `close()` instead of being
silently discarded by the background thread.

Generated HOST orchestration entries accept an internal `_domain_provider`
keyword. Normal dispatch leaves it unset and continues to call
`orch.allocate_domain`. Persistent dispatch supplies a provider keyed by the
compiled program and generated domain name. Existing generated artifacts must
be regenerated before they can use persistent execution.

## Fresh-window semantics

`alloc_window_buffer` currently means fresh, zero-filled scratch storage for
each dispatch. Reusing the physical allocation must preserve that semantic,
especially for monotonic notify/wait counters.

Before reusing a program's retained domain, PyPTO synchronously zeros every
local window on every participating worker. A 1 MiB read-only host chunk is
allocated before the chip workers fork and is copied repeatedly for larger
windows. The first dispatch uses the runtime's freshly initialized allocation
and does not perform this reset.

This avoids model-specific epoch parameters, but the reset copy is part of
each repeated request's host overhead. Measure it against dynamic domain
allocation before enabling persistent execution in production.

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

This implementation does not modify Simpler. It uses the public domain, copy,
and scope APIs, plus the private orchestration drain and Worker cleanup APIs,
to establish a device-completion boundary between queue requests while keeping
the outer `Worker.run()` alive. A future public Simpler request-boundary API
should encapsulate that drain and cleanup sequence.
