# Put and Get: Tensor-Level Push and Pull

Move whole window slices between ranks with `pld.tensor.put` and
`pld.tensor.get` — the tensor-level point-to-point moves, push vs pull.

> **Prerequisites:** [10-remote_load_store](10-remote_load_store.md). Two
> devices.

**Suggested reading order:** 01 → 02 → 03 → 04 → 05 → **06** — this page is step 06.

## The idea

Step 05 moved one *tile* at a time with `remote_load`/`remote_store`. The
tensor-level primitives `pld.tensor.put` and `pld.tensor.get` move a whole
slice of window memory between ranks in one call, and the runtime handles
staging and (for large transfers) chunking and pipelining for you.

The difference is **who initiates**:

| Primitive | Direction | Initiator | Result |
| --------- | --------- | --------- | ------ |
| `pld.tensor.put(dst, peer, src, atomic=...)` | Push | The **sender** | Sender's `src` slice lands in peer's `dst` |
| `pld.tensor.get(dst, peer, src)` | Pull | The **receiver** | Peer's `src` slice lands in receiver's `dst` |

The example is the same ring shift as step 05, one step: `--mode put` pushes
into the next rank then reads its own `dst` back → `y[r] = x[(r-1) % N]`;
`--mode get` pulls the next rank's `src` → `y[r] = x[(r+1) % N]`.

**Cost card:** one step, one slice exchanged with one peer per rank. For small
slices this is latency-bound; for large slices the runtime's chunked +
pipelined staging overlaps the rounds and turns a latency-bound move into a
bandwidth-bound one (the same trick steps 07–09 use for all-reduce).

## Run it

```bash
# Push (default):
python examples/distributed/06_put_get.py -p a2a3sim -d 0,1

# Pull:
python examples/distributed/06_put_get.py -p a2a3sim -d 0,1 --mode get
```

Expected output:

```text
OK
```

## Walkthrough

The put side:

```python
@pl.jit.incore
def put_step(x, y, src, dst, signal):
    ctx = pld.get_comm_ctx(src)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)

    peer = (my_rank + 1) % nranks
    pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)

    pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    recv = pl.load(dst, [0, 0], [1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y
```

- **Stage.** The sender copies its local slice into its `src` window slice —
  `put` moves *window* memory to *window* memory, so the source must be in the
  window first.
- **Put.** `pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)`
  pushes `src` into peer's `dst`. `atomic` chooses the update mode; `None_`
  means an unconditional overwrite (the simple case).
- **Signal, wait, read.** After the put, the sender notifies the peer and
  waits for the rank that targets *it* — then reads its own `dst`, which the
  previous rank just wrote.

The get side is the receiver-initiated mirror:

```python
    local = pl.load(x, [0, 0], [1, SIZE])
    src = pl.store(local, [0, 0], src)

    peer = (my_rank + 1) % nranks
    pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

    pld.tensor.get(dst, peer=peer, src=src)

    recv = pl.load(dst, [0, 0], [1, SIZE])
    y = pl.store(recv, [0, 0], y)
```

Here every rank still stages its `src` and signals — but the *move* is a
`pld.tensor.get(dst, peer=peer, src=src)`: the receiver pulls the peer's `src`
into its own `dst` after the handshake. Same ring, opposite initiator.

**Chunking and pipelining.** For large transfers the runtime splits the slice
into chunks and pipelines the moves so the next chunk's staging overlaps the
current chunk's transfer. The tiny transfers in this example use the
full-slice form; the chapter reference documents the chunk-size rules.

## Edge cases

> **Fatal pitfall — put without a signal.** `put` is fire-and-forget from the
> sender's perspective; the sender must notify and the receiver must wait
> before reading the destination, or the read races the transfer. **Fix:** for
> every `put`/`get`, pair the notify on the sender side with a wait on the
> receiver side before the destination is read.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Receiver reads stale `dst` | Put/get raced the signal handshake | Notify after the move; wait before reading `dst` |
| Data lands in the wrong rank | `peer` computed wrong | `put` writes to `(r+1) % n`; `get` reads from `(r+1) % n` |
| Golden is one step behind | Pull vs push confusion | `put` mode: `y[r] = x[(r-1)]`; `get` mode: `y[r] = x[(r+1)]` |
| Large transfer stalls or overruns | Chunking rules ignored | Follow the chapter's chunk-size and pipelining constraints |
| `atomic` parameter omitted | Default is not always an overwrite | Pass `atomic=pld.AtomicType.None_` for a plain overwrite |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 06)
- [02-primitives](../distributed/02-primitives.md) §Put and Get — chunking and
  pipelining constraints
- [01-collectives](../distributed/01-collectives.md) — how collectives compose
  these moves (steps 07–15)
- Next step: [05-tutorials](05-tutorials.md) — steps 07–15 (all-reduce) are planned
