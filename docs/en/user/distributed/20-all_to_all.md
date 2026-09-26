# AllToAll: A Different Slice for Every Peer

Every rank sends a *distinct* slice to every peer and receives a distinct
slice from every peer — the most general of the point-to-point patterns — then
the builtin does it in one call.

> **Prerequisites:** [19-reduce_scatter](19-reduce_scatter.md). Any number of
> devices ≥ 2 (the examples use 2 and 4 sim devices).

**Suggested reading order:** 01 → … → 14 → **15** — this page is step 15.

## The idea

All the previous collectives move the *same* shape everywhere: broadcast moves
one slice to all, allgather moves each rank's slice to all, reduce-scatter
reduces. All-to-all is different: rank `r` sends a **different `N/P` slice to
every peer** — the slice for destination `d` is not the slice for destination
`e`. This is the personalized exchange.

| Aspect | AllToAll |
| ------ | -------- |
| Data in | `P` distinct chunks per rank (one per destination) |
| Data out | `P` distinct chunks per rank (one from each source) |
| Pattern | Push each dest chunk (`put`) → barrier → read back |
| Cost | `(P-1)/P · N` bytes received; no two ranks want the same bytes |

This is the pattern behind real dispatch/combine workloads — distributed MoE
sends each token-group to a different expert's rank; AllGather-GEMM pipelines
route per-shard data. The walkthrough's See-also links those applications by
name (without restating them).

## Run it

```bash
# Hand-rolled: put each dest chunk, barrier, read back.
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1

# Reveal: pld.tensor.all_to_all in one call.
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1 --mode builtin

# The same source at P=4:
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1,2,3
python examples/distributed/15_all_to_all.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

Expected output:

```text
OK
```

The golden is constructed so each chunk is unique: `input[r, d, j] =
r*1000 + d*100 + j` — the source, destination, and element are all encoded in
the value. Any routing mistake (wrong source, wrong destination) shows up as
a wrong value, not a subtle shape issue.

## Walkthrough

Both modes share one `[nr, SIZE]` window. Rank `r`'s chunk-for-`d` ends up at
row `r` of destination `d`'s window, then rank `d` reads row `src` of its
own window — in the hand-rolled kernel the caller does the write explicitly;
in the reveal below, the push happens inside the call instead. The
hand-rolled kernel uses step 06's `put`:

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # Phase 1 — push: write chunk-for-dest into dest's window at our row.
    for dest in pl.range(nranks):
        pld.tensor.put(data, dest, x, [my_rank, 0], [dest, 0], [1, SIZE])

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nranks):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.Set)
    for src in pl.range(nranks):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — read-back: row src of our window holds src's chunk for us.
    for src in pl.range(nranks):
        chunk = pl.load(data, [src, 0], [1, SIZE])
        y = pl.store(chunk, [src, 0], y)
    return y
```

- **The push is a loop over destinations.** `pld.tensor.put(data, dest, x,
  [my_rank, 0], [dest, 0], [1, SIZE])` writes `x`'s row `dest` into peer
  `dest`'s window at *our* row. Each iteration targets a different peer with a
  different source row — the "personalized" part.
- **`Set` not `AtomicAdd` for the notify.** Each rank is the only writer of
  its row (`[my_rank, 0]`), so the barrier here uses `Set`/`Ge(1)` — the
  single-writer form from step 04's walkthrough.
- **Read-back completes the exchange.** After the barrier, row `src` of our
  own window holds the chunk rank `src` intended for us.

The reveal replaces phases 1–3 with one call:

```python
    result = pld.tensor.all_to_all(x, data, signal)
    for src in pl.range(nranks):
        chunk = pl.load(result, [src, 0], [1, SIZE])
        y = pl.store(chunk, [src, 0], y)
    return y
```

- **The source is your local `x` (a plain `pl.Tensor`)** with row `d` = the
  chunk for destination `d` — the same layout the hand-rolled loop used.
- **The window becomes the result** (row `src` = the chunk from rank `src`),
  and `input`/`target` must be **separate** buffers.

### The IR diff (the teaching artifact)

Like step 13's push/pull reversal, the interesting part isn't which bytes
move — both modes move the same chunks to the same destinations — it's which
side of the transfer the barrier lands on.

- `--mode hand` lowers to the three phases above: `P` puts (one per
  destination), then the `Set`/`Ge(1)` barrier, then the read-back loop — the
  barrier comes *after* your own pushes but *before* you trust any peer's,
  because your reads would otherwise race their pushes.
- `--mode builtin` expands into the same shape — your per-destination chunks
  pushed via `pld.tile.put`, then the barrier, then the row read-back — but
  the barrier's role has flipped. Because the transfers come *first*, it is a
  **completion** barrier telling you every peer's push has landed, not a
  **readiness** barrier gating a read.
- The composite appends a **self-clearing epilogue** after the barrier that
  subtracts this call's credits back out, so the same signal is reusable by a
  later collective ([21-putting_it_together](21-putting_it_together.md)). The
  HOST builtin additionally chunks large transfers at the orchestration
  level — the InCore composite this example uses does not need to.
- That barrier-role flip is the lesson of this diff: whether a barrier gates
  a read or confirms a write depends on which side of the transfer it sits
  on, not on anything about barriers themselves.

**Cost card (per rank):** every rank sends a *different* `N/P` slice to each
peer — `(P-1)/P · N` bytes received, and no two ranks receive the same bytes.
This is why all-to-all is the most expensive point-to-point pattern to
schedule: every pair exchanges distinct data.

## Edge cases

> **Fatal pitfall — reusing one window for source and result.** All-to-all
> delivers its *result* in place — `target` is rebound as the window-as-result
> — but the *source* is a separate buffer: the API requires `input` and
> `target` to be different — and on the InCore path `x` is a plain
> `pl.Tensor` while `data` is the window, so they are two separate buffers of
> two different kinds, not one window used twice.
> Passing the same window as both lets the push overwrite chunks you still
> need to send. **Fix:** allocate a distinct target window (as the example
> does) — the builtin enforces this too.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Chunk for dest `d` lands in the wrong peer | `put` peer ≠ destination row | `put(data, dest, x, ..., [dest, 0], ...)` |
| Output has my own chunks, not peers' | Read-back from own rows | Read row `src` after the barrier |
| Source/target aliasing corruption | Same buffer as input and target | Separate `x` and `data` windows |
| One rank hangs | `Set` notify/wait mismatch | Notify writes row `my_rank` of peer; wait reads row `src` of self |
| Wrong values at P=4 only | Missing barrier before read-back | Barrier between the put loop and the read loop |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 15)
- [01-collectives](../distributed/01-collectives.md) §AllToAll — the full API
- [03-execution](../distributed/03-execution.md) — `DistributedWorker` and
  device-side staging for production all-to-all
- `examples/runtime/distributed_callback.py` — host-side runtime-bound
  callbacks around an L3 distributed program
- [11-put_get](11-put_get.md) — the `put`/`get` substrate this step builds on
- More advanced applications (not restated here): pypto-lib
  [#869](https://github.com/hw-native-sys/pypto-lib/pull/869) (AllGather-GEMM,
  an allgather pattern from step 13) and the DeepSeek-V4 distributed MoE
  dispatch/combine (an all-to-all pattern from this step)
- [04-debugging](04-debugging.md) — the canonical failure catalog for distributed programs
- Next step: [21-putting_it_together](21-putting_it_together.md) — compose
  three collectives in one kernel
