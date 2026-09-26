# ReduceScatter: All-to-Chunks

Every rank stages all the chunks; every rank ends with the reduced chunk at
its own index — the reduce-scatter half of two-phase all-reduce — then the
builtin does it in one call.

> **Prerequisites:** [18-allgather](18-allgather.md). Any number of devices
> ≥ 2 (the examples use 2 and 4 sim devices).

**Suggested reading order:** 01 → … → 13 → **14** — this page is step 14.

## The idea

Reduce-scatter is the mirror of allgather: instead of every rank receiving
*all* slices, every rank receives **one reduced chunk** — the chunk at its own
index, reduced (here: summed) across all ranks.

| Aspect | ReduceScatter |
| ------ | ------------- |
| Data in | Every rank's full set of `P` chunks (`N` elements) |
| Data out | Rank `r` ends with `Σ_k chunk_r(inputs[k])` — `N/P` elements |
| Pattern | Stage all chunks → barrier → sum your chunk across peers |
| Cost | `(P-1)/P · N` bytes received — the first half of two-phase |

This is exactly the **first half of step 09's two-phase all-reduce**. Step 13
built the second half (allgather); this step builds the first. Together they
are the two-phase schedule you already ran as one builtin.

## Run it

```bash
# Hand-rolled: stage all chunks, barrier, sum your chunk across peers.
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1

# Reveal: pld.tensor.reduce_scatter in one call.
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1 --mode builtin

# The same source at P=4:
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1,2,3
python examples/distributed/14_reduce_scatter.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

Expected output:

```text
OK
```

The golden is per-rank: `out[r]` must equal the element-wise sum of chunk `r`
across all ranks — a *different* chunk per rank, so a rank that reduces the
wrong chunk fails.

## Walkthrough

Both modes share one `[nr, SIZE]` window: each rank stages chunk `c` at row
`c` and reduces row `my_rank`. The hand-rolled kernel:

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    # Phase 1 — stage every chunk at its row, so each peer can read it.
    for c in pl.range(nranks):
        chunk = pl.load(x, [0, c * SIZE], [1, SIZE])
        data = pl.store(chunk, [c, 0], data)

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nranks):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    for src in pl.range(nranks):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — reduce: sum row my_rank across every peer.
    acc = pl.load(data, [my_rank, 0], [1, SIZE])
    for peer in pl.range(nranks):
        if peer != my_rank:
            recv = pld.tile.remote_load(data, peer=peer, offsets=[my_rank, 0], shape=[1, SIZE])
            acc = pl.add(acc, recv)
    return pl.store(acc, [0, 0], y)
```

- **You stage ALL chunks, not just your own.** Every rank publishes the whole
  `[nr, SIZE]` matrix so any peer can read the specific chunk it needs. This
  is the opposite of allgather's one-slice stage — the data you publish is
  `P` slices, the data you consume is one.
- **The reduction is a local loop.** The remote reads are *adds*, not stores:
  `acc` accumulates chunk `my_rank` from every peer. The order of the loop
  differs between ranks, which is why the golden uses a tolerance (reduction
  order is not the same as torch's).

The reveal replaces phases 2–3 with one call:

```python
    for c in pl.range(nranks):
        chunk = pl.load(x, [0, c * SIZE], [1, SIZE])
        data = pl.store(chunk, [c, 0], data)

    data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
    acc = pl.load(data, [my_rank, 0], [1, SIZE])
    return pl.store(acc, [0, 0], y)
```

- **`op=` picks the reduction — the full family on this path.** The parameter
  defaults to `pld.ReduceOp.Sum`; on the InCore composite rail this example
  uses, `Max`, `Min`, and `Prod` are supported too, the same dispatch
  `pld.tensor.allreduce` uses. The restriction is narrower elsewhere: the
  **HOST builtin** rail (a different lowering, not exercised by this step)
  lowers `Sum`/FP32 only and rejects anything else with `builtin.tensor.
  reduce_scatter op must be ReduceOp.Sum (got int N)`. Don't carry that
  HOST-only restriction across to the InCore form you're writing here.
- **Row `my_rank` of the window is your reduced chunk** — the same
  row-per-chunk layout the hand-rolled version used.

### The IR diff (the teaching artifact)

- `--mode hand` lowers to the four phases above: `P` stores, the ready
  barrier, and `P-1` remote loads accumulated with adds — then the result goes
  straight out to `y`.
- `--mode builtin` expands into that shape **plus a second barrier you did not
  write**. Because the composite writes the reduced chunk back into
  `target[my_rank]` rather than out to `y`, it has a write-after-read hazard
  your version does not: a fast rank could overwrite its own row while a
  slower peer is still reading it. So the expansion is ready barrier →
  reduce → **post-reduce barrier** → `tile.store` into row `my_rank`, and the
  signal carries **2 credits per call** instead of one.
- That is the lesson of this diff: the extra barrier is not overhead the
  compiler failed to optimise away — it is the price of the in-place
  window-as-result form. Your version avoids it only by writing somewhere the
  peers never read.
- As with the other composites, a self-clearing epilogue subtracts both
  credits back out, so the signal is reusable on the next call.

**Cost card (per rank):** you receive `(P-1)/P · N` bytes and end with `N/P`
reduced elements — the first half of two-phase all-reduce (step 09), whose
second half (allgather) you built in step 13.

## Edge cases

> **Fatal pitfall — reducing your own chunk from the wrong place.** The
> accumulation must start from *your* window row (which includes your own
> contribution) and then add every peer's row. If you instead reduce only the
> remote rows, your own contribution is missing and the golden fails by a
> known amount. **Fix:** seed `acc` with `pl.load(data, [my_rank, 0], ...)`
> before the peer loop.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Golden off by your own chunk | Accumulator seeded from zeros, not your row | Load `[my_rank, 0]` before the peer loop |
| Every rank gets the same chunk | Reduced row fixed to `[0, 0]` | Reduce row `my_rank` |
| Wrong chunk boundaries | Chunk offset arithmetic wrong | Chunk `c` at `[0, c*SIZE]` in `x`, row `c` in `data` |
| Result differs from torch (tolerance ok) | Reduction order differs per rank | Compare with a tolerance, not exact equality |
| `builtin.tensor.reduce_scatter op must be ReduceOp.Sum` | Called the **HOST builtin** path with `op=Max`/`Min`/`Prod` | Use `pld.ReduceOp.Sum` on the HOST path, or use the InCore composite path (this step's example), which supports the full family |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 14)
- [01-collectives](../distributed/01-collectives.md) §ReduceScatter — the full API
- [14-allreduce_two_phase](14-allreduce_two_phase.md) — the two-phase all-reduce
  this step is the first half of
- [04-debugging](04-debugging.md) — the canonical failure catalog for distributed programs
- Next step: [20-all_to_all](20-all_to_all.md) — a different slice for every peer
