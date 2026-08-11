# AllGather: All-to-All Slices

Every rank publishes its slice, every rank ends with the rank-ordered
concatenation of all slices — the all-gather half of two-phase all-reduce —
then the builtin does it in one call.

> **Prerequisites:** [17-broadcast](17-broadcast.md). Any number of devices
> ≥ 2 (the examples use 2 and 4 sim devices).

**Suggested reading order:** 01 → … → 12 → **13** — this page is step 13.

## The idea

Allgather inverts broadcast's asymmetry: every rank is both a producer and a
consumer. Each rank contributes one slice (`N/P` elements), and every rank
ends with the **rank-ordered concatenation** `[x[0], x[1], …, x[P-1]]`.

| Aspect | AllGather |
| ------ | --------- |
| Data in | Every rank's slice |
| Data out | Concatenation of all slices, on **every** rank |
| Pattern | Stage your slice → barrier → read every peer's slice |
| Cost | Each rank sends `N/P` to every peer: `(P-1)/P · N` received |

You met this pattern once before: step 09's two-phase all-reduce is
reduce-scatter **followed by allgather**. This step builds the allgather half
on its own; step 14 builds the reduce-scatter half.

## Run it

```bash
# Hand-rolled: stage, barrier, remote_load every peer.
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1

# Reveal: pld.tensor.allgather in one call.
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1 --mode builtin

# The same source at P=4:
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1,2,3
python examples/distributed/13_allgather.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

Expected output:

```text
OK
```

The golden is the rank-ordered concatenation — identical on every rank — so
any rank producing the wrong *order* (or its own slice) fails.

## Walkthrough

Both modes share one `[nr, SIZE]` window: each rank stages at its own row and
reads every row back. The hand-rolled kernel:

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)

    # Phase 1 — stage this rank's slice into its own row.
    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [my_rank, 0], data)

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nr):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    for src in pl.range(nr):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — gather: pull every peer's row into the output.
    for peer in pl.range(nr):
        recv = pld.tile.remote_load(data, peer=peer, offsets=[peer, 0], shape=[1, SIZE])
        y = pl.store(recv, [0, peer * SIZE], y)
    return y
```

- **Row `my_rank` is your slot.** Staging at `[my_rank, 0]` instead of
  broadcast's single root slot is what makes the exchange symmetric: every
  rank writes a distinct row, so no two ranks ever collide.
- **The gather is a loop over peers** — `remote_load` peer `p` at row `p`,
  stored at output offset `p * SIZE`. The output is the rank-ordered
  concatenation, which is why the loop order (and the offset arithmetic)
  matter: slot `p` must hold rank `p`'s slice.

The reveal replaces phases 2–3 with one call — the push-based form:

```python
    data = pld.tensor.allgather(x, data, signal)   # stage + barrier + gather

    for src in pl.range(nr):
        chunk = pl.load(data, [src, 0], [1, SIZE])
        y = pl.store(chunk, [0, src * SIZE], y)
```

- **The source is your local `x` (a plain `pl.Tensor`), not the window.** The
  push-based allgather stages it for you; the target window becomes the
  `[nr, SIZE]` result (row `src` = rank `src`'s slice).
- **Same row-per-rank layout** as the hand-rolled version — the builtin
  chooses the same schedule.

### The IR diff (the teaching artifact)

- `--mode hand` lowers to the three phases above: one store into your row,
  the notify/wait barrier, and `P` `remote_load`s — one per peer.
- `--mode builtin` expands into the same shape: your slice staged, the ready
  barrier on the `[nr, 1]` signal, and the per-peer reads. The composite adds
  nothing — it is the hand-rolled schedule, expressed by the compiler. (For
  larger slices the lowering may chunk the transfers, the pipelining step 11
  mentioned; the tiny slices here use the full-slice form.)

**Cost card (per rank):** each rank sends `N/P` bytes to every peer, so each
rank receives `(P-1)/P · N` bytes — the gather half of two-phase all-reduce
(step 09) moved `(P-1)/P · N` per phase too.

## Edge cases

> **Fatal pitfall — gathering into the wrong slot.** If the output offset does
> not match the peer's rank (`y[peer]` written from peer `p` but offset by
> `p+1`), every rank is internally consistent and the golden still fails —
> order is the contract. **Fix:** offset `peer * SIZE` for peer `peer`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Rows in the wrong order | Output offset ≠ peer rank | Store peer `p` at `[0, p * SIZE]` |
| Every rank shows its own slice | Read from own window instead of peers | `remote_load` each `peer` at `[peer, 0]` |
| `pld.tensor.allgather` source rejected | A tile passed as the source | Pass a plain `pl.Tensor` (or a `DistributedTensor`) |
| Concatenation has gaps/overlaps | Stage/gather offset mismatch | Stage row `my_rank`; read row `peer` |
| Stale data at P=4 | Barrier missing between stage and gather | Notify/wait covers all `nr` peers before the read loop |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 13)
- [01-collectives](../distributed/01-collectives.md) §AllGather — the full API
- [14-allreduce_two_phase](14-allreduce_two_phase.md) — the gather half of the
  two-phase all-reduce this step isolates
- Next step: [19-reduce_scatter](19-reduce_scatter.md) — the reduce half
