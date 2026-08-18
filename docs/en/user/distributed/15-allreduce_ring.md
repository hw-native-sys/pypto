# All-Reduce 3: Ring — Constant Per-Step Size

The two-phase shape from step 09, but each rank only ever moves data with its
**left neighbour**. Instead of reading `P-1` peers per stage, the chunks rotate
around the ring: `2 * (P-1)` steps, each moving one `N/P`-sized chunk — so the
per-step transfer stays `N/P` as `P` grows under weak scaling (for a fixed
`N` the chunk shrinks instead). Synchronization is
neighbour-local too: each rank notifies its **right** neighbour after a store
and waits on its **left** neighbour before a read — no full barrier per round.

> **Prerequisites:** [14-allreduce_two_phase](14-allreduce_two_phase.md).
> Four sim devices recommended (the constant per-step size is only observable
> at P≥4); `SIZE` must divide evenly by the rank count.

**Suggested reading order:** 01 → … → 09 → **10** — this page is step 10.

## The idea

Two-phase halves mesh's traffic but still reads every peer per stage — O(P)
peers, each a `N/P` chunk. The ring removes the "every peer" part: arrange the
ranks in a circle and pass chunks to the **left neighbour**. The same
reduce-scatter + all-gather split now takes `P-1` steps each:

- **Reduce-scatter (P-1 steps):** in each step a chunk travels one hop and is
  added at its destination. After `P-1` steps every rank holds the reduced
  chunk it owns.
- **All-gather (P-1 steps):** the reduced chunks keep circulating, and each
  rank copies every chunk as it passes. After `P-1` more steps every rank has
  the full result.

Total bytes equal two-phase (`2 * (P-1) / P * N`), but each step moves only
`N/P` — under weak scaling (a workload that grows with `P`) the per-step size
stays constant as `P` grows, which is what keeps the ring efficient at large
world sizes; for a fixed `N` the chunk would shrink instead.

## Run it

```bash
# P=4 (the constant per-step size shows here) and P=2:
python examples/distributed/10_allreduce_ring.py -p a2a3sim -d 0,1,2,3
python examples/distributed/10_allreduce_ring.py -p a2a3sim -d 0,1
```

Expected output:

```text
OK
```

## Walkthrough

The kernel is monolithic (one InCore function) and the signal generalises the
two-phase idea: `[2 * (nr-1), nr]` — **one row per round**. The chunk index
arithmetic is the heart of the schedule:

```python
left = (my_rank - 1 + nranks) % nranks      # never negative

# Reduce-scatter: (nr-1) steps.
for s in pl.range(nranks - 1):
    step = s + 1
    recv_add_idx = (my_rank - step - 1 + nranks) % nranks
    left_send_idx = (left - step + nranks) % nranks
    # Wait for the left neighbour's round-s chunk (signal row s), then:
    pld.system.wait(signal, offsets=[s, left], expected=1, cmp=pld.WaitCmp.Ge)
    recv = pld.tile.remote_load(scratch, peer=left,
                                offsets=[0, left_send_idx * chunk],
                                shape=[1, chunk])
    acc = pl.load(scratch, [0, recv_add_idx * chunk], [1, chunk])
    acc = pl.add(acc, recv)
    scratch = pl.store(acc, [0, recv_add_idx * chunk], scratch)
    # The store stages next round's send: signal the right neighbour (row s+1).
    pld.system.notify(signal, peer=right, offsets=[s + 1, my_rank],
                      value=1, op=pld.NotifyOp.AtomicAdd)

# All-gather: (nr-1) steps (rows nranks-1 .. 2*(nranks-1)-1), copying the
# left neighbour's send chunk into the local chunk.
```

- **`left = (my_rank - 1 + nranks) % nranks`.** The `+ nranks` keeps the
  dividend non-negative — a bare `(my_rank - 1) % nranks` yields `-1` at
  rank 0 under truncating modulo (the step-06 lesson, now on the index side).
- **Chunks rotate, not ranks.** In round `s`, the chunk you add from the left
  and the chunk you forward both shift by one (`- step`), so every chunk
  visits every rank exactly once per phase.
- **Neighbour-ready handshakes, not barriers.** Each round gets its own signal
  row — notify the **right** neighbour after a store, wait on the **left**
  neighbour before a `remote_load`. The monotonic `Ge(1)` counters never leak
  between rounds (the step-09 discipline, now at `2*(P-1)` rows), and only the
  two adjacent ranks ever synchronize: O(P) signals **per rank** (O(P²)
  system-wide) — versus the O(P²) per rank a full-mesh barrier per round would
  cost.

**Cost card (per rank):** `2 * (P-1) / P * N` total — the same as two-phase —
but in `2*(P-1)` steps of `N/P` bytes each. Under weak scaling the per-step
size stays **constant as P grows** — unlike mesh's `(P-1) * N` per step —
which is the ring's reason to exist.

## Edge cases

> **Fatal pitfall — a negative left-neighbour index.** `(my_rank - 1) %
> nranks` is `-1` at rank 0 under truncating modulo — an invalid peer the
> `remote_load` then targets. **Fix:** always write
> `(my_rank - 1 + nranks) % nranks` (at P=2 rank 0's left neighbour is rank 1;
> the `+ nranks` keeps the dividend non-negative).

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Hang at P=2 | Negative dividend in the left-neighbour index | `(my_rank - 1 + nranks) % nranks` |
| Wrong chunks in the result | Chunk index arithmetic off by one in a round | Trace `recv_add_idx` / `left_send_idx` for `s=0` on paper |
| Two handshakes share a signal row | Row index reused across RS and AG | RS rows `0..P-2`, AG rows `P-1..2(P-1)-1` |
| Result correct at P=2 only | P=2 has a single round, hiding rotation bugs | Run P=4 and check every chunk position |
| Same result on every rank but ≠ torch sum | Reduction order differs (not a bug) | Compare with a tolerance |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 10)
- [14-allreduce_two_phase](14-allreduce_two_phase.md) — the two-phase shape (step 09)
- [01-collectives](01-collectives.md) §AllReduce — reference (Ring Mode, signal shape, `Sum`/`FP32`)
- Next step: [16-allreduce_reveal](16-allreduce_reveal.md) — the builtin that picks mesh or ring for you
