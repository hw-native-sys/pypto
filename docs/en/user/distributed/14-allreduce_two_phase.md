# All-Reduce 2: Two-Phase — Reduce-Scatter Then All-Gather

The same result as step 08, in roughly half the remote traffic. Instead of
every rank reading every peer's *full* slice, split the slice into `P`
chunks: first **reduce-scatter** so rank `r` owns the reduced chunk `r`, then
**all-gather** so every rank collects every chunk.

> **Prerequisites:** [13-allreduce_mesh](13-allreduce_mesh.md). Four sim
> devices recommended (the traffic saving vs mesh is only observable at P≥4);
> `SIZE` must divide evenly by the rank count.

**Suggested reading order:** 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → **09** — this page is step 09.

## The idea

Mesh transfers `(P-1) * N` bytes per rank because every rank reads every
peer's *entire* slice. But the sum only needs `N` result values per rank — so
most of what mesh moves is duplicate work. Two-phase restructures the same
sum into two stages, each moving `N/P`-sized pieces:

1. **Reduce-scatter (RS):** rank `r` is the *owner* of chunk `r`. Every rank
   reads every peer's chunk `r` (a `N/P`-sized piece, not the full slice) and
   sums locally. After this, rank `r` alone holds the reduced chunk `r`.
2. **All-gather (AG):** every rank reads every peer's reduced chunk, one per
   rank, and assembles the full result.

Each stage moves `(P-1) * N/P` bytes, so the total is `2 * (P-1) / P * N` —
roughly half of mesh's traffic, at the price of a second barrier.

## Run it

```bash
# P=4 (the saving vs mesh shows here) and P=2:
python examples/distributed/09_allreduce_two_phase.py -p a2a3sim -d 0,1,2,3
python examples/distributed/09_allreduce_two_phase.py -p a2a3sim -d 0,1
```

Expected output:

```text
OK
```

## Walkthrough

Same `@pl.program` class form as step 08, but now with a **rank-count
factory** — step 08 needed none. The factory is what makes `nr` a compile-time
constant, and this step is the first that requires one: the chunk size
`SIZE // nr` is a **tile shape**, and tile shapes must be known when the kernel
is compiled. There are also **two** windows now (`data` for the staged inputs,
`result` for the reduced chunks) and a **two-row signal** (`[2, nr]` — one row
per barrier round):

```python
# Phase 1 — stage this rank's slice into its window slot.
local = pl.load(x, [0, 0], [1, SIZE])
data = pl.store(local, [0, 0], data)

# Barrier A (signal row 0) — all inputs staged before the RS reads.
# (notify all peers / wait all peers, as in step 08, but on row 0)

# Phase 2 — reduce-scatter: rank r owns chunk r of the result.
acc = pl.load(data, [0, my_rank * chunk], [1, chunk])
for peer in pl.range(nranks):
    if peer != my_rank:
        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, my_rank * chunk],
                                    shape=[1, chunk])
        acc = pl.add(acc, recv)
result = pl.store(acc, [0, my_rank * chunk], result)

# Barrier B (signal row 1) — all reduced chunks staged before the AG reads.

# Phase 3 — all-gather: rank r reads every rank's reduced chunk.
for c in pl.range(nranks):
    recv = pld.tile.remote_load(result, peer=c, offsets=[0, c * chunk], shape=[1, chunk])
    y = pl.store(recv, [0, c * chunk], y)
```

- **Chunk ownership.** The RS loop indexes every peer's window at
  `[0, my_rank * chunk]` — each rank reads the *same* chunk (its own) from
  every peer. After the loop, rank `r`'s `result` holds the reduced chunk `r`.
- **The AG reads one chunk per peer** at `[0, c * chunk]` from peer `c` — the
  pieces already reduced in the RS — and writes them into the output in order.
- **Two barriers, two signal rows.** The `[2, nr]` signal gives each barrier
  its own row, so the monotonic counters (`Ge(1)`) don't leak between rounds.

**Cost card (per rank):** `2 * (P-1) / P * N` bytes — `(P-1)` reads of `N/P`
bytes in the RS, `(P-1)` reads of `N/P` bytes in the AG. Roughly half of
mesh's `(P-1) * N`, at the cost of one extra barrier round.

## Edge cases

> **Fatal pitfall — reusing one signal row for both barriers.** The counters
> are monotonic: after barrier A, `Ge(1)` is already satisfied on that row, so
> barrier B returns immediately and the AG reads can race the RS stores.
> **Fix:** give each round its own row (the `[2, nr]` signal) — the same
> discipline the ring step generalises to one row per round.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| AG reads stale chunk data | Both barriers used the same signal cell | One signal row per round (`[2, nr]`) |
| Wrong result only at P≥4 | Chunk size mismatch (`SIZE % P != 0`) | Run P where `SIZE` divides evenly (2, 4) |
| Result has chunk `r` in the wrong position | AG assembled chunks out of order | Read chunk `c` from peer `c`, write at `[0, c * chunk]` |
| All ranks hold the RS result but not the sum | AG missing (only reduce-scatter ran) | Add the AG loop: every rank reads every reduced chunk |
| Same result on every rank but ≠ torch sum | Reduction order differs (not a bug) | Compare with a tolerance |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 09)
- [13-allreduce_mesh](13-allreduce_mesh.md) — the baseline this step improves (step 08)
- [01-collectives](01-collectives.md) §AllReduce — reference (Mesh Mode, Ring Mode)
- Next step: [15-allreduce_ring](15-allreduce_ring.md) — same bytes, constant per-step size
