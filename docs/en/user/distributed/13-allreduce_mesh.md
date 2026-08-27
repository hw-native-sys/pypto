# All-Reduce 1: Mesh — Every Rank Reads Every Peer

The first collective of the ladder. Every rank contributes its slice; every
rank ends with the element-wise **sum of all slices**. The mesh algorithm is
the simplest spelling of that: one barrier, then each rank `remote_load`s
every peer's slice and accumulates locally.

> **Prerequisites:** [12-dynamic_rank_count](12-dynamic_rank_count.md). Four
> sim devices recommended — at two ranks the three all-reduce variants of
> steps 08-10 collapse into the same exchange, and their differences are only
> observable at P=4.

**Suggested reading order:** 01 → 02 → 03 → 04 → 05 → 06 → 07 → **08** — this page is step 08.

## The idea

All-reduce is the operation every collective comparison starts from: you have
`P` ranks, each holding a slice; after the call every rank holds the reduction
of *all* slices. Step 11 reveals the builtin; here you build it by hand, and
the cost card you develop is the reason the builtin exists and what it chooses
between.

The mesh is the naive baseline: **every rank reads every peer**. That is
O(P) remote traffic per rank — simple, round-heavy, and the yardstick the
two-phase and ring steps are measured against.

## Run it

```bash
# P=4 (the comparison steps require it) and P=2:
python examples/distributed/08_allreduce_mesh.py -p a2a3sim -d 0,1,2,3
python examples/distributed/08_allreduce_mesh.py -p a2a3sim -d 0,1
```

Expected output:

```text
OK
```

## Walkthrough

The rank count is never a compile-time constant here. It is
`NR = pl.dynamic("NR")` in the annotations and `pld.world_size()` in the host
body, so one module-level `@pl.program` serves any world size picked with `-d`
— no rank-count factory. This mirrors
`tests/st/distributed/collectives/test_l3_allreduce.py`, the system test for
this same collective.

Steps 01-07 use the `@pl.jit` family, and the class form here is a
presentational choice rather than a requirement. `signal` is a window shaped
`[pld.world_size(), 1]`, whose row count no static rule can fold. `@pl.jit`
gives such a dim a synthesized dynamic dimension, declares it via `pl.dynamic`
in the program it generates, and the kernel binds it from the actual argument's
descriptor — structurally what the class form below writes by hand as `NR`,
differing only in the symbol's name. The class form is used here because it
makes that shape explicit next to the system test it mirrors.

Steps 09 and 10 switch for a reason that *is* binding: their chunk size
`SIZE // nr` is a **tile shape**, and tile shapes must be known when the kernel
is compiled, so those genuinely need a compile-time rank count and a factory.
That limit applies to either decorator family — a dynamic dim reaching a tile
shape is rejected downstream by `InitMemRef`, not by the frontend. A signal row
count is not a tile shape, which is why it can stay dynamic here.

The kernel is the four phases every hand-rolled collective shares:

```python
# Phase 1 — stage this rank's slice into its window slot.
local = pl.load(x, [0, 0], [1, SIZE])
data = pl.store(local, [0, 0], data)

# Phase 2 — barrier: notify every peer, wait on every peer slot.
for peer in pl.range(nranks):
    if peer != my_rank:
        pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                          value=1, op=pld.NotifyOp.AtomicAdd)
for src in pl.range(nranks):
    if src != my_rank:
        pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)

# Phase 3 — accumulate: start from our own slice, add every peer's slice.
acc = pl.load(data, [0, 0], [1, SIZE])
for peer in pl.range(nranks):
    if peer != my_rank:
        recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
        acc = pl.add(acc, recv)

# Phase 4 — stage-out: the accumulated result is this rank's output.
y = pl.store(acc, [0, 0], y)
```

- **Phase 2 is the step-04 barrier, verbatim.** Each rank owns a dedicated row
  (`offsets=[my_rank, 0]`); `AtomicAdd`/`Ge(1)` passes only once every peer has
  staged. Without it, Phase 3 could `remote_load` a peer's slice before that
  peer's store lands.
- **Phase 3 is the mesh itself.** Start from your own slice, then `remote_load`
  every other rank's slice and add. Note the symmetry: every rank does this,
  so every rank ends with the same sum.

**Cost card (per rank):** `(P-1) * N` bytes — one full slice per peer, read by
every rank. Round-heavy: `P-1` remote reads plus one barrier. This O(P) traffic
is exactly why the two-phase and ring variants exist.

## Edge cases

> **Fatal pitfall — a missing barrier lets the load race the store.** If you
> drop Phase 2, a rank can read a peer's window slot before that peer's
> `pl.store` has landed, mixing stale/zero data into the sum. The race is
> timing-dependent, so it may pass at P=2 and fail at P=4. **Fix:** the
> notify/wait handshake must complete before any `remote_load`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Sum includes zeros at some ranks | Barrier missing/incorrect; read raced the store | Barrier (notify all / wait all) before Phase 3 |
| Wrong result only at P=4 | P=2 hides the race (single peer) | Run P≥4; check the barrier covers every peer |
| Same result on every rank but ≠ torch sum | Reduction order differs (not a bug) | Compare with a tolerance (the example already does) |
| `InitMemRef requires static shape ... is dynamic` | A runtime-sized dim reached a **tile** shape (e.g. a chunk size derived from the rank count) | Give that kernel a compile-time rank count via a factory, as steps 09-10 do; tile shapes cannot be runtime-sized in either decorator family |
| Golden fails with a huge diff | Slices summed in the wrong place (e.g. own slice counted twice) | Stage once; accumulate from your own slice, then peers |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 08)
- [01-collectives](01-collectives.md) §AllReduce — the reference for mesh mode
- [09-barrier](09-barrier.md) — the notify/wait barrier reused here (step 04)
- [10-remote_load_store](10-remote_load_store.md) — `remote_load` (step 05)
- Next step: [14-allreduce_two_phase](14-allreduce_two_phase.md) — the same result in roughly half the traffic
