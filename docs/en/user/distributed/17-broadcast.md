# Broadcast: One-to-All

Root rank's slice reaches every rank — weights, configs, a global row — then
the builtin does it in one call.

> **Prerequisites:** [16-allreduce_reveal](16-allreduce_reveal.md). Any number
> of devices ≥ 2 (the examples use 2 and 4 sim devices).

**Suggested reading order:** 01 → … → 11 → **12** — this page is step 12.

## The idea

Broadcast is the simplest collective with a distinguished rank: the **root**
owns the data and every other rank ends with a copy of it. This is the
"load the weights once, share them" pattern — the data is the same on every
rank afterwards, so there is nothing to reduce.

| Aspect | Broadcast |
| ------ | --------- |
| Data in | Root's slice only (other ranks' input is ignored) |
| Data out | Root's slice on **every** rank |
| Pattern | Root stages → barrier → every rank reads root |
| Cost | Root writes `N` bytes, every peer reads them: `(P-1)·N` total, one step |

You already have every primitive this needs — step 04's barrier and step 05's
`remote_load`. The only new idea is that the *read target* is fixed: everyone
reads rank 0.

## Run it

```bash
# Hand-rolled: root stages, barrier, every rank remote_loads root.
python examples/distributed/12_broadcast.py -p a2a3sim -d 0,1

# Reveal: pld.tensor.broadcast in one call.
python examples/distributed/12_broadcast.py -p a2a3sim -d 0,1 --mode builtin

# The same source at P=4 (comparisons need >2 ranks):
python examples/distributed/12_broadcast.py -p a2a3sim -d 0,1,2,3
python examples/distributed/12_broadcast.py -p a2a3sim -d 0,1,2,3 --mode builtin
```

Expected output:

```text
OK
```

The golden checks the contract precisely: every rank's output equals root's
slice, **and** no non-root rank's input leaked into any output.

## Walkthrough

The hand-rolled kernel is the three-phase pattern from the substrate — nothing
new, just a fixed `peer`:

```python
@pl.function(type=pl.FunctionType.InCore)
def hand_step(self, x, y, data, signal, root):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)

    # Phase 1 — stage-in: root only writes its slice into the window.
    if my_rank == root:
        local = pl.load(x, [0, 0], [1, SIZE])
        data = pl.store(local, [0, 0], data)

    # Phase 2 — barrier: notify every peer, wait on every peer slot.
    for peer in pl.range(nr):
        if peer != my_rank:
            pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    for src in pl.range(nr):
        if src != my_rank:
            pld.system.wait(signal, offsets=[src, 0], expected=1,
                            cmp=pld.WaitCmp.Ge)

    # Phase 3 — broadcast: pull root's slice into local output.
    recv = pld.tile.remote_load(data, peer=root, offsets=[0, 0], shape=[1, SIZE])
    return pl.store(recv, [0, 0], y)
```

- **The stage is conditional.** Only `my_rank == root` writes the window —
  the `if` on the runtime rank scalar is a normal control-flow branch. Other
  ranks leave the window untouched and wait.
- **The barrier is the same one from step 04** — dedicated-row
  `AtomicAdd`/`Ge(1)`. It exists so no rank `remote_load`s root's slice before
  root staged it.
- **The read is `remote_load(data, peer=root, ...)`** — step 05's primitive
  with the peer fixed to the root. Nothing about broadcast is a new
  primitive; it is a *usage* of the ones you built.

The reveal replaces phases 2–3 with one call:

```python
    if my_rank == ROOT_RANK:
        local = pl.load(x, [0, 0], [1, SIZE])
        data = pl.store(local, [0, 0], data)

    data = pld.tensor.broadcast(data, signal, root=ROOT_RANK)
    acc = pl.load(data, [0, 0], [1, SIZE])
    return pl.store(acc, [0, 0], y)
```

- **`root=` must be a compile-time constant** (a Python `int`, here
  `ROOT_RANK = 0`), not a runtime scalar — the lowering needs to know the root
  statically. That is also why the hand-rolled kernel takes `root` as a scalar
  but the builtin does not.
- **Non-root slots are ignored on input** — you may leave them uninitialised;
  only the root's slot is read after the call. The golden asserts this: non-root
  inputs never leak.

### The IR diff (the teaching artifact)

Compile with pass dumps enabled and diff the two modes' lowered IR:

- `--mode hand` lowers to exactly the three phases above — one `remote_load`
  from the root, guarded by the notify/wait barrier.
- `--mode builtin` lowers to the same shape: the composite expands into the
  ready barrier on the `[nr, 1]` signal plus the root read. The diff is
  "your schedule, expressed by the compiler" — nothing extra, no hidden
  traffic.

**Cost card (per rank):** root writes `N` bytes; every peer reads them —
`(P-1)·N` total bytes in one step. This is the cheapest collective per byte
because every rank wants the *same* bytes.

## Edge cases

> **Fatal pitfall — non-root data leaking into the output.** If the kernel
> broadcasts from `my_rank`'s own slice instead of the root's, every rank ends
> with *its own* data and the golden fails silently (each rank is internally
> consistent). **Fix:** the read target is always the root:
> `remote_load(data, peer=ROOT_RANK, ...)`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Every rank outputs its own slice | Broadcast read from `my_rank`, not the root | Read `peer=ROOT_RANK` |
| Non-root inputs appear in some output | Root staging missing → stale window | Stage only under `if my_rank == root` |
| `root` kwarg rejected at compile | Runtime scalar passed to `pld.tensor.broadcast` | Pass a Python `int` constant |
| Remote read returns zeros | `remote_load` before root staged / no barrier | Load mode: stage (root) → barrier → load |
| Only one rank's row correct at P=4 | Hand-rolled read raced the barrier | Confirm the notify/wait loop covers all `nr` peers |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 12)
- [01-collectives](../distributed/01-collectives.md) §Broadcast — the full API
- [02-primitives](../distributed/02-primitives.md) §Tile-Level RMA — the
  `remote_load` the hand-rolled version builds on
- Next step: [18-allgather](18-allgather.md) — all-to-all slices
