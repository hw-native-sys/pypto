# Remote Load / Store: Tile-Level RMA

Move one slice between ranks with `pld.tile.remote_load` and
`pld.tile.remote_store` — the two sides of a one-step ring shift.

> **Prerequisites:** [09-barrier](09-barrier.md). Two devices.

**Suggested reading order:** 01 → 02 → 03 → 04 → **05** → 06 — this page is step 05.

## The idea

A window is symmetric: you can reach *any* rank's slice of it, not just your
own. **Tile-level RMA** exposes that directly. `pld.tile.remote_load(...)`
pulls a peer's slice into a local tile; `pld.tile.remote_store(...)` pushes a
local tile into a peer's slice. These are the same operations the hand-rolled
all-reduce in the chapter reference is built from.

The example is a **one-step ring shift** behind a barrier: every rank shifts
its data one position around the ring. `--mode load` shows the *pull* side
(each rank remote-loads the next rank's slice → `y[r] = x[(r+1) % N]`);
`--mode store` shows the *push* side (each rank remote-stores into the next
rank's slice, then reads its own back → `y[r] = x[(r-1) % N]`).

**Cost card:** one step moves one slice of `N` bytes to (or from) one peer per
rank, in one communication round, a single remote read or write.
Latency-bound: the cost is the round trip, not the bytes.

## Run it

```bash
# Pull side (default):
python examples/distributed/05_remote_load_store.py -p a2a3sim -d 0,1

# Push side:
python examples/distributed/05_remote_load_store.py -p a2a3sim -d 0,1 --mode store
```

Expected output:

```text
OK
```

## Walkthrough

```python
@pl.jit.incore
def shift_by_load(x, y, data, signal):
    ctx = pld.get_comm_ctx(data)
    my_rank = pld.rank(ctx)
    nranks = pld.nranks(ctx)

    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)

    signal = pld.tensor.barrier(signal)

    peer = (my_rank + 1) % nranks
    recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
    y = pl.store(recv, [0, 0], y)
    return y
```

- **Stage in.** Every rank copies its local `x` slice into its own window
  slice with an ordinary `pl.load`/`pl.store` — RMA reads *window* memory, so
  the data must be in the window first.
- **Barrier.** The barrier from step 04 (here as the revealed builtin) orders
  the exchange: no rank remote-loads before every rank has staged.
- **The remote load.** `pld.tile.remote_load(data, peer=peer, offsets=[0, 0],
  shape=[1, SIZE])` pulls peer's window slice into a local tile, exactly like a
  local load — but from a peer's memory. `peer = (my_rank + 1) % nranks` is
  plain `INT32` scalar arithmetic, which is legal on the AI core (unlike FP32
  scalar arithmetic — step 01).

The store side swaps the move for a push:

```python
    local = pl.load(x, [0, 0], [1, SIZE])
    peer = (my_rank + 1) % nranks
    pld.tile.remote_store(local, data, peer=peer, offsets=[0, 0])

    signal = pld.tensor.barrier(signal)

    back = pl.load(data, [0, 0], [1, SIZE])   # rank (r-1) just wrote OUR slice
    y = pl.store(back, [0, 0], y)
```

`remote_store` takes the local tile, the window, and the peer — and pushes.
After the barrier, each rank reads *its own* window slice, which the previous
rank just wrote. The same barrier that orders loads also orders stores.

**`DistributedTensor` vs `Tensor`.** Only the window-bound
`pld.DistributedTensor` is visible to other ranks. `x` and `y` are plain
`pl.Tensor` — local inputs/outputs that no peer can reach. The rule is
structural: anything you want to share must flow through a
`pld.DistributedTensor` view of a window buffer.

## Pushing a computed value from a tensor-level kernel

The example above is a `@pl.jit.incore` (tile-level) kernel, so `pl.load`
produces a `Tile` and `pld.tile.remote_store` takes it directly. In a
tensor-level `@pl.jit` kernel there are no tiles to name — every value is a
`pl.Tensor` — so the push is spelled `pld.tensor.remote_store`:

```python
@pl.jit
def push_scaled(x, win, peer):
    with pl.at(level=pl.Level.CORE_GROUP):
        scaled = pl.mul(x[0:ROWS, 0:COLS], 2.0)
        pld.tensor.remote_store(scaled, win, peer, [0, 0])
        # ...then pld.system.notify() to release it, as always.
```

Both spellings compile to the same single remote write. The value goes
**straight from on-core memory to the peer** — you do not have to store it back
to global memory and push from there, which would cost a round trip and leave
the store and the transfer on different pipes with nothing ordering them.

If you don't want to think about which one you're in, the short form
`pld.remote_store(src, target, peer, offsets)` picks the right one from the
operand you hand it.

Pass `atomic=pld.AtomicType.Add` (either form) to make the push a **combine** —
`peer_region += src` — instead of an overwrite. That is what an all-to-all
combine wants: every rank's contribution accumulates in place. It needs an
fp32/bf16/fp16/int32/int16/int8 dtype, the same set `pl.store` accepts.

Use [`pld.tensor.put`](11-put_get.md) instead when you are moving a **bulk**
global-memory region: `put` streams through a staging tile and has the
`chunk_rows` / `chunk_cols` / `pipeline` knobs, so it is not limited to what
fits on-core. `remote_store` moves what you already have on-core, in one write.

## Edge cases

> **Fatal pitfall — RMA before the ordering barrier.** `remote_load` reads
> *window* memory, so the peer must have staged its slice first: **load** mode
> = stage, then barrier, then `remote_load`. The **store** ordering is the
> mirror: `remote_store` first, *then* barrier, so no receiver reads your
> window slice before the write lands. A barrier before a store does not help —
> the store still races the receiver's read.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Remote read returns zeros | `remote_load` before the peer staged / no barrier | Load mode: stage, barrier, then `remote_load` |
| Shift goes the wrong direction | Pull vs push semantics confused | `load` mode: read `(r+1)`; `store` mode: write to `(r+1)` and read self |
| Scalar `FP32` arithmetic error in `peer` math | Scalar float ops on the AI core | Keep index math in `INT32` (`(r+1) % n`), cast only for data ops |
| Type mismatch on the window param | `DistributedTensor` used as `Tensor` | Annotate shared buffers as `pld.DistributedTensor[...]` |
| One rank reads the previous rank's stale data | Barrier skipped or placed wrong | Load mode: barrier between stage and load; store mode: barrier after the store, before reading |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 05)
- [02-primitives](../distributed/02-primitives.md) §Tile-Level RMA — the full
  `pld.tile.*` surface
- [01-collectives](../distributed/01-collectives.md) — all-reduce is these
  moves plus an add (steps 08–10)
- Next step: [11-put_get](11-put_get.md) — tensor-level push/pull
