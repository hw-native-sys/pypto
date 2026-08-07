# Window Buffer: The Memory Substrate

Allocate a symmetric window buffer, view it as a `DistributedTensor`, and
read/write your own slice — no communication yet, but every later step moves
data through exactly this object.

> **Prerequisites:** [07-programming_model](07-programming_model.md). Two
> devices.

**Suggested reading order:** 01 → 02 → **03** → 04 → 05 → 06 — this page is step 03.

## The idea

Distributed memory in `pld` is **symmetric**: every rank allocates the *same*
window buffer at the same virtual address, so "the buffer" is one object that
every rank can reach — its own slice locally, peers' slices through RMA. The
window buffer is an HCCL buffer with a **signal tail** that the runtime
reserves for cross-rank signaling (steps 04–06 use it).

Two calls create it. `pld.alloc_window_buffer(...)` allocates the buffer; a
`pld.window(...)` call gives you a `pld.DistributedTensor` view of it — the
type that is visible to peers. This step does the simplest possible thing with
a window: load your own slice, store it back to your own slice, read it again.
Nothing is shared yet, and `y == x` is the golden — but the object every future
step communicates through is now on the table.

## Run it

```bash
python examples/distributed/03_window_buffer.py -p a2a3sim -d 0,1
```

Expected output:

```text
OK
```

## Walkthrough

```python
SIZE = 256          # 1 KiB per rank -- below the 4 KiB window floor

@pl.jit.incore
def window_roundtrip(
    x: pl.Tensor[[1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
):
    local = pl.load(x, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)      # write own slice
    back = pl.load(data, [0, 0], [1, SIZE])   # read own slice back
    y = pl.store(back, [0, 0], y)
    return y
```

The kernel's third parameter is a `pld.DistributedTensor` — the window-bound
type. `pl.store`/`pl.load` against it read and write this rank's own slice of
the symmetric window, exactly like a local tensor. Nothing about the kernel
changes the fact that it is distributed: the *type* is what tells the compiler
this buffer lives in shared window memory.

```python
@pl.jit.host
def window_program(
    x: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        per_rank(x[r], y[r], data, device=r)
```

The **host orchestrator owns the window**. `alloc_window_buffer` runs once,
before the dispatch loop — every rank's runtime allocates the same buffer at
the same address. Inside the loop, `pld.window(...)` produces this rank's
`DistributedTensor` view, which is passed down to the kernel.

**The 4 KiB floor.** The window buffer is padded up to at least 4 KiB
regardless of your data size. Here `[1, SIZE]` of `FP32` is `SIZE * 4 = 1 KiB`
per rank, yet the buffer costs 4 KiB — the extra space is the signal tail plus
alignment. This is the first budget constraint of distributed programming: tiny
windows do not cost what their shape suggests.

## Edge cases

> **Fatal pitfall — treating the window as a plain tensor.** `pl.Tensor` and
> `pld.DistributedTensor` are different types. Passing a `DistributedTensor`
> where a local tensor is expected (or vice versa) fails at compile time, not
> run time. **Fix:** use `pld.DistributedTensor[...]` for anything allocated
> with `alloc_window_buffer`, and reserve `pl.Tensor` for local inputs/outputs.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| `alloc_window_buffer` used with a local `Tensor` type | Window view and local tensor types confused | Annotate window params as `pld.DistributedTensor[...]` |
| Per-rank data is 1 KiB but the buffer is bigger | The 4 KiB floor (signal tail + alignment) | Budget 4 KiB minimum per window, not `size * dtype` |
| Window allocated inside the rank loop | Re-allocating per dispatch instead of once | Hoist `alloc_window_buffer` above the loop; call `window(...)` inside |
| Reading a peer's slice as a local load | Forgetting a window is shared | Local loads see only your slice; peers need RMA (steps 05–06) |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 03)
- [02-primitives](../distributed/02-primitives.md) §Window Buffer Management —
  the full API
- [00-model](../distributed/00-model.md) §Glossary — window buffer, signal
- Next step: [09-barrier](09-barrier.md) — signals only, no data
