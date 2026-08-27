# The Reveal: `pld.tensor.allreduce` in One Call

You built all-reduce three ways by hand — mesh, two-phase, ring. Now the
builtin: `pld.tensor.allreduce(data, signal, op=..., mode=...)` replaces the
whole schedule with one call, and the IR diff shows what it lowers to: the
hand-rolled pattern you wrote, or a better one.

> **Prerequisites:** [13-allreduce_mesh](13-allreduce_mesh.md) ·
> [14-allreduce_two_phase](14-allreduce_two_phase.md) ·
> [15-allreduce_ring](15-allreduce_ring.md). Four sim devices recommended.

**Suggested reading order:** 01 → … → 10 → **11** — this page is step 11.

## The idea

The reveal discipline, completed: step 04 revealed the barrier after you built
it; this step reveals the collective after you built three of them. The point
of the hand-rolled steps was not that you should write all-reduce by hand —
it was that you should know *what the builtin chooses between*.

`pld.tensor.allreduce` is the builtin for the whole ladder: it takes your
window and signal, does the barrier + cross-rank reduce + store-back, and
hands the reduced slice back. `mode=` picks the algorithm — `"mesh"` (default)
or `"ring"`. The same golden as steps 08-10; none of the schedule.

## Run it

```bash
# Both modes, at P=4 (and P=2):
python examples/distributed/11_allreduce_reveal.py -p a2a3sim -d 0,1,2,3
python examples/distributed/11_allreduce_reveal.py -p a2a3sim -d 0,1,2,3 --mode ring
python examples/distributed/11_allreduce_reveal.py -p a2a3sim -d 0,1
```

Expected output:

```text
OK
```

## Walkthrough

Your stage-in and stage-out stay; only the middle is replaced:

```python
# Phase 1 — stage this rank's slice into its window slot.
local = pl.load(x, [0, 0], [1, SIZE])
data = pl.store(local, [0, 0], data)

# Phase 2 — the builtin: barrier + reduce + store-back, in one call.
data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode=mode)

# Phase 3 — stage-out: the reduced slice back to the local output.
recv = pl.load(data, [0, 0], [1, SIZE])
y = pl.store(recv, [0, 0], y)
```

- **The signal is yours, and its shape tells you which mode you're in.** Mesh
  takes `[nr, 1]`; ring takes `[2*(nr-1), nr]` — exactly the row-per-round
  signal step 10 taught. The factory folds both `nr` and `mode` in, so one
  source builds either variant (the class-form pattern from steps 08-10).
- **Why a factory here is not the steps 09/10 story.** Those two need a
  compile-time `nr` because their chunk size `SIZE // nr` is a **tile shape**.
  Here the builtin owns the chunking, so nothing is a tile shape sized by the
  rank count, and `nr` could stay dynamic. What has to be fixed when the kernel
  is traced is `mode`: it picks the lowering *and* the signal layout, and mesh
  and ring are two different shapes rather than two extents of one. `nr` is
  folded in beside it so a single source can spell both layouts.
- **What the builtin accepts (read this twice):** `pld.tensor.allreduce` — the
  **InCore composite** lowering used here — takes the full `ReduceOp` family
  (`Sum`/`Max`/`Min`/`Prod`) and `FP16`/`FP32` in **both** modes — the mesh and
  ring ST suites run every operator and `FP16` through the real pipelines.
  A narrower `Sum`+`FP32`-only contract exists, but only on the separate
  **HOST builtin** ring path (`builtin.tensor.allreduce_ring`, not used by
  this tutorial) — see `01-collectives.md` §AllReduce.

### The IR diff (the teaching artifact)

Compile with pass dumps enabled and diff the lowered IR of this step against
your hand-rolled programs:

- `--mode mesh` expands into **step 08's pattern**: a ready barrier on the
  `[nr, 1]` signal, then `remote_load` + accumulate chunks — the mesh you
  wrote, possibly chunked to UB-sized tiles.
- `--mode ring` expands into **step 10's shape**: `2*(nr-1)` rounds on the
  `[2*(nr-1), nr]` signal, chunked `N/P` transfers — but the builtin lowers
  each round to a **full-mesh barrier** (`EmitNotifyAll`/`EmitWaitAll`), not
  the neighbour-ready handshake you wrote in step 10. That difference is the
  point of the diff: same schedule and chunks, different synchronization.

Same golden, same signal conventions, same four-phase shape — the diff is
"your schedule, expressed by the compiler", which is the whole teaching point.

**Cost card (per rank):** whatever mode you picked — `(P-1) * N` (mesh) or
`2 * (P-1) / P * N` in `N/P` steps (ring). The builtin chooses the algorithm;
you still choose the mode.

## Edge cases

> **Fatal pitfall — the wrong signal shape for the mode.** `mode="ring"`
> strictly validates the `[2*(nr-1), nr]` signal. Mesh validates that a static
> column count is 1 — so a static ring-shaped signal is rejected — but it does
> not validate the row count; it only reads the slots it indexes. **Fix:** keep
> mesh → `[nr, 1]`, ring → `[2*(nr-1), nr]`, and don't share one signal window
> across modes.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Compile error about the signal shape | Ring-mode signal not `[2*(nr-1), nr]`, or a static non-`[·,1]` mesh signal | Mesh `[nr, 1]`, ring `[2*(nr-1), nr]` — one window per mode |
| Builtin ring shows more sync cost than your hand-rolled ring | The builtin emits a full-mesh barrier per round (O(P²) per rank); your handshake is neighbour-local (O(P) per rank) | Hand-roll the ring (step 10) when per-round sync cost matters |
| Same result on every rank but ≠ torch sum | Reduction order differs (not a bug) | Compare with a tolerance |
| Result is your own slice, unreduced | The call result wasn't rebound (`data =`) | `data = pld.tensor.allreduce(data, signal, ...)` — in-place rebind |
| Builtin slower than your hand-rolled mesh | Tiny payload: mesh is for small messages | Use ring for payloads ≳ 16 KiB (see `01-collectives.md`) |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 11)
- [01-collectives](01-collectives.md) §AllReduce — the reference for both modes and the signal shapes
- [13-allreduce_mesh](13-allreduce_mesh.md) / [15-allreduce_ring](15-allreduce_ring.md) — what each mode lowers to
- [02-primitives](02-primitives.md) — the substrate the builtin is built from
- Next: steps 12-16 in [05-tutorials](05-tutorials.md) cover the other collectives
