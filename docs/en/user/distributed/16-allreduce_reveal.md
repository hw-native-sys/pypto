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
- **What the builtin narrows (read this twice):** the InCore composite this
  example calls (explicit signal) accepts the full `ReduceOp` family
  (`Sum`/`Max`/`Min`/`Prod`) and `FP16`/`FP32` in **both** modes — the
  `Sum`+`FP32`-only restriction belongs to the **HOST builtin** ring path, not
  to this InCore form. The ring intrinsic ST suite exercises `Max`/`Min`/`Prod`
  and `FP16`.

### The IR diff (the teaching artifact)

Compile with pass dumps enabled and diff the lowered IR of this step against
your hand-rolled programs:

- `--mode mesh` expands into **step 08's pattern**: a ready barrier on the
  `[nr, 1]` signal, then `remote_load` + accumulate chunks — the mesh you
  wrote, possibly chunked to UB-sized tiles.
- `--mode ring` expands into **step 10's pattern**: `2*(nr-1)` rounds of
  neighbour-ready handshakes on the `[2*(nr-1), nr]` signal, chunked `N/P`
  transfers — the ring you wrote, with the push/read side chosen by the
  lowering.

Same golden, same signal conventions, same four-phase shape — the diff is
"your schedule, expressed by the compiler", which is the whole teaching point.

**Cost card (per rank):** whatever mode you picked — `(P-1) * N` (mesh) or
`2 * (P-1) / P * N` in `N/P` steps (ring). The builtin chooses the algorithm;
you still choose the mode.

## Edge cases

> **Fatal pitfall — the wrong signal shape for the mode.** Pass a `[nr, 1]`
> signal with `mode="ring"` (or the ring shape with mesh) and the lowering
> rejects it — the signal *is* the contract for which schedule the builtin
> runs. **Fix:** mesh → `[nr, 1]`; ring → `[2*(nr-1), nr]`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Compile error about the signal shape | Signal shape does not match the mode | Mesh `[nr, 1]`, ring `[2*(nr-1), nr]` |
| `mode="ring"` + `Max`/`Prod`/`FP16` rejected on the HOST path | The `Sum`+`FP32` limit is host-builtin-only | Use this explicit-signal InCore form, or mesh |
| Same result on every rank but ≠ torch sum | Reduction order differs (not a bug) | Compare with a tolerance |
| Result is your own slice, unreduced | The call result wasn't rebound (`data =`) | `data = pld.tensor.allreduce(data, signal, ...)` — in-place rebind |
| Builtin slower than your hand-rolled mesh | Tiny payload: mesh is for small messages | Use ring for payloads ≳ 16 KiB (see `01-collectives.md`) |

## See also

- [05-tutorials](05-tutorials.md) — the tutorial index (this step = row 11)
- [01-collectives](01-collectives.md) §AllReduce — the reference for both modes and the signal shapes
- [13-allreduce_mesh](13-allreduce_mesh.md) / [15-allreduce_ring](15-allreduce_ring.md) — what each mode lowers to
- [02-primitives](02-primitives.md) — the substrate the builtin is built from
- Next: steps 12-16 in [05-tutorials](05-tutorials.md) cover the other collectives
