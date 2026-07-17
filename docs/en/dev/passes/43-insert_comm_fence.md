# InsertCommFence Pass

## Overview

`InsertCommFence` implements the *data-before-signal* obligation the latest PTOAS
pushes onto the compiler — a cross-rank write must be visible to the peer before
the `pld.system.notify` that signals it — with two ops at two granularities:

1. `system.cacheinvalid` is **per address**: one is emitted **immediately after
   every publishing write**, invalidating the cache lines of exactly that write's
   region.
2. `system.fence` is **per notify**: a single GM barrier before a notify orders
   *all* prior writes, so one fence covers however many writes precede it.

```text
remote_store(a); remote_store(b); notify
  -> remote_store(a); cacheinvalid(a); remote_store(b); cacheinvalid(b); fence; notify
```

- `system.cacheinvalid` lowers to `pto.cmo.cacheinvalid … single_cache_line`.
- `system.fence` lowers to `pto.fence.barrier_all #pto.fence_scope<gm>` — a GM
  barrier with DDR observability, stronger than a bare `pto.barrier <PIPE_ALL>`.

Because the two share nothing, they are two **independent traversals**
(`CacheInvalidInserter`, then `FenceInserter`) — order is irrelevant (a
cacheinvalid is inert to the fence analysis).

### cacheinvalid is placed at the write site (always in scope)

Placing the cacheinvalid immediately after its write means the target `Var` is
trivially in scope (the write just used it) — whether it is a window parameter, an
alias (`dv = pl.tensor.view(win); remote_store(dv)`), a loop-carried `iter_arg`, or
a value defined inside a branch. There is **no cross-scope tracking and nothing is
ever silently dropped**: the per-write cacheinvalid always lands next to its write,
at every nesting level. (Contrast the earlier design, which batched cacheinvalids
before the notify and had to reason about which targets were still in scope there.)

The cacheinvalid currently covers the **whole target tensor** (region `[0, …]`
offsets over the tensor's full shape), reusing the tensor type's dim exprs.
Narrowing to the precise written sub-region — the write's own `(shapes, offsets)`
args are right there at the write site — is a planned follow-up.

## Position in the pipeline

```text
... -> MaterializeRuntimeScopes -> ClassifyIterArgCarry -> InsertCommFence   (last)
```

It runs **last** in the Default pipeline, after every statement-reordering pass
(`SkewCrossCorePipeline`, `LowerPipelineLoops`, `CanonicalizeIOOrder`, ...). A
`system.fence` has no operands and no dependency edges, so an earlier insertion
could be moved away from its notify; running last keeps the fence adjacent to the
notify through codegen. The passes before it (`MaterializeRuntimeScopes`,
`ClassifyIterArgCarry`) only touch Orchestration functions, so the InCore IR this
pass sees is exactly what codegen lowers.

## What counts as a publishing write

Decided by `op_predicates::IsPublishingWrite`:

| Case | Condition |
| ---- | --------- |
| `pld.tile.remote_store` / `pld.tile.put` / `pld.tensor.put` | remote write (always) |
| `tile.store` | destination tensor (arg 2) is a window-bound `DistributedTensorType` (a peer can `remote_load` it) |
| `pld.tile.get` / `pld.tensor.get` | local destination (arg 0) is window-bound |
| any call to an unregistered op name | conservative — a user function not analysed interprocedurally |

A `remote_load` (result is a tile, no GM write) and a `tile.store` into a plain
`Tensor` are **not** publishing writes.

## Algorithm

Two independent `O(N)` traversals.

### Traversal 1 — `CacheInvalidInserter` (per-address, structural only)

A plain structural rewrite: after each publishing-write child of a `SeqStmts`,
append a whole-tensor `cacheinvalid` for that write's target. A bare
single-statement branch/loop body that *is* a publishing write is wrapped in
place (`body -> { body; cacheinvalid }`). No control-flow analysis, no `pending`
state. Idempotent: a write already followed by its cacheinvalid is left alone.

### Traversal 2 — `FenceInserter` (per-notify, control-flow aware)

**Pass 1 — subtree summaries (bottom-up / post-order, memoized).** Each statement
records three pure structural bits:

- `opens_with_notify` — some path reaches a notify before any write/fence;
- `may_end_with_write` — the subtree may exit with an uncovered publishing write;
- `transparent` — some path falls through touching no write/fence/notify.

Loops are always `transparent` (they may iterate zero times); `opens` / `may_end`
come from the body.

**Pass 2 — forward insertion.** A single `pending` bool flows in execution order.
At each `SeqStmts`, before a child that `opens_with_notify`, a fence is emitted if
`pending` (then `pending` clears) — **except before an `if`**, which is recursed
into so each branch fences at its own real notify:

```text
remote_store; cacheinvalid; notify         -> ...; fence; notify           (straight line)
remote_store; cacheinvalid; for p: notify  -> ...; fence; for p: notify     (hoist before loop)
remote_store; cacheinvalid; if c: notify   -> ...; if c: { fence; notify }  (inside the branch)
```

The `if` differs from the loop on purpose. A notify inside an `if` is
*conditional*, so pushing the barrier into the taken branch is strictly more
precise than hoisting an unconditional barrier before the `if` (no barrier runs
when the branch is not taken). A loop whose body does **not** end with a write
(e.g. a `for p: notify` barrier) *does* hoist, so one fence before the loop covers
all iterations instead of one per iteration.

A loop whose body **may end with a write** is entered with
`pending || may_end_with_write(body)`, so a write at the tail of one iteration
fences the notify at the head of the next — the ring-allreduce back-edge (the tail
write's own cacheinvalid was already placed by traversal 1). Such a loop already
fences at its own head, so an incoming pending write is **not** hoisted before it:
the head fence on iteration 0 covers that write too (a hoisted fence would be a
redundant second barrier before the same notify). Concretely
`store; for { notify; store }` emits a single fence, at the loop head — not one
before the loop plus one inside.

```text
for s: { for p: notify(...); ...; store(win); cacheinvalid(win) }
     -> for s: { fence; for p: notify(...); ...; store(win); cacheinvalid(win) }
```

An existing `system.fence` clears `pending`, so the pass is **idempotent** and
takes a user-written fence as the complete barrier (no duplicate fence added).

## Codegen interaction

`InsertCommFence` supersedes the former unconditional drain barrier emitted after
every TPUT in `MakePutCodegenPTO` (a PTOAS#872 workaround). That codegen barrier
is removed: the fence now fires only when a notify actually follows, and it adds
the DDR-observability drain a pipe barrier cannot. The TPUT/TGET **pre** barriers
and the TGET **tail** barrier are unrelated (in-core RAW ordering, not
data-before-signal) and are kept.

## Consumers

None downstream in the pipeline. PTO codegen lowers the inserted `system.cacheinvalid`
and `system.fence` via their existing op handlers; no other pass reasons about them.
