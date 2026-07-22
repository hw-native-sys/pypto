# InsertCommFence Pass

## Overview

`InsertCommFence` implements the *data-before-signal* memory-consistency contract
the latest PTOAS enforces in its `pto-memory-consistency` pass and pushes onto the
compiler. The contract is **two-sided**:

- **Publish side.** A `pto.comm.tnotify` requires an explicit
  `pto.fence.barrier_all #pto.fence_scope<gm>` after the matching
  `pto.cmo.cacheinvalid` release marker and before the signal — a cross-rank write
  must be visible to the peer before the notify that signals it.
- **Consume side.** A cacheable GM load after `pto.comm.twait` (or a successful
  `pto.comm.ttest`) requires an explicit `pto.cmo.cacheinvalid all
  #pto.address_space<gm>` first, so the reader sees the peer's fresh write.

Both cache markers are the **same** `system.cacheinvalid` op, in two forms
distinguished by arity:

| Form | IR | Lowers to |
| ---- | -- | --------- |
| Region | `system.cacheinvalid(tensor, shapes, offsets)` | `pto.cmo.cacheinvalid … single_cache_line` |
| Whole-GM | `system.cacheinvalid()` (no args) | `pto.cmo.cacheinvalid all #pto.address_space<gm>` |

`system.fence` lowers to `pto.fence.barrier_all #pto.fence_scope<gm>` — a GM
barrier with DDR observability, stronger than a bare `pto.barrier <PIPE_ALL>`.

## What the pass inserts

A single forward structural traversal (`InsertCommMarkers`) inserts, per op:

- **after each publishing write** — a whole-tensor **region** `system.cacheinvalid`
  of the written target, **immediately followed by a `system.fence`**. ptoas
  requires the fence to directly follow the release marker, so both land at the
  write site (one per address), not deferred to the notify;
- **before each bare barrier notify** — a **whole-GM** `system.cacheinvalid` +
  `system.fence`. A *bare barrier notify* is one with no pending fenced publishing
  write (a pure barrier signal, or a notify whose write lives in a prior loop so
  `pending` was reset). A notify that *does* have a pending fenced write needs
  nothing — the write's own `cacheinvalid; fence` is its release marker;
- **after each wait** — a **whole-GM** `system.cacheinvalid` (the consume-side
  invalidate before the next cacheable read).

```text
remote_store(a); remote_store(b); notify
  -> remote_store(a); cacheinvalid(a); fence; remote_store(b); cacheinvalid(b); fence; notify

for p: (if p != me: notify)                 (bare barrier notify — no write)
  -> for p: (if p != me: cacheinvalid(); fence; notify)

for c: store; for p: notify                 (write and notify in separate loops)
  -> for c: (store; cacheinvalid; fence);
     for p: (cacheinvalid(); fence; notify)

wait; read
  -> wait; cacheinvalid(); read
```

Why the fence sits at the write, not the notify: ptoas matches the fence to the
release marker **lexically and locally**. A fence placed near a notify whose data
was released by a `cacheinvalid` in a *different* loop is rejected — the fence must
immediately follow that `cacheinvalid`. Emitting `cacheinvalid; fence` together at
every write satisfies this for a following notify no matter how far away (even
across loop nests), and a straight-line `store; notify` still lowers to the
canonical `store; cacheinvalid; fence; notify`.

The region cacheinvalid currently covers the **whole target tensor** (region
`[0, …]` offsets over the tensor's full shape), reusing the tensor type's dim
exprs. Narrowing to the precise written sub-region — the write's own
`(shapes, offsets)` are right there at the write site — is a planned follow-up.

### Markers land where the write / wait / notify is (always in scope)

Placing the region cacheinvalid immediately after its write means the target `Var`
is trivially in scope (the write just used it) — whether it is a window parameter,
an alias (`dv = pl.tensor.view(win); remote_store(dv)`), a loop-carried `iter_arg`,
or a value defined inside a branch. There is **no cross-scope tracking and nothing
is ever silently dropped**: every marker lands next to the op that needs it, at
every nesting level. A bare single-statement branch/loop body is wrapped in place
(`body -> { markers; body; markers }`); after the first run the body is a
`SeqStmts`, so the pass stays idempotent.

## Position in the pipeline

```text
... -> MaterializeRuntimeScopes -> ClassifyIterArgCarry -> InsertCommFence   (last)
```

It runs **last** in the Default pipeline, after every statement-reordering pass
(`SkewCrossCorePipeline`, `LowerPipelineLoops`, `CanonicalizeIOOrder`, ...). The
inserted ops have no operands and no dependency edges, so an earlier insertion
could be moved away from its notify/wait; running last keeps them adjacent through
codegen. The passes before it only touch Orchestration functions, so the InCore IR
this pass sees is exactly what codegen lowers.

## What counts as a publishing write

Decided by `op_predicates::IsPublishingWrite`:

| Case | Condition |
| ---- | --------- |
| `pld.tile.remote_store` / `pld.tile.put` / `pld.tensor.put` | remote write (always) |
| `tile.store` | destination tensor (arg 2) is a window-bound `DistributedTensorType` (a peer can `remote_load` it) |
| `pld.tile.get` / `pld.tensor.get` | local destination (arg 0) is window-bound |
| any call to an unregistered op name | conservative — a user function not analysed interprocedurally |
| `Submit` (task launch) | conservative — treated as a write for ordering |

A `remote_load` (result is a tile, no GM write) and a `tile.store` into a plain
`Tensor` are **not** publishing writes. A plain-tensor store therefore emits no
region cacheinvalid, but a notify after it is still a *bare barrier notify* and
gets the whole-GM cacheinvalid + fence.

## Algorithm — one forward traversal with a `pending` bool

A single `pending` bool flows in execution order and decides the notify marker:

- a **publishing write** sets `pending = true` (it emitted its own `cacheinvalid;
  fence`);
- a **notify** or **fence** clears `pending = false`;
- a **wait** / anything else leaves it unchanged.

At each notify: if `pending` is true, a fenced write already precedes it, so
nothing is added; if `pending` is false the notify is a *bare barrier notify* and
gets its own whole-GM `cacheinvalid; fence`.

`pending` is **conservative across control flow** so an uncertain path emits the
safe whole-GM marker rather than trusting a region marker that may not precede:

- **`if`** — each branch is marked with its own incoming `pending`; after the `if`,
  `pending` is the **AND** of the two branch outcomes (a write on only one path is
  not proven). A conditional notify with no proven pending write thus gets its own
  whole-GM marker inside its branch.
- **loops** — the body is entered with `pending = false`, and `pending` is reset to
  false after the loop. This is required for correctness, not just conservatism:
  ptoas checks the release marker **lexically**, so a loop-head notify cannot rely
  on a fence carried in from before the loop (iteration 0) or from the previous
  iteration's tail write (the back-edge) — neither lexically precedes it in the
  body. Clearing `pending` forces every loop-body notify to get its own marker.

```text
remote_store; notify                 -> remote_store; cacheinvalid; fence; notify
remote_store; for: notify             -> remote_store; cacheinvalid; fence;
                                         for: { cacheinvalid(); fence; notify }
for: { notify; store }                -> for: { cacheinvalid(); fence; notify;
                                                store; cacheinvalid; fence }
```

An existing region `cacheinvalid` **immediately followed by a fence** after a
write, and an existing whole-GM cacheinvalid immediately after a wait, are
recognized and **not duplicated**, so the pass is idempotent.

## Codegen interaction

`InsertCommFence` supersedes the former unconditional drain barrier emitted after
every TPUT in `MakePutCodegenPTO` (a PTOAS#872 workaround). That codegen barrier is
removed: the fence now fires only when a notify actually follows, and it adds the
DDR-observability drain a pipe barrier cannot. The TPUT/TGET **pre** barriers and
the TGET **tail** barrier are unrelated (in-core RAW ordering, not
data-before-signal) and are kept.

## Consumers

None downstream in the pipeline. PTO codegen lowers the inserted `system.cacheinvalid`
(region or whole-GM by arity) and `system.fence` via their existing op handlers; no
other pass reasons about them.
