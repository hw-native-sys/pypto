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

Verified empirically on ptoas 0.50, the contract reduces to **two purely-local
rules** — the *notify* itself needs no marker. A single structural traversal
(`InsertCommMarkers`), with no control-flow state, inserts:

- **after each publishing write** — a whole-tensor **region** `system.cacheinvalid`
  of the written target, **immediately followed by a `system.fence`**. ptoas ties
  the release fence to this cacheinvalid; any later `tnotify` that publishes the
  data is satisfied by it, so both markers land at the write site (one per
  address);
- **after each wait** — a **whole-GM** `system.cacheinvalid` (the consume-side
  invalidate before the next cacheable read);
- **notify** — nothing.

```text
remote_store(a); remote_store(b); notify
  -> remote_store(a); cacheinvalid(a); fence; remote_store(b); cacheinvalid(b); fence; notify

for c: store; for p: notify                 (write and notify in separate loops)
  -> for c: (store; cacheinvalid; fence);
     for p: notify

wait; read
  -> wait; cacheinvalid(); read
```

Why the notify needs no marker, and the fence sits at the write: ptoas associates
the required release fence with a publishing write's `cacheinvalid`, not with the
notify. So a `tnotify` that releases data written earlier — even in a *different*
loop — is already satisfied by that write's `cacheinvalid; fence`; the fence does
**not** need to sit next to the notify. A pure barrier notify (no data at all)
requires nothing. (This was verified by removing the notify-side markers from the
ring-allreduce `.pto` and confirming ptoas 0.50 still accepts it; removing the
wait-side `cacheinvalid all`, by contrast, is rejected.)

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
(`body -> { body; markers }`); after the first run the body is a `SeqStmts`, so the
pass stays idempotent.

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
`Tensor` are **not** publishing writes — no marker is emitted for them, and the
notify after them needs none either.

## Algorithm — one structural traversal, no flow state

Both rules are purely local, so the pass carries **no** control-flow state (no
`pending` bool, no `if`/loop analysis, no notify classification):

- at each **publishing write**, append `region cacheinvalid; fence`;
- at each **wait**, append `cacheinvalid()`;
- **notify** is left untouched.

`if`/`for`/`while` bodies are visited normally; the only special handling is
wrapping a bare single-statement body (a write/wait that is the sole body of an
`if`/`for` without an enclosing `SeqStmts`) so its marker still lands. Because both
rules are local and append-only, control flow is irrelevant: a write inside one
loop is correctly marked whether or not its notify lives in another.

```text
remote_store; notify                 -> remote_store; cacheinvalid; fence; notify
remote_store; for: notify             -> remote_store; cacheinvalid; fence; for: notify
for: { notify; store }                -> for: { notify; store; cacheinvalid; fence }
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
