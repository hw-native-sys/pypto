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

- **after each local publishing write** — a window-bound `tile.store`, or a `get`
  into a local destination: a whole-tensor **region** `system.cacheinvalid` of the
  written target, **immediately followed by a `system.fence`**;
- **after each remote publishing write** — `remote_store` / `put`: only a
  `system.fence` (see the note below on where its cacheinvalid comes from);
- **after each opaque publishing write** — a `Submit`, or a call to an
  unregistered user function whose body is not analysed here (no single
  addressable region): a conservative **whole-GM** `system.cacheinvalid()` +
  `system.fence`;
- **after each wait** — a **whole-GM** `system.cacheinvalid` (the consume-side
  invalidate before the next cacheable read);
- **notify** — nothing.

A region `system.cacheinvalid(target)` addresses `target`'s **local** base, which
is correct for a local-window store. The **remote** writes `remote_store` / `put`
write to a **peer-offset** GM address (`local_ptr + delems(peer)`) that the local
target view does not address — a local-target cacheinvalid would clean the wrong
line. The peer offset is only known during codegen (`EmitCommRemoteView`) and is
**not yet expressible in the IR**, so the peer-region `pto.cmo.cacheinvalid
<peer_view> single_cache_line` is emitted by the op's codegen as a **workaround**.
The GM release **fence**, however, is always an explicit `system.fence` op inserted
by this pass — the codegen must not embed it, keeping the release ordering
IR-visible and uniform. (Giving the peer-region cacheinvalid a first-class IR
representation, so the pass can own the whole marker, is a follow-up.)

```text
store(win_a); store(win_b); notify        (local window stores)
  -> store(win_a); cacheinvalid(win_a); fence; store(win_b); cacheinvalid(win_b); fence; notify

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

## Which writes the pass marks

| Case | cacheinvalid | fence |
| ---- | ------------ | ----- |
| `tile.store` into a window-bound `DistributedTensorType` (a peer can `remote_load` it) | pass (local region, dst arg 2) | pass |
| `pld.tile.get` / `pld.tensor.get` (peer read into a local destination) | pass (local region, dst arg 0) | pass |
| `pld.tile.remote_store` / `pld.tile.put` / `pld.tensor.put` (peer-offset write) | **codegen** (peer region — IR workaround) | **pass** |

A `remote_load` (result is a tile, no GM write) and a `tile.store` into a plain
`Tensor` are **not** publishing writes — no marker at all.

## Algorithm — one structural traversal, no flow state

The pass carries **no** control-flow state (no `pending` bool, no `if`/loop
analysis, no notify classification):

- at each **local publishing write**, append `region cacheinvalid; fence`;
- at each **remote publishing write** (`remote_store` / `put`), append `fence`
  only (the peer-region cacheinvalid is codegen's — see below);
- at each **wait**, append `cacheinvalid()`;
- **notify** is left untouched.

`if`/`for`/`while` bodies are visited normally; the only special handling is
wrapping a bare single-statement body (a write/wait that is the sole body of an
`if`/`for` without an enclosing `SeqStmts`) so its marker still lands. Because the
rules are local and append-only, control flow is irrelevant: a write inside one
loop is correctly marked whether or not its notify lives in another.

```text
store(win); notify                   -> store(win); cacheinvalid(win); fence; notify
store(win); for: notify               -> store(win); cacheinvalid(win); fence; for: notify
for: { notify; store(win) }           -> for: { notify; store(win); cacheinvalid(win); fence }
```

An existing region `cacheinvalid` **immediately followed by a fence** after a
write, and an existing whole-GM cacheinvalid immediately after a wait, are
recognized and **not duplicated**, so the pass is idempotent.

## Codegen interaction

`remote_store` and `put` codegen each emit their peer-region
`pto.cmo.cacheinvalid <peer_view>` right after the store (the peer offset is only
known there and is not yet expressible in the IR — a **workaround**; the pass
supplies the paired GM fence as a `system.fence` op, so it does *not* emit the
fence here). `put` additionally keeps a tail `pto.barrier <PIPE_ALL>` **between**
the TPUT and its cacheinvalid: TPUT is a DMA, and the GM fence orders memory but
does not drain the MTE pipe that issued the DMA, so without this barrier the
following notify can fire before the (possibly atomic) TPUT has landed at the peer
— `test_l3_put` atomic-add / subregion flake on device without it, and
device-testing shows an MTE3-scoped barrier is not enough (only `PIPE_ALL` is
stable). A PTOAS#872 workaround; remove once PTOAS drains the tput itself. The
TPUT/TGET **pre** barriers and the TGET **tail** barrier are likewise kept.

**Follow-up:** the peer-region cacheinvalid for remote writes is the one marker
still emitted in codegen rather than as an IR op, because the peer address
(`local_ptr + delems(peer)`) has no IR representation yet. A first-class IR form
would let InsertCommFence own the complete remote-write marker (cacheinvalid +
fence), matching the local-write path.

## Consumers

None downstream in the pipeline. PTO codegen lowers the inserted `system.cacheinvalid`
(region or whole-GM by arity) and `system.fence` via their existing op handlers; no
other pass reasons about them.
