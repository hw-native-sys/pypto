# Scopes and Placement

Where work runs: marking a region as device work, grouping co-scheduled cores, and
spreading a kernel across blocks.

> **Prerequisites:** [Functions and Programs](01-functions.md) and
> [Programming Model § execution model](../03-programming-model.md#the-execution-model).

## Concept

Placement answers one question: **which piece of hardware runs this code.**

There are four constructs, all written with `with` (or `for`), and they compose:

| Construct | Places work on |
| --------- | -------------- |
| `pl.at` | A core group — marks a region as device work |
| `pl.cluster` | One physical cluster — co-schedules a Cube and a Vector kernel |
| `pl.spmd` | `n` blocks — the same kernel, once per block |
| `pl.split_aiv` | Two AIV lanes — splits one region across both |

The alternative to `pl.at` is writing a separate `@pl.jit.incore` function and calling it.
They produce the same thing: `pl.at` is outlined into exactly such a function during
compilation. Use the scope when the region is short and belongs where it is written; use a
separate function when it deserves a name or is called from more than one place.

Placement is not the same as **ordering** — what must finish before a task starts. The
runtime derives that from the parameter directions in [Types](00-types.md) and the buffers
each task touches. That machinery, and the interfaces for steering it by hand, are
[Tasks and Ordering](../tasks/index.md); this page is only about where code lands.

## Quickstart: mark a region as device work

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
X = torch.randn(256, 128, dtype=torch.float32)
Y = torch.randn(256, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def scale(
    x: pl.Tensor[[256, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.mul(x, 2.0)
    return out


out = torch.zeros(256, 128, dtype=torch.float32)
scale(X, out, config=CFG)
torch.testing.assert_close(out, X * 2.0, rtol=1e-4, atol=1e-4)
```

| Element | What it does |
| ------- | ------------ |
| `@pl.jit` | The entry point — control plane, which cannot hold operators itself |
| `with pl.at(level=pl.Level.CORE_GROUP)` | Marks the region as device work, giving the operators somewhere legal to live |
| `pl.mul(x, 2.0)` | Runs on the core group |

Without the `pl.at`, this kernel fails to compile with `Misplaced tensor op ... should be
inside InCore block` — see [Functions and Programs](01-functions.md).

## Mechanics

### `pl.at`

`level=` picks the hierarchy level. `pl.Level.CORE_GROUP` is the one that produces an
InCore scope, and the region becomes its own kernel function during compilation.

Two optional keywords shape that outlined kernel:

| Keyword | Meaning |
| ------- | ------- |
| `optimizations=[pl.split(mode)]` | Cross-core split mode for the outlined kernel |
| `optimizations=[pl.cross_core_slot(slot_num=N)]` | Ring depth of the automatic cross-core pipeline |
| `name_hint="..."` | Name for the outlined function |

Entries in `optimizations=` must be written inline at the call site — the parser reads the
AST, so a list built up in a variable is not accepted. `pl.split` and `pl.cross_core_slot`
are orthogonal and combine freely: one splits the work, the other sizes the channel.

```python
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                          pl.cross_core_slot(slot_num=4)]):
    ...
```

Omitting `cross_core_slot` keeps the default ring depth of 2 slots per active direction —
enough to double-buffer the handoff while leaving on-chip room for the tiles themselves.
Raise it when the producing core should be able to run further ahead.

### SPMD

`pl.spmd(n)` runs the same kernel on `n` blocks. Two forms, differing in whether the body
reads the block index:

```python
# Dispatch form — the body launches a kernel defined elsewhere. `self.kernel`
# means this form needs @pl.program; from @pl.jit, use the loop form below.
with pl.spmd(4):
    out = self.kernel(a, b, out)
```

The loop form is the one a `@pl.jit` entry can write directly:

<!-- doctest: run -->
```python
@pl.jit
def spmd_add(
    a: pl.Tensor[[256, 128], pl.FP32],
    b: pl.Tensor[[256, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    for i in pl.spmd(2):                    # `i` binds the block index
        off = i * 128
        out = pl.store(
            pl.add(pl.load(a, [off, 0], [128, 128]), pl.load(b, [off, 0], [128, 128])),
            [off, 0],
            out,
        )
    return out


out = torch.zeros(256, 128, dtype=torch.float32)
spmd_add(X, Y, out, config=CFG)
torch.testing.assert_close(out, X + Y, rtol=1e-4, atol=1e-4)
```

A `with pl.spmd(n):` body that neither reads the block index nor dispatches a kernel is
rejected — every block would be doing identical work.

When a hard `pl.system.syncall` is involved, size the launch from the device rather than
from a literal: pass `pl.system.available_cluster_count()` (mixed or cube-only kernels) or
`pl.system.available_aiv_count()` (vector-only), written inline at the call site.

### Clusters and AIV lanes

`with pl.cluster():` groups AIC and AIV kernels so they are co-scheduled on the same
physical cluster, producing a `Group` function.

`for aiv_id in pl.split_aiv(2, mode=...):` splits one region across the two AIV lanes. It
belongs to mixed-kernel programming — AIC and AIV cooperating inside one function — which
the tutorials chapter covers end to end.

`mode=` picks how the two lanes divide the work:

| `mode=` | Each lane gets |
| ------- | -------------- |
| `pl.SplitMode.UP_DOWN` / `LEFT_RIGHT` | Half of every tile (rows / cols) — data-parallel |
| `pl.SplitMode.NONE` | The **full** body; you dispatch disjoint work via `aiv_id` — task-parallel |

**Opening one region changes the rules for the whole function.** The regions then own
every placement decision for vector work, so vector compute has to live inside a region:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        ...                                    # phase 1 — per-lane work via aiv_id
    pl.system.syncall(core_type=pl.KernelType.MIX)  # barrier: outside, runs on both
    mm = pl.matmul(q, k)                       # cube work: outside, runs on AIC
    for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        out = pl.add(pl.aiv_shard(mm), bias)   # phase 2 — full-width vector work
```

One region per vector phase. `mode=NONE` is the wrapper for a phase you do **not** want
halved: both lanes run the full body, which is what un-regioned vector code did anyway, so
wrapping it changes the text and not the execution. Cube ops and barriers stay outside.

`mm` is cube-produced and read on the vector lane, so it crosses the AIC/AIV boundary —
`pl.aiv_shard` is what says so. The next section explains why that is required.

A function with **no** `pl.split_aiv` at all is unaffected — write it exactly as before.

**Every crossing in one function must agree on split-vs-no-split.** All the
`pl.aiv_shard` / `pl.aic_gather` calls in a function ride a single cross-core pipe, and the
hardware fixes that pipe as either split or un-split for its whole lifetime. So a
`mode=NONE` region that crosses the boundary cannot sit beside an `UP_DOWN` or `LEFT_RIGHT`
region that also crosses it:

```python
for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    a = pl.exp(pl.aiv_shard(mm0))         # crossing, no split
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
    b = pl.exp(pl.aiv_shard(mm1))         # crossing, split      -> rejected
```

Two **different** split axes are fine, because the axis is chosen per transfer — only
split-vs-no-split belongs to the pipe:

```python
for r in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
    a = pl.exp(pl.aiv_shard(mm0))
for c in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
    b = pl.exp(pl.aiv_shard(mm1))         # accepted
```

A region that carries **no** crossing is free to use any mode — the `mode=NONE` region that
only pins a `pld.system.notify` to the vector lane never touches the pipe. When two phases
genuinely need different transports, put them in separate `pl.at(level=pl.Level.CORE_GROUP)`
scopes: each becomes its own function, and so gets its own pipe.

### Name every tile that crosses a region edge

**Once a function opens a region, a tile crossing a region edge must say so.** The
boundary between the two cores is yours to place in manual mode, so the compiler stops
choosing it for you:

| Direction | Where it is written | Op |
| --------- | ------------------- | -- |
| Cube value read on the vector lane (C->V) | at the top of the region | `pl.aiv_shard(x)` |
| Vector value read on the cube lane (V->C) | inside the region, before the read | `pl.aic_gather(x)` |

```python
mm = pl.matmul(q, k)                        # cube, outside every region
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    v = pl.exp(pl.aiv_shard(mm))            # C->V: named
    kv = pl.aic_gather(v)                   # V->C: named
out = pl.matmul(kv, w)                      # cube again, outside
```

Drop either call and the crossing still *works* — the compiler emits the same transfer
either way — but it is then a boundary nobody chose, in a program whose whole point is
that you chose it. So it is rejected instead; the diagnostic names the value and the op
that reads it.

**In a `mode=NONE` region both ops cross without splitting.** There is no split axis to
halve or re-join, so the shape passes through unchanged — `pl.aiv_shard` of a `[128, 128]`
tile is a `[128, 128]` tile. Only in `UP_DOWN` / `LEFT_RIGHT` do they also halve (shard)
and re-join (gather).

**Only gather a value both AIV lanes agree on.** The hardware requires both sub-lanes to
take part in a no-split handshake, and they share one destination slot with no per-lane
offset. Nothing arbitrates between them: both lanes push, so if they hold *different*
values the cube receives an **unspecified** one of the two. Not lane 0's — unspecified.

There is no way to select a lane here. Guarding the *production* of the value does not
help: lane 1 still reaches the push and still sends whatever its tile holds. So a
`pl.aic_gather` out of a `mode=NONE` region is well-defined only when the value is
lane-uniform — computed identically on both lanes, or made identical before the gather. If
the lanes must contribute different data to the cube, this construct cannot express it;
route it through GM and order it yourself, or use a data-parallel (`UP_DOWN` /
`LEFT_RIGHT`) region, where each lane owns a declared half and the gather re-joins them.

The compiler does not check any of this.

**GM traffic is not covered by any of this.** These rules are about *tile* values crossing
a region edge. A GM tensor belongs to no lane, so no boundary op can express a crossing
through one — `pld.tensor.put` takes a GM tensor by signature. AIC and AIV run
asynchronously. `ExpandMixedKernel` handles one narrow C->V case automatically: a unique
cube `tile.store` producer whose same-origin vector `tile.load` is in the same body or a
nested body. Every other GM handoff — including V->C, communication ops, and sibling bodies
— stays yours. `syncall` alone only aligns arrival: publish the producer's cache lines and
issue a GM fence before the barrier, then invalidate the consumer's cache before it reads.
For a buffer that may span multiple cache lines, use the conservative whole-GM
`pl.system.cacheinvalid()` form; the tensor-region overload currently covers only the cache
line containing the view's base address.

### Put cross-rank comm ops in a region

A region also decides placement for ops that have no lane of their own.
`pld.system.notify` is core-agnostic — the hardware runs TNOTIFY on either core —
so in a kernel that mixes cube and vector work the compiler emits it on **both**
the AIC and the AIV lane. Wrap the comm phase in a region and it is pinned to the
vector lane instead:

```python
for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    pld.tensor.put(dst=win, peer=peer, src=out,
                   dst_offsets=[0, 0], src_offsets=[0, 0], shape=[16, 256])
    pld.system.notify(target=sig, peer=peer, offsets=[0, 0], value=1,
                      op=pld.NotifyOp.AtomicAdd)
```

The cube copy is the dangerous one: the AIC lane can reach the notify before the
AIV lane's put has landed the data, publishing a signal for bytes that are not
there yet. A region removes it.

### Shard once-only side effects across the two AIV lanes

**A `mode=NONE` region body runs on BOTH AIV sub-lanes.** That is the whole point
of the mode — the region is not "one lane", it is two lanes running the same
code, and you dispatch the disjoint work with the loop's `aiv_id`. The snippet
above is therefore still incomplete: it fires *one* notify per lane, i.e. **two**
notifies for the same peer.

An op whose side effect must happen once per logical occurrence — a
`pld.system.notify` above all — must be either **sharded by `aiv_id`** or
**guarded to one lane**:

```python
# sharded: each lane takes a different set of peers
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    for owner in pl.range(aiv_id, NUM_PEERS, 2):
        pld.system.notify(target=sig, peer=owner, offsets=[0, 0], value=1,
                          op=pld.NotifyOp.AtomicAdd)

# guarded: lane 0 does it, lane 1 skips
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    if aiv_id == 0:
        pld.system.notify(target=sig, peer=peer, offsets=[0, 0], value=1,
                          op=pld.NotifyOp.AtomicAdd)
```

**The guarded form carries an ordering obligation the sharded one does not.**
The two AIV lanes run asynchronously — nothing orders lane 0 against lane 1. So
a lane-0-guarded notify may publish *before* lane 1's writes have landed, and
the peer then reads data that is still in flight. It is safe only when the data
the signal releases was written by lane 0 itself. If lane 1 contributes any of
it, order the two lanes explicitly before the notify, or prefer the sharded form
— there each lane releases only what it wrote, so the question does not arise.

**What goes wrong without it.** `NotifyOp.AtomicAdd` accumulates into the peer's
slot. Two lanes notifying the *same* peer make that rank's counter read `2` when
one rank has arrived. A `pld.system.wait` waiting for two arrivals is released by
one — so that rank runs ahead and reads a buffer whose data has not landed. The
symptom is wrong numbers on one rank, intermittently, at multi-rank runtime; it
does not reproduce on one rank and it does not look like a synchronisation bug.

### What the compiler does and does not do here

| Behaviour | What it means |
| --------- | ------------- |
| **Does** | Keeps a region's comm ops off the **cube** lane. Outside a region they are duplicated onto the AIC lane as well. |
| **Does not** | Check the lane-sharding. A notify that both AIV lanes run against the same peer compiles cleanly and **is not diagnosed**. |

The compiler cannot diagnose it: the correct form and the wrong form produce the
same single statement in the AIV function, differing only in whether `aiv_id`
reached the call's arguments. Getting this right is the author's job.

### Cross-lane ordering is yours too

AIC and AIV run **asynchronously**. A boundary op orders the one value it carries — that
is what the transfer is — but nothing orders a cube-lane write against a vector-lane read
of the same **GM buffer**. Publish the producer's cache lines and issue a GM fence, place a
cross-core barrier between the phases, then invalidate the consumer's cache before it
reads. The barrier by itself synchronizes arrival only. A region places work on a lane; it
does not sequence the two lanes against each other.

The conservative sequence below uses whole-GM cache maintenance and the soft barrier, so
it is safe for multi-cache-line buffers and partial occupancy. `sync_ws` is an exclusive,
zero-initialized 16-element `INT32` GM tensor, and `participant_count` is the total number
of participating AIC and AIV cores.

```python
pl.system.cacheinvalid()  # publish all producer cache lines
pl.system.fence()         # wait until they are visible in GM
pl.system.syncall(
    mode=pl.SyncAllMode.SOFT,
    core_type=pl.KernelType.MIX,
    gm_workspace=sync_ws,
    used_cores=participant_count,
)                         # synchronize arrival only
pl.system.cacheinvalid()  # consumer invalidates before reading
```

For a finer handoff than a whole-barrier rendezvous, `pl.system.sync_set` / `pl.system.sync_wait`
raise and await a single cross-core event. In a **mixed** InCore kernel, pin each one to the lane
that must run it with `core_type=pl.KernelType.AIC` or `core_type=pl.KernelType.AIV`; in an
explicitly typed AIC or AIV kernel the lane is already known, so omit the argument.

```python
pl.system.sync_set(0, pipe=pl.PipeType.MTE3, core_type=pl.KernelType.AIV)   # raised on AIV
pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)    # awaited on AIC
```

## Edge Cases

> **Fatal pitfall:** `pl.spmd` is an assertion, not a request. You are telling the compiler
> the blocks are independent. If they are not, the result is a race — not a diagnostic.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`Misplaced tensor op ... should be inside InCore block`** | Operators sit directly in the `@pl.jit` body | Wrap them in `with pl.at(level=pl.Level.CORE_GROUP):` |
| **`with pl.spmd(n):` body rejected** | It neither reads the block index nor dispatches a kernel | Read `pl.tile.get_block_idx()`, or call a kernel |
| **Most `pl.write` stores vanish, a different set each run** | Concurrent instances write different elements of one 64-byte cache line — the line, not the element, is what reaches DDR | Give each instance whole 64-byte lines, or write from `pl.spmd(1)`; see [Memory](03-memory.md#scalar-writes-from-concurrent-task-instances) |
| **`optimizations=` rejected** | Built up in a variable — the parser reads the AST | Write the list inline at the call site |
| **Printed IR cannot be reparsed** | A device-size query was bound to a name before use | Write the call inline where it is used |
| **`vector op '...' sits outside every pl.split_aiv region`** | The function opens a region, so the regions own vector placement | Wrap that phase in `for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):` |
| **`cube op '...' inside a pl.split_aiv region`** | A region body is AIV work | Move the `pl.matmul` out of the region |
| **`'x' is produced on the CUBE lane ... reads it on the VECTOR lane inside one`** | An unnamed C->V crossing into a region | Read it as `pl.aiv_shard(x)` at the top of the region |
| **`'x' is defined inside a pl.split_aiv region but ... reads it on the CUBE lane outside`** | An unnamed V->C crossing out of a region | Gather it inside the region: `x = pl.aic_gather(x)` |
| **`'pl.aiv_shard' crosses the AIC/AIV boundary under ... but ... earlier in this function crosses it under ...`** | One function's crossings mix `mode=NONE` with a split mode; they share one cross-core pipe | Make every crossing agree on split-vs-no-split, drop the crossing from one region, or split the phases into separate `pl.at(level=pl.Level.CORE_GROUP)` scopes |
| **The cube reads one lane's value at random** | A V->C crossing out of a `mode=NONE` region — both lanes push, one shared slot, no arbitration, **not diagnosed** | Gather only a lane-uniform value; use a data-parallel region if the lanes hold different halves |
| **A peer's signal counter reads twice what it should** | Both AIV lanes ran the same `pld.system.notify` — **not diagnosed** | Shard the notify by `aiv_id`, or guard it with `if aiv_id == 0:` |
| **A rank reads stale data after its `pld.system.wait` returns** | Either the double-notify above, or an incomplete cache-publication/fence/barrier/invalidation sequence between the cube and vector phases | Shard the notify; add the full GM handoff sequence |

## See Also

- [Functions and Programs](01-functions.md) — the alternative to `pl.at`: a separate `@pl.jit.incore` function.
- [Control Flow](02-control-flow.md) — the loops these scopes sit inside.
- [Memory and Data Movement](03-memory.md) — what the placed code does with buffers.
- [Tasks and Ordering](../tasks/index.md) — when the placed work runs relative to everything else.
- [OutlineIncoreScopes](../../dev/passes/08-outline_incore_scopes.md) — how `pl.at` becomes a function.
- [ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md) — what `pl.split` drives.
