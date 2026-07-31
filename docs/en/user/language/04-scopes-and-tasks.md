# Scopes and Tasks

Where work is placed, and how the dependency graph the runtime executes gets its shape.

> **Prerequisites:** [Functions and Programs](01-functions.md) and
> [Programming Model § execution model](../03-programming-model.md#the-execution-model).

## Concept

Two independent questions, often confused because both are spelled with `with`:

**Placement** — which hardware runs this code. `pl.at` puts a region on a core group,
`pl.cluster` groups co-scheduled Cube and Vector kernels, `pl.spmd` fans one kernel across
many blocks, `pl.split_aiv` splits a region across the two AIV lanes.

**Ordering** — what must finish before this task starts. That is the *runtime scope*
(`pl.scope` / `pl.manual_scope`) plus the per-task edges (`deps=`, `no_dep`,
`predicate=`).

The default for ordering is automatic. The runtime tracks every task's buffers in an
OverlapMap and derives edges from overlapping accesses, using the parameter directions
from [Types](00-types.md). You reach for the manual surface when auto-tracking derives an
edge that is not real (serializing work that could overlap), or cannot derive one that is
(because the relationship is not visible in buffer overlap).

The critical property, stated once: **statement order expresses nothing.** Two dispatches
written one after another are ordered only if something — an overlapping buffer, or an
explicit edge — says so.

## Quickstart: placement, then an explicit edge

```python
import pypto.language as pl

@pl.jit
def two_stage(
    x: pl.Tensor[[256, 128], pl.FP32],
    scratch: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    # Placement: run this region on a core group, and name its producer TaskId.
    with pl.at(level=pl.Level.CORE_GROUP) as t1:
        scratch = pl.mul(x, 2.0)

    # Ordering: this dispatch waits on t1 explicitly.
    with pl.at(level=pl.Level.CORE_GROUP, deps=[t1]):
        out = pl.add(scratch, scratch)

    return scratch, out
```

Here the `scratch` buffer overlaps, so auto-tracking would have derived the same edge —
`deps=[t1]` is redundant and harmless. It becomes load-bearing when the relationship is
*not* a buffer overlap.

## Mechanics

### Placement

| Construct | Creates | Use for |
| --------- | ------- | ------- |
| `with pl.at(level=...)` | InCore scope (`CORE_GROUP`) or Hierarchy scope | Marking a region as device work without writing a separate function |
| `with pl.cluster()` | Cluster scope → a `Group` function | Co-scheduling AIC and AIV kernels on one physical cluster |
| `with pl.spmd(n)` / `for i in pl.spmd(n)` | SPMD scope | Fanning one kernel across `n` blocks |
| `for aiv_id in pl.split_aiv(2, mode=...)` | `SplitAivScopeStmt` region | Splitting a region across the two AIV lanes |

`pl.at` keyword arguments, all optional:

| Argument | Level | Meaning |
| -------- | ----- | ------- |
| `optimizations=[pl.split(mode)]` | Placement | Cross-core split mode for the outlined kernel |
| `optimizations=[pl.cross_core_slot(slot_num=N)]` | Placement | Ring depth of the automatic cross-core pipe |
| `deps=[tid, ...]` | Ordering, TaskId | Explicit producer edges |
| `no_dep_args=[t, ...]` | Ordering, arg slot | Captured tensors excluded from dependency tracking |
| `dumps=[t, ...]` | Debug | Tensors to mark for selective dump |
| `allow_early_resolve=True` | Scheduling | Let consumers pre-stage before this task completes |
| `name_hint="..."` | Cosmetic | Name of the outlined function |

`optimizations=` entries must be written inline at the call site — the parser reads the
AST, so a list built in a variable is not accepted. `pl.split` and `pl.cross_core_slot`
are orthogonal and combine freely: one partitions work, the other sizes a channel.

```python
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                          pl.cross_core_slot(slot_num=4)]):
    ...
```

Omitting `cross_core_slot` keeps the default ring depth: 8 slots when one direction is
live, 4 per direction when both are. `pl.split(slot_num=...)` is deprecated — it forces a
split mode you may not want; use the two entries separately.

### SPMD

Three forms, differing in whether the body reads the block index and whether you capture
a TaskId:

```python
# 1. Dispatch form — body launches a pre-defined kernel.
with pl.spmd(4):
    out = self.kernel(a, b, out)

# 2. Loop form — body is auto-outlined; `i` binds the block index.
for i in pl.spmd(4):
    off = i * 128
    out = pl.store(pl.add(pl.load(a, [off, 0], [128, 128]),
                          pl.load(b, [off, 0], [128, 128])), [off, 0], out)

# 3. Capture form — same bodies as form 1, plus a producer TaskId.
with pl.spmd(4, deps=[prev_tid]) as tid:
    ...
```

A `with pl.spmd(n):` body that neither reads the block index nor dispatches a kernel is
rejected — every block would run identical work. `deps=` is only available on the `as tid`
form.

Size the launch from the device rather than a literal when a hard `pl.system.syncall` is
involved: pass `pl.system.available_cluster_count()` (mixed or cube-only) or
`pl.system.available_aiv_count()` (vector-only), written inline at the call site.

### Runtime scopes

A runtime scope (`PTO2_SCOPE`) is a resource and dependency-tracking boundary: it bounds
OverlapMap tracking and gives a per-scope heap level, so nested scopes reclaim memory
independently. The runtime provides an implicit top-level scope, so **writing scopes is
tuning, never a correctness requirement.**

| Mode | Meaning |
| ---- | ------- |
| `pl.scope()` / `ScopeMode.AUTO` | OverlapMap auto-tracking on |
| `pl.scope(mode=pl.ScopeMode.MANUAL)`, alias `pl.manual_scope()` | Auto-tracking off — you declare every edge |

Rules: scopes belong in Orchestration functions, not InCore ones. `mode=AUTO` is only
allowed under `@pl.function(auto_scope=False)` — in the default the compiler owns AUTO
placement. `MANUAL` is allowed either way. An AUTO scope may not nest inside a MANUAL one,
and a `manual_scope` may not nest inside another.

### Submitting tasks

`pl.submit` dispatches a kernel and hands back its producer TaskId. It is a parser
construct — calling it outside a decorated body raises.

```python
with pl.manual_scope():
    scratch, tid = pl.submit(self.stage1, x, scratch)
    out, _       = pl.submit(self.stage2, scratch, out, deps=[tid])
```

`pl.spmd_submit` is the SPMD sibling, with a required `core_num=` keyword:

```python
out, tid = pl.spmd_submit(self.kernel, x, core_num=8, sync_start=True, deps=[prev])
```

Both must be unpacked as a 2-tuple: element 0 is the kernel's result(s), element 1 is the
TaskId. Both work in **auto and manual scope** — `deps=` is orthogonal to OverlapMap
tracking, and the final fanin is the union of derived and explicit edges. In auto scope,
use `deps=` as a precision tool for edges the runtime cannot infer; in `manual_scope`, use
it for every edge.

### Opting out of dependency tracking

Three granularities, coarsest first:

| Construct | Scope of the opt-out |
| --------- | -------------------- |
| `with pl.manual_scope():` | Every submit in the region |
| `pl.create_tensor(..., manual_dep=True)` | One tensor, for its whole lifetime |
| `pl.no_dep(t)` at a call argument | One tensor, for one task |

`no_dep_args=[t]` on a `pl.at` scope is the same assertion as `pl.no_dep(t)`, for the case
where the kernel call is synthesized by the outliner and there is no syntactic argument
slot to wrap. It is legal for captures the body mutates as well as read-only ones — you
are asserting that sibling tasks touch disjoint regions.

`deps=` takes TaskIds; `no_dep_args=` takes tensors. They describe different things.

### Dispatch predicates

`predicate=` skips a task whose need is only known at runtime. The scheduler evaluates it
at the dispatch point — after dependencies are satisfied, so the value is current — and
when false retires the task inline without sending it to a core, while still settling
fanin and fanout so consumers unlock.

```python
with pl.spmd(1) as gate_tid:
    row_count = self.gate(row_count)

with pl.spmd(4, deps=[gate_tid], predicate=(row_count[0, 0] > 0)) as tid:
    out = self.expert(x, out)
```

The comparison is matched **syntactically and never evaluated** in orchestration — reading
it there would stall on `wait_for_tensor_ready`, exactly what the predicate avoids. Only
`tensor[indices] OP int-literal` is expressible, with one comparison and no chaining,
arithmetic, or boolean combination. Reduce anything richer to a single gate value in a
prior kernel.

**Contract:** the operand tensor's producer must be one of this task's `deps=`, or the
dispatch-point read may see a stale value. The parser checks this where it is statically
provable; the rest is yours.

### Scheduling hints

`allow_early_resolve=True` opts a task in as a speculative early-dispatch producer: the
scheduler may pre-stage its consumers on idle cores before it finishes. It is
producer-side — a consumer pre-stages only once all of its producers are flagged. Pure
scheduling optimization, no effect on results; it pays off on critical paths of many short
tasks.

## Edge Cases

> **Fatal pitfalls:**
>
> - Writing one dispatch after another does not order them. If the relationship is not
>   visible as a buffer overlap the runtime can see, there is no edge, and the two tasks
>   may overlap. The result is a race that reproduces intermittently and vanishes under a
>   debugger. State the edge with `deps=`.
> - `predicate=` over a tensor whose producer is not in `deps=` reads whatever happened to
>   be there. Nothing reports it — the task is skipped, or not, based on stale data.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Results change between runs** | Two tasks that must be ordered have nothing expressing it | Add `deps=[producer_tid]` |
| **Tasks serialize that should overlap** | Auto-tracking derived an edge from an overlap that is not a real dependency | `pl.no_dep(t)` at the argument, or `no_dep_args=[t]` on the scope |
| **`pl.submit is a DSL parser construct and cannot be called directly`** | Used outside a decorated body | Move it into a `@pl.function` / `@pl.jit` body |
| **`with pl.spmd(n):` body rejected** | It neither reads the block index nor dispatches a kernel | Read `pl.tile.get_block_idx()`, or call a kernel |
| **`deps=` rejected on `pl.spmd`** | Only the `as tid` form accepts it | `with pl.spmd(n, deps=[...]) as tid:` |
| **`optimizations=` rejected** | Built in a variable — the parser reads the AST | Write the list inline at the call site |
| **`DeprecationWarning` from `pl.split(slot_num=...)`** | Deprecated spelling | `optimizations=[pl.split(MODE), pl.cross_core_slot(slot_num=N)]` |
| **Nested-scope rejection** | AUTO inside MANUAL, or `manual_scope` inside `manual_scope` | Flatten; the runtime forbids both |
| **`predicate` / `allow_early_resolve` rejected under `pl.cluster()`** | A cluster-nested `pl.spmd` never produces a Submit | Move the hint out of the cluster |

## See Also

- [Types § parameter directions](00-types.md#parameter-directions) — what auto dependency derivation reads.
- [Control Flow](02-control-flow.md) — loops that contain these scopes.
- [Directives](05-directives.md) — `dump_tag` and the debugging surface referenced by `dumps=`.
- [AutoDeriveTaskDependencies](../../dev/passes/36-auto_derive_task_dependencies.md) — how edges are derived.
- [OutlineIncoreScopes](../../dev/passes/08-outline_incore_scopes.md) — how `pl.at` becomes a function.
- [MaterializeRuntimeScopes](../../dev/passes/42-materialize_runtime_scopes.md) — AUTO scope placement.
- [ExpandMixedKernel](../../dev/passes/20-expand_mixed_kernel.md) — what `pl.split` drives.
