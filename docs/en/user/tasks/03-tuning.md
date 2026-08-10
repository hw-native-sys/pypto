# Refining the Graph

Removing an edge that is not real, skipping a task that is not needed, and letting the
scheduler start early.

> **Prerequisites:** [Declaring an edge](02-submit.md).

## Concept

The previous pages built the graph. This one changes it in three directions, and they are
genuinely different operations — reaching for the wrong one is the usual mistake:

| You want | Reach for |
| -------- | --------- |
| An inferred edge gone, because it is not a real dependency | An **opt-out**: `manual_scope`, `manual_dep=True`, or `pl.no_dep` |
| The task not to run at all when a runtime value says so | A **dispatch predicate**: `predicate=` on `pl.submit` / `pl.spmd_submit` / `pl.spmd` |
| The same graph, dispatched sooner | A **scheduling hint**: `allow_early_resolve=` |

Only the first changes correctness guarantees. The predicate changes what runs; the hint
changes nothing but timing.

## Quickstart: three granularities of opt-out

```python
with pl.manual_scope():                              # whole region: every task inside
    ...

t = pl.create_tensor(..., manual_dep=True)           # one tensor, its entire lifetime

with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[shared]) as tid:   # one tensor, one task
    ...
```

| Construct | Scope of the opt-out | Available in |
| --------- | -------------------- | ------------ |
| `with pl.manual_scope():` | Every task in the region | `@pl.jit`, `@pl.function` |
| `pl.create_tensor(..., manual_dep=True)` | One tensor, for as long as it lives | `@pl.jit`, `@pl.function` |
| `pl.at(..., no_dep_args=[t])` | One tensor, for one task | `@pl.jit`, `@pl.function` |
| `pl.no_dep(t)` at a call argument | One tensor, for one task | `@pl.program` classes |

Prefer the narrowest one that expresses the claim. Opting one argument out says "this
argument of this task has no conflict"; `manual_scope` says "I own the entire graph here",
which is a much larger promise.

> Fragments: each line belongs in an Orchestration function body.

## Mechanics

### `pl.no_dep`

A parser-recognised marker at a kernel-call argument position — at runtime it returns the
tensor unchanged. It makes the runtime skip **both** the OverlapMap dependency lookup *and*
the producer insert for that argument.

It is legal whether the callee declares the parameter `In`, `Out` or `InOut`, because what
you are asserting is out-of-band: that there is no read-after-write, write-after-write or
write-after-read conflict on that slot. The motivating case is a write whose target offset
is data-dependent — the compiler cannot prove disjointness, but the allocation protocol
guarantees it.

Wrapping a call argument this way needs an explicit `self.<kernel>` call, so it belongs to
the `@pl.program` form. In a `@pl.jit` function the equivalent is `no_dep_args=[t]` on the
enclosing `pl.at` scope — which is also what you use for kernel calls the outliner
synthesises, where there is no syntactic argument slot to wrap.

`deps=` takes TaskIds; `no_dep_args=` takes tensors. They are not two spellings of one
thing.

### `predicate=`

Carried by `pl.submit`, `pl.spmd_submit` and `pl.spmd` — but **not** by `pl.at`. In a
`@pl.jit` function `pl.spmd` is the form that has it; reach for it rather than `pl.at` when
the region needs a predicate.

Skips a task whose need is only known at run time. The scheduler evaluates the comparison
at the **dispatch point** — after dependencies are satisfied, so the value is current
without an orchestration-time wait. When it is false the task is retired inline and never
reaches a core, while its fanin and fanout still settle so downstream consumers unlock
normally.

```python
out, tid = pl.spmd_submit(self.expert_ffn, tokens, out, core_num=N,
                          deps=[gather_tid],
                          predicate=(row_count[e] > 0))
```

The comparison is **matched syntactically and never evaluated**. In this position
`row_count[e] > 0` is a spec handed to the scheduler, not a `tensor.read` plus a compare —
reading it in the orchestration would mean waiting on the tensor, which is the thing the
predicate exists to avoid.

Only `tensor[indices] OP int-literal` is expressible: one comparison, with `==` `!=` `>`
`<` `>=` `<=`. No chained comparisons, no arithmetic, no boolean combination — the runtime
supports a single comparison. Reduce anything richer to one gate value in a prior kernel
and predicate on that.

**Contract:** the operand tensor's producer must be one of this task's `deps=`, so the
dispatch-point read sees the current value. The parser enforces this where it is statically
provable; beyond that it is yours to honour. On a `pl.spmd` region this forces the `as tid`
form: only that spelling accepts `deps=` at all — `with pl.spmd(n, deps=[...]):` and
`for i in pl.spmd(n, deps=[...]):` are rejected outright. A bare `with pl.spmd(n,
predicate=...):` does parse, but it has no way to name the producer, so honouring the
contract is on you.

### `allow_early_resolve=`

Flags a task as a speculative early-dispatch producer: the scheduler may pre-stage its
consumers onto idle cores before it finishes, releasing them with a doorbell the moment it
does. It is a **producer-side** hint — a consumer only pre-stages once *all* of its
producers are flagged or already complete.

Pure scheduling: results are unaffected. It pays off on a critical path built from many
short tasks and is harmless otherwise. A `sync_start` SPMD task cannot itself be pre-staged
block by block, but flagging it still lets its consumers pre-stage.

### `pl.system.task_dummy`

A dependency join with no work: it takes `deps=[...]` and returns a TaskId, so several
producers can be collapsed into one handle that later tasks name.

```python
gate = pl.system.task_dummy(deps=[tid_a, tid_b])
out, _ = pl.submit(self.consumer, x, out, deps=[gate])
```

Like `pl.submit`, it is a parser construct — calling it outside a decorated body raises.
Note the spelling: it lives under `pl.system`, not at top level.

## Edge Cases

> **Fatal pitfalls:**
>
> - `pl.no_dep` is an assertion the compiler cannot check. If the regions are not actually
>   disjoint you have removed a real edge, and the result is a race — the same class of bug
>   as never declaring the edge at all.
> - `predicate=` over a tensor whose producer is not in this submit's `deps=` reads whatever
>   happened to be in memory. Nothing reports it; the task is skipped or not, on stale data.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Race after adding `no_dep`** | The regions are not disjoint after all | Remove the marker; the edge it deleted was real |
| **`pl.no_dep` breaks metadata inference under `@pl.jit`** | The wrapper hides the tensor from `@pl.jit`'s shape/dtype inference | Use `no_dep_args=[t]` on the enclosing `pl.at` scope instead |
| **Predicate rejected by the parser** | Only `tensor[indices] OP int-literal` is expressible | Reduce to one gate value in a prior kernel, predicate on that |
| **Predicated task runs when it should not** | The operand's producer is not in `deps=` | Add the producer's TaskId to `deps=` |
| **`predicate` / `allow_early_resolve` rejected under `pl.cluster()`** | A `pl.spmd` nested in a cluster produces no Submit to carry the hint | Move the hint out of the cluster |
| **`allow_early_resolve` changed nothing** | A consumer pre-stages only when *all* its producers are flagged | Flag the other producers too, or accept it does not apply |
| **`pl.task_dummy` is not defined** | It lives under `pl.system` | Call `pl.system.task_dummy(deps=[...])` |

## See Also

- [Declaring an edge](02-submit.md) — where `predicate=` and `allow_early_resolve=` are spelled.
- [Runtime scopes](01-scopes.md) — the coarsest opt-out, and why it is rarely the right one.
- [Types § parameter directions](../language/00-types.md#parameter-directions) — the declaration `no_dep` overrides.
- [Operations](../ops/01-catalog.md) — catalog entries for `no_dep` and the system operators.
