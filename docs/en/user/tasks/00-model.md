# The Dependency Model

What a task is, how the runtime works out the order between tasks, and the one property
that surprises everybody.

> **Prerequisites:** [Types § parameter directions](../language/00-types.md#parameter-directions).

## Concept

A **task** is one kernel dispatch. The runtime does not execute your orchestration function
statement by statement — it builds a dependency graph of tasks and runs whatever is ready.

The graph is derived, not written. The runtime keeps an *OverlapMap*: for each task it
records the buffers that task touches and how it touches them, using the direction each
parameter declares. Two tasks that touch the same buffer get an edge whose direction falls
out of theirs — a reader waits for the writer before it, a writer waits for the readers and
the writer before it. Two tasks that touch nothing in common get no edge and may run at the
same time.

That leads to the property this whole chapter exists for:

> **Statement order expresses nothing.** Two dispatches written one after the other are
> ordered only if *something* says so — a buffer overlap the runtime can see, or an edge
> you declared. Adjacency in the source is not that something.

## Quickstart: an edge nobody had to declare

```python
import pypto.language as pl

@pl.jit.incore
def twice(x: pl.Tensor[[256, 128], pl.FP32], out: pl.Out[pl.Tensor[[256, 128], pl.FP32]]):
    out[:] = pl.add(x, x)
    return out

@pl.jit
def two_stage(
    x: pl.Tensor[[256, 128], pl.FP32],
    scratch: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    scratch = twice(x, scratch)        # writes scratch
    out = twice(scratch, out)          # reads scratch  -> ordered after the write
    return scratch, out
```

| What the runtime sees | Consequence |
| --------------------- | ----------- |
| Task 1 declares `scratch` as `Out` | It is recorded as the producer of `scratch` |
| Task 2 declares the same buffer as `In` | A read-after-write edge is derived |
| Nothing else overlaps | No other edges; anything independent stays parallel |

No `deps=`, no scope, no annotation beyond the directions already on the signatures. This
is the case that needs nothing from this chapter.

## Mechanics

### What a direction means for ordering

| Direction | The runtime concludes |
| --------- | --------------------- |
| In (default) | Reads it. Orders after the task that last wrote it |
| `pl.Out[...]` | Writes it, does not read it. Orders after prior readers and the prior writer |
| `pl.InOut[...]` | Both. Orders against everything that touched it |

This is why a wrong direction is a correctness bug rather than a style issue: declaring an
`InOut` buffer as `Out` tells the runtime nothing needs to finish before this task writes
it. See [Types § parameter directions](../language/00-types.md#parameter-directions).

### What the inference cannot see

The OverlapMap works on buffers, so it infers exactly the relationships that *are* buffer
overlaps. Two situations fall outside that:

**An overlap that is not a dependency.** Sibling tasks writing disjoint regions of one
tensor overlap as far as the map is concerned, so they are serialized. If the disjointness
is real but not provable — a data-dependent write offset, say — the inference is
conservative and you lose the parallelism. [Refining the graph](03-tuning.md) covers the
opt-outs.

**A dependency that is not an overlap.** If task B must follow task A for a reason that
never shows up as a shared buffer, nothing derives that edge. You declare it — see
[Declaring an edge](02-submit.md).

### Automatic and explicit edges compose

They are not alternatives. An explicit `deps=[...]` on a submit is added on top of whatever
the runtime already inferred:

```text
final wait set  =  auto-tracked edges  ∪  explicit deps=
```

So `deps=` works inside an ordinary auto scope, as a precision tool for the edges the
inference cannot reach. Turning the inference *off* is a separate decision, and a heavier
one — that is [`pl.manual_scope`](01-scopes.md).

## Edge Cases

> **Fatal pitfall:** if nothing expresses an ordering, the tasks may overlap — and the
> result is a race that reproduces intermittently and disappears under a debugger or a
> print. Neither statement adjacency nor "it worked when I ran it" is evidence of an edge.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Results differ between runs** | Two tasks that must be ordered have nothing expressing it | Declare it with `deps=` — see [Declaring an edge](02-submit.md) |
| **Work that should overlap runs serially** | The inference saw an overlap that is not a real dependency | Opt the argument out — see [Refining the graph](03-tuning.md) |
| **A buffer holds stale data in a later task** | The producing task was not ordered before the consumer | Check the directions first; a wrong direction silently removes the edge |
| **Adding a `print` or a debugger makes it correct** | Timing changed, not semantics | The missing edge is still missing — do not treat this as fixed |

## See Also

- [Runtime scopes](01-scopes.md) — the boundary this inference runs inside, and how to switch it off.
- [Declaring an edge](02-submit.md) — naming a task so a later one can wait on it.
- [Types § parameter directions](../language/00-types.md#parameter-directions) — the declaration the inference reads.
- [AutoDeriveTaskDependencies](../../dev/passes/39-auto_derive_task_dependencies.md) — the compiler-side half of this.
