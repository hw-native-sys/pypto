# Shaping the Task Graph

Getting the runtime to execute the graph you meant, rather than the one it inferred.

> **Prerequisites:** [Your first operator](00-elementwise.md).
> **Companion file:** `examples/intermediate/07_task_graph.py`.
> **Reference:** [Tasks and Ordering](../tasks/index.md).

## What you are building

A multi-task program whose dependency graph you can state, tighten, and loosen on purpose.

## The one property to internalise

The runtime does not execute your orchestration function statement by statement. It builds
a dependency graph and runs whatever is ready.

> **Statement order expresses nothing.** Two dispatches written one after the other are
> ordered only if *something* says so — a buffer overlap the runtime can see, or an edge
> you declared.

Where the edges come from by default: the runtime records which buffers each task touches
and how, using each parameter's direction, and derives an edge wherever two tasks touch the
same buffer.

## Step 1: the edge you get for free

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def inferred(
    x: pl.Tensor[[128, 128], pl.FP32],
    scratch: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1"):
        scratch[:] = pl.add(x, x)       # writes scratch
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2"):
        out[:] = pl.add(scratch, scratch)   # reads scratch
    return scratch, out

torch.manual_seed(0)
x = torch.randn(128, 128)
scratch = torch.zeros(128, 128)
out = torch.zeros(128, 128)
inferred(x, scratch, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, (x + x) + (x + x), rtol=1e-5, atol=1e-5)
```

Stage 1 declares `scratch` as an output, so it is recorded as its producer. Stage 2 reads
the same buffer, so a read-after-write edge is derived. Nothing was declared, and the
ordering is guaranteed.

This is the case that needs nothing from this page. Reach for the rest only when the
inference is wrong in one of two directions.

## Step 2: saying the edge yourself

If task B must follow task A for a reason that never shows up as a shared buffer, nothing
derives that edge — you have to declare it. The mechanism is a TaskId on the producer and
`deps=` on the consumer.

Shown here on the *same* pair as step 1, so you can still check the result:

```python
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1") as first:
        scratch[:] = pl.add(x, x)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2", deps=[first]):
        out[:] = pl.add(scratch, scratch)
```

`as first` binds the region's TaskId; `deps=[first]` makes the consumer wait on it.

Be clear about what this example does and does not prove. These two regions share
`scratch`, so the edge was **already** inferred in step 1 — writing it out changes nothing
here. It is the mechanism on a case you can verify, not a case that needs it. You need it
when the two tasks share no buffer at all, and there the result cannot be checked by
comparing against a golden: the wrong answer is a race, not a number.

**Explicit edges compose with automatic ones.** The final wait set is the union:

```text
final wait set  =  auto-tracked edges  ∪  explicit deps=
```

So `deps=` is a precision tool that works inside an ordinary auto scope. It does not
require, and does not imply, a manual scope.

## Step 3: an inferred edge that is not real

The opposite failure: the OverlapMap works on buffers, so sibling tasks writing *disjoint
regions* of one tensor look like they overlap, and get serialized. Three ways out, from
narrowest to widest:

| Construct | Opts out |
| --------- | -------- |
| `pl.at(..., no_dep_args=[t])` | One tensor, for one task |
| `pl.create_tensor(..., manual_dep=True)` | One tensor, for its whole lifetime |
| `with pl.manual_scope():` | Every task in the region — you declare all edges |

Prefer the narrowest that expresses the claim.

> **Fatal pitfall:** every one of these is an assertion the compiler cannot check. If the
> regions are not actually disjoint, you have deleted a real edge and bought a race — the
> same defect as never declaring it.

## Step 4: skipping work, and starting early

Two more knobs, and they are not the same kind of thing:

**`predicate=`** — do not run this task at all when a runtime value says so. Evaluated at
the dispatch point, so the value is current without an orchestration-time wait. Available
on `pl.submit`, `pl.spmd_submit` and `pl.spmd` — not on `pl.at`.

```python
with pl.spmd(4, deps=[gather_tid], predicate=(row_count[e] > 0)) as tid:
    ...
```

Only `tensor[indices] OP int-literal` is expressible — one comparison. Reduce anything
richer to a gate value in a prior kernel.

**`allow_early_resolve=True`** — a pure scheduling hint. The scheduler may pre-stage this
task's consumers before it finishes. It changes timing, never results.

| Construct | Changes correctness | Changes what runs | Changes timing |
| --------- | ------------------- | ----------------- | -------------- |
| Opt-outs (`no_dep`, `manual_dep`) | **Yes** | No | Yes |
| `predicate=` | No | **Yes** | Yes |
| `allow_early_resolve=` | No | No | **Yes** |

Reaching for the wrong row is the usual mistake. Only the first can corrupt your results.

## Step 5: waiting on many producers

One TaskId names one task. To wait on a loop's worth, collect them into a `pl.array` of
`pl.TASK_ID` and pass the array:

```python
tids = pl.array.create(branches, pl.TASK_ID)
for branch in pl.parallel(branches):
    out, tid = pl.submit(self.producer, data, branch, out)
    tids[branch] = tid
out, _ = pl.submit(self.consumer, data, out, deps=[tids])
```

Fragment — `pl.submit` needs a `self.<kernel>` callee, so this form lives in a
`@pl.program` class. From `@pl.jit`, use `pl.at`/`pl.spmd` blocks with `deps=`. See
[Declaring an edge](../tasks/02-submit.md).

`pl.system.task_dummy(deps=[...])` does the same join with no work, returning one TaskId
that stands for several producers.

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **Results differ between runs** | Two tasks that must be ordered have nothing expressing it | Declare it with `deps=` |
| **Adding a print makes it correct** | Timing changed, not semantics | The missing edge is still missing |
| **Work that should overlap runs serially** | An inferred overlap that is not a real dependency | Opt the argument out — step 3 |
| **`pl.at` rejects `predicate=`** | `pl.at` has no predicate | Use `pl.spmd` or a submit form |
| **A consumer waits only on the last producer** | One TaskId reused instead of collected | Collect into a `pl.TASK_ID` array |

## Next

[Tuning the schedule](05-scheduling-tuning.md) — now that the graph is what you meant,
find out what the runtime actually did with it.
