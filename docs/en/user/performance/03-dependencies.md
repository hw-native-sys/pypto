# Managing Dependencies

Why the runtime sometimes runs in sequence what could have run at once, and what to do
about it.

> **Prerequisites:** [Tasks and Ordering](../tasks/index.md) — this page assumes you know
> what `deps=`, `manual_scope` and `no_dep_args` *are*. Here they are tuning decisions.

## What the runtime infers, exactly

Task dependencies are derived at submit time from the tensor arguments each task carries.
Being precise about the rule is worth the paragraph, because every false edge in this
chapter comes out of it:

| Step | Applies to | Effect |
| ---- | ---------- | ------ |
| **Creator retention** | *Every* tensor argument, any direction | An edge on the task that created that tensor |
| **Producer lookup** | `INPUT` / `INOUT` only | An edge on the current registered producer of any **overlapping** region |
| **Producer registration** | `INOUT` / `OUTPUT_EXISTING` | This task becomes the registered producer for that buffer |

Which gives you the two classic hazards, and one gap:

- **RAW** — a reader looks up the current writer and takes an edge. Tracked.
- **WAW** — a new writer takes an edge on the prior writer, then replaces it. Tracked.
- **WAR** — a writer overwriting a buffer a pure reader may still be reading. **Not
  tracked.** A writer would have to find every in-flight reader, which is a per-write walk
  over a reader set on the hot path. If you need that ordering, you own it.

Loop kinds sit on top of this: `pl.range` is sequential, `pl.parallel` asserts the
iterations are independent. `pl.parallel` is an **assertion, not a request** — it does not
remove the edges above, it promises you have not created any that matter.

## Serialization you did not ask for

### The accumulator chain

The most common one. A sequential loop whose iterations write the same buffer produces a
WAW chain, one edge per iteration, and that is *correct* — the writes really do land in one
place.

```python
for i in pl.range(N):
    with pl.at(level=pl.Level.CORE_GROUP):   # writes `acc` every iteration
        ...                                  # → WAW edge on iteration i-1
```

It becomes a performance bug when the iterations write **disjoint regions** of that buffer
and only look like they collide. The producer lookup is an *overlap* test over buffer
addresses; a region it cannot prove disjoint is treated as overlapping.

This is also why a `pl.range` outer loop wrapping a `pl.parallel` inner loop often
disappoints: the inner iterations may well overlap each other, but the outer loop's shared
output buffer still chains the iterations together, and the parallelism you declared inside
never gets a chance to show.

**The fix is to say what the compiler cannot prove**, at the narrowest scope that expresses
it:

| Scope of the claim | Construct |
| ------------------ | --------- |
| One tensor, one task | `pl.at(..., no_dep_args=[t])` |
| One tensor, its whole lifetime | `pl.create_tensor(..., manual_dep=True)` |
| Every task in a region | `with pl.manual_scope():` |

Prefer the narrowest one that works. Each is an assertion the compiler **cannot check** —
if those regions do overlap after all, you have not fixed a serialization, you have created
an intermittent race that reproduces on someone else's machine.

**There is a fourth option, and it is the only one that needs no assertion.** Slice the
output in the orchestration and pass each slice to its InCore function, so the tasks no
longer share a buffer at all:

```python
for i in pl.range(N):
    part = pl.slice(out, [TILE, COLS], [i * TILE, 0])   # a distinct region per iteration
    with pl.at(level=pl.Level.CORE_GROUP):
        ...                                             # writes `part`, not `out`
```

Now the regions are disjoint *by construction*, and the runtime derives that rather than
being told it. **Cost:** the extra orchestration-level tensors are themselves work — more
arguments to register and more entries to walk — so dependency resolution takes longer per
task. On a graph that was already dispatch-bound ([01](01-task-granularity.md)) that can
cost more than the serialization it removed. Measure both ends.

### Readers that serialize each other

The other direction, and it usually arrives as a well-intentioned fix. Because WAR is not
tracked, a reader that must finish before a later overwrite has no edge protecting it. The
tempting move is to promote the reader from `INPUT` to `INOUT`, which does create the
edge — an `INOUT` registers as a writer, so the overwrite takes a WAW edge on it.

**And it serializes every other reader of that buffer.** Each `INOUT` reader becomes the
registered producer in turn, so the second takes a WAW edge on the first. A tensor read
concurrently by several tasks loses that concurrency entirely, to buy one anti-dependency.

Declare the edge explicitly instead — `deps=[reader_tid]` on the writer — and leave the
readers as `INPUT`.

**How to confirm either fix:** `enable_dep_gen=True` and compare the graph — the removed
edge should be gone and nothing else should have moved. Then the swimlane, because a graph
that fans out does not prove tasks overlapped; a saturated ring can still serialize them.
Check both.

## Fine-grained edges you have to build yourself

Some dependencies are not visible as a buffer overlap at all, and no amount of inference
will find them. `models/qwen3_14b/decode_fwd.py` in pypto-lib is the reference for what
that looks like at scale — a decode layer wired almost entirely by hand.

The pattern worth taking from it: **hoist the TaskId array out to orchestration scope** so
that a consumer running *after* a `manual_scope` can still gate on tasks created *inside*
it.

```python
# Declared before the manual scope — so a later, outside consumer can read it
down_tids = pl.array.create(DOWN_ON * K_SPLITS, pl.TASK_ID)

with pl.manual_scope():
    # ... the loop fills down_tids[k] as it submits each down_proj task
    ...

# After the scope: gate the consolidated writer on those producers
with pl.at(level=pl.Level.CORE_GROUP,
           deps=[down_tids[k] for k in range(DOWN_ON * K_SPLITS)]):
    ...
```

Inside `pl.manual_scope()` nothing registers in the tensormap at all — producer
registration is a no-op there — so every edge in the region is one you wrote. That is the
point of it: on a path where the inferred edges were mostly false, declaring the true ones
is less work than removing the wrong ones.

> The model uses the per-index form shown above rather than passing the whole array. Both
> spellings exist ([fan-in through a TaskId
> array](../tasks/02-submit.md#fan-in-through-a-taskid-array)); if a whole-array `deps=`
> does not produce the edges you expect in a hoisted-across-scope case like this one, the
> per-index list is the form that model relies on.

**Cost:** every edge is now your responsibility, including the ones that were previously
correct for free. A missing edge in a manual scope is a race, not an error message.

**How to confirm:** `enable_dep_gen=True`, and read the graph against what you intended —
this is the one case where reading the whole graph, not a diff of it, is the check.

## Deciding

```text
Tasks are serialized but should not be
├─ They write the same buffer, genuinely       → not a bug; merge them or restructure
├─ They write disjoint regions of one buffer   → no_dep_args / manual_dep, narrowest first
├─ A reader was promoted to INOUT for ordering → revert it; use deps= on the writer
└─ The edge is not about buffers at all        → manual_scope + explicit deps
```

## See also

- [Tasks and Ordering § the dependency model](../tasks/00-model.md) — the full derivation.
- [Tuning the schedule](../tutorials/05-scheduling-tuning.md) — the same reasoning applied
  step by step to one kernel.
