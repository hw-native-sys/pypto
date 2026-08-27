# LegalizeGraphBoundary Pass

Makes every `FunctionType::Graph` function legal for the `host_build_graph`
runtime to record and replay: hoists the boundary scalars a Graph body derives
out to its call sites, and rejects the boundaries the runtime would decline to
cache.

## Overview

The `host_build_graph` runtime records a Graph function's task topology on its
first call and replays that recording afterwards. Replay patches only two
things: the addresses of the boundary tensors, and the values of the boundary
scalars. Everything else — node count, shapes, dependency edges, block counts —
is frozen into the recorded Definition.

That makes two classes of problem possible, and both are silent at runtime:

| Problem | What the runtime does | What this pass does |
| ------- | --------------------- | ------------------- |
| A boundary scalar is *derived* inside the region | Classifies it as static data and freezes the first call's value into the recording. No warning, ever. | **Step A** — hoists the computation to the call site |
| The boundary itself is not cacheable | Declines to cache and silently runs the region as ordinary tasks | **Step D** — rejects it at compile time |

The first produces wrong answers. The second produces correct answers with none
of the intended speedup — invisible to any numerical test, which is why the
checks live here rather than being left to a runtime log line.

## Step A — derived boundary scalars

A boundary scalar is tracked by **pointer identity**. During recording the
runtime anchors the address of each `args.scalar(k)` slot; on replay it re-reads
those addresses. A value the body computes has no slot:

```python
@pl.function(type=pl.FunctionType.Graph)
def layer(self, cur, wq, layer_idx: pl.Scalar[pl.INDEX]):
    base = layer_idx * 5120          # <- derived: no argument slot
    ...                              #    frozen at the first call's value
```

Step A rewrites this so the value arrives as a parameter instead:

```python
# after the pass, conceptually:
def layer(self, cur, wq, layer_idx, base):   # base is now a real boundary scalar
    ...

# and at each call site:
self.layer(cur, wq_view(i), i, i * 5120)     # the arithmetic moved out here
```

A value is hoistable when its whole expression tree bottoms out in the Graph's
own scalar parameters and constants — exactly the set a call site can recompute,
since it already supplies those parameters. Scalar arithmetic in PyPTO is a
`BinaryExpr` / `UnaryExpr` node rather than a `Call`, so the check recurses
through those two base classes and treats everything else as a leaf.

A scalar that reaches a task but is *not* hoistable — because it depends on a
task output, a tensor read, or a runtime query — is rejected with a message
naming the variable and explaining why the value cannot be reconstructed.

New parameters are **appended**, not prepended: `CoreTaskArgs` requires every
tensor argument to precede every scalar one.

## Step D — boundary legality

| Check | Why |
| ----- | --- |
| At least one tensor parameter | A graph with an empty boundary has nothing to patch on replay; the runtime refuses to cache it |
| At most 128 tensor parameters | `GRAPH_MAX_TENSOR_ARGS` — the boundary is a fixed-size `GraphTaskArgs` |
| At most 64 scalar parameters | `GRAPH_MAX_SCALAR_ARGS`. Checked after Step A, which *adds* scalar parameters, so a signature that fit before hoisting can stop fitting after |
| No `Out` tensor parameter | `Out` means the runtime allocates the buffer; a recorded graph's boundary tensors must already exist so replay can patch their addresses |
| Scalar parameters are `In` | A boundary scalar is passed by value and replayed from the call site |
| Returns only its own parameters | `rt_submit_graph` yields a valid task id only on a cache *hit*, so nothing can depend on a graph call's result. `return c` for an `InOut` parameter is the in-place spelling and is fine; a computed value is not |
| Between 1 and 1024 launched tasks | `graph_execution_storage_layout` refuses a node count of zero as well as one over `GRAPH_MAX_NODES`. A launch in a loop counts once per iteration, not once per call site, and `system.task_dummy` counts too — it lowers to `rt_submit_dummy_task`, and `ExpandManualPhaseFence` inserts them automatically. Allocations record nodes too, counted as an *upper* bound so that passing this check means the runtime will accept the Graph. Codegen collects every eligible create in a statement list — an intervening launch does not close the batch — and packs them at most `kAllocTensorsArgs` (16) to an `alloc_tensors`. Two of its three ineligibility rules cannot fire here (a shape reading a local is already rejected as non-constant; an already-declared var cannot recur under SSA), so those creates are counted exactly. The third can — an injected GM pipe buffer leaves the shared batch when its `core_num` reads a body-local, which only the emitter's use-resolution knows — so each of those is charged its worst case of one node. The batch size and the GM-pipe predicate are shared with the emitter in `utils/alloc_batching.h` rather than restated here |
| No allocation under a runtime loop or branch | Each records a node, so a count that varies between calls is a topology that varies |
| Allocation shapes are compile-time constants | Recording copies the shape into the node and derives the buffer address from it; a shape reading a boundary scalar is frozen at the first call's value |
| No `tensor.full` in the region | Orchestration codegen has no lowering for it and rejects it as a misplaced tensor op |
| No launch under a runtime loop, `while` or `if` | The recording fixes the topology on the first call and replays it unchanged, so a launch count or branch that can differ between calls would silently replay call one's shape |
| No scalar computed inline at a task argument | Step A hoists *named* derived values; an expression written inline at the call has no name to hoist and no boundary slot, so it would be frozen at the first call's value |
| No Graph calls a Graph | The runtime cannot record a graph from inside one it is already recording |
| Every parameter supplied at the call site | A `Submit` may normally pass a prefix and let the runtime allocate the tail `Out` params; a Graph has no such tail |
| No explicit dependencies on the launch | An explicit dependency edge makes the launch uncacheable, so the region would silently run as ordinary tasks |
| No dispatch predicate on the launch | A predicate on a graph launch is neither honoured nor rejected — the runtime silently zeroes it, so the region would run unconditionally |

## Position in the pipeline

Runs after the final `Simplify` and immediately before
[`MaterializeRuntimeScopes`](46-materialize_runtime_scopes.md).

That position is forced from both sides. `DeriveCallDirections` and
`AutoDeriveTaskDependencies` must already have run, so argument directions and
cross-task edges are known. `MaterializeRuntimeScopes` must not yet have run, so
no scope wrapper has been placed around the statements Step A moves.

## Pass properties

- **Requires**: `SplitIncoreOrch`, `CallDirectionsResolved`
- **Produces**: `GraphBoundaryLegalized`, `CallDirectionsResolved`

`CallDirectionsResolved` is re-declared because the pass rewrites call arguments
and their direction attrs; `MaterializeRuntimeScopes`, which runs next, requires
that property.

## Not yet handled

Derived *tensor slices* of a boundary tensor are handled separately. An
allocation inside the region is allowed here, subject to the constant-shape rule
above; hoisting one to the call site is a later change.

## See also

- [Pass Manager](00-pass_manager.md) — full pipeline order
- [MaterializeRuntimeScopes](46-materialize_runtime_scopes.md) — runs immediately after
