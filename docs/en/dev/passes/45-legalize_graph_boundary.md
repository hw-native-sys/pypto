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
| A boundary scalar is *derived* inside the region | Classifies it as static data and freezes the first call's value into the recording. No warning, ever. | **Step A** — hoists the computation to the call site, or leaves it alone when the frozen value is provably the right one |
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

New parameters are **appended**, not prepended: `CoreTaskArgs` requires every
tensor argument to precede every scalar one.

### A bare rename is deleted, not accepted

`n = batch` computes nothing, so there is nothing to hoist — but leaving it alone
reintroduces the very bug Step A exists to prevent. Codegen emits a surviving
alias as a **value copy**:

```cpp
const uint64_t& batch = args.scalar(0);   // the slot, by reference
int64_t n = batch;                        // a copy, at a different address
g0_params_t0.add_scalar(n);               // the copy is what the task receives
```

Recording classifies a scalar by the *address its value came from*, comparing
against `&boundary_args->scalar(i)`. The copy matches nothing, so it is recorded
as `STATIC_VALUE` and every later replay reuses the first call's number.

Step A therefore substitutes the name away and erases the binding, so the task
reads `add_scalar(batch)` — the slot itself. Chains collapse to their root in one
pass (`a = p; b = a;` sends both readers to `p`), and an alias of a *hoisted*
value lands on that value's new parameter. A rename that somehow survives is
rejected rather than waved through, by both Step D and the verifier.

### Hoistable vs replay-invariant

Not being hoistable is not the same as being illegal. Hoistability answers *"can
the call site recompute this?"*; the runtime only needs *"is this the same on
every call?"*, which is strictly weaker. Freezing a value into the recording is a
wrong answer only when the value can **differ** between calls.

| Property | Question | Consequence |
| -------- | -------- | ----------- |
| hoistable | can the call site recompute it? | Step A moves it out; it gets a real argument slot |
| replay-invariant | is it the same on every call? | it stays where it is, frozen — correctly |

`ReplayInvariantSet` (`utils/graph_replay_invariant.h`) draws the second line.
Three seeds, closed under scalar arithmetic and over names bound to an invariant
value:

| Seed | Why replay reproduces it |
| ---- | ------------------------ |
| a literal | trivially |
| the induction variable of a **constant-trip** loop | recording walks the loop once and bakes each iteration's literal into that iteration's own node; constant bounds mean every later call walks the identical sequence |
| `tensor.dim` of a boundary tensor parameter, with a literal axis | `graph_boundary_matches` compares each boundary tensor's `ndims`, `shapes` and `strides` against the recorded `GraphBoundarySignature` and declines the cached graph on any mismatch, so a boundary shape cannot change within one recording |

**Scalar parameters are deliberately excluded from the invariant set.** The
runtime patches a boundary scalar's slot on every call — which is exactly what
makes one a legal *task argument* and an illegal *frozen view offset*.

This is what admits a tiled kernel at all. A slab offset `i * TILE` cannot be
hoisted: the value does not exist at the call site. Every projection, MLP and
attention loop in a decoder layer is indexed that way, so rejecting it excluded
the whole shape.

`DataType::TASK_ID` operands are skipped outright rather than classified. A task
id is never a boundary scalar — the recording captures dependency structure
itself, and `graph_boundary_matches` refuses any call carrying an explicit
dependency (`explicit_dep_count() != 0`), so an id produced outside the region
can never reach a replay. This matters because the scalar check runs over
*every* call's arguments, so `seeds[0] = seed` — an `array.update_element` into a
`pl.array` of `TASK_ID` — used to read as a task consuming a boundary scalar
while the same id written straight into `deps=[...]` was accepted.

A scalar that reaches a task and is neither hoistable, a boundary parameter, nor
replay-invariant — because it depends on a task output or a tensor read — is
rejected with a message naming the variable.

## Step B — derived slices of a boundary tensor

Replay patches a boundary tensor's **address**. A view taken *inside* the region
is re-derived from whatever the recording froze, so it must be taken at the call
site instead:

```python
wl = pl.tensor.slice(w, [128, 128], [layer_idx * 128, 0])   # inside the region
```

Step B moves that slice out and passes the result in as an additional boundary
tensor. Each slice site becomes its own parameter with its own fixed shape,
which is what the runtime's `BOUNDARY_VIEW` classification requires — it matches
on same-buffer plus offset, with the shape playing no part, so a view whose shape
varied between calls could not be classified at all.

The hoisted statements are emitted **scalars first, then tensors**, because a
slice's offset is typically a Step A scalar and the binding has to precede its
use. The *parameter* order is the reverse — tensors before scalars — which is
what `CoreTaskArgs` requires. A view of a region-local tensor stays put.

**A view of a hoisted view is hoisted too.** Once `wl` moves out it is a boundary
parameter, so `wr = slice(wl, ...)` is in exactly the position `wl` was. The body
is SSA in definition order, so one forward pass reaches the whole chain — a view
can only name a source defined before it. Leaving `wr` behind is silent:
`graph_rebind_tensor` patches the buffer address from `wl` but keeps the offset
recorded on the first call.

**A hoisted view must have a compile-time-constant shape.** Replay copies a
view's `shapes` and `strides` straight from the recorded template and patches
only `buffer_addr` and `start_offset`, so an extent read from a boundary scalar
would apply the first call's shape to a later call's buffer. An extent that is
replay-invariant is accepted for the same reason it is elsewhere: it cannot
differ, so freezing it changes nothing.

**A view has three outcomes, not two.** Replay restores a `BOUNDARY_VIEW` as
`boundary.start_offset + packed_offset`, where `packed_offset` is the delta
recorded on the **first** call; only the address, size, owner and version are
patched. So the offset is frozen, and that is wrong exactly when it can move:

| Every non-source operand is | Outcome | Why |
| --------------------------- | ------- | --- |
| derivable | hoisted to the call site | the caller can rebuild the view |
| replay-invariant only | **left in place** | frozen == correct, and the call site has no name for it |
| neither, or a mix of the two | rejected | the frozen delta would be call one's |

The mixed case is the one that matters. `off = layer_idx + i * TILE` can neither
be hoisted (`i` does not exist at the call site) nor frozen (`layer_idx` is
patched per call), and it is what a tiled region produces if it indexes weights
by both. Anything built on top of a view left in place — a view of it, or an
alias — is held to the same rule rather than hoisted, since the call site has no
name for any of them either.

**A varying offset is fine, and that is not obvious.** Codegen clamps a runtime
view to `min(declared, source.shapes[i] - offset[i])`, so the *actual* shape is
offset-dependent even when the IR extent is constant. It does not reach replay
because a hoisted view is passed as its **own** boundary tensor rather than
re-derived in the region: `graph_tensor_from_boundary` tries `BOUNDARY_EXACT`
across all boundary tensors before any `BOUNDARY_VIEW`, and a node consuming the
view matches the view itself, so `graph_rebind_tensor` replaces the whole
`GraphTensor` — `shapes`, `strides` and `extent_elem` included. The frozen-shape
behaviour belongs to `BOUNDARY_VIEW`, which is the in-region case this step
hoists out. Requiring the offset to be provably in bounds would reject the
motivating case, a per-layer `layer_idx * 5120`.

## Step C — allocations inside the region

`pl.create_tensor` **is allowed** in the region, subject to one rule: its shape
must be a compile-time constant.

Codegen lowers it into a batched `alloc_tensors`, and the runtime records that as
a kernel-less node — the same shape `submit_dummy_task` records — so it does not
poison the recording. It does count toward the node limit, and one under a
runtime loop or branch is a topology that varies between calls; both are Step D's
concern.

What recording cannot reproduce is a *shape* read from a boundary scalar. The
extent is copied into the node and the buffer's address derived from it, and
replay never re-runs the body, so a later call with a larger extent is handed the
first call's buffer — a wrong address layout rather than a fallback. Step D
rejects that, and `tensor.full` outright, which orchestration codegen has no
lowering for.

Hoisting a region-local allocation to a boundary parameter is *not* done: it
would add a second `InOut` parameter, and the return-alias mapping requires the
callee's `ReturnStmt` to name a parameter directly in order to disambiguate which
one a tensor return aliases — an invariant a synthesised parameter does not
satisfy. Automating it needs that mapping reworked first.

## Step D — boundary legality

| Check | Why |
| ----- | --- |
| The compilation targets `host_build_graph` | `GraphTaskArgs` and `rt_submit_graph` exist only in that runtime's orchestration API, and codegen emits them unconditionally, so a Graph built against the default `tensormap_and_ringbuffer` yields orchestration C++ that names undeclared symbols. Reported here, against the function the user wrote, rather than as a C++ error in generated code |
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
| No *call-varying* scalar computed inline at a task argument | Step A hoists *named* derived values; an expression written inline at the call has no name to hoist and no boundary slot, so it would be frozen at the first call's value. A replay-invariant expression is accepted — the freeze is harmless |
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

The `GraphBoundaryLegalized` verifier re-derives the topology, node count and
hoisting post-conditions independently, so a later pass that reintroduces a
rejected state is caught. It shares exactly one thing with this pass:
`ReplayInvariantSet`. That is not a decision this pass made and the verifier
rubber-stamps — it is a reading of the runtime's own contract, and two
hand-written copies of it could disagree, leaving the verifier rejecting IR the
pass had just produced.

## Not yet handled

Automatically hoisting a region-local allocation to a boundary parameter — one
is allowed in place, subject to the constant-shape rule above — and packing a
boundary of more than 128 tensors into a scratch arena.

## See also

- [Pass Manager](00-pass_manager.md) — full pipeline order
- [MaterializeRuntimeScopes](46-materialize_runtime_scopes.md) — runs immediately after
