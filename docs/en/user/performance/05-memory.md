# Memory

The runtime's four rings, why scope placement decides which one your tasks land in, and how
to size them.

> **Prerequisites:** [Runtime scopes](../tasks/01-scopes.md) — this page assumes you know
> what a scope *is*, and treats it as a memory knob.

## The four rings

The runtime does not have one pool of task resources; it has **four independent ones**, and
a task's scope nesting depth picks which it uses:

```text
ring_idx = min(scope_depth, 3)

scope depth 0 ──► ring 0 ─┐
scope depth 1 ──► ring 1  │  each with its own task-slot window,
scope depth 2 ──► ring 2  │  output heap, and dependency-edge pool,
scope depth 3+ ─► ring 3 ─┘  reclaimed FIFO, independently
```

Each ring is a separate mapping with its own cursor and FIFO reclamation pointer, so
inner-scope tasks never share a FIFO head with outer-scope, longer-lived allocations. That
is the whole point of the design: a short-lived task in a deep scope can be reclaimed
without waiting on a long-lived allocation from the top level.

Each ring holds three separately-sized resources:

| Resource | Holds | Runs out as |
| -------- | ----- | ----------- |
| `task_window` | In-flight task slots | A capacity error naming the task window |
| `heap` | Output auto-allocation bytes | An allocation failure |
| `dep_pool` | Dependency-edge entries | A capacity error naming the dep pool |

## Why the default placement can waste them

By default the compiler owns scope placement: `MaterializeRuntimeScopes` wraps **the whole
function body, and each `for` body and each `if` then/else body** in its own AUTO scope.
That is a reasonable default — but it means your ring assignment is a side effect of your
control-flow shape, not of anything you decided.

```python
@pl.function(type=pl.FunctionType.Orchestration)   # auto_scope=True (default)
def orch(self, a, out):
    for i in pl.range(4):
        out = self.kernel(a, out)
    return out
```

becomes

```python
@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
def orch(self, a, out):
    with pl.scope():            # depth 0 — function body
        for i in pl.range(4):
            with pl.scope():    # depth 1 — loop body: every task lands in ring 1
                out = self.kernel(a, out)
        return out
```

A flat kernel — one function body, no loops or branches worth wrapping — puts **everything
in ring 0** and leaves rings 1–3 completely unused. The three idle rings are still mapped;
you paid for them and got nothing. The failure mode is asymmetric and unhelpful: ring 0
hits its ceiling and reports a capacity error while three-quarters of the resource sits
free next door.

## Rebalancing by hand

Opt out of compiler placement and put the scopes where the work is:

```python
@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
def orch(self, a, out):
    with pl.scope():
        # ... phase 1 tasks, ring 0
        with pl.scope():
            ...   # phase 2 tasks, ring 1 — reclaimed independently of phase 1
        return out
```

`@pl.jit`, `@pl.jit.host` and `@pl.jit.inline` accept `auto_scope=False`; `.incore` and
`.opaque` reject it, since they are outlined into standalone kernels with no orchestration
body for a scope to live in.

**Cost:** with `auto_scope=False` the pass inserts **nothing**, so every scope in the
function is now yours to place — including the ones the compiler was adding for free. This
is a placement decision only: an AUTO scope keeps auto dependency tracking on, so
rebalancing rings does not change your dependency semantics. (`MANUAL` mode does, and that
is [a different chapter](../tasks/01-scopes.md).)

**How to confirm:** scope stats — below. Peaks should spread across the rings instead of
stacking on one.

## Measuring before you size

Never resize a ring you have not measured. `RunConfig(enable_scope_stats=True)` records
per-scope peak usage of task-window slots, heap bytes, dep-pool entries, and tensormap
entries:

```python
cfg = RunConfig(platform="a2a3", enable_scope_stats=True, save_kernels=True)
```

```text
<work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

It is NDJSON: line 1 is run metadata, every later line is one scope sample. The metadata
line carries `task_window_max`, `heap_max` and `dep_pool_max` as arrays **indexed by ring
0..3** — the fastest way to confirm what sizing the run actually got. Render the whole
thing with the runtime's plotter:

```bash
python simpler_setup/tools/scope_stats_plot.py <...>/scope_stats/scope_stats.jsonl
```

Read it for two things:

- **A peak sitting at capacity** is a ceiling — that ring is the constraint.
- **Peaks well below capacity, on one ring only** is the imbalance above: rebalance scopes
  before you enlarge anything.

## Sizing the rings

When measurement says a ring is genuinely too small, three `RunConfig` fields size them.
Each takes a scalar (broadcast to all four rings) or a **list of exactly 4** ints sizing
rings 0..3 independently, where a `0` entry leaves that ring at its default:

| Field | Unit | Per-entry constraint |
| ----- | ---- | -------------------- |
| `ring_task_window` | In-flight task slots | Power of two, `>= 4` |
| `ring_heap` | **Bytes** | Power of two, `>= 1024` |
| `ring_dep_pool` | Dependency-edge entries | In `[4, INT32_MAX]` |

```python
cfg = RunConfig(
    platform="a2a3",
    ring_task_window=[8192, 16384, 131072, 524288],
    ring_heap=[134217728, 268435456, 402653184, 536870912],
)
```

Leaving a field `None` (the default) defers to the runtime's `PTO2_RING_*` environment
variables or its compile-time default, so you can also experiment without touching source.

**Cost:** memory, and the arithmetic is per ring — a scalar you meant as "just make it
bigger" is applied four times. Sizing the rings is also the *second* fix: a task window
that overflows because one scope holds thousands of tasks is better split into two scopes
than grown. The runtime says so itself when it fails — *"raise `ring_task_window`
(`PTO2_RING_TASK_WINDOW`) or split the scope"*.

**How to confirm:** the metadata line of a fresh `scope_stats.jsonl` shows the new sizes,
and the peak that was pinned at capacity is no longer pinned.

## On-chip buffers are a different problem

Everything above is host-visible runtime state. The tiles inside your kernel are planned
separately, and when *those* run out the error comes from the compiler, not the runtime.
`pypto.tools.memory_map` renders that allocation as HTML — address across, lifetime down,
IR alongside. Its input is a **pass dump**, not a run:

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

prog = kernel.lower(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
```

```bash
DUMP=path/to/output_dir/passes_dump/NN_after_SomePass.py
python -m pypto.tools.memory_map "$DUMP" -o map.html
```

Read it for tiles alive longer than they need to be, and for the headroom that decides
whether a deeper [pipeline](04-incore.md#double-buffering) or a deeper cross-core ring will
fit.

> Under `memory_planner=PTOAS` the compiler skips `AllocateMemoryAddr` entirely, so the
> pass dump carries no assigned offsets and this tool has nothing to draw. Compare end to
> end instead.

## See also

- [Runtime scopes](../tasks/01-scopes.md) — scopes as a dependency-semantics choice.
- [Tuning the InCore function](04-incore.md) — what consumes the on-chip side.
