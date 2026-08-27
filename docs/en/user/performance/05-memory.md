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

**Deep nesting fails the same way, at the other end.** The mapping saturates —
`min(scope_depth, 3)` — so scope depth 3, 4, 5 and everything below all land on **ring 3**:

```text
depth  0    1    2    3    4    5    6 ...
ring   0    1    2    3 ── 3 ── 3 ── 3   ← every deeper scope piles onto one ring
```

A kernel with several nested loops therefore concentrates its innermost — and usually most
numerous — tasks on a single ring, which is exactly the ring that overflows. Flattening the
nesting, or hoisting scopes so the deep levels are not each wrapped, spreads them back
out.

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

N, TR, TC = 4, 64, 128
ROWS = N * TR

torch.manual_seed(0)
A = torch.randn(ROWS, TC, dtype=torch.float32)


def check(kernel, cfg):
    out = torch.zeros(ROWS, TC, dtype=torch.float32)
    kernel(A, out, config=cfg)
    torch.testing.assert_close(out, A * 2.0 + 1.0, rtol=1e-4, atol=1e-4)
```

## Rebalancing by hand

Opt out of compiler placement and put the scopes where the work is:

The depths are the whole point: phase 1 sits in the outer scope, phase 2 one level deeper,
so they land on different rings. Two *sibling* scopes would not do this — `ring_idx` is a
function of depth alone.

<!-- doctest: run -->
```python
@pl.jit(auto_scope=False)
def manual_placement(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.scope():                       # depth 0 — phase 1's tasks, ring 0
        for i in pl.range(N):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(a, [i * TR, 0], [TR, TC])
                pl.store(pl.mul(t, 2.0), [i * TR, 0], out)
        with pl.scope():                   # depth 1 — phase 2 moves to ring 1
            with pl.at(level=pl.Level.CORE_GROUP):
                for i in pl.range(N):
                    t = pl.load(out, [i * TR, 0], [TR, TC])
                    pl.store(pl.add(t, 1.0), [i * TR, 0], out)
    return out


check(manual_placement, RunConfig(platform="__PLATFORM__"))
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
# the runtime submodule ships the plotter
python runtime/simpler_setup/tools/scope_stats_plot.py \
    <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
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

<!-- doctest: run -->
```python
sized = RunConfig(
    platform="__PLATFORM__",
    ring_task_window=[8192, 16384, 131072, 524288],
    ring_heap=[134217728, 268435456, 268435456, 536870912],
)


@pl.jit
def scaled(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(N):
            t = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.add(pl.mul(t, 2.0), 1.0), [i * TR, 0], out)
    return out


check(scaled, sized)
```

Leaving a field `None` (the default) defers to the runtime's compile-time default.
The process-wide `PTO2_RING_*` environment variables are retired and no longer read,
so `RunConfig` is the only way to size the rings.

**Cost:** memory, and the arithmetic is per ring — a scalar you meant as "just make it
bigger" is applied four times. Sizing the rings is also the *second* fix: a task window
that overflows because one scope holds thousands of tasks is better split into two scopes
than grown. The runtime says so itself when it fails — *"raise `ring_task_window`
(`runtime_env.ring_task_window`) or split the scope"*.

**How to confirm:** the metadata line of a fresh `scope_stats.jsonl` shows the new sizes,
and the peak that was pinned at capacity is no longer pinned.

## Keeping a streaming operand out of the cache

Not every buffer a kernel reads deserves cache. A weight matrix that each byte is
read from exactly once cannot hit, yet it still sweeps the last-level cache and
evicts the small activation working set that *does* get reread. Two operands, one
cache, opposite needs.

The compiler cannot tell them apart — reuse is not reliably visible to it, and a
wrong guess is a silent multi-percent regression rather than an error. So you
declare it:

```python
with pl.at(level=pl.Level.CORE_GROUP):
    pl.set_cache_policy(weights, pl.CachePolicy.BYPASS)
    # every read of `weights` in this scope is now declared streaming;
    # the activations keep the default cached path.
    acc = pl.matmul(activations, weights, out_dtype=pl.FP32)
```

`pl.set_cache_policy` is a standalone statement at the top level of a
`pl.at(level=pl.Level.CORE_GROUP, ...)` or `pl.spmd(...)` body — anywhere among
that body's own statements, not necessarily the first line. It covers every read
of that tensor in the scope, so tensor-level code needs no change at the access
sites — which matters because those reads are implicit: `pl.matmul`,
`pl.assemble` and subscript slicing issue loads without a call you could
annotate.

A declaration on a non-`CORE_GROUP` `pl.at` scope parses and is carried, but
nothing lowers it into a load today, so it has no effect — those scopes do not
become device kernels.

When you *are* writing tile-level code and already name the load, put it there
instead:

```python
tile = pl.load(weights, [n0, k0], [256, 512], cache=pl.CachePolicy.BYPASS)
```

An explicit `cache=` on a load always wins over the scope declaration, in both
directions — so `cache=pl.CachePolicy.DEFAULT` opts one access back into the
cache inside an otherwise-bypassing scope.

**Cost:** correctness is yours to guarantee. `BYPASS` asserts two things: this
tensor has no reuse worth caching, *and* nothing writes those bytes while the
kernel runs. Mixing a cached write with a bypassing read of the same bytes is a
coherency bug, which is why this is never a default and never inferred. The
compiler rejects the case it can see — declaring `BYPASS` on a tensor the scope
writes is an error — but it cannot prove the general case.

**Current status:** the toolchain has no bypass path yet
([PTOAS#1356](https://github.com/hw-native-sys/PTOAS/issues/1356)). A `BYPASS`
declaration is accepted and carried, warns at compile time, and generates exactly
the same code as an ordinary cached read. Declaring it now costs nothing and
starts working when that lands.

## See also

- [Runtime scopes](../tasks/01-scopes.md) — scopes as a dependency-semantics choice.
- [Tuning the InCore function](04-incore.md) — what consumes the on-chip side.
