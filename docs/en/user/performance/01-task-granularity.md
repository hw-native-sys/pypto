# Task Granularity

Dispatch is not free. Sizing InCore functions so the cores spend their time computing
rather than waiting to be told what to compute.

> **Prerequisites:** [Reading the swimlane](00-swimlane.md).

## The cost you are paying

Every `pl.at` block is one task. The runtime must, for each of them, resolve its
dependencies, place it on a core, write a descriptor, and observe its completion. That work
happens on the AICPU while the AICore waits.

Two components show up on the swimlane:

| Component | Where it shows | Rough scale |
| --------- | -------------- | ----------- |
| Pickup latency | `[dispatch, start]` gap on a core | ~0.8 µs per switch |
| Scheduler not keeping up | Idle core while a *ready, undispatched* task exists | Workload-dependent |

The second is the one that hurts, and it is measurable rather than theoretical. On the
sample the runtime's scheduler-overhead model is documented against — qwen3-14b
`decode_layer`, a2a3, **542 tasks** — the analysis reports AIC idle-with-ready-work at
**15.0%** of the makespan and AIV at **10.3%**. Those are not universal numbers, and the
point is not their size: it is that a workload built from many small tasks can spend a
double-digit share of its wall clock on task administration.

**Symptom:** narrow bars, wide gaps, and `sched_overhead_analysis` reporting a large
`has_overhead`. If your bars are wide and the gaps are thin, this page has nothing for
you — go to [Tuning the InCore function](04-incore.md).

## Growing a task

Three ways, in rough order of how often they apply.

### a. Larger tiling

**When it applies:** the kernel is doing a fixed amount of work per task and the tiles are
small enough that the transfer is inefficient too.

**How:** raise the tile shape the task works on.

```python
# Before — one task per 64x64 tile
with pl.at(level=pl.Level.CORE_GROUP):
    tile_a = pl.load(a, [0, 0], [64, 64])
    tile_b = pl.load(b, [0, 0], [64, 64])
    pl.store(pl.add(tile_a, tile_b), [0, 0], c)

# After — one task covering 4x the elements
with pl.at(level=pl.Level.CORE_GROUP):
    tile_a = pl.load(a, [0, 0], [128, 128])
    tile_b = pl.load(b, [0, 0], [128, 128])
    pl.store(pl.add(tile_a, tile_b), [0, 0], c)
```

**Run it:** `python examples/advanced/04_task_granularity.py --mode larger_tiles` — compare against `--mode many_small_tasks`, which does the same work in four
tasks instead of two.

**Cost:** on-chip buffer footprint, quadratically in a 2D tile. A tile that no longer fits
alongside its co-residents pushes the allocator into either failing or giving up a
pipeline stage — see [Memory](05-memory.md).

**How to confirm:** the swimlane, for wider bars *and* proportionally narrower gaps.
Also check `report/perf_hints.log`: if PH001 was flagging your loads, a wider innermost
dimension should make those lines disappear.

### b. A loop inside the InCore function

**When it applies:** the work is already chunked, and the chunking loop sits *outside* the
`pl.at` block — so each chunk pays a full dispatch.

**How:** move the loop inside. The tile shape stays the same; only the offset moves.

```python
# Before — N tasks, one per chunk
for i in range(ROWS // TILE_ROWS):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        ...

# After — one task, N iterations inside it
with pl.at(level=pl.Level.CORE_GROUP):
    for i in pl.range(ROWS // TILE_ROWS):
        tile_a = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        tile_b = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        pl.store(pl.add(tile_a, tile_b), [i * TILE_ROWS, 0], c)
```

`examples/beginner/02_elementwise.py` (`chunked_add`) is this pattern end to end.

**Run it:** `python examples/advanced/04_task_granularity.py --mode loop_inside` — `--mode many_small_tasks` is the four-dispatch form it replaces.

**Cost:** the chunks are now strictly ordered within one core. If they were independent and
you had cores to spare, you have traded parallelism for dispatch savings — which is the
wrong trade when cores are idle. It also makes the loop a candidate for
[double buffering](04-incore.md), which is usually where the win comes back.

**How to confirm:** the `N` nodes in `deps.json` collapse into one — the task count drops
by `N - 1` — and the swimlane shows one wide bar in place of the staircase.

### c. Merging several InCore functions

**When it applies:** consecutive tasks in the graph are a producer/consumer chain over
data that could have stayed on-chip.

**How:** put the operations in one `pl.at` block, so the intermediate never round-trips
through GM.

```python
# Before — two tasks, and `s` goes out to GM and back
with pl.at(level=pl.Level.CORE_GROUP):
    s = pl.add(pl.load(a, [0, 0], [TR, TC]), pl.load(b, [0, 0], [TR, TC]))
    pl.store(s, [0, 0], scratch)
with pl.at(level=pl.Level.CORE_GROUP):
    pl.store(pl.exp(pl.load(scratch, [0, 0], [TR, TC])), [0, 0], out)

# After — one task, `s` stays on chip
with pl.at(level=pl.Level.CORE_GROUP):
    s = pl.add(pl.load(a, [0, 0], [TR, TC]), pl.load(b, [0, 0], [TR, TC]))
    pl.store(pl.exp(s), [0, 0], out)
```

**Run it:** `python examples/advanced/04_task_granularity.py --mode merged_chain` — `--mode two_tasks_via_gm` is the same chain before merging.

**Cost:** the merged task holds every intermediate live at once.

> **Merging across engines is not this.** Putting a cube op and a vector op in one scope
> additionally needs a split mode — without `pl.split(...)` the buffers do not fit and the
> compiler refuses the scope. That case is [a mixed
> kernel](02-runtime-overhead.md#build-a-mixed-kernel); read it before merging a `matmul`
> with the vector op that consumes it.

**How to confirm:** the merged task disappears from `deps.json` as a separate node, and the
GM traffic for the intermediate disappears from the kernel.

## The other direction: too coarse

Granularity is not monotone. A single card has a fixed number of cores — on Ascend910B,
**48 vector and 24 cube** — and a task occupies one of them.

```text
too many tiny tasks          right                    too few big tasks
├─┤ ├─┤ ├─┤ ├─┤ ├─┤          ├─────┤├─────┤           ├──────────────────┤
 gaps dominate               cores busy               cores 2..47 idle
 → dispatch-bound            → compute-bound          → parallelism-bound
```

If merging drops you below the core count, you have moved the bottleneck rather than
removed it, and the swimlane makes it obvious: bars are wide, gaps are gone, and most core
lanes are simply **empty**.

Note that [SPMD](02-runtime-overhead.md#use-spmd) does not remove this trade-off. Like
`pl.parallel`, it is a way of *describing* the work — one dispatch that fans out across many
blocks — and how much each block does is still your decision. What it changes is the price
of the description: `N` blocks cost one dispatch instead of `N`. The granularity question
stays yours either way.

## Deciding

```text
Wide gaps between narrow bars?
├─ Cores mostly idle, few tasks         → tasks too coarse: split, or use SPMD
├─ Cores busy, gaps between every bar   → tasks too fine: grow via a, b, or c
└─ Gaps only at specific points         → not granularity: see 03-dependencies
```

## See also

- [Runtime overhead](02-runtime-overhead.md) — reducing per-task cost instead of task count.
- [Tuning the InCore function](04-incore.md) — making the bar itself shorter.
