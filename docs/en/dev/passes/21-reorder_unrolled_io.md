# ReorderUnrolledIO Pass

Inside each `unroll_replicated` region, lifts `tile.load` to the top and sinks `tile.store` to the bottom — subject to the SSA dependency graph — to enable symmetric ping-pong buffering.

## Overview

After `PartialUnrollTileLoops` produces an outer `ForStmt` whose body is a `SeqStmts` of `F` cloned bodies, the natural emission order is `[load_0, compute_0, store_0, load_1, compute_1, store_1, …]`. With this layout, sibling clones' tile live ranges are sequential — `MemoryReuse` happily coalesces them into a single buffer, defeating ping-pong.

This pass reorders each marked `SeqStmts` so:

- Each `tile.load` floats to the earliest position the dependency graph permits.
- Each `tile.store` sinks to the latest position the dependency graph permits.
- Compute statements settle in the middle.

The result is `[loads…, compute…, stores…]` whenever the dataflow allows. Sibling clones' input tiles are co-live near the top, output tiles co-live near the bottom — `MemoryReuse` cannot coalesce them, so each clone keeps its own MemRef and ping-pong buffering becomes possible.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**Pipeline position**: After `PartialUnrollTileLoops`, before `InitMemRef` (slot 20.6). Running before `InitMemRef` keeps SSAForm intact for the dependency analysis.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::ReorderUnrolledIO()` | `passes.reorder_unrolled_io()` | Program-level |

```python
from pypto import passes
result = passes.reorder_unrolled_io()(program)
```

## Algorithm

A priority-aware stable topological sort. Each top-level statement of the marked `SeqStmts` is categorized:

| Category | Priority | Examples |
| -------- | -------- | -------- |
| `Load` | 0 (emit first) | `AssignStmt(tile, Call("tile.load", …))` |
| `Compute` | 1 | Anything else inside the region |
| `Store` | 2 (emit last) | `AssignStmt(_, Call("tile.store", …))` or `EvalStmt(Call("tile.store", …))` |

At each step:

- Among statements whose predecessors are all already emitted (`ready`):
  - If any non-store is ready → emit the smallest `(category, original_index)`. Loads (cat 0) win over compute (cat 1).
  - Otherwise → emit the smallest-index store.

Stores are deferred because they are only chosen when no load or compute is available — combined with the load-priority rule, the pass produces `[loads…, compute…, stores…]` whenever the dataflow allows.

Worked example — input `[load_0, compute_0, store_0, load_1, compute_1, store_1]` with each clone's compute reading its load and each store reading its compute:

```text
ready={load_0, load_1}        non-store ready → emit load_0
ready={load_1, compute_0}     non-store ready → emit load_1   (cat 0 < cat 1)
ready={compute_0, compute_1}  non-store ready → emit compute_0
ready={compute_1, store_0}    non-store ready → emit compute_1
ready={store_0, store_1}      no non-store    → emit store_0
ready={store_1}               no non-store    → emit store_1
```

Output: `[load_0, load_1, compute_0, compute_1, store_0, store_1]`.

## Correctness

The reorder is a topological sort over the SSA def-use dependency graph, so it preserves all dataflow. Soundness rests on two utilities from `stmt_dependency_analysis.h`:

1. `CheckInOutUseDiscipline(region, program)` — aborts compilation if any user-function call passes a variable as `InOut`/`Out` and a later statement reads that same variable. The discipline (RFC #1026) guarantees that physical-memory mutations are mirrored by SSA version changes, so SSA def-use captures all real dependencies.
2. `BuildStmtDependencyGraph(region, program)` — produces a sound def-use DAG over the region's top-level statements, given the discipline holds.

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| Operates only inside `ForStmt` carrying `attrs_["unroll_replicated"]` | The pass would otherwise reorder unrelated SeqStmts in unintended ways |
| Region must satisfy the InOut-use discipline | Required for sound dataflow analysis (see RFC #1026) |
| Aborts on cyclic dependency graph | Should be impossible for an SSA region; raised as `INTERNAL_CHECK` |

## Example

**Before** (input from `PartialUnrollTileLoops`):

```python
for i in pl.range(0, 8, 4, attrs={"unroll_replicated": 4}):
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128])
    tile_y_0 = pl.tile.add(tile_x_0, 1.0)
    pl.tile.store(tile_y_0, [i * 128], output)
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128])
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    pl.tile.store(tile_y_1, [(i + 1) * 128], output)
    # ... k=2, k=3 ...
```

**After**:

```python
for i in pl.range(0, 8, 4, attrs={"unroll_replicated": 4}):
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128])
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128])
    tile_x_2 = pl.tile.load(input_a, [(i + 2) * 128], [128])
    tile_x_3 = pl.tile.load(input_a, [(i + 3) * 128], [128])
    tile_y_0 = pl.tile.add(tile_x_0, 1.0)
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    tile_y_2 = pl.tile.add(tile_x_2, 1.0)
    tile_y_3 = pl.tile.add(tile_x_3, 1.0)
    pl.tile.store(tile_y_0, [i * 128], output)
    pl.tile.store(tile_y_1, [(i + 1) * 128], output)
    pl.tile.store(tile_y_2, [(i + 2) * 128], output)
    pl.tile.store(tile_y_3, [(i + 3) * 128], output)
```

All four `tile_x_k` are now co-live up to the last load, and all four `tile_y_k` are co-live up to the first store. `MemoryReuse` (running next) cannot merge them — each gets a distinct MemRef.

## Related

- [`PartialUnrollTileLoops`](20-partial_unroll_tile_loops.md) — produces the `unroll_replicated` marker this pass consumes
- [`MemoryReuse`](16-memory_reuse.md) — runs after this pass; benefits from the co-live tiles
- RFC #1025 — design document
- RFC #1026 / PR #1029 — InOut-use discipline + dependency analysis utility
