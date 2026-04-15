# PartialUnrollTileLoops Pass

Lowers `pl.range(N, unroll=F)` at the tile level: replicates the loop body `F` times per outer iteration to enable ping-pong buffering, while keeping the outer loop sequential.

## Overview

`pl.unroll(N)` fully expands a loop into `N` body copies at slot #1 (before SSA). Users reach for this not because they want `N` copies but because they need distinct tile MemRefs — `MemoryReuse` would otherwise coalesce sequentially-live tiles into a single buffer, defeating ping-pong execution.

`PartialUnrollTileLoops` provides the targeted knob: replicate the body `F` times (typically 2–4) at the tile level, leaving an outer loop of `N/F` iterations. Each clone gets fresh def-vars (SSA preserved) and operates on independent tiles, which downstream `MemoryReuse` cannot merge.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**Pipeline position**: After `NormalizeReturnOrder`, before `InitMemRef` (slot 20.5). Late enough that all tile-structural decisions are made; early enough that `InitMemRef`/`MemoryReuse` see distinct tile vars per clone.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::PartialUnrollTileLoops()` | `passes.partial_unroll_tile_loops()` | Function-level |

```python
from pypto import passes
result = passes.partial_unroll_tile_loops()(program)
```

## DSL Syntax

```python
# Replicate the body 4 times per outer iteration; outer loop runs 16 iters with stride 4.
for i in pl.range(64, unroll=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)
```

## Behavior

For `for i in range(start, stop, step)` with `attrs_["unroll_factor"] = F` and trip count `T = (stop - start) / step`:

- **Clean divide** (`T % F == 0`): one outer loop of `T/F` iterations, body is a `SeqStmts` of `F` clones; the outer loop carries `attrs_["unroll_replicated"] = F`.
- **Remainder** (`T % F != 0`): outer replicated loop covers `(T // F) * F` iterations; a trailing remainder loop covers the leftover `T % F` iterations with the original stride. The remainder loop carries no marker.
- **Cloning**: each clone uses `DeepClone(body, {loop_var → new_var + k * step}, clone_def_vars=true)`. Fresh def-vars per clone keep SSA intact and give `MemoryReuse` distinct tile identities to work with.

## Constraints (first cut)

| Constraint | Reason |
| ---------- | ------ |
| `start`, `stop`, `step` must be compile-time integer constants | Trip count needed to size the main/remainder split |
| `iter_args` / `init_values` not allowed | Loop-carried state across replicated copies needs SSA-aware renaming not yet implemented |
| `unroll` and `chunk` are mutually exclusive on `pl.range` | Different optimization axes; combining them adds semantic ambiguity without a clear use case |

## Example

**Before** (input IR with `unroll_factor=4` attr on the ForStmt):

```python
for i in pl.range(0, 8, 1, attrs={"unroll_factor": 4}):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)
```

**After**:

```python
for i in pl.range(0, 8, 4, attrs={"unroll_replicated": 4}):
    # k=0 clone
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x_0, [i * 128], output)
    # k=1 clone
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128])
    pl.tile.store(tile_x_1, [(i + 1) * 128], output)
    # k=2, k=3 clones similarly
```

The downstream `ReorderUnrolledIO` pass picks up `unroll_replicated`-tagged loops and clusters loads at the top, stores at the bottom — making the cloned input tiles co-live so `MemoryReuse` keeps them in distinct buffers.

## Related

- [`ReorderUnrolledIO`](21-reorder_unrolled_io.md) — consumes the `unroll_replicated` marker
- [`UnrollLoops`](01-unroll_loops.md) — full-unroll pass at slot #1, kept as the primary `pl.unroll(N)` lowering
- RFC #1025 — design document
