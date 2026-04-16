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

For `for i in range(start, stop, step)` with `attrs_["unroll_factor"] = F`:

- **Main loop**: stride `F*step`, body is a `SeqStmts` of `F` clones; the outer loop carries `attrs_["unroll_replicated"] = F`.
- **Cloning**: each clone uses `DeepClone(body, {loop_var → new_var + k * step}, clone_def_vars=true)`. Fresh def-vars per clone keep SSA intact and give `MemoryReuse` distinct tile identities to work with.

Two lowering modes — static vs dynamic — differ only in how the main loop's `stop` and the remainder are computed.

### Static bounds — all of `start`, `stop`, `step` are compile-time integers

With trip count `T = (stop - start) / step`:

- Main loop stops at `start + (T // F) * F * step`.
- If `T % F != 0`, a single **tail branch** is emitted: a trip-1 `ForStmt` tagged `unroll_replicated = T % F`, containing `T % F` cloned bodies at offsets `start + (T // F) * F * step + j * step` for `j ∈ [0, T%F)`. No runtime dispatch is needed — the remainder count is known.

### Dynamic bounds — `start` and/or `stop` are runtime Exprs (`step` still static, positive)

- Compute the total trip count as `trip_iters = ceil_div(stop - start, step)`. When `step == 1` this collapses to `stop - start` and the pass emits the shorter form.
- Let `main_iters = trip_iters / factor` (floor-div) and materialize `main_end = start + main_iters * (factor * step)` as a fresh SSA `AssignStmt` (named `unroll_main_end`).
- Main loop is `for i in range(start, main_end, F*step)`.
- Materialize `rem_iters = trip_iters - main_iters * factor` as a fresh SSA `AssignStmt` (named `unroll_rem`). When `step == 1` this is equivalent to `stop - main_end`, and the pass emits that shorter form. The remainder is dispatched through a cascaded IfStmt chain:

  ```text
  if rem_iters == 1:    <1 clone>                      # outermost
  else if rem_iters == 2: <2 clones marked unroll_replicated=2>
  else if rem_iters == 3: <3 clones marked unroll_replicated=3>
  # ...
  else if rem_iters == F-1: <F-1 clones>
  # rem_iters == 0 falls through — no tail work.
  ```

  Each branch's body is a trip-1 `ForStmt` tagged `unroll_replicated = k`, so `ReorderUnrolledIO` reorders each branch internally the same way it does the main loop. SSA stays clean: each branch is self-contained; no conditionally-defined var escapes its IfStmt.

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| `step` must be a compile-time integer constant | Main loop's stride and per-clone offsets both require `factor * step` as an integer |
| Dynamic bounds require `step > 0` | The dynamic trip-count formula assumes positive step; negative-step ranges must use static bounds |
| `unroll` and `chunk` are mutually exclusive on `pl.range` | Different optimization axes; combining them adds semantic ambiguity without a clear use case |
| `unroll=` only on `pl.range()` | Scoped feature; `pl.parallel()` / `pl.unroll()` have different semantics |

### Loop-carried state (`iter_args`)

`iter_args` / `init_values` are supported. Loop-carried state threads sequentially through the `F` replicated clones: clone `k` consumes the previous clone's yielded expressions as its iter-arg substitutes, and only the last clone's `YieldStmt` is kept to feed the next outer iteration. When a tail follows, the main loop's `return_vars` become fresh SSA names that seed the tail's `iter_args`; the tail inherits the original outer loop's `return_vars` so downstream references stay valid.

Under dynamic bounds, every `IfStmt` in the cascade carries `return_vars` matching the iter-arg types and every branch ends with a `YieldStmt`. The innermost `else` yields the main-loop `return_vars` unchanged — that is the `rem == 0` no-op fall-through.

## Examples

### Static — trip count known (`N=10`, `F=4`)

```python
# Before
for i in pl.range(0, 10, 1, attrs={"unroll_factor": 4}):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)

# After: main loop covers [0, 8), single tail branch for the leftover 2 iters
for i in pl.range(0, 8, 4, attrs={"unroll_replicated": 4}):
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128]); pl.tile.store(tile_x_0, [i * 128], output)
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128]); pl.tile.store(tile_x_1, [(i + 1) * 128], output)
    tile_x_2 = pl.tile.load(input_a, [(i + 2) * 128], [128]); pl.tile.store(tile_x_2, [(i + 2) * 128], output)
    tile_x_3 = pl.tile.load(input_a, [(i + 3) * 128], [128]); pl.tile.store(tile_x_3, [(i + 3) * 128], output)

for _tail_iter_2 in pl.range(0, 1, 1, attrs={"unroll_replicated": 2}):
    tile_x_4 = pl.tile.load(input_a, [8 * 128], [128]); pl.tile.store(tile_x_4, [8 * 128], output)
    tile_x_5 = pl.tile.load(input_a, [9 * 128], [128]); pl.tile.store(tile_x_5, [9 * 128], output)
```

### Static with `iter_args` — loop-carried accumulator (`N=10`, `F=4`)

```python
# Before
for i, (a,) in pl.range(0, 10, 1, init_values=(s0,), attrs={"unroll_factor": 4}):
    b = a + i
    r = pl.yield_(b)

# After: main loop threads a → b → b_1 → b_2 → b_3, then tail seeds a_tail from r_main
for i, (a,) in pl.range(0, 8, 4, init_values=(s0,), attrs={"unroll_replicated": 4}):
    b = a + i
    b_1 = b + (i + 1)
    b_2 = b_1 + (i + 2)
    b_3 = b_2 + (i + 3)
    r_main = pl.yield_(b_3)

for _tail_iter_2, (a,) in pl.range(0, 1, 1, init_values=(r_main,), attrs={"unroll_replicated": 2}):
    b_4 = a + 8
    b_5 = b_4 + 9
    r = pl.yield_(b_5)
```

### Dynamic — runtime stop `n`

```python
# Before
for i in pl.range(0, n, 1, attrs={"unroll_factor": 4}):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)

# After
unroll_main_end: pl.Scalar[pl.INDEX] = ((n - 0) // 4) * 4 + 0
for i in pl.range(0, unroll_main_end, 4, attrs={"unroll_replicated": 4}):
    <4 clones as above>

unroll_rem: pl.Scalar[pl.INDEX] = n - unroll_main_end
if unroll_rem == 1:
    for _tail_iter_1 in pl.range(0, 1, 1, attrs={"unroll_replicated": 1}):
        tile_x_t0 = pl.tile.load(input_a, [unroll_main_end * 128], [128])
        pl.tile.store(tile_x_t0, [unroll_main_end * 128], output)
else:
    if unroll_rem == 2:
        for _tail_iter_2 in pl.range(0, 1, 1, attrs={"unroll_replicated": 2}):
            <2 clones at offsets unroll_main_end + 0, unroll_main_end + 1>
    else:
        if unroll_rem == 3:
            for _tail_iter_3 in pl.range(0, 1, 1, attrs={"unroll_replicated": 3}):
                <3 clones at offsets unroll_main_end + 0, +1, +2>
```

Every main-loop iteration AND every tail branch carries the `unroll_replicated` marker, so `ReorderUnrolledIO` uniformly clusters loads at the top, stores at the bottom — making the cloned input tiles co-live so `MemoryReuse` keeps them in distinct buffers. Ping-pong buffering works for both the bulk (main) and the tail.

## Related

- [`ReorderUnrolledIO`](21-reorder_unrolled_io.md) — consumes the `unroll_replicated` marker
- [`UnrollLoops`](01-unroll_loops.md) — full-unroll pass at slot #1, kept as the primary `pl.unroll(N)` lowering
- RFC #1025 — design document
