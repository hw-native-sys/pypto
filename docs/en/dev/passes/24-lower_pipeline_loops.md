# LowerPipelineLoops Pass

Lowers `pl.pipeline(N, stage=F)` at the tile level: replicates the loop body `F` times per outer iteration to enable ping-pong buffering, while keeping the outer loop sequential.

## Overview

`pl.unroll(N)` fully expands a loop into `N` body copies at slot #1 (before SSA). Users reach for this not because they want `N` copies but because they need distinct tile MemRefs — `MemoryReuse` would otherwise coalesce sequentially-live tiles into a single buffer, defeating ping-pong execution.

`pl.pipeline(N, stage=F)` is the user-facing surface for that targeted knob: replicate the body `F` times (typically 2–4) at the tile level, leaving an outer loop of `N/F` iterations. Each clone gets fresh def-vars (SSA preserved) and operates on independent tiles, which downstream `MemoryReuse` cannot merge.

Internally, `pl.pipeline(...)` emits `ForStmt(kind=ForKind::Pipeline, attrs={"pipeline_stages": F})`. The bidirectional structural invariant `kind == Pipeline ⇔ pipeline_stages attr present` (verified by `PipelineLoopValid`) holds at every observable IR state — the kind and the attr always travel together. `LowerPipelineLoops` triggers when `F > 1`; it replicates the body and downgrades the attr to `1` as the post-lowering marker, keeping `ForKind::Pipeline` for the downstream `CanonicalizeIOOrder` pass to scope on. Re-running `LowerPipelineLoops` is a no-op (it sees `factor == 1` and skips).

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**Pipeline position**: After [`NormalizeReturnOrder`](23-normalize_return_order.md), before `CanonicalizeIOOrder` and `InitMemRef` (slot 20.5). Late enough that all tile-structural decisions are made; early enough that `CanonicalizeIOOrder` / `InitMemRef` / `MemoryReuse` see distinct tile vars per clone.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerPipelineLoops()` | `passes.lower_pipeline_loops()` | Function-level |

```python
from pypto import passes
result = passes.lower_pipeline_loops()(program)
```

## DSL Syntax

```python
# Replicate the body 4 times per outer iteration; outer loop runs 16 iters with stride 4.
for i in pl.pipeline(64, stage=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)
```

`pl.pipeline` accepts the same positional args as `pl.range` — `(stop)`, `(start, stop)`, or `(start, stop, step)` — plus the required `stage=` kwarg. `stage=` and `chunk=` are mutually exclusive. `init_values=` is supported.

## Behavior

For `ForStmt(kind=Pipeline, attrs={"pipeline_stages": F}, start, stop, step, body)` with `F > 1`:

- **Main loop**: stride `F*step`, body is a `SeqStmts` of `F` clones, kind still `ForKind::Pipeline` (marker), attr downgraded to `1`. The (kind, attr) pair stays together so `PipelineLoopValid` holds and the IR survives print/parse round-trip (renders as `pl.pipeline(..., stage=1)`).
- **Cloning**: each clone uses `DeepClone(body, {loop_var → new_var + k * step}, clone_def_vars=true)`. Fresh def-vars per clone keep SSA intact and give `MemoryReuse` distinct tile identities to work with.

`stage=1` is a no-op trigger: the pass leaves the loop intact (kind and attr stay) and only recurses into the body to lower nested pipelines. `CanonicalizeIOOrder` then scopes on the kept marker, applies the IO reorder, and demotes/strips on exit. The same path handles user-written `pl.pipeline(stage=1)` and the post-lowering output of a `factor>1` invocation — both ask for IO reorder without further replication, so re-running `LowerPipelineLoops` is naturally idempotent.

Two lowering modes — static vs dynamic — differ only in how the main loop's `stop` and the remainder are computed.

### Static bounds — all of `start`, `stop`, `step` are compile-time integers

With trip count `T = (stop - start) / step`:

- Main loop stops at `start + (T // F) * F * step`.
- If `T % F != 0`, a **bare `SeqStmts`** of `T % F` cloned bodies at offsets `start + (T // F) * F * step + j * step` (for `j ∈ [0, T%F)`) is flattened directly into the outer scope. No runtime dispatch and no wrapper are needed — the remainder count is known.
- When the source loop has `iter_args`, trailing `AssignStmt`s bind the outer loop's `return_vars` to the tail's final yielded expressions so downstream references stay valid.

### Dynamic bounds — `start` and/or `stop` are runtime Exprs (`step` still static, positive)

- Compute the total trip count as `trip_iters = ceil_div(stop - start, step)`. When `step == 1` this collapses to `stop - start` and the pass emits the shorter form.
- Let `main_iters = trip_iters / factor` (floor-div) and materialize `main_end = start + main_iters * (factor * step)` as a fresh SSA `AssignStmt` (named `unroll_main_end`).
- Main loop is `for i in range(start, main_end, F*step)`.
- Materialize `rem_iters = trip_iters - main_iters * factor` as a fresh SSA `AssignStmt` (named `unroll_rem`). When `step == 1` this is equivalent to `stop - main_end`, and the pass emits that shorter form. The remainder is dispatched through a cascaded IfStmt chain:

  ```text
  if rem_iters == 1:    <1 clone>
  else if rem_iters == 2: <2 clones>
  else if rem_iters == 3: <3 clones>
  # ...
  else if rem_iters == F-1: <F-1 clones>
  # rem_iters == 0 falls through — no tail work.
  ```

  Each branch body is a bare `SeqStmts` of `k` cloned bodies (followed by a `YieldStmt` when the source loop has `iter_args`). The enclosing `IfStmt` carries `return_vars`; at the outermost level these are the original outer loop's `return_vars`, at inner levels they are fresh vars fed upward via successive `YieldStmt`s. SSA stays clean: each branch is self-contained; no conditionally-defined var escapes its IfStmt.

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| `step` must be a compile-time integer constant | Main loop's stride and per-clone offsets both require `stage * step` as an integer |
| Dynamic bounds require `step > 0` | The dynamic trip-count formula assumes positive step; negative-step ranges must use static bounds |
| `stage=` and `chunk=` are mutually exclusive on `pl.pipeline` | Different optimization axes; combining them adds semantic ambiguity without a clear use case |
| `stage=` is only accepted on `pl.pipeline()` | Scoped feature; `pl.range()` / `pl.parallel()` / `pl.unroll()` have different semantics |

## Examples

### Static — trip count known (`N=10`, `F=4`)

```python
# Before
for i in pl.pipeline(0, 10, 1, stage=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)

# After: main loop covers [0, 8) with kind=Pipeline (marker), attr downgraded to
# stage=1; bare tail clones flattened into the outer scope.
for i in pl.pipeline(0, 8, 4, stage=1):
    tile_x_0 = pl.tile.load(input_a, [i * 128], [128]); pl.tile.store(tile_x_0, [i * 128], output)
    tile_x_1 = pl.tile.load(input_a, [(i + 1) * 128], [128]); pl.tile.store(tile_x_1, [(i + 1) * 128], output)
    tile_x_2 = pl.tile.load(input_a, [(i + 2) * 128], [128]); pl.tile.store(tile_x_2, [(i + 2) * 128], output)
    tile_x_3 = pl.tile.load(input_a, [(i + 3) * 128], [128]); pl.tile.store(tile_x_3, [(i + 3) * 128], output)

tile_x_4 = pl.tile.load(input_a, [8 * 128], [128]); pl.tile.store(tile_x_4, [8 * 128], output)
tile_x_5 = pl.tile.load(input_a, [9 * 128], [128]); pl.tile.store(tile_x_5, [9 * 128], output)
```

### Dynamic — runtime stop `n`

```python
# Before
for i in pl.pipeline(0, n, 1, stage=4):
    tile_x = pl.tile.load(input_a, [i * 128], [128])
    pl.tile.store(tile_x, [i * 128], output)

# After
unroll_main_end: pl.Scalar[pl.INDEX] = ((n - 0) // 4) * 4 + 0
for i in pl.pipeline(0, unroll_main_end, 4, stage=1):  # post-lowering marker
    <4 clones as above>

unroll_rem: pl.Scalar[pl.INDEX] = n - unroll_main_end
if unroll_rem == 1:
    tile_x_t0 = pl.tile.load(input_a, [unroll_main_end * 128], [128])
    pl.tile.store(tile_x_t0, [unroll_main_end * 128], output)
else:
    if unroll_rem == 2:
        <2 clones at offsets unroll_main_end + 0, unroll_main_end + 1>
    else:
        if unroll_rem == 3:
            <3 clones at offsets unroll_main_end + 0, +1, +2>
```

After this pass, `CanonicalizeIOOrder` runs scoped to the pipeline loop's body, clusters loads at the top and stores at the bottom, and demotes the outer loop's `kind_` to `Sequential` — making the cloned input tiles co-live so `MemoryReuse` keeps them in distinct buffers. Ping-pong buffering applies to both the bulk main loop and the tail clones.

## Related

- [`CanonicalizeIOOrder`](25-canonicalize_io_order.md) — the IO-order canonicalization pass that runs next, scoped to pipeline bodies
- [`UnrollLoops`](02-unroll_loops.md) — full-unroll pass at slot #1, kept as the primary `pl.unroll(N)` lowering
