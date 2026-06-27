# CanonicalizeIOOrder Pass

Scoped to `SeqStmts` **inside a `ForKind::Pipeline` body**, reorders statements along a **same-core hardware-unit stage ladder** (scalar → load → compute → store) — subject to the SSA dependency graph. Clustering the replicated clones' loads up front issues both stages' prefetches before the computes, so the MTE load engine runs ahead of the Vector/Cube compute engine (double-buffer overlap); the compute/store tier is then ordered by pipeline **stage** so each stage stores its output right after its compute. Note: buffer *separation* between stages is no longer a side effect of this clustering — it is an explicit [`MemoryReuse`](32-memory_reuse.md) constraint (`pipeline_membership`); this pass only shapes the **schedule**. Cross-core (cube/vector) pipelines are software-pipelined upstream by [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md) and reach this pass as `ForKind::Sequential`, so there is no cross-core handling here. Loops that are not pipelined are left untouched.

## Overview

After `LowerPipelineLoops` produces an outer `ForStmt` (kind=Pipeline marker) whose body is a `SeqStmts` of `F` cloned bodies, the natural emission order is `[scalar_0, load_0, compute_0, store_0, scalar_1, load_1, compute_1, store_1, …]` (each clone's address arithmetic precedes its own load). In that layout each clone's load is issued only after the previous clone's store, so the MTE load engine cannot run ahead of the compute engine — no prefetch overlap.

This pass reorders `SeqStmts` **inside a `ForKind::Pipeline` body** (including nested `IfStmt` branch bodies inside the pipeline scope) so:

- Each scalar-producing compute (typically address arithmetic) floats to the earliest position the dependency graph permits, so it unblocks downstream loads.
- Each `tile.load` / `tile.read` floats to the earliest position the dependency graph permits.
- Tile compute statements settle in the middle.
- Each `tile.store` / `tile.write` sinks to the latest position the dependency graph permits.

The result is `[scalars…, loads…, per-stage (compute, store)…]` whenever the dataflow allows — e.g. a `stage=2` body emits `load load compute_s0 store_s0 compute_s1 store_s1`. Clustering the loads gives prefetch overlap (the MTE engine runs ahead); ordering the compute/store tier by stage means each stage's output is stored right after its compute, freeing that buffer before the next stage and cutting both on-chip pressure and the cross-iteration load↔store coupling. Buffer separation between stages is enforced separately by `MemoryReuse` (see [32-memory_reuse.md](32-memory_reuse.md)).

Lifting scalar compute is what unlocks the load cluster: without it, each clone's address-arithmetic assign would be classified as ordinary compute and rank by original position — interleaving between sibling loads and pinning them in their original groups. With scalar compute as the highest-priority category, all sibling clones' address arithmetic emits first, all dependent loads become ready together, and the loads naturally cluster.

### Cross-core (AIC↔AIV) — handled upstream

Cross-core (cube/vector) pipeline loops are software-pipelined by [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md), which runs *before* `LowerPipelineLoops` and rewrites every cross-core loop to `ForKind::Sequential`. They therefore never reach this pass as a `ForKind::Pipeline` body, and `CanonicalizeIOOrder` has **no cross-core handling** — `tpush`/`tpop` are ordinary tile compute here, not reordered into any cross-core tier. This pass only clusters the **same-core** stages (scalar → load → compute → store) of the remaining same-core pipeline loops (GM→L1, L1→L0, nested matmul) for ping-pong.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure.

**Pipeline position**: After `LowerPipelineLoops`, before `InitMemRef` (slot 20.6). Running before `InitMemRef` keeps SSAForm intact for the dependency analysis. On exit the pass demotes the outer pipeline loop's `kind_` from `ForKind::Pipeline` → `ForKind::Sequential` and strips any stale `pipeline_stages` attr — `ForKind::Pipeline` is a transient marker that must not survive past this pass.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::CanonicalizeIOOrder()` | `passes.canonicalize_io_order()` | Program-level |

```python
from pypto import passes
result = passes.canonicalize_io_order()(program)
```

## Algorithm

A priority-aware stable topological sort applied to every `SeqStmts` of two or more statements **inside a `ForKind::Pipeline` body**. The mutator maintains a pipeline-depth counter: it increments on entry to a `ForKind::Pipeline` loop, decrements on exit, and reorders `SeqStmts` only when the counter is non-zero. Each top-level statement is categorized:

| Category | Priority | Hardware unit | Examples |
| -------- | -------- | ------------- | -------- |
| `ScalarCompute` | 0 (emit first) | scalar | `AssignStmt` whose LHS is a `ScalarType` (e.g. `off = i * 64`) |
| `Load` | 1 | MTE ingress (GM→L1/L0) | `AssignStmt(_, Call("tile.load", …))` / `tile.read` / L1→L0 `tile.extract` |
| `TileCompute` | 2 | CUBE/Vec compute | Everything else (matmul loops, elementwise, `tile.move`, `tpush`/`tpop` — see note) |
| `Store` | 3 (emit last) | MTE egress (L1/L0→GM) | `tile.store` / `tile.write` (AssignStmt or EvalStmt) |

`tile.read` is classified as `Load` even though it produces a scalar — it's I/O against a tile and belongs in the load tier alongside `tile.load`. The LHS-type check only applies once the RHS is determined not to be a recognized I/O op.

Cross-core `tpush`/`tpop` carry no special category — they fall through to `TileCompute` and keep their program order among siblings (cross-core software-pipelining is done upstream by [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md); see *Cross-core (AIC↔AIV)* above).

At each step, among statements whose predecessors are all already emitted (`ready`), the pass emits the one with the smallest `(tier, stage, sub, original_index)` — where `tier` is 0 for scalar compute, 1 for load, 2 for tile-compute/store; `stage` is the statement's `pipeline_membership` (so a tile def, and the store that consumes it, share a stage); and `sub` is 0 for compute, 1 for store. Loads (tier 1) thus cluster before all compute/store, and within the compute/store tier each stage's compute precedes its store before the next stage begins. Non-pipeline regions carry no membership (`stage` empty), so the tier/sub ordering reduces to the prior scalar → load → compute → store ladder.

Worked example — input `[scalar_0, load_0, compute_0, store_0, scalar_1, load_1, compute_1, store_1]` with each clone's load reading its scalar, each compute reading its load, each store reading both its scalar and compute:

```text
ready={scalar_0, scalar_1}              emit scalar_0    (cat 0, idx 0)
ready={load_0, scalar_1}                emit scalar_1    (cat 0 < cat 1)
ready={load_0, load_1}                  emit load_0      (cat 1, idx 1 < 5)
ready={load_1, compute_0}               emit load_1      (cat 1 < cat 2)
ready={compute_0, compute_1}            emit compute_0
ready={compute_1, store_0}              emit compute_1   (cat 2 < cat 6)
ready={store_0, store_1}                emit store_0
ready={store_1}                         emit store_1
```

Output: `[scalar_0, scalar_1, load_0, load_1, compute_0, compute_1, store_0, store_1]`.

## Correctness

The reorder is a topological sort over the SSA def-use dependency graph, so it preserves all dataflow. Soundness rests on two utilities from `stmt_dependency_analysis.h`:

1. `CollectInOutUseDisciplineDiagnostics(region, program)` — reports any user-function call that passes a variable as `InOut`/`Out` while a later statement still reads it. Since PR #1039 this is a structural IR invariant (RFC #1026): every function in valid IR satisfies it. The pass runs this check once per function — not per `SeqStmts`, since variable scopes don't cross function boundaries — and skips reordering for any function that reports a violation (to stay sound even under `VerificationLevel.NONE`).
2. `BuildStmtDependencyGraph(region, program)` — produces a sound def-use DAG over the region's top-level statements, given the discipline holds. The pass passes `nullptr` for `program` since the discipline check has already been performed at function scope.

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| Function must satisfy the InOut-use discipline | Required for sound dataflow analysis (structural invariant since PR #1039); per-function check skips reordering otherwise |
| Aborts on cyclic dependency graph | Should be impossible for an SSA region; raised as `INTERNAL_CHECK` |

## Example

**Before** (input from `LowerPipelineLoops` — note the outer loop still carries the `kind=Pipeline` marker, and the per-clone scalar address-arithmetic assigns):

```python
for i in pl.pipeline(0, 8, 4, stage=1):  # kind=Pipeline (marker); attr=1 post-LowerPipelineLoops
    off_0: pl.Scalar[pl.INDEX] = i * 128
    tile_x_0 = pl.tile.load(input_a, [off_0], [128])
    tile_y_0 = pl.tile.add(tile_x_0, 1.0)
    pl.tile.store(tile_y_0, [off_0], output)
    off_1: pl.Scalar[pl.INDEX] = (i + 1) * 128
    tile_x_1 = pl.tile.load(input_a, [off_1], [128])
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    pl.tile.store(tile_y_1, [off_1], output)
    # ... k=2, k=3 ...
```

**After** (kind demoted to Sequential; body reordered):

```python
for i in pl.range(0, 8, 4):  # kind=Sequential
    off_0: pl.Scalar[pl.INDEX] = i * 128
    off_1: pl.Scalar[pl.INDEX] = (i + 1) * 128
    off_2: pl.Scalar[pl.INDEX] = (i + 2) * 128
    off_3: pl.Scalar[pl.INDEX] = (i + 3) * 128
    tile_x_0 = pl.tile.load(input_a, [off_0], [128])
    tile_x_1 = pl.tile.load(input_a, [off_1], [128])
    tile_x_2 = pl.tile.load(input_a, [off_2], [128])
    tile_x_3 = pl.tile.load(input_a, [off_3], [128])
    tile_y_0 = pl.tile.add(tile_x_0, 1.0)
    pl.tile.store(tile_y_0, [off_0], output)
    tile_y_1 = pl.tile.add(tile_x_1, 1.0)
    pl.tile.store(tile_y_1, [off_1], output)
    tile_y_2 = pl.tile.add(tile_x_2, 1.0)
    pl.tile.store(tile_y_2, [off_2], output)
    tile_y_3 = pl.tile.add(tile_x_3, 1.0)
    pl.tile.store(tile_y_3, [off_3], output)
```

All four `off_k` lift first to unblock the loads, which then cluster (prefetch overlap — the MTE engine runs ahead of compute). The compute/store tier is ordered by stage, so each `tile_y_k` is stored right after its compute, freeing that output buffer before the next stage. The buffers' stage *separation* (each clone keeping a distinct MemRef) is enforced by `MemoryReuse` via `pipeline_membership`, not by this ordering.

## Related

- [`LowerPipelineLoops`](28-lower_pipeline_loops.md) — upstream producer of replicated regions that benefit from this pass; leaves `ForKind::Pipeline` as the scope marker this pass consumes
- [`MaterializeTensorStrides`](30-materialize_tensor_strides.md) — runs immediately after this pass (when inserted into the default pipeline); fills implicit `TensorView` strides before `InitMemRef` consumes them
- [`MemoryReuse`](32-memory_reuse.md) — runs after this pass; enforces stage buffer separation explicitly via `pipeline_membership` (this pass only shapes the schedule)
- RFC #1026 / PR #1029 — InOut-use discipline + dependency analysis utility
