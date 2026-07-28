# Performance Tuning: Loop & Pipeline

> **Status:** skeleton — outline only; prose/examples TBD.

## `pl.pipeline`

> Per-loop opt-in. Effect differs for non-mixed vs mixed (cross-core) loops —
> cover **both**.

### Non-mixed (same-core) loops

- Overlaps load/compute/store stages within one core (GM→L1, L1→L0, nested matmul
  stage loops — no `tpush`/`tpop`).
- `LowerPipelineLoops` replicates the stages (`stage=F`).

### Mixed (cross-core C→V→C) loops

- `SkewCrossCorePipeline` skews the cube (AIC) and vector (AIV) halves so the two
  cores overlap instead of stalling on each other.
- Pass is default-on, but only acts on loops you marked `pl.pipeline(stage=F)`
  with `F > 1`.
- `stage` → skew depth `D = max(2, stage - 1)` (depth-2 by default).
- Role-dependent: producer-role single round-trip → real producer skew;
  consumer-role / multi-round-trip / dynamic bounds / trip < 2 → demoted to
  Sequential (overlap then comes from the peer core's skew).

### When mixed pipelining is suboptimal: manual cross-core sync

- `pl.system.sync_set(event_id, pipe=..., core_type="aic"|"aiv")`
- `pl.system.sync_wait(event_id, pipe=..., core_type="aic"|"aiv")`
- Hand-control the Cube/Vector cross-core events when auto-skew doesn't produce
  the desired overlap.

## `pl.unroll`

- Unroll factor; trade-off (code size vs scheduling freedom).

## `pl.parallel`

- Mark independent iterations for parallel execution.

## Verifying the effect

- swimlane → [DFX › Swimlane](../dfx/01-swimlane.md)
