# Performance Tuning: Split & Parallel

> **Status:** skeleton — hierarchy + knob list only; prose/examples TBD.
> Single-chip work distribution. (Multi-**card** → [Distributed Guide](../../distributed/00-guide.md).)

## Mixed kernel (cube + vector split)

> Tensor-level interface only. Cross-core push/pop/free primitives are
> pass-inserted (`ExpandMixedKernel`) — not user-facing.

Knobs to cover:

- `pl.split(mode)` on an InCore scope (`pl.at(..., optimizations=[pl.split(...)])`)
- `pl.SplitMode` — `UP_DOWN` / `LEFT_RIGHT` / `NONE`
- `pl.split_aiv(2, mode=...)`

## On-chip multi-block parallelism (`pl.spmd` family)

> Single-chip, multi-block (SPMD grid dispatch) — not multi-card distribution.

Knobs to cover:

- `pl.spmd` — `with` / `for i in` / `as tid` forms
- `pl.cluster`
- `pl.at`
- `pl.spmd_submit`
- `pl.tile.get_block_idx()` / `get_block_num()`
- `pl.system.syncall(core_type=..., mode="hard"|"soft")` — all-core barrier across
  SPMD blocks (`hard` = full-occupancy FFTS; `soft` = GM-polling, partial-occupancy)

(`deps=` / `allow_early_resolve=` / `predicate=` → [Dependency & Dispatch](03-dependency-dispatch.md).)

## Verifying the effect

- Mixed kernel (cube/vector overlap) → **in-core profiling** — a mixed kernel is
  one task, so the AIC/AIV overlap is internal, *not* visible on the swimlane.
- SPMD multi-block (across-core balance) → **swimlane** →
  [DFX › Swimlane](../dfx/01-swimlane.md)
