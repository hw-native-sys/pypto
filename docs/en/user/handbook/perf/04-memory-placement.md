# Performance Tuning: Memory Placement

> **Status:** skeleton — outline only; prose/examples TBD.

## Tile-level memory hierarchy & movement engines

- Recap the hierarchy: DDR → Vec / Mat(L1) → Left/Right(L0A/B) → Acc(L0C) / Bias
  (link [Language Guide › Memory](../../02-language_guide.md#memory-and-data-movement)).
- Which hardware engine performs each move (load / move / store paths, e.g.
  MTE load-in, MTE move, FIXPIPE out).
- **Why this matters:** each move is a lane in the in-core swimlane — knowing the
  engine → data-move mapping is how you read a per-kernel swimlane.

## Compiler L1 (Mat) reuse — and its constraints

- Automatic and **compiler-owned** (loop-invariant matmul-operand Mat residency,
  PR #2080) — not a user API.
- Applies only to compiler-generated tensor-level `pl.matmul` operand loads
  (`GM → Mat → Left/Right → matmul`); user-authored
  `tile.load(..., target_memory=Mat)` is **not** hoisted.
- Constraints that must all hold for reuse to trigger:
  - sequential, statically-bounded, non-empty loop; candidate runs unconditionally
  - loop-invariant offsets / shapes / dependencies
  - no ordering boundary (calls, cross-core, sync, cache maintenance, stores,
    intervening control flow)
  - compiler-owned storage root; no aliasing writable root
  - extended lifetime fits backend Mat/L0 capacity
- If any constraint fails → falls back to a per-iteration `GM → Mat` load (no reuse).

## L0 tiling constraints (`AutoTileMatmulL0`)

- Chooses L0 tiles + ping/pong buffering **within a single matmul**.
- Cannot retain operands across an outer loop by itself (why L1 residency is a
  separate pass).
- Capacity / tiling limits — TBD.

## Placement tip: 512B alignment

- L2 read minimum granularity is **512B**.
- Keep contiguous data **512B-aligned** for best L2 read efficiency.

## The one user knob: `target_memory` (optional)

- `pl.load(..., target_memory=pl.Mem.*)` / `pl.move(..., target_memory=...)`
- `pl.Mem`: DDR · Vec · Mat · Left · Right · Acc · Bias
- Only to override / steer inference when the automatic placement is not ideal.

## Verifying the effect

- in-core profiling / swimlane → [Measuring Impact](05-measuring-impact.md)
