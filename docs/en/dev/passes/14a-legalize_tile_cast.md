# LegalizeTileCast Pass

Expands `tile.cast` `(src, dst)` pairs that the active `pto.tcvt` profile cannot emit as a single instruction into the shortest chain of native casts.

## Overview

For each `var = tile.cast(...)`:

1. Look up the backend-specific adjacency table (from pto-isa `tcvt` Supported Conversions for A5 / A2A3).
2. Already native: leave unchanged (including FIXPIPE-foldable `FP32→BF16/FP16` with `mode=rint`).
3. Non-native: BFS for a shortest path; among equal-length paths prefer "same byte-width → float, then adjust width".

Typical A5 results: `INT32→FP16` → `INT32→FP32→FP16`; `FP16→BF16` → `FP16→FP32→BF16`.

Unreachable pairs hard-fail with src/dst/arch in the diagnostic.

**Requires / Produces / Invalidates**: none (empty `PassProperties`).

## When it runs

Default pipeline:

```text
lower_composite_ops → flatten_tile_nd_to_2d → legalize_tile_cast → auto_tile_matmul_l0
```

## API

| C++ | Python |
| --- | ------ |
| `pass::LegalizeTileCast()` | `passes.legalize_tile_cast()` |
