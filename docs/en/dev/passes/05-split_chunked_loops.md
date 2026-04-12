# SplitChunkedLoops Pass

Splits loops with `chunk` into nested outer/inner loops under one of two policies.

## Overview

This pass transforms a for loop created with `chunk=C` into a pair of nested loops: an outer loop over chunk indices and an inner loop iterating within each chunk. Two codegen policies are supported:

- **`guarded`** (default) — emit a single outer loop of `ceil(T/C)` chunks plus an inner loop of `C`, and wrap the body in `if (idx < stop)` (or `idx > stop` for negative step). Out-of-range iterations become no-ops. A single kernel is emitted.
- **`leading_full`** — emit a full-chunk loop of `T/C` chunks plus a separate remainder loop of `T % C` iterations. Two sibling loops are emitted.

Both policies run after SSA conversion and propagate `iter_args` through the generated loops.

**Requires**: `TypeChecked`, `SSAForm`.

**When to use**: Runs automatically in the default pipeline after `FlattenCallExpr` and before `InterchangeChunkLoops`. Use `chunk=` on `pl.range()`, `pl.parallel()`, or `pl.unroll()` inside a `with pl.auto_incore():` scope. Chunked loops outside `auto_incore` are not split.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::SplitChunkedLoops()` | `passes.split_chunked_loops()` | Function-level |

```python
from pypto import passes
result = passes.split_chunked_loops()(program)
```

## DSL Syntax

Chunked loops must be wrapped in `with pl.auto_incore():`:

```python
with pl.auto_incore():
    # Default (guarded): single kernel with if-guard
    for i in pl.range(10, chunk=5):
        x = pl.add(x, 1.0)

    # Explicit guarded (same as default)
    for i in pl.parallel(n, chunk=4, chunk_policy="guarded"):
        x = pl.add(x, 1.0)

    # Explicit leading_full: peels remainder into separate loop
    for i in pl.range(7, chunk=5, chunk_policy="leading_full"):
        x = pl.add(x, 1.0)

    # iter_args are supported under both policies
    for i, (s,) in pl.range(10, init_values=(x,), chunk=5):
        s = pl.add(s, 1.0)
        s = pl.yield_(s)
```

## Choosing a Policy

| Criterion | Prefer `guarded` | Prefer `leading_full` |
| --------- | ---------------- | --------------------- |
| Dynamic bound (`stop` not a compile-time constant) | ✅ — single kernel preserves loop-carried state across the boundary | ❌ — remainder kernel receives iter_args as input-only copies, breaking cross-iteration accumulation |
| Static bound, trip_count known divisible | Slightly redundant guard | ✅ — no guard, no remainder |
| Want minimum kernel count under `pl.auto_incore()` | ✅ | Produces 2 kernels per chunked loop |
| Want to eliminate masked iterations inside the hot loop | ❌ | ✅ — full chunks run unconditionally |

`guarded` is the default because (1) it preserves `add_inout()` accumulation under dynamic bounds and (2) it avoids doubling the kernel count under `pl.auto_incore()`.

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| `step`, `chunk` must be integer constants | Needed at compile time |
| `chunk` must be a positive integer | Non-positive sizes are invalid |
| `step` may be negative (descending loop) | `guarded` adapts the predicate to the step sign |
| `start`, `stop` may be dynamic expressions under `guarded` | Trip count becomes `max(abs(stop - start), 0) / abs(step)` |
| Chunked loop must be inside `pl.auto_incore()` | Only `auto_incore`-scoped loops are split |
| `chunk` may be combined with `init_values` | Both policies thread iter_args through the generated loops |

## Algorithm

Let `T = ceil(max(|stop - start|, 0) / |step|)` and `C = chunk`.

### `guarded` (default)

1. `n_total = ceil(T / C)` — static when bounds are const, otherwise `(T + C - 1) // C`.
2. Emit outer loop `for out_var in [0, n_total)` and inner loop `for in_var in [0, C)`.
3. Compute `idx = start + (out_var * C + in_var) * step` (substituted into body).
4. Wrap the visited body in an `IfStmt` whose condition is:
   - `idx < stop` when `step > 0`
   - `idx > stop` when `step < 0`
5. **Without iter_args** — IfStmt has no else branch; skipped iterations are no-ops.
6. **With iter_args** — IfStmt gets `return_vars` acting as phi nodes: the then-branch keeps the user body's trailing `YieldStmt` (updated values), the else-branch yields the inner iter_args unchanged. The inner loop's trailing `YieldStmt` references the IfStmt's phi vars, so loop-carried state threads through both guarded and skipped iterations.

### `leading_full`

1. `n_full = T // C`, `n_rem = T % C`.
2. Emit outer loop `for out_var in [0, n_full)` and inner loop `for in_var in [0, C)` with `idx = start + (out_var * C + in_var) * step`. Skip if `n_full == 0`.
3. If `n_rem > 0`, emit a remainder loop `for rem_var in [0, n_rem)` with `idx = start + (n_full * C + rem_var) * step`. Its `init_values` chain from the outer loop's `return_vars` (or from the original init if no full-chunk loop was emitted).
4. Remap the original `return_vars` to the final loop's `return_vars`.

Both paths preserve the original `ForKind` (Sequential, Parallel, or Unroll) on inner and outer/remainder loops.

## Auto-Name Abbreviations

Printed IR uses the compact auto-name format `base__qualifier_role_vN`. Abbreviated qualifiers:

| Abbreviation | Meaning | Emitted by |
| ------------ | ------- | ---------- |
| `co` | chunk_outer | both policies |
| `ci` | chunk_inner | both policies |
| `cr` | chunk_rem (remainder) | `leading_full` only |
| `cg` | chunk_guard (IfStmt phi) | `guarded` with iter_args only |

Examples: `i__co_idx_v0` (outer index), `x__ci_iter_v1` (inner iter_arg), `x__cr_rv_v1` (remainder return var), `x__cg_rv_v1` (IfStmt phi var).

## Examples

### `guarded`, divisible (`chunk=5`, trip_count=10)

**After**:

```python
for i__co_idx_v0, (x__co_iter_v1,) in pl.range(2, init_values=(x__ssa_v0,)):
    for i__ci_idx_v0, (x__ci_iter_v1,) in pl.range(5, init_values=(x__co_iter_v1,)):
        if i__co_idx_v0 * 5 + i__ci_idx_v0 < 10:
            x__ssa_v3 = pl.tensor.add(x__ci_iter_v1, 1.0)
            x__cg_rv_v1 = pl.yield_(x__ssa_v3)
        else:
            x__cg_rv_v1 = pl.yield_(x__ci_iter_v1)
        x__ci_rv_v1 = pl.yield_(x__cg_rv_v1)
    x__co_rv_v1 = pl.yield_(x__ci_rv_v1)
return x__co_rv_v1
```

### `guarded`, dynamic bound (`chunk=4`, `stop=n`)

**After** (single kernel, `n_total = (n + 3) // 4`):

```python
for i__co_idx_v0, (x__co_iter_v1,) in pl.range((n + 3) // 4, init_values=(x__ssa_v0,)):
    for i__ci_idx_v0, (x__ci_iter_v1,) in pl.range(4, init_values=(x__co_iter_v1,)):
        if i__co_idx_v0 * 4 + i__ci_idx_v0 < n:
            x__ssa_v3 = pl.tensor.add(x__ci_iter_v1, 1.0)
            x__cg_rv_v1 = pl.yield_(x__ssa_v3)
        else:
            x__cg_rv_v1 = pl.yield_(x__ci_iter_v1)
        x__ci_rv_v1 = pl.yield_(x__cg_rv_v1)
    x__co_rv_v1 = pl.yield_(x__ci_rv_v1)
return x__co_rv_v1
```

### `leading_full`, non-divisible (`chunk=5`, trip_count=7)

**After** (two sibling loops):

```python
for i__co_idx_v0, (x__co_iter_v1,) in pl.range(1, init_values=(x__ssa_v0,)):
    for i__ci_idx_v0, (x__ci_iter_v1,) in pl.range(5, init_values=(x__co_iter_v1,)):
        x__ssa_v3 = pl.tensor.add(x__ci_iter_v1, 1.0)
        x__ci_rv_v1 = pl.yield_(x__ssa_v3)
    x__co_rv_v1 = pl.yield_(x__ci_rv_v1)
for i__cr_idx_v0, (x__cr_iter_v1,) in pl.range(2, init_values=(x__co_rv_v1,)):
    x__ssa_v4 = pl.tensor.add(x__cr_iter_v1, 1.0)
    x__cr_rv_v1 = pl.yield_(x__ssa_v4)
return x__cr_rv_v1
```

## LoopOrigin Tagging

| LoopOrigin | Description | Emitted by |
| ---------- | ----------- | ---------- |
| `Original` | Regular user loop (default) | — |
| `ChunkOuter` | Outer loop over chunk indices | both policies |
| `ChunkInner` | Inner loop within a chunk | both policies |
| `ChunkRemainder` | Remainder loop for leftover iterations | `leading_full` only |

Access via `for_stmt.attrs.get("loop_origin")` (Python) or `for_stmt->GetAttr<LoopOrigin>("loop_origin")` (C++).

## Pipeline Position

```text
UnrollLoops → ConvertToSSA → FlattenCallExpr → SplitChunkedLoops → InterchangeChunkLoops → OutlineIncoreScopes → ...
```

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `TypeChecked`, `SSAForm` |
| Produced | `TypeChecked`, `SSAForm` |
| Invalidated | (none) |
