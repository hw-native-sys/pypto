# SplitChunkedLoops Pass

Splits loops with `chunk` into nested outer/inner loops for chunked iteration.

## Overview

This pass transforms for loops created with `chunk=C` into nested loops: an outer loop over chunk indices and an inner loop within each chunk. When the trip count is not divisible by the chunk size, a remainder loop is appended. Runs after SSA conversion and propagates `iter_args` through the generated nested loops.

**Requires**: TypeChecked, SSAForm properties.

**When to use**: Runs automatically in the default pipeline after `FlattenCallExpr` and before `InterchangeChunkLoops`. Use `chunk=` on `pl.range()`, `pl.parallel()`, or `pl.unroll()` inside a `with pl.auto_incore():` scope to split a loop into chunks. Chunked loops outside `auto_incore` are not split.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::SplitChunkedLoops()` | `passes.split_chunked_loops()` | Function-level |

**Python usage**:

```python
from pypto import passes

result = passes.split_chunked_loops()(program)
```

## DSL Syntax

Chunked loops must be wrapped in `with pl.auto_incore():` to be split:

```python
with pl.auto_incore():
    # Chunked sequential loop: 10 iterations in chunks of 5
    for i in pl.range(10, chunk=5):
        x = pl.add(x, 1.0)

    # Chunked parallel loop: inner loop is parallel, outer is sequential
    for i in pl.parallel(8, chunk=4):
        x = pl.add(x, 1.0)

    # Chunked unroll loop: inner loop is unrolled, outer is sequential
    for i in pl.unroll(12, chunk=4):
        x = pl.add(x, 1.0)
```

Chunked loops outside `auto_incore` are rejected earlier by the DSL parser, so this pass only sees valid chunked loops that are already inside `auto_incore`.

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| `start`, `stop`, `step`, `chunk` must be integer constants | Values needed at compile time |
| `chunk` must be a positive integer | Non-positive chunk sizes are invalid |
| `chunk` cannot be used with `init_values` in DSL | User-specified iter_args not supported on chunked loops |
| Chunked loops must be inside `pl.auto_incore()` | Only loops within `auto_incore` scope are split |

## Algorithm

Given a chunked loop in SSA form:

```text
for i, (x__iter_v1=x__ssa_v0,) in range(start, stop, step, chunk=C) -> (x__rv_v2,):
    x__ssa_v3 = add(x__iter_v1, 1.0)
    yield(x__ssa_v3)
```

1. Compute `trip_count = ceil((stop - start) / step)`
2. `num_full_chunks = trip_count // C`, `remainder = trip_count % C`
3. Generate outer loop with iter_args initialized from original init values
4. Generate inner loop with iter_args fed from outer iter_args, body substitution: `i = start + (i_out * C + i_in) * step`
5. Inner loop yields to inner return_vars; outer loop yields inner return_vars to outer return_vars
6. If `remainder > 0`, generate remainder loop with iter_args chained from outer return_vars
7. Remap original return_vars to the final loop's return_vars

The inner loop preserves the original `ForKind` (Sequential, Parallel, or Unroll). The outer loop is always Sequential.

## Auto-Name Abbreviations

The printed IR examples use the compact auto-name format `base__qualifier_role_vN`.
To keep generated names readable, some qualifiers are abbreviated:

| Abbreviation | Meaning |
| ------------ | ------- |
| `co` | `chunk_outer` |
| `ci` | `chunk_inner` |
| `cr` | `chunk_rem` / chunk remainder |

Examples:

- `i__co_idx_v0` = outer chunk loop index
- `x__ci_iter_v1` = inner chunk iter_arg
- `x__cr_rv_v1` = remainder-loop return var

Roles such as `idx`, `iter`, `rv`, and `ssa` keep their full spelling because they are already short and commonly used across passes.

## Example

**Before** (printed SSA IR form; not valid DSL source since `chunk` + `init_values` is forbidden in the DSL):

```python
@pl.program
class Before:
    @pl.function
    def main(self, x__ssa_v0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i__idx_v0, (x__iter_v1,) in pl.range(10, init_values=(x__ssa_v0,), chunk=5):
            x__ssa_v3 = pl.tensor.add(x__iter_v1, 1.0)
            x__rv_v2 = pl.yield_(x__ssa_v3)
        return x__rv_v2
```

**After** (nested loops, divisible):

```python
@pl.program
class After:
    @pl.function
    def main(self, x__ssa_v0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i__co_idx_v0, (x__co_iter_v1,) in pl.range(2, init_values=(x__ssa_v0,)):
            for i__ci_idx_v0, (x__ci_iter_v1,) in pl.range(
                5, init_values=(x__co_iter_v1,)
            ):
                x__ssa_v3 = pl.tensor.add(x__ci_iter_v1, 1.0)
                x__ci_rv_v1 = pl.yield_(x__ssa_v3)
            x__co_rv_v1 = pl.yield_(x__ci_rv_v1)
        return x__co_rv_v1
```

**With remainder** (`chunk=5`, trip_count=7):

```python
# Generates: outer(0,1) * inner(0,5) + remainder(0,2)
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

Each generated loop is tagged with a `LoopOrigin` annotation indicating how it was produced:

| LoopOrigin | Description |
| ---------- | ----------- |
| `Original` | Regular loop (default, not generated by splitting) |
| `ChunkOuter` | Outer loop iterating over chunk indices |
| `ChunkInner` | Inner loop iterating within a chunk |
| `ChunkRemainder` | Remainder loop for leftover iterations |

Access via `for_stmt.loop_origin` (Python) or `for_stmt->loop_origin_` (C++). Downstream passes can use this to distinguish generated loops from user-written ones.

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
