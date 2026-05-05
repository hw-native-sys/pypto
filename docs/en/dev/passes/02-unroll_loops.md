# UnrollLoops Pass

Expands `ForKind::Unroll` loops at compile time by inlining the loop body for each iteration value.

## Overview

This pass statically unrolls for loops created with `pl.unroll()`, replacing them with repeated copies of the loop body where the loop variable is substituted with each iteration's constant value.

**Requires**: TypeChecked property.

**When to use**: Runs automatically in the default pipeline before `ConvertToSSA`. Use `pl.unroll()` when the loop trip count is a compile-time constant and you want the body duplicated for each iteration.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::UnrollLoops()` | `passes.unroll_loops()` | Function-level |

**Python usage**:

```python
from pypto import passes

# Create and apply the pass
result = passes.unroll_loops()(program)
```

## DSL Syntax

```python
# Basic unroll: body is duplicated 4 times with i = 0, 1, 2, 3
for i in pl.unroll(4):
    x = pl.add(x, i)

# With start/stop/step: body is duplicated for i = 0, 2, 4
for i in pl.unroll(0, 6, 2):
    x = pl.add(x, i)
```

## Constraints

| Constraint | Reason |
| ---------- | ------ |
| `start`, `stop`, `step` must be integer constants | Values needed at compile time |
| `step` must be non-zero | Prevents infinite loops |
| `init_values` cannot be used with `pl.unroll()` | No loop-carried state in unrolled loops |

## Example

**Before** (with unroll loop):

```python
@pl.program
class Before:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.unroll(3):
            x = pl.add(x, i)
        return x
```

**After** (loop expanded):

```python
@pl.program
class After:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        x = pl.add(x, 0)
        x = pl.add(x, 1)
        x = pl.add(x, 2)
        return x
```

## Pipeline Position

UnrollLoops runs **once** in `Default` and `DebugTileOptimization`, before control flow structuring:

```text
UnrollLoops → CtrlFlowTransform → ConvertToSSA → FlattenCallExpr → SplitChunkedLoops → InterchangeChunkLoops → OutlineIncoreScopes → ...
```

UnrollLoops expands non-chunked `pl.unroll()` loops (skipping chunked unroll loops which retain `chunk` for `SplitChunkedLoops` to handle later).

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `TypeChecked` |
| Produced | `TypeChecked` |
| Invalidated | (none) |
