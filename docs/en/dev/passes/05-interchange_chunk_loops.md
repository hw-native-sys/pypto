# InterchangeChunkLoops Pass

Reorders nested ChunkOuter/ChunkInner loop pairs and inserts `InCore` scopes for downstream outlining.

## Overview

After `SplitChunkedLoops` splits chunked loops into nested `ChunkOuter→ChunkInner` pairs, the structure for nested chunked loops is:

```text
i_out[ChunkOuter] → i_in[ChunkInner,Parallel] → j_out[ChunkOuter] → j_in[ChunkInner,Parallel] → body
```

This pass reorders so all outer loops are on top and wraps the inner loops + body in `ScopeStmt(InCore)`:

```text
i_out[ChunkOuter] → j_out[ChunkOuter] → InCore{ i_in[ChunkInner] → j_in[ChunkInner] → body }
```

**Requires**: TypeChecked, SSAForm properties.

**When to use**: Runs automatically in the default pipeline after `SplitChunkedLoops` and before `OutlineIncoreScopes`. Only operates on loops inside `pl.auto_incore()` scope. The `AutoInCore` scope is consumed (removed) by this pass.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::InterchangeChunkLoops()` | `passes.interchange_chunk_loops()` | Function-level |

**Python usage**:

```python
from pypto import passes

result = passes.interchange_chunk_loops()(program)
```

## Constraints

| Constraint | Behavior |
| ---------- | -------- |
| SSA-only | Runs after `SplitChunkedLoops` (requires `SSAForm`) |
| Parallel-only interchange | Only interchanges when ALL ChunkInner loops have `ForKind::Parallel` |
| Sequential chunked loops | Not interchanged, but wrapped in InCore if inside `auto_incore` |
| Existing InCore | If chain body already contains `ScopeStmt(InCore)`, skip |
| Requires `auto_incore` scope | Only loops inside `ScopeStmt(AutoInCore)` are processed; the scope is consumed |

## Algorithm

1. **Collect chain** — Starting from a `ChunkOuter` ForStmt, walk into nested ForStmt body. Build list of `(ForStmt, LoopOrigin)` entries. Stop at non-ForStmt, `Original` loop, or `ScopeStmt`.

2. **Guard checks** — Verify all ChunkInner loops are Parallel. Check no existing InCore scope in innermost body.

3. **Separate** — Split chain into `outers` (ChunkOuter) and `inners` (ChunkInner).

4. **Reconstruct** (inside-out build):
   - Visit the innermost body
   - Wrap inners around body (preserving order), reconnecting iter_args
   - Wrap in `ScopeStmt(ScopeKind::InCore)`
   - Wrap outers around InCore (preserving order), reconnecting iter_args and yields

5. **Handle remainders** — `ChunkRemainder` loops: recurse into body. Wrap standalone parallel remainder sub-loops in InCore.

## Example

**Before** (after SplitChunkedLoops, all parallel):

```python
for i_out, (x_outer,) in pl.range(2, init_values=(x_0,)):        # ChunkOuter
    for i_in, (x_ia,) in pl.parallel(4, init_values=(x_outer,)):   # ChunkInner
        for j_out, (y_outer,) in pl.range(3, init_values=(x_ia,)):  # ChunkOuter
            for j_in, (y_ia,) in pl.parallel(4, init_values=(y_outer,)):  # ChunkInner
                z = pl.add(y_ia, 1.0)
                y_ia_rv = pl.yield_(z)
            y_outer_rv = pl.yield_(y_ia_rv)
        x_ia_rv = pl.yield_(y_outer_rv)
    x_outer_rv = pl.yield_(x_ia_rv)
return x_outer_rv
```

**After** (InterchangeChunkLoops):

```python
for i_out, (x_l0,) in pl.range(2, init_values=(x_0,)):        # ChunkOuter
    for j_out, (x_l1,) in pl.range(3, init_values=(x_l0,)):    # ChunkOuter
        with pl.incore():                                               # InCore inserted
            for i_in, (x_l2,) in pl.parallel(4, init_values=(x_l1,)):  # ChunkInner
                for j_in, (x_l3,) in pl.parallel(4, init_values=(x_l2,)):  # ChunkInner
                    z = pl.add(x_l3, 1.0)
                    x_l3_rv = pl.yield_(z)
                x_l2_rv = pl.yield_(x_l3_rv)
        x_l1_rv = pl.yield_(x_l2_rv)
    x_l0_rv = pl.yield_(x_l1_rv)
return x_l0_rv
```

## Remainder Handling

For non-divisible trip counts, remainder loops get InCore wrapping:

```python
for i_rem, (...) in pl.parallel(2, init_values=(...)):   # ChunkRemainder
    for j_out, (...) in pl.range(3, init_values=(...)):   # Interchange applied
        with pl.incore():
            for j_in, (...) in pl.parallel(4, init_values=(...)):
                body
    with pl.incore():                                            # Remainder wrapped
        for j_rem, (...) in pl.parallel(2, init_values=(...)):
            body
```

## Non-Chunk Statement Handling

When `auto_incore` is consumed, statements that were not handled by chunk interchange (standalone tensor ops, non-chunked loops, sequential chunked loops that failed the parallel guard) are wrapped in `ScopeStmt(InCore)` to ensure they get outlined into InCore functions by `OutlineIncoreScopes`.

Consecutive non-InCore statements are grouped into a single `ScopeStmt(InCore)`. Control flow statements (`YieldStmt`, `ReturnStmt`) are never wrapped.

**Example** — standalone op + parallel chunk:

```python
# Before (inside auto_incore, after SplitChunkedLoops)
with pl.auto_incore():
    x = pl.add(x, 1.0)                           # standalone op
    for i_out in pl.range(2):                     # ChunkOuter (parallel inner)
        for i_in in pl.parallel(4):
            x = pl.add(x, 2.0)

# After InterchangeChunkLoops
with pl.incore():                                 # standalone wrapped
    x = pl.add(x, 1.0)
for i_out in pl.range(2):                         # interchanged chunk
    with pl.incore():
        for i_in in pl.parallel(4):
            x = pl.add(x, 2.0)
```

**Example** — sequential chunk (fails interchange guard):

```python
# Before
with pl.auto_incore():
    for i_out in pl.range(2):                     # ChunkOuter (sequential inner)
        for i_in in pl.range(4):                  # ChunkInner, Sequential → fails guard
            x = pl.add(x, 1.0)

# After — entire chain wrapped in InCore
with pl.incore():
    for i_out in pl.range(2):
        for i_in in pl.range(4):
            x = pl.add(x, 1.0)
```

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
