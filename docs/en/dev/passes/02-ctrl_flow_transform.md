# CtrlFlowTransform Pass

Transforms `break` and `continue` statements into equivalent structured control flow (if-else + while loops) so that downstream passes and code generation can operate on structured IR without unstructured jumps.

## Overview

PTO codegen emits MLIR-style SCF (structured control flow), which does not directly support `break` or `continue`. This pass eliminates both by rewriting loops into equivalent structured forms.

**Applies to**: InCore-type functions only (InCore, AIC, AIV). Orchestration/Host functions are skipped because they can use `break`/`continue` natively.

**Requires**: `TypeChecked`

**Produces**: `TypeChecked`, `StructuredCtrlFlow`

**When to use**: Runs automatically in the default pipeline after `UnrollLoops` and before `ConvertToSSA`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::CtrlFlowTransform()` | `passes.ctrl_flow_transform()` | Function-level |

**Python usage**:

```python
from pypto.pypto_core import passes

result = passes.ctrl_flow_transform()(program)
```

## Algorithm

The pass runs two phases in order:

| Phase | Eliminates | Strategy |
| ----- | ---------- | -------- |
| 1 | `continue` | Restructure remaining body into `if-else` |
| 2 | `break` | Convert `ForStmt` to `WhileStmt` with break flag |

**Phase 1 must run before Phase 2** because `continue` elimination preserves the loop type (ForStmt/WhileStmt), while `break` elimination converts ForStmt to WhileStmt. Running Phase 1 first keeps the transformation simpler.

### Phase 1: Continue Elimination

Remaining statements after a `continue` are moved into an `else` branch. The loop type is preserved.

**Before**:

```python
for i in pl.range(n):
    A
    if cond:
        continue
    B
    C
```

**After**:

```python
for i in pl.range(n):
    A
    if cond:
        pass  # nothing
    else:
        B
        C
```

This works identically for `WhileStmt`. Multiple `continue` statements are handled by repeated application (innermost first).

### Phase 2: Break Elimination

A `ForStmt` containing `break` is converted to a `WhileStmt` with an auxiliary break flag variable.

**Before**:

```python
for i in pl.range(start, stop, step):
    A
    if cond:
        break
    B
    C
```

**After**:

```python
i = start
__break_N: bool = False
while i < stop and not __break_N:       # i > stop for negative step
    A
    if cond:
        __break_N = True
    else:
        B
        C
    if not __break_N:
        i = i + step
```

Key details:

| Aspect | Behavior |
| ------ | -------- |
| Break flag naming | `__break_N` where N is a unique counter |
| While condition | `And(i < stop, Not(__break_N))` for positive step; `And(i > stop, Not(__break_N))` for negative step |
| Iterator advancement | Guarded by `if not __break_N` to prevent advancing past the break point |
| WhileStmt break | Same pattern but without the for-to-while conversion |

### Combined Break + Continue

When a loop contains both `break` and `continue`, Phase 1 eliminates all `continue` statements first, then Phase 2 eliminates `break` statements on the already-transformed IR.

**Before**:

```python
for i in pl.range(n):
    A
    if cond1:
        continue
    B
    if cond2:
        break
    C
```

**After Phase 1** (continue eliminated):

```python
for i in pl.range(n):
    A
    if cond1:
        pass
    else:
        B
        if cond2:
            break
        C
```

**After Phase 2** (break eliminated):

```python
i = 0
__break_0: bool = False
while i < n and not __break_0:
    A
    if cond1:
        pass
    else:
        B
        if cond2:
            __break_0 = True
        else:
            C
    if not __break_0:
        i = i + 1
```

## Pipeline Position

CtrlFlowTransform runs after UnrollLoops and before ConvertToSSA:

```text
UnrollLoops -> CtrlFlowTransform -> ConvertToSSA -> FlattenCallExpr -> SplitChunkedLoops -> ...
```

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `TypeChecked` |
| Produced | `TypeChecked`, `StructuredCtrlFlow` |
| Invalidated | (none) |
