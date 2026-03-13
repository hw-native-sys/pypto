# LowerBreakContinue Pass

Lowers `BreakStmt` and `ContinueStmt` in InCore/AIC/AIV functions into equivalent structured control flow.

## Overview

PTO and CCE codegen backends do not implement visitors for `BreakStmt`/`ContinueStmt`. This pass rewrites them before codegen:

- **Continue**: Restructures into `IfStmt` with phi-node `return_vars`. The continue path yields current iter_arg values; the normal path yields the original values. A single trailing `YieldStmt` uses the phi results.
- **Break**: Converts `ForStmt` to `WhileStmt` with a `__brk_flag` IterArg. The break path sets the flag to `True`; the while condition checks `not __brk_flag`.
- **Both**: Converts to `WhileStmt` (break requires it), then handles continue via the phi-node approach within the while body.

**Requirements**:

- Input IR must be in SSA form (SSAForm required and preserved)
- InCore scopes must be outlined (SplitIncoreOrch required and preserved)

**When to use**: Run after InferTileMemorySpace and before InitMemRef.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerBreakContinue()` | `passes.lower_break_continue()` | Function-level |

**Factory function**:

```cpp
Pass LowerBreakContinue();
```

**Python usage**:

```python
from pypto import passes

lowered = passes.lower_break_continue()(program)
```

## Properties

| Property | Required | Produced | Invalidated |
| -------- | -------- | -------- | ----------- |
| SSAForm | Yes | Yes | — |
| SplitIncoreOrch | Yes | Yes | — |

## Algorithm

### Continue Lowering (ForStmt)

```python
# Before:
for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
    if i < 5:
        continue
    y = pl.add(x_iter, x_iter)
    x_iter = pl.yield_(y)

# After (phi-node approach):
for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
    if i < 5:
        __phi_0 = pl.yield_(x_iter)       # IfStmt branch: yield current value
    else:
        y = pl.add(x_iter, x_iter)
        __phi_0 = pl.yield_(y)            # IfStmt branch: yield computed value
    x_iter = pl.yield_(__phi_0)           # Loop's trailing yield uses phi
```

### Break Lowering (ForStmt → WhileStmt)

```python
# Before:
for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
    if i > 5:
        break
    y = pl.add(x_iter, x_iter)
    x_iter = pl.yield_(y)

# After:
for (__brk_flag, __lv, x_iter) in pl.while_(init_values=(False, 0, x_0)):
    pl.cond(__lv < 10 and not __brk_flag)
    if __lv > 5:
        __phi_0, __phi_1, __phi_2 = pl.yield_(True, __lv + 1, x_iter)
    else:
        y = pl.add(x_iter, x_iter)
        __phi_0, __phi_1, __phi_2 = pl.yield_(False, __lv + 1, y)
    __brk_flag, __lv, x_iter = pl.yield_(__phi_0, __phi_1, __phi_2)
```

## Scope

- Only targets `IsInCoreType()` functions (InCore, AIC, AIV)
- Orchestration functions are left unchanged
- Processes nested loops bottom-up — break/continue in each loop is resolved at that loop's level
- Handles multiple continues and nested break+continue in the same loop

## Pipeline Position

```text
... → InferTileMemorySpace → LowerBreakContinue → InitMemRef → ...
```
