# InlineFunctions Pass

Eliminates `FunctionType.Inline` functions by splicing their bodies at every call site.

## Overview

Functions decorated as `@pl.function(type=pl.FunctionType.Inline)` (or via the JIT-side `@pl.jit.inline`) are *source-level utilities*: each call site expands into a fresh, alpha-renamed copy of the body, with formal parameters substituted by actual-argument expressions. After this pass, no `FunctionType.Inline` function and no `Call` to one survives in the program — subsequent passes treat the spliced code as if it had been written inline at the call site.

Runs as the **first** pass in `OptimizationStrategy.Default` so downstream passes (`UnrollLoops`, `OutlineIncoreScopes`, …) never observe Inline functions.

**Produces**: `IRProperty.InlineFunctionsEliminated`.

**Requires**: nothing — runs on a freshly parsed program.

**When to use**: Always, as part of the default pipeline. The pass is a no-op when no Inline functions exist.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::InlineFunctions()` | `passes.inline_functions()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

inline_pass = passes.inline_functions()
program_inlined = inline_pass(program)
```

## Algorithm

1. **Collect** all functions with `func_type == FunctionType::Inline`.
2. **Cycle-detect** the Inline → Inline call graph; raise `pypto::ValueError` naming the cycle if one is found.
3. **Iterate to fixpoint** — each iteration walks every function (including the Inline ones, so that nested Inline-calls-Inline expands transitively):
   - For every top-level `LHS = inline_call(args)` or `EvalStmt(inline_call(args))` in a function body:
     - Build the param-substitution map (formal `Var` → actual `Expr`).
     - Alpha-rename every locally-bound `Var` in the inlined body to a fresh name (`<orig>_inline<counter>`) to avoid collisions across multiple call sites.
     - Splice the renamed-and-substituted body's statements before the call site.
     - Replace the call with: `LHS = renamed_return` (single-return) or `LHS = MakeTuple([renamed_returns...])` (multi-return). When `LHS` resolves to the same `Var` as the substituted return value, the assignment is omitted to avoid a redundant SSA copy.
4. **Drop** all Inline functions from the program.

The pass uses a single underscore (`_inline`) in the rename suffix because `__` is reserved by the IR's auto-naming convention (see `auto_name_utils.h`).

## Example

### Single call site

**Before**:

```python
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Inline)
    def helper(self, x):
        y = pl.mul(x, x)
        return y

    @pl.function
    def main(self, a):
        z = self.helper(a)
        return z
```

**After**:

```python
@pl.program
class P:
    @pl.function
    def main(self, a):
        y_inline0 = pl.mul(a, a)
        z = y_inline0
        return z
```

### Multiple call sites

Each site is independently alpha-renamed, so locals never collide:

**Before**:

```python
@pl.function(type=pl.FunctionType.Inline)
def square(self, x):
    y = pl.mul(x, x)
    return y

@pl.function
def main(self, a, b):
    a2 = self.square(a)
    b2 = self.square(b)
    return pl.add(a2, b2)
```

**After**:

```python
@pl.function
def main(self, a, b):
    y_inline0 = pl.mul(a, a)
    a2 = y_inline0
    y_inline1 = pl.mul(b, b)
    b2 = y_inline1
    return pl.add(a2, b2)
```

### Inline body containing `pl.at`

The scope is preserved verbatim and gets outlined by `OutlineIncoreScopes` later in the pipeline, exactly as if it had been written at the call site.

## Edge cases

| Case | Behaviour |
| ---- | --------- |
| Inline function with no callers | Silently removed from the program. |
| Inline function as program entry | Not detected as an error here — but no Call to it exists, so it is removed in the cleanup phase like any other no-caller function. |
| Inline calls Inline (transitive) | Iteratively expanded to fixpoint. |
| Recursive Inline (self or mutual) | `pypto::ValueError` raised before any splicing, with the cycle named (`a -> b -> a`). |
| Multi-return inline | `LHS = MakeTuple([rets...])` emitted at the call site. Subsequent `Simplify` may fold `TupleGetItemExpr(MakeTuple(...), i)`. |
| Nested call to Inline (e.g. `pl.add(inline_fn(x), y)`) | Not handled in v1 — left as-is. The `InlineFunctionsEliminated` verifier flags any surviving Call. |

## Verification

The `InlineFunctionsEliminated` `PropertyVerifier` (registered against `IRProperty.InlineFunctionsEliminated`) confirms:

1. No `Function` with `func_type == FunctionType::Inline` remains.
2. No `Call` whose callee resolves to one survives.

## See also

- `python/pypto/jit/decorator.py` — `@pl.jit.inline` is the user-facing front end (`_SubFunctionDecorator("inline", ...)`).
- [02-unroll_loops](02-unroll_loops.md) — runs immediately after.
- [10-outline_incore_scopes](10-outline_incore_scopes.md) — handles the `pl.at` scopes that survive splicing.
