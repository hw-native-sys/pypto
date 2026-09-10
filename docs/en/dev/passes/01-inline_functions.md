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
     - Alpha-rename every locally-bound `Var` in the inlined body to a fresh name (`<orig>_inline<counter>`, with any trailing `_` trimmed off `<orig>`) to avoid collisions across multiple call sites.
     - Splice the renamed-and-substituted body's statements before the call site.
     - Wire up the callee's trailing return value according to the call-site form: `LHS = renamed_return` (single-return assign; omitted when `LHS` resolves to the same `Var` as the substituted value, to avoid a redundant SSA copy), per-element `TupleGetItemExpr` substitution instead of a `MakeTuple` binding (multi-return assign), a fresh `ReturnStmt` (`return inline_call(...)`), or a fresh `EvalStmt` when the value is discarded but its evaluation is observable (`EvalStmt` call site — see [Edge cases](#edge-cases)).
4. **Drop** all Inline functions from the program.

The pass uses a single underscore (`_inline`) in the rename suffix because `__` is reserved by the IR's auto-naming convention (see `auto_name_utils.h`).

A single-underscore suffix is not sufficient on its own: `<orig>` may itself end in `_` — `_` is Python's throwaway name and the documented loop variable of `for _ in pl.split_aiv(...)` — and plain concatenation would then fuse two individually-legal underscores into the reserved delimiter. `FreshName` therefore joins through `auto_name::JoinNameSuffix`, which trims that tail: `_` renames to `_inline7`, not `__inline7`. A `<orig>` that *already* contains `__` is passed through unchanged, so an author-written `a__b` stays the user-facing error `ValidateBaseName` reports rather than being silently normalized.

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
| Multi-return inline | No `LHS = MakeTuple([rets...])` is emitted — orchestration codegen cannot lower `MakeTuple`. The cloned return values are recorded against the LHS `Var` and downstream `TupleGetItemExpr(LHS, i)` uses are rewritten to value `i`, leaving the LHS binding unreferenced (see `SpliceInlineCallAsTupleSub`). |
| Nested call to Inline (e.g. `pl.add(inline_fn(x), y)`) | Not handled in v1 — left as-is. The `InlineFunctionsEliminated` verifier flags any surviving Call. |
| `EvalStmt(inline_call(...))` — return value ignored | The value is discarded, its **evaluation** is not. See [Discarding a return value](#discarding-a-return-value) below. |

## Discarding a return value

An `EvalStmt` call site — `self.wrapper(x, out)` with no LHS — has nowhere to put the callee's trailing return value. Dropping that **value** is correct; dropping its **evaluation** is not, because evaluating it can write through `Out` / `InOut` arguments, launch a task, block on a signal, or set up hardware. Each discarded value is therefore classified:

| Discarded value | Behaviour |
| --------------- | --------- |
| A `Call` — any callee, cross-function or builtin | Re-emitted as an `EvalStmt`, in return order. The fixpoint loop expands a cross-function one on its next iteration when that callee is also Inline; otherwise it stays an ordinary dispatch, exactly as if the author had written it at the call site. |
| A `Submit` | Re-emitted as an `EvalStmt`. A task launch is effectful whatever its callee does. |
| Anything else that hides no call — a `Var`, a constant | Dropped. |
| A value that is not itself call-like but *wraps* a call — scalar arithmetic such as `self.bump(n) + 1`, a `MakeTuple`, a `TupleGetItemExpr` | `pypto::ValueError`. It cannot become an `EvalStmt`, and deleting it would delete the nested call with it. Return that call directly, or bind the wrapper's result at the call site. |

**Why every call, rather than only the ones that write.** Nothing in the IR answers "is this call safe to delete". The nearest registry data, `OpRegistryEntry::WritesAnyArg`, answers whether an operator writes *through an argument*, and keying deletion on it is wrong in both directions:

- Most operators are simply unclassified — 263 of 315 at the time of writing, among them `tile.tpush_to_aiv` and `system.aic_initialize_pipe`, which `dce::IsSideEffectOp` lists as side-effecting. `OpRegistryEntry::HasDeclaredArgEffects` exists precisely so an analysis can tell "declared to write nothing" from "nobody looked yet".
- A *positive* `no_arg_writes()` verdict does not mean deletable either. `pld.system.wait` blocks until a signal slot satisfies a threshold, `pld.system.defer_wait` registers a completion condition, and `system.set_ffts` hands the FFTS unit its workspace pointer — all three declare `no_arg_writes()` while carrying synchronization or hardware-setup semantics.

So the pass keeps every call. A discarded genuinely pure call survives as a dead `EvalStmt`, which the pipeline carries harmlessly. Narrowing this needs a real "safely deletable" operator property, declared per operator rather than inferred from writes.

**Before**:

```python
@pl.function(type=pl.FunctionType.Inline)
def writeout(self, t, out: pl.Out[...]):
    return pl.tile.store(t, [0, 0], out)   # the write IS the return expression

@pl.function(type=pl.FunctionType.InCore)
def kernel(self, a, out: pl.Out[...]):
    t = pl.tile.load(a, [0, 0], [64, 64])
    self.writeout(t, out)                  # return value ignored
    return out
```

**After** — the store survives:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(self, a, out: pl.Out[...]):
    t = pl.tile.load(a, [0, 0], [64, 64])
    pl.tile.store(t, [0, 0], out)
    return out
```

## Verification

The `InlineFunctionsEliminated` `PropertyVerifier` (registered against `IRProperty.InlineFunctionsEliminated`) confirms:

1. No `Function` with `func_type == FunctionType::Inline` remains.
2. No `Call` whose callee resolves to one survives.

## See also

- `python/pypto/jit/decorator.py` — `@pl.jit.inline` is the user-facing front end (`_SubFunctionDecorator("inline", ...)`).
- [02-unroll_loops](02-unroll_loops.md) — runs immediately after.
- [09-outline_incore_scopes](09-outline_incore_scopes.md) — handles the `pl.at` scopes that survive splicing.
