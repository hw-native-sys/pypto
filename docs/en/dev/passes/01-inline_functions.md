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
     - Build the param-substitution map (formal `Var` → actual `Expr`). The map applies at use-sites **and** def-sites, so a callee rebinding `out = pl.tensor.assemble(out, ...)` rebinds the caller's actual `Var` — but only for a param that carries an in-place contract (see [Rebinding a parameter](#rebinding-a-parameter)). Three kinds of actual arg are instead bound to a fresh `<param>_inline<counter>` `Var` ahead of the spliced body, and that `Var` is substituted: the arg of a rebound param that is not an assignable `Var` (a slice `c[r]`, an `IterArg`, a computed scalar); the arg of a rebound pass-by-value param (a plain scalar that is neither `pl.Out` nor `pl.InOut`); and any computed tensor / tile arg (a `Call` such as `a[r]`), which Python evaluates once at the call site. Other args — `Var`s bound to read-only params, scalar expressions, constants — are substituted directly, so shape expressions that read a param still fold.
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

## Rebinding a parameter

Because substitution reaches def-sites, a callee that rebinds one of its own parameters could land that rebinding on the caller's `Var`. Whether it *should* depends on the parameter's calling convention:

| Parameter | Rebinding lands on | Why |
| --------- | ------------------ | --- |
| `pl.Out[...]` / `pl.InOut[...]` | the caller's `Var` | The author opted into the in-place contract explicitly. |
| Any tensor / tile / `Array` param | the caller's `Var` | These params are *handles*, and the rebinding **is** the in-place update: `c[...] = v` parses as `c = pl.tensor.assemble(c, ...)` and `a[i] = v` as `a = pl.array.update_element(a, i, v)`. `@pl.jit.inline` strips `pl.Out` / `pl.InOut` from shaped params for exactly this reason, so direction alone cannot be the test. Binding a temporary here would also emit a bare `arr_inline0 = arr` alias, which orchestration codegen cannot declare — an array `Var` comes from `array.create` or an alias onto a backing array, never from its type. |
| A plain scalar | a fresh `<param>_inline<counter>` `Var` | Pass-by-value, as in Python. |

The last row is the one that is easy to get wrong. Given:

```python
@pl.function(type=pl.FunctionType.Inline)
def bump(self, n: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    n = n + 1
    return n

@pl.function
def main(self, k: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    m: pl.Scalar[pl.INDEX] = self.bump(k)
    return k + m                              # k + (k + 1)
```

the call site binds a temporary first, so the caller's `k` stays readable afterwards:

```python
n_inline0: pl.Scalar[pl.INDEX] = k
n_inline0 = n_inline0 + 1
m: pl.Scalar[pl.INDEX] = n_inline0
s: pl.Scalar[pl.INDEX] = k + m                # the ORIGINAL k
```

Substituting `k` at the def-site instead would splice `k = k + 1` into the caller and silently compute `2k + 2`. Note that the JIT specializer's alpha-rename does **not** cover this: it renames a rebinding only at the scope depth where the name was first bound, so a rebind inside `pl.range` / `pl.spmd` reaches this pass as a genuine def-site of the param.

## Edge cases

| Case | Behaviour |
| ---- | --------- |
| Inline function with no callers | Silently removed from the program. |
| Inline function as program entry | Not detected as an error here — but no Call to it exists, so it is removed in the cleanup phase like any other no-caller function. |
| Inline calls Inline (transitive) | Iteratively expanded to fixpoint. |
| Computed tensor / tile arg, e.g. `f(a[r], c[r])` where the callee reads `x` and writes `c[0:4, j] = v` | Each arg is bound once at the call site — `x_inline0 = a[r]`, `c_inline1 = c[r]`, then `c_inline1 = pl.tensor.assemble(c_inline1, v, ...)` — the IR the parser emits when the caller names the slices itself. Substituting `c[r]` would put a `Call` on the LHS of the rebinding; substituting `a[r]` would re-evaluate it inside the callee's `pl.spmd` / `pl.pipeline` bodies and move it into the outlined kernel. Each slice is a view of its source, so the write reaches the caller's `c`. |
| Rebound pass-by-value param, e.g. an inline callee doing `n = n + 1` on a scalar param | The arg is bound once at the call site (`n_inline0 = k`) and the rebinding stays on that temporary, so the caller's `k` is unchanged afterwards. See [Rebinding a parameter](#rebinding-a-parameter). |
| Recursive Inline (self or mutual) | `pypto::ValueError` raised before any splicing, with the cycle named (`a -> b -> a`). |
| Multi-return inline | No `LHS = MakeTuple([rets...])` is emitted — orchestration codegen cannot lower `MakeTuple`. The cloned return values are recorded against the LHS `Var` and downstream `TupleGetItemExpr(LHS, i)` uses are rewritten to value `i`, leaving the LHS binding unreferenced (see `SpliceInlineCallAsTupleSub`). |
| Nested call to Inline (e.g. `pl.add(inline_fn(x), y)`, or the `array.update_element(arr, i, inline_fn(x))` the parser desugars `arr[i] = inline_fn(x)` into) | Hoisted onto an `AssignStmt` of its own, then spliced in the same iteration — see [Nested call sites](#nested-call-sites). |
| Nested call to a **tuple-returning** Inline, a `WhileStmt` condition, an `IterArg` init value, or a bare (non-`SeqStmts`) `ForStmt` / `IfStmt` body | Not hoisted. The `InlineFunctionsEliminated` verifier reports the surviving Call at its own source line right after this pass. |
| `EvalStmt(inline_call(...))` — return value ignored | The value is discarded, its **evaluation** is not. See [Discarding a return value](#discarding-a-return-value) below. |

## Nested call sites

`HandleTopLevelInlineCall` recognises a call site only when the `Call` **is** the whole statement value — `LHS = f(...)`, `EvalStmt(f(...))`, `return f(...)`. A `Call` anywhere else would be skipped while step 5 dropped the callee anyway, leaving a reference to a deleted function that only failed at `GenerateOrchestration preconditions` with `references undefined function`.

Such calls come from ordinary DSL, sometimes without the user writing a nested call at all. `arr[i] = f(x)` has no IR statement of its own; the parser desugars it to a functional update:

```python
arr[i] = f(x)                              # what the user writes
arr = pl.array.update_element(arr, i, f(x))  # what the parser stores — f is now an argument
```

`NestedInlineCallHoister` therefore runs over each statement's own expressions before the call-site match, pulling every nested inline `Call` onto a fresh `t__inline_arg_vN` binding placed before that statement:

```python
# before                                   # after the hoist, before the splice
k = self.half(n) + 1                       t__inline_arg_v0 = self.half(n)
                                           k = t__inline_arg_v0 + 1
```

`SpliceHoisted` then splices each hoisted binding immediately, so a hoist and the splice it enables land in the same fixpoint iteration and the `inline_fns.size() + 1` iteration bound still holds.

Three properties worth keeping in mind when editing this:

- **A call already in top-level position is left alone** (`HoistInArgs` rewrites only its arguments). Hoisting it would add a redundant copy to every existing call site and churn every before/after test.
- **Bodies are not touched.** The hoister rewrites only the statement's own expressions; `InlineCallsMutator` still recurses into loop and branch bodies, so a hoist inside a body lands inside that body.
- **Every hoisted position is evaluated exactly once**, at the point the hoisted statement lands — a call argument, a binary operand, a loop bound, an `if` condition, a `yield` value. A `WhileStmt` condition is not, which is why it is excluded: hoisting it would evaluate the spliced body once instead of per iteration.
- **A tuple-returning callee is never hoisted.** `SpliceInlineCallAsTupleSub` deliberately emits no `tmp = ...` binding — it records the cloned return values against the LHS `Var` and rewrites downstream `TupleGetItemExpr(tmp, i)` uses instead. A nested consumer holds `tmp` itself rather than a `TupleGetItemExpr`, so hoisting would leave the temp undefined: `return self.pair(x), y` printed `t__inline_arg_v0__FREE_VAR`. Leaving the `Call` in place keeps the pre-hoist behaviour, and the verifier names it.

This is deliberately narrower than `FlattenCallExpr` (pass 06), which performs the same hoist for *all* calls. That pass declares `.required = {SSAForm, NormalizedStmtStructure}`, both established after this one, so it cannot simply run first.

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
- [09-outline_incore_scopes](10-outline_incore_scopes.md) — handles the `pl.at` scopes that survive splicing.
