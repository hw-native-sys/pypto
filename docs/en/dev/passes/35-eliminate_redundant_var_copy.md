# EliminateRedundantVarCopy

Copy-propagates redundant `X = Y` Var rebinds in Orchestration functions:
rewrites every use of `X` to `Y` and drops the copy statement.

- **Position**: 35th pass, immediately after [`DeriveCallDirections`](34-derive_call_directions.md)
- **Scope**: `FunctionType::Orchestration` bodies only
- **Properties**: requires `SplitIncoreOrch`, `CallDirectionsResolved`; preserves both
- **Source**: `src/ir/transforms/eliminate_redundant_var_copy_pass.cpp`

## Why

After outlining, SSA conversion, and direction derivation, an orchestration body
accumulates pure SSA rebinds whose value is just another `Var`/`IterArg`:

```python
score__ssa_v1: pl.Tensor[[64], pl.FP32] = score__rv_v2   # same physical buffer
```

`X` and `Y` name the *same physical buffer* — an orchestration `Tensor` is a
handle, not a value, so the rebind copies nothing. Left in the IR, these aliases
forced the orchestration codegen to reason about them at emit time:

- A rebind whose two sides collapsed onto the same param-rooted emit name
  produced `auto X = X;`, which gcc rejects with
  *"use of 'X' before deduction of 'auto'"*.
- A rebind inside a `pl.manual_scope` emitted a block-local `Tensor X = Y;`
  whose name dies at the closing brace, so a task placed **after** the scope
  named an out-of-scope `X` and the generated `.cpp` failed to compile
  (issues [#1697](https://github.com/hw-native-sys/pypto/issues/1697) /
  [#1713](https://github.com/hw-native-sys/pypto/issues/1713)).

Codegen papered over both with an emit-time band-aid (`FIXME(#1281)`), violating
the strict 1-to-1 translation contract in
[`00-pto_codegen.md`](../codegen/00-pto_codegen.md). This pass removes from the
IR every alias it can prove safe, so codegen no longer has to reason about them.
See [Limitations](#limitations) for the residual case the band-aid still covers.

## Transform

For each `AssignStmt` `X = Y` where `Y` is a `Var`/`IterArg`, fold (rewrite all
uses of `X` to `Y`, delete the statement) when **all** guards hold:

| # | Guard | Rationale |
| - | ----- | --------- |
| 1 | Enclosing function is `Orchestration` | InCore/Group/Spmd bodies carry no such rebinds; distributed `host_orch` has its own codegen |
| 2 | Neither `X` nor `Y` is a carry lvalue | Codegen manages `iter_args` / `return_vars` as C++ locals reassigned across iterations and branch phases; collapsing one breaks the snapshot |
| 3 | `X` and `Y` share a buffer root | Guarantees the rebind is a pure alias, not a real data move |
| 4 | `Y`'s defining region encloses every read of `X` | A source defined inside a `manual_scope` is dead at an after-scope reader; folding onto it would relocate the #1713 failure rather than fix it |

Buffer identity reuses `BufferRootCollector` (`orchestration_analysis.h`), the
same oracle `DeriveCallDirections` uses.

Guard 4 is what resolves #1713. A scope-internal rebind chain rooted at an
**outer** base folds all the way to that base, so the after-scope reader names an
enclosing variable and no block-local alias is emitted:

```python
# Before
with pl.scope(mode=pl.ScopeMode.MANUAL):
    x__rv_v2 = base          # block-local alias of an outer tensor
    x__rv_v5 = x__rv_v2
r = self.rd(x__rv_v5)        # after-scope reader -> `x__rv_v5` is dead in C++

# After
with pl.scope(mode=pl.ScopeMode.MANUAL):
    pass
r = self.rd(base)            # resolves in the enclosing frame
```

Conversely a source *defined inside* the scope is left alone, since folding would
not make it any more visible:

```python
with pl.scope(mode=pl.ScopeMode.MANUAL):
    p__ssa_v0 = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
    x__ssa_v1 = p__ssa_v0    # kept: `p__ssa_v0` is scope-local
r = self.rd(x__ssa_v1)
```

Copy chains are resolved transitively: `X = Y; Z = X` makes every use of both `X`
and `Z` name `Y`. Candidates sharing an ultimate source are dropped as a group,
so a chain never half-folds.

Not folded: a value that is a `Call`/`Submit`/`MakeTuple`/`TupleGetItemExpr`
(these are real definitions), and TaskId scalars (no buffer root), whose alias
mapping orchestration codegen still tracks.

## Effect on generated code

A loop carry that is only ever an alias of an `pl.Out` param disappears entirely,
together with its self-assignment:

```cpp
// Before
Tensor k_proj_rv = ext_k_proj;
for (...) {
    Tensor k_proj_iter = k_proj_rv.view(...);
    rt_submit_task(...);              // writes the buffer in place (add_inout)
    Tensor k_proj_next = k_proj_rv;
    k_proj_rv = k_proj_next;          // no-op
}

// After
for (...) {
    Tensor k_proj_iter = ext_k_proj.view(...);
    rt_submit_task(...);
}
```

## Limitations

Guard 2 refuses any copy touching a carry lvalue, so a **post-loop rebind of a
loop carry** (`score__ssa_v1 = score_rv`) is left in the IR. When that loop sits
inside a `pl.manual_scope` and the copy is read after the scope, orchestration
codegen still needs its `FIXME(#1281)` emit-time guard to remap the name.

Folding that case correctly requires distinguishing a post-loop carry read (the
carry is stable, so collapsing is safe) from an in-loop copy taken before the
loop's yield (collapsing would alias the carry's *later* value — see
`test_manual_scope_in_loop_carry_copy_keeps_snapshot`). Codegen resolves this
today by *hoisting* the carry's `Tensor carry = init;` declaration out of the
manual-scope body, a decision this pass cannot see from the IR. Retiring the
band-aid therefore means moving that hoist into the IR as well; until then the
guard stays and this pass simply never produces the aliases it does fold.

## Complexity

Four linear traversals (buffer roots, carries, copy candidates, scope/def-use
map) plus a rewrite, with hash-map lookups — `O(N log N)`, satisfying
`pass-complexity.md`.

## Interaction with the printer

The pass may fold away every statement of a region (e.g. a `manual_scope` whose
body was only rebinds). An empty body must still print an indented block, so
`IRPythonPrinter` emits `pass` for it; otherwise the printed IR would not
re-parse and the pipeline's roundtrip verification would reject it.

## Tests

`tests/ut/ir/transforms/test_eliminate_redundant_var_copy_pass.py` — before/after
structural comparison for each guard: folds a param copy, an SSA copy chain, an
in-place-written source, and the #1713 outer-base chain across a `manual_scope`;
keeps a loop carry, a scope-local source, a `Call` RHS, and InCore bodies.

## See also

- [`34-derive_call_directions.md`](34-derive_call_directions.md) — runs immediately before; supplies `arg_directions` and the `BufferRootCollector` idiom
- [`05-simplify.md`](05-simplify.md) — scalar constant propagation only; never restructures statements
- [`../codegen/01-orchestration_codegen.md`](../codegen/01-orchestration_codegen.md) — the 1-to-1 emit contract this pass upholds
