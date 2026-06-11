# MaterializeRuntimeScopes Pass

Inserts explicit AUTO `RuntimeScopeStmt` nodes into Orchestration functions so
that PTO orchestration codegen emits `PTO2_SCOPE()` 1:1 from the IR instead of
deriving the scope structure from `for` / `if` statements — unless the function
opts out with `@pl.function(auto_scope=False)`, in which case the user places
every scope by hand.

## Overview

The simpler runtime wraps regions of an orchestration routine in `PTO2_SCOPE()`
blocks (auto dependency tracking via the OverlapMap). It also provides an
implicit top-level scope, so scopes are a **tuning / placement** mechanism, never
a correctness requirement — a function may end up with zero compiler scopes.

By default (`auto_scope=True`) the compiler owns scope placement: for every
`FunctionType::Orchestration` function this pass inserts AUTO `RuntimeScopeStmt`
(`manual_ = false`) nodes wrapping the whole function body and each `ForStmt`
body and `IfStmt` then/else body (suppressed inside a manual scope, since the
runtime forbids AUTO nested in MANUAL). Codegen then emits `PTO2_SCOPE` **only**
from `RuntimeScopeStmt` nodes — staying 1:1 with the IR (see
[orchestration codegen](../codegen/01-orchestration_codegen.md)).

Under `@pl.function(auto_scope=False)` the pass inserts **nothing**: the user
places scopes with `with pl.scope()` / `with pl.scope(mode=pl.ScopeMode.MANUAL)`,
which the parser materialises directly into the IR. This is the knob for
controlling scope granularity (ring isolation), MANUAL dependency regions, and
full takeover.

After materialising scopes in the default mode, the pass marks the function
`auto_scope=False` (scopes are now placed). This makes the pass idempotent and
lets the output round-trip: the inserted `with pl.scope()` blocks parse back only
under `auto_scope=False` (the parser rejects hand-placed AUTO scopes in the
default mode, where the compiler owns placement).

**When to use**: last pass in the `Default` and `DebugTileOptimization`
strategies, after the final `Simplify`. Running dead last means no other
transform has to reason about the inserted scope wrappers.

**Scope**: only `Orchestration` functions are modified. InCore / AIC / AIV /
Group / Spmd bodies are never scope-wrapped by codegen, so they are returned
unchanged.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeRuntimeScopes()` | `passes.materialize_runtime_scopes()` | Function-level |

```python
from pypto.pypto_core import passes

scoped = passes.materialize_runtime_scopes()(program)
```

## Behavior

| Function | for/if + function body | Hand-placed `with pl.scope()` |
| -------- | ---------------------- | ----------------------------- |
| `auto_scope=True` (default) | Auto-wrapped in AUTO scope (suppressed inside a manual scope) | Rejected by the parser (use `auto_scope=False`) |
| `auto_scope=False` | Not auto-wrapped (pass is a no-op) | The only scopes; `with pl.scope(mode=MANUAL)` and the `manual_scope` alias also allowed |

In the default mode the `InsertAutoScopeMutator` walks the body:

1. On entering a **manual** `RuntimeScopeStmt`, a depth counter is incremented;
   AUTO insertion is suppressed while the counter is non-zero (AUTO-in-MANUAL is
   forbidden). AUTO scopes do not suppress nesting.
2. Each `ForStmt` body is wrapped in `RuntimeScopeStmt(manual=false)` unless
   already AUTO-wrapped; each `IfStmt` then/else body likewise.
3. The whole function body is then wrapped in one outermost AUTO scope, and the
   function is marked `auto_scope=False`.

## Example

```python
# Before — default auto_scope=True
@pl.function(type=pl.FunctionType.Orchestration)
def orch(self, a, out):
    for i in pl.range(4):
        out = self.kernel(a, out)
    return out
```

```python
# After MaterializeRuntimeScopes (marked auto_scope=False; round-trips)
@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
def orch(self, a, out):
    with pl.scope():            # function body
        for i in pl.range(4):
            with pl.scope():    # loop body
                out = self.kernel(a, out)
        return out
```

```python
# Opt out and place scopes yourself (coarser granularity here: one scope)
@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
def orch(self, a, out):
    with pl.scope():
        for i in pl.range(4):
            out = self.kernel(a, out)
        return out
```

A trailing return-var `yield` stays inside the scope; the printer recurses
through the AUTO scope so the `var = pl.yield_(...)` assignment LHS is preserved,
and the parser treats the yield inside `pl.scope()` as the enclosing for/if's
return-var.

## Verification

**Tests**: `tests/ut/ir/transforms/test_materialize_runtime_scopes.py` (auto-mode
wrapping, manual-scope suppression, idempotency, opt-out no-op / scope
preservation, AUTO-in-default rejection) and
`tests/ut/language/parser/test_scope_parsing.py` (`pl.scope()` parse / round-trip
/ mode / nesting / opt-out rules). The full orchestration codegen suite
(`tests/ut/codegen/test_orchestration_codegen.py`) verifies the emitted
`PTO2_SCOPE` output is byte-identical to the previous codegen-driven behavior.

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SplitIncoreOrch`, `CallDirectionsResolved` |
| Produced | `RuntimeScopesMaterialized` |
| Invalidated | — |
