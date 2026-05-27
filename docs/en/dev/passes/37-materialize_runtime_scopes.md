# MaterializeRuntimeScopes Pass

Inserts explicit AUTO `RuntimeScopeStmt` nodes into Orchestration functions so
that PTO orchestration codegen emits `PTO2_SCOPE()` 1:1 from the IR instead of
deriving the scope structure from `for` / `if` statements.

## Overview

The simpler runtime wraps regions of an orchestration routine in `PTO2_SCOPE()`
blocks (auto dependency tracking via the OverlapMap). Historically the
orchestration codegen decided *where* to emit those blocks from the statement
structure: it implicitly wrapped the whole function body, every `ForStmt` body,
and every `IfStmt` branch body in a `PTO2_SCOPE()` — suppressing the wrap inside
a manual scope, because the runtime forbids AUTO nested in MANUAL.

That embedded scope *policy* inside the printer. This pass moves the policy into
the IR. For every `FunctionType::Orchestration` function it inserts explicit
AUTO `RuntimeScopeStmt` (`manual_ = false`) nodes:

- wrapping the entire function body, and
- wrapping each `ForStmt` body and each `IfStmt` then/else body,

while skipping insertion anywhere inside a manual `RuntimeScopeStmt`. Codegen
then emits `PTO2_SCOPE` **only** from `RuntimeScopeStmt` nodes — staying 1:1 with
the IR (see [orchestration codegen](../codegen/01-orchestration_codegen.md)).

The inserted scopes have a DSL surface, `with pl.auto_scope():`, so the IR
round-trips through the printer/parser like any other construct.

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

## Algorithm

`InsertAutoScopeMutator` walks each Orchestration function body:

1. On entering a **manual** `RuntimeScopeStmt`, a depth counter is incremented;
   AUTO insertion is suppressed while the counter is non-zero (the runtime
   forbids AUTO nested in MANUAL). AUTO scopes do not suppress nesting.
2. For each `ForStmt`, the body is replaced with `RuntimeScopeStmt(manual=false,
   body)` unless already AUTO-wrapped.
3. For each `IfStmt`, the then/else bodies are each wrapped the same way.

After the mutator runs, the whole function body is wrapped in one outermost AUTO
scope (mirroring the always-on outermost `PTO2_SCOPE()` codegen used to emit).
The wrap is idempotent — an already-AUTO body is left as-is.

| Source | Action |
| ------ | ------ |
| Orchestration function body | Wrapped in one AUTO `RuntimeScopeStmt` |
| `ForStmt` body (not in manual scope) | Wrapped in AUTO `RuntimeScopeStmt` |
| `IfStmt` then/else body (not in manual scope) | Each wrapped in AUTO `RuntimeScopeStmt` |
| Any body inside a manual `RuntimeScopeStmt` | Left bare |
| Non-Orchestration function | Returned unchanged |

## Example

```python
# Before
@pl.function(type=pl.FunctionType.Orchestration)
def orch(self, a, out):
    for i in pl.range(4):
        out = self.kernel(a, out)
    return out
```

```python
# After MaterializeRuntimeScopes
@pl.function(type=pl.FunctionType.Orchestration)
def orch(self, a, out):
    with pl.auto_scope():            # function body
        for i in pl.range(4):
            with pl.auto_scope():    # loop body
                out = self.kernel(a, out)
        return out
```

A trailing return-var `yield` stays inside the scope; the printer recurses
through the AUTO scope so the `var = pl.yield_(...)` assignment LHS is preserved,
and the parser treats the yield inside `pl.auto_scope()` as the enclosing
for/if's return-var.

## Verification

**Tests**: `tests/ut/ir/transforms/test_materialize_runtime_scopes.py` (function
body + for/if wrapping, manual-scope suppression, idempotency, non-Orchestration
untouched) and `tests/ut/language/parser/test_auto_scope_parsing.py`
(`pl.auto_scope()` parse / round-trip / nesting restriction). The full
orchestration codegen suite (`tests/ut/codegen/test_orchestration_codegen.py`)
verifies the emitted `PTO2_SCOPE` output is byte-identical to the previous
codegen-driven behavior.

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SplitIncoreOrch`, `CallDirectionsResolved` |
| Produced | `RuntimeScopesMaterialized` |
| Invalidated | — |
