# HoistScopeLocalAllocs Pass

Marks each `tensor.create` that sits directly in a `pl.manual_scope` body and is
*enclosing-scope-valid* with the `hoistable_alloc` attribute, so the
orchestration codegen hoists the buffer's declaration one C++ scope out instead
of recovering the hoist set from emit-time indent arithmetic (issue #1697).

## Overview

A `pl.manual_scope` lowers to a `PTO2_SCOPE(PTO2ScopeMode::MANUAL) { ... }` C++
block. A manual scope is a **scheduling** region, not a **storage** region: a
buffer allocated inside it may be read by a task placed *after* the block. If the
buffer's declaration stayed inside the block, the C++ local would die at the
closing brace and the after-scope reader would fail to compile.

`alloc_tensors(...)` carries no scheduling dependency, so emitting it one level
out — ahead of the `PTO2_SCOPE(MANUAL) {` header — is semantically inert and
keeps the buffer in scope both inside the block and at every after-scope reader.

This pass makes "which creates get hoisted" an explicit IR fact. Previously the
orchestration codegen recovered it at emit time from two signals: a
`scope_hoist_sink_` pointer (are we buffering a manual scope?) and an indent
comparison (`IsAtManualScopeBodyIndent()` — is this the scope's *direct* body?),
plus an on-the-fly `ShapeDependsOnLocalVars` analysis. The indent heuristic and
the shape analysis are exactly the analysis the ["strict 1-to-1
codegen"](../codegen/01-orchestration_codegen.md) contract asks to move into a
pass.

**When to use**: after
[`MaterializeRuntimeScopes`](41-materialize_runtime_scopes.md), so the
manual-scope boundary is an explicit `RuntimeScopeStmt(manual=True)` edge rather
than an indent level. Only `FunctionType::Orchestration` functions are touched.

## What gets marked

A `tensor.create` `Call` is stamped `hoistable_alloc = True` exactly when:

1. Its `AssignStmt` is a **direct** child of a `RuntimeScopeStmt(manual=True)`
   body (a create nested in a for/if *within* the scope is left in place — it
   belongs to the loop/branch C++ scope, not the manual block).
2. Its result shape references **no** `Var` defined inside that manual-scope body
   — i.e. the buffer is *enclosing-scope-valid*. A shape dimension that depends
   on a scope-local value could not be evaluated one level out, so such a create
   stays put.

Nested manual scopes are handled: each scope stamps only its own direct-body
creates, computed against its own body-local def set, so a buffer hoisted out of
an inner scope lands in the outer scope's body.

The pass never rewrites structure — it only appends the attr to the create
`Call`. The physical relocation (routing the `alloc_tensors` batch into the
enclosing scope, and remapping aliasing kernel outputs) stays in codegen; this
pass only supplies the decision.

## Stamped attribute

| Key | Type | Meaning |
| --- | ---- | ------- |
| `hoistable_alloc` | `bool` | `True` on a `tensor.create` `Call` = hoist this buffer's declaration out of its enclosing `pl.manual_scope`. Absent = leave in place. |

The key round-trips through the printer/parser (see the `python_printer`
op-attr allowlist and `ast_parser._parse_op_attrs`).

## Example

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        scratch = pl.create_tensor([64], dtype=pl.FP32)  # hoistable
        a, a_tid = pl.submit(self.k, x, scratch)
    b, _ = pl.submit(self.k, a, x)                        # reads `a` after the scope
    return b
```

After the pass the `scratch` create carries `attrs={"hoistable_alloc": True}`.
Codegen emits its `alloc_tensors` batch *before* the `PTO2_SCOPE(MANUAL) {`
header, so the `const Tensor& scratch = ...;` declaration stays live for readers
placed after the block.

A create nested in a `for` within the scope, or one whose shape references a
value computed inside the scope, is **not** marked and stays inside the block.

## Pass properties

| - | Properties |
| - | ---------- |
| Required | `CallDirectionsResolved`, `RuntimeScopesMaterialized` |
| Produced | `HoistableAllocsMarked`, `RuntimeScopesMaterialized` |
| Invalidated | — |

`HoistableAllocsMarked` is a codegen precondition (see
`VerifyOrchestrationCodegenPreconditions`) and has a registered property
verifier: an enclosing-scope-valid `tensor.create` in a manual-scope body with no
`hoistable_alloc` attr means the pass never ran, and codegen would leave the
buffer declared inside the block — an after-scope reader would then reference an
out-of-scope C++ local (#1697).

## See also

- [MaterializeRuntimeScopes](41-materialize_runtime_scopes.md) — materializes the `RuntimeScopeStmt(manual=True)` edge this pass keys on
- [ClassifyIterArgCarry](42-classify_iter_arg_carry.md) — the sibling attr-stamping pass that runs just before
- [Orchestration codegen](../codegen/01-orchestration_codegen.md) — the consumer of the stamped attr
- [Pass manager](00-pass_manager.md)
