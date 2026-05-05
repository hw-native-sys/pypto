# OutlineHierarchyScopes Pass

Outlines Hierarchy scopes into separate functions that carry `level` and `role` metadata.

## Overview

This pass transforms `ScopeStmt(Hierarchy)` nodes — produced by
`with pl.at(level=..., role=...)` — into separate
`Function(Opaque, level, role)` definitions and replaces each scope with a
Call to the outlined function.

> **DSL note**: `pl.at(level=pl.Level.CORE_GROUP, ...)` is a special case at
> the parser level — it produces an `InCore` (or `AutoInCore`) scope rather
> than a Hierarchy scope, and is handled later by `OutlineIncoreScopes` (see
> `python/pypto/language/dsl_api.py:984`). Hierarchy scopes at `CORE_GROUP`
> can still arise from direct IR construction; the C++ name table below
> covers all `Level` values for completeness.

**Requirements**:

- Input IR must be in SSA form (run `ConvertToSSA` first); SSAForm is preserved
  (produced) by this pass
- Only processes `Opaque` functions; other function types are left unchanged
- Should run **before** `OutlineIncoreScopes` and `OutlineClusterScopes`

**When to use**: Run after `InterchangeChunkLoops` and before the InCore /
Cluster outlining passes when the IR contains `with pl.at(level=..., role=...)`
regions that should be lifted into level/role-tagged functions for downstream
hierarchy-aware lowering.

**Parent function type is preserved.** Unlike `OutlineIncoreScopes`, this pass
does **not** promote the parent to `Orchestration`. Hierarchy is orthogonal to
`FunctionType`: the outlined child stays `Opaque` and carries the level/role on
its function metadata.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::OutlineHierarchyScopes()` | `passes.outline_hierarchy_scopes()` | Program-level |

**Factory function**:

```cpp
Pass OutlineHierarchyScopes();
```

**Python usage**:

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_hierarchy_scopes()
program_outlined = outline_pass(program)
```

## Algorithm

1. **Iterate functions**: For each function in the program, skip non-`Opaque`
   functions and emit them unchanged.
2. **Build symbol table**: Collect parameter types/objects and known names via
   `outline_utils::VarCollector` over the function body.
3. **Outline Hierarchy scopes**: Walk the body with
   `outline_utils::ScopeOutliner` configured for `ScopeKind::Hierarchy`. For
   each scope encountered:
   - Determine inputs (variables defined outside the scope but used inside).
   - Determine outputs (variables defined inside the scope but used after).
   - Recurse into nested Hierarchy scopes (inner scopes are outlined first and
     replaced with calls before the outer scope is lifted).
4. **Create outlined function**: Build a new `Function(FunctionType::Opaque)`
   carrying the scope's `level` and `role`, with parameters = inputs and
   returns = outputs.
5. **Replace scope with Call**: Substitute the original `ScopeStmt(Hierarchy)`
   with a Call to the outlined function plus `AssignStmt`s for each output.
6. **Assemble program**: Prepend all outlined functions before the originals
   and return a new `Program`. Parent function types are unchanged.

## Naming

Outlined functions follow `{parent}_{level}[_{role}]_{counter}`. The level
component is lowercase; the role suffix is omitted when the scope has no role.

| Level enum | Suffix |
| ---------- | ------ |
| `AIV` | `aiv` |
| `AIC` | `aic` |
| `CORE_GROUP` | `core_group` |
| `CHIP_DIE` | `chip_die` |
| `CHIP` | `chip` |
| `HOST` | `host` |
| `CLUSTER_0` | `cluster0` |
| `CLUSTER_1` | `cluster1` |
| `CLUSTER_2` | `cluster2` |
| `GLOBAL` | `global` |

| Role enum | Suffix |
| --------- | ------ |
| `Orchestrator` | `orch` |
| `Worker` | `worker` |

Examples:

- `pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator)` → `main_host_orch_0`
- `pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)` → `main_global_orch_0`
- `pl.at(level=pl.Level.CHIP)` → `main_chip_0`

**Level aliases** (`POD = CLUSTER_0`, `NODE = HOST`, `UMA = CHIP`, etc.)
resolve to the canonical underlying name. For example,
`pl.at(pl.Level.POD)` produces `main_cluster0_0`.

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function  # Opaque function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y
```

**After**:

```python
@pl.program
class After:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main_host_orch_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

    @pl.function  # Opaque (unchanged — parent type is preserved)
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = self.main_host_orch_0(x)
        return y
```

> **Note:** SubWorker scopes (``role=pl.Role.SubWorker``) are not supported as
> inline ``with pl.at(...)`` blocks. Declare a SubWorker via
> ``@pl.function(level=..., role=pl.Role.SubWorker)`` as a self-contained
> function (no ``self`` parameter) inside ``@pl.program``; its body is captured
> as an :class:`InlineStmt` and embedded directly in the IR.

## Nested Hierarchy Example

Nested Hierarchy scopes are outlined recursively. Inner scopes are extracted
first and replaced with calls inside the outer outlined function, producing
chained names like `main_global_orch_0_host_orch_0`.

**Before**:

```python
@pl.program
class Before:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
            with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
        return z
```

**After**:

```python
@pl.program
class After:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main_global_orch_0_host_orch_0(
        self, y: pl.Tensor[[64], pl.FP32]
    ) -> pl.Tensor[[64], pl.FP32]:
        z: pl.Tensor[[64], pl.FP32] = pl.mul(y, y)
        return z

    @pl.function(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)
    def main_global_orch_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        z: pl.Tensor[[64], pl.FP32] = self.main_global_orch_0_host_orch_0(y)
        return z

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        z: pl.Tensor[[64], pl.FP32] = self.main_global_orch_0(x)
        return z
```

`InCore` scopes nested inside a Hierarchy scope are preserved — they are left
untouched here and lifted later by `OutlineIncoreScopes` (which runs against
the resulting Opaque hierarchy functions).

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineHierarchyScopes();
```

**Implementation**: `src/ir/transforms/outline_hierarchy_scopes_pass.cpp`

- Drives `outline_utils::ScopeOutliner` with `ScopeKind::Hierarchy` and
  `FunctionType::Opaque`
- Builds the symbol table via `outline_utils::VarCollector`
- Preserves the parent function type via `MutableCopy`
- Prepends outlined functions to the program's function list

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_hierarchy_scopes", &pass::OutlineHierarchyScopes,
           "Create a pass that outlines Hierarchy scopes into separate level/role functions");
```

**Tests**: `tests/ut/ir/transforms/test_outline_hierarchy_scopes.py`

- Basic single-scope outlining (with and without role)
- Multiple scopes per function (independent counter)
- Nested Hierarchy scopes (recursive outlining and chained naming)
- InCore / Cluster scopes are not touched
- Multiple inputs / multiple outputs / no outputs
- Outlining inside control flow
- Parent function type preserved as `Opaque`
- Level alias resolution (`POD` → `cluster0`)
- Independent counters across functions
- Print → re-parse round-trip
- `OutlineIncoreScopes` runs cleanly on the hierarchy-outlined output
- `HierarchyOutlined` property verifier behaviour

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm |
| Produced | SSAForm, HierarchyOutlined |
| Invalidated | — |

## HierarchyOutlined Property Verifier

The `HierarchyOutlined` IRProperty asserts that no `ScopeStmt(Hierarchy)`
remains anywhere in the program where this pass is responsible — that is, in
`Opaque` functions. The verifier (`HierarchyOutlinedPropertyVerifierImpl`)
walks each `Opaque` function body and reports any leftover Hierarchy scope
with the message `"Hierarchy ScopeStmt found in function (should have been
outlined)"`.

Hierarchy scopes inside non-`Opaque` functions are intentionally **not**
flagged, mirroring the pass's own scope: this pass does not process them, so
the verifier should not require them to be absent.

## Relationship to Sibling Outlining Passes

| Aspect | OutlineHierarchyScopes | OutlineIncoreScopes | OutlineClusterScopes |
| ------ | ---------------------- | ------------------- | -------------------- |
| Scope kind | `ScopeKind::Hierarchy` | `ScopeKind::InCore` | `ScopeKind::Cluster` / standalone `ScopeKind::Spmd` |
| Output function type | `FunctionType::Opaque` (with `level` / `role`) | `FunctionType::InCore` | `FunctionType::Group` / `FunctionType::Spmd` |
| Naming pattern | `{func}_{level}[_{role}]_{n}` | `{func}_incore_{n}` | `{func}_cluster_{n}` / `{func}_spmd_{n}` |
| Promotes parent to | *(unchanged)* | Orchestration | *(unchanged)* |
| Processes | Opaque functions only | Opaque functions only | Opaque + Orchestration |
| Pipeline position | 8 (before InCore / Cluster) | 9 | 10 |
