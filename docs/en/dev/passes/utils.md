# Shared Pass Utilities

Reusable utilities in `include/pypto/ir/transforms/utils/` for passes.

## Variable Collectors (`var_collectors.h`)

**Header:** `#include "pypto/ir/transforms/utils/var_collectors.h"`
**Namespace:** `pypto::ir::var_collectors`

### Quick Reference

| Utility | What it collects |
| ------- | ---------------- |
| `VarDefUseCollector` | Def sites AND use sites in a single pass. `.GetAllVarRefs()` returns the union. |
| `VarDefCollector` | Alias for `VarDefUseCollector` — read `.var_defs`. |
| `VarUseCollector` | Alias for `VarDefUseCollector` — read `.var_uses`. |
| `VarRefCollector` | Alias for `VarDefUseCollector` — use `.GetAllVarRefs()`. |
| `CollectStmtDefinedVars()` | Vars visible after a single statement. Non-recursive. |
| `CollectVarDefsInOrder()` | Same scope as VarDefCollector but ordered (DFS). Appends to output vector or returns new one. |
| `CollectAssignDefs()` | AssignStmt var\_ only (no loop vars). Recursive. |
| `CollectTypeVars()` | Vars in type shapes (dynamic dims). Walks type tree. |
| `VisitTypeExprFields()` | Dispatch visitor over type expr fields. |
| `GetSortedVarRefs()` | Deterministic sort by name + ID. |

### Choosing the Right Collector

```text
What do you need?
  |
  |-- Both defs and uses in one pass? ------> VarDefUseCollector
  |-- All vars referenced in a subtree? ----> VarDefUseCollector (.GetAllVarRefs())
  |-- Only vars defined in a subtree? ------> VarDefCollector (read .var_defs)
  |-- Only vars used (read, not defined)? --> VarUseCollector (read .var_uses)
  |-- Only AssignStmt definitions? ----------> CollectAssignDefs()
  |-- Vars output by a single statement? ---> CollectStmtDefinedVars()
  |-- Ordered list of defs (DFS)? -----------> CollectVarDefsInOrder()
  |-- Dynamic shape vars from types? -------> CollectTypeVars()
  |-- Deterministic iteration order? -------> GetSortedVarRefs()
```

### Semantic Differences

**VarDefUseCollector** collects both *definitions* and *uses* in a single traversal:

| Statement | `var_defs` | `var_uses` |
| --------- | ---------- | ---------- |
| `AssignStmt` | `var_` | RHS `value_` |
| `ForStmt` | `loop_var_`, `iter_args_`, `return_vars_` | `start_`, `stop_`, `step_`, `chunk_size_`, initValues |
| `WhileStmt` | `iter_args_`, `return_vars_` | `condition_`, initValues |
| `IfStmt` | `return_vars_` | `condition_` |

**CollectAssignDefs** collects only `AssignStmt::var_` — useful for
SSA analysis where loop variables and return variables are handled
separately.

**CollectStmtDefinedVars** is non-recursive and collects vars
*visible after* a single statement. It includes `AssignStmt::var_`
and control-flow `return_vars_` but excludes `loop_var_` and
`iter_args_` (which are scoped to the loop body).

For valid IR, `GetAllVarRefs()` (var\_defs ∪ var\_uses) equals the set
of all variable pointers in the subtree.  `VarRefCollector` is a
backward-compatible alias for `VarDefUseCollector`.

### Usage Examples

```cpp
#include "pypto/ir/transforms/utils/var_collectors.h"

using namespace pypto::ir;

// Example 1: Find vars defined in a scope but used after it
var_collectors::VarDefUseCollector collector;
collector.VisitStmt(scope_body);

// var_defs: vars defined inside the scope
// var_uses: vars used (read) inside the scope
// Inputs = uses not satisfied by local defs:
for (const Var* use : collector.var_uses) {
  if (!collector.var_defs.count(use)) {
    // 'use' comes from the enclosing scope — it's an input
  }
}

// Example 2: Classify assigned vars for SSA conversion
auto assigned = var_collectors::CollectAssignDefs(loop_body);
for (const Var* var : assigned) {
  if (outer_scope.count(var)) {
    // Loop-carried variable — needs iter_arg
  }
}

// Example 3: Build definition index with deterministic ordering
auto stmt_defs = var_collectors::CollectStmtDefinedVars(stmt);
for (const auto* def :
     var_collectors::GetSortedVarRefs(stmt_defs)) {
  def_index[def] = stmt_position;
}

// Example 4: Collect dynamic shape vars from function params
for (const auto& param : func->params_) {
  auto type_vars =
      var_collectors::CollectTypeVars(param->GetType());
  // type_vars has Var nodes like N, M in Tensor[[N, M], FP32]
}
```

### Type Expression Visitors

`VisitTypeExprFields(visitor, type)` dispatches a visitor over all
expression fields in a type: `TensorType::shape_`,
`TensorView::{valid_shape, stride}`, `TileType::shape_`,
`TileView::{valid_shape, stride, start_offset}`, and `TupleType`
elements (recursively). Use this when you need to find dynamic
dimension variables embedded in type annotations.

`CollectTypeVars(type)` is a convenience wrapper that returns all
`Var` pointers found in a type's expression fields.

## Other Shared Utilities

| Header | Utilities |
| ------ | --------- |
| `transform_utils.h` | `SubstituteExpr/Stmt`, `CollectDefVars`, `FindYieldStmt`, `FlattenToStmts` |
| `loop_state_repair.h` | `BuildDefMap`, loop rebuild helpers, `StripDeadIterArgs` |
| `scope_outline_utils.h` | `VarCollector`, `StoreTargetCollector`, `ScopeOutliner` |
| `auto_name_utils.h` | SSA name generation, rename maps, name parsing |
| `parent_stmt_analysis.h` | Parent-child statement mapping |
| `dead_code_elimination.h` | Dead code removal within functions |
