# Shared Pass Utilities

Reusable utilities in `include/pypto/ir/transforms/utils/` for passes.

## Variable Collector (`var_collectors.h`)

**Header:** `#include "pypto/ir/transforms/utils/var_collectors.h"`
**Namespace:** `pypto::ir::var_collectors`

### Quick Reference

| Utility | What it collects |
| ------- | ---------------- |
| `VarDefUseCollector` | All defs, uses, assign-only defs, and ordered defs in a single pass. |
| `CollectStmtDefinedVars()` | Vars visible after a single statement. Non-recursive. |
| `CollectTypeVars()` | Vars in type shapes (dynamic dims). Walks type tree. |
| `VisitTypeExprFields()` | Dispatch visitor over type expr fields. |
| `GetSortedVarRefs()` | Deterministic sort by name + ID. |

### VarDefUseCollector Fields

| Field | Content |
| ----- | ------- |
| `var_defs` | All def sites (unordered set). |
| `var_uses` | All use sites (unordered set). |
| `var_defs_ordered` | Def sites in DFS pre-order (vector). |
| `var_assign_defs` | AssignStmt LHS only (unordered set). |
| `GetAllVarRefs()` | Returns `var_defs ∪ var_uses`. |

### What Each Statement Populates

| Statement | `var_defs` / `var_defs_ordered` | `var_assign_defs` | `var_uses` |
| --------- | ------------------------------- | ----------------- | ---------- |
| `AssignStmt` | `var_` | `var_` | RHS `value_` |
| `ForStmt` | `loop_var_`, `return_vars_`, `iter_args_` | — | bounds, `chunk_size_`, initValues |
| `WhileStmt` | `return_vars_`, `iter_args_` | — | `condition_`, initValues |
| `IfStmt` | `return_vars_` | — | `condition_` |

### Usage Examples

```cpp
#include "pypto/ir/transforms/utils/var_collectors.h"

using namespace pypto::ir;

// Single traversal gives defs, uses, and ordered defs
var_collectors::VarDefUseCollector collector;
collector.VisitStmt(scope_body);

// Inputs = uses not satisfied by local defs
for (const Var* use : collector.var_uses) {
  if (!collector.var_defs.count(use)) {
    // 'use' comes from the enclosing scope
  }
}

// SSA: find assign-only defs (excludes loop vars, iter_args)
for (const Var* v : collector.var_assign_defs) {
  // candidate for loop-carried state or escaping var
}

// Deterministic def ordering for rename maps
for (const Var* def : collector.var_defs_ordered) {
  rename_map[def] = next_name();
}
```

### Type Expression Visitors

`VisitTypeExprFields(visitor, type)` dispatches a visitor over all
expression fields in a type. `CollectTypeVars(type)` is a convenience
wrapper returning all `Var` pointers found. These operate on types
(not IR statements), so they remain free functions.

## Other Shared Utilities

| Header | Utilities |
| ------ | --------- |
| `transform_utils.h` | `SubstituteExpr/Stmt`, `CollectDefVars`, `FindYieldStmt`, `FlattenToStmts` |
| `loop_state_repair.h` | `BuildDefMap`, loop rebuild helpers, `StripDeadIterArgs` |
| `scope_outline_utils.h` | `VarCollector`, `StoreTargetCollector`, `ScopeOutliner` |
| `auto_name_utils.h` | SSA name generation, rename maps, name parsing |
| `parent_stmt_analysis.h` | Parent-child statement mapping |
| `dead_code_elimination.h` | Dead code removal within functions |
