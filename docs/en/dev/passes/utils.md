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
| `ForStmt` | `loop_var_`, `return_vars_`, `iter_args_` | — | bounds, `chunk_config_->size`, initValues |
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

## MemRef Collector (`memref_collectors.h`)

**Header:** `#include "pypto/ir/transforms/utils/memref_collectors.h"`
**Namespace:** `pypto::ir::memref_collectors`

### Quick Reference

| Utility | What it collects |
| ------- | ---------------- |
| `MemRefWithSpaceCollector` | Unique (MemRef, MemorySpace) pairs from TileType variables. Class-based for multi-visit use. |
| `CollectMemRefsWithSpace()` | All (MemRef, MemorySpace) pairs from a statement. |
| `CollectNonDDRMemRefsWithSpace()` | Non-DDR (MemRef, MemorySpace) pairs from a statement. |
| `CollectShapedTypeMemRefs()` | MemRefPtrs from any ShapedType (Tensor or Tile) in an expression. |
| `CollectUsedBasePtrs()` | Raw base Ptr pointers from MemRefs in TileType/TensorType variables. |

### Usage Examples

```cpp
#include "pypto/ir/transforms/utils/memref_collectors.h"

using namespace pypto::ir;

// Collect all MemRefs with their memory spaces
auto memrefs = memref_collectors::CollectMemRefsWithSpace(func->body_);

// Collect non-DDR MemRefs (e.g., for tile.alloc emission)
auto non_ddr = memref_collectors::CollectNonDDRMemRefsWithSpace(func->body_);

// Multi-visit: collect from both params and body
memref_collectors::MemRefWithSpaceCollector collector(/*skip_ddr=*/true);
for (const auto& param : func->params_) collector.VisitExpr(param);
collector.VisitStmt(func->body_);
// Results in collector.memrefs

// Collect from expressions (works with both TensorType and TileType)
auto expr_memrefs = memref_collectors::CollectShapedTypeMemRefs(expr);

// Raw base Ptr set for fast membership checks (unused alloc detection)
auto used = memref_collectors::CollectUsedBasePtrs(func->body_);
```

## Other Shared Utilities

| Header | Utilities |
| ------ | --------- |
| `transform_utils.h` | `Substitute`, `CollectDefVars`, `FindYieldStmt`, `FlattenToStmts`, `IsComputeTensorOp` |
| `loop_state_repair.h` | `BuildDefMap`, loop rebuild helpers, `StripDeadIterArgs` |
| `scope_outline_utils.h` | `VarCollector`, `StoreTargetCollector`, `ScopeOutliner`, `ScopeKindAbsenceVerifier` |
| `auto_name_utils.h` | SSA name generation, rename maps, name parsing |
| `parent_stmt_analysis.h` | Parent-child statement mapping |
| `dead_code_elimination.h` | Dead code removal within functions |
