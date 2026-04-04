# 共享 Pass 工具函数

`include/pypto/ir/transforms/utils/` 中的可复用工具。

## 变量收集器 (`var_collectors.h`)

**头文件:** `#include "pypto/ir/transforms/utils/var_collectors.h"`
**命名空间:** `pypto::ir::var_collectors`

### 快速参考

| 工具 | 收集内容 |
| ---- | -------- |
| `VarDefUseCollector` | 单次遍历收集所有定义、使用、仅赋值定义和有序定义。 |
| `CollectStmtDefinedVars()` | 语句后可见的变量。非递归。 |
| `CollectTypeVars()` | 类型形状中的变量（动态维度）。遍历类型树。 |
| `VisitTypeExprFields()` | 在类型表达式字段上分派 visitor。 |
| `GetSortedVarRefs()` | 按名称+ID 确定性排序。 |

### VarDefUseCollector 字段

| 字段 | 内容 |
| ---- | ---- |
| `var_defs` | 所有定义点（无序集合）。 |
| `var_uses` | 所有使用点（无序集合）。 |
| `var_defs_ordered` | 定义点的 DFS 前序遍历（vector）。 |
| `var_assign_defs` | 仅 AssignStmt 左值（无序集合）。 |
| `GetAllVarRefs()` | 返回 `var_defs ∪ var_uses`。 |

### 各语句填充内容

| 语句 | `var_defs` / `var_defs_ordered` | `var_assign_defs` | `var_uses` |
| ---- | ------------------------------- | ----------------- | ---------- |
| `AssignStmt` | `var_` | `var_` | 右值 `value_` |
| `ForStmt` | `loop_var_`、`return_vars_`、`iter_args_` | — | 边界、`chunk_size_`、initValues |
| `WhileStmt` | `return_vars_`、`iter_args_` | — | `condition_`、initValues |
| `IfStmt` | `return_vars_` | — | `condition_` |

### 使用示例

```cpp
#include "pypto/ir/transforms/utils/var_collectors.h"

using namespace pypto::ir;

// 单次遍历同时获取定义、使用和有序定义
var_collectors::VarDefUseCollector collector;
collector.VisitStmt(scope_body);

// 输入 = 使用但未在本地定义的变量
for (const Var* use : collector.var_uses) {
  if (!collector.var_defs.count(use)) {
    // 'use' 来自外层作用域
  }
}

// SSA：查找仅赋值定义（不含循环变量、iter_args）
for (const Var* v : collector.var_assign_defs) {
  // 循环携带状态或逃逸变量的候选
}

// 确定性定义排序用于重命名映射
for (const Var* def : collector.var_defs_ordered) {
  rename_map[def] = next_name();
}
```

### 类型表达式访问器

`VisitTypeExprFields(visitor, type)` 在类型的所有表达式字段上
分派 visitor。`CollectTypeVars(type)` 是便捷包装器，返回所有
`Var` 指针。这些操作类型（非 IR 语句），因此保留为自由函数。

## 其他共享工具

| 头文件 | 工具 |
| ------ | ---- |
| `transform_utils.h` | `SubstituteExpr/Stmt`、`CollectDefVars`、`FindYieldStmt`、`FlattenToStmts` |
| `loop_state_repair.h` | `BuildDefMap`、循环重建辅助函数、`StripDeadIterArgs` |
| `scope_outline_utils.h` | `VarCollector`、`StoreTargetCollector`、`ScopeOutliner` |
| `auto_name_utils.h` | SSA 名称生成、重命名映射、名称解析 |
| `parent_stmt_analysis.h` | 父子语句映射 |
| `dead_code_elimination.h` | 函数内死代码消除 |
