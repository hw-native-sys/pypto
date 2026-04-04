# 共享 Pass 工具函数

`include/pypto/ir/transforms/utils/` 中的可复用工具。

## 变量收集器 (`var_collectors.h`)

**头文件:** `#include "pypto/ir/transforms/utils/var_collectors.h"`
**命名空间:** `pypto::ir::var_collectors`

### 快速参考

| 工具 | 收集内容 |
| ---- | -------- |
| `VarDefUseCollector` | 定义点和使用点，单次遍历。`.GetAllVarRefs()` 返回并集。 |
| `VarDefCollector` | `VarDefUseCollector` 的别名 — 读取 `.var_defs`。 |
| `VarUseCollector` | `VarDefUseCollector` 的别名 — 读取 `.var_uses`。 |
| `VarRefCollector` | `VarDefUseCollector` 的别名 — 使用 `.GetAllVarRefs()`。 |
| `CollectStmtDefinedVars()` | 语句后可见的变量。非递归。 |
| `CollectVarDefsInOrder()` | 同 VarDefCollector 但有序（DFS）。追加到输出 vector 或返回新 vector。 |
| `CollectAssignDefs()` | 仅 AssignStmt var\_（不含循环变量）。递归。 |
| `CollectTypeVars()` | 类型形状中的变量（动态维度）。遍历类型树。 |
| `VisitTypeExprFields()` | 在类型表达式字段上分派 visitor。 |
| `GetSortedVarRefs()` | 按名称+ID 确定性排序。 |

### 选择合适的收集器

```text
你需要什么？
  |
  |-- 单次遍历同时获取定义和使用？ ------> VarDefUseCollector
  |-- 子树中所有引用的变量？ ------------> VarDefUseCollector（.GetAllVarRefs()）
  |-- 仅定义的变量？ --------------------> VarDefCollector（读取 .var_defs）
  |-- 仅使用（读取，非定义）的变量？ ----> VarUseCollector（读取 .var_uses）
  |-- 仅 AssignStmt 定义？ ----------> CollectAssignDefs()
  |-- 单条语句输出的变量？ -----------> CollectStmtDefinedVars()
  |-- 有序定义列表（DFS）？ ----------> CollectVarDefsInOrder()
  |-- 类型中的动态形状变量？ ---------> CollectTypeVars()
  |-- 确定性迭代顺序？ --------------> GetSortedVarRefs()
```

### 语义差异

**VarDefUseCollector** 在单次遍历中同时收集*定义*和*使用*：

| 语句 | `var_defs` | `var_uses` |
| ---- | ---------- | ---------- |
| `AssignStmt` | `var_` | 右值 `value_` |
| `ForStmt` | `loop_var_`、`iter_args_`、`return_vars_` | `start_`、`stop_`、`step_`、`chunk_size_`、initValues |
| `WhileStmt` | `iter_args_`、`return_vars_` | `condition_`、initValues |
| `IfStmt` | `return_vars_` | `condition_` |

**CollectAssignDefs** 仅收集 `AssignStmt::var_` — 适用于 SSA
分析，循环变量和返回变量需单独处理。

**CollectStmtDefinedVars** 非递归，收集单条语句*之后可见*的变量。
包含 `AssignStmt::var_` 和控制流 `return_vars_`，但不包含
`loop_var_` 和 `iter_args_`（它们的作用域仅限循环体内）。

对于合法 IR，`GetAllVarRefs()`（var\_defs ∪ var\_uses）等于子树中
所有变量指针的集合。`VarRefCollector` 是 `VarDefUseCollector` 的
向后兼容别名。

### 使用示例

```cpp
#include "pypto/ir/transforms/utils/var_collectors.h"

using namespace pypto::ir;

// 示例 1：单次遍历确定作用域的输入变量
var_collectors::VarDefUseCollector collector;
collector.VisitStmt(scope_body);

// var_defs: 作用域内定义的变量
// var_uses: 作用域内使用（读取）的变量
// 输入 = 使用但未在本地定义的变量：
for (const Var* use : collector.var_uses) {
  if (!collector.var_defs.count(use)) {
    // 'use' 来自外层作用域 — 是输入
  }
}

// 示例 2：为 SSA 转换分类赋值变量
auto assigned = var_collectors::CollectAssignDefs(loop_body);
for (const Var* var : assigned) {
  if (outer_scope.count(var)) {
    // 循环携带变量 — 需要 iter_arg
  }
}

// 示例 3：使用确定性排序构建定义索引
auto stmt_defs = var_collectors::CollectStmtDefinedVars(stmt);
for (const auto* def :
     var_collectors::GetSortedVarRefs(stmt_defs)) {
  def_index[def] = stmt_position;
}

// 示例 4：从函数参数中收集动态形状变量
for (const auto& param : func->params_) {
  auto type_vars =
      var_collectors::CollectTypeVars(param->GetType());
  // type_vars 包含 Tensor[[N, M], FP32] 中的 N、M
}
```

### 类型表达式访问器

`VisitTypeExprFields(visitor, type)` 在类型的所有表达式字段上
分派 visitor：`TensorType::shape_`、
`TensorView::{valid_shape, stride}`、`TileType::shape_`、
`TileView::{valid_shape, stride, start_offset}` 以及 `TupleType`
元素（递归）。

`CollectTypeVars(type)` 是一个便捷包装器，返回类型表达式字段中
所有 `Var` 指针。

## 其他共享工具

| 头文件 | 工具 |
| ------ | ---- |
| `transform_utils.h` | `SubstituteExpr/Stmt`、`CollectDefVars`、`FindYieldStmt`、`FlattenToStmts` |
| `loop_state_repair.h` | `BuildDefMap`、循环重建辅助函数、`StripDeadIterArgs` |
| `scope_outline_utils.h` | `VarCollector`、`StoreTargetCollector`、`ScopeOutliner` |
| `auto_name_utils.h` | SSA 名称生成、重命名映射、名称解析 |
| `parent_stmt_analysis.h` | 父子语句映射 |
| `dead_code_elimination.h` | 函数内死代码消除 |
