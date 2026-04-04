# 共享 Pass 工具函数

`include/pypto/ir/transforms/utils/` 中的可复用工具。

## 变量收集器 (`var_collectors.h`)

**头文件:** `#include "pypto/ir/transforms/utils/var_collectors.h"`
**命名空间:** `pypto::ir::var_collectors`

### 快速参考

| 工具 | 收集内容 |
| ---- | -------- |
| `VarDefCollector` | 所有定义点（AssignStmt var、loop\_var、iter\_args、return\_vars）。递归。 |
| `VarRefCollector` | 所有变量引用（定义和使用点）。递归。 |
| `VarUseCollector` | 仅使用点（跳过 AssignStmt 左值）。递归。 |
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
  |-- 子树中所有定义的变量？ ---------> VarDefCollector
  |-- 子树中所有引用的变量？ ---------> VarRefCollector
  |-- 仅使用（读取，非定义）的变量？ --> VarUseCollector
  |-- 仅 AssignStmt 定义？ ----------> CollectAssignDefs()
  |-- 单条语句输出的变量？ -----------> CollectStmtDefinedVars()
  |-- 有序定义列表（DFS）？ ----------> CollectVarDefsInOrder()
  |-- 类型中的动态形状变量？ ---------> CollectTypeVars()
  |-- 确定性迭代顺序？ --------------> GetSortedVarRefs()
```

### 语义差异

**VarDefCollector** 收集语句引入的*所有*变量：

| 语句 | 收集的变量 |
| ---- | ---------- |
| `AssignStmt` | `var_` |
| `ForStmt` | `loop_var_`、`iter_args_`、`return_vars_` |
| `WhileStmt` | `iter_args_`、`return_vars_` |
| `IfStmt` | `return_vars_` |

**CollectAssignDefs** 仅收集 `AssignStmt::var_` — 适用于 SSA
分析，循环变量和返回变量需单独处理。

**CollectStmtDefinedVars** 非递归，收集单条语句*之后可见*的变量。
包含 `AssignStmt::var_` 和控制流 `return_vars_`，但不包含
`loop_var_` 和 `iter_args_`（它们的作用域仅限循环体内）。

**VarRefCollector** 与 **VarUseCollector**：

- `VarRefCollector` 捕获子树中*每个* Var 指针，
  包括定义点的左值变量
- `VarUseCollector` 跳过 `AssignStmt::var_`（被定义的变量），
  仅捕获读取点

### 使用示例

```cpp
#include "pypto/ir/transforms/utils/var_collectors.h"

using namespace pypto::ir;

// 示例 1：查找在作用域中定义但在之后使用的变量
var_collectors::VarRefCollector refs;
refs.VisitStmt(after_scope_stmts);

var_collectors::VarDefCollector defs;
defs.VisitStmt(scope_body);

std::vector<VarPtr> outputs;
for (const Var* def : defs.var_defs) {
  if (refs.var_refs.count(def)) {
    outputs.push_back(/* resolve VarPtr */);
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
