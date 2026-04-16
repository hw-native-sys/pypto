# OutlineHierarchyScopes Pass

将非 `CORE_GROUP` 的 `HierarchyScopeStmt` 区域提取为独立的 `Opaque` 函数，
并把作用域的 level/role 元信息带到提取出的函数上。

## 概述

该 Pass 把每个 `level_` 不为 `Level.CORE_GROUP` 的 `HierarchyScopeStmt`
变换为独立的 `Function` 定义，并将原作用域替换为对该函数的 `Call`。提取出
的函数类型恒为 `FunctionType::Opaque`；父函数的类型保持不变。

| 作用域 `level_` | 本 Pass 是否处理 | 提取出的函数类型 | 父函数类型（Pass 后） |
| --------------- | ---------------- | ---------------- | --------------------- |
| `Level.HOST`、`Level.CLUSTER`、`Level.GLOBAL`、... | 是 | `FunctionType::Opaque` | 保持不变 |
| `Level.CORE_GROUP` | **否 —— 有意跳过** | *（由 [`OutlineIncoreScopes`](06-outline_incore_scopes.md) 处理）* | *（由下一个 Pass 提升为 `Orchestration`）* |

`CORE_GROUP` 作用域在本 Pass 中被有意保留；紧接着执行的
[`OutlineIncoreScopes`](06-outline_incore_scopes.md) 会把它们提取为
`Function(InCore)` 并将父函数由 `Opaque` 提升为 `Orchestration`。

**前置条件**：

- 输入 IR 必须为 SSA 形式（需先运行 `ConvertToSSA`）。本 Pass 保留
  （产生）SSA 形式。
- 处理 `Opaque` 函数。已经为 `Orchestration`、`InCore`、`AIC`、`AIV`、
  `Group` 的函数保持不变。

**使用时机**：在 `ConvertToSSA`/`FlattenCallExpr` 之后运行，当 IR 中包含
非 `CORE_GROUP` 层级的 `with pl.at(level=...):` 作用域需要提取为独立辅助
函数时使用。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::OutlineHierarchyScopes()` | `passes.outline_hierarchy_scopes()` | 程序级 |

**工厂函数**：

```cpp
Pass OutlineHierarchyScopes();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_hierarchy_scopes()
program_outlined = outline_pass(program)
```

## 算法

1. **扫描 Hierarchy 作用域**：在每个 `Opaque` 函数体中查找所有 `level_`
   **不为** `CORE_GROUP` 的 `HierarchyScopeStmt` 节点。
2. **分析输入/输出**：复用 scope_outline_utils 辅助工具计算外部定义、内部
   使用的变量（输入）以及内部定义、外部使用的变量（输出）。
3. **创建提取函数**：将作用域体提取为新的 `Function`：
   - 参数 = 输入变量
   - 返回值 = 输出变量
   - 函数体 = 作用域体
   - `func_type_` = `Opaque`
   - 将 `role_` 元信息复制到函数 attrs。
4. **替换作用域**：将原 `HierarchyScopeStmt` 替换为对提取函数的 `Call` +
   绑定返回值的若干 `AssignStmt`。
5. **保持父函数类型**：本 Pass 不修改父函数的 `func_type_`。对
   `CORE_GROUP` 作用域的父函数提升由
   [`OutlineIncoreScopes`](06-outline_incore_scopes.md) 负责。
6. **加入程序**：将提取出的函数前置到程序的函数列表中。

**命名规则**：`{原函数名}_{level}_{计数器}`（例如 `main_host_0`、
`main_global_0`）。若 `HierarchyScopeStmt.name_hint` 非空，则直接使用该
name_hint。

## 示例

### 非 CORE_GROUP 层级（HOST）

**之前**：

```python
@pl.program
class Before:
    @pl.function  # Opaque
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST):
            y = helper(x)
        return y
```

**之后**（父函数仍为 `Opaque`，提取函数也是 `Opaque`）：

```python
@pl.program
class After:
    @pl.function  # 未变
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = self.main_host_0(x)
        return y

    @pl.function  # Opaque
    def main_host_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = helper(x)
        return y
```

### 多输出

```python
with pl.at(level=pl.Level.HOST):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
# out_a 与 out_b 都在作用域之后被使用
x = out_a + out_b
```

提取后的函数体变为：

```python
out_a, out_b = self.main_host_0(a, b, out)  # 多返回值
x = out_a + out_b
```

### CORE_GROUP 作用域会被跳过

```python
@pl.function  # Opaque
def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):   # <-- 本 Pass 不处理
        tile = pl.load(x, [0], [64])
        result = pl.store(tile, [0], x)
    return result
```

本 Pass 会把上述 `CORE_GROUP` 作用域原样保留。下一个流水线 Pass
[`OutlineIncoreScopes`](06-outline_incore_scopes.md) 会把它提取为
`Function(InCore)` 并把父函数提升为 `Orchestration`。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineHierarchyScopes();
```

**实现文件**：`src/ir/transforms/outline_hierarchy_scopes.cpp`

- 使用公共 `scope_outline_utils` 计算输入/输出
- 对每个非 `CORE_GROUP` 作用域构造新的 `Function(Opaque)`
- 将 `role_` 元信息复制到提取函数的 attrs
- 从不修改父函数的 `func_type_`

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_hierarchy_scopes", &pass::OutlineHierarchyScopes,
           "Outline non-CORE_GROUP HierarchyScopeStmt regions into Opaque functions");
```

**测试**：`tests/ut/ir/transforms/test_outline_hierarchy_scopes.py`

- 测试非 `CORE_GROUP` 作用域 → `Opaque` 函数 + 父函数不变
- 测试 `CORE_GROUP` 作用域保持原样不被处理
- 测试输入/输出分析
- 测试同一父函数中多个非 `CORE_GROUP` 作用域
- 测试 SSA 保留

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | `SSAForm` |
| 产生 | `SSAForm` |
| 失效 | — |

`HierarchyOutlined` 现由紧随其后的
[`OutlineIncoreScopes`](06-outline_incore_scopes.md) Pass 产生，它负责
处理剩余的 `CORE_GROUP` 作用域。

## 流水线位置

```text
... → ConvertToSSA → NormalizeStmtStructure → FlattenCallExpr →
OutlineHierarchyScopes → OutlineIncoreScopes → OutlineClusterScopes →
ConvertTensorToTileOps → ...
```
