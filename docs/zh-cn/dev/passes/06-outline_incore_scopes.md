# OutlineIncoreScopes Pass

将 `level_ == CORE_GROUP` 的 `HierarchyScopeStmt` 区域提取为独立的
`Function(InCore)` 定义，并把外层父函数由 `Opaque` 提升为 `Orchestration`。

## 概述

该 Pass 专门处理 `HierarchyScopeStmt` 的 `CORE_GROUP` 形式 —— 即由
`with pl.at(level=pl.Level.CORE_GROUP):` 引入的 per-core-group 内核区域。
对每个此类作用域，它都会提取出一个新的 `Function`，`func_type_` 为
`FunctionType::InCore`，并将原作用域替换为对该函数的 `Call`。只要从某个
父函数中提取出至少一个 `CORE_GROUP` 作用域，就把该父函数的 `func_type_`
由 `Opaque` 提升为 `Orchestration`。

本 Pass 是 [`OutlineHierarchyScopes`](05-outline_hierarchy_scopes.md) 在
`CORE_GROUP` 方向的对应 Pass，后者只处理非 `CORE_GROUP` 层级，生成
`Function(Opaque)` 且不修改父函数类型。

| 作用域 `level_` | 提取出的函数类型 | 父函数类型（Pass 后） |
| --------------- | ---------------- | --------------------- |
| `Level.CORE_GROUP` | `FunctionType::InCore` | `Opaque` 提升为 `Orchestration` |
| 其他层级 | *（本 Pass 不处理；已由 `OutlineHierarchyScopes` 提取）* | — |

当 `CORE_GROUP` 作用域携带 `split_` 优化提示时，会把该提示复制到提取出的
`InCore` 函数 attrs 中，供下游 Pass（特别是
[`ExpandMixedKernel`](11-expand_mixed_kernel.md)）在决定如何拆分 AIC /
AIV 核时使用。

**前置条件**：

- 输入 IR 必须为 SSA 形式（需先运行 `ConvertToSSA`）。本 Pass 保留
  （产生）SSA 形式。
- 期望 `OutlineHierarchyScopes` 已经运行过，因此当前只剩下 `CORE_GROUP`
  的 `HierarchyScopeStmt` 节点需要处理。
- 仅处理 `Opaque` 函数（其中可能残留 `CORE_GROUP` 作用域）。已经为
  `Orchestration`、`InCore`、`AIC`、`AIV`、`Group` 的函数保持不变。

**使用时机**：在 [`OutlineHierarchyScopes`](05-outline_hierarchy_scopes.md)
之后、[`OutlineClusterScopes`](07-outline_cluster_scopes.md) 之前运行。
本 Pass 完成后，`HierarchyOutlined` 属性成立：`Opaque` / `Orchestration`
函数中不再残留任何 `HierarchyScopeStmt` 节点。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::OutlineIncoreScopes()` | `passes.outline_incore_scopes()` | 程序级 |

**工厂函数**：

```cpp
Pass OutlineIncoreScopes();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_incore_scopes()
program_outlined = outline_pass(program)
```

## 算法

1. **扫描 CORE_GROUP 作用域**：在每个 `Opaque` 函数体中查找所有 `level_`
   为 `CORE_GROUP` 的 `HierarchyScopeStmt` 节点。
2. **分析输入/输出**：复用 scope_outline_utils 辅助工具计算外部定义、内部
   使用的变量（输入）以及内部定义、外部使用的变量（输出）。
3. **创建 InCore 函数**：将作用域体提取为新的 `Function`：
   - 参数 = 输入变量
   - 返回值 = 输出变量
   - 函数体 = 作用域体
   - `func_type_` = `InCore`
   - 将 `role_` 复制到函数 attrs
   - 若作用域携带 `split_` 优化提示，将其复制到函数的 `split` attr
     （由 `ExpandMixedKernel` 消费）
4. **替换作用域**：将原 `HierarchyScopeStmt` 替换为对提取出 InCore 函数的
   `Call` + 绑定返回值的若干 `AssignStmt`。
5. **父函数提升**：若父函数中至少有一个 `CORE_GROUP` 作用域被提取，则将
   该父函数由 `Opaque` 重标记为 `Orchestration`。
6. **加入程序**：将提取出的 InCore 函数前置到程序的函数列表中。

**命名规则**：`{原函数名}_core_group_{计数器}`（例如
`main_core_group_0`）。提取出的 InCore 函数在 attrs 中使用 `_incore_`
风格的名称后缀，在打印的 IR 中便于识别。若
`HierarchyScopeStmt.name_hint` 非空，则直接使用该 name_hint。

## 示例

### CORE_GROUP → InCore + Orchestration

**之前**（假设 `OutlineHierarchyScopes` 已完成，非 CORE_GROUP 作用域已经
被提取；CORE_GROUP 作用域仍内联在 `main` 中）：

```python
@pl.program
class Before:
    @pl.function  # Opaque
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = x + 1

        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(y, [0], [64])
            tile_sq = pl.mul(tile, tile)
            result_tile = tile_sq + 1
            result = pl.store(result_tile, [0], x)

        z = result + 2
        return z
```

**之后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Orchestration)  # 已升级
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = x + 1
        result = self.main_core_group_0(y, x)  # 调用提取出的 InCore 函数
        z = result + 2
        return z

    @pl.function(type=pl.FunctionType.InCore)  # 提取出
    def main_core_group_0(self, y: pl.Tensor[[64], pl.FP32],
                          x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        tile = pl.load(y, [0], [64])
        tile_sq = pl.mul(tile, tile)
        result_tile = tile_sq + 1
        result = pl.store(result_tile, [0], x)
        return result
```

### 带 split 提示的 CORE_GROUP

```python
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
    ...
```

提取出的 `InCore` 函数 attrs 中会携带该 `split` 提示，供后续
`ExpandMixedKernel` 读取以决定 AIC+AIV 拆分方式。

### 多输出

```python
with pl.at(level=pl.Level.CORE_GROUP):
    a_tile = pl.load(a, [0], [64])
    b_tile = pl.load(b, [0], [64])
    c_tile = pl.add(a_tile, b_tile)
    out_a = pl.store(c_tile, [0], out)
    out_b = pl.mul(c_tile, 2.0)
# out_a 与 out_b 都在作用域之后被使用
x = out_a + out_b
```

提取后，父函数体变为：

```python
out_a, out_b = self.main_core_group_0(a, b, out)  # 多返回值
x = out_a + out_b
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineIncoreScopes();
```

**实现文件**：`src/ir/transforms/outline_incore_scopes.cpp`

- 使用公共 `scope_outline_utils` 计算输入/输出
- 对每个 `CORE_GROUP` 作用域构造新的 `Function(InCore)`
- 将 `role_` / `split_` 元信息复制到提取函数的 attrs
- 当从某父函数中至少提取出一个 `CORE_GROUP` 作用域时，将该父函数
  由 `Opaque` 重标记为 `Orchestration`

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_incore_scopes", &pass::OutlineIncoreScopes,
           "Outline CORE_GROUP HierarchyScopeStmt regions into Function(InCore) "
           "and promote the parent function to Orchestration");
```

**测试**：`tests/ut/ir/transforms/test_outline_incore_scopes.py`

- 测试 `CORE_GROUP` 作用域 → `InCore` 函数 + 父函数升级为 `Orchestration`
- 测试 `split_` 透传到提取出的 InCore 函数
- 测试输入/输出分析
- 测试同一父函数中多个 `CORE_GROUP` 作用域
- 测试 SSA 保留

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | `SSAForm` |
| 产生 | `SSAForm`, `HierarchyOutlined` |
| 失效 | — |

`HierarchyOutlined` 由本 Pass 产生（而非
[`OutlineHierarchyScopes`](05-outline_hierarchy_scopes.md)）：两次 outline
Pass 全部结束后，`Opaque`/`Orchestration` 函数中不再残留任何
`HierarchyScopeStmt` 节点。

## 流水线位置

```text
... → ConvertToSSA → NormalizeStmtStructure → FlattenCallExpr →
OutlineHierarchyScopes → OutlineIncoreScopes → OutlineClusterScopes →
ConvertTensorToTileOps → ...
```
