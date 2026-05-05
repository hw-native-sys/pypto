# OutlineHierarchyScopes Pass

将 Hierarchy 作用域提取为携带 `level` 与 `role` 元数据的独立函数。

## 概述

该 Pass 将 `ScopeStmt(Hierarchy)` 节点（由 `with pl.at(level=..., role=...)`
生成）变换为独立的 `Function(Opaque, level, role)` 定义，并将原作用域替换为
对提取函数的调用。

> **DSL 说明**：`pl.at(level=pl.Level.CORE_GROUP, ...)` 在解析层是一个特例
> —— 它生成的是 `InCore`（或 `AutoInCore`）作用域，而非 Hierarchy 作用域，
> 后续由 `OutlineIncoreScopes` 处理（参见
> `python/pypto/language/dsl_api.py:984`）。`CORE_GROUP` 层级的 Hierarchy
> 作用域仍可能通过直接构造 IR 产生；下方 C++ 命名表覆盖所有 `Level` 枚举值
> 以保持完整。

**前置条件**：

- 输入 IR 必须为静态单赋值 (SSA) 形式（需先运行 `ConvertToSSA`）；该 Pass
  保持（产生）SSAForm
- 仅处理 `Opaque` 函数；其他类型的函数保持不变
- 应在 `OutlineIncoreScopes` 与 `OutlineClusterScopes` **之前**运行

**使用时机**：在 `InterchangeChunkLoops` 之后、InCore / Cluster 提取 Pass
之前运行。当 IR 中包含 `with pl.at(level=..., role=...)` 区域，需要将其提升
为带有层级 / 角色标签的函数以供后续层级感知 (hierarchy-aware) 下沉时使用。

**父函数类型保持不变**。与 `OutlineIncoreScopes` 不同，该 Pass **不会**将父
函数提升为 `Orchestration`。Hierarchy 与 `FunctionType` 是正交的：提取出的
子函数仍为 `Opaque`，仅在函数元数据上携带 `level` / `role`。

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

1. **遍历函数**：对程序中的每个函数，跳过非 `Opaque` 函数并原样输出。
2. **建立符号表**：通过 `outline_utils::VarCollector` 在函数体上收集形参的
   类型 / 对象与已知名字。
3. **提取 Hierarchy 作用域**：用配置为 `ScopeKind::Hierarchy` 的
   `outline_utils::ScopeOutliner` 遍历函数体。对遇到的每个作用域：
   - 确定输入（在作用域外定义、在作用域内使用的变量）。
   - 确定输出（在作用域内定义、在作用域之后仍被使用的变量）。
   - 递归处理嵌套的 Hierarchy 作用域（先提取内层并替换为调用，再处理外层）。
4. **创建提取函数**：构造新的 `Function(FunctionType::Opaque)`，携带作用域的
   `level` 与 `role`，参数 = 输入，返回值 = 输出。
5. **替换为 Call**：用对提取函数的 Call 加上对应的 `AssignStmt` 替换原始的
   `ScopeStmt(Hierarchy)`。
6. **组装程序**：将所有提取出的函数前置在原始函数之前，返回新的 `Program`。
   父函数的类型保持不变。

## 命名

提取函数的命名遵循 `{父函数}_{level}[_{role}]_{计数器}`。`level` 部分为小写；
当作用域无 `role` 时省略 `role` 后缀。

| Level 枚举 | 后缀 |
| ---------- | ---- |
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

| Role 枚举 | 后缀 |
| --------- | ---- |
| `Orchestrator` | `orch` |
| `Worker` | `worker` |

示例：

- `pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator)` → `main_host_orch_0`
- `pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator)` → `main_global_orch_0`
- `pl.at(level=pl.Level.CHIP)` → `main_chip_0`

**层级别名**（如 `POD = CLUSTER_0`、`NODE = HOST`、`UMA = CHIP` 等）会解析为
其规范的底层名称。例如 `pl.at(pl.Level.POD)` 生成 `main_cluster0_0`。

## 示例

**之前**：

```python
@pl.program
class Before:
    @pl.function  # Opaque function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y
```

**之后**：

```python
@pl.program
class After:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def main_host_orch_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

    @pl.function  # Opaque（保持不变 —— 父函数类型不被提升）
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = self.main_host_orch_0(x)
        return y
```

> **注意**：SubWorker 作用域（``role=pl.Role.SubWorker``）不再支持以内联
> ``with pl.at(...)`` 形式书写。请使用
> ``@pl.function(level=..., role=pl.Role.SubWorker)`` 在 ``@pl.program`` 内
> 声明一个自包含函数（不带 ``self`` 参数），其函数体会以 :class:`InlineStmt`
> 的形式直接嵌入 IR。

## 嵌套层级示例

嵌套的 Hierarchy 作用域会被递归提取。内层作用域先被提取并替换为对应的 Call，
然后才处理外层，从而生成形如 `main_global_orch_0_host_orch_0` 的链式名称。

**之前**：

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

**之后**：

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

嵌套在 Hierarchy 作用域内的 `InCore` 作用域会被保留 —— 该 Pass 不会触碰它们；
它们将在随后的 `OutlineIncoreScopes` 中（针对生成的 Opaque 层级函数）被提取。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass OutlineHierarchyScopes();
```

**实现文件**：`src/ir/transforms/outline_hierarchy_scopes_pass.cpp`

- 以 `ScopeKind::Hierarchy` 与 `FunctionType::Opaque` 驱动
  `outline_utils::ScopeOutliner`
- 通过 `outline_utils::VarCollector` 构建符号表
- 通过 `MutableCopy` 保持父函数类型不变
- 将提取出的函数前置加入程序的函数列表

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("outline_hierarchy_scopes", &pass::OutlineHierarchyScopes,
           "Create a pass that outlines Hierarchy scopes into separate level/role functions");
```

**测试**：`tests/ut/ir/transforms/test_outline_hierarchy_scopes.py`

- 基本单作用域提取（带与不带 `role`）
- 同一函数中的多个作用域（独立计数器）
- 嵌套 Hierarchy 作用域（递归提取与链式命名）
- InCore / Cluster 作用域不被触碰
- 多输入 / 多输出 / 无输出
- 控制流内部的作用域提取
- 父函数类型保持为 `Opaque`
- 层级别名解析（`POD` → `cluster0`）
- 跨函数的独立计数器
- 打印 → 再解析的往返一致性
- `OutlineIncoreScopes` 在层级提取后的输出上能正常运行
- `HierarchyOutlined` 属性验证器行为

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | SSAForm |
| 产生 | SSAForm, HierarchyOutlined |
| 失效 | — |

## HierarchyOutlined 属性验证器

`HierarchyOutlined` IRProperty 断言：在该 Pass 负责处理的范围 —— 即
`Opaque` 函数 —— 中，不再存在任何 `ScopeStmt(Hierarchy)`。验证器
（`HierarchyOutlinedPropertyVerifierImpl`）遍历每个 `Opaque` 函数体，并对
任何遗留的 Hierarchy 作用域报告诊断信息：
`"Hierarchy ScopeStmt found in function (should have been outlined)"`。

非 `Opaque` 函数中的 Hierarchy 作用域**不会**被标记，这与该 Pass 自身的处理
范围一致：既然该 Pass 不处理这些函数，验证器也不应要求它们的作用域被消除。

## 与其他 Outline Pass 的关系

| 方面 | OutlineHierarchyScopes | OutlineIncoreScopes | OutlineClusterScopes |
| ---- | ---------------------- | ------------------- | -------------------- |
| 作用域类型 | `ScopeKind::Hierarchy` | `ScopeKind::InCore` | `ScopeKind::Cluster` / standalone `ScopeKind::Spmd` |
| 输出函数类型 | `FunctionType::Opaque`（带 `level` / `role`） | `FunctionType::InCore` | `FunctionType::Group` / `FunctionType::Spmd` |
| 命名模式 | `{func}_{level}[_{role}]_{n}` | `{func}_incore_{n}` | `{func}_cluster_{n}` / `{func}_spmd_{n}` |
| 提升父函数为 | *（不变）* | Orchestration | *（不变）* |
| 处理对象 | 仅 Opaque 函数 | 仅 Opaque 函数 | Opaque + Orchestration |
| 流水线位置 | 8（在 InCore / Cluster 之前） | 9 | 10 |
