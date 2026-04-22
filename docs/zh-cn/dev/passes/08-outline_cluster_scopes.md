# OutlineClusterScopes Pass

将 Cluster 作用域提取为 Group 函数，并将独立的 Spmd 作用域提取为 Spmd 函数。

## 概述

该 Pass 将 `ClusterScopeStmt` 节点变换为独立的 `Function(Group)` 定义，并将原作用域替换为对提取函数的调用。它还会把未嵌套在 Cluster 内部的 standalone `SpmdScopeStmt` 提取为 `Function(Spmd)`。Group 函数表示共享同一物理集群 (Cluster) 资源的协同调度 AIC（Cube）+ AIV（Vector）内核组，而 Spmd 函数保留 standalone 调度所需的 `core_num` / `sync_start` 语义。

**前置条件**：

- 输入 IR 必须为静态单赋值 (SSA) 形式（需先运行 ConvertToSSA）
- 仅处理 Opaque 和 Orchestration 函数

**使用时机**：在 `OutlineIncoreScopes` 之后运行，当 IR 包含需要提取的 `with pl.cluster():` 作用域或 standalone `with pl.spmd(...):` / `for i in pl.spmd(...)` 作用域时使用。loop-form 是解析器对 `SpmdScopeStmt(body=InCoreScopeStmt(...))` 的语法糖；`OutlineIncoreScopes` 先把 InCore 体提取为独立函数，使 Spmd 体变成单次函数调用，之后本 pass 再把它提升为 `Function(Spmd)`。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::OutlineClusterScopes()` | `passes.outline_cluster_scopes()` | 程序级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

outline_pass = passes.outline_cluster_scopes()
program_outlined = outline_pass(program)
```

## 算法

1. **扫描 Cluster 作用域**：在 Opaque/Orchestration 函数中查找所有 `ClusterScopeStmt` 节点
2. **提取 Cluster 作用域**：将每个 Cluster 作用域体提取为 `Function(func_type=Group)`
3. **扫描 standalone Spmd 作用域**：在变换后的函数体中查找所有未嵌套在 Cluster 内部的 `SpmdScopeStmt` 节点
4. **提取 standalone Spmd 作用域**：将每个 standalone Spmd 作用域体提取为 `Function(func_type=Spmd)`，并把 `core_num` / `sync_start` 复制到函数 attrs
5. **展开 Group 内嵌 Spmd**：对于 `pl.cluster(): with pl.spmd(...): ...`，保留单一 Group 函数，并把 `core_num` / `sync_start` 提升到 Group attrs
6. **替换作用域**：将作用域语句替换为对提取函数的调用 + 输出赋值
7. **添加到程序**：将提取的函数前置到程序的函数列表中

**命名规则**：`{原函数名}_cluster_{计数器}`（例如 `main_cluster_0`）

## 示例

**之前**：

```python
@pl.program
class Before:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.cluster():
            with pl.incore():
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y
```

**之后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Group)
    def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.incore():
            y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = self.main_cluster_0(x)
        return y
```

注意：Cluster 内部的 InCore 作用域在提取的 Group 函数中被保留。可以先运行 `OutlineIncoreScopes` 提取 InCore 作用域再进行聚簇，也可以之后在 Group 函数内提取。

## Standalone Spmd 示例

**之前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64], pl.FP32],
               out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        tile = pl.load(x, [0], [64])
        out = pl.store(pl.add(tile, tile), [0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(4, sync_start=True):
            out = self.kernel(x, out)
        return out
```

**之后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Spmd, attrs={"core_num": 4, "sync_start": True})
    def main_spmd_0(self, x: pl.Tensor[[64], pl.FP32],
                    out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        out = self.kernel(x, out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        out = self.main_spmd_0(x, out)
        return out
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现文件**：`src/ir/transforms/outline_cluster_scopes_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_outline_cluster_scopes.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | TypeChecked, SSAForm |
| 产生 | SSAForm, ClusterOutlined |
| 失效 | — |

## 与 OutlineIncoreScopes 的关系

| 方面 | OutlineIncoreScopes | OutlineClusterScopes |
| ---- | ------------------- | -------------------- |
| 作用域类型 | `ScopeKind::InCore` | `ScopeKind::Cluster` / standalone `ScopeKind::Spmd` |
| 输出函数类型 | `FunctionType::InCore` | `FunctionType::Group` / `FunctionType::Spmd` |
| 命名模式 | `{func}_incore_{n}` | `{func}_cluster_{n}` / `{func}_spmd_{n}` |
| 提升父函数为 | Orchestration | *（不变）* |
| 处理对象 | 仅 Opaque 函数 | Opaque + Orchestration |
