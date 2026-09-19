# OutlineClusterScopes Pass

将 Cluster 作用域提取为 Group 函数，并将独立的 Spmd 作用域提取为 Spmd 函数。

## 概述

该 Pass 将 `ClusterScopeStmt` 节点变换为独立的 `Function(Group)` 定义，并将原作用域替换为对提取函数的调用。它还会把未嵌套在 Cluster 内部的 standalone `SpmdScopeStmt` 提取为 `Function(Spmd)`。Group 函数表示共享同一物理集群 (Cluster) 资源的协同调度 AIC（Cube）+ AIV（Vector）内核组，而 standalone 调度语义（`core_num` / `sync_start`）则挂到合成出的调度点上 —— 参见[启动规格 (launch spec) 的存放位置](#启动规格-launch-spec-的存放位置)。

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
4. **提取 standalone Spmd 作用域**：将每个 standalone Spmd 作用域体提取为 `Function(func_type=Spmd)`，并把 `core_num` / `sync_start` 挂到合成出的**调度点**（Call attrs；`as tid` 作用域则用 `Submit` 字段）——绝不挂在被提取的函数上
5. **展开 Group 内嵌 Spmd**：对于 `pl.cluster(): with pl.spmd(...): ...`，保留单一 Group 函数，把 `core_num` / `sync_start` 挂到其**调度点**（由于该规格是从被调用方内部提取的，需经调度点实参回译），并在 Group 上打上自包含的 `spmd_unwrapped` 标记
6. **替换作用域**：将作用域语句替换为对提取函数的调用 + 输出赋值
7. **添加到程序**：将提取的函数前置到程序的函数列表中

**命名规则**：`{原函数名}_cluster_{计数器}`（例如 `main_cluster_0`）

**参数化显式返回**：与 `OutlineIncoreScopes` 相同，只要某个
tensor 输出经由参数回写，外提的 Group/Spmd 函数就返回自身参数——store 目标
直接返回参数，其余输出通过共享的 `return_lineage` 工具追踪；只有 kernel 内
部分配的输出保留其 SSA 值。这维持 `ReturnParamsExplicit` 不变量，使编排代
码生成按指针同一性建立返回值到实参的映射。

## 示例

**之前**：

```python
@pl.program
class Before:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.cluster():
            with pl.at(level=pl.Level.CORE_GROUP):
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
        return y
```

**之后**：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.Group)
    def main_cluster_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
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
    @pl.function(type=pl.FunctionType.Spmd)
    def main_spmd_0(self, x: pl.Tensor[[64], pl.FP32],
                    out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        out = self.kernel(x, out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[64], pl.FP32],
             out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        out = self.main_spmd_0(x, out, attrs={"core_num": 4, "sync_start": True})
        return out
```

### 启动规格 (launch spec) 的存放位置

`core_num` / `sync_start` 挂在**调度点 (dispatch)** 上，而不是被提取出的函数上：

| 调度形态 | 载体 |
| -------- | ---- |
| 普通 `Call`——独立的 `with pl.spmd(...)` / `for i in pl.spmd(...)`，且不带任何 Submit 专属元数据 | `attrs["core_num"]`（`ExprPtr`）与 `attrs["sync_start"]`（`bool`，仅为真时发出） |
| `Submit`——独立的 `with pl.spmd(...)` / `for i in pl.spmd(...)`，带 `as tid`，或带 `deps=` / `allow_early_resolve=` / `predicate=` 之一（此时 outliner 会合成 TaskId Var） | 一等字段 `Submit::core_num_` / `sync_start_` |
| `pl.cluster(): with pl.spmd(...)` 产生的 `Group` | 同上，挂在调度点；Group 上只保留 `spmd_unwrapped` 标记 |

`core_num` 是在*调度方*函数作用域中求值的表达式，可能引用调用者的局部标量（例如
`m = pl.tensor.dim(a, 0)` 之后的 `pl.spmd(m // 16)`）。而 `Function` 是封闭作用域
(closed scope)：其引用的每个 Var 都必须能解析到自身的参数或函数体内的定义。因此把该表达式
存到被调用函数上，会产生一个引用了自己未绑定名字的 Function —— 打印出来是一个在任何函数体
绑定这些名字之前就被求值的装饰器（程序无法被重新解析），并且对逐函数遍历的 visitor / mutator
不可见。把启动规格保留在调度点即可同时消除这三个问题，且无需任何特殊处理：现有的 Call attr
打印/解析编解码器本就能往返一个通用 `ExprPtr`，其引用的 Var 对 def-use 与 DCE 而言也只是
普通的局部使用。

Group 情形多一步：其规格是从被调用方*内部*提取的，其中的计数引用的是 Group 的**形参**
（cluster 提取时把调用者的标量捕获成了形参）。`LaunchSpecStamper` 通过调度点做
`params_[i] -> args_[i]` 映射回译，使该 attr 引用的 Var 在调用点确实存活。

留在 Group 上的是 `spmd_unwrapped`（`bool`）—— 它确实属于函数作用域：它陈述的是该函数
自身函数体的性质，不引用任何外部内容。它告诉启动点的消费者：调度这个 Group 启动的是其
函数体所调用的 kernel，而不是把 Group 自身当作 mixed kernel —— 这正是占用率校验器过去
依据「是否存在 `core_num` attr」所作的区分。

Orchestration 代码生成通过 `EffectiveLaunchSpec` 读取该规格：优先取调度点自身的 attrs；
启动函数回退路径保留，用于手写或反序列化 IR 中仍在函数上写常量 `core_num` 的情形。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现文件**：`src/ir/transforms/outline_cluster_scopes_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_outline_cluster_scopes.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | SSAForm |
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
