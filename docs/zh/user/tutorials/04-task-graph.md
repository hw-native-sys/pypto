# 塑形任务图

让运行时执行你想要的那张图，而不是它推出来的那张。

> **前置**：[第一个算子](00-elementwise.md)。
> **配套文件**：`examples/intermediate/07_task_graph.py`。
> **参考**：[任务与定序](../tasks/index.md)。

## 你要做的东西

一个多任务程序，它的依赖图你能有意地声明、收紧、放松。

## 必须内化的那条性质

运行时并不逐条执行你的编排函数。它建起一张依赖图，谁就绪就跑谁。

> **语句顺序什么都不表达。** 一前一后写下的两次派发，只有当**某样东西**这么说了才是有序的 —— 一次运行时看得见的缓冲区重叠，或者一条你声明的边。

默认的边从哪来：运行时记下每个任务触碰了哪些缓冲区、以什么方式触碰（依据每个参数的方向），凡两个任务触碰同一缓冲区就推出一条边。

## 第 1 步：白得的那条边

```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

@pl.jit
def inferred(
    x: pl.Tensor[[128, 128], pl.FP32],
    scratch: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1"):
        scratch = pl.assemble(scratch, pl.add(x, x), [0, 0])       # 写 scratch
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2"):
        out = pl.assemble(out, pl.add(scratch, scratch), [0, 0])   # 读 scratch
    return scratch, out

torch.manual_seed(0)
x = torch.randn(128, 128)
scratch = torch.zeros(128, 128)
out = torch.zeros(128, 128)
inferred(x, scratch, out, config=RunConfig(platform="a2a3sim"))
assert torch.allclose(out, (x + x) + (x + x), rtol=1e-5, atol=1e-5)
```

阶段 1 把 `scratch` 声明为输出，于是它被记为该缓冲区的生产者。阶段 2 读同一块缓冲区，于是推出一条写后读的边。什么都没声明，顺序却有保证。

这是本页什么都不需要的情形。只有当推导在下面两个方向之一出错时，才去用其余部分。

## 第 2 步：推导看不见的边

如果任务 B 必须排在任务 A 之后，而这个理由从不体现为共享缓冲区，那就没有任何东西会推出它。绑定生产者的 TaskId 并点名：

```python
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1") as first:
        scratch = pl.assemble(scratch, pl.add(x, x), [0, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2", deps=[first]) as second:
        out = pl.assemble(out, pl.add(scratch, scratch), [0, 0])
```

`as first` 绑定该区域的 TaskId；`deps=[first]` 让消费者等它。

**显式边与自动边是叠加的。** 最终等待集合是并集：

```text
最终等待集合  =  自动跟踪的边  ∪  显式 deps=
```

所以 `deps=` 是一个在普通 auto 作用域里就能用的精修工具。它不要求、也不蕴含手动作用域。

## 第 3 步：推出来但并不真实的边

相反的失效：OverlapMap 基于缓冲区工作，所以写同一张量**互不相交区域**的兄弟任务看起来是重叠的，于是被串行化。三种退出方式，由窄到宽：

| 构造 | 退出范围 |
| ---- | -------- |
| `pl.at(..., no_dep_args=[t])` | 单个张量，仅对单个任务 |
| `pl.create_tensor(..., manual_dep=True)` | 单个张量，其整个生命周期 |
| `with pl.manual_scope():` | 区域内每一个任务 —— 所有边归你声明 |

优先选能表达该断言的最窄那一种。

> **致命陷阱：** 以上每一种都是编译器无法检验的断言。如果那些区域其实并非互不相交，你就删掉了一条真实的边、买来一个竞态 —— 与压根没声明它是同一类缺陷。

## 第 4 步：跳过工作，以及提前开始

还有两个旋钮，而它们不是同一类东西：

**`predicate=`** —— 当运行期的值这么说时，这个任务根本不要跑。在**派发点**求值，所以取到的值是最新的，而不必在编排期等待。可用于 `pl.submit`、`pl.spmd_submit` 与 `pl.spmd` —— `pl.at` 上没有。

```python
with pl.spmd(4, deps=[gather_tid], predicate=(row_count[e] > 0)) as tid:
    ...
```

可表达的只有 `tensor[indices] OP int 字面量` —— 单个比较。更复杂的条件请在前一个 kernel 里归约成一个门控值。

**`allow_early_resolve=True`** —— 纯调度提示。调度器可以在本任务完成前把它的消费者预置好。它改变时序，从不改变结果。

| 构造 | 改变正确性 | 改变跑什么 | 改变时序 |
| ---- | ---------- | ---------- | -------- |
| 退出跟踪（`no_dep`、`manual_dep`） | **是** | 否 | 是 |
| `predicate=` | 否 | **是** | 是 |
| `allow_early_resolve=` | 否 | 否 | **是** |

拿错行是最常见的错误。只有第一行能损坏你的结果。

## 第 5 步：等待多个生产者

一个 TaskId 只指代一个任务。要等待一个循环产出的全部生产者，把它们收进一个 `pl.TASK_ID` 的 `pl.array` 并把数组传进去：

```python
tids = pl.array.create(branches, pl.TASK_ID)
for branch in pl.parallel(branches):
    out, tid = pl.submit(self.producer, data, branch, out)
    tids[branch] = tid
out, _ = pl.submit(self.consumer, data, out, deps=[tids])
```

片段 —— `pl.submit` 要求被调方是 `self.<kernel>`，所以这种写法存在于 `@pl.program` 类里。从 `@pl.jit` 出发请用带 `deps=` 的 `pl.at` / `pl.spmd` 块。见 [声明一条边](../tasks/02-submit.md)。

`pl.system.task_dummy(deps=[...])` 做同样的汇聚但不做任何工作，返回一个代表若干生产者的 TaskId。

## Edge Cases

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **多次运行结果不同** | 两个必须有序的任务没有任何东西表达这个顺序 | 用 `deps=` 声明 |
| **加一条 print 就正确了** | 变的是时序，不是语义 | 那条缺失的边仍然缺失 |
| **本该重叠的工作串行执行** | 推出了一次并非真实依赖的重叠 | 让该实参退出跟踪 —— 第 3 步 |
| **`pl.at` 拒绝 `predicate=`** | `pl.at` 没有谓词 | 改用 `pl.spmd` 或某种 submit 写法 |
| **消费者只等到了最后一个生产者** | 复用了一个 TaskId 而没有收集 | 收进 `pl.TASK_ID` 数组 |

## 下一步

[调度调优](05-scheduling-tuning.md) —— 图已经是你想要的了，接下来看运行时实际拿它做了什么。
