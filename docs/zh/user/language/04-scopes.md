# 作用域与放置

代码在哪里执行：把一个区域标记为设备工作、把协同调度的核归组、把一个 kernel 铺开到多个 block。

> **前置**：[函数与程序](01-functions.md) 与 [编程模型 § 执行模型](../03-programming-model.md#执行模型)。

## Concept

放置回答一个问题：**这段代码由哪块硬件执行。**

一共四个构造，都用 `with`（或 `for`）书写，且可以组合：

| 构造 | 把工作放到 |
| ---- | ---------- |
| `pl.at` | 一个核组 —— 把区域标记为设备工作 |
| `pl.cluster` | 同一个物理 cluster —— 让 Cube 与 Vector kernel 协同调度 |
| `pl.spmd` | `n` 个 block —— 同一个 kernel，每个 block 跑一份 |
| `pl.split_aiv` | 两条 AIV lane —— 把一个区域切开分给两条 |

`pl.at` 的替代写法是单独写一个 `@pl.jit.incore` 函数再调用它。两者产生的东西一样：编译时 `pl.at` 正是被 outline 成这样一个函数。区域短、且就该写在原地时用作用域；值得起个名字、或者要被多处调用时写成独立函数。

放置与**定序**不是一回事 —— 定序说的是本任务开始之前什么必须先完成。运行时是从 [类型](00-types.md) 里的参数方向、以及每个任务触碰的缓冲区推导出来的。那套机制和手工干预它的接口会有独立章节；本页只讲代码落在哪里。

## Quickstart：把一个区域标记为设备工作

```python
import pypto.language as pl

@pl.jit
def scale(
    x: pl.Tensor[[256, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out = pl.mul(x, 2.0)
    return out
```

| 元素 | 作用 |
| ---- | ---- |
| `@pl.jit` | 入口 —— 控制面，它自己不能放算子 |
| `with pl.at(level=pl.Level.CORE_GROUP)` | 把该区域标记为设备工作，给算子一个合法的容身之处 |
| `pl.mul(x, 2.0)` | 在核组上执行 |

去掉 `pl.at`，这个 kernel 会编译失败并报 `Misplaced tensor op ... should be inside InCore block` —— 见 [函数与程序](01-functions.md)。

## Mechanics

### `pl.at`

`level=` 选择层级。`pl.Level.CORE_GROUP` 是产生 InCore 作用域的那个，该区域在编译时会变成一个独立的 kernel 函数。

两个可选关键字塑造这个被 outline 的 kernel：

| 关键字 | 含义 |
| ------ | ---- |
| `optimizations=[pl.split(mode)]` | 被 outline kernel 的跨核切分模式 |
| `optimizations=[pl.cross_core_slot(slot_num=N)]` | 自动跨核管道的环深度 |
| `name_hint="..."` | outline 出的函数名 |

`optimizations=` 的条目必须在调用点内联书写 —— 解析器读的是 AST，因此用变量拼出来的列表不被接受。`pl.split` 与 `pl.cross_core_slot` 彼此正交、可自由组合：一个切分工作，一个给通道定尺寸。

```python
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                          pl.cross_core_slot(slot_num=4)]):
    ...
```

省略 `cross_core_slot` 就保持默认环深度：单方向活跃时 8 个 slot，双方向都活跃时每方向 4 个。

### SPMD

`pl.spmd(n)` 让同一个 kernel 在 `n` 个 block 上执行。两种形式，区别在于函数体是否读 block 索引：

```python
# Dispatch form — the body launches a kernel defined elsewhere.
with pl.spmd(4):
    out = self.kernel(a, b, out)

# Loop form — the body is auto-outlined and `i` binds the block index.
for i in pl.spmd(4):
    off = i * 128
    out = pl.store(pl.add(pl.load(a, [off, 0], [128, 128]),
                          pl.load(b, [off, 0], [128, 128])), [off, 0], out)
```

一个既不读 block 索引、也不派发 kernel 的 `with pl.spmd(n):` 体会被拒绝 —— 那样每个 block 都在做完全相同的工作。

当涉及硬 `pl.system.syncall` 时，请按设备实际规模而非字面量来定启动规模：传 `pl.system.available_cluster_count()`（混合或纯 cube kernel）或 `pl.system.available_aiv_count()`（纯 vector kernel），并在调用点内联书写。

### Cluster 与 AIV lane

`with pl.cluster():` 把 AIC 与 AIV kernel 归组，让它们在同一个物理 cluster 上协同调度，产生一个 `Group` 函数。

`for aiv_id in pl.split_aiv(2, mode=...):` 把一个区域切开分给两条 AIV lane。它属于混合 kernel 编程（AIC 与 AIV 在同一个函数里协作），教程章会端到端地讲。

## Edge Cases

> **致命陷阱：** `pl.spmd` 是一个断言，不是请求。你是在告诉编译器这些 block 彼此独立。如果它们其实不独立，结果是竞态，而不是一条诊断。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`Misplaced tensor op ... should be inside InCore block`** | 算子直接写在 `@pl.jit` 体内 | 包进 `with pl.at(level=pl.Level.CORE_GROUP):` |
| **`with pl.spmd(n):` 体被拒绝** | 它既不读 block 索引也不派发 kernel | 读 `pl.tile.get_block_idx()`，或调用一个 kernel |
| **`optimizations=` 被拒绝** | 用变量拼出来的 —— 解析器读的是 AST | 在调用点内联书写该列表 |
| **printed IR 无法被重新解析** | 设备规模查询在使用前被绑定到了名字上 | 在使用处内联书写该调用 |

## See Also

- [函数与程序](01-functions.md) —— `pl.at` 的替代写法：独立的 `@pl.jit.incore` 函数。
- [控制流](02-control-flow.md) —— 包含这些作用域的循环。
- [内存与数据搬运](03-memory.md) —— 被放置的代码拿缓冲区做什么。
- [OutlineIncoreScopes](../../dev/passes/09-outline_incore_scopes.md) —— `pl.at` 如何变成函数。
- [ExpandMixedKernel](../../dev/passes/22-expand_mixed_kernel.md) —— `pl.split` 驱动的是什么。
