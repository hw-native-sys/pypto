# 作用域与任务

工作被放在哪里执行，以及运行时真正执行的那张依赖图是怎么成形的。

> **前置**：[函数与程序](01-functions.md) 与 [编程模型 § 执行模型](../03-programming-model.md#执行模型)。

## Concept

两个彼此独立的问题，常被混为一谈，因为它们都用 `with` 书写：

**放置（placement）** —— 这段代码在哪块硬件上跑。`pl.at` 把一个区域放到核组上，`pl.cluster` 把协同调度的 Cube 与 Vector kernel 归组，`pl.spmd` 把一个 kernel 铺开到多个 block，`pl.split_aiv` 把一个区域切到两条 AIV lane 上。

**定序（ordering）** —— 本任务开始之前什么必须先完成。这由*运行时作用域*（`pl.scope` / `pl.manual_scope`）加上每个任务的边（`deps=`、`no_dep`、`predicate=`）决定。

定序的默认行为是自动的。运行时在 OverlapMap 里跟踪每个任务的缓冲区，用 [类型](00-types.md) 里的参数方向从访问重叠中推导出边。你会去用手动接口，是因为自动跟踪推出了一条并不真实存在的边（把本可并行的工作串行化了），或者推不出一条真实存在的边（因为那个关系并不体现为缓冲区重叠）。

有一条关键性质，只说一次：**语句顺序什么都不表达。** 一前一后写下的两次派发，只有当某样东西 —— 一次缓冲区重叠，或一条显式边 —— 这么说了，它们才是有序的。

## Quickstart：先放置，再加一条显式边

```python
import pypto.language as pl

@pl.jit
def two_stage(
    x: pl.Tensor[[256, 128], pl.FP32],
    scratch: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
    out: pl.Out[pl.Tensor[[256, 128], pl.FP32]],
):
    # Placement: run this region on a core group, and name its producer TaskId.
    with pl.at(level=pl.Level.CORE_GROUP) as t1:
        scratch = pl.mul(x, 2.0)

    # Ordering: this dispatch waits on t1 explicitly.
    with pl.at(level=pl.Level.CORE_GROUP, deps=[t1]):
        out = pl.add(scratch, scratch)

    return scratch, out
```

这里 `scratch` 缓冲区是重叠的，自动跟踪本来也会推出同一条边 —— `deps=[t1]` 是冗余且无害的。当那个关系**不是**缓冲区重叠时，它才开始承重。

## Mechanics

### 放置

| 构造 | 产生 | 用于 |
| ---- | ---- | ---- |
| `with pl.at(level=...)` | InCore 作用域（`CORE_GROUP`）或 Hierarchy 作用域 | 不写独立函数就把一个区域标为设备工作 |
| `with pl.cluster()` | Cluster 作用域 → 一个 `Group` 函数 | 让 AIC 与 AIV kernel 在同一物理 cluster 上协同调度 |
| `with pl.spmd(n)` / `for i in pl.spmd(n)` | SPMD 作用域 | 把一个 kernel 铺开到 `n` 个 block |
| `for aiv_id in pl.split_aiv(2, mode=...)` | `SplitAivScopeStmt` 区域 | 把一个区域切到两条 AIV lane 上 |

`pl.at` 的关键字参数，全部可选：

| 参数 | 层面 | 含义 |
| ---- | ---- | ---- |
| `optimizations=[pl.split(mode)]` | 放置 | 被 outline kernel 的跨核切分模式 |
| `optimizations=[pl.cross_core_slot(slot_num=N)]` | 放置 | 自动跨核管道的环深度 |
| `deps=[tid, ...]` | 定序，TaskId 级 | 显式生产者边 |
| `no_dep_args=[t, ...]` | 定序，实参槽级 | 排除在依赖跟踪之外的被捕获张量 |
| `dumps=[t, ...]` | 调试 | 标记要做选择性 dump 的张量 |
| `allow_early_resolve=True` | 调度 | 允许消费者在本任务完成前预置 |
| `name_hint="..."` | 外观 | outline 出的函数名 |

`optimizations=` 的条目必须在调用点内联书写 —— 解析器读的是 AST，因此用变量拼出来的列表不被接受。`pl.split` 与 `pl.cross_core_slot` 彼此正交、可自由组合：一个切分工作，一个给通道定尺寸。

```python
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                          pl.cross_core_slot(slot_num=4)]):
    ...
```

省略 `cross_core_slot` 就保持默认环深度：单方向活跃时 8 个 slot，双方向都活跃时每方向 4 个。`pl.split(slot_num=...)` 已弃用 —— 它逼你顺带指定一个未必想要的切分模式；请分开写这两个条目。

### SPMD

三种形式，区别在于函数体是否读 block 索引、以及你是否捕获 TaskId：

```python
# 1. Dispatch form — body launches a pre-defined kernel.
with pl.spmd(4):
    out = self.kernel(a, b, out)

# 2. Loop form — body is auto-outlined; `i` binds the block index.
for i in pl.spmd(4):
    off = i * 128
    out = pl.store(pl.add(pl.load(a, [off, 0], [128, 128]),
                          pl.load(b, [off, 0], [128, 128])), [off, 0], out)

# 3. Capture form — same bodies as form 1, plus a producer TaskId.
with pl.spmd(4, deps=[prev_tid]) as tid:
    ...
```

一个既不读 block 索引、也不派发 kernel 的 `with pl.spmd(n):` 体会被拒绝 —— 那样每个 block 都在做完全相同的工作。`deps=` 只在 `as tid` 形式下可用。

当涉及硬 `pl.system.syncall` 时，请按设备实际规模而非字面量来定启动规模：传 `pl.system.available_cluster_count()`（混合或纯 cube kernel）或 `pl.system.available_aiv_count()`（纯 vector kernel），并在调用点内联书写。

### 运行时作用域

运行时作用域（`PTO2_SCOPE`）是资源与依赖跟踪的边界：它界定 OverlapMap 的跟踪范围，并给出每作用域的堆层级，因此嵌套作用域各自独立回收内存。运行时提供了一个隐式顶层作用域，所以**写作用域是调优手段，从来不是正确性要求**。

| 模式 | 含义 |
| ---- | ---- |
| `pl.scope()` / `ScopeMode.AUTO` | OverlapMap 自动跟踪开启 |
| `pl.scope(mode=pl.ScopeMode.MANUAL)`，别名 `pl.manual_scope()` | 自动跟踪关闭 —— 每条边由你声明 |

规则：作用域属于 Orchestration 函数，不属于 InCore 函数。`mode=AUTO` 只在 `@pl.function(auto_scope=False)` 下允许 —— 默认情况下 AUTO 的放置由编译器掌管。`MANUAL` 两种情况下都允许。AUTO 作用域不得嵌套在 MANUAL 之内，`manual_scope` 也不得嵌套在另一个 `manual_scope` 之内。

### 提交任务

`pl.submit` 派发一个 kernel 并交还它的生产者 TaskId。它是解析器构造 —— 在被装饰的函数体之外调用会抛异常。

```python
with pl.manual_scope():
    scratch, tid = pl.submit(self.stage1, x, scratch)
    out, _       = pl.submit(self.stage2, scratch, out, deps=[tid])
```

`pl.spmd_submit` 是它的 SPMD 兄弟，`core_num=` 为必填关键字：

```python
out, tid = pl.spmd_submit(self.kernel, x, core_num=8, sync_start=True, deps=[prev])
```

两者都必须按二元组解包：元素 0 是 kernel 的结果，元素 1 是 TaskId。两者在 **auto 与 manual 作用域下都可用** —— `deps=` 与 OverlapMap 跟踪正交，最终 fanin 是推导边与显式边的并集。在 auto 作用域里，把 `deps=` 当作补齐运行时推不出来的边的精修工具；在 `manual_scope` 里，用它声明每一条边。

### 退出依赖跟踪

三种粒度，由粗到细：

| 构造 | 退出范围 |
| ---- | -------- |
| `with pl.manual_scope():` | 区域内所有 submit |
| `pl.create_tensor(..., manual_dep=True)` | 单个张量，其整个生命周期 |
| `pl.no_dep(t)` 写在调用实参处 | 单个张量，仅对单个任务 |

`pl.at` 作用域上的 `no_dep_args=[t]` 与 `pl.no_dep(t)` 是同一个断言，用于 kernel 调用由 outliner 合成、没有语法上的实参槽可包裹的场合。它对被函数体修改的捕获和只读捕获同样合法 —— 你是在断言兄弟任务触碰的是互不相交的区域。

`deps=` 收 TaskId，`no_dep_args=` 收张量。二者描述的不是一回事。

### 派发谓词

`predicate=` 用于跳过那些"需不需要做"只有运行期才知道的任务。调度器在派发点求值 —— 此时依赖已满足，所以取到的值是最新的 —— 为假时把任务就地退休、根本不下发到核上，同时仍然结算 fanin 与 fanout，使下游消费者照常解锁。

```python
with pl.spmd(1) as gate_tid:
    row_count = self.gate(row_count)

with pl.spmd(4, deps=[gate_tid], predicate=(row_count[0, 0] > 0)) as tid:
    out = self.expert(x, out)
```

这个比较是**按语法匹配、从不求值**的：在编排里读它会卡在 `wait_for_tensor_ready` 上，而那正是谓词要避免的事。可表达的只有 `tensor[indices] OP int 字面量`，单个比较，不支持链式比较、算术或布尔组合。更复杂的条件请在前一个 kernel 里归约成单个门控值，再对它做谓词。

**契约：** 操作数张量的生产者必须在本任务的 `deps=` 之中，否则派发点读到的可能是陈旧值。解析器在静态可证的范围内做检查，其余由你负责。

### 调度提示

`allow_early_resolve=True` 把任务标记为可推测早派发的生产者：调度器可以在它完成之前把它的消费者预置到空闲核上。这是生产者侧的提示 —— 消费者只有在它*所有*生产者都被标记后才会预置。纯调度优化，不影响结果；在由大量短任务构成的关键路径上收益明显。

## Edge Cases

> **致命陷阱：** 一前一后写两次派发并不会让它们有序。如果那个关系没有体现为运行时能看见的缓冲区重叠，就不存在边，两个任务可能重叠执行。结果是一个偶发复现、一上调试器就消失的竞态。请用 `deps=` 把边写出来。

> **致命陷阱：** 对一个生产者不在 `deps=` 里的张量使用 `predicate=`，读到的是当时恰好在那里的东西。没有任何提示 —— 任务被跳过或不被跳过，取决于陈旧数据。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **多次运行结果不同** | 两个必须有序的任务没有任何东西表达这个顺序 | 加 `deps=[producer_tid]` |
| **本可并行的任务被串行化** | 自动跟踪从一次并非真实依赖的重叠推出了边 | 在实参处 `pl.no_dep(t)`，或在作用域上 `no_dep_args=[t]` |
| **`pl.submit is a DSL parser construct and cannot be called directly`** | 在被装饰函数体之外使用 | 移进 `@pl.function` / `@pl.jit` 体内 |
| **`with pl.spmd(n):` 体被拒绝** | 它既不读 block 索引也不派发 kernel | 读 `pl.tile.get_block_idx()`，或调用一个 kernel |
| **`pl.spmd` 上的 `deps=` 被拒绝** | 只有 `as tid` 形式接受它 | 写成 `with pl.spmd(n, deps=[...]) as tid:` |
| **`optimizations=` 被拒绝** | 用变量拼出来的 —— 解析器读的是 AST | 在调用点内联书写该列表 |
| **`pl.split(slot_num=...)` 触发 `DeprecationWarning`** | 已弃用的写法 | 改为 `optimizations=[pl.split(MODE), pl.cross_core_slot(slot_num=N)]` |
| **嵌套作用域被拒绝** | AUTO 嵌套在 MANUAL 内，或 `manual_scope` 套 `manual_scope` | 拍平；运行时禁止这两种 |
| **`predicate` / `allow_early_resolve` 在 `pl.cluster()` 下被拒绝** | cluster 内嵌的 `pl.spmd` 不产生 Submit | 把该提示移出 cluster |

## See Also

- [类型 § 参数方向](00-types.md#参数方向) —— 自动依赖推导读的是什么。
- [控制流](02-control-flow.md) —— 包含这些作用域的循环。
- [编译期指令](05-directives.md) —— `dumps=` 所引用的 `dump_tag` 与调试接口。
- [AutoDeriveTaskDependencies](../../dev/passes/36-auto_derive_task_dependencies.md) —— 边是怎么推导出来的。
- [OutlineIncoreScopes](../../dev/passes/08-outline_incore_scopes.md) —— `pl.at` 如何变成函数。
- [MaterializeRuntimeScopes](../../dev/passes/42-materialize_runtime_scopes.md) —— AUTO 作用域的放置。
- [ExpandMixedKernel](../../dev/passes/20-expand_mixed_kernel.md) —— `pl.split` 驱动的是什么。
