# 精修依赖图

去掉一条并不真实的边、跳过一个并不需要的任务，以及让调度器提前开始。

> **前置**：[声明一条边](02-submit.md)。

## Concept

前面几页把图建了起来。本页从三个方向改动它，而这三者是**真正不同**的操作 —— 拿错工具是最常见的错误：

| 你想要 | 该用 |
| ------ | ---- |
| 去掉一条推出来的边，因为它不是真实依赖 | **退出跟踪**：`manual_scope`、`manual_dep=True` 或 `pl.no_dep` |
| 当运行期的值这么说时，这个任务根本不要跑 | **派发谓词**：`predicate=` |
| 同一张图，但更早派发 | **调度提示**：`allow_early_resolve=` |

只有第一类改变正确性保证。谓词改变的是"跑什么"；提示除了时序什么都不改。

## Quickstart：三种粒度的退出

```python
with pl.manual_scope():                              # whole region: every task inside
    ...

t = pl.create_tensor(..., manual_dep=True)           # one tensor, its entire lifetime

with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[shared]) as tid:   # one tensor, one task
    ...
```

| 构造 | 退出范围 | 可用于 |
| ---- | -------- | ------ |
| `with pl.manual_scope():` | 区域内每一个任务 | `@pl.jit`、`@pl.function` |
| `pl.create_tensor(..., manual_dep=True)` | 单个张量，其整个生命周期 | `@pl.jit`、`@pl.function` |
| `pl.at(..., no_dep_args=[t])` | 单个张量，仅对单个任务 | `@pl.jit`、`@pl.function` |
| 调用实参处的 `pl.no_dep(t)` | 单个张量，仅对单个任务 | 仅 `@pl.program` 类 |

优先选能表达该断言的**最窄**那一种。让单个实参退出跟踪说的是"这个任务的这个实参没有冲突"；`manual_scope` 说的是"这块区域的整张图归我"，是大得多的承诺。

> 均为片段：每一行都应位于一个 Orchestration 函数体内。

## Mechanics

### `pl.no_dep`

一个由解析器识别的标记，写在 kernel 调用的实参位置 —— 运行期它原样返回该张量。它让运行时对这个实参**同时**跳过 OverlapMap 的依赖查询**和**生产者插入。

无论被调方把该参数声明为 `In`、`Out` 还是 `InOut`，它都合法，因为你断言的是一件带外的事：这个槽位上不存在写后读、写后写或读后写冲突。典型场景是写偏移由数据决定的写入 —— 编译器无法证明不相交，但分配协议保证了它。

这样包裹调用实参需要一个显式的 `self.<kernel>` 调用，因此它属于 `@pl.program` 写法。在 `@pl.jit` 函数里对应的写法是在外层 `pl.at` 作用域上写 `no_dep_args=[t]` —— 这也正是 kernel 调用由 outliner 合成、没有语法上的实参槽可包裹时所用的写法。

`deps=` 收 TaskId，`no_dep_args=` 收张量。二者不是一件事的两种拼写。

### `predicate=`

只有 `pl.submit` 与 `pl.spmd_submit` 带谓词 —— `pl.at` 没有 `predicate=` 关键字，所以这一项只能从 `@pl.program` 写法够到。

用来跳过那些"需不需要做"只有运行期才知道的任务。调度器在**派发点**求值 —— 此时依赖已满足，所以取到的值是最新的，而不必在编排期等待。为假时任务就地退休、根本不下发到核上，同时其 fanin 与 fanout 照常结算，下游消费者正常解锁。

```python
out, tid = pl.spmd_submit(self.expert_ffn, tokens, out, core_num=N,
                          deps=[gather_tid],
                          predicate=(row_count[e] > 0))
```

这个比较是**按语法匹配、从不求值**的。在这个位置上，`row_count[e] > 0` 是交给调度器的一份声明，而不是一次 `tensor.read` 加一次比较 —— 在编排里读它意味着要等这个张量，而那正是谓词想避免的事。

可表达的只有 `tensor[indices] OP int 字面量`：单个比较，运算符为 `==` `!=` `>` `<` `>=` `<=`。不支持链式比较、算术或布尔组合 —— 运行时只支持单个比较。更复杂的条件请在前一个 kernel 里归约成一个门控值，再对它做谓词。

**契约：** 操作数张量的生产者必须在本次 submit 的 `deps=` 之中，这样派发点读到的才是当前值。解析器在静态可证的范围内强制这一点；其余由你负责。

### `allow_early_resolve=`

把任务标记为可推测早派发的生产者：调度器可以在它完成之前把它的消费者预置到空闲核上，等它一完成就用门铃放行。这是**生产者侧**的提示 —— 消费者只有在它*所有*生产者都被标记（或已完成）之后才会预置。

纯调度行为：不影响结果。在由大量短任务构成的关键路径上收益明显，其余情况下无害。`sync_start` 的 SPMD 任务自身不能被逐 block 预置，但标记它仍然能让它的消费者预置。

### `pl.system.task_dummy`

一个不做任何工作的依赖汇聚点：它接受 `deps=[...]` 并返回一个 TaskId，因此可以把若干生产者收敛成一个句柄，供后续任务点名。

```python
gate = pl.system.task_dummy(deps=[tid_a, tid_b])
out, _ = pl.submit(self.consumer, x, out, deps=[gate])
```

和 `pl.submit` 一样，它是解析器构造 —— 在被装饰函数体外调用会抛异常。注意拼写：它在 `pl.system` 下，不在顶层。

## Edge Cases

> **致命陷阱：**
>
> - `pl.no_dep` 是一个编译器无法检验的断言。如果那些区域其实并不相交，你就删掉了一条真实的边，结果是竞态 —— 与压根没声明这条边是同一类缺陷。
> - 对一个生产者不在本次 submit `deps=` 里的张量使用 `predicate=`，读到的是内存里当时恰好存在的东西。没有任何提示；任务被跳过或不被跳过，取决于陈旧数据。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **加了 `no_dep` 之后出现竞态** | 那些区域其实并不相交 | 去掉这个标记；它删掉的那条边是真的 |
| **`@pl.jit` 下 `pl.no_dep` 破坏元数据推断** | 包裹层让 `@pl.jit` 的 shape/dtype 推断看不到该张量 | 改在外层 `pl.at` 作用域上写 `no_dep_args=[t]` |
| **谓词被解析器拒绝** | 可表达的只有 `tensor[indices] OP int 字面量` | 在前一个 kernel 里归约成一个门控值，对它做谓词 |
| **被谓词的任务在不该跑时跑了** | 操作数的生产者不在 `deps=` 里 | 把生产者的 TaskId 加进 `deps=` |
| **`pl.cluster()` 下 `predicate` / `allow_early_resolve` 被拒绝** | cluster 内嵌的 `pl.spmd` 不产生可承载该提示的 Submit | 把提示移出 cluster |
| **`allow_early_resolve` 没有任何效果** | 消费者只有在它*所有*生产者都被标记后才预置 | 把其余生产者也标记上，或接受它在此不适用 |
| **`pl.task_dummy` 未定义** | 它在 `pl.system` 下 | 调用 `pl.system.task_dummy(deps=[...])` |

## See Also

- [声明一条边](02-submit.md) —— `predicate=` 与 `allow_early_resolve=` 写在哪里。
- [运行时作用域](01-scopes.md) —— 最粗的退出方式，以及为什么它很少是对的选择。
- [类型 § 参数方向](../language/00-types.md#参数方向) —— `no_dep` 所覆盖的那个声明。
- [算子](../ops/01-catalog.md) —— `no_dep` 与 system 算子的目录条目。
