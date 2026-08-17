# 任务粒度

派发不是免费的。把 InCore 函数的尺寸定在「核把时间花在算，而不是花在等着被告知算什么」的那个点上。

> **前置**：[读泳道图](00-swimlane.md)。

## 你正在付的那笔钱

每一个 `pl.at` 块就是一个任务。运行时要为它们每一个解析依赖、放置到核上、写描述符、观察完成。这些活跑在 AICPU 上，而 AICore 在等。

泳道图上会看到两个分量：

| 分量 | 出现在哪 | 量级 |
| ---- | -------- | ---- |
| 取件延迟 | 某个核上的 `[dispatch, start]` 间隙 | 每次切换约 0.8 µs |
| 调度器没跟上 | 存在**就绪但未派发**任务时的空闲核 | 取决于负载 |

真正疼的是第二个，而且它是可测的、不是理论上的。在运行时调度开销模型所依据的那个样本上 —— qwen3-14b `decode_layer`，a2a3，**542 个任务** —— 分析报告 AIC 的「空闲且有就绪活」占 makespan 的 **15.0%**，AIV 为 **10.3%**。这些不是普适数字，重点也不在它们的大小：重点是**一个由大量小任务堆出来的负载，可以把两位数比例的墙上时间花在任务管理上**。

**症状：** 条窄、隙宽，并且 `sched_overhead_analysis` 报出一个很大的 `has_overhead`。如果你的条很宽、隙很细，本页对你没用 —— 去 [InCore 函数调优](04-incore.md)。

## 把一个任务变大

三种办法，大致按适用频率排列。

### a. 更大的 tiling

**何时适用：** kernel 每个任务干的活是固定的，而 tile 小到连搬运也不划算。

**怎么做：** 提高任务处理的 tile 形状。

```python
# 之前 —— 每个 64x64 tile 一个任务
with pl.at(level=pl.Level.CORE_GROUP):
    tile_a = pl.load(a, [0, 0], [64, 64])
    tile_b = pl.load(b, [0, 0], [64, 64])
    pl.store(pl.add(tile_a, tile_b), [0, 0], c)

# 之后 —— 一个任务覆盖 4 倍元素
with pl.at(level=pl.Level.CORE_GROUP):
    tile_a = pl.load(a, [0, 0], [128, 128])
    tile_b = pl.load(b, [0, 0], [128, 128])
    pl.store(pl.add(tile_a, tile_b), [0, 0], c)
```

**跑一下：** `python examples/advanced/04_task_granularity.py --mode larger_tiles` —— 与 `--mode many_small_tasks` 对照，后者把同样的活切成四个任务而不是两个。

**代价：** 片上缓冲占用，对 2D tile 是平方增长。一个再也放不下同居者的 tile，会把分配器逼到要么失败、要么让出一级流水 —— 见 [内存](05-memory.md)。

**怎么确认：** 泳道图上条更宽**并且**隙按比例更窄。同时看 `report/perf_hints.log`：如果之前 PH001 在标你的 load，更宽的最内维应该让那些行消失。

### b. 把循环放进 InCore 函数里

**何时适用：** 活本来就是分块的，而分块循环在 `pl.at` 块**外面** —— 于是每一块都付一次完整派发。

**怎么做：** 把循环挪进去。tile 形状不变，只有偏移在动。

```python
# 之前 —— N 个任务，每块一个
for i in range(ROWS // TILE_ROWS):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        ...

# 之后 —— 一个任务，里面 N 次迭代
with pl.at(level=pl.Level.CORE_GROUP):
    for i in pl.range(ROWS // TILE_ROWS):
        tile_a = pl.load(a, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        tile_b = pl.load(b, [i * TILE_ROWS, 0], [TILE_ROWS, COLS])
        pl.store(pl.add(tile_a, tile_b), [i * TILE_ROWS, 0], c)
```

`examples/beginner/02_elementwise.py`（`chunked_add`）就是这个模式的完整版本。

**跑一下：** `python examples/advanced/04_task_granularity.py --mode loop_inside` —— `--mode many_small_tasks` 就是它替换掉的四次派发形式。

**代价：** 这些块现在在一个核内被严格定序了。如果它们本来互相独立、而你又有空核，你就是拿并行度换了派发开销 —— 核在空转时这笔交易是亏的。不过它也让这个循环成为 [double buffer](04-incore.md) 的候选，而收益通常从那里回来。

**怎么确认：** `deps.json` 里那 `N` 个节点塌缩成一个 —— 任务数减少 `N - 1` —— 泳道图上原来的阶梯变成一根宽条。

### c. 合并多个 InCore 函数

**何时适用：** 图里连续的几个任务是一条生产者/消费者链，而中间数据本可以留在片上。

**怎么做：** 把这些操作放进同一个 `pl.at` 块，中间结果就不再往返 GM。

```python
# 之前 —— 两个任务，`s` 出 GM 再回来
with pl.at(level=pl.Level.CORE_GROUP):
    s = pl.add(pl.load(a, [0, 0], [TR, TC]), pl.load(b, [0, 0], [TR, TC]))
    pl.store(s, [0, 0], scratch)
with pl.at(level=pl.Level.CORE_GROUP):
    pl.store(pl.exp(pl.load(scratch, [0, 0], [TR, TC])), [0, 0], out)

# 之后 —— 一个任务，`s` 留在片上
with pl.at(level=pl.Level.CORE_GROUP):
    s = pl.add(pl.load(a, [0, 0], [TR, TC]), pl.load(b, [0, 0], [TR, TC]))
    pl.store(pl.exp(s), [0, 0], out)
```

**跑一下：** `python examples/advanced/04_task_granularity.py --mode merged_chain` —— `--mode two_tasks_via_gm` 是合并之前的同一条链。

**代价：** 合并后的任务要同时持有每一个中间结果。

> **跨引擎合并不是这一招。** 把一个 cube 操作和一个 vector 操作放进同一个作用域，还需要一个 split 模式 —— 没有 `pl.split(...)` 的话缓冲放不下，编译器会拒绝这个作用域。那个情形是 [mix kernel](02-runtime-overhead.md#构建-mix-kernel)；把 `matmul` 和消费它的 vector 操作合并之前先读它。

**怎么确认：** 被合并的任务不再作为独立节点出现在 `deps.json` 里，中间结果的 GM 流量从 kernel 里消失。

## 另一个方向：太粗

粒度不是单调的。单卡的核数是固定的 —— Ascend910B 上是 **48 个 vector 与 24 个 cube** —— 而一个任务占一个核。

```text
太多小任务                  合适                      太少大任务
├─┤ ├─┤ ├─┤ ├─┤ ├─┤        ├─────┤├─────┤            ├──────────────────┤
 隙占主导                   核都在忙                  核 2..47 空闲
 → 派发受限                 → 计算受限                → 并行度受限
```

如果合并让你掉到核数以下，你只是把瓶颈搬了个家而不是消掉它，而泳道图会说得很明白：条很宽、隙没了、而大多数核的泳道干脆是**空的**。

注意 [SPMD](02-runtime-overhead.md#使用-spmd) 并不会消掉这个取舍。它和 `pl.parallel` 一样，只是**描述**工作的一种方式 —— 一次派发扇出到很多 block —— 每个 block 干多少活仍然由你决定。它改变的是这种描述的价钱：N 个 block 只付一次派发而不是 N 次。粒度这个问题两种写法下都还是你的。

## 怎么判断

```text
窄条之间有宽隙？
├─ 核大多空闲、任务很少          → 任务太粗：拆分，或用 SPMD
├─ 核都在忙、但每根条之间都有隙  → 任务太细：用 a、b、c 放大
└─ 只在特定位置有隙              → 不是粒度问题：见 03-dependencies
```

## 参见

- [运行时开销](02-runtime-overhead.md) —— 降低每个任务的代价，而不是任务数量。
- [InCore 函数调优](04-incore.md) —— 把条本身变短。
