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

下面这些 kernel 每次 CI 都会被执行，所以它们是真货而不是草图。它们共用这段准备：

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

ROWS, COLS = 256, 128
SMALL, LARGE = 64, 128          # tile rows before and after (a)
CFG = RunConfig(platform="__PLATFORM__")

torch.manual_seed(0)
A = torch.randn(ROWS, COLS, dtype=torch.float32)
B = torch.randn(ROWS, COLS, dtype=torch.float32)

def fresh():
    return torch.zeros(ROWS, COLS, dtype=torch.float32)
```

### a. 更大的 tiling

**何时适用：** kernel 每个任务干的活是固定的，而 tile 小到连搬运也不划算。

**怎么做：** 提高任务处理的 tile 形状。

改前 —— 四个任务，每个处理一块 `[64, 128]`。`pl.unroll` 在编译期展开，所以每次迭代各自产生一个 `pl.at` 块、也就是各自一次派发：

<!-- doctest: run -->
```python
@pl.jit
def many_small_tasks(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    for i in pl.unroll(ROWS // SMALL):
        with pl.at(level=pl.Level.CORE_GROUP):
            ta = pl.load(a, [i * SMALL, 0], [SMALL, COLS])
            tb = pl.load(b, [i * SMALL, 0], [SMALL, COLS])
            pl.store(pl.add(ta, tb), [i * SMALL, 0], c)
    return c

c = fresh()
many_small_tasks(A, B, c, config=CFG)
torch.testing.assert_close(c, A + B, rtol=1e-4, atol=1e-4)
```

改后 —— 同样这些行改用 `[128, 128]` 的 tile，于是两个任务而不是四个。什么都没挪，只是 tile 覆盖的元素更多：

<!-- doctest: run -->
```python
@pl.jit
def larger_tiles(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    for i in pl.unroll(ROWS // LARGE):
        with pl.at(level=pl.Level.CORE_GROUP):
            ta = pl.load(a, [i * LARGE, 0], [LARGE, COLS])
            tb = pl.load(b, [i * LARGE, 0], [LARGE, COLS])
            pl.store(pl.add(ta, tb), [i * LARGE, 0], c)
    return c

c = fresh()
larger_tiles(A, B, c, config=CFG)
torch.testing.assert_close(c, A + B, rtol=1e-4, atol=1e-4)
```

这里只有行轴在放大（`COLS` 已经是满宽），所以是 2 倍。两个轴一起放大，任务数就按各自的倍数缩减，片上占用也随之上去。

**代价：** 片上缓冲占用，对 2D tile 是平方增长。一个再也放不下同居者的 tile，会把分配器逼到要么失败、要么让出一级流水 —— 见 [内存](05-memory.md)。

**怎么确认：** 泳道图上条更宽**并且**隙按比例更窄。同时看 `report/perf_hints.log`：如果之前 PH001 在标你的 load，更宽的最内维应该让那些行消失。

### b. 把循环放进 InCore 函数里

**何时适用：** 活本来就是分块的，而分块循环在 `pl.at` 块**外面** —— 于是每一块都付一次完整派发。

**怎么做：** 把循环挪进去。tile 形状不变，只有偏移在动。

`pl.range` 是设备侧循环，所以整体只有一次派发。tile 尺寸不变，动的只是偏移 —— 与上面的 `many_small_tasks` 对照，那是同样的活拆成四次派发：

<!-- doctest: run -->
```python
@pl.jit
def loop_inside(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(ROWS // SMALL):
            ta = pl.load(a, [i * SMALL, 0], [SMALL, COLS])
            tb = pl.load(b, [i * SMALL, 0], [SMALL, COLS])
            pl.store(pl.add(ta, tb), [i * SMALL, 0], c)
    return c

c = fresh()
loop_inside(A, B, c, config=CFG)
torch.testing.assert_close(c, A + B, rtol=1e-4, atol=1e-4)
```

`examples/beginner/02_elementwise.py`（`chunked_add`）就是这个模式的完整版本。

**代价：** 这些块现在在一个核内被严格定序了。如果它们本来互相独立、而你又有空核，你就是拿并行度换了派发开销 —— 核在空转时这笔交易是亏的。不过它也让这个循环成为 [double buffer](04-incore.md) 的候选，而收益通常从那里回来。

**怎么确认：** `deps.json` 里那 `N` 个节点塌缩成一个 —— 任务数减少 `N - 1` —— 泳道图上原来的阶梯变成一根宽条。

### c. 合并多个 InCore 函数

**何时适用：** 图里连续的几个任务是一条生产者/消费者链，而中间数据本可以留在片上。

**怎么做：** 把这些操作放进同一个 `pl.at` 块，中间结果就不再往返 GM。

改前 —— 两个任务，`s` 要经 GM 往返一趟；改后 —— 一个任务，`s` 始终留在片上：

<!-- doctest: run -->
```python
@pl.jit
def two_tasks_via_gm(a: pl.Tensor, b: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        s = pl.add(pl.load(a, [0, 0], [LARGE, COLS]), pl.load(b, [0, 0], [LARGE, COLS]))
        pl.store(s, [0, 0], scratch)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.exp(pl.load(scratch, [0, 0], [LARGE, COLS])), [0, 0], out)
    return scratch, out

@pl.jit
def merged_chain(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        s = pl.add(pl.load(a, [0, 0], [LARGE, COLS]), pl.load(b, [0, 0], [LARGE, COLS]))
        pl.store(pl.exp(s), [0, 0], out)
    return out

expected = torch.exp(A[:LARGE] + B[:LARGE])
scratch, out = torch.zeros(LARGE, COLS), torch.zeros(LARGE, COLS)
two_tasks_via_gm(A[:LARGE], B[:LARGE], scratch, out, config=CFG)
torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-4)

out = torch.zeros(LARGE, COLS)
merged_chain(A[:LARGE], B[:LARGE], out, config=CFG)
torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-4)
```

这里两处比较只放松了相对界限，取 `rtol=1e-3`：设备上的 `exp` 自身带有约 `1e-4` 的相对误差，
沿用上面逐元素块里的 `1e-4` 量的就是算子精度，而不是这次变换。`atol` 仍是 `1e-4` —— 输出较小的
那些元素靠它兜底，那里的松紧和别处一样。

**代价：** 合并后的任务要同时持有每一个中间结果。

> **跨引擎合并不是这一招。** 把一个 cube 操作和一个 vector 操作放进同一个作用域，还需要一个 split 模式 —— 没有 `pl.split(...)` 的话缓冲放不下，编译器会拒绝这个作用域。那个情形是 [mix kernel](02-runtime-overhead.md#build-a-mixed-kernel)；把 `matmul` 和消费它的 vector 操作合并之前先读它。

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

注意 [SPMD](02-runtime-overhead.md#use-spmd) 并不会消掉这个取舍。它和 `pl.parallel` 一样，只是**描述**工作的一种方式 —— 一次派发扇出到很多 block —— 每个 block 干多少活仍然由你决定。它改变的是这种描述的价钱：N 个 block 只付一次派发而不是 N 次。粒度这个问题两种写法下都还是你的。

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
