# 分布式性能

多于一个 rank 之后有什么不同：一项新的主导开销、一种新的串行化方式，以及需要换种读法的指标。

> **前置**：[读泳道图](00-swimlane.md) 与 [分布式编程](../distributed/index.md)。第 [01](01-task-granularity.md)–[06](06-host.md) 页里的一切仍然逐 rank 适用 —— 本页只讲**额外多出来**的部分。

## 三件不一样的事

1. **通信是一项你原本没有的开销。** 它常常是最大的单项，而闭环第 2 步会把它显示为 device 时间——一段改 kernel 碰不到的 device 时间。
2. **最慢的那个 rank 决定节奏。** 一个起步晚的 rank 会拖住它参与的每一次 collective，所以值得关注的数字是**跨 rank 的离散度**，不是均值。
3. **一个均值把这两件事都藏起来了。** 这正是 benchmark 提供三种分组方式的原因。

## 读指标

`BenchmarkStats` 把同一批派发按三种方式分组。选错分组正是偏斜被漏掉的原因：

| 方法 | 返回 | 回答 |
| ---- | ---- | ---- |
| `per_round(metric)` | 每轮一个值 | 整个系统变快了吗？ |
| `per_rank(metric)` | 每个 rank → 一组值 | 是不是某个 rank 一直更慢？ |
| `per_dispatch(metric)` | `{(pid, slot): [round0, ...]}` | 哪个派发槽位是离群点？ |

每个都接受 `metric` = `"device"` / `"host"` / `"effective"`：

- **`device`** —— 该次派发的设备挂钟时间。
- **`host`** —— 主机挂钟时间。与 `device` 对比可以找出派发开销。
- **`effective`** —— device 域中 `orch` 与 `sched` 两个 span 的**并集**，即运行时的 "Effective" 指标。两个 span 共享本次调用的设备时钟原点，所以这个并集在一次派发内是有意义的。在 `*sim` 平台与非 profiling 构建上它返回 `0.0` —— 这里的 0.0 意思是「没采集」，不是「瞬间完成」。

`per_dispatch` 的键是一轮之内的派发**槽位**而不是轮次 —— 所以同一个 rank 的重复派发或异构派发彼此分开。配合 `dispatch_tasks()` 给槽位贴标签。它**仅限 L3**：在 L2 上返回 `{}`。

**先看 `per_rank`。** 如果各 rank 很齐，那 `per_round` 才是该优化的数字。如果不齐，那么在偏斜被搞清楚之前，任何 collective 调优都不重要 —— 你优化的会是等待，而不是工作。

## collective 算法：mesh vs ring

各 collective 接受 `mode=`，默认 `"mesh"`：

| `mode=` | 算法 | 窗口数 |
| ------- | ---- | ------ |
| `"mesh"`（默认） | 直接交换 | **O(P)** |
| `"ring"` | 分块 reduce-scatter + allgather | **O(1)** |

窗口数就是这个权衡本身。`"mesh"` 直接交换，每个对端要一个窗口，所以它的缓冲区占用随 rank 数增长。`"ring"` 跑一个 `2(P−1)` 步的调度，只需常数个窗口，用步数换内存。

- **何时用 `"ring"`：** rank 很多，或者窗口内存是约束。
- **代价：** `2(P−1)` 个顺序步骤 —— 对延迟更敏感，且在小载荷下更差，因为那时步数比搬运的字节更主导。
- **怎么开：** 在 collective 上写 `mode="ring"` —— 但改动从来不止这一个参数。signal 合成（host 形式能省略 `signal` 就靠它）**仅限 mesh**；而且对 `NR` 个 rank，`LowerHostTensorCollectives` 还要求：
  - 一个显式的、绑定 window 的 **INT32** signal，秩为 2，形状 `[2*(NR-1) + 1, NR]`；
  - `src` 形状**静态可知** —— 动态的 host-ring extent 会被拒绝；
  - `numel(src)` 是 `NR` 的整数倍，因为该调度会把它切成 `NR` 块。

  这三条都是编译期错误，报错信息会点名是哪一条，所以切错了会明确失败而不是悄悄错。参见 [集合通信](../distributed/01-collectives.md)。
- **怎么确认：** `per_round("device")` 看总量，`per_rank` 确认这个改动不是把开销挪到了某一个 rank 上。

## 让通信与计算重叠

大致形态是：collective 和别的任务一样也是一个任务，所以由**图**决定有没有东西能与它并行。如果某个 rank 手上有独立的计算，它就不该在一次 collective 里闲着。

- **怎么确认：** 依赖图（[`enable_dep_gen`](../tutorials/05-scheduling-tuning.md)）会显示那段计算是 collective 的**兄弟**还是**后代**。后代无法与它重叠，无论 collective 怎么调。

## 跨 rank 只付一次 setup

`DistributedWorker` 的文档写的是「prepare once, dispatch many」—— 它持有一个已初始化的 level-3 worker 以及全部 setup 产物。对性能有两个推论：

- **复用 worker。** 若干 program 注册到同一个 `DistributedWorker` 上就共享这份 setup，于是只有第一次运行要付。形态见 `examples/runtime/multi_program_kv_cache.py`。
- **让分片数据常驻。** `alloc_stacked_tensor(worker_ids=...)` 给每个 worker 放一个分片并留在那里，省掉每次启动都要散发一遍不会变的权重。

*这两件事都在 **host** span 里确认。* setup 与常驻从不改变 device 时间，所以只看 `device` 的度量会报告「毫无改善」。

## 进攻顺序

1. **先 `per_rank`。** 偏斜会让其余每个数字都失效。
2. **然后是图** —— 到底有没有计算被允许与 collective 并行？
3. **然后是算法** —— mesh 还是 ring，按 rank 数与窗口内存来选。
4. **然后是 setup 与常驻** —— 只在 host span 里可见。
5. **然后才是逐 rank 的 kernel 工作** —— 第 [01](01-task-granularity.md)–[06](06-host.md) 页。

第 1 步先于第 3 步是最省时间的一条。在某个 rank 迟到的情况下调 collective，优化的是一条没人在等的队列。

## 边界情况

| 症状 | 可能原因 | 去哪看 |
| ---- | -------- | ------ |
| **`effective` 处处是 `0.0`** | `*sim` 平台或非 profiling 构建 | 属预期 —— 改用 `device` |
| **`per_round` 改善了，用户却没感觉** | 开销被挪到了某一个 rank 上 | `per_rank` |
| **`ring` 比 `mesh` 慢** | rank 少，或载荷小 —— `2(P−1)` 步占主导 | rank 数与载荷大小 |
| **collective 从不与计算重叠** | 那段计算在图里是 collective 的后代 | `enable_dep_gen` |
| **加 rank 之后内存失败** | `mesh` 需要 O(P) 个窗口 | 改用 `mode="ring"` |

## 参见

- [读泳道图](00-swimlane.md) —— 逐任务计时，与单卡共用。
- [任务粒度](01-task-granularity.md) 及其后 —— 仍然逐 rank 适用。
- [分布式编程](../distributed/index.md) —— API 面本身。
- [集合通信](../distributed/01-collectives.md) —— 各 collective 的语义。
