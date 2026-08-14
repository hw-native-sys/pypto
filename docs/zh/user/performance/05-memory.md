# 内存

运行时的四个环、scope 放置为何决定你的任务落到哪一个环，以及怎么给它们定尺寸。

> **前置**：[运行时作用域](../tasks/01-scopes.md) —— 本页假定你知道作用域**是什么**，并把它当作一个内存旋钮来讲。

## 四个环

运行时没有一个统一的任务资源池，它有**四个互相独立的**，而任务的作用域嵌套深度决定用哪一个：

```text
ring_idx = min(scope_depth, 3)

作用域深度 0 ──► ring 0 ─┐
作用域深度 1 ──► ring 1  │  各自拥有任务槽窗口、
作用域深度 2 ──► ring 2  │  输出堆、依赖边池，
作用域深度 3+ ─► ring 3 ─┘  各自按 FIFO 独立回收
```

每个环是一块独立映射，有自己的游标和 FIFO 回收指针，所以内层作用域的任务永远不会和外层作用域那些长命分配共用一个 FIFO 头。这正是这套设计的全部意义：深作用域里的一个短命任务，不必等着顶层的长命分配才能被回收。

每个环持有三种分别定尺寸的资源：

| 资源 | 装什么 | 耗尽时的表现 |
| ---- | ------ | ------------ |
| `task_window` | 在飞的任务槽 | 一条点名 task window 的容量错误 |
| `heap` | 输出自动分配的字节 | 分配失败 |
| `dep_pool` | 依赖边条目 | 一条点名 dep pool 的容量错误 |

## 默认放置为什么会浪费它们

默认情况下作用域放置由编译器拥有：`MaterializeRuntimeScopes` 会把**整个函数体、以及每个 `for` 体和每个 `if` 的 then/else 体**各自包进一个 AUTO 作用域。这是个合理的默认 —— 但它意味着你的环分配是控制流形状的副作用，而不是你决定的任何事情。

```python
@pl.function(type=pl.FunctionType.Orchestration)   # auto_scope=True（默认）
def orch(self, a, out):
    for i in pl.range(4):
        out = self.kernel(a, out)
    return out
```

会变成

```python
@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
def orch(self, a, out):
    with pl.scope():            # 深度 0 —— 函数体
        for i in pl.range(4):
            with pl.scope():    # 深度 1 —— 循环体：里面每个任务都落在 ring 1
                out = self.kernel(a, out)
        return out
```

一个扁平的 kernel —— 一个函数体，没有值得包裹的循环或分支 —— 会把**所有东西塞进 ring 0**，让 ring 1–3 完全闲置。那三个闲置的环仍然被映射着；你付了钱，什么也没拿到。失效方式是不对称且不友好的：ring 0 撞到天花板报出容量错误，而隔壁四分之三的资源空着。

## 手工再平衡

退出编译器放置，把作用域放到活所在的地方：

```python
@pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
def orch(self, a, out):
    with pl.scope():
        # ... 阶段 1 的任务，ring 0
        with pl.scope():
            ...   # 阶段 2 的任务，ring 1 —— 与阶段 1 独立回收
        return out
```

`@pl.jit`、`@pl.jit.host`、`@pl.jit.inline` 接受 `auto_scope=False`；`.incore` 与 `.opaque` 拒绝它 —— 它们被外提成独立 kernel，没有可供作用域存在的编排函数体。

**代价：** 带上 `auto_scope=False` 之后该 pass **什么都不插**，于是这个函数里每一个作用域都归你放 —— 包括那些编译器原本免费加的。这是一个纯放置决策：AUTO 作用域仍然保持自动依赖跟踪开启，所以再平衡环并不改变你的依赖语义。（`MANUAL` 模式会改，那是[另一章](../tasks/01-scopes.md)的事。）

**怎么确认：** 见下面的 scope stats。峰值应该分散到各个环，而不是堆在一个上。

## 定尺寸之前先度量

绝不要给一个你没度量过的环改尺寸。`RunConfig(enable_scope_stats=True)` 记录每个作用域在任务槽、堆字节、依赖池条目、tensormap 条目上的峰值：

```python
cfg = RunConfig(platform="a2a3", enable_scope_stats=True, save_kernels=True)
```

```text
<work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
```

它是 NDJSON：第 1 行是运行元数据，之后每一行是一个作用域样本。元数据行里的 `task_window_max`、`heap_max`、`dep_pool_max` 是**按 ring 0..3 索引**的数组 —— 这是确认本次运行实际拿到什么尺寸最快的办法。整体渲染用运行时自带的绘图脚本：

```bash
python simpler_setup/tools/scope_stats_plot.py <...>/scope_stats/scope_stats.jsonl
```

读它看两件事：

- **顶到容量的峰值**是天花板 —— 那个环就是约束。
- **只有某一个环峰值很高、其余远低于容量**就是上面说的不均衡：先再平衡作用域，再考虑加大任何东西。

## 给环定尺寸

当度量说明某个环确实太小时，三个 `RunConfig` 字段负责定尺寸。每个都接受一个标量（广播到全部四个环）或**恰好 4 个** int 的列表分别对应 ring 0..3，其中值为 `0` 的项表示该环保持默认：

| 字段 | 单位 | 每项约束 |
| ---- | ---- | -------- |
| `ring_task_window` | 在飞任务槽 | 2 的幂，`>= 4` |
| `ring_heap` | **字节** | 2 的幂，`>= 1024` |
| `ring_dep_pool` | 依赖边条目 | 落在 `[4, INT32_MAX]` |

```python
cfg = RunConfig(
    platform="a2a3",
    ring_task_window=[8192, 16384, 131072, 524288],
    ring_heap=[134217728, 268435456, 402653184, 536870912],
)
```

字段留空（默认 `None`）会回落到运行时的 `PTO2_RING_*` 环境变量或它的编译期默认值，所以你也可以不改源码来做实验。

**代价：** 内存，而且算术是按环算的 —— 你以为「就整体大一点」的那个标量，会被应用四次。给环加尺寸也是**第二顺位**的修法：一个因为某个作用域里塞了上千个任务而溢出的任务窗口，拆成两个作用域比把它撑大更好。运行时失败时自己就是这么说的 —— *「raise `ring_task_window`（`PTO2_RING_TASK_WINDOW`）or split the scope」*。

**怎么确认：** 新一份 `scope_stats.jsonl` 的元数据行显示新尺寸，而原来顶在容量上的那个峰值不再顶着。

## 片上缓冲是另一个问题

上面这一切都是 host 可见的运行时状态。kernel 里的那些 tile 是另行规划的，**它们**耗尽时报错来自编译器而不是运行时。`pypto.tools.memory_map` 把那份分配渲染成 HTML —— 横轴地址、纵轴生命期、旁边是 IR。它的输入是一份 **pass dump**，不是一次运行：

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

prog = kernel.lower(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
```

```bash
DUMP=path/to/output_dir/passes_dump/NN_after_SomePass.py
python -m pypto.tools.memory_map "$DUMP" -o map.html
```

读它看两样：活得比需要更久的 tile，以及决定一条更深的[流水](04-incore.md#double-buffer)或更深的跨核环能不能放下的余量。

> `memory_planner=PTOAS` 下编译器完全跳过 `AllocateMemoryAddr`，于是 pass dump 里没有已分配的偏移，这个工具无从绘制。改用端到端对比。

## 参见

- [运行时作用域](../tasks/01-scopes.md) —— 把作用域当作依赖语义选择来看。
- [InCore 函数调优](04-incore.md) —— 片上这一侧的消耗方。
