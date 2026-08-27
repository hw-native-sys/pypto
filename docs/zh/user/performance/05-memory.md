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

**嵌套过深会以同样的方式、从另一头翻车。** 这个映射是饱和的 —— `min(scope_depth, 3)` —— 所以作用域深度 3、4、5 以及更深的全都落到 **ring 3** 上：

```text
深度   0    1    2    3    4    5    6 ...
ring   0    1    2    3 ── 3 ── 3 ── 3   ← 更深的作用域全都堆到同一个环上
```

于是一个有好几层嵌套循环的 kernel，会把它最内层、通常也是数量最多的那批任务集中到单独一个环上，而那正是会溢出的那个环。把嵌套压平，或者把作用域上提、别让每一层深的都被包一次，就能重新摊开。

下面这些 kernel 每次 CI 都会被执行，所以它们是真货而不是草图。它们共用这段准备：

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

N, TR, TC = 4, 64, 128
ROWS = N * TR

torch.manual_seed(0)
A = torch.randn(ROWS, TC, dtype=torch.float32)


def check(kernel, cfg):
    out = torch.zeros(ROWS, TC, dtype=torch.float32)
    kernel(A, out, config=cfg)
    torch.testing.assert_close(out, A * 2.0 + 1.0, rtol=1e-4, atol=1e-4)
```

## 手工再平衡

退出编译器放置，把作用域放到活所在的地方：

深度才是关键：阶段 1 待在外层作用域，阶段 2 再往里嵌一层，于是二者落在不同的环上。两个**同级**作用域做不到这一点 —— `ring_idx` 只由深度决定。

<!-- doctest: run -->
```python
@pl.jit(auto_scope=False)
def manual_placement(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.scope():                       # depth 0 — phase 1's tasks, ring 0
        for i in pl.range(N):
            with pl.at(level=pl.Level.CORE_GROUP):
                t = pl.load(a, [i * TR, 0], [TR, TC])
                pl.store(pl.mul(t, 2.0), [i * TR, 0], out)
        with pl.scope():                   # depth 1 — phase 2 moves to ring 1
            with pl.at(level=pl.Level.CORE_GROUP):
                for i in pl.range(N):
                    t = pl.load(out, [i * TR, 0], [TR, TC])
                    pl.store(pl.add(t, 1.0), [i * TR, 0], out)
    return out


check(manual_placement, RunConfig(platform="__PLATFORM__"))
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
# 绘图脚本随 runtime 子模块一起提供
python runtime/simpler_setup/tools/scope_stats_plot.py \
    <work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl
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

<!-- doctest: run -->
```python
sized = RunConfig(
    platform="__PLATFORM__",
    ring_task_window=[8192, 16384, 131072, 524288],
    ring_heap=[134217728, 268435456, 268435456, 536870912],
)


@pl.jit
def scaled(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(N):
            t = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.add(pl.mul(t, 2.0), 1.0), [i * TR, 0], out)
    return out


check(scaled, sized)
```

字段留空（默认 `None`）会回落到运行时的编译期默认值。进程级的 `PTO2_RING_*` 环境变量已经退役、不再被读取，因此 `RunConfig` 是给环设尺寸的唯一途径。

**代价：** 内存，而且算术是按环算的 —— 你以为「就整体大一点」的那个标量，会被应用四次。给环加尺寸也是**第二顺位**的修法：一个因为某个作用域里塞了上千个任务而溢出的任务窗口，拆成两个作用域比把它撑大更好。运行时失败时自己就是这么说的 —— *「raise `ring_task_window`（`runtime_env.ring_task_window`）or split the scope」*。

**怎么确认：** 新一份 `scope_stats.jsonl` 的元数据行显示新尺寸，而原来顶在容量上的那个峰值不再顶着。

## 让流式操作数绕过缓存

并不是内核读到的每个缓冲区都值得占用缓存。一个每个字节只被读一次的权重矩阵不可能命中缓存，
却仍会冲刷末级缓存（last-level cache），把真正会被重复读取的那一小块激活工作集挤出去。
两个操作数、一个缓存、相反的诉求。

编译器无法区分它们 —— 复用（reuse）对它并不可靠可见，而猜错的代价是几个百分点的性能悄悄
劣化，而不是一个报错。所以由你来声明：

```python
with pl.at(level=pl.Level.CORE_GROUP):
    pl.set_cache_policy(weights, pl.CachePolicy.BYPASS)
    # 本作用域内对 `weights` 的每次读取都被声明为流式；
    # 激活值仍走默认的带缓存路径。
    acc = pl.matmul(activations, weights, out_dtype=pl.FP32)
```

`pl.set_cache_policy` 是写在 `pl.at(level=pl.Level.CORE_GROUP, ...)` 或
`pl.spmd(...)` 作用域体**顶层**的独立语句 —— 位于该 body 自身的语句之中即可，不必是第一行。
它覆盖该作用域内对这个张量的每一次读取，因此张量层代码在访问点上无需任何改动 —— 这一点很
关键，因为那些读取是隐式的：`pl.matmul`、`pl.assemble` 和下标切片都会发出 load，却没有可供
标注的调用点。

写在非 `CORE_GROUP` 的 `pl.at` 作用域上的声明可以解析并被携带，但目前没有任何环节会把它下降
到 load，因此不产生效果 —— 那些作用域不会变成设备侧 kernel。

当你写的是 tile 层代码、已经显式写出了 load 时，就标在那里：

```python
tile = pl.load(weights, [n0, k0], [256, 512], cache=pl.CachePolicy.BYPASS)
```

load 上显式的 `cache=` 永远优先于作用域声明，两个方向都成立 —— 所以在一个整体绕过缓存的
作用域里，`cache=pl.CachePolicy.DEFAULT` 可以把某一次访问单独放回缓存路径。

**代价：** 正确性由你保证。`BYPASS` 断言了两件事：这个张量没有值得缓存的复用，**并且**内核
运行期间没有任何一方写这些字节。对同一段字节混用带缓存的写和绕过缓存的读是一致性
（coherency）缺陷 —— 这正是它绝不作为默认、也绝不由编译器推断的原因。编译器会拒绝它能看见的
那种情况（在作用域会写入的张量上声明 `BYPASS` 是错误），但无法证明一般情形。

**当前状态：** 工具链尚无 bypass 通路
（[PTOAS#1356](https://github.com/hw-native-sys/PTOAS/issues/1356)）。`BYPASS` 声明会被接受
并向下传递，在编译期给出告警，生成的代码与普通带缓存读取完全一致。现在就写上不会有任何代价，
等该 issue 落地后即可自动生效。

## 参见

- [运行时作用域](../tasks/01-scopes.md) —— 把作用域当作依赖语义选择来看。
- [InCore 函数调优](04-incore.md) —— 片上这一侧的消耗方。
