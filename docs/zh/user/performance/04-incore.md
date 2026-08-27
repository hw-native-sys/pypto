# InCore 函数调优

把条本身变短：让搬运与计算重叠、换算法、迎合硬件的粒度，以及——当别的都赢不了时——干脆不用 PyPTO 写这个 kernel。

> **前置**：前面几页。如果条与条之间的间隙仍然是主要开销，本页为时过早。

下面这些 kernel 每次 CI 都会被执行，所以它们是真货而不是草图。它们共用这段准备：

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

NT, TR, TC = 8, 64, 128          # tiles in the loop, tile rows, tile cols
ROWS = NT * TR
CFG = RunConfig(platform="__PLATFORM__")

# Cycle through binary-exact values in a stable range on every host architecture.
indices = torch.arange(ROWS * TC, dtype=torch.int64)
A = (indices % 3 - 1).to(torch.float32).reshape(ROWS, TC)


def check(kernel):
    out = torch.zeros(ROWS, TC, dtype=torch.float32)
    kernel(A, out, config=CFG)
    torch.testing.assert_close(out, torch.exp(A), rtol=1e-4, atol=1e-4)
```

## Double buffer

**何时适用：** kernel 内的一个循环在 load → 计算 → store 之间交替，而核卡在搬运上，因为只有一块 buffer 可以载入。

### 用 `pl.pipeline`

编译器托管的形式。它把循环体在每次外层迭代里复制 `stage` 份，使第 `i+1` 次迭代的 load 与第 `i` 次的计算重叠：

<!-- doctest: run -->
```python
@pl.jit
def single_buffer(a: pl.Tensor, out: pl.Out[pl.Tensor]):     # the baseline
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(NT):
            tile = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


@pl.jit
def pipelined(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.pipeline(NT, stage=2):
            tile = pl.load(a, [i * TR, 0], [TR, TC])
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


check(single_buffer)
check(pipelined)
```

外层循环于是以 `stage * step` 为步长推进，尾部有一次 tail dispatch 覆盖不能被 `stage` 整除的余数。深度通常取 2–4。

**代价：** 循环体所暂存的每一块 buffer 都要有 `stage` 份同时存活。这是片上内存耗尽最常见的单一原因，而编译器会明说，不会悄悄少给：

```text
[perf_hint PH-MR-001] software pipelining requested depth 4 ... but only 2 of 4 buffers
fit (... B per stage, ... B free) — stages 2 apart share storage and serialize.
```

读法是：*你要了 4，你拿到 2*。这条提示接着会告诉你该动哪个杠杆 —— 把每级 tile 缩到它给出的字节预算内，或者把深度降到实际放得下的那个值。

**怎么确认：** `report/perf_hints.log` 里这条提示消失，并且 [L0 trace](#l0-指令级-trace) 里 MTE2 泳道与计算泳道重叠，而不是交替。

### 用显式 slot

手工管理的**放置**方式，适用于你希望轮转严格按你写的来 —— 通常是因为自然的暂存结构与 `pl.pipeline` 复制出来的不一致。注意它不是什么：`pl.pipeline` 把循环重构成一个调度，而 slot 只是消掉了会阻止重叠的同缓冲冲突。循环本身仍是顺序的，所以重叠与否要去 [L0 trace](#l0-指令级-trace) 里确认，而不是假定换个写法就买到了。`pl.MemRef("name", slots=N)` 在一块分配里预留 `N` 个等大的槽，用一个普通索引表达式按迭代挑一个：

<!-- doctest: run -->
```python
@pl.jit
def explicit_slots(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(NT):
            tile: pl.Tile[[TR, TC], pl.FP32, pl.MemRef("ub", slots=2)[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * TR, 0], [TR, TC], target_memory=pl.Mem.Vec
            )
            pl.store(pl.exp(tile), [i * TR, 0], out)
    return out


check(explicit_slots)
```

用内联的 `pl.MemRef("name", slots=2)` 写法，不要把声明绑到一个 Python 变量上 —— `@pl.jit` 会在一个全新的模块命名空间里重新解析生成的源码，那里没有这个变量。

**代价，而且取决于内存规划器：**

| 规划器 | slot 的下降形式 | 一次迭代内两个 slot 同时存活 |
| ------ | --------------- | ---------------------------- |
| `PYPTO`（默认） | 烘焙地址（`alloc_tile`） | 支持 |
| `PTOAS` | 一个 `alloc_multi_tile` 区域 + 每次使用一个 `multi_tile_get` | **codegen 拒绝** |

PTOAS 的这个拒绝是刻意的，在你围绕它做设计之前值得理解：ptoas 只保护一次迭代里的**第一个** `multi_tile_get`，于是第二个 slot 会在下一次迭代覆盖它的同时被读。这件事在 codegen 开始拒绝之前，是在设备上被实测出错的。**一次迭代一个 slot 存活**才是区域形式存在的目的，也是你如果可能换规划器时该写的形状。

## 看清片上预算

上面两种形式花的是同一样稀缺资源：片上缓冲空间。`pypto.tools.memory_map` 把那份分配渲染成 HTML —— 横轴地址、纵轴生命期、旁边是 IR —— 于是你能看清一条更深的流水得塞进什么样的空间里。它的输入是一份 **pass dump**，不是一次运行：

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

compiled = kernel.compile(*args, config=RunConfig(dump_passes=PassDumpLevel.EXPLICIT))
```

```bash
DUMP=path/to/output_dir/passes_dump/NN_after_SomePass.py
python -m pypto.tools.memory_map "$DUMP" -o map.html
```

读它看两样：活得比需要更久的 tile，以及决定再加一级流水或更深的跨核环能不能放下的余量。

> `memory_planner=PTOAS` 下编译器完全跳过 `AllocateMemoryAddr`，于是 pass dump 里没有已分配的偏移，这个工具无从绘制。改用端到端对比。

## 算法改动

有些 kernel 既不是搬运受限也不是派发受限，而是形状与机器不匹配。典型例子是一个 `M`/`N` 太小填不满 cube、而 `K` 很长的 matmul —— 切分归约维能给出输出维给不了的并行度：

```python
for ks in pl.parallel(SPLITS):
    ...   # 每个 split 归约自己那段 K；之后再合并各部分和
```

`examples/advanced/01_split_k.py` 是完整版本，[matmul 教程](../tutorials/02-matmul.md) 讲了它什么时候划算。

**代价：** split-K 的累加顺序不同，用了原子加之后这个顺序甚至逐次运行都不固定。要预期末位差异；在把它当 bug 之前，先核对累加顺序。

## L0 指令级 trace

**何时适用：** kernel 就是瓶颈，而你想知道是**哪条流水**。

编译期提示说的是编译器怀疑什么；L2 泳道图说的是任务怎么被调度。两者都没有显示核在指令层面做了什么。`incore-profiling` skill（来自 `pypto-user` 插件）把每个生成的 kernel 放到 Ascend 算子模拟器上跑，采集 cycle 级 trace：

先安装（`claude plugin install pypto-user@pypto-skills`），然后调用这个 skill；它会在一个已构建的
case 上驱动 `incore_profile.py`：

```text
/incore-profiling --build-dir build_output/<case> --target a2a3
```

该脚本属于插件而不属于本仓库，所以没有可以直接运行的仓内路径。

原始输出很杂。仓库内的工具把它清理成一份按流水分道、可用 Perfetto 查看的 trace：

```bash
TRACE="<build-dir>/kernel_insight_all_funcs_<ts>/funcs/<kernel>/collect/out"
python -m pypto.tools.clean_sim_trace "$TRACE"/OPPROF_* -o trace-out
```

它写出 `trace.clean.json`，泳道按数据流顺序排列 —— **MTE2 → MTE1 → CUBE → VECTOR → FIXPIPE → MTE3** —— 外加 `instr_metrics.json`，含逐指令的流水、cycle 数与 vector 利用率。

**怎么读：** 逐流水的 cycle 拆解就是「这个 kernel 到底在干什么」的答案。全是 MTE2 的 kernel 是搬运受限（给它加 double buffer）；全是 VECTOR 而利用率很低的，是形状不适合 vector 单元；matmul kernel 上出现 `CUBE = 0` cycle，意味着 trace 是退化的，而不是 matmul 免费。

**前置条件是实打实的：** 一个含 `ptoas/` kernel 的已构建 case、一套支持 TL 的 CANN、以及 `msopprof` worker。skill 会预检这三项，并以明确的消息提前失败。

## 硬件粒度

最常见的那一条，编译器每次编译都替你检查。`PH001`（`TileInnermostDimGranularity`）检查每一个 `tile.load` / `tile.store`，标出最内维小于该 backend 推荐搬运粒度的那些 —— a2a3 上是 **512 B 的 L2 cache line**：

```text
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost dim = 64B
(tile fp32[16], target_memory=Mat); moves 1024B as 16 x 64B rows; recommended >= 512B
for backend a2a3 (L2 cache line = 512B). Consider increasing tile shape on the
innermost axis. at examples/intermediate/05_assemble.py:70:5
```

有两点让这条提示可用而不是噪声：

- **按 `moves …` 子句排序，不要按条数。** 一个 `[1024, 64]` 的权重面板和一个 `[16, 64]` 的激活面板产生看上去一模一样的提示，而流量差 64 倍。区分它们的正是这个子句。
- **惩罚的量级是真实的。** `b_trans` matmul 那种情形 —— GM→Mat 的权重 load 搬的是 128 B 的行，而推荐值是 512 B —— 实测有 **16–25%** 的惩罚。

**怎么改：** 加宽 tile 的最内轴，或者转置一下，让被搬运的那一维是连续的。如果这个 tiling 是刻意的、而且你测过，就用 `disabled_diagnostics` 关掉这项检查，而不是忍着噪声。

## 逃到手写 kernel

**何时适用：** 你已经有一个调好的 AscendC kernel，或者这个 kernel 上 PyPTO 的 codegen 追不上手写代码。

`@pl.function(external_source=...)` 用一份手写的 C++ `.cpp` 来支撑一个 `AIC` / `AIV` 函数。该函数的函数体是一个裸 `...` —— 只有签名 —— 编排照常调用它，而编译器跳过对它的 PyPTO codegen，改为编译所引用的源码。

```python
@pl.function(type=pl.FunctionType.AIV, external_source="kernels/my_kernel.cpp")
def my_kernel(x: pl.Tensor[[128, 128], pl.FP16], out: pl.Out[pl.Tensor[[128, 128], pl.FP16]]):
    ...
```

相对路径相对于定义它的文件所在目录解析。完整约定见 [函数 § 外部 kernel](../language/01-functions.md)。

**代价：** 对这个函数而言你把编译器抛在了身后 —— 没有 layout 推断、没有内存规划、没有性能提示，周围 IR 形状变了时也没有任何保护。签名从此是一份由你手工维护的契约。

## 参见

- [内存](05-memory.md) —— double buffer 需要的那些 buffer 从哪来。
- [精度](../precision/index.md) —— 当算法改动动了数值时。
