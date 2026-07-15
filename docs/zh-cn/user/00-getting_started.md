# PyPTO 入门指南

## 什么是 PyPTO？

PyPTO 是一个基于 Python 的 Ascend NPU 内核编程框架。使用 `pypto.language` 模块编写计算内核，PyPTO 将其编译为优化的设备代码。

```python
import pypto.language as pl
from pypto import ir
```

所有内核代码使用 `pl` 命名空间。`ir` 模块提供编译和 IR 工具。

## Hello World：向量加法（张量级别）

最简单的内核操作 **Tensor（张量）** —— DDR 内存中的高级数组。PyPTO 自动处理数据搬运和内存分配。

```python
import pypto.language as pl
from pypto import ir

@pl.function
def vector_add(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
    return result
```

**逐行说明：**

| 行 | 功能 |
| -- | ---- |
| `@pl.function` | 将 Python 函数体解析为 PyPTO IR |
| `a: pl.Tensor[[64], pl.FP32]` | 输入：一维张量，64 个元素，32 位浮点 |
| `pl.add(a, b)` | 逐元素加法（分发到张量加法） |
| `return result` | 函数返回一个张量 |

装饰器执行后，`vector_add` 是一个 `ir.Function` 对象 —— 不是 Python 可调用函数。打印 IR：

```python
print(vector_add.as_python())
```

## Tile 内核：Load-Compute-Store

要进行硬件级控制，使用 **Tile（数据块）** —— 片上内存缓冲区。显式地从 DDR 加载数据、在片上计算、然后将结果存回。

```python
@pl.function
def vector_add_tile(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
    output: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    # 从 DDR 加载到片上（Vec 内存）
    a_tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
    b_tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])

    # 片上计算
    result: pl.Tile[[64], pl.FP32] = pl.add(a_tile, b_tile)

    # 存回 DDR
    out: pl.Tensor[[64], pl.FP32] = pl.store(result, [0], output)
    return out
```

**与张量版本的关键区别：**

| 概念 | 张量级别 | Tile 级别 |
| ---- | -------- | --------- |
| 数据位置 | DDR（自动） | 显式 load/store |
| 类型 | `pl.Tensor` | `pl.Tile`（片上） |
| 输出参数 | 返回值 | `pl.Out[pl.Tensor[...]]` |
| 内存控制 | 编译器决定 | 用户决定 |

**`pl.load(tensor, offsets, shapes)`** 从 DDR Tensor 拷贝一个区域到片上 Tile。

**`pl.store(tile, offsets, output_tensor)`** 将 Tile 拷贝回 DDR。

## 循环与累加

使用 `pl.range()` 进行循环。通过 `init_values` 实现循环携带值（累加器）：

```python
@pl.function
def sum_elements(
    a: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[1], pl.FP32]:
    zero: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (acc,) in pl.range(64, init_values=(zero,)):
        elem: pl.Tensor[[1], pl.FP32] = pl.slice(a, [1], [i])
        new_acc: pl.Tensor[[1], pl.FP32] = pl.add(acc, elem)
        acc_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_acc)

    return acc_out
```

**`init_values` 工作原理：**

1. `init_values=(zero,)` —— 累加器的初始值
2. `for i, (acc,)` —— `i` 是循环变量，`acc` 是当前累加器
3. `pl.yield_(new_acc)` —— 将 `new_acc` 作为下一次迭代的累加器
4. 循环结束后，`acc_out` 保存最终值

简单循环（无累加器）：

```python
for i in pl.range(10):
    # i 从 0 到 9
    ...

for i in pl.range(0, 100, 2):
    # i 从 0 到 98，步长 2
    ...
```

## 多函数程序

使用 `@pl.program` 将多个相互调用的函数组合在一起：

```python
@pl.program
class VectorAddProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        a_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        b_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.add(a_tile, b_tile)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(
            result, [0, 0], output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor(
            [128, 128], dtype=pl.FP32
        )
        c = self.kernel_add(a, b, c)
        return c
```

**关键概念：**

| 概念 | 说明 |
| ---- | ---- |
| `@pl.program` | 装饰类 → 成为 `ir.Program` |
| `self` | 必需的第一个参数；从 IR 中去除 |
| `self.kernel_add(...)` | 程序内跨函数调用 |
| `FunctionType.InCore` | 在 AICore 上运行（计算内核） |
| `FunctionType.Orchestration` | 在主机端运行（任务图协调器） |

**函数类型（FunctionType）：**

- **`Opaque`**（默认）—— 无特定执行上下文
- **`InCore`** —— AICore 计算内核；使用 load/store 进行数据搬运
- **`Orchestration`** —— 主机端函数，创建张量并调度 InCore 任务

## 编译

编译程序以生成设备代码：

```python
from pypto.backend import BackendType

output_dir = ir.compile(
    VectorAddProgram,
    strategy=ir.OptimizationStrategy.Default,
    dump_passes=True,
    backend_type=BackendType.Ascend910B,
)
print(f"Generated code in: {output_dir}")
```

**`ir.compile()` 参数：**

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `program` | （必需） | 要编译的 `ir.Program` |
| `output_dir` | `None` → `<base>/<name>_<timestamp>` | 代码生成、报告及（开启 dump 时）各 pass IR 的输出目录。`<base>` 取自 `PYPTO_PROG_BUILD_DIR` 环境变量，未设置时为 `build_output` |
| `strategy` | `OptimizationStrategy.Default` | Pass 流水线预设（`Default` 或 `DebugTileOptimization`） |
| `dump_passes` | `True` | 为 `True` 时，在每个 pass 后将 IR 快照写入 `output_dir/passes_dump/` |
| `backend_type` | `BackendType.Ascend910B` | Pass 与代码生成的目标硬件（`Ascend910B` 或 `Ascend950`） |
| `skip_ptoas` | `False` | 为 `True` 时跳过 ptoas，只生成原始 `.pto`（MLIR），不生成已编译的 C++ 包装代码 |
| `verification_level` | `None` | 可选的 `ir.VerificationLevel` 覆盖；`None` 表示使用默认（或环境变量 `PYPTO_VERIFY_LEVEL`） |

`DebugTileOptimization` 只是用于观察 PTO tile 流水线的调试捷径。除非你正在
专门排查策略选择或 pass 顺序，否则应优先使用 `Default`。

**不编译直接查看 IR：**

```python
# 打印单个函数
print(vector_add.as_python())

# 打印整个程序
print(VectorAddProgram.as_python())

# 省略中间类型标注（简洁模式）
print(vector_add.as_python(concise=True))
```

## 在 worker 上复用权重（DeviceTensor）

当同一个大张量被多次内核调用复用 —— 例如前向计算每个 batch 都要用到的权重矩阵 ——
每次都重新上传会浪费带宽。`ChipWorker.alloc_tensor` 在 device 上分配一块常驻内存，并返回
一个 `DeviceTensor` 句柄；`CompiledProgram` 接受它替代 `torch.Tensor` 入参。runtime
把这块 buffer 视为已经驻留在 device 上，对该入参跳过 H2D 与 D2H 拷贝。

```python
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig

compiled = ir.compile(MyKernel)

with ChipWorker(config=RunConfig(platform="a2a3sim")) as w:
    weight = w.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    for batch in batches:
        out = torch.empty(batch.shape[0], 4096, dtype=torch.float16)
        compiled(batch, weight, out)
    w.free_tensor(weight)
```

### 注意事项

- `DeviceTensor` 永远不会被拷回 host。如果内核写入了它，需要在同一个 ChipWorker
  实例上显式调用 `w.copy_from(host_ptr, t.data_ptr, t.nbytes)` 读回结果。
- 必须在 ChipWorker 关闭之前用 `w.free_tensor(t)` 释放句柄，否则该内存会泄漏到
  ChipWorker 生命周期结束。
- 只有分配它的那个 ChipWorker 实例可以使用该 buffer。

### 显式 dispatch（`worker.run`、`worker.register`）

上面的 `with ChipWorker(): compiled(...)` 隐式模式依赖 `ContextVar` 发现：块内任何
`compiled(...)` 调用都会找到当前活跃的 worker 并复用它。这对脚本写法很方便，但 worker
对象本身被藏起来了 —— 库代码需要把 worker 传来传去，或者常驻服务想预注册多个 kernel
时，应该显式地驱动 dispatch：

```python
worker = ChipWorker(config=RunConfig(platform="a2a3sim"))
try:
    out = worker.run(compiled, a, b)                 # 单次
    handle = worker.register(compiled)               # 预注册
    for _ in range(1000):                            # 热循环，无 cid lookup
        handle(a, b, out)
finally:
    worker.close()                                   # cid + DeviceTensor 统一释放
```

`worker.register(compiled)` 立即触发 `compile_and_assemble` + simpler `register`，
配置错误会在这里抛出而不是到第一次 dispatch 才暴露。返回的 `RegistrationHandle` 是
可调用的、支持 `with handle:` 作用域清理，也有 `handle.unregister()` 用于显式提前
关闭。对同一个 `compiled.chip_callable` 多次 `register` 返回的是同一个 cid 的别名；
真正的 simpler 反注册在 `worker.close()` 里集中做。

`@pl.jit` 内核走同样的流程，先经过 `JITFunction.compile()`：

```python
@pl.jit
def add_kernel(a, b, out): ...

compiled = add_kernel.compile(sample_a, sample_b, sample_out)
handle = worker.register(compiled)
for batch in stream:
    handle(batch.a, batch.b, batch.out)
```

`compile()` 只读取每个张量参数的 shape/dtype —— 从不触碰内容 —— 所以这些样例
张量纯粹是元数据载体。

### 从签名编译（无需样例张量）

当每个张量参数都**完整注解**了 shape 和 dtype 时，`compile()` 可以直接从签名
读出整个 shape 契约 —— **不传任何位置参数**即可，样例张量全部省掉：

```python
HIDDEN, VOCAB = 4096, 152064
M = pl.dynamic("M")          # 运行期动态维

@pl.jit
def prefill_fwd(
    hidden: pl.Tensor[[M, HIDDEN], pl.BF16],
    lm_head: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[M, VOCAB], pl.FP32]],
): ...

# 没有 torch.empty(...) 占位张量 —— shape 全部来自注解。
compiled = prefill_fwd.compile()
```

对于签名很大的内核，这是更符合直觉的路径：shape 契约只在签名一处声明，而不是
再写成一长串一次性的 `torch.empty(...)`。细节：

- **静态维**（`HIDDEN`、`VOCAB` …）来自注解常量。
- **动态维**（`pl.dynamic` / `bind_dynamic`）无需给值 —— 编译产物与具体 extent
  无关，`compile()` 与等价的 `compile(sample_tensors)` 共享同一 cache 条目。
- **标量参数**在签名里没有值 —— 用关键字参数传入，例如
  `kernel.compile(num_tokens=128)`。
- **bare `pl.Tensor`**（无 shape）无从读取，会给出明确报错；请补全
  `pl.Tensor[[...], dtype]` 注解，或回退到 `compile(*sample_tensors)`。

完整三种使用模式（推理服务、训练循环、register/dispatch 开销验证）见
`examples/runtime/explicit_dispatch.py`。

### 读取单次 launch 的计时

`worker.run` / `handle(...)` 只返回张量输出，不再暴露单次 launch 的计时对象。
runtime 以 `[STRACE]` 日志标记的形式输出每次运行的 host/device 计时（simpler
PR #1177，在 `SIMPLER_DFX` 下默认开启）；用 simpler 的 `strace_timing` /
`device_log_timing` 工具解析这些标记，而不是读取返回值。需要 per-task 的 device
计时时，开启 L2 swimlane DFX（`RunConfig(enable_l2_swimlane=True)`）并读取
`l2_swimlane_records.json`。

### 性能基准（`benchmark`）

对于 register-once + 多轮（rounds）模式，`pypto.runtime.benchmark` 封装了循环
与聚合：它注册 *compiled* 一次并发起 `rounds` 次廉价 launch（不再每轮重付
register/load），读取每次 launch 的 `[STRACE]` 标记并返回 `BenchmarkStats`：

```python
from pypto.runtime import benchmark

stats = benchmark(compiled, [a, b, c], rounds=100, warmup=3,
                  platform="a2a3", device_id=0)
print(stats.device_wall_us_median, stats.device_wall_us_min, len(stats.samples))
```

常见情况传 `platform=` / `device_id=`；需要 `block_dim` / `aicpu_thread_num` 等
精细控制时传完整的 `RunConfig`（通过 `config=`）——两者不能同时给。聚合指标同时
以 `device_wall_us_*` 和更短的 `device_us_*` 两套命名暴露，`samples` 是原始
`device_wall_us` 列表的别名。

`benchmark` 从 `[STRACE]` 标记读取计时（simpler PR #1177）：它在 worker 生命周期内
将 runtime 日志级别提升到 `v9`，并在测量循环期间以 fd 级别捕获 `stderr`，因此循环
期间产生的 stderr 会被转存到临时文件，而非实时打印。`device_wall_us` 在 L2 单芯片
运行时是真实的 NPU 墙钟（分布式见下方 L3 说明）；在未开启 `SIMPLER_HOST_STRACE` 的
runtime 上或 `*sim` 平台上为 `0`（用 `stats.all_zero_device` 判断）。

除聚合值外，每次测量 launch 的完整 `[STRACE]` span 树保存在 `stats.invocations`
（`TraceInvocation` 列表，已排除 warmup）。可用分支连接符渲染——单次 launch，或跨所有
launch 求均值并标注每个节点的离散度（`spread` 取 `"stdev"`（默认）、`"minmax"`、
`"both"` 或 `"none"`）：

```python
stats.print_tree(launch=0)            # 某次 launch 的嵌套 span 树
stats.print_mean_tree(spread="both")  # 每节点均值 + ±stdev + [min..max]
```

```text
mean of 20 launches (warmup 5 excluded); each node: mean ±stdev [min..max]:
simpler_run                71784.1us  ±6797.5  [66482.4..89832.6]
|- bind                    27943.6us  ±4163.7  [24836.7..37713.3]
|- runner_run               3030.8us   ±184.4    [2822.3..3694.7]
|  `- device_wall [dev]     2005.2us    ±74.6    [1875.1..2173.2]
|     `- graph_build [dev]  1634.8us    ±64.6    [1490.2..1777.6]
`- validate                40697.7us  ±3063.5  [38606.3..48200.6]
```

嵌套关系由点分 span 名重建,因此设备域 span（`...device_wall.*`,标 `[dev]`）会正确挂在
其 host 父节点下。每个节点是一段**墙钟窗口而非时间划分**:子节点可能并发重叠（如 `orch`/
`sched` 并行）或处于不同时钟域（`runner_run` 是 host 墙钟、`device_wall` 是 NPU 墙钟）,
故子节点时长之和不必等于父节点。要取原始 span 用
`stats.invocations[i].by_name()[<name>].dur_us`。

`benchmark` 也接受 L3 的 `DistributedCompiledProgram`（经 `compiled.prepare()`
打开）：传共享内存 host 张量（或 `DeviceTensor`），并省略 `platform=` / `device_id=`
（设备集在编译期由 `distributed_config` 固定）。L3 没有单一的 DAG 级 device 墙钟，
因此计时由各 rank 的 chip 子进程标记折叠成逐轮样本——headline `device_wall_us[k]`
是各卡该轮 dispatch device 墙钟之和再跨卡取 max。四个指标统一查询：

```python
stats.per_round("device" | "host" | "effective" | "union")  # -> 每轮一个值
stats.per_rank("device" | "host" | "effective")             # -> {pid: 每轮一个值}
```

这两个视图都是**按 rank 按轮**聚合的：每个值是该 rank 该轮内多次 dispatch 的
**求和**（一张卡串行执行它的多次 dispatch），因此是"每 rank 每轮"的量，**不是**逐
dispatch 的量。当某 rank 每轮恰好只有 1 次 dispatch 时，求和即那唯一一次 dispatch 的值；
无论哪种情况，要看逐次 dispatch 明细都读 `stats.rounds_dispatches[k][pid]`（见下）。

`effective` 是 orch∪sched 的设备执行窗口（每卡 L2 Effective）；`union` 是跨卡 host
时间轴并集窗口（能反映起跑错位——host 域，含派发开销）。可导航的
`round -> rank -> [dispatch]` 网格是 `stats.rounds_dispatches`，每个
`TraceInvocation` 暴露 `.task`（callable 标识）、`.device_wall_us`、`.host_wall_us`、
`.effective_us`。纯 device 的跨卡端到端墙钟目前无法从标记恢复。若 dispatch 形状非
确定，则 `stats.fallback_flattened` 被置位，per-rank / `union` 视图为空。

### 分布式（L3+）程序

`ir.compile` 对 L3+ 分布式程序返回的 `DistributedCompiledProgram` 与 `CompiledProgram`
一样接受 `DeviceTensor` 入参：用 worker 常驻 buffer 替代 `torch.Tensor`，runtime 即对该
参数跳过 H2D/D2H。这是在 generate 循环的多次 dispatch 之间保持大块静态权重常驻的推荐做法。

```python
import torch
from pypto.runtime import DeviceTensor

compiled = ir.compile(MyDistributedProgram)   # 返回 DistributedCompiledProgram
weight = DeviceTensor(dev_ptr, (1024, 4096), torch.float16)   # 调用方自管 buffer
compiled(x, weight, out)                       # weight：无 H2D/D2H 拷贝
```

#### 跨多次 dispatch 复用 setup（`prepare()`）

`compiled(*args)` 每次调用都会跑完整的分布式 setup（逐 chip 装配、构造 simpler Worker 并 fork）。
对反复 dispatch 同一程序的常驻服务（如 generate 循环），可调用一次 `compiled.prepare()` 得到
一个 `DistributedWorker` 句柄：setup 只做一次，多次 dispatch 复用同一个 worker。

可变的 per-call IO buffer 是**在 `prepare()` 之前分配的共享内存 host 张量**并原地复用，这样子进程的写入
对父进程可见。不可变输入也可以保留为普通的 CPU 连续张量：必须在 fork 前将同一个对象传给
`compiled.prepare(inherited_host_tensors=[...])`，runtime 会保留其引用并通过继承的写时复制映射读取。
该例外仅适用于输入；输出、原地修改张量以及未注册的非共享张量仍会被拒绝。大块静态权重也可以用
`rt.alloc_tensor` 一次性上传，但其 `init` 源仍必须是 `prepare()` 之前分配的共享张量。

```python
compiled = ir.compile(MyDistributedProgram)

# 共享内存 host buffer —— 必须在 prepare() 之前分配
host_x = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_out = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_weight = load_weight().share_memory_()

with compiled.prepare() as rt:                  # setup 只跑一次
    weight = rt.alloc_tensor(host_weight.shape, host_weight.dtype, init=host_weight)
    for step in generate_steps:
        host_x.copy_(next_input(step))          # 原地刷新输入
        rt(host_x, weight, host_out)            # host shm IO + 常驻权重
        consume(host_out)                       # 直接读输出
    rt.free_tensor(weight)
# 退出时自动 rt.close()
```

#### 把权重按卡切分常驻（`alloc_stacked_tensor`）

当 HOST orchestrator 把一个 `[B, N, M]` 权重按首维切片并分发到每张卡——即规范写法
`for r in range(world_size): child(x[r], device=r)`——直接传整块 host 张量会在**每次**
dispatch 都把 `x[r]` 切片重新上传到对应卡。要让每个分片**只上传一次**并常驻在自己那张卡上,
用 `rt.alloc_stacked_tensor` 构造一个 `StackedDeviceTensor`:

```python
host_w = load_weight().share_memory_()           # [B, N, M],B == world_size
host_a = torch.zeros((B, N, M), dtype=...).share_memory_()
host_out = torch.zeros((B, N, M), dtype=...).share_memory_()

with compiled.prepare() as rt:
    w = rt.alloc_stacked_tensor(host_w)          # 第 i 片上传到第 i 张卡,只传一次
    for step in steps:
        host_a.copy_(next_input(step))
        rt(host_a, w, host_out)                  # x[r] 解析到常驻的第 r 片
        consume(host_out)
    rt.free_stacked_tensor(w)
```

内部每个分片 `host_w[i]` 都成为一个 worker 常驻的 `DeviceTensor`,因此生成代码里的
`x[r]` 取下标会跳过 H2D 上传(`child_memory`)。分片在 `close()` 时自动释放,也可提前用
`free_stacked_tensor` 释放。

和单个 `DeviceTensor` 一样,`StackedDeviceTensor` 也不会被自动拷回。若要一次把每个分片
当前的设备内容读回主机——例如某一步结束时读回常驻的 KV cache——可用
`rt.copy_stacked_from(w, host_out)`,即 `alloc_stacked_tensor` 的对称读回接口。`host_out`
原地填充(`host_out[i]` 接收第 `i` 片);与上传源一样,它必须是形状和 dtype 与该 stack
匹配、且在 `prepare()` **之前**分配的 CPU、连续、**共享内存** `[B, *tail]` 张量
(调用 `.share_memory_()`):D2H 拷贝在 fork 出的 chip worker 中执行,只能写它在 fork
时继承的主机内存。

首维就是分片维,`B` 必须等于程序分发到的卡数。默认第 `i` 片落在第 `i` 个 worker 上
(对应 `device=r`)。如果程序用的是**非恒等**放置——置换或子集卡(如 `device=2*r`,或字面量
`device=1` / `device=0`)——就要传匹配的 `worker_ids`,其中 `worker_ids[i]` 是程序提交
`x[i]` 那次任务所用的 worker:

```python
# orchestrator 把 x[0] 分发到卡 1、x[1] 分发到卡 0
w = rt.alloc_stacked_tensor(host_w, worker_ids=[1, 0])
```

`worker_ids` 必须互不相同且落在 `[0, world_size)` 内;与程序的 `device=` 不匹配会把分片放到
错误的卡上、读到垃圾数据。

`rt.alloc_tensor(..., worker_id=r)` 同样接受非默认的 `worker_id`,可把单个常驻
`DeviceTensor` 放到任意卡(`free_tensor` 时传相同的 `worker_id`)。

#### 在同一个 worker 上调度多个程序（multi-program）

Serving 场景需要把 prefill 和 decode 作为两个独立的 HOST 程序,共享同一个 L3
worker 和同一份设备常驻 KV cache。把一组兼容的 `DistributedCompiledProgram`
以列表形式传给 `DistributedWorker`,或等价地用
`prefill.prepare(extra_compiled=[decode])`——它们会在同一个 worker 上一次性
准备好,再用 `rt.run(compiled, *args)` 选择分发哪一个。各程序必须使用相同的
platform、runtime 和 device ids。多程序模式下 `rt(*args)` 这个快捷方式会被禁用
(目标程序有歧义)——一律用 `rt.run(...)`。worker 常驻的 `DeviceTensor`
(如 KV cache)在两个程序的多次 dispatch 之间始终有效。

可运行的端到端示例见
[`examples/runtime/multi_program_kv_cache.py`](../../../examples/runtime/multi_program_kv_cache.py)。

```python
from pypto.runtime import DistributedWorker, RunConfig

cfg = RunConfig(platform="a2a3", distributed_config=dc)
prefill_c = prefill.compile(host_prompt, kv_sample, config=cfg)   # @pl.jit.host kernel:
decode_c = decode.compile(host_token, kv_sample, host_logits, config=cfg)  # 只编译不下发

with DistributedWorker([prefill_c, decode_c]) as rt:    # 一个 worker,一次 fork
    kv_cache = rt.alloc_tensor(kv_shape, torch.float16)  # 两个程序共享常驻
    rt.run(prefill_c, host_prompt, kv_cache)             # 写入 KV cache
    for _ in range(max_new_tokens):
        rt.run(decode_c, host_token, kv_cache, host_logits)  # 读取/更新 KV cache
```

## 下一步

- **[语言指南](01-language_guide.md)** —— 类型、操作、控制流、内存和编译的完整参考
- **[操作参考](02-operation_reference.md)** —— 所有 `pl.*` 操作的查找表
