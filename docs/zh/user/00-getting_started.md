# 在设备上运行

数据常驻 worker、显式派发、性能测量，以及分布式程序的运行。

> **本页处于过渡状态。** 其入门部分已迁至 [快速上手](02-quickstart.md)。这里保留的是设备执行
> 与运行时内容，待 `execution/`、`performance/`、`distributed/` 三章落地后会再次搬迁：
>
> | 小节 | 去向 |
> | ---- | ---- |
> | 常驻设备张量、显式派发、从签名编译 | `execution/01-run.md` |
> | 单次 launch 计时、`benchmark` | `performance/06-host.md` |
> | 分布式（L3+）执行 | `distributed/03-execution.md` |
>
> 这里的内容都没有废弃，只是地址是临时的。
>
> **前置**：[快速上手](02-quickstart.md)，以及一台有设备或模拟器平台的机器。与快速上手不同，
> 下面的例子会真正派发到硬件。

## 在 worker 上复用权重（DeviceTensor）

当同一个大张量被多次内核调用复用 —— 例如前向计算每个 batch 都要用到的权重矩阵 ——
每次都重新上传会浪费带宽。`ChipWorker.alloc_tensor` 在 device 上分配一块常驻内存，并返回
一个 `DeviceTensor` 句柄；`CompiledProgram` 接受它替代 `torch.Tensor` 入参。runtime
把这块 buffer 视为已经驻留在 device 上，对该入参跳过 H2D 与 D2H 拷贝。

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig

ROWS, COLS = 128, 128
PLATFORM = "__PLATFORM__"


@pl.jit
def add_kernel(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        ta = pl.load(a, [0, 0], [ROWS, COLS])
        tb = pl.load(b, [0, 0], [ROWS, COLS])
        pl.store(pl.add(ta, tb), [0, 0], out)
    return out


torch.manual_seed(0)
A = torch.randn(ROWS, COLS, dtype=torch.float32)
B = torch.randn(ROWS, COLS, dtype=torch.float32)

# A DeviceTensor carries no shape/dtype for the @pl.jit specializer to read, so
# the resident-weight pattern below runs a *compiled* program rather than the
# jit entry directly.
compiled = ir.compile(add_kernel.lower(A, B, torch.zeros(ROWS, COLS)), platform=PLATFORM)
```

<!-- doctest: run -->
```python
cfg = RunConfig(platform=PLATFORM)

with ChipWorker(config=cfg) as w:
    resident = w.alloc_tensor((ROWS, COLS), torch.float32, init=B)  # stays on device
    for _ in range(3):                                              # three "batches"
        out = torch.zeros(ROWS, COLS, dtype=torch.float32)
        w.run(compiled, A, resident, out)
        torch.testing.assert_close(out, A + B, rtol=1e-4, atol=1e-4)
    w.free_tensor(resident)
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

<!-- doctest: run -->
```python
worker = ChipWorker(config=RunConfig(platform=PLATFORM))
try:
    handle = worker.register(compiled)               # eager registration
    out = torch.zeros(ROWS, COLS, dtype=torch.float32)
    for _ in range(3):                               # hot loop, no cid lookup
        handle(A, B, out)
    torch.testing.assert_close(out, A + B, rtol=1e-4, atol=1e-4)
finally:
    worker.close()                                   # cids + DeviceTensors released
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
- **标量参数**在签名里没有值 —— 用关键字参数传入。传字面量会把该值**特化**进
  产物，例如 `kernel.compile(num_tokens=128)` 编出的内核只认 128。改传
  `pl.RUNTIME` —— `kernel.compile(num_tokens=pl.RUNTIME)` —— 则**不特化**：该参数
  在生成的程序里仍是真正的 `pl.Scalar` 参数，值在 dispatch 时给出；它与动态维一样
  不进 cache key，一份产物服务所有取值。该值要通过编译产物给出 —— `compiled(...)`
  或 `worker.register(compiled)` 拿到的 handle —— 而不是直接调用内核：
  `kernel(x, out, 128)` 会按 128 重新特化并编出另一份产物。`pl.RUNTIME` 也可以写成
  签名默认值（`num_tokens: pl.Scalar[pl.INT32] = pl.RUNTIME`），这样每个 `compile()`
  调用点都不必再传关键字。
- **bare `pl.Tensor`**（无 shape）无从读取，会给出明确报错；请补全
  `pl.Tensor[[...], dtype]` 注解，或回退到 `compile(*sample_tensors)`。

完整三种使用模式（推理服务、训练循环、register/dispatch 开销验证）见
`examples/runtime/explicit_dispatch.py`。

### 读取单次 launch 的计时

`worker.run` / `handle(...)` 只返回张量输出，不再暴露单次 launch 的计时对象。
runtime 以 `[STRACE]` 日志标记的形式输出每次运行的 host/device 计时（simpler
PR #1177，在 `SIMPLER_DFX` 下默认开启）；用 simpler 的 `strace_timing` /
`device_log_timing` 工具解析这些标记，而不是读取返回值。需要 per-task 的 device
计时时，开启 L2 swimlane DFX（`RunConfig(enable_chip_swimlane=True)`）并读取
`chip_swimlane_records.json`。

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

常见情况传 `platform=` / `device_id=`；需要 `aicpu_thread_num` 等
精细控制时传完整的 `RunConfig`（通过 `config=`）——两者不能同时给。聚合指标同时
以 `device_wall_us_*` 和更短的 `device_us_*` 两套命名暴露，`samples` 是原始
`device_wall_us` 列表的别名。

`compiled` 不必来自一次新编译：`CompiledProgram.from_dir(work_dir)` 能从已有的
构建目录重建出可 benchmark 的程序，于是可以手改生成的 orchestration cpp /
`.pto` 后直接重测，无需重新编译——但要配合
[对重放的单芯片构建做 benchmark](../dev/03-runtime-replay.md#对重放的单芯片构建做-benchmark)
里的 `.pto` 重建与二进制缓存失效两步，否则改动可能没被编译进去。

`benchmark` 从 `[STRACE]` 标记读取计时（simpler PR #1177）：它在 worker 生命周期内
将 runtime 日志级别设为 `timing`，并在测量循环期间以 fd 级别捕获 `stderr`，因此循环
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
chip.run                   71784.1us  ±6797.5  [66482.4..89832.6]
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

`benchmark` 也接受 L3 的 `DistributedCompiledProgram`，并会自行打开 prepared
worker。参数应传共享内存 host 张量；外部分配的常驻张量属于另一个 worker，不能传入。
同时省略 `platform=` / `device_id=`（设备集在编译期由 `distributed_config` 固定）。
L3 没有单一的 DAG 级 device 墙钟，因此计时由各 rank 的 chip 子进程标记折叠成逐轮
样本——headline `device_wall_us[k]` 是各卡该轮 dispatch device 墙钟之和再跨卡取 max。
四个指标统一查询：

```python
stats.per_round("device" | "host" | "effective" | "union")  # -> 每轮一个值
stats.per_rank("device" | "host" | "effective")             # -> {pid: 每轮一个值}
stats.per_dispatch("device" | "host" | "effective")         # -> {(pid, slot): 每轮一个值}
```

`per_round` / `per_rank` 是**按 rank 按轮**聚合的：每个值是该 rank 该轮内多次
dispatch 的**求和**（一张卡串行执行它的多次 dispatch），因此是"每 rank 每轮"的忙时量，
**不是**逐 dispatch 的量。

`per_dispatch` 是**不做融合**的视图——它不求和。它以 `(pid, slot)` 为键，`slot` 是该次
dispatch 在本 rank 该轮内的序号。这样一个 rank 每轮的多次 dispatch 会各自保留一条序列，
而不是被加成一个数。

slot 要能代表某次 dispatch，前提是该 rank 每轮发出的 callable **顺序也相同**。dispatch
数恒定并不保证这一点，因此解析时会校验：若某个 slot 在各轮中出现过不同的 task，则置位
`stats.unstable_dispatch_slots`，逐 dispatch 视图返回空，而不是把不同 kernel 平均到
第一轮的标签下。轮次边界不受影响，`per_rank` / `per_round` 仍然有效。`stats.dispatch_tasks()` 给出每个 slot 实际执行的编排函数
名，`stats.dispatch_groups()` 返回其每轮的 `TraceInvocation`。

```python
stats.per_dispatch("device")   # {(4242, 0): [4.1, 3.8, ...], (4242, 1): [6.3, 6.5, ...]}
stats.dispatch_tasks()         # {(4242, 0): "prefill_orch", (4242, 1): "decode_orch"}
```

marker 本身不带名字，只带 `hid` —— 该 callable 编排 `.so` 的 ELF Build-ID（仍可通过
`TraceInvocation.task` 读到）。pypto 对交给 runtime 的同一份 `.so` 字节重算该 Build-ID，
并在装配时与该编排生成的名字配对，从而还原名字（`TraceInvocation.task_name`）。配对不可得时
退化为原始哈希：`*sim` 平台（其 host 用 runtime `callable_id` 而非 Build-ID 填充 `hid`），
或该 callable 是在别的进程里装配的。

均值树视图同样按 dispatch 区分：L3 下 `print_mean_tree()` **按 `(pid, slot)` 各输出一棵
树**，不再把一个 rank 的不同 kernel 平均进同一棵树；`pid=` / `slot=` 可收窄到某一次
dispatch。`format_tree()` 的 launch 表头也带上 `round=` / `slot=`。`mean_invocation()`
只返回一棵树，因此除非用 `pid=` / `slot=` 选定单次 dispatch，否则会抛错。

`effective` 是 orch∪sched 的设备执行窗口（每卡 L2 Effective）；`union` 是跨卡 host
时间轴并集窗口（能反映起跑错位——host 域，含派发开销）。可导航的
`round -> rank -> [dispatch]` 网格是 `stats.rounds_dispatches`，每个
`TraceInvocation` 暴露 `.task`（callable 标识）、`.device_wall_us`、`.host_wall_us`、
`.effective_us`。纯 device 的跨卡端到端墙钟目前无法从标记恢复。若 dispatch 形状非
确定，则 `stats.fallback_flattened` 被置位，per-rank / `union` 视图为空。

### 分布式（L3+）程序

完整的分布式编程模型——从 `alloc_window_buffer` 到 `allreduce`、`barrier`、
`broadcast` 等集合通信——见 [分布式编程](distributed/00-model.md)。下面展示
InCore kernel（执行平面）的 mesh allreduce Hello World 示例；完整的可运行程序
还需包含 host 编排器、`ir.compile` 及分布式 worker 设置，详见上述指南：

```python
import pypto.language as pl
import pypto.language.distributed as pld

NR = pl.dynamic("NR")

@pl.program
class HelloAllReduce:
    @pl.function(type=pl.FunctionType.InCore)
    def reduce_step(
        self,
        inp: pl.Tensor[[1, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
        data: pl.InOut[pld.DistributedTensor[[1, 256], pl.FP32]],
        signal: pl.InOut[pld.DistributedTensor[[NR, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 256], pl.FP32]:
        ctx = pld.get_comm_ctx(data)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)

        # 1. Stage-in：将本地输入复制到本 rank 的 window 分片。
        data = pl.store(pl.load(inp, [0, 0], [1, 256]), [0, 0], data)

        # 2. Barrier：通知每个对端，然后等待每个对端。
        for peer in pl.range(nranks):
            if peer != my_rank:
                pld.system.notify(signal, peer=peer, offsets=[my_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        for src in pl.range(nranks):
            if src != my_rank:
                pld.system.wait(signal, offsets=[src, 0],
                                expected=1, cmp=pld.WaitCmp.Ge)

        # 3. 计算：加载自身分片，remote-load 每个对端，累加。
        acc = pl.load(data, [0, 0], [1, 256])
        for peer in pl.range(nranks):
            if peer != my_rank:
                peer_tile = pld.tile.remote_load(
                    data, peer=peer, offsets=[0, 0], shape=[1, 256])
                acc = pl.add(acc, peer_tile)

        # 4. Stage-out：将累加器写入本地输出。
        out = pl.store(acc, [0, 0], out)
        return out
```

指南包含**逐行解读、ring allreduce 权衡、notify/wait 握手模式以及调试表格**。
完整章节见 [distributed/index.md](distributed/index.md)。

L3+ 常驻张量必须由执行它的同一个 prepared `DistributedWorker` 分配：
`DeviceTensor` 使用它的 `alloc_tensor`，`StackedDeviceTensor` 使用它的
`alloc_stacked_tensor`。这些接口会在每个张量或分片上保留无地址 wire ABI 所需的
Simpler owner `Buffer`；手工用裸指针构造的张量无法安全跨过该边界。one-shot
`compiled(...)` 只接受 host `torch.Tensor` 参数，并会拒绝这两种常驻参数。

```python
import torch
compiled = ir.compile(MyDistributedProgram)   # 返回 DistributedCompiledProgram
with compiled.prepare() as rt:
    weight = rt.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    rt(x, weight, out)                         # weight：每次 dispatch 不再 H2D/D2H
```

#### 跨多次 dispatch 复用 setup（`prepare()`）

`compiled(*args)` 每次调用都会跑完整的分布式 setup（逐 chip 装配、构造 simpler Worker 并 fork）。
对反复 dispatch 同一程序的常驻服务（如 generate 循环），可调用一次 `compiled.prepare()` 得到
一个 `DistributedWorker` 句柄：setup 只做一次，多次 dispatch 复用同一个 worker。

per-call dispatch IO buffer 仍是**在 `prepare()` 之前分配的共享内存 host 张量**并原地复用，
这样子进程的写入对父进程可见。通过 `rt.alloc_tensor(init=...)`、
`rt.alloc_stacked_tensor(...)`、`rt.copy_to(...)`、`rt.copy_from(...)` 和
`rt.copy_stacked_from(...)` 执行的显式常驻上传与读回，会经过 runtime 管理的共享
Buffer staging。host 端为 `torch.Tensor` 时只需是 CPU 连续张量，可以在 `prepare()`
后创建；无需 `.share_memory_()`、fork 前分配或 `inherited_host_tensors`。低层
`copy_to`/`copy_from` 接口接收 host 张量的 `.data_ptr()` 和字节数。

```python
from pypto.runtime import DistributedWorker

compiled = ir.compile(MyDistributedProgram)

# 共享内存 host buffer —— 必须在 prepare() 之前分配
host_x = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_out = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()

with DistributedWorker(compiled) as rt:
    # 显式上传源可以是在 prepare() 后创建的普通张量
    host_weight = load_weight().contiguous()
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
host_a = torch.zeros((B, N, M), dtype=...).share_memory_()
host_out = torch.zeros((B, N, M), dtype=...).share_memory_()

with DistributedWorker(compiled) as rt:
    host_w = load_weight().contiguous()          # prepare() 后的普通 CPU 张量
    w = rt.alloc_stacked_tensor(host_w)          # 第 i 片上传到第 i 张卡,只传一次
    for step in steps:
        host_a.copy_(next_input(step))
        rt(host_a, w, host_out)                  # x[r] 解析到常驻的第 r 片
        consume(host_out)
    rt.free_stacked_tensor(w)
```

内部每个分片 `host_w[i]` 都成为一个保留 owner `Buffer` 的 worker 常驻 `DeviceTensor`,
因此生成代码里的 `x[r]` 会直接构造 wire Tensor 并跳过 H2D 上传。分片在 `close()` 时自动释放,也可提前用
`free_stacked_tensor` 释放。

和单个 `DeviceTensor` 一样,`StackedDeviceTensor` 也不会被自动拷回。若要一次把每个分片
当前的设备内容读回主机——例如某一步结束时读回常驻的 KV cache——可用
`rt.copy_stacked_from(w, host_out)`,即 `alloc_stacked_tensor` 的对称读回接口。`host_out`
原地填充(`host_out[i]` 接收第 `i` 片);它必须是形状和 dtype 与该 stack
匹配的 CPU 连续 `[B, *tail]` 张量。它可以在 `prepare()` 之后分配，因为 D2H
会经过 runtime 管理的共享 Buffer staging；不需要 `.share_memory_()` 或
`inherited_host_tensors`。

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

## 另请参阅

- [快速上手](02-quickstart.md) —— 编写并编译在这里派发的那些 kernel。
- [编程模型](03-programming-model.md) —— 为什么决定执行顺序的是运行时而不是语句顺序。
- [运行时 DFX](../dev/03-runtime-dfx.md) —— 本页计时与性能分析背后的诊断开关。
- [逐任务 Ring Sizing](../dev/05-runtime-ring-sizing.md) —— 调整运行时的逐任务环形缓冲区。
- [持久化 L3 执行](../dev/06-persistent-l3.md) —— 在多个已 prepare 的分布式程序间复用同一个 worker。
- [运行时文档](https://hw-native-sys.github.io/simpler/) —— 运行时自身的内部机制。
