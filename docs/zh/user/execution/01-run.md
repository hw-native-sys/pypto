# 运行

> **旧 Python 接口示例：** 本页为兼容现有用法保留 `platform=` 调用，
> 这些调用会发出 `DeprecationWarning`；新代码应使用 `backend_type` 和
> `execution_mode`。参见[目标映射](../../dev/08-entry-points.md#迁移旧-platform-示例)。

派发一个 `CompiledProgram`，并让常驻数据真的常驻。

## 概念

`CompiledProgram` 是一个句柄：指向编译产物，外加运行时启动它们所需的元数据。`ChipWorker` 持有设备连接与那些注册；派发要么是隐式的 —— 在 `@pl.jit` 函数上直接 `kernel(*args)` —— 要么是显式的，经由 worker，适用于库代码需要把 worker 传来传去、或服务化运行时想预注册许多 kernel 的场合。

早点搞清楚的一件事是：每次 launch 有什么东西跨过了 PCIe。默认每个张量实参都会 H2D 拷进去、再 D2H 拷回来。`DeviceTensor` 让一块 buffer 同时免掉这两次拷贝 —— 常驻权重或 KV cache 能成立，靠的就是它。

## 快速上手：把权重留在设备上

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

`alloc_tensor` 返回一个 `DeviceTensor`，编译后的程序在任何接受 `torch.Tensor` 的位置都接受它。运行时把这块 buffer 当作已经常驻，对该实参跳过 H2D 与 D2H。

## 机制

### `CompiledProgram` 的契约

| 成员 | 给你什么 |
| ---- | -------- |
| `output_dir` | 产物在哪 |
| `platform` / `backend_type` | 它是为什么构建的；worker 会校验前者 |
| `param_names` / `output_indices` / `has_return` | 调用形状，供自行绑定实参的 harness 使用 |
| `program` | 交给 `compile` 的那份 `Program` —— 通常是未经 pass 的，且经 `from_dir` 重建后为 `None` |
| `chip_callable` / `runtime_name` / `runtime_config` | 运行时侧的句柄 |
| `validate_ir` | 逐 pass 的语义对比（[精度](../precision/00-workflow.md)） |
| `from_dir` / `load` | 从已保存的产物目录重建句柄 |

> **`compiled.program` 不是产出那些产物的那份 IR。** 它是你交给 `compile` 的那个 `Program`，原样存下；codegen 实际跑的那份变换后程序并不保留，而经 `from_dir` 重建的句柄根本没有 program。
>
> 在通常的 `ir.compile(MyProgram)` 路径上，那份输入是未经 pass 的 IR。但不必然如此 —— 上面的准备代码编译的是 `add_kernel.lower(...)`，那本身已经降级过，所以**在那里** `compiled.program` 是过了 pass 的。这个属性不对此做任何承诺，它只是把收到的东西交回来。要专门拿降级后的 IR，用 `kernel.lower(*args)` 或读一份 pass dump。

### 显式派发

`worker.run(compiled, *args)` 是一次性的。`worker.register(compiled)` 返回一个跳过逐次查找的句柄，热循环要的就是它：

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

`register` 触发一次装配与加载；返回的句柄才是你每次 launch 调用的东西。`close()` 释放这些注册，以及调用方忘记释放的 `DeviceTensor`。

### `DeviceTensor`

| 规则 | 细节 |
| ---- | ---- |
| **由 worker 分配** | `w.alloc_tensor(shape, dtype, init=...)` |
| **不会自动拷回** | 用 `w.copy_from(host_ptr, t.data_ptr, t.nbytes)` 读回来 |
| **显式释放** | `w.free_tensor(t)`；`close()` 是兜底，不是方案 |
| **绑定在它的 worker 上** | 不能移交给另一个 `ChipWorker` |

A2/A3 上可用 `exportable=True` 分配 VMM 内存，再调用
`worker.export_buffer(tensor)`。同机消费者在自己的当前 ACL context 中通过
`simpler.buffer_export.BufferMapping` 导入返回的租约；原 worker 仍持有内存所有权。
调用方必须保持 worker 和 mapping 存活。释放 tensor 或关闭 worker 前，依次完成：
等待外部访问结束、注销外部资源、成功关闭 mapping、调用
`worker.release_export(exported)`。仍有导出租约时，free 和 close 会拒绝清理。
`BufferExportError` 通过 `error.exported` 提供尚未确认的租约，调用
`release_export(error.exported)` 确认并释放它。`BufferImportError` 通过
`error.mapping` 提供可重试清理的映射；应在导入 context 中完成清理后再释放源租约。

真实控制超时会保留原始载荷。迟到回执到达后重试 `release_export`，将对账同一条
RETAIN/RELEASE，而不重新执行原命令。这只恢复清理能力，不重新开放已隔离的计算
endpoint；应关闭该 worker 并创建新实例。分配回执未决时 close 会拒绝；分配回滚
未完成的资源仍由 worker 持有，可再次 close。VMM free 失败会撤销使用资格，同时
保留仅用于清理的身份：可重试 `free_tensor` 或 close，不可继续计算、复制或导出该 tensor。

共享范围限定为同机可信进程。native 导出关闭 PID 白名单校验，租约是生命周期合同，
不代表目标进程授权。同一 backing 应复用一个消费者 mapping；重复导入同一 shareable
handle 受 CANN 自身约束。

### 影响派发的 `RunConfig` 字段

`RunConfig` 同时携带编译侧与运行时侧设置；编译侧那些见[上一页](00-compile.md)。派发时要紧的是：

| 字段 | 效果 |
| ---- | ---- |
| `platform` / `device_id` | 用哪块设备，以及 worker 会接受哪种产物 |
| `enable_chip_swimlane` / `enable_dep_gen` / `enable_pmu` / `enable_dump_args` / `enable_scope_stats` | DFX 采集（[性能](../performance/00-swimlane.md)） |
| `ring_task_window` / `ring_heap` / `ring_dep_pool` | 运行时环的尺寸（[内存](../performance/05-memory.md)） |
| `aicpu_thread_num` | AICPU 线程数覆盖 |

**`RunConfig` 里有些字段属于 harness，不属于派发。** `rtol` / `atol`、`golden_data_dir`、`save_kernels` 与 `codegen_only` 是由系统测试 harness 读取的 —— 那条路径会编译、生成 golden 并比对。走 `compiled(...)`、`worker.run(...)` 或注册句柄时，它们不起作用；**尤其是 `codegen_only=True` 在这条路径上并不会阻止派发**，别指望用它来避免一次 launch。（`save_kernels_dir` 是例外：`RunConfig.compile_kwargs()` 会把它作为 `ir.compile` 的 `output_dir` 转发。）

## 边界情况

| 现象 | 原因 | 修法 |
| ---- | ---- | ---- |
| **worker 在首次派发前就拒绝该程序** | 产物的 `platform` 与 worker 不一致 | 用你要派发的平台去编译 |
| **`missing inferred tensor metadata for parameter`** | 把 `DeviceTensor` 传给了 `@pl.jit` 入口 | 派发**已编译**的程序；特化器读不到它的 shape/dtype |
| **设备内存随 launch 增长** | `DeviceTensor` 没释放 | `free_tensor`，或用 `with` 圈住 worker |
| **派发不给 host/device 拆分** | `execution_time` 是整段墙上时间 | 用 `pypto.runtime.benchmark` 取 `device_wall_us` / `host_wall_us` |

## 参见

- [编译](00-compile.md) —— 产出本页所派发的产物。
- [快速上手](../00-getting_started.md) —— 同样的内容，最短形式。
- [Host](../performance/06-host.md) —— 当该缩的是 host 那一段时。
- [分布式执行](../distributed/03-execution.md) —— 多 rank 的 worker。
