# 运行

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
| `program` | 降级后的 IR，用于 `as_python()` 或 torch codegen |
| `chip_callable` / `runtime_name` / `runtime_config` | 运行时侧的句柄 |
| `build_orch_args` / `build_call_config` | 显式派发需要的两个构造器 |
| `validate_ir` | 逐 pass 的语义对比（[精度](../precision/00-workflow.md)） |
| `from_dir` / `load` | 从已保存的产物目录重建句柄 |

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

### 影响派发的 `RunConfig` 字段

`RunConfig` 同时携带编译侧与运行时侧设置；编译侧那些见[上一页](00-compile.md)。派发时要紧的是：

| 字段 | 效果 |
| ---- | ---- |
| `platform` / `device_id` | 用哪块设备，以及 worker 会接受哪种产物 |
| `rtol` / `atol` | golden 对比的容差 |
| `save_kernels` / `save_kernels_dir` | 保留产物目录而不是用临时目录 |
| `codegen_only` | 生成代码后停下，不执行 |
| `enable_chip_swimlane` / `enable_dep_gen` / `enable_pmu` / `enable_dump_args` / `enable_scope_stats` | DFX 采集（[性能](../performance/00-swimlane.md)） |
| `ring_task_window` / `ring_heap` / `ring_dep_pool` | 运行时环的尺寸（[内存](../performance/05-memory.md)） |
| `aicpu_thread_num` | AICPU 线程数覆盖 |
| `golden_data_dir` | 复用或固化 golden `.pt` 数据 |

## 边界情况

| 现象 | 原因 | 修法 |
| ---- | ---- | ---- |
| **worker 在首次派发前就拒绝该程序** | 产物的 `platform` 与 worker 不一致 | 用你要派发的平台去编译 |
| **`missing inferred tensor metadata for parameter`** | 把 `DeviceTensor` 传给了 `@pl.jit` 入口 | 派发**已编译**的程序；特化器读不到它的 shape/dtype |
| **设备内存随 launch 增长** | `DeviceTensor` 没释放 | `free_tensor`，或用 `with` 圈住 worker |
| **`run()` 不给 host/device 拆分** | `execution_time` 是整段墙上时间 | 用 `pypto.runtime.benchmark` 取 `device_wall_us` / `host_wall_us` |

## 参见

- [编译](00-compile.md) —— 产出本页所派发的产物。
- [快速上手](../00-getting_started.md) —— 同样的内容，最短形式。
- [Host](../performance/06-host.md) —— 当该缩的是 host 那一段时。
- [分布式执行](../distributed/03-execution.md) —— 多 rank 的 worker。
