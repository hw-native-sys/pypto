# 编译

把一个程序或一个 `@pl.jit` kernel 变成设备 kernel 与主机编排。

## 概念

编译会在你的 IR 上跑完整条 pass 流水，为每个 InCore 函数生成 PTO，交给 **ptoas** 产出设备二进制，并发出启动它们的主机编排。你拿回来的是一个 `CompiledProgram` —— 一个指向产物目录的句柄，外加运行时派发所需的元数据。

入口有两个，区别只在 IR 从哪来。`ir.compile(program)` 接受一个 `@pl.program` 类；`kernel.compile(*args)` 接受一个 `@pl.jit` 函数加上样例实参，按它们的形状特化之后做同样的事。

## 快速上手：编译并保留产物

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
A = torch.randn(128, 128, dtype=torch.float32)
B = torch.randn(128, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def add(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.add(a, b)
    return out


out = torch.zeros(128, 128, dtype=torch.float32)
compiled = add.compile(A, B, out, config=CFG)

print("artifacts in:", compiled.output_dir)
assert compiled.platform == CFG.platform
assert compiled.param_names == ["a", "b", "out"]
```

`compile()` 的位置参数是 **kernel 自己的参数**，不是编译选项。`add.compile(skip_ptoas=True)` 会按 kernel 签名去绑定并抛出 `TypeError: got an unexpected keyword argument`；编译选项走 `config=RunConfig(...)`。

## 机制

### `ir.compile` 的参数

十八个，其中四个承担了大部分决策，其余的默认值你很少会动。它们全部是关键字参数 ——
只有 `program` 是位置参数。

| 参数 | 默认 | 决定什么 |
| ---- | ---- | -------- |
| `program` | — | 要编译的 `ir.Program` |
| `output_dir` | `None` | 产物落点；`None` 表示 `PYPTO_PROG_BUILD_DIR` 或 `build_output` 下的 `<name>_<timestamp>` |
| `strategy` | `Default` | pass 流水。`Default` 是唯一策略 |
| `dump_passes` | `True` | 每个 pass 之后的 IR 快照 —— 见下 |
| `backend_type` | `Ascend910B` | pass 与 codegen 的目标（`Ascend910B` / `Ascend950`） |
| `platform` | `None` | 产物面向的执行平台；必须与派发它的 worker 一致 |
| `skip_ptoas` | `False` | 停在 `.pto`（MLIR），不构建设备二进制 |
| `verification_level` | `None` | `NONE` / `BASIC` / `ROUNDTRIP`；`None` 交给 `PYPTO_VERIFY_LEVEL`，否则 `BASIC` |
| `memory_planner` | `None` → `DSA_RP` | `PYPTO` / `DSA_RP` / `PTOAS` —— 谁规划片上缓冲（[内存](../performance/05-memory.md)）；当前 `PassContext` 可覆盖该回退值 |
| `diagnostic_phase` | `None` | 警告与性能提示在哪个阶段设门 |
| `disabled_diagnostics` | `None` | 关掉特定检查，而不是全部 |
| `profiling` | `False` | 逐阶段编译计时写入 `report/pipeline_profile.{txt,json}` |
| `distributed_config` | `None` | 按 rank 编译 HOST 级程序（[分布式](../distributed/index.md)） |
| `analyze_auto_scopes_for_deps` | `False` | 对 AUTO 作用域做编译器推导依赖 |
| `enable_pypto_l0c_double_buffer` | `None` | L0C double buffer |
| `emit_source_loc` | `None` | 把 DSL 源位置带进发出的 `.pto` |
| `dump_ptoas_passes` | `False` | 同时 dump ptoas 自己的 pass IR |
| `runtime` | `None` | 面向哪个 Simpler 运行时 ABI —— `TENSORMAP_AND_RINGBUFFER` 或 `HOST_BUILD_GRAPH`（`@pl.jit.graph` 需要它 —— [函数](../language/01-functions.md)）；`None` 继承当前 `PassContext`。派发它的 worker 必须与之匹配 |

### Pass dump

`dump_passes` 接受 bool 或 `PassDumpLevel`：

| 值 | 效果 |
| -- | ---- |
| `False` / `PassDumpLevel.NONE` | 不出快照 |
| `True` / `PassDumpLevel.CONCISE` | 每个 pass 一份快照 |
| `PassDumpLevel.EXPLICIT` | 同上，并解析出隐式 tile layout 与分布式 window buffer |

它们落在 `<output_dir>/passes_dump/NN_after_<PassName>.py`，按执行顺序编号。[内存图](../tools/02-memory-map.md)读的正是 `EXPLICIT`；当问题是「哪个 pass 改了这个」时，你要的也是它。

### 产物目录里有什么

| 路径 | 内容 |
| ---- | ---- |
| `passes_dump/` | 逐 pass 的 IR 快照（当 `dump_passes` 要求时） |
| `ptoas/` | 每个 InCore 函数的 `.pto`，旁边是 ptoas 产出的 `.cpp` |
| `kernels/` | 编译好的设备 kernel |
| `report/perf_hints.log` | 编译期性能提示（[性能](../performance/index.md)） |
| `report/pipeline_profile.*` | 逐阶段编译计时（当 `profiling=True`） |
| `dfx_outputs/` | 由 DFX 开关在**运行**时写出，不是编译产物 |

### `JITFunction.compile`

`@pl.jit` 平时把特化 + 编译 + 派发融成一次 `kernel(*args)` 调用。`compile(*sample_args)` 在编译后停下，返回 JIT 缓存持有的那个 `CompiledProgram` —— 所以之后用同一个特化键调用会拿到同一个对象。

它暴露了完整的提取面，这正是直接驱动运行时的 harness 所需要的：`chip_callable`、`runtime_name`、`runtime_config`、`output_dir`、`platform`、`output_indices`、`param_names`、`orchestration_names`、`has_return`。实参编排不属于这个面 —— 交给 [`ChipWorker`](01-run.md#显式派发) 派发，它会替你完成。

`lower(*args)` 比它早停一站：只跑 pass 并返回 `Program`，不写任何产物 —— 也就意味着没有 `passes_dump/`。它适合 [torch codegen](../tools/01-torch-codegen.md)，那里要的就是 IR 本身；而[内存图](../tools/02-memory-map.md)读的是磁盘上的 dump，因此需要 `compile()`。

## 边界情况

| 现象 | 原因 | 修法 |
| ---- | ---- | ---- |
| **`TypeError: got an unexpected keyword argument`** | 编译选项被当位置参数传给了 `compile()` | 放进 `config=RunConfig(...)` |
| **worker 拒绝该产物** | 编译期 `platform` 与 worker 的不一致 | 用你将要派发的平台去编译 |
| **没有 `passes_dump/`** | `lower()` 不写产物；或 `dump_passes=False` | 改用 `compile()`，或传 `dump_passes=PassDumpLevel.EXPLICIT` |
| **内存图没东西可画** | `memory_planner=PTOAS` 跳过 `AllocateMemoryAddr`，dump 里没有偏移 | 改用端到端对比 |

> **`verification_level` 是调试杠杆，不是该常年调高的默认值。** 默认是 `BASIC` —— 除非 `PYPTO_VERIFY_LEVEL` 另有指定，因为该参数留 `None` 时会交给它。`ROUNDTRIP` 会额外重新解析每份 dump，明显更慢。怀疑 IR 畸形时调高，用完调回去。

## 参见

- [运行](01-run.md) —— 派发本页产出的东西。
- [Pass Manager](../../dev/passes/00-pass_manager.md) —— `strategy` 选择的那条流水。
- [调试](../tools/00-debugging.md) —— 读本页 dump 出来的 IR。
- [精度](../precision/index.md) —— 编译成功但数值不对时。
