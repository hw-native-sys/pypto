# 编译与执行入口

每个入口做什么，以及它属于哪一层。

PyPTO 的编译与执行入口比它拥有的概念要多，而且好几个名字横跨抽象层：有六个不同的
函数叫 `compile`，而 `run` 既是运行时 worker 句柄上的方法，也是 `PassPipeline` 上的
方法。本页是这份地图——每个入口接受什么、产出什么、什么时候该用它。

## 四个层次

每个入口都恰好属于四层中的一层。名字没有表达出它属于哪一层时，混乱就产生了：

```text
  define              compile                 artifact                execute
  ────────────────────────────────────────────────────────────────────────────
  @pl.program  ──┐                        CompiledProgram      ChipWorker.run()
  @pl.function   ├─>  ir.compile()   ──>   Distributed-        DistributedWorker.run()
  @pl.jit      ──┘    kernel.compile()     CompiledProgram     compiled(*args)
                      kernel.lower()  ──>  ir.Program
```

产物层之下还有一个**装配层（assembly layer）**——把 `.pto` 与 `.cpp` 变成可加载的
二进制。它是内部实现，没有受支持的入口。

## 如何选择入口

| 你手上有 | 你想要 | 用 |
| -------- | ------ | -- |
| 一个 `@pl.jit` 内核和 torch 张量 | 结果 | `kernel(*args, config=...)` |
| 一个 `@pl.jit` 内核 | 产物，用于反复派发 | `kernel.compile(*args, config=...)` |
| 一个 `@pl.jit` 内核 | IR，不做 codegen | `kernel.lower(*args)` |
| 一个 `@pl.program` 类 | 产物 | `ir.compile(program, ...)` |
| 一个 `CompiledProgram` | 派发一次 | `compiled(*args, config=...)` |
| 一个 `CompiledProgram` 和常驻设备的数据 | 派发多次 | `ChipWorker.run(compiled, *args)` |
| 一个 `DistributedCompiledProgram` | 派发多次 | `compiled.prepare()` → `DistributedWorker.run(...)` |
| 磁盘上的一个构建目录 | 派发一次，不重新编译 | `execute_compiled(work_dir, args, ...)` |
| 一个 `CompiledProgram` | 计时派发 | `benchmark(compiled, args, ...)` |

## 编译

| 入口 | 层 | 位置 |
| ---- | -- | ---- |
| `ir.compile` | 编译驱动 | [`ir/compile.py`](../../../python/pypto/ir/compile.py) |
| `JITFunction.compile` | 特化 + 驱动 | [`jit/decorator.py`](../../../python/pypto/jit/decorator.py) |
| `JITFunction.lower` | 只做特化，止于 `ir.Program` | 同上 |
| `device_runner.compile_and_assemble` | 装配 | [`runtime/device_runner.py`](../../../python/pypto/runtime/device_runner.py) |
| `device_runner.compile_single_kernel` / `compile_single_orchestration` | 装配 | 同上 |
| `KernelCompiler.compile_incore` | 调用 ptoas | [`runtime/kernel_compiler.py`](../../../python/pypto/runtime/kernel_compiler.py) |

只有 `ir.compile` 是受支持的入口；表中其余项列在这里，是为了让 traceback 里出现的
名字能被定位到某一层。它还遮蔽了 Python 内置的 `compile`，因此更推荐
`from pypto import ir` 后调用 `ir.compile`，而不是直接导入这个名字。

它的参数在[编译](../user/execution/00-compile.md)中有说明。

## 执行

| 入口 | 层 | 接受什么 | 位置 |
| ---- | -- | -------- | ---- |
| `CompiledProgram.__call__` | 产物 | 产物句柄 | [`ir/compiled_program.py`](../../../python/pypto/ir/compiled_program.py) |
| `Worker.run` / `ChipWorker.run` / `DistributedWorker.run` | 执行 | 产物 + worker | [`runtime/worker.py`](../../../python/pypto/runtime/worker.py) |
| `runtime.execute_compiled` | 执行 | 产物目录 | [`runtime/runner.py`](../../../python/pypto/runtime/runner.py) |
| `execute_distributed_compiled` | 执行 | 产物目录 | [`runtime/distributed_runner.py`](../../../python/pypto/runtime/distributed_runner.py) |
| `device_runner.execute_on_device` | 装配 | 已装配的二进制 | [`runtime/device_runner.py`](../../../python/pypto/runtime/device_runner.py) |
| `execute_artifact_dir` / `execute_batch_manifest` | CLI | 产物目录 | [`runtime/execute_artifact.py`](../../../python/pypto/runtime/execute_artifact.py) |

`run` 本身不说明你在哪一层，接收者才说明。`ChipWorker.run` 与
`DistributedWorker.run` 派发产物，二者都实现 `Worker.run`。`PassPipeline.run`
与此无关——它变换一个 `Program`——`PassManager.run_passes` 同理。

`execute_compiled` 与 `execute_distributed_compiled` 分别是 L2 与 L3 的
"目录驱动"路径。它们完全跳过 PyPTO 编译，[replay](03-runtime-replay.md)
正是靠这一点成立的。

## 用同一个 config 驱动两个阶段

`RunConfig` 同时承载编译期与派发期的设置。`compile_kwargs()` 把编译期那一半提取为
`ir.compile` 的关键字参数，因此一个 config 对象可以驱动两个阶段：

```python
config = RunConfig(platform="a2a3")
compiled = ir.compile(program, **config.compile_kwargs())
compiled(*tensors, config=config)
```

它的字段分三类：

| 由谁读取 | 字段 |
| -------- | ---- |
| `ir.compile`，经 `compile_kwargs()` | `strategy`、`backend_type`、`platform`、`memory_planner`、`dump_passes`、`dump_ptoas_passes`、`compile_profiling`、`diagnostic_phase`、`disabled_diagnostics`、`analyze_auto_scopes_for_deps`、`save_kernels_dir`（作为 `output_dir`）、`distributed_config` |
| 派发 | `device_id`、`aicpu_thread_num`、`ring_*` 覆写项（[Ring 尺寸](05-runtime-ring-sizing.md)）、DFX 开关（[DFX](03-runtime-dfx.md)） |
| 仅系统测试 harness | `rtol`、`atol`、`golden_data_dir`、`save_kernels`、`codegen_only` |

`@pl.jit` 路径保留了自己的映射 `jit.decorator._run_config_compile_kwargs`，
它省略了 `platform` 与 `backend_type`，因为该路径会单独转发这两项。

## 哪些是内部实现

以下没有稳定性保证。它们不在任何 `__all__` 中，PyPTO 之外的代码不应导入：

- `pypto.ir.compile` —— `_ensure_orchestration_headers`
- `pypto.runtime.device_runner` —— `compile_and_assemble`、`compile_single_kernel`、
  `compile_single_orchestration`、`execute_on_device`
- `pypto.runtime.kernel_compiler` —— `KernelCompiler`、`compile_incore`
- `pypto.runtime.runner` —— `_build_call_config`、`_coerced_to_orch_args`、
  `_DfxOpts`
- `pypto.runtime.tensor_arg`、`pypto.runtime.elf_parser`、`pypto.runtime._binary_cache`

`DistributedCompiledProgram` 与 `DistributedConfig` 是公开的，但没有从 `pypto.ir`
重新导出；请从 `pypto.ir.distributed_compiled_program` 导入。

## 命令行入口

| 命令 | 用途 |
| ---- | ---- |
| `pypto-ir-trace` | 把 pass dump 渲染成交互式 lowering trace（[IR lowering trace](07-ir-lower-trace.md)） |
| `python -m pypto.runtime.execute_artifact` | 执行一个产物目录或一份 batch manifest |
| `python -m pypto.runtime.debug.replay` | 手改生成代码后重跑一个构建目录（[Replay](03-runtime-replay.md)） |
| `python build_output/<dir>/debug/run.py` | 每次构建都会生成的自包含复现脚本 |
| `python -m pypto.tools.memory_map` | 把片上内存渲染成 HTML（[Memory Map](07-memory-map.md)） |
| `python -m pypto.tools.clean_sim_trace` | 把仿真器 dump 转成可读 trace（[Trace 清洗](04-simulator-trace-cleaning.md)） |

## 参见

- [编译](../user/execution/00-compile.md) —— `ir.compile` 的参数与产物目录。
- [运行](../user/execution/01-run.md) —— `CompiledProgram`、`ChipWorker`、`DeviceTensor`。
- [运行时 DFX 开关](03-runtime-dfx.md) —— 五个诊断子特性。
- [每任务 Ring 尺寸](05-runtime-ring-sizing.md) —— 三个逐次派发的 ring 覆写项。
