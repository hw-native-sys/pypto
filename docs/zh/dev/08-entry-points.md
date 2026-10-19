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
| 磁盘上的一个构建目录 | 派发一次，不重新编译 | `CompiledProgram.from_dir(work_dir)(*args, config=...)` |
| 一个 `CompiledProgram` | 计时派发 | `benchmark(compiled, args, ...)` |

## 内部 torch 适配

[Kernel mode 集成基础](runtime/kernel-mode.md) 校验 NPU metadata 并构造每次调用
的 frame，不改变上面的公开入口，也不提供 kernel launch 接口。

## 编译

| 入口 | 层 | 位置 |
| ---- | -- | ---- |
| `ir.compile` | 编译驱动 | [`ir/compile.py`](../../../python/pypto/ir/compile.py) |
| `JITFunction.compile` | 特化 + 驱动 | [`jit/decorator.py`](../../../python/pypto/jit/decorator.py) |
| `JITFunction.lower` | 只做特化，止于 `ir.Program` | 同上 |
| `device_runner._compile_and_assemble` | 装配 | [`runtime/device_runner.py`](../../../python/pypto/runtime/device_runner.py) |
| `device_runner._compile_single_kernel` / `_compile_single_orchestration` | 装配 | 同上 |
| `KernelCompiler.compile_incore` | 调用 ptoas | [`runtime/kernel_compiler.py`](../../../python/pypto/runtime/kernel_compiler.py) |

只有 `ir.compile` 是受支持的入口；表中其余项列在这里，是为了让 traceback 里出现的
名字能被定位到某一层。它还遮蔽了 Python 内置的 `compile`，因此更推荐
`from pypto import ir` 后调用 `ir.compile`，而不是直接导入这个名字。它接受的每个
选项都是关键字参数；`program` 是唯一的位置参数。

它的参数在[编译](../user/execution/00-compile.md)中有说明。

它可能返回的两种产物类型，以及选定分布式那一种的配置，都从 `pypto.ir` 导出：

```python
from pypto.ir import CompiledProgram, DistributedCompiledProgram, DistributedConfig
```

定义它们的模块 `pypto.ir.distributed_compiled_program` 仍然可以导入 —— `pypto.runtime`
直接从那里取，以避开导入环 —— 但用户代码与测试应当从 `pypto.ir` 取这三个名字。

## 执行

| 入口 | 层 | 接受什么 | 位置 |
| ---- | -- | -------- | ---- |
| `CompiledProgram.__call__` | 产物 | 产物句柄 | [`ir/compiled_program.py`](../../../python/pypto/ir/compiled_program.py) |
| `Worker.run` / `ChipWorker.run` / `DistributedWorker.run` | 执行 | 产物 + worker | [`runtime/worker.py`](../../../python/pypto/runtime/worker.py) |
| `CompiledProgram.from_dir` / `DistributedCompiledProgram.from_dir` | 产物 | 产物目录 | [`ir/compiled_program.py`](../../../python/pypto/ir/compiled_program.py) |
| `runtime.execute_compiled`（已弃用） | 执行 | 产物目录 | [`runtime/runner.py`](../../../python/pypto/runtime/runner.py) |
| `execute_distributed_compiled`（已弃用） | 执行 | 产物目录 | [`runtime/distributed_runner.py`](../../../python/pypto/runtime/distributed_runner.py) |
| `device_runner._execute_on_device` | 装配 | 已装配的二进制 | [`runtime/device_runner.py`](../../../python/pypto/runtime/device_runner.py) |
| `execute_artifact_dir` / `execute_batch_manifest` | CLI | 产物目录 | [`runtime/execute_artifact.py`](../../../python/pypto/runtime/execute_artifact.py) |

`run` 本身不说明你在哪一层，接收者才说明。`ChipWorker.run` 与
`DistributedWorker.run` 派发产物，二者都实现 `Worker.run`。`PassPipeline.run`
与此无关——它变换一个 `Program`——`PassManager.run_passes` 同理。

派发一个目录而不是一个活的句柄，正是 [replay](03-runtime-replay.md) 成立的前提：
PyPTO 编译被完全跳过。`from_dir` 就是抵达那里的方式 —— 它从持久化的 sidecar 重建
产物句柄，而调用这个句柄走的路径，与一个从未离开内存的句柄完全相同。

`execute_compiled` 与 `execute_distributed_compiled` 曾是它在 L2 与 L3 的写法，
现已**弃用**。两者仍转发到同一份实现，并各自发出 `DeprecationWarning`。

L3 那个是纯粹改名 —— 它本来就是 `from_dir` 加一次调用：

```python
# before
execute_distributed_compiled(work_dir, args, config=cfg, platform="a2a3")

# after
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode

ir.DistributedCompiledProgram.from_dir(
    work_dir, backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD,
)(*args, config=cfg)
```

**L2 那个不是**，因为两条路径的优先级规则不同：

| 设置项 | `execute_compiled` | `CompiledProgram.__call__` |
| ------ | ------------------ | -------------------------- |
| `platform` | 显式传入的参数 | 传了 config 就取 `config.platform`，否则取产物自带的 |
| `device_id` / `dfx` / `aicpu_thread_num` | 显式传入的参数 | 一律取自 `config` |
| ring 覆写项 | 取自 `config` | 取自 `config` |

也就是说，仅为 ring 尺寸而传的 `config` 会顺带接管其余各项；而
`RunConfig.platform` 默认为 `a2a3sim`，直接丢掉那些显式参数会把这次运行悄悄挪到
仿真器上。正确做法是把它们并入 config：

```python
# before —— 显式参数优先，cfg 只被用来读 ring 尺寸
execute_compiled(work_dir, args, platform="a2a3", device_id=0, config=cfg)

# after —— 由 cfg 承载全部执行期设置
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode

cfg = dataclasses.replace(
    cfg, backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD, device_id=0,
)
ir.CompiledProgram.from_dir(work_dir)(*args, config=cfg)
```

`dfx` 与 `aicpu_thread_num` 不需要翻译：各个 DFX 开关与 `aicpu_thread_num`
本来就是 `RunConfig` 的字段。传给
`from_dir(backend_type=..., execution_mode=...)` 的目标用于**不带 config** 的调用。

## 迁移旧 platform 示例

部分指南和可运行示例保留 `platform=`，用于展示对现有调用方及基于 platform 的
测试配置的兼容性。这些示例明确标记为**旧接口示例**：该 Python 参数会发出
`DeprecationWarning`，将此警告视为错误时会失败。新 Python 代码应从
`pypto.backend` 导入 `BackendType`，从 `pypto.runtime` 导入 `ExecutionMode`，
然后用下表中的两个字段替换 `platform=`：

| 旧 `platform` | `backend_type` | `execution_mode` |
| ------------- | -------------- | ---------------- |
| `"a2a3"` | `BackendType.Ascend910` | `ExecutionMode.ONBOARD` |
| `"a2a3sim"` | `BackendType.Ascend910` | `ExecutionMode.SIM` |
| `"a5"` | `BackendType.Ascend950` | `ExecutionMode.ONBOARD` |
| `"a5sim"` | `BackendType.Ascend950` | `ExecutionMode.SIM` |

此映射适用于 `RunConfig`、`CompileOptions`、`ir.compile`、两种产物的 `from_dir`
方法及 `dataclasses.replace(config, ...)`。CLI 的 `--platform` 参数、warmup JSON、
产物元数据和派生的 `.platform` 属性保留现有字符串格式，不触发 Python API 弃用警告。

## 用同一个 config 驱动两个阶段

`RunConfig` 同时承载编译期与派发期的设置。`compile_kwargs()` 把编译期那一半提取为
`ir.compile` 的关键字参数，因此一个 config 对象可以驱动两个阶段：

```python
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

config = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.ONBOARD)
compiled = ir.compile(program, **config.compile_kwargs())
compiled(*tensors, config=config)
```

它的字段分三类，而每一类都是一个类型：

| 由谁读取 | 类型 | 字段 |
| -------- | ---- | ---- |
| `ir.compile` | `CompileOptions` | `backend_type`、`execution_mode`（兼容旧 `platform`）、`strategy`、`dump_passes`、`dump_ptoas_passes`、`profiling`（`compile_profiling`）、`diagnostic_phase`、`disabled_diagnostics`、`analyze_auto_scopes_for_deps`、`output_dir`（`save_kernels_dir`）、`memory_planner`、`distributed_config` |
| 派发 | `RunOptions` | `platform`、`device_id`、`aicpu_thread_num`、`ring_*` 覆写项（[Ring 尺寸](05-runtime-ring-sizing.md)），以及内嵌的 `DfxOptions` |
| 一次派发采集哪些诊断 | `DfxOptions` | `enable_chip_swimlane`、`enable_dump_args`、`enable_pmu`、`enable_dep_gen`、`enable_scope_stats`（[DFX](03-runtime-dfx.md)） |
| 仅系统测试 harness | —— | `rtol`、`atol`、`golden_data_dir`、`save_kernels`、`codegen_only` |
| 无人读取 —— 派生 | —— | `backend_type`，是 RunConfig 的 `arch` 存储字段的构造及读取别名 |

**使用 `backend_type` 和 `execution_mode` 选择目标。** 架构决定编译 lowering
与 codegen；执行方式决定仿真或真机二进制的编译及派发。`RunConfig`、
`CompileOptions` 和 `ir.compile` 均支持这两个参数：

```python
from pypto.backend import BackendType
from pypto.runtime import CompileOptions, ExecutionMode, RunConfig

config = RunConfig(
    backend_type=BackendType.Ascend950,
    execution_mode=ExecutionMode.ONBOARD,
)
compiled = ir.compile(program, **config.compile_kwargs())
# Equivalent direct compilation:
compiled = ir.compile(
    program, backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD,
)
```

默认目标仍为 Ascend910 + SIM。编译产物提供两个目标属性；
`from_dir(..., execution_mode=...)` 可以覆盖持久化的执行方式，
`from_dir(..., backend_type=...)` 则保留持久化的执行方式，同时更新后端和派生 platform。
两个维度均可独立覆盖，且不会改写元数据。已有 Worker 绑定和缓存限制仍然适用；
重新加载不会重新运行代码生成。

不使用旧 `platform` 选择目标时，`CompileOptions` 要求执行方式为 `ExecutionMode`
枚举值。省略 `execution_mode` 使用默认 SIM；显式传入 `None` 或字符串会抛出
`TypeError`。

`RunConfig`、`CompileOptions`（含位置参数形式）、`ir.compile`、
`CompiledProgram.from_dir` 和 `DistributedCompiledProgram.from_dir` 的
`platform=` 输入已弃用，调用时发出 `DeprecationWarning`；请改用
`backend_type` 和 `execution_mode`。目前未指定移除版本。读取派生的 `.platform`
属性仍不产生警告，RunConfig 的 `arch` 字段保持兼容。旧 platform
仍决定两个轴，并优先于 backend_type；RunConfig 对两者冲突保留原有警告。
`ir.compile` 拒绝同时显式指定但不一致的 execution_mode 和 platform。
RunConfig、CompileOptions 的旧 platform 优先于两个轴，以兼容
`dataclasses.replace(config, platform=...)`。不传 platform 时，backend_type
现在会真正选择架构，RunConfig 不再丢弃它。

RunConfig 暂时用 arch 存储架构，backend_type 是构造及读取别名；
CompileOptions 保存 backend_type、execution_mode。配置向编译器的映射传递
`backend_type` 和 `execution_mode`，因此新接口编译不会产生旧接口警告。产物文件、
CLI 兼容路径、缓存 key 与 Simpler SDK 调用继续使用内部规范平台编码，
因此新旧写法对应同一缓存身份。本次不改变运行配置默认值或进程级后端选择限制。

`RunConfig.compile_options()` / `run_options()` / `dfx_options()` 是这个聚合体上的视图，
而 `compile_kwargs()` 就是 `compile_options().as_compile_kwargs()`。`CompileOptions`
按 `ir.compile` 的叫法命名字段 —— 是 `output_dir` 而不是 `save_kernels_dir` ——
因为它存在的意义就是用编译器自己的词汇说出编译侧；它也可以独立使用：

```python
from pypto.runtime import CompileOptions, ExecutionMode

compiled = ir.compile(program, **CompileOptions(execution_mode=ExecutionMode.ONBOARD).as_compile_kwargs())
```

**三个里只导出两个**，因为只有两个是调用方真能交出去的东西。`CompileOptions` 如上，
解包进 `ir.compile`；`DfxOptions` 是 `execute_compiled`、`execute_artifact_dir`、
`execute_batch_manifest` 的 `dfx=` 参数。`RunOptions` 两者都不是：每个派发入口 ——
`CompiledProgram.__call__`、`ChipWorker.run` 以及分布式那一对 —— 收的都是 `RunConfig`，
再自己经 `run_options()` 取派发侧那一半。直接把它交进去会抛 `AttributeError`，
所以在那些签名放宽之前，它保持内部（`pypto.runtime.runner.RunOptions`）。放宽本身是另一件
迁移：那意味着主派发 API 的 `config=` 参数要变成联合类型，而只改一部分入口比不改更糟。

`RunConfig` 保留全部字段与全部调用方；这三个类型是它底下的词汇。有一个单测把这个划分钉成
**完备**的 —— 每个 `RunConfig` 字段要么被某个视图认领，要么属于那五个 harness 专用字段 ——
于是以后新加的字段不会悄悄两边都不属于。把那五个挪去 harness 是本次**没有**做的一步：
它们也出现在 `pypto-lib` 的构造调用里。

`compile_kwargs()` 是通往 `ir.compile` 参数的唯一映射。`@pl.jit` 路径过去另有一份
副本，省略 `platform` 与 `backend_type` 并单独转发 platform；现在它和其他调用方一样
调用 `compile_kwargs()`，于是给一边加的开关不会在另一边悄悄缺席。`lower()` 仍有自己
那份更窄的映射 —— 它止步于 codegen 之前，对准的是 pass 流水线而不是 `ir.compile`。

## 层与层之间的箭头朝哪

`ir` 产出产物，`runtime` 派发它。两者之间的箭头本该单向，而且大体上是单向的 ——
但 `CompiledProgram` 既是编译**产物**又是执行**句柄**，于是
`pypto.ir.compiled_program` 通过十处函数内 import 反向伸向 `pypto.runtime`。

这些延迟导入通常被说成是为了打破导入环。逐个实测下来，真正如此的只有一处：

| 延迟导入 | 能提到模块级吗？ | 真实原因 |
| -------- | ---------------- | -------- |
| `runtime.runner`（6 处） | 能 | 分层选择 —— 不让派发层进入 `pypto.ir` 的导入图 |
| `runtime.distributed_runner`（2 处） | 能 | 同上 |
| `runtime.debug.run_script_writer` | **现在**能了 | 曾是真环：它把 `ParamInfo` 从 `compiled_program` 读回去 |
| `runtime.device_runner` | 不能 | 它在导入期就需要可选的 `simpler` 包 |

那唯一的真环已经消除。参数元数据移到了
[`ir/param_info.py`](../../../python/pypto/ir/param_info.py) —— 一个不从
`pypto.runtime` 导入任何东西的叶子模块 —— 重放脚本生成器改从那里读；
`compiled_program` 重新导出这些名字，所以别处什么都不用动。有一个单测把这个叶子性质
钉住，因为往里拉进任何一个 runtime 导入都会让环回来。

`pypto.runtime` 直接导入 `pypto.ir.distributed_compiled_program` —— 箭头唯一反向的
那一处 —— **确实**是为了避开导入环，保持原样。

彻底反转这个依赖，是比这些 import 语句大得多的问题。它意味着 `CompiledProgram` 不再可调用、
`ir.compile` 改为返回一个由 `runtime` 包装的描述符 —— 那会改掉每个 example 和两个下游仓库
都在用的 API 返回类型。这份 import 清单是那个双重身份的症状而非成因；而把那八处能提的都提到
模块级，只会让耦合**更强** —— 那等于把 `pypto.runtime.runner` 放上 `import pypto.ir` 的关键路径。

## 哪些是内部实现

以下没有稳定性保证。它们不在任何 `__all__` 中，PyPTO 之外的代码不应导入：

- `pypto.ir.compile` —— `_ensure_orchestration_headers`
- `pypto.ir.compiled_program` —— `CompiledProgram._build_orch_args`、
  `CompiledProgram._build_call_config`
- `pypto.runtime.runner` —— `_execute_compiled`、`_execute_golden_case`、
  `_build_call_config`、`_coerced_to_orch_args`、`RunOptions`
- `pypto.runtime.distributed_runner` —— `_execute_distributed`
- `pypto.runtime.device_runner` —— 整个装配层：`_compile_and_assemble`、
  `_compile_single_kernel`、`_compile_single_orchestration`、`_execute_on_device`
- `pypto.runtime.kernel_compiler` —— `KernelCompiler`、`compile_incore`
- `pypto.runtime.tensor_arg`、`pypto.runtime.elf_parser`、`pypto.runtime._binary_cache`

装配层带下划线，是为了让 traceback 一眼看出它来自边界的哪一侧：一个必须拼出 `_`
的导入，是一个做过决定的导入。`KernelCompiler` / `compile_incore` 以及最后一行的三个
模块是例外 —— 它们靠模块而非名字划为内部，稳定性保证同样为零。

这两个实参构造器把用户实参编排成 simpler 的 `TaskArgs` 与 `CallConfig`。
`ChipWorker.run` 与 `ChipWorker.register` 是抵达这条路径的受支持方式 —— 它们会替你调用
构造器；而自行构造 `simpler.worker.Worker` 的调用方，仍可从公开面拿到 `chip_callable`、
`runtime_name` 与 `runtime_config`。

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
