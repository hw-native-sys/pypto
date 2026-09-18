# Kernel mode 集成基础

公开 JIT eager 入口借用 NPU 参数，通过可选 native torch_npu adapter 提交。
kernel 执行要求每个进程先调用一次 `pypto.torch.init(...)`。显式 `.compile()` 返回 program 对象。

## 公开 JIT eager 入口

本集成分支中，`op(x, scale, out)` 固定进入 kernel mode。进程须先调用一次
`pypto.torch.init(...)`（见[kernel 执行设置](#kernel-execution-setup)）。调用方随后传入真实 NPU Tensor 和完整
Out/InOut，不设置 decorator mode，也不显式编译 kernel。每次调用先校验参数、快照本次
类型化 Scalar 与当前 stream。graph capture 要求每个算子特化提前完成 warmup。首次有效
调用编译 kernel 产物，并在 `init` 创建的 Worker 上 prepare callable；后续匹配调用复用产物和注册。
不同算子共享同一个 Worker。运行时 Scalar 值与 stream 变化不重新编译，constexpr 变化可选择不同产物。

```python
import pypto.torch

# op is a @pl.jit entry; the caller selected the current NPU device.
pypto.torch.init()
x = torch.ones((16, 16), device="npu")
out = torch.empty_like(x)
op(x, 2.0, out)
```

当前支持 A2/A3 的 `tensormap_and_ringbuffer`，包含非默认 stream 和 taskQueue 开关。
可选的 `config=CompileOptions(...)` 提供编译选项，并计入特化 key。拒绝 `RunConfig`、
CPU/Meta/Fake Tensor 及 Worker 自有 handle，不会据此切换执行路径。native launch 要求 rank 1–5 和正 uint32 extent/stride。
A5、HBG 执行仍属后续工作；直接 JIT 和注册后的 torch.ops 均支持 warmup 后的 ACLGraph capture/replay，见下文。自动清理目前按下文
torch_npu 2.6.0.post2 的退出契约接入；其他框架版本在完成退出协议验证前拒绝 native kernel
初始化。此功能仍限于集成分支。

Host/模拟器和分布式执行使用显式 program 编译：

```python
program = op.compile(host_x, 2.0, host_out, config=program_config)
program(host_x, 3.0, host_out, config=program_config)
```

仅编译不认领进程执行模式，program 与 kernel 执行须使用独立进程。两种产物使用独立缓存身份。
正式 program 调用（包括恢复对象和 orchestration 子入口）必须传入全部 Out/InOut，返回 `None`，
不分配省略的输出。底层显式 Worker API 保留原有内存管理行为。kernel 调用返回 `None` 或 IR
return alias 指定的原始 Tensor 对象，不分配输出、不额外执行 warmup。

## kernel 执行设置 {#kernel-execution-setup}

执行信息属于进程状态，而不是调用参数：

```python
import pypto
import pypto.torch
import torch_npu
from pypto.runtime import CompileOptions

torch_npu.npu.set_device(3)
pypto.torch.init(aicpu_thread_num=4)  # device=None uses the current torch_npu device
pypto.configure_cache(pypto.CacheConfig(enabled=True))  # optional process cache policy
op(x, 2.0, out)  # no execution information at the call site
op(x, 2.0, out, config=CompileOptions(analyze_auto_scopes_for_deps=True))  # compile options only
```

`init(*, device=None, platform="a2a3", runtime="tensormap_and_ringbuffer",
aicpu_thread_num=0, enable_chip_swimlane=0, enable_dep_gen=False, output_dir=None)` 在每个进程中调用一次，须早于首次直接 JIT 或注册 torch.ops 的 kernel 调用，
且在 graph capture 之外。认领 kernel mode 之前，它依次检查目标、torch_npu 当前设备、框架版本以及
当前没有进行中的 capture；随后认领 kernel mode、安装框架退出集成，并在生命周期线程上初始化进程
Worker。它不编译也不 prepare 任何算子，入图的每个特化仍需 warmup。

| 情形 | 行为 |
| ---- | ---- |
| `init` 前发起 kernel 调用 | 参数绑定、依赖发现与编译之前报 `RuntimeError`；进程不认领 mode，之后仍可调用 `init` |
| 以相同配置重复调用 `init` | 直接返回 |
| 以不同配置调用 `init` | `ValueError`，不创建第二个 Worker |
| `device` 与 torch_npu 当前设备不同 | `ValueError`；`init` 从不切换设备 |
| 在 graph capture 中调用 `init` | 任何状态变化之前报 `RuntimeError` |
| native 初始化失败 | 保留失败状态；需使用新进程 |
| 注册、Fake/Meta 调用、编译器追踪、program 编译 | 不需要 `init` |

执行信息不随调用传入，graph capture、编译器追踪和 dispatcher 注册都不会固化设备或 runtime。
直接调用和 `register()` 拒绝 `RunConfig` 与 `CompileOptions.distributed_config`：编译选项使用 `CompileOptions`，持久缓存策略使用
`pypto.configure_cache`。`CompileOptions.platform` 若不是默认值，必须等于绑定的平台。外层
`PassContext` 的 runtime 必须与绑定值一致；没有外层上下文时，eager 编译使用 `init` 绑定的 runtime。

## Kernel 泳道图采集 {#kernel-swimlane-collection}

在初始化时配置诊断，使用 `begin_dfx()` / `end_dfx()` 包围需要测量的调用。
两个边界接口必须在 ACLGraph capture 外调用，且会同步当前 torch_npu stream
（包括 host taskQueue）。一个窗口中的调用、replay 和结束操作必须使用同一 stream，
并与本进程其他 kernel 活动串行执行。

设备端只在 begin/end 窗口内开启诊断记录；预热和其他窗口外调用正常执行，
不会产生诊断记录。

```python
pypto.torch.init(enable_chip_swimlane=4, enable_dep_gen=True, output_dir="dfx")
op(x, 2.0, out)  # warm up before measuring
pypto.torch.begin_dfx()
op(x, 2.0, out)  # alternatively: replay an already captured graph
pypto.torch.end_dfx()
```

`enable_chip_swimlane` 接受 0..4（`True` 表示 4）；`enable_dep_gen` 可独立开启。
任一诊断开启时必须提供 `output_dir`。这些选项属于进程固定配置，重复 `init`
必须使用相同配置。目录不存在时会创建。第一个窗口直接输出
`chip_swimlane_records.json` 和/或 `deps.json`，后续窗口输出到
`window_1/`、`window_2/` 等子目录。嵌套窗口、没有 begin 就 end、换 stream
结束均会报错；关闭进程前应结束每个已开启的窗口。

```bash
python -m simpler_setup.tools.swimlane_converter \
    dfx/chip_swimlane_records.json -o dfx/merged_swimlane.json
```

合并文件可用 Perfetto 打开。依赖采集会增加开销；精确测时可在独立进程中
单独采集同一算子的依赖图，并通过转换器 `--deps-json` 指定文件。
改变 init 配置需要新进程。一个窗口可以包含多次调用，但转换器要求窗口内
`(core_id, reg_task_id)` 组合唯一，可能因不同调用复用 ID 而拒绝转换。
需要转换泳道图时，每个窗口只测一次 PyPTO 调用。
当前支持 A2/A3 TMR，包括预热后的 graph replay；begin/end 本身不支持 capture。

## Callable 身份与热路径测量

每个已加载的 `KernelArtifact` 缓存完整 Simpler callable 与 PyPTO ABI 的摘要。
并发 eager 与 capture 查询共享这一身份，描述符构建失败不会缓存结果。
capture 使用 `loaded_identity()`，绝不加载或编译产物。进程内注册仍属于对应 Worker
代次（generation）；缓存的只是不可变二进制的身份。

`tests/st/runtime/kernel/test_hot_path.py` 在 taskQueue 开关两种配置下，为直接 JIT 与
注册后的 `torch.ops` 各入口、各变体默认记录 64 次 Host 返回耗时。JUnit 属性包含
微秒单位的 p50/p99、参数校验/cache key/描述符计数，以及单/双生产线程吞吐。
未缓存对照（uncached control）在同一构建中恢复逐次描述符哈希，以隔离该项成本。
warmup 与批次 drain 不计入延迟样本。生产线程吞吐使用 16 次调用的批次，
串行切换 stream 并 drain，包含该交接耗时；不代表无约束的多 stream 并发吞吐。
耗时作为测量证据，不设 CI 通过阈值。小样本默认仅检查计数与正确性，尾延迟不作为性能验收结果。
完整测量通过 pytest 参数 `--kernel-perf-samples=1024` 启用。保持每批 16 次调用，
保留 backlog 计数契约，同时减少 PR CI 中的重复工作。

当前固定版本的 Simpler 在前一 caller stream 的 serial tail 未完成时拒绝切换 stream，
返回 `PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE`（`-1002`）。应用须在切换 caller stream 前
确保前一条 stream 已完成；adapter 的 Host launch 锁本身不能保证设备完成。
这一限制也约束双生产线程实验，移除 Host 锁不能使无约束的多 stream 提交得到支持。

taskQueue 开启时，使用阻塞的 Host callback，在另一条 stream 上精确保留
0/16/256 个初始 pending ticket。每个测量批次增加 16 个 ticket，随后释放 gate 并 drain。
另有排入队列的测试专用 stream fence，保证 foreground launch 到达 Simpler 前
background stream 已完成；fence 在 gate 释放后执行，不计入 Host 延迟样本。
测试断言每批 `done()` 调用次数为 `16 * pending + 120`，显示当前全列表扫描成本。
taskQueue 关闭时 gate 会内联执行，因此不运行该阻塞队列实验；普通延迟与生产线程测量
覆盖开关两种配置。一个已销毁图的四个 ticket 当前会在下次 eager 提交中产生四次设备同步，
另有计数断言记录该行为。队列回收、图同步合并及去锁仍为独立改造，须满足生命周期、
异步失败及 Simpler 并发契约。

这些用例需要以 `PYPTO_BUILD_TORCH_NPU_TESTS=ON` 构建。该选项向可选 adapter 添加
私有 `_test_counters()` 和 `_test_reset_counters()` 接口；普通构建不包含这些接口与
计数增量。计数只覆盖 adapter 的 `Done()` 调用及设备同步尝试，不代表框架/SDK 的全部调用。
对照与缓存变体使用同一计数构建。kernel eager 设备 CI job 执行这些用例并上传 JUnit 证据。

## 调用元数据与所有权

[`CallSignature`](../../../../python/pypto/torch/interop.py) 一次性复制共享的
`ir.param_info.ParamInfo` 签名。`describe_call(args)` 复用 `bind_complete_args`，
所有 Out/InOut 参数必须在查询框架上下文前完整传入。返回别名（return alias）使用
完整参数列表中的 Tensor 索引，不能指向 Scalar 或不存在的参数。

每次调用创建新的不可变 `CallFrame`：

| 字段 | 含义 |
| ---- | ---- |
| `tensors` | 按签名顺序保存 Tensor，包含原始 `param_index`、方向、dtype、shape、stride、format、逻辑地址和 storage 边界。 |
| `scalars` | 按签名顺序保存本次类型化基本值及原始 `param_index`；可变 ctypes 输入被复制。 |
| `device_index` | 本次 NPU 当前设备，与全部 Tensor 核对。 |
| `stream` | 本次 torch_npu 当前 stream 对象，由 frame 保持引用。 |
| `return_tensors` | 经校验的返回别名选择的原始调用方对象。 |

Tensor/Scalar 的排列不定义 native ABI 布局。后续 native 参数编码需同时消费这些值
和经过校验的 kernel descriptor。地址、Scalar 实际值和 stream 属于每次调用状态，
不进入编译 key 或持久 metadata。

frame 持有每个 Tensor 及其 storage 对象。`alias_result()` 返回 `None`、单个已有
Tensor 或已有 Tensor 元组，不分配业务输出。仅有 Python 引用不能保护 frame 释放后
的设备异步使用；native 队列所有权和 allocator stream 记录由下文 launch 层提供。
frame 使用期间，调用方不能 resize 或使借用的 storage 失效。

## 校验规则

- 接受同一设备上的真实 NPU torch Tensor。拒绝 CPU、Meta/Fake Tensor 和 Worker
  自有对象，不自动选择 program mode。
- 对照 `ParamInfo` 校验 dtype 和 rank。静态 carrier 维度必须匹配；`-1` 维度使用
  本次大小。打包 FP4 直接使用 `ParamInfo` 已提供的 carrier shape，不再次展开。
- 要求 base NCHW (0) 或 ND (2) 格式的连续 strided view，保留非零 storage offset
  和逻辑地址。其他格式、转置 view、未解析的 conjugate/negative view 及无效 storage
  边界明确报错，不进行复制或格式转换。
- 空 view 不访问任何元素，因此非负 storage offset 可以超过 storage 容量；
  仅对非空 view 校验访问范围上界。
- 允许完全相同的 Tensor view 别名、同一 storage 上互不相交的 view，以及重叠的
  只读 view。涉及 Out/InOut 的部分重叠需要更丰富的别名契约，当前明确拒绝。
- 使用 program Scalar 类型映射复制 Python 或类型匹配的 ctypes Scalar。
  整数必须位于声明类型范围内，不将浮点数静默截断成整数。这不改变 JIT 的运行时
  Scalar 或常量分类。
- autograd 开启时拒绝需要梯度的 Tensor；`torch.no_grad()` 推理可以借用模型参数，
  但不提供 backward 实现。

参数校验后，适配器在**每次调用**查询当前 device/stream，要求当前设备与 Tensor
设备一致，并核对 stream 设备。不切换设备、不缓存首个 stream、不读取 `npu_stream`，
也不同步或入队。这些上下文查询可能初始化 torch_npu 自身的框架上下文，但不创建或
初始化 Simpler/PyPTO Worker。

## 内部 schema 与 Fake/Meta 辅助

`pypto.torch.registration.RegistrationSignature` 复制相同的 `ParamInfo` carrier shape
及返回参数索引。`schema(name)` 将 Out/InOut Tensor 标记为可写，并把每个 Tensor
返回值关联到对应输入的别名集合（alias set）。全部参数均须传入，不推导输出分配。
Scalar 输入映射为 dispatcher 的 `SymInt`、`float` 或 `bool`。`SymInt` 接受普通整数，
并在 dispatch 中保留符号整数，包括没有具体值提示的符号。Scalar 校验保留 dtype
范围检查，不将符号转换为 Python 整数。拒绝直接返回只读输入、Scalar 输出、
纯 Scalar 算子、非法名称与返回别名，以及 UINT64 Scalar：dispatcher 的有符号整数
类型无法表达完整 UINT64 范围。返回别名必须指向 Out/InOut Tensor，因为 dispatcher
schema 检查器不允许直接返回只读输入对象。

`fake(*args)` 接受 FakeTensor 或 Meta Tensor，校验 dtype、rank、静态 shape、
连续性、设备一致性及仅推理约束，然后返回声明的原始输入对象。动态维度及符号整数
Scalar 保持符号形式。直接返回输入保留 stride、storage offset 与别名身份，包括
空切片。该辅助不读取 storage 或地址，不查询 NPU format/device/stream，不分配
业务输出，也不调用 Worker。抽象 Tensor 无法证明真实物理布局与 storage 重叠情况，
因此这些校验仍由真实调用适配器执行。

`define(library, name)` 仅在调用方持有的 `torch.library.Library` 中定义 schema
并注册 fake kernel；调用方负责保留 library 对象及其注册生命周期。重复定义，包括
同名不同签名，均抛出 PyTorch 的重复定义错误，不替换已有定义。导入或重新加载模块
不注册算子。下文的公开 `register` 辅助负责持有 Library 并接入实际设备实现。
如果当前 PyTorch 缺少 `torch.library.register_fake`，则回退到
`torch.library.impl_abstract`（PyTorch 2.2–2.3 提供），并保留调用方 Library 的
生命周期管理。两种 API 均不可用时，在安装 schema 前明确报错。该可选辅助要求
至少存在其中一种 API；此回退不新增 PyTorch 2.0–2.1 支持，也不调整整个包的最低依赖版本。

测试使用临时 namespace 和 CPU 实现夹具。在 PyTorch 2.6 上，无 dispatcher 返回值
的纯修改 schema 通过全部
[`torch.library.opcheck`](https://docs.pytorch.org/docs/2.6/library.html#torch.library.opcheck)
检查以及 `torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)`；
测试 wrapper 在调用算子后返回调用方传入的输出 Tensor。通过注册后的 `torch.ops`
验证有具体值提示（backed）和无具体值提示（unbacked）的整数符号原样到达 fake kernel，
且不添加等值 guard；另以 shape 派生的 Scalar 验证不同输入尺寸复用同一编译图。
API 选择测试在 PyTorch 2.6 上模拟旧注册入口，验证 Fake/Meta dispatch、重复定义拒绝
及 Library 清理；不据此宣称旧版 PyTorch 已通过端到端编译器兼容性验证。
不透明的带返回别名 schema 仍分别验证 schema 正确性与 Fake/Meta 行为。
公开注册通过下述分解（decomposition）支持函数化（functionalization）；仅定义带别名的
不透明 schema 不足以支持编译执行。当前不提供 autograd。

## 将 JIT kernel 注册到 torch.ops

```python
from pypto.torch import init, register

# op has fully shaped @pl.jit annotations, including Out/InOut directions.
registered = register(op, "my_kernels::op")  # needs no init
init()
registered(x, 2.0, out)
torch.ops.my_kernels.op(x, 3.0, out)
```

`register(kernel, name, *, constexpr=None, config=None)` 从 `kernel.specialize()` 和
与目标无关的返回别名分析派生 schema。注册不构建 binary、不分配 Tensor、不查询 NPU
上下文、不要求 `pypto.torch.init`，也不初始化 Worker。真实 NPU 调用须先完成 `init`，随后进入与
`kernel(...)` 相同的隐式编译、注册、当前 stream 提交及共享 Worker 路径。返回 Tensor 就是返回原始 Out/InOut 对象，支持重复
别名。全部运行时参数都必须提供，即使 Python 函数有默认值；不推导输出分配。不注册 CPU 执行。

Tensor 注解必须给出 shape/dtype；声明的动态维度保持动态。`constexpr={"block": 16}`
为一个算子名固定编译期参数，省略时使用签名默认值；运行时 Scalar 仍保留在 dispatcher
schema 中。其他 constexpr 变体使用其他名称。可选 `config=CompileOptions(...)` 计入算子身份
（省略时等同于 `CompileOptions()`），并用于每次 JIT 调用。拒绝 `RunConfig`：设备与 runtime 来自 `pypto.torch.init`，从不来自注册。constexpr 和 config 都不是运行时 `torch.ops` 参数。

PyPTO 在进程生命周期内持有注册 Library。同名、同一个 JIT 对象、相同 constexpr 和编译选项的
重复请求返回已有 overload；并发请求共享一个定义。不同 JIT 对象/绑定、外部已有定义和
内部名称冲突均报错，不替换已有算子。普通重复 import 命中模块缓存，不重复注册；重新加载
应用模块产生新的 JIT 对象时应使用新名称。重新加载注册辅助本身保留已有 Library。
`_pypto_` 前缀的算子名供内部 dispatcher 使用；注册中途失败会撤销本次定义。

公开算子保留准确 mutation/alias schema，其 `CompositeImplicitAutograd` 实现调用内部
纯修改算子，再返回原始参数。内部算子的 PrivateUse1 实现调用既有 JIT 入口，并配有 Fake/Meta
校验。这样 PyTorch 可对修改执行函数化，而无需处理不透明的返回别名；wrapper 不创建业务输出，
框架编译变换可以管理自己的中间缓冲区。真实和抽象执行都校验注册时的 shape/dtype 契约；
Fake/Meta 不编译、不 prepare、不 launch。尽管 dispatch key 名称包含 Autograd，此入口仅供推理：
开启梯度且 Tensor requires_grad 时明确拒绝；`torch.no_grad()` / `torch.inference_mode()`
下可使用这类 Tensor，但不提供 backward。

已验证的编译路径是 PyTorch 2.6 `torch.compile(backend="aot_eager", fullgraph=True)`，
CPU 夹具覆盖输出修改与重复别名；真实 NPU 测试覆盖直接/注册调用复用产物和 handle、taskQueue
开关，以及编译图中 kernel 前后的框架算子。已 warmup 的 `aot_eager` 调用还经过 NPUGraph
capture/replay 验证，包含注册 kernel 周围的框架算子。必须在 capture 外用相同受 guard 约束的输入
和选项调用 compiled wrapper；仅 warmup 底层 JIT 不会编译框架 wrapper。不据此宣称所有 compiler
backend 或自动图捕获已支持。

## 可选依赖与范围

`import pypto.torch` 导出 `register`。导入该包、`interop` 或 `registration` 模块不请求 torch_npu、
Simpler 或 native launch 扩展；`torch` 仍是 PyPTO 的常规依赖。真正描述 NPU 调用时
才按需加载 `torch_npu`，缺失时给出针对性的错误信息。

直接调用与注册后的 launch 路径均需要可选 native adapter。
直接 JIT 与注册后的 torch.ops capture 均要求提前 warmup，并共用图生命周期集成。

## 进程 kernel Worker 与注册

内部 `runtime.kernel.context.get_process_kernel_state()` 持有由 `pypto.torch.init`
通过 `ensure_worker` 创建的进程唯一 kernel Worker。kernel 调用只读取已绑定配置（`require_config`、
`bound_worker`），从不初始化 Worker；未调用 `init` 时报未初始化错误。所有算子共享此管理器；算子、Scalar 值或 caller stream 变化不会新建
Worker。`KernelConfig` 固定 platform、runtime、device、AICPU 线程数和 DFX 配置，其他常驻资源
暂用 simpler 默认值。配置不兼容时报错，不额外创建 Worker。

集成 SDK 固定为 `46b92250f2f5fa6e7166085f91e3fc7b7e74ca46`。实际 Python 接口为
`simpler.task_interface.ChipWorker.kernel_init`、`kernel_prepare_callable`、
`kernel_begin_dfx`、`kernel_end_dfx` 和 `finalize`。PyPTO 内部 adapter
使用这些已有方法；init/prepare 不接收 caller stream，native context generation 和
callable ID 均由 simpler 分配。调用线程须已绑定框架当前设备。初始化使用已安装的
runtime 二进制并检查能力，不编译业务算子、不分配业务输出。该 pin 的 HBG kernel
初始化不受支持，HBG 二进制编译成功不代表可执行。

该 pin 新增两项 PyPTO 依赖的要求。onboard 的 `tensormap_and_ringbuffer` 构建必须产出
独立的 kernel-mode AICore ELF（`aicore_kernel_mode.o`）：此时
`RuntimeBuilder.get_binaries` 会报告 `kernel_aicore_required` 及对应的
`kernel_aicore_path`，缺少该产物时 `kernel_init` 直接报错，不回退到 program 模式的
二进制。注册也不再同步——成功只表示 callable 镜像已上传且驻留已记录，并不表示设备已
加载 orchestration；该加载属于此 callable 的首次 launch，因此设备侧拒绝会在调用方排空
那次 launch 时浮现，而不是从 prepare 抛出。simpler 现在还允许在 ACLGraph capture 内注册
（init 仍须在 capture 外完成），并提供 init 前的
`ChipWorker.probe_kernel_mode_supported(bins)`；PyPTO 两者都尚未使用，仍要求在 capture
外完成 warmup。

管理器状态包括 UNINITIALIZED、INITIALIZING、READY、FAILED、CLOSING、CLOSED。
并发初始化共享结果；初始化失败保留错误及部分 Worker 供清理，不自动重建。PID 检查在
获取可能继承的锁之前执行：fork 子进程不得使用已初始化的管理器或注册项。若 fork 前
完全未初始化 kernel，仅重建未使用的 Python 状态；native 初始化后应使用独立 spawn
进程。

`ensure_callable(artifact, config)` 加载产物后，使用 simpler 既有 descriptor helper
对完整序列化 ChipCallable、Tensor 签名、target/runtime 及 PyPTO ABI 描述符求摘要。
注册 identity 不使用 ELF 展示用短 hash、路径或 Python 对象地址。同 identity 的并发
请求共享一次 prepare 的结果或错误，不同 identity 在同一 Worker 分别注册。prepare
失败不发布注册项，可重试；不通过执行一次算子来 warmup。每个 `KernelRegistration`
保持 callable、artifact、manager 的强引用，并检查 PID、管理器 generation 和注册表
成员关系；native handle 和注册项不写入磁盘缓存。

PyPTO 的显式 chip/distributed Worker 和一次性 program runner 在 native 初始化前
声明 program mode，kernel 初始化声明 kernel mode。该声明属于整个进程，在失败或
close 后仍保留；切换模式须使用独立进程。simpler 提供 native 单 context 模式防护及
重复 kernel context 拒绝。直接使用第三方 simpler 对象会绕过 PyPTO 的进程检查，不能
据此在同一进程混用 program/kernel 执行。

## 自动框架退出

PyPTO 在 native Worker 初始化前接入 torch_npu 既有退出边界。torch_npu 2.6.0.post2 的
[`_npu_shutdown`](https://github.com/Ascend/pytorch/blob/eef1d5ae62b9118ae78bf2d7084e6fba1b13058f/torch_npu/__init__.py)
依次调用 `_C._npu_shutdown_synchronize()`、销毁 process group、调用 `_C._npu_shutdown(success)`。
PyPTO 包装这两个 native 属性，在首次同步前完成一次清理；teardown 包装也覆盖跳过同步步骤的
显式拆除。保留原函数及参数，不额外注册 `atexit(worker.close)`；普通导入和 program 编译不安装钩子。

框架 native [teardown](https://github.com/Ascend/pytorch/blob/eef1d5ae62b9118ae78bf2d7084e6fba1b13058f/torch_npu/csrc/InitNpuBindings.cpp)
清理 allocator 后调用 `NpuSysCtrl::Finalize`，后者销毁 event、stream 和 device。
kernel 清理须在这些操作之前完成。native `GetInitFlag()` 检查已拆除的框架且不初始化设备。
私有退出接口和顺序依赖版本，因此当前仅允许 torch_npu 2.6.0.post2 初始化 native kernel。

每个 native kernel Worker 持有一个持续运行的 daemon 生命周期线程。该线程借用调用方的 ACL
context，执行 Worker 构造/init、callable prepare 和 finalize，不创建或 reset 设备。
因此即使 `pypto.torch.init` 在已结束的短生命周期线程上执行，也满足 simpler 的 init-owner-thread 规则。
热路径仍走已有 native torch 队列。close 成功后停止并 join 此线程；失败时保留线程以便重试。
采用 daemon 是因为 Python 会在框架退出钩子之前 join 非 daemon 线程；正常清理由钩子显式 join，
发生在框架拆除之前。

内部 `close()` 可由其他线程请求：先禁止新增工作，等待初始化、在途 prepare/admission，排空
已接受的 eager ticket，再在 native owner 线程 finalize。并发 close 共享完成结果；重复框架
通知不重试或重复 finalize。初始化/prepare 内的重入 close 被拒绝。成功后清空注册、使 handle
失效且不能重新初始化；未初始化的管理器不创建 Worker、不执行 native close。算子 GC 不触发关闭。

drain/finalize 失败保留 Worker、注册项及必要引用。自动清理不能完成或框架已拆除时，发出诊断，
停止接纳，并使用故意不释放的 native Python 引用保活管理器，防止后续解释器/module 清理在
ACL context 已销毁后执行 Worker/event 析构。框架退出仍继续；此为失败回退，不代表安全释放
已经完成。已证明清理成功但需报告较早提交错误的情况单独报告。fork 子进程不关闭继承的 Worker；
`os._exit`、signal、解释器崩溃等异常终止不保证清理。

对于含有 PyPTO 调用的 `torch_npu.npu.NPUGraph`，退出先禁止新增 replay，排空框架队列与设备工作，
reset 存活图，再释放 graph ticket 并关闭 Worker。不新增用户必须配对的 close/shutdown 接口。

## warmup 后的 JIT 与 torch.ops 图捕获

须在 warmup 之前调用 `pypto.torch.init`；在 capture 中调用 `init`，或未调用 `init` 就捕获 kernel 调用，
均会被拒绝。capture 前必须对**每个算子及特化**执行 warmup。shape、dtype 或 constexpr 变化可能选择新特化，
需要重新 warmup；运行时 Scalar 值变化不需要。warmup 会实际执行算子，因此若捕获计算依赖
InOut/输出初值，需恢复被 warmup 改写的状态。仅编译或命中磁盘缓存不代表已经在当前进程 Worker 注册。
Device 准备成功后，进程 Worker 按算子、特化及标量 ABI 类型持有注册记录（owning registration）。
capture 独立于 JIT 编译缓存查询该记录，因此设置 `PYPTO_PROG_BUILD_DIR`、传入
`CompileOptions(output_dir=...)`，或在 warmup 后清空编译缓存，都不会丢失已准备的 callable。
capture 不会编译、加载二进制或再次 prepare。prepare 失败不会发布记录，Worker 关闭后其注册记录均失效。
capture 外的诊断 eager/program 调用仍按原有策略重新编译。
单独设置 `PYPTO_PROG_BUILD_DIR` 不再视为诊断请求。持久缓存默认关闭，重复 eager 调用
仍可复用编译结果与注册记录。通过 `PYPTO_CACHE=1` 或 `CacheConfig(enabled=True)`
开启跨进程复用；环境策略使用构建目录的 `.pypto-cache` 子目录，除非另有指定。
后续进程仍需在各自 Worker 中 prepare，目前即使 READY 命中也需支付安装目录遍历成本。

```python
# op_a and op_b are @pl.jit entries; x, y, out are caller-owned NPU tensors.
pypto.torch.init()
op_a(x, y)
op_b(y, out)
torch.npu.synchronize()
# Restore any InOut state changed by warmup here.
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    op_a(x, y)
    op_b(y, out)
graph.replay()
```

注册算子沿用同一契约。匹配的直接 JIT warmup 可供 torch.ops capture 使用，反之亦然：
kernel 对象、constexpr 绑定及编译/Worker 配置必须一致。在一张图中混用两个入口也复用
同一个 Worker 与既有 callable 注册。`register()` 只定义 dispatcher 元数据，不等于 warmup。

```python
from pypto.torch import register

registered_a = register(op_a, "my_graph::a")
registered_b = register(op_b, "my_graph::b")
pypto.torch.init()
registered_a(x, y)
registered_b(y, out)
torch.npu.synchronize()
# Restore any InOut state changed by warmup here.
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    torch.ops.my_graph.a(x, y)
    torch.ops.my_graph.b(y, out)
graph.replay()
```

dispatcher Scalar 使用 Python `int`/`float`/`bool`；直接 JIT 还接受类型化 ctypes 值。
两者最终形成相同的类型化 ABI 快照。带数值校验的完整双算子示例见
[`examples/runtime/torch_kernel_capture.py`](../../../../examples/runtime/torch_kernel_capture.py)，
通过 `--entry torch_ops`（默认）或 `--entry jit` 选择调用入口。

capture 只查找已有产物与已完成注册，不初始化 Worker、不加载或注册新二进制；执行信息在 capture 前已由
`init` 绑定，图中不会记录它。缺少 warmup 时明确报错
`Kernel capture requires warmup outside capture for this specialization`。当前 pin 的 Simpler 注册会同步
内部 AICPU 流，该流在已有算子加入 capture 后不能被同步。冷双算子捕获见
[Simpler #2255](https://github.com/hw-native-sys/simpler/issues/2255)，流分离修复暂缓；不自动回退到 eager。

replay 直接执行已捕获的设备工作，不经过 Python JIT、编译或注册。Tensor 地址与类型化 Scalar 值是
捕获时的快照：修改该地址处的 storage 内容会影响 replay，重绑 Python Tensor 或修改 ctypes Scalar
不会改变图参数。不同 stream 之间的数据依赖由调用方建立。

captured ticket 持有 native Tensor/Storage，直到 ACL 图的销毁回调触发且设备静默已得到确认。
不能使用 eager 完成 event 证明以后不会 replay；销毁回调只标记 Host 状态，不调用 Python 或 ACL 清理。
reset 和正常退出先排空工作再释放图资源；算子/图 GC 都不会关闭进程 Worker。

torch_npu 2.6.0.post2 上首次有效 capture 包装 `NPUGraph.capture_end`、`replay`、`reset`，
仅以弱引用跟踪包含 PyPTO 调用的图。退出先禁止这些图的新 replay，再同步、reset 存活图、释放 ticket、
关闭 Worker，最后才允许框架拆除。未结束/失败的 capture 或排空失败保留资源并报告已有退出警告。
直接使用底层 ACL 图 API、预先保存的未包装框架方法及其他框架版本不在此生命周期契约内。

## 验证

### 集成分支 CI

目标为 `feat/kernel-mode-integration-test` 的 PR 会运行 `Kernel Mode CI`。
必验阶段包括 pre-commit（不含 clang-tidy）、关闭 native adapter 的 kernel 定向 CPU 回归、
固定工具链解析，以及 native adapter 构建和定向设备测试。
CPU UT 覆盖 JIT 路由、ABI/编译器/产物契约、Worker 与 shutdown 状态、torch
互操作/注册/launch/capture、JUnit 证据校验，以及 PR #2785（09A）的可选依赖导入隔离。
显式用例清单位于 `.github/scripts/kernel-mode-cases.sh`；该 PR workflow 不运行仓库全量
UT 或设备矩阵，完整测试仍可手动运行。CPU 测试与 pre-commit 独立启动，设备 job
只等待工具链解析；最终结果仍要求每个 job 全部通过。

每个设备任务只需一张 NPU 卡，使用共享的 `[self-hosted, linux, npu]` runner 池、现有
`setup-ci-job` bundle 环境，以及通过 runner 的 `DEVICE_ID` 分配设备的 `task-submit`。
独立环境安装 Torch 2.6.0 和 torch_npu 2.6.0.post2，要求 C++11 ABI，
并从当前检出的源码构建 adapter 和仅供测试的队列 gate。安装脚本按 runner 架构选择包：
ARM64 使用 Torch CPU 索引及固定 SHA256 的 torch_npu CPython 3.10 wheel；x86_64 使用 Torch
`cpu-cxx11-abi` 索引及 Ascend C++11 ABI zip 中的 CPython 3.10 wheel。
两种 torch_npu 包均来自官方 `v7.1.0.2-pytorch2.6.0` 发布，PyPI 未发布该版本。
发布包每次下载最多允许十分钟，连接超时为 15 秒，连续一分钟低于 1 KiB/s 则中止，
使持续传输的慢下载能够完成，同时避免停滞的连接一直等待。
最多尝试四次，中断后从已下载的字节继续；若服务端拒绝断点续传，则下一次从头下载。
只有完整下载成功的文件才会交给安装步骤。
设备预检复用 runtime 的框架版本检查，接受 C++11 ABI 构建后缀，仍要求版本为
2.6.0.post2。即使预检在生成 JUnit 之前失败，完整任务日志也会上传；缺失或失败的
JUnit 报告仍使 job 失败。任务在构建前检查 Torch ABI，并从
`.github/requirements/kernel-mode.txt` 安装固定版本的 CANN 9 TBE 基础依赖及
torch_npu wheel 未声明的 PyYAML，并在构建 adapter 前检查 torch_npu 和 TBE 导入。
仅加载 CANN 环境变量不会向任务虚拟环境安装这些 Python 依赖。native 构建使用
`build/kernel-native`，与共享 setup 在 `build/` 中的 scikit-build wheel 缓存分开，
并在分配设备前检查源码目录的 core、adapter 及测试 gate 能否导入。Simpler 与 pto-isa
来自 submodule pin，ptoas 来自 `toolchain/versions.env`。CANN 使用 runner 的
`CANN_ROOT`，须满足 adapter 已说明的环境前提。设备操作均在任务分配范围内执行。

两个矩阵 job `device-tests (eager)` 和 `device-tests (capture)` 分别独立运行
eager、stream、生命周期及 program 回归，和 warmup 后的 capture/replay。
runner 和设备资源允许时可并行执行；每个 job 独立检出、配置环境、构建、分配设备、
检查报告并清理自身任务。`fail-fast: false` 保证一组失败后另一组仍可完成。
制品分别命名为 `kernel-device-eager-results` 和 `kernel-device-capture-results`，
各自只保存对应测试组的证据。PR 选择 10 个 eager 用例和 6 个 capture 用例：

- Eager：两种队列模式的直接 JIT、显式 program 回归、注册入口 `aot_eager`、
  两种队列模式的提交失败、延迟 Host callback、正常进程退出，以及两个小样本热路径计数用例。
- Capture：冷调用拒绝、关闭队列的直接 JIT replay、注册入口 storage 保活与退出、
  混合入口多算子图，以及 `aot_eager` capture。选择关键契约，不展开完整笛卡尔积矩阵。

所有已选择的设备用例均须通过，不能跳过。显式 pytest node ID 保证用例缺失或改名时
收集失败。Pytest 在分配的卡上串行运行，用例按需创建隔离进程。

制品保留 JUnit、实际芯片名称和 device id、源码及 SDK revision、
Python/Torch/torch_npu/nanobind 版本、`npu-smi` 输出，以及安装环境提供的 CANN
版本文件。设备报告缺失、为空、格式错误、失败或存在跳过都会使报告检查失败。
最终 `Kernel Mode required results` 即使上游失败仍会运行，并要求包括两个设备矩阵
job 在内的全部阶段成功；
跳过或取消的 job 不能满足该检查。若需由 GitHub 强制限制合入，应将其配置到
集成分支的合入规则；仅定义 workflow 不会修改仓库规则。

这里交付当前 A2/A3 平台族 TRB 的 CI 接线，不代表完整平台验收。制品记录实际测试的
芯片，一种芯片通过不能同时充当 A2 和 A3 两份结果。A5、HBG 和完整平台矩阵仍待
09C 完成；capture 继续要求事先 warmup。

共享 UT fixture `run_without_optional_runtime` 启动独立 Python 进程，通过导入查找器
（import finder）阻止 `torch_npu`、`simpler`、`simpler_setup`、`_task_interface` 和
`pypto._torch_npu`。
同时捕获 import 语句与 `importlib.import_module()` 动态导入，即使调用方捕获了
`ImportError`，该次尝试仍使检查失败。显式检查在 `PYTHONOPTIMIZE=1` 和 `2` 下仍然生效，
负对照覆盖这两种优化级别及普通执行。子进程关闭 PyTorch backend 自动加载，以隔离
PyPTO 自身行为与已安装的框架插件。查找器的负向测试确保 CPU runner 即使原本就未安装
这些依赖，也不会把导入尝试误报为通过。

这些检查覆盖包导入/重载、program 配置和注册后的 Fake/Meta dispatch，直接进入现有
kernel 定向 UT CI，无需可选 runtime 依赖或新增设备 job。它们不验证 native adapter 构建或
设备执行；这些验证仍属于集成分支。

`tests/ut/torch/test_interop.py` 使用真实 CPU storage 和模拟的 NPU device 标签，
只替换框架 format/context 查询。覆盖 view offset、alias、独立 Scalar/stream 快照、
所有权、非法输入和禁止导入可选 runtime 依赖的隔离进程。

`tests/st/runtime/kernel/test_torch_interop.py` 核对真实 NPU Tensor、非默认 stream、
offset view 和非连续输入拒绝行为。没有真实 NPU 或所选平台为模拟器时跳过。
这些是 metadata 测试，不是 PyPTO kernel 执行测试。

`tests/ut/runtime/test_kernel_context.py` 覆盖共享初始化/注册、配置冲突、并发 prepare
失败传播、过期/fork handle、owner 保活、初始化线程关闭与重试。
`tests/st/runtime/kernel/test_kernel_context.py` 在隔离进程中使用真实 A2/A3 TRB
对两个 DSL callable 执行 init/prepare/close，并验证 native 重复 kernel context 拒绝
和 HBG 能力拒绝。测试要求固定版本 runtime 二进制及已预留的 NPU，不执行 PyPTO
kernel，也不验证 capture。

## 内部 torch 队列提交（04B）

`pypto.torch.launch.enqueue(registration, args)` 接受进程管理器准备好的注册项及逻辑签名顺序的完整参数。
每次校验注册项并生成独立 frame，通过可选 native torch_npu 扩展提交，返回既有输出别名；返回只表示
Host 接纳，不表示设备执行完成。本入口不编译、不 prepare、不创建 Worker、不分配业务输出。
公开 JIT 和 torch.ops 设备实现均复用此路径；两个入口的 capture 均要求提前 warmup。

扩展使用固定 SDK 的 `ChipStorageTaskArgs` 头文件构造两个独立参数池。混合签名 `(x, scale, out)`
对应两个 Tensor 和一个 Scalar；组装及二进制恢复的 ChipCallable 签名包含 `IN, OUT, SCALAR`，
program 代码生成的 Tensor direction 元数据保持原语义。Scalar 按实际类型的对象字节零扩展为 u64，
保留浮点位模式及有符号整数宽度。Tensor 使用含 storage offset 的逻辑地址；native launch 当前要求
rank 1..5、正 u32 extent/stride 和基础格式。空 view 可描述 metadata，但暂不可提交执行。

`OpCommand::RunOpApiV2` 将 native callback 纳入当前 framework stream 的 Host 队列；关闭 taskQueue
时同一 callback 同步执行。callback 只调用正式 C++ `ChipWorker::kernel_launch`，不执行 Python、
JIT 或 prepare。Tensor、Storage、参数 POD、callable ID 和 stream 都在提交前捕获。不会缓存首个
调用的 stream，也不读取会排空队列的 Python `npu_stream` 属性；capture 查询使用不排空队列的 stream。

进程管理器在 enqueue **之前**持有 ticket，并串行处理 Host 接纳。native Tensor/Storage owner 覆盖
延迟 callback；提交前按唯一 Storage 执行 allocator `recordStream`，包括别名参数。Simpler 建立
caller-stream join 后记录逐次 completion event，覆盖设备使用。后续调用查询 event 回收已完成 ticket，
不排空 Host 队列；内部 `state.drain()` 或内部 `close()` 等待在途提交，最后一个 ticket 可保留到
该边界。close 拒绝新增工作，等待 prepare/admission，排空 ticket 后才 finalize Worker。

同步提交错误及异步 callback 错误通过 framework 和 ticket wait 传播。回收旧 ticket 或 drain 时
观察到错误后，Worker 与同步 enqueue 失败一样进入 `FAILED`。后续任意算子调用均报
`Kernel Worker is failed, expected ready`，异常 cause 保留首次观察到的提交错误。
在回收时首次发现异步错误的调用会报告较早的异步 launch 失败，不 prepare 或 enqueue 新 ticket。
close 期间发现失败时保持 `CLOSING`，由清理过程完成收尾。失败或部分 enqueue 的 ticket
继续持有 Worker、参数及 Storage：caller stream 等待失败不能证明内部 stream 已静止。内部 close 排空 Host callback，并仅在错误路径执行全设备同步，证明内部 stream
已静止后再 finalize、释放 owner，同时重新抛出原提交错误。quiescence 或 teardown 失败时继续保留
全部 owner 以便重试 close。不会隐式重新初始化；自动框架退出通过上述集成完成。

### 可选扩展构建

默认 `PYPTO_BUILD_TORCH_NPU=OFF`，普通构建不发现或链接 torch_npu。导入 `pypto.torch.launch` 无需
扩展；实际 enqueue 缺扩展时给出明确构建提示。

```bash
source .claude/skills/testing/load-env.sh
# Source the installed CANN set_env.sh to set ASCEND_HOME_PATH.
cmake -S . -B build -DPYPTO_BUILD_TORCH_NPU=ON
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
```

使用当前环境匹配的 torch/torch_npu headers、libraries，以及 `ASCEND_HOME_PATH` 下的 CANN；要求
C++11 libstdc++ ABI 和兼容的 nanobind 构建。扩展嵌入并校验精确 Simpler revision。
Simpler Python 模块隐藏 C++ symbols，因此本扩展直接编译固定 SDK 的 Worker 实现，并通过 nanobind
已注册的 `ChipWorker` 类型接入；不复制 ABI 定义、不提取私有 context 地址、不创建额外 Worker。
切换 SDK、framework 或编译器 ABI 后须同时重建两个模块。

确定性 Host 队列阻塞测试使用独立、不安装的辅助模块。仅测试时开启
`-DPYPTO_BUILD_TORCH_NPU_TESTS=ON`，并把 `build/torch_npu_tests` 加入 `PYTHONPATH`。
生产扩展不包含阻塞队列或故障注入接口。

`tests/ut/torch/test_launch.py` 覆盖分发、Scalar 编码、别名、缺失/不兼容扩展；管理器 UT 覆盖保活、
回收、失败接纳及关闭顺序。`tests/st/runtime/kernel/test_torch_launch.py` 用真实 DSL callable 验证
A → PyPTO → B、非默认 stream、taskQueue 开关、offset view、逐次 Scalar、GC/分配压力、close、
阻塞 callback 快照及 native 错误注入。需预留 A2/A3 NPU 并从本 worktree 构建两个扩展；不声明 A5 或
ACLGraph 已验收。

`tests/ut/torch/test_registration.py` 覆盖 schema mutation/alias 契约、Fake/Meta
与符号输入、隔离导入、重复定义，以及不依赖真实 kernel executor 的测试内
dispatcher/compiler 集成，以及注册生命周期/冲突、回滚、推理限制和 JIT 入口路由。
`tests/st/runtime/kernel/test_torch_ops.py` 验证公开注册的真实 NPU eager/aot_eager、taskQueue
开关及共享 Worker/产物/callable；使用正常进程退出，不要求用户 close。

`tests/ut/torch/test_init.py` 覆盖显式初始化：绑定当前设备、幂等与配置冲突，以及不支持的目标、设备不一致、未验证的框架版本和进行中的 capture 在认领 kernel mode 之前被拒绝。

`tests/ut/jit/test_kernel_eager.py` 覆盖公开入口、参数绑定前的未初始化错误、`RunConfig` 拒绝、`CompileOptions` 特化、绑定目标/设备/runtime 校验、提前拒绝、Scalar 快照和缓存隔离。`tests/st/runtime/kernel/test_jit_eager.py` 验证未初始化错误、Worker 在 `init` 中初始化、真实 InOut 多次更新、constexpr 变体、共享 Worker 和隔离进程中的显式 program 执行。

`tests/ut/runtime/test_kernel_shutdown.py` 覆盖线程归属、初始化竞态、幂等 close、框架顺序、失败保活、
版本拒绝和 fork。`tests/st/runtime/kernel/test_kernel_shutdown.py` 使用普通 Python 子进程正常退出，
避免 multiprocessing 的 `os._exit` 绕过退出钩子，业务路径不手动 close/drain。覆盖 taskQueue 开关、
已结束的首调线程、两个算子、延迟 callback、部分初始化、重复通知及 finalize 失败后跨框架 teardown 保活。

`tests/st/runtime/kernel/test_capture.py` 对直接 JIT 和 torch.ops 执行同一矩阵，覆盖缺失 warmup
拒绝、单/多算子、多图/跨流、持久缓存复用、Scalar 快照、storage 保活、图 reset/GC/重建及
在途 replay 的正常进程退出，分别验证 taskQueue 开关。跨入口用例验证一种入口 warmup、另一种
入口 capture，以及同图混用；另有用例拒绝未调用 `init` 就捕获 kernel 调用，以及在 capture 中调用 `init`。
replay 不允许重新进入 Python JIT、编译或 prepare。
`tests/st/runtime/kernel/test_torch_ops.py` 还验证已 warmup 的 `aot_eager` 调用进入 capture，
包含周围框架算子及输出 alias。
`tests/ut/torch/test_capture.py` 覆盖图持有和退出接纳；管理器/JIT 测试验证 capture 不初始化、编译或注册。
