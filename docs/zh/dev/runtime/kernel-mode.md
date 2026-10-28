# Torch eager kernel 执行

直接调用 `@pl.jit` 函数时，必须先完成 `pypto.torch.init()`，随后在当前 Torch NPU
stream 上执行 eager kernel。显式 `op.compile()` 保留 program 语义。两条路径共用
specialization、编译、产物存储及下述参数契约。

## 公共签名与 L2 ABI

`ParamInfo` 是 program 执行、Torch 实参校验、注册元数据和 assembly 共用的不可变
参数快照。shape 固化为 tuple，JSON 保留数组。共享 helper 校验名称唯一性和外部返回
别名；索引对应包含 Scalar 的完整参数列表，允许重复。L2 参数池容量单独在单芯片边界
检查，不限制分布式元数据。

不再保留独立 kernel ABI descriptor 或 SDK 字段。编译身份复用 `ArtifactKey` 和
`ToolchainIdentity`，包含请求的 PROGRAM 或 KERNEL 契约。参数 dtype/layout 序列化
保持共享；IR 构造和 pass 负责参数合法性，调用适配器负责 Scalar 范围和借用存储校验。

## 公共编译产物

内部 `CompiledArtifact` 持有单芯片目录、平台、`ArtifactMetadata` 和延迟 callable 加载。
`CompiledProgram` 通过组合保留 live IR、路径兼容、program 返回式分配和 runner。
参数名称、输出分配位置和返回数量从公共 metadata 派生，不在包装层重复保存。
目标属性使用产物的有效平台，包括恢复时显式指定的目标覆盖。
多 orchestration 的子入口共用该产物，父对象保留入口选择；分布式执行保留现有包装层。
公共编译器只写入一次 sidecar，包装结果时不再重写；直接从 live IR 构造的兼容路径
仍会创建 sidecar。

公开 `ir.compile()` 与内部 `_compile_artifact(contract=KERNEL)` 通过 `_compile_to_files()`
共用 passes/codegen；该步骤仅返回 lowered IR 和构建信息，不创建执行包装对象。
`_compile_artifact()` 只接受单芯片单入口，始终返回 `CompiledArtifact`。
公开 `compile()` 负责单程序、多入口和分布式包装，以及无法提取规范 lowered 签名时
的 program 兼容回退。
公开 `compile()` 显式请求 PROGRAM，直接 JIT 调用请求 KERNEL。公共缓存 key、manifest
能力及恢复消费者均携带该契约，防止复用不兼容的产物。不新增公开 `compile_kernel()`。`contract` 选择 PROGRAM/KERNEL
调用规则，`execution_mode` 独立选择目标平台的 SIM/ONBOARD。
kernel 要求单芯片、单入口、显式提供 Out/InOut buffer，
且 Tensor 返回必须别名到外部参数。现有 C++ 返回血缘分析从原始 IR 生成有序别名，
允许重复。该契约检查外部入口签名，不增加独立的内部写入及别名扫描；内部 submit
仍可省略由 runtime 分配的 Out 参数。lowering 后核对外部参数 ABI：数量、顺序、
名称、方向、dtype 和 storage。
外部参数名称去除 SSA 版本后缀。
HBG Host Tensor 数据读写被拒绝，设备 submit predicate 保留原语义；program 编译
仍支持原有分配及 Host 控制流程。

metadata 区分逻辑调用签名 `params` 与 assembly 签名 `lowered_params`，均使用
`ParamInfo`。PROGRAM 参数绑定和输出分配使用 lowering 前的原始 `params`，
从产物恢复以及选择多 orchestration 子入口时也保持这一行为；`lowered_params`
仍用于 assembly ABI 校验。Torch 按逻辑 shape 校验；NZ lowering 按元素单位验证无 padding 的
blocked 表示，dtype 不变，因此字节数一致。动态 NZ 支持逻辑 `[B, R, C]` 的 batch
维度；blocked shape 不作为 Torch 的逻辑 shape。

`compiled_meta.json` schema 3 保存两种签名、返回数量、别名、平台/backend 和已验证
执行能力。生产者仅声明本次请求的能力，不自动声明双模式兼容。schema 1/2 旧 program
metadata 仅按 PROGRAM 恢复，不推断 kernel 契约。kernel 必需字段缺失、索引非法或
能力不一致时要求重编译。`from_dir()` 仅读取 metadata，不加载 native。

`ArtifactRuntime` 仍是唯一缓存晋级/加载实现。单芯片恢复先创建 `CompiledArtifact`，
PROGRAM 消费者再加执行包装，内部 KERNEL 消费者直接获取产物。manifest、构建锁、
prebuilt 校验和阶段晋级完全复用。加载串行化并在 assembly 成功后发布状态，不初始化、
注册或执行 Worker。缓存晋级保持源路径身份稳定，运行输出位于不可变缓存外；注册
handle、stream、输入 Tensor 和运行资源不进入公共产物。

program 提交构造无地址的 `TaskArgs`，由 Simpler 解析为 `ChipStorageTaskArgs`。
Torch adapter 通过公开 Worker 接口提交 `TaskArgs`，使用 `Buffer.wrap`
和 Simpler Tensor 视图描述借用的设备存储。Scalar 的值和类型通过公开 `TaskArgs`
typed scalar 接口传入，由 Simpler 完成 ABI 编码。
`ScalarArgument` 仅保留校验后的值和 dtype，不执行编码。

Runtime 能力检查在显式 Torch kernel 初始化时执行。生命周期层支持 Worker 初始化和
callable 注册及 eager 提交；公开算子注册和 DFX 窗口留待后续迁移。

## 进程初始化与生命周期

内部 `runtime/kernel` 包按资源归属组织：

- `session.py`：配置、runtime 能力检查和借用上下文会话（`_BorrowedWorkerSession`），
  将公开 `simpler.Worker` 的生命周期操作绑定到 Torch context。
- `context.py`：进程状态、并发注册复用和 pending ticket 清理。
- `owner.py`：生命周期 owner 线程。
- `callable.py`：共享 artifact 身份及绑定上下文的注册记录。
- `pending.py`：callback 结果发布及单次调用的资源保留。
- `arguments.py`：会话拥有的 storage 注册及单次 TaskArgs 租用。

session 是生命周期适配层，不是独立执行引擎。Simpler 负责 callable 注册和执行，
进程管理器负责调用准入和缓存。

`pypto.torch.init()` 显式借用当前 torch_npu 设备上下文，支持 onboard `a2a3` 的
`tensormap_and_ringbuffer` 和 `host_build_graph`。显式 `device` 必须匹配当前设备，
初始化不会切换设备。graph capture 在占用进程模式前抛出 `NotImplementedError`。
共享 `CallConfig.validate()` 在 preflight 中检查 native ring-size 约束。
配置错误、能力缺失和 runtime 身份不匹配也在 preflight 阶段拒绝；内部异常保持原类型。

配置包含 AICPU 线程数、swimlane/dep-gen、输出目录、ring task-window/heap/dep-pool
及可选 `workspace_budget_bytes`。ring 标量广播到四个 scope-depth ring。诊断配置
要求输出目录，本阶段只保留配置，公开 DFX 窗口后续实现。workspace budget 遵循公开
Worker 的最小 1024 字节要求，不包含借用的 Tensor 存储。

runtime submodule 跟随 upstream 的 Simpler revision
`531765c210813b019e9d6a491e3ace92d357fb26`。实际 native binding 的身份仍由既有
`check_runtime_pin` 检查，包括身份缺失和显式跳过规则。能力检查要求公开 borrowed init、
caller-stream submit、Buffer/TaskArgs、RunHandle 和 `flush_diagnostics` 接口；
不查询旧 kernel supported bit 或私有 slot。接口检查和 UT 不构成设备执行正确性验证。

进程管理器持有一个 Worker 和非零 context-generation token，状态包括 UNINITIALIZED、
INITIALIZING、READY、FAILED、CLOSING、CLOSED。同配置初始化幂等，冲突及递归初始化
拒绝，并发初始化共享结果。native 初始化失败保留 Worker 和模式以便清理；关闭后不允许
切换到 program。program 模式在 runner 执行前（包括 DFX dependency-capture 子进程）
及直接初始化 Worker 时占用，导入、编译和 metadata 加载
均不占用模式。一次性分布式执行在将 host Tensor 转换为共享内存或清理已有 dispatch
诊断文件之前占用 program 模式。kernel 初始化后 fork 的子进程必须改用 spawn；未使用的管理器及其继承的锁
在单线程的 at-fork 子进程回调中统一替换一次，早于子进程中的并发首次访问。
父进程未声明模式或使用 PROGRAM 模式时，模式门禁也在 at-fork 子进程回调中重置。
子进程中竞争的 PROGRAM/KERNEL 声明共享同一把锁：一种模式获准，另一种在 native Worker
初始化前抛出异常。继承的 KERNEL 声明不会被重置。

生命周期 owner thread 在调用公开
`Worker(level=2).init(prewarm_config=..., borrowed_context_generation=...)`、
`Worker.register(callable)` 和 `Worker.close()` 前绑定借用的 RTS context。
Simpler 管理 runtime binaries、slot 和 native 清理。注册复用 `CompiledArtifact`
的公共加载器，校验 KERNEL 能力及 platform/runtime，保存 opaque `CallableHandle`。
缓存产物复用既有 `ArtifactKey`，私有产物使用对象身份，避免目录覆写后误用旧 handle；
不定义新散列。并发 prepare 共享成功或失败；registry busy 和容量
错误原样传播，不新增 drain、wait 或自动淘汰。

pending ticket 在入队前持有 registration 和每次调用的资源。callback adapter
向 ticket 发布 RunHandle，无需获取管理器锁。普通调用只轮询完成的 ticket；尚未执行的
callback 视为未完成。`RunHandle.done` 为真后仍调用 `wait()` 完成 native 退休清理，
随后才释放借用 source。显式 drain/shutdown 可同步框架队列，按 run 退休、source 释放、
Worker 关闭的顺序清理。callback、退休或 source 释放失败均保留 ticket 和 Worker，
不假定设备已静止。

可选 `_torch_npu` 扩展提供生命周期/context、NZ storage 查询、allocator 保护及 Torch 队列桥接。用匹配的
Torch、torch_npu、CANN 头文件及库，通过 `-DPYPTO_BUILD_TORCH_NPU=ON` 构建；
它不链接 Simpler C++ 实现或独立 kernel ABI。默认 core 构建不要求这些可选依赖。
即使启用 `PYTHONOPTIMIZE`，配置阶段仍会执行 C++11 ABI 检查。
关闭逻辑包装已验证的框架 teardown hooks（torch_npu `2.6.0.post2`、`2.10.0.post2`，
允许构建后缀），在框架拆除前关闭 PyPTO。无法安全清理时通过 native anchor 保留资源
直至进程退出，避免 context 销毁后运行析构。
fork 子进程在框架拆除前通过 native anchor 保留继承的状态，不获取继承的锁，也不关闭父进程的 Worker。

## 直接 JIT 调用

```python
import pypto.torch
from pypto.runtime import CompileOptions, ExecutionMode

pypto.torch.init()
# x and out are real NPU tensors; every Out/InOut buffer is supplied.
result = op(x, out, config=CompileOptions(execution_mode=ExecutionMode.ONBOARD))
```

初始化必须先于参数绑定和编译成功完成。省略 `config` 或使用 `CompileOptions` 的默认
平台时，按初始化绑定的目标编译。直接调用拒绝非默认且不匹配的平台、`RunConfig`
及分布式配置。没有活跃 `PassContext` 时，编译使用 `init` 绑定的 runtime，包括 HBG；
存在活跃 context 时，其 runtime 必须与初始化配置一致。解析后的平台和 runtime
也用于标识缓存产物。`op.compile(..., config=run_config)`
和 `op.warmup(...)` 保留 program 语义，不要求 kernel 初始化；同一进程内 program
与 kernel 执行仍互斥。

首次 eager 调用完成 specialization、逻辑签名校验、公共 passes/codegen、产物加载及
assembly、callable 注册。后续调用复用兼容产物和 handle，runtime Scalar 改变不重编译。
返回值为调用方 Tensor 的别名，允许重复别名，不分配业务输出。即使已 eager 预热，
capture/replay 也在编译或注册前抛出 `NotImplementedError`。

kernel 调用通过 `runtime.kernel.arguments.BorrowedArguments`，从校验后的 NPU
CallFrame 构造 TaskArgs，并管理本次调用的 storage 租用，不进入 program 的
`runtime.tensor_arg.make_tensor_arg`。pending ticket 持有 BorrowedArguments
直到退休，再调用其 `release()` 方法。
storage 按基址和完整容量通过 `Buffer.wrap(..., location=worker.device_location)`
注册。Tensor view 保留字节 offset 和 stride；原生 NZ 使用经过校验的物理 blocked
维度。同一 storage 的 view 和重叠 pending 调用共享会话 source，每次调用持有租用
直到 run 退休；最后一个租用通过 `Worker.release_buffer()` 关闭 source 并清除缓存导入。
释放失败时保留租用供后续重试。Tensor 使用 INPUT/OUTPUT_EXISTING/
INOUT 和显式 NONE transfer，Scalar 通过带类型的 ctypes 值传给公开
`TaskArgs.add_scalar`，不隐式执行 H2D。

`RunOpApiV2` 入队前，Python pending ticket 已持有 BorrowedArguments、Torch stream、
callback、配置、Worker 和 callable。native `enqueue` 函数校验 context、记录 allocator
stream 并捕获 native stream，不再创建 native ticket 或持有 Tensor 引用。
入队前设置 torch_npu 的进程级 `g_used_aclop` 标志，让支持的 2.6/2.10
版本在后续队列排空时释放 GIL；该标志保持到进程结束。
入队期间也释放调用线程的 GIL；callback 获取
GIL 后调用公开 Python `Worker.submit(..., caller_stream=...)`，再发布 RunHandle。
不使用同步 `Worker.run`、私有 native submit 或 slot API。native 队列状态不拥有
Python 引用；pending owners 和 callback closure 仅在退休成功后、持有 GIL 时释放。

外层 JIT 返回、callback 提交返回、设备完成是三个不同阶段。普通 enqueue 和未完成
ticket 轮询不会 drain、synchronize 或等待 callback/run。已完成的 run 仍需 `wait()`
退休后才释放 source；显式 shutdown 可以同步保留的 Torch stream 并等待 run。Simpler submit 自身
可能等待容量或前驱。提交或退休失败、完成状态未知时保留所有者。

UT 使用真实初始化状态转换、JIT passes/codegen、公共产物发布/恢复及二进制校验，
仅模拟外部编译器输出和 native/framework 边界。UT 不证明可选 adapter 真实编译、
实际 taskQueue/GIL/锁行为、allocator 安全或 NPU 数值正确性。下述依赖设备的 ST
覆盖数值执行和队列排空回归；跳过的用例不能证明设备执行正确。

## 调用元数据与所有权

[`CallSignature`](../../../../python/pypto/torch/interop.py) 持有共享的不可变
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

Tensor/Scalar 的排列不定义 native ABI 布局。native 参数编码同时消费这些值
和经过校验的产物 metadata。地址、Scalar 实际值和 stream 属于每次调用状态，
不进入编译 key 或持久 metadata。

frame 持有每个 Tensor 及其 storage 对象。`alias_result()` 返回 `None`、单个已有
Tensor 或已有 Tensor 元组，不分配业务输出。仅有 Python 引用不能保护 frame 释放后
的设备异步使用；native 队列所有权和 allocator stream 记录属于独立的 launch 层工作。
frame 使用期间，调用方不能 resize 或使借用的 storage 失效。

## 校验规则

- 接受同一设备上的真实 NPU torch Tensor。拒绝 CPU、Meta/Fake Tensor 和 Worker
  自有对象，不自动选择 program mode。
- BOOL Tensor 在编译或源注册前被拒绝：固定版本运行时缺少借用 BOOL Tensor 的类型映射。
  BOOL Scalar 仍受支持。
- 对照 `ParamInfo` 校验 dtype 和 rank。静态 carrier 维度必须匹配；`-1` 维度使用
  本次大小。打包 FP4 直接使用 `ParamInfo` 已提供的 carrier shape，不再次展开。
- 要求 base NCHW (0) 或 ND (2) 格式的连续 strided view，保留非零 storage offset
  和逻辑地址。其他格式、转置 view、未解析的 conjugate/negative view 及无效 storage
  边界明确报错，不进行复制或格式转换。
- 显式 `layout="NZ"` 且 `platform="a2a3"` 时允许只读 FP16/BF16/INT8 的原生
  format 29 输入；物理维度必须对应完整、无 padding、零 storage offset 的 NZ 矩阵。
  声明 NZ 的参数拒绝 base-format storage，适配器不转换布局。
  物理描述符查询按需加载可选 native adapter；UT 仅替换该查询边界。
  签名校验直接比较逻辑 Torch shape 与逻辑 `ParamInfo`；物理查询独立校验
  blocked 维度和字节容量。
- `CallSignature` 接受空 view：它们不访问任何元素，因此非负 storage offset 可以
  超过 storage 容量；仅对非空 view 校验访问范围上界。eager launch 的运行时描述符
  契约更严格，要求 rank 为 1..5，extent/stride 为正 u32 值。
  即使 frame 校验成功，零尺寸 Tensor 仍会在编译或注册前被拒绝；空 view 的互操作
  测试仅覆盖 frame 校验。
- 允许字节区间完全一致的别名（可具有不同 dtype/shape）、同一 storage 上互不相交的 view，以及重叠的
  只读 view。涉及 Out/InOut 的部分重叠需要更丰富的别名契约，当前明确拒绝。
- 使用 program Scalar 类型映射复制 Python 或类型匹配的 ctypes Scalar。
  整数必须位于声明类型范围内，不将浮点数静默截断成整数，拒绝有限浮点值溢出。这不改变 JIT 的运行时
  Scalar 或常量分类。
- autograd 开启时拒绝需要梯度的 Tensor；`torch.no_grad()` 推理可以借用模型参数，
  但不提供 backward 实现。

参数校验后，适配器在**每次调用**查询当前 device/stream，要求当前设备与 Tensor
设备一致，并核对 stream 设备。不切换设备、不缓存首个 stream、不读取 `npu_stream`，
也不同步或入队。这些上下文查询可能初始化 torch_npu 自身的框架上下文，但不创建或
初始化 Simpler/PyPTO Worker。

## 内部 schema 与 Fake/Meta 辅助

`pypto.torch.registration.RegistrationSignature` 持有相同的不可变 `ParamInfo` carrier shape
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
不注册算子。`pypto.torch` 不新增公开注册 API，本阶段也不安装真实设备 kernel。
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
带返回别名的 schema 分别
验证 schema 正确性与 Fake/Meta 行为，不把这些检查视为该类算子已支持函数化
（functionalization）或编译执行。正式 kernel 注册、设备执行、autograd 与最终编译器
接线仍属于后续工作。

## 可选依赖与范围

`import pypto.torch` 导出 `init`，初始化后的直接 JIT 调用通过 eager 路径提交 kernel。导入该包、`interop` 或 `registration` 模块不请求 torch_npu、
Simpler 或 native launch 扩展；`torch` 仍是 PyPTO 的常规依赖。真正描述 NPU 调用时
才按需加载 `torch_npu`，缺失时给出针对性的错误信息。

公共编译、metadata/缓存加载、调用 frame 所有权、进程初始化和 eager 提交均已实现。
直接 JIT 调用使用 Torch 队列桥接，显式 `compile()` 保留 program 执行语义。
单元测试通过模拟执行边界验证队列顺序和资源生命周期。native adapter 构建、真实设备
数值正确性及 allocator 安全仍需设备验证；ACLGraph capture/replay 仍不支持。

## 验证

共享 UT fixture `run_without_optional_runtime` 启动独立 Python 进程，通过导入查找器
（import finder）阻止 `torch_npu`、`simpler`、`simpler_setup`、`_task_interface` 和
`pypto._torch_npu`。
同时捕获 import 语句与 `importlib.import_module()` 动态导入，即使调用方捕获了
`ImportError`，该次尝试仍使检查失败。显式检查在 `PYTHONOPTIMIZE=1` 和 `2` 下仍然生效，
负对照覆盖这两种优化级别及普通执行。子进程关闭 PyTorch backend 自动加载，以隔离
PyPTO 自身行为与已安装的框架插件。查找器的负向测试确保 CPU runner 即使原本就未安装
这些依赖，也不会把导入尝试误报为通过。

这些检查覆盖包导入/重载、program 配置和注册后的 Fake/Meta dispatch，直接进入现有
全量 UT CI，无需可选 runtime 依赖或新增设备 job。它们不验证 native adapter 构建或
设备执行；这些验证仍属于集成分支。

`tests/ut/torch/test_interop.py` 使用真实 CPU storage 和模拟的 NPU device 标签，
只替换框架 format/context 查询。覆盖 view offset、alias、独立 Scalar/stream 快照、
所有权、非法输入和禁止导入可选 runtime 依赖的隔离进程。

`tests/st/runtime/kernel/test_torch_interop.py` 核对真实 NPU Tensor、非默认 stream、
offset view 和非连续输入拒绝行为。没有真实 NPU 或所选平台为模拟器时跳过。
这些测试包含 metadata 验证，以及使用 native adapter 的子进程队列回归测试：
验证持有 GIL 的 `stream.query()` 和 `empty_cache()` 能完成队列排空。
队列回归只执行 Python callback，不启动 PyPTO 设备 kernel。

`tests/ut/torch/test_registration.py` 覆盖 schema mutation/alias 契约、Fake/Meta
与符号输入、隔离导入、重复定义，以及不依赖真实 kernel executor 的测试内
dispatcher/compiler 集成。

`tests/st/runtime/kernel/test_torch_eager.py` 新增 eager 数值执行 ST，覆盖单次调用、
八次具有数据依赖的连续调用，以及 Torch NPU/PyPTO 交错调用。三种场景均覆盖默认和
非默认 stream、两种受支持的 runtime，并在独立解释器中执行以隔离进程执行模式。
CI 使用单独的 pytest 步骤，确保父进程尚未初始化设备。
所有结果在最后一次 stream 同步后与 CPU 参考值比较，连续调用和交错调用之间不插入
同步。输出缓冲区预填 NaN，并改变运行时 scalar，以检测漏写和过期参数。

```bash
python -m pytest tests/st/runtime/kernel/test_torch_eager.py --platform=a2a3 --device=0 -v
```

12 个执行用例需要真实 NPU、`torch_npu` 和可选 `pypto._torch_npu` adapter；缺少
这些依赖或指定 `--codegen-only` 时跳过。另有两个 codegen 冒烟用例，分别为两种
runtime 完整编译测试 kernel，但不验证设备执行。执行用例被跳过时，CI 通过不能
证明 eager 数值执行正确。
