# 外部 Kernel 契约（内部 v1）

本文是任务 34.1–34.4 和 [RFC #3293](https://github.com/hw-native-sys/pypto/issues/3293) 的接口基线。
范围是数据交接与可执行 IR，不是公开 provider 注册 API。下文的 CANNBotDSL JIT 集成支持经过校验的 A5 子集。

## 职责

| 组件 | 负责 | 不负责 |
| ---- | ---- | ------ |
| Provider | DSL 识别、编译、元数据转换 | JIT AST 状态和 runtime wrapper |
| 通用契约（Contract） | 逻辑签名、效果、公开返回和存储关系 | 实际 tensor 地址和可变请求状态 |
| JIT | 参数绑定、特化、效果与返回规范化 | 在通用语义中增加 provider 专用分支 |
| 执行适配层（Runtime bridge） | 原生 ABI 转换、目标检查、任务完成保证 | 将公开返回等同于所有被写参数 |

实现位于 `python/pypto/jit/external/`；bridge 协议位于
`python/pypto/backend/external_bridge.py`。导入这些模块不会导入可选 DSL。
内部注册与捕获已在 `registry.py` 和 `compatibility.py` 实现。
JIT 已实现调用规范化和签名变体；34.4 将实现原生 wrapper 和运行期绑定检查。

## Provider 发现与捕获

`ProviderRegistry` 的作用域由调用者显式管理。注册时分别提供轻量的
`matches(object) -> bool` 识别函数和 `load() -> CapturingKernelProvider` 工厂。
注册与发现阶段都不会加载编译器、捕获输入或编译 kernel。识别函数只能检查对象，
不能导入可选 DSL。框架不隐式扫描插件，也不安装全局 kernel 缓存。

```python
from pypto.jit.external.registry import ProviderRegistry

# ExampleKernel and ExampleProvider are supplied by the integration.
registry = ProviderRegistry()
registry.register("example", matches=lambda obj: isinstance(obj, ExampleKernel),
                  load=ExampleProvider)
ref = registry.discover(kernel)  # None when no provider recognizes the object.
# Explicit selection is available when multiple providers recognize an object.
ref = registry.bind(kernel, provider="example")
captured = ref.capture(request)  # Validated signature and target are now known.
prepared = captured.prepare()
```

多个 provider 同时匹配时明确报错，注册顺序不决定优先级。重复名称和非布尔识别结果也会报错。
显式绑定直接选择指定 provider，不执行自动识别。`KernelRef` 强引用 kernel 和注册项；
registry 不持有已发现的 kernel 或捕获结果。注销只影响后续发现，已有引用仍然有效。
每个注册项的工厂只加载一次，包括并发捕获场景。Provider 必须支持独立并发请求，
或自行串行化编译器调用。

捕获型 provider 实现 `capture(kernel, request) -> CapturedInputs` 和
`prepare(captured, request) -> PreparedKernel`。`CapturedInputs` 持有不可变的、
由 provider 编码的字节快照及 `PreparationIdentity`；摘要必须与快照字节一致。
Provider 将相关源码、引用常量和依赖编码到快照中，并在准备阶段解码快照。
`complete` 仍是 provider 对完整性的承诺，摘要校验无法证明未声明的依赖已被捕获。
`False` 不产生可复用的准备键。`CapturedKernel.prepare()` 每次重新准备；
任务 34.6 将接入 artifact store 复用。输出校验拒绝 provider 改变请求或捕获身份。
原有仅提供 prepare 的 `ExternalKernelProvider` 协议继续供直接调用者使用；
registry 使用支持 capture 的协议。

`make_request` 复制参数、常量、选项容器并校验其不可变内容。
`request_from_context` 是兼容边界：仅从 `SpecializeContext` 复制签名元数据和字面量常量，
规范化共享动态维度名称，并将不精确的范围保留为动态维度。
声明的维度按 DSL 符号字面量共享，即使 Python 别名不同也一样；
合成维度（synthesized dimension）仅在引用同一实例时共享，
因此复用局部变量名不会给独立的范围强加相等约束。
长度未经证明的切片使用独立符号，即使元数据回退为父张量的动态维度也不例外；
全切片保留已证明的相等关系。
同一不精确张量元数据的重复别名共享轴符号，独立视图则不共享。
缩短视图在回退边界之外保留独立的运行时维度身份。整形状读取和基于描述符的 reshape
保持这些轴为动态维度，包括跨普通 JIT 依赖的情况。局部视图的描述符读取可能获得独立符号：
parser 不一定能证明它与最初用于指定视图大小的 scalar 相等。完整下标切片保留可证明的相等关系。
省略的 layout 规范化为 ND，不支持的常量会明确报错。旧 context 无法证明 stride 和 offset，
因此分别保留为未约束和运行期绑定。

现有 `__pypto_jit__()` 钩子可返回稳定的 `JITFunction`；发现阶段检查返回类型，
编译继续由普通 JIT 路径执行。`legacy_builder(ref, target, lower)` 将接收 context 的
builder 回调转换成 capture/prepare，再显式调用 `lower(PreparedKernel)` 生成旧 builder 结果。
Provider 不接触实际 context、AST、globals 或 tensor 地址。这是 PR #3231 中 builder 路径的
适配边界，不安装该 PR 的 runtime wrapper，也不让任意注册对象直接在 JIT 程序中执行。
将任意注册对象连接到可执行 runtime wrapper，以及真实 DSL 迁移，仍属于后续任务。

## CANNBotDSL 准备阶段（任务 34.5，第一阶段）

`pypto.jit.external.providers.cannbotdsl.CANNBotDSLProvider` 通过可选包的公开
`inspect_kernel`、`export_device` 和 `get_include` API 实现直接调用
`prepare(kernel, request)`。导入适配器不会导入编译器。本阶段支持 A5/CCEC、
无默认值的显式位置参数、tensor 和运行时 scalar 描述符，以及命名 constexpr 值。
Tensor 请求必须采用 ND 布局，明确给出正步长且最内层步长为 1；未约束的步长会被
拒绝，不会推断为连续布局。共享维度符号保持共享；仅用作长度的符号允许为零，
用作任一步长的符号最小值为 1，即使该符号同时用作长度。暂不接受编译选项。

例如，行步长动态的 `(rows, 32)` tensor 可描述为
`TensorSpec(("rows", 32), DataType.FP32, strides=("pitch", 1))`。
运行时 view offset 仍保留在请求中；下述调用适配器在构造提供方的首个有效元素指针时，
只应用一次 offset。运行时绑定校验必须证明得到的地址位于存储边界内。

结果为使用 `cannbotdsl.device_callable` v1 描述符 ABI 的 `PreparedKernel`，
保留导出的物理参数顺序。公开返回为空；每个 tensor 写入都产生内部写回，覆盖范围
保守设为未知，避免仅凭访问掩码丢弃未写入数据的依赖。AIC/AIV/MIX 参与核来自导出，
其中 MIX 包含两个 AIV lane。
逻辑运行时契约中的 scalar 元素类型必须与请求一致，不能仅比较字节宽度。
格式错误的导出（包括缺少 `manifest` 或 `source` 属性）会在交接前抛出 `ContractError`。

准备阶段还会校验 v1 描述符布局：字节偏移 0 处为 8 字节指针，随后依次紧密排列
int64 动态 shape 字段和动态 stride 字段，整体按 8 字节对齐。静态维度不占用描述符
槽位。字段偏移、各数组内的槽位、轴号、总大小及 scalar 类别和位宽必须与请求一致。
数值元数据严格区分类型，不接受以布尔值或浮点数代替整数偏移和大小。

逻辑契约 schema 6 会单独校验，因为物理 `-1` 标记无法表示维度相等关系。Tensor
元素类型、GM 地址空间、shape/stride 表达式及维度边界必须与请求完全一致；额外或
更强的约束会被拒绝。支持的 view 约定为首个有效元素指针、消费端应用 offset、正步长
且元素不重叠；不支持跨参数写重叠。这些检查只验证声明，下述绑定校验器负责验证
实际地址、边界和重叠关系；runtime bridge 必须在每次提交前调用它。

产物拥有 `device_callable.hpp`、完整的 `device_manifest.json`，以及经过摘要校验的
DSL 扩展头文件副本。Manifest 保留描述符布局、view 约束、状态修改、独占资源、
缓存行写隔离、完成条件和二进制加载要求。CANN SDK 头文件仍由消费端工具链提供。
每次准备都重新导出；由于尚未建立完整的 Python 闭包/输入快照，身份明确标记为不完整。
因此该 provider 尚未注册到仅支持 capture 的 registry，也不声明 preparation 缓存复用。

A5 TMR bridge 支持下述单 block AIC/AIV/MIX 子集的直接执行、普通 JIT 自动发现和显式绑定。
完整输入捕获、持久复用及性能验收仍单独跟踪。准备和原生二进制编译不使用缓存。

### 普通 JIT 自动发现

在普通 `@pl.jit` 中可以直接调用未配置 host launch 的 `@cb.kernel` 对象：

```python
@cb.kernel
def native_kernel(x, out):
    cb.mem_copy(out, x)

@pl.jit
def pipeline(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    native_kernel(x, out)
    return out
```

AIC、AIV、MIX 均无需注册或编写 `pl.jit.extern` 声明。自动发现从 globals 和 closure
解析直接调用的名称，支持导入别名。模块属性或绑定方法先赋给名称，再定义 JIT 函数，
例如 `native_kernel = module.kernel` 或 `native_kernel = instance.kernel`。
公开 `inspect_kernel()` API 提供参数名，调用点元数据提供 tensor/scalar 类型。
参数必须为无默认值的显式位置参数；已配置 host launch 的对象会被拒绝。
PyPTO 不遍历原生函数体，导入 PyPTO 也不会导入 CANNBotDSL。

适配器按对象身份在一次编译请求内去重，支持不可哈希对象，并持有 kernel 和内部签名声明。
不同请求间重新绑定的 kernel 会重新发现；这不代表完整输入捕获，也不会启用缓存。

原有显式声明仍可使用：

```python
from pypto.jit.external.providers.cannbotdsl import CANNBotDSLProvider

@pl.jit.extern(binding=CANNBotDSLProvider().bind(native_kernel))
def native(x: pl.Tensor, out: pl.Tensor):
    ...

@pl.jit
def pipeline(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    native(x, out)
    return out
```

参数名与顺序必须与原生 kernel 一致。声明无返回值，使用普通 tensor 注解；读写方向与
隐藏 SSA 回写由经过校验的原生导出决定。AIC/AIV/MIX 拓扑也从导出推导，MIX 生成为
包含一个 AIC 和双 lane AIV 成员的 Group。调用前后可以使用普通 PyPTO 生产者与消费者。

使用 `compile(..., config=RunConfig(backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD))` 或携带相同配置的普通 JIT 执行，
运行时必须为 TMR。绑定暂不支持 `specialize()`/`lower()`；可查看编译后的程序。首版要求正的静态
ND shape、连续紧凑 strides、首元素地址 64 字节对齐，且整个视图占据完整的 64 字节缓存行。
每次提交前均检查 shape/stride/alignment/存储边界，包括已加载程序的复用；签名不是实际
存储的证明。启动检查要求单 block 且无同步启动。动态 shape、非连续或非对齐视图会明确失败。

每次编译重新准备，在程序输出目录的独立子目录保存拥有所有权的导出源码，因此准备结束后
仍可调用 `CompiledProgram.load()`。由于 Python 闭包捕获尚不完整，含绑定的程序绕过两级
JIT 缓存。每个原生成员也绕过原始及剥离后二进制缓存，使用编译器选定的 SDK 重新执行
零栈与 text-only ELF 准入。可复用的准备阶段身份仍由任务 34.6 负责。

这类编排的全部任务均关闭 early dispatch，包括普通 PyPTO 消费者，确保依赖等待生产者完成
及 MIX 全部 FIN。显式 `allow_early_resolve=True` 提示不能覆盖此要求。
原生视图覆盖完整缓存行，保证逻辑不重叠时也不存在缓存行伪共享。
逐核串行执行与桥接层入口/出口的流水线排空、状态恢复共同保证 callback 局部资源独占。

A5 系统测试在三种拓扑的真实 matmul/vec kernel 前后加入普通 JIT 计算，并编译、加载和重复
执行。每次更换输入、用 NaN 填充输出；MIX 消费者同时使用两个原生结果。
完整输入捕获、持久复用及性能验收仍为后续工作。

### A5 AIC/AIV/MIX runtime 路径

`TMRBridge.lower(prepared, target)` 在 `a5/ccec` 的 TMR v1 上支持
`cannbotdsl.device_callable` v1。支持单 AIC、单 AIV，或一个硬件配对的 AIC + 两个
AIV 组（MIX）。要求无隐式 workspace，以及与导出器完全一致的资源、状态、加载和完成声明。
支持 AIV UB 和 AIC L1/L0A/L0B/L0C 中静态且互不重叠的分配。Bridge 校验整数大小与偏移、
空间和核的归属、对齐（UB 为 32 字节，Cube 空间为 512 字节）、容量和每个空间的精确最高使用地址。
容量上限为 UB 248 KiB、L1 512 KiB、L0A/L0B 各 64 KiB、L0C 256 KiB。
暂不支持动态分配、重叠的分配记录、共享 SSBUF 或其他内存空间。
未知或更强的要求在编译前被拒绝。每次提交只能包含一个逻辑 block，且不得同步启动。

TMR 持久化 executor 在每个物理核上同步、串行调用 kernel，排队任务也不例外。
回调执行期间，executor 不使用 kernel 的本地内存或同步事件，因此该执行区间提供
资源独占；不同核之间不通过全局锁串行化。Wrapper 在入口前排空继承的流水线和标量
GM 写入，并设置 atomic-none。导出函数返回后，再次排空流水线、刷新标量 GM、执行
DDR barrier，恢复 ASC/DMA/padding/quantization/mask 默认状态和保存的 CTRL，
以 atomic-none 状态返回。此后 executor 才能发布该参与者的 FIN 或执行下一个回调。
AIC 入口还会清除继承的 HF32 模式；两个编译版本均恢复保存的 CTRL。
TMR 将 MIX 任务放入同一物理 cluster，仅在三个参与者的 FIN 全部到达后完成任务。
复用现有完成计数器实现整组汇合，不把本地 pipe barrier 当作跨核 barrier。
Kernel 作者仍须保证组内同步和写入缓存行的独占。

使用 `pypto.backend.cannbotdsl_runtime.compile_task(task, compiler,
pto_isa_root=...)` 按 `task.prepared.execution.cores` 顺序返回 text 镜像元组：AIC/AIV
返回一个镜像，MIX 返回 `(aic_image, aiv_image)`。每个镜像注册为 `CoreCallable`，
MIX 的两个 AIV lane 复用 AIV 镜像。所有镜像均通过准入后才返回元组。它从编译器选定的 CANN SDK
获取 C API 头文件，在编译期间保留拥有所有权的源码，并强制执行 text-only ELF 准入和
静态零栈报告检查。Text-only 编译省略可选的帧指针维护，使无寄存器溢出的 SIMD VF 调用
能够通过同样的零栈检查；实际栈使用仍然被拒绝。旧 `configure_context` helper 仍拒绝该二进制策略。使用上述 JIT 集成，
即可在普通编译路径中保留完整的准入要求。

直接 runtime 系统测试覆盖 FP16 Cube matmul（FP32 累加）、SIMD FP32 `(x + y) * x`，
以及同时执行两个计算流水线的 MIX 任务。MIX 生产任务的每个 AIV lane 负责一个 128 元素 tile。
独立 SIMD 任务和后续依赖 AIV 任务都在单个 AIV 上处理两个 tile（256 个元素）；依赖任务在整组
完成后读取 MIX 的两个输出。A5 向量寄存器为 256 字节，因此每个 tile 分两次执行 64 元素的
寄存器计算。重复执行时更换输入并用 NaN 填充输出，以检测陈旧数据。
这些用例使用分配的片上 buffer 和真实计算指令，尚不覆盖组内 Cube 到 vector 的 buffer 交接。
仅编译模式会构建设备和编排二进制；数值验收需要在 A5 设备上运行相同测试。

`build_submitter(tasks, function_ids)` 返回拥有所有权的编排头文件。每个任务对应一个
function-ID 元组，顺序与编译镜像一致；例如 `((10,), (20, 21), (30,))` 表示
AIC → MIX → AIV 批次。ID 必须非负，不同 task/core 镜像的 ID 必须互不重复。
相等的 `ExternalTask` 重复出现时，同一个 core 可以复用相同 ID；批次内同一 ID
映射到不同 task 或 core 会被拒绝。每个编译镜像需注册到对应 ID。
生成函数接收
`std::array<CoreTaskArgs*, N>` 并返回已提交任务的 ID。在任何提交前，它校验所有 launch
和绑定；然后通过有序批次辅助层关闭提前派发，并为每对冲突的 64 字节缓存行增加基于
完成的 `DEP_WAIT` 依赖。实际提交使用 `rt_submit_task(MixedKernels, args)`，未使用的
槽位设为 -1；MIX 的第一个 ID 填入 AIC 槽位，第二个 ID 同时填入两个 AIV 槽位，
返回的单个任务 ID 覆盖整组完成。
编排构建必须物化全部头文件，并将 callable、参数描述符和 tensor 存储保留到执行完成。
批次外的调用，包括 PyPTO 生产者和消费者，仍须自行建立依赖或 GM 隔离。
应包含返回的 `SourceArtifact.entry`，不要使用固定文件名。Runtime 入口文件名包含
批次与注册 ID 的摘要，因此多个 submitter 可以共享物化目录和翻译单元。

Host 测试验证 wrapper 排空/恢复顺序、真实 TMR 提交 API 和固定版本 scheduler 的整组完成计数器；
可选 CCEC 测试将静态及动态 AIC/AIV/MIX
导出通过完整准入流程。专项 runtime 测试
`tests/st/runtime/external_kernel/test_cannbotdsl_runtime.py` 构建各拓扑的依赖链和
AIC → MIX → AIV 链，MIX 测试参与者写入不同缓存行，
在 A5 上检查重复执行的数值结果。`--codegen-only` 编译设备和编排二进制并构造 callable，
但不启动设备。两种模式均需要固定版本的 Simpler 包和 CANN 工具链：在构造 callable
之前，`KernelCompiler` 就已经依赖 `simpler_setup`。`tests/st/codegen` CI 任务不安装
Simpler 或 CANN，也不运行该直接 runtime 测试。不能据仅编译结果宣称设备完成可见性或性能验收通过。

### 仅加载 text 的二进制准入

`KernelCompiler.compile_incore(..., text_only=True)` 为具有 `kernel_entry` wrapper
的设备可调用源码提供显式启用的 A5 编译路径。它生成独立的函数/数据 section 和编译器
栈用量报告，再按入口顺序链接并回收不可达 section。这样可以删除 `g_tilingKey` 等未被
引用的 SDK 数据；被引用的数据仍须通过最终校验。

返回链接后的 ELF 前，编译器要求仅有一个非空、可执行的 `.text`，全局 `kernel_entry`
位于其首字节，不包含其他非空的已分配 section、未处理的重定位或未解析符号，且 section
表和符号表有效，包括索引 0 处的空 section。所有 section 名称必须为 ASCII，避免加载器
解码名称时忽略无效字符并选择不同的字节。非空的 `.ascend.meta*` section 会被拒绝，因为 kernel 启动元数据可能
要求额外预留 SU/SIMT 栈或 UB。每个保留的函数都必须具有编译器生成的**静态零栈用量**报告。缺失报告、
动态栈帧和正栈用量均会被拒绝，因为持久化 runtime 当前没有每任务编译器栈预留契约。
具有完全相同非空代码范围的函数别名共享一个栈帧，至少需要一份报告，且所有可用的别名报告
都必须显示静态零栈用量。大小为零的符号独立校验。
链接器删除的函数报告不占用预算。编译、报告生成和检查在同一私有构建目录内完成，成功
或失败都会清理该目录。

此模式支持 A5 AIC/AIV 编译；拒绝模拟器及独立编译的 VPTO helper，因为其栈证据不在本次
校验范围内。普通编译保留既有行为。该检查验证编译器产物结构，不能证明任意机器码与
位置无关。它不会启用描述符 JIT 调用、预留本地内存/同步资源、落实批次排序或证明设备
完成可见性；bridge 仍须在提交前建立这些保证。后续接入产物缓存时，必须将该构建策略
纳入身份并保留准入证据；此直接编译路径不查找缓存。

`test_kernel_compiler.py` 覆盖无效二进制和栈报告、失败清理，并在环境可用时使用 CCEC
编译链接真实 CANNBotDSL 静态/动态导出。这些是二进制准入检查，不代表设备执行验收。

### CANNBotDSL 调用适配器（仅编译）

`pypto.backend.cannbotdsl_abi.build_call_adapter(prepared)` 生成拥有全部内容的
`SourceArtifact`，包含 `call_adapter.hpp` 及原生源码、manifest 和头文件。
它不读取 provider 目录，也不导入可选编译器。生成代码前，适配器会重新核对
manifest 与 prepared contract、原生 ABI、执行元数据及源码/头文件摘要的一致性。

生成的设备函数消费 TMR 参数存储（tensor 在前，scalar 位模式槽位在后），保留导出
的物理参数顺序，只应用一次元素 offset，并填充动态 shape/stride 数组。C++ 断言
将实际函数签名、POD 布局、指针和数组类型、偏移、大小及对齐与 manifest 对照检查。
即使原生整型载体的有符号属性与逻辑类型不同，也保持其位模式。Tensor 元素类型必须
与请求一致，不能只比较字节宽度。浮点 scalar 保留
负零和 NaN payload。Bool 槽位的低字节被规范化为合法的 C++ bool。TMR 限制为
最多 32 个 tensor、16 个 scalar，以及 1..5 维 tensor；不支持 FP16/BF16 scalar 槽位。

这是可调用头文件，**不是 `ExternalTask` 或可加载的 runtime 入口**：它有意不定义
`kernel_entry`；上面的公共 runtime bridge 提供入口和执行策略。调用方必须在每次调用时运行绑定
校验器，并落实保留的资源、缓存行、栈、二进制加载及完成条件。Host C++ 测试实际执行
参数组装及 offset 处理，可选 CCEC 测试编译真实公开导出；两者均不代表设备执行或
runtime 完成正确性验收。

### CANNBotDSL 逐次调用绑定校验器（binding checker）

`pypto.backend.cannbotdsl_abi.build_binding_checker(prepared)` 生成包含
`binding_check.hpp` 的自持有 `SourceArtifact`。重新核验 prepared manifest 及源码/头文件
摘要后，它为实际 TMR `CoreTaskArgs` 生成校验函数。使用 A5 TMR 编排头文件包含它，
并在打包或提交同一组绑定前调用其导出符号。生成过程不导入可选编译器，也不捕获
任何实际 tensor 地址。每次调用均须重新执行校验，包括命中产物缓存时。

它复用 TMR 对参数数量、已有输出存储、rank/dtype、静态和共享维度、正步长、offset
约束、存储边界及算术溢出的检查。CANNBotDSL 对齐检查作用于首个元素地址
（`buffer.addr + start_offset * element_bytes`），至少满足元素的自然对齐要求。
`c_pointer` bridge 保留独立的分配基地址对齐规则。

对每个非空 tensor，校验器按 stride 排序各轴、忽略长度为 1 的轴，并要求每次跨步
越过此前各轴覆盖的跨度。这能证明带 padding 和轴置换的布局不重叠；无法通过该保守
检查证明的交错布局会被拒绝。不同参数的字节包围区间只有在两者均不写入时才允许
重叠。空 view 不访问字节，但仍要求 stride、offset 合法，指针可表示且满足对齐。
这些检查在 release 构建中仍然生效。

生成校验器前会检查 TMR 元数据表示范围。`INDEX` 使用 runtime 的 `INT64` tensor
标签；E4M3FN 和 E8M0 使用各自独立的 FP8 标签。E5M2 因 TMR 缺少对应标签而被拒绝。
不支持 FP8 runtime scalar，也不支持 FP16/BF16 runtime scalar。

通过校验器**不代表已允许设备执行**：字节级不重叠无法证明任务间的 64 字节缓存行
隔离。资源所有权、栈/加载校验与任务完成保证需要使用上述共享 runtime bridge；
JIT 集成也执行同样的检查。

### CANNBotDSL 跨调用缓存行冲突

`pypto.backend.cannbotdsl_abi.build_conflict_checker(first, second)` 生成包含两侧绑定
校验器的自持有 `conflict_check.hpp`。两个 `PreparedKernel` 可以有不同的签名、dtype
和访问效果。两侧 manifest 必须声明 64 字节 `serialize_conflicting_cache_lines` GM
隔离策略；缺失或未知策略会被拒绝。可选字段 `within_group` 必须为
`kernel_author_must_ensure`，表示协作 kernel 组内的同步仍由 kernel 作者保证。
导出符号接受两个 `CoreTaskArgs`，返回这对调用
是否需要串行化。

无论是否已经发现冲突，返回前都会完整校验两侧调用。它根据实际分配地址、元素
offset、shape、stride 和元素宽度计算各访问 tensor 的保守字节包围区间，再比较包含
末端的缓存行编号，避免在地址空间顶部向上取整溢出。空 tensor 和 `Access.NONE`
不产生访问。允许 read/read 共享；只要共享缓存行且至少一侧写入，就返回 `true`。
带 stride 的 view 中的空隙也保守地计入范围。

例如，两个单元素 FP32 输出 view 分别位于 `0x2000` 和 `0x2004`，虽然字节不重叠，
但同属 `0x2000` 缓存行，因此返回 `true`。第二个 view 移到 `0x2040` 且其他访问独立
时返回 `false`。一侧输出与另一侧输入共享缓存行也要求串行化。指针 offset 在计算
缓存行之前应用。

这是调度判定函数，不是锁或提交 API。调用方必须检查所有可能并发的调用对，并在
返回 `true` 时保证前一调用完成后才开始后一调用；仅检查相邻调用不足以保证安全。
绑定在执行完成前必须保持不变，存储必须保持有效；绑定变化后必须重新检查。
返回 `false` 仅证明这一对调用没有 GM 冲突，不证明本地资源、栈/加载要求或完成
可见性。协议之外的调用也需要依赖或隔离策略。下述批量辅助层将判定结果转换为
TMR 依赖。JIT 子集则要求视图覆盖完整缓存行，并使用等待任务完成的普通数据依赖；
任意非连续视图的调度仍不支持。

### CANNBotDSL 有序批量提交

`pypto.backend.cannbotdsl_submit.build_ordered_submitter(prepared)` 接受非空的
prepared kernel tuple，在 `SourceArtifact.entry` 指定的位置生成自持有头文件。
文件名包含批次身份，避免不同批次互相覆盖；相同的支持头文件可以共享。生成函数接受
`std::array<CoreTaskArgs*, N>` 和同步回调
`(size_t index, CoreTaskArgs& args) -> TaskId`，按批次顺序返回提交得到的 ID。
回调必须提交对应位置的 prepared kernel，同步消费传入的参数和依赖，并返回新任务
ID。辅助层不负责加载 kernel 或授权执行；回调所接入的 runtime 必须先满足这些要求。

第一次回调之前，辅助层验证所有参数及已有依赖、检查所有调用对并分配依赖存储。
每个有冲突的先前调用都贡献 `DEP_WAIT`，包括非相邻调用。不相交和 read/read
共享不增加依赖。已有依赖的 flags（包括 `DEP_RETAIN`）保持有效；匹配到已有边时
补充 `DEP_WAIT`，不会移除 retain。输出分别位于缓存行 A、B、A 的批次只增加
第一项到第三项的顺序约束，中间项仍可独立执行。

辅助层临时替换各 `CoreTaskArgs` 的依赖数组并禁用提前调度。每次回调之后恢复原有
依赖指针、flags 和提前调度设置，异常展开时也恢复。回调不得保留临时指针或修改
批次绑定。提交失败、无效或重复任务 ID 会阻止后续提交；已经提交的任务无法回滚，
因此回调失败不代表批次原子失败。

消费者提交前，所有生产者必须在同一个有效 TMR scope 内保持可引用。绑定必须保持
不变，底层存储必须存活到执行完成。每次调用重新计算冲突，不缓存依赖实际地址的
结果。辅助层只约束本批次，其他可能并发的调用仍需独立的依赖或隔离策略。它不证明
组内安全、资源所有权、栈限制、二进制加载或完成可见性。上述直接 runtime 路径将
该辅助层与 `TMRBridge.lower`、受检查的编译以及真实 TMR 提交组合，用于支持的子集。
显式 JIT 绑定和原生对象自动发现使用相同的受限视图策略。

## 请求与编译结果

`KernelRequest` 包含有序 `ParameterSpec`、编译期常量、`CompileTarget` 和有效选项。
`TensorSpec` 记录 dtype、shape、layout、stride、offset 和对齐。维度可以是非负整数或请求内的符号名。
stride 和 offset 以元素为单位；空 stride 元组表示未约束，不代表连续。`offset=None` 表示运行期绑定。
scalar 必须声明 `DataType`；TaskId 不是设备 scalar 参数。

记录使用冻结 dataclass 和不可变 tuple/frozenset。校验器拒绝嵌套可变值、重复名称和非有限常量。
声明为 tuple 的字段必须使用实际 tuple，包括 ABI 参数、tensor 维度、结果绑定和 include 路径；
不可变集合不能替代有序 tuple。常量和选项条目也必须使用 `(name, value)` tuple；
capability 集合仍使用无序 frozenset。
容器成员和标量元数据也必须符合声明类型。仅 `SourceFile.content` 接受 bytes；
格式错误的元数据会在语义校验或摘要序列化前触发 `ContractError`。
规范化签名不能携带 `SpecializeContext`、AST、Python 源码表达式或本次调用的数据地址。

`ExternalKernelProvider.prepare(kernel, request)` 接受 provider 持有的不透明引用，返回 `PreparedKernel`：

| 字段 | 约定 |
| ---- | ---- |
| `request` | 原始规范化输入，包含目标、工具链和选项 |
| `contract` | 逻辑参数、效果、公开返回、写回和别名许可 |
| `artifact` | 持有的源码/头文件字节、bundle 相对路径、SHA-256、入口、符号及有序 include 路径 |
| `native_abi` | 独立版本化的物理参数表示 |
| `execution` | 核成员、lane、启动要求及完成保证 |
| `identity` | provider/adapter 版本、源码依赖摘要及完整性承诺 |

每个 `SourceFile.content` 持有不可变 `bytes`，生命周期与 bundle 一致。Provider 在返回前对所有
源码/头文件生成字节快照，因此即使 provider 临时目录已删除或缓存租约已释放，
`RuntimeBridge.lower()` 仍可使用 `PreparedKernel`。`ExternalTask.source` 同样使用这一自包含表示
保存 bridge 生成的源码。

`validate_prepared` 检查每份内容是否与其 SHA-256 一致。顶层 `artifact_manifest.json` 名称由现有
artifact store 保留，不得用于 payload 或目录。Bridge 将已校验字节写入私有目录，并在磁盘上检查
路径/符号链接边界。持久化发布、恢复和租约仍复用 `ArtifactStore`；bundle 不是另一套缓存。
文件路径不能是其他文件的祖先路径：同时包含 `include` 和 `include/header.h` 的 bundle 会被拒绝，
与文件枚举顺序无关。
include 目录及其所有祖先路径也不能与 bundle 中的文件重合；include 目录可以包含普通 bundle 头文件。

## 公开返回与写回

每个 tensor 参数必须有且只有一个 `TensorEffect`。每个被写参数必须有一个 `WritebackBinding`，
与 Python 函数是否返回该参数无关。`ReturnBinding` 独立描述每个公开 tensor 返回；允许重复或只读参数返回。
v1 返回与完整的实参 view 共享存储，不一定对应完整底层分配。新分配、设备 scalar 返回以及任意返回 view
表达式不属于 v1 支持范围。

```python
from pypto.jit.external.contracts import (
    Access, Coverage, KernelContract, ParameterSpec, ReturnBinding,
    TensorEffect, TensorSpec, WritebackBinding,
)
from pypto.jit.external.normalization import plan_call
from pypto.pypto_core import DataType

parameters = tuple(
    ParameterSpec(name, TensorSpec((32,), DataType.FP32))
    for name in ("x", "state", "out")
)
contract = KernelContract(
    parameters,
    (
        TensorEffect(0, Access.READ),
        TensorEffect(1, Access.READ_WRITE, Coverage.PARTIAL),
        TensorEffect(2, Access.WRITE, Coverage.FULL),
    ),
    public_returns=(ReturnBinding(2),),
    writebacks=(WritebackBinding(1), WritebackBinding(2)),
)
plan = plan_call(contract)
assert plan.result_parameters == (1, 2)
assert plan.public_slots == (1,)
assert plan.writeback_slots == ((1, 0), (2, 1))
```

对公开调用 `result = stage(x, state, out)`，内部调用返回更新后的 `state` 和 `out`，仅将后者投影为
`result`。JIT 已根据计划改写后续 state 使用，并通过现有 inline/存储分析跨边界传播写入。
运行 pass 前必须将投影落实为 IR；`NormalizeReturnOrder` 后不能继续使用旧的结果序号。

IR 原型复用 `external_return_params` 中对本函数参数的引用，以及 `param_directions` 和返回类型。
该表示可以在参数重建后保持关系，避免按 shape 或输出位置猜测别名。无需新增 IR 节点或 binding。
参见[已有 extern 返回声明](04-external-kernels.md)。

## 部分写入、view 与别名

| 情况 | v1 规则 |
| ---- | ------- |
| 只读或无访问 | 保守映射为 `In`，不因未使用而消除依赖 |
| 仅写且完整覆盖实参 | `Out`，仍接受现有调用方向推导 |
| 读写、部分或未知覆盖 | `InOut`，保留传入依赖 |
| 部分/未知写入且不保证保留未覆盖字节 | 拒绝 |
| 完整写入 view | 不能推导为完整写入根分配 |
| 不同参数重叠，且至少一个被写 | 必须有显式 `AliasPermission` 和 bridge 支持 |
| 没有许可且重叠未知 | 调用前证明或检查不重叠，否则拒绝 |

缺失/未知访问效果会被拒绝；已知写入但覆盖范围未知可以保守处理。别名许可保证 kernel 内部的安全性，
不能仅靠任务调度顺序代替。缓存命中后也必须检查实际别名和对齐。

不能把 `ReturnedParamIndices == None` 当作新分配的证据。完整参数返回分析会在改变 view 时停止；
存储区域分析回答的是另一个问题。必须保留 view 的 shape、stride 和 offset。
零次循环和未执行的分支保留传入存储及依赖。

## 原生 ABI 与执行

`NativeABI` 指定 bridge 理解的布局名称和版本，参数元组按**物理顺序**排列。
每个 `ABIParameter` 包含逻辑参数索引、值种类、字节大小、字节对齐、可选 tensor 轴和元素/字节单位。
scalar 按位复制，不做数值转换。descriptor 内部布局由 ABI 名称/版本定义；名称有效不代表 bridge 支持它。
数据指针指向分配基址；view offset 只能应用一次。
使用 `DATA_POINTER` 的被访问 tensor，其运行期绑定或符号 offset 必须有 `OFFSET` 字段；
符号维度必须有对应轴的 `DIMENSION` 字段，未约束或符号 stride 必须有对应轴的 `STRIDE` 字段。
固定整数约束可在 provider 编译时使用。受支持的 descriptor ABI 在 descriptor 中携带这些信息；
未使用的指针（`Access.NONE`）不需要 view 元数据槽位。

`ExecutionRequirements` 声明 AIC/AIV 成员和 AIV lane 数。独立 AIV 使用一个 lane，MIX 可以使用一个或两个。
全核驻留要求同步启动；全核 MIX 要求两个 AIV lane。逻辑启动上下文必须显式声明，不能用物理核号替代。

v1 仅支持 `ENTRY_RETURN` 完成保证：入口返回时所有工作已完成，写入已可见。
bridge 必须在 runtime 标记任务完成前插入所需等待和 cache 操作。不支持的异步完成方式应明确拒绝。

`validate_target` 检查编译目标、ABI 名称/版本以及语义和执行能力。从实际效果和启动字段推导的要求，
不能通过省略 capability 标签隐藏。它不证明每次调用的别名、维度、对齐和驻留情况；bridge 仍须检查。
runtime target 字段与 provider 记录采用相同的严格类型校验。支持的 ABI 条目必须是 frozenset 中的
`(str, int)` tuple，名称非空且版本为正整数；不能利用 Python 数值相等规则将 bool 或 float 当作
有效 ABI 版本。runtime 名称和版本也必须非空。

## 原始设备镜像入口

硬件 kernel（A2/A3 和 A5 的 AIC、AIV）以原始 `.text` 字节加载。
`kernel_entry` 必须位于非空、可执行 `.text` 段的偏移零处。
链接参数 `-e kernel_entry` 只选择 ELF 入口元数据，不会移动函数。
因此，保留了更早的非内联函数的源码 wrapper 即使链接成功，也可能违反加载器 ABI。

PyPTO 在 `compile_incore` 返回前校验链接后的 AI Core ELF；对于源码旁缓存的 ELF，
也会在提取代码或发布缓存前再次校验。非零入口偏移会报错，并给出入口地址、`.text` 地址
和偏移。应让链接后的入口位于最前，或移除前面的非内联函数；必须检查最终 ELF，不能仅
依赖源码顺序。CPU simulator 共享库通过符号解析 `kernel_entry`，不使用此检查。

普通 extern 校验不会施加 VPTO helper 对其他分配段的禁令，例如 AscendC 的 block-local
存储段。Helper 构建仍执行 VPTO 的附加自包含镜像校验。

可变二进制缓存 context 使用 schema 3：旧 stamp 会使原始代码缓存和 ELF sidecar 均失效，
即使文件摘要匹配也不能复用。已发布的 `BINARY_READY` manifest 使用 schema 2，包含分布式
父级和子级 manifest。旧 prebuilt manifest 会在构造 callable 前被拒绝，需要从源码重新
构建这些产物。仅凭原始代码字节无法证明原 ELF 的入口位置。新生成且通过校验的缓存仍可复用。

## 版本与缓存身份

未知 schema 版本和必需能力触发 `ContractError`。原生 ABI 版本独立于契约和 host runtime ABI。
新增语义必须显式检查能力，不能根据 provider 名称推断。

`preparation_key` 覆盖规范化签名、常量/选项、目标/工具链、provider/adapter 版本以及源码依赖摘要。
命名常量和选项的顺序不影响身份；物理参数顺序和 include 搜索顺序影响身份。
依赖身份不完整时返回 `None`，要求重新 prepare。`prepared_digest` 对已校验结果生成摘要，包含效果、ABI 和执行要求。
下游可执行产物 key 还须包含 bridge 版本、runtime 和有效构建选项。锁、发布和生命周期复用 `ArtifactStore`。

## 验证证据与限制

`test_external_contract.py` 覆盖两个独立 fake provider、返回/写回计划、不可变请求、诊断、ABI 顺序/单位、
目标能力检查、身份失效、provider 目录清理后的源码交接、内容完整性及保留 manifest 路径。
`test_external_ir_semantics.py` 对规范化形式执行往返序列化、默认 pass pipeline 和编排代码生成：
显式公开返回与隐藏状态写回、inline、零次/多次循环、分支、AIC/AIV/MIX，以及 view 写入到根读取的依赖。

这些是 CPU 编译器检查，不能证明设备数值正确性、重叠执行安全或完整 runtime 完成行为。
后续接入任务及定向设备测试负责这些验证。已有 extern API 继续使用兼容路径。

`test_external_registry.py` 覆盖惰性加载、匹配歧义、生命周期、不可变捕获、并发加载和
builder 请求隔离；`test_extern_kernel.py` 中的旧钩子回归执行完整 JIT 编译。

## 调用规范化与签名特化（任务 34.3）

JIT 在解析与 SSA 转换前，将外部调用的公开返回值和隐藏 tensor 写回分别投影到
现有 IR。公开返回为 void 的调用仍捕获内部更新；inline 和存储分析传播共享存储的
写入，同时保留 helper 局部名称重绑定的语义。循环与分支中的调用留在原来的执行位置，
零次循环和未执行分支不会调用 kernel。视图参数仅求值一次。

内部 `SpecializeContext.external_contract` 接收已绑定 extern 声明的已准备契约。
契约的公开返回参数映射必须与声明一致：调用方元数据在此交接之前已推导，
不一致的映射会被拒绝，避免后续函数使用过期的结果元数据。
既有 extern 从经验证的参数方向和返回别名生成同样的调用计划。部分或未知覆盖写入
使用 `InOut`，保留输入依赖；公开结果顺序、重复别名、只读别名及 Python tuple
返回注解保持原意，原生 ABI 参数顺序不变。所有投影在 pass 执行前成为 IR，不能在
`NormalizeReturnOrder` 之后继续使用原始结果槽位编号。
写回投影保留实际参数的描述符类型，包括通过局部别名读取的动态维度。
仅有 InOut 参数且使用裸 `-> pl.Tensor` 注解的旧式 stub，以第一个 InOut 缓冲区
作为公开结果；无返回注解的 void stub 仍只产生隐藏写回。

外部调用须为独立语句，或赋值/return 的完整右值。嵌套表达式、条件、循环迭代器和
上下文管理器头部中的外部调用会报明确错误，应先将调用结果绑定到局部变量。
对 provider 契约，静态可识别的同根别名和视图在缺少 `AliasPermission` 时会被拒绝，
但同一父视图坐标系中具有常量区间且可证明互不重叠的一维切片可以通过。未知坐标、
启用 clamp 的切片、不同父视图以及 reshape 改变的坐标系仍保守处理。
这包括 tuple/list 解包产生的别名、不可变 tuple 绑定的常量索引投影,
以及使用关键字传入源张量的视图。
重绑定更新别名来源；不确定的控制流合流丢弃静态证明。该检查不能证明不同输入缓冲区
或跨 helper 的参数互不重叠，运行时 bridge 仍须在每次调用（包括缓存命中）检查这些
情况。既有 extern 保留原有别名策略。

含 extern 的调用图按 constexpr 值、tensor shape/dtype/layout 和 scalar dtype
特化依赖及普通 helper。等价签名复用生成函数，不同签名生成不同函数。动态维度按
相等关系规范化，不依赖符号拼写或采样边界；缩短视图的独立运行时维度保持独立。
省略布局与显式 `ND` 保持区分：前者可继承下游声明，后者优先于下游声明。
局部分配保留已知布局，包括显式或默认的 `ND`。
裸 `pl.Scalar` helper/extern 参数从调用点继承类型，显式 scalar 注解保留声明类型。
字面量、显式类型的 `pl.const`、算术、cast、维度读取及局部 scalar 参数遵循 parser 的 scalar 类型提升规则。
分支合流仅在两条路径类型一致时保留 scalar 类型。
传入采样参数时也会携带声明的 scalar dtype；scalar 值不会拆分缓存。
缺失或不支持的外部参数注解会在诊断中指出函数和参数名称。
函数和调用索引限定在单次请求内，每个调用者变体执行一次元数据遍历，支持直接传入
组合的 slice/reshape 表达式，避免为每个调用点重新遍历前缀。
inline helper 直接返回 extern 调用时,保留与先绑定局部变量再返回相同的结果元数据。
直接视图参数的返回别名保留元数据，可继续传给后续 JIT 函数。同一未改变的运行时
长度用于多个切片时共享 extent 符号；scalar 重绑定和不确定的控制流合流使该相等关系失效。

`test_external_normalization.py` 从 JIT 源码验证嵌套 inline、控制流、部分写入、
视图和重绑定的 IR/代码生成，以及签名隔离、嵌套调用和重叠别名诊断。这些是 CPU
编译器验证，不代表设备数值正确性或安全重叠执行。任意注册对象与可执行 wrapper 的
连接、原生 ABI 和运行时检查仍由后续任务 34.4/34.5 完成。

## TMR c_pointer 桥接（任务 34.4）

`pypto.backend.tmr_bridge.TMRBridge` 将 `c_pointer` ABI v1 转为 A5/CCEC 的 TMR
任务入口。目标使用 `RuntimeTarget(runtime="tmr", version="1", ...)`；版本是内部
runtime 接口版本，不是 runtime 软件包版本。Provider 提供可包含的、返回 `void` 的
设备函数，不能提供 host launch stub；桥接器不导入可选 DSL。

原生参数可按任意物理顺序排列；TMR 仍先传 tensor、再传 scalar。Scalar 按位传递，
C++ 类型须匹配声明。维度、stride 和 offset 使用 1/2/4/8 字节无符号整数，并按
声明转换元素/字节单位。数据指针指向分配基址，原生函数恰好应用一次显式传入或
编译期固定的 offset。Tensor 对齐约束作用于该分配基址，与 offset 无关；
不要求计算后的视图地址满足相同对齐。编译检查参数数量、类型、大小和对齐。

`lower()` 返回拥有源码字节的任务及独立 host 检查器。
`with materialize(task) as native:` 建立私有源码租约；
`native.configure_context(context)` 校验请求，并给既有 extern 特化附加源码、include、
契约及检查器。保持租约直到既有 runtime 完成源码编译/发布。AIC/AIV、mixed Group
和向量 lane 数来自执行契约。注册对象接入仍属任务 34.5，持久化
复用属任务 34.6。CANNBotDSL 支持上文的直接 runtime、自动 JIT 发现及显式绑定路径。

每次调用都在实际参数打包后、提交前检查 rank/dtype、固定及共享维度、正 stride、
offset、对齐、存储/地址边界和原生元数据位宽，支持普通调用、显式 submit 及参数重排的
SPMD/Group。写入参数未获许可的重叠按保守字节区间拒绝；支持不相交的兄弟视图及
显式别名许可。产物缓存命中仍保留检查。调用原生函数前先关闭 atomic 模式，再保存
CTRL，确保恢复控制状态不会重新启用继承的 atomic 模式；并在 AIC
上清除继承的 HF32 模式；AIV 保留该控制位。设备入口返回前排空流水线、回写缓存并执行
DDR 屏障，然后恢复控制状态并关闭 atomic 模式。

此 c_pointer 路径拒绝 descriptor ABI；专用 runtime 路径处理支持的 CANNBotDSL 子集。
其他明确拒绝的情况包括非 ND 布局、1–5 之外的 rank、FP16/BF16 scalar、其他
架构/runtime、逻辑 launch context、同步/全驻留启动、runtime 新分配的输出槽及
录制 Graph 的重放。输出须在提交前已有存储。CPU 测试用固定 runtime 头文件编译并
执行参数转换和检查；mock 完成指令仅验证顺序，不证明设备可见性，设备验收仍属任务 34.7。
