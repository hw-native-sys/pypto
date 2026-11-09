# 外部 Kernel 契约（内部 v1）

本文是任务 34.1–34.3 和 [RFC #3293](https://github.com/hw-native-sys/pypto/issues/3293) 的接口基线。
范围是数据交接与可执行 IR 原型，不是公开 provider 注册 API，也不代表已经接入 CANNBotDSL runtime。

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
