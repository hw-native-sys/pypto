# 外部 Kernel 契约（内部 v1）

本文是任务 34.1 和 [RFC #3293](https://github.com/hw-native-sys/pypto/issues/3293) 的接口基线。
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
34.2 实现注册与绑定，34.3 实现全程序规范化，34.4 实现原生 wrapper 和运行期绑定检查。

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
`result`。34.3 还需要替换后续 state 使用，并跨 inline 边界传播效果。计划本身不改写 Python 源码。
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

这些是 CPU 编译器检查，不能证明设备数值正确性、自动源码写回传播、重叠执行安全或完整 runtime 完成行为。
后续接入任务及定向设备测试负责这些验证。已有 extern API 继续使用兼容路径。
