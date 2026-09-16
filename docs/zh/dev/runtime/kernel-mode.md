# Kernel mode 集成基础

内部 torch 适配器校验借用的 NPU 参数，并通过可选 native torch_npu 扩展提交已准备好的
kernel 注册项，不新增公开 kernel 执行入口。已有 JIT、编译 program 和 Worker 调用保持当前行为。

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

`import pypto.torch` 不导出执行 API。导入该包、`interop` 或 `registration` 模块不请求 torch_npu、
Simpler 或 native launch 扩展；`torch` 仍是 PyPTO 的常规依赖。真正描述 NPU 调用时
才按需加载 `torch_npu`，缺失时给出针对性的错误信息。

内部 launch 路径需要可选 native adapter。公开 JIT 入口切换与 torch.ops 注册仍是后续工作。
eager 提交拒绝 graph capture，尚未提供 ACLGraph 生命周期契约。

## 进程 kernel Worker 与注册

内部 `runtime.kernel.context.get_process_kernel_state()` 持有进程唯一的延迟初始化
kernel Worker。所有算子共享此管理器；算子、Scalar 值或 caller stream 变化不会新建
Worker。`KernelConfig` 固定 platform、runtime、device 和 AICPU 线程数，其他常驻资源
暂用 simpler 默认值。配置不兼容时报错，不额外创建 Worker。

集成 SDK 固定为 `b5a0ea0c941576e4e9c409b7be5130a607c4f9dc`。实际 Python 接口为
`simpler.task_interface.ChipWorker.kernel_init`、`kernel_prepare_callable` 和
`finalize`，目标 L2 `Worker(execution_mode="kernel")` 尚未提供。PyPTO 内部 adapter
使用这些已有方法；init/prepare 不接收 caller stream，native context generation 和
callable ID 均由 simpler 分配。调用线程须已绑定框架当前设备。初始化使用已安装的
runtime 二进制并检查能力，不编译业务算子、不分配业务输出。该 pin 的 HBG kernel
初始化不受支持，HBG 二进制编译成功不代表可执行。

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

内部管理器 `close()` 为终止操作，须在初始化线程执行。它停止新增注册及提交，
等待在途 prepare/admission，排空已接纳 eager ticket，然后 finalize Worker。close 失败保留 owner 与
注册记录供初始化线程重试，但 handle 已不可用；成功后清空注册、使 handle 失效，不能
重新初始化。关闭未初始化的管理器不做 native 工作。不增加析构或裸 `atexit` close，
框架退出时序另行接入；这些是内部基础原语，不要求普通算子用户手动管理生命周期。

## 验证

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
公开 JIT 入口与 torch.ops 注册仍由后续 PR 接线。当前 eager 提交明确拒绝 graph capture。

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
不排空 Host 队列；内部 `state.drain()` 或初始化线程上的 `close()` 等待在途提交，最后一个 ticket 可保留到
该边界。close 拒绝新增工作，等待 prepare/admission，排空 ticket 后才 finalize Worker。

同步提交错误及异步 callback 错误通过 framework 和 ticket wait 传播。失败或部分 enqueue 的 ticket
继续持有 Worker、参数及 Storage：caller stream 等待失败不能证明内部 stream 已静止。内部初始化线程上的 close 排空 Host callback，并仅在错误路径执行全设备同步，证明内部 stream
已静止后再 finalize、释放 owner，同时重新抛出原提交错误。quiescence 或 teardown 失败时继续保留
全部 owner 以便重试 close。不会隐式重新初始化；自动框架退出串接仍留给后续工作。

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
dispatcher/compiler 集成。
