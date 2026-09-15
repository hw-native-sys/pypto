# Kernel mode 集成基础

内部 torch 适配器为后续 kernel executor 描述借用的 NPU 参数，不新增公开 kernel
执行入口。已有 JIT、编译 program 和 Worker 调用保持当前行为。

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
的设备异步使用；native 队列所有权和 allocator stream 记录属于独立的 launch 层工作。
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

## 可选依赖与范围

`import pypto.torch` 不导出执行 API。导入该包或 `interop` 模块不请求 torch_npu、
Simpler 或 native launch 扩展；`torch` 仍是 PyPTO 的常规依赖。真正描述 NPU 调用时
才按需加载 `torch_npu`，缺失时给出针对性的错误信息。

本基础能力覆盖 metadata 校验和 Python 调用 frame 所有权，不表示 kernel launch、
taskQueue 顺序、allocator 安全、eager 数值执行或 ACLGraph 已可用。公开入口切换
必须等待 native adapter 和 runtime 集成完成。

## 进程 kernel Worker 与注册

内部 `runtime.kernel.context.get_process_kernel_state()` 持有进程唯一的延迟初始化
kernel Worker。所有算子共享此管理器；算子、Scalar 值或 caller stream 变化不会新建
Worker。`KernelConfig` 固定 platform、runtime、device 和 AICPU 线程数，其他常驻资源
暂用 simpler 默认值。配置不兼容时报错，不额外创建 Worker。

集成 SDK 固定为 `29a1cd405645ab65e8f26c1b1f18c8622e5bf8b9`。实际 Python 接口为
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

内部管理器 `close()` 为终止操作，须在初始化线程执行，且调用方已排空 launch 和图使用。
它停止新增注册、等待在途 prepare，然后 finalize Worker。close 失败保留 owner 与
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
