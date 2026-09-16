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
如果当前 PyTorch 缺少 `torch.library.register_fake`，则在安装 schema 前明确报错。

测试使用临时 namespace 和 CPU 实现夹具。在 PyTorch 2.6 上，无 dispatcher 返回值
的纯修改 schema 通过全部
[`torch.library.opcheck`](https://docs.pytorch.org/docs/2.6/library.html#torch.library.opcheck)
检查以及 `torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)`；
测试 wrapper 在调用算子后返回调用方传入的输出 Tensor。通过注册后的 `torch.ops`
验证有具体值提示（backed）和无具体值提示（unbacked）的整数符号原样到达 fake kernel，
且不添加等值 guard；另以 shape 派生的 Scalar 验证不同输入尺寸复用同一编译图。
带返回别名的 schema 分别
验证 schema 正确性与 Fake/Meta 行为，不把这些检查视为该类算子已支持函数化
（functionalization）或编译执行。正式 kernel 注册、设备执行、autograd 与最终编译器
接线仍属于后续工作。

## 可选依赖与范围

`import pypto.torch` 不导出执行 API。导入该包、`interop` 或 `registration` 模块不请求 torch_npu、
Simpler 或 native launch 扩展；`torch` 仍是 PyPTO 的常规依赖。真正描述 NPU 调用时
才按需加载 `torch_npu`，缺失时给出针对性的错误信息。

本基础能力覆盖 metadata 校验、Python 调用 frame 所有权及内部 schema/Fake 辅助，
不表示 kernel launch、
taskQueue 顺序、allocator 安全、eager 数值执行或 ACLGraph 已可用。公开入口切换
必须等待 native adapter 和 runtime 集成完成。

## 验证

`tests/ut/torch/test_interop.py` 使用真实 CPU storage 和模拟的 NPU device 标签，
只替换框架 format/context 查询。覆盖 view offset、alias、独立 Scalar/stream 快照、
所有权、非法输入和禁止导入可选 runtime 依赖的隔离进程。

`tests/st/runtime/kernel/test_torch_interop.py` 核对真实 NPU Tensor、非默认 stream、
offset view 和非连续输入拒绝行为。没有真实 NPU 或所选平台为模拟器时跳过。
这些是 metadata 测试，不是 PyPTO kernel 执行测试。

`tests/ut/torch/test_registration.py` 覆盖 schema mutation/alias 契约、Fake/Meta
与符号输入、隔离导入、重复定义，以及不依赖真实 kernel executor 的测试内
dispatcher/compiler 集成。
