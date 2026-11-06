# 标量 SIMT IR

规范 SIMT IR 用一个无返回值的设备辅助函数和显式 launch 语句表达执行域，
底层原语贴近 PTOAS。内联 `simt_scope` 语法及 outlining 属于独立的上层实现。

## 函数与 launch

`FunctionType.SIMT` 追加到函数类型枚举末尾，序列化值为 9。
SIMT helper 没有运行时任务 level、role、binding requirement 或返回值。
`is_incore_type(SIMT)` 为 false，因为 helper 不作为运行时任务提交；
`is_device_function_type(SIMT)` 为 true。

```python
launch = ir.SimtLaunchStmt(
    ir.GlobalVar("helper"),
    [ir.ConstInt(n, DataType.INT32, span) for n in (64, 1, 1)],
    [src, dst],
    [updated_dst],
    [1],
    span,
)
```

`block_dims` 必须包含三个正整型常量；布尔值、动态值及超过 INT32_MAX 的乘积
在构造时被拒绝。`args` 是使用点，`results` 是定义点。
递增的 `result_param_indices` 将每个输出恰好映射到一个 InOut 参数；
输出保留相应实参的完整类型和存储元数据。

该语句支持 visitor/mutator、结构相等与哈希、MessagePack 序列化、深拷贝、
定义前使用检查及 SSA 转换。Python printer 输出如下规范语法：

```python
updated_dst = pl.simt.launch(
    self.helper, src, dst, block_dims=(64, 1, 1), result_param_indices=(1,)
)
```

文本 parser 可以读取该形式。这是 IR 往返格式；后续上层界面通过 outlining
推导输出映射。

## 标量操作

| 操作 | 结果 | 契约 |
| ---- | ---- | ---- |
| `simt.thread_idx`、`block_idx`、`block_dim`、`grid_dim` | INT32 | 必须显式传入 typed `SimtAxis.X/Y/Z` kwarg |
| `simt.lane_id` | INT32 | 无 axis |
| `simt.load(storage, indices)` | 元素标量 | Vec/UB Tile 或 GM Tensor |
| `simt.store(storage, indices, value)` | VoidType | 标量类型与存储元素类型一致 |

Python IR wrapper 位于 `pypto.ir.op.simt`。索引规范化为 `MakeTuple`，每个逻辑
维度对应一个整数或 index 标量。store 放入 `EvalStmt`，不能作为赋值右值。
本阶段不支持短向量及不足一个字节的存储元素。

Tensor 表达直接 GM 访问，没有隐式 Tile 搬运。ND/DN 视图沿用已有 stride 和
偏移元数据，不支持分形布局。显式非 Vec 的 Tile 被拒绝。
GM 接受 PTOAS 的 `l1cache="cache"/"uncache"` 以及分别验证的 load/store
`l2cache` 策略集合；UB 不接受 cache 属性。

## 效果与验证

`analyze_simt_access(helper)` 按参数返回物理读写摘要。
store 的物理效果是 `Write`；launch 的可写参数在逻辑上是 `InOut`，
因为未写元素保留旧内容。launch 输出是更新后的存储版本，不是标量返回值或任务句柄。

`SimtValid` 结构验证器检查 helper 签名、操作域、实参类型、可写映射及重复的
可写实参。helper 接受标量及借用的 Tile/Tensor，禁止分配、重绑存储句柄，
也不能通过控制流结果携带存储句柄。函数体只允许标量控制流及 SIMT builtin。
禁止 Out 参数、普通 Call 或运行时 Submit 调用 helper，以及嵌套 launch。
调用者必须是 AIV 或展开前的 InCore。VMI 寄存器不是可接受的存储操作数；
helper 借用 UB 或 GM 存储。
活动 VMI 寄存器作用域内的 launch 也会被拒绝。混合核展开将 SIMT launch 放在
AIV 上，避免在不同核上重复执行。

## 编译器准备

默认流水线现在会在内存规划前保留存储语义：

- [MaterializeSimtOutputs](../passes/35-materialize_simt_outputs.md) 为每个
  可写 Tile 初始化独立副本。部分写入保留未触及元素，旧 Tile 值在分支和循环中
  仍可读取。
- [SpecializeSimtHelpers](../passes/36-specialize_simt_helpers.md) 根据完整块
  维度、类型及布局和目标创建变体。动态元数据成为标量实参，分配地址不进入特化键。
- GM 结果保留原 Tensor 标识。`simt.store` 使用独立于标量控制器和 DMA 存储的
  `Simt` 写通道。核函数参数方向以 InOut 传播到外层 Call 和 Submit，再由普通
  任务依赖分析处理。即使后续读取使用旧 GM 句柄，设备语句调度也保留 launch 顺序。
- 混合核拆分将 launch 保留在 AIV，并在清理前保留 AIC 借用的 GM 句柄。
  Tensor 视图保留形状、步长和布局，包括通过循环和分支传递的视图与别名。
  仅处理 GM 句柄不会产生额外的 Cube 核函数。

复制前会拒绝已知的可写实参重叠。不同 GM 实参在运行时仍可能别名，调用者负责
保证不重叠。辅助函数参数借用存储，不独立分配。自动选择 Buffer IR 时，SIMT
核函数仍保留在 Functional IR。

## 自动资源推导与实现边界

`infer_simt_resources(launch)` 经 `PassContext` 查询目标 `BackendHandler`。
Ascend950 的三个维度乘积不得超过 2,048，寄存器数量按以下规则推导：

| 线程数 | 最大寄存器数 |
| ------ | ------------ |
| 1–256 | 128 |
| 257–512 | 64 |
| 513–1,024 | 32 |
| 1,025–2,048 | 16 |

用户不提供 `simt_max_threads` 或 `simt_max_regs`。其他后端明确拒绝 SIMT 资源推导。

规范 IR、验证、存储准备、内存规划集成、资源特化及 A5 可执行子集已实现。
[SIMT lowering 与执行](../codegen/05-simt-lowering.md)说明 ABI、同步、支持类型
和验收测试。独立的[原生 SIMT 探针](../codegen/04-simt-native-validation.md)
保留此前 P0 ABI 测试。更高层的编写入口及 outlining 仍属于独立工作。
