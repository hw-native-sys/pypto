# SpecializeSimtHelpers

根据实际 launch 维度、存储元数据和编译目标创建 SIMT 辅助函数变体。辅助函数仍是
无返回值的设备函数，不会变成运行时任务。

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::SpecializeSimtHelpers()` | `passes.specialize_simt_helpers()` | 程序 |

**前置条件：** SSAForm。
**产生：** SimtSpecialized。

## 特化键与元数据

特化键 (specialization key) 包含原辅助函数、完整的三个块维度、规范化参数类型及
布局，以及目标架构。仅线程数乘积相等不足以共享：`(64, 1, 1)` 和 `(8, 8, 1)`
会产生不同变体。辅助函数类型会移除具体分配标识和地址；运行时标量值和循环迭代
次数不进入特化键。

动态形状、步长、有效形状和偏移表达式在需要时成为显式标量实参。若已有标量参数
提供该表达式，则复用该参数。使用不同运行时步长变量的调用者可以共享同一变体。
辅助函数体不能保留未解析的捕获。编译器使用现有 Tile/Tensor 类型和标量参数，
无需新增指针或视图 ABI。

变体带有编译器所有的函数属性 `simt_block_dims`、`simt_target`、
`simt_max_threads` 和 `simt_max_regs`。资源根据目标处理器的
`infer_simt_resources` 规则推导，用户不指定寄存器或线程限制。
`SimtSpecialized` 对照每个 launch 验证这些属性。序列化和 Python 打印保留它们。

等价变体通过有界结构签名索引共享辅助函数。重复运行此 pass 保留已有变体，不会
重复添加元数据实参。原始辅助函数定义仍保留在程序中。

## 执行顺序

默认流水线在 `MaterializeSimtOutputs` 后、`InitMemRef` 前运行此 pass。
更早的 `FlattenTileNdTo2D` 同时改写调用者的 Tile 类型与辅助函数的标量索引。
其初始 SIMT 支持要求隐式紧凑 Tile 布局和静态正数前导维度。GM Tensor 秩保持
不变。后续内存规划仅分配调用者存储。

资源表及编译器准备与 PTOAS/设备执行的边界见[标量 SIMT IR](../ir/11-simt.md)。
