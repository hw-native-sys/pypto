# 显式 PTO 目标 IR

> **状态：** `MaterializePTOTileHandles` 与 `LowerTileToPTOIR` 已加入
> `Default` 和 `DebugTileOptimization` 流水线末尾。受支持的函数由
> `PTOIRPrinter` 输出。包含尚未覆盖 Tile 操作族的函数会被标记为
> `pto.target_lowering_deferred`，并整体保留在旧路径；不会出现只下沉半个函数的
> 混合 IR。这是用于确认完善程度的默认启用，当前仍未达到生产可用水平。

## 范围

本设计只适用于 PTO 核内（InCore）代码生成。编排（Orchestration）C++ 代码生成
具有不同的对象身份和生命周期要求，不会下沉到这套目标 IR。

当前目标是将以下决策移出 PTO codegen：

- 每个逻辑 Tile 结果写入哪个 PTO tile buffer；
- 每个 `pto.alloc_tile` handle 定义在哪里；
- 每个目标操作的输入和输出 handle；
- 每次传输使用的 Tensor 分区 offset 和 extent。

printer 应机械地消费这些决策，不能从 `AssignStmt`、`TileType::memref_`、
planner 专用 alias 映射或当前输出的 SSA 名称中恢复它们。

## 值模型

实验路径不会替换 SSA，也不会引入一套独立的非 SSA IR。它把逻辑值与可变目标
存储区分开：

| 概念 | IR 表示 | 语义 |
| ---- | ------- | ---- |
| 逻辑 Tile 值 | `TileType` | 目标下沉前使用的值语义 Tile IR |
| 目标 buffer handle | `PTOTileBufType` | 命名可变 PTO tile buffer 的 SSA handle |
| 物理存储元数据 | `pto.alloc_tile` operand | 可选地址和有效行/列范围 |
| 目标操作 | `EvalStmt` 中的 `pto.*` call | 按 operand schema 读写显式 handle |

例如：

```text
%lhs = pto.alloc_tile(...)
%rhs = pto.alloc_tile(...)
%out = pto.alloc_tile(...)
pto.tadd(%lhs, %rhs, %out)
```

`%lhs`、`%rhs` 和 `%out` 各自只有一个 SSA 定义。`pto.tadd` 没有返回值，因为它
通过 `%out` 写入；注册的 operand role 和 memory effect 把 `%lhs/%rhs` 声明为
读，把 `%out` 声明为写。因此，可变 buffer 语义不要求破坏 handle SSA。

`PTOTileBufType` 是内部目标下沉状态，不是面向用户的 DSL 类型，也不能与
`TileType` 互换。

## 分阶段下沉

已实现路径由两个 IR Pass 和一个目标 printer 组成：

```text
Logical Tile SSA
  |
  | MaterializePTOTileHandles (Step 3)
  v
Logical Tile SSA + explicit PTO handle plan
  |
  | LowerTileToPTOIR (Step 4)
  v
Destination-passing PTO target IR
  |
  | PTOIRPrinter
  v
PTO-ISA MLIR
```

### Step 3：物化 PTO Tile Handle

`MaterializePTOTileHandles` 保留逻辑 Tile 程序，同时添加经过验证的逻辑值到
buffer 规划：

- 每个受支持的 Tile producer 都有一个支配它的 `pto.alloc_tile` 定义；
- 每个逻辑 Tile operand 映射到有序输入 handle 列表；
- 每个逻辑 Tile 结果映射到一个输出 handle；
- 过渡 call attr 携带输入/输出映射；
- `PTOHandlesMaterialized` 验证支配、唯一性、精确映射，以及结构化
  `ForStmt`/`IfStmt` 的 region 映射；
- `tile.slice` 产生显式、非分配型的 `pto.subview` handle；
- loop carry 和分支 phi 使用显式共享 handle；仅在存储并不共享时插入 `pto.tmov`。

分配 operand 取决于选择的内存规划器（memory planner）：

| Planner | `pto.alloc_tile` operand |
| ------- | ------------------------ |
| `PYPTO` | `(byte_offset: i64, valid_row: index, valid_col: index)` |
| `PTOAS` | `(valid_row: index, valid_col: index)` |

这些 attr 是临时下沉状态，由 Step 4 消费并删除；`PTOIRPrinter` 不会读取它们。

### Step 4：下沉逻辑 Tile 操作

`LowerTileToPTOIR` 将支持的逻辑操作改写为目标传递（destination-passing）call，
并删除对应的 `TileType` 定义。

改写前：

```text
a = tile.load(input_a, [0, 0])
b = tile.load(input_b, [0, 0])
root = tile.sqrt(a)
added = tile.add(root, b)
result = tile.mul(added, b)
output_result = tile.store(result, [0, 0], output)
return output_result
```

改写后：

```text
a_buf = pto.alloc_tile(...)
pto.tload(input_a, [0, 0], [16, 16], a_buf)

b_buf = pto.alloc_tile(...)
pto.tload(input_b, [0, 0], [16, 16], b_buf)

root_buf = pto.alloc_tile(...)
pto.tsqrt(a_buf, root_buf)

added_buf = pto.alloc_tile(...)
pto.tadd(root_buf, b_buf, added_buf)

result_buf = pto.alloc_tile(...)
pto.tmul(added_buf, b_buf, result_buf)

pto.tstore(result_buf, [0, 0], [16, 16], output)
output_result = output
return output_result
```

最后的 Tensor alias 保留逻辑 `tile.store` 的 Tensor 结果，但不会重新引入 Tile
结果。传输 offset 和 valid extent 是普通 IR operand，不由 printer 重建。

## 目标操作契约

每个目标 op 都注册了 `PTOOpSpec`，用于描述 operand role、memory effect、类型约束
和结果放置：

| 操作 | 输入 operand | 元数据 | 输出 | IR 结果 |
| ---- | ------------ | ------ | ---- | ------- |
| `pto.alloc_tile` | 无 | 可选地址、有效行/列 | 分配的 handle | `PTOTileBufType` |
| `pto.tload` | Tensor | offset、extent | tile buffer | 无 |
| `pto.tstore` | tile buffer | offset、extent | Tensor | 无 |
| `pto.tsqrt` | 一个 tile buffer | 无 | tile buffer | 无 |
| 一元/二元/tile-scalar `pto.t*` | 一到两个 tile buffer，可带 scalar | 无 | tile buffer | 无 |
| `pto.trowsum` | 输入与临时 tile buffer | 无 | tile buffer | 无 |
| `pto.subview` | 源 tile buffer | shape、offset、valid shape | view handle | `PTOTileBufType` |

分配 op 必须作为 `AssignStmt` 的 value。目标传递 op 必须具有 `UnknownType`，并直接
出现在 `EvalStmt` 中。

## 验证契约

两个 IR 属性定义阶段边界：

| 属性 | 必须满足的不变量 |
| ---- | ---------------- |
| `PTOHandlesMaterialized` | 逻辑 Tile IR 仍存在，但每个受支持 producer/operand 都有精确且满足支配关系的 handle 映射 |
| `PTOBufferized` | InCore 目标 IR 不含逻辑 Tile 值，所有目标 buffer 使用都显式且满足支配关系 |

`PTOBufferized` verifier 检查：

- 不存在逻辑 Tile 参数、赋值、返回值或非 PTO Tile call；
- 每个 tile-buffer handle 都由已注册的 handle op（`pto.alloc_tile`/
  `pto.subview`）或函数参数定义；
- 每次 handle 使用都由其定义支配；
- 目标结果位置和 operand 类型符合 `PTOOpSpec`；
- `tload`/`tstore` 的 offset/extent rank 非空、相等，并与 Tensor rank 一致；
- handle 支配关系会递归检查 `ForStmt` 和 `IfStmt`；
- PTO 目标 call 不会逃逸到 Orchestration 函数。

### 默认流水线接入

Pass property 契约已与 planner 解耦。两个目标下沉 Pass 都不再无条件要求
`AllocatedMemoryAddr`，因为 PTOAS planner 会跳过 `AllocateMemoryAddr` 并生成无地址
allocation。Step 3 只在当前 planner 为 PYPTO 时检查 MemRef 是否含常量 byte offset。

`PTOHandlesMaterialized` 和 `PTOBufferized` 已加入默认验证属性集合，正常流水线会验证
两个过渡边界。Step 4 把信息转移到显式目标 operand 后，会使逻辑 Tile/MemRef 属性
失效。

verifier 允许 `PTOTileBufType` 函数参数作为支配性的 handle 定义，但当前实验
printer 只接受 Tensor 和 Scalar 参数。Step 3 不会创建 buffer 参数，因此已实现
范围内部一致；在允许 buffer handle 跨函数边界前，必须对齐 verifier/printer 契约。

## Printer 边界

`PTOCodegen::Generate` 会逐函数选择 printer。包含目标传递 PTO call 的函数交给
`PTOIRPrinter`；带 `pto.target_lowering_deferred` 标记的函数整体保留在旧逻辑 Tile
路径。因此，多函数 Program 可以同时包含完整下沉函数和 deferred 函数，而不会把
deferred 函数错误地交给目标 printer。

对目标 IR，`PTOIRPrinter` 只能执行格式化和局部展开：

| Printer 职责 | 必须已在 IR 中显式表达 |
| ------------ | ---------------------- |
| MLIR 名称和类型拼写 | allocation 位置和可选地址 |
| 常量和 Tensor view 格式化 | 输入/输出 buffer 身份 |
| 根据传输 operand 输出 `partition_view` 语法 | offset 和 valid extent |
| 根据 `PTOOpSpec` 输出 `ins(...)` / `outs(...)` | alias 和 destination 决策 |

它不能读取 `TileType`、从外层 assignment 恢复 destination，或维护 planner 专用的
结果 buffer 推断状态。

## 支持范围

当前实现支持：

- 静态 shape 的 rank-N Tensor view/传输和物理 rank-2 Tile buffer；
- 静态或动态 rank-2 valid shape；
- `load`/`store`、`create`/`full`、`slice`、带 allocation 的 `reshape`、基本
  一元/二元/tile-scalar 运算、move/fillpad 和 row sum；
- 显式 SPMD block/subblock 参数；
- 结构化 `ForStmt`/`IfStmt`，包括 scalar carry/result 和 Tile carry/phi handle；
- 结构化 region 内的 Tensor scalar read/write；
- PYPTO 显式地址和 PTOAS 无地址 allocation。

在 PTOAS 下，分支局部 producer 直接写共享 phi/carry handle；pass-through 值和只含
元数据的 view 在确实需要拷贝时使用显式 `pto.tmov`。PYPTO 保留基于地址的 alias
plan。`pto.subview` 的静态 valid extent 会逐维写入结果 `PTOTileBufType`。

跨核 pipe、动态 shape Tensor 参数、`tile.sum` 等一般归约、`WhileStmt` 和其他 Tile
操作族尚未进入目标 IR。这类函数会在插入任何 handle 前标记为
`pto.target_lowering_deferred`，并整体走旧路径。

## 完善计划

### Phase A：默认诊断启用

已完成：

1. 将 property 契约与 planner 解耦，同时保留 PYPTO 对常量地址的条件检查；
2. 在两套受维护 Tile 流水线末尾运行 Step 3 和 Step 4；
3. 在正常流水线中验证两个过渡属性；
4. 将目标 IR 交给 `PTOIRPrinter`，并分别用 PYPTO level-3 和 PTOAS level-2 模式
   通过真实 `ptoas` 编译；
5. 在修改前把不受支持的 InCore 函数整体标记为 deferred，不产生逻辑/目标混合 IR。

### Phase B：直线操作覆盖

只有在定义 operand role、memory effect、结果放置、lowering、verifier 规则、shadow
comparison 和 `ptoas` 覆盖后，才能增加新的操作族。

### Phase C：控制流与别名

`IfStmt` 和 `ForStmt` 已具有显式 Step-3 handle plan：scalar 保持 SSA `scf`
result/carry，Tensor 保持引用 alias，Tile 使用共享 handle 或显式 `pto.tmov`。
`WhileStmt` 及更一般的 region/block 传递仍需实现，并在两个 planner 下验证。

### Phase D：生产就绪门槛

保持默认启用以持续发现覆盖缺口，但在代表性真实 kernel 和系统测试通过、shadow
差异均可解释、且旧 `PTOCodegen` 中对应的 destination/handle 恢复代码可以删除前，
不能把该路径视为生产就绪。

## 验证证据

已实现的静态范围通过了：

- 新旧 PTO 操作的有序比较；
- `PTOBufferized` 正向和负向 verifier 测试；
- 二进制序列化往返；
- `ptoas` 接受 PYPTO/level-3 MLIR；
- `ptoas` 接受 PTOAS/level-2 MLIR；
- 静态 elementwise kernel 从默认流水线分别以 PYPTO 和 PTOAS 模式通过真实
  `ptoas` 编译；
- 聚焦的 Step 3/Step 4 与 PassManager 测试；
- 默认启用后的 codegen 基线，并以显式的整函数 deferred 边界隔离不受支持的范围。

初次默认启用时，`tests/ut/codegen/test_pto_codegen.py` 通过 61/86。补齐 rank-N
传输、动态 valid shape、view/slice、基本操作族、SPMD 参数和结构化
`ForStmt`/`IfStmt` 后，同一隔离 worktree 测试达到 86/86。这只是单元测试覆盖，
不是生产就绪结论。进一步修复 PTOAS 控制流输出共享、逐维 subview valid 类型、
scalar `min`/`max` 以及私有 SPMD 参数边界后，完整 `tests/ut/codegen/` 达到
573/573，聚焦的 target op/pass/property/verifier 测试达到 82/82。跨核目标 IR、
`WhileStmt`、更多操作族及代表性系统/PTOAS 验证仍未完成。

## 实现位置

| 组件 | 位置 |
| ---- | ---- |
| 目标 buffer 类型和过渡 attr | `include/pypto/ir/type.h`、`include/pypto/ir/expr.h` |
| 目标 op schema | `src/ir/op/pto_ops/` |
| Step 3 | `src/ir/transforms/materialize_pto_tile_handles_pass.cpp` |
| Step 4 | `src/ir/transforms/lower_tile_to_pto_ir_pass.cpp` |
| Property verifier | `src/ir/verifier/verify_pto_ir.cpp` |
| 机械式 printer | `src/codegen/pto/pto_ir_printer.cpp` |
| 新旧路径分发 | `src/codegen/pto/pto_codegen.cpp` |

另见 [PTO 代码生成](00-pto_codegen.md)、
[MaterializePTOTileHandles](../passes/43-materialize_pto_tile_handles.md)、
[LowerTileToPTOIR](../passes/44-lower_tile_to_pto_ir.md)、
[#1956](https://github.com/hw-native-sys/pypto/issues/1956) 和
[#2032](https://github.com/hw-native-sys/pypto/issues/2032)。
