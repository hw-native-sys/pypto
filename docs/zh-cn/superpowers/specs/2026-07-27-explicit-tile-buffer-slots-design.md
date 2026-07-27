# 显式 Tile Buffer Slot 设计

**Issue：** #2131

## 摘要

PyPTO 将提供一等公民、具有 tuple 风格使用方式的片上 tile buffer 集合。用户可以
创建固定数量的同构物理 slot，通过运行时整数表达式选择其中一个 slot，并把支持的
tile producer 直接绑定到该 slot。L1、L0B 和 L0C 使用彼此独立的 buffer 集合，
因此嵌套循环可以按不同粒度轮换它们，同时阻止 `MemoryReuse` 合并不同 slot。

同一套 IR 同时支持两种内存规划器。PyPTO planner 为整个集合分配一个基地址，
PTOAS planner 则不写入地址，由 `PlanMemory` 完成放置。两者均生成
`pto.alloc_multi_tile`，并通过 `pto.multi_tile_get` 在运行时选择 slot。

## 目标

- 为跨循环迭代的 buffer 集合和 slot 提供稳定身份。
- 支持 `iteration % 2` 这类运行时 slot 选择表达式。
- 支持嵌套 pipeline 中彼此独立的 L1、L0B 和 L0C 轮换。
- 将 load、extract、move、matmul 和 matmul-accumulate 的结果直接写入选中的
  destination slot。
- slot 再次使用时保留正确的 RAW、WAR 和 WAW 依赖。
- 禁止 `MemoryReuse` 合并显式 slot，或把无关 buffer 合并进显式 buffer 集合。
- 同时支持 `MemoryPlanner.PYPTO` 和 `MemoryPlanner.PTOAS`。
- Python API 保持 tuple 风格，但 IR 不使用 `TupleType` 表示 buffer 集合。

## 非目标

- 不允许用户指定物理字节地址；地址放置由所选内存规划器负责。
- 不支持元素类型不同的任意 buffer 集合。
- 首版不为所有 tile producer 增加 `out=`。
- 不替换现有的 `pl.pipeline` 自动 buffering 策略。
- 不新增独立 example 文件；功能说明和示例放入现有文档与测试体系。

## 用户 API

### 创建与选择 buffer

```python
l1_buffers = pl.create_tile_buffers(
    2, [128, 512], pl.BF16, pl.Mem.Mat
)
l0b_buffers = pl.create_tile_buffers(
    2, [128, 128], pl.BF16, pl.Mem.Right
)
l0c_buffers = pl.create_tile_buffers(
    2, [16, 128], pl.FP32, pl.Mem.Acc
)

for stack in pl.range(STACKS):
    l1_slot = l1_buffers[stack % 2]
    b_l1 = pl.load(
        b,
        [stack * K, 0],
        [K, STACK_N],
        target_memory=pl.Mem.Mat,
        out=l1_slot,
    )

    for col in pl.range(0, STACK_N, L0_N):
        sub = col // L0_N
        l0_index = sub % 2
        l0b_slot = l0b_buffers[l0_index]
        l0c_slot = l0c_buffers[l0_index]

        b_l0 = pl.tile.extract(
            b_l1,
            0,
            col,
            [K, L0_N],
            target_memory=pl.Mem.Right,
            out=l0b_slot,
        )
        acc = pl.tile.matmul(q_l0, b_l0, out=l0c_slot)
        out = pl.store(acc, [stack * M, col], out)

        pl.tile.release(l0b_slot)
        pl.tile.release(l0c_slot)

    pl.tile.release(l1_slot)
```

`create_tile_buffers` 返回 Python `TileBufferSet` wrapper。该对象支持
`len(buffers)` 和 `buffers[index]`，但底层表达式类型是
`TileBufferSetType`，而不是 `TupleType`。索引操作返回普通 `Tile` wrapper，
其底层表达式是 `tile.buffer_slot` 调用。

### 支持显式 destination 的操作

首版公开的 `out=` 接口包括：

- `pl.load(..., out=slot)`
- `pl.tile.extract(..., out=slot)`
- `pl.tile.move(..., out=slot)`
- `pl.tile.matmul(..., out=slot)`
- `pl.tile.matmul_acc(..., out=slot)`

未提供 `out=` 的现有调用保持当前签名和行为。Python 层把带 `out=` 的调用降低为
独立的 destination-form IR op：

```text
tile.load_into(..., destination)
tile.extract_into(..., destination)
tile.move_into(..., destination)
tile.matmul_into(..., destination)
tile.matmul_acc_into(..., destination)
```

使用独立 IR op 可以避免改变现有 op 的含义，并让所有 pass 和 codegen 都能明确
识别 destination binding，而不需要事后猜测。

## 类型模型

### `TileBufferSetType`

`TileBufferSetType` 是一等公民 IR 类型，包含以下字段：

- 每个 slot 的静态 shape；
- 元素 dtype；
- tile view/layout；
- 片上 memory space；
- 编译期 slot count；
- 可选的 group `MemRef`，由 PyPTO planner 路径中的 `InitMemRef` 填充。

它表示存储资源，而不是 tile 数据值。普通 tile op 必须拒绝该类型；只有
`tile.buffer_slot` 等 buffer-set op 可以接收它。

本设计不复用 `TupleType`：tuple 元素是彼此独立的 SSA 值，可以具有不同类型和
allocation identity，并且主要用于结构化选择；buffer 集合则表示一个同构 allocation
group，并支持运行时索引。

### 选中 slot 的类型

`tile.buffer_slot(set, index)` 返回集合中单个 slot 对应的 `TileType`。
`InitMemRef` 运行后，该 tile 携带一个 slot `MemRef`。它与集合共享 allocation
base，并具有符号化相对偏移：

```text
index * aligned_slot_size
```

group `MemRef` 的 offset 为零，size 为 `count * aligned_slot_size`。allocator 只
放置 group root；slot 的相对偏移保持符号形式，由 `pto.multi_tile_get` 消费，
不会形成独立 allocation。

destination-form op 的结果复用 destination slot 的 `MemRef`。这样既能在后续使用中
保留 allocation lineage，又能让 codegen 将 producer 的 `outs` operand 绑定到
选中的 slot handle。

## IR 操作

```text
tile.create_buffer_set(shape)
    attrs: dtype, target_memory, count, layout options
    result: TileBufferSetType

tile.buffer_slot(buffer_set, index)
    result: per-slot TileType

tile.release(slot)
    result: none; lifetime marker consumed before PTO codegen
```

创建 op 映射到 `pto.alloc_multi_tile`，slot 选择映射到
`pto.multi_tile_get`。`tile.release` 是前端生命周期元数据，没有对应的 PTO 指令。

## 生命周期与依赖语义

对集合执行索引操作会取得一次 slot lease。lease 在以下两个位置中较早的位置结束：

1. 显式的 `tile.release(slot)` marker；
2. 未提供 release marker 时的最后一次 SSA 使用。

lease 结束前，该 slot 以及所有绑定到它的 destination-form 结果的读操作都必须完成。
显式 release 后继续使用属于非法行为。再次取得同一物理 slot 会创建新的 lease；
依赖分析必须保留来自上一个 lease 的 WAR 或 WAW 边。

显式 release 是可选的。它可以缩短编译器保守推断出的生命周期，但不能删除仍在使用
该 slot 的操作所产生的真实依赖。

选中 slot 的依赖 key 为：

```text
(buffer-set allocation identity, normalized slot index)
```

只有当编译器能够证明两个规范化 slot index 不同时，才认为两次访问相互独立；如果
无法证明，则保守地认为它们可能 alias。

## Pipeline Lowering 与调度

显式 buffer op 参与现有 pipeline lowering，不替换 `pl.pipeline`。

对于两级 inner pipeline：

```python
for col in pl.pipeline(0, 512, 128, stage=2):
    index = (col // 128) % 2
    b = l0b_buffers[index]
    c = l0c_buffers[index]
    b_value = pl.tile.extract(..., out=b)
    acc = pl.tile.matmul(q, b_value, out=c)
    pl.store(acc, ...)
```

`LowerPipelineLoops` 为两个 stage 复制循环体，同时保留 buffer-set identity。标量
化简把两个 clone 中的 index 规范化为不同 slot。随后 `CanonicalizeIOOrder` 可以
生成以下顺序：

```text
TEXTRACT(..., out=B0)
TEXTRACT(..., out=B1)
TMATMUL(..., out=C0)
TMATMUL(..., out=C1)
TSTORE(C0)
TSTORE(C1)
```

下一次复用 `B0` 或 `C0` 时，真实的循环携带依赖仍然存在。嵌套 L1 和 L0 pipeline
使用不同 buffer-set identity，因此其 modulo 表达式和轮换 cadence 相互独立。

## 内存规划器行为

### PyPTO planner

`InitMemRef` 为每个 `TileBufferSetType` 创建一个 allocation root，其 size 为对齐后的
单 slot footprint 乘以 `count`。`MemoryReuse` 将该 root 视为显式、不可合并的
allocation。`AllocateMemoryAddr` 为整个 group 分配一个基地址。

PTO codegen 生成：

```text
%buffers = pto.alloc_multi_tile addr = %group_addr ...
%slot = pto.multi_tile_get %buffers[%index] ...
```

不得为各个 slot 额外生成 `pto.alloc_tile`。

### PTOAS planner

同一套 IR 在没有 PyPTO 分配地址的情况下到达 codegen：

```text
%buffers = pto.alloc_multi_tile ...
%slot = pto.multi_tile_get %buffers[%index] ...
```

group placement 由 PTOAS `PlanMemory` 负责。两种 planner 的 destination 和
lifetime 语义完全一致。

## 校验与错误

- `count` 必须处于 `[2, 16]`，与 `pto.multi_tile_buf` 一致。
- 单 slot shape 必须非空、静态且所有维度为正数。
- memory space 只能是 `Vec`、`Mat`、`Left`、`Right`、`Acc` 或 `Bias`。
- 常量 index 超出 `[0, count)` 时在编译期报错。
- 动态 index 必须是整数 dtype。运行时范围由用户保证，标准写法是 `% count`。
- destination 与 slot 的 shape、dtype、memory space、valid shape 和 layout 必须与
  对应 producer 兼容。
- buffer set 不能直接作为普通 tile op 的输入。
- 显式 release 后不能继续使用 lease。
- 除非正常依赖分析能够对 producer 排序，否则同一 lease 不能成为两个重叠 producer
  的 destination。
- 显式 buffer-set allocation 必须参与现有的各 memory space 容量校验。

由源程序导致的错误使用面向用户的检查，携带源 span，并同时报告实际值与期望值。
内部 pass 不变量使用 `INTERNAL_CHECK_SPAN`。

## Pass 与组件改动

- IR type、reflection、structural equality/hash、serialization、binding 和 stub 增加
  `TileBufferSetType`。
- Python typing 与 parser 增加 tuple 风格的 `TileBufferSet` wrapper。
- tile memory op 增加创建、选择、release 和 destination form。
- type inference 校验 destination compatibility。
- `LowerPipelineLoops` 和标量化简在复制 stage 时保留并规范化 slot selection。
- statement dependency analysis 理解 buffer set 与 slot index 的联合身份。
- `InitMemRef` 物化 group allocation 和 slot lineage。
- `MemoryReuse` 禁止显式 group root 参与合并。
- `AllocateMemoryAddr` 在 PyPTO planner 下放置 group root。
- PTO codegen 生成带地址或不带地址的 multi-buffer op。
- lifetime verifier 在 release marker 被删除前检查 release 和 alias 误用。

## 测试策略

### 类型与前端测试

- 构造并检查 `TileBufferSetType`。
- 验证 structural equality 与 structural hash。
- 验证 printer/parser 和 serialization round trip。
- 验证 `len(buffers)` 与动态 `buffers[index]` lowering。
- 拒绝将 buffer set 直接当作 tile 值使用。

### 校验测试

- 拒绝 count 1 和 17。
- 拒绝空 shape、动态 shape 和非正维度。
- 拒绝越界常量 index 和非整数动态 index。
- 分别拒绝 destination shape、dtype、memory space 和 layout 不匹配。
- 拒绝 release 后继续使用。

### 内存 pass 测试

- 两个 `[16, 128]` FP32 Acc slot 形成一个 16 KiB allocation group。
- `MemoryReuse` 不合并不同 slot 或不同 group。
- 独立 L1、L0B 和 L0C group 保留不同 allocation identity。
- 现有容量诊断包含显式 group。

### Pipeline 回归测试

- 复现 issue 中 L1 以 512 token、L0 以 128 token 轮换的两级 cadence。
- 验证 lowering 后保留两个 Right slot 和两个 Acc slot。
- 验证两次 extract、两次 matmul、两次 store 的调度顺序。
- 验证下一次复用 slot 0 前的依赖仍然存在。

### Codegen 测试

- PyPTO planner 生成带 `addr` 的 `pto.alloc_multi_tile`。
- PTOAS planner 生成不带 `addr` 的同一 op。
- 两者都生成动态 `pto.multi_tile_get`。
- 不生成多余的单 slot `pto.alloc_tile`、Acc-to-Acc `TMOV` 或与 destination 断开的
  allocation。

文档更新覆盖中英文 DSL 文档和受影响的 pass 文档，两种语言中的代码示例保持一致。

## 备选方案

### 给 `TileType` 增加 `count`

该方案可以减少 IR type class 数量，但所有现有 `As<TileType>()` 判断都会同时接受
buffer 集合。如果遗漏新的 `count == 1` 校验，整个集合可能被静默传给 compute op，
或由 type inference 错误传播 count。独立类型能让这类遗漏立即失败。

### 复用 `TupleType`

tuple 元素具有独立的 value identity 和 allocation identity，通常通过结构化方式选择；
它无法表示同构 allocation group 或运行时 slot selection。本设计只复用 Python 层的
tuple 风格语法。

### 为 `pl.pipeline` 增加 annotation

仅增加 pipeline buffer count annotation 的改动更小，但用户无法通过不同的嵌套索引
选择 slot，也不能显式绑定单个 destination，因此无法满足 issue #2131 的通用 buffer
identity contract。

### 使用 marker 固定静态地址

静态 pin 可以在 PyPTO planner 下保护不同 buffer，但无法表达运行时 slot selection，
并要求用户负责平台相关地址。本设计继续由 planner 负责地址放置。
