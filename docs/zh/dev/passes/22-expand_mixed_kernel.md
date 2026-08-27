# ExpandMixedKernel Pass

将混合 InCore 函数展开为独立的 AIC（Cube）+ AIV（Vector）内核，并包装在 Group 函数中。非混合 InCore 函数的 FunctionType 会被转换为 AIC 或 AIV。

## 概述

在 `OutlineIncoreScopes` 和 `ConvertTensorToTileOps` 之后，InCore 函数可能同时包含 Cube 操作（`tile.matmul`、`tile.gemv` 等）和 Vector 操作（`tile.add`、`tile.exp` 等）。部分操作如 `tile.load`、`tile.store`、`tile.move`、`tile.reshape` 根据其 tile 操作数的 MemorySpace 被分类为 Cube 或 Vector。包含两侧操作的函数是**混合 InCore 函数**。硬件要求 Cube 和 Vector 操作在不同的核心类型上运行，因此该 Pass 将它们拆分为：

- **AIC 函数**（`FunctionType::AIC`）— 仅包含 Cube + 共享操作
- **AIV 函数**（`FunctionType::AIV`）— 仅包含 Vector + 共享操作
- **Group 函数**（`FunctionType::Group`）— 依次调用 AIC 和 AIV，替换原始函数

当已有 Group 函数调用该 InCore 函数时（如来自 `OutlineClusterScopes`），该 Pass 会**就地改写该 Group 函数**以直接调用 AIC + AIV，避免产生冗余的 Group 包装。若 standalone `Spmd` 包装函数调用该 InCore，则该 Pass 会保留 `Spmd` 包装，并在其下方创建新的 `Group` 被调函数，使 launch 语义继续保留在 `FunctionType::Spmd` 上。

对于**非混合 InCore 函数**（纯 Cube 或纯 Vector），该 Pass 将 `FunctionType::InCore` 转换为对应的类型，无需拆分：

- 纯 Cube → `FunctionType::AIC`
- 纯 Vector 或仅包含共享操作 → `FunctionType::AIV`

该 Pass 执行后，程序中不再存在 `FunctionType::InCore` 函数。

### 手写 Group 的共享参数 ABI

一次运行时 `MixedKernels` 提交会创建一个协同调度的 AIC/AIV task，其中所有
active lane 共用同一份 `CoreTaskArgs` payload。因此，两个已编译 kernel wrapper
必须按完全相同的 tensors-first 参数布局解包。由 mixed-kernel 拆分器生成的 Group
天然满足这一约束，因为两个成员调用都会转发完整的 Group 签名。

手写 `pl.cluster()` 则可能提交源签名不同的成员，例如
`cube(q, k, v, sink, task)` 与 `vec(out, task)`；outline 后的 Group 会捕获这些值的
并集。本 Pass 会把两个 DSL 成员都归一化为该 Group 签名，并改写两个内层 `Call`
或 `Submit`，使其转发完整参数列表。未使用的参数仍是合法的 wrapper 参数；克隆后的
函数体依旧只引用原调用实际使用的值。

当两个成员 kernel 都只属于该 Group 时，保留其原函数名。如果任一 kernel 还有其他
调用点，Pass 会生成一对 Group 私有 adapter，并把
`import_peer_buffer(peer_func=...)` 改写为 adapter 的 peer 名称，从而为其他调用者保留
原 ABI。外部源码和 runtime-bound 成员无法安全克隆函数体；若它们的参数布局不同，
Pass 会拒绝该 Group，并要求两个声明使用相同签名。

归一化发生在 `InjectGMPipeBuffer` 之前，因此后端注入的 `__gm_pipe_buffer` 会追加到
已经一致的两个成员签名末尾。该过程不需要 `sync_start`：AIC 与 AIV 是同一个 mixed
task 的 subslot，而 `sync_start` 控制的是多 block SPMD 启动准入。

## 不可切分的转置：报错而非降级

当内核含有一个**交换了切分轴**的 `tile.transpose` 时,**请求的向量切分会被以 `ValueError` 拒绝**。`tile.transpose` 交换两个轴,于是每个 lane 的切分数据迁移到了*另一个*维度,而 `SplitVectorKernel` 仍按*原*切分轴减半 —— 它无法正确定型这种转置,结果形状错误并会算错。这与切分模式(UP_DOWN dim 0 / LEFT_RIGHT dim 1)和 dtype 都无关。

切分是用户掌控的性能决策,因此 pass 不会悄悄把它丢掉(那会编译出比用户要求更慢的内核),而是 fail-loud、指出出错的那个 transpose,并给出两条修改方向:

1. **删掉切分** —— 设 `attrs={"split": pl.SplitMode.NONE}`(或移除 `pl.split(...)` 优化)。这与用户本可直接请求的不拆内核完全一致。
2. **消除该 transpose** —— 例如把「转置后按行取」改成直接列切片 `pre[:, h:h+1]`。这样切分得以保留,内核仍拿到请求的加速,同时也避开了 pto-isa FP 转置在设备上的 tail-path 算错。

`split_axis::FindTransposeSplitHazard` 在 `ExpandMixedFunction` 开头检测:标记**第一个**在切分轴上非 singleton 的 `tile.transpose` 源(若源在切分轴上是 singleton,则不携带切分数据 —— 即广播 no-op 情形 —— 保持切分;动态的非 `ConstInt` extent 视为非 singleton,保守标记)。

该整函数检查只读取**单个** `func->GetSplitMode()`,无法表达多模式函数。当 `ExpandMixedKernel` 运行时,任何一等公民 `SplitAivScopeStmt` 区域均已被 [`LowerAutoVectorSplit`](21-lower_auto_vector_split.md)(pass 20)消费并擦除,后者已用**每个区域**各自的切分轴校验其转置风险,并在函数上打 `split_aiv_region_validated`。因此本 pass 对携带 `split_aiv_region_validated` 的函数跳过单一函数级模式的转置检查(AUTO 整函数路径保持不变);作用域节点永不到达此处 —— 只剩逐算子的 `aiv_shard` / `aic_gather` 标记。

CV 边界的跨核心数据传输通过将显式 `tile.move` 操作拆分为 `tpush`/`tpop` 对来处理：

| 方向 | AIC 侧 | AIV 侧 |
| ---- | ------ | ------ |
| Cube→Vector（如 Acc→Vec） | `tpush_to_aiv(source_tile)` | `dest_var = tpop_from_aic()` |
| Vector→Cube（如 Vec→Mat/Left/Right） | `dest_var = tpop_from_aiv()` | `tile.move` 适配 fractal 布局，然后 `tpush_to_aic(adapted_tile)` |

**Fractal TileView 布局**：跨核传输 tile 的 TileView 由 `BuildCrossCoreTransferView` 根据目标内存空间计算，不同后端的映射不同：

Ascend950（a5）——硬件跨核 pipe 直接按 fractal 布局传输数据：

| 方向 | Push/Pop TileView (blayout, slayout) | 名称 |
| ---- | ------------------------------------ | ---- |
| Vec->Left | col_major, row_major | NZ |
| Vec->Right | col_major, row_major | NZ |
| Vec->Mat | 需要显式在move中设置，否则同src | — |
| Mat/Acc->Vec | 需要显式在move中设置，否则同src | — |

`Vec->Right` 在边界上按 NZ 传输，而不是 Right 操作数最终的 ZN：a5 通过
`TINSERT_IMPL<TInsertMode::NZ>` 把 Vec tile 插入 Mat FIFO，因此桥接 tile 必须保持
NZ。ZN 属于 cube 侧在 pop 之后所做的 `Mat → Right` `tile.move`，比本表描述的边界晚
一步——与下面 a2a3 表格中写明的拆分相同。

Ascend910B（a2a3）——跨核传输经过 GM → Mat，Mat 仅支持 NZ 布局。因此 Left 和 Right 目标在传输边界统一使用 NZ，最终的 Left/Right 布局由后续 `Mat → Left/Right` `tile.move`（MTE1）处理：

| 方向 | Push/Pop TileView (blayout, slayout) | 名称 |
| ---- | ------------------------------------ | ---- |
| Vec->Left | col_major, row_major | NZ |
| Vec->Right | col_major, row_major | NZ |
| Vec->Mat | 保持原始布局 | — |
| Mat/Acc->Vec | 保持原始布局 | — |

在两种后端上，AIV 推送侧（V→C）都会在 `tpush_to_aic` 前插入一个 `tile.move` 将源 tile 转换为所需的 fractal 布局。`tile.move` 辅助函数（`CreateMove`）在结果类型携带 TileView 时会传播 `blayout`/`slayout` kwargs。

### 手写 pipe 同样获得该适配

上述规则描述的是边界移动路径，它只能看到本 pass 在展开 InCore 函数时构建的 pipe。完全手写的
pipe（`pl.reserve_buffer`、`pl.{aic,aiv}_initialize_pipe` 与 `pl.tpush_to_aic`）不会经过该
路径。在 `RequiresVtoCFractalAdapt()` 成立的后端上，这样的推送会把裸 ND tile 送进一个被 cube
按 fractal 解释的 FIFO，导致弹出 tile 的每个元素都错位。

`AdaptManualVtoCPush` 负责补上这个缺口。它作为本 pass 的最后一个阶段运行，遍历本 pass **产出**
的每一个 AIV 函数，而不是它收到的那些。这个区别很关键：`tile.tpush_to_aic` 声明的 affinity 是
`CoreAffinity::VECTOR`，因此手写推送合法地出现在 InCore 体内，而这样的函数体正是在本 pass 中才
变成 AIV 函数 —— 纯向量体经由非混合转换成为 AIV，混合体则把该语句带进展开后的 AIV 半边；两者都
发生在逐函数循环之后，所以更早的钩子仍会漏掉那条裸 ND 推送。

三条性质保证这次扫描既省又安全：

- **固定边界，无需跨函数分析。** 适配器取 cube 侧传输内存
  `GetBoundaryTpopMemory(CoreSide::AIC)`，而不是到对端函数里定位匹配的 `tpop`。这是精确而非
  近似的，因为 `BuildCrossCoreTransferView` 把 `Mat`、`Left`、`Right` 映射到同一个 fractal
  视图：无论消费者弹到哪里，它期望的布局都正是这里产出的那个。
- **幂等，且尊重作者的手写。** 源 tile 已经携带边界视图的推送会被跳过。重复运行本 pass 不会叠加，
  手工完成该 move 的程序保留自己的写法，边界移动路径已适配过的推送也不会被再插一次。
- **后端按需查询。** `RequiresVtoCFractalAdapt()` 在 mutator 内部、遇到第一条 V→C 推送时才读取，
  因此没有手写推送的程序无需配置后端即可走过该阶段。

重写后的推送保留原调用的 kwargs —— 丢掉 `id` 会让多 pipe 程序坍缩到同一个 FIFO 上。

### 经 GM 中转的跨核依赖

`tile.move` 并非数据跨越 CV 边界的唯一途径。当一条 lane 用 `tile.store` 写入某个 GM 张量、另一条 lane 用 `tile.load` 读取同一张量时，数据是*经由 GM*跨核的。由于这两个算子都不是 `tile.move`，CV 边界检测无法识别该依赖；若不加栅栏，拆分后的两个内核会在共享 GM 区域上发生数据竞争（issue #1433）。

`CollectGmCrossLaneSyncs` 检测此类 store/load 配对，并生成一个**纯同步**握手：数据仍照常经 GM 流动，同时由生产侧（store 之后）的 `tpush` 与消费侧（load 之前）的配对 `tpop` 建立缺失的 happens-before 关系。被 pop 出的 tile 仅作为栅栏令牌，随后由 `FinalizeTpopTfrees` 立即释放；`BuildAutomaticPipeSetup` 随后注入与 `tile.move` 边界相同的 pipe setup。

| 生产侧（写 GM） | AIC 侧 | AIV 侧 |
| --------------- | ------ | ------ |
| Cube store → Vector load | `tile.store ...; tpush_to_aiv(stored_tile)` | `tok = tpop_from_aic(); tfree_to_aic(tok); tile.load ...` |

为避免死锁，仅当满足以下条件时才生成握手：(1) 该 GM 源张量恰好只有一个生产侧 store；(2) 对侧 load 位于**同一结构化作用域**（同一循环/分支，使 `tpush`/`tpop` 执行次数一致）；(3) store 先于 load。跨越不同循环或分支的配对保持不变。

仅对 **Cube→Vector** 方向（cube `tile.store` → vector `tile.load`）插入栅栏。AIC 侧的 `tpush` 直接发送被存储的 tile（与常规边界 C2V 推送在两种后端上的行为一致），AIV 侧 `tpop` 落在 `Vec`。反向的 Vector→Cube 方向需要 `tile.move` 边界路径在 `tpush_to_aic` 前所做的 V→C fractal 布局适配；在该方向发送未经适配的原始 tile 会破坏跨核传输契约，因此 V2C 的 GM 交换保持不加栅栏。

当拆分后的内核包含跨核 `tpush`/`tpop` 时，该 Pass 还会自动在函数前缀补齐前端 pipe setup：

- 消费侧插入 `system.reserve_buffer(...)`
- 生产侧插入 `system.import_peer_buffer(...)`
- 两侧分别插入 `system.aic_initialize_pipe(...)` / `system.aiv_initialize_pipe(...)`

此外，Pass 还会在每条消费侧 `tpop` 链后插入
`system.tfree_to_aic(...)` / `system.tfree_to_aiv(...)`。

这些 setup 参数由拆分后的函数体自动推导：

- `dir_mask`：`C2V=1`、`V2C=2`、双向=`3`
- `id`：自动生成 setup 时省略，因此 PTOAS 使用默认 frontend pipe id `0`
- `slot_size`：所有方向中 tile 字节大小的最大值（`shape * dtype bits / 8`）
- `slot_num`：环形缓冲深度——每个方向默认均为 `2`，且始终显式发射；可按 scope 覆盖（见下文）
- `buffer_size`：`slot_num * slot_size`
- buffer 名称：`<func>_c2v_slot_buffer` / `<func>_v2c_slot_buffer`
- reserve-buffer 的 `base`：插入时统一使用 `AUTO`，随后由 `AllocateMemoryAddr` 解析成显式地址

当跨核方向使用了不同大小的 tile 时，Pass 会取所有观察到的 tile 字节大小的最大值作为 `initialize_pipe` 的公共 `slot_size`。较小 tile 写入时不会填满整个槽位，但不影响硬件正确性。用户手写程序仍然可以通过给 `initialize_pipe` 以及匹配的 `tpush` / `tpop` / `tfree` 传入不同 `id` 来创建多条独立 pipe。

### 覆盖槽位数（`slot_num`）

默认槽位数（`cross_core_pipe::kDefaultAutoPipeSlotNum` = 2）可以通过
`pl.cross_core_slot(slot_num=N)` 优化条目按 scope 调整：

```python
# 加深自动插入的环，让 cube 能比 vector 核跑得更超前，
# 代价是消费侧的 buffer 空间。
with pl.spmd(4, optimizations=[pl.cross_core_slot(slot_num=4)]):
    ...

# 也支持 pl.at 形式，并可与彼此独立的 split 模式组合：
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                          pl.cross_core_slot(slot_num=16)]):
    ...
```

该值作为 `slot_num` scope 属性记录，由 `OutlineIncoreScopes` 传播到被 outline
函数的 `slot_num` 属性，并在此 Pass 读取。它同时决定保留 buffer 的大小
（`slot_size * slot_num`）与发射的 `initialize_pipe` 的 `slot_num` 属性，使 PTOAS
与自动保留的 buffer 保持一致。无论 scope 是否覆盖该值，自动 setup **始终**发射该属性：
缺少该子句时 PTOAS 会按 `dir_mask` 自行推导深度（单向 8、双向 4），从而越过按更浅
的环预留的 buffer。只有手写的 `pl.system.{aic,aiv}_initialize_pipe` 仍会走到该回退路径。`slot_num`
必须为正数，且与拆分**正交**——它只决定数据通道的大小，不划分计算。它对 scope 实际
使用的方向都生效（cube->vector、vector->cube 或双向），并适用于任何驱动 pipe 的
scope：完全没有 split 的内核同样会驱动一条（a2a3 上通过双 AIV 派发——见下文）。
仅当被 outline 的 scope 最终没有跨核操作时才会被忽略。

**已废弃：** `pl.split(MODE, slot_num=N)` 仍接受该槽位数，但会发出
`DeprecationWarning`。该写法迫使作者写出并不需要的 split 模式——即
`pl.split(pl.SplitMode.NONE, slot_num=N)` 惯用法——而 `OutlineIncoreScopes`
现在会在任何携带 `pl.split_aiv` 区域的 scope 上拒绝它。请把该值迁移到
`pl.cross_core_slot(slot_num=N)`；打印器已统一输出为该形式。

> 将逻辑环深与更小的本地驻留窗口解耦（`local_slot_num < slot_num`，仅 a2/a3 的
> 优化）目前尚未在自动拆分路径暴露——如需该能力请使用手动的
> `pl.aic_initialize_pipe` / `pl.aiv_initialize_pipe` 系统算子。
>
> 该手动路径同样仅限 a2/a3，且不只限于 `local_slot_num < slot_num` 的情形：
> 950 前端 pipe lowering 下，ptoas 会拒绝 `pto.{aic,aiv}_initialize_pipe` 上的
> `local_slot_num` 操作数（无论取值为何），因此在 a5 上必须完全省略该操作数。

对于消费侧跨核 tile，该 Pass 还会保证每个 `tile.tpop_*` 都有匹配的
`system.tfree_*`。若现有 `tfree` 明显过早，Pass 会在不重排彼此独立 `tpop`
链的前提下，将它延后到同一个 block 中更靠后的语句之后。若 AIC 侧的
`tile.tpop_from_aiv` 结果还需要做同侧 `tile.move`（例如 Mat -> Left/Right/Bias），
则生成的 `system.tfree_to_aiv(...)` 会改写为释放 canonical 的 popped tile，并可被延后到这条 carrier 链之后。

对于 Ascend910B（a2a3），该 Pass 也支持**没有函数级 split 模式**的 mixed kernel（`split` 未设置或为
`SplitMode.None`）。此时 Pass 会保留单个 AIV kernel body，并为其打上**双 AIV 派发**标记；后续 lowering
会基于运行时 `subblock_idx` 插入分支：AIV lane 0 执行原始函数体，AIV lane 1 重放跨核握手，并将 tile-producing
replay 路径强制为 `valid_shape=[0, 0]`，同时屏蔽可见的 `tile.store` 写回。这样
`pl.at(level=CORE_GROUP)` 形式的 no-split mixed kernel 也能保持 AIC/AIV
握手对称，同时让第二个同步 lane 避免真实 DMA/计算工作。

此 replay 路径仅适用于**非 `split_aiv`** 的 mixed kernel。携带 `pl.split_aiv` 区域的函数——任意模式，包括任务并行的
`pl.SplitMode.NONE`——会被 [`LowerAutoVectorSplit`](21-lower_auto_vector_split.md)（pass 20）标记 `split_aiv`，因此
[`SplitVectorKernel`](24-split_vector_kernel.md) 会将其走 split 路径（两个 AIV lane 经由 `dual_aiv_dispatch` 派发），
而非上述 lane-0-only 的 replay。对 `NONE` 区域这正是所需：两个 lane 都运行**完整**函数体，处理由 `aiv_id` 分派的
不相交工作——丢弃 lane-1 的 store 会导致误编译。

**前置条件**：

- 输入 IR 必须具有 tile 操作（需先运行 `ConvertTensorToTileOps`）
- 输入 IR 必须已提取 InCore 作用域（需先运行 `OutlineIncoreScopes`）
- Tile 操作必须已展平为 2D（需先运行 `FlattenTileNdTo2D`）
- Tile 内存空间必须已推断（需先运行 `InferTileMemorySpace`）
- 跨核 Fractal TileView 分配在 Ascend950 和 Ascend910B 后端均受支持

**使用时机**：在 `InferTileMemorySpace` 之后运行，当 InCore 函数可能同时包含 Cube 和 Vector tile 操作时使用。

> **注意**：该 Pass 已经在默认 tile 优化流水线中启用。调试问题或自定义 PassPipeline 时，也可以通过
> `passes.expand_mixed_kernel()(program)` 显式调用。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::ExpandMixedKernel()` | `passes.expand_mixed_kernel()` | 程序级 |

**Python 用法**：

```python
from pypto.pypto_core import passes

expand_pass = passes.expand_mixed_kernel()
program_expanded = expand_pass(program)
```

## 算法

```text
阶段 1 — 预扫描：
  识别哪些 InCore 函数已有 Group 调用者，以及哪些调用者仍然需要保留原始函数名。

阶段 2 — 展开每个 InCore 函数 F：
  1. 递归分类所有语句的亲和性（包括循环/条件内部）
  2. 检测 CV 边界移动：跨越 cube↔vector 内存空间的 tile.move 操作
     （记录在独立的 boundary_moves 映射中，而不是作为单独的亲和性枚举值）
  2a. 检测经 GM 中转的跨核 store/load 配对（CollectGmCrossLaneSyncs）：
      一条 lane 上的 tile.store 与另一条 lane 上读取同一 GM 张量来源的
      tile.load，调度为 tpush/tpop 同步栅栏（见上文“经 GM 中转的跨核依赖”）
  3. 如果不是混合的（没有 CUBE 操作或没有 VECTOR 操作，且没有边界移动）：
     将 FunctionType 转换为 AIC（纯 Cube）或 AIV（纯 Vector / 仅共享操作）
  4. 构建 AIC 函数体：保留 CUBE + SHARED 语句，删除 VECTOR，递归处理 MIXED 循环
     - 对于边界移动（Cube→Vector）：生成 tpush_to_aiv(source_tile)
     - 对于边界移动（Vector→Cube）：生成 dest_var = tpop_from_aiv()，携带 fractal TileView
  4a. 对由算子驱动的边界（tile.aiv_shard / tile.aic_gather），根据该算子携带的
      MODE 与全宽 tile 的 extent，推导 tpush/tpop 对使用的 pto-isa split CODE
      （BoundaryTransportSplitCode -> split_axis::ShardSplitCode / GatherSplitCode）
      ——见下文“split code”
  5. 构建 AIV 函数体：对称（保留 VECTOR + SHARED，删除 CUBE）
     - 对于边界移动（Cube→Vector）：生成 dest_var = tpop_from_aic()，携带 fractal TileView
     - 对于边界移动（Vector→Cube）：生成 tile.move 适配 fractal 布局，然后 tpush_to_aic(adapted_tile)
  5a. 通过 BuildCrossCoreTransferView 为所有边界 tpop 结果类型和 tpush 前的 tile.move 操作分配 fractal TileView：
      - Ascend950：Left→NZ，Right→ZN，Mat/Vec→保持原始
      - Ascend910B：Left→NZ，Right→NZ（Mat 仅支持 NZ），Mat/Vec→保持原始
  6. 修复两侧函数体中的循环携带状态
     - 删除在当前核心侧无用的 dead iter_args
     - 为保留下来的 iter_args 补回缺失的 init value 定义
     - 当分支局部值被裁剪后，将悬空 yield 改写为 identity yield
     - 将悬空的 tile.store 结果变量（被 AIC 侧拆分裁剪的 SSA 版本）重映射到对应的输出参数
  7. 对两个函数体运行死代码消除（递归进入循环）
  8. 再次归一化循环携带状态，因为 DCE 可能移除仅用于过渡的 SHARED 后续引用，
     使某些 iter_arg 到这一步才变成可删除
  9. 再次运行死代码消除，清理第二次裁剪后暴露出的 init-value 链
 10. 保证每条消费侧 `tpop` 链都有匹配的
      `system.tfree_to_aic` / `system.tfree_to_aiv`，并在需要时把明显过早的 free 延后到同一个 block 内更靠后的位置
 11. 如果拆分后的函数体包含跨核 tile 操作，且尚未带有 setup，
      则推导并 prepend reserve/import/initialize_pipe 前缀
 12. 创建 AIC 函数（无返回值）和 AIV 函数（原始返回值）
     - 对于 Ascend910B 上的 no-split mixed kernel，为生成的 AIV 打上 dual-dispatch 元数据，
       使后续 lowering 在两个 vector lane 上派发同一个 AIV kernel，并将第二个 lane 的 tile replay
       路径改写为 `valid_shape=[0, 0]`
 13. 如果满足以下任一条件，同时创建 Group 函数（调用 AIC 和 AIV）：
     - 仍有非 Group 调用者需要保留原始函数名（`needs_preserved_name`，例如 standalone Spmd 包装）
     - 尚不存在 Group 调用者（`!has_group_caller`）
 14. 如果已存在 Group 调用者且无需保留名称：跳过额外的 Group 包装

阶段 3 — 改写 Group 调用者：
  对于每个调用了已拆分 InCore 的 Group 函数，将 InCore 调用替换为
  AIC 调用 + AIV 调用序列（AIC 用 EvalStmt，AIV 用 AssignStmt）。

阶段 4 — 归一化手写 mixed Group 的 ABI：
  对于 AIC/AIV 调用未同时转发完整 Group 签名的 Group，把两个 DSL 成员函数体
  重建到同一份 canonical 参数列表上，并改写两个 Call/Submit 以传入完整列表。
  独占成员保留原函数名；否则创建 Group 私有 adapter，并同步重定向 peer_func。
```

**亲和性分类**：

| 亲和性 | 操作 | 分类规则 |
| ------ | ---- | -------- |
| CUBE | `tile.matmul`、`tile.matmul_acc`、`tile.matmul_bias`、`tile.gemv`、`tile.gemv_acc`、`tile.gemv_bias`、`tile.batch_matmul`、`tile.batch_matmul_acc` | 始终为 CUBE（按操作名） |
| CUBE 或 VECTOR | `tile.load` | 按 `target_memory` kwarg：cube 侧内存（Mat、Left、Right、Acc、Bias）→ CUBE；Vec → VECTOR |
| CUBE 或 VECTOR | `tile.store`、`tile.reshape` | 按源 tile 的 `memory_space`：cube 侧内存 → CUBE；Vec → VECTOR |
| MIXED | 跨越 cube↔vector 内存的 `tile.move` | 跨核心侧的叶子移动 —— 同时记录在 `boundary_moves` 映射中（见下文） |
| CUBE 或 VECTOR | `tile.move`（同侧） | 按源 tile 的 `memory_space` |
| VECTOR | 所有其他 `tile.*` 操作（`tile.add`、`tile.exp`、`tile.sub` 等） | 始终为 VECTOR（按操作名） |
| VECTOR | `pld.tile.put`、`pld.tile.get`、`pld.tensor.put`、`pld.tensor.get` | 显式声明 `set_core_affinity(VECTOR)` —— TPUT/TGET 经由 VEC 暂存 tile 中转，ptoas 强制校验该约束 |
| SHARED | 非 tile 操作、函数调用、控制流、标量操作 | — |
| SHARED | `pld.system.notify`、`pld.system.wait` | 按 ISA 属于核无关操作（纯标量/GM），因此不声明亲和性。`notify` 额外声明了 `set_no_duplicate()`（对两种 `NotifyOp` 形式都生效）—— cube 侧的副本可能在向量 lane 的 TPUT 落盘数据之前就释放对端。`wait` 则不声明：它会*阻塞*，其 cube 侧副本是有实际作用的 |
| MIXED | 包含 CUBE 和 VECTOR 子语句的复合语句 | — |
| VECTOR | **任何打上 `attrs["core_placement"] = "aiv"` 的调用** | 区域放置——凌驾于上述所有规则之上。见下文 |

**区域放置凌驾于推断之上。** `SHARED` 不再无条件复制到两条 lane 上。作者写在 `pl.split_aiv`
区域内的语句会携带 `attrs["core_placement"] = "aiv"`——由
[`LowerAutoVectorSplit`](21-lower_auto_vector_split.md) 在擦除区域包装前打上（该文档说明了
打标范围与理由）——`ClassifyCallAffinity` 将这类调用解析为 `VECTOR`，因此它只落在 AIV lane
上。正是这一点阻止了 `pld.system.notify`（核无关，因而 `SHARED`）同样落到 cube lane 上——
在那里它可能在向量 lane 的 TPUT 把该信号所释放的数据落盘之前就发布信号。

对*未被放置*的那一类，没有任何检查会拒绝：写在所有区域之外的通信算子会被复制到两条 lane 上，
且不会有任何诊断。该标记也不能让它只运行一次——本 pass 发出的 AIV 函数带有
`dual_aiv_dispatch`（当后端确实需要两条子 lane 时，即 Ascend910B 的 no-split 混合 kernel），
此时其函数体会在两条 AIV 子 lane 上都运行，把只应发生一次的副作用在两条子
lane 之间分片是作者的职责（见[作用域与放置](../../user/language/04-scopes.md)）。

该覆盖是完备的，并在三种情况下拒绝覆盖——每种都是因为“该调用的 lane 并非由区域决定”：

| 本征亲和性 | 结果 | 原因 |
| ---------- | ---- | ---- |
| **自述** lane（`set_core_affinity` 或 `core_type` kwarg） | 保持不变 | `tile.create` 按策略为 `SHARED`，以便两条 lane 都能声明该缓冲；`system.syncall(core_type="mix")` 需要两核会合。声明优先于放置 |
| `MIXED` | 保持不变 | `aiv_shard` / `aic_gather` 与跨 C/V 的 `tile.move` **就是**那次传输——强制为 `VECTOR` 会让 tpush 失去配对的 tpop |
| `CUBE` | 保持不变 | 区域内的 cube 计算是作者错误，由检查 (a) 报告；不覆盖即让它照旧留在 cube lane 上，而不是在关闭验证时把它错误地搬到向量 lane |

pass 20 只给不属于上述三类的调用打标，因此编译器产出的 IR 中这些分支永不触发；写成完备形式
是因为该属性就是能经受 print → parse 往返的普通 IR。亲和性汇总读取完毕后，本 pass 会把该
属性从它发出的每个函数上**剥除**（包括仅被改型或原样透传的函数），因此它不会到达任何后续
pass 或打印输出。

**CV 边界检测**：当 `tile.move` 的源 tile 内存和目标内存位于不同核心侧时，该移动为 CV 边界。Cube 侧内存：Mat、Left、Right、Acc、Bias。Vector 侧内存：Vec。同侧移动（如 Mat→Left）按其源内存照常分类。边界叶子移动在亲和性上被标记为 `MIXED`，并额外记录在独立的 `boundary_moves` 映射中；跨核方向（Cube→Vector vs Vector→Cube）由 `CollectCVBoundaryMoves`、`BuildCoreBody` 等调用点通过 `ClassifyMoveDirection` 即时恢复。

**嵌套结构处理**：包含混合操作的 ForStmt、IfStmt 和 WhileStmt 会被复制到 AIC 和 AIV 函数体中，内部内容递归裁剪。

**拆分后的循环状态修复**：在构建 AIC/AIV 函数体时，Pass 会先保留共享的控制流骨架，因此某一侧可能暂时留下多余的 iter_args、缺失的 init value 定义，或引用已被裁剪分支局部值的 yield。Pass 会先在 DCE 前按固定顺序修复这些情况，再在 DCE 后做一次循环状态归一化，因为某些仅用于过渡的共享别名会在 DCE 后消失，进而让相应 iter_arg 变成真正可删除。最后再运行一次 DCE，清理第二次裁剪后暴露出的 init-value 链。

**Group 包装函数的参数化返回**：新建的 Group 包装函数在所有
返回位置都能追踪到参数回写（经 `return_lineage::ReturnedParamIndices`）时，
直接返回自身参数——AIV 调用作为纯 `EvalStmt` 发出，包装函数直接
`return out_0`，从而维持 `ReturnParamsExplicit` 不变量。只有当某个返回位置
不是参数回写时，才回退到旧的 `result = aiv_call(); return result` 形式。

## split code

`tile.aiv_shard` / `tile.aic_gather` 携带作者选择的 **mode**（`0` / `1` / `2`）；本 pass
生成的传输对携带的是 pto-isa 的 `TileSplitAxis` **code**（`0`..`4`），后者还表达了两个
AIV lane 的*运行时* extent 之间的关系——消费侧正是靠它在 FIFO 槽位中定位 lane 1 的数据段：

| Code | pto-isa | Lane 1 的数据段起点 | 要求 |
| ---- | ------- | ------------------- | ---- |
| 1 / 2 | `TILE_UP_DOWN` / `TILE_LEFT_RIGHT` | `e1 * pitch` | `e0 == e1` |
| 3 / 4 | `TILE_UP_DOWN_ODD` / `TILE_LEFT_RIGHT_ODD` | `(e1 + 1) * pitch` | `e0 == e1 + 1` |

`BoundaryTransportSplitCode` 从**全宽** tile 上读取 lane extent
`eL = clamp(V - L*S, 0, S)`——shard 取其操作数（Cube→Vector），gather 取其结果
（Vector→Cube）。`S` 是分区步长：默认为该 tile 的物理 half；当
[LowerAutoVectorSplit](21-lower_auto_vector_split.md) 对 ragged 边界做了均分时，则取算子上
`lane_stride=S` 属性。因此奇数切分轴（奇数物理 box、偶数 box 内的奇数 valid extent，或均分
后的奇数 valid 区域）会取奇数 code；lane 1 为空时保持偶数 code；pto-isa 无法摆放的 ragged
extent（只可能出现在 box 分区上）则被拒绝，并给出可行的取值。两侧 lane 的函数体用相同输入执行同一推导，因此 AIC 与 AIV 侧永远一致。
Vector→Cube 的 gather 没有奇数形态（pto-isa 的向量侧 producer 按 lane 1 自身的 extent
偏移），因此只使用偶数 code。完整推导见
[LowerAutoVectorSplit](21-lower_auto_vector_split.md)。

## 示例 1：InCore 没有已有 Group 调用者

当 Orchestration 直接调用 InCore 时，创建新的 Group 包装函数。

**之前**（经过 `InferTileMemorySpace` 之后）：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute_incore_0(self, x: pl.Tensor[[16, 128], pl.BF16],
                         y: pl.Tensor[[128, 128], pl.BF16],
                         out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
                         ) -> pl.Tensor[[16, 128], pl.FP32]:
        x_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
        y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_mat, target_memory=pl.Mem.Left)
        y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.Mem.Right)
        z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
        z_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(z_tile, target_memory=pl.Mem.Vec)
        out_0 = pl.store(z_vec, [0, 0], out_0)
        return out_0

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_incore_0(x, y, out_0)
```

**之后**（概念性 — 实际 IR 包含所有变量的类型注解；为简洁起见，这里省略自动生成的 pipe setup 前缀）：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_incore_0_aic(self, x, y, out_0):
        x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.Mem.Mat)  # CUBE：加载到 Mat
        y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.Mem.Mat) # CUBE：加载到 Mat
        x_left = pl.move(x_mat, target_memory=pl.Mem.Left)   # CUBE：Mat→Left（同侧）
        y_right = pl.move(y_mat, target_memory=pl.Mem.Right)  # CUBE：Mat→Right（同侧）
        z_tile = pl.matmul(x_left, y_right)              # CUBE 操作
        pl.tile.tpush_to_aiv(z_tile, split=0)        # 边界移动：推送 Acc tile 到 AIV

    @pl.function(type=pl.FunctionType.AIV)
    def compute_incore_0_aiv(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # tpop 结果携带 fractal TileView（Vec 目标 → 保持原始布局）
        z_vec: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView()] = pl.tile.tpop_from_aic(split=0)
        out_0 = pl.store(z_vec, [0, 0], out_0)           # VECTOR 操作
        pl.system.tfree_to_aic(z_vec)                    # 释放已消费的跨核槽位
        return out_0

    @pl.function(type=pl.FunctionType.Group)
    def compute_incore_0(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        self.compute_incore_0_aic(x, y, out_0)
        self.compute_incore_0_aiv(x, y, out_0)
        return out_0  # param-return: the AIV result writes through out_0

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_incore_0(x, y, out_0)  # 调用 Group（同名）
```

## 示例 2：InCore 有已有 Group 调用者

当 `OutlineClusterScopes` 已经创建了调用 InCore 的 Group 函数时，该 Pass 改写已有 Group，而不创建新的包装函数。若 `OutlineClusterScopes` 创建的是 standalone `Spmd` 包装，则该 Pass 会保留该 `Spmd` 函数，并把它改为调用新的同名 `Group` 函数。

**之前**：

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute_incore_0(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # ... 混合 Cube + Vector 操作 ...

    @pl.function(type=pl.FunctionType.Group)
    def compute_group(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        result = self.compute_incore_0(x, y, out_0)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_group(x, y, out_0)
```

**之后** — 已有 Group 被改写，不存在 `compute_incore_0` Group 包装：

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_incore_0_aic(self, x, y, out_0):
        # ... Cube 操作 + tpush ...

    @pl.function(type=pl.FunctionType.AIV)
    def compute_incore_0_aiv(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # ... tpop + Vector 操作 ...

    @pl.function(type=pl.FunctionType.Group)
    def compute_group(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        self.compute_incore_0_aic(x, y, out_0)       # 改写后：AIC 调用
        result = self.compute_incore_0_aiv(x, y, out_0)  # 改写后：AIV 调用
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_group(x, y, out_0)  # 不变
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

**实现文件**：`src/ir/transforms/expand_mixed_kernel_pass.cpp`

**Python 绑定**：`python/bindings/modules/passes.cpp`

**测试**：`tests/ut/ir/transforms/test_expand_mixed_kernel.py`

## Pass 属性

| 属性 | 值 |
| ---- | -- |
| 所需 | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred, NormalizedStmtStructure |
| 产生 | SSAForm, MixedKernelExpanded, NormalizedStmtStructure, HardSyncallOccupancyValid, AccCompactValid |
| 失效 | AccCompactValid |

`HardSyncallOccupancyValid` 在此产生，并非因为本 pass 做了什么改写，而是因为它把每个 kernel 的 `FunctionType` 解析为 AIV/AIC/Group——这正是硬 syncall 占用率 verifier 所依赖的前置条件。该 verifier 只在本 pass 之后触发一次。

`AccCompactValid` 在此失效并重新产生：Cube->Vector 边界的 `tile.move` 在本 pass 中被重建为 tpush/tpop 对，并带上新构造的消费者类型，因此 Acc compact 契约必须在这份新 IR 上重新校验，而不能沿用 `InferTileMemorySpace` 的结论。

## 属性验证器

`MixedKernelExpandedPropertyVerifier` 会检查：

- 剩余的 `FunctionType::InCore` 函数不再同时包含 Cube 和 Vector tile 操作
- AIC 中的 `tile.tpop_from_aiv` 必须落到 `MemorySpace::Mat`
- AIV 中的 `tile.tpop_from_aic` 必须落到 `MemorySpace::Vec`
- 带跨核 `tpush`/`tpop` 的 AIC/AIV 函数必须同时具备所需的 pipe setup
- 每个 AIC/AIV `tile.tpop_*` 都必须有匹配的 `system.tfree_*`
- 跨核 tile 操作的 tile 形状必须是静态已知的（自动 pipe setup 所需）

这样，常见错误（缺少 `initialize_pipe`、缺少 `reserve_buffer` / `import_peer_buffer`、缺少 `tfree`、
非静态 tile 形状、跨核 `tpop`/`tfree` 不匹配）会在 `ExpandMixedKernel` 之后立即报出，而不是拖到 PTO codegen /
`ptoas` 阶段。当前 verifier 只检查按 tile 值/op 的配对关系，并不证明 `tfree` 一定落在该 tile 的真实最后一次使用之后。

## 设计决策

| 决策 | 理由 |
| ---- | ---- |
| 基于 move 的 CV 边界检测 | 显式 `tile.move` 操作标记边界——无需脆弱的变量数据流分析 |
| 使用 `boundary_moves` 映射（而非独立枚举值） | 边界状态可由 `ClassifyMoveDirection` 在调用点即时推导；以侧表存储可让 `CoreAffinity` 保持四个执行侧分类（CUBE / VECTOR / SHARED / MIXED） |
| 数据移动操作基于 MemorySpace 分类 | `tile.load`/`tile.store`/`tile.move`/`tile.reshape` 根据其操作的内存空间服务于 Cube 或 Vector；`InferTileMemorySpace` 在该 Pass 之前已设置此信息 |
| tpop 结果携带 Fractal TileView | tpop 结果类型直接携带 fractal TileView（NZ/ZN），而非剥离布局——下游 Pass 和 codegen 无需额外推断即可看到正确布局 |
| AIV 侧 tpush 前插入 `tile.move` | V→C 传输要求数据为 fractal 布局；在 `tpush_to_aic` 前插入显式 `tile.move` 使布局转换在 IR 中可见 |
| `CreateMove` 传播布局 kwargs | 当结果类型携带 TileView 时，`blayout`/`slayout` 作为 kwargs 转发，使生成的 `tile.move` 调用自描述 |
| Group 保留原始函数名 | 无已有 Group 调用者时：Orchestration 调用点无需修改——不需要重写调用点 |
| 改写已有 Group 调用者 | 当 Group 已调用 InCore 时（如来自 `OutlineClusterScopes`）：就地改写以调用 AIC + AIV，避免冗余的 Group→Group 嵌套 |
| 保留 standalone Spmd 包装 | 当 standalone `Spmd` 调用 InCore 时：保留 `FunctionType::Spmd`，在其下创建 `Group` 被调函数，并继续由 Spmd 持有 `core_num` / `sync_start` |
| 参数复制到所有三个函数 | 简化连接；DCE 在下游 Pass 中移除未使用的参数 |
| 递归处理复合语句 | 正确拆分 `ForStmt`、`IfStmt`、`WhileStmt` 内部的混合操作 |
| 两阶段拆分后循环状态修复 | 先保证 loop-carried state 合法，再在 DCE 移除死共享别名后重新裁剪 iter_arg，最后再跑一次 DCE 清理暴露出的 init-value 链 |
| 自动生成 pipe setup | tensor 级 mixed kernel 无需手写 `reserve_buffer` / `import_peer_buffer` / `initialize_pipe`；Pass 会根据跨核 tile 操作自动推导 |
| 自动生成 tfree 链 | 消费侧拆分内核会补齐缺失的 `tfree`、将其改写为释放 canonical 的 popped tile，并在需要时把明显过早的 free 延后到同一个 block 内更靠后的位置；但不会重排彼此独立的 `tpop` 链顺序 |
| Max-slot-size 策略 | 取所有 tile 字节大小的最大值作为单一 `initialize_pipe.slot_size`，对齐后端自动生成 setup 的假设，并保留旧有双向 `dir_mask=3` 行为 |
