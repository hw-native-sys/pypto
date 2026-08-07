# LowerPipelineToSlots Pass

把 `pl.pipeline(N, stage=F)` 循环改为让**一份**循环体轮转同一个分配的 `F` 个 slot，而不是把循环体复制 `F` 份。

## 概述

`pl.pipeline(N, stage=F)` 表达的是乒乓缓冲的诉求。[`LowerPipelineLoops`](29-lower_pipeline_loops.md) 用**复制**来兑现：把循环体复制 `F` 份，每份都有全新的定义变量，于是各份的 tile 是彼此独立的 MemRef，`MemoryReuse` 不允许把它们合并。这条路可行，但代价是 `F` 倍的代码量、一套静态/动态余数派发，以及为了隔开各份副本而存在的 `pipeline_membership` 机制。

本 pass 用 ptoas 本来就认识的形式表达同一个意图。`pl.MemRef(name, slots=F)` 的含义正是**"一个分配、F 个等大 slot、本次使用取第 k 个"**（见[Slots](../language/00-python_syntax.md#slots)），而 PTO codegen 恰好把它下降为 `pto.alloc_multi_tile` + `pto.multi_tile_get`。于是循环只保留**一份**循环体，每个按 stage 私有的缓冲变成合成声明的第 `iv % F` 个 slot：

```python
# Before（本 pass 看到的形态）
for i in pl.pipeline(64, stage=2):
    x: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.tile.load(a, [i * 128], [128])
    pl.tile.store(x, [i * 128], out)

# After —— 单份循环体，边界与步长原封不动，kind 降级
for i in pl.range(64):
    x: pl.Tile[[128], pl.FP32, pl.MemRef("pipe_x", slots=2)[i % 2], pl.Mem.Vec] = \
        pl.tile.load(a, [i * 128], [128])
    pl.tile.store(x, [i * 128], out)
```

**不引入新的 IR op，也不新增用户可见开关。** 合成出来的 MemRef 与作者手写的声明形状完全一致，因此 [`InitMemRef`](32-init_memref.md) 走同一条路径解析它，codegen 也分辨不出这个轮转是作者写的还是编译器推导的。

由于边界、步长和 `iter_args` 都没有改动，不存在需要派发的余数——动态 trip count 完全不需要特殊处理。

**依赖属性**：SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、TileMemoryInferred、NormalizedStmtStructure。

**流水位置**：在 [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md) 之后，紧接 [`LowerPipelineLoops`](29-lower_pipeline_loops.md) 之前。足够晚，内存空间已推断、tile 结构已定型；又足够早，`InitMemRef` 还没有给这些 tile 分配编译器自己的 MemRef。

## 两个 pass 是互补关系，不是二选一

两者都会执行，且按此顺序。本 pass 只接手能证明安全的循环并将其降级；**凡是它不接手的循环都保持 `ForKind::Pipeline`**，由 `LowerPipelineLoops` 照旧复制。不会因为本 pass 的存在而让任何循环失去乒乓——matmul L0 stage 循环、嵌套 pipeline、形状特殊的循环都仍然走复制路径。

这与 [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md) 的做法同构：它处理跨核 pipeline 循环，其余原样留下。

## 自门控于 `memory_planner=PTOAS`

在默认的 PyPTO planner 下，本 pass 对每个函数都原样返回，因此那条路径保持**字节一致**。

这个门控划的是"region 在哪里被**发出**"，而不是"ptoas 在哪里**能用** region"。PTO codegen 的 `PlanMultiBufferRegions` 在 PyPTO planner 下直接返回，所以在那条路上合成的轮转只会落成一条运行时地址的普通 `alloc_tile`——正确，但拿不到本变换赖以生效的 slot 分析。

**限制不在 ptoas。** 给定 `pto.alloc_multi_tile addr = <常量 base>`，ptoas 0.55 在 `--pto-level=level3` 下推导出的 per-slot 动态 event 同步与 level2 一致——在一个预取循环上实测，归一化 event id 后同步算子序列逐条相同（prime 两个 event、按 slot 键控的 `wait_flag`/`set_flag`、两个 drain）。ptoas 0.54 则不会，这正是较早的
[PTOAS#1106](https://github.com/hw-native-sys/PTOAS/issues/1106) 所描述的情形。

因此放宽门控要做的是 PyPTO 侧而非上游的工作：地址分配器需要为 region base 预留 `slot_count * slot_size`，codegen 需要把该地址发到 region 上。这属于后续工作；在它落地之前，本 pass 只覆盖 codegen 路径已经存在的那个 planner。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::LowerPipelineToSlots()` | `passes.lower_pipeline_to_slots()` | 函数级 |

```python
from pypto import passes
with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
    result = passes.lower_pipeline_to_slots()(program)
```

## 行为

对于 `F > 1` 且通过下列全部门槛的 `ForStmt(kind == ForKind::Pipeline, attrs["pipeline_stages"] == F)`：

1. 把每个候选 tile 的 `TileType` 重绑到一个新的 pinned `MemRef(name, slots=F)`，`slot_index = iv % F`。定义与其全部使用点在一趟遍历中改写完成——IR 处于 SSA 形式，任何使用都不会出现在其定义之前。
2. `kind_` 变为 `ForKind::Sequential`，同时剥掉 `pipeline_stages`。两者始终同进同出，故 `PipelineLoopValid` 不变式（`kind == Pipeline` 当且仅当 `pipeline_stages` 存在）在每个可观察状态都成立。
3. 其余一概不动：不新增、不删除、不重排任何语句，`start` / `stop` / `step` / `iter_args` 全部保持原样。

`F == 1` 要么是用户手写的 `pl.pipeline(stage=1)`，要么是上一次 `LowerPipelineLoops` 留下的标记。两种情况都不需要多缓冲，因此 (kind, attr) 这一对保持完整，留给 `CanonicalizeIOOrder` 作用域使用。

### 哪些 tile 取 slot

循环体顶层、且作者尚未自行绑定的**全部** `tile.load` 结果。

- **只取 load。** 需要保持私有的是 load 缓冲——这样第 `i+1` 次迭代的预取才能与第 `i` 次的计算重叠；计算中间结果可以合并。这与 [`MemoryReuse`](34-memory_reuse.md) 通过 `pipeline_load_tiles` 划出的界线一致。给**所有** tile 都开 `F` 份私有缓冲会在真实 kernel 上撑爆片上预算——`stage=4` 的 RMSNorm 需要 `4 x 67 KB > 188 KB` UB。`tile.read` **不在其中**：它返回的是标量元素而非 tile，没有可轮转的缓冲。
- **顶层。** 仅限循环体 `SeqStmts` 的直接成员；嵌套在内层循环或 `if` 中的 load 属于那个区域。
- **不做循环不变性过滤。** 顶层未绑定的 load 一律入选，包括实参从未提到归纳变量的那些。是否循环不变无法用归纳变量判断：经由循环携带的 `IterArg` 寻址的 load，每次迭代读到的数据都不同，却从不出现 `iv`；一旦同循环内另有候选把循环降级，跳过它就会让它既拿不到 slot、也进不了复制。而对真正循环不变的 load 开槽也不比回退更亏——`LowerPipelineLoops` 同样会把它的缓冲复制 `F` 份。
- 作者已经绑定到声明分配的 tile 仍归作者所有。

### 合格性

codegen 对无法描述的 region 是**硬拒**而非降级，因为退回逐 slot 的 `alloc_tile` 会让 ptoas 把这些 slot 规划到彼此头上。所以这里的每一条门槛都对应 `PlanMultiBufferRegions` 的一个 blocker：合成一个可疑的 region 会把今天能正常编译的 kernel 变成编译失败。

| 门槛 | 原因 |
| ---- | ---- |
| `F` 落在 `[2, 16]` | ptoas `multi_tile_buf` 的 slot 数上下界 |
| 内存空间为 Vec / Mat / Acc | ptoas 接受的 slot 空间 |
| 静态 valid shape | 一个 region 为其所有 slot 声明唯一的静态 extent |
| 未被带入 phi | 被 yield 的 tile，或被用作嵌套循环 `init_values` 的 tile，都会让那个 phi 共享它的 MemRef。两者殊途同归——前者经由 `YieldStmt`，后者经由 `IterArg::initValue_`。判定基于**别名根**：`InitMemRef` 会让裸的 `a = b` tile 拷贝、以及 view / 原地算子的结果共享同一个 MemRef，因此 yield 一个别名与直接 yield 原 tile 一样会把槽位带进 phi |
| 未被 view / 原地 op 消费 | 这类结果**就是**其源的缓冲，会落到同一分配上却带着不同的 `tile_buf` 类型 |
| `step == 1` 且 `start % F == 0` | 见下 |
| 没有被拒绝的外层 pipeline 循环 | 见下 |
| slot 放得进该内存空间 | 见下 |

**拒绝以循环为单位，而非以 tile 为单位。** 上表中的四条 tile 级门（内存空间、静态 valid
shape、phi、view / 原地算子）只对**真正想要 slot** 的 load 生效——即顶层、且作者尚未自行绑定的
那些。只要有一个这样的 load 触发其中任意一条，**整个循环**都会被拒绝，即使同一循环体内还有其他
合格的 load。只丢掉那一个 load 是行不通的：任何幸存的 candidate 仍会把循环降级为 `Sequential`，
被挡住的那个 load 于是既拿不到本 pass 的 slot、也进不了 `LowerPipelineLoops` 的复制，
`pl.pipeline(stage=F)` 所要求的按 stage 私有缓冲就被静默丢失了。只有作者已绑定的 tile 会被跳过
而不影响整个循环——因为为它拒绝循环反而会把该声明推上复制路径，而复制路径会拒绝它。

**为什么 slot 索引必须字面上是 `iv % F`。** ptoas 依据 slot 索引的**仿射形式**来判定哪些访问共享一个 slot，而这个匹配正是轮转拿到 per-slot 动态 event id 的依据——喂给它一个折叠后的字节偏移会让分析失效。一般形式 `((iv - start) / step) % F` 必须物化为中间 SSA 值，有丢掉本变换赖以生效的那个分析的风险。索引无法直接写成该形式的循环一律留给复制路径。

**为什么 slot 必须放得下。** 声明出来的 slot 是**钉住**的：`InitMemRef` 按 `F * slot_size`
给出分配，ptoas 不得复用其中任何一部分，因此这些字节由本 pass 直接负责。否则一个有多个合格 load
的循环会把占用乘上 `F`，而 ptoas 对放不下的 region 是**硬报错** `overflow`，并不会降级。复制路径
则会降级：`MemoryReuse` 的容量闸门会下调实际双缓冲深度（`F_g = min(depth_g, ⌊C_s / slot_g⌋)`）
并跨组 shed 直到放得下——所以拒绝等于把循环交给一条"会缩、不会挂"的路径。

预算按内存空间统计，累加该循环的全部候选，并在**整个函数**范围内累计——被开槽的内层循环，其 region
与被开槽的外层是同时存活的。预算初值来自**作者已声明**的分配：那些同样带 `is_pinned_`，ptoas 也不能
复用；忽略它们就会放行那种单看自己放得下、两者相加却溢出的合成 region。容量取自 `Backend::GetMemSize(space)`；容量未知的空间（未配置 backend）
不设闸门，与 `MemoryReuse` 的做法一致。本闸门只约束**本 pass 钉住**的部分：未被开槽的 tile 仍由
ptoas 带生命周期复用地规划，那部分本 pass 无法建模。

**为什么被拒绝的外层循环会使其下方全部失去资格。** 那个循环会被复制，其 `F` 个副本会在同一个循环体内各自选取同一分配的一个 slot。ptoas 只为一次迭代中**第一个** `multi_tile_get` 推导 per-slot WAR 保护，因此 codegen 拒绝这种形状（[PTOAS#1118](https://github.com/hw-native-sys/PTOAS/issues/1118)）。

## 生成的 PTO IR

```mlir
%pipe_t_mb = pto.alloc_multi_tile valid_row = %c64_index valid_col = %c64_index
           : !pto.multi_tile_buf<!pto.tile_buf<loc=vec, dtype=f32, rows=64, cols=64, ...>, count=2>
scf.for %i = %c0_index to %c4_index step %c1_index {
  %0 = arith.remsi %i, %c2_index : index
  %t = pto.multi_tile_get %pipe_t_mb[%0] : !pto.multi_tile_buf<..., count=2> -> !pto.tile_buf<...>
  pto.tload ins(...) outs(%t : ...)
  ...
}
```

循环按原步长前进且只有一份循环体，region 不带 `addr`——由 ptoas `PlanMemory` 放置，这正是它能把第 *i* 次迭代的 load 与第 *i-1* 次的计算重叠起来的原因。

## 相关

- [`LowerPipelineLoops`](29-lower_pipeline_loops.md) —— 复制路径，仍然处理本 pass 拒绝的每一个循环
- [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md) —— 跨核 pipeline 循环上的同构做法
- [`InitMemRef`](32-init_memref.md) —— 解析合成出来的声明
- [PTO codegen](../codegen/00-pto_codegen.md) —— 把 slot 下降为 ptoas region
- [Python 语法：Slots](../language/00-python_syntax.md#slots) —— 同一声明的手写形式
