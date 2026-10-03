# LowerPipelineToSlots Pass

把合格的 `pl.pipeline(N, stage=F)` 循环下降为现有 slot MemRef。默认 PyPTO 路径保留 unroll + reorder；`enable_software_pipeline=True` 开启固定地址的显式预取。原有 PTOAS planner 轮转路径单独保留。

## 确定性的嵌套数据流规则

同一生产者调度器处理带或不带子 pipeline 的单向 FIFO 循环。
辅助 GM 加载采用声明作用域的 stage 和 S-1 提前量；固定 unroll 工作没有额外的两槽特化。

对于可证明安全的嵌套流水，每个分配在各层 pipeline 作用域都有版本坐标。
外层 stage=A、内层 stage=B 时预留 A*B 个版本；再嵌套 stage=C 时预留 A*B*C。
版本数量由 stage 决定，不由子循环迭代次数决定；外层自身的分配仍只有 A 个版本。

InitMemRef 之后，MemoryReuse 在各层统一使用共享生命周期分析和算子别名契约决定复用。计算产生的 store 结果可覆盖
最后使用且兼容的输入；store 延长该存储的异步生命周期，不强制独立 output ring。
必须共享的 view 保持源身份；无法安全复用输入的结果保留普通分配，不由调度 pass 创建 output ring。

物理布局把外层坐标放在最快变化的维度：两层为
`(child_iteration % B) * A + (root_iteration % A)`，更深层使用 stage 的混合进制余数。
PyPTO 预留完整乘积区域。运行时或带条件的调度使用固定地址父 buffer 与显式 slot subview，
保留全部物理槽及坐标，不依赖 PTOAS 私有事件边界契约。

无外层条件保护的定长子循环使用连续序号
`parent * child_trip_count + child_iteration`。先预载 `S-1` 个输入，随后
在每个子 phase 发起超前 `S-1` 次迭代的输入，包括跨父迭代边界的输入。
未来加载保留自己的地址与有效性条件，并检查未来父迭代未越界。提前量还受同一物理槽
两次使用的最短距离限制，避免短子循环覆盖仍存活的父版本；分配深度不缩减。
父层计算顺序不变。无法证明可跨越外层条件时，保留局部预取。

InitMemRef 物化存储，MemoryReuse 决定复用，AllocateMemoryAddr 检查最终容量。
调度 pass 不维护另一套容量预算，也不降低声明的槽数。

## 原有轮转方案概述

`pl.pipeline(N, stage=F)` 表达的是乒乓缓冲的诉求。[`LowerPipelineLoops`](31-lower_pipeline_loops.md) 用**复制**来兑现：把循环体复制 `F` 份，每份都有全新的定义变量，于是各份的 tile 是彼此独立的 MemRef，`MemoryReuse` 不允许把它们合并。这条路可行，但代价是 `F` 倍的代码量、一套静态/动态余数派发，以及为了隔开各份副本而存在的 `pipeline_membership` 机制。

本 pass 用 ptoas 本来就认识的形式表达同一个意图。`pl.MemRef(name, slots=F)` 的含义正是**"一个分配、F 个等大 slot、本次使用取第 k 个"**（见[槽位](../language/00-python_syntax.md#槽位)），而 PTO codegen 恰好把它下降为 `pto.alloc_multi_tile` + `pto.multi_tile_get`。于是循环只保留**一份**循环体，每个按 stage 私有的缓冲变成合成声明的第 `iv % F` 个 slot：

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

**原有轮转方案不需要新的 IR op 或额外开关。** 合成出来的 MemRef 与作者手写的声明形状完全一致，因此 [`InitMemRef`](34-init_memref.md) 走同一条路径解析它，codegen 也分辨不出这个轮转是作者写的还是编译器推导的。

由于边界、步长和 `iter_args` 都没有改动，不存在需要派发的余数——动态 trip count 完全不需要特殊处理。

**依赖属性**：SSAForm、SplitIncoreOrch、IncoreTileOps、TileOps2D、TileMemoryInferred、NormalizedStmtStructure。

**流水位置**：在 [`SkewCrossCorePipeline`](29-skew_cross_core_pipeline.md) 之后，紧接 [`LowerPipelineLoops`](31-lower_pipeline_loops.md) 之前。足够晚，内存空间已推断、tile 结构已定型；又足够早，`InitMemRef` 还没有给这些 tile 分配编译器自己的 MemRef。

对于有界局部嵌套 pipeline，同一规划器保留各子循环的 stage，并专门化为共享的
仿射槽流。详见[嵌套作用域规则与限制](29-skew_cross_core_pipeline.md#嵌套局部-pipeline-作用域)。

## 两个 pass 是互补关系，不是二选一

两者都会执行，且按此顺序。本 pass 只接手能证明安全的循环并将其降级；**凡是它不接手的循环都保持 `ForKind::Pipeline`**，由 `LowerPipelineLoops` 照旧复制。不会因为本 pass 的存在而让任何循环失去乒乓——matmul L0 stage 循环、嵌套 pipeline、形状特殊的循环都仍然走复制路径。

这与 [`SkewCrossCorePipeline`](29-skew_cross_core_pipeline.md) 的做法同构：它处理跨核 pipeline 循环，其余原样留下。

## 可选的 PyPTO 软件流水

按编译开启新调度，DSL 保持不变：

```python
import pypto.language as pl
from pypto.runtime import RunConfig

@pl.jit
def vec_add(x: pl.Tensor[[64, 1024], pl.FP32],
            y: pl.Out[pl.Tensor[[64, 1024], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.pipeline(64, stage=3):
            y[i:i + 1, :] = pl.add(x[i:i + 1, :], 1.0)
    return y

vec_add.compile(config=RunConfig(enable_software_pipeline=True, codegen_only=True))
```

`PassContext([], enable_software_pipeline=True)` 和
`ir.compile(..., enable_software_pipeline=True)` 使用同一配置。context 默认值为 `False`；
compile/RunConfig 未指定时继承当前 context。已有 context 时显式传入编译选项会报错。
生效的开关参与 JIT 缓存键，并通过 profiling 和 pass dump 的嵌套 context 传递。
首版要求 PyPTO 内存规划器和现有 Tile IR 流水线，不支持开发中的 Buffer IR 路径。

单条局部数据流有 `S` 个槽位时，预取距离为 `P = S - 1`：

1. 预加载逻辑迭代 `0 .. P-1`。
2. 对 `t = 0 .. N-P-1`，先把 `t+P` 加载到 `(t+P) % S`，再使用 `t % S` 计算、写回 `t`。
3. 计算、写回剩下的 `P` 次迭代。

变换复用普通语句、循环和 `MemRef` 槽位元数据，通过 `tile.create` 获取已填充存储的句柄，
不增加 IR 节点或算子。主循环降为顺序循环，后续 unroll 和 IO reorder 不再改动其调度。
槽位下标局部生成 `arith.remui`，不改变一般标量取模的语义。计算结果复用输入槽位时，
codegen 也复用已定义的 tile 句柄；分别发射等价的槽位取模 SSA 可能妨碍 PTOAS 推导动态 event 调度。

PyPTO 预留完整 region 并分配基地址。在 PTOAS level3 下，
静态直线调度生成 `pto.alloc_multi_tile addr = ...` 和 `pto.multi_tile_get`。
运行时或带条件的调度使用 `pto.alloc_tile addr = <base>` 和 `pto.subview`，保留完整槽分配及已证明的
静态有效维度。两者都不使用 PTOAS 内存规划；最终容量检查在 AllocateMemoryAddr 完成。
单层与嵌套流水在 InitMemRef 之后，由 MemoryReuse 统一使用既有算子契约和共享生命周期分析：

- GM load 拥有 `S` 槽输入区域。
- 原地安全的计算可以继续使用最后一次读取的输入槽。INT32 到 FP32 等同元素位宽转换、
  完整的一维 reshape 别名复用 `TileBufSignature` 的物理布局证明。仍存活的别名、禁止别名
  的参数或不兼容布局会阻止复用。结果的整个别名族必须适合完整槽；较窄的 view 保留普通存储。
- MemoryReuse 决定末端 store 源是否复用输入版本；否则使用普通分配，不由调度 pass 创建 output ring。
  同时存在 store 和 compute 消费者的值仍不支持。
- 独立中间量及 reduction scratch 保持普通分配。常驻的完整 Vec tile 可以留在循环外。
  可写参数必须在 registry 声明为 workspace，且由未绑定的 `tile.create` 独立分配；
  同时作为循环数据读取的 workspace 会被拒绝。已有 reduction 复用原有的
  `LaneInvariantArg::Scratch` 契约。
  这些分配均计入容量检查。

例如，accumulator load 可经 cast、ReLU、列广播乘法原地复用同一槽；row-sum 的输出和
scratch 独立；最后的乘法可复用已最后使用的 KV-scale 槽。三槽 128x64 score epilogue
只需 accumulator 与 scale 两个 ring，无需复制所有中间量。

强制别名 view 由 `InitMemRef` 继承源分配。Codegen 只在已证明时使用静态有效维度，
用现有 `pto.treshape` 表达槽位 view，并在每个直线语句区域共享规范槽位下标 SSA。
不会把动态或部分有效维度替换成完整形状。

局部路径支持有效范围完整的二维平坦 Vec tile、2–4 个 stage、非负常量起点和正常量步长。
动态边界使用受保护的预载与归一化循环；有溢出风险的路径保留原调度。
有界嵌套作用域遵循前述存储和预取规则。不支持的副作用、真正的数据递推和无法证明的
别名关系会使整条循环回退。单层静态循环的次数小于预载距离时也保留既有路径。

前置的[联合阶段](29-skew_cross_core_pipeline.md#单向-fifo-的联合流水线)
使用标准 GM-entry pipe 操作处理合格的单向 AIC→AIV 循环。
消费者保留单份外层循环体和动态槽位选择；一般反馈和多消息调度不在本优化范围内。

**性能限制：** 原版 PTOAS 可能对槽位互不重叠的动态 subview 仍插入保守同步。
它不会移除必要同步，但可能阻止 MTE2 与向量计算重叠，见
[PTOAS #1587](https://github.com/hw-native-sys/PTOAS/issues/1587)。
因此，开启该变换不保证性能提升。

## 原有 PTOAS planner 轮转

新开关关闭时，`memory_planner=PTOAS` 保留本文原有的单循环体 `iv % F` 轮转。
默认 PyPTO planner 且新开关关闭时，本 pass 保持所有循环不变。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::LowerPipelineToSlots()` | `passes.lower_pipeline_to_slots()` | 函数级 |

```python
from pypto import passes
with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
    result = passes.lower_pipeline_to_slots()(program)
```

## 原有轮转方案的行为

对于 `F > 1` 且通过下列全部门槛的 `ForStmt(kind == ForKind::Pipeline, attrs["pipeline_stages"] == F)`：

1. 把每个候选 tile 的 `TileType` 重绑到一个新的 pinned `MemRef(name, slots=F)`，`slot_index = iv % F`。定义与其全部使用点在一趟遍历中改写完成——IR 处于 SSA 形式，任何使用都不会出现在其定义之前。
2. `kind_` 变为 `ForKind::Sequential`，同时剥掉 `pipeline_stages`。两者始终同进同出，故 `PipelineLoopValid` 不变式（`kind == Pipeline` 当且仅当 `pipeline_stages` 存在）在每个可观察状态都成立。
3. 其余一概不动：不新增、不删除、不重排任何语句，`start` / `stop` / `step` / `iter_args` 全部保持原样。

`F == 1` 要么是用户手写的 `pl.pipeline(stage=1)`，要么是上一次 `LowerPipelineLoops` 留下的标记。两种情况都不需要多缓冲，因此 (kind, attr) 这一对保持完整，留给 `CanonicalizeIOOrder` 作用域使用。

### 哪些 tile 取 slot

循环体顶层、且作者尚未自行绑定的**全部** `tile.load` 结果。

- **只取 load。** 需要保持私有的是 load 缓冲——这样第 `i+1` 次迭代的预取才能与第 `i` 次的计算重叠；计算中间结果可以合并。这与 [`MemoryReuse`](36-memory_reuse.md) 通过 `pipeline_load_tiles` 划出的界线一致。给**所有** tile 都开 `F` 份私有缓冲会在真实 kernel 上撑爆片上预算——`stage=4` 的 RMSNorm 需要 `4 x 67 KB > 188 KB` UB。`tile.read` **不在其中**：它返回的是标量元素而非 tile，没有可轮转的缓冲。
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

**为什么被拒绝的外层循环会使其下方全部失去资格。** 那个循环会被复制，其 `F` 个副本会在同一个循环体内各自选取同一分配的一个 slot——这是 PTO codegen 在该 planner 下拒绝的形状（见[每轮迭代只用一个槽位](../codegen/00-pto_codegen.md#多槽位声明映射为一块-ptoas-区域ptoas-模式)）。

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

原有轮转循环按原步长前进且只有一份循环体，region 不带 `addr`，由 PTOAS 放置。保留 region 和槽位身份才能推导逐槽位同步，实现迭代间重叠。

## 相关

- [`LowerPipelineLoops`](31-lower_pipeline_loops.md) —— 复制路径，仍然处理本 pass 拒绝的每一个循环
- [`SkewCrossCorePipeline`](29-skew_cross_core_pipeline.md) —— 跨核 pipeline 循环上的同构做法
- [`InitMemRef`](34-init_memref.md) —— 解析合成出来的声明
- [PTO codegen](../codegen/00-pto_codegen.md) —— 把 slot 下降为 ptoas region
- [Python 语法：槽位](../language/00-python_syntax.md#槽位) —— 同一声明的手写形式

### 动态边界调度

符合条件的动态循环保留 guarded preload 和一条归一化循环体。
PyPTO 内部调度属性不会作为 assembler 契约输出。
运行时或带条件的循环使用固定地址父 buffer 与显式 slot subview，保留槽数、固定分配基址及已证明的静态有效维度。
PTOAS 为这些标准操作插入同步，不需要私有能力开关；A5 保留既有 fallback。
保守同步可能使传输串行，因此 overlap 和数值正确性需要分别验证。
