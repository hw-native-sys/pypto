# MemoryReuse Pass

利用依赖分析识别内存复用机会，并移除冗余的 alloc 操作。

## 概述

该 Pass 通过分析变量生命周期和依赖关系来实现内存共享。在同一内存空间中，生命周期不重叠的变量可以共享内存引用 (MemRef) 对象，从而减少内存占用。

应用 MemRef 共享后，该 Pass 还会**移除冗余的 `tile.alloc` 语句 (Statement)**——即那些不再被任何 TileType 变量引用的 MemRef 对应的 alloc 语句。

**核心要点**：

- 生命周期不重叠的变量可以复用内存
- 只有在同一内存空间中的变量才能共享 MemRef
- 生命周期通过 def-use 分析确定
- 共享完成后，已无引用的 MemRef 及其 alloc 语句会被清理

**使用时机**：这是 `MemoryPlanner.PYPTO` 的机会性复用阶段，在
[`MaterializeSemanticAliases`](32-materialize_semantic_aliases.md) 之后、
[`AllocateMemoryAddr`](34-allocate_memory_addr.md) 之前运行。
`MemoryPlanner.DSA_RP` 会跳过它，以便 DSA-RP 求解器仍能看到独立 buffer；
`MemoryPlanner.PTOAS` 也会跳过它，因为生命周期复用由 ptoas 负责。三种模式中的
循环 carry 与原地强制别名都已由 `MaterializeSemanticAliases` 建立。

## 规划器边界

`MemoryReuse` 在地址分配前先选择共享的 MemRef 身份。`DSA_RP` 则保留这些独立
身份，并在 `AllocateMemoryAddr` 中结合容量与复用惩罚联合选择地址。二者同时运行
会在 DSA-RP 评估之前删除候选方案。生命周期干涉、语义 no-alias 规则与目标
hazard 等正确性事实，在 DSA-RP 问题中仍然是硬约束。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MemoryReuse()` | `passes.memory_reuse()` | 函数级 |

**工厂函数**：

```cpp
Pass MemoryReuse();
```

**Python 用法**：

```python
from pypto.pypto_core import passes

reuse_pass = passes.memory_reuse()
program_optimized = reuse_pass(program)
```

## 算法

1. **生命周期分析**：遍历完整 IR 树（包括嵌套控制流体内的语句）通过 def-use 分析计算变量生命周期。在循环外定义但在循环内使用的变量，其生命周期会延展到循环结束（循环感知延展）
2. **干涉检查**：识别生命周期重叠的变量
3. **MemRef 共享**（全局「最大优先 + first-fit」装箱，`IdentifyReuseOpportunities`）：在每个内存空间内，按 **大小从大到小** 装箱；后续每个区间加入第一个其全部成员都能与之共享的缓冲区（生命周期不重叠 + hazard / no-alias 安全，见 `can_share`）。缓冲区的分配大小由其首个（最大）成员固定，因此之后纳入更小的成员是「免费」的 —— 且 *后定义的较大区间* 现在可以承载 *先定义的较小区间*。（此前的定义序贪心带有单向的大小门槛 `source.size >= target.size`，因此两个生命周期不相交、但较小者先定义的 tile 永远无法合并。）每个成员被重定位到的「代表」是该缓冲区的最大成员；由于 InitMemRef 会把所有 `tile.alloc` 提升到函数体头部，代表的 alloc 支配整个函数，因此代表即使定义在其部分成员之后也是安全的。由于装箱器不再按程序序处理，每个成对门槛（hazard、no-alias）都会在两个方向上检查。
4. **循环携带变量重对齐**（`AlignLoopCarriesToInitMutator`）：共享（步骤 3）只会重写由 `AssignStmt` 定义的变量（producer/init），而循环携带的 `iter_arg`/`return_var` 节点被排除在生命周期/共享映射之外、仍保留原始 MemRef。本步骤**自外向内**遍历 `ForStmt`，将每个循环的 `iter_arg`/`return_var` 重对齐到其（已复用的）`initValue` 的 MemRef，并在递归前写入 `var_remap_`，使嵌套循环能观察到已修正的外层 `iter_arg` 作为其 init。若缺少本步骤，被复用的**嵌套流水化 `matmul_acc`** 累加器会分裂到两个 Acc 缓冲区，导致步骤 5 插入非法的 `acc→acc tile.move`，被 Ascend 910B 的 ptoas 拒绝（[#1352](https://github.com/hw-native-sys/pypto/issues/1352)）
5. **累加器 if-phi 合并**（`TopDownRetargeter::CoalesceAccumulatorIfPhis`）：`LowerPipelineLoops` 会把 stage-2 的 K 循环剥离成 `if`-phi，其活跃分支是就地累加的 `matmul_acc`（位于累加器缓冲区），而失效的 `if k==0` 分支是位于*不同* Acc 缓冲区上的全新 `matmul` seed。若不处理，步骤 6 会尝试用 `acc→acc tile.move` 协调二者 —— 产生第二个同时存活的 L0C 缓冲区（溢出），且 ptoas 也会拒绝（不存在合法的 Acc→Acc `tmov`）。本步骤通过 `reuses_input` 的 producer 识别就地累加分支，并把*另一*分支的 seed 重定向到累加器缓冲区，使两个分支共享同一缓冲区、不再产生 move（符合 `mad_acc` 共享 `%dst` 的语义）。仅作用于 `Acc`；重定向是**强制的**（被拒绝的重定向会触发 `INTERNAL_CHECK`，绝不退化为 move —— 因为不存在合法的 Acc→Acc move）。它会跳过*全局* dead-at-assign 活跃性检查（否则会因 if 之后合法的 phi 消费者而误判拒绝），但仅在验证分支互斥真正所需的两个前提之后：(a) seed 的 producer 是词法上位于该分支**内部**的 `Call`（经由分支透传的 if 前值会无条件执行，从而破坏 sibling 就地分支所读取的累加器），以及 (b) **限定分支范围** 的 `IsTargetDeadAtAssign`（在所属 `if` 处停止）确认分支内 seed 之后没有对累加器缓冲区的尾部读取。任一前提不满足时，该 phi 保持未合并，步骤 6 会明确失败，而不会生成不受支持的 Acc→Acc IR
6. **Yield 修复**：修复控制流返回变量的 MemRef 不一致。Acc→Acc 拷贝不合法，因此残留的不匹配 Acc 循环携带值会触发内部错误，而不是生成 `tile.move`：
   - **ForStmt**：确保 4 个循环携带变量（initValue、iter_arg、yield value、return_var）共享同一个 MemRef。若 MemRef 不同则在 yield 前插入 `tile.move`
   - **IfStmt**：修补 return_vars 使其 MemRef 与 yield value 一致
7. **恒等拷贝缓冲区归一化**（`NormalizeIdentityCopyBuffersMutator`）：在步骤 5 重定向累加器 if-phi 后，对（已被移动的）return_var 的下游裸 `Var` SSA 恒等拷贝可能仍携带合并前的缓冲区（例如 `c_phi` 移到 `mem_acc_5` 后，`c: …mem_acc_17 = c_phi`）。`x = y` 拷贝（值为裸 `Var` 而非 `Call`）是纯重命名、必须与 `y` 共用缓冲区，因此本次单向前向遍历把这类拷贝的 LHS 重定型到 RHS 的 MemRef，并替换 LHS 的下游使用。无不一致时为空操作
8. **移除冗余 alloc**：收集仍被 TileType 变量引用的所有 MemRef，然后移除不再使用的 `tile.alloc` 语句

**复用条件**：

- 生命周期不重叠（无干涉）。当 `prev.last_use <= curr.def` 时，两个变量不重叠（即源的最后使用可以和目标的定义在同一语句，因为在同一语句内输入先于输出被消费）
- 相同内存空间
- 缓冲区大小取其**最大**成员；由于按最大优先装箱，后纳入的成员都不大于代表，故无需显式字节大小检查（复用方向也不再被限制为「先定义且更大」）
- **No-alias 守护**（算子语义）：定义复用变量的算子可以禁止其输出与某些输入操作数共享缓冲区——因为硬件在**写输出的同时读取**这些输入,原地写会中途破坏该算子。三个来源汇入同一个"每个输出禁止 alias 的输入集合"（`ForbidAliasCollector`）：
  - `not_inplace_safe()` —— 该算子无法以 `src == dst` 运行，因此其输出不得 alias **任何**输入操作数。
  - `forbid_output_alias(i)` —— 该算子对其值操作数 in-place-safe，但在写输出时读取**某个特定**操作数，因此输出不得 alias 该操作数的缓冲区。
  - **升精度 `tile.cast`**（直接在 `ForbidAliasCollector` 处理）—— 输出 dtype 比输入**更宽**时,cast 无法原地：元素 `i` 在 `i*in_bytes` 处读、`i*out_bytes` 处写,写指针超前于读指针,冲掉尚未转换的输入。降精度 / 同宽 cast 仍 in-place-safe（保留下方的跨 dtype 复用）。

  MemoryReuse 拒绝将输出放到任一禁止操作数的**物理缓冲区**上,并通过 reuse-map 合并**与** VIEW 继承（`reshape`/`slice` 共享其源的 MemRef base）解析每个操作数——因此间接到达的禁止操作数（其 owner tile 被复用到别的缓冲区,或经 view 占用）也能被捕获。对于返回 tuple 的算子，约束会从 tuple 临时变量传播到每个 tile 类型的 `TupleGetItem` 结果。

  当前声明 no-alias 约束的算子：

  | 算子 | 约束 | 为何输出不能 alias 输入 |
  | ---- | ---- | ----------------------- |
  | `tile.recip`、`tile.rsqrt` | `not_inplace_safe` | 高精度路径在写输出时读取输入**和** tmp scratch |
  | `tile.row_sum` / `row_max` / `row_min` | `not_inplace_safe` | `TROW*` 在写规约输出 `[M, 1]` 时读取整行输入 + tmp scratch |
  | `tile.mrgsort_format1` | `not_inplace_safe` | 归并排序 intrinsic 要求 `src != dst` |
  | `tile.fmod`、`tile.fmods` | `not_inplace_safe` | `TFMOD`/`TFMODS` 按 `a - trunc(a/b)*b` 计算，先用 `dst = a/b` 覆盖输出，再重新读取原始 `src0`（`a`）做最后的减法；当 `dst == src0` 时该减法读到的是已被覆盖的商，导致每个元素都算成 `0` |
  | `tile.transpose` | `not_inplace_safe` | `pto.ttrans` 非 in-place 安全：a2a3 非对齐标量路径直接从 `src` 写 `dst`（不经 tmp 暂存），`dst == src` 会边写边读损坏数据。输出始终分配新 buffer（InitMemRef 也不会为其继承输入的 buffer）。 |
  | `tile.sel` | `forbid_output_alias(0)`（mask）、`(3)`（tmp） | `TSEL` 在写 `dst` 时读取 mask + tmp scratch |
  | `tile.sels` | 感知 target | `TSELS` 始终要求 `dst` 与 predicate mask 分离，并允许复用 `src` 或 `tmp`；A2/A3 会先将 scalar 写入 `tmp`，再通过 `set_cmpmask` 读取它，之后才写 `dst`，因此 `tmp` 可以 alias `dst`，但不得与 mask/src 重叠；A5 保留但不读取 ABI 中的 `tmp`，允许其 alias 任一操作数 |
  | `tile.prelu` | 感知 target | A2/A3 的 `TPRELU` 在写 `dst` 时读取 `src`、`slope` 与 `tmp`，因此是 `not_inplace_safe`；A5 保留 ABI 要求的 `tmp` 操作数但不读取它，所以 `dst` 可复用 `tmp`，但不可复用仍参与运算的 `src`/`slope` |
  | `tile.{row,col}_expand{,_mul,_add,_sub,_div}` | `forbid_output_alias(1)`（广播向量） | 行/列向量（arg 1）会被**每个**输出行/列重读,输出若 alias 它则在第一行/列后被覆盖 |
  | `tile.cast`（仅升精度） | 输出 ≠ 输入缓冲区（条件式,在 `ForbidAliasCollector`） | 更宽的输出写指针超前于读指针（见上） |

- **流水线 stage 守卫**（容量门控，仅针对复制路径）：`pl.pipeline(stage=F)` 将循环体复制 `F` 份以实现 ping-pong，`LowerPipelineLoops` 给每个副本产生 tile 的 `Call` 打上 `pipeline_membership` `(group, stage)`（见 [28-lower_pipeline_loops.md](28-lower_pipeline_loops.md)）。本守卫只作用于这条复制路径。在 `memory_planner=PTOAS` 下，合格循环会先被 [`LowerPipelineToSlots`](27-lower_pipeline_to_slots.md) 接手：它给每个 load 一个以 `iv % F` 索引的 `pl.MemRef(..., slots=F)`，循环体只有一份——没有副本，也没有 `pipeline_membership`，而且该流水线本就跳过 MemoryReuse。以下描述仅适用于被那个 pass 拒绝的循环。`F` 份副本在调度器下并发执行，因此它们程序序不相交的生命周期**不是**安全的复用信号——把并发副本合并到同一块缓冲会注入一条虚假的写后读（write-after-read），使各 stage 串行化（即 #1475 的 cube matmul 操作数坍缩）。MemoryReuse 因此在**每个内存空间**（包括 L0 matmul 空间 Left/Right/Acc/Bias，且无论 tile 是 load 还是 `tile.move` 的结果）都把并发副本保持在**不同的缓冲**中，最多到**可负担的双缓冲深度** `F_g = min(depth_g, ⌊C_s / slot_g⌋)`：stage `k` 的副本落在残数 `ordinal(k) mod F_g`（**稠密**的 stage 序号，因此稀疏 stage ID 如 `{0, 2}` 不会因 `2 mod 2 == 0 mod 2` 而错误合并），因此并发副本永不共享（放得下时是完整 ping-pong，空间紧张时尽量分散）。分离是否放得下由**精确的按空间分配器足迹**（`SpaceFootprint`，与 `AllocateMemoryAddr` 共享——按构造保证一致）决定，而非估算。当某空间在所有 group 的可负担深度下仍然溢出时，采用**优雅的跨 group 削减**：将某个 group 的深度降低一个残数并重新打包（按 `MaxRelief` 启发式选择 group——优先释放最多字节，平局取最小 group id）；若削减耗尽，则整体回退到 **legacy 重新打包**（`force_legacy`），从而绝不会在 legacy 打包本可放下的情况下溢出。容量未知的空间（未配置 backend）使用 legacy 判据，因此容量门控路径绝不比 legacy 更差。当门控把某个 group 的深度降到其请求的 `stage=` 之下（或某空间触发 legacy 回退）时，MemoryReuse 通过统一诊断通道发出诊断——一条 `PH-MR-001` **性能提示**（回退情形则为 **warning**），指出请求深度与实际深度以及修复方式（把每 stage 的 tile 缩小到 `≤ C_s / stage`，或把 `stage=` 降到能放下的值）——因此容量导致的串行化绝不会静默发生。复用决策完成后，MemoryReuse 会剥离已消费的 `pipeline_membership` attr，使其不会带到下游 pass 或 codegen。
- `DSA_RP` 跳过 `MemoryReuse`；它把请求的流水线深度表示为硬分离，并在
  `AllocateMemoryAddr` 中执行仅放宽流水线意图的回退。

**不再有 shape / dtype / TileView 兼容性门槛**：共享同一物理 MemRef 的 tile 可以携带**不同**的 shape、dtype 或 `TileView` 属性。PTO codegen 为每个 tile 绑定一条 per-variable 的 `alloc_tile`，因此每个别名都以各自的静态 shape / dtype / layout / `valid_shape` 声明共享基址。这允许例如：

- 跨 dtype 复用 —— BF16 tile 复用已死亡的 FP32 tile 的缓冲区（例如跨 `tile.cast`）；
- `tile.fillpad` 输出复用其输入，以及两个 `pad` 不同的 fillpad 输出共享一个缓冲区；
- N-D tile 在 `valid_shape` 不同的情况下共享缓冲区（各自在自己的 `alloc_tile` 上保留各自的 `valid_shape`）；
- L0 cube 输入 `Left` / `Right` 中 shape 不同的子 tile 共享同一槽位（例如 fused-attention QK 的 `Right` `[k, SEQ]` 被 PV 的 `Right` `[k', HEAD]` 复用，将 L0B 峰值减半 —— issue #1595）。

  早期版本以 `AreTileTypesCompatible`（shape / dtype / view 匹配，外加一个狭窄的 L0 字节复用例外）作为门槛；该门槛已移除。对读-写同体（read-while-write）算子的正确性现由上面的 no-alias 守护精确处理，而不再依赖粗粒度的整块匹配。

**Alloc 清理**：

MemRef 共享完成后，部分 MemRef 对象变为无引用状态（其变量现在指向不同的共享 MemRef）。该 Pass 遍历周围的 `SeqStmts`，移除所有左值 MemRef 指针不在仍使用集合中的 `tile.alloc` `AssignStmt`。

## 声明式分配

复用是机会主义的：任意两个生命周期不重叠的 tile 都是合并到同一块 buffer 的候选。就容量而言
这是正确的默认策略，但它并非没有代价——共用一块 buffer 的两个 tile 之间被强加了一条源码从未
要求的 WAR 依赖，硬件因此必须串行执行本可由调度器重叠的工作。

在 tile 注解中引用一个声明好的 `pl.MemRef("name")`，作者即可把某块分配从 packer 手里收回。
InitMemRef 将其物化为 `tile.alloc(..., pinned=True)`（见
[InitMemRef](31-init_memref.md#声明式分配)），本 pass 随后视其为封闭的：pinned 区间在
first-fit 打包中自开一个槽位，之后每个候选在扫描槽位时都会跳过它。（隔离是打包循环内的
per-slot 标记，而非又一个 `can_share` 门——`can_share` 是 O(M²) 打包的最内层，且每次 shed
都会重跑，所以该判定按区间解析一次，而不是按配对解析。）具体而言：

- 作者绑定到**不同** buffer 的 tile 永远不会被合并，无论其生命周期多么不相交——保持彼此独立
  正是绑定的目的。
- 作者绑定到**同一** buffer 的 tile 在 InitMemRef 阶段已共享同一 base，并保持如此。
- 未绑定的 tile 照旧参与打包，且绝不会被塞进声明式分配。

代价由作者承担：pin 是用容量换并行度，pin 过头的 kernel 会在 `AllocateMemoryAddr` 处硬报错，
而不是被静默合并回去。

**重叠检查。** 独立绑定到**同一槽位**的两个 tile 不得同时存活——那不是复用，而是后写的数据
摧毁了先前 tile 仍需要的内容。该检查按**槽位**而非按分配进行：`pl.MemRef(slots=N)` 声明中位于
*不同*槽位的两个 tile 本来就应该同时存活（这正是 ping-pong 的用途），只有落在同一槽位的 tile
才会互相破坏。下标是运行期表达式时（`l0c[i % 2]`）没有静态的槽位归属可比，该检查被跳过——轮转
的正确性由作者负责——而与其他分配之间的隔离依然成立。该检查由本 pass 负责，因为这里才有生命周期信息
（`ComputeLifetimes`）；判据与 packer 自己的 `var_overlap` 一致，因此**首尾相接是允许的**
（一个 tile 的最后一次读，可以正是产出下一个成员的那条语句）。通过继承而非绑定落到该 buffer
上的 tile——view、原地结果、裸 SSA 别名——被排除在外：它们与源 tile 是同一份数据，重叠是预期
行为。

由于隔离保证由本 pass 提供，而 ptoas 会整体替换本 pass，因此 `memory_planner=PTOAS` 下的
单参数 `pl.MemRef(...)` 会在 InitMemRef 处**直接报错**，而不是"分配了但不隔离"地静默生效——只分配不
隔离，恰好会把作者写这条绑定所要避免的合并原样还回去。
`memory_planner=DSA_RP` 同样会跳过本 pass，但会在分配问题中保留该契约：
每个声明式分配都会与同一内存空间中的其他所有分配建立硬分离；若同一声明中独立绑定的
成员同时存活，则在求解前直接报错。

## Ascend910B load + tpop_from_aic 危害

在 `SplitMode` 非 `None` 的 Ascend910B AIV 函数中，如果某个 writer **同时**消费 `tile.load` 的结果（或其合法 view 派生）**和** `tile.tpop_from_aic` 的值，则它的输出不能与该 load 结果落在同一块物理 buffer 上。让 writer 的输出原地复用 load buffer 会在该硬件上产生**静默的错误结果**。

MemoryReuse 掌管所有 buffer 合并决策，因此它从源头上阻止这种危害共享的形成，而不依赖后续的拆分。当 guard 生效时，复用决策在以下条件**同时满足**时被阻止：

- writer 的定义 op 消费了 `tile.tpop_from_aic` 的值，**且**
- 它将要原地复用的那个 buffer 成员（其 last use 正是该 writer 的定义语句）是 load 派生的。

该 guard 由 `BackendHandler::RequiresSplitLoadTpopWorkaround()`（仅 Ascend910B 为 true）以及函数为 split-AIV 这两个条件门控；在其他任何 backend / 函数类型下输入集合为空，复用行为不变。writer 仍可自由复用任何**非** load buffer —— 只有 load + tpop 的原地组合会被拒绝。（该 guard 此前由独立的 `LegalizePTOBufferReuse` pass 在事后拆分 buffer 来实现，现已并入 MemoryReuse。）

## 示例

### MemRef 共享与 Alloc 清理

**之前**（InitMemRef 之后）：

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
mem_vec_2: MemRefType = tile.alloc(Vec, -1, 16384, 2)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# tile_a last use ↑
tile_c: Tile[[64, 64], FP32, memref=mem_vec_2] = tile.load(...)
# ]
```

**之后**（tile_c 复用了 tile_a 的 mem_vec_0，mem_vec_2 的 alloc 被移除）：

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
# mem_vec_2 alloc removed — no longer referenced
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
tile_c: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
# tile_c now shares mem_vec_0 with tile_a
# ]
```

### 生产者-消费者复用

当某个变量的最后一次使用恰好发生在定义另一个新变量的同一条语句上（生产者-消费者关系）时，
新变量可以复用旧变量的内存，因为输入在输出写入之前被读取：

```python
# Before:
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.create(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.muls(tile_a, 0.0)
# tile_a.last_use == tile_b.def → reuse allowed

# After:
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.create(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.muls(tile_a, 0.0)
# tile_b reuses mem_vec_0
```

### 生命周期重叠（不可复用）

**之前/之后**（无变化——alloc 语句保留）：

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.load(...)
tile_c: Tile[[64, 64], FP32, memref=...] = tile.add(tile_a, tile_b)
# tile_a and tile_b are both live here → cannot reuse
# ]
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`

```cpp
Pass MemoryReuse();
```

**实现文件**：`src/ir/transforms/memory_reuse_pass.cpp`

- `LifetimeAnalyzer` 遍历完整 IR 树计算变量生命周期（包括嵌套控制流）
- `ComputeLifetimes` 构建 MemRef 共享组和生命周期区间
- `IdentifyReuseOpportunities` 查找复用候选
- `ApplyMemRefSharing` 通过 `MemRefSharingMutator` 更新 MemRef 指针
- `TopDownRetargeter::CoalesceAccumulatorIfPhis` 通过把失效分支的 seed 重定向到就地累加器缓冲区，合并被剥离的循环携带累加器 `if`-phi，使 `YieldFixupMutator` 不再产生非法的 `acc→acc tile.move`（见算法步骤 5）
- `YieldFixupMutator` 修复 ForStmt/IfStmt 在复用后的 yield/return_var MemRef 不一致（合法时插入 `tile.move`；拒绝残留的 Acc→Acc 不一致）
- `NormalizeIdentityCopyBuffersMutator` 协调累加器 if-phi 合并后 LHS/RHS 缓冲区不一致的裸 `Var` SSA 恒等拷贝（见算法步骤 7）
- `UsedMemRefCollector` 收集共享后仍被引用的 MemRef 指针
- `RemoveUnusedAllocStatements` 从 `SeqStmts` 中过滤掉冗余的 `tile.alloc` 语句

**Python 绑定**：`python/bindings/modules/passes.cpp`

```cpp
passes.def("memory_reuse", &pass::MemoryReuse, "Memory reuse optimization");
```

**测试**：`tests/ut/ir/transforms/test_memory_reuse.py`

- 测试非重叠生命周期的 MemRef 共享复用
- 测试重叠生命周期不复用
- 测试内存空间隔离
- 测试字节大小兼容性
- 测试跨 dtype / 跨 `TileView` 复用（现已允许：BF16↔FP32、fillpad 输出↔输入、`valid_shape` 不同）
- 测试 no-alias 守护（`TestForbidOutputAlias` + `TestInplaceOps`），上表每条约束一个用例：
  - `tile.recip` / `tile.rsqrt` / `tile.row_sum` —— 输出不得 alias 输入（`not_inplace_safe`）
  - `tile.sel` —— 输出不得 alias mask / tmp（`forbid_output_alias`）
  - `tile.sels` —— 输出始终不得 alias mask；A2/A3 与 A5 均允许 tmp/输出 alias，但 A2/A3 backend 仍会拒绝 tmp 与 mask/src 重叠
  - `tile.prelu` —— A2/A3 输出不得 alias 任一输入；A5 输出仅可 alias 未使用的 `tmp`
  - `tile.col_expand_mul` —— 输出不得 alias 广播向量
  - 升精度 `tile.cast` —— 输出不得 alias（更窄的）输入
  - 经 VIEW 间接到达的禁止操作数也被遵守（物理缓冲区解析）
- 测试切片操作的 MemRef 共享保持
- 测试冗余 alloc 语句移除
- 测试控制流生命周期分析（ForStmt 内嵌套 IfStmt、分支变量共享）
