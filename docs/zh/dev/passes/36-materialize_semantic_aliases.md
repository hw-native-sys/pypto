# MaterializeSemanticAliases Pass

将**语义要求**必须是同一块分配的 buffer 归一到同一个 MemRef —— 通过把每个循环
carried 的 `iter_arg`/`initValue` MemRef 沿 yield/producer 链向下传播来实现。

## 概述

内存规划区分两种 buffer 共享：

- **强制别名（语义要求）：** 循环累加器、或原地算子的结果**必须**落在同一块
  buffer——写"下一个"值必须更新 carried buffer，否则循环无法累加。这是正确性,
  不是优化。
- **机会别名（可选）：** 生命周期不冲突的两块独立 buffer *可以*共享存储以省内存,
  属于优化。

本 pass 只处理**强制别名**。它从 [`MemoryReuse`](37-memory_reuse.md) 中拆出
（原来是那个 pass 的 "Step 0"），以便机会性的生命周期复用可以被独立跳过：

- `MemoryPlanner.DSA_RP` 保留独立分配身份，交给进程内 DSA-RP 求解器放置。
- `MemoryPlanner.PTOAS` 把生命周期复用和地址分配交给 ptoas。

**使用时机**：在 [`InitMemRef`](35-init_memref.md)（创建 MemRef）之后、所选内存
规划器之前运行。它总是运行。`PYPTO` 随后运行
[`MemoryReuse`](37-memory_reuse.md)；`DSA_RP` 在
[`AllocateMemoryAddr`](38-allocate_memory_addr.md) 中消费这些分配身份。

## API

| C++ | Python | 级别 |
| --- | ------ | ---- |
| `pass::MaterializeSemanticAliases()` | `passes.materialize_semantic_aliases()` | 函数级 |

```python
from pypto.pypto_core import passes

program = passes.materialize_semantic_aliases()(program)
```

## 算法

`InitMemRef` 已经让循环 carried 的 `iter_arg` 和 `return_var` 与 `initValue`
（累加器 buffer）共享同一 MemRef，但 yield 值的*生产者* —— 例如计算 `acc_next`
的 `tile.add` —— 仍被分配了自己的新 MemRef。本 pass 补上这个缺口：

1. **自顶向下重定向**（`TopDownRetargeter`）：对每个 `ForStmt`，取每个 `iter_arg`
   的规范 MemRef 作为目标，推送到 yield 值及其 producer 链上（跟随原地
   `output-reuses-input` 算子与 view 输入）。`IfStmt` 的返回值被推送到两个分支的
   yield，然后应用收集到的类型改写。遍历会进入 `WhileStmt` 的循环体，因此外层
   标量 while 不会使嵌套 For carry 绕过既有的重定向及活跃性检查。改变 producer 的
   目标之前，还会检查通过外层循环可重复的目标物理窗口独立读取，包括内层 For 的
   词法生命周期之外的读取。仅把 initializer 绑定为 carry 不算独立的数据观测。
   实际读取 For carry 时，会沿嵌套 For 绑定追溯到 initializer 的逻辑定义，保证每次
   进入外层 While 时仍能取得原始 seed。While carry 会沿回边更新，因此保留自身的定义。
2. **规范化 peeled accumulator phi**：按后序访问嵌套 `IfStmt`，同时识别直接的
   in-place accumulator producer，以及由分支外 accumulator seed 驱动的分支内
   loop。当且仅当一个分支是 accumulator continuation 时，把另一个分支的局部
   seed、phi result、alias 和嵌套 loop carry 重定向到 reused input 的规范 `Acc`
   allocation。accumulator loop 和 sibling seed 都必须位于各自分支内，而且 target
   在 seed 分支剩余部分必须已 dead。无论 continuation 是直接的
   `tile.matmul_acc` 还是分支内 loop，它复用的 input 及所有 bare/metadata alias 在
   `if` 之后都不能存在绕过 phi 的独立读取；否则在 continuation 不执行的 sibling
   路径上，重定向后的 producer 会覆盖仍可观察的值。后序遍历也会进入 `WhileStmt`
   的循环体。For/While 的 carry 参数及结果与 initializer 属于同一别名族，避免通过
   carry 名称漏掉对 initializer 的独立读取。即使读取位于 `if` 之前，只要外层循环
   能在分支之后再次执行该读取，也会阻止合并。重复读取索引只考虑在读取句柄定义之后
   才进入的循环，因此每轮在循环体内重新生成的累加器仍可安全合并。别名和循环区间
   索引均预先建立，分析复杂度保持 O(N log N)。
3. **规范化语义 identity chain**（`NormalizeIdentityCopyBuffersMutator`）：让 bare SSA
   alias 跟随 source，让注册了 `output_reuses_input` 的结果跟随指定的写回输入，
   让注册的首输入 buffer 别名操作跟随当前的第一个输入。目标是在重定向后统一维护
   所有必需的别名关系，而不是孤立修正某个 producer。单次前向遍历会把每个修正后的
   结果替换到下游使用中，因此写回结果后接 `transpose_view`、`reshape` 或 `slice`
   时，整条链始终使用正确的存储，无需不动点迭代。这样可在任何 memory planner
   观察 lifetime 或 PTOAS 发射 tile handle 之前消除 lowering 引入的类型漂移。

这些规则覆盖带 MemRef 的 shaped result，包括 tile 和 tensor 写回。裸 SSA 重命名
直接跟随 source 的完整类型，不复制数据；调用结果重定型保留结果的 shape、dtype、
view/distribution 元数据。显式 `output_reuses_input` 契约优先于首输入继承。
首输入 buffer 别名判定要求同时满足 `OutputMemoryInheritsInput()` 和
`IsInplaceSafe()`；只继承 memory space 并不意味着共享 buffer。
写回结果继承当前写回输入的完整 MemRef
窗口。首输入别名保留结果的窗口大小，使用当前输入的 allocation、字节偏移，加上
由操作计算的 view 相对偏移构造窗口；相对偏移只应用一次，同时继承输入的流水线
slot 身份；tile 结果还会跟随输入的 memory space。从操作重新计算相对偏移，
而不是在陈旧的绝对偏移上累加，使归一化保持幂等，并能处理同一 allocation 内部
的窗口变化与跨 allocation 变化。Buffer 别名是一种存储契约：`tile.assemble`
虽然会写数据，结果也必须跟随第一个输入。真正搬运数据、不能原地执行的
`tile.transpose` 和独立的 `tile.extract` 结果保持自己的存储。

在 legacy Tile 和 Buffer IR 两种模式下，`TensorType`/`DistributedTensorType`
的 For/While 参数和返回值都是当前 initializer 的可变 view。归一化会对齐它们的
存储，同时保留各边界结果的 shape、view 元数据和分布式窗口身份，也涵盖零次
迭代路径。独立产生的 yield tensor 保留自己的存储。Tile carry 仍使用单独的
规范缓冲区及 yield 修复规则；yield tile 需要写入 carry 缓冲区时会插入显式传输。

yield 修复前使用的归一化形式会把未决 `IfStmt` 结果加入初始集合，并沿裸别名、
写回结果和首输入别名传播未决状态，也会经过 tensor 循环参数和返回值继续传播。
所有依赖它的调用都不会被锚定到临时窗口。
控制流边界确定后，后续归一化遍历再修复整条链。Buffer IR 在索引旧值观测之前，
以及隔离循环入口之后，也使用这一延迟形式，因为分支存储尚未确定；边界确定后的
归一化无需延迟处理集合。yield 修复前后的归一化遍历见
[MemoryReuse 算法步骤 6](37-memory_reuse.md#算法)。

本 pass 跳过 orchestration 类函数，包括 graph body。其他函数即使 `Compute`
没有返回重定向改写，仍会执行别名归一化，独立修复已有的类型漂移。

## 与 codegen 的关系

### 分阶段启用的 Buffer IR 流水线

`PassContext(enable_buffer_ir=True)` 启用 Buffer IR 迁移中的存储合法化部分。
该选项默认 true；先由 `SelectBufferIRFunctions` 选择整个 kernel，
仅对选中函数启用 Buffer 特有存储改写。C++ 访问器为 `GetEnableBufferIR()`，Python 访问器为
`get_enable_buffer_ir()`。编译、IR dump 和 profiling 保留当前选项，所有内存
规划器的 JIT 缓存键都会区分该选项的值。

启用后，`MaterializeSemanticAliases` 会在 `PYPTO`、`DSA_RP` 或 `PTOAS`
规划内存之前建立显式分支目标：

1. 如果两个分支已经 yield 同一物理窗口，则保留该窗口。
2. 否则，为结果分配独立的规范目标（canonical destination）。从分支外传入并
   yield 的输入保持原存储，因此 `IfStmt` 之后仍存活的输入不会被分支写回覆盖。
3. 如果分支内的直接 producer 输出存储没有别名、没有固定分配，且注册的算子
   契约允许，则将 producer 重定向到新目标。View 和必须与输入共享存储的算子
   保留原来的存储关系。
4. 在其余分支的 yield 前插入显式 `tile.move`，并移除 producer 重定向后不再
   使用的分配。分支复制和已有的 For carry 修复会在三种规划器之前执行；
   `PYPTO` 在复用后修复新增的不一致，同时保留已声明的 phi 目标。

分支规划会在同一次前向遍历中，在相邻 phi 的存储选择之间传播必需别名，使后续
phi 看到当前窗口，而不是已废弃的窗口。For/While carry 在访问循环体之前跟随
当前 initializer，循环结果在访问之后跟随这些规范 carry，涵盖零次迭代路径。
存储句柄计数随计划中的分配变化及回滚同步更新，使可选的 producer 重定向保持保守。

支持透明作用域 `SplitAivScopeStmt` 和 `RuntimeScopeStmt` 中的末尾 yield。
传输保留在同一作用域内，紧邻 yield 之前；嵌套控制流区域的 yield 不会被替换。

例如，`if flag: yield a; else: yield b` 的 `a` 和 `b` 在分支后仍然存活时，
会新增一个结果分配，并在两个分支各复制一次。两个独立的分支内逐元素 producer
则可以直接写入同一个结果分配，无需复制。PTOAS 因此收到显式分支传输，不需要
由 codegen 再选择目标或补充复制。

新增的分支分析采用固定次数的 IR 遍历和索引查找，时间复杂度为 O(N log N)，
空间复杂度为 O(N)，不在 IR 上附加持久化别名表。累加器分支仍使用已有的受保护
合并逻辑；剩余分歧 `Acc` 分支会报错，因为不支持 Acc 到 Acc 的复制。
下文介绍循环输入隔离、并行传输、While carry 及复用后的存储验证。这些步骤
建立规范的存储边界；完整的初始化验证和异步生命周期验证仍由独立工作完成。

### 默认流水线

PTO codegen 把解析到*同一*物理 MemRef window（`base` + `byte_offset` + `size` +
pipeline-slot 元数据）的变量渲染成同一个 `tile_buf` handle，因此本 pass 之后,
循环累加器会发出原地的
`pto.tadd ins(%acc, %t) outs(%acc)`，而不是写到独立的 `%acc_next`。
`memory_planner=DSA_RP` 会把每个所得分配身份变成一个 DSA buffer；
`memory_planner=PTOAS` 则让 codegen 不带物理地址发射该身份，交给 ptoas
`PlanMemory`。
参见 [PTO 代码生成 — 由谁规划内存](../codegen/00-pto_codegen.md)。

## 说明

- view / 部分 view 保留各自的 `byte_offset`/`size` 元数据。在 `DSA_RP` 下，共享
  同一 `base` 的所有成员属于同一个物理分配；规划器整体移动该分配，并在回写时保留
  每个成员的相对偏移。仅共享 `base` 不足以建立 must-alias 关系：互不相交的 byte
  window 和不同 pipeline slot 会保持独立，直到 producer 被安全地重定向到精确的
  canonical window。
- 在默认（`PYPTO`）流水线里,本 pass 加上 `MemoryReuse` 组合起来等于原来单个
  `MemoryReuse` pass 的行为。
- `DSA_RP` 与 `PTOAS` 都跳过这里的机会性 MemRef 合并；二者都不能撤销本 pass
  建立的强制别名关系。
- 在 legacy `PTOAS` 路径中，若两个分支 yield 同一 tile 字节窗口（allocation、
  offset、size、slot 数量及索引相同），且两个 yield 与合并结果具有相同且已确定的
  存储空间（MemorySpace），合并结果保留该窗口，无需复制。语义别名会跟随修正后的
  结果，同时保留各自的类型及视图元数据。已注册的视图把相对字节偏移重新关联到
  修正后的输入，保留自身窗口大小并继承输入的流水线 slot 身份；后续完整窗口写回
  结果继续跟随该视图。复用具有多种类型别名的 codegen 句柄时，还必须满足实际声明的 SSA 类型
  匹配、两个 yield 的窗口及当前绑定一致，以及句柄定义支配（dominance）当前 if。
  若先前的合并选择了新的支配句柄，后续合并使用两个 yield 当前共同绑定的句柄，
  即使身份映射仍指向此前被拒绝的兄弟分支句柄。
  不同分支窗口沿用原有处理。
- accumulator-phi 规范化会在 lifetime planning 之前对所有 memory planner 运行。
  legacy `PYPTO` 路径在机会性 reuse 之后会再运行一次，因为 reuse 可能引入新的
  carry/phi mismatch。
- 新的 matmul accumulator 推荐使用单个
  `tile.matmul_acc(..., init_cond=...)`。为兼容已有手写 kernel，peeled
  `matmul`/`matmul_acc` 分支仍受支持，并由本 pass 规范化。

开发阶段的流水线在共享及复用后存储协调完成后、地址分配之前运行 `VerifyTileStorage`。
符号存储一致性及独立的实际地址重叠检查见[存储属性约束](99-verifier.md#tile-存储属性)。

### 显式循环与分支传输

开启 `enable_buffer_ir=True` 时，共享协调首先隔离仍会被独立观察的循环输入。
入口复制、yield 传输和快照生成前均检查目标是否支持对应的移动；
不支持的 Mat、Left、Right、Acc 同空间传输会给出早期诊断。
通过原始输入或元数据别名进行的数据读取可能需要入口复制；复制在 `ForStmt` 或
`WhileStmt` 之前执行，保留零次迭代的语义。初始 carry 窗口重叠时，会创建独立
存储。只建立元数据视图不算数据读取。只有在读取句柄的逻辑定义之后进入的外层循环，才可能跨迭代重复观察同一值。
通过循环区间索引保护此类读取，包括内层循环之前和兄弟分支中的读取。
每次外层迭代重新生成的 seed 不会仅因本次迭代中较早的读取而触发隔离；
直接别名和元数据视图继承 seed 的定义位置。循环区域之外的读取不会在其内部重复。
前向观察仍使用两种分支顺序排除互斥的兄弟分支。分析不逐层遍历祖先，保持 O(N log N)。

在重定向生产者之前，For 和 While 的初始值、iter_arg、结果及结果视图会对齐。
分支和循环 yield 使用同一个并行传输调度器：先保存所有可能与目标重叠的源，
再写入目标。交换、循环依赖、fanout 和源的部分重叠都通过 O(N log N) 的索引
查询处理；scratch 分配在地址规划前就已显式存在。复用后的协调再次使用相同
调度算法，地址分配阶段不能新增 scratch 或传输。

如果所需的同空间复制无法实现，会提前报错。例如，不能通过 Acc 到 Acc 的复制
保留仍然活跃的 Acc 输入。兼容 carry 仍可使用已有的带条件检查的累加器生产者
合并。无法判定的视图地址和同时重叠的目标窗口必须在最终 Buffer lowering 前
解决。未选中 kernel 和显式 Off 模式保持原有 Tile 行为。

默认 Buffer 路径在存储合法化前选择完整支持的 kernel，详见
[整函数选择](55-lower_tile_to_buffer.md#默认整函数选择)。
