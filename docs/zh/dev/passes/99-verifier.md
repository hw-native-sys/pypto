# IR 验证器 (Verifier)

可扩展的验证系统，通过可插拔属性验证器和诊断报告来验证 PyPTO 中间表示 (IR) 的正确性，并与 Pass 系统集成。

## 概述

| 组件 | 描述 |
| ---- | ---- |
| **PropertyVerifier (C++)** | 验证规则的基类 |
| **PropertyVerifierRegistry (C++)** | IRProperty → PropertyVerifier 工厂的单例映射，提供验证/报告 API |
| **Diagnostic** | 结构化的错误/警告报告，包含严重级别、位置和消息 |
| **VerificationError** | 验证失败时抛出的异常 |

### 关键特性

- **可插拔规则系统**：可通过自定义验证规则进行扩展
- **基于属性的验证**：选择性属性集——精确验证所需内容
- **结构性属性 (Structural Properties)**：TypeChecked、BreakContinueValid、NoRedundantBlocks、UseAfterDef、OutParamNotShadowed、NoNestedInCore、InOutUseValid、PipelineLoopValid、ArrayNotEscaped、ManualDepsOnSubmitOnly、AtomicAddDtypeValid 和 AccStorePhaseValid 由 `VerificationInstrument` 在每个 Pass 执行前后验证；在流水线启动时，`PassPipeline` 仅验证与 `GetVerifiedProperties()` 共有的轻量子集
- **双重验证模式**：收集诊断信息或在首个错误时抛出异常
- **Pass 集成**：可作为优化流水线中的 Pass 使用
- **全面的诊断信息**：收集所有问题及源码位置

## 架构

### 结构性属性 vs 流水线属性

| 类别 | 示例 | 行为 |
| ---- | ---- | ---- |
| **结构性** | TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid, AccStorePhaseValid | 始终为真。由 `VerificationInstrument` 在每个 Pass 执行前后验证；与 `GetVerifiedProperties()` 共有的子集还会在流水线启动时验证。不在 PassProperties 中声明。 |
| **流水线** | SSAForm, NoNestedCalls, HasMemRefs, ... | 由 Pass 产生/失效。按 Pass 声明的契约验证。 |

`GetStructuralProperties()` 返回 `{TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid, AccStorePhaseValid}`。这些由 `VerificationInstrument` **在每个 Pass 执行前后验证**。在**流水线启动时**，`PassPipeline::Run()` 仅额外验证与 `GetVerifiedProperties()` 共有的轻量子集（`GetStructuralProperties().Intersection(GetVerifiedProperties())`）——因此例如 `ArrayNotEscaped` 会在每个 Pass 前后验证，但不会在流水线启动时验证。由于没有 Pass 在 `required`/`produced`/`invalidated` 中声明它们，`VerificationInstrument` 将它们与 Pass 声明的属性合并，确保没有 Pass 破坏这些基本不变量。

### 验证规则系统

验证器使用**插件架构**，每个 `PropertyVerifier` 子类是一个独立的规则：

- 规则按注册顺序在所有函数上运行
- 每个规则独立运行——一个规则的失败不影响其他规则
- 规则接收 `ProgramPtr`，并在内部决定是遍历函数还是检查程序级属性
- 可以通过 `IRPropertySet` 选择性地包含规则

### 诊断系统

| 字段 | 类型 | 用途 |
| ---- | ---- | ---- |
| `severity` | `DiagnosticSeverity` | 错误或警告 |
| `rule_name` | `string` | 检测到问题的规则 |
| `error_code` | `int` | 数字错误标识符 |
| `message` | `string` | 人类可读的描述 |
| `span` | `Span` | 源码位置信息 |

### 与 Pass 系统的集成

1. **自动属性验证**：`PassPipeline` 使用 `PropertyVerifierRegistry` 在每个 Pass 执行后检查产生的属性（由 `PassContext` 中的 `VerificationLevel` 控制）。与 `GetVerifiedProperties()` 共有的轻量结构性属性子集在流水线启动时检查。详见 [Pass 管理器](00-pass_manager.md)。
2. **`VerificationInstrument`**：一个 `PassInstrument`，通过 `PassContext` 验证属性。在每个 Pass 执行前，检查 Pass 声明的 `required` 属性。在每个 Pass 执行后，检查 Pass 声明的 `produced` 属性**加上所有结构性属性**——确保没有 Pass 破坏基本的 IR 不变量。

`run_verifier()` 工具函数创建一个独立的 `Pass`，用于自定义流水线中的临时使用，但它**不是**默认优化策略的一部分。

## 内置规则

| 规则名称 | IRProperty | 用途 |
| -------- | ---------- | ---- |
| **SSAVerify** | SSAForm | 无多重赋值、无名称遮蔽、无缺失 yield、作用域违规、基数检查 |
| **TypeCheck** | TypeChecked | 类型种类/数据类型/形状/大小一致性 |
| **NoNestedCall** | NoNestedCalls | 参数、条件、范围中无嵌套调用表达式 |
| **BreakContinueCheck** | BreakContinueValid | break/continue 仅在顺序/while 循环中 |
| **UseAfterDefCheck** | UseAfterDef | 每个 Var 使用均由定义支配（参数、AssignStmt、循环变量、iter_arg、return_var） |
| **NormalizedStmtStructure** | NormalizedStmtStructure | 展平嵌套 `SeqStmts` 并解包单子节点 `SeqStmts` |
| **NoRedundantBlocks** | NoRedundantBlocks | 无单子节点或嵌套的 `SeqStmts` |
| **SplitIncoreOrch** | SplitIncoreOrch | Opaque 函数中不残留 `InCoreScopeStmt` 节点 |
| **IncoreTileOps** | IncoreTileOps | InCore 函数使用 tile 操作（无张量级操作残留） |
| **HasMemRefs** | HasMemRefs | 所有 TileType 变量已初始化 MemRef |
| **AllocatedMemoryAddr** | AllocatedMemoryAddr | 所有 MemRef 在缓冲区限制内具有有效地址 |
| **OutParamNotShadowed** | OutParamNotShadowed | Out/InOut 参数未被张量创建操作重新赋值 |
| **NoNestedInCore** | NoNestedInCore | 无嵌套 InCore 作用域（`InCoreScopeStmt` 内含 `InCoreScopeStmt`） |
| **InOutUseValid** | InOutUseValid | 作为 InOut/Out 传入用户函数调用的变量，在调用之后不得再被读取（RFC #1026）。Group 类型函数体目前跳过，待后续完善。 |
| **PipelineLoopValid** | PipelineLoopValid | 每个 `ForStmt` 上的双向不变量：`kind_ == ForKind::Pipeline` ⇔ 含有 `pipeline_stages` 属性。任一方向失败即表示 pipeline 循环格式错误。 |
| **ArrayNotEscaped** | ArrayNotEscaped | `ArrayType` 不得作为任何函数参数或返回类型出现（会通过 `TupleType` 递归检查）。`ArrayType` 是归属于所在函数的片上标量寄存器堆 / C 栈存储——让它跨越函数边界会泄漏栈指针，因此只能在函数体内创建并就地使用。 |
| **ManualDepsOnSubmitOnly** | ManualDepsOnSubmitOnly | 任何普通跨函数 `Call`（GlobalVar callee）都不得携带 `attrs["manual_dep_edges"]`——手动依赖边只存在于类型化的 `Submit::deps_` 字段中。Op call（`system.task_dummy`）作为 codegen fanin 契约保留该 attr，属于豁免。 |
| **OrchestrationReferencesResolved** | OrchestrationReferencesResolved | `FunctionType::Orchestration` 函数体内每一个非 builtin Call 必须对应到 Program 中存在的 Function。取代 codegen 端原本在生成时抛错的 `ValidateOrchestrationReferences` 遍历。 |
| **RuntimeScopesMaterialized** | RuntimeScopesMaterialized | 每个 `FunctionType::Orchestration` 函数满足 `attrs_["auto_scope"] == false`，即 `MaterializeRuntimeScopes` 插入显式 `RuntimeScopeStmt` 后打上的标记（或由用户 `@pl.function(auto_scope=False)` 声明）。编排 codegen 仅从这类节点发射 `PTO2_SCOPE()`；跳过该 Pass 会保留 `auto_scope=True` 并静默遗漏作用域。**由** `MaterializeRuntimeScopes` **产生**并列入 `GetVerifiedProperties()`，故 `PassPipeline` 在该 Pass 之后自动验证。 |
| **AssignTypeSymmetry** | AssignTypeSymmetry | 每个 `AssignStmt(var, value)` 满足 `structural_equal(var.type, value.type)`。覆盖 dtype、shape 以及 tile_view/tensor_view；此外比较 TileType 的 `memory_space`（TensorType 没有 `memory_space`）和 DistributedTensorType 的 `window_buffer`；元组赋值逐元素递归比较。**不包含** `memref_`——`structural_equal` 将其视为绑定在 Var 上的内存分配细节，由 `HasMemRefs` / `AllocatedMemoryAddr` 负责。用于捕获只修改赋值一侧类型的 Pass（例如 #1262 的 TileType memory_space、#1278 的 tile_view）。已在 `PropertyVerifierRegistry` 注册，但尚未加入 `GetStructuralProperties()`——可通过 `PropertyVerifierRegistry::verify` 或将该属性加入 `VerificationInstrument` 按需运行。 |
| **AivSplitValid** | AivSplitValid | 针对一等公民 `SplitAivScopeStmt` 区域的结构性检查，以该节点本身为准（因此嵌套 / 多模式函数会逐区域检查）。持有至少一个区域的函数即进入**手动模式（manual mode）**：此时区域对向量计算的放置具有决定权。**(a)** **任何**区域内都不得有 cube 计算——理由有两条互相独立，故不分模式一律触发：数据并行区域无法对 `matmul` 做向量切分（每个 AIV lane 只持有半块 tile），而任何区域（含任务并行）**本身就是** AIV lane 的函数体，cube 计算不属于其中。**(b)** 不得有在**切分轴**上折叠的向量归约（`tile.row_*` / `tile.col_*` / `tile.sum` / `tile.max` / `tile.min`）——它会产生每 lane 的部分结果（错误编译）。与 (a) 不同，该理由确实与切分轴相关，故 (b) 仍只在数据并行模式下生效。**(c)** `aiv_shard` / `aic_gather` 必须出现在区域内部（`tile.*` 与面向作者的 `tensor.*` 两种形式皆然）。在任务并行的 `mode=NONE` 区域中它们是**被接受的**：没有可切分的轴时，它们仍然承载检查 (f)/(g) 所要求的含义——这个值跨越了 AIC/AIV 边界——且其 `split=0` 类型推导会原样保留形状，既不折半也不拼合。**(d)** **边界内存契约**：`tile.aiv_shard` 为 `Acc → Vec`，`tile.aic_gather` 为 `Vec → Mat`。这两个算子本身*就是*跨核传输，因此操作数必须位于生产侧 lane、结果必须位于消费侧 lane；内存尚未解析前会跳过检查，故同一检查在整个窗口内都安全。该契约与模式无关——任务并行的跨越与数据并行的跨越横跨的是同样两条 lane——因此它在每个区域中都生效，`NONE` 也不例外。**(e)** 手动模式函数中，所有区域**之外**不得有 VECTOR 亲和的算子——区域既已对放置具有决定权，这样的算子既未被区域钉在 AIV lane 上、也不是 cube 计算，没有确定的归属。共有三类排除项：`tile.load` / `tile.store` 是**编译器自身**产生的区域外输出——`ConvertTensorToTileOps` 会把 tensor 级算子的 load/store 对提升到承载其计算的区域之外；以及通过 `set_core_affinity` **显式声明**了所在 lane 的算子（`system.syncall(core_type="aiv_only")`、`system.sync_set/sync_wait(core_type="aiv")`、`pld.tile.put` / `get`）——它们的 lane 本就不是推断出来的，区域也无从进一步消歧。**没有**任何区域的函数不受影响。**(f)** V->C：在区域内定义、却在区域外的 **cube** lane 上被读取的值，必须是 `tile.aic_gather` 的结果。**(g)** C->V：在所有区域之外由 cube 产出、却在区域内的**向量** lane 上被读取的值，必须经由 `tile.aiv_shard` 抵达。两个方向本来就都能下降——编译器无论如何都会发出 `split=0` 的 tpush/tpop 对——这恰恰是要检查它们的原因：手动模式的存在意义就是让作者（而非编译器）来放置 AIC/AIV 边界，而没被写出来的边界就是没人选择过的边界。(f)/(g) 在**定义侧与消费侧**都沿用 (e) 的 `tile.load` / `tile.store` 排除项，理由相同：那一对正是编译器自己提升到区域之外的。跨 C/V 的 `tile.move` 按其投递方向计为消费者，因此在 `InferTileMemorySpace` 验证点上这两项检查依然生效——此时隐式跨越已经变成了这样一次 move。**刻意不检查：**从 `mode=NONE` 区域向外的 V->C 跨越的 lane 规则——ISA 要求两条 AIV 子 lane 都参与 no-split 握手，而它们共用同一个目标槽位、没有每 lane 偏移，且两者之间没有任何仲裁，因此当两条 lane 持有不同的值时，cube 收到的是二者之一且不确定是哪一个；保证被 gather 的值 lane-uniform 是作者的职责，只作文档说明、编译器不合成。同样刻意不检查的还有只应发生一次的副作用（`pld.system.notify`）在两条 AIV 子 lane 之间的分片。区域无法表达「恰好一次」——AIV 函数带有 `dual_aiv_dispatch`，其函数体会在两条 AIV 子 lane 上都运行——而正确写法（按 `aiv_id` 分片）与错误写法（两条 lane 通知同一个 peer）在 IR 上完全相同。这两条规则都改为面向作者写入[作用域与放置](../../user/language/04-scopes.md)。**由** `OutlineIncoreScopes` **产生**，随后由 `ConvertTensorToTileOps` 与 `InferTileMemorySpace` 失效并重新产生，最后由 `LowerAutoVectorSplit` **失效**（后者擦除区域节点）。`PassPipeline` 只验证*产生*的属性（`passes.cpp`），因此正是这两次重新产生才为检查 (d)、(e)、(f)、(g) 提供了真正生效的验证点——在 `OutlineIncoreScopes` 处边界仍是不带内存空间的 `tensor.*` 形式（(d) 空转），tensor 级计算也归类为 `SHARED`，于是 (e)、(f)、(g) 找不到 VECTOR 或 CUBE 算子。被检查的算子都是带非空 `op_` 的普通 `Call`；`Submit` 会被正确跳过。**修复方式**：(a) 将 cube 算子移出区域；(b) 在非切分轴上归约，或先用 `tile.aic_gather` 汇聚回完整 tile；(c) 把边界算子移进它所属的区域内；(d) 只对 cube 产出的值做 shard——向量产出的值（`pl.load` / `pl.full`）本就位于 AIV lane，应去掉 `pl.aiv_shard`，交由隐式 affinity 门控的切分来折半；(e) 把该阶段包进独立区域——`for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):` 让两条 lane 都运行完整函数体，正是全宽计算的任务并行写法；(f) 在区域内 gather（`x = pl.aic_gather(x)`），在区域外读取 gather 后的值；(g) 在区域开头以 `pl.aiv_shard(x)` 读取。 |
| **HardSyncallOccupancy** | HardSyncallOccupancyValid | 硬（FFTS）形式的 `system.syncall` 会等待其 `core_type` 的**每一个**物理核到达 barrier，因此外层 SPMD 启动必须同时满足两个独立保证：(1) **满占用**——恰好填满这些核（`N != required` 即报错，覆盖部分占用与超占用）；(2) **`sync_start=True`**——所有 block 同时驻留，因为非 sync_start 启动可能分波次派发 block，即便满占用也会使 barrier 死锁。任一缺失都会在设备上死锁（AICore 超时 507018）并使设备残留需复位。**由** `ExpandMixedKernel` **产生**（该 Pass 解析出每个被启动 kernel 的 `FunctionType`——检查所依赖的前提）并列入 `GetVerifiedProperties()`，故 `PassPipeline` 在该 Pass 之后立即自动验证一次。覆盖所有块数可被识别的 SPMD 启动点：`FunctionType::Spmd` 函数（作用域式 `pl.spmd`）、带 `core_num` 属性的 `FunctionType::Group` 函数（`pl.cluster()` 内嵌 `pl.spmd`）、以及带 `core_num` 的 `Submit`（`pl.spmd_submit`）。启动宽度分两类：编译期字面量（对照 SoC 物理核数校验），或**启动形状查询**——`pl.system.available_cluster_count()` / `pl.system.available_aiv_count()`，可直接传入或经其绑定的 Var 传入。查询在设备上解析为本次运行自身的几何，因而天然满占用，无需再比数量；它同时是可移植写法：字面量即便与本 SoC 静态核数相符，只要运行时设备报告的可用核数不同就是错的——可用核更多时块数不足，更少时块数超额（后者会被 runtime 直接拒绝）。用错核类型的查询（纯 AIV 启动却用 cluster 数、混合启动却用 AIV 数）会被拒绝。按启动点及其直接被调：独立 **AIV** kernel + `aiv_only` → `required = GetCoreCount(VECTOR)`；独立 **AIC** kernel + `aic_only` → `required = GetCoreCount(CUBE)`；独立 AIV/AIC kernel 若用了**不匹配**的 `core_type`（含默认的 `mix`）直接报错——单核类型启动下另一种核零参与，barrier 永远无法完成；**Group**（混合 kernel）→ 任意 barrier `core_type` 的 `required = GetCoreCount(CUBE)` 个 core-group（每 group 一个 AIC，填满全部 group 即填满全部核），检查其 AIC/AIV 子 kernel 中的硬 syncall（复制的 `mix` 只报一次）。当后端未配置、`core_num` 既非字面量也非启动形状查询、或没有 SPMD 启动点启动该 kernel（占用率是启动期属性）时跳过。**修复方式**：用匹配的 `core_type` 满占用启动并设置 `sync_start=True`（推荐直接用对应的启动形状查询来定尺寸），或对部分占用改用 `pl.system.syncall(mode="soft", ...)`（GM 轮询）。 |
| **AccToGmStoreValid** | AccToGmStoreValid | 每个源 tile 位于 **Acc** 的 `tile.store`，其目标 GM tensor 的 dtype 必须是 cube fix-pipe 能够收窄到的类型：Ascend910B 与 Ascend950 均为 `INT32/FP32/FP16/BF16`（`BackendHandler::SupportsAccToGmDtype`，与 pto-isa 非量化分支的 `CheckAcc2gm` 白名单以及 ptoas `pto.tstore` 的 "acc tstore dst element type" 规则一致；该钩子仍按后端区分，因为两个集合是各自 pin 目标的独立事实）。`INT8`/`INT16` 目标会被拒绝。**由** `InferTileMemorySpace` **产生**并列入 `GetVerifiedProperties()`，因此 `PassPipeline` 在该 pass 之后立即自动验证。这个位置是关键的：合法性取决于 tile 的**内存空间**，而非用户可见的 dtype，因此无法更早判定——同一段 DSL 程序，当 matmul 结果经由 Vec（显式 `pl.cast` 在向量单元中收窄）时合法，留在 Acc 时则非法。缺少此检查时程序会一路走到 ptoas，由后者在 `pto.tstore` 校验阶段针对生成的 `.pto` 中某一行报错；而且这样一个 store 往往还会连带触发另一个看似无关的算子（int8 零初始化会下降为同样非法 dtype 的 `pto.texpands`），于是这个迟到的诊断只报出两个症状，从不指出根因。当后端未配置（无可校验的依据）或 tile 内存空间尚未解析时跳过。**修复方式**：先经向量单元收窄——用 `pl.cast` 将 matmul 结果转为目标 dtype 后再存储——或者累加到 `INT32`/`FP32` tensor 后再转换。 |
| **AtomicAddDtypeValid** | AtomicAddDtypeValid | 每个写入全局内存的原子加操作，其目标 dtype 必须是后端 store 流水能够合并的类型。该校验在一处覆盖全部原子写入点：`tile.store`、`tensor.assemble`、`pld.tensor.put`、`pld.tile.put`、`pld.tensor.remote_store` 和 `pld.tile.remote_store`。只有 `bf16` 因后端而异——pto-isa 将其下降为 `SetAtomicAdd<bfloat16_t>` -> `set_atomic_bf16`，Ascend910B（A2/A3）支持而 Ascend950（A5）不支持（`BackendHandler::SupportsBf16AtomicAdd`）；其余硬件原子加 dtype（`FP32/FP16/INT32/INT16/INT8`）在所有后端均可用，由各算子 deducer 以后端无关的方式把关。远程 put 路径与本地 store 是**同一套机制**而非并行机制：pto-isa 的 comm `TPut` 通过 VEC 暂存 tile 流式传输，并用 `TSTORE_IMPL<..., AtomicAdd>` 落盘每个分块，而 `remote_store` 直接发射 `pto.tstore`，因此一个判定式即可管住全部写入点；而 ptoas 自身没有原子 dtype 规则（`TPutOp::verify` 只检查元素类型一致性与 shape），缺少此检查时程序会一路走到生成代码中的 pto-isa `static_assert`，而那段代码并非用户所写。列入 **`GetStructuralProperties()`**，且不由任何 Pass 产生：这里不依赖任何下降结果（atomic kwarg 与目标 dtype 在用户自己的 IR 中即已存在），因此 `PassPipeline` 在 `pipeline_input` 阶段验证，错误携带原始 `Span`。当后端未配置（无可校验的依据）时跳过。**修复方式**：累加到 `FP32` tensor，在归约完成后再转换为 `bf16`。 |

| **AccStorePhaseValid** | AccStorePhaseValid | 校验 A2/A3 unit flag 的双向配对协议。每个带 `acc_phase="final"` 的 `tile.gemv`、`tile.gemv_acc` 或 `tile.gemv_bias`，都必须由 `tile.store(..., st_phase="final")` 对同一个 SSA 值（允许普通 SSA 别名）恰好消费一次；反向地，每个 final store 也必须有一个仍未消费的 final 生产者。配对必须位于同一个直线控制流区域：分支或循环体内部完整配对是合法的，但 obligation 不得跨越 if/loop/scope 边界，否则单边分支或零次循环会使 flag 状态取决于执行路径。生产者 final 下降为 check-and-set，store final 下降为 check-and-clear；漏掉 store 会遗留置位并让后续生产者卡住，缺少生产者的 final store 则会永久等待从未置位的 flag。该属性列入 **`GetStructuralProperties()`** 与 `GetVerifiedProperties()`，因此 `PassPipeline` 会在 `pipeline_input` 阶段、任何下降或内存规划之前校验带用户源码 Span 的 IR；`VerificationInstrument` 也会确保后续 Pass 维持该不变量。**修复方式**：绑定 final GEMV 的结果，并在同一直线区域内用 `pl.store(result, ..., st_phase="final")` 排空同一个值。 |

### SSAVerify

**错误类型** (`ssa::ErrorType`)：

| 错误码 | 名称 | 描述 |
| ------ | ---- | ---- |
| 1 | `MULTIPLE_ASSIGNMENT` | 变量在同一作用域中被多次赋值 |
| 2 | `NAME_SHADOWING` | 变量名遮蔽了外层作用域的变量 |
| 3 | `MISSING_YIELD` | ForStmt 或 IfStmt 缺少必需的 YieldStmt |
| 4 | `ITER_ARGS_RETURN_VARS_MISMATCH` | ForStmt/WhileStmt 中 iter_args 数量 != return_vars 数量 |
| 5 | `YIELD_COUNT_MISMATCH` | YieldStmt 值数量 != iter_args/return_vars 数量 |
| 6 | `SCOPE_VIOLATION` | 变量在其定义作用域之外被使用 |
| 7 | `MISPLACED_YIELD` | YieldStmt 出现在作用域中尾部以外的位置 |

### TypeCheck

**错误类型** (`typecheck::ErrorType`)：

| 错误码 | 名称 | 描述 |
| ------ | ---- | ---- |
| 101 | `TYPE_KIND_MISMATCH` | 类型种类不匹配（如 ScalarType 与 TensorType） |
| 102 | `DTYPE_MISMATCH` | 数据类型不匹配（如 INT64 与 FLOAT32） |
| 103 | `SHAPE_DIMENSION_MISMATCH` | 形状维度数不匹配 |
| 104 | `SHAPE_VALUE_MISMATCH` | 形状维度值不匹配 |
| 105 | `SIZE_MISMATCH` | 控制流分支中向量大小不匹配 |
| 106 | `IF_CONDITION_MUST_BE_SCALAR` | IfStmt/WhileStmt 条件必须是 ScalarType |
| 107 | `FOR_RANGE_MUST_BE_SCALAR` | ForStmt 范围必须是 ScalarType |
| 108 | `CONDITION_MUST_BE_BOOL` | IfStmt/WhileStmt 条件 dtype 必须是 BOOL |
| 109 | `TENSOR_PADDING_MISMATCH` | Tensor 填充元数据不匹配 |
| 110 | `DISTRIBUTED_WINDOW_IDENTITY_MISMATCH` | DistributedTensor 引用了不同的窗口缓冲区 |
| 111 | `TILE_VIEW_MISMATCH` | 有效 TileView 元数据不匹配 |

### NoNestedCall

| 名称 | 描述 |
| ---- | ---- |
| `CALL_IN_CALL_ARGS` | 调用表达式嵌套在另一个调用的参数中 |
| `CALL_IN_IF_CONDITION` | 调用表达式在 if 语句条件中 |
| `CALL_IN_FOR_RANGE` | 调用表达式在 for 循环范围中 |
| `CALL_IN_BINARY_EXPR` | 调用表达式在二元表达式中 |
| `CALL_IN_UNARY_EXPR` | 调用表达式在一元表达式中 |

### UseAfterDefCheck

**错误类型** (`use_after_def::ErrorType`)：

| 错误码 | 名称 | 描述 |
| ------ | ---- | ---- |
| 401 | `USE_BEFORE_DEF` | 变量在当前作用域中任何定义之前被引用 |

**作用域规则：**

- 函数参数在整个函数体内可见
- `AssignStmt`：LHS 变量在 RHS 求值后进入作用域
- `ForStmt`：`loop_var` 和 `iter_args` 仅在循环体内可见；`return_vars` 在循环结束后进入外层作用域
- `WhileStmt`：`iter_args` 在条件和循环体内可见；`return_vars` 在循环结束后进入外层作用域
- `IfStmt`：
  - **SSA/phi 形式（存在 `return_vars`）**：then/else 分支内新定义的局部变量**不**传播到外层作用域，只有 `return_vars` 在 if 结束后进入外层作用域
  - **泄漏模式（无 `return_vars`）**：then/else 分支内定义的变量**可能泄漏**到外层作用域；该形式通常由 Python 解析器在无 `yield` 的情况下生成，后续由 `ConvertToSSA`/`SSAVerify` 负责将其转换并检查合法性

## PropertyVerifierRegistry

**头文件**：`include/pypto/ir/verifier/property_verifier_registry.h`

将 `IRProperty` 值映射到 `PropertyVerifier` 工厂的单例注册表。由 `PassPipeline` 用于在 Pass 执行前/后自动验证属性。

| 方法 | 描述 |
| ---- | ---- |
| `GetInstance()` | 获取单例实例 |
| `Register(prop, factory)` | 为属性注册验证器工厂 |
| `GetVerifier(prop)` | 创建验证器实例（若未注册则返回 nullptr） |
| `HasVerifier(prop)` | 检查是否已注册验证器 |
| `VerifyProperties(properties, program)` | 验证一组属性，返回诊断信息 |
| `VerifyOrThrow(properties, program)` | 验证并在出错时抛出 VerificationError |
| `GenerateReport(diagnostics)` | 静态方法——将诊断信息格式化为可读报告 |

## C++ API 参考

### PropertyVerifier 接口

| 方法 | 签名 | 描述 |
| ---- | ---- | ---- |
| `GetName()` | `std::string GetName() const` | 返回唯一的规则标识符 |
| `Verify()` | `void Verify(const ProgramPtr&, std::vector<Diagnostic>&)` | 检查程序并追加诊断信息 |

### 结构性属性和默认属性

| 函数 | 返回值 | 描述 |
| ---- | ------ | ---- |
| `GetStructuralProperties()` | `{TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid, AccStorePhaseValid}` | 由 `VerificationInstrument` 在每个 Pass 执行前后验证的不变量（与 `GetVerifiedProperties()` 共有的子集还会在流水线启动时验证） |
| `GetDefaultVerifyProperties()` | `{SSAForm, TypeChecked, NoNestedCalls, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, TileTypeCoherence, ArrayNotEscaped}` | `run_verifier()` 的默认属性集 |
| `GetVerifiedProperties()` | `{SSAForm, TypeChecked, MixedKernelExpanded, AllocatedMemoryAddr, BreakContinueValid, NoRedundantBlocks, InOutUseValid, CallDirectionsResolved, ManualDepsOnSubmitOnly, ReturnParamsExplicit, AivSplitValid, HardSyncallOccupancyValid, IterArgCarryClassified, RuntimeScopesMaterialized, AccToGmStoreValid, AtomicAddDtypeValid, AccStorePhaseValid}` | `PassPipeline` 自动验证的轻量级属性集 |

### RunVerifier Pass 工厂

```cpp
Pass RunVerifier(const IRPropertySet& properties);
```

创建一个 `Pass`，使用 `PropertyVerifierRegistry` 验证指定的属性。

## Python API 参考

**模块**：`pypto.pypto_core.passes`

### PropertyVerifierRegistry

| 方法 | 参数 | 返回值 | 描述 |
| ---- | ---- | ------ | ---- |
| `verify(properties, program)` | `IRPropertySet, Program` | `list[Diagnostic]` | 收集诊断信息 |
| `verify_or_throw(properties, program)` | `IRPropertySet, Program` | `None` | 出错时抛出异常 |
| `generate_report(diagnostics)` | `list[Diagnostic]` | `str` | 格式化诊断信息 |

### 辅助函数

| 函数 | 返回值 | 描述 |
| ---- | ------ | ---- |
| `get_default_verify_properties()` | `IRPropertySet` | `run_verifier()` 的默认属性集 |
| `get_structural_properties()` | `IRPropertySet` | 结构性不变量属性 |

### run_verifier 函数

| 参数 | 类型 | 默认值 | 描述 |
| ---- | ---- | ------ | ---- |
| `properties` | `IRPropertySet \| None` | `None` | 要验证的属性（None → 默认属性集） |
| **返回值** | `Pass` | - | 验证器 Pass 对象 |

## 使用示例

### 基本验证

```python
from pypto.pypto_core import passes

# Verify default properties
props = passes.get_default_verify_properties()
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)

if diagnostics:
    report = passes.PropertyVerifierRegistry.generate_report(diagnostics)
    print(report)
```

### 选择性验证

```python
# Verify only specific properties
props = passes.IRPropertySet()
props.insert(passes.IRProperty.SSAForm)
props.insert(passes.IRProperty.TypeChecked)
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
```

### 禁用检查

```python
# Start from default set and remove what you don't want
props = passes.get_default_verify_properties()
props.remove(passes.IRProperty.SSAForm)
diagnostics = passes.PropertyVerifierRegistry.verify(props, program)
```

### 使用异常处理错误

```python
props = passes.get_default_verify_properties()
try:
    passes.PropertyVerifierRegistry.verify_or_throw(props, program)
    print("Program is valid")
except Exception as e:
    print(f"Verification failed: {e}")
```

### 在自定义流水线中使用

```python
# Create verifier pass (defaults to get_default_verify_properties())
verify_pass = passes.run_verifier()
result = verify_pass(program)

# Or with custom properties
props = passes.get_default_verify_properties()
props.remove(passes.IRProperty.SSAForm)
verify_pass = passes.run_verifier(properties=props)
result = verify_pass(program)
```

## 添加自定义规则

### 实现步骤

1. 继承 `PropertyVerifier`，实现 `GetName()` 和 `Verify()`
2. 创建返回 `PropertyVerifierPtr` 的工厂函数
3. 在构造函数中向 `PropertyVerifierRegistry` 注册
4. 添加 Python 绑定和类型存根（可选）

### 准则

- 使用 `IRVisitor` 系统地遍历 IR 节点
- 保持规则聚焦——一个规则检查一类问题
- 避免副作用——仅读取 IR 并写入诊断信息
- 创建描述性诊断信息，包含严重级别、规则名称、错误码、消息和 span

## 相关组件

- **Pass 系统**（`00-pass_manager.md`）：验证器作为 Pass 集成，PropertyVerifierRegistry 由 PassPipeline 使用
- **IR 构建器**（`../ir/06-builder.md`）：构造验证器验证的 IR
- **类型系统**（`../ir/02-types.md`）：TypeCheck 规则根据类型系统进行验证
- **错误处理**（`../02-error-handling.md`）：异常体系、断言宏（`CHECK`、`INTERNAL_CHECK_SPAN`）以及 `Diagnostic` / `VerificationError` 定义

## 测试

测试覆盖在 `tests/ut/ir/transforms/test_verifier.py` 中：有效/无效程序验证、基于属性的选择、异常与诊断模式、Pass 集成、诊断字段访问、报告生成、结构性/默认属性集。

UseAfterDef 专项覆盖在 `tests/ut/ir/transforms/test_verify_use_after_def.py` 中：有效程序（参数、链式赋值、for 循环体、循环后 return_var）、无效程序（先用后定义、循环变量越界、分支定义不可见于外层）、错误码/规则名验证、结构性属性成员验证。
