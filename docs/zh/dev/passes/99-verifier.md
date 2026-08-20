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
- **结构性属性 (Structural Properties)**：TypeChecked、BreakContinueValid、NoRedundantBlocks、UseAfterDef、OutParamNotShadowed、NoNestedInCore、InOutUseValid、PipelineLoopValid、ArrayNotEscaped、ManualDepsOnSubmitOnly 和 AtomicAddDtypeValid 由 `VerificationInstrument` 在每个 Pass 执行前后验证；在流水线启动时，`PassPipeline` 仅验证与 `GetVerifiedProperties()` 共有的轻量子集
- **双重验证模式**：收集诊断信息或在首个错误时抛出异常
- **Pass 集成**：可作为优化流水线中的 Pass 使用
- **全面的诊断信息**：收集所有问题及源码位置

## 架构

### 结构性属性 vs 流水线属性

| 类别 | 示例 | 行为 |
| ---- | ---- | ---- |
| **结构性** | TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid | 始终为真。由 `VerificationInstrument` 在每个 Pass 执行前后验证；与 `GetVerifiedProperties()` 共有的子集还会在流水线启动时验证。不在 PassProperties 中声明。 |
| **流水线** | SSAForm, NoNestedCalls, HasMemRefs, ... | 由 Pass 产生/失效。按 Pass 声明的契约验证。 |

`GetStructuralProperties()` 返回 `{TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid}`。这些由 `VerificationInstrument` **在每个 Pass 执行前后验证**。在**流水线启动时**，`PassPipeline::Run()` 仅额外验证与 `GetVerifiedProperties()` 共有的轻量子集（`GetStructuralProperties().Intersection(GetVerifiedProperties())`）——因此例如 `ArrayNotEscaped` 会在每个 Pass 前后验证，但不会在流水线启动时验证。由于没有 Pass 在 `required`/`produced`/`invalidated` 中声明它们，`VerificationInstrument` 将它们与 Pass 声明的属性合并，确保没有 Pass 破坏这些基本不变量。

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
| **RuntimeScopesMaterialized** | RuntimeScopesMaterialized | 每个 `FunctionType::Orchestration` 函数满足 `attrs_["auto_scope"] == false`，即 `MaterializeRuntimeScopes` 插入显式 `RuntimeScopeStmt` 后打上的标记（或由用户 `@pl.function(auto_scope=False)` 声明）。编排 codegen 仅从这类节点发射 `SIMPLER_SCOPE()`；跳过该 Pass 会保留 `auto_scope=True` 并静默遗漏作用域。**由** `MaterializeRuntimeScopes` **产生**并列入 `GetVerifiedProperties()`，故 `PassPipeline` 在该 Pass 之后自动验证。 |
| **AssignTypeSymmetry** | AssignTypeSymmetry | 每个 `AssignStmt(var, value)` 满足 `structural_equal(var.type, value.type)`。覆盖 dtype、shape 以及 tile_view/tensor_view；此外比较 TileType 的 `memory_space`（TensorType 没有 `memory_space`）和 DistributedTensorType 的 `window_buffer`；元组赋值逐元素递归比较。**不包含** `memref_`——`structural_equal` 将其视为绑定在 Var 上的内存分配细节，由 `HasMemRefs` / `AllocatedMemoryAddr` 负责。用于捕获只修改赋值一侧类型的 Pass（例如 #1262 的 TileType memory_space、#1278 的 tile_view）。已在 `PropertyVerifierRegistry` 注册，但尚未加入 `GetStructuralProperties()`——可通过 `PropertyVerifierRegistry::verify` 或将该属性加入 `VerificationInstrument` 按需运行。 |
| **AivSplitValid** | AivSplitValid | 针对一等公民 `SplitAivScopeStmt` 区域的结构性检查，以该节点本身为准（因此嵌套 / 多模式函数会逐区域检查）。持有至少一个区域的函数即进入**手动模式（manual mode）**：此时区域对向量计算的放置具有决定权。**(a)** **任何**区域内都不得有 cube 计算——理由有两条互相独立，故不分模式一律触发：数据并行区域无法对 `matmul` 做向量切分（每个 AIV lane 只持有半块 tile），而任何区域（含任务并行）**本身就是** AIV lane 的函数体，cube 计算不属于其中。**(b)** 不得有在**切分轴**上折叠的向量归约（`tile.row_*` / `tile.col_*` / `tile.sum` / `tile.max` / `tile.min`）——它会产生每 lane 的部分结果（错误编译）。与 (a) 不同，该理由确实与切分轴相关，故 (b) 仍只在数据并行模式下生效。**(c)** `aiv_shard` / `aic_gather` 必须出现在区域内部（`tile.*` 与面向作者的 `tensor.*` 两种形式皆然）。在任务并行的 `mode=NONE` 区域中它们是**被接受的**：没有可切分的轴时，它们仍然承载检查 (f)/(g) 所要求的含义——这个值跨越了 AIC/AIV 边界——且其 `split=0` 类型推导会原样保留形状，既不折半也不拼合。**(d)** **边界内存契约**：`tile.aiv_shard` 为 `Acc → Vec`，`tile.aic_gather` 为 `Vec → Mat`。这两个算子本身*就是*跨核传输，因此操作数必须位于生产侧 lane、结果必须位于消费侧 lane；内存尚未解析前会跳过检查，故同一检查在整个窗口内都安全。该契约与模式无关——任务并行的跨越与数据并行的跨越横跨的是同样两条 lane——因此它在每个区域中都生效，`NONE` 也不例外。**(e)** 手动模式函数中，所有区域**之外**不得有 VECTOR 亲和的算子——区域既已对放置具有决定权，这样的算子既未被区域钉在 AIV lane 上、也不是 cube 计算，没有确定的归属。共有三类排除项：`tile.load` / `tile.store` 是**编译器自身**产生的区域外输出——`ConvertTensorToTileOps` 会把 tensor 级算子的 load/store 对提升到承载其计算的区域之外；以及通过 `set_core_affinity` **显式声明**了所在 lane 的算子（`system.syncall(core_type="aiv_only")`、`system.sync_set/sync_wait(core_type="aiv")`、`pld.tile.put` / `get`）——它们的 lane 本就不是推断出来的，区域也无从进一步消歧。**没有**任何区域的函数不受影响。**(f)** V->C：在区域内定义、却在区域外的 **cube** lane 上被读取的值，必须是 `tile.aic_gather` 的结果。**(g)** C->V：在所有区域之外由 cube 产出、却在区域内的**向量** lane 上被读取的值，必须经由 `tile.aiv_shard` 抵达。两个方向本来就都能下降——编译器无论如何都会发出 `split=0` 的 tpush/tpop 对——这恰恰是要检查它们的原因：手动模式的存在意义就是让作者（而非编译器）来放置 AIC/AIV 边界，而没被写出来的边界就是没人选择过的边界。(f)/(g) 在**定义侧与消费侧**都沿用 (e) 的 `tile.load` / `tile.store` 排除项，理由相同：那一对正是编译器自己提升到区域之外的。跨 C/V 的 `tile.move` 按其投递方向计为消费者，因此在 `InferTileMemorySpace` 验证点上这两项检查依然生效——此时隐式跨越已经变成了这样一次 move。**刻意不检查：**从 `mode=NONE` 区域向外的 V->C 跨越的 lane 规则——ISA 要求两条 AIV 子 lane 都参与 no-split 握手，而它们共用同一个目标槽位、没有每 lane 偏移，且两者之间没有任何仲裁，因此当两条 lane 持有不同的值时，cube 收到的是二者之一且不确定是哪一个；保证被 gather 的值 lane-uniform 是作者的职责，只作文档说明、编译器不合成。同样刻意不检查的还有只应发生一次的副作用（`pld.system.notify`）在两条 AIV 子 lane 之间的分片。区域无法表达「恰好一次」——AIV 函数带有 `dual_aiv_dispatch`，其函数体会在两条 AIV 子 lane 上都运行——而正确写法（按 `aiv_id` 分片）与错误写法（两条 lane 通知同一个 peer）在 IR 上完全相同。这两条规则都改为面向作者写入[作用域与放置](../../user/language/04-scopes.md)。 **(h)** **放置位置** —— `pl.split_aiv` 开启的是 CORE_GROUP 级区域，因此只能写在 CORE_GROUP scope（`pl.at(level=pl.Level.CORE_GROUP)`）内，或写在 Opaque 函数顶层（解析器会为其合成同样的 scope）；**不得**写在已声明为 `pl.FunctionType.InCore` 的函数内。判据是**来源（provenance）**：区域之所以能合法出现在 InCore 函数中，只可能是 `OutlineIncoreScopes` 把外围的 CORE_GROUP scope 提取成了该函数——而提取器会在它新建的函数上打 `split_aiv` 标记（见 `scope_outline_utils.h`，`LowerAutoVectorSplit` 会重新打一次）。因此本检查是「InCore 函数带有区域，但它不是提取器产生的函数」。之所以按来源而非按形状判断：解析器必须把 InCore 函数中的顶层区域**裸露**地发出（否则打印一个被提取的函数再重新解析就无法重建同样的 IR），而基于形状的判据（「区域嵌套在仍存在的 InCore scope 中」）只在解析器为每个顶层区域加包装时才成立，一旦不再加包装就会悄无声息地不再拒绝任何东西。`LowerAutoVectorSplit` 保留其下降后守卫，作为本检查未遍历的 scope 种类的兜底；在此处报错可让诊断比原先提前 12 个 pass。 **(i)** **边界结果不得跨越循环回边** —— `pl.aiv_shard` / `pl.aic_gather` 的结果不得作为循环 `iter_arg` 的初始值，也不得作为回边上 yield 回该 `iter_arg` 的值。边界结果是按 lane 的（半宽），而 `LowerAutoVectorSplit` 的半宽扫描与 `ExpandMixedKernel` 的边界折叠都不会穿过循环 phi 追踪取值：扫描看到的是 `IterArg` 而非定义它的 shard，于是把已经折半的 tile 误判为全宽；而 `FixupIterArgInitValues`（`loop_state_repair.cpp`）在替换边界 tpop 的 `DeepClone` 之前运行，把仍为原始形态的初始值变量判定为未定义，并把该 pass 刚刚折叠掉的 `tile.aiv_shard` 重新拉回——最终表现为一条指名作者从未写过的 SSA 的内部错误。两端都要检查，因为无论哪一端出问题循环都能通过类型检查：以半宽 `tile.full` 作种子、从第 1 次迭代起由 shard 供给的循环体同样会击穿这些 pass，而初始值一侧毫无迹象。 **(j)** **边界结果属于产生它的那个区域** —— 在其定义区域之外的**数据并行**（`UP_DOWN` / `LEFT_RIGHT`）区域内读取 `pl.aiv_shard` / `pl.aic_gather` 的结果会被拒绝。这类区域会沿切分轴把它计算的每个 tile 折半，并把每个 store 偏移本地化到 lane，因此一个已经按 lane 的值会被折半两次、偏移两次——偏移损坏是静默的，而形状损坏会完全逃出 pypto，最终由 ptoas 报出 `'pto.tcvt' op expects src and dst to have compatible shapes`。与 (b) 同理只在数据并行模式下生效：`mode=NONE` 区域没有切分轴、不改写任何东西，因此**可以**消费别处产生的边界结果——这正是跨核通信 kernel 的写法，该豁免正是为了让它保持合法。边界算子本身不作为消费者判定（它*就是*那次跨越，与 (f)/(g) 一致）；`tile.load` / `tile.store` 豁免在此**不**适用，因为该豁免针对的是被 `ConvertTensorToTileOps` 提升到区域**外**的算子，而 (j) 只考察区域**内**的消费者。 **(k)** **一个函数一条跨核 pipe——因而只有一种传输类别** —— 同一函数中不得同时存在 no-split 跨越（`split=0`，来自 `mode=NONE` 区域）与 split 跨越。函数里的每个 `pl.aiv_shard` / `pl.aic_gather` 都跑在**同一条**逻辑 pipe 上：`BuildAutomaticPipeSetup`（`cross_core_pipe.cpp`）每侧只发出一对 `reserve_buffer` + `initialize_pipe`，且 `dir_mask` 是**合并**的，因此 C→V 与 V→C 也共用它。pto-isa 把 no-split 作为 pipe **类型**的参数（`TPipe<FlagID, Dir, SlotSize, SlotNum, LocalSlotNum, IsNoSplit>`），它选择的是另一套握手协议（`ShouldNoSplitC2VConsumerLaneParticipate` 以 `Pipe::is_no_split` 为门控），因此一条 pipe 在其整个生命周期内要么走 no-split 协议、要么走 split 协议。PTOAS 在 `PTOInferValidatePipeInitPass` 中执行同一约束：它按 pipe 从其使用者推断出唯一的 `nosplit` 布尔量，并把冲突拒绝为 `'pto.initialize_l2g2l_pipe' op cannot mix 'split = 0' with 'split = 1', 'split = 2', 'split = 3', or 'split = 4' on the same logical pipe`——一个作者从未写过的内部算子，来自 DSL 根本没有提及的后端。切分**轴**则确实是逐次传输的（`TALLOC` / `TPUSH` / `TPOP` 各自以 `TileSplitAxis` 作为模板参数），因此 `UP_DOWN` 与 `LEFT_RIGHT` 共用一条 pipe，是**被接受的**：这条界线画在 `SplitMode::None` 与其余模式之间——PTOAS 也正是这样划分的——而不是画在「模式不同」上。判据是每个边界算子自身的 `split` kwarg，而非外围区域的模式，这一点同时在三处都是对的：解析器按**最内层** `pl.split_aiv` 打上该 kwarg（嵌套因此自动成立），被提取的 `pl.tile.aiv_shard(t, split=N)` 形式在完全没有区域时也带着它，而**不含**任何跨越的区域不贡献传输、因而可以保留任意模式——这正是通信 kernel 的写法：`mode=NONE` 区域的存在只是为了把 `pld.system.notify` 钉在向量 lane 上。每个函数只报一次，报在两个冲突算子中较后的那个上，并在消息中指出另一个的位置。 **由** `OutlineIncoreScopes` **产生**，随后由 `ConvertTensorToTileOps` 与 `InferTileMemorySpace` 失效并重新产生，最后由 `LowerAutoVectorSplit` **失效**（后者擦除区域节点）。`PassPipeline` 只验证*产生*的属性（`passes.cpp`），因此正是这两次重新产生才为检查 (d)、(e)、(f)、(g) 提供了真正生效的验证点——在 `OutlineIncoreScopes` 处边界仍是不带内存空间的 `tensor.*` 形式（(d) 空转），tensor 级计算也归类为 `SHARED`，于是 (e)、(f)、(g) 找不到 VECTOR 或 CUBE 算子。被检查的算子都是带非空 `op_` 的普通 `Call`；`Submit` 会被正确跳过。**修复方式**：(a) 将 cube 算子移出区域；(b) 在非切分轴上归约，或先用 `tile.aic_gather` 汇聚回完整 tile；(c) 把边界算子移进它所属的区域内；(d) 只对 cube 产出的值做 shard——向量产出的值（`pl.load` / `pl.full`）本就位于 AIV lane，应去掉 `pl.aiv_shard`，交由隐式 affinity 门控的切分来折半；(e) 把该阶段包进独立区域——`for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):` 让两条 lane 都运行完整函数体，正是全宽计算的任务并行写法；(f) 在区域内 gather（`x = pl.aic_gather(x)`），在区域外读取 gather 后的值；(g) 在区域开头以 `pl.aiv_shard(x)` 读取；(i) 把 `pl.aiv_shard` 调用移进循环体、紧挨其消费者，或改为跨循环携带**全宽**的 `pl.matmul` 累加器并在使用处再 shard（这会多占一组 L0C 累加器，可能超出 Acc 预算）；(j) 把读取边界结果的算子移进产生它的区域，或把 `pl.aiv_shard` 调用移进消费区域；(k) 让函数内所有跨越在 split / no-split 上取得一致（给 `mode=NONE` 那个阶段一个 split 模式，或把 split 阶段改为 `pl.SplitMode.NONE`），或者去掉其中一个区域里的跨越，又或者把两个阶段拆进各自的 CORE_GROUP scope——它们会被提取成各自的函数，从而各得一条 pipe。 |
| **HardSyncallOccupancy** | HardSyncallOccupancyValid | 硬（FFTS）形式的 `system.syncall` 会等待其 `core_type` 的**每一个**物理核到达 barrier，因此外层 SPMD 启动必须同时满足两个独立保证：(1) **满占用**——恰好填满这些核（`N != required` 即报错，覆盖部分占用与超占用）；(2) **`sync_start=True`**——所有 block 同时驻留，因为非 sync_start 启动可能分波次派发 block，即便满占用也会使 barrier 死锁。任一缺失都会在设备上死锁（AICore 超时 507018）并使设备残留需复位。**由** `ExpandMixedKernel` **产生**（该 Pass 解析出每个被启动 kernel 的 `FunctionType`——检查所依赖的前提）并列入 `GetVerifiedProperties()`，故 `PassPipeline` 在该 Pass 之后立即自动验证一次。覆盖所有块数可被识别的 SPMD 启动点：`FunctionType::Spmd` 函数（作用域式 `pl.spmd`）、带 `core_num` 属性的 `FunctionType::Group` 函数（`pl.cluster()` 内嵌 `pl.spmd`）、以及带 `core_num` 的 `Submit`（`pl.spmd_submit`）。启动宽度分两类：编译期字面量（对照 SoC 物理核数校验），或**启动形状查询**——`pl.system.available_cluster_count()` / `pl.system.available_aiv_count()`，可直接传入或经其绑定的 Var 传入。查询在设备上解析为本次运行自身的几何，因而天然满占用，无需再比数量；它同时是可移植写法：字面量即便与本 SoC 静态核数相符，只要运行时设备报告的可用核数不同就是错的——可用核更多时块数不足，更少时块数超额（后者会被 runtime 直接拒绝）。用错核类型的查询（纯 AIV 启动却用 cluster 数、混合启动却用 AIV 数）会被拒绝。按启动点及其直接被调：独立 **AIV** kernel + `aiv_only` → `required = GetCoreCount(VECTOR)`；独立 **AIC** kernel + `aic_only` → `required = GetCoreCount(CUBE)`；独立 AIV/AIC kernel 若用了**不匹配**的 `core_type`（含默认的 `mix`）直接报错——单核类型启动下另一种核零参与，barrier 永远无法完成；**Group**（混合 kernel）→ 任意 barrier `core_type` 的 `required = GetCoreCount(CUBE)` 个 core-group（每 group 一个 AIC，填满全部 group 即填满全部核），检查其 AIC/AIV 子 kernel 中的硬 syncall（复制的 `mix` 只报一次）。当后端未配置、`core_num` 既非字面量也非启动形状查询、或没有 SPMD 启动点启动该 kernel（占用率是启动期属性）时跳过。**修复方式**：用匹配的 `core_type` 满占用启动并设置 `sync_start=True`（推荐直接用对应的启动形状查询来定尺寸），或对部分占用改用 `pl.system.syncall(mode=pl.SyncAllMode.SOFT, ...)`（GM 轮询）。 |
| **AccToGmStoreValid** | AccToGmStoreValid | 每个源 tile 位于 **Acc** 的 `tile.store`，其目标 GM tensor 的 dtype 必须是 cube fix-pipe 能够收窄到的类型：Ascend910B 与 Ascend950 均为 `INT32/FP32/FP16/BF16`（`BackendHandler::SupportsAccToGmDtype`，与 pto-isa 非量化分支的 `CheckAcc2gm` 白名单以及 ptoas `pto.tstore` 的 "acc tstore dst element type" 规则一致；该钩子仍按后端区分，因为两个集合是各自 pin 目标的独立事实）。`INT8`/`INT16` 目标会被拒绝。**由** `InferTileMemorySpace` **产生**并列入 `GetVerifiedProperties()`，因此 `PassPipeline` 在该 pass 之后立即自动验证。这个位置是关键的：合法性取决于 tile 的**内存空间**，而非用户可见的 dtype，因此无法更早判定——同一段 DSL 程序，当 matmul 结果经由 Vec（显式 `pl.cast` 在向量单元中收窄）时合法，留在 Acc 时则非法。缺少此检查时程序会一路走到 ptoas，由后者在 `pto.tstore` 校验阶段针对生成的 `.pto` 中某一行报错；而且这样一个 store 往往还会连带触发另一个看似无关的算子（int8 零初始化会下降为同样非法 dtype 的 `pto.texpands`），于是这个迟到的诊断只报出两个症状，从不指出根因。当后端未配置（无可校验的依据）或 tile 内存空间尚未解析时跳过。**修复方式**：先经向量单元收窄——用 `pl.cast` 将 matmul 结果转为目标 dtype 后再存储——或者累加到 `INT32`/`FP32` tensor 后再转换。 |
| **AccCompactValid** | AccCompactValid | L0C compact 契约的两半。**(a)** 凡是 lhs 有效行数使 `mad` 的 pitch 与累加器物理行数不同的 `tile.matmul_acc` / `tile.matmul_mx_acc`，其累加的 buffer 必须是 `CompactMode::normal`。`mad` 取 L0A 操作数的有效行数作为 M，并以 `ceil(M/16)*16` 的 N-fractal stride 写出乘积（pto-isa `TMatmul.hpp`）；而读取方在 tile 非 compact 时按编译期物理 `Rows` 推导 stride，只有 compact 时才重新计算 `ceil(validRow/16)*16`（`tstore_common.hpp`）。非 compact 的累加器会以从未写入过的 pitch 读回，第一个之后的每个 N-fractal 都会错位——这正是 #2470 的 store 路径与 #2510 的 Cube→Vector push 路径，两者都以设备上的错误数值出现，且各层均无诊断。检查放在累加算子上，是因为比较所需的两半都在手边（`mad` 取 M 的 lhs，以及结果原地别名的那块 buffer），而且它是唯一**继承**而非推导 compact 的算子，因此正是以非 compact 种子起头的链路丢失该模式之处。检查**读取方**则是错的：`tile.store` 无法区分 `mad` 写出的累加器与某个 `tile.load` 按物理 pitch 填充的 Acc tile。仅当 pitch **确实不同**时才报错——`ceil(validRow/16)*16 == Rows` 时二者一致（有效行填满 box，或单个 fractal 行块的 box 无论有效行为多少都打包成 16，例如 `[16, N]` 的 gemv 累加器只有一行有效）；`AccPitchesCoincide` 与盖章逻辑共用，两者不会漂移。**(b)** compact 累加器的打包 pitch 必须在其别名上保持不变：`tile.set_validshape` 只改元数据且刻意保留 `compact`，因此跨 fractal 边界重新声明有效行数（`mad` 以 pitch 32 写了 17 行，别名却改称 16 行）会让字节仍按 32 打包，而每个 compact 读取方此时都按 16 推导。这样的别名会被拒绝——请在数据离开 L0C 之后再窄化结果。**全 box** 的源是例外：`AutoTileMatmulL0` 用 `tile.create(compact=True)` 声明种子并随即窄化，而尚未写入任何内容的 buffer 不存在重新解释；无法判定的一对 extent 也会被放过而非拒绝。**(c)** fractal 空间（`Left` / `Right` / `Acc`）之外的 tile 一律不得携带 compact：compact 描述的就是 N-fractal pitch，UB tile 没有这种 pitch，pto-isa 的 Vec 路径也从不读取 `TileData::Compact`——因此把 C2V pop 的 Vec 侧标成 compact，轻则完全无效，重则在确实读取该标记的算子（`TMov`、`TFillPad`）上造成静默的布局变化。该规则只检查 Var 类定义与参数，不检查 call 结果类型，因为 `AssignStmt` 两侧绑定同一类型。**由** `InferTileMemorySpace` **产生**（内存空间解析后才能识别 Acc tile），并由 `ExpandMixedKernel` 重新产生——后者会把边界 `tile.move` 重建为 tpush/tpop 对，消费侧类型是新构造的；两者都列入 `GetVerifiedProperties()`。**修复方式**：此处失败是编译器 bug 而非用户错误——`tile.matmul` 通过 `StampCompactForNarrowedAccRows` 推导该模式，合成的累加器种子则通过 `tile.create(..., compact=True)` 声明；丢失它的链路一定有某个种子或别名从未携带过它。 |
| **TileMemoryInferred** | TileMemoryInferred | 每个**由 `AssignStmt` 绑定的** `TileType` 变量都已解析出 `memory_space_`，**并且** `Call`（位于 `AssignStmt` 或 `EvalStmt` 中）的每个受约束实参所在的空间都在该算子注册的 `input_constraints`（`set_input_memory`）允许范围内。该访问器只遍历这两类语句，因此 `ForStmt` 的 `iter_args_` / `return_vars_` 以及 `IfStmt` 的 `return_vars_` 标注**不在**检查范围内。后者才是关键：违反声明输入空间的算子没有合法下降路径，症状总是出现在很远的下游。`tile.cast` 要求 `Vec`；若喂入 `Acc` 实参，cube→vector 的切分点就没有边界 `tile.move`，于是 `ExpandMixedKernel` 切分出的 kernel 中，cast 引用了只在 cube 侧定义的变量——失败要到 11 个以上 pass 之后才浮现：或是 `MemoryReuse` 中非法的 `Acc->Acc tile.move`，或是 PTO codegen 的 `no MLIR mapping for MemRef base`，两者都不会指出真正出错的算子。该校验器直接读取 `TileType` 标注，而非分析阶段的 `var_memory_` 映射，因此对分析阶段漏记的空间依然如实报告。**由** `InferTileMemorySpace` **产生**并列入 `GetVerifiedProperties()`，故 `PassPipeline` 在该 pass 之后立即自动验证。**修复方式**：该 pass 会自行插入所需的 `tile.move`；此处报错属于 `InferTileMemorySpace` 的编译器缺陷，而非用户编写错误。 |
| **AtomicAddDtypeValid** | AtomicAddDtypeValid | 每个写入全局内存的原子加操作，其目标 dtype 必须是后端 store 流水能够合并的类型。该校验在一处覆盖全部原子写入点：`tile.store`、`tensor.assemble`、`pld.tensor.put`、`pld.tile.put`、`pld.tensor.remote_store` 和 `pld.tile.remote_store`。只有 `bf16` 因后端而异——pto-isa 将其下降为 `SetAtomicAdd<bfloat16_t>` -> `set_atomic_bf16`，Ascend910B（A2/A3）支持而 Ascend950（A5）不支持（`BackendHandler::SupportsBf16AtomicAdd`）；其余硬件原子加 dtype（`FP32/FP16/INT32/INT16/INT8`）在所有后端均可用，由各算子 deducer 以后端无关的方式把关。远程 put 路径与本地 store 是**同一套机制**而非并行机制：pto-isa 的 comm `TPut` 通过 VEC 暂存 tile 流式传输，并用 `TSTORE_IMPL<..., AtomicAdd>` 落盘每个分块，而 `remote_store` 直接发射 `pto.tstore`，因此一个判定式即可管住全部写入点；而 ptoas 自身没有原子 dtype 规则（`TPutOp::verify` 只检查元素类型一致性与 shape），缺少此检查时程序会一路走到生成代码中的 pto-isa `static_assert`，而那段代码并非用户所写。列入 **`GetStructuralProperties()`**，且不由任何 Pass 产生：这里不依赖任何下降结果（atomic kwarg 与目标 dtype 在用户自己的 IR 中即已存在），因此 `PassPipeline` 在 `pipeline_input` 阶段验证，错误携带原始 `Span`。当后端未配置（无可校验的依据）时跳过。**修复方式**：累加到 `FP32` tensor，在归约完成后再转换为 `bf16`。 |

### InParamWritten

> 这是[参数方向推导](../ir/08-param-directions.md)所述整条链的最后一环。

**警告**：`DiagnosticCheck::InParamWritten` —— 声明为 `In` 的参数被其所在函数体写入。

这是一个**警告，而非 `IRProperty`**，这个区分是实质性的而非措辞问题。见下文"为何不是属性"。

**它证明什么，不证明什么。** 每个推导方向的 pass 都要构建一份"这次调用写哪个实参"
的集合，而该集合来自算子在注册表上的声明（`set_arg_effect`，参见
[算子](../ir/05-operators.md#参数效应argument-effects)）以及各被调函数自身的
`param_directions_`。本检查读取的是**同样这两份声明**，报告它们与参数自身 `In` 声明
相矛盾之处。因此它是一道**针对已声明写语义的一致性检查**，而不是对写语义的独立发现。

这个区分很重要，因为催生这项工作的故障恰恰是**声明缺失**：从未声明效应的算子会被读成
纯消费者，它的写入消失，参数停留在 `In`，不会对它发出 RAW 边，症状不在编译期暴露，而是
在设备上表现为竞争或调度死锁。`pld.system.notify` 就是这样上线的（#2391），而
`tile.mscatter` 在本检查写下时仍处于同样状态。**本校验器无法发现这一类问题。** 对于没有
声明效应的算子，`CallWriteTargets` 返回空集，检查保持沉默——它读的正是那份缺失的声明。

这个缺口在别处也没有被堵住，注册表门禁到底覆盖多远需要说准。`ValidateArgEffects` 只对
两种形态开火：

- 声明了 `set_output_reuses_input(N)` 却没有对实参 `N` 分类的算子；
- 声明了 write channel 却不通过任何实参写入的算子。

**两者皆无**的算子——既无复用契约、也无 write channel——两道门都不碰。`pld.system.notify`
正是这种形态：去掉它的 `set_arg_effect`，它会同时静默通过注册门禁**和**本校验器，与 #2391
当时一模一样。因此这两项检查合起来**并未**普遍封住最初那类生产故障；它们覆盖的是"已经
对自己有所声明、但声明不完整"的算子。

本检查真正带来的是：`tile.mscatter` 与 `pld.system.notify` 补上声明之后，它阻止调用方把
目标参数重新声明为 `In`；并且它覆盖全部跨函数调用——那里被调函数自身的签名就是声明，
不涉及任何注册表条目。

**运行方式。** 注册为 `DiagnosticCheck::InParamWritten`，在
`DiagnosticPhase::PostPipeline` 上作为**警告**运行。
它只能经诊断注册表触达——这正是它作为警告的体现：

```python
checks = passes.DiagnosticCheckSet()
checks.insert(passes.DiagnosticCheck.InParamWritten)
diagnostics = passes.DiagnosticCheckRegistry.run_checks(
    checks, passes.DiagnosticPhase.POST_PIPELINE, program
)
```

**为何不是属性。** 属性是编译器可以担保的论断，而这一条担保不了。该检查必须在
`DeriveCallDirections`（pass 37）之后运行——在那之前 wrapper 的签名读作 `In` 是合法的；
而 `InitMemRef`（pass 31）声明了 `.invalidated = {IRProperty::SSAForm}`，此后无人重建。
**流水线中不存在既在 pass 37 之后、又处于 SSA 形式的位置。** 下文的 buffer lineage 在汇合点
不做合并，其精确性只在"每个名字一个定义"时成立，因此在它实际收到的 IR 上，两个方向都可能出错：

- 分支内建立的 view 会把 lineage 泄漏过汇合点，分支之后的写入可能被归咎于只有该路径才命名的
  buffer；
- `BufferRootCollector` 预先扫描整个函数体，重新绑定的名字只有一份最终映射，却也被套用到更早的
  写入上。

两种形态都钉在 `tests/ut/ir/verifier/test_in_param_written.py` 里，第一种是 strict `xfail`，
一旦修好该测试就会失败。**报告是"去看一眼"的信号，沉默则什么都不证明。** 要让它健全，需要
真正的控制流数据流分析（汇合点合并 + 候选集 lineage）——那是另一项工作，也是这棵已有三套别名
模型的树里的第四套。

三个关键取舍：

- **作用于完成后的程序，而非某个 pass 之后。** Group/Spmd wrapper 把参数转发给内层
  kernel，在 `DeriveCallDirections` 的 phase 0 把有效方向写回 IR 之前，它自己的签名对
  内层 kernel 会写的参数读作 `In` 是合法的。该不变量只在流水线跑完后成立。
- **跳过 Orchestration 函数。** 它们的方向是用户的声明，其参数就是 host ABI——纯 `Out`
  参数会在 return 风格调用中由 host 自动分配，因此翻转它是用户要做的迁移，而不是编译器
  要补完的推导。
- **是警告而非错误。** 它不会凭空造出写入，但确实会报告今天可以正常编译运行的程序；
  升级路径就是上面的 `IRProperty`——等报告清零之后。

**零拷贝 view 会被追溯。** 经参数的 view 写入，就是对该参数的写入：

```python
view = pl.tile.slice(acc, [8, 128], [0, 0])   # acc 声明为 In
view = pl.tile.assemble(view, src, [0, 0])    # 写的是 acc 的 buffer
```

判定一个值指向哪块 buffer 的是两份共享声明，都不是本地维护的清单：`ResultAliasedArgIndex`
（算子返回它更新过的那个实参——`tensor.assemble`、`tensor.write`、各集合通信），以及
`op_predicates::IsBufferAliasingViewOp`，后者读取 `OutputMemoryInheritsInput() &&
IsInplaceSafe()`——即零拷贝 view，它们不更新任何东西，因此也不声明复用契约。
`tile.transpose` 凭自身的 `not_inplace_safe()` 注册被第二个判据排除：它把数据置换进一块
全新 buffer，输出并不别名输入；日后任何以同样方式注册的 inherit-input 算子也会被自动
排除，无需改动此处。

`tensor.slice` 正属于这类 view，因此写入参数的某个 slice **会**被报告——尽管
`BufferRootCollector` 有意把它映射为全新 root。该链在校验器内部解析而非改动那份共享分析，
因为另有三个 pass 共用它，为它们一并放宽"何为别名"是另一项改动。

**SSA 是前置条件。** lineage 是单一环境、在汇合点不做合并，其正确性恰好依赖"每个名字只有
一个定义"——而 `PostPipeline` 保证了这一点。在 SSA 之前的 IR 上它两个方向都不成立：分支中
重新指向某个名字会把 lineage 泄漏过汇合点（分支未走时，写入根本到不了那块 buffer，却被
归咎于它）；而普通的 `t = buf1; ...; t = buf2` 会让 `BufferRootCollector` 用同一份最终映射
覆盖两块不同 buffer。直接调用方必须先做转换。

lineage **不**跨 phi（`return_vars_` / `iter_args_`）传递，因此分支或循环之后经 view 的写入
会漏报——这是安全的方向。来源在记录绑定时即已解析，因此查表只需一次读取，遍历对函数体
保持线性。

**修复**：把该参数声明为 `pl.Out`（只写不读）或 `pl.InOut`（既读又写）。对本检查报出的
问题，补 `.set_arg_effect(...)` **不是**修复手段——builtin 能出现在这里正是因为它的效应
已经声明，而跨函数写入方是用户函数，根本没有 `REGISTER_OP` 块。缺失效应属于上文所述的
注册表缺口，本检查看不见它。

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
| `GetStructuralProperties()` | `{TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid, ArrayNotEscaped, ManualDepsOnSubmitOnly, AtomicAddDtypeValid}` | 由 `VerificationInstrument` 在每个 Pass 执行前后验证的不变量（与 `GetVerifiedProperties()` 共有的子集还会在流水线启动时验证） |
| `GetDefaultVerifyProperties()` | `{SSAForm, TypeChecked, NoNestedCalls, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, TileTypeCoherence, ArrayNotEscaped}` | `run_verifier()` 的默认属性集 |
| `GetVerifiedProperties()` | `{SSAForm, TypeChecked, MixedKernelExpanded, AllocatedMemoryAddr, BreakContinueValid, NoRedundantBlocks, InOutUseValid, CallDirectionsResolved, ManualDepsOnSubmitOnly, ReturnParamsExplicit, AivSplitValid, TileMemoryInferred, HardSyncallOccupancyValid, IterArgCarryClassified, RuntimeScopesMaterialized, DistTensorCtxMaterialized, GraphBoundaryLegalized, AccToGmStoreValid, AccCompactValid, AtomicAddDtypeValid}` | `PassPipeline` 自动验证的轻量级属性集 |

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
