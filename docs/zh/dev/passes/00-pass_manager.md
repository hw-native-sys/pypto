# Pass、PassContext、PassPipeline 和 PassManager

用于组织和执行中间表示 (IR) 变换 Pass 的框架，支持属性 (Property) 跟踪、插桩和基于策略的优化流水线，作用于 Program。

## 概述

| 组件 | 描述 |
| ---- | ---- |
| **Pass (C++)** | 独立的 Program -> Program 变换类，带有属性声明 |
| **IRProperty / IRPropertySet** | 可验证 IR 属性的枚举 + 位集合（SSAForm、HasMemRefs 等） |
| **PassInstrument / PassContext** | 插桩回调（Pass 执行前/后），使用线程局部上下文栈 |
| **PassPipeline (C++)** | 按顺序执行的有序 Pass 序列 |
| **PassManager (Python)** | 高层管理器，使用 PassPipeline，支持基于策略的优化 |

### 关键特性

- **属性跟踪**：Pass 声明所需、产生和失效的属性
- **插桩**：PassContext 持有 PassInstrument，在每个 Pass 执行前/后运行
- **运行时验证**：VerificationInstrument 根据实际 IR 检查属性
- **基于策略的流水线**：预配置的优化级别（`Default`）
- **不可变变换**：返回新的 IR 节点，不就地修改

## IRProperty 系统

### IRProperty 枚举

**头文件**：`include/pypto/ir/transforms/ir_property.h`

| 属性 | 描述 |
| ---- | ---- |
| `SSAForm` | IR 处于静态单赋值 (SSA) 形式 |
| `TypeChecked` | IR 已通过类型 (Type) 检查 |
| `NoNestedCalls` | 无嵌套调用表达式 (Expression) |
| `NormalizedStmtStructure` | 语句 (Statement) 结构已规范化 |
| `NoRedundantBlocks` | 无单子节点或嵌套的 SeqStmts |
| `SplitIncoreOrch` | InCore 作用域已提取为独立函数 |
| `HasMemRefs` | 变量上已初始化内存引用 (MemRef) 对象 |
| `IncoreTileOps` | InCore 函数使用 tile 操作（tile 类型、load/store） |
| `AllocatedMemoryAddr` | 所有 MemRef 在缓冲区限制内具有有效地址 |
| `MixedKernelExpanded` | 混合 InCore 函数已拆分为 AIC + AIV + Group |
| `ClusterOutlined` | Cluster 作用域已提取为 Group 函数 |
| `TileOps2D` | InCore 函数中所有 tile 操作的 tile 维度 ≤2D |
| `TileMemoryInferred` | InCore 函数中 `TileType::memory_space_` 已填充 |
| `BreakContinueValid` | break/continue 仅出现在 sequential/while 循环中 |
| `UseAfterDef` | 所有变量使用都被其定义支配 (dominate) |
| `HierarchyOutlined` | Hierarchy 作用域已提取为 level/role 函数 |
| `StructuredCtrlFlow` | 不存在 BreakStmt/ContinueStmt——只有结构化控制流 |
| `VectorKernelSplit` | 带 split 模式的 AIV 函数已调整 tpop 形状与 store 偏移 |
| `OutParamNotShadowed` | Out/InOut 参数未被创建 tensor 的算子重新赋值 |
| `NoNestedInCore` | 无嵌套 InCore 作用域（ScopeStmt 内再嵌 ScopeStmt） |
| `InOutUseValid` | 调用之后不再读取以 InOut/Out 传入的变量（RFC #1026） |
| `PipelineLoopValid` | 双向不变量：`ForStmt.kind_ == Pipeline` ⇔ 带有 `pipeline_stages` 属性 |
| `PipelineResolved` | 不再残留 `ForKind::Pipeline`；由 CanonicalizeIOOrder 产生 |
| `CallDirectionsResolved` | 每个非 builtin Call 都带有显式的 `attrs['arg_directions']` |
| `TileTypeCoherence` | 每个 TileType 都具有规范的 tile_view（隐式视图存储为 nullopt） |
| `InlineFunctionsEliminated` | 不再残留 `FunctionType::Inline` 函数及对其的 Call |
| `OrchestrationReferencesResolved` | `FunctionType::Orchestration` 函数体内每一个非 builtin Call 必须对应到 Program 中存在的 Function |
| `TensorViewCanonical` | TensorView 规范性已验证（弱模式：允许空 stride；严格模式：要求已材料化，RFC #1300 §2.2） |
| `ArrayNotEscaped` | ArrayType 不会出现在函数参数或返回类型中 |
| `CommDomainScopesMaterialized` | host_orch 函数体已被 CommDomainScopeStmt 包裹，且 `pld.tensor.window` 结果类型带有 `DistributedTensorType::window_buffer_` 反向引用 |
| `DistTensorCtxMaterialized` | host orchestration 之外不再残留 `pld.system.get_comm_ctx`；每个 chip-orchestration / device 通信上下文都是可追溯到参数的显式 CommCtxType SSA 值 |
| `RuntimeScopesMaterialized` | Orchestration 函数带有显式的 RuntimeScopeStmt 节点，codegen 不再隐式生成 `SIMPLER_SCOPE()` 包裹 |
| `AssignTypeSymmetry` | 每个 AssignStmt 满足 `structural_equal(var->GetType(), value->GetType())`（memref 作为分配细节被排除） |
| `ManualDepsOnSubmitOnly` | 普通跨函数 Call 不携带 `attrs["manual_dep_edges"]`——手写依赖边只存在于 `Submit::deps_` |
| `ReturnParamsExplicit` | InCore/Group/Spmd 的 tensor 返回值按指针恒等引用函数参数（#1702） |
| `UnrollResolved` | 不再残留 `ForKind::Unroll`；由 UnrollLoops 产生 |
| `AivSplitValid` | SplitAivScopeStmt 区域结构合法：区域内无 cube 计算与 split 轴 reduce，边界算子只出现在区域内 |
| `HardSyncallOccupancyValid` | 每个硬 (FFTS) `system.syncall` 都在满占用下启动——部分或超额启动会导致设备侧死锁 (507018) |
| `IterArgCarryClassified` | Orchestration 中每个带 iter_args 的 ForStmt 都带有 `iter_arg_rebind_<i>` 携带方案，codegen 直接读取而不再重新推导 |
| `AccToGmStoreValid` | 每个源 tile 位于 Acc 的 `tile.store` 所写 GM dtype 都能被后端 fix-pipe 收窄 |
| `AtomicAddDtypeValid` | 每个写入 GM 的 atomic-add 的目标 dtype 都能被后端 store pipe 合并 |

### IRPropertySet

基于位集合的高效集合，支持 `Insert`、`Remove`、`Contains`、`ContainsAll`、`Union`、`Difference`、`ToString`。

### 声明新属性 (Property)

一个枚举项需要在四处分别书写，而构建过程不会把它们关联起来：

| 层 | 文件 | 形式 |
| -- | ---- | ---- |
| 枚举 | `include/pypto/ir/transforms/ir_property.h` | `MyProperty,` 并附 `///<` 描述 |
| 名称 | `src/ir/transforms/ir_property.cpp` | `case IRProperty::MyProperty: return "MyProperty";` |
| 绑定 (Binding) | `python/bindings/modules/passes.cpp` | `.value("MyProperty", IRProperty::MyProperty, "<doc>")` |
| 类型存根 (Stub) | `python/pypto/pypto_core/passes.pyi` | `MyProperty = ...` |

四处都要添加，且顺序与枚举的声明顺序一致。缺少绑定的属性仍能编译，`str(IRPropertySet)` 也仍能正确打印其名称
（由上表的 switch 渲染），但只要集合中包含该属性，`IRPropertySet.to_list()` 就会抛出
`ValueError: <n> is not a valid IRProperty`，该集合的每个 Python 调用方都会随之失败。
`tests/lint/check_ir_property_parity.py`（一个 pre-commit 钩子）负责保持这四份列表一致。

### PassProperties

```cpp
struct PassProperties {
  IRPropertySet required;      // Preconditions
  IRPropertySet produced;      // New properties guaranteed after running
  IRPropertySet invalidated;   // Properties this pass breaks
};
```

## 各 Pass 的属性声明

| Pass | 所需 | 产生 | 失效 |
| ---- | ---- | ---- | ---- |
| InlineFunctions | — | InlineFunctionsEliminated | — |
| UnrollLoops | — | UnrollResolved | — |
| CtrlFlowTransform | — | StructuredCtrlFlow | — |
| ConvertToSSA | — | SSAForm | NormalizedStmtStructure |
| Simplify | — | — | — |
| NormalizeStmtStructure | — | NormalizedStmtStructure | — |
| FlattenCallExpr | SSAForm, NormalizedStmtStructure | SSAForm, NoNestedCalls, NormalizedStmtStructure | — |
| OutlineHierarchyScopes | SSAForm | SSAForm, HierarchyOutlined, OrchestrationReferencesResolved | — |
| OutlineIncoreScopes | SSAForm | SSAForm, SplitIncoreOrch, AivSplitValid | — |
| OutlineClusterScopes | SSAForm | SSAForm, ClusterOutlined | — |
| ConvertTensorToTileOps | SSAForm, SplitIncoreOrch, NormalizedStmtStructure | SSAForm, IncoreTileOps, NormalizedStmtStructure, AivSplitValid | AivSplitValid |
| OptimizeOrchTensors | SplitIncoreOrch, IncoreTileOps | SplitIncoreOrch, IncoreTileOps | — |
| LowerCompositeOps | — | — | — |
| FlattenTileNdTo2D | SSAForm, IncoreTileOps, NormalizedStmtStructure | SSAForm, TileOps2D, NormalizedStmtStructure | — |
| LegalizeTileCast | — | — | — |
| AutoTileMatmulL0 | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure | — |
| CanonicalizeTileSlice | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, NormalizedStmtStructure | — |
| InferTileMemorySpace | SSAForm, IncoreTileOps, SplitIncoreOrch, NormalizedStmtStructure | SSAForm, TileMemoryInferred, NormalizedStmtStructure, AivSplitValid, AccToGmStoreValid | AivSplitValid |
| InsertMxScaleAddr | SSAForm, IncoreTileOps, SplitIncoreOrch, NormalizedStmtStructure, TileMemoryInferred | SSAForm, IncoreTileOps, SplitIncoreOrch, NormalizedStmtStructure, TileMemoryInferred | — |
| ResolveBackendOpLayouts | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, NormalizedStmtStructure | — |
| LowerAutoVectorSplit | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred, NormalizedStmtStructure, AivSplitValid | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | AivSplitValid |
| ExpandMixedKernel | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, MixedKernelExpanded, NormalizedStmtStructure, HardSyncallOccupancyValid | — |
| InjectGMPipeBuffer | SSAForm, MixedKernelExpanded, NormalizedStmtStructure | SSAForm, MixedKernelExpanded, NormalizedStmtStructure | — |
| SplitVectorKernel | SSAForm, MixedKernelExpanded | SSAForm, VectorKernelSplit, NormalizedStmtStructure | — |
| StampTfreeSplit | SplitIncoreOrch | — | — |
| NormalizeReturnOrder | SplitIncoreOrch, IncoreTileOps | ReturnParamsExplicit | — |
| SkewCrossCorePipeline | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | — |
| LowerPipelineToSlots | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | — |
| LowerPipelineLoops | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | — |
| CanonicalizeIOOrder | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure, PipelineResolved | — |
| MaterializeTensorStrides | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred, NormalizedStmtStructure, TensorViewCanonical | — |
| InitMemRef | SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred | HasMemRefs, NormalizedStmtStructure | SSAForm |
| MaterializeSemanticAliases | SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D, NormalizedStmtStructure | NormalizedStmtStructure | — |
| MemoryReuse | SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D, NormalizedStmtStructure | NormalizedStmtStructure | — |
| AllocateMemoryAddr | SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D | AllocatedMemoryAddr | — |
| FoldNoOpReshape | SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D | — | — |
| FuseCreateAssembleToSlice | SplitIncoreOrch | — | — |
| DeriveCallDirections | SplitIncoreOrch | CallDirectionsResolved | — |
| AutoDeriveTaskDependencies | SplitIncoreOrch, CallDirectionsResolved | CallDirectionsResolved | — |
| ExpandManualPhaseFence | NoNestedCalls, NormalizedStmtStructure, CallDirectionsResolved | NoNestedCalls, NormalizedStmtStructure, CallDirectionsResolved | — |
| SynthesizeAllReduceSignals | — | — | — |
| MaterializeCommDomainScopes | — | CommDomainScopesMaterialized | — |
| LowerHostTensorCollectives | CommDomainScopesMaterialized | CommDomainScopesMaterialized | — |
| MaterializeDistTensorCtx | CommDomainScopesMaterialized, ReturnParamsExplicit | CommDomainScopesMaterialized, DistTensorCtxMaterialized | — |
| MaterializeRuntimeScopes | SplitIncoreOrch, CallDirectionsResolved | RuntimeScopesMaterialized | — |
| ClassifyIterArgCarry | CallDirectionsResolved, RuntimeScopesMaterialized | IterArgCarryClassified, RuntimeScopesMaterialized | — |
| InsertCommFence | SplitIncoreOrch | — | — |
| MaterializeValidShapeSymbols | — | — | — |

本表按 `Default` 策略的执行顺序列出全部已注册 Pass。新增 Pass 或修改属性声明时，请同步更新
此处对应的行。

多数 `PassProperties` 常量位于 `include/pypto/ir/transforms/pass_properties.h`，但该头文件并非
完整清单：`kFuseCreateAssembleToSliceProperties` 就声明在
`src/ir/transforms/fuse_create_assemble_to_slice_pass.cpp` 内部。真正把 Pass *名字* 与其属性绑定
起来的是 `CreateFunctionPass` / `CreateProgramPass` 调用点——`Pass::GetName()` 与该常量正是在那里
汇合。因此请从调用点而非仅从头文件重新生成表格行，否则会静默漏掉 pass 内部的局部声明。

> **注意**：VerifySSA 和 TypeCheck 是**属性验证器 (PropertyVerifier)**（验证规则），不是 Pass。它们通过 `VerificationInstrument` 或 `run_verifier()` 工具函数运行——参见[验证器](99-verifier.md)。这也是没有任何 Pass 声明 `TypeChecked` 的原因：它是**结构性**属性（`GetStructuralProperties()`），在流水线入口的 IR 上验证一次，而不是由某个 Pass 建立。

## C++ Pass 基础设施

### Pass 类

```cpp
class Pass {
  ProgramPtr operator()(const ProgramPtr& program) const;  // checks PassContext
  std::string GetName() const;
  IRPropertySet GetRequiredProperties() const;
  IRPropertySet GetProducedProperties() const;
  IRPropertySet GetInvalidatedProperties() const;
};
```

`Pass::operator()` 检查 `PassContext::Current()` 并在实际变换前后运行插桩。

### 使用属性创建 Pass

```cpp
namespace pass {
Pass YourPass() {
  return CreateFunctionPass(TransformFunc, "YourPass",
      {.required = {IRProperty::SSAForm},
       .produced = {IRProperty::SomeProperty},
       .invalidated = {IRProperty::AnotherProperty}});
}
}
```

## PassContext 和插桩

**头文件**：`include/pypto/ir/transforms/pass_context.h`

### PassInstrument

Pass 插桩回调的抽象基类：

```cpp
class PassInstrument {
  virtual void RunBeforePass(const Pass& pass, const ProgramPtr& program) = 0;
  virtual void RunAfterPass(const Pass& pass, const ProgramPtr& program) = 0;
  virtual std::string GetName() const = 0;
};
```

### VerificationInstrument

使用 `PropertyVerifierRegistry` 验证属性的具体插桩：

```cpp
class VerificationInstrument : public PassInstrument {
  explicit VerificationInstrument(VerificationMode mode);
  // BEFORE: verify required properties before pass
  // AFTER: verify produced properties after pass
  // BEFORE_AND_AFTER: both
};
```

### CallbackInstrument

轻量级插桩，调用用户提供的回调，适用于无需子类化 `PassInstrument` 的临时插桩（IR 转储、日志记录、性能分析）：

```cpp
class CallbackInstrument : public PassInstrument {
  using Callback = std::function<void(const Pass&, const ProgramPtr&)>;
  explicit CallbackInstrument(Callback before_pass = nullptr,
                              Callback after_pass = nullptr,
                              std::string name = "CallbackInstrument");
};
```

```python
# Python: dump IR after each pass
def after_pass(p, program):
    print(f"After {p.get_name()}")

with passes.PassContext([passes.CallbackInstrument(after_pass=after_pass)]):
    pipeline.run(program)
```

`run_passes(dump_ir=True)` 内部使用 `CallbackInstrument` 在每个 Pass 后转储 IR，将验证委托给 C++ 流水线。在已有 `PassContext` 内调用时，转储模式会保留外层上下文的插桩（如用户提供的 `VerificationInstrument`）和验证级别，将转储插桩追加到组合列表中。

**转储详细级别（`PassDumpLevel`）。** `dump_passes` 开关（位于 `ir.compile`、`RunConfig` 以及 `run_passes` 的 `dump_ir`）接受一个 `PassDumpLevel` 枚举——为向后兼容也接受 `bool`（`True` → `CONCISE`，`False` → `NONE`）：

| 级别 | 含义 |
| ---- | ---- |
| `NONE` | 不进行逐 Pass 转储。 |
| `CONCISE` | 简洁规范 IR（默认）；最利于逐 Pass 对比 diff。 |
| `EXPLICIT` | 完全解析的转储——对布局自描述（issue #2088）。 |

默认（`CONCISE`）下，转储的 `pl.Tile` 注解在其 `blayout`/`slayout`/`fractal` 与所属内存空间的*隐式*视图相同时会将其省略，而规范 IR 将隐式视图存储为 `nullopt`——因此即便某个 tile 的真实布局并不平凡，它也可能完全不打印 `TileView`（例如一个 `pl.Mem.Acc` tile 实际上是 `blayout=col_major, slayout=row_major, fractal=1024`）。`EXPLICIT` 让每个转储的 tile 从 `GetEffectiveTileView` 打印其完全解析的布局，并展示 `pld.DistributedTensor` 携带、但简洁形式会丢弃的 `window_buffer` 反向引用——从而仅凭打印出的 IR 即可定位布局/别名缺陷。`EXPLICIT` 转储仍能重新解析为完全相同的 IR：tile 布局会规范化回 `nullopt`（与隐式视图相同的显式视图），而 window buffer 标记是一个信息性的尾随字符串,解析器在重新加载时会将其剥离(真实引用会从 `pld.tensor.window` 重新推导)。这样 `compiled.validate_ir()`(会重新加载每个转储)仍能正常工作。以编程方式使用时，向 `python_print(...)` 传入 `explicit_layout=True`。

```python
from pypto.ir import PassDumpLevel
from pypto.runtime import RunConfig

RunConfig(dump_passes=PassDumpLevel.EXPLICIT)   # 完全解析的转储
RunConfig(dump_passes=True)                     # == PassDumpLevel.CONCISE
```

### ReportInstrument

承载流水线落盘产物所在的目录。它自身不观察任何 Pass —— `DiagnosticInstrument` 读取它的 `output_dir` 来决定把 `perf_hints.log` 追加到哪里：

```cpp
class ReportInstrument : public PassInstrument {
  explicit ReportInstrument(std::string output_dir);
  const std::string& GetOutputDir() const;
};
```

```python
instrument = passes.ReportInstrument("/path/to/report")

with passes.PassContext([instrument]):
    pipeline.run(program)
```

`compile()` 会自动创建一个指向 `build_output/<name>/report/` 的实例。

内存占用不再由这里生成，改为用 `python -m pypto.tools.memory_map` 从 pass dump 渲染 —— 见[内存地图](../07-memory-map.md)。

### RoundtripInstrument

打印→解析 roundtrip 验证插桩。每次 Pass 执行后：

1. 通过 `python_print()` 将结果 IR 打印为 Python DSL 文本
2. 通过 `parse()` 将文本解析回 IR `Program`
3. 断言 `structural_equal(original, reparsed)` —— 失败则说明 printer 或 parser 无法忠实表示该 Pass 输出的 IR

```python
from pypto.pypto_core import passes
from pypto.ir.instruments import make_roundtrip_instrument

with passes.PassContext([make_roundtrip_instrument()]):
    result = passes.convert_to_ssa()(program)
```

**已知的非致命情况**（插桩跳过检查，不报错）：

| 情况 | 行为 | 原因 |
| ---- | ---- | ---- |
| Printer `InternalError`（如 `ForKind::Unroll` + SSA `iter_args`） | `UserWarning`，跳过 roundtrip | 该过渡状态无合法 DSL 语法 |
| 原始 IR 中的 `UnknownType`（手动 `ir.Call(ir.Op(...))` 构造） | 静默跳过 | 解析时 C++ 推断出具体类型，属于类型改善而非 bug |
| `tensor.add(x, scalar)` → roundtrip 后变为 `tensor.adds` | 静默跳过 | Python API 会自动将标量 RHS dispatch 到 `tensor.adds` |
| `tile.load` 3-arg → roundtrip 后变为 4-arg | 静默跳过 | C++ 要求 4 个参数；手动构造 3-arg 由 printer 规范化 |
| 动态 shape Var 在 return types 中的指针不匹配 | 静默跳过 | `structural_equal` 无法在函数体外追踪 Var |

**单元测试中默认开启**，通过 `tests/ut/conftest.py`（见下方[测试Fixture](#测试fixture)）。可通过 `PYPTO_VERIFY_LEVEL=basic` 或 `PYPTO_VERIFY_LEVEL=none` 关闭。

### PassContext

线程局部上下文栈，支持 `with` 风格的嵌套。同时持有插桩和 Pass 配置（如验证级别）：

```cpp
class PassContext {
  explicit PassContext(std::vector<PassInstrumentPtr> instruments,
                       VerificationLevel verification_level = VerificationLevel::Basic);
  void EnterContext();      // push onto thread-local stack
  void ExitContext();       // pop from stack
  VerificationLevel GetVerificationLevel() const;
  static PassContext* Current();  // get active context
};
```

**所有 Pass 相关配置都应放在 PassContext 中**——参见 `.claude/rules/pass-context-config.md`。

### Python 用法

```python
from pypto.pypto_core import passes

# Enable verification for a block of code
with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
    result = passes.convert_to_ssa()(program)  # instruments fire automatically

# Disable automatic verification for a block
with passes.PassContext([], passes.VerificationLevel.NONE):
    result = pipeline.run(program)  # no automatic verification

# Nesting: inner context overrides outer
with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
    with passes.PassContext([]):  # disable instruments for this block
        result = some_pass(program)  # no verification
```

### 测试Fixture

所有单元测试通过 `tests/ut/conftest.py` 自动启用属性验证**和 roundtrip 验证**。roundtrip 在测试中默认开启，以便自动发现 printer/parser 不对称问题。

```python
@pytest.fixture(autouse=True)
def pass_verification_context():
    level_str = os.environ.get("PYPTO_VERIFY_LEVEL", "roundtrip").lower()
    instruments = []
    if level_str != "none":
        instruments.append(passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER))
    if level_str == "roundtrip":
        from pypto.ir.instruments import make_roundtrip_instrument
        instruments.append(make_roundtrip_instrument())
    with passes.PassContext(instruments):
        yield
```

通过环境变量控制：

| `PYPTO_VERIFY_LEVEL` | 属性验证 | Roundtrip |
| -------------------- | -------- | --------- |
| `roundtrip`（测试默认值） | ✅ BEFORE_AND_AFTER | ✅ |
| `basic` | ✅ BEFORE_AND_AFTER | ❌ |
| `none` | ❌ | ❌ |

### PassPipeline (C++)

```cpp
class PassPipeline {
  void AddPass(Pass pass);
  ProgramPtr Run(const ProgramPtr& program) const;  // executes passes in order
  std::vector<std::string> GetPassNames() const;
  std::vector<Pass> GetPasses() const;
};
```

`PassPipeline` 是有序 Pass 对象及其名称的单一事实来源。`GetPasses()` 返回轻量的 Pass 句柄副本，用于检查或组合新的 pipeline。每个 Pass 的 `operator()` 检查活跃的 `PassContext` 以获取插桩。

### 自动验证

当 `VerificationLevel` 为 `Basic`（默认值）时，流水线会自动验证 `GetVerifiedProperties()`（`src/ir/transforms/ir_property.cpp`）列出的**轻量级属性**，每次产生时各验证一次。这可以在无需手动设置 `PassContext` 的情况下捕获常见的 IR 错误。

该集合——以及 `GetStructuralProperties()` 和 `GetDefaultVerifyProperties()`——的成员在文字描述中被重复列出了三份：`ir_property.h` 中每个声明上的 `Returns {...}` 子句，以及各语言版本 [Verifier](99-verifier.md) 文档中的一行汇总。`tests/lint/check_property_set_doc_parity.py`（一个 pre-commit 钩子）负责让这些副本与 C++ 初始化列表保持一致：仅把属性加进初始化列表而不更新副本，既能编译也能通过 CI，但开发者实际阅读的每一份列表都会少一项。

**工作原理**：

1. 在流水线入口验证 `GetStructuralProperties() ∩ GetVerifiedProperties()`——这些不变量在任何 Pass 运行前就应在用户自己的 IR 上成立
2. 每个 Pass 执行后，验证它所*产生*且位于 `GetVerifiedProperties()` 中、尚未验证过的属性
3. 当某个 Pass *失效*了这样一个属性时，将其从已验证集合中移除，以便后续的产生者重新验证
4. 出错时抛出 `VerificationError`

**使用 `Default` 策略时**（共 23 次检查；两个集合都声明在 `ir_property.cpp` 中，因此该时序完全由它们与上面的逐 Pass 属性表推导得出）：

| 验证时机 | 验证的属性 |
| -------- | ---------- |
| 流水线入口 | TypeChecked, BreakContinueValid, NoRedundantBlocks, InOutUseValid, ManualDepsOnSubmitOnly, AtomicAddDtypeValid |
| ConvertToSSA | SSAForm |
| OutlineIncoreScopes | AivSplitValid |
| ConvertTensorToTileOps | AivSplitValid *（重新验证——本 Pass 会先失效它，参见 [10](10-convert_tensor_to_tile_ops.md)）* |
| InferTileMemorySpace | AivSplitValid *（重新验证）*、TileMemoryInferred、AccToGmStoreValid、AccCompactValid |
| ExpandMixedKernel | MixedKernelExpanded、HardSyncallOccupancyValid、AccCompactValid *（重新验证）* |
| NormalizeReturnOrder | ReturnParamsExplicit |
| AllocateMemoryAddr | AllocatedMemoryAddr |
| DeriveCallDirections | CallDirectionsResolved |
| MaterializeDistTensorCtx | DistTensorCtxMaterialized |
| LegalizeGraphBoundary | GraphBoundaryLegalized |
| MaterializeRuntimeScopes | RuntimeScopesMaterialized |
| ClassifyIterArgCarry | IterArgCarryClassified |

因此，一个 Pass 少声明 `produced` 不只是文档写错——它会悄悄地从该时序中抹掉一次验证。

**通过 `PassContext` 控制**：

```python
from pypto import ir
from pypto.pypto_core import passes

# Disable automatic verification via PassContext
with passes.PassContext([], passes.VerificationLevel.NONE):
    pipeline.run(program)

# Or per-compilation
ir.compile(program, verification_level=ir.VerificationLevel.NONE)

# Environment variable (default when no PassContext): PYPTO_VERIFY_LEVEL=none|basic|roundtrip
```

**验证级别的确定方式**：

1. 如果 `PassContext` 处于活跃状态 -> 使用其 `verification_level`（默认：Basic）
2. 如果没有 `PassContext` -> 使用 `GetDefaultVerificationLevel()`（读取 `PYPTO_VERIFY_LEVEL` 环境变量，默认：Basic）

## Python PassManager

**文件**：`python/pypto/ir/pass_manager.py`

### API

| 方法 | 描述 |
| ---- | ---- |
| `get_strategy(strategy)` | 获取按策略配置的 PassManager |
| `run_passes(program, dump_ir, output_dir, prefix)` | 通过 PassPipeline 执行 Pass |
| `get_pass_names()` | 获取所有 Pass 的名称 |
| `passes` / `pass_names` | 从底层 PassPipeline 派生的只读快照 |

### 用法

```python
from pypto import ir
from pypto.pypto_core import passes

# Default usage
pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default)
result = pm.run_passes(program)

# With verification via PassContext
with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
    result = pm.run_passes(program)
```

### 策略补充说明

`Default` 的 PTO tile 阶段顺序为：

1. [`LowerCompositeOps`](12-lower_composite_ops.md)
2. [`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md)
3. [`LegalizeTileCast`](15-legalize_tile_cast.md)（把目标 ISA 无法用单条 `pto.tcvt` 表达的 `tile.cast` 展开为原生 cast 链）
4. [`AutoTileMatmulL0`](16-auto_tile_matmul_l0.md)
5. [`CanonicalizeTileSlice`](17-canonicalize_tile_slice.md)
6. `InferTileMemorySpace`
7. [`InsertMxScaleAddr`](19-insert_mx_scale_addr.md)（Ascend950 MX 路径；在内存空间解析后插入内部 scale 地址绑定）
8. [`ResolveBackendOpLayouts`](20-resolve_backend_op_layouts.md)（pass 内部已自动归一化语句结构）
9. [`LowerAutoVectorSplit`](21-lower_auto_vector_split.md)（在用自动拆分下降路径；在 ExpandMixedKernel 之前把 AUTO `pl.split` 混合 InCore 函数转换为显式 `split_aiv` 形态）
10. `ExpandMixedKernel`
11. [`InjectGMPipeBuffer`](23-inject_gm_pipe_buffer.md)
12. [`SplitVectorKernel`](24-split_vector_kernel.md)（仅为 split_aiv 函数打属性 + 处理无拆分双 AIV 路径）
13. [`StampTfreeSplit`](25-stamp_tfree_split.md)（把每个跨核 tpop 的 split/pipe-id 复制到与之配对的 tfree 算子上）
14. `NormalizeReturnOrder`
15. [`SkewCrossCorePipeline`](27-skew_cross_core_pipeline.md)（cube/vector 跨核软流水 skew；紧接在 LowerPipelineLoops 之前运行）
16. [`LowerPipelineToSlots`](28-lower_pipeline_to_slots.md)（把合格的 `pl.pipeline` 循环体改为轮转一个分配的多个 slot，而不是复制；自门控于 `memory_planner=PTOAS`，未处理的循环原样留给 `LowerPipelineLoops`）
17. [`LowerPipelineLoops`](29-lower_pipeline_loops.md)
18. [`CanonicalizeIOOrder`](30-canonicalize_io_order.md)
19. [`MaterializeTensorStrides`](31-materialize_tensor_strides.md) —— 自 RFC #1300 P6 起接入默认 pipeline
20. `InitMemRef`
21. [`MaterializeSemanticAliases`](33-materialize_semantic_aliases.md)（语义强制别名：循环 carry / 原地；总是运行）
22. `MemoryReuse`
23. `AllocateMemoryAddr`
24. [`FoldNoOpReshape`](36-fold_no_op_reshape.md)
25. [`FuseCreateAssembleToSlice`](37-fuse_create_assemble_to_slice.md)
26. [`DeriveCallDirections`](38-derive_call_directions.md)
27. [`AutoDeriveTaskDependencies`](39-auto_derive_task_dependencies.md)（runtime scope 编译器依赖；AUTO-scope 分析需要显式开启）
28. [`ExpandManualPhaseFence`](40-expand_manual_phase_fence.md)（manual-scope phase-fence TaskId 依赖压缩）
29. [`SynthesizeAllReduceSignals`](41-synthesize_allreduce_signals.md)（分布式：host allreduce optional signal -> explicit internal signal IR）
30. [`MaterializeCommDomainScopes`](42-materialize_comm_domain_scopes.md)（分布式：构造 WindowBuffer 并写 CommDomainScopeStmt wrappers in each host_orch body；无通信程序为 no-op）
31. [`LowerHostTensorCollectives`](43-lower_host_tensor_collectives.md)（host-level tensor collectives -> internal builtin chip dispatches）
32. [`MaterializeDistTensorCtx`](44-materialize_dist_tensor_ctx.md)（为 DistributedTensor 参数显式物化 CommCtx 参数/实参）
33. `Simplify`
34. [`LegalizeGraphBoundary`](45-legalize_graph_boundary.md)（把 Graph 体从边界标量派生出来的值上提到调用点，并拒绝 host_build_graph runtime 无法录制的边界；无 Graph 函数的程序为 no-op）
35. [`MaterializeRuntimeScopes`](46-materialize_runtime_scopes.md)（插入 AUTO RuntimeScopeStmt，使 orchestration codegen 1:1 emit SIMPLER_SCOPE）
36. [`ClassifyIterArgCarry`](47-classify_iter_arg_carry.md)（把每个 ForStmt iter_arg 标注为平凡别名 / 重绑定 carry，并为 manual-scope TaskId fence 数组定尺）
37. [`InsertCommFence`](48-insert_comm_fence.md)（在每个发布性写入与释放它的 pld.system.notify 之间插入整张 tensor 的 system.cacheinvalid + GM system.fence；跑在最后，使插入的 op 一路到 codegen 都紧邻其 notify）

[`ResolveBackendOpLayouts`](20-resolve_backend_op_layouts.md) 会根据
backend 注册的 layout 元数据修复受约束的逐元素 tile 操作。对于当前 PTO
上要求 `row_major` 的逐元素算子，它会在受约束 use-site 把 `[N, 1]`
向量操作数改写成 `[1, N]` 的 `tile.reshape`，其 layout 由目标 shape
自动推导为 `row_major`，并在需要时把结果 reshape 回原始向量 shape。

[`NormalizeReturnOrder`](26-normalize_return_order.md) 对 InCore 函数的 `ReturnStmt::value_` 重新排序，使
`return[i]` 对应声明顺序中第 i 个 `Out`/`InOut` 参数，并同步更新调用点的
`TupleGetItemExpr` 索引。这样编排代码生成可以直接通过
`out_indices[i]` 查找输出参数，而不需要追踪 `tile.store`/yield 链。该 pass
放在 `InitMemRef` 之前，在所有核函数拆分之后、内存分配之前执行。

`Simplify` 执行算术恒等式折叠（`x + 0 → x`、`x * 1 → x`）、常量表达式求值、
基于循环变量边界与 if 分支条件的范围感知重写，以及通过单赋值绑定的标量
常量传播。作为最后一步，它会运行一次 **保守的标量 DCE**：任何 LHS 为
标量 `Var`、RHS 表达式中任意位置都不含 `Call` 的 `AssignStmt`，在 LHS
没有其他引用时会被移除。只要 RHS 表达式中（顶层或嵌套在算术子树里）出现
任何 `Call`，该赋值都会被保留 —— 目前 IR 还没有纯度注解，调用可能带有
可观察的副作用。该 DCE 步骤会递归进入
`ForStmt`/`IfStmt`/`WhileStmt`/`ScopeStmt` 的 body，以便同时清理嵌套
块中的死标量。

### 直接使用 PassPipeline

```python
from pypto.pypto_core import passes

pipeline = passes.PassPipeline()
pipeline.add_pass(passes.convert_to_ssa())
pipeline.add_pass(passes.init_mem_ref())
pipeline.add_pass(passes.memory_reuse())

# Execute
result = pipeline.run(program)

# Inspect pass properties
p = passes.convert_to_ssa()
print(p.get_name())                  # "ConvertToSSA"
print(p.get_produced_properties())   # {SSAForm}
```

## 添加新 Pass

1. 在 `passes.h` 中**声明**：`Pass YourNewPass();`
2. 在 `src/ir/transforms/` 中**实现**，带有 `PassProperties`
3. 在 `python/bindings/modules/passes.cpp` 中添加 **Python 绑定**
4. **属性声明**：在工厂函数中设置 required/produced/invalidated
5. 在 `python/pypto/pypto_core/passes.pyi` 中添加**类型存根**
6. 如果是策略的一部分，在 PassManager 中**注册**
7. 在 `tests/ut/ir/transforms/` 中添加**测试**

## 测试

- `tests/ut/ir/transforms/test_ir_property.py` — IRProperty/IRPropertySet 测试
- `tests/ut/ir/transforms/test_pass_pipeline.py` — Pipeline、PassContext、插桩和自动验证测试
- `tests/ut/ir/transforms/test_pass_manager.py` — PassManager 向后兼容性测试
- `tests/ut/conftest.py` — 为所有测试启用 BEFORE_AND_AFTER 验证的 autouse fixture
