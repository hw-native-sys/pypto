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
- **基于策略的流水线**：预配置的优化级别（`Default`、`DebugTileOptimization`）
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
| `ClusterOutlined` | Cluster 作用域已提取为 Group 函数 |
| `HasMemRefs` | 变量上已初始化内存引用 (MemRef) 对象 |
| `IncoreTileOps` | InCore 函数使用 tile 操作 |
| `MixedKernelExpanded` | 混合 InCore 函数已拆分为 AIC + AIV + Group |
| `AllocatedMemoryAddr` | 所有 MemRef 在缓冲区限制内具有有效地址 |
| `TileTypeCoherence` | 每个 TileType 都具有规范的 tile_view（隐式视图存储为 nullopt） |

### IRPropertySet

基于位集合的高效集合，支持 `Insert`、`Remove`、`Contains`、`ContainsAll`、`Union`、`Difference`、`ToString`。

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
| UnrollLoops | TypeChecked | TypeChecked | — |
| CtrlFlowTransform | TypeChecked | TypeChecked, StructuredCtrlFlow | — |
| ConvertToSSA | TypeChecked | TypeChecked, SSAForm | NormalizedStmtStructure |
| FlattenCallExpr | SSAForm | SSAForm, NoNestedCalls | NormalizedStmtStructure |
| SplitChunkedLoops | TypeChecked, SSAForm | TypeChecked, SSAForm | — |
| InterchangeChunkLoops | TypeChecked, SSAForm | TypeChecked, SSAForm | — |
| NormalizeStmtStructure | TypeChecked | TypeChecked, NormalizedStmtStructure | — |
| OutlineIncoreScopes | TypeChecked, SSAForm | SplitIncoreOrch | — |
| OutlineClusterScopes | TypeChecked, SSAForm | ClusterOutlined | — |
| ConvertTensorToTileOps | SplitIncoreOrch | IncoreTileOps | — |
| FlattenTileNdTo2D | SSAForm, IncoreTileOps | SSAForm, TileOps2D | — |
| AutoTileMatmulL0 | SSAForm, IncoreTileOps, TileOps2D | SSAForm, IncoreTileOps, TileOps2D | — |
| ResolveBackendOpLayouts | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D | NormalizedStmtStructure |
| ExpandMixedKernel | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D | SSAForm, MixedKernelExpanded | — |
| NormalizeReturnOrder | SplitIncoreOrch, IncoreTileOps | — | — |
| InitMemRef | TypeChecked, SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D | HasMemRefs | SSAForm |
| MemoryReuse | TypeChecked, SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D | — | — |
| AllocateMemoryAddr | TypeChecked, SplitIncoreOrch, IncoreTileOps, HasMemRefs, TileOps2D | AllocatedMemoryAddr | — |
| FuseCreateAssembleToSlice | — | — | — |
| DeriveCallDirections | SplitIncoreOrch | CallDirectionsResolved | — |
| Simplify | — | — | — |

> **注意**：VerifySSA 和 TypeCheck 是**属性验证器 (PropertyVerifier)**（验证规则），不是 Pass。它们通过 `VerificationInstrument` 或 `run_verifier()` 工具函数运行——参见[验证器](99-verifier.md)。

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

### ReportInstrument

在指定 Pass 执行后生成报告文件的插桩。使用 `ReportGeneratorRegistry` 分发报告生成：

```cpp
class ReportInstrument : public PassInstrument {
  explicit ReportInstrument(std::string output_dir);
  void EnableReport(ReportType type, std::string trigger_pass);
};
```

```python
# Python: 在 AllocateMemoryAddr 后生成内存报告
instrument = passes.ReportInstrument("/path/to/report")
instrument.enable_report(passes.ReportType.Memory, "AllocateMemoryAddr")

with passes.PassContext([instrument]):
    pipeline.run(program)
```

`compile()` 会自动创建 `ReportInstrument`，在 `build_output/<name>/report/` 目录中生成内存报告。

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
};
```

`PassPipeline` 是简单的有序 Pass 列表。每个 Pass 的 `operator()` 检查活跃的 `PassContext` 以获取插桩。

### 自动验证

当 `VerificationLevel` 为 `Basic`（默认值）时，流水线会自动对一组**轻量级属性**各验证一次。这可以在无需手动设置 `PassContext` 的情况下捕获常见的 IR 错误。

**验证的属性**：`{SSAForm, TypeChecked, AllocatedMemoryAddr}`

**工作原理**：

1. 每个 Pass 执行后，检查是否产生了尚未检查的已验证属性
2. 使用 `PropertyVerifierRegistry` 验证这些属性
3. 出错时抛出 `VerificationError`
4. 跟踪已验证属性以避免重复检查

**使用 `Default` 策略时**：

| Pass 执行后 | 验证的属性 | 累计 |
| ----------- | ---------- | ---- |
| ConvertToSSA | SSAForm, TypeChecked | 2 |
| FlattenCallExpr | *(TypeChecked 已验证——跳过)* | 2 |
| AllocateMemoryAddr | AllocatedMemoryAddr | 3 |

**总计：3 次属性检查**（每个属性恰好验证一次）。

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

`Default` 和 `DebugTileOptimization` 共享的 PTO tile 阶段顺序为：

1. `FlattenTileNdTo2D`
2. [`AutoTileMatmulL0`](15-auto_tile_matmul_l0.md)
3. `InferTileMemorySpace`
4. `ResolveTransposeLayout`
5. [`ResolveBackendOpLayouts`](18-resolve_backend_op_layouts.md)
6. `NormalizeStmtStructure`
7. `ExpandMixedKernel`
8. [`InjectGMPipeBuffer`](20-inject_gm_pipe_buffer.md)
9. [`SplitVectorKernel`](21-split_vector_kernel.md)
10. `NormalizeReturnOrder`
11. [`LowerPipelineLoops`](23-lower_pipeline_loops.md)
12. [`CanonicalizeIOOrder`](24-canonicalize_io_order.md)
13. `InitMemRef`
14. `MemoryReuse`
15. [`LegalizePTOBufferReuse`](27-legalize_pto_buffer_reuse.md)
16. `AllocateMemoryAddr`
17. `FuseCreateAssembleToSlice`
18. [`DeriveCallDirections`](30-derive_call_directions.md)
19. `Simplify`

`DebugTileOptimization` 只是用于排查 PTO tile 阶段的调试策略，会跳过
tensor-only 前缀 pass。正常编译和非 strategy 专项测试都应优先使用
`Default`，以保证主维护流水线持续被覆盖。

[`ResolveBackendOpLayouts`](18-resolve_backend_op_layouts.md) 会根据
backend 注册的 layout 元数据修复受约束的逐元素 tile 操作。对于当前 PTO
上要求 `row_major` 的逐元素算子，它会在受约束 use-site 把 `[N, 1]`
向量操作数改写成 `[1, N]` 的 `tile.reshape`，其 layout 由目标 shape
自动推导为 `row_major`，并在需要时把结果 reshape 回原始向量 shape。

[`NormalizeReturnOrder`](22-normalize_return_order.md) 对 InCore 函数的 `ReturnStmt::value_` 重新排序，使
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
