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
- **结构性属性 (Structural Properties)**：TypeChecked、BreakContinueValid、NoRedundantBlocks、UseAfterDef、OutParamNotShadowed、NoNestedInCore、InOutUseValid 和 PipelineLoopValid 在流水线启动时由 `PassPipeline` 验证，并由 `VerificationInstrument` 在每个 Pass 执行前后验证
- **双重验证模式**：收集诊断信息或在首个错误时抛出异常
- **Pass 集成**：可作为优化流水线中的 Pass 使用
- **全面的诊断信息**：收集所有问题及源码位置

## 架构

### 结构性属性 vs 流水线属性

| 类别 | 示例 | 行为 |
| ---- | ---- | ---- |
| **结构性** | TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid | 始终为真。在流水线启动时验证，并由 `VerificationInstrument` 在每个 Pass 执行前后验证。不在 PassProperties 中声明。 |
| **流水线** | SSAForm, NoNestedCalls, HasMemRefs, ... | 由 Pass 产生/失效。按 Pass 声明的契约验证。 |

`GetStructuralProperties()` 返回 `{TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid}`。这些在 `PassPipeline::Run()` 中**于流水线启动时验证**，并由 `VerificationInstrument` **在每个 Pass 执行前后验证**。由于没有 Pass 在 `required`/`produced`/`invalidated` 中声明它们，`VerificationInstrument` 将它们与 Pass 声明的属性合并，确保没有 Pass 破坏这些基本不变量。

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

1. **自动属性验证**：`PassPipeline` 使用 `PropertyVerifierRegistry` 在每个 Pass 执行后检查产生的属性（由 `PassContext` 中的 `VerificationLevel` 控制）。结构性属性在流水线启动时检查。详见 [Pass 管理器](00-pass_manager.md)。
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
| **OrchestrationReferencesResolved** | OrchestrationReferencesResolved | `FunctionType::Orchestration` 函数体内每一个非 builtin Call 必须对应到 Program 中存在的 Function。取代 codegen 端原本在生成时抛错的 `ValidateOrchestrationReferences` 遍历。 |

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
| `GetStructuralProperties()` | `{TypeChecked, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore, InOutUseValid, PipelineLoopValid}` | 在流水线启动时及每个 Pass 执行前后验证的不变量 |
| `GetDefaultVerifyProperties()` | `{SSAForm, TypeChecked, NoNestedCalls, BreakContinueValid, NoRedundantBlocks, UseAfterDef, OutParamNotShadowed, NoNestedInCore}` | `run_verifier()` 的默认属性集 |
| `GetVerifiedProperties()` | `{SSAForm, TypeChecked, AllocatedMemoryAddr, BreakContinueValid, NoRedundantBlocks, InOutUseValid}` | `PassPipeline` 自动验证的轻量级属性集 |

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
