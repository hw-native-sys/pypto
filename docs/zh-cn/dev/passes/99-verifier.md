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
- **结构性属性 (Structural Properties)**：TypeChecked 和 BreakContinueValid 在流水线启动时验证一次，不在每个 Pass 中声明
- **双重验证模式**：收集诊断信息或在首个错误时抛出异常
- **Pass 集成**：可作为优化流水线中的 Pass 使用
- **全面的诊断信息**：收集所有问题及源码位置

## 架构

### 结构性属性 vs 流水线属性

| 类别 | 示例 | 行为 |
| ---- | ---- | ---- |
| **结构性** | TypeChecked, BreakContinueValid | 始终为真。在流水线启动时验证。不在 PassProperties 中声明。 |
| **流水线** | SSAForm, NoNestedCalls, HasMemRefs, ... | 由 Pass 产生/失效。按 Pass 声明的契约验证。 |

`GetStructuralProperties()` 返回 `{TypeChecked, BreakContinueValid}`。这些在 `PassPipeline::Run()` 中**于流水线启动时验证一次**。由于没有 Pass 在 `required`/`produced`/`invalidated` 中声明它们，它们在整个过程中保持已验证状态。

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
2. **`VerificationInstrument`**：一个 `PassInstrument`，通过 `PassContext` 在 Pass 执行前/后验证所需/产生的属性。

`run_verifier()` 工具函数创建一个独立的 `Pass`，用于自定义流水线中的临时使用，但它**不是**默认优化策略的一部分。

## 内置规则

| 规则名称 | IRProperty | 用途 |
| -------- | ---------- | ---- |
| **SSAVerify** | SSAForm | 无多重赋值、无名称遮蔽、无缺失 yield |
| **TypeCheck** | TypeChecked | 类型种类/数据类型/形状/大小一致性 |
| **NoNestedCall** | NoNestedCalls | 参数、条件、范围中无嵌套调用表达式 |
| **BreakContinueCheck** | BreakContinueValid | break/continue 仅在顺序/while 循环中 |
| **NormalizedStmtStructure** | NormalizedStmtStructure | 函数体为 SeqStmts，连续赋值包装在 OpStmts 中 |
| **FlattenedSingleStmt** | FlattenedSingleStmt | 无单元素 SeqStmts/OpStmts |
| **SplitIncoreOrch** | SplitIncoreOrch | Opaque 函数中不残留 InCore ScopeStmts |
| **IncoreTileOps** | IncoreTileOps | InCore 函数使用 tile 操作（无张量级操作残留） |
| **HasMemRefs** | HasMemRefs | 所有 TileType 变量已初始化 MemRef |
| **AllocatedMemoryAddr** | AllocatedMemoryAddr | 所有 MemRef 在缓冲区限制内具有有效地址 |

### SSAVerify

**错误类型** (`ssa::ErrorType`)：

| 错误码 | 名称 | 描述 |
| ------ | ---- | ---- |
| 1 | `MULTIPLE_ASSIGNMENT` | 变量在同一作用域中被多次赋值 |
| 2 | `NAME_SHADOWING` | 变量名遮蔽了外层作用域的变量 |
| 3 | `MISSING_YIELD` | ForStmt 或 IfStmt 缺少必需的 YieldStmt |

### TypeCheck

**错误类型** (`typecheck::ErrorType`)：

| 错误码 | 名称 | 描述 |
| ------ | ---- | ---- |
| 101 | `TYPE_KIND_MISMATCH` | 类型种类不匹配（如 ScalarType 与 TensorType） |
| 102 | `DTYPE_MISMATCH` | 数据类型不匹配（如 INT64 与 FLOAT32） |
| 103 | `SHAPE_DIMENSION_MISMATCH` | 形状维度数不匹配 |
| 104 | `SHAPE_VALUE_MISMATCH` | 形状维度值不匹配 |
| 105 | `SIZE_MISMATCH` | 控制流分支中向量大小不匹配 |
| 106 | `IF_CONDITION_MUST_BE_SCALAR` | IfStmt 条件必须是 ScalarType |
| 107 | `FOR_RANGE_MUST_BE_SCALAR` | ForStmt 范围必须是 ScalarType |

### NoNestedCall

| 名称 | 描述 |
| ---- | ---- |
| `CALL_IN_CALL_ARGS` | 调用表达式嵌套在另一个调用的参数中 |
| `CALL_IN_IF_CONDITION` | 调用表达式在 if 语句条件中 |
| `CALL_IN_FOR_RANGE` | 调用表达式在 for 循环范围中 |
| `CALL_IN_BINARY_EXPR` | 调用表达式在二元表达式中 |
| `CALL_IN_UNARY_EXPR` | 调用表达式在一元表达式中 |

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
| `GetStructuralProperties()` | `{TypeChecked, BreakContinueValid}` | 在流水线启动时验证的不变量 |
| `GetDefaultVerifyProperties()` | `{SSAForm, TypeChecked, NoNestedCalls, BreakContinueValid}` | `run_verifier()` 的默认属性集 |
| `GetVerifiedProperties()` | `{SSAForm, TypeChecked, AllocatedMemoryAddr, BreakContinueValid}` | `PassPipeline` 自动验证的轻量级属性集 |

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
- **错误处理**（`include/pypto/core/error.h`）：Diagnostic 和 VerificationError 定义

## 测试

测试覆盖在 `tests/ut/ir/transforms/test_verifier.py` 中：有效/无效程序验证、基于属性的选择、异常与诊断模式、Pass 集成、诊断字段访问、报告生成、结构性/默认属性集。
