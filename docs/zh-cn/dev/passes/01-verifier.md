# IR 验证器 (Verifier)

可扩展的验证系统，通过可插拔规则和诊断报告来验证 PyPTO 中间表示 (IR) 的正确性，并与 Pass 系统集成。

## 概述

| 组件 | 描述 |
| ---- | ---- |
| **PropertyVerifier (C++)** | 验证规则的基类 |
| **IRVerifier (C++)** | 管理规则集合并对 Program 执行验证 |
| **PropertyVerifierRegistry (C++)** | IRProperty -> PropertyVerifier 工厂的单例映射 |
| **Diagnostic** | 结构化的错误/警告报告，包含严重级别、位置和消息 |
| **VerificationError** | 在抛出模式下验证失败时抛出的异常 |

### 关键特性

- **可插拔规则系统**：可通过自定义验证规则进行扩展
- **选择性验证**：可按用例单独启用/禁用规则
- **双重验证模式**：收集诊断信息或在首个错误时抛出异常
- **Pass 集成**：可作为优化流水线中的 Pass 使用
- **全面的诊断信息**：收集所有问题及源码位置
- **基于属性的验证**：注册表将 IRProperty 值映射到验证器，用于流水线自动检查

## 架构

### 验证规则系统

验证器使用**插件架构**，每个 `PropertyVerifier` 子类是一个独立的规则：

- 规则按注册顺序在所有函数上运行
- 每个规则独立运行——一个规则的失败不影响其他规则
- 规则接收 `ProgramPtr`，并在内部决定是遍历函数还是检查程序级属性
- 可以选择性地启用/禁用规则，无需移除

### 验证模式

| 模式 | 方法 | 行为 | 使用场景 |
| ---- | ---- | ---- | -------- |
| **诊断收集** | `Verify()` | 收集所有错误/警告，返回向量 | 需要完整错误列表、构建工具、生成报告 |
| **快速失败** | `VerifyOrThrow()` | 在首个错误时抛出 VerificationError | 流水线验证、测试、开发 |

**模式选择指南**：

- IDE/工具集成使用 `Verify()`——用户希望看到所有问题
- 流水线中使用 `VerifyOrThrow()`——在无效 IR 上立即失败
- 测试中使用 `VerifyOrThrow()`——通过异常处理实现清晰的通过/失败

### 诊断系统

**Diagnostic 结构**：

| 字段 | 类型 | 用途 |
| ---- | ---- | ---- |
| `severity` | `DiagnosticSeverity` | 错误或警告 |
| `rule_name` | `string` | 检测到问题的规则 |
| `error_code` | `int` | 数字错误标识符 |
| `message` | `string` | 人类可读的描述 |
| `span` | `Span` | 源码位置信息 |

**严重级别**：

- `Error`：IR 无效，必须修复
- `Warning`：IR 有效但可能存在问题

**报告生成**：`GenerateReport()` 将诊断信息格式化为人类可读的报告，包含计数、分组和位置详情。

### 与 Pass 系统的集成

验证器通过 `run_verifier()` 集成到 Pass 流水线中：

- **返回**：一个 `Pass` 对象（Program -> Program 变换）
- **行为**：验证程序，记录诊断信息，出错时抛出异常
- **配置**：接受 `disabled_rules` 参数
- **流水线位置**：通常插入在变换之后以验证输出

**设计考虑**：验证器 Pass 是**透明的**——如果验证通过，它返回未更改的输入程序，因此可以安全地插入流水线的任何位置。

## 内置规则

| 规则名称 | IRProperty | 用途 |
| -------- | ---------- | ---- |
| **SSAVerify** | SSAForm | 无多重赋值、无名称遮蔽、无缺失 yield |
| **TypeCheck** | TypeChecked | 类型种类/数据类型/形状/大小一致性 |
| **NoNestedCall** | NoNestedCalls | 参数、条件、范围中无嵌套调用表达式 |
| **NormalizedStmtStructure** | NormalizedStmtStructure | 函数体为 SeqStmts，连续赋值包装在 OpStmts 中 |
| **FlattenedSingleStmt** | FlattenedSingleStmt | 无单元素 SeqStmts/OpStmts |
| **SplitIncoreOrch** | SplitIncoreOrch | Opaque 函数中不残留 InCore ScopeStmts |
| **IncoreBlockOps** | IncoreBlockOps | InCore 函数使用块操作（无张量级操作残留） |
| **HasMemRefs** | HasMemRefs | 所有 TileType 变量已初始化 MemRef |
| **AllocatedMemoryAddr** | AllocatedMemoryAddr | 所有 MemRef 在缓冲区限制内具有有效地址 |

### SSAVerify

**设计目标**：强制执行 PyPTO IR 正确性所依赖的 SSA 不变量。

**错误类型** (`ssa::ErrorType`)：

| 错误码 | 名称 | 描述 |
| ------ | ---- | ---- |
| 1 | `MULTIPLE_ASSIGNMENT` | 变量在同一作用域中被多次赋值 |
| 2 | `NAME_SHADOWING` | 变量名遮蔽了外层作用域的变量 |
| 3 | `MISSING_YIELD` | ForStmt 或 IfStmt 缺少必需的 YieldStmt |

**检测细节**：

- **MULTIPLE_ASSIGNMENT**：跟踪每个作用域中的所有变量声明。如果一个变量名在同一作用域内出现在多个 AssignStmt 节点中，则报告错误。
- **NAME_SHADOWING**：维护作用域栈。当进入嵌套作用域（ForStmt、IfStmt）时，如果任何新变量名与外层作用域的名称匹配，则报告错误。
- **MISSING_YIELD**：验证循环和条件块在 IR 结构语义要求的位置包含至少一个 yield 语句。

**重要性**：SSA 形式使优化 Pass 能够对变量生命周期和依赖关系做出假设。违反 SSA 可能导致不正确的变换。

### TypeCheck

**设计目标**：捕获可能导致运行时错误或生成无效代码的类型不匹配。

**错误类型** (`typecheck::ErrorType`)：

| 错误码 | 名称 | 描述 |
| ------ | ---- | ---- |
| 101 | `TYPE_KIND_MISMATCH` | 类型种类不匹配（如 ScalarType 与 TensorType） |
| 102 | `DTYPE_MISMATCH` | 数据类型不匹配（如 INT64 与 FLOAT32） |
| 103 | `SHAPE_DIMENSION_MISMATCH` | 形状维度数不匹配 |
| 104 | `SHAPE_VALUE_MISMATCH` | 形状维度值不匹配 |
| 105 | `SIZE_MISMATCH` | 控制流分支中向量大小不匹配 |

**检测细节**：

- **TYPE_KIND_MISMATCH**：检查操作是否接收了正确类别的类型（标量、张量 (Tensor)、元组等）。
- **DTYPE_MISMATCH**：验证操作间数据类型的一致性（如 Add 的所有操作数必须具有相同的 dtype）。
- **SHAPE_DIMENSION_MISMATCH**：确保张量操作接收的输入具有兼容的维度数。
- **SHAPE_VALUE_MISMATCH**：在需要的地方验证特定维度大小是否匹配（如矩阵乘法的维度）。
- **SIZE_MISMATCH**：在控制流（if/else、循环）中，确保变量向量在分支间具有一致的大小。

### NoNestedCall

**错误类型** (`NestedCallErrorType`)：

| 名称 | 描述 |
| ---- | ---- |
| `CALL_IN_CALL_ARGS` | 调用表达式嵌套在另一个调用的参数中 |
| `CALL_IN_IF_CONDITION` | 调用表达式在 if 语句条件中 |
| `CALL_IN_FOR_RANGE` | 调用表达式在 for 循环范围中 |
| `CALL_IN_BINARY_EXPR` | 调用表达式在二元表达式中 |
| `CALL_IN_UNARY_EXPR` | 调用表达式在一元表达式中 |

## PropertyVerifierRegistry

**头文件**：`include/pypto/ir/transforms/property_verifier_registry.h`

将 `IRProperty` 值映射到 `PropertyVerifier` 工厂的单例注册表。由 `PassPipeline` 用于在 Pass 执行前/后自动验证属性。每个验证器与其对应的 Pass 放在一起（如 `CreateSplitIncoreOrchPropertyVerifier` 在 `outline_incore_scopes_pass.cpp` 中），而注册表在启动时将它们连接在一起。工厂声明在 `verifier.h` 中。

| 方法 | 描述 |
| ---- | ---- |
| `GetInstance()` | 获取单例实例 |
| `Register(prop, factory)` | 为属性注册验证器工厂 |
| `GetVerifier(prop)` | 创建验证器实例（若未注册则返回 nullptr） |
| `HasVerifier(prop)` | 检查是否已注册验证器 |
| `VerifyProperties(properties, program)` | 验证一组属性，返回诊断信息 |

所有 9 个内置属性在构造函数中预注册。

## C++ API 参考

**头文件**：`include/pypto/ir/transforms/verifier.h`

### PropertyVerifier 接口

实现自定义验证规则的基类。

| 方法 | 签名 | 描述 |
| ---- | ---- | ---- |
| `GetName()` | `std::string GetName() const` | 返回唯一的规则标识符 |
| `Verify()` | `void Verify(const ProgramPtr&, std::vector<Diagnostic>&)` | 检查程序并追加诊断信息 |

每个验证器接收 `ProgramPtr`，并在内部决定是遍历函数还是检查程序级属性。验证器应追加诊断信息，而不是抛出异常。

### IRVerifier 类

管理验证规则并执行验证。

#### 构造和配置

| 方法 | 描述 |
| ---- | ---- |
| `IRVerifier()` | 构造不含规则的空验证器 |
| `static IRVerifier CreateDefault()` | 工厂方法——返回包含 SSAVerify 和 TypeCheck 规则的验证器 |
| `void AddRule(PropertyVerifierPtr rule)` | 注册验证规则（重复名称时忽略） |

#### 规则管理

| 方法 | 描述 |
| ---- | ---- |
| `void EnableRule(const std::string& name)` | 启用先前禁用的规则（未找到时无操作） |
| `void DisableRule(const std::string& name)` | 按名称禁用规则——验证时将被跳过 |
| `bool IsRuleEnabled(const std::string& name) const` | 检查规则当前是否启用 |

#### 验证执行

| 方法 | 返回值 | 抛出 | 描述 |
| ---- | ------ | ---- | ---- |
| `Verify(const ProgramPtr&)` | `std::vector<Diagnostic>` | 否 | 运行所有启用的规则，收集所有诊断信息 |
| `VerifyOrThrow(const ProgramPtr&)` | `void` | `VerificationError` | 运行验证，发现错误时抛出异常 |

#### 报告

| 方法 | 描述 |
| ---- | ---- |
| `static std::string GenerateReport(const std::vector<Diagnostic>&)` | 将诊断信息格式化为可读报告，包含计数和详情 |

**报告格式**：摘要行包含错误/警告计数，后跟每条诊断信息的详细列表，包括规则名称、严重级别、位置和消息。

## Python API 参考

**模块**：`pypto.pypto_core.passes`

### IRVerifier 类

C++ IRVerifier 的 Python 绑定，使用 snake_case 命名。

#### 工厂和构造

| 方法 | 描述 |
| ---- | ---- |
| `IRVerifier()` | 创建空验证器（通常不直接使用） |
| `IRVerifier.create_default()` | 静态方法——返回启用默认规则的验证器 |

#### 规则管理

| 方法 | 参数 | 描述 |
| ---- | ---- | ---- |
| `enable_rule(name)` | `name: str` | 启用已禁用的规则 |
| `disable_rule(name)` | `name: str` | 按名称禁用规则 |
| `is_rule_enabled(name)` | `name: str` | 检查规则是否启用（返回 `bool`） |

#### 验证

| 方法 | 参数 | 返回值 | 抛出 | 描述 |
| ---- | ---- | ------ | ---- | ---- |
| `verify(program)` | `program: Program` | `list[Diagnostic]` | 否 | 收集所有诊断信息 |
| `verify_or_throw(program)` | `program: Program` | `None` | 异常 | 出错时抛出异常 |

#### 报告

| 方法 | 参数 | 返回值 | 描述 |
| ---- | ---- | ------ | ---- |
| `generate_report(diagnostics)` | `diagnostics: list[Diagnostic]` | `str` | 静态方法——格式化诊断信息 |

### run_verifier 函数

创建验证器 Pass 的工厂函数，用于 PassManager。

| 参数 | 类型 | 默认值 | 描述 |
| ---- | ---- | ------ | ---- |
| `disabled_rules` | `list[str] \| None` | `None` | 要禁用的规则名称列表 |
| **返回值** | `Pass` | - | 验证器 Pass 对象 |

**用法**：`verify_pass = passes.run_verifier(disabled_rules=["TypeCheck"])`

### Diagnostic 类型

表示单个验证问题的只读结构。

| 字段 | 类型 | 描述 |
| ---- | ---- | ---- |
| `severity` | `DiagnosticSeverity` | `Error` 或 `Warning` |
| `rule_name` | `str` | 检测到问题的规则名称 |
| `error_code` | `int` | 数字标识符 |
| `message` | `str` | 人类可读的描述 |
| `span` | `Span` | 源码位置 |

### DiagnosticSeverity 枚举

| 值 | 含义 |
| -- | ---- |
| `DiagnosticSeverity.Error` | IR 无效 |
| `DiagnosticSeverity.Warning` | 可能存在问题但有效 |

## 使用示例

### 基本验证

```python
from pypto import ir
from pypto.pypto_core import passes

# Build program (assume 'program' is constructed)
verifier = passes.IRVerifier.create_default()
diagnostics = verifier.verify(program)

if diagnostics:
    report = passes.IRVerifier.generate_report(diagnostics)
    print(report)
```

### 禁用规则

```python
# Create verifier and disable specific rules
verifier = passes.IRVerifier.create_default()
verifier.disable_rule("TypeCheck")  # Skip type checking

# Only SSAVerify will run
diagnostics = verifier.verify(program)
```

### 使用异常处理错误

```python
verifier = passes.IRVerifier.create_default()

try:
    verifier.verify_or_throw(program)
    print("Program is valid")
except Exception as e:
    print(f"Verification failed: {e}")
```

### 作为 Pass 使用

```python
from pypto.ir import PassManager, OptimizationStrategy

# Verifier automatically included in Default strategy
pm = PassManager.get_strategy(OptimizationStrategy.Default)
result = pm.run_passes(program)  # Verifier runs after ConvertToSSA
```

### 自定义 Pass 配置

```python
from pypto.pypto_core import passes

# Create verifier pass with specific rules disabled
verify_pass = passes.run_verifier(disabled_rules=["SSAVerify"])

# Use in custom pipeline
result = verify_pass(program)
```

## 添加自定义规则

要使用领域特定检查扩展验证器，需实现自定义 PropertyVerifier。

### 实现步骤

**1. 创建规则类**（C++）

继承 `PropertyVerifier` 并实现必需的方法：

```cpp
#include "pypto/ir/transforms/verifier.h"

class MyCustomRule : public PropertyVerifier {
 public:
  std::string GetName() const override { return "MyCustom"; }

  void Verify(const ProgramPtr& program,
              std::vector<Diagnostic>& diagnostics) override {
    for (const auto& [gv, func] : program->functions_) {
      // Verification logic per function
    }
  }
};
```

#### 2. 创建工厂函数

```cpp
PropertyVerifierPtr CreateMyCustomRule() {
  return std::make_shared<MyCustomRule>();
}
```

#### 3. 注册规则

```cpp
// Add to default verifier in verifier.cpp CreateDefault():
verifier.AddRule(CreateMyCustomRule());

// Or register with PropertyVerifierRegistry for pipeline integration:
PropertyVerifierRegistry::GetInstance().Register(IRProperty::MyProp, CreateMyCustomRule);
```

**4. Python 绑定**（可选）

添加到 `python/bindings/modules/passes.cpp`：

```cpp
passes.def("create_my_custom_rule", &CreateMyCustomRule,
           "Create MyCustom verification rule");
```

**5. 类型存根**（可选）

添加到 `python/pypto/pypto_core/passes.pyi`：

```python
def create_my_custom_rule() -> PropertyVerifier: ...
```

### 准则

- 使用 `IRVisitor` 系统地遍历 IR 节点
- 保持规则聚焦——一个规则检查一类问题
- 避免副作用——仅读取 IR 并写入诊断信息
- 创建描述性诊断信息，包含严重级别、规则名称、错误码、消息和 span

### 集成点

| 位置 | 用途 |
| ---- | ---- |
| `src/ir/transforms/your_rule.cpp` | 实现 |
| `include/pypto/ir/transforms/passes.h` | 工厂声明（如需暴露） |
| `src/ir/transforms/verifier.cpp` | 添加到 `CreateDefault()` |
| `python/bindings/modules/passes.cpp` | Python 绑定 |
| `tests/ut/ir/transforms/test_verifier.py` | 测试用例 |

## 相关组件

- **Pass 系统**（`00-pass_manager.md`）：验证器作为 Pass 集成，PropertyVerifierRegistry 由 PassPipeline 使用
- **IR 构建器**（`../ir/06-builder.md`）：构造验证器验证的 IR
- **类型系统**（`../ir/02-types.md`）：TypeCheck 规则根据类型系统进行验证
- **错误处理**（`include/pypto/core/error.h`）：Diagnostic 和 VerificationError 定义

## 测试

测试覆盖在 `tests/ut/ir/transforms/test_verifier.py` 中：有效/无效程序验证、规则启用/禁用、异常与诊断模式、Pass 集成、诊断字段访问、报告生成。
