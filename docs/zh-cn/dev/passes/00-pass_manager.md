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
- **基于策略的流水线**：预配置的优化级别（Default/PTOAS）
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
| `FlattenedSingleStmt` | 单语句块已展平 |
| `SplitIncoreOrch` | InCore 作用域已提取为独立函数 |
| `HasMemRefs` | 变量上已初始化内存引用 (MemRef) 对象 |
| `IncoreBlockOps` | InCore 函数使用块操作 |
| `AllocatedMemoryAddr` | 所有 MemRef 在缓冲区限制内具有有效地址 |

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
| ConvertToSSA | TypeChecked | TypeChecked, SSAForm | NormalizedStmtStructure, FlattenedSingleStmt |
| FlattenCallExpr | TypeChecked | TypeChecked, NoNestedCalls | NormalizedStmtStructure, FlattenedSingleStmt |
| NormalizeStmtStructure | TypeChecked | TypeChecked, NormalizedStmtStructure | FlattenedSingleStmt |
| FlattenSingleStmt | TypeChecked | TypeChecked, FlattenedSingleStmt | NormalizedStmtStructure |
| OutlineIncoreScopes | TypeChecked, SSAForm | SplitIncoreOrch | — |
| ConvertTensorToBlockOps | SplitIncoreOrch | IncoreBlockOps | — |
| InitMemRef | TypeChecked, SSAForm, SplitIncoreOrch, IncoreBlockOps | HasMemRefs | SSAForm |
| BasicMemoryReuse | TypeChecked, SplitIncoreOrch, IncoreBlockOps, HasMemRefs | — | — |
| InsertSync | TypeChecked, SplitIncoreOrch, IncoreBlockOps, HasMemRefs | — | — |
| AllocateMemoryAddr | TypeChecked, SplitIncoreOrch, IncoreBlockOps, HasMemRefs | AllocatedMemoryAddr | — |
| RunVerifier | — | — | — |

> **注意**：VerifySSA 和 TypeCheck 是**属性验证器 (PropertyVerifier)**（验证规则），不是 Pass。它们通过 `RunVerifier` 或 `VerificationInstrument` 运行——参见[验证器](01-verifier.md)。

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

### 测试夹具

所有单元测试通过 `tests/ut/conftest.py` 自动在 BEFORE_AND_AFTER 验证模式下运行：

```python
@pytest.fixture(autouse=True)
def pass_verification_context():
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield
```

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

**使用默认策略时**：

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

# Environment variable (default when no PassContext): PYPTO_VERIFY_LEVEL=none|basic
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

### 直接使用 PassPipeline

```python
from pypto.pypto_core import passes

pipeline = passes.PassPipeline()
pipeline.add_pass(passes.convert_to_ssa())
pipeline.add_pass(passes.init_mem_ref())
pipeline.add_pass(passes.basic_memory_reuse())

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
- `tests/ut/conftest.py` — 为所有测试启用 AFTER 验证的自动使用夹具
