# 工具 Pass

用于中间表示 (IR) 结构的规范化和清理 Pass。

## 概述

这些工具 Pass 处理 IR 的规范化和清理任务：

1. **NormalizeStmtStructure**：确保一致的语句 (Statement) 结构
2. **VerifyNoNestedCall**：验证 (Verifier) 三地址码形式的 Pass

这些 Pass 通常由其他 Pass 内部使用，或用于特定的规范化需求。

## SeqStmts::Flatten

用于创建格式正确的 `SeqStmts` 节点的静态辅助方法。构造 `SeqStmts` 时应优先使用 `Flatten()` 以满足 `NoRedundantBlocks` 结构属性。

### SeqStmts::Flatten

```cpp
// 签名 (include/pypto/ir/stmt.h)
static StmtPtr SeqStmts::Flatten(std::vector<StmtPtr> stmts, Span span);
```

| 输入 | 输出 |
| ---- | ---- |
| `Flatten({a, SeqStmts({b, c}), d}, span)` | `SeqStmts({a, b, c, d})` |
| `Flatten({a}, span)` | `a`（解包） |
| `Flatten({}, span)` | `SeqStmts({})` |

嵌套的 `SeqStmts` 子节点会被吸收（展平）。单子节点结果会被解包。

### 在 IRMutator 中的使用

基础 `IRMutator::VisitStmt_(SeqStmtsPtr)` 会自动调用 `SeqStmts::Flatten()`。所有继承自 `IRMutator` 的 Pass 无需额外操作即可生成格式正确的 IR。

### 在 Pass 中的使用

直接构造 `SeqStmts`（而非通过 mutator）的 Pass 应使用 `Flatten()`：

```cpp
// ✅ 正确 — 始终格式正确
return SeqStmts::Flatten(std::move(new_stmts), op->span_);

// ❌ 错误 — 可能产生单子节点或嵌套 SeqStmts
return std::make_shared<SeqStmts>(new_stmts, op->span_);
```

### NoRedundantBlocks（结构属性）

`NoRedundantBlocks` 是一个 **结构属性** — 在流水线开始时验证，预期始终成立。检查项：

| 检查 | SeqStmts |
| ---- | -------- |
| 单子节点（应解包） | 是 |
| 嵌套（应展平） | 是 |

---

## NormalizeStmtStructure

确保 IR 处于具有一致结构的规范化形式。

**前置条件**：需要 TypeChecked 属性 (Property)。

### 用途

通过以下方式规范化语句结构：

1. 展平嵌套的 `SeqStmts`
2. 解包单子节点的 `SeqStmts`
3. 让 `AssignStmt`/`EvalStmt` 直接作为 `SeqStmts` 的子节点

### API

| C++ | Python |
| --- | ------ |
| `pass::NormalizeStmtStructure()` | `passes.normalize_stmt_structure()` |

### 算法

1. **展平嵌套**：将嵌套的 `SeqStmts` 吸收到父节点中
2. **解包单子节点**：直接返回单个子节点（不使用冗余的 `SeqStmts` 包装）
3. **保留控制流**：保持 `IfStmt`/`ForStmt`/`WhileStmt` 的主体结构

### 示例

**之前**：

```python
def func(...):
    x = 1  # Direct AssignStmt
```

**之后**：

```python
def func(...):
    AssignStmt(x, 1)  # 单子节点 SeqStmts 会被解包
```

**之前**：

```python
SeqStmts([
    AssignStmt(a, 1),  # Consecutive operations
    AssignStmt(b, 2),
    IfStmt(...)
])
```

**之后**：

```python
SeqStmts([
    AssignStmt(a, 1),
    AssignStmt(b, 2),
    IfStmt(...)
])
```

### 实现

**工厂函数**：`pass::NormalizeStmtStructure()`
**文件**：`src/ir/transforms/utils/normalize_stmt_structure.cpp`
**测试**：`tests/ut/ir/transforms/test_normalize_stmt_structure_pass.py`

---

## Verify NoNestedCall（IRVerifier 的一部分）

验证 IR 处于三地址码形式（无嵌套调用）。

### 用途

此验证规则（IRVerifier 的一部分）检查 FlattenCallExpr Pass 是否已成功运行。它检测以下情况：

- `CALL_IN_CALL_ARGS`：调用参数中包含调用
- `CALL_IN_IF_CONDITION`：if 条件中包含调用
- `CALL_IN_FOR_RANGE`：for 范围中包含调用
- `CALL_IN_BINARY_EXPR`：二元表达式 (Expression) 中包含调用
- `CALL_IN_UNARY_EXPR`：一元表达式中包含调用

### API

通过 PropertyVerifierRegistry 验证（非独立 Pass）：

```python
# Verify with default properties (includes NoNestedCalls)
verify_pass = passes.run_verifier()

# Or exclude NoNestedCalls from verification
props = passes.get_default_verify_properties()
props.remove(passes.IRProperty.NoNestedCalls)
verify_pass = passes.run_verifier(properties=props)
```

### 实现

**文件**：`src/ir/verifier/verify_no_nested_call_pass.cpp`
**规则名称**：`"NoNestedCall"`
**测试**：`tests/ut/ir/transforms/test_verifier.py`

---

## 使用模式

### 规范化流水线

```python
# Typical normalization sequence
program = passes.normalize_stmt_structure()(program)
```

### 验证

```python
# Verify three-address code form
verifier = passes.run_verifier()  # Includes NoNestedCall by default
verified_program = verifier(program)  # Throws if nested calls found
```

---

## 使用时机

| Pass | 使用时机 |
| ---- | -------- |
| **NormalizeStmtStructure** | 在需要扁平 `SeqStmts` 结构的 Pass 之前 |
| **VerifyNoNestedCall** | 在 FlattenCallExpr 之后确保正确性 |

## 实现文件

| Pass | 头文件 | 实现文件 | 测试 |
| ---- | ------ | -------- | ---- |
| NormalizeStmtStructure | `passes.h` | `normalize_stmt_structure.cpp` | `test_normalize_stmt_structure_pass.py` |
| VerifyNoNestedCall | `passes.h` | `verify_no_nested_call_pass.cpp` | `test_verifier.py` |
