# 工具 Pass

用于中间表示 (IR) 结构的规范化和清理 Pass。

## 概述

这些工具 Pass 处理 IR 的规范化和清理任务：

1. **NormalizeStmtStructure**：确保一致的语句 (Statement) 结构
2. **FlattenSingleStmt**：移除不必要的嵌套
3. **VerifyNoNestedCall**：验证 (Verifier) 三地址码形式的 Pass

这些 Pass 通常由其他 Pass 内部使用，或用于特定的规范化需求。

## NormalizeStmtStructure

确保 IR 处于具有一致结构的规范化形式。

**前置条件**：需要 TypeChecked 属性 (Property)。

### 用途

通过以下方式规范化语句结构：

1. 将函数/if/for 的主体包装在 SeqStmts 中
2. 将 SeqStmts 内连续的 AssignStmt/EvalStmt 包装在 OpStmts 中

### API

| C++ | Python |
| --- | ------ |
| `pass::NormalizeStmtStructure()` | `passes.normalize_stmt_structure()` |

### 算法

1. **确保 SeqStmts**：将非 SeqStmts 的主体包装在 SeqStmts 中
2. **分组操作**：将连续的 AssignStmt/EvalStmt 包装在 OpStmts 中
3. **保留控制流**：保持 IfStmt/ForStmt/WhileStmt 不被包装

### 示例

**之前**：

```python
def func(...):
    x = 1  # Direct AssignStmt (not in SeqStmts)
```

**之后**：

```python
def func(...):
    SeqStmts([OpStmts([AssignStmt(x, 1)])])
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
    OpStmts([AssignStmt(a, 1), AssignStmt(b, 2)]),  # Wrapped in OpStmts
    IfStmt(...)
])
```

### 实现

**工厂函数**：`pass::NormalizeStmtStructure()`
**文件**：`src/ir/transforms/normalize_stmt_structure.cpp`
**测试**：`tests/ut/ir/transforms/test_normalize_stmt_structure_pass.py`

---

## FlattenSingleStmt

递归展平单语句块以简化 IR。

**前置条件**：需要 TypeChecked 属性。

### 用途

移除不必要的嵌套：

- 仅包含一条语句的 SeqStmts -> 该语句本身
- 仅包含一条语句的 OpStmts -> 该语句本身
- 递归应用

**注意**：该 Pass 不强制要求 Function/IfStmt/ForStmt 的主体必须是 SeqStmts。如果它们仅包含一条语句，也会被展平。

### API

| C++ | Python |
| --- | ------ |
| `pass::FlattenSingleStmt()` | `passes.flatten_single_stmt()` |

### 算法

1. **遍历 IR**：访问所有 SeqStmts 和 OpStmts 节点
2. **检查数量**：如果节点恰好包含一条语句
3. **替换**：将节点替换为该单条语句
4. **递归**：持续处理直到没有更多单语句块

### 示例

**之前**：

```python
SeqStmts([OpStmts([AssignStmt(x, 1)])])
```

**之后**：

```python
AssignStmt(x, 1)
```

**之前**：

```python
SeqStmts([OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])])
```

**之后**：

```python
OpStmts([AssignStmt(x, 1), AssignStmt(y, 2)])
# Only outer SeqStmts flattened, OpStmts preserved (has 2 statements)
```

### 实现

**工厂函数**：`pass::FlattenSingleStmt()`
**文件**：`src/ir/transforms/flatten_single_stmt.cpp`
**测试**：`tests/ut/ir/transforms/test_flatten_single_stmt_pass.py`

---

## Verify NoNestedCall（RunVerifier 的一部分）

验证 IR 处于三地址码形式（无嵌套调用）。

### 用途

此验证规则（RunVerifier 的一部分）检查 FlattenCallExpr Pass 是否已成功运行。它检测以下情况：

- `CALL_IN_CALL_ARGS`：调用参数中包含调用
- `CALL_IN_IF_CONDITION`：if 条件中包含调用
- `CALL_IN_FOR_RANGE`：for 范围中包含调用
- `CALL_IN_BINARY_EXPR`：二元表达式 (Expression) 中包含调用
- `CALL_IN_UNARY_EXPR`：一元表达式中包含调用

### API

属于 `RunVerifier` Pass 的一部分（非独立 Pass）：

```python
# Enable/disable via RunVerifier
verifier_pass = passes.run_verifier(disabled_rules=["NoNestedCall"])
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
program = passes.flatten_single_stmt()(program)
```

### 变换后清理

```python
# After a pass that might create single-statement blocks
program = some_transformation_pass()(program)
program = passes.flatten_single_stmt()(program)  # Clean up
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
| **NormalizeStmtStructure** | 在需要一致 SeqStmts/OpStmts 结构的 Pass 之前 |
| **FlattenSingleStmt** | 在变换之后清理不必要的嵌套 |
| **VerifyNoNestedCall** | 在 FlattenCallExpr 之后确保正确性 |

## 实现文件

| Pass | 头文件 | 实现文件 | 测试 |
| ---- | ------ | -------- | ---- |
| NormalizeStmtStructure | `passes.h` | `normalize_stmt_structure.cpp` | `test_normalize_stmt_structure_pass.py` |
| FlattenSingleStmt | `passes.h` | `flatten_single_stmt.cpp` | `test_flatten_single_stmt_pass.py` |
| VerifyNoNestedCall | `passes.h` | `ir_verifier.cpp` | `test_verifier.py` |
